"""FastAPI application for Lifinity House Price Prediction API."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.predictor import Predictor
from api.schemas import BatchRequest, BatchResponse, PredictionResponse, PropertyInput
from lifinity.config import get_project_root

logger = logging.getLogger("lifinity.api")

# Global predictor instance
predictor: Predictor | None = None


def log_request_jsonl(
    endpoint: str,
    input_fields: dict[str, Any] | list[dict[str, Any]],
    prediction: Any,
    latency_ms: float,
    warnings: list[str] | list[list[str]],
) -> None:
    """Log request details as a single JSON line to logs/requests.jsonl."""
    try:
        root = get_project_root()
        logs_dir = root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "requests.jsonl"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "input_fields": input_fields,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "warnings": warnings,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to log request to jsonl: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler to initialize the predictor ONCE."""
    global predictor
    try:
        predictor = Predictor()
        logger.info(
            "Successfully loaded Predictor (model_type=%s, source=%s)",
            predictor.model_type,
            predictor.model_source,
        )
    except Exception as exc:
        logger.error("Failed to load Predictor during lifespan startup: %s", exc)
        predictor = None
    yield


app = FastAPI(
    title="Lifinity - House Price Prediction API",
    version="1.0.0",
    description="Production machine learning API for predicting house sale prices in Ames, Iowa.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean JSON formatting for 422 validation errors."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        msg = err.get("msg", "Validation error")
        errors.append(f"{loc}: {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors, "message": "Validation failed for request input."},
    )


@app.get("/")
async def root():
    """Root endpoint presenting API info and documentation link."""
    return {
        "name": "Lifinity - House Price Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API Root Info"},
            {"path": "/health", "method": "GET", "description": "Health Check"},
            {"path": "/model-info", "method": "GET", "description": "Model Metadata & Evaluation Metrics"},
            {"path": "/predict", "method": "POST", "description": "Single House Price Prediction"},
            {"path": "/predict/batch", "method": "POST", "description": "Batch House Price Prediction (1-500)"},
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "model_loaded": False,
                "detail": "Model predictor failed to initialize.",
            },
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": predictor.model_type,
        "model_source": predictor.model_source,
    }


@app.get("/model-info")
async def model_info():
    """Returns final model type, blend weights, test metrics, training sample size, and git commit."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    m = predictor.metrics
    return {
        "final_model_type": m.get("final_model_type", predictor.model_type),
        "final_model_weights": m.get("final_model_weights", {}),
        "test_metrics": {
            "rmse_log": m.get("rmse_log"),
            "mae": m.get("mae"),
            "mape": m.get("mape"),
            "r2_log": m.get("r2_log"),
            "r2_dollar": m.get("r2_dollar"),
        },
        "n_train": m.get("n_train"),
        "trained_commit": m.get("git_commit"),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PropertyInput):
    """Predict price for a single property input."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    try:
        record_dict = input_data.model_dump(by_alias=True)
        results = predictor.predict([record_dict])
        res = results[0]

        response = PredictionResponse(
            predicted_price=res["predicted_price"],
            price_range_low=res["price_range_low"],
            price_range_high=res["price_range_high"],
            model_type=res["model_type"],
            model_version=res["model_version"],
            fields_defaulted=res["fields_defaulted"],
            warnings=res["warnings"],
            latency_ms=res["latency_ms"],
        )

        log_request_jsonl(
            endpoint="/predict",
            input_fields=record_dict,
            prediction=res["predicted_price"],
            latency_ms=res["latency_ms"],
            warnings=res["warnings"],
        )

        return response
    except Exception as exc:
        logger.error("Prediction execution error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal prediction failure. Safe error."
        )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request_body: BatchRequest | list[PropertyInput]):
    """Predict prices for a batch of 1-500 property records."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    try:
        if isinstance(request_body, BatchRequest):
            houses = request_body.houses
        else:
            houses = request_body

        records = [h.model_dump(by_alias=True) for h in houses]
        results = predictor.predict(records)

        responses = []
        batch_warnings = []
        total_latency = 0.0

        for res in results:
            responses.append(
                PredictionResponse(
                    predicted_price=res["predicted_price"],
                    price_range_low=res["price_range_low"],
                    price_range_high=res["price_range_high"],
                    model_type=res["model_type"],
                    model_version=res["model_version"],
                    fields_defaulted=res["fields_defaulted"],
                    warnings=res["warnings"],
                    latency_ms=res["latency_ms"],
                )
            )
            batch_warnings.append(res["warnings"])
            total_latency += res["latency_ms"]

        log_request_jsonl(
            endpoint="/predict/batch",
            input_fields=records,
            prediction=[r["predicted_price"] for r in results],
            latency_ms=round(total_latency, 2),
            warnings=batch_warnings,
        )

        return BatchResponse(predictions=responses)
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal batch prediction failure. Safe error."
        )
