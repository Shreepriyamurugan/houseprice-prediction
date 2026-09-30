"""Pydantic schemas for the FastAPI House Price Prediction API."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Allowed categorical sets extracted directly from data/raw/train.csv
ALLOWED_NEIGHBORHOODS = (
    "Blmngtn",
    "Blueste",
    "BrDale",
    "BrkSide",
    "ClearCr",
    "CollgCr",
    "Crawfor",
    "Edwards",
    "Gilbert",
    "IDOTRR",
    "MeadowV",
    "Mitchel",
    "NAmes",
    "NPkVill",
    "NWAmes",
    "NoRidge",
    "NridgHt",
    "OldTown",
    "SWISU",
    "Sawyer",
    "SawyerW",
    "Somerst",
    "StoneBr",
    "Timber",
    "Veenker",
)

ALLOWED_MS_ZONINGS = ("C (all)", "FV", "RH", "RL", "RM")

ALLOWED_SALE_CONDITIONS = ("Abnorml", "AdjLand", "Alloca", "Family", "Normal", "Partial")

QualityRating = Literal["Ex", "Gd", "TA", "Fa", "Po"]
CentralAirRating = Literal["Y", "N"]


class PropertyInput(BaseModel):
    """Property features input for house price prediction."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
    )

    # REQUIRED fields
    OverallQual: int = Field(
        ...,
        ge=1,
        le=10,
        description="Overall material and finish quality (1-10)",
        examples=[7],
    )
    GrLivArea: float = Field(
        ...,
        ge=300.0,
        le=6000.0,
        description="Above grade (ground) living area sq ft (300-6000)",
        examples=[1710.0],
    )
    Neighborhood: str = Field(
        ...,
        description="Physical locations within Ames city limits",
        examples=["CollgCr"],
    )

    # OPTIONAL fields
    OverallCond: int | None = Field(
        None, ge=1, le=10, description="Overall condition rating (1-10)"
    )
    YearBuilt: int | None = Field(
        None, ge=1870, le=2026, description="Original construction date"
    )
    YearRemodAdd: int | None = Field(
        None, ge=1870, le=2026, description="Remodel date"
    )
    TotalBsmtSF: float | None = Field(
        None, ge=0.0, le=6500.0, description="Total square feet of basement area"
    )
    first_flr_sf: float | None = Field(
        None,
        ge=0.0,
        le=5000.0,
        alias="1stFlrSF",
        description="First Floor square feet",
    )
    second_flr_sf: float | None = Field(
        None,
        ge=0.0,
        le=2500.0,
        alias="2ndFlrSF",
        description="Second floor square feet",
    )
    GarageCars: int | None = Field(
        None, ge=0, le=5, description="Size of garage in car capacity"
    )
    FullBath: int | None = Field(
        None, ge=0, le=4, description="Full bathrooms above grade"
    )
    HalfBath: int | None = Field(
        None, ge=0, le=2, description="Half baths above grade"
    )
    BsmtFullBath: int | None = Field(
        None, ge=0, le=3, description="Basement full bathrooms"
    )
    BsmtHalfBath: int | None = Field(
        None, ge=0, le=2, description="Basement half bathrooms"
    )
    BedroomAbvGr: int | None = Field(
        None, ge=0, le=8, description="Bedrooms above grade"
    )
    TotRmsAbvGrd: int | None = Field(
        None, ge=2, le=15, description="Total rooms above grade (does not include bathrooms)"
    )
    Fireplaces: int | None = Field(
        None, ge=0, le=4, description="Number of fireplaces"
    )
    LotArea: float | None = Field(
        None, ge=1000.0, le=250000.0, description="Lot size in square feet"
    )
    LotFrontage: float | None = Field(
        None, ge=20.0, le=350.0, description="Linear feet of street connected to property"
    )
    YrSold: int | None = Field(
        None, ge=2006, le=2026, description="Year Sold"
    )
    MoSold: int | None = Field(
        None, ge=1, le=12, description="Month Sold"
    )
    MSZoning: str | None = Field(
        None, description="Identifies the general zoning classification of the sale"
    )
    KitchenQual: QualityRating | None = Field(
        None, description="Kitchen quality"
    )
    ExterQual: QualityRating | None = Field(
        None, description="Evaluates the quality of the material on the exterior"
    )
    BsmtQual: QualityRating | None = Field(
        None, description="Evaluates the height of the basement"
    )
    CentralAir: CentralAirRating | None = Field(
        None, description="Central air conditioning"
    )
    SaleCondition: str | None = Field(
        None, description="Condition of sale"
    )

    @field_validator("Neighborhood")
    @classmethod
    def validate_neighborhood(cls, v: str) -> str:
        if v not in ALLOWED_NEIGHBORHOODS:
            allowed = ", ".join(ALLOWED_NEIGHBORHOODS)
            raise ValueError(f"Unknown Neighborhood '{v}'. Allowed values: {allowed}")
        return v

    @field_validator("MSZoning")
    @classmethod
    def validate_mszoning(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_MS_ZONINGS:
            allowed = ", ".join(ALLOWED_MS_ZONINGS)
            raise ValueError(f"Unknown MSZoning '{v}'. Allowed values: {allowed}")
        return v

    @field_validator("SaleCondition")
    @classmethod
    def validate_sale_condition(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_SALE_CONDITIONS:
            allowed = ", ".join(ALLOWED_SALE_CONDITIONS)
            raise ValueError(f"Unknown SaleCondition '{v}'. Allowed values: {allowed}")
        return v


class PredictionResponse(BaseModel):
    """Response schema for house price prediction."""

    predicted_price: float = Field(..., description="Predicted house price in USD")
    price_range_low: float = Field(
        ..., description="Approximate low bound based on test MAPE"
    )
    price_range_high: float = Field(
        ..., description="Approximate high bound based on test MAPE"
    )
    model_type: str = Field(..., description="Trained model architecture/type")
    model_version: str = Field(..., description="Model version or commit hash")
    fields_defaulted: int = Field(
        ..., description="Number of feature fields filled from defaults"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Out-of-range or domain warnings"
    )
    latency_ms: float = Field(..., description="Prediction inference latency in milliseconds")


class BatchRequest(BaseModel):
    """Batch prediction request containing 1-500 house records."""

    houses: list[PropertyInput] = Field(
        ..., min_length=1, max_length=500, description="List of property input records (1-500)"
    )


class BatchResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of prediction responses"
    )
