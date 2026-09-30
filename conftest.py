"""Global pytest configuration and early mocks."""

import sys
import types

# If Windows Application Control prevents loading _ssl.pyd DLL,
# mock _ssl early so ssl.py and dependent libraries (fastapi, requests, mlflow, etc.) import cleanly.
if "_ssl" not in sys.modules:
    try:
        import _ssl  # type: ignore
    except ImportError:

        class DummySSLError(Exception):
            pass

        class DummySSLContext:
            def __init__(self, protocol=0, *args, **kwargs):
                self.protocol = protocol
                self.options = 0
                self.verify_mode = 0

            def set_ciphers(self, ciphers):
                pass

            def load_verify_locations(self, *a, **kw):
                pass

            def set_default_verify_paths(self):
                pass

            def _set_npn_protocols(self, p):
                pass

            def _set_alpn_protocols(self, p):
                pass

            def wrap_socket(self, sock, *args, **kwargs):
                return sock

        mod = types.ModuleType("_ssl")
        attrs = {
            "_DEFAULT_CIPHERS": "DEFAULT:!aNULL:!eNULL:!MD5",
            "_OPENSSL_API_VERSION": 0x10101000,
            "OPENSSL_VERSION": "OpenSSL 1.1.1",
            "OPENSSL_VERSION_NUMBER": 0x10101000,
            "OPENSSL_VERSION_INFO": (1, 1, 1, 0, 0),
            "HAS_SNI": True,
            "HAS_ECDH": True,
            "HAS_NPN": True,
            "HAS_ALPN": True,
            "HAS_SSLv2": False,
            "HAS_SSLv3": False,
            "HAS_TLSv1": True,
            "HAS_TLSv1_1": True,
            "HAS_TLSv1_2": True,
            "HAS_TLSv1_3": True,
            "PROTOCOL_SSLv23": 2,
            "PROTOCOL_SSLv2": 0,
            "PROTOCOL_SSLv3": 1,
            "PROTOCOL_TLS": 2,
            "PROTOCOL_TLS_CLIENT": 16,
            "PROTOCOL_TLS_SERVER": 17,
            "PROTOCOL_TLSv1": 3,
            "PROTOCOL_TLSv1_1": 4,
            "PROTOCOL_TLSv1_2": 5,
            "OP_ALL": 0,
            "OP_NO_SSLv2": 0,
            "OP_NO_SSLv3": 0,
            "OP_NO_TLSv1": 0,
            "OP_NO_TLSv1_1": 0,
            "OP_NO_TLSv1_2": 0,
            "OP_NO_TLSv1_3": 0,
            "ALERT_DESCRIPTION_CLOSE_NOTIFY": 0,
            "SSL_ERROR_ZERO_RETURN": 0,
            "VERIFY_DEFAULT": 0,
            "CERT_NONE": 0,
            "CERT_OPTIONAL": 1,
            "CERT_REQUIRED": 2,
            "PROTO_MINIMUM_SUPPORTED": 0,
            "PROTO_SSLv3": 1,
            "PROTO_TLSv1": 3,
            "PROTO_TLSv1_1": 4,
            "PROTO_TLSv1_2": 5,
            "PROTO_TLSv1_3": 6,
            "PROTO_MAXIMUM_SUPPORTED": 6,
            "enum_certificates": staticmethod(lambda *a, **kw: []),
            "enum_crls": staticmethod(lambda *a, **kw: []),
            "SSLError": DummySSLError,
            "SSLZeroReturnError": DummySSLError,
            "SSLWantReadError": DummySSLError,
            "SSLWantWriteError": DummySSLError,
            "SSLSyscallError": DummySSLError,
            "SSLEOFError": DummySSLError,
            "SSLCertVerificationError": DummySSLError,
            "CertificateError": DummySSLError,
            "MemoryBIO": object,
            "_SSLContext": DummySSLContext,
            "_sslContext": DummySSLContext,
            "SSLSession": object,
            "SSLContext": DummySSLContext,
            "_ASN1Object": object,
            "txt2obj": staticmethod(lambda s, name=False: (1, "name", "long_name", s)),
            "nid2obj": staticmethod(lambda n: (n, "name", "long_name", "oid")),
            "RAND_status": staticmethod(lambda *a, **kw: 1),
            "RAND_add": staticmethod(lambda *a, **kw: None),
            "RAND_bytes": staticmethod(lambda n: b"\x00" * n),
            "RAND_pseudo_bytes": staticmethod(lambda n: (b"\x00" * n, True)),
        }
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules["_ssl"] = mod

import os
from pathlib import Path
import pytest
from lifinity.config import get_project_root


def pytest_runtest_setup(item: pytest.Item) -> None:
    ci_simulate = os.getenv("LIFINITY_CI_SIMULATE") == "1"
    root = get_project_root()

    if item.get_closest_marker("requires_data"):
        raw_csv = root / "data" / "raw" / "train.csv"
        proc_parquet = root / "data" / "processed" / "train.parquet"
        if ci_simulate or not (raw_csv.exists() or proc_parquet.exists()):
            pytest.skip("Test requires real data files (missing or LIFINITY_CI_SIMULATE=1)")

    if item.get_closest_marker("requires_model"):
        model_file = root / "models" / "model.joblib"
        if ci_simulate or not model_file.exists():
            pytest.skip("Test requires models/model.joblib (missing or LIFINITY_CI_SIMULATE=1)")

