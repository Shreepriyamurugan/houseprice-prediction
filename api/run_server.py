"""Server launcher with early _ssl fallback mock for blocked environment."""

import sys
import types

# Apply _ssl mock BEFORE any library (uvicorn, starlette, etc.) imports ssl
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

import uvicorn
sys.path.insert(0, r"C:\projects\lifinity")
from api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
