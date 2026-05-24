import numpy as np
import pytest
from mxfp4_cdna3 import ref as rr


def test_quant_dequant_roundtrip_zero():
    x = np.zeros((4, 32), dtype=np.float32)
    p, s = rr.quantize_mxfp4(x, axis=-1)
    y = rr.dequantize_mxfp4(p, s, axis=-1)
    assert np.allclose(y, x)


def test_quant_dequant_small_values():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 64), dtype=np.float32)
    p, s = rr.quantize_mxfp4(x, axis=-1)
    y = rr.dequantize_mxfp4(p, s, axis=-1)
    assert p.shape == (8, 32)
    assert s.shape == (8, 2)
    assert (np.abs(y - x) / (np.abs(x) + 1e-6)).mean() < 0.30


def test_quant_packing_bit_layout():
    x = np.array([[4.0, -4.0] * 16], dtype=np.float32)
    p, s = rr.quantize_mxfp4(x, axis=-1)
    assert s[0, 0] == 127
    assert p[0, 0] == 0xE6
    assert (p[0] == 0xE6).all()


def test_ref_gemm_identity_like():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((16, 32), dtype=np.float32)
    B = rng.standard_normal((32, 16), dtype=np.float32) * 0.1
    p, s = rr.quantize_mxfp4(B, axis=0)
    C = rr.ref_gemm(A, p, s)
    assert C.shape == (16, 16)
    assert np.isfinite(C).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
