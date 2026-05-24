from .ref import dequantize_mxfp4, preshuffle_b, quantize_mxfp4, ref_gemm

__version__ = "0.1.0"
__all__ = [
    "dequantize_mxfp4",
    "device_info",
    "gemm_a",
    "gemm_b",
    "gemm_c",
    "preshuffle_b",
    "quantize_mxfp4",
    "ref_gemm",
    "shape_supported_c",
]


def __getattr__(name):
    if name in ("gemm_a", "gemm_b", "gemm_c", "device_info", "shape_supported_c"):
        from . import kernels

        return getattr(kernels, name)
    raise AttributeError(name)
