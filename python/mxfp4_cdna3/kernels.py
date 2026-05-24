import torch  # noqa: F401  preloads libc10/libtorch_hip so _C.so's DT_NEEDED resolves

try:
    from . import _C as _ext
except ImportError as e:
    raise RuntimeError("mxfp4_cdna3._C not built; run `pip install -e .`") from e

from .ref import prepack_b_mxfp4  # noqa: F401  re-export for callers


def device_info(dev_id=0):
    return _ext.device_info(dev_id)


def shape_supported_c(M, N, K):
    return _ext.shape_supported_c(M, N, K)


def shape_supported_d(M, N, K):
    return _ext.shape_supported_d(M, N, K)


def shape_supported_e(M, N, K):
    return _ext.shape_supported_e(M, N, K)


def shape_supported_f(M, N, K):
    return _ext.shape_supported_f(M, N, K)


def shape_supported_g(M, N, K):
    return _ext.shape_supported_g(M, N, K)


def gemm_a(A, B_packed, B_scales, C=None):
    return _ext.gemm_a(A, B_packed, B_scales, C)


def gemm_b(A, B_packed, B_scales, C=None):
    return _ext.gemm_b(A, B_packed, B_scales, C)


def gemm_c(A, B_packed, B_scales, C=None):
    return _ext.gemm_c(A, B_packed, B_scales, C)


def gemm_d(A, B_packed, B_scales, C=None):
    return _ext.gemm_d(A, B_packed, B_scales, C)


def gemm_e(A, B_packed, B_scales, C=None):
    return _ext.gemm_e(A, B_packed, B_scales, C)


def gemm_f(A, B_packed, B_scales, C=None):
    return _ext.gemm_f(A, B_packed, B_scales, C)


def gemm_g(A, B_prep, Bs_prep, C=None):
    # B_prep / Bs_prep must come from prepack_b_mxfp4 (raw [K/2, N] does NOT work).
    return _ext.gemm_g(A, B_prep, Bs_prep, C)


def gemm_auto(A, B_packed, B_scales, C=None):
    # Shape-aware dispatch. J is the production synthesis (SW-scaled MXFP4 + LDS swizzle
    # + sched_group_barrier choreography); falls back to F for shapes only F covers,
    # then E for low-CTA grids, then C/B as last resort.
    M, K = A.shape
    N = B_packed.shape[1]
    if shape_supported_d(M, N, K):
        return gemm_j(A, B_packed, B_scales, C)
    if shape_supported_f(M, N, K):
        return gemm_f(A, B_packed, B_scales, C)
    if shape_supported_e(M, N, K):
        return gemm_e(A, B_packed, B_scales, C)
    if shape_supported_c(M, N, K):
        return gemm_c(A, B_packed, B_scales, C)
    return gemm_b(A, B_packed, B_scales, C)


def gemm_h(A, B_packed, B_scales, C=None):
    return _ext.gemm_h(A, B_packed, B_scales, C)


def gemm_i(A, B_packed, B_scales, C=None):
    return _ext.gemm_i(A, B_packed, B_scales, C)


def gemm_k(A, B_packed, B_scales, C=None):
    return _ext.gemm_k(A, B_packed, B_scales, C)


def gemm_j(A, B_packed, B_scales, C=None):
    return _ext.gemm_j(A, B_packed, B_scales, C)


def gemm_m(A, B_packed, B_scales, C=None):
    return _ext.gemm_m(A, B_packed, B_scales, C)
