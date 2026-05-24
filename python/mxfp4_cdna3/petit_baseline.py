"""Petit (CDNA3 MXFP4 GEMM, https://github.com/causalflow-ai/petit-kernel) baseline.

Petit's API takes a transposed B layout (`[N, K/2]` packed FP4, scales `[N, K/32]`) and a
NVFP4-style `global_scale`. For MXFP4 the global_scale is identity (1.0). Inputs are repacked
once via `repack_mxfp4` / `process_mxfp4_scales` (offline shuffle, Marlin-style).

Petit is optional. Install with:
    CMAKE_ARGS='-DCMAKE_PREFIX_PATH=/opt/rocm-7.2.1;<torch_dir>' \\
      pip install ./third_party/petit-kernel
"""

import logging

import torch

log = logging.getLogger("mxfp4_cdna3.petit_baseline")

try:
    import petit_kernel  # type: ignore

    _HAS_PETIT = True
except ImportError:
    petit_kernel = None
    _HAS_PETIT = False


def is_available() -> bool:
    return _HAS_PETIT


class PetitPreshuffle:
    """One-shot offline weight shuffle for the bench/validate timing loop."""

    def __init__(self, B_packed_kn: torch.Tensor, B_scales_kn: torch.Tensor, N: int, K: int):
        if not _HAS_PETIT:
            raise RuntimeError("petit_kernel not installed")
        # Our layout: B_packed [K/2, N] uint8, B_scales [K/32, N] uint8.
        # Petit's layout: [N, K/2] uint8 (viewed as int32 in groups of 4 bytes), [N, K/32].
        b_nk_u8 = B_packed_kn.t().contiguous()  # [N, K/2]
        s_nk_u8 = B_scales_kn.t().contiguous()  # [N, K/32]
        self.b_repacked = petit_kernel.repack_mxfp4(b_nk_u8.view(torch.int32), N, K)
        self.s_processed = petit_kernel.process_mxfp4_scales(s_nk_u8, N, K)
        self.global_scale = torch.ones(1, dtype=torch.float32, device=B_packed_kn.device)
        self.N = N
        self.K = K


def gemm_petit_mxfp4(
    A: torch.Tensor,
    pre: PetitPreshuffle,
    C: torch.Tensor | None = None,
    solution_id: int = -1,
) -> torch.Tensor:
    """A [M, K] bf16, pre = PetitPreshuffle(B). Returns bf16 [M, N] (Petit's native output)."""
    if not _HAS_PETIT:
        raise RuntimeError("petit_kernel not installed")
    M = A.size(0)
    out_bf16 = petit_kernel.mul_mxfp4_a16(
        A,
        pre.b_repacked,
        pre.s_processed,
        pre.global_scale,
        M,
        pre.N,
        pre.K,
        solution_id,
    )
    if C is not None:
        # Caller wants fp32 output for our bench/ref compatibility; cast in-place.
        C.copy_(out_bf16.to(torch.float32))
        return C
    return out_bf16
