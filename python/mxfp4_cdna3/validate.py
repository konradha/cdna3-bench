import argparse
import logging

import numpy as np

try:
    import torch
except ImportError as e:
    raise SystemExit("validate.py requires torch (ROCm build)") from e

from . import kernels as kk
from . import petit_baseline as petit
from . import ref as rr

log = logging.getLogger("mxfp4_cdna3.validate")


def _max_rel_err(out, ref, eps=1e-6):
    return float(np.max(np.abs(out - ref) / np.maximum(np.abs(ref), eps)))


def validate_shape(M, N, K, atol=1e-2, rtol=2e-2):
    device = torch.device("cuda:0")
    rng = np.random.default_rng(42)
    A_f32 = rng.standard_normal((M, K), dtype=np.float32) * 0.1
    A = torch.from_numpy(A_f32).to(torch.bfloat16).to(device)

    B_f32 = rng.standard_normal((K, N), dtype=np.float32) * 0.1
    B_packed_np, B_scales_np = rr.quantize_mxfp4(B_f32, axis=0)
    B_packed = torch.from_numpy(B_packed_np).to(device)
    B_scales = torch.from_numpy(B_scales_np).to(device)

    A_back = A.cpu().to(torch.float32).numpy()
    C_ref = rr.ref_gemm(A_back, B_packed_np, B_scales_np)

    strategies = {
        "A": kk.gemm_a,
        "B": kk.gemm_b,
        "C": kk.gemm_c if kk.shape_supported_c(M, N, K) else None,
        "D": kk.gemm_d if kk.shape_supported_d(M, N, K) else None,
        "E": kk.gemm_e if kk.shape_supported_e(M, N, K) else None,
        "F": kk.gemm_f if kk.shape_supported_f(M, N, K) else None,
        "H": kk.gemm_h if kk.shape_supported_d(M, N, K) else None,
        "I": kk.gemm_i if kk.shape_supported_d(M, N, K) else None,
        "K": kk.gemm_k if kk.shape_supported_d(M, N, K) else None,
        "J": kk.gemm_j if kk.shape_supported_d(M, N, K) else None,
        "M": kk.gemm_m if kk.shape_supported_d(M, N, K) else None,
    }
    strategies = {k: v for k, v in strategies.items() if v is not None}

    log.info("shape M=%d N=%d K=%d", M, N, K)
    all_ok = True
    for label, fn in strategies.items():
        try:
            C_out = fn(A, B_packed, B_scales).cpu().numpy()
        except RuntimeError as ex:
            log.error("  %s: FAILED %s", label, ex)
            all_ok = False
            continue
        max_abs = float(np.max(np.abs(C_out - C_ref)))
        max_rel = _max_rel_err(C_out, C_ref)
        ok = np.allclose(C_out, C_ref, atol=atol, rtol=rtol)
        log.info("  %s: max_abs=%.4e max_rel=%.4e pass=%s", label, max_abs, max_rel, ok)
        all_ok = all_ok and ok

    if kk.shape_supported_g(M, N, K):
        try:
            Bp_prep_np, Bs_prep_np = rr.prepack_b_mxfp4(B_packed_np, B_scales_np)
            Bp_prep = torch.from_numpy(Bp_prep_np).to(device)
            Bs_prep = torch.from_numpy(Bs_prep_np).to(device)
            C_out = kk.gemm_g(A, Bp_prep, Bs_prep).cpu().numpy()
            max_abs = float(np.max(np.abs(C_out - C_ref)))
            max_rel = _max_rel_err(C_out, C_ref)
            ok = np.allclose(C_out, C_ref, atol=atol, rtol=rtol)
            log.info("  G: max_abs=%.4e max_rel=%.4e pass=%s", max_abs, max_rel, ok)
            all_ok = all_ok and ok
        except RuntimeError as ex:
            log.error("  G: FAILED %s", ex)
            all_ok = False

    if petit.is_available():
        try:
            pre = petit.PetitPreshuffle(B_packed, B_scales, N, K)
            C_out = petit.gemm_petit_mxfp4(A, pre).to(torch.float32).cpu().numpy()
            max_abs = float(np.max(np.abs(C_out - C_ref)))
            max_rel = _max_rel_err(C_out, C_ref)
            ok = np.allclose(C_out, C_ref, atol=atol, rtol=rtol)
            log.info("  PETIT: max_abs=%.4e max_rel=%.4e pass=%s", max_abs, max_rel, ok)
            all_ok = all_ok and ok
        except Exception as ex:
            log.warning("  PETIT: SKIP %s", ex)
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=["256,256,128", "512,512,256", "1024,1024,256"])
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    ok_all = True
    for s in args.shapes:
        m, n, k = (int(x) for x in s.split(","))
        ok_all = validate_shape(m, n, k) and ok_all
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
