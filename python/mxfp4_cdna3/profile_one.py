import argparse
import logging

import numpy as np
import torch

from . import kernels as kk
from . import ref as rr

log = logging.getLogger("mxfp4_cdna3.profile_one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        choices=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "m"],
        required=True,
    )
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    device = torch.device("cuda:0")
    rng = np.random.default_rng(args.seed)
    A = (
        torch.from_numpy(rng.standard_normal((args.M, args.K), dtype=np.float32) * 0.1)
        .to(torch.bfloat16)
        .to(device)
    )
    B_f32 = rng.standard_normal((args.K, args.N), dtype=np.float32) * 0.1
    B_packed_np, B_scales_np = rr.quantize_mxfp4(B_f32, axis=0)
    C = torch.empty((args.M, args.N), dtype=torch.float32, device=device)

    if args.strategy == "g":
        Bp_np, Bs_np = rr.prepack_b_mxfp4(B_packed_np, B_scales_np)
        Bp = torch.from_numpy(Bp_np).to(device)
        Bs = torch.from_numpy(Bs_np).to(device)

        def call():
            kk.gemm_g(A, Bp, Bs, C)
    else:
        Bp = torch.from_numpy(B_packed_np).to(device)
        Bs = torch.from_numpy(B_scales_np).to(device)
        fn = {
            "a": kk.gemm_a,
            "b": kk.gemm_b,
            "c": kk.gemm_c,
            "d": kk.gemm_d,
            "e": kk.gemm_e,
            "f": kk.gemm_f,
            "h": kk.gemm_h,
            "i": kk.gemm_i,
            "k": kk.gemm_k,
            "j": kk.gemm_j,
            "m": kk.gemm_m,
        }[args.strategy]

        def call():
            fn(A, Bp, Bs, C)

    for _ in range(args.iters):
        call()
    torch.cuda.synchronize()
    log.info("%s M=%d N=%d K=%d iters=%d", args.strategy, args.M, args.N, args.K, args.iters)


if __name__ == "__main__":
    main()
