import argparse
import csv
import logging
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as e:
    raise SystemExit("bench.py requires torch (ROCm build)") from e

from . import kernels as kk
from . import petit_baseline as petit
from . import ref as rr

log = logging.getLogger("mxfp4_cdna3.bench")


def _alloc_inputs(M, N, K, device):
    rng = np.random.default_rng(0)
    A = torch.empty(M, K, dtype=torch.bfloat16, device=device).normal_(0.0, 1.0)
    B_f32 = rng.standard_normal((K, N), dtype=np.float32).astype(np.float32)
    B_packed_np, B_scales_np = rr.quantize_mxfp4(B_f32, axis=0)
    B_packed = torch.from_numpy(B_packed_np).to(device)
    B_scales = torch.from_numpy(B_scales_np).to(device)
    C = torch.empty(M, N, dtype=torch.float32, device=device)
    return A, B_packed, B_scales, C


def _bench_one(launch_fn, A, B_packed, B_scales, C, n_warmup, n_iter, rot=8):
    device = A.device
    B_packed_rot = [B_packed.clone() for _ in range(rot)]
    B_scales_rot = [B_scales.clone() for _ in range(rot)]

    torch.cuda.synchronize(device)
    for _ in range(n_warmup):
        launch_fn(A, B_packed_rot[0], B_scales_rot[0], C)
    torch.cuda.synchronize(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        idx = i % rot
        starts[i].record()
        launch_fn(A, B_packed_rot[idx], B_scales_rot[idx], C)
        ends[i].record()
    torch.cuda.synchronize(device)
    return [s.elapsed_time(e) * 1e6 for s, e in zip(starts, ends, strict=True)]


def _strategies_for(M, N, K):
    out = {
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
    return {k: v for k, v in out.items() if v is not None}


def run_sweep(shapes, out_csv, n_warmup=50, n_iter=100):
    device = torch.device("cuda:0")
    rows = []
    for M, N, K in shapes:
        log.info("shape M=%d N=%d K=%d", M, N, K)
        A, B_packed, B_scales, C = _alloc_inputs(M, N, K, device)
        strategies = _strategies_for(M, N, K)
        if kk.shape_supported_g(M, N, K):
            Bp_prep_np, Bs_prep_np = rr.prepack_b_mxfp4(
                B_packed.cpu().numpy(), B_scales.cpu().numpy()
            )
            Bp_prep = torch.from_numpy(Bp_prep_np).to(device)
            Bs_prep = torch.from_numpy(Bs_prep_np).to(device)

            def _g(A_, _Bp, _Bs, C_, _bp=Bp_prep, _bs=Bs_prep):
                return kk.gemm_g(A_, _bp, _bs, C_)

            strategies["G"] = _g

        if petit.is_available():
            try:
                pre = petit.PetitPreshuffle(B_packed, B_scales, N, K)

                def _petit(A_, _Bp, _Bs, C_, _pre=pre):
                    return petit.gemm_petit_mxfp4(A_, _pre, C_)

                strategies["PETIT"] = _petit
            except Exception as ex:
                log.warning("petit preshuffle skipped: %s", ex)

        for label, fn in strategies.items():
            try:
                ts = _bench_one(fn, A, B_packed, B_scales, C, n_warmup=n_warmup, n_iter=n_iter)
            except RuntimeError as ex:
                log.error("strategy %s FAILED: %s", label, ex)
                continue
            ts_sorted = sorted(ts)
            p50 = ts_sorted[len(ts) // 2]
            p95 = ts_sorted[int(len(ts) * 0.95)]
            tflops = (2.0 * M * N * K) / p50 / 1e3
            log.info("  %s: p50=%.1fus p95=%.1fus %.1fTFLOPS", label, p50 / 1e3, p95 / 1e3, tflops)
            for i, t in enumerate(ts):
                rows.append({"strategy": label, "M": M, "N": N, "K": K, "iter": i, "ns": t})

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "M", "N", "K", "iter", "ns"])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d samples)", out_csv, len(rows))


def plot_violins(csv_path, out_png):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["shape"] = df.apply(lambda r: f"{r.M}x{r.N}x{r.K}", axis=1)
    df["tflops"] = (2.0 * df["M"] * df["N"] * df["K"]) / df["ns"] / 1e3

    shapes = df["shape"].unique().tolist()
    strategies = sorted(df["strategy"].unique().tolist())
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(shapes)), 5))
    width = 0.8 / max(1, len(strategies))
    palette = {"A": "#888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
    for si, strat in enumerate(strategies):
        positions = [i + (si - (len(strategies) - 1) / 2) * width for i in range(len(shapes))]
        data = [df[(df["strategy"] == strat) & (df["shape"] == s)]["tflops"].values for s in shapes]
        parts = ax.violinplot(data, positions=positions, widths=width * 0.9, showmedians=True)
        color = palette.get(strat)
        for body in parts["bodies"]:
            if color:
                body.set_facecolor(color)
            body.set_alpha(0.7)
        ax.scatter([], [], color=color or "gray", label=f"Strategy {strat}")
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, rotation=20, ha="right")
    ax.set_ylabel("TFLOPS")
    ax.set_title("MXFP4 GEMM throughput (violins over 100 iters)")
    ax.legend()
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    log.info("wrote %s", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes",
        nargs="*",
        default=[
            "1024,11008,4096",
            "2048,11008,4096",
            "4096,11008,4096",
            "1024,4096,11008",
            "2048,4096,11008",
        ],
    )
    ap.add_argument("--out", type=str, default="results/bench.csv")
    ap.add_argument("--plot", type=str, default="results/bench_violins.png")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iter", type=int, default=100)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    run_sweep(shapes, args.out, n_warmup=args.warmup, n_iter=args.iter)
    plot_violins(args.out, args.plot)


if __name__ == "__main__":
    main()
