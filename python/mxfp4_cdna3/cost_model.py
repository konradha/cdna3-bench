import argparse
import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("mxfp4_cdna3.cost_model")


@dataclass(frozen=True)
class HwProfile:
    name: str
    num_cus: int
    num_xcds: int
    clock_ghz: float
    hbm_bw_tbps: float
    lds_bw_per_cu_gbps: float
    valu_lanes_per_cu: int = 64
    mfma_cycles_bf16_16x16x16: int = 16

    @property
    def mfma_peak_tflops(self) -> float:
        flops_per_mfma = 2 * 16 * 16 * 16
        per_cu_per_cycle = flops_per_mfma / self.mfma_cycles_bf16_16x16x16
        return per_cu_per_cycle * self.num_cus * self.clock_ghz * 1e9 / 1e12

    @property
    def valu_peak_gops(self) -> float:
        return self.valu_lanes_per_cu * self.num_cus * self.clock_ghz

    @property
    def lds_bw_aggregate_tbps(self) -> float:
        return self.lds_bw_per_cu_gbps * self.num_cus / 1000.0


MI300A = HwProfile("MI300A", 228, 6, 2.1, 5.3, 1075.0)
MI300X = HwProfile("MI300X", 304, 8, 2.1, 5.3, 1075.0)


@dataclass(frozen=True)
class ProblemShape:
    M: int
    N: int
    K: int

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    def hbm_bytes(self, extra_b_passes: int = 0) -> int:
        # A bf16, packed B (FP4 = 0.5 bytes/elt), B scales (1 byte / 32 elts), C fp32.
        a = self.M * self.K * 2
        bp = (self.K // 2) * self.N
        s = (self.K // 32) * self.N
        c = self.M * self.N * 4
        extra = extra_b_passes * self.K * self.N * 2  # bf16 scratch traffic for strategy A
        return a + bp + s + c + extra

    @property
    def n_mfma_16x16x16(self) -> int:
        return (self.M // 16) * (self.N // 16) * (self.K // 16)

    @property
    def dequant_calls(self) -> int:
        return (self.K * self.N) // 8


@dataclass(frozen=True)
class StrategyDesc:
    name: str
    block_m: int
    block_n: int
    block_k: int
    lds_kib_per_cta: int
    extra_b_hbm_passes: int  # A: writes + reads bf16 scratch B (= 2 extra passes of K*N*2)
    dequant_overlap: float  # 0..1 fraction of dequant_ns hidden behind compute
    achievable_eff: float  # 0..1 of mfma_peak_tflops actually sustained


STRATEGIES = (
    StrategyDesc(
        "A", 64, 128, 64, 48, extra_b_hbm_passes=2, dequant_overlap=0.0, achievable_eff=0.10
    ),
    StrategyDesc(
        "B", 64, 128, 64, 48, extra_b_hbm_passes=0, dequant_overlap=0.40, achievable_eff=0.15
    ),
    StrategyDesc(
        "C", 64, 128, 64, 48, extra_b_hbm_passes=0, dequant_overlap=0.50, achievable_eff=0.15
    ),
    StrategyDesc(
        "D", 128, 128, 64, 64, extra_b_hbm_passes=0, dequant_overlap=0.50, achievable_eff=0.20
    ),
)
STRATEGY_BY_NAME = {s.name: s for s in STRATEGIES}


@dataclass
class Prediction:
    strategy: str
    hw: str
    shape: ProblemShape
    compute_ns: float
    hbm_ns: float
    lds_ns: float
    dequant_ns_total: float
    dequant_ns_unhidden: float
    binding: str
    predicted_ns: float
    predicted_tflops: float

    def to_row(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "shape"}
        d.update({"M": self.shape.M, "N": self.shape.N, "K": self.shape.K})
        return d


def predict(strategy: StrategyDesc, shape: ProblemShape, hw: HwProfile) -> Prediction:
    compute_ns = shape.flops / (hw.mfma_peak_tflops * strategy.achievable_eff * 1e12) * 1e9
    hbm_ns = shape.hbm_bytes(strategy.extra_b_hbm_passes) / (hw.hbm_bw_tbps * 1e12) * 1e9
    # B/C/D read each MFMA operand from LDS: 1024 bytes/MFMA (A operand 512 + B operand 512).
    lds_bytes = shape.n_mfma_16x16x16 * 1024
    lds_ns = lds_bytes / (hw.lds_bw_aggregate_tbps * 1e12) * 1e9

    # Dequant VALU: 22 ops per 8 FP4 elements (LUT perm + cvt + scale + bf16 RNE).
    dequant_ops = shape.dequant_calls * 22
    dequant_ns_total = dequant_ops / (hw.valu_peak_gops * 1e9) * 1e9
    dequant_ns_unhidden = dequant_ns_total * (1.0 - strategy.dequant_overlap)

    compute_with_dequant = compute_ns + dequant_ns_unhidden
    candidates = {"compute": compute_with_dequant, "hbm": hbm_ns, "lds": lds_ns}
    binding = max(candidates, key=candidates.get)
    predicted_ns = candidates[binding]
    tflops = shape.flops / (predicted_ns * 1e3)
    return Prediction(
        strategy=strategy.name,
        hw=hw.name,
        shape=shape,
        compute_ns=compute_ns,
        hbm_ns=hbm_ns,
        lds_ns=lds_ns,
        dequant_ns_total=dequant_ns_total,
        dequant_ns_unhidden=dequant_ns_unhidden,
        binding=binding,
        predicted_ns=predicted_ns,
        predicted_tflops=tflops,
    )


def predict_all(shapes, hw: HwProfile = MI300A, strategies=STRATEGIES):
    return [predict(s, sh, hw) for sh in shapes for s in strategies]


def fit_efficiency(bench_csv: str, hw: HwProfile = MI300A) -> dict[str, float]:
    """Invert measured p50 -> achievable_eff per strategy. Median across shapes."""
    import pandas as pd

    df = pd.read_csv(bench_csv)
    p50 = df.groupby(["strategy", "M", "N", "K"])["ns"].median().reset_index()
    out = {}
    for strat, group in p50.groupby("strategy"):
        ratios = []
        for _, row in group.iterrows():
            shape = ProblemShape(int(row.M), int(row.N), int(row.K))
            peak_ns = shape.flops / (hw.mfma_peak_tflops * 1e12) * 1e9
            ratios.append(peak_ns / row.ns)
        out[strat] = float(sorted(ratios)[len(ratios) // 2])  # median
    return out


def write_predictions_csv(rows: list[Prediction], path: str):
    fields = list(rows[0].to_row().keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_row())


def plot_predictions_vs_measured(predictions_csv: str, measured_csv: str, out_png: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    pred = pd.read_csv(predictions_csv)
    meas = pd.read_csv(measured_csv)
    if "iter" in meas.columns:
        meas = (
            meas.groupby(["strategy", "M", "N", "K"])["ns"]
            .median()
            .reset_index()
            .rename(columns={"ns": "measured_ns"})
        )
    df = pred.merge(meas, on=["M", "N", "K", "strategy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    palette = {"A": "#888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
    for strat in sorted(df["strategy"].unique()):
        sub = df[df["strategy"] == strat]
        ax1.scatter(
            sub["predicted_ns"],
            sub["measured_ns"],
            color=palette.get(strat, "k"),
            label=f"Strategy {strat}",
            s=60,
        )
    lim_lo = min(df["predicted_ns"].min(), df["measured_ns"].min())
    lim_hi = max(df["predicted_ns"].max(), df["measured_ns"].max())
    ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", c="black", alpha=0.4, label="y=x")
    ax1.set_xlabel("predicted ns")
    ax1.set_ylabel("measured ns (p50)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Predicted vs measured")
    ax1.legend()

    # Stacked breakdown: which term binds per shape per strategy.
    df["shape"] = df.apply(lambda r: f"{r.M}x{r.N}x{r.K}", axis=1)
    binding_counts = df.groupby(["strategy", "binding"]).size().unstack(fill_value=0)
    binding_counts.plot.bar(ax=ax2, stacked=True, colormap="viridis")
    ax2.set_title("Binding constraint by strategy (across shapes)")
    ax2.set_ylabel("# shapes")
    ax2.tick_params(axis="x", rotation=0)

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
    ap.add_argument("--hw", choices=["MI300A", "MI300X"], default="MI300A")
    ap.add_argument("--bench", type=str, default=None, help="bench CSV; if set, refit efficiencies")
    ap.add_argument("--out", type=str, default="results/cost_predictions.csv")
    ap.add_argument("--plot", type=str, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    hw = {"MI300A": MI300A, "MI300X": MI300X}[args.hw]
    shapes = [ProblemShape(*(int(x) for x in s.split(","))) for s in args.shapes]

    strategies = STRATEGIES
    if args.bench:
        eff = fit_efficiency(args.bench, hw)
        log.info("fit_efficiency: %s", eff)
        strategies = tuple(
            StrategyDesc(
                s.name,
                s.block_m,
                s.block_n,
                s.block_k,
                s.lds_kib_per_cta,
                s.extra_b_hbm_passes,
                s.dequant_overlap,
                achievable_eff=eff.get(s.name, s.achievable_eff),
            )
            for s in STRATEGIES
        )

    rows = [predict(s, sh, hw) for sh in shapes for s in strategies]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(rows, str(out_path))
    log.info(
        "wrote %s (%d rows, hw=%s, peak=%.0f TF)", out_path, len(rows), hw.name, hw.mfma_peak_tflops
    )

    if args.plot and args.bench:
        plot_predictions_vs_measured(str(out_path), args.bench, args.plot)


if __name__ == "__main__":
    main()
