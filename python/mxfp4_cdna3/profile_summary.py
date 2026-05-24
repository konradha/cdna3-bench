import argparse
import contextlib
import csv
import logging
import statistics
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("mxfp4_cdna3.profile_summary")

# Counter names that aren't per-dispatch metadata; sums across dispatches.
_PMC_NAMES = {
    "SQ_INSTS_VALU",
    "SQ_INSTS_MFMA",
    "SQ_INSTS_LDS",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "GRBM_GUI_ACTIVE",
    "SQ_WAVES",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    "TCP_TCC_READ_REQ_sum",
    "TCP_TCC_WRITE_REQ_sum",
}


def _short(name: str) -> str:
    if not name:
        return "?"
    return name.split("::")[-1].split("(")[0].split("<")[0].strip().strip('"')


def _parse_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_trace(path: Path) -> dict[str, list[int]]:
    """Handles rocprofv3 kernel_trace.csv and rocprof v1 trace.csv."""
    durs: dict[str, list[int]] = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # v3: Kernel_Name + Start_Timestamp/End_Timestamp; v1: KernelName + BeginNs/EndNs
            name = r.get("Kernel_Name") or r.get("KernelName") or r.get("kernel_name")
            s = _parse_int(r.get("Start_Timestamp") or r.get("BeginNs"))
            e = _parse_int(r.get("End_Timestamp") or r.get("EndNs"))
            if name is None or s is None or e is None:
                continue
            durs[_short(name)].append(e - s)
    return durs


def parse_pmc(path: Path) -> dict[str, float]:
    """v3 counter_collection.csv (long) or v1 pmc.csv (wide)."""
    counters: dict[str, float] = defaultdict(float)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = rows[0].keys()
    if "Counter_Name" in cols or "counter_name" in cols:
        for r in rows:
            name = r.get("Counter_Name") or r.get("counter_name")
            val = r.get("Counter_Value") or r.get("counter_value")
            if not name or val is None:
                continue
            with contextlib.suppress(ValueError):
                counters[name] += float(val)
        return dict(counters)
    # Wide format (v1): pick counter columns that match known names, sum across rows.
    for r in rows:
        for k, v in r.items():
            if k in _PMC_NAMES and v not in (None, ""):
                with contextlib.suppress(ValueError):
                    counters[k] += float(v)
    return dict(counters)


def fmt_us(ns: float) -> str:
    return f"{ns / 1000:.1f}us"


def summarize_run(run_dir: Path) -> None:
    trace = sorted(
        list(run_dir.glob("trace/**/*kernel_trace.csv")) + list(run_dir.glob("trace.csv"))
    )
    stats = sorted(run_dir.glob("trace.stats.csv"))
    pmc = sorted(
        list(run_dir.glob("pmc/**/*counter_collection.csv")) + list(run_dir.glob("pmc.csv"))
    )

    durs: dict[str, list[int]] = defaultdict(list)
    for c in trace:
        for k, v in parse_trace(c).items():
            durs[k].extend(v)
    # If we have nothing per-dispatch, fall back to v1's aggregated stats (Name, Calls, TotalDurationNs, AverageNs).
    if not durs:
        for c in stats:
            with open(c) as f:
                for r in csv.DictReader(f):
                    name = _short(r.get("Name") or r.get("KernelName") or "")
                    calls = _parse_int(r.get("Calls")) or 0
                    total = _parse_int(r.get("TotalDurationNs")) or 0
                    avg = _parse_int(r.get("AverageNs")) or (total // calls if calls else 0)
                    if name and calls > 0:
                        durs[name].extend([avg] * calls)

    print(f"\n=== {run_dir.name} ===")
    if not durs:
        print("  (no kernel trace)")
        return
    total_wall = sum(sum(v) for v in durs.values())
    for k, v in sorted(durs.items(), key=lambda kv: -sum(kv[1])):
        p50 = statistics.median(v)
        total = sum(v)
        share = 100.0 * total / total_wall if total_wall else 0
        print(
            f"  {k:42s} n={len(v):4d}  p50={fmt_us(p50):>10}  "
            f"total={fmt_us(total):>10}  {share:5.1f}%"
        )

    if not pmc:
        return
    counters: dict[str, float] = defaultdict(float)
    for c in pmc:
        for k, v in parse_pmc(c).items():
            counters[k] += v
    if not counters:
        return

    mfma = counters.get("SQ_INSTS_MFMA", 0)
    valu = counters.get("SQ_INSTS_VALU", 0)
    lds = counters.get("SQ_INSTS_LDS", 0)
    vmem = counters.get("SQ_INSTS_VMEM", 0)
    salu = counters.get("SQ_INSTS_SALU", 0)
    active = counters.get("GRBM_GUI_ACTIVE", 0)
    waves = counters.get("SQ_WAVES", 0)
    tcc_hit = counters.get("TCC_HIT_sum", 0)
    tcc_miss = counters.get("TCC_MISS_sum", 0)
    tcp_rd = counters.get("TCP_TCC_READ_REQ_sum", 0)
    tcp_wr = counters.get("TCP_TCC_WRITE_REQ_sum", 0)

    print("  ---- PMC ----")
    if active and mfma:
        # gfx942: 16 cycles per v_mfma_f32_16x16x16_bf16_1k. mfma_issue_eff = MFMA*16 / active.
        # 1.0 = MFMA pipe saturated; <0.5 = stall-bound (deps on acc, sync, LDS).
        print(f"  mfma_issue_eff (MFMA*16 / GRBM_GUI_ACTIVE): {mfma * 16 / active:.3f}")
    if mfma:
        print(
            f"  valu/mfma: {valu / mfma:6.2f}   lds/mfma: {lds / mfma:6.2f}   "
            f"vmem/mfma: {vmem / mfma:6.2f}   salu/mfma: {salu / mfma:6.2f}"
        )
    if tcc_hit + tcc_miss:
        print(
            f"  L2 hit rate: {tcc_hit / (tcc_hit + tcc_miss):.3f} "
            f"(hits={tcc_hit:.0f} misses={tcc_miss:.0f})"
        )
    if tcp_rd + tcp_wr:
        print(f"  TCP->TCC traffic: rd={tcp_rd:.0f} wr={tcp_wr:.0f}")
    if waves:
        print(f"  SQ_WAVES: {waves:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="prof_out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(args.root)
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    if not runs:
        log.warning("no runs found under %s", root)
        return
    for r in runs:
        summarize_run(r)


if __name__ == "__main__":
    main()
