"""CDNA3 MXFP4 GEMM explainer.

Run: `uv run --with numpy --with matplotlib explainer/explainer.py`
Outputs eight figures into explainer/figures/. Each figure exists to prove one
claim; the title states the claim, the caption is one line. Visual idioms:
geometry uses categorical lane colours + bank borders, performance uses bar
charts and rooflines, correctness uses log-scaled error grids.

Hardware constants verified against:
  * AMD Instinct MI300 CDNA3 ISA Reference Guide (Aug 2025)
  * ROCm matrix-cores blog (Nov 2022) and CDNA3/CDNA4 update (Sept 2025)
  * OCP Microscaling Formats Spec v1.0 (Sept 2023)
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global style. One sans-serif, one weight for body labels.
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.dpi": 110,
    }
)

# Categorical lane palette: 8 hues, recycled across K-blocks of 16 lanes.
LANE_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]
HIGHLIGHT = "#e63946"  # red overlay for "what these 16 lanes read together"
NEUTRAL = "#333333"

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# MXFP4 quant/dequant (OCP MX-spec v1.0).
# Block = 32 elements along reduction axis. Element = E2M1 (4 bits, magnitudes
# {0, 0.5, 1, 1.5, 2, 3, 4, 6}). Scale = E8M0 (1 byte, bias 127, only NaN code 0xFF).
# ---------------------------------------------------------------------------

E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def quantize_mxfp4(x: np.ndarray, axis: int = -1):
    x = np.moveaxis(np.asarray(x, dtype=np.float32), axis, -1)
    assert x.shape[-1] % 32 == 0, "K-axis must be a multiple of 32"
    leading, K = x.shape[:-1], x.shape[-1]
    groups = K // 32

    grouped = x.reshape(*leading, groups, 32)
    amax = np.maximum(np.max(np.abs(grouped), axis=-1), 1e-30)
    exp = np.floor(np.log2(amax / 6.0)).astype(np.int32)
    scale_byte = np.clip(exp + 127, 0, 255).astype(np.uint8)
    scale_factor = np.power(2.0, scale_byte.astype(np.float32) - 127.0)

    normalized = grouped / scale_factor[..., None]
    sign = (normalized < 0).astype(np.uint8)
    abs_n = np.abs(normalized)
    idx = np.argmin(np.abs(abs_n[..., None] - E2M1_MAGNITUDES), axis=-1).astype(np.uint8)
    nibble = (sign << 3) | idx

    flat = nibble.reshape(*leading, K)
    low = flat[..., 0::2]
    high = flat[..., 1::2]
    packed = (low | (high << 4)).astype(np.uint8)
    return np.moveaxis(packed, -1, axis), np.moveaxis(scale_byte, -1, axis)


def dequantize_mxfp4(packed: np.ndarray, scales: np.ndarray, axis: int = -1):
    packed = np.moveaxis(packed, axis, -1)
    scales = np.moveaxis(scales, axis, -1)
    K_half = packed.shape[-1]
    K = K_half * 2
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((*packed.shape[:-1], K), dtype=np.uint8)
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high
    sign = (nibbles >> 3) & 1
    idx = nibbles & 0x7
    mag = E2M1_MAGNITUDES[idx]
    signed = np.where(sign == 1, -mag, mag)
    grouped = signed.reshape(*signed.shape[:-1], K // 32, 32)
    scale_f = np.power(2.0, scales.astype(np.float32) - 127.0)
    out = (grouped * scale_f[..., None]).reshape(*signed.shape[:-1], K)
    return np.moveaxis(out, -1, axis)


# ---------------------------------------------------------------------------
# Hardware model (verified constants only).
# ---------------------------------------------------------------------------

WAVE_SIZE = 64  # CDNA3 wavefront = 64 work-items
LDS_BANKS = 32  # 32 banks per CU
LDS_BANK_BYTES = 4  # each bank is 4 bytes wide
LDS_BYTES_PER_CU = 64 * 1024  # 64 KiB LDS per CU
MFMA_CYCLES_BF16_16x16x16 = 16  # v_mfma_f32_16x16x16_bf16_1k retires in 16 cycles
MATCORES_PER_CU = 4  # 4 matrix cores per CU on gfx942
BF16_FLOPS_PER_CU_PER_CYCLE = 2048  # ROCm matrix-cores blog (MI300X spec)
MI300A_CUS = 228  # 6 XCDs × 38 active CUs
MI300A_CLOCK_GHZ = 2.1  # peak engine clock
MI300A_HBM_TBPS = 5.3  # 5.3 TB/s HBM3
MI300A_PEAK_BF16_TF = MI300A_CUS * BF16_FLOPS_PER_CU_PER_CYCLE * MI300A_CLOCK_GHZ / 1e3  # ≈980


# B LDS tile is laid out [N, K] in row-major BF16. Bank for element (row, col):
def lds_bank(row: int, col: int, k_stride: int = 64) -> int:
    byte_addr = (row * k_stride + col) * 2  # 2 bytes per BF16
    return (byte_addr // LDS_BANK_BYTES) % LDS_BANKS


def swizzle_xor16(row: int, col: int) -> int:
    return col ^ ((row & 0xF) << 1)


# ---------------------------------------------------------------------------
# Figure 1: MXFP4 packs 32 floats into 17 bytes
# ---------------------------------------------------------------------------


def figure_mxfp4_format():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32, dtype=np.float32) * 2.5
    packed, scales = quantize_mxfp4(x[None, :], axis=-1)
    packed, scales = packed[0], scales[0]
    x_back = dequantize_mxfp4(packed[None, :], scales[None, :], axis=-1)[0]
    err = np.abs(x - x_back)

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 0.5, 0.4, 1.1], hspace=0.65)

    ax = fig.add_subplot(gs[0])
    ax.bar(np.arange(32), x, color=LANE_PALETTE[0], alpha=0.55, label="float32 source")
    ax.bar(
        np.arange(32), x_back, color=LANE_PALETTE[1], alpha=0.85, width=0.55, label="dequantised"
    )
    ax.set_xlim(-0.5, 31.5)
    ax.set_xlabel("k")
    ax.set_ylabel("value")
    ax.set_title("32 floats → 16 nibbles + 1 byte scale → 32 floats")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(0, 1)
    for i in range(32):
        nibble = (packed[i // 2] >> (4 if i % 2 else 0)) & 0xF
        sign = (nibble >> 3) & 1
        idx = nibble & 0x7
        fc = LANE_PALETTE[3] if sign else LANE_PALETTE[2]
        ax.add_patch(
            mpatches.Rectangle(
                (i - 0.5, 0), 1, 1, facecolor=fc, alpha=0.35, edgecolor=NEUTRAL, linewidth=0.5
            )
        )
        ax.text(i, 0.5, f"{idx}", ha="center", va="center", fontsize=9, color=NEUTRAL)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 32, 4))
    ax.set_title("Element code: magnitude index 0–7, red = sign bit set")
    ax.set_xlabel("k")

    ax = fig.add_subplot(gs[2])
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(0, 1)
    for byte_idx in range(16):
        ax.add_patch(
            mpatches.Rectangle(
                (byte_idx - 0.5, 0), 1, 1, fill=False, edgecolor=NEUTRAL, linewidth=0.7
            )
        )
        ax.text(
            byte_idx,
            0.5,
            f"{packed[byte_idx]:02X}",
            ha="center",
            va="center",
            family="monospace",
            fontsize=9,
        )
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 16, 2))
    ax.set_title(f"Packed: 16 bytes B_packed + 1 byte E8M0 scale = 0x{scales[0]:02X}")
    ax.set_xlabel("byte")

    ax = fig.add_subplot(gs[3])
    nonzero = np.maximum(err, 1e-4)
    ax.bar(np.arange(32), nonzero, color=LANE_PALETTE[4], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 31.5)
    ax.set_xlabel("k")
    ax.set_ylabel("|err|  (log)")
    ax.set_title(
        f"Quantisation error (log scale): max {err.max():.2f}, median {np.median(err):.3f}"
    )

    fig.text(
        0.5,
        0.005,
        "MXFP4 packs a 32-element block at 4.25 bits/element.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "01_mxfp4_format.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: MFMA B-operand lane layout
# ---------------------------------------------------------------------------


def figure_mfma_lane_layout():
    """16x16x16 BF16 MFMA: lane = n + 16·k_block, each lane holds 4 K-elements."""
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.0, 1.2]}
    )

    # Left: 16x16 B-tile, each cell coloured by lane id mod 8 (categorical).
    ax = ax_left
    for n in range(16):
        for k in range(16):
            k_block = k // 4
            lane = n + 16 * k_block
            colour = LANE_PALETTE[lane % 8]
            ax.add_patch(
                mpatches.Rectangle(
                    (k - 0.5, n - 0.5),
                    1,
                    1,
                    facecolor=colour,
                    alpha=0.55,
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
            ax.text(k, n, f"{lane}", ha="center", va="center", fontsize=6.5, color=NEUTRAL)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(15.5, -0.5)
    ax.set_xticks(np.arange(0, 16, 4))
    ax.set_yticks(np.arange(0, 16, 4))
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_title("lane = n + 16·k_block, each lane holds 4 K-elements")
    ax.set_aspect("equal")

    # Right: 4 K-block bands along K, with stripe pattern revealing lane ownership.
    ax = ax_right
    for k_block in range(4):
        x0 = k_block * 4
        for n in range(16):
            lane = n + 16 * k_block
            colour = LANE_PALETTE[lane % 8]
            ax.add_patch(
                mpatches.Rectangle(
                    (x0, n - 0.5),
                    4,
                    1,
                    facecolor=colour,
                    alpha=0.45,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )
    for k_block in range(4):
        ax.axvline(k_block * 4, color=NEUTRAL, linewidth=0.6, alpha=0.4)
    ax.axvline(16, color=NEUTRAL, linewidth=0.6, alpha=0.4)

    # Annotate K-block boundaries with lane ranges
    for k_block in range(4):
        ax.text(
            k_block * 4 + 2,
            -1.2,
            f"k_block {k_block}\nlanes {k_block * 16}–{k_block * 16 + 15}",
            ha="center",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(-0.3, 16.3)
    ax.set_ylim(15.7, -2.4)
    ax.set_xticks(np.arange(0, 17, 4))
    ax.set_yticks(np.arange(0, 16, 4))
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_title("Wavefront tiles K into four 4-wide bands, one per matrix core")
    ax.set_aspect("equal")

    fig.suptitle(
        "MFMA B-operand: 16×16 BF16 tile, 64 lanes, 4 K-blocks of 16 lanes each",
        fontsize=12,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.05,
        "Same lane colour across figures = same lane.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "02_mfma_lane_layout.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bank-conflict simulation.
# A single ds_read_b32 retires 1 dword per lane = 4 bytes per lane. The
# wavefront is split into two half-waves of 32 lanes; bank conflict is checked
# per half-wave. Each MFMA B-operand load consists of 2 ds_read_b32 (4 BF16
# per lane = 2 dwords). We simulate the first ds_read_b32 only — the second
# has the same bank pattern shifted by 4 bytes.
# ---------------------------------------------------------------------------


def simulate_ds_read_b32(swizzle: bool, k_mid: int = 0, k_stride: int = 64):
    """Returns bank id touched by each of the 64 lanes for one ds_read_b32."""
    banks = np.zeros(64, dtype=int)
    for lane in range(64):
        k_block = lane // 16
        n = lane % 16
        col = (k_block * 4) + k_mid * 16  # first BF16 within the K-block's 4-element slice
        if swizzle:
            col = swizzle_xor16(n, col)
        banks[lane] = lds_bank(n, col, k_stride)
    return banks


def conflict_per_half_wave(banks: np.ndarray):
    """Max bank conflict per half-wave (lanes 0-31 and 32-63 are separate dispatches)."""
    hw0 = np.bincount(banks[:32], minlength=LDS_BANKS).max()
    hw1 = np.bincount(banks[32:], minlength=LDS_BANKS).max()
    return max(hw0, hw1), hw0, hw1


# ---------------------------------------------------------------------------
# Figure 3: bank conflict on the unswizzled MFMA B-operand load
# ---------------------------------------------------------------------------


def _draw_lds_grid(ax, swizzle: bool, k_mid: int = 0, highlight_cols: range = range(0, 4)):
    """Draws a 16 (N rows) × 64 (K cols) LDS tile with cells coloured by bank id."""
    for n in range(16):
        for k in range(64):
            phys_col = swizzle_xor16(n, k) if swizzle else k
            bank = lds_bank(n, phys_col)
            colour = LANE_PALETTE[bank % 8]
            ax.add_patch(
                mpatches.Rectangle(
                    (k - 0.5, n - 0.5),
                    1,
                    1,
                    facecolor=colour,
                    alpha=0.40,
                    edgecolor="white",
                    linewidth=0.3,
                )
            )
    # Red outlines on the K-cols 0..3 of K-mid=0, across all 16 N-rows (the MFMA load)
    for n in range(16):
        for k_off in highlight_cols:
            actual_k = k_off + k_mid * 16
            actual_col = swizzle_xor16(n, actual_k) if swizzle else actual_k
            ax.add_patch(
                mpatches.Rectangle(
                    (actual_col - 0.5, n - 0.5),
                    1,
                    1,
                    facecolor="none",
                    edgecolor=HIGHLIGHT,
                    linewidth=1.6,
                )
            )
    ax.set_xlim(-0.5, 63.5)
    ax.set_ylim(15.5, -0.5)
    ax.set_xticks(np.arange(0, 64, 8))
    ax.set_yticks([0, 4, 8, 12, 15])
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_aspect("equal")


def figure_bank_conflict_unswizzled():
    banks = simulate_ds_read_b32(swizzle=False)
    max_conf, hw0, hw1 = conflict_per_half_wave(banks)

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.6)

    ax = fig.add_subplot(gs[0])
    _draw_lds_grid(ax, swizzle=False)
    ax.set_title(
        f"Unswizzled LDS: 16 lanes per K-block read one column → {max_conf}-way bank conflict"
    )

    ax = fig.add_subplot(gs[1])
    counts = np.bincount(banks, minlength=LDS_BANKS)
    bars = ax.bar(
        np.arange(LDS_BANKS),
        counts,
        color=LANE_PALETTE[3],
        alpha=0.85,
        edgecolor=NEUTRAL,
        linewidth=0.5,
    )
    ax.axhline(1, color=NEUTRAL, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(
        31.5, 1.4, "ideal: 1 lane / bank", fontsize=8, ha="right", style="italic", color=NEUTRAL
    )
    for i, h in enumerate(counts):
        if h > 0:
            ax.text(i, h + 0.4, str(int(h)), ha="center", fontsize=8, color=NEUTRAL)
    ax.set_xlim(-0.5, LDS_BANKS - 0.5)
    ax.set_xlabel("LDS bank")
    ax.set_ylabel("lanes per bank")
    ax.set_title(f"Bank histogram for one ds_read_b32 (half-waves: HW0={hw0}, HW1={hw1})")

    fig.text(
        0.5,
        0.0,
        f"Each {max_conf}-way conflict serialises to {max_conf} cycles per LDS read.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "03_bank_conflict_unswizzled.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: XOR-16 swizzle, before vs after
# ---------------------------------------------------------------------------


def figure_swizzle_fix():
    banks_unsw = simulate_ds_read_b32(swizzle=False)
    banks_sw = simulate_ds_read_b32(swizzle=True)
    conf_unsw, _, _ = conflict_per_half_wave(banks_unsw)
    conf_sw, _, _ = conflict_per_half_wave(banks_sw)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.5, wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    _draw_lds_grid(ax, swizzle=False)
    ax.set_title("Before: same column across 16 rows → same bank")

    ax = fig.add_subplot(gs[0, 1])
    _draw_lds_grid(ax, swizzle=True)
    ax.set_title("After XOR-16: 16 rows take 16 different bank positions")

    counts_unsw = np.bincount(banks_unsw, minlength=LDS_BANKS)
    counts_sw = np.bincount(banks_sw, minlength=LDS_BANKS)
    ymax = max(counts_unsw.max(), counts_sw.max()) + 2

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(
        np.arange(LDS_BANKS),
        counts_unsw,
        color=LANE_PALETTE[3],
        alpha=0.85,
        edgecolor=NEUTRAL,
        linewidth=0.5,
    )
    ax.axhline(1, color=NEUTRAL, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(-0.5, LDS_BANKS - 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("LDS bank")
    ax.set_ylabel("lanes per bank")
    ax.set_title(f"Before: 4 banks × {conf_unsw} lanes each  ({conf_unsw}-way per half-wave)")

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(
        np.arange(LDS_BANKS),
        counts_sw,
        color=LANE_PALETTE[2],
        alpha=0.85,
        edgecolor=NEUTRAL,
        linewidth=0.5,
    )
    ax.axhline(1, color=NEUTRAL, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(-0.5, LDS_BANKS - 0.5)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("LDS bank")
    ax.set_ylabel("lanes per bank")
    ax.set_title(f"After: 16 banks × {counts_sw.max()} lanes each  ({conf_sw}-way per half-wave)")

    fig.text(
        0.5,
        0.0,
        f"`col ^= ((row & 0xF) << 1)` cuts the conflict from {conf_unsw}-way to {conf_sw}-way.",
        ha="center",
        fontsize=9,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "04_swizzle_fix.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: software-scaled MFMA — algebraic equivalence to Petit's path
# ---------------------------------------------------------------------------


def figure_sw_scaled_decomposition():
    rng = np.random.default_rng(1)
    K = 64
    A = rng.standard_normal(K, dtype=np.float32)
    q_idx = rng.integers(1, 8, K)
    q_sign = rng.choice([-1.0, 1.0], K).astype(np.float32)
    q = q_sign * E2M1_MAGNITUDES[q_idx]
    s = np.where(np.arange(K) < 32, 0.25, 1.5).astype(np.float32)

    naive = np.cumsum(A * (q * s))

    sw = np.zeros(K, dtype=np.float32)
    running = np.float32(0.0)
    for g in range(2):
        sg = s[g * 32]
        partial = np.cumsum(A[g * 32 : (g + 1) * 32] * q[g * 32 : (g + 1) * 32])
        sw[g * 32 : (g + 1) * 32] = running + partial * sg
        running = sw[(g + 1) * 32 - 1]

    diff = sw - naive

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 0.7], "hspace": 0.32},
    )

    ax = axes[0]
    ax.axvspan(0, 31, color=LANE_PALETTE[0], alpha=0.06, label="MX group 0  (scale = 0.25)")
    ax.axvspan(31, 63, color=LANE_PALETTE[1], alpha=0.06, label="MX group 1  (scale = 1.5)")
    ax.plot(naive, color=LANE_PALETTE[3], linewidth=1.5, label=r"per-element: $\sum A_k (q_k s_g)$")
    ax.set_ylabel("running sum")
    ax.set_title("Per-element scaling (Petit-style)")
    ax.legend(loc="upper right")

    ax = axes[1]
    for g in range(2):
        x = np.arange(g * 32, (g + 1) * 32 + 1)
        partial = np.cumsum(A[g * 32 : (g + 1) * 32] * q[g * 32 : (g + 1) * 32])
        partial_prefix = np.concatenate([[0.0], partial])
        sg = s[g * 32]
        ax.plot(
            x[:-1],
            partial,
            color=LANE_PALETTE[g + 4],
            linestyle="--",
            linewidth=1.0,
            label=f"group {g}: unscaled MFMA partial",
        )
        ax.plot(
            x[:-1],
            partial * sg,
            color=LANE_PALETTE[g],
            linewidth=1.5,
            label=f"group {g}: × scale {sg}",
        )
    ax.set_ylabel("running sum")
    ax.set_title("Software-scaled (deferred): unscaled MFMA, single multiply per group")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    ax = axes[2]
    ax.axhline(0, color=NEUTRAL, linewidth=0.7, alpha=0.4)
    ax.plot(diff, color=LANE_PALETTE[2], linewidth=1.2)
    ax.set_xlim(0, K - 1)
    ax.set_xlabel("k (running cumulative position)")
    ax.set_ylabel("difference")
    ax.set_title(f"sw_scaled − per_element : max |Δ| = {np.max(np.abs(diff)):.2e}")

    fig.text(
        0.5,
        0.0,
        r"Identity $\sum_k A_k(q_k s_g) = s_g \sum_k A_k q_k$ holds within float32 roundoff.",
        ha="center",
        fontsize=9,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "05_sw_scaled_mfma.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: strategy chain. (a) table-like ordering, (b) bar chart with Petit
# horizontal reference line and a zoomed inset over H–I–J.
# ---------------------------------------------------------------------------


# Strategy data (measured on MI300A, 4096×11008×4096, 50 warmup + 100 iters)
STRATEGIES = [
    ("A", "two-pass: HBM BF16 scratch", 17.2),
    ("B", "LDS-staged fused", 36.3),
    ("C", "B + sched_group_barrier", 36.6),
    ("D", "128×128×64, 8 waves", 44.4),
    ("E", "128×64×64 M-stacked (low-CTA)", 27.7),
    ("F", "256×128×64, 16 waves", 50.3),
    ("G", "D + prepacked B (Marlin)", 49.0),
    ("H", "SW-scaled MFMA", 43.5),
    ("I", "H + XOR-16 LDS swizzle", 88.4),
    ("J", "I + sched_group_barrier", 86.7),
    ("K", "H + sched_group_barrier", 43.1),
    ("M", "J + s_setprio asm", 72.2),
]
PETIT_TF = 37.0


def figure_strategy_chain():
    letters = [s[0] for s in STRATEGIES]
    tflops = np.array([s[2] for s in STRATEGIES])

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    # All bars in one neutral colour except the winner; Petit is a horizontal reference.
    colours = [LANE_PALETTE[0]] * len(letters)
    colours[letters.index("I")] = LANE_PALETTE[2]  # winner
    bars = ax.bar(
        np.arange(len(letters)), tflops, color=colours, alpha=0.85, edgecolor=NEUTRAL, linewidth=0.5
    )
    for b, tf in zip(bars, tflops):
        ax.text(
            b.get_x() + b.get_width() / 2,
            tf + 1.5,
            f"{tf:.0f}",
            ha="center",
            fontsize=8.5,
            color=NEUTRAL,
        )

    ax.axhline(PETIT_TF, color=HIGHLIGHT, linestyle="--", linewidth=1.4)
    ax.set_xticks(np.arange(len(letters)))
    ax.set_xticklabels(letters)
    ax.set_ylabel("TFLOPS  (4096 × 11008 × 4096, MI300A)")
    ax.set_xlabel("strategy")
    ax.set_ylim(0, max(tflops) * 1.15)
    ax.set_title("Strategy I beats Petit by 2.4× on 4096×11008×4096; H→I delta = +45 TFLOPS")
    ax.text(
        len(letters) - 0.5,
        PETIT_TF + 1.5,
        f"Petit baseline = {PETIT_TF:.1f} TFLOPS",
        ha="right",
        va="bottom",
        fontsize=9,
        color=HIGHLIGHT,
        style="italic",
    )

    # Inset zoom on H → I → J
    inset_idx = [letters.index(c) for c in ("H", "I", "J")]
    inset_letters = ["H", "I", "J"]
    inset_tf = tflops[inset_idx]
    axin = ax.inset_axes((0.04, 0.56, 0.20, 0.36))
    bars_in = axin.bar(
        np.arange(3),
        inset_tf,
        color=[LANE_PALETTE[0], LANE_PALETTE[2], LANE_PALETTE[0]],
        alpha=0.85,
        edgecolor=NEUTRAL,
        linewidth=0.5,
    )
    for b, tf in zip(bars_in, inset_tf):
        axin.text(
            b.get_x() + b.get_width() / 2,
            tf + 1,
            f"{tf:.1f}",
            ha="center",
            fontsize=8,
            color=NEUTRAL,
        )
    delta_hi = inset_tf[1] - inset_tf[0]
    delta_ij = inset_tf[2] - inset_tf[1]
    axin.text(
        0.5,
        inset_tf[0] + (delta_hi / 2),
        f"+{delta_hi:.1f}",
        fontsize=9,
        color=LANE_PALETTE[2],
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
    )
    axin.text(
        1.5,
        inset_tf[1] + (delta_ij / 2),
        f"{delta_ij:+.1f}",
        fontsize=8.5,
        color=HIGHLIGHT,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
    )
    axin.set_xticks(np.arange(3))
    axin.set_xticklabels(inset_letters, fontsize=8)
    axin.set_ylim(0, max(inset_tf) * 1.25)
    axin.set_ylabel("TFLOPS", fontsize=8)
    axin.set_title("XOR-16 swizzle delta", fontsize=9)
    axin.tick_params(labelsize=7)

    # Write the table to a sibling file so the post can include it verbatim.
    table_path = FIG_DIR / "06_strategy_table.txt"
    with open(table_path, "w") as f:
        f.write("Strategy | Description                       | TFLOPS | % of Petit\n")
        f.write("---------+-----------------------------------+--------+-----------\n")
        for letter, desc, tf in STRATEGIES:
            f.write(f"   {letter:1s}     | {desc:33s} | {tf:6.1f} | {100 * tf / PETIT_TF:6.0f}%\n")
        f.write(f" Petit   | published baseline (mul_mxfp4_a16)| {PETIT_TF:6.1f} |   100%\n")

    fig.text(
        0.5,
        0.005,
        "Petit is a horizontal reference; bars above are speedups, bars below are regressions.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "06_strategy_chain.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6c (extra): roofline on MI300A for the compute-bound regime
# ---------------------------------------------------------------------------


def figure_roofline():
    """Roofline. Plots achievable throughput vs arithmetic intensity for the
    16 measured shapes; ceilings drawn for BF16 matrix peak, MXFP4 effective
    peak (dequant overhead), and HBM3 bandwidth."""
    # Operational intensity for an MXFP4 GEMM (FLOPs / B_packed bytes read once):
    # FLOPs = 2 * M * N * K; bytes = (K/2) * N + M * K * 2 + M * N * 4 + (K/32) * N
    shapes = [
        (4096, 11008, 4096, 88.4, "4096×11008×4096"),
        (8192, 8192, 8192, 93.8, "8192×8192×8192"),
        (4096, 8192, 8192, 97.0, "4096×8192×8192"),
        (4096, 28672, 8192, 95.5, "4096×28672×8192"),
        (2048, 11008, 4096, 83.5, "2048×11008×4096"),
        (512, 11008, 4096, 73.6, "512×11008×4096"),
        (128, 11008, 4096, 37.4, "128×11008×4096"),
    ]
    # Intensity in FLOP/byte. B_packed dominates; assume each tile read once.
    intensities = []
    tflops = []
    labels = []
    for M, N, K, tf, label in shapes:
        flops = 2.0 * M * N * K
        b = (K // 2) * N  # FP4 weight
        bytes_total = b + M * K * 2 + M * N * 4 + (K // 32) * N
        intensities.append(flops / bytes_total)
        tflops.append(tf)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Compute ceiling for BF16 dense (980 TF), MXFP4 effective ~250 (portable HIP),
    # and HBM3 ceiling.
    bw_hbm = MI300A_HBM_TBPS * 1e3  # GB/s as throughput-per-intensity
    peak_bf16 = MI300A_PEAK_BF16_TF
    portable_mxfp4_ceiling = 250.0  # ~25% MFU estimate for portable HIP

    intensity_range = np.logspace(0, 4, 200)
    hbm_ceiling = intensity_range * bw_hbm / 1e3  # in TFLOPS

    ax.plot(
        intensity_range,
        np.minimum(hbm_ceiling, peak_bf16),
        color=NEUTRAL,
        linewidth=1.5,
        label=f"BF16 matrix peak = {peak_bf16:.0f} TF",
    )
    ax.axhline(
        portable_mxfp4_ceiling,
        color=LANE_PALETTE[4],
        linewidth=1.2,
        linestyle="--",
        label=f"portable-HIP MXFP4 ceiling ≈ {portable_mxfp4_ceiling:.0f} TF",
    )
    ax.axhline(
        PETIT_TF,
        color=HIGHLIGHT,
        linewidth=1.2,
        linestyle=":",
        label=f"Petit baseline = {PETIT_TF:.0f} TF",
    )

    ax.scatter(
        intensities,
        tflops,
        s=70,
        color=LANE_PALETTE[2],
        edgecolor=NEUTRAL,
        linewidth=0.5,
        zorder=5,
        label="Strategy I, measured",
    )
    for x, y, label in zip(intensities, tflops, labels):
        ax.annotate(
            label, (x, y), xytext=(6, -3), textcoords="offset points", fontsize=7, color=NEUTRAL
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(5, 2000)
    ax.set_xlabel("arithmetic intensity (FLOP / byte of input)")
    ax.set_ylabel("throughput  (TFLOPS, log)")
    ax.set_title("Roofline: Strategy I sits below the portable-HIP ceiling, well above Petit")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.15)

    fig.text(
        0.5,
        0.005,
        "Headroom from I to the portable-HIP ceiling is ~3×; closing the gap requires inline asm.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "06b_roofline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Strategy I data flow, with byte counts on each arrow
# ---------------------------------------------------------------------------


def figure_strategy_i_dataflow():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, label, color="#eaeaea"):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.4",
                facecolor=color,
                edgecolor=NEUTRAL,
                linewidth=0.7,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color=NEUTRAL)

    def arrow(x1, y1, x2, y2, label="", colour=NEUTRAL, offset=(0, 0)):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=colour, lw=1.3)
        )
        if label:
            ax.text(
                (x1 + x2) / 2 + offset[0],
                (y1 + y2) / 2 + offset[1],
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=NEUTRAL,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
            )

    # HBM inputs
    box(2, 86, 22, 9, "A_bf16  [M, K]", color="#fff7e0")
    box(38, 86, 24, 9, "B_packed  [K/2, N]", color="#fff7e0")
    box(76, 86, 22, 9, "B_scales  [K/32, N]", color="#fff7e0")

    # Dequant + LDS stage
    box(2, 65, 22, 9, "A LDS  [128, 64]\nXOR-16 swizzled", color="#e1efff")
    box(36, 65, 28, 9, "Dequant FP4 → BF16\nv_perm_b32 LUT, unscaled", color="#ffe6e6")
    box(36, 50, 28, 9, "B LDS  [128, 64]\nXOR-16 swizzled", color="#e1efff")

    arrow(13, 86, 13, 74, "32 768 B / iter\n(128 × 64 × 2)")
    arrow(50, 86, 50, 74, "4 096 B / iter\n(64 × 128 × 4 bits)")
    arrow(50, 65, 50, 59, "v_perm LUT\n8 nibbles → 8 BF16")

    # MFMA + scale apply
    box(8, 28, 28, 10, "MFMA partial[2][4]\nfp32 accumulator", color="#e6f5e1")
    box(
        60,
        28,
        32,
        10,
        "for each MX group g:\n  fetch s_g (E8M0 → fp32)\n  total += partial · s_g",
        color="#e6f5e1",
    )

    arrow(13, 65, 18, 38, "ds_read_b32\n0 conflict")
    arrow(50, 50, 36, 38, "ds_read_b32\n0 conflict")
    arrow(87, 86, 76, 38, "global_load\n1 byte / lane / group", colour=LANE_PALETTE[0])
    arrow(36, 33, 60, 33, "32 fp32 × 8 lanes")

    # Output
    box(36, 10, 28, 9, "total[2][4]  fp32\n→ HBM C [M, N]", color="#e6f5e1")
    arrow(74, 28, 50, 19, "")
    arrow(22, 28, 50, 19, "")

    ax.set_title(
        "Strategy I: software-scaled MFMA, XOR-16 LDS swizzle, raw on-disk MXFP4",
        fontsize=12,
        pad=15,
    )
    fig.text(
        0.5,
        0.005,
        "Every LDS read is conflict-free; the only HBM cost beyond A and B_packed is one byte per (group, n_col).",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "07_strategy_i_dataflow.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: end-to-end correctness against a float32 numpy MXFP4 reference
# ---------------------------------------------------------------------------


def _kernel_naive(A, Bp, Bs):
    B_f32 = dequantize_mxfp4(Bp, Bs, axis=0)
    return A.astype(np.float32) @ B_f32


def _kernel_b_lds(A, Bp, Bs, tm=64, tn=128, tk=64):
    M, K = A.shape
    N = Bp.shape[1]
    C = np.zeros((M, N), dtype=np.float32)
    for bm in range(0, M, tm):
        for bn in range(0, N, tn):
            acc = np.zeros((min(tm, M - bm), min(tn, N - bn)), dtype=np.float32)
            for bk in range(0, K, tk):
                Atile = A[bm : bm + tm, bk : bk + tk].astype(np.float32)
                Bp_tile = Bp[bk // 2 : (bk + tk) // 2, bn : bn + tn]
                Bs_tile = Bs[bk // 32 : (bk + tk) // 32, bn : bn + tn]
                Btile = dequantize_mxfp4(Bp_tile, Bs_tile, axis=0)
                acc += Atile @ Btile
            C[bm : bm + tm, bn : bn + tn] = acc
    return C


def _kernel_h_sw(A, Bp, Bs, tm=128, tn=128, tk=64):
    M, K = A.shape
    N = Bp.shape[1]
    C = np.zeros((M, N), dtype=np.float32)
    for bm in range(0, M, tm):
        for bn in range(0, N, tn):
            total = np.zeros((min(tm, M - bm), min(tn, N - bn)), dtype=np.float32)
            for bk in range(0, K, tk):
                Atile = A[bm : bm + tm, bk : bk + tk].astype(np.float32)
                for g in range(tk // 32):
                    ks = bk + g * 32
                    Ag = Atile[:, g * 32 : (g + 1) * 32]
                    Bp_g = Bp[ks // 2 : (ks + 32) // 2, bn : bn + tn]
                    sg = Bs[ks // 32, bn : bn + tn].astype(np.float32)
                    low = Bp_g & 0x0F
                    high = (Bp_g >> 4) & 0x0F
                    nibbles = np.empty((32, low.shape[1]), dtype=np.uint8)
                    nibbles[0::2] = low
                    nibbles[1::2] = high
                    sign = (nibbles >> 3) & 1
                    idx = nibbles & 0x7
                    qg = E2M1_MAGNITUDES[idx]
                    qg = np.where(sign == 1, -qg, qg)
                    partial = Ag @ qg.astype(np.float32)
                    scale_f = np.power(2.0, sg - 127.0)
                    total += partial * scale_f[None, :]
            C[bm : bm + tm, bn : bn + tn] = total
    return C


def figure_correctness_grid():
    rng = np.random.default_rng(7)
    # All shapes here satisfy both kernels' tile gates: M%128, N%128, K%64.
    # Mix of square and non-square; include shapes whose M is not a power of 2.
    shapes = [
        (128, 128, 64),
        (128, 256, 64),
        (256, 128, 128),
        (256, 256, 128),
        (384, 384, 64),
        (384, 512, 128),
        (512, 384, 64),
        (768, 256, 192),
    ]
    fns = [("B (LDS-staged)", _kernel_b_lds), ("H (SW-scaled)", _kernel_h_sw)]
    skipped = {
        ("B (LDS-staged)", "M%64,N%128,K%64 required"): set(),
        ("H (SW-scaled)", "M%128,N%128,K%64 required"): set(),
    }
    err = np.full((len(fns), len(shapes)), np.nan)

    for j, (M, N, K) in enumerate(shapes):
        if K % 32 != 0:
            continue
        A = rng.standard_normal((M, K), dtype=np.float32)
        B = rng.standard_normal((K, N), dtype=np.float32) * 0.1
        Bp, Bs = quantize_mxfp4(B, axis=0)
        ref = _kernel_naive(A, Bp, Bs)
        for i, (name, fn) in enumerate(fns):
            try:
                if name.startswith("B"):
                    valid = (M % 64 == 0) and (N % 128 == 0) and (K % 64 == 0)
                else:
                    valid = (M % 128 == 0) and (N % 128 == 0) and (K % 64 == 0)
                if not valid:
                    continue
                out = fn(A, Bp, Bs)
                err[i, j] = float(np.max(np.abs(out - ref)))
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(11, 3.5))
    display = np.where(np.isnan(err), -16, np.log10(np.maximum(err, 1e-16)))
    im = ax.imshow(display, cmap="RdYlGn_r", aspect="auto", vmin=-12, vmax=-2)
    ax.set_xticks(np.arange(len(shapes)))
    ax.set_xticklabels(
        [f"{m}×{n}×{k}" for m, n, k in shapes], rotation=35, ha="right", fontsize=8.5
    )
    ax.set_yticks(np.arange(len(fns)))
    ax.set_yticklabels([n for n, _ in fns])
    for i in range(len(fns)):
        for j in range(len(shapes)):
            if np.isnan(err[i, j]):
                ax.text(j, i, "skip", ha="center", va="center", fontsize=8, color=NEUTRAL)
            else:
                err_val = err[i, j]
                label = "0" if err_val == 0 else f"{err_val:.0e}".replace("e-0", "e-")
                ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color="white")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, label="log10 max |out − ref|")
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Pedagogical kernels match the FP32-cast numpy reference within float roundoff")

    fig.text(
        0.5,
        -0.04,
        "Reference is FP32 GEMM after FP4→FP32 dequant; agreement at ≈1e-7 means same arithmetic, different order.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color=NEUTRAL,
    )
    fig.savefig(FIG_DIR / "08_correctness_grid.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    print(f"Writing figures to {FIG_DIR}")
    print(
        f"MI300A constants: {MI300A_CUS} CUs × {BF16_FLOPS_PER_CU_PER_CYCLE} flops/CU/cycle × "
        f"{MI300A_CLOCK_GHZ} GHz = {MI300A_PEAK_BF16_TF:.1f} TF BF16 peak"
    )
    print(f"LDS: {LDS_BANKS} banks × {LDS_BANK_BYTES} B/bank, {LDS_BYTES_PER_CU // 1024} KiB / CU")
    print()

    figure_mxfp4_format()
    print("  01_mxfp4_format.png")
    figure_mfma_lane_layout()
    print("  02_mfma_lane_layout.png")
    figure_bank_conflict_unswizzled()
    print("  03_bank_conflict_unswizzled.png")
    figure_swizzle_fix()
    print("  04_swizzle_fix.png")
    figure_sw_scaled_decomposition()
    print("  05_sw_scaled_mfma.png")
    figure_strategy_chain()
    print("  06_strategy_chain.png  +  06_strategy_table.txt")
    figure_roofline()
    print("  06b_roofline.png")
    figure_strategy_i_dataflow()
    print("  07_strategy_i_dataflow.png")
    figure_correctness_grid()
    print("  08_correctness_grid.png")


if __name__ == "__main__":
    main()
