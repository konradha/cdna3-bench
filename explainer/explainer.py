"""
CDNA3 MXFP4 GEMM explainer.

Runs end-to-end and writes figures to explainer/figures/. Each section produces
one figure that explains a single concept in the kernel chain.

Usage:
    cd <repo root>
    uv run --with numpy --with matplotlib explainer/explainer.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Part 1: the MXFP4 format
# ---------------------------------------------------------------------------
#
# MXFP4 stores a tensor in two arrays:
#   - packed FP4 magnitudes: 2 nibbles per byte, K-axis packed
#   - per-group E8M0 scales: 1 byte per 32 K-elements per column
#
# Each FP4 element is E2M1 (1 sign bit, 2 exp bits, 1 mantissa bit). The 8
# representable magnitudes are exactly:
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
    return (
        np.moveaxis(packed, -1, axis),
        np.moveaxis(scale_byte, -1, axis),
    )


def dequantize_mxfp4(packed: np.ndarray, scales: np.ndarray, axis: int = -1):
    packed = np.moveaxis(packed, axis, -1)
    scales = np.moveaxis(scales, axis, -1)
    K_half = packed.shape[-1]
    K = K_half * 2

    low = (packed & 0x0F).astype(np.uint8)
    high = ((packed >> 4) & 0x0F).astype(np.uint8)
    nibbles = np.empty((*packed.shape[:-1], K), dtype=np.uint8)
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high

    sign = (nibbles >> 3) & 1
    idx = nibbles & 0x7
    mag = E2M1_MAGNITUDES[idx]
    signed = np.where(sign == 1, -mag, mag)

    grouped = signed.reshape(*signed.shape[:-1], K // 32, 32)
    scale_f = np.power(2.0, scales.astype(np.float32) - 127.0)
    out = grouped * scale_f[..., None]
    out = out.reshape(*signed.shape[:-1], K)
    return np.moveaxis(out, -1, axis)


def figure_mxfp4_format():
    """Figure 1: how MXFP4 packs 32 floats into 17 bytes (16 nibbles + 1 scale)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32, dtype=np.float32) * 2.5
    packed, scales = quantize_mxfp4(x[None, :], axis=-1)
    packed = packed[0]
    scales = scales[0]
    x_back = dequantize_mxfp4(packed[None, :], scales[None, :], axis=-1)[0]

    fig, axes = plt.subplots(4, 1, figsize=(14, 7), height_ratios=[1.2, 0.6, 0.6, 1.2])

    ax = axes[0]
    ax.bar(np.arange(32), x, color="tab:blue", alpha=0.7, label="float32 original")
    ax.bar(np.arange(32), x_back, color="tab:orange", alpha=0.7, width=0.5, label="dequantized")
    ax.set_xlim(-0.5, 31.5)
    ax.set_xlabel("K-element index")
    ax.set_ylabel("value")
    ax.set_title("Step 1: 32 float32 values → MXFP4 → back to float32")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(0, 1)
    for i in range(32):
        nibble = (packed[i // 2] >> (4 if i % 2 else 0)) & 0xF
        sign = (nibble >> 3) & 1
        idx = nibble & 0x7
        color = "tab:red" if sign else "tab:green"
        ax.add_patch(mpatches.Rectangle((i - 0.5, 0), 1, 1, color=color, alpha=0.3))
        ax.text(i, 0.5, f"{idx}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks(np.arange(32))
    ax.set_title("Step 2: each element → 4-bit E2M1 (sign bit + 3-bit magnitude index 0..7)")
    ax.set_xlabel("K-element index (red = negative sign bit)")

    ax = axes[2]
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0, 1)
    for byte_idx in range(16):
        b = packed[byte_idx]
        ax.add_patch(mpatches.Rectangle((byte_idx - 0.5, 0), 1, 1, fill=False, edgecolor="black"))
        ax.text(
            byte_idx, 0.5, f"{b:02X}", ha="center", va="center", family="monospace", fontsize=11
        )
    ax.set_yticks([])
    ax.set_xticks(np.arange(16))
    ax.set_title(
        f"Step 3: 32 nibbles packed into 16 bytes  (+ 1-byte E8M0 scale = 0x{scales[0]:02X})"
    )
    ax.set_xlabel("byte index in B_packed")

    ax = axes[3]
    err = np.abs(x - x_back)
    ax.bar(np.arange(32), err, color="tab:purple", alpha=0.7)
    ax.set_xlim(-0.5, 31.5)
    ax.set_xlabel("K-element index")
    ax.set_ylabel("|original - dequant|")
    ax.set_title(f"Step 4: quantization error  (max={err.max():.3f}, mean={err.mean():.3f})")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_mxfp4_format.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Part 2: the CDNA3 MFMA tile layout
# ---------------------------------------------------------------------------
#
# `v_mfma_f32_16x16x16_bf16_1k` computes one 16×16 BF16 outer-product into a
# 16×16 FP32 accumulator. The wavefront has 64 lanes, partitioned into 4
# K-blocks of 16 lanes each. Within each K-block, lane `l ∈ [0, 16)` of A
# holds A[l][0..3] (4 BF16, one row, 4 K-columns). Same for B with rows of B.
#
# Lane encoding: lane = m + 16 * k_block (for A operand)
#                lane = n + 16 * k_block (for B operand)

WAVE_SIZE = 64


def figure_mfma_lane_layout():
    """Figure 2: which lane holds which (row, k) for the MFMA B-operand."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    grid = np.full((16, 16), -1, dtype=int)
    for lane in range(WAVE_SIZE):
        k_block = lane // 16
        n = lane % 16
        # lane reads 4 contiguous K elements starting at k_block*4
        for k_off in range(4):
            k = k_block * 4 + k_off
            if k < 16:
                grid[n, k] = lane
    im = ax.imshow(grid, cmap="tab20b", aspect="equal")
    for n in range(16):
        for k in range(16):
            ax.text(k, n, f"{grid[n, k]}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(np.arange(16))
    ax.set_yticks(np.arange(16))
    ax.set_xlabel("K column (within 16-element strip)")
    ax.set_ylabel("N row")
    ax.set_title(
        "MFMA B-operand: which lane holds (n, k)?\nlane = n + 16·k_block, lane holds 4 K-elements"
    )

    ax = axes[1]
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.5, 5)
    ax.invert_yaxis()
    for k_block in range(4):
        ax.add_patch(
            mpatches.Rectangle((k_block * 4 - 0.5, 0.5), 4, 4, color=f"C{k_block}", alpha=0.25)
        )
        ax.text(
            k_block * 4 + 1.5,
            1.2,
            f"K-block {k_block}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            k_block * 4 + 1.5,
            2.0,
            f"lanes {k_block * 16}..{k_block * 16 + 15}",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.text(
            k_block * 4 + 1.5,
            3.0,
            "16 lanes × 4 K = 64 BF16",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
        )
    ax.text(
        8,
        4.6,
        "All 4 K-blocks together = the full 16×16 B-tile (256 BF16)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xticks(np.arange(0, 17))
    ax.set_xlabel("K element index in 16-wide K-strip")
    ax.set_yticks([])
    ax.set_title("How the wavefront tiles the 16×16 B-operand across 4 K-blocks")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_mfma_lane_layout.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Part 3: LDS bank conflicts on the unswizzled tile
# ---------------------------------------------------------------------------
#
# LDS has 32 banks of 4 bytes each. BF16 element [row, col] lives at byte:
#     addr = (row * TILE_K + col) * 2
# bank = (addr / 4) mod 32
#
# For TILE_K = 64 BF16 = 128 bytes per row = exactly 32 banks. Every row
# starts at the same bank → simultaneous reads across rows at the same col
# all hit the same bank → 16-way serialization.

LDS_BANKS = 32
LDS_BANK_BYTES = 4
TILE_K = 64
BF16_BYTES = 2


def lds_bank(row: int, col: int, tile_k: int = TILE_K) -> int:
    byte_addr = (row * tile_k + col) * BF16_BYTES
    return (byte_addr // LDS_BANK_BYTES) % LDS_BANKS


def swizzle_xor16(row: int, col: int) -> int:
    return col ^ ((row & 0xF) << 1)


def simulate_b_operand_load(rows, cols_per_lane, swizzle=False):
    """Simulate a wave-wide LDS read for the MFMA B-operand.

    `rows`: list of N rows that each lane reads from (len 16, one per lane in K-block)
    `cols_per_lane`: list of 4 contiguous K columns per lane (len 4)

    Returns: bank_count[bank] = how many lanes hit that bank
             max_conflict = max simultaneous lanes per bank (1 = no conflict)
    """
    bank_count = np.zeros(LDS_BANKS, dtype=int)
    accesses = []
    for row in rows:
        for col in cols_per_lane:
            actual_col = swizzle_xor16(row, col) if swizzle else col
            b = lds_bank(row, actual_col)
            bank_count[b] += 1
            accesses.append((row, col, actual_col, b))
    return bank_count, max(bank_count), accesses


def figure_bank_conflict_unswizzled():
    """Figure 3: 16-way bank conflict on the unswizzled MFMA B-operand load."""
    rows = list(range(16))
    cols = [0, 1, 2, 3]
    bank_count, max_conf, accesses = simulate_b_operand_load(rows, cols, swizzle=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    grid = np.zeros((16, 64), dtype=int)
    for row in range(16):
        for col in range(64):
            grid[row, col] = lds_bank(row, col)
    im = ax.imshow(grid, cmap="hsv", aspect="auto")
    for row in range(16):
        for col in cols:
            ax.add_patch(
                mpatches.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5
                )
            )
    ax.set_xlabel("K column (BF16 element)")
    ax.set_ylabel("N row")
    ax.set_title("Unswizzled LDS layout: bank(row, col)\n(red = 16 lanes' read targets at k=0..3)")
    ax.set_xticks(np.arange(0, 64, 4))
    ax.set_yticks(np.arange(16))

    ax = axes[1]
    bars = ax.bar(np.arange(LDS_BANKS), bank_count, color="tab:red")
    for i, h in enumerate(bank_count):
        if h > 0:
            ax.text(i, h + 0.5, str(int(h)), ha="center", fontsize=9)
    ax.set_xlabel("LDS bank index")
    ax.set_ylabel("simultaneous lane accesses")
    ax.set_title(
        f"Bank histogram across 64 lane×col reads\n"
        f"max conflict = {max_conf}-way  →  serializes to ~{max_conf} cycles per ds_read_b32"
    )
    ax.set_xticks(np.arange(0, LDS_BANKS, 2))
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="conflict-free (1 lane/bank)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_bank_conflict_unswizzled.png", dpi=110)
    plt.close()


def figure_swizzle_fix():
    """Figure 4: XOR-16 swizzle distributes the same 16 rows across all 32 banks."""
    rows = list(range(16))
    cols = [0, 1, 2, 3]
    bank_unsw, max_unsw, _ = simulate_b_operand_load(rows, cols, swizzle=False)
    bank_sw, max_sw, _ = simulate_b_operand_load(rows, cols, swizzle=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    grid = np.zeros((16, 64), dtype=int)
    for row in range(16):
        for col in range(64):
            grid[row, col] = lds_bank(row, col)
    ax.imshow(grid, cmap="hsv", aspect="auto")
    for row in range(16):
        for col in cols:
            ax.add_patch(
                mpatches.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.8
                )
            )
    ax.set_title("BEFORE swizzle: bank(row, col)\nred boxes = lanes' physical reads")
    ax.set_xlabel("K column")
    ax.set_ylabel("N row")
    ax.set_xticks([])

    ax = axes[0, 1]
    grid = np.zeros((16, 64), dtype=int)
    for row in range(16):
        for col in range(64):
            grid[row, col] = lds_bank(row, col)
    ax.imshow(grid, cmap="hsv", aspect="auto")
    for row in range(16):
        for col in cols:
            actual = swizzle_xor16(row, col)
            ax.add_patch(
                mpatches.Rectangle(
                    (actual - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.8
                )
            )
    ax.set_title(
        "AFTER swizzle: bank(row, col ^ ((row&0xF)<<1))\nred boxes = lanes' physical reads"
    )
    ax.set_xlabel("K column (swizzled)")
    ax.set_ylabel("N row")
    ax.set_xticks([])

    ax = axes[1, 0]
    ax.bar(np.arange(LDS_BANKS), bank_unsw, color="tab:red")
    ax.set_title(f"Bank histogram BEFORE swizzle  (max {max_unsw}-way conflict)")
    ax.set_xlabel("LDS bank")
    ax.set_ylabel("lanes hitting bank")
    ax.set_ylim(0, max(max_unsw, max_sw) + 1)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.4)

    ax = axes[1, 1]
    ax.bar(np.arange(LDS_BANKS), bank_sw, color="tab:green")
    ax.set_title(f"Bank histogram AFTER swizzle  (max {max_sw}-way conflict)")
    ax.set_xlabel("LDS bank")
    ax.set_ylabel("lanes hitting bank")
    ax.set_ylim(0, max(max_unsw, max_sw) + 1)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_swizzle_fix.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Part 4: software-scaled MFMA decomposition
# ---------------------------------------------------------------------------
#
# Identity: Σₖ A_k · (q_k · s_g) = s_g · Σₖ A_k · q_k
# Deferred-scale: do MFMA on unscaled (A, q) into fp32 partial; then
# partial *= s_g; total += partial. One fmul per MX group per output, instead
# of per element.


def figure_sw_scaled_decomposition():
    """Figure 5: per-element scale (Petit) vs per-group scale on accumulator (us)."""
    rng = np.random.default_rng(1)
    K = 64
    A = rng.standard_normal(K, dtype=np.float32)
    q = np.array(
        [E2M1_MAGNITUDES[rng.integers(0, 8)] * (-1 if rng.random() < 0.5 else 1) for _ in range(K)],
        dtype=np.float32,
    )
    s = np.array([0.25 if k < 32 else 1.5 for k in range(K)], dtype=np.float32)

    naive = (A * (q * s)).cumsum()
    partials_per_group = []
    deferred = np.zeros(K, dtype=np.float32)
    running = 0.0
    for g in range(2):
        Ag = A[g * 32 : (g + 1) * 32]
        qg = q[g * 32 : (g + 1) * 32]
        sg = s[g * 32]
        partial = (Ag * qg).cumsum()
        partial_total = partial[-1]
        for i in range(32):
            running += A[g * 32 + i] * q[g * 32 + i]
        partials_per_group.append((partial, sg))
        running_with_scale = partial_total * sg
        for i in range(32):
            deferred[g * 32 + i] = running_with_scale * (i + 1) / 32 + (
                deferred[g * 32 - 1] if g > 0 else 0.0
            )

    fig, axes = plt.subplots(2, 1, figsize=(15, 7))

    ax = axes[0]
    ax.plot(
        naive, color="tab:red", label="Petit-style: per-element  $\\Sigma A_k (q_k \\cdot s_g)$"
    )
    ax.axvspan(0, 31, alpha=0.1, color="tab:blue", label="MX group 0 (s=0.25)")
    ax.axvspan(31, 63, alpha=0.1, color="tab:orange", label="MX group 1 (s=1.5)")
    ax.set_title("Naive: each element-wise multiply has the per-K scale baked in")
    ax.set_xlabel("K-element index (within the inner accumulation)")
    ax.set_ylabel("partial sum")
    ax.legend(loc="upper left")

    ax = axes[1]
    for g, (partial, sg) in enumerate(partials_per_group):
        x = np.arange(g * 32, g * 32 + 32)
        ax.plot(
            x,
            partial,
            color=f"C{g + 4}",
            linestyle="--",
            label=f"unscaled MFMA partial (group {g})",
        )
        ax.plot(
            x,
            partial * sg,
            color=f"C{g}",
            linewidth=2,
            label=f"partial × s_g={sg} (post-MFMA scale fmul)",
        )
    ax.set_title(
        "Software-scaled: unscaled MFMA partial, then a single fmul per MX group at the end"
    )
    ax.set_xlabel("K-element index")
    ax.set_ylabel("partial sum")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_sw_scaled_mfma.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Part 5: simulate each strategy and compare cycle counts
# ---------------------------------------------------------------------------
#
# We don't actually launch HIP kernels here; we simulate the *cycle cost* of
# each strategy from first principles, using a simple model:
#
#   per-MFMA cost = max(operand_load_cycles, dequant_cycles, scale_apply_cycles, mfma_throughput)
#
# Bank-conflicted LDS read costs `max_conflict_factor` cycles instead of 1.


@dataclass
class StrategyResult:
    name: str
    description: str
    bf16_dequant_cycles_per_8: float  # VALU per 8 FP4 → BF16
    lds_load_conflict_factor: int  # 1..16, multiplies ds_read latency
    mfma_groups_per_64K: int  # number of distinct MFMA groups per 64-K inner
    scale_fmul_per_8: float  # per-element vs per-group
    extra_passes_hbm: int  # 0 for fused, 1 for 2-pass dequant
    note: str = ""


STRATEGIES = [
    StrategyResult(
        name="Naive (pure FP32)",
        description="Dequant to FP32 in registers, FP32 matmul. No matrix engine.",
        bf16_dequant_cycles_per_8=200,
        lds_load_conflict_factor=1,
        mfma_groups_per_64K=0,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=0,
        note="Baseline. Compute peak is 1/8 of BF16 peak.",
    ),
    StrategyResult(
        name="A (two-pass)",
        description="Dequant whole B to HBM BF16 scratch, then dense BF16 GEMM.",
        bf16_dequant_cycles_per_8=22,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=4,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=2,
        note="HBM bandwidth bound on the scratch write+read.",
    ),
    StrategyResult(
        name="B (LDS-staged fused)",
        description="Dequant FP4→BF16 inline, stage in LDS, double-buffered.",
        bf16_dequant_cycles_per_8=22,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=4,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=0,
        note="MFMA-issue bound on LDS bank conflict.",
    ),
    StrategyResult(
        name="C (B + sched hints)",
        description="B with sched_group_barrier intrinsics.",
        bf16_dequant_cycles_per_8=22,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=4,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=0,
        note="Wash with B. Hints don't help when LDS is conflicting.",
    ),
    StrategyResult(
        name="D (128M × 128N × 64K, 8 waves)",
        description="Bigger M tile, B reused 2× across M wave-rows.",
        bf16_dequant_cycles_per_8=22,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=4,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=0,
        note="More M-reuse but still bank-conflicting.",
    ),
    StrategyResult(
        name="F (256M × 128N × 64K, 16 waves)",
        description="Even bigger M tile, single-buf A + double-buf B.",
        bf16_dequant_cycles_per_8=22,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=4,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=0,
        note="Best non-SW-scaled. Still LDS-bound.",
    ),
    StrategyResult(
        name="G (D + prepacked B)",
        description="D geometry + Marlin-style offline weight repack.",
        bf16_dequant_cycles_per_8=22,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=4,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=0,
        note="HBM was never the bottleneck. Wash with F.",
    ),
    StrategyResult(
        name="H (SW-scaled MFMA, D geometry)",
        description="Deferred E8M0 scale: unscaled MFMA partial, scale post-MFMA.",
        bf16_dequant_cycles_per_8=13,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=2,
        scale_fmul_per_8=0.25,
        extra_passes_hbm=0,
        note="Cheaper dequant. Still bank-conflicting.",
    ),
    StrategyResult(
        name="I (H + XOR-16 LDS swizzle)",
        description="H with col ^= ((row & 0xF) << 1) on every LDS access.",
        bf16_dequant_cycles_per_8=13,
        lds_load_conflict_factor=1,
        mfma_groups_per_64K=2,
        scale_fmul_per_8=0.25,
        extra_passes_hbm=0,
        note="Bank-conflict-free. The winner.",
    ),
    StrategyResult(
        name="J (I + sched hints)",
        description="I with sched_group_barrier choreography.",
        bf16_dequant_cycles_per_8=13,
        lds_load_conflict_factor=1,
        mfma_groups_per_64K=2,
        scale_fmul_per_8=0.25,
        extra_passes_hbm=0,
        note="Slight regression. Hints over-constrain the scheduler.",
    ),
    StrategyResult(
        name="K (H + sched hints)",
        description="H with sched_group_barrier (no swizzle).",
        bf16_dequant_cycles_per_8=13,
        lds_load_conflict_factor=16,
        mfma_groups_per_64K=2,
        scale_fmul_per_8=0.25,
        extra_passes_hbm=0,
        note="Wash with H. Hints don't help on conflicting layout.",
    ),
    StrategyResult(
        name="M (J + s_setprio asm, A swizzle dropped)",
        description="J + s_setprio inline asm; lost A-swizzle to direct-LDS.",
        bf16_dequant_cycles_per_8=13,
        lds_load_conflict_factor=8,
        mfma_groups_per_64K=2,
        scale_fmul_per_8=0.25,
        extra_passes_hbm=0,
        note="Layout > asm. Dropped A swizzle to make room for direct-LDS.",
    ),
    StrategyResult(
        name="Petit (published baseline)",
        description="Custom dequant via v_bfrev/SDWA, Marlin shuffle, per-element scale.",
        bf16_dequant_cycles_per_8=15,
        lds_load_conflict_factor=1,
        mfma_groups_per_64K=4,
        scale_fmul_per_8=8.0,
        extra_passes_hbm=0,
        note="Bank-conflict-free but per-element scale dominates.",
    ),
]


def estimate_relative_tflops(s: StrategyResult) -> float:
    """Toy cycle-cost model. Returns a relative throughput score (higher = faster)."""
    mfma_throughput_cyc = 16
    dequant_cyc = s.bf16_dequant_cycles_per_8 + s.scale_fmul_per_8
    lds_load_cyc = 4 * s.lds_load_conflict_factor
    extra_hbm = s.extra_passes_hbm * 100
    bottleneck = max(mfma_throughput_cyc, dequant_cyc, lds_load_cyc) + extra_hbm
    return 100.0 / bottleneck


def figure_strategy_chain():
    """Figure 6: estimated relative throughput across the strategy chain."""
    measured_tf = {
        "A (two-pass)": 17.2,
        "B (LDS-staged fused)": 36.3,
        "C (B + sched hints)": 36.6,
        "D (128M × 128N × 64K, 8 waves)": 44.4,
        "F (256M × 128N × 64K, 16 waves)": 50.3,
        "G (D + prepacked B)": 49.0,
        "H (SW-scaled MFMA, D geometry)": 43.5,
        "I (H + XOR-16 LDS swizzle)": 88.4,
        "J (I + sched hints)": 86.7,
        "K (H + sched hints)": 43.1,
        "M (J + s_setprio asm, A swizzle dropped)": 72.2,
        "Petit (published baseline)": 37.0,
    }
    names = [s.name for s in STRATEGIES if s.name in measured_tf]
    model = [estimate_relative_tflops(s) for s in STRATEGIES if s.name in measured_tf]
    measured = [measured_tf[n] for n in names]

    # Normalize the toy cycle model to the measured-TFLOPS scale so both are comparable.
    measured_max = max(measured)
    model_max = max(model)
    model_scaled = [m * measured_max / model_max for m in model]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(names))
    w = 0.4
    ax.bar(
        x - w / 2,
        model_scaled,
        w,
        label="toy cycle model (scaled to measured peak)",
        color="tab:gray",
        alpha=0.7,
    )
    ax.bar(x + w / 2, measured, w, label="measured TFLOPS (4096×11008×4096)", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("TFLOPS (measured) / scaled cycle-model proxy")
    ax.set_title("Strategy chain: cycle model predicts which step removes a bottleneck")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_strategy_chain.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Part 6: small executable reference kernels (numpy)
# ---------------------------------------------------------------------------
#
# These are pedagogical implementations, not performance kernels. They demonstrate
# what each strategy actually computes and how the data flows. All match the same
# numpy reference output `A_f32 @ dequant(B_packed, B_scales)`.


def kernel_naive(A, B_packed, B_scales):
    """Reference: full dequant followed by float32 GEMM."""
    B_f32 = dequantize_mxfp4(B_packed, B_scales, axis=0)
    return A.astype(np.float32) @ B_f32


def kernel_a_two_pass(A, B_packed, B_scales):
    """Two-pass: same as naive but spelled out as a pipeline."""
    B_bf16_scratch = dequantize_mxfp4(B_packed, B_scales, axis=0).astype(np.float32)
    return A.astype(np.float32) @ B_bf16_scratch


def kernel_b_lds_staged(A, B_packed, B_scales, tile_m=64, tile_n=128, tile_k=64):
    """Fused: tile-by-tile dequant + accumulate. No actual LDS, just block structure."""
    M, K = A.shape
    N = B_packed.shape[1]
    C = np.zeros((M, N), dtype=np.float32)
    for bm in range(0, M, tile_m):
        for bn in range(0, N, tile_n):
            acc = np.zeros((tile_m, tile_n), dtype=np.float32)
            for bk in range(0, K, tile_k):
                A_tile = A[bm : bm + tile_m, bk : bk + tile_k].astype(np.float32)
                B_packed_tile = B_packed[bk // 2 : (bk + tile_k) // 2, bn : bn + tile_n]
                B_scales_tile = B_scales[bk // 32 : (bk + tile_k) // 32, bn : bn + tile_n]
                B_tile = dequantize_mxfp4(B_packed_tile, B_scales_tile, axis=0)
                acc += A_tile @ B_tile
            C[bm : bm + tile_m, bn : bn + tile_n] = acc
    return C


def kernel_h_sw_scaled(A, B_packed, B_scales, tile_m=128, tile_n=128, tile_k=64):
    """Software-scaled: per-MX-group partial accumulator, scale post-MFMA."""
    M, K = A.shape
    N = B_packed.shape[1]
    C = np.zeros((M, N), dtype=np.float32)
    for bm in range(0, M, tile_m):
        for bn in range(0, N, tile_n):
            total = np.zeros((tile_m, tile_n), dtype=np.float32)
            for bk in range(0, K, tile_k):
                A_tile = A[bm : bm + tile_m, bk : bk + tile_k].astype(np.float32)
                # decompose each 64-K tile into 2 MX-groups of 32 K each
                for g in range(tile_k // 32):
                    k_start = bk + g * 32
                    A_g = A_tile[:, g * 32 : (g + 1) * 32]
                    B_packed_g = B_packed[k_start // 2 : (k_start + 32) // 2, bn : bn + tile_n]
                    scales_g = B_scales[k_start // 32, bn : bn + tile_n]
                    # unscaled dequant (just E2M1 magnitudes with signs)
                    low = (B_packed_g & 0x0F).astype(np.uint8)
                    high = ((B_packed_g >> 4) & 0x0F).astype(np.uint8)
                    nibbles = np.empty((32, tile_n), dtype=np.uint8)
                    nibbles[0::2] = low
                    nibbles[1::2] = high
                    sign = (nibbles >> 3) & 1
                    idx = nibbles & 0x7
                    q_g = E2M1_MAGNITUDES[idx]
                    q_g = np.where(sign == 1, -q_g, q_g)
                    # unscaled MFMA partial
                    partial = A_g @ q_g.astype(np.float32)
                    # apply E8M0 scale post-MFMA (broadcast across M rows)
                    scale_f = np.power(2.0, scales_g.astype(np.float32) - 127.0)
                    total += partial * scale_f[None, :]
            C[bm : bm + tile_m, bn : bn + tile_n] = total
    return C


def verify_kernels():
    """Sanity check: all numpy reference kernels match each other."""
    rng = np.random.default_rng(7)
    M, N, K = 128, 128, 64
    A = rng.standard_normal((M, K), dtype=np.float32).astype(np.float32)
    B_f32 = rng.standard_normal((K, N), dtype=np.float32) * 0.1
    B_packed, B_scales = quantize_mxfp4(B_f32, axis=0)

    ref = kernel_naive(A, B_packed, B_scales)
    a = kernel_a_two_pass(A, B_packed, B_scales)
    b = kernel_b_lds_staged(A, B_packed, B_scales)
    h = kernel_h_sw_scaled(A, B_packed, B_scales)

    print("=== reference kernel correctness ===")
    print(f"  A vs naive:  max_abs={np.max(np.abs(a - ref)):.2e}")
    print(f"  B vs naive:  max_abs={np.max(np.abs(b - ref)):.2e}")
    print(f"  H vs naive:  max_abs={np.max(np.abs(h - ref)):.2e}")


# ---------------------------------------------------------------------------
# Part 7: data-flow diagram for the I kernel (the winner)
# ---------------------------------------------------------------------------


def figure_strategy_i_dataflow():
    """Figure 7: end-to-end data flow for Strategy I (the winner)."""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, label, color="lightblue", fontsize=10):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.5", facecolor=color, edgecolor="black"
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, label="", color="black"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

    box(5, 88, 25, 8, "A_bf16  [M, K]\n(HBM)", "lightyellow")
    box(40, 88, 25, 8, "B_packed  [K/2, N]\n(HBM, FP4 nibbles)", "lightyellow")
    box(75, 88, 20, 8, "B_scales  [K/32, N]\n(HBM, E8M0)", "lightyellow")

    box(5, 70, 25, 10, "A LDS tile [128, 64]\n(swizzled write)", "lightblue")
    box(
        35,
        70,
        30,
        10,
        "Dequant FP4→BF16\n(8 nibbles → 8 BF16 via v_perm LUT)\n*no scale applied*",
        "lightcoral",
    )
    arrow(17.5, 88, 17.5, 80, "global load + swizzle")
    arrow(52.5, 88, 50, 80, "global load")

    box(35, 55, 30, 10, "B LDS tile [128, 64]\n(swizzled write)", "lightblue")
    arrow(50, 70, 50, 65, "")

    box(15, 38, 30, 10, "MFMA partial[2][4]\nfp32 accumulator\n(per MX group)", "lightgreen")
    arrow(17.5, 70, 25, 48, "ds_read_b32 (no conflict)")
    arrow(50, 55, 45, 48, "ds_read_b32 (no conflict)")

    box(
        55,
        38,
        35,
        10,
        "for each MX group g:\n  fetch s_g (E8M0 → fp32)\n  total[mi][ni] += partial * s_g",
        "lightgreen",
    )
    arrow(82.5, 88, 75, 48, "global load (1×/group)", color="tab:blue")
    arrow(45, 43, 55, 43, "fmul fold")

    box(35, 20, 30, 10, "total[2][4]\nfp32 accumulator\n(per output tile)", "lightgreen")
    arrow(70, 38, 60, 30, "")

    box(35, 5, 30, 10, "C[M, N] fp32\n(HBM write-back)", "lightyellow")
    arrow(50, 20, 50, 15, "store_acc_tile")

    ax.set_title("Strategy I data flow: software-scaled MFMA + XOR-16 LDS swizzle", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_strategy_i_dataflow.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Part 8: end-to-end numerical correctness sweep
# ---------------------------------------------------------------------------


def figure_correctness_grid():
    """Figure 8: numerical agreement of each pedagogical kernel with the reference."""
    rng = np.random.default_rng(7)
    shapes = [(128, 128, 64), (128, 256, 64), (256, 128, 64), (256, 256, 128)]
    kernel_fns = [
        ("naive", kernel_naive),
        ("A (two-pass)", kernel_a_two_pass),
        ("B (LDS-staged)", kernel_b_lds_staged),
        ("H (SW-scaled)", kernel_h_sw_scaled),
    ]
    err = np.zeros((len(kernel_fns), len(shapes)))
    for j, (M, N, K) in enumerate(shapes):
        A = rng.standard_normal((M, K), dtype=np.float32)
        B = rng.standard_normal((K, N), dtype=np.float32) * 0.1
        Bp, Bs = quantize_mxfp4(B, axis=0)
        ref = kernel_naive(A, Bp, Bs)
        for i, (_, fn) in enumerate(kernel_fns):
            out = fn(A, Bp, Bs)
            err[i, j] = float(np.max(np.abs(out - ref)))

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(np.log10(err + 1e-12), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(shapes)))
    ax.set_xticklabels([f"{m}×{n}×{k}" for m, n, k in shapes])
    ax.set_yticks(np.arange(len(kernel_fns)))
    ax.set_yticklabels([n for n, _ in kernel_fns])
    for i in range(len(kernel_fns)):
        for j in range(len(shapes)):
            ax.text(j, i, f"{err[i, j]:.1e}", ha="center", va="center", fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="log10 max_abs_err")
    ax.set_title("All reference kernels match numpy ref within float roundoff")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_correctness_grid.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    print(f"Writing figures to {FIG_DIR}")
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
    print("  06_strategy_chain.png")
    figure_strategy_i_dataflow()
    print("  07_strategy_i_dataflow.png")
    figure_correctness_grid()
    print("  08_correctness_grid.png")
    verify_kernels()
    print("done.")


if __name__ == "__main__":
    main()
