import numpy as np

E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _nearest_e2m1(x):
    sign = (x < 0).astype(np.uint8)
    mag = np.abs(x)
    diffs = np.abs(mag[..., None] - E2M1_MAGNITUDES[None, :])
    idx = np.argmin(diffs, axis=-1).astype(np.uint8)
    return idx, sign


def quantize_mxfp4(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    G = 32
    if x.shape[axis] % G != 0:
        raise ValueError(f"axis {axis} length {x.shape[axis]} not divisible by {G}")

    x = np.moveaxis(x, axis, -1)
    orig_shape = x.shape
    n_groups = x.shape[-1] // G
    x = x.reshape(*x.shape[:-1], n_groups, G)

    max_abs = np.max(np.abs(x), axis=-1)
    safe = np.maximum(max_abs, 1e-38)
    log2_safe = np.ceil(np.log2(safe / 6.0))
    log2_safe = np.clip(log2_safe, -127, 127)
    scale_exp = log2_safe.astype(np.int32)
    scale_byte = (scale_exp + 127).astype(np.uint8)
    scale_f32 = np.power(2.0, scale_exp.astype(np.float32))

    x_scaled = x / scale_f32[..., None]
    em, sign = _nearest_e2m1(x_scaled)
    nibble = ((sign << 3) | em).astype(np.uint8)

    nibble = nibble.reshape(*nibble.shape[:-1], G)
    low = nibble[..., 0::2]
    high = nibble[..., 1::2]
    packed = ((high & 0x0F) << 4) | (low & 0x0F)

    packed = packed.reshape(*orig_shape[:-1], n_groups * (G // 2))
    packed = np.moveaxis(packed, -1, axis)
    scale_byte = np.moveaxis(scale_byte, -1, axis)
    return np.ascontiguousarray(packed.astype(np.uint8)), np.ascontiguousarray(
        scale_byte.astype(np.uint8)
    )


def dequantize_mxfp4(packed, scales, axis=-1):
    packed = np.moveaxis(packed, axis, -1)
    scales = np.moveaxis(scales, axis, -1)
    G = 32
    n_packed_per_group = G // 2

    n_groups = packed.shape[-1] // n_packed_per_group
    packed = packed.reshape(*packed.shape[:-1], n_groups, n_packed_per_group)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((*packed.shape[:-1], G), dtype=np.uint8)
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high

    sign = (nibbles >> 3) & 0x1
    em = nibbles & 0x7
    mag = E2M1_MAGNITUDES[em]
    val = mag * np.where(sign == 1, -1.0, 1.0).astype(np.float32)

    scale_f32 = np.power(2.0, scales.astype(np.int32) - 127).astype(np.float32)
    out = val * scale_f32[..., None]
    out = out.reshape(*out.shape[:-2], n_groups * G)
    return np.moveaxis(out, -1, axis)


def ref_gemm(A_bf16, B_packed, B_scales):
    A_f32 = A_bf16.astype(np.float32) if A_bf16.dtype != np.float32 else A_bf16
    B_f32 = dequantize_mxfp4(B_packed, B_scales, axis=0)
    return A_f32 @ B_f32


def preshuffle_b(B_packed, B_scales, N_tile=128, K_tile=128):
    return np.ascontiguousarray(B_packed), np.ascontiguousarray(B_scales)


def prepack_b_mxfp4(B_packed, B_scales, n_tile=128):
    # Strategy G's prepacked layout:
    #   B_prep_flat[nb, chunk, ni, b]   where chunk = K-byte / 4, b = K-byte % 4
    #   Bs_prep_flat[nb, k_scale, ni]   where k_scale = K-elt / 32
    # Per CTA n_tile, all bytes contiguous in HBM -> coalesced linear reads (no stride-N hops).
    K_half, N = B_packed.shape
    K = K_half * 2
    assert N % n_tile == 0, "N must be a multiple of n_tile"
    n_blocks = N // n_tile
    prep = np.ascontiguousarray(B_packed).reshape(K // 8, 4, n_blocks, n_tile).transpose(2, 0, 3, 1)
    prep_scales = (
        np.ascontiguousarray(B_scales).reshape(K // 32, n_blocks, n_tile).transpose(1, 0, 2)
    )
    return (
        np.ascontiguousarray(prep).reshape(-1),
        np.ascontiguousarray(prep_scales).reshape(-1),
    )
