#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace mxfp4_cdna3 {

// v_perm_b32: byte-select 4 of 8 bytes via 4-byte selector.
__device__ __forceinline__ uint32_t amdgcn_perm(uint32_t s1, uint32_t s0, uint32_t sel) {
  return __builtin_amdgcn_perm(s1, s0, sel);
}

union v2f {
  float a[2];
  uint64_t u;
};

// v_cvt_pk_f32_fp8: 2 packed FP8(E4M3) -> 2 FP32 (gfx942 native).
__device__ __forceinline__ v2f amdgcn_cvt_pk_f32_fp8(uint32_t src) {
  v2f r;
  r.a[0] = __builtin_amdgcn_cvt_f32_fp8(src, 0);
  r.a[1] = __builtin_amdgcn_cvt_f32_fp8(src, 1);
  return r;
}

// E8M0 byte -> FP32 multiplier (shl-23 places exponent, zero mantissa).
__device__ __forceinline__ float e8m0_to_fp32(uint8_t scale) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(scale) << 23);
}

// 8 packed E2M1 -> 8 bf16. Path: FP4 -> FP8 via perm-LUT -> FP32 (hw cvt) -> scale -> bf16.
// gfx942 has no direct FP4->bf16 hw path.
__device__ __forceinline__ void dequant_8_mxfp4_to_bf16(uint32_t packed8, uint8_t scale,
                                                        __hip_bfloat16* __restrict__ out) {
  const uint32_t em_lo = packed8 & 0x07070707u;
  const uint32_t em_hi = (packed8 & 0x70707070u) >> 4;

  // LUT pre-stores -magnitude in FP8 FNUZ (bias 8, what gfx942 v_cvt_f32_fp8 expects):
  //   index {0..7} -> {-0, -0.5, -1, -1.5, -2, -3, -4, -6}.
  // Triton third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp upcast8xMxfp4_SW (CDNA3 branch).
  const uint32_t resLutLo = 0xC4C0B800u;
  const uint32_t resLutHi = 0xD4D0CCC8u;

  const uint32_t fp8_e = amdgcn_perm(resLutHi, resLutLo, em_lo);
  const uint32_t fp8_o = amdgcn_perm(resLutHi, resLutLo, em_hi);

  // FP4 sign bits: even positions {0,2,4,6} live at byte-bit 3; odd {1,3,5,7} at byte-bit 7.
  // 32-bit shift-left by 4 moves even signs into byte-bit 7 (LUT byte boundaries are preserved
  // because each byte's high nibble is the next FP4's low bits, which we don't touch here).
  // AND-mask (sign|0x7F): keeps LUT's preset negative sign when FP4 sign=1; clears to + when 0.
  const uint32_t s_even = (packed8 << 4) | 0x7F7F7F7Fu;
  const uint32_t s_odd = packed8 | 0x7F7F7F7Fu;
  const uint32_t fp8_es = fp8_e & s_even;
  const uint32_t fp8_os = fp8_o & s_odd;

  v2f e01 = amdgcn_cvt_pk_f32_fp8(fp8_es & 0xFFFFu);
  v2f e23 = amdgcn_cvt_pk_f32_fp8(fp8_es >> 16);
  v2f o01 = amdgcn_cvt_pk_f32_fp8(fp8_os & 0xFFFFu);
  v2f o23 = amdgcn_cvt_pk_f32_fp8(fp8_os >> 16);

  const float scl = e8m0_to_fp32(scale);
  float fp32[8] = {e01.a[0] * scl, o01.a[0] * scl, e01.a[1] * scl, o01.a[1] * scl,
                   e23.a[0] * scl, o23.a[0] * scl, e23.a[1] * scl, o23.a[1] * scl};

  // FP32 -> bf16 RNE.
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t u = __builtin_bit_cast(uint32_t, fp32[i]);
    uint32_t rne = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
    out[i] = __builtin_bit_cast(__hip_bfloat16, static_cast<uint16_t>(rne));
  }
}

}  // namespace mxfp4_cdna3
