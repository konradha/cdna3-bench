#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace mxfp4_cdna3 {

// SW-scaled MXFP4: dequant FP4 -> BF16 with the *unscaled* E2M1 magnitude only.
// All 8 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} are exactly representable in BF16,
// so this skips the FP8 FNUZ + FP32 cvt + RNE path. The per-(K-group, N-col) E8M0
// scale is applied once per partial accumulator post-MFMA — mathematically equivalent
// to per-element scaling and matches CDNA4's v_mfma_scale_* semantics in software.

__device__ __forceinline__ uint32_t h_amdgcn_perm(uint32_t s1, uint32_t s0, uint32_t sel) {
  return __builtin_amdgcn_perm(s1, s0, sel);
}

// BF16 bit-patterns for E2M1 magnitudes (em = 0..7):
//   em=0: 0x0000  em=1: 0x3F00  em=2: 0x3F80  em=3: 0x3FC0
//   em=4: 0x4000  em=5: 0x4040  em=6: 0x4080  em=7: 0x40C0
// Packed as 4 low-bytes / 4 high-bytes per uint32 (em groups 0..3 and 4..7).
__device__ __forceinline__ void dequant_8_mxfp4_to_bf16_unscaled(uint32_t packed8,
                                                                 __hip_bfloat16* out) {
  constexpr uint32_t LUT_LO_S0 = 0xC0800000u;  // em 0..3 low-bytes
  constexpr uint32_t LUT_LO_S1 = 0xC0804000u;  // em 4..7 low-bytes
  constexpr uint32_t LUT_HI_S0 = 0x3F3F3F00u;  // em 0..3 high-bytes
  constexpr uint32_t LUT_HI_S1 = 0x40404040u;  // em 4..7 high-bytes

  const uint32_t em_lo = packed8 & 0x07070707u;         // even FP4 positions {0,2,4,6}
  const uint32_t em_hi = (packed8 & 0x70707070u) >> 4;  // odd  FP4 positions {1,3,5,7}

  // Lookup magnitude low/high bytes via v_perm_b32.
  const uint32_t mlo_lo = h_amdgcn_perm(LUT_LO_S1, LUT_LO_S0, em_lo);
  const uint32_t mlo_hi = h_amdgcn_perm(LUT_HI_S1, LUT_HI_S0, em_lo);
  const uint32_t mhi_lo = h_amdgcn_perm(LUT_LO_S1, LUT_LO_S0, em_hi);
  const uint32_t mhi_hi = h_amdgcn_perm(LUT_HI_S1, LUT_HI_S0, em_hi);

  // Interleave each em's low/high byte into bf16 pairs (2 bf16 per uint32).
  const uint32_t bf_lo01 = h_amdgcn_perm(mlo_hi, mlo_lo, 0x05010400u);
  const uint32_t bf_lo23 = h_amdgcn_perm(mlo_hi, mlo_lo, 0x07030602u);
  const uint32_t bf_hi01 = h_amdgcn_perm(mhi_hi, mhi_lo, 0x05010400u);
  const uint32_t bf_hi23 = h_amdgcn_perm(mhi_hi, mhi_lo, 0x07030602u);

  // Interleave even/odd FP4 positions to restore native order (0,1,2,3,4,5,6,7).
  const uint32_t out01 = h_amdgcn_perm(bf_hi01, bf_lo01, 0x05040100u);
  const uint32_t out23 = h_amdgcn_perm(bf_hi01, bf_lo01, 0x07060302u);
  const uint32_t out45 = h_amdgcn_perm(bf_hi23, bf_lo23, 0x05040100u);
  const uint32_t out67 = h_amdgcn_perm(bf_hi23, bf_lo23, 0x07060302u);

  // Sign bits: FP4 bit 3 / 7 / 11 / ... -> BF16 bit 15 / 31 / 47 / ... (within each uint32).
  const uint32_t s01 = ((packed8 << 12) & 0x00008000u) | ((packed8 << 24) & 0x80000000u);
  const uint32_t s23 = ((packed8 << 4) & 0x00008000u) | ((packed8 << 16) & 0x80000000u);
  const uint32_t s45 = ((packed8 >> 4) & 0x00008000u) | ((packed8 << 8) & 0x80000000u);
  const uint32_t s67 = ((packed8 >> 12) & 0x00008000u) | ((packed8) & 0x80000000u);

  reinterpret_cast<uint32_t*>(out)[0] = out01 ^ s01;
  reinterpret_cast<uint32_t*>(out)[1] = out23 ^ s23;
  reinterpret_cast<uint32_t*>(out)[2] = out45 ^ s45;
  reinterpret_cast<uint32_t*>(out)[3] = out67 ^ s67;
}

// E8M0 byte -> FP32 scale (shl-23 packs exponent, zero mantissa).
__device__ __forceinline__ float h_e8m0_to_fp32(uint8_t scale) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(scale) << 23);
}

}  // namespace mxfp4_cdna3
