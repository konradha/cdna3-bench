#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace mxfp4_cdna3 {

constexpr int MX_GROUP_SIZE = 32;
constexpr int WAVE_SIZE = 64;

// A/B/C: 64M x 128N x 64K, 4 waves at 2x2 (per-wave 32M x 64N = 2x4 MFMA tiles).
constexpr int SC_BLOCK_M = 64;
constexpr int SC_BLOCK_N = 128;
constexpr int SC_BLOCK_K = 64;
constexpr int SC_WAVES = 4;
constexpr int SC_MFMA_M = 16;
constexpr int SC_MFMA_N = 16;
constexpr int SC_MFMA_K = 16;  // gfx942: v_mfma_f32_16x16x16_bf16_1k

// D: 128M x 128N x 64K, 8 waves at 4x2 -> 1 CTA/CU, 8 waves/CU.
constexpr int D_BLOCK_M = 128;
constexpr int D_BLOCK_N = 128;
constexpr int D_BLOCK_K = 64;
constexpr int D_WAVES = 8;

// E: 128M x 64N x 64K, 4 waves M-stacked. Coverage for low-CTA grids (small N).
constexpr int E_BLOCK_M = 128;
constexpr int E_BLOCK_N = 64;
constexpr int E_BLOCK_K = 64;

// F: 256M x 128N x 64K, 16 waves at 8x2. Single-buf A (32 KiB), double-buf B (32 KiB).
// B-tile reused across 256 M rows = 2x D's reuse; 16 waves/CU -> highest MFMA density target.
constexpr int F_BLOCK_M = 256;
constexpr int F_BLOCK_N = 128;
constexpr int F_BLOCK_K = 64;

// G: 128M x 128N x 64K, 8 waves at 4x2. Same geometry as D but consumes a prepacked B layout
// [n_tile_idx, chunk_idx, n_in_tile, byte] so each CTA reads a contiguous HBM range per k_outer.
constexpr int G_BLOCK_M = 128;
constexpr int G_BLOCK_N = 128;
constexpr int G_BLOCK_K = 64;
constexpr int G_WAVES = 8;

hipError_t launch_strategy_a(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_b(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_c(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_d(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_e(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_f(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_g(const void* A_bf16, const uint8_t* B_packed_prep,
                             const uint8_t* B_scales_prep, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_h(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_i(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_k(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_j(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);
hipError_t launch_strategy_m(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream);

inline bool shape_supported_c(int M, int N, int K) {
  return (M % SC_BLOCK_M == 0) && (N % SC_BLOCK_N == 0) && (K % SC_BLOCK_K == 0) &&
         (K % MX_GROUP_SIZE == 0);
}
inline bool shape_supported_d(int M, int N, int K) {
  return (M % D_BLOCK_M == 0) && (N % D_BLOCK_N == 0) && (K % D_BLOCK_K == 0) &&
         (K % MX_GROUP_SIZE == 0);
}
inline bool shape_supported_e(int M, int N, int K) {
  return (M % E_BLOCK_M == 0) && (N % E_BLOCK_N == 0) && (K % E_BLOCK_K == 0) &&
         (K % MX_GROUP_SIZE == 0);
}
inline bool shape_supported_f(int M, int N, int K) {
  return (M % F_BLOCK_M == 0) && (N % F_BLOCK_N == 0) && (K % F_BLOCK_K == 0) &&
         (K % MX_GROUP_SIZE == 0);
}
inline bool shape_supported_g(int M, int N, int K) {
  return (M % G_BLOCK_M == 0) && (N % G_BLOCK_N == 0) && (K % G_BLOCK_K == 0) &&
         (K % MX_GROUP_SIZE == 0);
}

}  // namespace mxfp4_cdna3
