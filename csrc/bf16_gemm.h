#pragma once

#include <hip/hip_bf16.h>

#include "common.h"
#include "strategy_c_dequant.h"

namespace mxfp4_cdna3 {

using bf16 = __hip_bfloat16;
using v4f32 = __attribute__((ext_vector_type(4))) float;
using v4i16 = __attribute__((ext_vector_type(4))) short;  // bf16 bit-pattern carrier for MFMA

constexpr int THREADS_PER_BLOCK = WAVE_SIZE * SC_WAVES;                             // 256
constexpr int A_TILE_PASSES = (SC_BLOCK_M * SC_BLOCK_K) / (THREADS_PER_BLOCK * 2);  // 8
constexpr int B_TILE_PASSES = (SC_BLOCK_N * SC_BLOCK_K) / (THREADS_PER_BLOCK * 2);  // 16

__device__ __forceinline__ v4f32 mfma_16x16x16_bf16(v4i16 a, v4i16 b, v4f32 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ v4i16 pack_bf16x4(const bf16* p) {
  v4i16 r;
  r[0] = __builtin_bit_cast(short, p[0]);
  r[1] = __builtin_bit_cast(short, p[1]);
  r[2] = __builtin_bit_cast(short, p[2]);
  r[3] = __builtin_bit_cast(short, p[3]);
  return r;
}

// MFMA A-operand: lane = m + 16*k_block; lane holds A[m][k_block*4 + 0..3]. LDS is [M][K].
__device__ __forceinline__ v4i16 lds_load_a_16x16(bf16 (*lds)[SC_BLOCK_K], int wave_m_off,
                                                  int m_mfma_off, int k_mid) {
  const int lane = threadIdx.x % WAVE_SIZE;
  const int row = wave_m_off + m_mfma_off + (lane % 16);
  const int k = (lane / 16) * 4 + k_mid * 16;
  v4i16 a;
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = __builtin_bit_cast(short, lds[row][k + i]);
  return a;
}

// MFMA B-operand: lane = n + 16*k_block; lane holds B[n][k_block*4 + 0..3]. LDS is [N][K].
__device__ __forceinline__ v4i16 lds_load_b_16x16(bf16 (*lds)[SC_BLOCK_K], int wave_n_off,
                                                  int n_mfma_off, int k_mid) {
  const int lane = threadIdx.x % WAVE_SIZE;
  const int row = wave_n_off + n_mfma_off + (lane % 16);
  const int k = (lane / 16) * 4 + k_mid * 16;
  v4i16 b;
#pragma unroll
  for (int i = 0; i < 4; ++i) b[i] = __builtin_bit_cast(short, lds[row][k + i]);
  return b;
}

// A global: row-major [M][K]. 256 threads x A_TILE_PASSES x 2 elt = full tile.
__device__ __forceinline__ void global_load_a_tile_to_lds(const bf16* __restrict__ A,
                                                          bf16 (*lds)[SC_BLOCK_K], int block_m,
                                                          int k_offset, int M, int K) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int pass = 0; pass < A_TILE_PASSES; ++pass) {
    const int linear = pass * THREADS_PER_BLOCK + tid;
    const int row = linear / (SC_BLOCK_K / 2);
    const int col = (linear % (SC_BLOCK_K / 2)) * 2;
    const int gm = block_m + row;
    const int gk = k_offset + col;
    if (gm < M && gk < K) {
      lds[row][col] = A[gm * K + gk];
      lds[row][col + 1] = A[gm * K + gk + 1];
    }
  }
}

// B global (pre-dequanted bf16): row-major [K][N].
__device__ __forceinline__ void global_load_b_bf16_tile_to_lds(const bf16* __restrict__ B,
                                                               bf16 (*lds)[SC_BLOCK_K], int block_n,
                                                               int k_offset, int N, int K) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int pass = 0; pass < B_TILE_PASSES; ++pass) {
    const int linear = pass * THREADS_PER_BLOCK + tid;
    const int n_in_tile = linear / (SC_BLOCK_K / 2);
    const int k_in_tile = (linear % (SC_BLOCK_K / 2)) * 2;
    const int gn = block_n + n_in_tile;
    const int gk = k_offset + k_in_tile;
    if (gn < N && gk < K) {
      lds[n_in_tile][k_in_tile] = B[gk * N + gn];
      lds[n_in_tile][k_in_tile + 1] = B[(gk + 1) * N + gn];
    }
  }
}

// 256 threads -> 128 (n_in_tile) x 2 (k_seg). Each thread: 4 dequant calls of 8 bf16 = 32 bf16.
__device__ __forceinline__ void dequant_b_tile_to_lds(const uint8_t* __restrict__ B_packed,
                                                      const uint8_t* __restrict__ B_scales,
                                                      bf16 (*lds)[SC_BLOCK_K], int block_n,
                                                      int k_offset, int N, int K) {
  const int tid = threadIdx.x;
  const int n_in_tile = tid % SC_BLOCK_N;
  const int k_seg = tid / SC_BLOCK_N;
  const int n_col = block_n + n_in_tile;
  if (n_col >= N) return;

  const int scale_row = (k_offset + k_seg * 32) / MX_GROUP_SIZE;
  const uint8_t scale = B_scales[scale_row * N + n_col];

  const int k_byte_base = (k_offset + k_seg * 32) / 2;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int k_byte = k_byte_base + q * 4;
    uint32_t packed = static_cast<uint32_t>(B_packed[(k_byte + 0) * N + n_col]) |
                      (static_cast<uint32_t>(B_packed[(k_byte + 1) * N + n_col]) << 8) |
                      (static_cast<uint32_t>(B_packed[(k_byte + 2) * N + n_col]) << 16) |
                      (static_cast<uint32_t>(B_packed[(k_byte + 3) * N + n_col]) << 24);
    bf16 out8[8];
    dequant_8_mxfp4_to_bf16(packed, scale, out8);
    const int k_lds = k_seg * 32 + q * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) lds[n_in_tile][k_lds + i] = out8[i];
  }
}

// Per wave: 2 mi (M=32) x 4 ni (N=64) acc tiles.
__device__ __forceinline__ void mfma_outer_product(v4i16 a_op[2], v4i16 b_op[4],
                                                   v4f32 (&acc)[2][4]) {
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      acc[mi][ni] = mfma_16x16x16_bf16(a_op[mi], b_op[ni], acc[mi][ni]);
    }
  }
}

// MFMA acc: 4 fp32/lane along M-tile rows; lane%16 = N-tile col.
__device__ __forceinline__ void store_acc_tile(float* __restrict__ C, const v4f32 (&acc)[2][4],
                                               int block_m, int wave_m_off, int block_n,
                                               int wave_n_off, int M, int N) {
  const int lane = threadIdx.x % WAVE_SIZE;
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int n_col = block_n + wave_n_off + ni * 16 + (lane % 16);
      const int m_base = block_m + wave_m_off + mi * 16 + (lane / 16) * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m_row = m_base + i;
        if (m_row < M && n_col < N) C[m_row * N + n_col] = acc[mi][ni][i];
      }
    }
  }
}

// MI300X: 304 CUs / 38 = 8 XCDs; MI300A: 228/38 = 6. Override via MXFP4_NUM_XCDS.
__host__ inline int detect_num_xcds(hipStream_t stream) {
  if (const char* e = std::getenv("MXFP4_NUM_XCDS")) {
    int v = std::atoi(e);
    if (v > 0) return v;
  }
  int dev = 0;
  (void)hipStreamGetDevice(stream, &dev);
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return 8;
  const int xcds = prop.multiProcessorCount / 38;
  return (xcds > 0) ? xcds : 8;
}

// XCD-locality remap (CK gemm_tile_partitioner). Fallback to linear when num_groups isn't a
// multiple of num_xcds — naive formula otherwise collides indices and leaves blocks unwritten.
__device__ __forceinline__ int remap_xcd(int linear_id, int num_groups, int num_xcds) {
  if (num_groups % num_xcds != 0) return linear_id;
  const int per_xcd = num_groups / num_xcds;
  return (linear_id % num_xcds) * per_xcd + (linear_id / num_xcds);
}

}  // namespace mxfp4_cdna3
