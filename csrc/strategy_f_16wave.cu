// F: 256M x 128N x 64K, 16 waves at 8x2. Single-buf A (32 KiB) + double-buf B (32 KiB) = 64 KiB.
// 1 CTA/CU, 16 waves/CU. B-tile reused across 256 M rows (2x D).
// Pipeline: A reloaded once per k_outer; B[buf^1] dequant overlaps MFMA on B[buf].

#include "bf16_gemm.h"

namespace mxfp4_cdna3 {

constexpr int F_WAVES = 16;
constexpr int F_THREADS = F_WAVES * WAVE_SIZE;                         // 1024
constexpr int F_A_PASSES = (F_BLOCK_M * F_BLOCK_K) / (F_THREADS * 2);  // 8
constexpr int F_B_CHUNKS = (F_BLOCK_N * F_BLOCK_K) / 8;                // 1024
// F_B_CHUNKS == F_THREADS: one dequant chunk per thread.

__device__ __forceinline__ void f_load_a_tile(const bf16* __restrict__ A, bf16 (*lds)[F_BLOCK_K],
                                              int block_m, int k_offset, int M, int K) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int pass = 0; pass < F_A_PASSES; ++pass) {
    const int linear = pass * F_THREADS + tid;
    const int row = linear / (F_BLOCK_K / 2);
    const int col = (linear % (F_BLOCK_K / 2)) * 2;
    const int gm = block_m + row;
    const int gk = k_offset + col;
    if (gm < M && gk < K) {
      lds[row][col] = A[gm * K + gk];
      lds[row][col + 1] = A[gm * K + gk + 1];
    }
  }
}

__device__ __forceinline__ void f_dequant_b_tile(const uint8_t* __restrict__ B_packed,
                                                 const uint8_t* __restrict__ B_scales,
                                                 bf16 (*lds)[F_BLOCK_K], int block_n, int k_offset,
                                                 int N) {
  const int chunk = threadIdx.x;
  const int n_in_tile = chunk % F_BLOCK_N;
  const int k8 = (chunk / F_BLOCK_N) * 8;
  const int n_col = block_n + n_in_tile;
  const int k_global = k_offset + k8;
  const uint8_t scale = B_scales[(k_global / MX_GROUP_SIZE) * N + n_col];
  const int k_byte = k_global / 2;
  uint32_t packed = static_cast<uint32_t>(B_packed[(k_byte + 0) * N + n_col]) |
                    (static_cast<uint32_t>(B_packed[(k_byte + 1) * N + n_col]) << 8) |
                    (static_cast<uint32_t>(B_packed[(k_byte + 2) * N + n_col]) << 16) |
                    (static_cast<uint32_t>(B_packed[(k_byte + 3) * N + n_col]) << 24);
  bf16 out8[8];
  dequant_8_mxfp4_to_bf16(packed, scale, out8);
#pragma unroll
  for (int i = 0; i < 8; ++i) lds[n_in_tile][k8 + i] = out8[i];
}

__global__ __launch_bounds__(F_THREADS) void strategy_f_kernel(const bf16* __restrict__ A,
                                                               const uint8_t* __restrict__ B_packed,
                                                               const uint8_t* __restrict__ B_scales,
                                                               float* __restrict__ C, int M, int N,
                                                               int K, int num_xcds) {
  const int grid_m = M / F_BLOCK_M;
  const int grid_n = N / F_BLOCK_N;
  const int linear = blockIdx.y * grid_m + blockIdx.x;
  const int remapped = remap_xcd(linear, grid_m * grid_n, num_xcds);
  const int block_m = (remapped % grid_m) * F_BLOCK_M;
  const int block_n = (remapped / grid_m) * F_BLOCK_N;

  const int wave = threadIdx.x / WAVE_SIZE;
  const int wave_m_off = (wave / 2) * 32;
  const int wave_n_off = (wave % 2) * 64;

  __shared__ bf16 a_lds[F_BLOCK_M][F_BLOCK_K];
  __shared__ bf16 b_lds[2][F_BLOCK_N][F_BLOCK_K];

  v4f32 acc[2][4] = {};
  const int outer_iters = K / F_BLOCK_K;
  int buf = 0;

  f_load_a_tile(A, a_lds, block_m, 0, M, K);
  f_dequant_b_tile(B_packed, B_scales, b_lds[0], block_n, 0, N);
  __syncthreads();

  for (int k_outer = 0; k_outer < outer_iters; ++k_outer) {
    if (k_outer + 1 < outer_iters) {
      f_dequant_b_tile(B_packed, B_scales, b_lds[buf ^ 1], block_n, (k_outer + 1) * F_BLOCK_K, N);
    }

#pragma unroll
    for (int k_mid = 0; k_mid < 4; ++k_mid) {
      v4i16 a_op[2];
      a_op[0] = lds_load_a_16x16(a_lds, wave_m_off, 0, k_mid);
      a_op[1] = lds_load_a_16x16(a_lds, wave_m_off, 16, k_mid);
      v4i16 b_op[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b_op[ni] = lds_load_b_16x16(b_lds[buf], wave_n_off, ni * 16, k_mid);
      }
      mfma_outer_product(a_op, b_op, acc);
    }

    // A is single-buf: MFMA must finish reading a_lds before next k_outer overwrites it.
    __syncthreads();
    if (k_outer + 1 < outer_iters) {
      f_load_a_tile(A, a_lds, block_m, (k_outer + 1) * F_BLOCK_K, M, K);
    }
    buf ^= 1;
    __syncthreads();
  }

  store_acc_tile(C, acc, block_m, wave_m_off, block_n, wave_n_off, M, N);
}

hipError_t launch_strategy_f(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream) {
  if (!shape_supported_f(M, N, K)) return hipErrorInvalidValue;
  const int num_xcds = detect_num_xcds(stream);
  dim3 grid(M / F_BLOCK_M, N / F_BLOCK_N, 1);
  dim3 block(F_THREADS, 1, 1);
  strategy_f_kernel<<<grid, block, 0, stream>>>(static_cast<const bf16*>(A_bf16), B_packed_fp4,
                                                B_scales_e8m0, C, M, N, K, num_xcds);
  return hipGetLastError();
}

}  // namespace mxfp4_cdna3
