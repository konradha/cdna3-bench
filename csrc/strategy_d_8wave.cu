// 8-wave 128M x 128N x 64K LDS-staged fused dequant GEMM. B dequant reused across 2x M wave-rows.

#include "bf16_gemm.h"

namespace mxfp4_cdna3 {

constexpr int D_THREADS = D_WAVES * WAVE_SIZE;                         // 512
constexpr int D_A_PASSES = (D_BLOCK_M * D_BLOCK_K) / (D_THREADS * 2);  // 8
constexpr int D_B_CHUNKS = (D_BLOCK_N * D_BLOCK_K) / 8;                // 1024
constexpr int D_B_CHUNKS_PER_THREAD = D_B_CHUNKS / D_THREADS;          // 2

__device__ __forceinline__ void d_load_a_tile(const bf16* __restrict__ A, bf16 (*lds)[D_BLOCK_K],
                                              int block_m, int k_offset, int M, int K) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int pass = 0; pass < D_A_PASSES; ++pass) {
    const int linear = pass * D_THREADS + tid;
    const int row = linear / (D_BLOCK_K / 2);
    const int col = (linear % (D_BLOCK_K / 2)) * 2;
    const int gm = block_m + row;
    const int gk = k_offset + col;
    if (gm < M && gk < K) {
      lds[row][col] = A[gm * K + gk];
      lds[row][col + 1] = A[gm * K + gk + 1];
    }
  }
}

// Chunk-based: 1024 (n, k8) chunks total, 2/thread. n = chunk%128, k8 = (chunk/128)*8.
__device__ __forceinline__ void d_dequant_b_tile(const uint8_t* __restrict__ B_packed,
                                                 const uint8_t* __restrict__ B_scales,
                                                 bf16 (*lds)[D_BLOCK_K], int block_n, int k_offset,
                                                 int N) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int slot = 0; slot < D_B_CHUNKS_PER_THREAD; ++slot) {
    const int chunk = tid + slot * D_THREADS;
    const int n_in_tile = chunk % D_BLOCK_N;
    const int k8 = (chunk / D_BLOCK_N) * 8;
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
}

// 8 waves at 4x2: wave_m_off in {0,32,64,96}, wave_n_off in {0,64}. Per wave: 32M x 64N (2x4 MFMA).
__global__ __launch_bounds__(D_THREADS) void strategy_d_kernel(const bf16* __restrict__ A,
                                                               const uint8_t* __restrict__ B_packed,
                                                               const uint8_t* __restrict__ B_scales,
                                                               float* __restrict__ C, int M, int N,
                                                               int K, int num_xcds) {
  const int grid_m = M / D_BLOCK_M;
  const int grid_n = N / D_BLOCK_N;
  const int linear = blockIdx.y * grid_m + blockIdx.x;
  const int remapped = remap_xcd(linear, grid_m * grid_n, num_xcds);
  const int block_m = (remapped % grid_m) * D_BLOCK_M;
  const int block_n = (remapped / grid_m) * D_BLOCK_N;

  const int wave = threadIdx.x / WAVE_SIZE;
  const int wave_m_off = (wave / 2) * 32;
  const int wave_n_off = (wave % 2) * 64;

  __shared__ bf16 a_lds[2][D_BLOCK_M][D_BLOCK_K];
  __shared__ bf16 b_lds[2][D_BLOCK_N][D_BLOCK_K];

  v4f32 acc[2][4] = {};
  const int outer_iters = K / D_BLOCK_K;
  int buf = 0;

  d_load_a_tile(A, a_lds[0], block_m, 0, M, K);
  d_dequant_b_tile(B_packed, B_scales, b_lds[0], block_n, 0, N);
  __syncthreads();

  for (int k_outer = 0; k_outer < outer_iters; ++k_outer) {
    const int k_offset = k_outer * D_BLOCK_K;
    if (k_outer + 1 < outer_iters) {
      d_load_a_tile(A, a_lds[buf ^ 1], block_m, k_offset + D_BLOCK_K, M, K);
      d_dequant_b_tile(B_packed, B_scales, b_lds[buf ^ 1], block_n, k_offset + D_BLOCK_K, N);
    }

#pragma unroll
    for (int k_mid = 0; k_mid < 4; ++k_mid) {
      v4i16 a_op[2];
      a_op[0] = lds_load_a_16x16(a_lds[buf], wave_m_off, 0, k_mid);
      a_op[1] = lds_load_a_16x16(a_lds[buf], wave_m_off, 16, k_mid);

      v4i16 b_op[4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        b_op[ni] = lds_load_b_16x16(b_lds[buf], wave_n_off, ni * 16, k_mid);
      }
      mfma_outer_product(a_op, b_op, acc);
    }

    buf ^= 1;
    __syncthreads();
  }

  store_acc_tile(C, acc, block_m, wave_m_off, block_n, wave_n_off, M, N);
}

hipError_t launch_strategy_d(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream) {
  if (!shape_supported_d(M, N, K)) return hipErrorInvalidValue;
  const int num_xcds = detect_num_xcds(stream);
  dim3 grid(M / D_BLOCK_M, N / D_BLOCK_N, 1);
  dim3 block(D_THREADS, 1, 1);
  strategy_d_kernel<<<grid, block, 0, stream>>>(static_cast<const bf16*>(A_bf16), B_packed_fp4,
                                                B_scales_e8m0, C, M, N, K, num_xcds);
  return hipGetLastError();
}

}  // namespace mxfp4_cdna3
