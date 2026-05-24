// G: 128M x 128N x 64K, 8 waves at 4x2, D's geometry + prepacked B for linear-coalesced HBM reads.
// B_prep layout: [n_tile_idx, chunk_idx (= K_byte/4), n_in_tile, 4 bytes].
// Per CTA per k_outer iter -> 8 contiguous chunks x 128 ni x 4 bytes = 4096 bytes linear in HBM.

#include "bf16_gemm.h"

namespace mxfp4_cdna3 {

constexpr int G_THREADS = G_WAVES * WAVE_SIZE;                                         // 512
constexpr int G_A_PASSES = (G_BLOCK_M * G_BLOCK_K) / (G_THREADS * 2);                  // 8
constexpr int G_CHUNKS_PER_ITER = G_BLOCK_K / 8;                                       // 8
constexpr int G_CHUNK_PAIRS_PER_THREAD = (G_CHUNKS_PER_ITER * G_BLOCK_N) / G_THREADS;  // 2

__device__ __forceinline__ void g_load_a_tile(const bf16* __restrict__ A, bf16 (*lds)[G_BLOCK_K],
                                              int block_m, int k_offset, int M, int K) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int pass = 0; pass < G_A_PASSES; ++pass) {
    const int linear = pass * G_THREADS + tid;
    const int row = linear / (G_BLOCK_K / 2);
    const int col = (linear % (G_BLOCK_K / 2)) * 2;
    const int gm = block_m + row;
    const int gk = k_offset + col;
    if (gm < M && gk < K) {
      lds[row][col] = A[gm * K + gk];
      lds[row][col + 1] = A[gm * K + gk + 1];
    }
  }
}

// Dequant a 128 x 64 tile from a contiguous 4096-byte slice of B_prep.
// tid -> (n_in_tile = tid % 128, chunk_pair = tid / 128 in {0,1,2,3}); each thread does 2 chunks.
__device__ __forceinline__ void g_dequant_b_tile(const uint8_t* __restrict__ B_tile,
                                                 const uint8_t* __restrict__ Bs_tile,
                                                 bf16 (*lds)[G_BLOCK_K]) {
  const int tid = threadIdx.x;
  const int n_in_tile = tid % G_BLOCK_N;
  const int chunk_pair = tid / G_BLOCK_N;   // 0..3
  const int scale_group = chunk_pair >> 1;  // chunks 0..3 -> group 0; 4..7 -> group 1
  const uint8_t scale = Bs_tile[scale_group * G_BLOCK_N + n_in_tile];

#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int chunk = chunk_pair * 2 + q;
    const uint8_t* src = B_tile + chunk * G_BLOCK_N * 4 + n_in_tile * 4;
    const uint32_t packed = *reinterpret_cast<const uint32_t*>(src);
    bf16 out8[8];
    dequant_8_mxfp4_to_bf16(packed, scale, out8);
    const int k_lds = chunk * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) lds[n_in_tile][k_lds + i] = out8[i];
  }
}

__global__ __launch_bounds__(G_THREADS) void strategy_g_kernel(const bf16* __restrict__ A,
                                                               const uint8_t* __restrict__ B_prep,
                                                               const uint8_t* __restrict__ Bs_prep,
                                                               float* __restrict__ C, int M, int N,
                                                               int K, int num_xcds) {
  const int grid_m = M / G_BLOCK_M;
  const int grid_n = N / G_BLOCK_N;
  const int linear = blockIdx.y * grid_m + blockIdx.x;
  const int remapped = remap_xcd(linear, grid_m * grid_n, num_xcds);
  const int block_m = (remapped % grid_m) * G_BLOCK_M;
  const int n_tile_idx = remapped / grid_m;
  const int block_n = n_tile_idx * G_BLOCK_N;

  const int wave = threadIdx.x / WAVE_SIZE;
  const int wave_m_off = (wave / 2) * 32;
  const int wave_n_off = (wave % 2) * 64;

  __shared__ bf16 a_lds[2][G_BLOCK_M][G_BLOCK_K];
  __shared__ bf16 b_lds[2][G_BLOCK_N][G_BLOCK_K];

  v4f32 acc[2][4] = {};
  const int outer_iters = K / G_BLOCK_K;
  int buf = 0;

  // Bytes per CTA across all K, then per k_outer iter.
  const size_t b_per_iter = static_cast<size_t>(G_CHUNKS_PER_ITER) * G_BLOCK_N * 4;  // 4096
  const size_t bs_per_iter = static_cast<size_t>(G_BLOCK_K / 32) * G_BLOCK_N;        // 256
  const size_t b_base = static_cast<size_t>(n_tile_idx) * (K / 8) * G_BLOCK_N * 4;
  const size_t bs_base = static_cast<size_t>(n_tile_idx) * (K / 32) * G_BLOCK_N;

  g_load_a_tile(A, a_lds[0], block_m, 0, M, K);
  g_dequant_b_tile(B_prep + b_base, Bs_prep + bs_base, b_lds[0]);
  __syncthreads();

  for (int k_outer = 0; k_outer < outer_iters; ++k_outer) {
    if (k_outer + 1 < outer_iters) {
      const size_t off = static_cast<size_t>(k_outer + 1);
      g_load_a_tile(A, a_lds[buf ^ 1], block_m, (k_outer + 1) * G_BLOCK_K, M, K);
      g_dequant_b_tile(B_prep + b_base + off * b_per_iter, Bs_prep + bs_base + off * bs_per_iter,
                       b_lds[buf ^ 1]);
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

hipError_t launch_strategy_g(const void* A_bf16, const uint8_t* B_packed_prep,
                             const uint8_t* B_scales_prep, float* C, int M, int N, int K,
                             hipStream_t stream) {
  if (!shape_supported_g(M, N, K)) return hipErrorInvalidValue;
  const int num_xcds = detect_num_xcds(stream);
  dim3 grid(M / G_BLOCK_M, N / G_BLOCK_N, 1);
  dim3 block(G_THREADS, 1, 1);
  strategy_g_kernel<<<grid, block, 0, stream>>>(static_cast<const bf16*>(A_bf16), B_packed_prep,
                                                B_scales_prep, C, M, N, K, num_xcds);
  return hipGetLastError();
}

}  // namespace mxfp4_cdna3
