// Two-pass: dequant B to HBM bf16 scratch, then plain bf16 MFMA gemm. Zero overlap.

#include "bf16_gemm.h"

namespace mxfp4_cdna3 {

// B_packed[K/2,N] + B_scales[K/32,N] -> B_bf16[K,N]. Block: 128 N-cols x 32 K-rows (one MX group).
__global__ void dequant_b_kernel(const uint8_t* __restrict__ B_packed,
                                 const uint8_t* __restrict__ B_scales, bf16* __restrict__ B_bf16,
                                 int N, int K) {
  const int n_base = blockIdx.x * 128;
  const int k_base = blockIdx.y * 32;
  const int tid = threadIdx.x;
  const int n_in_tile = tid % 128;
  const int k_chunk8 = tid / 128;  // 0..1
  const int n_col = n_base + n_in_tile;
  if (n_col >= N) return;

  const uint8_t scale = B_scales[(k_base / MX_GROUP_SIZE) * N + n_col];

#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int k_in_group = k_chunk8 * 16 + half * 8;
    if (k_base + k_in_group + 7 >= K) break;
    const int k_byte = (k_base + k_in_group) / 2;
    uint32_t packed = static_cast<uint32_t>(B_packed[(k_byte + 0) * N + n_col]) |
                      (static_cast<uint32_t>(B_packed[(k_byte + 1) * N + n_col]) << 8) |
                      (static_cast<uint32_t>(B_packed[(k_byte + 2) * N + n_col]) << 16) |
                      (static_cast<uint32_t>(B_packed[(k_byte + 3) * N + n_col]) << 24);
    bf16 out8[8];
    dequant_8_mxfp4_to_bf16(packed, scale, out8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      B_bf16[(k_base + k_in_group + i) * N + n_col] = out8[i];
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void bf16_gemm_kernel(const bf16* __restrict__ A,
                                                                      const bf16* __restrict__ B,
                                                                      float* __restrict__ C, int M,
                                                                      int N, int K, int num_xcds) {
  const int grid_m = M / SC_BLOCK_M;
  const int grid_n = N / SC_BLOCK_N;
  const int linear = blockIdx.y * grid_m + blockIdx.x;
  const int remapped = remap_xcd(linear, grid_m * grid_n, num_xcds);
  const int block_m = (remapped % grid_m) * SC_BLOCK_M;
  const int block_n = (remapped / grid_m) * SC_BLOCK_N;

  const int wave = threadIdx.x / WAVE_SIZE;
  const int wave_m_off = (wave / 2) * 32;
  const int wave_n_off = (wave % 2) * 64;

  __shared__ bf16 a_lds[2][SC_BLOCK_M][SC_BLOCK_K];
  __shared__ bf16 b_lds[2][SC_BLOCK_N][SC_BLOCK_K];

  v4f32 acc[2][4] = {};
  const int outer_iters = K / SC_BLOCK_K;
  int buf = 0;

  global_load_a_tile_to_lds(A, a_lds[0], block_m, 0, M, K);
  global_load_b_bf16_tile_to_lds(B, b_lds[0], block_n, 0, N, K);
  __syncthreads();

  for (int k_outer = 0; k_outer < outer_iters; ++k_outer) {
    const int k_offset = k_outer * SC_BLOCK_K;
    if (k_outer + 1 < outer_iters) {
      global_load_a_tile_to_lds(A, a_lds[buf ^ 1], block_m, k_offset + SC_BLOCK_K, M, K);
      global_load_b_bf16_tile_to_lds(B, b_lds[buf ^ 1], block_n, k_offset + SC_BLOCK_K, N, K);
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

hipError_t launch_strategy_a(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream) {
  if (!shape_supported_c(M, N, K)) return hipErrorInvalidValue;

  void* B_bf16_raw = nullptr;
  const size_t bf16_bytes = static_cast<size_t>(K) * N * sizeof(uint16_t);
  hipError_t e = hipMallocAsync(&B_bf16_raw, bf16_bytes, stream);
  if (e != hipSuccess) return e;

  dim3 d_grid((N + 127) / 128, K / MX_GROUP_SIZE, 1);
  dim3 d_block(256, 1, 1);
  dequant_b_kernel<<<d_grid, d_block, 0, stream>>>(B_packed_fp4, B_scales_e8m0,
                                                   static_cast<bf16*>(B_bf16_raw), N, K);

  const int num_xcds = detect_num_xcds(stream);
  dim3 g_grid(M / SC_BLOCK_M, N / SC_BLOCK_N, 1);
  dim3 g_block(THREADS_PER_BLOCK, 1, 1);
  bf16_gemm_kernel<<<g_grid, g_block, 0, stream>>>(
      static_cast<const bf16*>(A_bf16), static_cast<const bf16*>(B_bf16_raw), C, M, N, K, num_xcds);

  hipError_t free_err = hipFreeAsync(B_bf16_raw, stream);
  hipError_t last = hipGetLastError();
  return (last != hipSuccess) ? last : free_err;
}

}  // namespace mxfp4_cdna3
