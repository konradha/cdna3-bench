// C: B + sched_group_barrier hints.

#include "bf16_gemm.h"

namespace mxfp4_cdna3 {

// sched_group_barrier masks (LLVM AMDGPU User Guide).
constexpr int SGB_VALU = 0x002;
constexpr int SGB_MFMA = 0x008;
constexpr int SGB_VMEM_READ = 0x020;
constexpr int SGB_DS_READ = 0x100;

__global__ __launch_bounds__(THREADS_PER_BLOCK) void strategy_c_kernel(
    const bf16* __restrict__ A, const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ B_scales, float* __restrict__ C, int M, int N, int K,
    int num_xcds) {
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
  dequant_b_tile_to_lds(B_packed, B_scales, b_lds[0], block_n, 0, N, K);
  __syncthreads();

  for (int k_outer = 0; k_outer < outer_iters; ++k_outer) {
    const int k_offset = k_outer * SC_BLOCK_K;
    if (k_outer + 1 < outer_iters) {
      global_load_a_tile_to_lds(A, a_lds[buf ^ 1], block_m, k_offset + SC_BLOCK_K, M, K);
      __builtin_amdgcn_sched_group_barrier(SGB_VMEM_READ, 8, 0);
      dequant_b_tile_to_lds(B_packed, B_scales, b_lds[buf ^ 1], block_n, k_offset + SC_BLOCK_K, N,
                            K);
      __builtin_amdgcn_sched_group_barrier(SGB_VALU, 16, 0);
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
      __builtin_amdgcn_sched_group_barrier(SGB_DS_READ, 6, 0);

      mfma_outer_product(a_op, b_op, acc);
      __builtin_amdgcn_sched_group_barrier(SGB_MFMA, 8, 0);
    }

    buf ^= 1;
    __syncthreads();
  }

  store_acc_tile(C, acc, block_m, wave_m_off, block_n, wave_n_off, M, N);
}

hipError_t launch_strategy_c(const void* A_bf16, const uint8_t* B_packed_fp4,
                             const uint8_t* B_scales_e8m0, float* C, int M, int N, int K,
                             hipStream_t stream) {
  if (!shape_supported_c(M, N, K)) return hipErrorInvalidValue;
  const int num_xcds = detect_num_xcds(stream);
  dim3 grid(M / SC_BLOCK_M, N / SC_BLOCK_N, 1);
  dim3 block(THREADS_PER_BLOCK, 1, 1);
  strategy_c_kernel<<<grid, block, 0, stream>>>(static_cast<const bf16*>(A_bf16), B_packed_fp4,
                                                B_scales_e8m0, C, M, N, K, num_xcds);
  return hipGetLastError();
}

}  // namespace mxfp4_cdna3
