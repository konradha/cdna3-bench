#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include "common.h"

namespace mxfp4_cdna3 {

namespace {

hipStream_t check_inputs(const torch::Tensor& A, const torch::Tensor& B_packed,
                         const torch::Tensor& B_scales, const torch::Tensor& C, int64_t M,
                         int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && B_packed.is_cuda() && B_scales.is_cuda() && C.is_cuda(),
              "all tensors must be on GPU");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bfloat16");
  TORCH_CHECK(B_packed.scalar_type() == at::kByte, "B_packed must be uint8");
  TORCH_CHECK(B_scales.scalar_type() == at::kByte, "B_scales must be uint8");
  TORCH_CHECK(C.scalar_type() == at::kFloat, "C must be float32");
  TORCH_CHECK(A.dim() == 2 && A.size(0) == M && A.size(1) == K, "A shape");
  TORCH_CHECK(B_packed.dim() == 2 && B_packed.size(0) == K / 2 && B_packed.size(1) == N,
              "B_packed shape");
  TORCH_CHECK(B_scales.dim() == 2 && B_scales.size(0) == K / 32 && B_scales.size(1) == N,
              "B_scales shape");
  TORCH_CHECK(C.dim() == 2 && C.size(0) == M && C.size(1) == N, "C shape");
  TORCH_CHECK(A.is_contiguous() && B_packed.is_contiguous() && B_scales.is_contiguous() &&
                  C.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
  return at::cuda::getCurrentCUDAStream();
}

using LaunchFn = hipError_t (*)(const void*, const uint8_t*, const uint8_t*, float*, int, int, int,
                                hipStream_t);

torch::Tensor dispatch(LaunchFn fn, const char* name, const torch::Tensor& A,
                       const torch::Tensor& B_packed, const torch::Tensor& B_scales,
                       c10::optional<torch::Tensor> C_opt) {
  const int64_t M = A.size(0), K = A.size(1), N = B_packed.size(1);
  torch::Tensor C =
      C_opt.has_value() ? *C_opt : torch::empty({M, N}, A.options().dtype(at::kFloat));
  auto stream = check_inputs(A, B_packed, B_scales, C, M, N, K);
  auto err = fn(A.data_ptr(), static_cast<const uint8_t*>(B_packed.data_ptr()),
                static_cast<const uint8_t*>(B_scales.data_ptr()), static_cast<float*>(C.data_ptr()),
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), stream);
  TORCH_CHECK(err == hipSuccess, name, ": ", hipGetErrorString(err));
  return C;
}

torch::Tensor gemm_a(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  return dispatch(launch_strategy_a, "launch_strategy_a", A, B_packed, B_scales, C);
}
torch::Tensor gemm_b(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  return dispatch(launch_strategy_b, "launch_strategy_b", A, B_packed, B_scales, C);
}
torch::Tensor gemm_c(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_c(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy C requires M%64==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_c, "launch_strategy_c", A, B_packed, B_scales, C);
}
torch::Tensor gemm_d(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_d(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy D requires M%128==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_d, "launch_strategy_d", A, B_packed, B_scales, C);
}
torch::Tensor gemm_e(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_e(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy E requires M%128==0, N%64==0, K%64==0");
  return dispatch(launch_strategy_e, "launch_strategy_e", A, B_packed, B_scales, C);
}
torch::Tensor gemm_f(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_f(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy F requires M%256==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_f, "launch_strategy_f", A, B_packed, B_scales, C);
}

torch::Tensor gemm_h(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_d(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy H requires M%128==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_h, "launch_strategy_h", A, B_packed, B_scales, C);
}

torch::Tensor gemm_i(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_d(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy I requires M%128==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_i, "launch_strategy_i", A, B_packed, B_scales, C);
}

torch::Tensor gemm_k(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_d(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy K requires M%128==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_k, "launch_strategy_k", A, B_packed, B_scales, C);
}

torch::Tensor gemm_j(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_d(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy J requires M%128==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_j, "launch_strategy_j", A, B_packed, B_scales, C);
}

torch::Tensor gemm_m(const torch::Tensor& A, const torch::Tensor& B_packed,
                     const torch::Tensor& B_scales, c10::optional<torch::Tensor> C) {
  TORCH_CHECK(shape_supported_d(static_cast<int>(A.size(0)), static_cast<int>(B_packed.size(1)),
                                static_cast<int>(A.size(1))),
              "Strategy M requires M%128==0, N%128==0, K%64==0");
  return dispatch(launch_strategy_m, "launch_strategy_m", A, B_packed, B_scales, C);
}

torch::Tensor gemm_g(const torch::Tensor& A, const torch::Tensor& B_prep,
                     const torch::Tensor& Bs_prep, c10::optional<torch::Tensor> C_opt) {
  const int64_t M = A.size(0), K = A.size(1);
  TORCH_CHECK(A.is_cuda() && B_prep.is_cuda() && Bs_prep.is_cuda(), "all tensors on GPU");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A bf16");
  TORCH_CHECK(B_prep.scalar_type() == at::kByte, "B_prep uint8");
  TORCH_CHECK(Bs_prep.scalar_type() == at::kByte, "Bs_prep uint8");
  TORCH_CHECK(K % 2 == 0 && B_prep.numel() % (K / 2) == 0,
              "B_prep size not consistent with K/2 byte stride");
  const int64_t N = B_prep.numel() / (K / 2);
  TORCH_CHECK(Bs_prep.numel() == K * N / 32, "Bs_prep size != K*N/32");
  TORCH_CHECK(shape_supported_g(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)),
              "Strategy G requires M%128==0, N%128==0, K%64==0");
  TORCH_CHECK(A.is_contiguous() && B_prep.is_contiguous() && Bs_prep.is_contiguous(),
              "tensors contiguous");
  torch::Tensor C =
      C_opt.has_value() ? *C_opt : torch::empty({M, N}, A.options().dtype(at::kFloat));
  TORCH_CHECK(C.scalar_type() == at::kFloat && C.dim() == 2 && C.size(0) == M && C.size(1) == N &&
                  C.is_contiguous(),
              "C shape/contig");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto err = launch_strategy_g(A.data_ptr(), static_cast<const uint8_t*>(B_prep.data_ptr()),
                               static_cast<const uint8_t*>(Bs_prep.data_ptr()),
                               static_cast<float*>(C.data_ptr()), static_cast<int>(M),
                               static_cast<int>(N), static_cast<int>(K), stream);
  TORCH_CHECK(err == hipSuccess, "launch_strategy_g: ", hipGetErrorString(err));
  return C;
}

py::dict device_info(int64_t dev_id) {
  hipDeviceProp_t prop{};
  auto err = hipGetDeviceProperties(&prop, static_cast<int>(dev_id));
  TORCH_CHECK(err == hipSuccess, "hipGetDeviceProperties: ", hipGetErrorString(err));
  py::dict d;
  d["name"] = std::string(prop.name);
  d["gcn_arch_name"] = std::string(prop.gcnArchName);
  d["multi_processor_count"] = prop.multiProcessorCount;
  d["total_global_mem"] = static_cast<int64_t>(prop.totalGlobalMem);
  d["shared_mem_per_block"] = static_cast<int64_t>(prop.sharedMemPerBlock);
  d["warp_size"] = prop.warpSize;
  return d;
}

bool shape_supported_c_py(int64_t M, int64_t N, int64_t K) {
  return shape_supported_c(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}
bool shape_supported_d_py(int64_t M, int64_t N, int64_t K) {
  return shape_supported_d(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}
bool shape_supported_e_py(int64_t M, int64_t N, int64_t K) {
  return shape_supported_e(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}
bool shape_supported_f_py(int64_t M, int64_t N, int64_t K) {
  return shape_supported_f(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}
bool shape_supported_g_py(int64_t M, int64_t N, int64_t K) {
  return shape_supported_g(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

}  // namespace
}  // namespace mxfp4_cdna3

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_a", &mxfp4_cdna3::gemm_a, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_b", &mxfp4_cdna3::gemm_b, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_c", &mxfp4_cdna3::gemm_c, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_d", &mxfp4_cdna3::gemm_d, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_e", &mxfp4_cdna3::gemm_e, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_f", &mxfp4_cdna3::gemm_f, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_g", &mxfp4_cdna3::gemm_g, py::arg("A"), py::arg("B_prep"), py::arg("Bs_prep"),
        py::arg("C") = py::none());
  m.def("gemm_h", &mxfp4_cdna3::gemm_h, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_i", &mxfp4_cdna3::gemm_i, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_k", &mxfp4_cdna3::gemm_k, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_j", &mxfp4_cdna3::gemm_j, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("gemm_m", &mxfp4_cdna3::gemm_m, py::arg("A"), py::arg("B_packed"), py::arg("B_scales"),
        py::arg("C") = py::none());
  m.def("device_info", &mxfp4_cdna3::device_info, py::arg("dev_id") = 0);
  m.def("shape_supported_c", &mxfp4_cdna3::shape_supported_c_py, py::arg("M"), py::arg("N"),
        py::arg("K"));
  m.def("shape_supported_d", &mxfp4_cdna3::shape_supported_d_py, py::arg("M"), py::arg("N"),
        py::arg("K"));
  m.def("shape_supported_e", &mxfp4_cdna3::shape_supported_e_py, py::arg("M"), py::arg("N"),
        py::arg("K"));
  m.def("shape_supported_f", &mxfp4_cdna3::shape_supported_f_py, py::arg("M"), py::arg("N"),
        py::arg("K"));
  m.def("shape_supported_g", &mxfp4_cdna3::shape_supported_g_py, py::arg("M"), py::arg("N"),
        py::arg("K"));
}
