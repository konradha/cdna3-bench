import os
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"

IS_ROCM = torch.version.hip is not None

if not IS_ROCM:
    print("WARNING: torch is not a ROCm build. Building stub ext.", file=sys.stderr)

ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
PYTORCH_ROCM_ARCH = os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")

sources = [str(CSRC / "mxfp4_ext.cpp")]
if IS_ROCM:
    sources += [
        str(CSRC / "strategy_a_two_pass.cu"),
        str(CSRC / "strategy_b_lds_dequant.cu"),
        str(CSRC / "strategy_c_interleaved.cu"),
        str(CSRC / "strategy_d_8wave.cu"),
        str(CSRC / "strategy_e_coverage.cu"),
        str(CSRC / "strategy_f_16wave.cu"),
        str(CSRC / "strategy_g_prepacked.cu"),
        str(CSRC / "strategy_h_swscale.cu"),
        str(CSRC / "strategy_i_swizzle.cu"),
        str(CSRC / "strategy_k_sched.cu"),
        str(CSRC / "strategy_j_synthesis.cu"),
        str(CSRC / "strategy_m_asm.cu"),
    ]
else:
    stub_path = CSRC / "stub_no_rocm.cpp"
    if not stub_path.exists():
        stub_path.write_text(
            '#include "common.h"\n'
            "namespace mxfp4_cdna3 {\n"
            "hipError_t launch_strategy_a(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_b(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_c(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_d(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_e(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_f(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_g(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_h(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_i(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_k(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_j(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "hipError_t launch_strategy_m(const void*, const uint8_t*, const uint8_t*,\n"
            "                             float*, int, int, int, hipStream_t)"
            " { return hipErrorNotSupported; }\n"
            "}\n"
        )
    sources.append(str(stub_path))

hipcc_flags = [
    "-O2",
    "-std=c++17",
    "-Wno-unused-function",
    "-Wno-deprecated-declarations",
    "-DUSE_ROCM",
    "-D__HIP_PLATFORM_AMD__",
    "-mllvm",
    "-greedy-reverse-local-assignment=1",
    "-mllvm",
    "--amdgpu-use-amdgpu-trackers=1",
]

ext_modules = [
    CUDAExtension(
        name="mxfp4_cdna3._C",
        sources=sources,
        include_dirs=[str(CSRC), str(Path(ROCM_PATH) / "include")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-Wno-unused-function"],
            "nvcc": hipcc_flags if IS_ROCM else ["-O3"],
        },
    )
]

setup(
    name="mxfp4_cdna3",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
    install_requires=["numpy>=1.26", "matplotlib>=3.8", "pandas>=2.1"],
    zip_safe=False,
)
