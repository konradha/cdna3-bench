#pragma once
// Per-lane dword copy. The gfx940+ __builtin_amdgcn_global_load_lds path requires
// AS-qualified pointer types that HIP's generic-AS C++ pointers won't reinterpret_cast to
// on ROCm 7.2.1 clang; until the right addrspacecast pattern is verified for this toolchain
// we issue a plain VGPR-staged dword load, which clang lowers to a single GLOBAL_LOAD_DWORD
// followed by DS_WRITE_B32 (vs the 1-instruction GLOBAL_LOAD_LDS form).

#include <hip/hip_runtime.h>

#include <cstdint>

namespace mxfp4_cdna3 {

__device__ __forceinline__ void global_load_dword_to_lds(const void* __restrict__ gptr,
                                                         void* lptr) {
  *reinterpret_cast<uint32_t*>(lptr) = *reinterpret_cast<const uint32_t*>(gptr);
}

}  // namespace mxfp4_cdna3
