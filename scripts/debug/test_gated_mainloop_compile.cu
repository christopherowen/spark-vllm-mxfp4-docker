// Compile test for gated mainloop
// nvcc -std=c++17 -I/workspace/flashinfer/3rdparty/cutlass/include \
//      -I/workspace/flashinfer/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include \
//      --dry-run test_gated_mainloop_compile.cu

#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp>

// Include our gated extension
#include <cutlass_extensions/gemm/collective/sm120_blockscaled_mma_gated_array_tma.hpp>

// Basic smoke test - just verify the dispatch policy is defined
static_assert(
    cutlass::gemm::collective::MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated<
        4, 1, cute::Shape<cute::_1, cute::_1, cute::_1>, 
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::IsGated == true,
    "Gated dispatch policy should have IsGated=true"
);

int main() {
    return 0;
}
