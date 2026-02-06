// Compile test for gated FC1 kernel infrastructure
//
// Build command (inside container):
// nvcc -std=c++17 -E -x cu \
//   -DLOGICAL_TILE_M=64 -DLOGICAL_TILE_N=128 -DSWAP_AB=0 \
//   -I/workspace/flashinfer/3rdparty/cutlass/include \
//   -I/workspace/flashinfer/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include \
//   -I/workspace/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers \
//   test_gated_kernel_compile.cu

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>

// Include our gated infrastructure
#include "cutlass_extensions/gemm/collective/sm120_blockscaled_mma_gated_array_tma.hpp"
#include "cutlass_extensions/epilogue/sm120_gated_swiglu_epilogue.hpp"
#include "cutlass_extensions/gemm/kernel/sm120_gemm_gated_array_tma_warpspecialized.hpp"

// Verify dispatch policy is defined correctly
static_assert(
    cutlass::gemm::collective::MainloopSm120ArrayTmaWarpSpecializedBlockScaledGated<
        4, 1, cute::Shape<cute::_1, cute::_1, cute::_1>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::IsGated == true,
    "Gated dispatch policy should have IsGated=true"
);

// Verify AccumPair is defined
using TestAccum = float;
using TestAccumPair = cutlass::gemm::kernel::AccumPair<TestAccum>;

static_assert(sizeof(TestAccumPair) == 2 * sizeof(TestAccum),
    "AccumPair should contain two accumulators");

// Verify epilogue is defined
using TestEpilogue = cutlass::epilogue::Sm120GatedSwiGLUEpilogue<
    cute::Shape<cute::_64, cute::_128, cute::_128>,
    cutlass::bfloat16_t,
    cute::Stride<cute::_1, int64_t>,
    float
>;

int main() {
    // Just a compile test - verify types instantiate
    TestAccumPair accum_pair;
    accum_pair.clear();
    
    return 0;
}
