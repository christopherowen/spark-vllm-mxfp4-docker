// Diagnostic to print SharedStorage sizes for different tile configurations
// Compile: nvcc -std=c++17 -arch=sm_121 -I/workspace/flashinfer/3rdparty/cutlass/include \
//          -I/workspace/flashinfer/include -I/workspace/flashinfer/csrc -o print_smem_sizes print_smem_sizes.cu

#include <cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp>
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

// Minimal types for testing
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue8m0_t;
using ElementAccumulator = float;
using ElementOutput = cutlass::bfloat16_t;

template<int TileM, int TileN, int TileK, int Stages>
void print_smem_breakdown() {
    using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<_1, _1, _1>;
    
    // Standard dispatch policy
    using DispatchPolicy = cutlass::gemm::MainloopSm120ArrayTmaWarpSpecializedBlockScaled<
        Stages, 1, ClusterShape, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;
    
    // Compute layout sizes directly
    constexpr int smem_A_elements = TileM * TileK * Stages;
    constexpr int smem_B_elements = TileN * TileK * Stages;
    
    // Scale factors: based on group size 32
    constexpr int sf_group_size = 32;
    constexpr int sf_M_groups = (TileM + 31) / 32;  // atoms in M
    constexpr int sf_N_groups = (TileN + 31) / 32;  // atoms in N
    constexpr int sf_K_groups = TileK / sf_group_size;
    
    constexpr int smem_SFA_elements = sf_M_groups * sf_K_groups * Stages * 128;  // 128 SF per tile
    constexpr int smem_SFB_elements = sf_N_groups * sf_K_groups * Stages * 128;
    
    // Element sizes
    constexpr int sizeof_A = 1;  // FP8 = 1 byte
    constexpr int sizeof_B = 1;  // FP4 packed as bytes for addressing
    constexpr int sizeof_SF = 1; // E8M0 = 1 byte
    
    // Raw sizes
    int smem_A_bytes = smem_A_elements * sizeof_A;
    int smem_B_bytes = (TileN * TileK / 2) * Stages;  // FP4 is 4 bits, packed 2 per byte
    int smem_SFA_bytes = smem_SFA_elements * sizeof_SF;
    int smem_SFB_bytes = smem_SFB_elements * sizeof_SF;
    
    // Alignment padding (each array aligned to 128 or 1024 bytes)
    auto align_up = [](int size, int alignment) {
        return ((size + alignment - 1) / alignment) * alignment;
    };
    
    // Standard version uses alignas(1024) for A/B, 128 for SF
    int smem_A_aligned = align_up(smem_A_bytes, 1024);
    int smem_B_aligned = align_up(smem_B_bytes, 1024);
    int smem_SFA_aligned = align_up(smem_SFA_bytes, 128);
    int smem_SFB_aligned = align_up(smem_SFB_bytes, 128);
    
    int standard_total = smem_A_aligned + smem_B_aligned + smem_SFA_aligned + smem_SFB_aligned;
    
    // Gated version uses alignas(128) for all arrays
    int gated_smem_A = align_up(smem_A_bytes, 128);
    int gated_smem_B = align_up(smem_B_bytes, 128);
    int gated_smem_SFA = align_up(smem_SFA_bytes, 128);
    int gated_smem_SFB = align_up(smem_SFB_bytes, 128);
    int gated_smem_Aux = gated_smem_B;  // Same as B
    int gated_smem_SFAux = gated_smem_SFB;  // Same as SFB
    
    int gated_total = gated_smem_A + gated_smem_B + gated_smem_SFA + gated_smem_SFB + 
                      gated_smem_Aux + gated_smem_SFAux;
    
    std::cout << "\n=== Tile " << TileM << "x" << TileN << "x" << TileK 
              << ", Stages=" << Stages << " ===" << std::endl;
    std::cout << "Standard (4 arrays, 1024-byte alignment for A/B):" << std::endl;
    std::cout << "  smem_A:   " << smem_A_bytes << " -> " << smem_A_aligned << " B (aligned)" << std::endl;
    std::cout << "  smem_B:   " << smem_B_bytes << " -> " << smem_B_aligned << " B (aligned)" << std::endl;
    std::cout << "  smem_SFA: " << smem_SFA_bytes << " -> " << smem_SFA_aligned << " B (aligned)" << std::endl;
    std::cout << "  smem_SFB: " << smem_SFB_bytes << " -> " << smem_SFB_aligned << " B (aligned)" << std::endl;
    std::cout << "  TOTAL:    " << standard_total << " B (" << standard_total/1024.0 << " KB)" << std::endl;
    
    std::cout << "Gated (6 arrays, 128-byte alignment):" << std::endl;
    std::cout << "  smem_A:     " << smem_A_bytes << " -> " << gated_smem_A << " B" << std::endl;
    std::cout << "  smem_B:     " << smem_B_bytes << " -> " << gated_smem_B << " B" << std::endl;
    std::cout << "  smem_SFA:   " << smem_SFA_bytes << " -> " << gated_smem_SFA << " B" << std::endl;
    std::cout << "  smem_SFB:   " << smem_SFB_bytes << " -> " << gated_smem_SFB << " B" << std::endl;
    std::cout << "  smem_Aux:   " << smem_B_bytes << " -> " << gated_smem_Aux << " B" << std::endl;
    std::cout << "  smem_SFAux: " << smem_SFB_bytes << " -> " << gated_smem_SFAux << " B" << std::endl;
    std::cout << "  TOTAL:      " << gated_total << " B (" << gated_total/1024.0 << " KB)" << std::endl;
    
    std::cout << "Device limit: 101376 B (99.0 KB)" << std::endl;
    std::cout << "Gated fits? " << (gated_total <= 101376 ? "YES" : "NO") 
              << " (need to save " << (gated_total - 101376) << " B)" << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "SMEM SIZE ANALYSIS FOR GATED FC1" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    print_smem_breakdown<128, 128, 128, 2>();
    print_smem_breakdown<64, 128, 128, 2>();
    print_smem_breakdown<64, 64, 128, 2>();
    print_smem_breakdown<32, 128, 128, 2>();
    print_smem_breakdown<32, 64, 128, 2>();
    
    std::cout << "\n==================================================" << std::endl;
    std::cout << "OBSERVED ACTUAL VALUES (from runtime errors):" << std::endl;
    std::cout << "128x128x128: 129,024 B" << std::endl;
    std::cout << " 64x128x128: 112,640 B" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}
