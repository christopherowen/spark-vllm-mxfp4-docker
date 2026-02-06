#!/usr/bin/env python3
"""
Diagnostic tool to analyze shared memory usage in the gated FC1 kernel.

This script computes the theoretical SMEM sizes for different tile configurations
to help identify wastage and find a working configuration.
"""

import math

def compute_smem_sizes(tile_m, tile_n, tile_k, stages=2, element_bytes=1, sf_bytes=1, group_size=32):
    """
    Compute shared memory sizes for gated MXFP4 mainloop.
    
    SMEM arrays (gated has 6 vs 4 for standard):
    - smem_A: activations (FP8, 1 byte per element)
    - smem_B: linear weights (FP4, 0.5 bytes per element, but stored as bytes)
    - smem_SFA: scale factors for A
    - smem_SFB: scale factors for B
    - smem_Aux: gate weights (FP4)
    - smem_SFAux: scale factors for Aux
    
    For block-scaled, each 32 elements share one scale factor.
    """
    
    # Each stage needs to hold one tile
    # For FP8 activations: M x K elements per stage
    # For FP4 weights: K x N elements per stage (packed 2 per byte)
    
    # A (activations): FP8, M x K per stage
    smem_A_per_stage = tile_m * tile_k * element_bytes  # FP8 = 1 byte
    
    # B (weights): FP4 packed, K x N per stage
    # FP4 is 4 bits, so 2 elements per byte
    smem_B_per_stage = (tile_k * tile_n) // 2  # FP4 packed
    
    # Scale factors: one per group_size elements
    # SFA: covers M x K tile, but scales are per-K-group
    # Actually for block-scaled, the layout is more complex
    # Let's assume: (M / atom_m) * (K / group_size) scale factors
    # For SM120, atom is typically 32x32
    
    # Scale factor dimensions depend on the block scaling layout
    # Typically: ceil(tile_k / group_size) * ceil(tile_m / 32) for A
    #           ceil(tile_k / group_size) * ceil(tile_n / 32) for B
    
    sf_k_blocks = tile_k // group_size
    sf_m_blocks = tile_m // 32  # 32 is typical atom size
    sf_n_blocks = tile_n // 32
    
    smem_SFA_per_stage = sf_k_blocks * sf_m_blocks * 128  # 128 bytes for scale factor tile
    smem_SFB_per_stage = sf_k_blocks * sf_n_blocks * 128
    
    # Aux (gate weights): same as B
    smem_Aux_per_stage = smem_B_per_stage
    smem_SFAux_per_stage = smem_SFB_per_stage
    
    # Total per stage
    standard_per_stage = smem_A_per_stage + smem_B_per_stage + smem_SFA_per_stage + smem_SFB_per_stage
    gated_per_stage = standard_per_stage + smem_Aux_per_stage + smem_SFAux_per_stage
    
    # Total with staging
    standard_total = standard_per_stage * stages
    gated_total = gated_per_stage * stages
    
    # Add alignment overhead (each array aligned to 128 bytes)
    # 4 arrays for standard, 6 for gated
    alignment_overhead_standard = 4 * 128 * stages
    alignment_overhead_gated = 6 * 128 * stages
    
    # TensorMap storage (fixed overhead)
    tensormap_standard = 4 * 64  # 4 TMA descriptors, ~64 bytes each
    tensormap_gated = 6 * 64
    
    # Pipeline storage (barriers, etc.)
    pipeline_overhead = 256 * stages
    
    return {
        'tile': f'{tile_m}x{tile_n}x{tile_k}',
        'stages': stages,
        'smem_A_per_stage': smem_A_per_stage,
        'smem_B_per_stage': smem_B_per_stage,
        'smem_SFA_per_stage': smem_SFA_per_stage,
        'smem_SFB_per_stage': smem_SFB_per_stage,
        'smem_Aux_per_stage': smem_Aux_per_stage,
        'smem_SFAux_per_stage': smem_SFAux_per_stage,
        'standard_per_stage': standard_per_stage,
        'gated_per_stage': gated_per_stage,
        'standard_total_raw': standard_per_stage * stages,
        'gated_total_raw': gated_per_stage * stages,
        'standard_total_with_overhead': standard_total + alignment_overhead_standard + tensormap_standard + pipeline_overhead,
        'gated_total_with_overhead': gated_total + alignment_overhead_gated + tensormap_gated + pipeline_overhead,
    }


def main():
    device_max_smem = 101376  # 101KB for SM121
    
    print("=" * 80)
    print("GATED FC1 SMEM ANALYSIS")
    print(f"Device max shared memory: {device_max_smem} bytes ({device_max_smem / 1024:.1f} KB)")
    print("=" * 80)
    
    # Test various tile configurations
    configs = [
        (128, 128, 128, 2),  # Standard tile
        (128, 128, 128, 1),  # Single stage (if allowed)
        (64, 128, 128, 2),
        (128, 64, 128, 2),
        (64, 64, 128, 2),
        (64, 64, 128, 1),
        (32, 128, 128, 2),
        (32, 64, 128, 2),
        (128, 128, 64, 2),  # Reduce K (may break TMA)
    ]
    
    print("\nTheoretical SMEM estimates (simplified model):")
    print("-" * 80)
    print(f"{'Tile':<16} {'Stages':<7} {'Standard':<12} {'Gated':<12} {'Fits?':<6} {'Overhead':<10}")
    print("-" * 80)
    
    for tile_m, tile_n, tile_k, stages in configs:
        result = compute_smem_sizes(tile_m, tile_n, tile_k, stages)
        gated = result['gated_total_with_overhead']
        standard = result['standard_total_with_overhead']
        fits = "YES" if gated <= device_max_smem else "NO"
        overhead = gated - result['gated_total_raw']
        
        print(f"{result['tile']:<16} {stages:<7} {standard:>10} B  {gated:>10} B  {fits:<6} {overhead:>8} B")
    
    print("\n" + "=" * 80)
    print("ACTUAL SMEM VALUES FROM RUNTIME (observed):")
    print("-" * 80)
    print("128x128x128, stages=2: SharedStorageSize=129,024 B (from error)")
    print(" 64x128x128, stages=2: SharedStorageSize=112,640 B (from error)")
    print(" 64x 64x128, stages=2: ? (illegal instruction, not SMEM)")
    print("-" * 80)
    
    # Reverse engineer the actual formula
    print("\n" + "=" * 80)
    print("REVERSE ENGINEERING ACTUAL SMEM FORMULA:")
    print("-" * 80)
    
    # From 128x128x128 -> 129,024 bytes
    # From 64x128x128 -> 112,640 bytes
    # Delta: 129,024 - 112,640 = 16,384 bytes = 16KB
    # Tile M changed from 128 to 64 (halved)
    # So halving M saves 16KB
    
    print("Observed deltas:")
    print(f"  128x128x128 = 129,024 B")
    print(f"   64x128x128 = 112,640 B")
    print(f"  Delta (halving M): {129024 - 112640} B = {(129024-112640)/1024:.1f} KB")
    print()
    print(f"To fit in {device_max_smem} B, need to reduce by: {129024 - device_max_smem} B = {(129024 - device_max_smem)/1024:.1f} KB")
    print()
    
    # If halving M saves 16KB, and we need to save 28KB...
    needed_reduction = 129024 - device_max_smem
    reduction_per_half_m = 16384
    
    print("Options to reduce SMEM:")
    print(f"  1. Reduce M from 128 to 64: saves {reduction_per_half_m} B")
    print(f"  2. Reduce M from 128 to 32: saves ~{2*reduction_per_half_m} B (estimated)")
    print(f"  3. Reduce N similarly")
    print(f"  4. Reduce stages from 2 to 1 (if allowed): saves ~50%")
    print(f"  5. Reduce K (but breaks TMA SF layout)")
    print()
    
    # What if we use 64x64x128?
    # If halving M saves 16KB and halving N saves another ~16KB...
    # 64x64x128 should be around 129,024 - 16,384 - 16,384 = 96,256 B
    # But we got "illegal instruction" not SMEM error - so maybe it does fit but has other issues
    
    print("Prediction for 64x64x128:")
    estimated_64x64x128 = 129024 - 16384 - 16384
    print(f"  Estimated: ~{estimated_64x64x128} B ({estimated_64x64x128/1024:.1f} KB)")
    print(f"  Fits in {device_max_smem} B? {'YES' if estimated_64x64x128 <= device_max_smem else 'NO'}")
    print()
    print("  BUT: 64x64x128 failed with 'illegal instruction', NOT SMEM error!")
    print("  This suggests SMEM is OK, but there's a different issue (TMA, MMA atom, etc.)")
    
    print("\n" + "=" * 80)
    print("INVESTIGATION NEEDED:")
    print("-" * 80)
    print("1. Check if 64x64x128 tile is compatible with SM120 MMA atoms")
    print("2. Look at SmemLayout generation for the scale factor arrays")
    print("3. Check if there's a minimum tile constraint in the collective")
    print("4. Run compute-sanitizer on 64x64x128 to get more info on illegal instruction")
    print("=" * 80)


if __name__ == "__main__":
    main()
