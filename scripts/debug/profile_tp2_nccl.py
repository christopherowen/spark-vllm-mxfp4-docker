#!/usr/bin/env python3
"""Profile NCCL collective times in vLLM decode.

This script patches vLLM's communication ops to measure actual NCCL times.
Run with the vLLM server already loaded, then send a few requests.

Usage:
    # Start vLLM server with TP=2
    # Then in another terminal:
    docker exec -it vllm-dev python3 /workspace/profile_tp2_nccl.py
"""

import time
import torch
import functools
from collections import defaultdict

# Global timing storage
nccl_times = defaultdict(list)
enabled = False

def patch_distributed_ops():
    """Monkey-patch vLLM's distributed ops to add timing."""
    global enabled
    
    try:
        from vllm.distributed import parallel_state
        from vllm.distributed import communication_op
    except ImportError as e:
        print(f"Cannot import vllm.distributed: {e}")
        return False
    
    # Patch all_gather
    original_all_gather = communication_op.tensor_model_parallel_all_gather
    
    @functools.wraps(original_all_gather)
    def timed_all_gather(input_, dim=-1):
        if enabled:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_all_gather(input_, dim)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            nccl_times['all_gather'].append({
                'time_ms': elapsed,
                'shape': tuple(input_.shape),
                'dtype': str(input_.dtype),
            })
            return result
        return original_all_gather(input_, dim)
    
    communication_op.tensor_model_parallel_all_gather = timed_all_gather
    
    # Patch reduce_scatter if it exists
    if hasattr(communication_op, 'tensor_model_parallel_reduce_scatter'):
        original_reduce_scatter = communication_op.tensor_model_parallel_reduce_scatter
        
        @functools.wraps(original_reduce_scatter)
        def timed_reduce_scatter(input_, dim=-1):
            if enabled:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = original_reduce_scatter(input_, dim)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                nccl_times['reduce_scatter'].append({
                    'time_ms': elapsed,
                    'shape': tuple(input_.shape),
                    'dtype': str(input_.dtype),
                })
                return result
            return original_reduce_scatter(input_, dim)
        
        communication_op.tensor_model_parallel_reduce_scatter = timed_reduce_scatter
    
    # Patch all_reduce
    if hasattr(communication_op, 'tensor_model_parallel_all_reduce'):
        original_all_reduce = communication_op.tensor_model_parallel_all_reduce
        
        @functools.wraps(original_all_reduce)
        def timed_all_reduce(input_):
            if enabled:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = original_all_reduce(input_)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                nccl_times['all_reduce'].append({
                    'time_ms': elapsed,
                    'shape': tuple(input_.shape),
                    'dtype': str(input_.dtype),
                })
                return result
            return original_all_reduce(input_)
        
        communication_op.tensor_model_parallel_all_reduce = timed_all_reduce
    
    print("Patched distributed ops for timing")
    enabled = True
    return True


def print_summary():
    """Print timing summary."""
    print("\n" + "=" * 60)
    print("NCCL Timing Summary")
    print("=" * 60)
    
    for op_name, timings in nccl_times.items():
        if not timings:
            continue
        
        times = [t['time_ms'] for t in timings]
        shapes = set(str(t['shape']) for t in timings)
        
        print(f"\n{op_name}:")
        print(f"  Count: {len(times)}")
        print(f"  Total: {sum(times):.3f}ms")
        print(f"  Mean:  {sum(times)/len(times):.3f}ms")
        print(f"  Min:   {min(times):.3f}ms")
        print(f"  Max:   {max(times):.3f}ms")
        print(f"  Shapes: {shapes}")
    
    total_nccl = sum(sum(t['time_ms'] for t in timings) for timings in nccl_times.values())
    print(f"\nTotal NCCL time: {total_nccl:.3f}ms")


def clear_times():
    """Clear timing data."""
    global nccl_times
    nccl_times = defaultdict(list)


def enable_timing():
    """Enable timing collection."""
    global enabled
    enabled = True
    

def disable_timing():
    """Disable timing collection."""
    global enabled
    enabled = False


# Alternative: Use environment variable to enable profiling in running server
def install_hook():
    """Install the profiling hooks. Call this from the vLLM server process."""
    return patch_distributed_ops()


if __name__ == "__main__":
    print("This module provides NCCL timing utilities.")
    print()
    print("To use in a running vLLM server:")
    print("1. Import this module in the server process")
    print("2. Call install_hook() to patch the distributed ops")
    print("3. Run some requests")
    print("4. Call print_summary() to see results")
    print()
    print("Or add to your vLLM startup:")
    print("  import profile_tp2_nccl")
    print("  profile_tp2_nccl.install_hook()")
