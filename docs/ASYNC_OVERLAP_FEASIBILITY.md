# Async/Overlap Communication Feasibility Analysis

## Current Architecture

```
TransformerBlock.forward():
    1. input_layernorm(hidden_states, residual)     # Local compute
    2. attn(hidden_states, positions)               # Contains all_reduce (BLOCKING)
    3. post_attention_layernorm(hidden_states, residual)  # Needs step 2's output
    4. mlp(hidden_states)                           # MoE forward
    return output, residual                         # Next layer uses this
```

The attention forward is:
```python
# OAIAttention.forward()
qkv, _ = self.qkv_proj(hidden_states)     # Column parallel, no comm
q, k, v = qkv.split(...)                   # Local
q, k = self.rotary_emb(positions, q, k)    # Local
attn_output = self.attn(q, k, v)           # Attention (local)
output, _ = self.o_proj(attn_output)       # Row parallel, ALL_REDUCE HERE ← BLOCKING
return output
```

## Dependency Chain

```
Layer N:
  QKV_proj → Attention → O_proj_GEMM → [ALL_REDUCE] → residual_add → LayerNorm → MoE
                                              ↓
                                    Must complete before
                                              ↓
Layer N+1:                              depends on residual from N
```

**Key insight:** The residual add requires the fully reduced output from all_reduce. 
There is no "free" compute that can happen while waiting for all_reduce.

## Overlap Opportunities Analysis

### Option 1: Overlap All-Reduce with Next Layer's QKV
**Idea:** Start all_reduce async, begin next layer's QKV projection in parallel.

**Problem:** Layer N+1's input is `output + residual` from Layer N. We can't start 
Layer N+1 until Layer N's all_reduce completes and residual is added.

**Feasibility:** ❌ Not possible without architecture changes

### Option 2: Overlap All-Reduce with Attention Compute (Previous Op)
**Idea:** If we could pipeline the O_proj GEMM with the all_reduce...

**Problem:** GEMM must complete before all_reduce can start (all_reduce reduces 
the GEMM output).

**Feasibility:** ❌ Fundamental dependency

### Option 3: Fuse LayerNorm into All-Reduce Callback
**Idea:** Use NCCL's async callback to trigger LayerNorm immediately when reduce completes.

**Problem:** 
- NCCL doesn't have callbacks
- Would need custom CUDA kernel that fuses reduce + residual + layernorm
- Complex to implement and maintain

**Feasibility:** 🟡 Possible but high effort (custom fused kernel)

### Option 4: Restructure to Post-Norm (Instead of Pre-Norm)
**Idea:** Change architecture so LayerNorm happens BEFORE the all_reduce.

```python
# Current (Pre-Norm with residual)
hidden = all_reduce(o_proj(attn))
hidden = hidden + residual
hidden = layernorm(hidden)

# Restructured (LayerNorm before reduce) - NOT MATHEMATICALLY EQUIVALENT
hidden = layernorm(o_proj(attn))  # Can't do this - layernorm needs full tensor
hidden = all_reduce(hidden)
```

**Problem:** LayerNorm computes mean/std across the feature dimension. With partial 
tensors (before all_gather on last dim), this gives wrong results.

**Feasibility:** ❌ Mathematically incorrect

### Option 5: Sequence Parallelism (Already Partial Support)
**Idea:** Instead of reducing after O_proj, keep sequence dimension split across GPUs.

vLLM has `is_sequence_parallel` flag in MLPBlock. With sequence parallelism:
- Sequence is split across GPUs
- Attention: all-to-all instead of all_reduce
- Reduces total communication volume

**Status:** Partially implemented in vLLM for MoE (`parallel_config.use_sequence_parallel_moe`)

**Feasibility:** ✅ Could be extended, but may not reduce latency for small batches

### Option 6: Pipeline Parallelism (PP) Instead of TP
**Idea:** Split layers across nodes instead of splitting within layers.

```
Node 1: Layers 0-17 (no TP communication within forward)
Node 2: Layers 18-35 (no TP communication within forward)
Inter-node: Only at pipeline stage boundaries
```

**Benefits:**
- Eliminates 36 per-layer all_reduce calls
- Replaces with ~1 send/recv per micro-batch

**Costs:**
- Pipeline bubble (some GPU idle time)
- More complex scheduling

**Feasibility:** ✅ Supported by vLLM with `--pipeline-parallel-size 2`

## Implementation Effort Summary

| Approach | Effort | Latency Improvement | Recommended |
|----------|--------|---------------------|-------------|
| NCCL tuning (env vars) | Low | 5-10% | ✅ Try first |
| Pipeline Parallelism | Low | ~30-50% | ✅ If acceptable bubble |
| Sequence Parallelism | Medium | Variable | 🟡 For high throughput |
| Fused reduce+layernorm kernel | High | 10-20% | ❌ Too complex |
| Full async restructure | Very High | 20-30% | ❌ Major rewrite |

## Recommended Path

### Immediate (No code changes)
1. Test NCCL env var tuning (LL128, Tree, fewer channels)
2. Test `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1`

### Short-term (Configuration change)
3. Try PP=2, TP=1 instead of PP=1, TP=2
   ```bash
   vllm serve ... --pipeline-parallel-size 2 --tensor-parallel-size 1
   ```
   This eliminates 36 all_reduce calls per token.

### Medium-term (Code changes)
4. Extend sequence parallelism support in attention
5. Consider hybrid PP+TP strategies

## PP vs TP Trade-offs for Your Setup

| Metric | TP=2, PP=1 | TP=1, PP=2 |
|--------|------------|------------|
| Collectives per token | 37 (all_reduce) | 1-2 (send/recv) |
| Latency overhead | ~2.5ms | ~0.1-0.5ms |
| GPU utilization | High | Lower (pipeline bubble) |
| Memory per GPU | Lower (model split) | Higher (full layers) |
| Batch size sensitivity | Less sensitive | More sensitive |

For decode (batch=1), PP=2 may actually be faster due to fewer collectives.

## Conclusion

**The fundamental problem is architectural:** Pre-norm transformers with TP have 
strict sequential dependencies that prevent overlapping all_reduce with useful compute.

**Most practical solutions:**
1. NCCL tuning for marginal gains
2. PP=2 instead of TP=2 for significant gains (if model layers fit)
3. Accept the overhead as cost of inter-node TP
