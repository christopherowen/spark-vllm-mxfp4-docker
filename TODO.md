Upstream:
- The bug is pre-existing in vLLM main - not introduced by PR-31740. Both main and PR-31740 are missing o_data_type.
  The FlashInfer plan() function added o_data_type parameter, but the vLLM fast_plan_decode wasn't updated to pass it, causing positional argument shift.
  This bug would affect anyone using this code path with the current FlashInfer version. The mxfp4_wip branch correctly added None, # o_data_type to fix it.
- Ensure non_blocking is a boolean (older PyTorch accepts None, newer versions don't) <- flashinfer
- TRTLLM exclusion on sm121 for attention when using flashinfer backend
- issue report on the issue that caused the cutlass error

Experiment:
- Full-Scale Mode for SM12x Activation Quantization:
  Currently using identity scales (all 0x7F) which avoids layout issues.
  For workloads with wide dynamic range, implement proper per-block absmax scaling:
  1. Compute per-block max absolute value
  2. Convert to float_ue8m0_t scale factor
  3. Write scales in CUTLASS LayoutSFA (swizzled/tiled), NOT row-major
  4. Requires deriving LayoutSFA from instantiated kernel
  See: https://docs.nvidia.com/cuda/cutlass for Sm1xxBlockScaledConfig details
- Tree-based sampling for Eagle3
- tile sizes (64?  192?)
- shapes (other than 1x1x1)?
- mxfp8 data passed between layers
- Cuda graphs when using a draft model
- test async scheduling
- wire bg16, mxfp8, mxfp4
- investigate GDS support
- LORA support?
- address nvfp4
- Looking at the SM120 launcher, it builds a fixed epilogue using cutlass::epilogue::TmaWarpSpecialized. It doesn't seem to handle the FINALIZE fusion mode that the generic launcher supports.
For the initial testing, this might be okay - we can start with the basic mode and add FINALIZE support later if needed. Let me check if the current test uses FINALIZE mode:
- Clean up self-referential comments like mxfp4_wip

- (EngineCore_DP0 pid=285379) DEBUG 01-10 18:05:25 [model_executor/.../quantization/mxfp4.py:370] MXFP4 linear layer is not implemented - falling back to UnquantizedLinearMethod.
- (EngineCore_DP0 pid=285379) DEBUG 01-10 18:05:25 [model_executor/.../quantization/mxfp4.py:385] MXFP4 attention layer is not implemented. Skipping quantization for this layer.

- add finalize fusion support
- support swap?
- configure the cutlass autotuner
- Investigate mxfp8_quantize location in the pipeline
- Review comments for unnecessary optimization pivot entries

- Investigate PDL for the DGX Spark and the gated MoE kernel.
