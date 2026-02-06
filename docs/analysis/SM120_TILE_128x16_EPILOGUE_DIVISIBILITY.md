# SM120 MoE `(128,16)`: epilogue tile divisibility (`CTA_N=16`) and runtime trap

## TL;DR

- `(tile_mn) = (128,16)` is **non-swap** (native \(M \ge 64\)), so this isolates the issue to **small `CTA_N=16`** rather than `swap_ab`.
- **Initial failure (fixed)**: compile-time `static_assert`:
  - **`EPI_TILE_N must divide CTA_N`** (because `CTA_N=16` but `EPI_TILE_N=32`).
- **Current failure (unfixed)**: after forcing `EPI_TILE_N=16`, the kernel **builds** but **traps at runtime**:
  - `cuda-gdb`: **`CUDA_EXCEPTION_27 Warp Illegal Instruction Parameter`**
  - The trap is in the **SM120 block-scaled TMA mainloop** (`MainloopSm120ArrayTmaWarpSpecializedBlockScaled ... PingpongBlockScaledSm120 ...`), not in “epilogue arithmetic”.
- Companion background (epilogue concepts + C usage + SMEM “need” model):
  - `docs/analysis/EPILOGUE_STUDY_SM12X.md`
- Related prior small-`CTA_N=16` failure mode (different exception class):
  - `(16,256)` (swap/transposed) hit **`Warp Barrier Arrival Mismatch`** under `cuda-gdb`.

---

## Repro

Run inside the dev container (`vllm-dev`) since host Python in this repo typically doesn’t have `torch`.

Suggested environment (keeps errors synchronous + enables existing launcher debug logging):

```bash
export PYTHONPATH=/workspace/flashinfer:/workspace/vllm
export CUDA_LAUNCH_BLOCKING=1
export TLLM_LOG_LEVEL=DEBUG
export TLLM_SM120_MOE_DUMP_TMA=1
python3 /workspace/scripts/debug/repro_tile_128x16_gptoss.py
```

Expected outcomes (current state):

- The JIT build should succeed (no `EPI_TILE_N` divisibility `static_assert`).
- The run may fail as:
  - `SM120 MXFP4 MoE: run failed: Error Internal (CUDA: no error)` from the launcher, and/or
  - a GPU trap captured by `cuda-gdb` (`Warp Illegal Instruction Parameter`).

---

## Where the assert comes from (CUTLASS)

For SM90/SM120 TMA warp-specialized epilogues, CUTLASS requires the epilogue tile to evenly partition the CTA tile:

```text
static_assert(size<1>(CtaTileMNK{}) % size<1>(shape(EpilogueTile{})) == 0, "EPI_TILE_N must divide CTA_N");
```

This is implemented in:

- `cutlass/include/cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp`
- `cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`

So when the kernel has `CTA_N=16`, an `EPI_TILE_N=32` epilogue is invalid.

---

## Update: epilogue tile forced to `N=16` (build now succeeds)

We implemented a local override in the SM120 MXFP4 MoE launcher so that standard-mode kernels choose:

- `EPI_TILE_M=64`
- `EPI_TILE_N = min(32, CTA_N)` → for `(128,16)`, `EPI_TILE_N=16`

This removes the `EPI_TILE_N must divide CTA_N` compile-time blocker for `CTA_N=16`.

However, it does **not** resolve the runtime trap described below.

---

## Runtime failure under `cuda-gdb`: illegal instruction inside SM120 block-scaled mainloop

With the epilogue divisibility issue fixed, `cuda-gdb` shows the first hard failure as:

- `CUDA Exception: Warp Illegal Instruction Parameter`
- signal `CUDA_EXCEPTION_27`

and it reports the trap at:

- PC: `0x325b86920` (exception report) / `0x325b86670` (thread focus)

The faulting kernel is a CUTLASS `device_kernel<GemmUniversal<...>>` instantiation whose type includes:

- `MainloopSm120ArrayTmaWarpSpecializedBlockScaled`
- `KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120`
- tile shape fragment: `... C<128>, C<16>, C<128> ...`

This indicates the current blocker for `(128,16)` is **a mainloop/kernel execution hazard for small `CTA_N=16`**, not the epilogue tile divisibility constraint.

### New debug signal (2026-01-30): tensormap canary shows we get past tensormap commit

We built a special JIT variant that writes a **device-side canary** into the tensormap pool
*after* `tensormaps_cp_fence_release(...)` (i.e., after tensormap update/commit) and *before*
entering the mainloop TMA load loop.

Under `cuda-gdb`, the kernel still traps, but we can now inspect global memory *in the debugger*
despite the stream being poisoned for host `cudaMemcpy`.

For the trapping CTA `blockIdx=(12,0,0)` with `sm_count=48`, we dumped the per-CTA plane-3 descriptor
at:

- base `workspace=0xfffc2caf3580`
- `plane=3 (SFB)` offset \( (12 + 3\cdot 48)\cdot 128 = 0x4E00 \)
- address `0xfffc2caf8380`

The last 32-bit word of that descriptor contains the canary value:

```text
0xfffc2caf83f0:  0x00000000 0x00000000 0x00000000 0xca0a0000
```

**Implication**: the kernel reached “tensormap commit + fence release” before trapping, so the
runtime failure is not explained solely by “descriptor pool never initialized”.

Additionally, in this canary-enabled build the trapped PC we disassembled shows the failing
instruction sequence includes:

```text
SYNCS.PHASECHK.TRANS64.TRYWAIT P0[R4+URZ+0x14ca0],R9
@!P0 BRA ...
```

So the trap may be happening during a **transaction wait / phase check** in the mainloop pipeline
(rather than at the first observed `UTMALDG.*` from earlier traces).

### Disassembly at the faulting PC (what instruction actually traps)

Using `cuda-gdb` to disassemble around the exception PC `0x325b86920` shows the trap occurs **on a TMA load instruction**:

```text
0x0000000325b86900 ...:  R2UR UR19,R13
0x0000000325b86910 ...:  UTMALDG.4D desc[UR42][UR16][UR46]
0x0000000325b86920 ...:  UTMALDG.4D desc[UR42][UR8][UR44]   <-- traps (CUDA_EXCEPTION_27)
0x0000000325b86930 ...:  @P0 BRA 0x325b86640
```

Interpretation:

- This is **not** a tcgen MMA opcode; it is a **TMA load** (`UTMALDG.4D`) inside the SM120 block-scaled mainloop.
- “Illegal Instruction Parameter” at `UTMALDG.4D` strongly suggests the **TMA descriptor and/or coordinates** used by this kernel instance are invalid for this tile/path (or violate a hardware constraint for this TMA mode).

### New: the faulting operand points *inside the tensormap workspace*

In the same `cuda-gdb` session:

- Launcher reported: `workspace=0xfffc2caf3580 size=43008`
- Register snapshot at the trap shows `UR44 = 0xfffc2cafcb80` (via `UR44/UR45`)
- Therefore: `UR44 - workspace = 0x9600`

So the trap’s 4D TMA load is consuming a descriptor/handle that resolves to **an aligned 128B slot inside the allocated workspace**. This is useful because it lets us correlate the failure to a *specific descriptor slot index*:

- `0x9600 / 0x80 = 0x12C = 300`

This narrows “why is it tripping?” to: *why does the descriptor/metadata at slot 300 become invalid (or why is the kernel using the wrong slot) specifically for `CTA_N=16`?*

### Trace: inputs to the faulting `UTMALDG.4D`

From the kernel SASS around the faulting offset (`+39968` / `/*9c20*/`), the trap is on:

```text
/*9c10*/ UTMALDG.4D [UR16], [UR46], desc[UR42] ;          // succeeds
/*9c20*/ UTMALDG.4D [UR8],  [UR44], desc[UR42] ;          // traps (CUDA_EXCEPTION_27)
```

Immediately before those loads, the kernel computes the addresses for the 4D loads as:

```text
/*9b70*/ R2UR UR8, R4
/*9bc0*/ UIADD3 UR16, UPT, UPT, UR8, 0x12000, URZ
/*9bd0*/ UIADD3 UR8,  UPT, UPT, UR8, 0x12800, URZ
...
/*9c10*/ UTMALDG.4D [UR16], [UR46], desc[UR42]
/*9c20*/ UTMALDG.4D [UR8],  [UR44], desc[UR42]
```

Key observation:

- `desc[UR42]` is shared across both loads; the first 4D load succeeds, so the descriptor itself is likely valid.
- The failure is therefore most likely tied to the second load’s **destination address** (`UR8`) and/or its **coordinate/address operand** (`UR44`).

`cuda-gdb` register dump at the trap (lane 0) shows:

- `UR42 = 0x0` (descriptor handle/register; `cuda-gdb`’s raw printout for `UR` regs is not always semantically meaningful for `desc[]` operands)
- `UR16 = 0x12600`, `UR8 = 0x12e00` (these are the computed 4D destinations used by the two loads)
- `UR46 = 0x2cafb380`, `UR44 = 0x2cafcb80` (the second 4D load uses `UR44`)

Next question to answer (still pending): what does `UR44` represent for this `UTMALDG.4D` form (coordinate vs source pointer vs descriptor subfield),
and why is it invalid specifically for `CTA_N=16`?

### Trace: where `UR44` comes from (producer chain)

Tracing further backwards in SASS shows `UR44` is **not** computed near the `UTMALDG.4D`; it is produced earlier from a base pointer plus a stride derived from the grid size:

```text
/*9370*/ LDCU UR44, c[0x0][0xb84] ;                    // loads 48 for this run (gridDim.x)
...
/*93b0*/ UIMAD.WIDE UR48, UR44, 0x80, UR50 ;           // UR48 = UR50 + UR44 * 0x80
/*93c0*/ UIMAD.WIDE UR46, UR44, 0x80, UR48 ;           // UR46 = UR48 + UR44 * 0x80
/*93d0*/ UIMAD.WIDE UR44, UR44, 0x80, UR46 ;           // UR44 = UR46 + UR44 * 0x80
```

For this run:

- `gridDim.x = 48`
- `48 * 0x80 = 0x1800` (6144 bytes)

`cuda-gdb` register dumps are consistent with this:

- `UR44 = UR46 + 0x1800`
- `UR46 = UR48 + 0x1800`
- `UR48 = UR50 + 0x1800`

So `UR50/UR48/UR46/UR44` form a **sequence of four pointers/handles** separated by a stride of `gridDim.x * 128B`.

Implication:

- The trapping `UTMALDG.4D ... [UR44] ...` is using the **4th slot** in this stride sequence, while the prior 4D load uses the **3rd slot** (`UR46`).
- This strongly suggests the invalid parameter is related to how these per-grid slots are provisioned/used (e.g. an out-of-range slot, wrong base pointer, or a schedule that assumes fewer/more slots than what is allocated/initialized).

### Trace: where `UR50` comes from (base of the slot array)

`UR50` is loaded from kernel parameter constant memory, then offset by the CTA’s linear index:

```text
/*81e0*/ LDCU.64 UR50, c[0x0][0x810] ;                 // base pointer (kernel param)
/*81d0*/ S2UR UR7, SR_CTAID.X ;
/*8210*/ S2UR UR6, SR_CTAID.Y ;
/*8220*/ UIMAD UR6, UR6, UR15, UR7 ;                   // UR6 = ctaid.y * gridDim.x + ctaid.x
/*8230*/ UIMAD.WIDE UR50, UR6, 0x80, UR50 ;            // UR50 = base + UR6 * 128B
```

Then the “plane” pointers are derived from `UR50` using `gridDim.x * 128B` as the plane stride:

- `UR48 = UR50 + 1*(gridDim.x*128B)`
- `UR46 = UR50 + 2*(gridDim.x*128B)`
- `UR44 = UR50 + 3*(gridDim.x*128B)`

So **`UR50` is the per-CTA slot base**, and `UR44` is the **4th plane** for that same CTA slot.

### New (2026-01-30): `UR50` can point *inside* the overall GEMM workspace

CUTLASS partitions the single `gemm_workspace` buffer into sub-regions (scheduler / mainloop / epilogue). The SM120
blockscaled mainloop’s `Params::tensormaps` pointer refers to the **mainloop sub-region**, not necessarily the start
of the overall workspace allocation.

In a newer `(128,16)` `cuda-gdb` trap on SM121:

- Launcher printed: `workspace = 0x00000003_3046a680` (size 43008)
- Trap regs included: `UR50 = 0x00000003_3046f480`

So:

- `UR50 - workspace = 0x4e00`

This means the 4-plane tensormap array starts at **`workspace + 0x4e00`** for this kernel instance.

### New (2026-01-30): `cuTensorMapEncodeTiled` can fail for CTA\_N=16 before we ever reach `UTMALDG`

After an experimental change that forced a smaller SMEM swizzle for CTA\_N=16, the build printed a host-side
descriptor encoding failure:

```text
TMA Desc Addr:   0xffffe966bd80
format         14
dim            2
gmem_address   0
globalDim      (128,16,1,1,1)
globalStrides  (0,0,0,0,0)
boxDim         (32,16,1,1,1)
swizzle        1
Error: Failed to initialize the TMA descriptor 1
```

Even though this was produced by an experimental branch, it’s highly relevant: it provides a “C-side” explanation
for why the `(128,16)` kernel later traps with `CUDA_EXCEPTION_27` — the TMA descriptor configuration for the tile
can be **invalid to encode** (or encoded with invalid strides) for CTA\_N=16.

This makes the remaining debugging question very concrete:

- what is the data structure living at `c[0x0][0x810]` (workspace / descriptor pool / barrier pool), and
- why does the **4th plane** (`UR44`) become invalid specifically for `CTA_N=16` at the point where the kernel issues `UTMALDG.4D [UR8],[UR44],desc[...]`?

---

## Hypothesis (high confidence): the 4th plane is the SFB tensormap, and its layout is wrong for small `N`

The producer chain `UR50/UR48/UR46/UR44` looks like **four 128B slots** (stride \(0x80\)) with plane stride \(gridDim.x * 0x80\).
This matches CUTLASS's block-scaled mainloop workspace layout for tensormaps:

- A tensormap: `gmem_tensormap[sm_idx]`
- B tensormap: `gmem_tensormap[sm_idx + 1*sm_count]`
- SFA tensormap: `gmem_tensormap[sm_idx + 2*sm_count]`
- **SFB tensormap**: `gmem_tensormap[sm_idx + 3*sm_count]`  ← **the 4th plane**

In `cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`, the grouped-kernel setup currently initializes `layout_SFB` using the **SFA** helper:

```cpp
layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(init_M, init_N, init_K, 1));
layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(init_M, init_N, init_K, 1)); // suspicious
```

But the correct helper for SFB is `tile_atom_to_shape_SFB(...)`, which uses \(N\) (not \(M\)) when building the scale-factor layout.
This bug is mostly latent when \(M \approx N\) (e.g. 128×128 tiles), but becomes catastrophic for **small `N`** (e.g. 128×16):

- the **SFB tensormap descriptor describes the wrong tensor extents/strides** (as if \(N\) were \(M\))
- a later `UTMALDG.4D` using that SFB descriptor can trip **`Warp Illegal Instruction Parameter`**

### Proposed fix (needs applying in the FlashInfer CUTLASS tree)

Change grouped init to:

```cpp
layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(init_M, init_N, init_K, 1));
```

I wrote a patch file in this repo to apply to FlashInfer's CUTLASS checkout:
- `patches/cutlass_fix_layout_sfb_grouped.patch`

---

## Update (2026-01-29): after applying the CUTLASS `layout_SFB` fix, `(128,16)` still traps at the same instruction

We applied the `layout_SFB = tile_atom_to_shape_SFB(...)` fix directly in the local FlashInfer CUTLASS headers and rebuilt the JIT module.
The repro (`/workspace/scripts/debug/repro_tile_128x16_gptoss.py`) **still hits**:

- `CUDA_EXCEPTION_27 Warp Illegal Instruction Parameter`
- `errorpc = 0x325b86920`
- kernel: `MainloopSm120ArrayTmaWarpSpecializedBlockScaled ... PingpongBlockScaledSm120 ... C<128>,C<16>,C<128> ...`

### Disassembly around `errorpc`

`cuda-gdb` confirms the **faulting instruction is unchanged**:

```text
0x325b868f0: UTMALDG.2D desc[UR42][UR4][UR48]
0x325b86900: R2UR UR19,R13
0x325b86910: UTMALDG.4D desc[UR42][UR16][UR46]   // succeeds
0x325b86920: UTMALDG.4D desc[UR42][UR8][UR44]    // traps (CUDA_EXCEPTION_27)
0x325b86930: @P0 BRA 0x325b86640
```

### Register snapshot (lane 0) at the trap (key fields)

- `UR16 = 0x12600`, `UR8 = 0x12e00` (TMA destination addresses)
- `UR46 = 0x2cafb380` (3rd plane pointer)
- `UR44 = 0x2cafcb80` (4th plane pointer; the one used by the trapping `UTMALDG.4D`)
- `UR48 = 0x2caf9b80`
- `UR50 = 0x2caf8380`
- `UR15 = 0x30` (48) (consistent with `gridDim.x = 48`)

### Note: stride “expected (1,K) vs actual (K,1)” can be tuple-order, not a real mismatch

One recurring source of confusion while debugging swap/transpose is that CUTLASS packed stride tuples are often stored in **mode order** (e.g. `[N,K]`) rather than “matrix shape order” (e.g. `[K,N]`).

Example for B being **ColumnMajor KxN**:

- In matrix terms, ColumnMajor KxN means: `stride_K=1`, `stride_N=K`, i.e. **(1, K)** if you list strides as `(K,N)`.
- But if CUTLASS stores StrideB as a tuple in `[N,K]` order, the same layout prints as **(K, 1)**.

So seeing:

```text
B(ColumnMajor KxN) expected (1,256) actual (256,1)
```

does **not** automatically mean the memory layout is wrong; it can mean we are comparing against the wrong axis ordering.

To make this actionable, we updated the SM120 launcher debug print to report both interpretations and tell you explicitly whether the tuple matches:

- as `[K,N]` (expected `(1,K)`), or
- as `[N,K]` (expected `(K,1)`).

### Interpretation

This means:

- the `layout_SFB` fix is **not sufficient** to eliminate the trap for `(128,16)`, and/or
- the fix does not materially change the descriptors used by this code path (e.g. because the launcher provides `layout_SFB` via `tma_inputs` and the failing descriptor is still being built/updated incorrectly elsewhere).

**Next step**: dump and validate the actual per-plane `cute::TmaDescriptor` contents (especially plane 4 / `UR44`) from the `tensormaps` workspace at runtime, to determine which descriptor field is invalid for this tile (dims/strides/box/format).

---

## New tooling: dump the `tensormaps` workspace from the SM120 launcher (post-`initialize`, pre-`run`)

To make the “plane” hypothesis concrete, we added a launcher debug path that copies the `gemm_workspace` buffer
back to host (device→host) and prints the first 32 bytes of each descriptor plane for a chosen `sm_idx`.

Enable:

```bash
export TLLM_LOG_LEVEL=DEBUG
export TLLM_SM120_MOE_DUMP_TMA=1
export TLLM_SM120_MOE_DUMP_TENSORMAPS=1
export TLLM_SM120_MOE_DUMP_TENSORMAPS_SMIDX=0
```

What it prints (example for `(128,16)`):

- `grid=(48,1,1)` and `hw_info.sm_count=48`
- `expected=24576` bytes for the 4-plane descriptor pool:
  - \(4 * sm\_count * sizeof(TmaDescriptor)\) = \(4 * 48 * 128\) = 24576
- per-plane offsets for `sm_idx=0`:
  - `A`:   `off=0`
  - `B`:   `off=6144`  (\(48*128\))
  - `SFA`: `off=12288` (\(2*48*128\))
  - `SFB`: `off=18432` (\(3*48*128\))

### Observation (important): all 4 planes are zeros before `run`

On `(128,16)` (non-swap), after `gemm.initialize` succeeds we see:

```text
[dump] tensormaps[A] ... bytes[0..15]=00 ... 00
[dump] tensormaps[B] ... bytes[0..15]=00 ... 00
[dump] tensormaps[SFA] ... bytes[0..15]=00 ... 00
[dump] tensormaps[SFB] ... bytes[0..15]=00 ... 00
```

This indicates that **the `gemm_workspace` tensormap pool is not pre-initialized on the host** (at least in this code path).
That may be *expected* (kernel populates/updates descriptors itself), but it’s a key fact to keep in mind when interpreting the SASS:

- the trap is on a `UTMALDG.4D ... [UR44] ...` where `UR44` matches the **4th plane address** pattern,
  but the underlying memory at that plane is currently all zeros *before* launch.

### Next steps enabled by this dump

- Compare the tensormap pool **after** some safe point that is known to populate it (if we can instrument a “descriptor init only” pass).
- Add a second dump immediately after `gemm.run(stream)` for a *working* tile (e.g. `(64,128)`) to see whether the pool becomes non-zero on success.
- If the pool is supposed to be non-zero before the first `UTMALDG.*`, add a host-side init kernel / memcpy path to populate it from `tma_load_*` descriptors.

---

## Working-tile comparison: `(64,128)` succeeds and shows non-zero tensormap bytes

We ran `scripts/debug/repro_tile_64x128_gptoss.py` (same gpt-oss-like dims, tile_mn=(64,128)) with:

```bash
export TLLM_LOG_LEVEL=DEBUG
export TLLM_SM120_MOE_DUMP_TMA=1
export TLLM_SM120_MOE_DUMP_TENSORMAPS=1
export TLLM_SM120_MOE_DUMP_TENSORMAPS_POSTRUN=1
export TLLM_SM120_MOE_DUMP_TENSORMAPS_SMIDX=0
```

Result: `run=Success` and `SUCCESS torch.Size([128, 6144])`.

### Key observation

- For the failing tile `(128,16)`, **all 4 planes were zero** before `run`.
- For the working tile `(64,128)`, we see **non-zero** descriptor bytes in at least A/B/SFB on the `pre_run` dump (first 32 bytes shown),
  indicating that **this code path *can* produce non-zero tensormap descriptors in the workspace**.

Excerpt (A/B/SFB show non-zero header bytes; SFA remained zero in the first 32B window):

```text
tensormaps(pre_run)[A]   off=0     bytes[0..15]=80 9d 68 0c 32 e0 00 00 10 45 04 00 00 18 00 00
tensormaps(pre_run)[B]   off=6144  bytes[0..15]=80 9d 68 0c 32 e0 00 00 10 45 04 00 00 18 00 00
tensormaps(pre_run)[SFA] off=12288 bytes[0..15]=00 00 00 00 00 00 00 00 10 45 04 00 00 00 00 00
tensormaps(pre_run)[SFB] off=18432 bytes[0..15]=80 a5 d4 0a 32 e0 00 00 10 60 04 00 80 01 00 00
```

### Implication for the `(128,16)` trap

This strengthens the hypothesis that `(128,16)` is failing because **some descriptor plane (or its update path) is not being initialized/committed correctly** before the mainloop issues `UTMALDG.4D ... [UR44] ...`.

The immediate next experiment is to:

- compare the same `pre_run` dump for `(128,16)` across different `sm_idx` values (e.g. `0`, `1`, `2`, `47`) to rule out “only some slots get initialized”, and
- dump more bytes per descriptor (beyond 32B) for both a working and failing tile to see which descriptor fields differ.

---

## `(128,16)` across multiple `sm_idx`: still all-zero in every plane

We ran the failing repro multiple times in a single loop, varying:

- `TLLM_SM120_MOE_DUMP_TENSORMAPS_SMIDX = 0, 1, 2, 47`

Result: **all tested slots show all-zero bytes for all four planes** (A/B/SFA/SFB) in the `pre_run` dump.

Example (sm_idx=1 shows correct per-plane offsets but all zeros):

```text
tensormaps(pre_run)[A]   sm_idx=1 off=128   bytes[0..15]=00 ... 00
tensormaps(pre_run)[B]   sm_idx=1 off=6272  bytes[0..15]=00 ... 00
tensormaps(pre_run)[SFA] sm_idx=1 off=12416 bytes[0..15]=00 ... 00
tensormaps(pre_run)[SFB] sm_idx=1 off=18560 bytes[0..15]=00 ... 00
```

And sm_idx=47 (end of range) likewise:

```text
tensormaps(pre_run)[A]   sm_idx=47 off=6016  bytes[0..15]=00 ... 00
tensormaps(pre_run)[B]   sm_idx=47 off=12160 bytes[0..15]=00 ... 00
tensormaps(pre_run)[SFA] sm_idx=47 off=18304 bytes[0..15]=00 ... 00
tensormaps(pre_run)[SFB] sm_idx=47 off=24448 bytes[0..15]=00 ... 00
```

### Interpretation

This rules out “only some SM slots get initialized” as the cause for `(128,16)`.
Instead, for this tile/path, the tensormap workspace **appears to remain uninitialized (or cleared)** across the entire pool before `run`,
which is consistent with the kernel later dereferencing a descriptor pointer that resolves to zeroed fields and tripping `UTMALDG.4D` with `Illegal Instruction Parameter`.

### Next steps

- Dump **more than 32B** per descriptor (e.g. 128B) for the working tile `(64,128)` and failing tile `(128,16)`, to see whether the apparent zeros are truly entire-descriptor zeros. ✅ (implemented; see below)
- If `(128,16)` descriptors are truly all-zero at launch, add an explicit **host-side initialization** of the 4-plane descriptor pool (copy `tma_load_*` descriptors into `gemm_workspace`) before launching the kernel, to confirm whether the crash is simply “descriptor not initialized before first use.”

### Full 128B-per-descriptor dump results (2026-01-29)

We increased the tensormap dump to print the **full 128 bytes** for each `cute::TmaDescriptor` (8 × 16B lines) per plane (A/B/SFA/SFB).

- `(128,16)` (failing): **all zeros across the full 128B** for **all four planes** in `pre_run` (confirming it’s not just “first 32B are zero”).
- `(64,128)` (working): shows **meaningful non-zero** descriptor content for A/B and SFB in `pre_run` (SFA may remain partially/mostly zero depending on the selected layout/shape), confirming the dump path and plane indexing are correct.

### Comparison to prior small-`CTA_N=16` failures

- `(16,256)` (swap/transposed) under `cuda-gdb` produced **`Warp Barrier Arrival Mismatch`** (synchronization divergence).
- `(128,16)` (non-swap/standard) produces **`Warp Illegal Instruction Parameter`** (trap during execution).

So `CTA_N=16` is still problematic, but the **observed exception class differs by tile/path**.

---

## Why SM100 “supports it”

SM100’s ptr-array/TMA epilogue includes **explicit residue/OOB predication** driven by the runtime \(M,N\) problem shape (so it can safely handle “partial tiles” at boundaries), and its builder logic also tends to pick an `EPI_TILE_N` that divides `CTA_N` (otherwise it falls back to `EPI_TILE_N = CTA_N`).

Relevant CUTLASS files:

- `cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp` (residue-based predication)
- `cutlass/include/cutlass/epilogue/collective/builders/sm100_builder.inl` (auto epilogue-tile selection)

SM120 is currently hitting the **SM90-style divisibility constraint**.

---

## Investigation: why does SM120 pick `EPI_TILE_N=32` for `CTA_N=16`?

Observed behavior for `(tile_mn)=(128,16)`:

- The kernel CTA tile is effectively `CTA_N=16`
- But the epilogue tile being instantiated is effectively `EPI_TILE_N=32`
- This mismatch causes the compile-time `static_assert`.

What we need to locate:

- The code path that selects/derives the epilogue tile for the SM120 MoE grouped GEMM instantiation (likely inside the SM120 MoE launcher or its CUTLASS “collective builder” glue).

Known SM120 launcher location used in related SM120 debugging:

- `flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.inl`

---

## Fix options (ordered by likely effort)

### Option A (preferred): override/choose `EPI_TILE_N=16` when `CTA_N=16`

Goal: keep the SM90/SM120 epilogue path intact, but ensure divisibility.

Implementation sketch:

- In the SM120 kernel configuration for `CTA_N=16`, force:
  - `EpilogueTile = make_tile(Int<...M...>{}, Int<16>{})` (or equivalent builder override)
  - or pick the epilogue “auto” logic so it *never* returns 32 when `CTA_N=16`.

Pros:

- Minimal semantic change.
- Directly addresses the compile-time failure.

Cons / risks:

- Might reduce epilogue efficiency vs larger `EPI_TILE_N` (but correctness + buildability first).

### Option B: change the SM120 configuration to avoid `CTA_N=16`

E.g., bump CTA_N to 32 for the `(128,16)` case so the existing `EPI_TILE_N=32` works.

Pros:

- Small code change if allowed by the tile-search/config system.

Cons:

- Likely defeats the purpose of `(128,16)` support and may regress perf/occupancy.
- Not a real “support small tiles” solution.

### Option C: port SM100-style remainder-capable epilogue behavior to SM120

Goal: make SM120 epilogue tolerant to epilogue subtile shapes that don’t evenly partition the CTA tile, via residue predication.

Pros:

- More general (could help other small/misaligned tiles).

Cons:

- High risk / larger change; SM120’s TMA/warp-specialized epilogue code is not the SM100 epilogue.
- Might require deeper changes in CUTLASS or a new epilogue policy.

---

## What “done” looks like

- **Build**: `(128,16)` JIT module compiles successfully (no `EPI_TILE_N must divide CTA_N` assert). ✅ (after epilogue override)
- **Run**: the repro script runs end-to-end without CUDA errors under `CUDA_LAUNCH_BLOCKING=1`. ❌ (still traps / internal error)
- **Correctness**: outputs match a BF16 reference within expected MXFP4 tolerances.
- **Sanitizers**: `compute-sanitizer` reports no illegal instruction / OOB.

