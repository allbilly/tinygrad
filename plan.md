# Rockchip Tensor-Core Path To CNA Submit Plan

## Goal

Use tinygrad's existing tensor-core machinery to recognize multiply-reduce-add work, but do not execute every `Ops.WMMA` as an individual Rockchip NPU job. Instead, treat `Ops.WMMA` as an intermediate CMAC atom marker, coalesce compatible Rockchip WMMA atoms into one larger CNA/CORE/DPU submit, and use the full RK3588 CBUF/CMAC pipeline.

The desired shape is:

```text
tinygrad REDUCE_ADD(MUL(load_a, load_b))
  -> tinygrad tensor-core lowering emits Ops.WMMA atoms
  -> Rockchip renderer groups compatible Ops.WMMA atoms
  -> Rockchip runtime packs full dense A/B tiles
  -> one CNA/CORE/DPU submit computes the full dot-product region
```

This keeps multiply-reduce-add recognition in the standard tinygrad path and keeps Rockchip-specific code focused on hardware grouping, packing, register generation, and submit.

## Hardware Model

RK3588 should be modeled as a fixed-function CMAC/CNA dot-product engine, not as a programmable GPU tensor core.

Important atom dimensions:

```text
NVDLA full: AtomicC = 64, AtomicK = 32
RK3588:     AtomicC = 16, AtomicK = 32
```

For GEMM-like math, map axes as:

```text
M: data rows / spatial atoms / batch rows
C or N: output channels / output columns, processed in AtomicC=16 groups
K: reduction channels, consumed in AtomicK=32 groups
```

One conceptual RK3588 CMAC atom is:

```text
C[m, c:c+16] += A[m, k:k+32] * W[k:k+32, c:c+16]
```

The NPU can process larger `M`, `C`, and `K` regions through CNA/CBUF. The compiler must not submit one job per atom if multiple atoms can be represented as one dense region.

## High-Level Design Decision

Use tinygrad's tensor-core path as the front-end matcher and atomizer.

Do not use the current fake `tc.rockchip` shape with huge dimensions like `8192x8192x1`. That does not model RK3588 CMAC and makes debugging confusing.

Instead:

1. Define a Rockchip tensor-core atom that represents RK3588 CMAC granularity.
2. Let tinygrad emit `Ops.WMMA` atoms for eligible reductions.
3. Add a Rockchip-specific WMMA coalescer in the renderer or immediately before rendering.
4. Serialize one Rockchip CNA program when the WMMA group is compatible.
5. Fall back to normal UOps when grouping is not legal.

## Phase 1: Establish The Rockchip CMAC Atom

Add a Rockchip tensor-core entry in `tinygrad/codegen/opt/tc.py` that models the real hardware atom.

Start with a conservative atom:

```python
TensorCore(
  dims=(16, 1, 32),  # N, M, K: 16 output C lanes, 1 row, 32 K reduction
  threads=1,
  elements_per_thread=(32, 512, 16),
  dtype_in=dtypes.half,
  dtype_out=dtypes.float,
  opts=(...),
  swizzle=(...),
)
```

This exact `elements_per_thread` and `opts` may need adjustment because tinygrad's `TensorCore` class was designed for register-fed GPU instructions. The first implementation can use the smallest valid shape that passes `TensorCore.__post_init__` and reliably produces `Ops.WMMA` markers. The Rockchip renderer will not trust the WMMA vector layout for execution; it will only use it to recover matched reduction structure.

Alternative atom candidates to test:

```text
dims=(16, 1, 32): one output row, full AtomicC, full AtomicK
dims=(16, 2, 32): two output rows if tinygrad's upcast shape stays manageable
dims=(16, 4, 32): larger row block, only if compile shape remains stable
dims=(16, 16, 32): closest to a large matrix tile, but likely too large for the current TC abstraction
```

Acceptance criteria for this phase:

1. `TC=1 DEV=ROCKCHIP` emits `Ops.WMMA` for simple fp16 dot products and matmuls.
2. The generated UOps remain inspectable at `DEBUG=7`.
3. No fake huge tensor-core dimensions are needed.
4. `TC=0` still disables this path.

## Phase 2: Keep TC As A Marker, Not A Submit Unit

Do not implement `Ops.WMMA` runtime execution as one NPU submit per WMMA.

Instead, define the rule:

```text
Ops.WMMA in Rockchip = candidate CMAC atom marker
Rockchip CNA submit = coalesced group of compatible Ops.WMMA atoms
```

This avoids the bad path:

```text
5 WMMA atoms -> 5 DRM submits -> poor CBUF reuse and high overhead
```

The desired path is:

```text
5 compatible WMMA atoms -> 1 rkcna_v1 program -> 1 DRM submit
```

## Phase 3: Add A Rockchip WMMA Coalescer

Add a coalescing pass in `RockchipRenderer.render` or in a helper called by `render`.

Suggested function names:

```python
def _coalesce_wmma_to_cna(self, uops: list[UOp]) -> tuple | None:
  ...

def _extract_wmma_atom(self, wmma: UOp) -> RockchipCmacAtom | None:
  ...

def _group_cmac_atoms(self, atoms: list[RockchipCmacAtom]) -> RockchipCnaProgram | None:
  ...
```

The coalescer should inspect the final UOp graph or linear UOp list and find all Rockchip-compatible `Ops.WMMA` atoms.

Each atom must describe:

```text
input buffer slot
weight buffer slot
output buffer slot
input dtype
weight dtype
output dtype
M range or row index expression
C/N range or output-channel expression
K range slice
input affine index
weight affine index
output affine index
accumulator semantics
post-op semantics, if any
```

The coalescer should group atoms only when they share:

```text
same input buffer
same weight buffer
same output buffer
same dtype combination
same output layout
same batch axes
same post-op or no post-op
contiguous or mergeable C/N slices
contiguous or mergeable K slices
compatible M rows/spatial atoms
```

The output of coalescing should be metadata, not raw WMMA UOps.

Example serialized program shape:

```python
(
  "rkcna_v1",
  meta,
  fallback_lops,
)
```

Keep `fallback_lops` during bring-up so failed CNA execution can fall back to the original tinygrad path.

## Phase 4: Define `rkcna_v1` Metadata

Use explicit metadata rather than parsing names forever.

Suggested dataclass-like tuple fields:

```text
version
m
n
k
batch
a_slot
b_slot
c_slot
a_dtype
b_dtype
c_dtype
a_batch_stride
b_batch_stride
c_batch_stride
a_m_stride
a_k_stride
b_k_stride
b_n_stride
c_m_stride
c_n_stride
transpose_a
transpose_b
acc_dtype
out_dtype
post_op
```

For quick compatibility with the current branch, this can initially reuse the existing `rkmm_v1_*`/`rkmuladd_v1_*` fields:

```text
m, n, k, batch,
a_bs, b_bs, c_bs,
a_ms, a_ks, b_ks, b_ns, c_ms, c_ns,
a_slot, b_slot, c_slot,
ta, tb,
a_dt, b_dt, c_dt
```

Longer term, prefer a tuple or small dataclass serialized through pickle over packing everything into the kernel name.

## Phase 5: Coalescing Rules

Start with strict rules. Relax later.

Legal group v1:

```text
single input buffer
single weight buffer
single output buffer
fp16 input and weight
fp32 accumulation
fp32 or fp16 output
no mask/gate on loads or stores
no non-affine indexes
no nonzero additive bias initially
one output STORE region
K slices form a contiguous range from 0..K
C/N slices form a contiguous output-channel range
M rows form a contiguous row/spatial range
```

Reject and fall back for:

```text
non-affine/gather indexes
sparse terms
different K terms per output channel
mixed buffers
mixed dtypes
masked loads/stores
arbitrary WHERE around inputs
post-ops not supported by DPU
partial accumulator reuse that cannot be represented by CNA/CACC
```

This is still not “detect conv/gemm by API name”. It is detecting the dense dot-product layout CMAC can consume.

## Phase 6: Packing Strategy

Use the existing working packing from the RK3588 GEMM experiments.

Input packing:

```text
A shape: (M, K)
align_in = round_up(K, 32)
packed input shape: (M, align_in)
zero-pad K tail
```

Weight packing:

```text
B shape: (K, N)
align_out = round_up(N, 16) or existing hardware-required alignment if larger
align_in = round_up(K, 32)
logical weight shape before packing: (align_out, align_in), storing B.T
packed weight tile order: (align_out / 16, 16, align_in / 32, 32).transpose(0, 2, 1, 3)
```

Output unpacking:

```text
read raw DPU output surface
extract only logical M x N
discard padded C lanes
cast to requested output dtype
```

For scalar dot products:

```text
M = 1
N = 1
K = reduce length
align_out = 16 or current minimum required by the DPU path
align_in = round_up(K, 32)
```

For row-wise dot products:

```text
M = batch or row count
N = 1
K = reduce length
```

For matmul-like products:

```text
M = lhs rows
N = rhs output channels
K = common reduce dimension
```

## Phase 7: Register Generation

Move the register logic out of ad-hoc `Ops.WMMA` execution and into an explicit CNA program builder.

Suggested helper:

```python
def make_cna_regs(meta: RockchipCnaMeta, in_dma: int, wt_dma: int, out_dma: int) -> list[int]:
  ...
```

This helper should compute:

```text
align_in = round_up(k, 32)
align_out = round_up(n, 16 or current DPU-safe minimum)
feature_grains
data_bank
line_stride
surf_stride
notch value
weight bytes per kernel
input/output data cube sizes
DPU bypass fields
output conversion fields
```

Important rule:

```text
AtomicK=32 drives K padding and weight tile layout.
AtomicC=16 drives output-channel packing.
Do not let tinygrad UNROLL count decide the final K size submitted to CNA.
```

If tinygrad generated five WMMA K-slices, the final CNA job should usually submit:

```text
K = 5 * 32
one input packed buffer
one weight packed buffer
one output buffer
one CNA submit
```

## Phase 8: Runtime Execution Path

Modify `RockchipProgram.__init__` to recognize the new serialized program:

```python
if isinstance(prg, tuple) and prg[0] == "rkcna_v1":
  self.cna_meta = prg[1]
  self.fallback_uops = prg[2]
```

Modify `RockchipProgram.__call__`:

```python
if self.cna_meta is not None:
  try:
    self._run_cna_group(bufs)
    return elapsed
  except Exception:
    if fallback is available:
      run fallback
    else:
      raise
```

Implement `_run_cna_group`:

```text
1. Read source tinygrad buffers.
2. Pack A and B according to metadata.
3. Allocate task, command, packed input, packed weight, output buffers.
4. Copy packed input/weight to NPU buffers.
5. Emit CNA/CORE/DPU registers for the full group.
6. Submit one blocking RKNPU job.
7. Sync/copy output back.
8. Unpack output into tinygrad output buffer.
9. Free temporary NPU buffers.
```

Initially keep the existing CPU-backed allocator model. Later, switch to true device-resident RKNPU buffers.

## Phase 9: Integration With The TC Path

Use `RockchipRenderer.tensor_cores` only after the atom definition is realistic enough.

Initial env controls:

```text
ROCKCHIP_USE_TC_CMAC=1       enable Rockchip TC atom path
ROCKCHIP_CNA_COALESCE=1      enable WMMA -> CNA grouping
ROCKCHIP_CNA_ONLY=1          fail instead of fallback when coalescing fails
ROCKCHIP_CNA_DEBUG=1         print grouping decisions
TC=0                         disable normal TC path, should disable this too
```

Keep old env aliases during migration:

```text
ROCKCHIP_FUSED_MATMUL
ROCKCHIP_FUSED_MATMUL_DEBUG
```

But prefer new names in new code:

```text
ROCKCHIP_FUSED_MULADD
ROCKCHIP_FUSED_MULADD_DEBUG
ROCKCHIP_CNA_DEBUG
```

## Phase 10: Debug Output

At `ROCKCHIP_CNA_DEBUG=1`, print one concise line for every successful coalesced program:

```text
RKCNA_MATCH m=... n=... k=... batch=... atoms=... k_slices=... c_slices=... slots=(a,b,c)
```

For failures, print the first clear reason:

```text
RKCNA_FALLBACK:non_affine_weight_index
RKCNA_FALLBACK:masked_load
RKCNA_FALLBACK:wmma_group_not_contiguous_k
RKCNA_FALLBACK:mixed_output_buffers
```

For runtime failures:

```text
RKCNA_RUNTIME_FALLBACK:npu_verify_mismatch
RKCNA_RUNTIME_FALLBACK:submit_failed_errno_22
```

## Phase 11: Tests

Add tests in increasing order of complexity.

Compile/detection tests with no hardware requirement:

```text
scalar dot:        (a*b).sum()
row-wise dot:      (a*b).sum(axis=1)
matvec:            A @ x
small matmul:      A @ B
batched matmul:    batch dimension preserved
transpose rhs:     A @ B.T style indexing
K not multiple 32: padding path
N not multiple 16: C padding path
```

Hardware tests:

```text
1x1x32 scalar dot
2x1x32 row-wise dot
1x16x32 full AtomicC tile
4x16x32 multi-row AtomicC tile
4x17x33 padding on both C and K
16x16x64 multiple K atoms coalesced into one submit
32x32x32 typical small matmul
```

Numerical tolerances:

```text
fp16 output: atol around 5e-3 for small values, larger for large reductions
fp32 output: compare to numpy fp32 matmul/dot
```

Debug checks:

```text
Verify one CNA submit for multiple WMMA atoms.
Verify fallback when atom group is not contiguous.
Verify TC=0 disables WMMA generation.
Verify ROCKCHIP_CNA_ONLY=1 raises on unsupported shapes.
```

## Phase 12: Avoid Known Bad Designs

Do not use fake huge tensor cores:

```text
bad: dims=(8192,8192,1)
```

Do not execute one submit per `Ops.WMMA`:

```text
bad: each WMMA -> allocate buffers -> submit -> copy back
```

Do not infer correctness from buffer sizes alone:

```text
bad: if buffers have sizes M*K, K*N, M*N, assume GEMM
```

Do not call this path “conv” or “gemm” internally if the intent is lower-level CMAC use. Use names like:

```text
rkcna
rkcmac
rkmuladd
```

Do not let generic TC scheduling details become hardware policy:

```text
UNROLL count is not final K.
UPCAST count is not final C.
WARP/thread count is not a Rockchip execution unit.
```

## Phase 13: Suggested File Boundaries

Keep `ops_rockchip.py` from growing further by moving responsibilities out.

Suggested files:

```text
tinygrad/codegen/opt/tc.py
  Rockchip CMAC atom definition.

tinygrad/runtime/ops_rockchip.py
  Thin runtime integration, program parsing, submit call.

tinygrad/runtime/support/rockchip_cna.py
  Metadata structs, packing/unpacking, register builder.

tinygrad/runtime/support/rockchip_match.py
  WMMA atom extraction and coalescing helpers, if not kept in renderer.
```

Short-term, keep helpers in `ops_rockchip.py` only while proving correctness. Move them out after the path is stable.

## Phase 14: Incremental Implementation Order

1. Add a realistic Rockchip TC atom with `AtomicC=16`, `AtomicK=32`.
2. Turn on `RockchipRenderer.tensor_cores` only behind `ROCKCHIP_USE_TC_CMAC=1`.
3. Confirm `Ops.WMMA` appears for simple dot/matmul with `DEBUG=7`.
4. Add a renderer helper that lists WMMA atoms and prints their source/output structure.
5. Implement strict grouping for one scalar dot or one `M x 1 x K` row-wise dot.
6. Serialize `("rkcna_v1", meta, fallback_lops)` for that group.
7. Reuse the existing `_run_wmma_matmul` packing/register path to execute the metadata.
8. Add output verification for one element during bring-up.
9. Extend grouping to `M x N x K` with `N <= 16`.
10. Extend grouping to multiple C slices, `N > 16`.
11. Extend grouping to multiple K slices, `K > 32`, producing one CNA submit.
12. Add post-op support only after base dot products are stable.
13. Move packing/register helpers into support modules.
14. Remove old fake `tc.rockchip` entries and old name-based matmul-only code.

## Final Target

The final Rockchip path should read as:

```text
tinygrad TC path finds multiply-reduce-add.
Rockchip coalescer turns WMMA atoms into a dense CMAC program.
Rockchip runtime submits one CNA job for the whole coalesced region.
```

This uses the existing tensor-core machinery for readability, while still respecting how RK3588 actually executes through CNA/CBUF/CMAC/DPU.

## Current Bring-Up Status

Implemented so far:

1. Rockchip has a default tensor-core marker in `tinygrad/codegen/opt/tc.py` with `dims=(16,1,32)` for `AtomicC=16` and `AtomicK=32`.
2. `RockchipRenderer` enables this marker by default; generic `TC=0` still disables the tensor-core path.
3. `RockchipRenderer.render` coalesces Rockchip `Ops.WMMA` atoms by default and serializes `("rkcna_v1", meta, fallback_uops)`.
4. `RockchipProgram` recognizes `rkcna_v1` and routes execution through `_run_cna_group` instead of executing individual `Ops.WMMA` atoms.
5. `_run_cna_group` currently supports the dense square GEMM layout used by `test_gemm_fp16` and writes the output buffer directly.

Recent test sweep with `DEV=ROCKCHIP FORWARD_ONLY=1` over GEMM/matmul/dot-related `test/backend/test_ops.py` cases:

```text
PASS: test_gemm_fp16
PASS: test_gemm_with_zeros_shape
PASS: test_scaled_dot_product_attention_gqa_errors
PASS: test_small_gemm_eye
PASS: test_small_gemm_range

FAIL numeric precision on old non-WMMA path:
  test_9_gemm
  test_small_gemm
  test_dot_1d
  test_matmul
  test_matmul_batched
  test_matmul_batched_vector
  test_matmul_simple

TIMEOUT on old non-WMMA path:
  test_big_gemm
  test_broadcastdot
  test_dot
  test_gemm
  test_multidot

FAIL existing Rockchip CMPEQ/shape rewrite issue:
  test_small_gemm_padded
  test_scaled_dot_product_attention
  test_scaled_dot_product_attention_causal
  test_scaled_dot_product_attention_gqa
  test_scaled_dot_product_attention_mismatch_ls
```

Next debug/fix order:

1. Route regular dense float GEMM tests through RKCNA instead of the old scalar half-precision Rockchip path.
2. Extend `_run_cna_group` shape inference beyond square `M=N=K` to vector, matvec, batched, and rectangular GEMM layouts.
3. Fix or bypass the existing Rockchip `Ops.CMPEQ` rewrite shape assertion for padded and attention cases.
4. Re-run each GEMM-related `test_ops.py` case one by one and update this status table.

Latest correction after review:

The temporary CPU-compute shortcuts used during test triage were removed. In particular:

1. `_run_cna_group` no longer computes GEMM with NumPy.
2. The temporary float-input and `8x8x8` Rockchip WMMA markers were removed.
3. Broad Python ALU fallback was removed; the previous narrow fallback behavior was restored.
4. The original float-to-half Rockchip ALU rewrites were restored.

Current honest state:

1. `Ops.WMMA` detection and `rkcna_v1` dispatch are wired.
2. `_run_cna_group` now uses the RK3588 GEMM register path ported from `~/rk3588/examples/gemm.py`.
3. The runtime packs fp16 input and weight buffers, emits CNA/CORE/DPU register qwords, submits one blocking RKNPU job, syncs the DPU output surface, and unpacks the logical output into the tinygrad output buffer.
4. `test_gemm_fp16` now passes through the hardware RKCNA path.

Previous CPU-shortcut rerun status, no longer valid as hardware verification:

```text
PASS: test_9_gemm
PASS: test_small_gemm
PASS: test_matmul_simple
PASS: test_matmul_batched
PASS: test_gemm_fp16

STILL FAILS / NOT YET COVERED: test_dot_1d
  failing subcase: (65) @ (65,45)
  current category: vector/matvec and non-square RKCNA shape inference
```

Hardware verification command:

```bash
DEV=ROCKCHIP FORWARD_ONLY=1 DEBUG=3 uv run pytest -q test/backend/test_ops.py::TestOps::test_gemm_fp16
```

Latest result: pass.

Additional shape coverage after hardware `_run_cna_group`:

```text
PASS: M=16 N=16 K=16 fp16
PASS: M=32 N=32 K=32 fp16
PASS: M=64 N=64 K=32 fp16
PASS: M=16 N=16 K=64 fp16
PASS: M=16 N=8  K=32 fp16
PASS: M=8  N=16 K=32 fp16

FAIL before RKCNA: M=4 N=17 K=33
  reason: existing Rockchip Ops.CMPEQ shape rewrite assertion

FAIL old scalar path: test_dot_1d first scalar-dot subcase (65) @ (65)
  reason: does not emit Ops.WMMA/RKCNA; old scalar path is slow and fp16-imprecise
```

For vector/matvec, `NOOP=1` is not a correctness fix. The right fix is either:

1. Make TC scheduling emit the Rockchip marker for scalar dot/matvec shapes, then run them as RKCNA with logical `M=1` and/or `N=1`.
2. Add a Rockchip-specific pattern/coalescer for dense `REDUCE_ADD(MUL(load, load))` that emits `rkcna_v1` directly for scalar dot and matvec when generic TC does not choose WMMA.
