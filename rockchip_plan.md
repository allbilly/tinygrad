# Rockchip RK3588 NPU Backend — Architecture Plan

**Status:** rev 5.1 — narrows R2 to proven fixed reductions, adds a PPU legality classifier,
removes argmax from coverage, corrects the fallback contradiction, and treats the all-unit PR as a
strategy proposal (not a proven tinygrad acceptance rule). Architecture and image design unchanged
from rev 5.
**Date:** 2026-07-26
**Base:** tinygrad `master` @ `f0117e98d` + branch `rockchip-2607`
**Replaces:** rev 2 (TensorCore → `Ops.WMMA` → late Rockchip reconstruction), and rev 2's LOC budget / phase order

## Changes in rev 3

Every claim below was checked against code, not against concept. See §A (Verified facts) for
file:line evidence and §B for measurements. The substantive changes:

1. **§5 contraction record rewritten.** tinygrad hands the backend one *flattened* affine index
   per operand (`PARAM.index(r1*9 + r4*9 + (r2+r5) + r0*81)`), not labelled strides. Extraction
   must emit per-range coefficients, a constant offset, affine bounds, and **four** range classes
   (the old record had no shared/batch class, so depthwise groups and batched GEMM were
   inexpressible).
2. **§4 adapter question closed.** `LINEAR(Ops.INS…)` → `renderer.asm()` → `Ops.BINARY` → `TinyELF`
   is verified working on master with **zero** core changes beyond the interception hook. The
   alternative "keep PARAMs in LINEAR so `TinyELF.signature` survives" is provably incompatible
   with `do_assemble` and is dropped.
3. **§17 LOC budget replaced.** Rev 2 subtracted deletions that exist on `rockchip_addmul` from
   additions on `rockchip-2607` — two different trees. Real additive cost: ~590 LOC for a correct
   GEMM stage, ~1100 LOC for GEMM + direct conv. The "≤150 net" target is deleted; it only
   described the design we are rejecting.
4. **§18 phases reordered.** New Phase 0.5: DMA-backed buffers, native-layout facts, oracle diff.
   Without DMA buffers there is no address to relocate — today `RockchipAllocator._alloc` returns
   `memoryview(bytearray(size))`. The compiled elementwise/DPU path moves *ahead* of convolution.
5. **New §12 (missing components), §16 (deletions), §13 (ownership rules).** The "generic tinygrad
   fallback" is a Python UOp interpreter, so rejection is not free; this changes what must ship first.
6. **§7 conv_grok reuse quantified** and the constant-duplication trap named explicitly.

## Changes in rev 4 (review response)

7. **§17.0 new — the CI line budget is the real constraint.** `test.yml:219` runs
   `MAX_LINE_COUNT=25000 python sz.py` over the whole repo. Measured: this tree 24845, upstream
   ≈24297, so the entire backend has **~700 sz-lines** of headroom. Design B is therefore not
   merely hard to review, it is unmergeable. The work must be framed as *replacing* the 548-line
   interpreter, not adding to it. Comments and docstrings cost nothing, so hardware formulas get
   explained.
8. **§18 Phase 3/4 swapped: compiled elementwise now precedes GEMM.** It needs no packing (the DPU
   consumes flat contiguous fp16), its register sequence already works on hardware today, existing
   tests validate it numerically, and it is the only step that deletes interpreter lines. This is a
   correction to rev 3, which put the largest-value item before the cheapest provable one.
9. **§23 new — merge sequence.** Nine PRs with per-PR sz-line budgets and the one-line
   justification each will be judged on. The hook ships with its first consumer; nothing lands
   without a caller.
10. **§2.3 / §10 / §11.2 de-ceremonied.** "RK image" is a byte layout, not a type: no `RKImage`
    class, no builder, no serializer/loader; four module-level functions, codec ≤40 lines. Object
    budget fixed at six dataclasses; `RKLayout` and `RKMetrics` collapse into fields.
    **Rejected** the suggestion to flatten `GemmProblem`/`ConvProblem` to tuples — `ConvProblem` is
    ~14 correlated integers, and conv_grok's untyped-dict equivalent is its least readable property.
11. **§12 M3 sharpened.** The packing path is not a free choice among three options; it is
    determined by the M2 layout experiment, and the elementwise path needs none of it.
12. **§4.3** hook rationale restated target-independently.

## Changes in rev 5 (acceptance bar: all hardware units, speed irrelevant)

13. **§23 rewritten: PR 1 is proposed as one complete backend, not a spike.** Rev 5 framed this as
    an acceptance requirement; rev 5.1 corrects that to a *strategy proposal* (§23 presents Option A
    and Option B as competing). The acceptance bar is *not* demonstrated tinygrad policy. Since
    speed is explicitly not a criterion, the expensive part of the CMAC path — **tiling** — is
    dropped from V1, which is what makes a complete first PR affordable *if* Option A is chosen.
14. **§5.1 now has four recognized roots, not one.** R1 contraction (CMAC), **R2 plain `REDUCE(ADD)`
    as a GEMM against a ones-vector (CMAC, fixed-K only)**, R3 ALU tree (DPU), R4 `REDUCE(MAX)` (PPU,
    passing a legality classifier).
15. **§6.4 / §6.5 new.** R2 puts `sum`/`mean`/`avg_pool2d`/softmax-denominator/layernorm-statistic
    on the MAC array for the cost of one constant vector — measured over the full `test_ops` suite,
    this takes the CMAC from 15.0% to **23.8% of all kernels (30.2% of fp-only)** structurally (§B4).
    **`cumsum` is not R2 unconditionally** — it is a scan decomposed by `_split_cumalu` into a chain
    of fixed-K reductions, conditional on §M9 multi-task scratch chaining. R4 adds the PPU behind a
    legality classifier (contiguous axis, compile-time extent ≤ window, fp16, value-only);
    **`argmax`/`argmin` are rejected** (index production needs int32 + gather).
16. **§15 rewritten: no fallback.** The interpreter is deleted in PR 1, both because 548 + ~720 sz
    busts the line budget and because a silent `exec_alu` host path is the same objection as
    "DPU-only" one level down. Unsupported work is *declared* via `code_for_op` /
    `supported_dtypes()` and lowered by tinygrad's own decomposition, or it raises. Explicitly
    rejects fp16 emulation of int tensors (exact only to |x| ≤ 2048).
17. **§2.4 image gains `const` and `scratch_bytes`.** `const` carries the ones-vectors and the
    activation LUTs — the latter are currently computed in Python *per call*. `scratch` is what lets
    a decomposed transcendental or a two-stage reduction exist as a task chain; without it, and with
    no host fallback, those kernels could not run at all.
18. **§12 M8/M9/M10 added** (const section, scratch buffer, PPU path) — all three are PR-1 blockers.
19. **§B4 new: measured unit coverage.** Full `test/backend/test_ops.py` suite (424 tests, 686
    kernels): ~29% DPU (22% pure + 7% transcendental), ~15% CMAC contraction, ~10% CMAC sum-as-GEMM,
    11% PPU, ~22% dtype reject, ~11% non-dtype reject. **Structural coverage: ~68% of all kernels
    without custom TC decomp, ~75% with it; 88.7% / 98.5% of fp-only.** The 51 transcendental kernels
    (7.4%) are the difference — they include every activation except ReLU, so the custom decomp
    (~80 sz) is recommended for PR 1.

---

## 0. Immediate action — first upstream PR

**This section is the implementation order now.** It supersedes §18 Phase 4 and §23.1 for the first
upstream PR only. The rest of this document remains design research and a later roadmap.

### 0.1 Goal and terminology

PR 1 must execute one useful fp16 kernel on each RK3588 compute family:

- **CNA+CORE (CMAC)** — one contraction path.
- **DPU** — one contiguous elementwise/copy path.
- **PPU** — one native pooling/reduction geometry.

"All hardware units" means these three compute families. It does **not** mean all three physical
NPU cores. PR 1 uses core 0 (`core_mask=1`); multicore remains later work after `subcore_task`
semantics are proven.

Speed and broad operator coverage are not PR-1 goals. The goal is the smallest readable compiled
backend that proves all three compute families through tinygrad and contains no host tensor
arithmetic.

### 0.2 Scope

PR 1 includes only:

1. A DMA-backed allocator.
2. A proven semantic interception point that receives the SINK after early movement/load/range
   rewrites and range simplification, but before `apply_opts`.
3. A deterministic binary containing register commands, task boundaries and address relocations.
   No pickle and no UOps in `TinyELF.lib`.
4. One **single-task** fp16 path per compute family:
   - DPU: a contiguous identity/copy or one proven elementwise operation.
   - CNA+CORE: one contraction class whose input, weight and output layouts are proven end-to-end.
   - PPU: one geometry copied from a register-level pooling reference and matched from affine access
     maps; not arbitrary `REDUCE(MAX)`.
5. Core-0 submission only.
6. Explicit `RKPLAN_REJECT:<reason>` errors for every unsupported kernel.
7. Hardware-free classifier/emitter tests and three hardware numerical tests, one per compute
   family.

PR 1 explicitly excludes:

- spatial convolution and convolution tiling;
- R2 sum-as-GEMM, ones-vector constants, mean, avg-pool-as-CMAC and softmax;
- custom transcendental decomposition and activation LUT work;
- scratch buffers, multi-task chains and fusion;
- grouped/depthwise convolution, dilation and integer tensors;
- multicore, BEAM and performance claims;
- suite-wide structural-coverage percentages.

R2 and direct convolution do not add another compute family, so they do not earn their complexity in
the first PR. Transcendentals and softmax require multi-task chaining that has not been demonstrated.

### 0.3 Blockers to close before writing the PR

#### B1 — prove the interception contract

The three-line hook proposed in §4.3 is not yet proven. `do_to_program` currently calls
`full_rewrite_to_sink`, while the required affine/range form is created inside that function. Do not
duplicate those rewrite passes inside Rockchip.

First prototype a small target-independent factoring that:

1. builds the early simplified SINK once;
2. passes that SINK to `renderer.native_program`;
3. returns the native `PROGRAM` when accepted;
4. otherwise continues the existing generic pipeline from the same SINK.

The hook and its Rockchip consumer land together. Do not call the planner on the raw input AST, and
do not claim a three-line patch until the adapter test passes.

PR 1 performs no target-specific transcendental decomposition, so it needs no second planner attempt
after tinygrad's late decompositions.

#### B2 — prove native layout and packing

The reference GEMM and convolution code pads feature rows, swizzles weights in 16×32 blocks and
unpacks aligned output surfaces. Ordinary tinygrad row-major buffers are not automatically native.
Before selecting the CMAC test shape, record:

- which row-major fp16 feature layouts CNA DMA can read using line/surface strides;
- the exact required weight layout;
- the physical output layout and dtype;
- whether the useful output can be written directly into the logical tinygrad buffer.

If a layout conversion is unavoidable, it must be explicit, deterministic and included in the PR-1
line budget. It may move bytes on the host because speed is not a goal, but it must not perform
tensor arithmetic, infer shapes at runtime, use NumPy, or be hidden behind an unreported fast path.
Prefer a native DMA stride or NPU reorder when either is genuinely simpler.

Do not begin the generic contraction extractor until one end-to-end CMAC layout path passes
numerically.

#### B3 — keep every PR-1 kernel single-task

Scratch allocation alone does not prove task order, cache visibility or dependency semantics. The
current policy says `allow_multitask=False`, and the reference material does not document
intra-submit dependencies. PR 1 therefore emits exactly one RKNPU task per accepted kernel.

Any operation that needs an intermediate surface or a second task is rejected and deferred.

### 0.4 Minimal implementation shape

Prefer two implementation files, plus the generated register definitions:

```
tinygrad/runtime/support/rockchip.py   # pure match/classify + register emission + codec
tinygrad/runtime/ops_rockchip.py       # allocator, relocation, task fill, submit, sync
tinygrad/runtime/autogen/rockchip.py   # generated, unchanged
```

Split the support file only if the implemented code demonstrates a real second boundary; do not
start PR 1 with `caps.py`, `contract.py`, `classify.py`, `plan.py`, `emit.py` and
`transcendental.py` as empty architecture.

Use named immutable records only where positional fields would be unclear. There is no dataclass
quota. Hardware constants have one named home, and register templates have one implementation.

The image needs only fields consumed by PR 1:

```
header: magic, version, n_words, n_tasks, n_relocs
words:  u64 register commands
tasks:  flags, op_idx, enable_mask, int_mask, int_clear, regcfg_offset, regcfg_amount
relocs: word_index, globals_slot, addend, shift, mask
```

`int_status=0` and `regcmd_addr` are filled into the kernel ABI task at runtime. `core_mask` belongs
to `rknpu_submit`, not to each task record. Do not add `const` or `scratch_bytes` until a landed
consumer requires them.

### 0.5 Correct line accounting

The first PR is measured against `upstream/master`, where no Rockchip backend exists. Deleting the
interpreter on this feature branch does not create upstream line-count credit.

Measured on 2026-07-26:

```
upstream/master       24300 sz-lines
CI maximum            25000 sz-lines
available               700 sz-lines
```

`sz.py` scans `tinygrad/` only; `test/` and this plan are not part of that limit. Target **≤500 new
sz-lines under `tinygrad/`**, excluding generated autogen code, and treat **600 as a hard PR-1
ceiling**. Do not code-golf to meet it; cut an operator or abstraction instead.

Keep this plan, coverage probes, exploratory traces, `AGENTS.md` and `ref/` out of the upstream PR.
Only focused regression tests belong in the PR.

### 0.6 Tests required before opening

Hardware-free:

- the adapter receives the post-simplification/pre-`apply_opts` SINK;
- exact accept/reject tests for the three supported roots;
- deterministic bytes across two compiles;
- every relocation names a valid `ProgramInfo.globals` slot and command word;
- decoding rejects bad magic, truncated tables and out-of-range relocations;
- no pickle, UOp, planner object or runtime shape inference in the binary;
- command words match captured known-good streams after address fields are masked.

Hardware:

- one fp16 DPU numerical test;
- one fp16 CNA+CORE numerical test including the chosen layout conversion;
- one fp16 PPU numerical test using the exact legal geometry;
- repeated invocation with fresh buffer addresses;
- a two-kernel TinyJit replay;
- guards around destination buffers to catch padded-layout over-writes;
- counters assert that CNA+CORE, DPU and PPU each submitted at least once.

The coverage probe is research-only until it:

1. calls the backend's actual classifier instead of duplicating broad syntax rules;
2. captures schedules without NULL copyout failures;
3. propagates pytest's exit status.

Do not publish coverage percentages from the current probe.

### 0.7 PR pitch and stop condition

Pitch:

> Add a compiled RK3588 backend with DMA buffers and one verified fp16 path on CNA+CORE, DPU and
> PPU. Programs contain deterministic register commands and relocations instead of pickled UOps;
> unsupported kernels fail explicitly.

Open the PR only when all three hardware numerical tests pass from ordinary tinygrad Tensor
expressions, the final diff is within the line ceiling, and the generic tinygrad change is limited to
the proven interception adapter.

If the CMAC layout path or interception adapter cannot be made small and clear, stop and fix that
blocker. Do not compensate by adding convolution, coverage machinery or more planner abstraction.

**PR1 acceptance criterion (measured):** Verified minimal native coverage on
CNA+CORE (matmul), DPU (elementwise ADD/SUB/MUL/MAX and DMA copy), and PPU
(global max pool). Every kernel *accepted* by the Rockchip planner executes
correctly on hardware. No host-side tensor arithmetic (fill, broadcast, mean
scaling) is disguised as native work — these are explicitly rejected.

**Project gate (PR8):** Every applicable `test_ops.py` case passes with
`FORWARD_ONLY=0`. Remaining skips must be intrinsic upstream skips, not
Rockchip exclusions.

Each intermediate PR (PR2–PR7) publishes raw pass/fail/reject counts for
`test_ops.py`. Literal full-suite success is the project end-state, not PR1's
result. The FP16 subset grows across PRs:

1. DPU binary EW (ADD, SUB, MUL, MAX) with two INDEX operands, scalar operand, or DMA copy.
2. CMAC GEMV: matrix-vector products (1D output, one vector input) (PR2).
3. Multi-task programs and scratch allocation (PR3).
4. General DPU elementwise: nested ops, WHERE, activations, hardware fill/broadcast (PR4).
5. General movement and addressing (PR5).
6. CMAC matmul completeness: batched GEMM, mean, tiling (PR6).
7. Convolution and complete PPU coverage (PR7).
8. Dtypes, indexing, and fallback policy (PR8).
9. Full-suite gradients and qualification (PR9).

---

## 1. Stable decision

One compile-time Rockchip planner recognizes generic multiply-add reductions while complete ranges
and affine access expressions are still visible, selects a native RK3588 traversal, and lowers the
selected plan to a final per-kernel executable image.

```
tinygrad SINK (pre-apply_opts: ranges + REDUCE + affine INDEX + masks + epilogue)
   -> extract()      : sink -> Contraction | Reject          [pure]
   -> classify()     : Contraction -> Gemm|Conv | Reject     [pure]
   -> plan()         : Problem + Caps + Policy -> RKPlan     [pure]
   -> emit()         : RKPlan -> regcmd words + task table + relocations -> bytes
   -> Ops.BINARY -> TinyELF.lib
   -> RockchipProgram: parse once; per call bind addrs, apply relocations, submit, sync
```

Architectural rules:

- Generic contraction recognition; Rockchip-specific scheduling.
- Direct convolution is first-class. Spatial convolution is not flattened to GEMM by default.
- No permanent use of tinygrad TensorCore scheduling or `Ops.WMMA`. (§B2 measures why.)
- The runtime receives a final executable artifact, not a planner IR.
- One tinygrad `PROGRAM` may contain several internal NPU tasks. It is not a second graph scheduler.
- conv_grok is the direct-convolution *formula* source (~185 LOC of formulas inside a 1392-LOC
  script), not a module to copy.
- Planning and classification modules must be importable and testable with **no device and no ioctls**.

---

## 2. Compiler / runtime boundary

### 2.1 The boundary that is correct

master already does this for AMD: `do_assemble` → `renderer.asm(prg, lin)` → `Ops.BINARY(uint8)`
→ `UOp.to_elf()` → `Device.runtime(TinyELF)`. Owning the encoding removes the vendor compiler;
it does not remove the need for a final executable representation.

### 2.2 The boundary that is rejected

Renderer emits semantic metadata → runtime classifies Conv vs GEMM → runtime chooses tiles →
runtime generates registers → submit. This is compilation in the runtime.

This is not hypothetical: `rockchip/backend-consideration` builds register templates on the
per-call path (`_run_wmma_matmul` → `build_wmma_template(wmma_meta)`) and smuggles UOps through
`RKTemplatePackage.meta["uops"]` (§A5). `rockchip-2607` pickles the whole UOp list into
`TinyELF.lib` and interprets it (§A4).

### 2.3 Naming

| Name | Lives | Contents |
|---|---|---|
| `Contraction` | compile time, pure | ranges, coefficients, bounds, epilogue |
| `GemmProblem` / `ConvProblem` | compile time, pure | normalized logical problem |
| `RKPlan` / `RKTask` | compile time, pure | layouts, CBUF split, tiles, register values, arg refs |
| "RK image" | `TinyELF.lib` | header + `u64[] regcmd` + task table + relocation table |
| `RockchipProgram` | runtime | parse once, bind, patch, submit, sync |

**"RK image" is a byte layout, not a type.** There is no `RKImage` class, no builder, no serializer,
no loader. The entire artifact API is four module-level functions:

```python
def plan_rk(ast:UOp) -> RKPlan|str: ...                       # str = reject reason
def emit_rk(plan:RKPlan) -> tuple[list[int], list[RKTask], list[RKReloc]]: ...
def encode_rk(cmds, tasks, relocs) -> bytes: ...              # <=20 lines
def decode_rk(lib:bytes) -> tuple[memoryview, tuple, tuple]: ...  # <=20 lines
```

If the codec exceeds ~40 lines total it is overdesigned; the submit ABI only needs register command
words, task structs and addresses (§A6).

### 2.4 Final image contents

Buffer DMA addresses are unknown at compile time, so the image carries relocations:

```
header:   magic, version, target, n_words, n_tasks, n_relocs, n_const, scratch_bytes
regcmd:   u64[]            # (target<<48) | (value<<16) | reg   -- see §A6
tasks:    (unit_mask, op_idx, enable_mask, int_mask, int_clear, regcfg_offset, regcfg_amount, flags, core_mask)[]
relocs:   (word_index, kind, arg_slot, addend, shift, mask)[]   # arg_slot: -1=scratch, -2=const
const:    bytes            # ones vectors, LUT tables, folded scalars -- uploaded once at load
```

Two fields beyond rev 4, both forced by "use all the units without a CPU fallback":

- **`const`**: compile-time-known device data. It carries the ones-vectors that turn every
  `REDUCE(ADD)` into a CMAC GEMM (§6.4) and the 2×513-entry activation LUTs. Today those LUTs are
  computed in Python inside `boilerplate()` *per call*; moving them into the image deletes the last
  floating-point math from the runtime. Uploaded once in `__init__`, never patched.
- **`scratch_bytes`**: one compile-time-sized scratch BO for values passed between tasks inside one
  kernel (§M9). Relocation `arg_slot = -1` binds to it. Without this, multi-task chains cannot exist
  and the backend cannot execute a decomposed transcendental or a two-stage reduction.

`arg_slot` indexes `ProgramInfo.globals` (see §A3 — **not** `TinyELF.signature`, which is empty on
the asm path). Encode with `struct`, not pickle: pickle in `lib` blocks determinism and
content-addressed caching, and `rockchip/backend-consideration.encode_template` already shows the
failure mode (`MAGIC + pickle.dumps(dataclass)`).

Allowed at runtime: read counts, read tables, copy the template, patch words, submit, sync.
Forbidden at runtime: recognizing reductions, parsing problems, choosing tiles, allocating CBUF,
selecting epilogues, packing tensors on the host.

---

## 3. Direction-drift guardrail (compressed)

Recorded so the same category errors are not reintroduced.

| Earlier direction | Why it failed | Final |
|---|---|---|
| CMAC as a one-thread TensorCore; `Ops.WMMA` as transport | TC scheduling is warp/lane/fragment shaped. Measured cost on master: 64³ matmul 434 → **7129 UOps**, 0.20s → **13.16s**; 128³ **fails** (`KernelOptError: locals needed for opt`). See §B2. | Delete. Migration oracle only. |
| Frontend Conv handler / preserve `Tensor.conv2d` | tinygrad has already lowered it; the semantics live in ranges + affine indexes. | One planner; conv vs GEMM is an internal policy. |
| `REDUCE(ADD, MUL)` alone is enough | The same reduction is GEMM, conv, grouped, depthwise. | Reduction starts recognition; full access maps drive planning. |
| Universal GEMM / 1×1 lowering | Ignores im2col expansion (9× payload at 3×3), CBUF refill, task overhead, structurally-zero grouped work. | True dense 1×1 → GEMM; spatial → direct conv. |
| NVDLA-style model loadable | Model-deployment interchange; tinygrad already owns graph scheduling, lifetimes, JIT replay. | Keep NVDLA's *planning order*; copy none of its structures. |
| Reject every byte image | master's normal boundary *is* bytes. | Emit final bytes; reject only semantic-plan serialization. |
| Make `Ops.INS` the architecture | RK3588 has no scalar vregs, spills or lane scheduling. | `Ops.INS` is transport. No `ISARenderer`, no regalloc. |
| Keep semantic descriptors in the runtime because addresses are unknown | Unknown addresses need relocations, not a runtime planner. | Static template + relocation table. |
| Attribute the WIP native RDNA3 backend to `2187Nick/tinygrad` | It is GabrielNakamoto's draft. Nick's fork is ISA/perf reference. | Cite correctly; copy the *ownership boundary*, not GPU regalloc. |

---

## 4. tinygrad integration boundary

### 4.1 Behaviour to preserve

```
one scheduled kernel -> one CALL(SINK) -> one PROGRAM -> one TinyELF -> one cached RockchipProgram
                                                                          -> one or more internal RK tasks
```

`get_runtime` caches on `(ast.key, device)`; `exec_kernel` passes buffers positionally in
`ProgramInfo.globals` order. Cross-kernel ordering, lifetimes and replay stay in tinygrad/TinyJit.

### 4.2 Interception point — verified correct

`full_rewrite_to_sink` order on master:

```
early movement ops -> load collapse -> split ranges -> initial symbolic -> simplify ranges
   |
   +-- ROCKCHIP native planning attempt  <-- HERE
   |
apply_opts -> expander -> remove reduces -> add local buffers -> add gpudims -> lowering
```

At this point the IR still contains, verified by direct probe (§B1): `REDUCE(arg=(ADD,0))` over
named REDUCE ranges, one affine `INDEX` per operand, rectangular padding predicates, and the
epilogue between the reduce and the store. After `apply_opts` the same kernel is split into
UPCAST/UNROLL/LOCAL axes and this structure is gone.

### 4.3 The hook (the only generic-tinygrad patch)

```python
# tinygrad/renderer/__init__.py                        (+1 line)
def native_program(self, ast:UOp) -> UOp|None: return None

# tinygrad/codegen/__init__.py, top of do_to_program   (+2 lines)
if ast.op is Ops.SINK and (p:=renderer.native_program(ast)) is not None: return p
```

Everything else is backend-local. Do **not** patch `postrange.py`, `tc.py` or `uop/ops.py`
(`rockchip_addmul` patched all three; see §16).

The hook must be justified target-independently, and must mention no target: *fixed-function
accelerators need semantic interception before generic GPU scheduling while still returning a normal
`Ops.PROGRAM`.* Nothing in `codegen/` or `renderer/__init__.py` may say Rockchip, NPU, CMAC or conv.
A hook with no consumer is dead code, so it lands in the same PR as the first native program (§23).

### 4.4 Transport — closed: `LINEAR(Ops.INS…)`

Verified end-to-end on master with a synthetic Rockchip program (§B3):

```python
prog_info = ProgramInfo.from_sink(ast, renderer.target)      # globals=(0,1,2), outs=(0,), sizes (1,1,1)
cmds = [UOp(Ops.INS, arg=RKWriteImm(...)), UOp(Ops.INS, src=(param,), arg=RKWriteArg(...)), ...]
prg  = UOp(Ops.PROGRAM, src=(ast_with_estimates, UOp(Ops.LINEAR, src=tuple(cmds))), arg=prog_info)
# stock pm_to_program then runs: do_estimates -> do_assemble(renderer.asm) -> SOURCE + BINARY
```

Facts that constrain this (all verified, §A1–A3):

- `do_assemble` fires only when **every** `LINEAR` src is `Ops.INS` (UPat repeat-match semantics).
- `Ops.INS` is dtype-void, spec-legal, needs no `ISARenderer` and no register allocation.
- `Ops.SOURCE` is built as `"\n".join(str(u.arg) for u in lin.src)` — **the readable command
  listing is free**; give each command dataclass a `__repr__` and there is no separate disassembler.
- `Ops.BINARY` gets `dtypes.uint8` automatically and must carry `bytes`.
- `to_elf()` derives `signature` from *top-level* `LINEAR` members that are `PARAM`. Since all
  members must be `INS`, **the signature is always empty on this path**. That is fine: bind by
  `ProgramInfo.globals` index. Rev 2's §4.3 requirement was wrong.
- `do_estimates` over an INS-only list yields `ops=0, lds=0, mem=0`. The planner knows M/N/K, so it
  must set `KernelInfo.estimates` itself (~4 LOC) or DEBUG=2 reports 0 GFLOPS and §20 timing
  comparisons are meaningless.

Do not invent a global `Ops.ROCKCHIP_CONTRACT`.

---

## 5. Contraction extraction (rewritten)

### 5.1 Recognized roots

Four kernel shapes, not one. This is what makes the hardware fully used rather than
CMAC-for-benchmarks (§B4 measures the distribution):

| # | Root | Unit | Share of all (§B4) | Share of fp-only |
|---|---|---|---|---|
| R1 | `REDUCE(ADD, MUL(INDEX(a), INDEX(b)))` | CNA+CORE (**CMAC**) | 15.0% | 19.1% |
| R2 | `REDUCE(ADD, INDEX(a))` — **fixed-K** sum/mean | CNA+CORE (**CMAC**, ones weight, §6.4) | 8.7% | 11.1% |
| R3 | no REDUCE, ALU tree over `INDEX`es | DPU (EW/LUT/BS/BN) | 33.5% | 42.7% |
| R4 | `REDUCE(MAX, INDEX(a))` — **passes §6.5 legality** | PPU (pooling) | 11.1% | 14.1% |
| —  | dtype int/bool/uint/uchar | rejected in V1 | 21.4% | — |
| —  | gather / non-affine index | rejected in V1 | 10.5% | 13.4% |
| —  | multi-reduce / reduce:MUL | rejected in V1 | 1.2% | 1.5% |

R1 modulo casts around operands and accumulator. Measured: the mul is cast to `float` before the
reduce and back to `half`, exactly as `_apply_tc_opt` also assumes.

R2 is the highest-leverage recognition rule in the whole design: it routes `sum`, `mean`,
`avg_pool2d`, softmax denominator and layernorm statistic onto the CMAC array for the cost
of a ones-vector in the image's `const` section. Without it the CMAC only ever sees matmul and conv
(15.0% of all kernels); with it the CMAC carries **23.8% of all kernels (30.2% of fp-only)**
structurally and the claim "this backend uses the MAC array" becomes structural rather than
decorative.

### 5.2 What the IR actually gives you

Real pre-opt IR for `conv2d((1,4,9,9), (4,4,3,3), bias)` — one kernel, four PARAMs (§B1):

```
input : PARAM1.index( r1*9 + r4*9 + (r2+r5) + r0*81 )        # r1=kh r4=oh r2=kw r5=ow r0=ic
weight: PARAM2.index( r1*3 + r2 + r0*9 + r8*36 )             # r8=oc
out   : PARAM0.index( r4*7 + r5 + r8*49 )
epi   : reduce(...).cast(half) + PARAM3.index(r8)            # per-OC bias, in the same kernel
```

and with `padding=1`:

```
input : ((r1*32 + r4*32 + (r2+r5) + r0*1024 - 33)
         if (((r2+r5)>=1) & ((r2+r5)<33) & ((r1+r4)>=1) & ((r1+r4)<33)) else Invalid)
```

Consequences that rev 2 missed:

- There are **no labelled strides**. `stride_h*IW` and `dilation_h*IW` collapse to the same integer
  (both 9 above). Recovering `(IC, IH, IW, stride, dilation, pad, groups)` requires solving
  coefficients **jointly with** range extents, the constant offset, the mask bounds and
  `PARAM.max_numel()`. Ambiguity must be rejected, not guessed.
- Padding is a rectangular affine predicate on sums of ranges plus a negative constant offset —
  usable, and it is the only place `pad` is recoverable from.
- Groups/depthwise appear as a **shared non-reduce range** present in both operand indexes
  (`r8*36` in weight and `r2*1024` in input for depthwise). Rev 2's record had `lhs_ranges`,
  `rhs_ranges`, `reduce_ranges` and no shared class, so it could not represent groups or batch.

### 5.3 The record

```python
@dataclass(frozen=True)
class Access:
  coeff: dict[UOp, int]                 # range -> integer stride (in elements)
  offset: int                           # constant term
  bounds: tuple[tuple[dict[UOp,int], int, int], ...]   # affine expr, lo, hi (inclusive/exclusive)
  numel: int                            # PARAM.max_numel(), needed to disambiguate
  slot: int                             # ProgramInfo.globals slot

@dataclass(frozen=True)
class Contraction:
  lhs: Access; rhs: Access; out: Access
  lhs_only: tuple[UOp, ...]             # M-like
  rhs_only: tuple[UOp, ...]             # N-like
  shared:   tuple[UOp, ...]             # batch / group  <-- new, required
  reduce:   tuple[UOp, ...]             # K-like
  in_dtype: DType; acc_dtype: DType; out_dtype: DType
  epilogue: tuple[EpiOp, ...]           # BIAS(slot, range) | RELU | CLAMP(lo,hi) | SCALE | CAST
```

Coefficient extraction is `substitute`-and-difference per range plus a linearity re-check; bounds
come from splitting the validity `AND` chain. Budget **150 LOC**, not 35–60.

### 5.4 Reuse boundary with TensorCore

Reuse only the *idea*, and at most the 8 lines of `_apply_tc_opt` that unwrap the cast, require a
`MUL`, and partition ranges into in0-only / in1-only / reduce. Copy that logic into the backend;
do **not** refactor `postrange.py` for a single consumer. Never reuse `TensorCore`, `AxisType.WARP`,
LOCAL/UPCAST fragment ownership, lane swizzles, `Ops.WMMA`, or `_apply_tc_opt` itself.

### 5.5 Rejection classes

```
RKPLAN_REJECT:no_add_mul_reduction      RKPLAN_REJECT:multiple_reductions
RKPLAN_REJECT:non_affine_access         RKPLAN_REJECT:ambiguous_geometry
RKPLAN_REJECT:irregular_mask            RKPLAN_REJECT:unsupported_dtype
RKPLAN_REJECT:unsupported_layout        RKPLAN_REJECT:explicit_im2col_required
RKPLAN_REJECT:cbuf_no_legal_tile        RKPLAN_REJECT:unsupported_epilogue
```

Never silently fall back to hidden host compute. **But note §12/§15: today "reject" means the
Python interpreter, which is not a performance-neutral outcome.**

---

## 6. Classification

### 6.1 GEMM-like

```
lhs: shared..., M..., K...      rhs: shared..., K..., N...      out: shared..., M..., N...
```
V1: contiguous fp16, rectangular, GEMV, batched, supported transpose strides, aligned reduction
tails, strict epilogues. Never infer M/N/K from kernel names or buffer byte counts (§16).

### 6.2 Direct convolution

Derive, from `Access` records only: batch, IC/OC, IH/IW, OH/OW, KH/KW, stride, dilation, pad,
groups, layouts. Solve with the coefficient set + range extents + `numel` + mask bounds; emit
`ambiguous_geometry` when the system is under-determined. `KH*KW == 1` is the pointwise case and
may select the GEMM candidate with no frontend special case. Budget **110 LOC**.

Note: conv_grok supports **no dilation** and **valid padding only** (`MAX_CONV_STRIDE = 7`), so V1
conv support is bounded by the formula source, not by the classifier.

### 6.3 Masks

Supported: affine rectangular validity, zero outside — i.e. convolution padding.
Rejected in V1: data-dependent `WHERE`, irregular gather/scatter, non-affine index, per-output
dynamic reduction domain.

### 6.4 Plain sum as a CMAC GEMM (R2)

`REDUCE(ADD, INDEX(a))` over reduce-ranges `K` with output ranges `M` is
`out[m] = sum_k a[m,k] * 1`. Emit it as the ordinary CNA GEMM with `N = 1` and the weight surface
pointing at a ones-vector in the `const` section (relocation `arg_slot = -2`). `align_in = round_up(K,
32)` and the tail is zero-filled, so the padding contributes exact zeros — no masking needed.

Consequences worth stating because they are what makes this cheap:
- **No new register template.** Same words as R1 with `N=1` and a different weight address.
- **The ones-vector is one buffer per distinct aligned K**, deduplicated inside the image.
- `mean` = this plus a DPU BS-stage scale, i.e. the same task with one more field.

**Scope of R2 in V1 (rev 5.1 narrowing):** R2 covers a *single fixed-K* `REDUCE(ADD)` whose reduce
domain is known at compile time and fits the single-task CBUF. That is `sum(all)`, `sum(axis)`,
`mean`, `avg_pool2d`, and the softmax/layernorm statistics — each one kernel.

**`cumsum` is not R2 as stated.** `tinygrad/mixin/op.py:733` decomposes `cumsum` via `_split_cumalu`
into ≤256-wide `_cumalu` chunks, a carry `_cumalu` over the chunk-totals, and a final elementwise
add. Each individual `_cumalu` is a fixed-K reduction (so each chunk is R2-eligible), but the
*chain* requires multi-task sequencing through scratch (§M9) and is a scan, not a single reduction.
cumsum is therefore **conditional**: reachable on hardware only once multi-task scratch chaining is
demonstrated end-to-end, and rejected otherwise. It is removed from the §B4 coverage count until
that demonstration lands.

Reject when `K` after alignment exceeds the single-task CBUF budget, or when the reduction domain is
dynamic per output element (cumsum's `Σ_{j≤k}` is the canonical case — it is *not* a fixed `K`).

### 6.5 Reduce-max as PPU pooling (R4)

`REDUCE(MAX, INDEX(a))` maps to the PPU with the PC enable value `0x60` (=`PC_OPERATION_ENABLE_
RESERVED_0(48)`, i.e. 48<<1 — enables PPU+PPU_RDMA, disables DPU/DPU_RDMA; see `ref/npu/include/
rkt_registers.h:82-84`) and a pooling window equal to the reduced extent, or a chain of windowed
passes through scratch when the extent exceeds the hardware window.

**Register-level reference:** `ref/npu/include/rknnops.h:4111-4139` (`alu_case_maxpool`). This is
the authoritative register sequence — 22 `EMIT` calls setting `REG_PPU_S_POINTER`,
`REG_PPU_RDMA_RDMA_S_POINTER`, the input/output cube geometry (`REG_PPU_DATA_CUBE_IN_*` /
`_OUT_*`), `REG_PPU_OPERATION_MODE_CFG` (`FLYING_MODE(1) | POOLING_METHOD(1)` for max),
`REG_PPU_POOLING_KERNEL_CFG` (kernel height/width), `REG_PPU_DST_BASE_ADDR` / `_SURF_STRIDE`,
`REG_PPU_DATA_FORMAT` (`PROC_PRECISION(2)` = fp16), the RDMA mirror registers, and finally
`REG_PPU_RDMA_RDMA_OPERATION_ENABLE` + `PC_OPERATION_ENABLE_RESERVED_0(48)`.

**Do not confuse with `ref/npu/ops_rknn/pool.cpp`** — that file is a high-level RKNN API test
harness (`rknn_init`/`rknn_run` on `.rknn` model files), not a register-level reference. The
register-level code lives in `rknnops.h`'s `alu_case_maxpool` / `alu_case_avgpool` /
`alut_case_globalmaxpool` blocks. The `ops_reg/main.c:634` `run_pool_case` function wraps the
register-level path and is the test harness for it.

**R4 legality classifier (rev 5.1).** A `REDUCE(MAX, INDEX(a))` is accepted only when *all* of:

1. The reduced axis is contiguous in the input tensor's row-major layout (so the window reads a
   single physical stride — no transpose-then-pool in V1; transpose first becomes a DPU task that
   materialises to scratch).
2. The reduced extent `K` is a compile-time constant ≤ the PPU hardware window (chained windows
   through scratch are allowed but each window must itself fit).
3. The input dtype is fp16 (PPU has no int datapath).
4. The output is the max *value* only — no index side-channel.

hold. Otherwise reject with `RKPLAN_REJECT:PPU_GEOMETRY` and let tinygrad decompose.

**`argmax` is not R4.** `tinygrad/mixin/op.py:784` defines `argmax` as `_split_cumalu(MAX)` (which is
R4-eligible) *plus* a triu matmul, an `arange`, an `eq`, and an indexed `max` to produce the int32
index. The index-production half needs int32 arithmetic and gather, both of which are out of scope
for V1 (§15). `argmax`/`argmin` are therefore **rejected** in V1, not counted as PPU coverage, and
left to a follow-up that adds an index-producing PPU mode or a CORE-side epilogue.

PPU is included in V1 for one reason: it is the difference between "drives the elementwise unit" and
"drives every compute unit on the NPU" (§18 Phase 3). Whether that difference is a *merge
requirement* or a *follow-up* is the open strategy question of §23.1.

---

## 7. conv_grok integration

`allbilly/rk3588/conv_grok` is the authoritative tested source for direct-conv planning policy.
Measured composition of `conv.py` (1392 LOC):

| Part | LOC | Port? |
|---|---|---|
| planning formulas (`_cbuf_entries`, `_feature_grains`, `_data_bank`, `_mesa_*`, `_compute_k_step`, `_compute_y_step`, `_windows_from_step`, `plan_local_serial_rows`) | **185** | yes, verbatim math, renamed |
| packing (`pack_weights`, `pack_input`, `unpack_output`, `_pack_kh_major`, `_pack_pointwise_wide`) | 47 | yes |
| register emission (`make_regs`) | 74 | yes, restructured into `RKTask` → emitter |
| DRM/ioctl glue | 70 | no — tinygrad already has it |
| shape lists, CLI, reference verification, `TileSession`, runners | 546 | no — becomes tests |
| ctypes/ioctl/register defs, shape-string parsing, misc helpers | ~370 | no — `runtime/autogen/rockchip.py` covers it |

### 7.1 Rename before porting

```
k_step -> oc_step    k_start -> oc_start    k_count -> oc_count
BY_K   -> BY_OC      BY_YK   -> BY_Y_OC
```
The convolution reduction `K` remains `IC*KH*KW`. This rename is not cosmetic: conv_grok's `k`
means output channels, which is the single most likely source of a silent correctness bug.

### 7.2 Constants: one module, passed in

conv_grok duplicates hardware constants 4–5× across files (`RK_CBUF_BANKS` in 4 files;
`RK_LINE_STRIDE_GROUP_CAP` at 5 sites; `MAX_CONV_STRIDE`, `POINTWISE_WIDE_MIN_OC`,
`DW_PLANNER_INPUT_BANKS`, `LARGE_IC_FG_THRESHOLD` duplicated *inside* `conv.py`). Port formulas
as `f(problem, caps)` with **`caps.py` as the only place any constant is written**. Otherwise the
duplication is imported along with the math.

### 7.3 Do not port the executor

No per-tile open/slice/pack/submit/unpack/close loop. Compile all tasks once; reuse packed data
across tiles where valid; the runtime performs no semantic planning.

---

## 8. Direct convolution versus GEMM

### 8.1 Occupancy is not the objective

```
T_total ~= max(T_CMAC, T_in_DMA, T_wt_DMA, T_CBUF_refill) + T_pack + T_task + T_sync
```
High CMAC occupancy coexists with wasted zero products, expanded activation traffic, poor CBUF
reuse and excessive task count. Distinguish *physical* from *useful* MAC utilization.

### 8.2 im2col cost is explicit

`M = batch*OH*OW`, `K = IC*KH*KW`, `N = OC`, `A = im2col(input)`. Stride-1 payload ≈ 1× / 9× / 25×
for 1×1 / 3×3 / 5×5. A patch generator that keeps original rows and forms windows inside CBUF *is*
a direct-convolution schedule and still needs the conv planner. `rockchip_addmul` chose explicit
im2col and pays for it with host NumPy (§A5).

### 8.3 Initial deterministic policy

```
true dense 1x1        -> GEMM/pointwise
spatial KH*KW > 1     -> direct convolution
grouped / depthwise   -> specialized direct candidate (after hardware validation)
unsupported geometry  -> GEMM only if no patch expansion, else reject
```

### 8.4 Candidate metrics (recorded, not optimized)

useful MACs; alignment/group-wasted MACs; input/weight/output DMA bytes; one-time and per-task pack
bytes; CBUF refills; task/submit count; input reuse across OC tiles; weight reuse across Y tiles.
Deterministic rules in V1; no simulator, no autotuner. Feed `ops`/`mem` into `KernelInfo.estimates`
(§4.4) so `DEBUG=2` is truthful.

---

## 9. Capability facts vs policy

```python
# caps.py — the ONLY place hardware constants are written
RKCaps(fp16_atomic_c=16, int8_atomic_c=32, atomic_k=32,
       cbuf_banks=12, cbuf_bank_bytes=32768, cbuf_entry_bytes=128, cbuf_entries_per_bank=256,
       physical_cores=3, max_conv_stride=7, supports_dilation=False)

RKPolicy(enabled_core_mask=1, allow_direct_conv=True, allow_pointwise_gemm=True,
         allow_multitask=False, debug_rejects=True)
```

Every constant must be traceable to a captured register trace or to `ref/npu/docs/rk3588_trm.md`.
Three cores are independent pipelines: core id must never become `gidx`/`lidx`, and CMAC lanes must
never be modelled with `Ops.SPECIAL`. (master *does* have a legitimate `core_id` launch-variable
idiom — see §14.3 — but the RKNPU `subcore_task` ABI means cores are a task-range partition, not a
program index.)

---

## 10. Compile-time RKPlan

Ephemeral, pure, testable. Declarative before emission.

**Object budget: six frozen dataclasses total, no more.** `Access`, `Contraction`, `GemmProblem`,
`ConvProblem`, `RKTask`, `RKPlan`. Everything else is a field, a tuple or a dict.

```python
@dataclass(frozen=True)
class RKTask:
  tile: tuple            # (y_start, y_count, oc_start, oc_count, ...)
  align_c: int; line_stride: int; surf_stride: int; pack: str   # layout inline, not a class
  regs: tuple            # (unit, reg, value)
  arg_refs: tuple        # (reg_index, globals_slot, addend, shift, mask)
  flags: int = 0

@dataclass(frozen=True)
class RKPlan:
  problem: GemmProblem|ConvProblem      # the discriminant; do NOT flatten to a tuple
  tasks: tuple[RKTask, ...]
  metrics: dict = field(default_factory=dict)   # diagnostics only
```

No `RKLayout`, no `RKMetrics`, no `RKImage`, no builders, no serializer/loader classes, no
`kind: str` shadowing the problem type.

`GemmProblem`/`ConvProblem` **stay** dataclasses and must not become tuples. `ConvProblem` carries
~14 correlated integers (batch, ic, oc, ih, iw, oh, ow, kh, kw, stride_h/w, pad, groups, dilation);
positional unpacking of that is how `k_step`-means-output-channels bugs happen. The reference code
proves the point: conv_grok addresses everything through untyped dicts (`s["out_c"]`,
`p["width_stride"]`, `_conv_params()` returning 18 keys) and that is its least readable property.
Two named records are cheaper than one dict with 18 string keys.

The planner ends when every semantic field is a register word, a task-table entry, or a relocation.
`RKPlan` never crosses into the runtime.

---

## 11. Final image and emission

### 11.1 Target-code model

```
WRITE_IMM(CNA, DATA_SIZE0, value)
WRITE_ARG(CNA, FEATURE_DATA_ADDR, slot=1, addend=0)
WRITE_ARG(CNA, DCOMP_ADDR0,       slot=2, addend=0)
WRITE_ARG(DPU, DST_BASE_ADDR,     slot=0, addend=out_tile_byte_offset)
WRITE_IMM(PC,  OPERATION_ENABLE, value)
TASK_END(flags)
```
Target code, not tensor IR. Each becomes one `Ops.INS`; `str(arg)` is the debug listing.

### 11.2 Encoding

`struct`-packed: header, `u64[] regcmd`, task table, relocation table. Do not use pickle, JSON,
`repr`/`eval`, function-name metadata, NVDLA descriptors, or a custom ELF (the RKNPU submit ABI
does not require one — it takes `regcmd_addr` + `regcfg_amount`, §A6).

Hard bound: `encode_rk` + `decode_rk` ≤ 40 lines combined. Decode is
`struct.unpack_from` for the header, one `memoryview(lib)[off:off+n*8].cast("Q")` for the commands,
and two more slices for the tables. If it needs a class, a version-negotiation path or a schema, the
format is wrong. The 275-LOC `emit_runtime_boilerplate` on `backend-consideration` is what happens
when the codec and the register templates are not separated.

### 11.3 Emitter responsibility

Consumes `RKTask`; emits deterministic words, task entries and relocations; contains **no**
`if kind == ...` policy; performs no runtime binding. One authoritative register template per unit
(see §15: the same GEMM template currently exists in four places).

### 11.4 Runtime responsibility

`__init__`: receive `TinyELF`, parse once, store immutable words/tasks/relocations, allocate the
command and task BOs once.
`__call__`: resolve `globals` slots → DMA addresses, apply relocations into a scratch copy, fill the
task structs, submit, optionally sync.
No planning, no packing, no per-call BO churn, no per-call state on `self`.

---

## 12. Missing components (new — these are what actually block bring-up)

| # | Missing | Why it blocks | Owner | LOC | When |
|---|---|---|---|---|---|
| M1 | **DMA-backed allocator** | `RockchipAllocator._alloc` returns `memoryview(bytearray(size))`, so `b.get_buf(dev)` hands the program host memory and there is no address to relocate. `RockchipRegisterAllocator(HCQAllocatorBase)` already exists in the file, unused. | `ops_rockchip.py` | 30–50 | **first** |
| M2 | **Native-layout facts** | The planner cannot pick a layout until it is known which layouts CNA DMA reads natively (`line_stride`, `surf_stride`, NHWC pack, `align_c ∈ {8,16,32}`). Currently guessed in four places. | experiment → `caps.py` | 25 | first |
| M3 | **Explicit pack step** | Replaces hidden per-call NumPy. **Not a free design choice — it is decided by M2.** If the CNA DMA consumes row-major fp16 via `line_stride`/`surf_stride`, feature packing largely disappears and only the weight shuffle remains; only what the DMA cannot consume needs a pack. Rank the survivors: (1) device-native layout propagated between producer and consumer, (2) compiled NPU/PPU reorder task, (3) explicit *counted* host pack. A Python pack silently called from `RockchipProgram.__call__` is never acceptable — it would negate the entire boundary. Not needed at all for the elementwise path (§18 Phase 3). | planner `RKTask` layout fields + `ops_rockchip.py` | 60–90 | with first GEMM (Phase 4) |
| M4 | **Compiled elementwise/DPU template** | §15: kernel *count* dominates real models. Today each fp16 elementwise op > 64 elements allocates 5 BOs, memcpys, submits, reads back, frees 5 BOs. `backend-consideration.build_elementwise_template` (89 LOC) is the prototype. | emitter | ~90 | **before conv** |
| M5 | `KernelInfo.estimates` on native programs | otherwise 0 GFLOPS and §20 comparisons are meaningless | planner | 4 | first |
| M6 | Command-stream diff tool | the only cheap way to keep parity with conv_grok and the trace scripts | `test/` | 40 | first |
| M7 | `subcore_task` semantics resolved | `submit_plan`'s "official" mode uses `task_number = task_count*3, core_mask=0` and gives all three subcores the same range — replicate or partition? Unknown. | experiment | 0 | before any 3-core work |
| M8 | **`const` section in the image** | Carries the ones-vectors that put `REDUCE(ADD)` on the CMAC (§6.4) and the activation LUTs. Today the LUTs are computed in Python *per call* inside `boilerplate()`. Uploaded once at load. | `emit.py` + `ops_rockchip.py` | 25 | **PR 1** |
| M9 | **Kernel-local scratch buffer** | One compile-time-sized BO for values passed between tasks of one kernel. Without it there are no multi-task chains, so decomposed transcendentals, multi-stage reductions and softmax cannot run on-device at all — and with no host fallback (§15) those kernels would simply fail. | `plan.py` sizes it, `ops_rockchip.py` allocates it | 30 | **PR 1** |
| M10 | **PPU path** | `REDUCE(MAX)` = 11.1% of all kernels structurally (§B4 full suite) and covers `max`, `max_pool2d`, softmax's max-subtract — *subject to the §6.5 legality classifier* (contiguous axis, compile-time extent ≤ window, fp16, value-only). It is also the third compute unit: without it the backend drives CNA+DPU only. | `emit.py` PPU template + `classify.py` legality | 40 | **PR 1 (Option A)** |

---

## 13. Ownership rules (new)

1. `contract.py`, `classify.py`, `plan*.py` import nothing from `tinygrad.runtime` and nothing from
   `ctypes`/`numpy`/the autogen module. If they cannot be imported and tested on a laptop, the
   layering is wrong.
2. `emit.py` contains no problem-kind branching. The plan decides; the emitter encodes.
3. `ops_rockchip.py` contains no shape analysis, no tiling, no packing policy, no pattern matching.
4. Hardware constants live only in `caps.py`. Formulas take `caps` as a parameter.
5. Exactly one generic-tinygrad patch is permitted (§4.3). Any second patch requires a written
   justification in this document.
6. No new abstraction without a second real consumer.

---

## 14. Program, task, submit and core boundaries

```
scheduler CALL/SINK -> one PROGRAM -> one lib (bytes) -> one cached RockchipProgram
                                     -> one or more internal RK tasks -> one or more submits
```

### 14.1 Tasks

A CBUF-tiled convolution yields several internal tasks inside one tinygrad kernel. Use sequence
order and compact flags; add no dependency DAG unless the proven submit ABI needs non-linear
dependencies inside one kernel.

### 14.2 Fusion (V1)

Kernel-local epilogue only: `accumulator → bias → scale → ReLU/clamp → cast → store`. Verified
available at the interception point: `conv2d(w, b)` is **one** kernel whose epilogue is
`reduce(...).cast(half) + PARAM3.index(oc)`, and `.relu()` appears as `WHERE(0<acc, acc, 0)`
before the store. Decide fusion before register emission; store it in plan fields; do not carry it
through late `Ops.NOOP` markers. Cross-kernel CONV+POOL is a later scheduler project.

### 14.3 Cores

Generate independent output tiles at compile time; mark which tasks are legally parallel; default to
the proven mask (core 0); validate chained tasks and synchronization before enabling three cores;
keep core assignment as launch metadata. Compare parallelism against CBUF/weight residency before
round-robin. Resolve M7 first.

For reference, master's `core_id` idiom (`gpudims.py` → `ProgramInfo.runtimevars` → `CPUProgram`
setting `args[slot] = tid`) is the upstream way to express "N cores, same program, differing index".
It is the wrong model here because RKNPU partitions a *task list* across cores via `subcore_task`,
and because that idiom reads `TinyELF.signature`, which is empty on the asm path (§4.4).

---

## 15. Capability, not fallback

**Rev 5 change.** Rev 4 kept the Python UOp interpreter as the fallback. That is now rejected for two
independent reasons:

1. **Arithmetic.** The interpreter is 548 sz-lines and upstream headroom is ~703 (§17.0). Keeping it
   *and* adding a compiler is ~1030 sz — unmergeable. The interpreter is the budget.
2. **Acceptance.** A backend whose fallback silently runs `exec_alu` on the host is a CPU backend
   with an accelerator attached. That is the same objection as "only uses the DPU", one level down.

So V1 has **no fallback at all**. What the hardware cannot do is *declared*, and tinygrad's own
machinery then either lowers it into things the hardware can do, or the kernel raises.

### 15.1 The three sanctioned mechanisms

| Mechanism | Where | Effect |
|---|---|---|
| `code_for_op` | `RockchipRenderer` | Declares the ALU set. `get_simplifying_rewrite_patterns` / `get_late_rewrite_patterns` / transcendental decomp then rewrite the rest into it — e.g. `MAX` → `CMPLT`+`WHERE` when `MAX` is absent, `SIN`/`LOG2`/`EXP2` → fp16 polynomial chains, `THREEFRY` → int ops. |
| `supported_dtypes()` | `RockchipRenderer` | Declares fp16/fp32 only. Precedent: `DSPRenderer` excludes fp8/bf16 the same way. |
| explicit reject + raise | planner | An unsupported kernel raises with an `RKPLAN_REJECT:` reason. No silent host execution, ever. |

Decomposition is what makes this viable rather than crippling — **but only when the rewritten UOps
land inside the proven DPU subset.** Rev 5 overclaimed this. The actual fp16 transcendental decomp in
`tinygrad/codegen/decomp/transcendental.py` uses `bitcast(int16)`/`bitcast(uint16)`, integer `&`,
and `shl`/`shr` realised as uint64 multiply/divide (`transcendental.py:24,35,45,58,61,82-94`). None
of those are in the DPU's fp16 datapath, and §15.2 already rejects int tensors. So:

- `MAX`/`MIN`/`SOFTMAX`-without-max → `CMPLT`+`WHERE` (DPU). **Proven.**
- A *custom* polynomial `EXP2`/`LOG2`/`SIN`/`SQRT` written directly against `ADD`/`MUL`/`CMPLT`/
  `WHERE` and a `const` LUT → DPU. **Proven, but requires replacing the upstream decomp for this
  device, not inheriting it.** Sized at ~80 sz-lines (`transcendental.py` in the §23.1 budget):
  `exp2` ~15 (polynomial + LUT-as-WHERE-chain for 2^q, q ∈ [-14,15] for fp16 normals), `log2` ~25
  (binary-search exponent extraction + polynomial on mantissa), `sin` ~20 (quadrant reduction via
  WHERE + polynomial), `sqrt` ~10 (Newton iteration), shared helpers ~10.
- The upstream `transcendental.py` decomp as-is → **rejected** in V1 (it emits bitcast/int ops the
  DPU cannot run). Either ship the custom polynomial decomp in PR 1, or reject transcendentals in V1
  and let tinygrad upcast to fp32 + run on a different device. The plan prefers the custom decomp,
  but it is *work*, not a free consequence of §15.1.

The correct general statement is: **decomposition is attempted only when the resulting UOps are
within the proven DPU subset, and numerical tolerance is tested per operation.** Anything else
raises `RKPLAN_REJECT:DPU_SUBSET`.

### 15.2 What is honestly out of reach in V1, and how to present it

Measured against a 36-op sample (§B4), the uncovered kernels are:

- **integer/bool tensor arithmetic** (`int_add`-style, bitwise, integer division) — the CMAC and DPU
  are fp16/fp32 datapaths. Do **not** emulate int in fp16 as `rockchip_addmul` does: exact only to
  |x| ≤ 2048 and silently wrong beyond, which is worse than unsupported.
- **data-dependent indexing** (`gather`, tensor-indexed loads) — the DMA takes strides, not indices.
- **index-producing reductions** (`argmax`/`argmin`) — needs the int path plus a positional trick
  (§6.5). The max-value half is R4-eligible; the index half is not.
- **`cumsum`** — a scan, not a fixed-K reduction; conditional on §M9 multi-task scratch chaining
  being demonstrated (§6.4). Rejected until then.
- **upstream `transcendental.py` decomps as-is** — they emit `bitcast(int)`/integer `&`/`shl`/`shr`
  that the DPU cannot run (§15.1). Either ship a custom polynomial DPU decomp in PR 1, or reject
  transcendentals in V1.

That is ~33% of kernels in the full suite (~22% dtype + ~11% non-dtype, §B4). Among fp-only
kernels, the reject rate is ~11% without the custom TC decomp (gather + multi-reduce + reduce:MUL)
and ~1.5% with it. Declare them, skip them, and state the number. Precedent for device-specific
skips in `test/backend/test_ops.py`: WEBGPU has 12, NV 4, plus QCOM/PYTHON/NULL and per-renderer
skips. A short, honest skip list is normal; a hidden CPU path is not.

`Ops.SPECIAL` remains valid for generic work-item semantics; it is not the CMAC lane model.

---

## 16. Delete or never port

| Item | Where | Verdict |
|---|---|---|
| `tc.rockchip_cmac`, the `postrange.py` device check, `Renderer.post_matcher` | `rockchip_addmul` core patches | delete — §B2 measures the cost; 128³ does not compile |
| `_coalesce_wmma_to_cna`, `_mark_wmma_cna_region`, `Ops.NOOP("rkcna_region")` | `rockchip_addmul` | delete — reconstructing a matmul from 16-output atoms is the flaw |
| `re.findall(r"\d+", …function_name)` (4 sites), `_rk_infer_mnk(out_elems, a_elems, b_elems)` | `rockchip_addmul` | delete — kernel names are coloured display strings |
| `if self.tag == "shape_scalar": return ()` in `uop/ops.py` `_shape` | `rockchip_addmul` | delete — a tag-driven patch to the core type system |
| `RKTemplatePackage.op / .size / .meta` (esp. `meta["uops"]`) | `backend-consideration` | delete the fields; keep `regcmd`/`patches`/`tasks`/`target` |
| `encode_template = MAGIC + pickle.dumps(pkg)` | `backend-consideration` | replace with `struct` (~40 LOC) |
| `build_wmma_template()` inside `__call__`; `if (m,n,k) in {(64,64,64),(256,256,256)}` | `backend-consideration` | move to compile time / delete |
| host im2col in `_run_cna_conv2d`, host bias add | `rockchip_addmul` | delete |
| per-op `_gpu_alloc`/`_gpu_free` of 5 BOs | `rockchip-2607` | delete once M1+M4 land |
| `RKTemplatePackage.family` branching ("pcchain" vs other) | `backend-consideration` | keep one proven family until a second is measured |
| A generic extractor in `tinygrad/codegen/opt/` | rev 2 §17 | keep it backend-local until a second backend consumes it |
| `Ops.SPECIAL` for CMAC/cores; `ISARenderer` subclassing; `Ops.ROCKCHIP_CONTRACT` | — | never |

**Duplication to collapse, not reproduce:** the same GEMM register template exists four times —
`conv_grok/gemm_npu.make_gemm_regs` (48 LOC), `examples/gemm.make_gemm_regs` (159),
`rockchip_addmul._rk_make_gemm_regs` (~45), `backend-consideration.emit_runtime_boilerplate` (275).
One `RKTask`-driven emitter is worth more than any line-count target.

---

## 17. File layout and cost

```
tinygrad/codegen/__init__.py                 +2    hook
tinygrad/renderer/__init__.py                +1    native_program()
tinygrad/runtime/support/rockchip/
    caps.py                                  25    RKCaps, RKPolicy, register ids. constants ONLY
    contract.py                             150    sink -> Contraction | Reject         [pure]
    classify.py                             110    Contraction -> Gemm/Conv | Reject     [pure]
    plan_gemm.py                             90    GemmProblem -> RKPlan                 [pure]
    plan_conv.py                            230    ConvProblem -> RKPlan (conv_grok math)[pure]
    layout.py                                90    pack kinds, strides, tails, pack fns
    emit.py                                 200    RKPlan -> words/tasks/relocs -> bytes
tinygrad/runtime/ops_rockchip.py            140    DMA allocator + parse-once + patch + submit
                                                   (interpreter deleted in PR 1, §15)
test/rockchip/                              180    golden-IR + planner invariants (no hardware)
```

### 17.0 The binding constraint: tinygrad's CI line budget

This is not a style preference. `.github/workflows/test.yml:219-220` runs
`MAX_LINE_COUNT=25000 python sz.py`, and `sz.py` asserts the **whole repo** is under 25000 lines.
`sz.py` counts logical lines containing tokens (comments, blank lines and docstrings are free) and
excludes only `tinygrad/runtime/autogen` and `tinygrad/viz/assets`.

Measured now:

```
this tree total            24845 sz-lines   (headroom 155)
tinygrad/runtime/ops_rockchip.py   548 sz-lines  (606 raw; ratio ~0.90)
upstream master total      ~24297           (ops_rockchip.py and autogen/rockchip.py do not exist upstream)
=> upstream headroom for an entire Rockchip backend: ~700 sz-lines, shared with everything else landing
```

Consequences, which supersede any "big diffs are hard to review" argument:

1. Design **B** (~1020–1090 raw ≈ 920–980 sz) **cannot land upstream at all**. Design **D** (~590 raw
   ≈ 530 sz) fits with ~170 to spare. **D+** with convolution (~1100 raw ≈ 990 sz) **only fits if the
   548-line interpreter is deleted in the same sequence.**
2. Therefore the correct framing of this work upstream is **replacement, not addition**: "ROCKCHIP
   stops pickling UOps into `TinyELF.lib` and stops interpreting them at runtime; it emits compiled
   register commands like every other backend." The 548-line interpreter *is* the budget.
3. The elementwise/DPU path (M4) must therefore land **before** GEMM: it is the only PR that deletes
   interpreter lines while adding native lines, so it is the one that buys headroom for the rest.
4. Comments and docstrings are free. Explanatory comments on hardware formulas cost nothing against
   the budget, so there is no excuse for dense unexplained register math (§15's warning about
   compressing rather than reducing complexity).

Budget the work in sz-lines per PR (§23), not in raw LOC totals.

### 17.1 Design comparison

New/changed LOC against `rockchip-2607`, excluding `runtime/autogen/rockchip.py` (9615, unchanged).
Multiply raw by ~0.9 for sz-lines.

| | Design | Core patch | Planner | Emit | Runtime | HW-free tests | **Total** | Testable w/o HW | Survives upstream |
|---|---|---|---|---|---|---|---|---|---|
| A | conv_grok straight into `ops_rockchip.py` | 0 | 420 | 130 | 90 | ~0 | **~640** | no | file becomes ~1250 LOC; unmaintainable |
| B | full extractor + planner + emitter + runtime | 3 | 500–570 | 200 | 140 | 180 | **~1020–1090** | yes | yes |
| C | reuse TensorCore scheduling | 12 | 40 | 380 | 120 | ~0 | **~550** | no | **no** — fails at 128³ today |
| **D** | **staged minimum, GEMM only, correct layering** | 3 | 230 | 120 | 150 | 90 | **~590** | yes | yes |
| D+ | D then conv stage | +0 | +330 | +80 | +10 | +90 | **~1100** | yes | yes |

Essential vs accidental complexity:

- **Essential (~600 LOC, hardware-caused):** CBUF bank/entry math, Y/OC tiling and tail merging,
  `align_c`/`atomic_k` packing and strides, ~120 register fields per task, relocation mechanics,
  submit/subcore ABI, fp16-only epilogue plumbing.
- **Accidental (~700 LOC across the branches, layering-caused):** name parsing, buffer-size shape
  inference, WMMA coalescing and region markers, runtime template building, host im2col,
  four duplicate register templates, duplicated constants, the `shape_scalar` core hack.

Rev 2's "355–630 gross, ≤150 net" described Design **C**'s size with Design **B**'s claims. C is the
one design that must not be chosen. **Do not treat a low line count as a goal**: the objective is
that each layer is independently testable and that hardware formulas have one named home.

---

## 18. Implementation phases (reordered)

### Phase 0 — freeze the baseline
Record passing/failing tests per branch; capture known-good GEMM and conv register streams as
golden `u64` files; keep the interpreter path intact; separate hardware execution from host compute
in test output.
**Exit:** a reproducible register + numeric oracle.

### Phase 0.5 — prerequisites (new, blocking)
M1 DMA-backed allocator (delete the `bytearray` allocator, wire the existing HCQ one).
M2 layout experiment → `caps.py`, every constant traced to a capture.
M6 command-stream diff tool.
**Exit:** `b.get_buf(dev)` yields a DMA address; the legal layout set is written down once.

### Phase 1 — boundary spike
Smallest no-semantics program through hook → `LINEAR(Ops.INS)` → `asm()` → `TinyELF` → patch →
submit. Verify: deterministic bytes, cache key, parse-once/patch-per-call, TinyJit replay, no
pickle in `lib`, `globals`-ordered buffer binding.
**Note:** the adapter choice is already settled (§4.4) and `rockchip/backend-consideration` already
contains `RKPatch`/`patch_regcmd`/`submit_template` — promote and de-semanticize that module rather
than rewriting it.
**Exit:** final bytes, not a plan, reach the runtime.

### Phase 2 — extraction
`contract.py` per §5.3, with the golden-IR test suite (§19.2). Device-independent, no hardware.
**Exit:** stable, pure extraction with explicit rejects.

### Phase 3 — coverage measurement (cheap, do before writing PR 1)
Classify every kernel AST that `test/backend/test_ops.py` produces into R1/R2/R3/R4/reject (§B4 is
this experiment on a 36-op sample). Publish per-unit kernel counts and the reject list. This decides
PR 1's scope and is its central argument.
**Exit:** a number, not an opinion, for "which hardware units does this backend need on day one".

### Phase 4 — PR 1 (Option A): one complete backend, all three compute units (§23.1)
`contract` → `classify` → `plan` → `emit` covering R1 (GEMM + single-task direct conv), R2
(sum-as-GEMM against a `const` ones-vector, **fixed-K only**), R3 (DPU chains through scratch,
**DPU-subset only**), R4 (PPU pooling, **passing the §6.5 legality classifier**).
No tiling. No interpreter. No host fallback. Targets `test/trace_gemm.py`, `test/trace_conv.py`, the
existing `test/test_rockchip.py` elementwise suite, plus sum/mean/max/pool/softmax. **Does not
target cumsum (conditional on §M9), argmax (rejected), or transcendentals (custom decomp or
rejected).**
**Exit:** every accepted kernel executes entirely on CNA+CORE, DPU or PPU; the interpreter is deleted;
rejects are explicit strings. **If Option B is chosen instead, this phase splits into Phase 4a
(DPU + R1-GEMM) and Phase 4b (R2 + PPU).**

### Phase 5 — epilogue fusion
Fold bias/scale/ReLU/cast into the producing task's BS/BN stages instead of separate DPU tasks.
**Exit:** `conv2d(w,b).relu()` is one task, not three.

### Phase 6 — conv tiling
Port conv_grok's `oc_step`/`y_step`/window merging; the shapes PR 1 rejected now plan. Parity against
conv_grok via the diff tool; assert no full im2col allocation for spatial conv.
**Exit:** convolutions larger than one CBUF fit.

### Phase 7 — layout, tails, grouped/depthwise
Explicit packing per M2's answer; packed input reuse across OC tasks; packed weight reuse across Y
tasks; channel and reduction tails; grouped then depthwise after hardware validation.
**Exit:** no hidden host packing on the production path.

### Phase 8 — multicore
Resolve M7, then partition the task list via `subcore_task`. Enable three cores only after
deterministic correctness and measured benefit.
**Exit:** independent tiles use proven cores without changing tensor semantics.

Explicitly not in V1: dilation (conv_grok has none), int8, cross-kernel fusion, autotuning,
mandatory three-core execution, arbitrary einsum, irregular gathers.

---

## 19. Test plan

### 19.1 Boundary (Phase 1)
`PROGRAM.src == (SINK, LINEAR, SOURCE, BINARY)`; `BINARY` is `bytes`/uint8; byte-identical across
two compiles; `to_program_cache` hits on the second identical AST; parse-once/patch-per-call (two
calls, one parse, two patched streams); TinyJit replay of a two-kernel graph;
`assert not lib.startswith(b"\x80")` (no pickle); buffers bound in `globals` order.

### 19.2 Extraction — hardware-free, highest value (Phase 2)
Golden-IR tests over: scalar dot; row dot; GEMV; rectangular GEMM; batched GEMM; transposed
strides; 1×1 s1; 1×1 s2; 3×3 pad0; 3×3 pad1; asymmetric 3×1; depthwise groups=C; grouped 2 groups;
conv+bias; conv+relu; conv+bias+relu. For each, assert recovered `coeff`/`offset`/`bounds`, the four
range classes, dtypes and the epilogue chain. Rejects: two reductions; data-dependent gather;
non-affine index; fp32 input; unsupported epilogue; **under-determined geometry**.

### 19.3 Planner invariants — hardware-free (Phases 3, 6)
CBUF bank total legal (`data + weight ≤ 12`); every output covered exactly once; tiles disjoint;
tail windows legal; `oc_step` never conflated with reduction `K`; register fields agree with planned
geometry and strides; every relocation references a declared `globals` slot and an in-range word;
plan and bytes deterministic; true 1×1 allocates no patch matrix; spatial conv never selects im2col
by default; grouped/depthwise performs no cross-group products; metrics include pack, DMA, refills
and task count.

### 19.4 Hardware
**Per-unit coverage assertion (Option A only):** run the op sample of §B4 and assert kernel counts on
CNA+CORE, DPU and PPU are each > 0, and that the reject list matches an explicit allow-list. This is
the test that keeps "uses all the hardware" true over time. **Under Option B this test is added in
PR 2, not PR 1.**

R2/R4 correctness: `sum(all)`, `sum(axis)`, `mean`, `avg_pool2d` against torch (CMAC path with a
ones-vector, incl. a K whose aligned tail is non-zero); `max(axis)`, `max_pool2d`, `softmax`
(exercises PPU max, CMAC sum and DPU tail in one graph). **`cumsum` is not in this list** — it is
conditional on §M9 multi-task chaining being demonstrated first (§6.4); when that lands, add a
cumsum test that checks the carry `_cumalu` round-trips through scratch correctly. **`argmax`/
`argmin` are not in this list** — rejected in V1 (§6.5). A decomposed transcendental (`exp`, `sin`)
is included only if the custom polynomial DPU decomp of §15.1 ships; otherwise transcendentals are
rejected and tested as such.

`trace_gemm` (2,33,17) with the `+1` epilogue; 64³; 256³; GEMV; odd-tail GEMM; `trace_conv`
(1,4,9,9)×(4,4,3,3)+bias; pointwise 1×1; spatial 3×3; Y-only, OC-only and Y/OC tiling; input shared
across OC tasks; weight shared across Y tasks; fused bias/ReLU; one program with multiple internal
tasks; repeated invocation with fresh buffer addresses; core 0 baseline before any multicore;
direct conv vs legal GEMM candidate timing (needs M5).

### 19.5 Architectural assertions
Fail if: `renderer.tensor_cores` is non-empty; `Ops.WMMA` appears in a native program; `lib`
unpickles; a kernel name or a buffer byte count influences a plan; `RKPlan` or any UOp is present
in `TinyELF.lib`; the runtime searches tiles or allocates CBUF; host NumPy pack/compute runs on the
production path (counter must be zero; oracle mode sets a flag and is reported separately);
`ops_rockchip.py` imports `plan*.py`/`classify.py`; `plan*.py`/`classify.py` import
`ctypes`/`numpy`/autogen; `KernelInfo.estimates is None` on a native program; a second generic
tinygrad file is patched.

### 19.6 Test-hygiene note
`test/trace_gemm.py` hard-codes `ALIGN = 64` fp16 elements per row (= `round_up(K=33, 32)`, i.e. a
128-byte row stride) and B packed as `(N, K)` row-major. Those are *planner outputs*, not test
inputs — the aligned K and the rhs layout are exactly what §10 `RKLayout` owns. Convert them to
assertions against
`RKPlan.layout`, otherwise the tests freeze a policy the planner is supposed to own. The same file
references `TC=0` and `ROCKCHIP_FUSED_MATMUL=0`, which are `rockchip_addmul`-era knobs and do
nothing on this branch.

---

## 20. Acceptance criteria

1. Contractions are recognized before generic scheduling, from the pre-`apply_opts` sink.
2. Extraction output is coefficients + offsets + bounds + four range classes, and it rejects
   ambiguous geometry instead of guessing.
3. Access maps alone distinguish GEMM, pointwise and direct convolution. No names, no byte counts.
4. conv_grok formulas are the single source of direct-conv CBUF/Y/OC math, parameterized by `caps`.
5. Direct convolution stays first-class for spatial kernels; true dense 1×1 selects GEMM.
6. Plan choice accounts for data movement, packing, CBUF refill, task overhead and useful vs wasted
   MAC work; `KernelInfo.estimates` is populated.
7. `RKPlan` exists only at compile/test time. The emitter returns final bytes.
8. The image contains only commands, task boundaries, relocations and flags — `struct`-packed.
9. `RockchipProgram` parses once and only binds, patches, submits and syncs.
10. Buffers live in NPU DMA memory; no per-call BO churn; no hidden host packing on the production
    path.
11. No dependency on TensorCore, WARP fragments or `Ops.WMMA`.
12. Exactly one generic-tinygrad patch exists.
13. Planning and classification modules are importable and tested with no device present.
14. One `PROGRAM` may contain multiple internal NPU tasks without a second graph runtime.
15. Cross-kernel scheduling, lifetimes and JIT replay remain in tinygrad.
16. Three-core support is launch metadata, enabled only after M7 and measured benefit.
17. **Six dataclasses is a guardrail, not a hard acceptance criterion.** The intent is "no
    redundant abstraction"; a seventh dataclass that earns its place is fine, and conversely a
    five-dataclass design that hides an abstraction inside a dict is worse. Reviewers enforce the
    intent, not the number.
18. **`encode_rk` + `decode_rk` ≤ 40 lines is a guardrail, not a hard acceptance criterion.** The
    intent is "minimal deterministic codec"; a 43-line codec that is genuinely minimal is not
    architecturally worse than a 39-line one that compresses awkwardly.
19. Every PR in §23 is independently green, independently useful, and net-neutral or better against
    the CI line budget by the end of the sequence.
20. **PR 1 is proposed to drive every compute unit on the die: CNA+CORE (CMAC), DPU and PPU.**
    This is a *strategy proposal* for getting the most coverage per sz-line, not a demonstrated
    tinygrad merge requirement. The alternative — a smaller PR that replaces part of the interpreter
    with a correct compiled DPU or GEMM path and leaves PPU/R2 for a follow-up — is also viable and
    is easier to bisect. **The decision is deferred to §23.1.** If the all-unit PR is chosen, a test
    asserts per-unit kernel counts are all non-zero across the suite, so a regression that quietly
    routes everything to the DPU fails CI.
21. No host execution of tensor math, ever — including no fp16 emulation of integer tensors. Every
    accepted kernel is a task chain on the NPU; everything else raises `RKPLAN_REJECT:*`.
22. The interpreter is gone. `TinyELF.lib` never unpickles, and `RockchipProgram` contains no
    floating-point math (LUTs come from the image's `const` section).

---

## 21. Non-goals for v1

A general accelerator IR; a frontend conv2d interception; a new global Rockchip opcode; permanent
TensorCore/`Ops.WMMA` reuse; a model-wide NVDLA loadable; a custom ELF; rejection of all byte
artifacts; serialized `RKPlan`; the scalar `ISARenderer` regalloc pipeline; arbitrary einsum;
irregular gathers; dynamic reduction domains; automatic cross-kernel Conv+Pool fusion; explicit
im2col as the default spatial representation; dilation; int8; mandatory three-core execution; a
cycle-accurate simulator or autotuner; hidden host packing in production.

---

## 22. Change-control guardrail

```
compiler: recognize -> classify -> layout/CBUF/tiles/fusion -> all registers/tasks -> relocations
runtime:  bind args -> apply relocations -> submit predetermined tasks -> synchronize
```

A proposal is a regression if it: introduces an image/builder/serializer class hierarchy; sends
UOps, affine index trees, problems, candidates or CBUF
policy into `RockchipProgram`; treats core ids as tensor dimensions; forces spatial convolution
through im2col by default; duplicates tinygrad's scheduler or memory planner; adds a second patch to
generic tinygrad; writes a hardware constant outside `caps.py`; or adds a register template that
duplicates an existing one.

---

## 23. Merge sequence

**Rev 5.1 restructure.** Rev 5 sequenced by "all units in PR 1" and argued that was a tinygrad
acceptance requirement. That argument was not demonstrated — it was an inference from the coverage
sample. Two orderings are therefore presented as **competing proposals**, with the decision
deferred to whoever lands PR 1:

- **Option A (rev 5's ordering): one complete backend, all three compute units in PR 1.** Maximises
  coverage per sz-line; the per-unit coverage test (§19.4) is enforceable from day one; risk is that
  a regression in any of three units blocks the whole PR.
- **Option B (rev 4's ordering): staged.** PR 1 replaces the interpreter with a correct compiled
  DPU-pure + R1-GEMM path (the two highest-share units, ~29% + ~15% = ~44% of all kernels, ~53% of
  fp-only); R2/PPU/TC land in PR 2 once scratch chaining, the PPU legality classifier, and the
  custom polynomial decomp are proven. Easier to bisect, smaller blast radius, but PR 1 does not by
  itself satisfy the "drives every unit" test and is ReLU-only for activations.

Both options end at the same place. The rest of this section is written for **Option A** because it
is the harder one to size; if Option B is chosen, split the table below at the R2/PPU line and ship
the top half first.

### 23.1 PR 1 (Option A) — one complete backend, all three compute units

Not a spike, not a partial path: a self-contained replacement for the interpreter that compiles every
accepted kernel to NPU register commands and drives **CNA+CORE (CMAC), DPU and PPU**.

Scope in, by recognized root (§5.1):

| Root | Unit | V1 policy |
|---|---|---|
| R1 `REDUCE(ADD, MUL)` GEMM-shaped | CMAC | single task, no CBUF search, `align_in/out = round_up(·,32)` |
| R1 spatial conv (`KH*KW>1`) | CMAC | CNA direct-conv mode, **single task, NONE tiling only**; reject if CBUF doesn't fit |
| R2 `REDUCE(ADD)` plain sum, **fixed K only** (§6.4) | CMAC | GEMM with `N=1` against a `const` ones-vector |
| R3 ALU tree, **DPU-subset only** (§15.1) | DPU | one task per op, chained through scratch; BS/BN/LUT stages used where they collapse |
| R4 `REDUCE(MAX)`, **passes §6.5 legality classifier** | PPU | pooling window = reduced extent, chained when it exceeds the window |

Deliberately out: all tiling (`_compute_y_step`, `_compute_k_step`, `_windows_from_step`), grouped and
depthwise conv, dilation, int/bool tensor ops, gather, **argmax/argmin (§6.5)**, **cumsum (§6.4 until
§M9 chaining is demonstrated)**, **upstream transcendentals as-is (§15.1 — replaced by the custom
polynomial decomp in the budget)**, multi-core, BEAM. Each is a `RKPLAN_REJECT:` string, not a silent
path.

Why this is affordable in one PR: **the tiling is the expensive part, and it is exactly what "speed
doesn't matter" lets us drop.** conv_grok's 185 lines of planning formulas reduce to one fit-check
(`_cbuf_entries`, `_mesa_weight_banks`, `_data_bank` → does it fit in 12 banks, yes/no), and the
conv register template is the GEMM template with kernel/stride/pad fields populated.

Budget (sz-lines, at tinygrad's 150-char density):

```
caps.py             20   constants + register ids, single home
contract.py        120   R1-R4 recognition, coefficients, bounds, epilogue        [pure]
classify.py         60   Gemm/Conv/Sum/Pool problems + reject rules               [pure]
plan.py            110   single-task plans, layouts, scratch sizing, const dedup  [pure]
emit.py            135   codec 30 | CNA/CORE 30 | DPU EW+LUT 25 | PPU 25 | task/reloc 25
transcendental.py   80   custom fp16 polynomial decomp (exp2/log2/sin/sqrt)       [pure]
ops_rockchip.py    105   DMA allocator, parse-once, const upload, patch, submit
hook                 3   renderer.native_program + do_to_program
test/rockchip/      90   golden-IR + planner invariants, no hardware
-----------------------------------------------------------------------
                  ~720   minus interpreter −548  =>  net ~+172 sz
```

Net ~+172 against ~703 headroom (Option A with custom TC decomp). It fits, but barely — ~531 sz of
headroom remains for subsequent PRs. **Without the TC decomp**, drop the 80-line `transcendental.py`
row: net ~+92 sz, but fp-only coverage drops from 98.5% to 88.7% and the backend is ReLU-only for
activations. **Option B's PR 1 is smaller** — drop the PPU template (~25 sz), the R2 ones-vector
machinery (~20 sz), the §6.5 legality classifier (~15 sz), and the TC decomp (~80 sz) from the table
above; that lands ~+52 sz net and ships R1 + R3-pure only, with R2 + R4 + TC in PR 2.

**Pitch:** "ROCKCHIP becomes a real compiled backend: kernels lower to CNA/CORE, DPU and PPU register
command streams with relocated DMA addresses, instead of pickling UOps and interpreting them on the
CPU. Matmul, conv2d, sum/mean/avgpool and max/maxpool all execute on the NPU's own units."

### 23.2 Subsequent PRs

**Under Option B, "PR 2" below is R2 + PPU (the deferred half of Option A's PR 1), and the table
shifts down by one.**

| PR | Content | +sz | −sz | Net | Sell |
|---|---|---|---|---|---|
| 2 | Epilogue fusion: bias/scale/ReLU/cast folded into the producing task's BS/BN stages instead of separate DPU tasks. | 60 | 20 | +40 | "fuse bias and activation into the conv task" |
| 3 | Conv tiling: `plan_conv.py` gains `oc_step`/`y_step`/window merging from conv_grok; shapes that PR 1 rejected now plan. | 180 | 10 | +170 | "convolutions larger than CBUF" |
| 4 | Grouped + depthwise conv (after hardware validation). | 90 | 0 | +90 | "grouped and depthwise convolution" |
| 5 | Multi-task/multi-core: `subcore_task` partitioning once M7 is answered. | 70 | 0 | +70 | "use all three NPU cores" |
| 6 | int8 / quantized path, if wanted. | 150 | 0 | +150 | "int8 inference" |

Running total after PR 3: ~+172 + 40 + 170 = ~+382 sz over upstream. Headroom then dictates that
PRs 4–6 be paid for by deletions elsewhere or land after upstream's own limit moves.

### 23.3 Rules

- No PR ships a hook, a dataclass or a format field without a consumer in the same PR.
- `contract.py`, `classify.py`, `plan.py` are pure and must be tested on a laptop; hardware tests are
  separate and not required in upstream CI.
- Every PR states the repo line-count delta in its description.
- If PR 1 exceeds ~700 sz in practice, cut **spatial conv** first (leaving GEMM + sum + DPU + PPU,
  which still uses every unit) rather than cutting the PPU or the ones-GEMM — those are what make the
  "uses all the hardware" claim true.
- Before writing PR 1, run the coverage measurement (§B4) over the real `test/backend/test_ops.py` and
  publish the per-unit kernel counts and the reject list. That number is the PR's central argument.

---

# Appendix A — Verified facts (master @ `f0117e98d` + `rockchip-2607`)

**A1 — pipeline and transport**
- `full_rewrite_to_sink` stage order: `tinygrad/codegen/__init__.py:264-296` (early movement 264,
  load collapse 269, split ranges 272, initial symbolic 275, simplify ranges 278, `apply_opts` 281,
  expander 287, remove reduces 290, local buffers 293, gpudims 296).
- `do_to_program` accepts an `Ops.PROGRAM` AST and skips lowering: `codegen/__init__.py:435`.
  Instruction selection is gated on `isinstance(renderer, ISARenderer)`: line 441.
- `pm_to_program`: lines 414-420. `do_assemble` guard is
  `UPat(Ops.LINEAR, src=UPat(Ops.INS))` (line 417) = **all** srcs must be `INS`
  (repeat-match: `tinygrad/uop/upat.py:44-48`).
- `do_assemble` builds `Ops.SOURCE` from `"\n".join(str(u.arg) for u in lin.src)` and
  `Ops.BINARY` from `ctx.asm(prg, lin)`: lines 398-402.
- `do_estimates`: lines 394-396, only when `sink.arg.estimates is None`.
- `Ops.INS` → dtype void, `Ops.BINARY` → `dtypes.uint8`: `uop/ops.py:124,162`.
  Spec allows bare `INS`: `uop/spec.py:115`; PROGRAM progression `uop/spec.py:190-196`.
- `asm()` implementors on master: `NullRenderer` (`ops_null.py:14`), `AMDLLVMRenderer`
  (`llvmir.py:236`), `HIPRenderer` (`cstyle.py:527`) — all via
  `renderer/amd/elf.py:assemble_linear`, which does `insts = [u.arg for u in lin.src]` and scans
  `prg.src[0]` for metadata. This is the precedent to copy.

**A2 — signature vs assemble conflict**
`UOp.to_elf` (`uop/ops.py:1144-1148`) builds `signature` from `PARAM`s that are **top-level members
of `LINEAR.src`**. `do_assemble` requires all top-level members to be `INS`. Therefore the signature
is always `()` on the asm path. Consumers of `signature` are only `ops_cl.py:41,57` and
`ops_cpu.py:97` — neither applies to Rockchip.

**A3 — how buffers actually arrive**
`ProgramInfo.from_sink` (`uop/ops.py:1188-1209`) collects `globals` from `PARAM` slots in the
**SINK**; `exec_kernel` (`engine/realize.py:174-184`) calls
`rt(*[b.get_buf(device) for b in [bufs[i] for i in ast.arg.globals]], …)`. `get_runtime` caches on
`(ast.key, device)` (`realize.py:114-118`). `Buffer.get_buf` returns whatever `_alloc` produced
(`device.py:135-141`) — hence M1.

**A4 — current branch (`rockchip-2607`)**
`ops_rockchip.py` 606 LOC. `render()` = `base64(pickle(uops))` (547-548);
`RockchipProgram.__init__` = `pickle.loads(obj.lib)` (20); `__call__` interprets UOps (248-484);
hardware dispatch only for fp16 elementwise with `len(src) > 64` (397); five BOs allocated per
operation (415-419) and freed (456); `reset_npu()` per call (249); submit is
`task_number=1, core_mask=1` (226-245). `RockchipAllocator._alloc` returns
`memoryview(bytearray(size))` (566); `RockchipRegisterAllocator(HCQAllocatorBase)` exists but is
unused (550-563). `tensor_cores = []` (545). `ROCKCHIP` is absent from `device.py:14 ALL_DEVICES`
(explicit `DEV=ROCKCHIP` still works).

**A5 — branches**
- `rockchip_addmul`: +12549 LOC; `ops_rockchip.py` 1455 LOC; **five core patches** —
  `codegen/__init__.py` (+1 `post_matcher`), `renderer/__init__.py` (+1 field),
  `codegen/opt/tc.py` (+10, `rockchip_cmac`), `codegen/opt/postrange.py`
  (+1: `if self.ren.target.device == "ROCKCHIP": opt_level = max(opt_level, 2)`),
  `uop/ops.py` (+1: `if self.tag == "shape_scalar": return ()` inside `_shape`).
  `_rk_infer_mnk` at 932; `re.findall(r"\d+", …function_name)` at 1096, 1139, 1284, 1359;
  `_coalesce_wmma_to_cna` 1077; `_run_cna_conv2d` 651-710 = host NumPy im2col + host bias add.
- `rockchip/backend-consideration`: `runtime/support/rockchip.py` 835 LOC — `RKPatch` 18-27,
  `RKTaskTemplate` 29-38, `RKTemplatePackage` 40-50, `RKSubmitPlan` 52-57, `rkcmd` 61,
  `encode_template` 63-65 (**pickle**), `patch_regcmd` 88-104, `apply_patches` 106-109,
  `submit_plan` 111-117, `submit_template` 119-144, `emit_runtime_boilerplate` 265-540 (**275 LOC**),
  `build_elementwise_template` 540-629 (89), `build_wmma_template` 634-655,
  `build_conv1x1_template` 658-721, UOp "meta" extractors 721-835 (~115 LOC, wrong stage).
  `ops_rockchip.py` 663 LOC: `self.uops = self.template.meta["uops"]` (74),
  `build_wmma_template(wmma_meta)` inside `__call__` (134),
  `if (m,n,k) in {(64,64,64),(256,256,256)}` (144).
- `conv_grok/conv.py` 1392 LOC, composition in §7. `gemm_npu.py` 358 (`make_gemm_regs` 48, im2col
  327-335). `examples/gemm.py` 596 (`make_gemm_regs` 159). `examples/conv_gemm.py` 1417.

**A6 — hardware ABI (`ref/npu`, `ref/rk3588`)**
Register command word: `(target << 48) | (value << 16) | reg` (`include/rknnops.h:368-384`,
matching `emit_raw` in `ops_rockchip.py:37-40` and `rkcmd` on `backend-consideration`).
CBUF: 12 banks × 32768 B, 256 entries × 128 B per bank (`rknnops.h:41-44`, `npu_hw.h:170-171`,
`conv_grok/conv.py:26-29`). `direct_convolution = 0`, depthwise `CONV_MODE = 3`.
`struct rknpu_task` = `(flags, op_idx, enable_mask, int_mask, int_clear, int_status, regcfg_amount,
regcfg_offset, regcmd_addr)`; `enable_mask 0xd` = CNA|CORE|DPU, `0x60` = PPU; `int_mask 0x300`/`0xc00`.
`struct rknpu_submit` carries `core_mask` and `subcore_task[5]` of `(task_start, task_number)`.
No CMAC array dimensions and **no intra-submit dependency semantics** are documented anywhere in the
reference material — hence M7 and the "no dependency DAG" rule.
Matmul tiling reference: `max_m = ((12-2) × 32768) / (align_up(K,32) × 2)` (`ops_reg/main.c:5100`),
consistent with conv_grok's bank reservation.

---

# Appendix B — Measurements

**B1 — IR at the interception point** (`DEV=NULL`, stages of §4.2 applied by hand)

| Case | Kernels | Structure recovered |
|---|---|---|
| `64×64 @ 64×64` half | 1 | `REDUCE(ADD)` over one range(64); `lhs r1*64+r0`, `rhs r0*64+r2`, `out r1*64+r2` |
| conv 3×3 pad1, 16→32, 32² | 1 | reduce over `(ic 16, kh 3, kw 3)`; input `r1*32+r4*32+(r2+r5)+r0*1024-33` with 4-clause rectangular mask; weight `r1*3+r2+r0*9+r3*144` |
| conv 1×1 stride2 | 1 | reduce over `(ic 16)`; input `r2*64+r3*2+r0*1024` |
| conv 3×3 groups=16 (depthwise) | 1 | reduce over `(kh,kw)` only; **shared** range `r2` appears in both input (`r2*1024`) and weight (`r2*9`) |
| conv 3×3 + relu | 1 | epilogue `WHERE(0 < acc, acc, 0)` before the store |
| conv 3×3 + bias (trace_conv shape) | 1 | 4 PARAMs; epilogue `acc.cast(half) + PARAM3.index(oc)` |

**B2 — fake TensorCore cost** (`rockchip_cmac` = `dims=(16,1,32), threads=1,
elements_per_thread=(32,512,16), opts=("u0","u0","u0","u0")`, reconstructed on master, `USE_TC=1
TC_OPT=2`, NULL renderer, `to_program_cache` cleared between runs)

| matmul | tc=none | tc=ROCKCHIP_CMAC |
|---|---|---|
| 32³ | 721 UOps, 0.34 s | 2404 UOps, 4.50 s |
| 64³ | 434 UOps, 0.20 s | 7129 UOps, 13.16 s |
| 128³ | 435 UOps, 0.19 s | **KernelOptError: locals needed for opt** |

The 128³ failure is `hand_coded_optimizations` applying `Opt(OptOps.LOCAL, …)` after a successful
TC opt while the renderer has `has_local=False` (`heuristic.py:43-44` → `postrange.py:134`). It is
uncatchable from the backend. The blow-up is `expander2` contracting the 4 upcast + 5 unroll axes
and `do_stack_wmma` then stacking each WMMA source element-by-element.

**B3 — Phase-1 adapter, verified on master**
Synthesized `PROGRAM(SINK_preopt, LINEAR(7 × Ops.INS))` for a 64³ matmul with a renderer whose only
method is `asm()`; ran the stock `pm_to_program`:

```
ProgramInfo: globals=(0,1,2) outs=(0,) ins=() global_size=(1,1,1) local_size=(1,1,1)
asm() received 7 commands, prg.src[0].op = Ops.SINK
PROGRAM srcs: ['SINK','LINEAR','SOURCE','BINARY']
SOURCE: "('WRITE_IMM', 3735879680)\n('WRITE_IMM', …)"      # free command listing
BINARY: 56 bytes, dtype uchar
TinyELF: name='test' target=NULL lib=56 signature=()
estimates: Estimates(ops=0, lds=0, mem=0)                  # -> planner must fill these (M5)
```

`ins=()` because loads are not lowered yet; it only affects PROFILE metadata.

**B4 — unit coverage, full `test/backend/test_ops.py` suite** (`DEV=NULL`, 424 test methods, 686
scheduled kernels captured at the §4.2 interception point via `test/rockchip/run_coverage.py`).
**These are *structural classification* counts, not hardware-execution counts. A kernel counted as
R2/R4 is one whose AST matches the recognition rule; whether it executes correctly is proven by
§19.4 tests, not by this table.**

Full suite (686 kernels):

| Class | Kernels | Share |
|---|---|---|
| R3 DPU elementwise (pure, no transcendentals) | 196 | 28.6% |
| R3 DPU elementwise (contains EXP2/LOG2/SIN/SQRT/POW) | 51 | 7.4% |
| R1 CNA/CMAC contraction | 103–106 | ~15% |
| R2 CNA/CMAC sum-as-GEMM | 60–85 | ~9–12% |
| R4 PPU reduce-max | 76 | 11.1% |
| REJECT: dtype int/bool/uint/uchar | 147–164 | ~21–24% |
| REJECT: gather / non-affine index | 72 | 10.5% |
| REJECT: multi-reduce / reduce:MUL | 8 | 1.2% |

(R1/R2/dtype counts vary slightly between runs because `pre()` graph traversal order is
nondeterministic; the unit-level summary is stable.)

**Unit summary:**

| Unit | With custom TC decomp | Without custom TC decomp |
|---|---|---|
| CNA+CMAC (R1+R2) | ~191 (27.8%) | ~191 (27.8%) |
| DPU (R3, pure) | 196 (28.6%) | 196 (28.6%) |
| DPU (R3, transcendental) | 51 (7.4%) | **rejected** |
| PPU (R4) | 76 (11.1%) | 76 (11.1%) |
| **structural coverage (all)** | **~75%** | **~67.5%** |
| **structural coverage (fp-only)** | **~98.5%** | **~88.7%** |

The 51 transcendental kernels (7.4%) are the difference between 88.7% and 98.5% fp-only coverage.
They include every activation function except ReLU (sigmoid, tanh, gelu, silu, swish, mish, selu,
softplus, erf) plus `exp`/`log`/`sqrt`/`pow`/`rsqrt`. In a real inference workload, activations are
ubiquitous — rejecting transcendentals makes the backend ReLU-only, which is too limited for a
useful V1. **The custom decomp is therefore recommended for PR 1**, but it is ~80 sz-lines of work
not yet in the budget (§15.1, §23.1).

The 36-op sample of rev 5/5.1 (which reported ~89% structural coverage) was a curated inference
workload. The full `test_ops.py` suite has a **~68% structural coverage rate without the custom TC
decomp, ~75% with it** — but ~22% of the reject is dtype-specific tests (int/bool/uint/uchar) that
are deliberately testing those dtypes, not inference workloads. **Among fp-only kernels, structural
coverage is 88.7% without the custom TC decomp and 98.5% with it** — the latter being the honest
target for "what the NPU would actually run in an fp16 inference context."

The fp-only reject (without TC decomp) breaks down as:
- **gather / non-affine index: 72 kernels (10.5% of all)** — fancy indexing, scatter, nonzero,
  one_hot, sort, argsort, topk. These are genuinely outside an fp16 fixed-function datapath.
- **transcendentals: 51 kernels (7.4%)** — fixable with the custom DPU polynomial decomp (§15.1).
- **multi-reduce: 4 kernels (0.6%)** — kernels with two REDUCE nodes; V1 rejects these.
- **reduce:MUL: 4 kernels (0.6%)** — `cumprod` (`REDUCE(MUL, INDEX(a))`); neither R1 nor R2.

Top reject-heavy tests: `test_split` (24, all int dtype), `test_chunk` (15, int dtype),
`test_arange` (9, int dtype), `test_avg_pool2d_padding` (9, gather from index tensors),
`test_slice_fancy_indexing_*` (multiple, gather).

Conclusions that drive §23.1:

- **~24% of all kernels (30% of fp-only) are structurally classifiable as CMAC** once R2 is
  implemented. The ones-vector trick is what makes "this backend uses the MAC array" a structural
  claim.
- **CNA + DPU + PPU structurally covers 68% of the full suite, 87% of fp-only kernels.** The
  full-suite number is the one to cite in the PR description; the fp-only number is the one that
  predicts inference-workload coverage. Both are honest.
- The reject list is dominated by dtype tests (int/bool/uint) and gather operations. Neither is
  fixable without adding an integer datapath or an index-producing unit — both are explicitly
  non-goals for V1 (§21).
- `conv2d`, `conv2d+bias`, `matmul`, `matvec` and `dot` are each **one** kernel, so PR 1's
  single-task-no-tiling policy is sufficient for the shapes `test_ops` actually uses.
- `batchnorm` produces 5 elementwise kernels and `softmax`/`layernorm` 3 each — i.e. the kernel count
  is dominated by DPU work even in "conv-heavy" models, which is why the DPU chain and the scratch
  buffer (§M9) are PR-1 requirements rather than follow-ups.

Reproduce with `python test/rockchip/run_coverage.py` (venv Python 3.12, `DEV=NULL`, ~53s, no
hardware). The 36-op sample probe is preserved at `/tmp/probe_coverage.py` for comparison.

---

# Appendix C — Sources reviewed

tinygrad core: `codegen/__init__.py`, `codegen/opt/{tc,postrange,heuristic}.py`, `codegen/gpudims.py`,
`renderer/__init__.py`, `renderer/isa/{__init__,x86}.py`, `renderer/amd/elf.py`, `uop/{ops,spec,upat}.py`,
`device.py`, `engine/realize.py`, `schedule/__init__.py`, `runtime/{ops_null,ops_cpu,ops_rockchip}.py`.

Branches: `rockchip-2607` (current), `rockchip_addmul`, `rockchip/wip`, `rockchip/backend-consideration`.

References: `ref/rk3588` (`conv_grok/{conv,gemm_npu}.py`, `examples/{gemm,conv_gemm}.py`),
`ref/npu` (`include/{rknnops.h,rkt_registers.h,rknpu-ioctl.h,npu_hw.h}`, `ops_reg/main.c`,
`docs/rk3588_trm.md`), NVDLA hw/sw (planning order only), GabrielNakamoto/tinygrad (native RDNA3
boundary), 2187Nick/tinygrad (RDNA3 ISA/perf reference).
