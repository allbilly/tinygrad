# RK3588 fixed-function NPU backend

This is the active coverage/research branch. It targets the complete forward
`test_ops.py` inventory through RK3588 engine tasks, with frontend-only methods
reported separately and unsupported kernels rejected before submission.
The frozen `rockchip-pr`, `rockchip-2608`, and `rockchip-2607` branches remain
minimal, architectural, and behavioral/register-programming references.

The current authoritative uncached strict census at `85eeab7f5` is 148 native,
40 frontend-only, 224 failed, and 13 upstream-skipped methods across the exact
425-method inventory. It completed without an NPU timeout, reset failure,
invalid submission, or process abort. Coverage details and durable artifact
hashes are recorded in `coverage.md`.

The compiler boundary is:

```text
post-early_simplify UOps
  -> Rockchip legality and affine analysis
  -> UOp-free typed plans
       RKALUStage(op: Ops)
       RKMaskStage
       RKLUTStage(lut: RKLUTId)
       RKContract
       RKProgram(ordered typed steps)
  -> RKImage commands and relocations
  -> DRM allocation, patch, submit, and wait
```

The native runtime does not import NumPy, narrow FP32 buffers, or execute tensor
semantics on the host. Native lowering returns a typed plan or a typed reject
at the legality decision, including detail, offending op, and a normalized
slot-independent graph fingerprint. `ROCKCHIP_FALLBACK=0` is the authoritative
mode for coverage and development. The explicit `RKPY` experiment is retained
only as a historical/debugging reference: it executes generic UOps on the CPU
and must never contribute to Rockchip pass totals.

Mixed-engine work is represented by a generic ordered `RKProgram`. Scratch is
declared once at program scope, and each typed DPU, CMAC, or PPU step is emitted
in order with constants and relocations remapped centrally. The former fixed
CMAC-prefix/DPU/main-CMAC/CMAC-suffix pipeline no longer constrains composition.

Lowering uses six named ordered passes for DPU arithmetic, affine reformat,
global sum, affine reduction, PPU reduction, and contraction. Every pass returns
exactly one of `NATIVE`, `NOT_APPLICABLE`, or `UNSUPPORTED`: unrelated passes
cannot overwrite a useful reject, while applicable failures are ranked by
specificity before telemetry is emitted.

The compiler is split by responsibility under `renderer/rockchip/`:
`ir.py` owns the typed UOp-free plans, `expr.py` owns math recipes and UOp
canonicalization, `affine.py` owns affine maps and reject fingerprints,
`emit.py` owns DPU/CMAC/PPU register emission, and `image.py` owns the versioned
image codec and relocations. The package entry contains resource planning,
ordered legalization, and renderer integration. Register emission imports no
UOp definitions and cannot recover source-graph semantics.

The frozen `rockchip-2607` branch is a behavioral and register oracle, not
evidence that all of its 425 passing methods ran on the NPU. Its later runtime
contains NumPy-backed `_run_host_*` implementations for generic elementwise and
reductions, variance, interpolation, losses, tangent, and large einsum. Only
paths that reach `DRM_IOCTL_RKNPU_SUBMIT` without host semantic preprocessing
are eligible to be ported as native capabilities.

## Declared contract

- static, contiguous FP16 elementwise ADD, SUB, MUL, MAX, and division;
- scalar operands and FP16 fills through the same ALU stages;
- FP16 `WHERE` with a directly representable less-than mask and finite arms;
- generated EXP2, EXP, sigmoid, refined SQRT/RSQRT, logarithm, inverse
  trigonometric, and inverse-hyperbolic LUT assets with declared domains;
- direct FP16 CMAC for `M=1`, `K=32`, and `4 <= N <= 16` when the right-hand
  input is already stored as `(N, 32)`;
- direct FP16 sums of 4–16 dense rows of length 32 using an image-owned ones
  vector and the same CMAC contract;
- dense FP16 global MAX/MIN through an ordered DPU block tree, CMAC lane
  reformat, and PPU spatial reduction, including a direct scalar scale;
- small static affine FP16 MAX reductions, including scalar multi-axis output,
  batched as eight PPU channels after cost-bounded CMAC reformatting;
- statically masked affine FP16 MAX reductions, with invalid coordinates mapped
  to an NPU-filled negative-infinity sentinel before CMAC/PPU reduction;
- larger affine FP16 MAX surfaces split into bounded, atom-aligned source
  windows shared greedily across consecutive PPU batches under the global
  stage/constant budgets; masked windows use a 192-lane cost target, while
  unmasked CMAC groups use compact atom-aligned subviews of a shared pack;
- affine MAX atoms that span more than K512 reduce each logical output into an
  aligned PPU scratch atom, then gather lane zero through sparse CMAC;
- one static affine FP16 input reformat/broadcast materialized by cost-bounded
  CMAC before the ordinary generic DPU expression;
- contiguous FP16 global sums whose power-of-two block decomposition fits a
  32-term aligned DPU/CMAC plan;
- two-level scalar ADD reductions whose affine axes form a proven dense input
  bijection, executed as ordered CMAC plans with the intermediate FP16 rounding
  boundary preserved in NPU scratch;
- dense scalar FP16 MUL reductions of 2–32 values, with every logical lane
  materialized into an addressable NPU atom before source-order DPU folding;
- affine FP16 MUL reductions with 2–32 terms, materialized as one full NPU
  surface per reduction coordinate and folded by vector DPU stages;
- statically masked affine FP16 MUL reductions with multiplicative-identity
  surfaces selected from an NPU-filled ones atom;
- direct aligned K64–K512 CMAC sums or scaled sums with generated
  hardware-packed weights and a single FP32 accumulation;
- global ReLU-sum through a DPU MAX-zero prepass and the same direct CMAC;
- static affine FP16 ADD reductions with at most 512 input and 128 output
  elements, using NPU zero-fill/copy preparation and sequential sparse CMAC
  tiles of at most 16 logical outputs; constant mean scaling is folded into
  the weights;
- larger static affine FP16 ADD reductions split into atom-aligned source
  windows, provided the scale is exactly FP16, the plan has at most 400 stages,
  and affine visits and unique constants remain within declared budgets;
- wide dense affine FP16 ADD reductions whose single-level windows are illegal,
  reduced into padded scalar atoms and compacted by a second CMAC level;
- static affine FP16 ADD reductions selecting among 2–8 input surfaces, with
  source-local CMAC selectors combined by ordinary DPU ADD stages;
- static affine FP16 movements with at most 512 source and 4,096 output
  elements: aligned runs use DPU atom copies, while arbitrary selector maps up
  to a 2 MiB generated-weight budget use the same sparse CMAC pipeline;
- sub-atom FP16 DPU stages program their logical channel count, so NPU-zeroed
  physical padding remains defined when a scalar or short tail feeds CMAC;
- one demonstrated two-kernel workload: direct `(1,32) @ (8,32).T`, followed
  by bounded sigmoid using generic ALU stages and two sigmoid LUT assets.

Native arithmetic is FP16. Int32 and FP32 are currently admitted only for
operation-specific constant fills; this does not claim general arithmetic for
either dtype. User-visible bool outputs, noncontiguous elementwise layouts,
reductions outside the proven static affine bounds, general contractions,
fused CMAC epilogues, general convolution/pooling, and gradients remain outside
the native contract. A fused CMAC epilogue is rejected rather than silently
dropped.

## EXP2 generation and characterization

Run `python extra/rockchip/gen_lut.py` to regenerate
`tinygrad/runtime/autogen/rockchip_lut.py`. The generated file records its
schema, payload SHA-256 digest, domain, interpolation parameters, exhaustive
simulator population, and simulator error bounds.

Every one of the 32,770 FP16 encodings in `[-2, 2]` is checked in the simulator
and on RK3588 hardware. The 2026-08-01 hardware sweep measured:

- maximum absolute error: `0.0010883808135986328` at `1.9560546875`;
- maximum relative error: `0.0008962760912254453` at `-1.9326171875`;
- maximum error versus correctly rounded FP16: one ULP;
- monotonic output over the complete declared domain;
- exact `exp2(0) == 1`.

Inputs outside the declared domain are not claimed by this first LUT contract.
The EXP implementation uses two sequential NPU LUT tasks: a broad table over
`[-2, 2]` and a higher-resolution local table over `[-0.25, 0.25]`. See
`extra/rockchip/lut.md` for the table scaling, selection, and tuning procedure.

## Validation

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest \
  test/null/test_native_program.py test/unit/test_rockchip_image.py \
  test/unit/test_rockchip_compiler.py -q -n12
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest test/device/test_rockchip.py -q
python -m ruff check .
python -m mypy tinygrad/
python extra/rockchip/gen_lut.py
```

The hardware suite is serial. Anything included in it is a promised capability
and has no skip on an RK3588 host.

The current committed strict census at `bc5c4353a` contains 137 native passes,
40 frontend-only passes, 235 failures, and 13 upstream skips across exactly 425
methods. It ran uncached with `ROCKCHIP_FALLBACK=0`; no prior native pass
regressed and there were no NPU timeouts, reset failures, invalid submissions,
or process aborts. First-reject counts are 64 unsupported-output-dtype, 48
plan-limit, 35 unsupported-layout, 30 requires-reformat, 25 unsupported-ALU,
23 unsupported-input-dtype, and 4 unsupported-reduction; six failures execute
natively but miss the numerical or frontend contract. Larger contraction
packs remain typed plan-limit rejects when they exceed the 2 MiB constant
allocation or affine-analysis budgets. Static conditional FP16 `tril`/`triu`
subcases are native, while their explicit boolean-output subcases remain honest
dtype rejections.

The nine exact transitions from the preceding 128-pass census are `conv1d`,
`conv2d`, `copysign`, both remaining MAX-pool dilation/smaller-stride families,
`medium_grouped_conv2d`, both simple Conv2D families, and padded Conv1D. This
also confirms two generic gains that were not claimed from focused testing.
The durable telemetry JSON and JUnit XML are stored under
`~/rk2608_backups/census-conv-bc5c4353a-20260803/`.

Tiled contraction legality is checked before coordinate enumeration: K,
source extents, and at most 65,536 affine output/reduction visits bound compiler
work. Oversized graphs reject before NPU submission.

The first three MAX gains folded into that census are `test_max_pool2d_simple`,
`test_max_pool2d_ceil_mode`, and
`test_max_pool2d_ceil_mode_output_size_reduce_by_one`. Their static affine MAX
selectors are packed by bounded CMAC tasks and reduced by PPU; ceil-mode
padding is an NPU-filled negative-infinity sentinel selected by compile-time
coordinate predicates. There is no operation-name path or host semantic work.
Dense single-axis global MAX remains on the proven padded DPU tree. The strict
hardware suite passed 58 tests plus 37 subtests at that milestone.

Windowed affine MAX subsequently makes `test_max_pool2d_bigger_stride` and
`test_max_pool2d_bigger_stride_dilation` native, including all nine subtests.
Each PPU output batch copies only its bounded source span and the compiler
accounts for actual hardware stages and unique constant payloads before
acceptance. The strict hardware suite passes 59 tests plus 37 subtests; the
gains are folded into the current `6133442c2` census.

The next cost milestone coalesces consecutive output batches that share a
bounded source span. Complete `test_max_pool2d_padding` passes all nine
subtests with 246–336 stages per plan. A rejected 128-lane configuration left
one plan at 400 stages and eventually aborted in the driver's reset ioctl; the
accepted 192-lane target passes both the full method in one process and the
later strict hardware suite. This gain is folded into the current 128-pass
census.

After the census, compact CMAC scratch subviews make all four
`test_max_pool2d_smaller_stride` subtests native. The two remaining plans fall
inside the existing 400-stage and 2 MiB limits without changing either bound.
The strict hardware suite remains 59 tests plus 37 subtests, and the inferred
native total is 129 pending the next complete census.

Wide affine-MAX output atoms subsequently make `test_max_pool2d_dilation`
native. Each logical result is reduced into an aligned scratch atom entirely on
DPU/CMAC/PPU, then the generic sparse-CMAC reformatter gathers dense output.
The strict suite passes 60 tests plus 37 subtests; the inferred native total is
130 pending the next complete census.

Sparse affine contraction subsequently admits block-diagonal output/input pair
sets and applies the 2 MiB ceiling to the emitted program's deduplicated
constants. The NPU computes a legal Cartesian superset and a static CMAC
selector writes only graph-proved outputs. Explicit physical tail allocation
accounts for every CMAC task writing 32 lanes at a 16-logical-lane stride; this
fixes an exact-page RHS selector timeout. All seven `test_conv2d` subtests pass
natively, and the strict suite passes 61 tests plus 39 subtests. The inferred
native total is 131 pending the next complete census.

The selector cost milestone compares full sparse and bounded windowed CMAC
plans by typed stage, unique-constant, and scratch costs. Empty tiles become one
NPU zero-fill, and every direct window boundary is destination-atom aligned.
When direct selection cannot satisfy both the K512 and atom constraints, a
generic two-level NPU plan gathers into padded source-local scratch and then
compacts into dense output. This makes `test_simple_padding_conv1d` native with
a 395-stage/856,464-byte plan and keeps the full strict suite green at 62 tests
plus 39 subtests. The inferred native total is 132 pending the next census.

The contraction M ceiling subsequently becomes resource-derived: `M<=128`
while `M*aligned_K<=4096`, with the completed plan still limited to 400 stages
and 2 MiB. This admits the bounded M81/K4/N4 1x1-convolution plan without
admitting oversized M128 work. Complete `test_simple_conv2d_1x1` passes
natively, the strict suite passes 63 tests plus 39 subtests, and the inferred
native total is 133 pending the next census.

Aligned windowed selectors subsequently address typed source subviews directly
when the complete physical K span is inside the declared allocation; unsafe
tail windows still use NPU scratch packing. This removes two DPU copy stages
per safe window for contraction and affine-reduction plans. Complete
`test_simple_conv2d` becomes native with a 242-stage/331,312-byte plan, the
strict suite passes 64 tests plus 42 subtests, and the inferred native total is
134 pending the next census.

Contraction compute subsequently tiles tall M surfaces into ordered tasks that
each satisfy `tile_M*aligned_K<=4096`. A generic multi-broadcast lowerer also
materializes every static affine FP16 source before ordinary DPU arithmetic,
covering K=1 graphs whose reduction simplifies away. All 14 `test_conv1d`
subtests become native, the strict suite passes 65 tests plus 44 subtests, and
the inferred native total is 135 pending the next census.

Generic reduction epilogues now keep the contraction result in typed NPU
scratch, materialize static affine operands with the selector planner, and
schedule ordinary DPU arithmetic into the final output. This is an ordered
`RKProgram` capability rather than a bias or convolution recognizer. Complete
`test_simple_conv2d_bias` passes natively with 258 stages, 347,696 constant
bytes, and 18,592 scratch bytes. The other three methods that previously first
rejected on epilogue legality now reach their honest contraction resource
bounds. The strict suite remains 65 tests plus 44 subtests, and the inferred
native total is 138 after the current 137-pass census.

Static predicates inside affine SUM bodies now become empty selector entries
instead of an attempted DPU expression. This is the generic masked-reduction
form used by short prefix sums and zero-padded windows. Complete
`test_small_cumsum` passes natively with two DPU pack stages and one CMAC task;
large quadratic selector surfaces and FP32 accumulation remain bounded rejects.
The strict suite expands to 66 tests plus 44 subtests, and the inferred native
total is 139 after the current 137-pass census.

Threshold `WHERE` expressions with a true signed-infinity arm now avoid the
undefined `infinity*0` selection form. The compiler clamps the unmasked source
to the threshold, constructs signed infinity as `mask/(1-mask)`, and adds the
two entirely through DPU stages. Both comparison directions preserve NaN and
opposite-infinity behavior on hardware. Complete `test_masked_fill` is native,
the strict suite expands to 67 tests plus 44 subtests, and the inferred native
total is 140 after the current 137-pass census.

Nested scalar sums subsequently use the generic ordered `RKProgram` rather
than an algebraic flattening. The compiler proves the combined affine source
map, reduces the inner axes to FP16 scratch, then reduces that materialized
surface into the scalar output. This preserves the source graph's intermediate
rounding and makes all three `test_sum_twice` variants native. The strict suite
expands to 68 tests plus 47 subtests, and the inferred native total is 141 after
the current 137-pass census.

Short scalar products subsequently use CMAC lane selection to obey the DPU's
16-byte source-address granularity, followed by source-order DPU MUL stages.
No host gather or arithmetic participates. Complete `test_const_reduce` is
native, the strict suite expands to 69 tests plus 50 subtests, and the inferred
native total is 142 after the current 137-pass census.

Affine products subsequently transpose short reductions into full term
surfaces through the generic selector planner and fold them with vector DPU
MUL stages. Complete `test_prod` is native, the strict suite expands to 70
tests plus 52 subtests, and the inferred native total is 143 after the current
137-pass census.

Static predicates in short affine products subsequently select either the
indexed input or an NPU-generated identity-one surface. Complete
`test_small_cumprod` is native, the strict suite expands to 71 tests plus 52
subtests, and the inferred native total is 144 after the current 137-pass
census.

Wide dense row sums subsequently invoke the generic two-level selector rather
than stopping after one-level window rejection. FP16 constant fills also tile
at the hardware-proven 32,768-lane width; an untiled 65,528-lane probe timed
out, while two tiles fill 65,536 lanes exactly. Complete `test_sum_collapse` is
native, the strict suite expands to 73 tests plus 52 subtests, and the inferred
native total is 147 after the current 146-pass census.

Static multi-source row sums subsequently follow compile-time `WHERE` branches
at every affine coordinate, emit one cost-bounded selector plan per FP16 input,
and combine the partial surfaces with DPU ADD. Complete
`test_sum_cat_collapse` is native with 178 stages, 71,680 constant bytes, and
3,168 scratch bytes; a nonconstant hardware test covers the same path. The
strict suite expands to 74 tests plus 52 subtests, and the inferred native total
is 148 after the current 146-pass census.

The next census exposed that two-level scaled reductions had silently dropped
their scale in the compaction level. K3 average-pooling outputs were therefore
the raw sum, exactly 9x or 6x too large, and nine long subcases exceeded the
method-level watchdog. The second CMAC level now carries exactly representable
scales; unrepresentable FP16 weight scales return a typed numerical-contract
reject before submission. A 0.25
wide scaled reduction passes hardware, the pooling method completes under its
original watchdog, and the strict suite passes 74 tests plus 54 subtests. This
is a correctness/reliability milestone, not a new complete-method pass.

The corrected complete census subsequently confirms both pending gains:
`test_sum_collapse` and `test_sum_cat_collapse` are the only transitions from
146 to 148 native methods, with no regression. It finishes in 2,526.78 seconds
without a timeout, reset failure, invalid submission, or process abort. The
fresh largest first-reject classes are output dtype (65), plan limit (53),
layout (35), input dtype (23), and reformat (18).

The subsequent windowed-reduction milestone does not claim another complete
method: it proves exact 2x2 affine average windows over a 1,232-element input,
while non-exact reciprocals reject. Ordered image composition deduplicates
identical constant payloads. Reset-per-program, a two-term approximate
reciprocal, and K=640 dynamic CMAC were tested and rejected; the full hardware
suite remains green with per-stage reset and the K<=512/400-stage bounds.

The RK3588 BN multiplier was then characterized as a possible pre-rounding
scale. Enabling BN with `BN_CFG=0x20040` and a register FP16 multiplier really
does scale the FP32 CMAC accumulator before its final FP16 store. Factoring
1/9 into two FP16 operands reduced the official 3x3 average-pool error to
0.09--0.31% one-ULP mismatches, but did not eliminate it; 3x2 remained at
2.15--2.57%. Clearing `fp32tofp16_en` and using the integer output-converter
scale/shift produced zeros, while a combined CMAC plus EW-divider task timed
out and recovered after reset. Strict non-exact-scale rejection therefore
remains correct. The complete rejected implementation and probe results are
preserved in `0134-WIP-fused-BN-two-factor-scale.patch` (SHA-256
`a472dcbfb2a610e0aa0afe86f0318ad7e293d97417cf11067a6e94473054c22c`).

## Current upstream blocker

The base master contains 24,968 counted lines. This research branch currently
contains 28,351, so `MAX_LINE_COUNT=25000 python sz.py` fails by 3,351 lines.
The exact 3,383-line delta is 3,378 counted Rockchip backend lines (3,218
renderer/compiler, 111 runtime, 33 historical Python-fallback adapter, and 16
telemetry support) plus the five-line generic native-program hook. Generated
register definitions, LUT payloads, and reproducible command data belong under
`runtime/autogen`; handwritten legality, layout, scheduling, and emission logic
remain counted. The generic hook can be reviewed separately, while the backend
needs real upstream line budget or independently useful in-tree reductions
before submission.
