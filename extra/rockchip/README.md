# RK3588 fixed-function NPU backend

This is the active coverage/research branch. It targets the complete forward
`test_ops.py` inventory through RK3588 engine tasks, with frontend-only methods
reported separately and unsupported kernels rejected before submission.
The frozen `rockchip-pr`, `rockchip-2608`, and `rockchip-2607` branches remain
minimal, architectural, and behavioral/register-programming references.

The current authoritative uncached strict census at `2385767d1` is 204 native,
40 frontend-only, 168 failed, and 13 upstream-skipped methods across the exact
425-method inventory. It completed in 3,375.14 seconds without an NPU timeout,
reset failure, invalid submission, process abort, numerical mismatch, or
unclassified failure. Coverage details, the checkout-local pytest-plugin
invocation, reject Pareto, plan-cost histograms, and durable artifact hashes
are recorded in `coverage.md`.

The PPU layout contract now names the actual `PPU_HWC` format and accepts every
hardware-characterized FP16 channel count from two through eight. This follows
the variable `align_c` register contract in the local `allbilly/rk3588`
`rknnops.h` oracle and is independently verified on RK3588. It does not change
the focused TestOps tally because the relevant softmax methods deliberately run
their tinygrad reference-gap side in FP32.

Every remaining failure is a typed native reject; there are no device or
unclassified frontend failures in the authoritative inventory. Dynamic tensor
indexing is still rejected.

The compiler boundary is:

```text
post-early_simplify UOps
  -> Rockchip legality and affine analysis
  -> UOp-free typed plans
       RKALUStage(op: Ops)
       RKMaskStage
       RKLUTStage(lut: RKLUTId)
       RKContractionPlan -> RKCMACTask
       RKConvPlan -> RKConvTask(s)
       RKPool
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
declared once at program scope, and each typed DPU, CMAC, CONV, or PPU step is emitted
in order with constants and relocations remapped centrally. The former fixed
CMAC-prefix/DPU/main-CMAC/CMAC-suffix pipeline no longer constrains composition.

`RKPlanCost` accounts for task and reset counts, emitted command words,
deduplicated constants, scratch, estimated engine reads/writes, and CMAC/DPU
work. Legal selector candidates still respect the 400-task and 2 MiB constant
ceilings, then compare reset overhead, MACs, traffic, command volume, constants,
and scratch. Runtime telemetry records exact task/command/reset counts and marks
plans over 64 tasks or 1 MiB of constants as `CORRECTNESS_FALLBACK`; these remain
honest native passes but are kept visible for replacement by direct engine paths.
The current serialized device contract passes 96 tests plus 66 subtests in
807.11 seconds with fallback disabled.

Lowering uses twenty-one named ordered strategies grouped into elementwise,
movement/reformat, sum/product/MAX reduction, and contraction families. Every
strategy returns exactly one of `NATIVE`, `NOT_APPLICABLE`, or `UNSUPPORTED`: unrelated passes
cannot overwrite a useful reject, while applicable failures are ranked by
specificity before telemetry is emitted.

The compiler is split by responsibility under `renderer/rockchip/`:
`ir.py` owns the typed UOp-free plans, `expr.py` owns math recipes and UOp
canonicalization, `affine.py` owns affine maps and reject fingerprints,
`access.py` owns compact semantic identity, affine, padding, periodic,
piecewise-affine, and final static-selector access maps,
`emit.py` owns DPU/CMAC/CONV/PPU register emission, and `image.py` owns the versioned
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
- fused FP16 lerp through CMAC FP32 operand materialization followed by the
  DPU BS/BN/EW pipeline, preserving the single FP32 intermediate boundary;
- FP16 `WHERE` with a directly representable less-than mask and finite arms;
- generated EXP2, EXP, sigmoid, periodic SIN, refined SQRT/RSQRT, logarithm,
  inverse-trigonometric, and inverse-hyperbolic LUT assets with declared domains;
- direct FP16 CMAC for `M=1`, `K=32`, and `4 <= N <= 16` when the right-hand
  input is already stored as `(N, 32)`;
- bounded affine FP16 contractions with NPU selector packing, tiled CMAC, and
  output compaction; a one-value-per-output-channel FP16 bias is gathered and
  converted to FP32 on the NPU, then fused through BRDMA before the first FP16
  writeback, with optional ReLU in the same DPU flying-data stage; one CMAC
  task may produce up to 128 logical channels in proven 32-channel groups,
  with each 16-channel FP16 block normally occupying one gapped 32-lane WDMA
  atom; a hardware-proven one-row compact mode writes those blocks contiguously;
- dense multi-row FP16 contraction LHS surfaces whose physical row stride is
  already the CMAC K alignment are consumed directly; a separately proven
  2,048-lane selector window packs dynamic RHS tiles without raising the global
  400-task or 2 MiB image limits;
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
- scalar FP16 global sums may commit the unchanged CMAC FP32 accumulator to a
  public FP32 WDMA surface; FP32 remains an output-only contract for this path;
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

Native arithmetic is FP16. Int32 and FP32 are admitted for operation-specific
constant fills, and scalar FP16 sums can retain their FP32 CMAC accumulator at
writeback; this does not claim general FP32 input or arithmetic support.
User-visible bool outputs, noncontiguous elementwise layouts,
reductions outside the proven static affine bounds, general contractions,
CMAC epilogues other than the proven channel-bias/optional-ReLU form, general
convolution/pooling, and gradients remain outside the native contract.

Accepted elementwise plans are also gated by structural numerical contracts.
Multistage `lerp` requires a future fused FP32-intermediate task; dynamic
`copysign` requires a path that preserves reciprocal sign for negative zero;
unreduced probability BCE exceeds the current staged LUT error bound; and an
arbitrary constant-base power needs a separately characterized scaled-EXP2
contract. These graphs reject before submission rather than returning sampled
answers outside TestOps tolerance. Proven EXP/activation canonicalizations and
the repaired zero-base power identity remain native.

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

Channel bias and optional ReLU now use a typed `RKEpilogue` on `RKCMACTask`.
The selector planner gathers the logical FP16 bias into aligned four-lane
atoms, ordinary DPU tasks convert those lanes into the FP32 format consumed by
BRDMA, and the CMAC emitter performs bias/ReLU on the flying FP32 accumulator.
This removes the incorrect intermediate FP16 rounding boundary without host
packing. The two official `test_biased_conv2d` kernels are now 87 stages each
(previously 92 and 91) and the complete method passes at its unchanged
tolerance. `test_simple_conv2d_bias` also remains green. The strict device
suite passes 76 tests plus 54 subtests in 729.98 seconds.

Lerp now uses a generic `RKFusedALUStage` rather than three rounded FP16
stages. CMAC writes the indexed `x` operand directly as FP32 in 32-lane tiles;
each following DPU task executes `BS(y-x)`, `BN(*z)`, and `EW(+x)` over one
eight-channel atom before the only FP16 writeback. BRDMA is channel-broadcast,
so the compiler emits one fused task per atom. A 1,575-element official kernel
uses 50 CMAC tasks plus 197 DPU tasks, 2,048 constant bytes, and 6,528 scratch
bytes, and is bit-exact against the FP32-intermediate reference. Partial DPU
atoms are submitted as eight channels because four-to-seven-channel fused
tasks time out; unused lanes stay outside the logical output. The unchanged
official `test_lerp` method passes in strict native-only mode.
The raw operand-format, conversion-width, and fused-pipeline probes remain
reproducible in `extra/rockchip/probe_lerp_pipeline.py`.
The complete strict Rockchip device suite passes 77 tests plus 54 subtests in
731.46 seconds with fallback disabled.

`extra/rockchip/probe_ppu_channels.py` records why dense row reductions cannot
simply lower the PPU channel field below eight: tightly packed HWC1/2/4 returns
wrong maxima, `NONALIGN` times out, and unaligned DPU source addends do not
reduce within an atom. HWC8 remains the declared direct PPU layout.

`extra/rockchip/probe_cmac_width.py` proves direct logical output widths through
128. Normal FP16 WDMA stores every 16-channel block in a 32-lane atom, while
`DPU_SURFACE_ADD=0x20` writes one-row CMAC results contiguously for every probed
N from 16 through 128. Multi-row output retains a padded physical row stride,
so compact emission is deliberately restricted to M=1. The same probe proves
that one CMAC selector can gather a stride-99 pattern through a 1,504-lane
input window exactly.

The tiled RHS pack alone may use that proven window; unrelated affine planners
retain their 512-lane cap. Zero-weight padded lanes may read only the known
4-KiB-rounded GEM tail. Together with direct contiguous LHS use and one DPU
copy from compact CMAC scratch, `1x64 @ 64x99` now has 399 tasks, 803,024 unique
constant bytes, and 33,056 scratch bytes without raising either global limit.
Random hardware output is bit-exact against FP32 accumulation followed by FP16,
and the unchanged strict `TestOps.test_matmul` passes in 48.05 seconds. The full
uncached census confirms this transition with no prior accepted method regressing
other than the deliberate full-domain `copysign` numerical-contract guard.

`extra/rockchip/probe_dpu_surface_reformat.py` preserves the rejected alternative:
aligned scalar DPU rows remain eight lanes apart, line-notch variants do not
change the gather, and DPU NONALIGN times out. A known-good DPU recovery passes
after reset. The complete strict device suite passes 79 tests plus 56 subtests
in 728.41 seconds with fallback disabled.

The first direct spatial-convolution milestone removes the nearest remaining
stage-limit blocker without raising the 400-task ceiling. An exact affine
matcher recognizes dense FP16 stride-one NCHW/OIHW convolution by enumerating
axis roles and verifying the complete input, weight, and output stride maps;
it does not inspect Tensor operation names. The initial hardware contract is
deliberately narrow: four or sixteen input channels, at most sixteen output channels,
spatial kernels up to 3x3, input dimensions at most sixteen, and at most four
batches. NPU selector tasks pack NCHW input and OIHW weights into proven CNA
surfaces, one typed `RKConvTask` task runs per batch, and selectors unpack
the physical C1/W/C2 output. No host packing or tensor semantics are involved.

For the official `(2,4,9,9) * (4,4,3,3)` graph, the old affine-CMAC plan needed
421 tasks: 310 for im2col-like input packing, 12 for weight packing, two useful
CMAC tiles, and roughly 81 output-layout tasks. The direct plan needs 92 tasks,
4,208 command words, 92 resets, 700,384 constant bytes, 5,344 scratch bytes,
and 294,632 estimated MACs. The global 400-task and 2 MiB constant ceilings are
unchanged. The first emitter attempt incorrectly assumed a four-lane CNA input
contract and produced tiny nonconforming values; comparison against the
preserved `conv_simple.py` oracle exposed the required eight-lane CNA channel
alignment, NHWC conversion mode, weight padding, DMA strides, and CVT field.
The corrected register map is identical to the known-good direct task; the
`RKImage` adds only its normal explicit PC enable word.

Unchanged strict `TestOps.test_simple_conv2d_batched` passes in 10.97 seconds.
A separate randomized hardware regression verifies numerical parity and that
telemetry records two `CONV` tasks. All 104 host tests plus three subtests pass,
mypy is clean across 225 modules, touched-file Ruff is clean, and the complete
serialized device suite passes 79 tests plus 56 subtests in 728.68 seconds.
The complete census below confirms this failure-to-native transition.

The second direct-convolution milestone adds the independently proven
16-input-channel layout. The exact affine matcher also accepts batch-one graphs
whose unit batch axis was removed by simplification. Channel-16 activations are
packed as C1/H/W/C2 with eight-lane inner atoms; weights use the matching
KH/KW/OC/IC physical order. The emitter selects the corresponding non-NHWC CNA
conversion, CBUF, and DMA-stride fields. The wider 1,504-lane selector window
is used only for this typed convolution input pack and remains bounded by the
same page-tail and zero-weight proof as the wide contraction pack.

The official `(1,16,9,9) * (16,16,3,3)` graph becomes one direct `CONV` task
plus NPU packing/unpacking: 275 total tasks, 12,652 command words, 275 resets,
899,072 constant bytes, 8,864 scratch bytes, and 1,341,344 estimated MACs.
Its register map matches the preserved known-good channel-16 task exactly.
Strict `TestOps.test_simple_conv2d_m4` passes in 30.99 seconds, and a randomized
hardware regression verifies numerical parity and one telemetry-visible
`CONV` task. All 105 host tests plus three subtests pass, mypy and touched-file
Ruff are clean, and the expanded serialized device suite passes 81 tests plus
56 subtests in 750.28 seconds. The complete census confirms this second
failure-to-native transition as well.

The complete uncached strict census at `c65396da1` finishes in 2,498.83 seconds
with **152 native**, **40 frontend-only**, **220 failed**, and **13
upstream-skipped** methods. Raw pytest reports 204 passed, 256 failed, 13
skipped, and 78 passing subtests. Relative to `d1437ad58`, exactly
`test_simple_conv2d_batched` and `test_simple_conv2d_m4` change from failure to
native pass; no previously accepted method regresses. The run completes with
no NPU timeout, invalid submission, reset failure, or process abort.

The durable telemetry and JUnit artifacts are stored under
`~/rk2608_backups/census-direct-conv-c65396da1-20260803/`. Their SHA-256 hashes
are `7dd09a9152e43af0fbc726ce71e4972d2bd3eff3a96c17d26c0d7d3f47a3a369`
and `427a54dc6a4b9fac478b0ab8b548c80740a78b5483c14d121801ca602b758867`.
Aggregating rejects retained by failed subcases reduces the apparent 15
missing method-level rejects to three true non-reject failures. The resulting
first-reject Pareto is 65 unsupported-output-dtype, 50 plan-stage-limit, 35
unsupported-layout, 23 unsupported-input-dtype, 18 requires-reformat, 13
numerical-contract, 10 unsupported-ALU, and three unsupported-reduction.

Telemetry schema version 2 now promotes the earliest reject attached to a
failed subcase into the method's `first_reject`, while retaining every original
event under its subcase. It also distinguishes `NATIVE_REJECT`,
`DEVICE_FAILURE`, `POST_EXECUTION_FAILURE`, and `NON_DEVICE_FAILURE`; failure
class and message are preserved for phase and subcase failures. Thus a
numerical mismatch or frontend/compiler exception is no longer reported as a
missing or fabricated native reject.

The census exposed one intermittent accepted numerical failure in the masked
product scan: the 20-output program was bit-exact in isolated reruns but
returned a corrupted result after the earlier mixed-engine census workload.
The proven 10-output, one-CMAC-tile path remains native and passes on hardware.
Outputs above sixteen now reject with `NUMERICAL_CONTRACT` before submission
until an alternating-engine stress test proves a stable multi-tile contract.
The unchanged focused `test_cumprod` therefore fails honestly at its first
20-element case instead of sometimes returning incorrect data.

FP32 constant fills still consume an FP16 DPU input even though WDMA writes a
wider public dtype, so legalization requires exact FP16 round-trip. Int32 fills
now use a different proven hardware contract: the FP16 arithmetic lane emits
zero and the DPU's 32-bit output-converter offset supplies the exact signed
int32 bit pattern. This covers the complete signed int32 constant range,
including `INT_MIN` and `INT_MAX`, without host conversion or a claim of
general integer arithmetic. FP32 `0.1` remains a typed numerical rejection.

Static affine movement now lowers to a first-class `RKReformat` target plan.
It records typed logical source and destination surfaces, the complete static
output-to-input map, and the selected implementation kind:
`COALESCED_DPU` or `SELECTOR_CMAC`. Its implementation is an ordered tuple of
already-legal, UOp-free engine steps; the emitter only serializes that choice.
This changes no command bytes or coverage, but makes selector CMAC an explicit
fallback that can be compared against future CNA/ERDMA/PPU implementations.
Strict transpose, permute, and flip hardware methods remain green. The complete
serialized device suite passes 81 tests plus 56 subtests in 746.38 seconds;
all 110 Rockchip host tests plus three subtests pass as well.

Windowed selector CMAC now uses the proven compact one-row 32-output contract
instead of emitting at most sixteen logical outputs per task. The 32-channel
weight/output surface is already covered by the exhaustive N=16..128 hardware
probe, and tail padding remains in scratch rather than a user allocation. This
halves many packing/unpacking task groups without changing the 400-task or 2
MiB ceilings. Batched direct convolution drops from 92 to 49 tasks and 9.82 to
5.23 seconds; channel-16 drops from 275 to 139 tasks and 29.37 to 14.87 seconds.
Both unchanged official methods remain bit-exact. The complete device suite
passes 81 tests plus 56 subtests in 700.77 seconds.

The complete uncached compact-selector census confirms that this optimization
does not change method outcomes: 152 native, 40 frontend-only, 220 failed, and
13 upstream-skipped. It completes 153.36 seconds faster than the preceding
direct-CONV census. The authoritative telemetry and JUnit artifacts are under
`~/rk2608_backups/census-wide-selector-d237777da-20260803/`, with SHA-256
`cfc3dffab84ac7c1c84c2fede9213a631d3d205d3a757408bc828896afb36e64` and
`8b77a964e12230ffa3115633162051b17819e5bf74b16a9de78b7e3e0aa26f03`.
Schema version 2 resolves 218 failures to native rejects and leaves only the
state-sensitive 20-element cumulative sum plus one pre-device Clang failure
without a typed first reject.

The accepted cumulative-sum mismatch is now closed conservatively. A static
prefix selector with more than sixteen outputs returns `NUMERICAL_CONTRACT`,
matching the proven one-tile boundary already used for masked cumulative
products. The ten-output prefix sum remains bit-exact on RK3588; unchanged
official `test_cumsum` now rejects its 20-output case before submission rather
than exposing the state-sensitive two-tile result.

Pure affine `RKReformat` and the already proven direct-CONV packers now use up
to 64 compact CMAC outputs per selector task. An 8x8 transpose becomes one
CMAC task, batched direct convolution falls from 49 to 28 tasks, and the
channel-16 case falls from 139 to 80. The complete device suite passes 81
tests plus 56 subtests in 680.95 seconds. Generic tiled contraction deliberately
keeps a 32-output selector fence: widening it exposed a 245-task transposed
convolution whose output missed the official tolerance, so that graph remains
the prior typed 415-stage rejection rather than an accepted numerical mismatch.

The proven rockchip-2608 periodic SIN implementation is now expressed through
the clean typed stage IR. Two generated payloads retain their reference
SHA-256 digests, while generic DPU arithmetic performs split `2*pi` range
reduction, local/broad selection, near-zero identity, and non-finite NaN
propagation. The result is a 56-task, seven-scratch NPU-only plan. The unchanged
strict `TestOps.test_sin` passes in 19.67 seconds and a boundary/wide-magnitude
hardware regression passes in 13.06 seconds. TAN is not enabled: the reference
native experiments remained outside strict tolerance and rockchip-2607 used a
NumPy host implementation. The complete uncached census confirms that SIN is
the sole method transition: the authoritative result is now 153 native, 40
frontend-only, 219 failed, and 13 upstream-skipped. The complete serialized
device suite passes 82 tests plus 56 subtests in 693.48 seconds with no timeout,
reset failure, invalid submission, or process abort.

The sole pre-device failure from that census is now resolved generically.
Clang numeric vector conversion covers shaped anonymous values as well as
explicit register vectors, avoiding an illegal `__fp164` to `float4` C cast.
Weakfloat SUM keeps its existing FP32 accumulator but commits the public result
to `strong_dtype(weakfloat)`, so `DEFAULT_FLOAT=HALF` returns FP16 consistently
with the other weak reductions.

This does not add host fallback or a Rockchip test-name special case. The
constant-only first subcase remains frontend work, while all four realized
programs in the unchanged method execute only RK3588 DPU/CMAC tasks (18, 1, 1,
and 6 tasks). Focused schema-v2 telemetry records `PASS_NATIVE` with
`ROCKCHIP_FALLBACK=0`. The full host selection passes 114 tests plus three
subtests, and the complete serialized device contract passes 82 tests plus 56
subtests in 694.38 seconds. The complete uncached census confirms this as the
only method transition, producing the authoritative 154/40/218/13 result.

## Current upstream blocker

The base master contains 24,968 counted lines. This research branch currently
contains 28,985, so `MAX_LINE_COUNT=25000 python sz.py` fails by 3,985 lines.
The exact 4,017-line delta is 4,008 counted Rockchip backend lines, the
five-line generic native-program hook, and four generic correctness lines.
Generated
register definitions, LUT payloads, and reproducible command data belong under
`runtime/autogen`; handwritten legality, layout, scheduling, and emission logic
remain counted. The generic hook can be reviewed separately, while the backend
needs real upstream line budget or independently useful in-tree reductions
before submission.

## Bounded exact static reformat maps

Late movement lowering can contain more than one RANGE with the same logical
axis number after a dimension is split. The old affine dictionary keyed only
by `RANGE.arg[0]`, so it collapsed distinct subaxes such as `(1,0)` and `(1,1)`
and reported false holes or collisions. Non-affine but compile-time maps using
floor division or modulo had the same limitation.

The reformat lowerer now retains full RANGE UOp identity for the exact path and
enumerates at most 65,536 coordinate visits. It accepts redundant writes only
when they select the same source element, proves a dense destination, and then
hands the immutable map to the existing coalesced-DPU/selector-CMAC cost
choice. The sparse whole-surface candidate remains bounded to 512 source lanes;
the later windowed candidate may address a larger total surface while keeping
each local source window at K512. All candidates retain the 4,096-output,
400-task, and 2 MiB constant ceilings. Dynamic gather/scatter indexes remain
non-evaluable and reject.

The complete uncached census records six native transitions:
`test_simple_repeat`, `test_repeat_interleave`, `test_roll`,
`test_pad_reshape`, `test_diag`, and `test_pad_circular_mode`. `test_repeat`,
`test_pad`, and `test_pad_slice` still fail later independent subcases and are
not claimed. Permanent compiler regressions cover split-axis repeat
multiplicity and exact modulo-roll mapping. A serial RK3588 regression executes
both maps exactly after mixed-engine work. All 115 Rockchip host tests plus
three subtests, mypy, and Ruff pass; the complete device suite passes 83 tests
plus 56 subtests in 695.02 seconds without a timeout, invalid submission, reset
failure, or process abort.

## Static CAST and large-source windowed reformats

Nearest interpolation computes compile-time source coordinates in floating
point and casts them to integer. The static coordinate evaluator previously
treated CAST as an identity, so valid fractional intermediate values reached
layout validation and rejected. It now applies the target dtype's constant
conversion, including the required truncation toward zero, before proving the
immutable output-to-input map.

The selector planner also no longer confuses total source extent with CMAC K.
Sparse whole-surface CMAC remains limited to 512 source lanes. A larger source
surface may use the already proven windowed selector only when every selected
local window fits the unchanged K512 contract; output, task, and constant
ceilings remain 4,096, 400, and 2 MiB. Thus the 2D interpolation schedule can
legally materialize its 780-to-858 first-axis step without enabling K>512.

Both unchanged official FP16 `test_interpolate_nearest` and
`test_interpolate_nearest_exact` pass with `ROCKCHIP_FALLBACK=0` in 30.92
seconds. Each realizes eleven efficient NPU kernels; the largest contains 17
tasks and 135,376 constant bytes. Permanent compiler tests cover exact nearest
coordinate truncation and the larger source surface. A serial hardware test
checks nearest and nearest-exact mappings exactly. All 117 Rockchip host tests
plus three subtests, mypy, and Ruff pass, and the complete device contract
passes 84 tests plus 58 subtests in 705.30 seconds without a timeout, invalid
submission, reset failure, or process abort.

The complete uncached census confirms these as the only two method
transitions, with no regression, and establishes the authoritative
162/40/210/13 result. All 210 remaining failures are `NATIVE_REJECT`;
unsupported-layout falls from 28 to 26 method-first rejects.

## Affine-mean numerical contract

Ceil-mode average pooling with `count_include_pad=False` reaches a ratio of two
static sibling ADD reductions: a masked input sum divided by a masked count.
Two NPU-only implementations were characterized. Materializing numerator and
count surfaces before DPU division, including an explicit FP16 reciprocal
boundary, differed from the official result by one FP16 ULP. Folding the
per-output reciprocal into row-scaled CMAC selector weights reduced some errors
but retained the same one-ULP violation. Neither implementation is accepted.

The compiler now recognizes this generic reduction-ratio family and returns
`NUMERICAL_CONTRACT` before submission, rather than letting the later nested
sum lowerer report an imprecise unsupported-reduction reason or returning a
wrong native result. The complete experimental source and test are preserved
under `~/rk2608_backups/wip-affine-mean-rounding-12875809d-20260803/`; its
source SHA-256 is
`7394136e361afcf1880ba9fa08a1001cbeb8ad2050b7f92fd8f9f1774fee150b`.
This milestone intentionally changes no pass count. A future native path must
reproduce the source reduction's accumulation order/precision rather than
relaxing the official tolerance.

## Periodic large-output reformat

Large static repeat maps no longer require one selector row for every logical
output. When the exact output-to-input map has an aligned repeated period, the
planner materializes one period through the existing bounded selector-CMAC
path and duplicates that proven native surface with aligned DPU copies. The
general 400-task and 2 MiB constant ceilings remain unchanged; nonperiodic maps
still use the ordinary costed selector path or reject.

Complete unchanged `test_repeat` passes with `ROCKCHIP_FALLBACK=0`. Its four
realized FP16 kernels contain 23, 29, 30, and 78 tasks. The 6,912-output case
drops from 164 tasks and 98,752 constant bytes to 29 tasks and 49,040 bytes;
the formerly rejected 20,736-output case uses 78 tasks and 19,120 bytes. The
last plan is deliberately classified `CORRECTNESS_FALLBACK`, while the other
three remain `EFFICIENT`. Focused schema-v2 telemetry is stored under
`~/rk2608_backups/focused-periodic-repeat-f41b6ceaf-20260803/`.

Permanent compiler tests enforce both unchanged resource ceilings and a
100-task ceiling for the largest periodic case. A serial mixed-engine RK3588
regression checks its complete 20,736-element result. All 121 Rockchip host
tests plus three subtests, mypy over 225 modules, and repository Ruff pass; the
complete device contract passes 84 tests plus 58 subtests in 721.14 seconds
without a timeout, invalid submission, reset failure, or process abort. The
complete uncached census confirms `test_repeat` as the only method transition,
with no regression, and establishes the authoritative 163/40/209/13 result.

The subsequent geometric-copy milestone keeps that census as the authoritative
method count but reduces repeated-surface duplication from one task per period
to a doubling chain. The 20,736-output repeat consequently falls from 78 tasks
to 14, and its permanent compiler ceiling is 16 tasks. The same native planner
now recognizes aligned constant runs: it materializes one eight-lane head per
run through selector CMAC and doubles the populated portion with aligned DPU
copies. This makes the unchanged 32-channel depthwise-convolution method's
32,768-element channel broadcast pass exactly without host expansion. The
result is still deliberately classified as a 305-task
`CORRECTNESS_FALLBACK`; no 400-task or 2 MiB ceiling was raised. All 122 host
tests plus three subtests, mypy over 225 modules, and Ruff pass. The complete
serial device contract passes 85 tests plus 58 subtests in 735.32 seconds with
no timeout, invalid submission, reset failure, or process abort. A new full
census is required before promoting the focused inferred count of 164 native
methods to an authoritative result.

### Frozen 2607/2608 branch re-audit

The frozen branches remain useful as hardware oracles, not as strict coverage
implementations. Rechecking their late dtype and movement milestones confirms
that 2608's public bool output, bool input, int32 comparison/WHERE/movement,
typed cast, prefix-repeat, and signed-copysign ABIs read or rewrite mapped GEMs
with NumPy in `RockchipProgram`. Likewise, 2607's cumulative-sum milestone
selects `_try_cumsum_host_subtasks`, and its final average-pool and broad dtype
families dispatch through named `_run_host_*` handlers. None of those paths is
eligible for `ROCKCHIP_FALLBACK=0` coverage.

The re-audit did recover one missing native legality rule. Both old compiler
generations record that PPU dimensions of nine and global-pool splits `(3,6)`,
`(6,3)`, and `(12,12)` are hardware-bad on RK3588. The clean shape selector now
filters those geometries before emitting a PPU task. Compiler regressions cover
all three classes, while the proven 4x4 HWC8 hardware path still passes exactly.
The old PPU-average reference is retained only as a future probe: its
`globalavg` validation uses loose `atol=0.25` plus host post-scaling, so it is
not evidence that the current strict affine-mean failures can pass unchanged.

### `allbilly/rk3588` `conv_grok` and `allbilly/npu` re-audit

The `conv_grok` snapshot at `40fae7b1ade1` provides reusable raw CNA facts:
input channels one through four use the NHWC path with an eight-lane hardware
alignment, both convolution stride fields are three bits (`1..7`), and
`CNA_CONV_CON3` encodes them as `(stride_y << 3) | stride_x`. Its width-stride,
DMA-stride, feature-grain, CBUF-bank, and weight-layout formulas remain useful
for future direct pointwise/channel tiling.

The clean compiler now uses those facts for typed FP16 direct spatial
convolution with `IC in {1,2,3,4,16}` and independent X/Y strides in `1..7`.
Logical NCHW inputs and OIHW weights are still packed and outputs unpacked by
NPU selector/DPU work; no invocation-time host transformation was imported.
For IC=3, each physical batch is padded to a 16-byte boundary before assigning
the next batch address. Without this padding the 1,848-byte regression surface
lost low address bits and corrupted alternating batches.

The reference's `pack_input`, `pack_weights`, `unpack_output`, grouped, and
depthwise orchestration are NumPy/CPU code and are not strict backend
implementations. Its sweep also validates with `atol=0.12, rtol=0.02`, so only
register/layout facts independently revalidated at unchanged official
tolerances are promoted. The `allbilly/npu` snapshots similarly remain register
catalogues: `cast.cpp` asks RKNN for float output and truncates on the host, and
`pool.cpp` performs host layout conversion/reference work. Neither is evidence
for native cast or pooling coverage.

A second audit found two narrower reusable mechanisms. `conv_grok/gemm_npu.py`
builds one PC-linked command chain for a homogeneous set of CMAC tiles and its
`TileSession` reuses command/task/input/weight/output GEM objects. These are
useful runtime-overhead references, but they do not solve layout legalization;
their input and weight surfaces were still packed with NumPy. They also reset
once around a homogeneous job, whereas the clean mixed-engine compiler has a
proven state-sensitive multi-tile failure and therefore keeps its per-stage
reset contract. `plan_depthwise_rows` is only an unexecuted target planner—the
reference explicitly retains per-channel CPU orchestration—so it is not copied
as hardware proof.

The raw `allbilly/npu` command catalogue does contain a hardware FP32 writeback
contract: FP16 input/proc with DPU `OUT_PRECISION=5` and no FP32-to-FP16 output
conversion. The clean CMAC emitter already used that mode for fused lerp
materialization. It is now independently verified for a 135-element FP16
scalar sum and exposed only as FP16 reduction to FP32 output. The official
`test_sum_dtype_arg` still switches its input to FP32 for the reference
comparison, so it honestly remains an unsupported-input-dtype failure and this
capability changes no census total. The raw comparison examples write FP16
zero/one masks and convert them through RKNN/host code; they do not establish a
public native bool surface.

Permanent compiler tests require four direct CNA tasks for batched RGB stride
2 and `(2,1)` plans, while a device regression checks batch-2 IC3 stride `(2,1)`
against the unchanged strict result and requires two CONV tasks. The focused
plans contain 118 and 152 total tasks respectively because selector packing is
still a correctness fallback. All 123 host tests plus eight subtests, mypy over
225 modules, and Ruff pass. The complete serial device contract passes 86
tests plus 58 subtests in 731.90 seconds with no timeout, invalid submission,
reset failure, or process abort.

### Direct pointwise CNA boundary

The same `conv_grok` register catalogue distinguishes 1x1 pointwise work from
spatial convolution. The clean emitter now uses pointwise CBUF allocation,
`CNA_CVT_CON0=1`, and `CORE_MISC_CFG=0x200`, while keeping FP16 WDMA output.
Early simplification flattens pointwise H/W into one axis, so legalization
recognizes the exact `(output-channel, spatial, input-channel)` affine
contraction and chooses a balanced hardware H/W factorization no larger than
32x32. This is geometry-derived and does not match a test name.

The unchanged IC4 9x9 official 1x1 method passes through one CNA task in a
19-task plan; the former selector-contraction implementation needed a much
larger correctness plan. A permanent IC16 8x8 regression passes at official
tolerance through one CNA task in a 21-task plan. The large IC16 32x32 case
still rejects because native NCHW-to-CNA packing exceeds the unchanged cost
contract.

A channel-split experiment avoided that pack by running sixteen IC1 CNA tasks,
then accumulating their FP16 outputs on the DPU. Its 336-task plan stayed
within all resource ceilings but failed the unchanged official comparison:
4,458/16,384 outputs mismatched and maximum absolute error was 0.02344 because
each partial was rounded before accumulation. The implementation is excluded
and preserved as `wip-pointwise-channel-split-fp16-rounding.patch` (SHA-256
`2ba6bcccf36fd83039a5c8dbfe448fb4c076713073b19b592cf8d13a46c74de7`). A
future solution needs direct input-layout conversion or non-FP16 partial
accumulation; it must not relax tolerance or host-pack the input.

All 125 host tests plus ten subtests pass, mypy checks all 225 modules, and
Ruff is clean. The complete serialized device contract passes 88 tests plus 58
subtests in 719.03 seconds without a timeout, invalid submission, reset failure,
or process abort.

### Per-channel depthwise CNA

The full `allbilly/rk3588` history adds one portable convolution strategy beyond
the raw register formulas: its hardware runner executes depthwise convolution
as independent group-1 channel tasks. The clean compiler now recognizes the
generic dense NCHW depthwise affine form, pads each physical task to the proven
minimum two output channels, and emits one CNA task per batch/channel. It does
not import the reference's NumPy packing, Python assembly, or loose tolerance.

The initial one-output-channel physical task timed out. With the required
two-channel tile, exactly half of the batch-2/channel-3 outputs were corrupt:
the failing channel planes began at byte offsets congruent to eight modulo 16.
The final plan therefore aligns every logical input plane to a 16-byte base
through an NPU selector pack, pads the second physical weight kernel with zero,
and compacts channel zero from each CNA output through another NPU selector.
These are physical legality rules, not shape- or test-name predicates.

The official `test_fancy_conv2d` passes at unchanged tolerance. Its complete
plan has 108 tasks, 292,576 constant bytes, 53,088 scratch bytes, and six CNA
tasks. This remains a `CORRECTNESS_FALLBACK` until direct layout conversion
replaces selector packing, but it stays below every existing plan ceiling. The
existing official depthwise method and all previous device regressions remain
green. All 126 host tests plus ten subtests pass, mypy and Ruff are clean, and
the complete serialized device contract passes 89 tests plus 58 subtests in
730.57 seconds without a timeout, invalid submission, reset failure, or process
abort. The focused result implies 166/40/206/13 pending a complete census.

The subsequent complete census confirms that exact transition; 166/40/206/13
is now the authoritative inventory.

## NVDLA, Mesa Rocket, and CBUF-pressure planning

The local NVDLA SW snapshot `79538ba1b52b` and Mesa `rocket` snapshot
`76c88ba66485` confirm that convolution tiling is fundamentally a shared-CBUF
allocation problem. NVDLA computes feature entries per input slice, reserves
weight banks, assigns the remaining banks to feature slices, and only then
derives partial-height tiles and their overlap. If the complete weight surface
does not fit, it combines partial-height with split-K. Mesa Rocket explicitly
states that its splitter is mostly taken from NVDLA and implements the same
full-input/full-weight, partial-input/full-weight, and partial-input/partial-
weight decisions.

`conv_grok` specializes that model to the empirically proven RK3588 FP16
formats. Its `k_step` responds to weight-bank pressure, its `y_step` responds
to the feature banks remaining after the selected weight tile, and simultaneous
pressure creates the Cartesian `BY_YK` schedule. The ten offline planner tests
pass, and classifying its 217-shape catalogue produces 49 `NONE`, 37 `BY_Y`,
24 `BY_K`, 51 `BY_YK`, 39 depthwise-serial, and 17 grouped-serial cases. The
clean compiler should therefore schedule direct convolution as:

```text
logical convolution
  -> choose physical input/weight/output formats
  -> compute FP16 entries per slice and weight banks
  -> choose K tile
  -> recompute remaining feature banks
  -> choose Y tile and overlap
  -> emit NONE/BY_Y/BY_K/BY_YK task windows with buffer offsets
```

Only the formulas and register contracts are reusable. `conv_grok` slices and
packs every tile with NumPy, Mesa Rocket converts tensors and weights on the
CPU, and NVDLA targets a different accelerator. None is evidence for host
packing in the strict backend. The local classifier also mixes conservative
empirical headroom with the raw bank formula, so every promoted RK3588 boundary
still requires an exact hardware regression.

NVDLA contributes two further compiler patterns. Surface-format legalization
intersects producer and consumer capabilities, while line/surface stride and
buffer offset are negotiated across all clients; concat and split may then use
one larger physical surface with different offsets. This is the model needed
for executable `RKLayout` legality and internal zero-copy composition. Its PDP
planner also models first/middle/last pooling tiles and overlap, complementing
the exact local RK3588 sliding-MAX PPU register probe. NVDLA's BDMA path is not
promoted: the semantic 3D implementation in this snapshot is disabled behind
`#if 0`. Mesa Rocket is likewise a register/planner reference, not a semantic
oracle: it is a UINT8 prototype with CPU layout transforms and a substantial
failure/skip inventory.

## Per-group CNA tiles

The generic grouped affine form is now recognized as independent `(batch,
group)` convolution tiles. Logical NCHW inputs and OIHW weights are packed by
selector CMAC, each small tile is checked against the proven single-weight-bank
CBUF contract, CNA executes the convolution, and selector/DPU work compacts the
physical output. No Python slicing, NumPy packing, test-name predicate, or
tolerance change participates.

The first hardware regression made only tile zero correct. Packing had appended
alignment lanes once at the end of the complete surface, while each CNA base
advanced by a per-tile aligned stride; all later tiles therefore started four
FP16 values late. Inserting padding after every batch/group tile fixes the
physical contract. The official batch-4/group-5/IC-per-group-3/OC-per-group-7
`test_grouped_conv2d` now passes unchanged through 112 tasks: 20 CONV, 90 CMAC,
and two DPU. It uses 1,819,392 constant bytes and 16,640 scratch bytes and is
correctly labeled `CORRECTNESS_FALLBACK`. This replaces the former 446-task
selector-contraction lower bound without raising any ceiling. At that focused
milestone it implied 167/40/205/13; the subsequent census also includes direct
sliding MAX and confirms 168/40/204/13. CBUF splitting is not needed
for these small compute tiles; the remaining cost is almost entirely physical
packing, which must be replaced by direct native reformatting rather than by a
wider task limit.

The full host suite passes 127 tests plus ten subtests, mypy checks all 225
tinygrad modules, and Ruff is clean. The serialized RK3588 suite passes 90
tests plus 58 subtests in 731.94 seconds without timeout, invalid submission,
reset failure, or process abort. The existing simple and medium grouped
TestOps methods also remain green.

## Direct sliding-MAX PPU

The local `experimental/pool.py` result was first treated only as proof for
2x2/stride-1 MAX. A direct register probe now varies the PPU input/output cube,
kernel, and stride fields while keeping dense FP16 HWC8 surfaces. It is bit
exact for 2x2, 3x3, 5x5, rectangular 3x2, and 3x3/stride-2 cases. NVDLA's PDP
split model remains useful for future tall surfaces, but these are direct
RK3588 measurements and require no inference from NVDLA hardware.

The compiler adds a typed `RKPool` plan rather than encoding sliding windows as
hundreds of global reductions. The initial legality rule recognizes only dense
valid FP16 MAX pooling with affine planar input/output, kernel dimensions 2--8,
stride components 1--8, and bounded static surfaces. It selector-packs planar
channels into HWC8 once, emits one PPU task per eight-channel group, and
selector-unpacks the result. Padding, dilation, ceil mode, and oversized
surfaces continue to use their existing proven paths or reject.

The unchanged 3x2x17x14, 5x5/stride-1 official
`test_max_pool2d_unit_stride` changes from a 1,380-stage reject to an exact
native pass. Its plan has 55 tasks: 40 CMAC pack/unpack tasks, 14 DPU tasks, and
one PPU task; it uses 2,311 command words, 1,273,248 constant bytes, and 7,616
scratch bytes. It remains a cost-visible `CORRECTNESS_FALLBACK`, but neither
the 400-task nor 2 MiB constant ceiling changed. Together with the preceding
grouped-CNA gain is now confirmed by the complete 168/40/204/13 census.

The full host suite passes 128 tests plus ten subtests, mypy checks all 225
modules, and Ruff is clean. The serialized device contract passes 91 tests plus
58 subtests in 730.76 seconds without timeout, invalid submission, reset
failure, or process abort. The previously passing smaller- and bigger-stride
official methods also pass all eight subcases under the new lowerer order.

## NHWC/HWIO convolution and CBUF pressure

The TensorFlow-style affine form used by `test_simple_conv2d_nhwc` is now
recognized from its coefficients rather than its test name: logical NHWC input
and HWIO weight surfaces are selector-packed into the proven C16/CNA formats,
CNA executes each batch/channel tile, and selector work restores logical NCHW
output. Missing or nonpositive affine spatial coefficients make this lowerer
`NOT_APPLICABLE`; a full device-suite regression caught and permanently tests
that boundary so grouped convolution remains owned by its existing lowerer.

`conv_grok` supplies the right general scheduling model: weight-bank pressure
chooses the output-channel (K) step, then feature entries consume the banks
remaining after that K tile and choose the Y step. Pressure in both dimensions
requires the Cartesian `BY_YK` schedule, including input overlap and physical
buffer offsets. The newly supported 2x9x9x10 / 3x3x10x20 case is only a `BY_K`
case: its complete feature surface fits, while the proven physical CNA output
contract divides 20 channels into 16+4, and every tile stays inside one weight
CBUF bank. It must not be described as evidence that feature-bank/Y splitting
is already implemented. Larger surfaces remain future typed Y/YK plan work.

The unchanged official method passes at `atol=1e-5` with one native kernel:

```text
tasks / resets       160 / 160
engine tasks         151 CMAC + 4 CONV + 5 DPU
command words        7,298
constant bytes       1,004,272
scratch bytes        21,696
native quality       CORRECTNESS_FALLBACK
```

No CPU tensor path, runtime packing, tolerance change, or resource-ceiling
increase was added. Selector packing still dominates the plan and its
17.10-second kernel time, so the next optimization is direct native layout
conversion plus typed CBUF Y/K windows—not wider selector or task limits. The
complete uncached census at `de0ac1406` confirms this is the only transition:
169 native, 40 frontend, 203 failed, and 13 upstream-skipped methods. No native
method regresses.

Regression gates pass: 130 host tests plus ten subtests, mypy over all 225
tinygrad modules, Ruff, and 92 serialized RK3588 device tests plus 58 subtests
in 749.63 seconds. The hardware run had no timeout, invalid submission, reset
failure, or process abort. The complete census likewise finishes without a
device-state error in 2,539.47 seconds. Its JSON SHA-256 is
`7f8d1ee6f46ddf35136903c12f1d4b768b6f1fbd0a9c89555825f8f38828aa45` and
its JUnit SHA-256 is
`2c08cc8240e440c933f402819669323ea8c4e3be1368a2c9eb93bd7aa8fb92af`.

## Logical K=65 in one K=96 CMAC tile

The tiled contraction compiler previously rejected every logical K above 64,
although the physical CMAC ABI already expresses K as 32-lane blocks. A direct
RK3588 probe now proves logical K=65 with zero-padded lanes in one physical
K=96 tile in both orientations: `(65,) @ (65,45)` and `(45,65) @ (65,)` match
FP32 accumulation rounded to FP16 with zero sampled difference. The plans use
138 and 152 NPU tasks respectively; most remain selector packing, so both are
cost-visible `CORRECTNESS_FALLBACK` paths rather than efficient contractions.

The unchanged `test_dot_1d` now executes its scalar case and those two K=65
cases natively, then rejects the first batched `(8,45,65) @ (65,)` layout before
submission. Its method-level status therefore remains `FAIL`; the current
169/40/203/13 census is unchanged. The remaining blocker is physical row/batch
packing, not K arithmetic or a reason to raise the 400-task limit.

Permanent compiler tests require the main CMAC task to use physical K=96, and
permanent device tests cover both orientations. Regression gates pass with 131
host tests plus 12 subtests, 93 serialized RK3588 tests plus 60 subtests in
782.89 seconds, full mypy, and Ruff. No CPU execution, tolerance change,
timeout, reset error, invalid submission, or process abort occurred.

## Rejected direct PPU average pooling

NVDLA and the local RKNN command catalogue agree that PPU average mode is the
sliding-pool task with method zero plus two FP17 reciprocal fields. The durable
`probe_ppu_average.py` sweep executes those registers through the current DRM
runtime using the documented 1/K encodings for K=1 through 8.

The engine is functional but does not satisfy TestOps' `rtol=1e-5`: 2x2, 3x3,
3x2, and 5x5 sweeps differ from PyTorch by up to 0.001953125, 0.0029296875,
0.001953125, and 0.0009765625 respectively. PyTorch is bit-identical to the
FP32-accumulate-then-FP16 reference for the same inputs. Direct PPU AVG is
therefore retained as a hardware characterization probe and is not selected by
the compiler. This does not change the authoritative 169/40/203/13 census.

## Multi-input pointwise affine reductions

Variance exposes a reusable reduction shape that the earlier affine selector
could not express: materialize two differently indexed FP16 surfaces, evaluate
`(x-broadcast(mean))^2` pointwise on DPU, then reduce static output rows on
CMAC. `lower_pointwise_affine_reduce_result` now builds exactly that typed NPU
program for up to four inputs and 65,536 visits. Direct `INDEX*INDEX`
contractions are explicitly not applicable, so GEMM and convolution retain
their FP32 CMAC/CNA accumulation contract.

The correction-equals-extent epilogue is also hardware-specific. Tinygrad
canonicalizes the nonnegative squared sum divided by zero to `sum*+inf`;
RK3588 multiplication by infinity after CMAC returns an invalid result, while
native `sum/+0` has the identical IEEE contract: positive sums become `+inf`
and an exactly zero sum becomes NaN. The compiler applies that equivalence only
to a recognized sum of a square.

A second RK3588 boundary appeared in the preceding singleton cases. Hazardous
non-finite or FDIV operations over a partial final atom can leave state that
changes a following CMAC program. The emitter splits only those stages into an
aligned prefix and a true sub-eight-lane tail. A blanket split was rejected
because it added up to twelve tasks to established plans; allocator-wide tail
zeroing was likewise rejected after perturbing seven CMAC/CNA regressions.

The unchanged `test_var_one_in_axis` and `test_std_one_in_axis` now pass in 9.18
and 17.20 seconds in focused execution. The complete device gate at that
milestone passed 94 tests plus 60 subtests in 792.84 seconds; host gates passed
132 tests plus 12 subtests, mypy over 225 modules, and Ruff. No CPU semantic
lane, tolerance change, or resource-ceiling increase was added. The complete
`2e40def50` census verifies both transitions plus the generic
`test_binary_crossentropy_logits_pos_weights` gain, producing the current
authoritative 172/40/200/13 tally.

## Constant-filled static reformat

Static affine reformat now recognizes an indexed-versus-constant WHERE rather
than limiting the inactive branch to zero. Finite values round once to the
FP16 destination contract, append one atom-aligned constant lane to an NPU
scratch copy, and select it through the existing CMAC reformatter. Non-finite fills deliberately never enter CMAC because zero
weights multiplied by infinity could poison ordinary rows with NaN; a second
finite selector creates a padding mask and DPU constructs signed infinity as
`+/-mask/(1-mask)` before adding it to the selected source.
NaN and an explicitly preserved negative-zero fill remain typed numerical
rejects until their hardware sign/payload behavior is characterized.

The unchanged official `test_pad` passes every zero, finite, positive-infinity,
negative-infinity, crop, and exception subcase in 11.10 seconds with only
native RK lanes. `test_pad_slice` likewise passes all 34 zero and value-3.456
crop/slice subcases in 15.30 seconds. A permanent mixed-workload hardware test
covers `5`, rounded `3.456`, `+inf`, and `-inf`. Host gates pass 133 tests plus
16 subtests, mypy over 225 modules, and Ruff; the serialized device gate passes
95 tests plus 64 subtests in 805.79 seconds without a timeout, reset failure,
invalid submission, or abort. The focused expected tally is 174 native / 40
frontend / 198 fail / 13 skip;
172/40/200/13 remains authoritative until the next complete census.

### Rejected FP16-only multi-source stack promotion

A generic static-WHERE planner successfully resolved three FP16 input surfaces,
emitted 20--32 task selector/DPU plans for every stack dimension, and passed a
permanent-style RK3588 hardware sweep. It does not make the unchanged official
`test_stack` method pass: that method is intentionally executed under the
FP32 reference context so its final scalar `3.14` assertion retains precision,
and general FP32 NPU input remains a proven hardware timeout. The experiment is
therefore excluded from the clean compiler rather than being counted as a
coverage gain. Its complete code and tests are preserved as
`wip-multi-source-reformat-fp32-contract.patch` (SHA-256
`2a6c591f305b8e12eda17d5c6d199340286e53386caf87b7d0653f430ca704a7`).

## Static selector-expression reformat

Reflect and replicate padding simplify into disjoint ADD/WHERE trees containing
several guarded indexes of the same FP16 surface. A generic movement lowerer
now evaluates only the static predicates and index arithmetic for every output
coordinate, requires exactly one selected index and one source parameter, and
hands the resulting mapping to the existing typed CMAC reformatter. It cannot
claim arithmetic, multi-input, dynamic-index, or non-FP16 graphs.

Representative 3D, 4D, and 5D compiler cases pass, as does a permanent RK3588
reflect/replicate sweep. The unchanged official `test_pad_reflect_mode` and
`test_pad_replicate_mode` methods both pass natively in 16.78 seconds combined,
with ten RK kernels each and no fallback. Host gates pass 134 tests plus 22
subtests, mypy, and Ruff; the serialized device gate passes 96 tests plus 66
subtests in 807.11 seconds. The focused expected tally is 176 native / 40
frontend / 196 fail / 13 skip; 172/40/200/13 remains authoritative until the
next complete census.

## Semantic reformat plan boundary

Reformat lowering now preserves two explicit levels. `RKReformatPlan` contains
only the logical source and destination surfaces, exact static mapping, and fill
contract. `RKLegalizedReformat` pairs that semantic plan with the chosen DPU or
selector-CMAC kind and a UOp-free physical `RKProgram`. Engine steps and scratch
resources no longer live inside the logical transform itself.

This is deliberately a no-coverage, no-register-change milestone. Frozen
coalesced-copy, selector, finite-fill, and infinity-fill RKImage SHA-256 goldens
remain byte-identical, as do their complete task/command/constant/scratch cost
records. The combined host gate passes 135 tests plus 26 subtests, mypy, and
Ruff; the serialized RK3588 gate passes 96 tests plus 66 subtests in 806.83
seconds. The split provides the boundary required to compare direct
CNA/PPU/ERDMA packing against selector CMAC before selecting physical work.

## Executable physical layout contracts

`RKLayout` now distinguishes linear, DPU feature, CMAC activation/weight, PPU
HWC8, CNA activation/weight, and CONV-output surfaces. It computes exact
physical byte extent, detects dense/view-compatible storage, tracks whether
padding has a known initializer, and conservatively answers legality for each
RK engine. Emitters validate those contracts before producing CMAC, PPU, or
CONV commands, so physical-format knowledge no longer exists only as repeated
shape checks inside individual lowering paths.

Every established direct convolution and PPU reduction/pooling surface is
annotated without changing its serialized commands. The focused host gate
passes 128 tests plus 26 subtests, mypy, and Ruff. Twelve serialized RK3588
CONV/PPU tests plus three subtests pass in 79.72 seconds. This is a no-coverage
milestone and leaves 172/40/200/13 authoritative; it prepares direct packing
and CBUF-pressure candidates to state their input/output contracts explicitly.

## Typed CBUF-pressure convolution tiling

The formula-only `conv_grok` scheduler is now represented by typed
`RKConvTiling` and `RKConvTile` plans in the clean compiler. FP16 feature rows
determine resident data banks and Y windows; packed weight bytes determine K
windows; simultaneous pressure emits the Cartesian `BY_YK` product. Every tile
records input/output Y geometry, K range, and its data/weight bank allocation,
and construction rejects any tile exceeding the twelve-bank RK3588 CBUF.

Five representative shapes reproduce `conv_grok` exactly: `NONE` 7x4,
RGB `BY_Y` 32x32, spatial `BY_K` 1x32, pointwise `BY_YK` 7x32, and the
TestOps 64x64 5x2 large-input geometry as `BY_Y` with 23 output rows. All ten
reference offline planner tests pass. The current NHWC compiler now consumes
the typed planner for its proven K-only 16+4 split; its unchanged device test
passes in 18.37 seconds. Plans containing Y windows remain typed rejects until
CNA input overlap and output-offset emission are proven on hardware.

## Semantic convolution plan boundary

Convolution now has the same two-level distinction as reformatting.
`RKConvPlan` owns logical geometry, legal packed surfaces, and the typed CBUF
tiling decision. `legalize_conv_plan` converts it to one or more physical
`RKConvTask` submissions. A Y tile advances the CNA input base by its
overlapping input-row offset and the DPU output base by its logical output-row
offset while retaining the backing surface stride. This is the behavior used
by Mesa Rocket's split tasks; it is materially different from copying each
tile into a compact host buffer.

The local `conv_grok` harness does copy and repack every tile with NumPy before
submission, so that portion is deliberately not imported. K tiles are also
still rejected by generic legalization because a channel slice is not
contiguous in the canonical packed-weight surface; an explicit native weight
reformat must precede them. Existing no-split direct convolution now passes
through the semantic plan boundary and its focused RK3588 tests remain green.
The complete serialized device gate passes 96 tests plus 66 subtests in
805.78 seconds with no timeout, reset error, invalid submission, or process
abort.

## Direct CNA zero padding

Convolution plans and tasks now retain four explicit zero-padding extents.
The emitter programs CNA `PAD_CON0` with the proven top/left offsets and
`PAD_CON1` with the FP16 zero value; bottom/right are represented by the exact
output geometry, as in Mesa Rocket and the independent `allbilly/rk3588`
register traces. Padding is limited to the four-bit hardware fields and padded
Y tiling remains rejected until its edge-tile overlap rules are proven.

The NCHW matcher accepts a masked feature load only after recovering its one
real affine address branch and exhaustively proving that the predicate selects
exactly `0 <= y < input_height` and `0 <= x < input_width` over every compiled
coordinate. It does not infer padding merely from a negative address constant.
Relocations now find their unique address-register commands instead of relying
on word numbers that changed when the two padding registers were inserted.

The unchanged official `test_padded_conv2d_bs1`, `test_padded_conv2d_p21`, and
`test_padded_conv2d_p22` methods pass on RK3588 at the stock tolerance. Their
batch-one compiler plans use 64--70 tasks; the batch-four official plans use
229--282 tasks and remain selector-heavy correctness fallbacks. This adds three
focused native methods without CPU execution, a tolerance change, or a task
ceiling increase. The expected focused tally is 179 native / 40 frontend / 193
fail / 13 skip; 172/40/200/13 remains authoritative until a complete census.
The host gate passes 132 tests plus 34 subtests, full mypy and touched-module
Ruff pass, and the complete serialized RK3588 gate passes 96 tests plus 66
subtests in 804.61 seconds without a timeout, reset error, invalid submission,
or process abort.

The same exact predicate proof now handles a four-axis output with only the
channel reduction axis present, which is the simplified form of padded 1x1
convolution. The unchanged official `test_padded_conv2d_1x1` passes natively in
23.10 seconds. Its batch-four plan has 193 tasks and 1,086,784 constant bytes;
using 64-output selector tiles keeps it below the unchanged 2 MiB ceiling,
whereas the earlier 128-output candidate did not fit. Focused expected coverage
is now 180 native / 40 frontend / 192 fail / 13 skip. The complete host gate
passes 132 tests plus 34 subtests, and all four direct padded-convolution
TestOps regressions pass together on RK3588 in 92.30 seconds.

The singleton input/weight case simplifies past contraction entirely into a
zero-masked padded surface multiplied by one runtime scalar. Generic
multi-broadcast lowering now gives each indexed input its enclosing conditional
surface, proves the mask statically, materializes false coordinates as zero,
and substitutes the complete surface into the DPU expression. The unchanged
official `test_simple_padding_conv2d` passes natively in 1.44 seconds through a
seven-task plan with 4,176 constant bytes. This is reusable masked-broadcast
composition, not a convolution-name special case. Focused expected coverage is
181 native / 40 frontend / 191 fail / 13 skip.

Static affine multi-source reduction now accepts both ADD and MAX. MAX is
legal only when every source contributes exactly one value to every output, so
selector-CMAC can never accidentally sum two values before the maximum. When a
source mapping is already the output identity, the plan uses that argument
directly rather than packing it. The unchanged official `test_stack_max`
therefore passes through one DPU MAX task in 0.73 seconds with no scratch or
constants. Focused expected coverage is 182 native / 40 frontend / 190 fail /
13 skip.

Runtime FP16 tensor power now lowers to a typed, NPU-only range-reduced
LOG2/multiply/EXP2 program. Two offline-generated tables encode the residual
`2**r` curve and integer power-of-two scale; native roundoff supplies integer
decomposition, while DPU masks repair negative-base parity, invalid fractional
negative bases, and all zero-base cases. The physical program has 376 tasks and
is explicitly a correctness fallback under the unchanged 400-task ceiling.

The unchanged official `test_pow_full` passes both `x**y` spellings in 158.71
seconds, and `test_pow_zero_tensor` passes `0**0`, `0**positive`, and
`0**negative` in 195.00 seconds. Calibration is tied to exact FP16 base values
and exponent sign after the clean single-consumption scheduler. A rejected
global residual-knot adjustment is retained in the generator: it fixed nine
observed lanes but regressed fourteen other lanes sharing interpolated knots.
No tolerance was changed.

Unmasked affine broadcast substitution now replaces every occurrence of the
source INDEX, which lets repeated exponent uses canonicalize correctly.
However, the official large broadcast-POW shapes cost 500 and 590 tasks after
packing, so new cost gates reject them with `PLAN_STAGE_LIMIT`; the ceiling was
not raised. Focused expected coverage is 184 native / 40 frontend / 188 fail /
13 skip, while 172/40/200/13 remains authoritative until a complete census.
The complete host gate passes 144 tests plus 34 subtests, full mypy and
touched-module Ruff pass, and the serialized RK3588 gate passes 96 tests plus
66 subtests in 807.23 seconds without a timeout, invalid submission, reset
failure, or process abort.

Constant fractional power now canonicalizes tinygrad's exact negative-domain
`WHERE(NaN, EXP2(LOG2(abs(x))*c))` decomposition into the same range-reduced
NPU recipe. A negative constant exponent is represented upstream as
`(1/x)**abs(c)`; the compiler restores `x**c` before legalization because DPU
FDIV saturates `1/0` and would otherwise hide the required positive infinity.
The constant recipe needs 209 tasks and omits the runtime-tensor calibration
groups. The unchanged official `test_pow_zero_const` passes all four subcases
in 102.53 seconds and `test_pow` passes all fourteen subcases in 158.06
seconds. Focused expected coverage is now 186 native / 40 frontend / 186 fail /
13 skip; the authoritative complete census remains 172/40/200/13. The expanded
host gate passes 145 tests plus 34 subtests, full mypy, and touched-module Ruff.

## Native piecewise tangent

The exact `sin(x)/sin(pi/2-x)` decomposition now canonicalizes to a typed,
UOp-free tangent recipe. Generated direct Q15 tables cover the local and middle
ranges, while a sine/local-cosine quotient and split pole distance preserve
accuracy near odd multiples of pi/2. Values beyond magnitude five first use the
newer Cody--Waite two-pi reducer, avoiding the old 2607 FP16 period-counter
failure at 1000 and 10000. NaN and infinities are repaired on the NPU.

The final plan has 148 tasks and nine scratch surfaces. The unchanged official
`test_tan` passes all dense, scalar, IEEE-special, and large-angle subcases in
66.92 seconds without host conversion or tolerance changes. Cached structural
hashes on the immutable expression DAG reduce compilation of this shared recipe
from roughly three minutes to 2.2 seconds. Focused expected coverage is 187
native / 40 frontend / 185 fail / 13 skip; 172/40/200/13 remains authoritative
until the complete uncached census is rerun. The host gate passes 141 tests plus
34 subtests, full mypy and touched-file Ruff pass, and the complete serialized
RK3588 gate passes 96 tests plus 66 subtests in 804.40 seconds. Its explicit
fallback-coherence test now uses still-unsupported cosine rather than tangent.

The first post-tangent census attempt exposed an unproven broadcast extension:
the two small tensor-POW broadcast layouts each missed one of twenty outputs by
0.015625, and the later census process aborted inside a reset after the long
partial-broadcast group. An exact replay confirmed all sixteen ADD/SUB/MUL/DIV
subcases remain native and correct, both large POW layouts already exceed the
cost ceiling, and both small POW layouts are numerical mismatches. Materialized
tensor-POW broadcasts now reject before submission with `NUMERICAL_CONTRACT`
until an output-domain calibration is proven. The unchanged official method
finishes with four typed POW rejects and sixteen passing subtests in 222.70
seconds without a device failure.

The subsequent 425-method run completed device-stably but exposed that the old
external 2607 pytest plugin mutates `dtypes.default_float`, which is now a
read-only property. From `test_exp` onward it corrupted reference setup and
reported 253 failures, so that tally is explicitly invalid. The research branch
now owns a checkout-local plugin using `Context(DEFAULT_FLOAT=...)`; focused
`test_exp` and `test_arange` both pass with the repaired setup. Future censuses
must use `PYTHONPATH=$PWD/test/rockchip` and never the frozen 2607 plugin.

The corrected uncached census at `2d4c34807` then completed all 425 methods in
3,138.18 seconds: 187 `PASS_NATIVE`, 40 `PASS_FRONTEND`, 185 typed native
rejects, and 13 `SKIP_UPSTREAM`. It has no numerical mismatch, unclassified
failure, device error, or regression from `2e40def50`. The fifteen new native
methods are `test_pad`, `test_pad_reflect_mode`, `test_pad_replicate_mode`,
`test_pad_slice`, `test_padded_conv2d_1x1`, `test_padded_conv2d_bs1`,
`test_padded_conv2d_p21`, `test_padded_conv2d_p22`, `test_pow`,
`test_pow_full`, `test_pow_zero_const`, `test_pow_zero_tensor`,
`test_simple_padding_conv2d`, `test_stack_max`, and `test_tan`. The 499 kernels
belonging to fully native methods include 457 efficient plans and 42 explicit
correctness fallbacks; the largest remains the unchanged 399-task matmul.
`sz.py` records 30,334 counted repository lines at this milestone. The main
Rockchip compiler concentration remains visible: package `__init__.py` is 2,682
lines and `expr.py` is 1,445 lines, so contraction/reformat/reduction extraction
remains cleanup work rather than being hidden under generated files.

A post-census two-level average probe rounded the final reciprocal scale only
when its relative FP16 representation error was at most `2^-11`. All four
unchanged `test_avg_pool2d` subcases then executed on RK3588, but 35--39% of
their outputs missed the required `rtol=1e-5`; maximum absolute error was
0.000977 and maximum relative error was 0.001215. The experiment is rejected,
the exact-scale compiler guard is restored, and the WIP remains archived rather
than being counted as native coverage.

Constant-base `0.7**x` no longer relaxes the uncharacterized scaled-EXP2
contract. Twenty generated Q15 bands cover the complete finite-result FP16
range, with DPU-only overflow and underflow repair. Exhaustive simulation over
all 63,488 finite FP16 encodings and a 2,048-encoding stratified RK3588 sweep
both have zero failures at `rtol=1e-3, atol=1e-6`; a permanent boundary test
passes on hardware. The plan has 245 stages and remains an explicit
correctness fallback. A clean isolated `test_pow_const` still rejects earlier
at its explicit FP32 `0**x` input, so this milestone does not claim a method or
census gain. A full-device exhaustive tensor was also rejected as evidence:
the current per-element scalar materialization produced a 7,872,512-byte
constant GEM that could not be mapped.

Two nearby low-hanging probes were rejected without changing limits. Raising
the pointwise affine-reduction output cap from 128 to 512 did not legalize
`test_strided_conv_transpose2d`: one case has dynamic unsupported input layout,
one has 528 outputs, and the stride-one case was already native at 335 tasks.
`test_simple_conv2d_1x1_m4` already uses a single unsplit CBUF tile; its blocker
is planar NCHW-to-CNA packing and output conversion, not CBUF bank pressure.

The next `test_softmin` probe proved a generic 241-task transformed row-MAX
plan bit-exact, then exposed the existing EXP clamp below -2. Three generated
negative EXP bands make 2,921/2,925 centered exponentials bit-exact on RK3588,
but the unchanged method still misses 271 outputs because intermediate FP16
normalization cannot reproduce Torch's fused higher-precision HALF softmin.
For comparison, the tinygrad CPU backend also misses 10 lanes under the same
HALF invocation. A BS-subtract-to-EW-LUT fusion register experiment submitted
but produced incorrect values, so it was rejected and archived. The active
compiler remains at the clean `187/40/185/13` census contract with no accepted
numerical failure.

The next dtype probe used the documented full-width `DPU_OUT_CVT_OFFSET`
register instead of trying to materialize large integers as FP16 constants.
Five exact RK3588 boundary values (`INT_MIN`, -1234, 0, 1234, and `INT_MAX`),
each crossing the 64-element WDMA tile boundary, match NumPy bit-for-bit. The
unchanged `test_maximum` now passes its former `INT_MAX` fill blocker and
advances to a distinct int32 identity-reformat rejection in the following
`maximum(x, INT_MIN)` subcase. Therefore this milestone expands the honest
fill-only hardware contract but does not yet claim a method-level census gain;
the authoritative tally remains `187/40/185/13`.

The immediately following int32 identity movement is also now hardware-proven.
A typed `RKCopyStage` selects int32 MRDMA, DPU process, and WDMA precision while
bypassing BS, BN, and EW semantics. A 65-value boundary vector containing both
signed extrema is bit-exact. This lets `maximum(x, INT_MIN)` simplify to a
single native DPU copy; `test_maximum` now proceeds to its public-bool case,
which was the next hardware probe.

All-int8 DPU operation proves that public bool storage itself is valid. Typed
bool fill and identity stages use 16 lanes per atom with BS/BN/EW bypassed;
bool `OR` maps to the DPU's int8 MAX engine. Sixty-five-lane fill, copy, and OR
boundaries are bit-exact. This is distinct from the archived FP16-mask-to-int8
conversion timeout. `test_maximum` now passes every bool subcase and advances
to its mixed int32/FP16 case, which remains an unsupported input conversion.

A direct mixed-precision probe did not loosen that boundary. One typed task
selected int32 MRDMA input, FP16 DPU processing/output, and FP16 scalar MAX;
the RK3588 submission timed out with errno 110. The code was removed and is
preserved as `wip-native-int32-fp16-dpu-max-timeout-22c974ac2.patch` (SHA-256
`cedab6b795ef7dc1e7b893d6082002dde0f73b9dd6e3bf1f239f1b13904c8f8f`). A
different proven converter engine is required for mixed int/float maximum.

The symmetric bool minimum decomposition is `NOT(OR(NOT a, NOT b))`. Exact
structural recognition maps that identity to int8 DPU multiplication, which is
equivalent to `AND` for public bool lanes. Its 65-lane RK3588 boundary is
bit-exact, and `test_minimum` likewise advances through every bool subcase
before reaching the same unproven mixed int32/FP16 conversion boundary.

Bool logical NOT also has a direct int8 implementation: the compiler recognizes
`CMPNE(x, True)` exactly and emits `1 - x` through the DPU subtraction ALU,
with an aligned immutable int8-one surface. The 65-lane boundary is bit-exact.
The official `test_logical_not` passes its bool-input subcase and then reaches
the separate FP16-comparison-to-public-bool conversion boundary.

Static bool movement whose map is either identity or zero now lowers to one
int8 DPU multiplication against an immutable byte mask. This is a semantic
`RKReformatPlan` with one physical task, not a selector-CMAC matrix. The full
unchanged `test_tril` and `test_triu` methods pass, including their final bool
subcases, for two focused method gains. The authoritative full census remains
`187/40/185/13`; current focused expectation is `189/40/183/13`.

The same committed bool-fill contract also makes the complete unchanged
`test_all_zero_axis` and `test_any_zero_axis` methods pass: their empty
reductions simplify to the correct bool identity constants and execute through
native int8 fill. Nonempty `all` and `any` still reject at bool reduction or
FP16-comparison conversion. Focused expectation is therefore
`191 native / 40 frontend / 181 failed / 13 skipped`; the full-census baseline
remains `187/40/185/13`.

The nested-sum lowerer now owns only graphs whose stored value is the outer
reduction. Sibling numerator/count reductions used by NLL and cross-entropy
return `NOT_APPLICABLE` instead of the misleading “nested ADD reduction
requires one FP16 scalar output” rejection. This changes diagnostics and pass
ownership only; it does not claim native loss coverage.

FP32 identity movement is deliberately narrower than FP32 arithmetic. The DPU
all-bypass path preserves aligned 16-byte atoms bit-for-bit, including NaNs and
signed zero, without claiming an FP32 processing pipeline. Static FP32
reformats therefore legalize only when every coalesced source and destination
run starts on a 16-byte atom; unaligned permutations reject before submission.

Multi-source FP16 movement now retains a typed source/index map before packing
the inputs into aligned scratch and applying the bounded CMAC selector. All
four three-input stack layouts pass custom hardware tests in nine tasks. The
official stack method remains rejected because its plugin contract is FP32 and
its final permutation needs unaligned word writes; no method gain is claimed.

FP32-to-int32 `BITCAST` is native without a conversion stage: the compiler
selects the proven int32 all-bypass transport and preserves the source words
verbatim. The unchanged strict `test_bitcast` method passes; this does not
expand the claimed FP32 arithmetic contract.

The exact integer identity `x | 0xFFFFFFFF == -1` is canonicalized to the
native int32 constant-fill plan. The unchanged `test_int_or` passes without
claiming a general integer bitwise ALU.

## Direct dense-row PPU MAX tree

`test_max_dont_collapse` is a dense 256x256 FP16 row reduction. Reinterpreting
each row as 32 HWC8 pixels needs no input packing. Two exact PPU tasks reduce
the width through `1x8/stride 8` and `1x4/stride 4`; a bounded CMAC transpose
then makes the eight surviving channels planar, and seven DPU MAX tasks fold
them to one value per row.

The complete plan has 41 tasks, 1,742 command words, 41 resets, 524,288
constant bytes, 27,648 scratch bytes, and 1,116,928 estimated MACs. A random
256x256 hardware boundary is bit-exact, and the unchanged official method
passes in strict native-only mode. The 400-task and 2 MiB ceilings are
unchanged. Focused expected coverage is now `197/40/175/13`; the authoritative
complete census remains `195/40/177/13` until rerun.

`probe_ppu_sliding_channels.py` also records the rejected alternatives.
Partial-channel sliding HWC2 completes but corrupts 172 of 192 outputs, while
the first one-row width-16 boundary timed out. Those geometries remain outside
the compiler contract; only the exact HWC8 one-row widths through eight are
enabled.

## Single-task compact cumulative sums

Structural ADD prefixes through 32 outputs now use one compact CMAC task rather
than the old fixed 16-output split. The compiler zero-pads the input into one
32-lane surface, emits one triangular selector matrix, and requests the proven
compact one-row WDMA layout. This removes the exact second-task transition that
made the former N=20 implementation state-sensitive.

Both N=10 and N=20 are bit-exact after an explicit DPU, CMAC, PPU, and CNA
stress sequence in the same process. The unchanged `test_cumsum` advances past
its 20-element case and now rejects the distinct strided 20x30 axis-zero prefix
at the unchanged 512-lane generic selector contract. This milestone therefore
claims no complete-method gain; focused expected coverage remains
`197/40/175/13`.

Grouped structural prefixes now receive their own bounded compiler contract:
at most one million inspected coordinates, 608 source lanes per CMAC window,
64 compact outputs per task, and the unchanged 400-task/2 MiB physical limits.
The selector proof requires every group to grow exactly from `[x0]` through
`[x0,...,xN]`; arbitrary masks retain the old affine budgets.

The complete unchanged `test_cumsum` passes natively in 107.09 seconds. Its
20x30 axes cost 16 and 10 tasks; the 20x30x40 inner-axis plan uses 377 tasks but
only 57,696 unique constant bytes. A permanent random axis-zero boundary proves
the 608-lane strided window. Focused expected coverage is now
`198/40/174/13`; the authoritative census remains `195/40/177/13` pending a
new complete run.

## Direct FP16 cumulative sums through 1,024 elements

The 512/1,022-element `test_simple_cumsum` failure was not a device timeout or
a missing arithmetic operation. Tinygrad's generic two-level scan splits inputs
above 512 into 256-element blocks, stores every block prefix in FP16, and then
adds an FP16 carry. Under `DEFAULT_FLOAT=HALF` that intermediate rounding makes
the unchanged 1,022-element test miss Torch's tolerance; the CPU backend showed
the same 33-lane failure.

FP16 ADD scans now remain in the direct generic formulation through 1,024
elements; MUL and MAX retain the existing split threshold. The Rockchip
legalizer proves only monotone contiguous prefix groups before applying the
wider contract. A standalone worst-tail probe with K padded to 1,024 was
bit-exact, and the complete direct plans cost:

| extent | tasks | constants | scratch |
|---|---:|---:|---:|
| 512 | 8 | 294,912 | 0 |
| 1,022 | 18 | 1,118,208 | 2,048 |

The unchanged `test_simple_cumsum` passes on both RK3588 and CPU with
`DEFAULT_FLOAT=HALF`. The permanent RK3588 512/1,022 random boundary meets the
official `rtol=1e-3, atol=1e-6` contract; together with the mixed-engine prefix
stress test, two device tests plus four subtests pass in 32.39 seconds. All 152
compiler/image tests pass with 53 subtests, mypy covers 228 source files, and
Ruff is clean. No CPU fallback, tolerance change, task-limit increase, or host
tensor transformation was added.

This is one focused complete-method gain. Expected coverage is now
`199 native / 40 frontend / 173 failed / 13 skipped`; the authoritative
uncached baseline remains `195/40/177/13` until rerun.

## Authoritative census after direct prefix lowering

The complete uncached native-only run at `746707b3e` establishes:

| outcome | methods |
|---|---:|
| PASS_NATIVE | 198 |
| PASS_FRONTEND | 40 |
| FAIL (typed native reject) | 174 |
| SKIP_UPSTREAM | 13 |

All 174 failures are classified native rejections. The 56m32s run had no
numerical mismatch, NPU timeout, invalid submission, reset failure, process
abort, or unclassified failure. Fully native methods executed 543 kernels:
495 efficient and 48 bounded correctness fallbacks. The existing ceilings
remain unchanged; the largest successful plan has 399 tasks and the largest
constant payload is 1,819,392 bytes.

`test_sum_pad_collapse`, `test_max_dont_collapse`, and grouped `test_cumsum`
are confirmed native with no regression. `test_simple_cumsum` is not counted:
the strict census plugin still deliberately changes that method's default
float contract to FP32 for an older CPU-reference workaround, so its first
kernel rejects honestly as `unsupported_output_dtype`. The focused FP16 method
does pass after the direct-prefix change. The plugin exception must be removed
and retested now that the generic CPU FP16 scan has the same direct formulation;
the authoritative count remains 198 until that verification is complete.

Artifacts are preserved under
`/home/orangepi/rk2608_backups/census-wide-prefix-746707b3e-20260804-223120`.
The telemetry JSON SHA-256 is
`5dad8510c28b0664c846f57269aea95955c90baf543cb6e3552a0c8b05eeb4d4` and
the JUnit XML SHA-256 is
`b78d65a0f23047791a7ab58e1da3bd14f0428dc1611c338e8b0dd290acb3284e`.

The obsolete `test_simple_cumsum` FP32 census exception is now removed. With
the normal strict plugin loaded from this checkout, the unchanged method passes
on CPU in 1.27 seconds and on RK3588 in 18.66 seconds under
`DEFAULT_FLOAT=HALF`; no backend dtype support changed. Focused expected
coverage is therefore `199/40/173/13`, while `198/40/174/13` remains the last
complete uncached census until the next full run.

## Static convex two-tap transforms

One-dimensional linear interpolation is now lowered as a generic static
linear transform rather than an interpolation-named runtime task. The compiler
symbolically evaluates the FP32 expression at each bounded output coordinate,
accepts only zero-bias convex rows containing one source value or two adjacent
source values, rounds the upper coefficient once to FP16, and derives the lower
coefficient with the same FP16 complement used by the reference formulation.
Runtime tensor values are never inspected by the host.

The existing windowed selector planner is generalized from zero/one matrices
to FP16 weighted matrices. It still bounds each CMAC source window to 512
lanes and retains the 400-task and 2 MiB global ceilings. Existing selectors
remain a wrapper over the weighted primitive and their compiler/device
regressions are unchanged. The official plans are small:

| input → output | tasks | constants | scratch |
|---|---:|---:|---:|
| 52 → 29 | 5 | 45,424 | 192 |
| 29 → 52 | 7 | 41,168 | 128 |

Both unchanged `test_interpolate_linear` and
`test_interpolate_linear_corners_aligned` pass with the strict plugin in 6.56
seconds. The former FP32 reference exceptions are removed because current
PyTorch supports the same FP16 inputs. A permanent random RK3588 test covers
both directions and both corner policies bit-exactly. All 153 compiler tests
plus 57 subtests pass, the existing nearest and conditional selector device
regressions pass, mypy checks 228 files, and Ruff is clean. There is no CPU
fallback, dynamic host packing, tolerance change, or limit increase.

Together with the verified FP16 simple-cumsum plugin correction, focused
expected coverage is `201 native / 40 frontend / 171 failed / 13 skipped`.
The last complete uncached census remains `198/40/174/13` until rerun.

## Authoritative census after static two-tap lowering

The complete uncached native-only run at `92846845f` establishes:

| outcome | methods |
|---|---:|
| PASS_NATIVE | 201 |
| PASS_FRONTEND | 40 |
| FAIL (typed native reject) | 171 |
| SKIP_UPSTREAM | 13 |

All 171 failures are classified native rejections and retain an exact first
reject. The 56m56s run had no numerical mismatch, NPU timeout, invalid
submission, reset failure, process abort, or unclassified failure. Fully
native methods executed 549 kernels: 500 efficient and 49 bounded correctness
fallbacks. No task, constant, or tolerance ceiling changed.

The run makes the FP16 `test_simple_cumsum` census correction and both static
two-tap interpolation methods authoritative. The first-reject Pareto is 53
unsupported output dtype, 35 plan-stage limit, 27 unsupported input dtype, 18
numerical contract, 18 unsupported layout, 12 unsupported ALU, six requires
reformat, and two unaligned row.

Artifacts are preserved under
`/home/orangepi/rk2608_backups/census-two-tap-telemetry-92846845f-20260805`.
The telemetry JSON SHA-256 is
`82ffa33a116d4d3d5ae8821647bd653ca504ebc990d0d757dd4a8726dad3e09a` and
the JUnit XML SHA-256 is
`04f984fe7e9729903a41c3b6d337406e48da6b6f7d1ae715a2f1d149e5b54026`.

## DRM discovery and rejected submission-buffer reuse

The runtime no longer assumes `/dev/dri/card1`. It enumerates DRM card nodes,
selects the node whose bound driver is `RKNPU` or mainline `rocket`, and honors
an explicit `ROCKCHIP_DEVICE` path. The hardware-test availability check uses
the same driver identity instead of accepting any unrelated `card1` node.

A per-program command/task GEM reuse experiment passed focused DPU, CMAC, PPU,
CONV, and mixed-engine tests, then produced two regressions early in the full
device suite. Per-stage allocation, blocking submission, and reset are restored
unchanged. The rejected implementation is preserved as
`wip-runtime-command-task-gem-reuse-device-regressions.patch` (SHA-256
`8cd9b0fca1381fb3398a1ce99b01acf73d6ad2befda47c55cd627c140f748d4b`).
After restoration, the affected CONV/PPU region passes four focused tests in
37.95 seconds. All 158 compiler/image tests plus 57 subtests pass, mypy checks
228 source files, and Ruff is clean. This milestone changes no coverage or
execution semantics.

## Rejected vector-matrix-as-convolution experiment

A dense FP16 `1xK @ KxN` contraction has a useful direct CNA interpretation:
the row-major matrix is already an `H=K,W=N,C=1` activation surface, while the
dynamic vector can be packed on the NPU as a `Kx1` one-channel convolution
kernel. This avoids transposing the entire dynamic matrix into CMAC weight
order. Hardware probes established a precise boundary: kernel heights 4, 8,
16, 17, 24, and 31 are bit-exact, while height 32 and the official height 128
time out in the CONV stage.

Splitting K=128 into `31+31+31+31+4` avoids the timeout and produces a compact
29-task plan, but each CONV tile materializes an FP16 partial. DPU addition of
those partials disagrees with the official single-accumulator result in 11 of
128 values (`max_abs=0.012695`, `max_rel=0.01415`). The hardware reference also
records that separate CONV tasks overwrite the destination rather than
accumulating partial sums. The experiment is therefore disabled; `test_matvec`
retains its typed stage-limit rejection. Its full implementation is preserved
as `wip-vector-matrix-cna-k31-fp16-partial-rounding.patch` (SHA-256
`4a5ee50cccb3847bc05d96d52fd9e5524dedd6e7119289f5765be0c7d3bb96de`). A future
retry requires a proven FP32 partial-output/accumulation path, not relaxed
tolerance or more FP16 tiles.

## Native atrous convolution

Spatial convolution plans and tasks now carry independent X/Y dilation. The
CBUF planner uses the effective kernel size `(kernel-1)*dilation+1`, while the
emitter writes Rockchip's five-bit `ATROUS_{X,Y}_DILATION` fields as the number
of inserted feature-map positions (`dilation-1`). Affine recognition derives
dilation from the feature access map instead of matching a Tensor operation or
test shape. Unmasked inputs may no longer acquire inferred trailing padding;
nonzero padding still requires an exact static zero-mask proof.

Both unchanged `test_dilated_conv2d` subcases—dilation 2x2 and 2x1—pass on
RK3588 with `ROCKCHIP_FALLBACK=0` in 35.61 seconds. The current schedules use
117 and 166 tasks respectively, including four real CONV tasks plus bounded
native packing, so this is an honest but selector-heavy correctness milestone.
A deterministic hardware regression checks the official tolerance and CONV
telemetry, while the compiler regression checks typed dilation and exact
`CNA_CONV_CON3` words. All 159 compiler/image tests plus 59 subtests pass;
five direct-CNA hardware regressions plus two dilation subtests pass in 52.48
seconds; mypy checks 228 files and Ruff is clean. The complete uncached census
confirms `202 native / 40 frontend / 170 failed / 13 skipped` with no
regression.

## Complete native-only census after atrous convolution

The uncached `2fb47ca2b` run makes the native atrous-convolution milestone
authoritative. Exactly one method changes relative to `92846845f`:
`test_dilated_conv2d` moves from `PLAN_STAGE_LIMIT` to `PASS_NATIVE`. All 170
remaining failures are typed native rejects with retained first rejects; there
are no device, numerical, fallback, or unclassified failures.

Fully native methods execute 549 kernels: 500 `EFFICIENT` and 49 bounded
`CORRECTNESS_FALLBACK` plans. Task-count buckets are 104 at one task, 238 at
2--8, 87 at 9--32, 73 at 33--64, nine at 65--128, 23 at 129--256, and 15 at
257--400. The maximum remains 399 tasks, 1,819,392 constant bytes, and
42,636.60 ms for one kernel; no task, constant, or tolerance ceiling changed.

Artifacts are preserved under
`/home/orangepi/rk2608_backups/census-atrous-2fb47ca2b-20260805`.
`test_ops_coverage.json` has SHA-256
`84da48e03bf39ff886581fc95f19a3963220d7ee36fbad69a0a28d6b1bb464d7` and
`junit.xml` has SHA-256
`ea12b7403eab76d42f555cbcee0d975f44cb2500713dfc72180ede93736edc50`.

## Rejected stride-one transpose-as-convolution lowering

A stride-one transposed convolution has an exact affine reformulation as an
ordinary padded convolution with I/O-transposed and spatially reversed weights.
The compiler recognized that structure without a test-name or shape match and
used only native selectors plus the proven CONV path. Simple, padded, and
dilated official methods all reached RK3588; no CPU tensor work was involved.

The path is disabled because it does not satisfy the official numerical
contract. `test_simple_conv_transpose2d` missed 164/968 outputs with maximum
absolute error 0.02344, and the first dilated subcase missed 176/1144 with
maximum absolute error 0.01563. The asymmetric-padding case additionally
exposed an invalid physical-padding schedule. These are accepted-plan errors,
not reasons to relax tolerance. The original typed stage-limit rejection is
restored exactly.

The complete implementation is preserved as
`wip-stride1-transpose-as-conv-numerical-contract.patch` with SHA-256
`0ea1f73a0896f284f9b1c728998884587351086a265cef25fd44dde64bef06d6`.
Future work must characterize the native deconvolution/Rubik schedule or make
the existing CMAC legalization compact enough while preserving accumulation
semantics. The authoritative census remains `202/40/170/13`.

## Bounded multi-source concatenation transpose

Large concatenation movement no longer requires one selector window to span
the complete source. The compiler first gathers directly from the original
NPU buffers in aligned output tiles. A following block transpose partitions
each output tile into source-index windows no wider than the proven 512-lane
CMAC input, emits one disjoint selector per window, and combines partials with
ordinary DPU ADD. Runtime tensor data is never read or packed by the host.

Candidate tile widths remain multiples of the eight-lane FP16 atom. The cost
model selects only schedules below the unchanged 400-task and 2 MiB constant
ceilings. For the official `(45,65)` three-source case, the gather is 83 tasks
and 216,512 constant bytes; the required `(3,45,65)->(45,3,65)` block
transpose is 323 tasks and 598,224 constant bytes. The unchanged
`test_multicat` passes all three dimensions in 112.20 seconds with
`ROCKCHIP_FALLBACK=0`.

All 160 compiler/image tests plus 59 subtests pass, four focused RK3588
movement tests plus nine subtests pass, mypy checks 228 source files, and Ruff
is clean. This is one focused expected gain (`203/40/169/13`); the last full
uncached census remains `202/40/170/13` pending its next run.

## Reusable submission buffers

Each compiled native program now allocates one maximum-sized command GEM and
one kernel-mapped task descriptor GEM. Every sequential stage overwrites and
submits those two buffers after the existing per-stage reset; program teardown
frees them with scratch and constant storage. A 323-task plan consequently no
longer creates and destroys 646 temporary GEMs during one invocation.

The reset and one-task blocking submission contracts are unchanged. DPU,
CMAC, PPU, CONV, mixed-engine, and repeated-program hardware regressions pass.
The unchanged `test_multicat` remains correct but measures 113.04 seconds,
essentially unchanged from 112.20 seconds, proving reset/device work dominates
this workload. This milestone claims resource-lifecycle cleanup, not a speed or
coverage gain; the expected tally remains `203/40/169/13`.

## Physical cost module

Physical-plan cost analysis now lives in `renderer/rockchip/cost.py` rather
than the lowering entry module. The estimator still consumes only legalized
`RKProgram` work and the emitted `RKImage`; task, command, reset, constant,
scratch, traffic, and MAC accounting are unchanged. This is a mechanical
module-boundary milestone with no compiler legality, plan selection, register,
runtime, or coverage change. All 160 compiler/image tests plus 59 subtests
pass with `-n12`, mypy checks 229 source files, and Ruff is clean.

## Census commit snapshot

Telemetry now snapshots the repository commit when pytest configures the
census rather than resolving `HEAD` during session teardown. Long RK3588 runs
can therefore be followed by independent milestone commits without causing
the finished JSON to identify code that the running Python process never
loaded. The focused telemetry suite passes six tests; execution and coverage
classification are unchanged.

## Rejected reusable-submission-buffer experiment

The reusable command/task GEM milestone above is retained in history as a
negative hardware result, but is no longer enabled. A complete serialized
native-only census using that runtime first timed out at method 37 and then
reported 16 device failures plus nine post-execution numerical failures. A
fresh-process DPU lerp regression also returned 33/33 incorrect elements while
the reusable buffers were enabled; restoring fresh command and task GEMs per
submission made the unchanged test pass immediately.

This is consistent with the RKNPU submission path retaining object/address or
command visibility state that the current userspace ABI does not explicitly
invalidate when the same GEM is overwritten. Per-program command/task reuse is
therefore not a proven optimization and must not be used. The polluted census
is preserved at
`/home/orangepi/rk2608_backups/census-post-multicat-053ec3b6d-20260805`;
its JSON identifies the later teardown-time `HEAD`, but the process-start code
was `053ec3b6d`. It is diagnostic evidence only, not an authoritative coverage
result. `test_multicat` did pass natively in that run and the 170 typed reject
population was unchanged.

## CBUF-bounded wide CMAC output

Direct M=1 CMAC output is now legal through 416 physical channels when the
feature and weight surfaces fit the twelve 32-KiB CBUF banks. The emitter
computes feature-bank pressure, reserves at least one data bank, and rejects a
weight surface larger than the remaining banks. The boundary follows directly
from `416*416*2 <= 11*32768`; 448 channels are rejected before emission.

The standalone hardware probe is exact for ordinary and compact FP16 output
at 160, 256, 384, and 416 channels. A unit regression accepts 416 and rejects
448. This capability does not claim a TestOps transition: a proposed large
`test_cat` gather reached native execution, but its following
`(3,45,585)->(45,3,585)` physical transpose still costs 495--885 tasks and
5.8--37.9 MiB of selector constants. The 400-task and 2 MiB ceilings remain
unchanged. The rejected planner is preserved as
`wip-large-cat-second-reformat-stage-limit-20260805.patch` with SHA-256
`249b4f6d08c499f8156de6dfa77cfcc41acc4f706fc2cdf34b729cd37a6caec4`.

## Native bool-mask WHERE conversion

The DPU can consume an int8/bool surface and write an FP16 mask when the whole
public input fits one eight-value atom. The typed `RKCastStage` represents only that
proven `(bool -> half)` contract; it is not a general cast opcode. A wider
single-task geometry converted only its first eight values, while a mismatched
16-channel RDMA geometry timed out. Starting a second conversion at byte eight
also corrupts its lanes because the int8 RDMA base is not atom-aligned, so all
three wider forms are excluded before submission.

Integer WHERE uses two proven pieces: evaluate the selection in FP16, then
convert one four-value atom to int32. Because a dense FP16 source would make
every other four-value tile start at an illegal half-atom address, longer
results are first placed into a native selector surface with four values plus
four padding lanes per group. Each final conversion consequently reads and
writes aligned 16-byte atoms. The host never reads or repacks tensor data.

The unchanged `TestOps.test_where` passes all scalar, comparison, broadcast,
and tensor-arm cases in 12.11 seconds with `ROCKCHIP_FALLBACK=0`. Focused
compiler and device tests cover the typed cast, image round trip, padded int32
materialization, a 100-element comparison, and a public bool condition. This
is one focused method gain; together with the earlier `test_multicat` result,
the expected tally is `204/40/168/13`. The authoritative complete census
remains `202/40/170/13` pending a clean rerun.

## Exact pointwise-negative affine MAX

Affine PPU MAX can now materialize the exact FP16 pointwise transform `x * -1`
before selector packing. This is the native primitive needed by MIN-style
reductions and by the first centered-softmin kernel; it is deliberately not an
arbitrary expression-before-MAX path. A `9x65` row reduction is bit-exact on
RK3588 and costs 28 tasks, 104,112 constant bytes, and 3,328 scratch bytes.

The shared 64-output CMAC selector also halves existing affine-MAX packing:
the `(3,4,5,6) -> max(axis=1)` compiler case uses 12 CMAC tasks instead of 24,
and the scalar multiaxis pool case uses one instead of two. Six focused MAX and
pool hardware regressions pass, while the complete compiler suite is 159
tests plus 59 subtests passing.

This exact first kernel does not make `test_softmin` native. Its final
multi-broadcast EXP has no proof that the dynamic centered input remains inside
the generated `[-2,2]` LUT domain. It now rejects explicitly with
`LUT_DOMAIN_UNPROVEN` instead of executing the LUT clamp and returning a
numerical mismatch. The previously archived negative-EXP bands still cannot
recover Torch's fused HALF normalization precision, so no failed LUT experiment
was restored and no coverage transition is claimed.

## Authoritative 204-pass native census

The complete uncached native-only census at `2385767d1` establishes:

| outcome | methods |
|---|---:|
| `PASS_NATIVE` | 204 |
| `PASS_FRONTEND` | 40 |
| `FAIL` | 168 |
| `SKIP_UPSTREAM` | 13 |
| total | 425 |

The two transitions from the preceding 202-pass census are exactly
`test_multicat` and `test_where`. All 168 failures are typed native rejects;
there is no CPU fallback, numerical mismatch, NPU timeout, invalid submission,
reset failure, process abort, or unclassified failure. `test_softmin` reaches
the exact negative-MAX kernel and then rejects its final dynamic EXP as
`LUT_DOMAIN_UNPROVEN`, confirming that the new capability does not weaken the
accepted numerical contract.

The 56m15s run executed 653 successful native kernels: 590 efficient and 63
bounded correctness fallbacks. At method level, 169 native methods are fully
efficient and 35 contain at least one correctness fallback. The largest
successful plan remains `test_matmul` at 399 tasks and 42.64 seconds; the
largest constant payload remains 1,819,392 bytes in `test_grouped_conv2d`.
Neither cost ceiling changed.

Durable artifacts are stored under
`/home/orangepi/rk2608_backups/census-negative-max-2385767d1-20260805/`.
The telemetry JSON SHA-256 is
`fddc4fc0fe64d5229481ef6d7695cf9e72d9199b8186be6718f1a7983f701cc1` and
the JUnit XML SHA-256 is
`1a89c93b75e5952c59960d229f5e0e86a423ebcfd11eedbe7a63ac2e19fc905f`.

## Stable atom-split cumulative product

The masked affine product lowerer no longer shares one multi-tile selector
schedule across cumulative-product outputs. It partitions the public output
into independent groups of at most sixteen FP16 values and materializes and
folds every group separately. Each full group begins on a 32-byte output atom,
so no selector state or output compaction crosses the hardware boundary that
previously corrupted the second tile after mixed-engine work.

The 20-element cumulative product is exact on RK3588 and remains inside the
existing limits at 249 tasks, 106,784 constant bytes, and 9,312 scratch bytes.
It is deliberately classified as a correctness fallback. Three explicit
`DPU -> CMAC -> PPU -> CONV -> cumprod(20) -> cumprod(10) -> cumprod(20)`
stress iterations pass, and the checked-in mixed-engine regression passes in
38.90 seconds. The unmasked product lowerer remains preferred for ordinary
products, so its existing cost does not regress.

This is not yet a full `TestOps.test_cumprod` transition. The unchanged method
continues with `(20,30)` and `(20,30,40)` scans; the first 600-output by
20-term surface exceeds the bounded masked-affine schedule and rejects with
`PLAN_STAGE_LIMIT`. No task, constant, compiler-work, or numerical tolerance
ceiling changed. The authoritative tally therefore remains
`204 PASS_NATIVE / 40 PASS_FRONTEND / 168 FAIL / 13 SKIP_UPSTREAM`.

## Rejected int8 PPU boolean reduction

The next boolean-reduction probe tested the hardware path directly instead of
adding a CPU bool pack. PPU spatial MAX accepts an int8 HWC16 input and submits
without error, but its output is one int16 lane per channel: reading the raw
surface as public byte bool produces `1,0,1,0,...` for an all-true vector. This
matches NVDLA PDP's documented use of int16 internal storage for int8 pooling.

A typed follow-up kept that int16 surface in scratch and configured DPU MRDMA
for int16 input with int8 processing/output. RK3588 timed out at the second
stage before producing the public bool result. The device recovered and the
unchanged wide FP16 DPU regression passed immediately afterward. Both compiler
experiments were removed; the exact WIP is preserved as
`wip-ppu-int8-reduce-int16-output-dpu-cast-timeout-20260805.patch` with SHA-256
`18dfaedc00bcfe54ea04aa7dffc549089be1f7bd9345d02e9f7263b728bb1964`.
The follow-up also set DPU `IN_PRECISION=int16` while retaining int8
processing/output, as required by the NVDLA converter model; it timed out at
the same stage. That one-line delta is preserved as
`wip-ppu-int16-to-int8-corrected-in-precision-timeout-20260805.patch` with
SHA-256 `5b5d467c6ce792d255832b25fbb76db331b2e210d068df7faa43b290bc2013ab`
and applies after the first WIP patch.

Consequently `all`/`any` remain typed output-dtype rejects. A future native
implementation needs a proven PPU/DPU int16-to-int8 writeback contract or a
different byte-output engine; it must not reinterpret the two-byte PPU surface
or pack it on the CPU. The authoritative census remains `204/40/168/13`.

## FP32 CMAC accumulator epilogue

RK3588 CMAC can preserve a partial accumulator in memory without first
rounding it to FP16. A direct hardware probe uses FP16 activation and weight
surfaces, adds a 32-lane FP32 BRDMA bias to the flying CMAC accumulator, and
writes the result as FP32. The result agrees with the FP32 reference within
`2.3841858e-7`. The typed emitter therefore permits the already-bounded
one-row, 32-lane FP32 CMAC output contract to carry `RKEpilogue`; the image has
an explicit fourth relocation for the FP32 accumulator surface.

`extra/rockchip/probe_cmac_fp32_accumulate.py` reproduces the accepted
contract. `extra/rockchip/probe_conv_fp32_partial.py` records the related CNA
boundary: a tall one-channel convolution writes exact FP32 partials, but the
proven BRDMA mode broadcasts one channel value over spatial positions. Using
4- or 8-channel RDMA geometry times out, while 16 channels submit but add only
the first partial value everywhere. That unsafe continuation probe is opt-in;
no compiler path uses it. K-split vector-matrix convolution consequently still
rejects until a per-spatial FP32 accumulation contract is proven.

This is a hardware-capability milestone, not a TestOps count change. The
authoritative census remains `204 PASS_NATIVE / 40 PASS_FRONTEND / 168 FAIL /
13 SKIP_UPSTREAM`, with no relaxed tolerance or CPU semantic path.

## Rejected FP32-accumulator output scaling

The DPU output converter cannot apply its integer multiplier, right shift, or
`MINUS_EXP` fields while `FP32TOFP16_EN` converts a flying CMAC accumulator.
`extra/rockchip/probe_cmac_fp32_output_scale.py` compares seven register
encodings: unit conversion, `1 >> 3`, `29127 >> 18`, `MINUS_EXP=3`, and the
same three shift forms with `CVT_TYPE=1`. Every variant produces byte-identical
FP16 output to unit conversion. This closes the apparent route for applying
non-exact average reciprocals after FP32 accumulation.

The previously proven FP32 CMAC epilogue remains useful for exact accumulator
continuation. Non-exact two-level average scales still reject with
`NUMERICAL_CONTRACT`; the compiler does not round the reciprocal, retry on the
CPU, or relax the official tolerance. This negative hardware milestone does
not change the authoritative `204/40/168/13` census.

## Rejected full-domain EXP2 substitution for Softmin

The generated full-domain EXP2 range reducer can replace the bounded EXP
recipe mechanically, and doing so makes the final multi-broadcast Softmin
kernel compile natively. It does not meet the existing numerical contract.
The unchanged ordinary EXP method misses 81 of 2,925 FP16 outputs with maximum
relative error `0.001703`; Softmin misses 516 of 2,925 with maximum relative
error `0.00321`. The official limits remain `rtol=0.001` (and Softmin
`atol=1e-7`).

The substitution is disabled and preserved as
`wip-full-domain-exp2-softmin-official-mismatch-20260805.patch` (SHA-256
`c44c769d345e89b3d633c76cf45684908403c119582b88b17f0d31f79ecf559a`). Softmin keeps
its pre-submission `LUT_DOMAIN_UNPROVEN` rejection. A future solution needs a
more accurate range-reduced exponential or fused normalization, not a broader
domain claim for the current tables.

## Proven flying-CONV ERDMA surface accumulation

`extra/rockchip/probe_conv_erdma_accumulate.py` proves that ERDMA can supply a
full FP16 per-spatial surface to the elementwise ADD following a flying CNA
convolution. For a one-channel, eight-position output, the accepted contract
uses DPU EW ADD `0x108202c0`, `ERDMA_CFG=0x40000008`, channel `15`, cube width
`2*N-1`, feature mode `0x17850`, and PC enable mask `0x1d`. The operand's live
FP16 value occupies every second 16-byte atom. Three consecutive submissions
were bit-exact against FP32 convolution-plus-add followed by one FP16 rounding,
and the known-good wide FP16 DPU fill passed immediately afterward.

This is distinct from the rejected BRDMA continuation probe: its proven mode
broadcasts one channel value across spatial positions, whereas this ERDMA mode
consumes the complete spatial operand. It is not yet a compiler path. Splitting
input channels would still round every convolution partial to FP16 before the
next addition, while the official convolution reference may require one FP32
accumulation. The compiler also lacks a first-class accumulator physical layout
and direct output compaction for this schedule. Those contracts must be proven
before promoting channel-split convolution; no CPU packing, tolerance change,
or task-limit increase is permitted. The authoritative census remains
`204 PASS_NATIVE / 40 PASS_FRONTEND / 168 FAIL / 13 SKIP_UPSTREAM`.

The deterministic offline follow-up
`extra/rockchip/probe_conv_ic_split_rounding.py` closes that immediate compiler
route for `test_simple_conv2d_1x1_m4`. Using the test's seeded FP16 tensors and
official `rtol=1e-3, atol=1e-6`, sequential one-channel partials miss 4,458 of
16,384 outputs and four-channel partials miss 2,286. A balanced tree still
misses 3,761 and 2,269 respectively. Therefore, the compiler must not use the
proven FP16 ERDMA continuation for this TestOps contraction. It needs a flying
FP32 per-spatial continuation, a single unsplit CNA accumulation over a legal
input layout, or another schedule with equivalent accumulation semantics.

## Rejected unaligned 32-bit DPU copy

`extra/rockchip/probe_unaligned_word_copy.py` tests the apparent low-cost path
for int32 `where_permute` and the remaining FP32 word movements: emit ordered,
overlapping four-lane DPU copies whose relocations begin at arbitrary four-byte
offsets. RK3588 ignores the low two word-index bits. Both source and destination
addresses behave as though rounded down to a 16-byte atom, so the final 5x5
transpose contains aligned source groups rather than the desired individual
words. The probe asserts this exact aligned-down result, and the known-good FP16
DPU fill passes immediately afterward.

Consequently, adding int32 to the compiler's FP32 atom-copy allowlist would
only move `test_where_permute` from a dtype reject to incorrect execution.
Unaligned FP32 negative slices and multi-source copies have the same physical
boundary. They require a real word shuffle/packing engine or an aligned staged
algorithm, not weaker compiler validation or CPU movement.

## Rejected 415-task padded transposed convolution

The first `test_padded_conv_transpose2d` subcase was characterized by raising
the 400-task ceiling to 500 only inside a one-off process. The existing
selector-CMAC plan then submitted all 415 tasks, but missed 101 output values at
the unchanged `rtol=1e-3, atol=1e-6`; maximum absolute error was `0.0234375`.
The known-good DPU fill passed afterward.

This means the case is not a fifteen-task optimization away from a valid pass.
The active 400-task rejection remains correct, and no ceiling changes. A native
solution needs a direct transposed-convolution/layout schedule that also
preserves the reference accumulation semantics; merely shrinking or admitting
the current selector schedule is insufficient.

## Rejected channel-packed FP16 mask output

The earlier public-bool experiment used the ordinary linear FP16 geometry:
eight processing lanes per spatial position and int8 WDMA. A second probe now
tests the other plausible layout directly: one spatial position with sixteen
FP16 input/processing channels and one sixteen-byte int8 output atom. It sets
FP16 input and processing precision, int8 output precision, channel 15 in DPU
and MRDMA, width zero, and a sixteen-lane WDMA write.

That submission also times out with errno 110. The runtime reset recovers the
device and a 32-lane FP16 fill passes immediately afterward. Mesa Rocket's
quantized output converter does not provide a missing configuration here: its
working path uses int8 input, processing, and output throughout, rather than
FP16 processing followed by int8 storage. The reproducible experiment is
`probe_fp16_mask_int8_channel.py`.

Public bool results derived from FP16 therefore remain a typed pre-submission
reject. Existing all-int8 bool fill/copy/AND/OR/NOT and bounded bool-WHERE paths
stay enabled because they use independently proven contracts. No method count,
resource ceiling, or tolerance changes; the authoritative census remains
`204 PASS_NATIVE / 40 PASS_FRONTEND / 168 FAIL / 13 SKIP_UPSTREAM`.

## Unreduced BCE needs a narrower numerical recipe

The probability-BCE guard was disabled only inside a serialized probe process
and the existing staged LUT expression was compared at the official
`rtol=1e-3, atol=1e-6`. It misses 39 of 320 outputs, with maximum absolute and
relative errors `0.001953125` and `0.00434`. Rewriting the same inputs to the
standard logits BCE expression is not sufficient: that native recipe misses 39
of 320 outputs, up to `0.0009765625` absolute and `0.005554` relative.

`probe_bce_unreduced.py` reproduces both measurements with the seeded TestOps
inputs. The active pre-submission numerical rejection remains correct. A future
implementation needs a narrower second-level logarithm/softplus LUT or another
measured fused recipe; canonicalization alone cannot make this method pass.

## Selector planner module boundary

Static selector construction, CMAC payload packing, bounded window/two-level
planning, program composition, and semantic-reformat legalization now live in
`renderer/rockchip/selector.py`; shared compiler limits live in `limits.py`.
The public lowering module falls from 3,320 to 3,038 physical lines. This is a
mechanical extraction: plan selection, cost ordering, constants, scratch,
command words, and RKImage bytes are unchanged.

The full compiler/image/telemetry suite passes 171 tests plus 59 subtests.
Mypy passes all thirteen Rockchip renderer modules and Ruff is clean. Four
serialized hardware regressions cover selector reformat, ordered DPU+CMAC sum,
K65-to-K96 contraction, and direct CONV; all pass with eleven subtests. This
milestone changes no coverage result, execution lane, limit, or tolerance.

## Shared lowering-analysis module boundary

The compile-time UOp helpers shared by reformat, reduction, contraction, and
convolution lowering now live in `renderer/rockchip/analysis.py`. The module
contains cast stripping, scalar coordinate evaluation, static linear-form and
conditional-index analysis, convolution-padding proofs, and epilogue matching.
It reads no tensor data and produces no target task; lowerers retain the same
typed results and physical schedules.

The compiler/image/telemetry suite again passes 171 tests plus 59 subtests.
Representative serialized RK3588 reformat, reduction, K65 contraction, and
direct-convolution tests pass with eleven subtests. This mechanical prerequisite
for splitting whole lowerer families changes no coverage or numerical contract.

## Rejected logarithmic cumulative-product schedule

A Hillis-Steele-style prefix product would reduce native task growth from one
materialized selector per prefix to logarithmic DPU multiply passes. The
deterministic `probe_cumprod_parallel.py` applies FP16 rounding after every
pass to the unchanged seeded TestOps shapes and compares at `rtol=1e-3,
atol=1e-6`.

The 20-element vector happens to pass, but the `(20,30)` axes miss 65 and 68
of 600 outputs, and the 40-element last-axis scan misses 1,766 of 24,000.
This is an offline numerical rejection before hardware submission. NumPy's
linear FP16 accumulator also misses 23/600, 55/600, and 2,111/24,000 outputs,
showing that simply preserving left-to-right FP16 association is insufficient
for Torch's contract on the larger shapes. The active proven 20-element tiled
implementation remains, and larger scans continue to reject instead of using
either inaccurate FP16 schedule.

## Rejected FP32 weighted-interpolation output

The static weighted-CMAC planner was generalized experimentally to emit the
proven 32-lane FP32 CMAC surface and to accept up to eight affine interpolation
taps. All three first bilinear kernels became native in 72, 47, and 23 tasks,
with 65,632, 10,336, and 20,608 constant bytes.

The complete unchanged method still rejects immediately afterward. Tinygrad's
interpolation schedule stores that FP32 affine result in an internal surface,
then launches a separate noncontiguous FP32-to-FP16 kernel for the public half
output. RK3588's direct FP32 DPU input path is already proven to time out, so
the first-kernel improvement cannot honestly complete the operation. The active
compiler is restored; the exact experiment is commit `90adfa2cb` and patch
`0263-WIP-probe-fp32-weighted-interpolation.patch`.

## Ordered lowering orchestration

The generic native/not-applicable/unsupported constructors, reduction-family
applicability predicate, rejection-priority policy, and ordered pass-selection
loop now live in `renderer/rockchip/lower.py`. Concrete lowerers and their
explicit registry remain in the public compiler module, so ordering is still
visible while the common typed orchestration no longer depends on any engine
implementation.

The extraction preserves rejection fingerprints, physical schedules, cost
ceilings, and RKImage bytes. The full compiler/image/telemetry suite, full-tree
Mypy and Ruff, and representative serialized reformat, reduction, K65
contraction, and direct-CONV hardware tests pass. Coverage remains the
authoritative `204 PASS_NATIVE / 40 PASS_FRONTEND / 168 NATIVE_REJECT / 13
SKIP_UPSTREAM` because no lowering rule changed.

## Reformat lowering module

Static convex two-tap, single-source, multi-source, and selector-expression
movement lowering now lives in `renderer/rockchip/reformat.py`. The module
turns static UOp address relationships into typed semantic reformat plans and
uses the shared selector planner only during physical legalization. It does not
read runtime tensor data or add a new execution lane.

The move preserves the concrete lowerer order and every target-plan decision,
while reducing the public compiler module from 2,864 to 2,522 physical lines.
Frozen image tests, the complete compiler/image/telemetry suite, full-tree
Mypy/Ruff, and representative RK3588 hardware methods remain green. The
authoritative native census is unchanged at `204/40/168/13`.

## Rejected K128 conv3d selector path

Allowing the generic tiled-CMAC lowerer to accept K=108 does not solve the
remaining conv3d methods. The unchanged `test_simple_conv3d` geometry reaches
the wider planner, which computes a minimum of 1,226 selector tasks before
payload generation. The existing ceiling is 400 tasks.

The K96 legality bound therefore remains active. Commit `a19d43fa0` and patch
`0267-WIP-probe-K128-tiled-contraction.patch` preserve the experiment. Future
conv3d work must use direct 3D physical layouts or another tile-proportional NPU
schedule; increasing the selector task ceiling would only hide the blocker.

## MAX and pooling lowering module

The PPU global-MAX, sliding-pool, dense-row-MAX, and general affine-MAX
lowerers now live in `renderer/rockchip/pool.py`. Their shared PPU geometry
guards, CMAC/HWC preparation, scratch planning, and typed rejection paths move
together, while the ordered registry remains visible in the public compiler.

The main compiler module falls from 2,522 to 2,039 physical lines. Frozen image
tests and the complete compiler/image/telemetry suite pass; full-tree Mypy and
Ruff are clean. RK3588 tests cover direct global and sliding PPU work plus
CMAC-prepared padded and batched affine MAX. This is a byte-preserving module
split, so native coverage remains `204/40/168/13`.

## Arithmetic reduction lowering module

The SUM/mean, nested SUM, MUL/product, masked prefix, multi-source, pointwise,
and general affine reduction lowerers now live in
`renderer/rockchip/reduce.py`. Their shared epilogue construction stays with
the schedules it finishes, while the direct contraction lowerer imports that
typed helper. The exact-FP16 constant predicate is shared through the pure
analysis module.

The public compiler module falls from 2,039 to 1,291 physical lines. Frozen
images, the complete compiler/image/telemetry suite, full-tree Mypy/Ruff, and
representative RK3588 SUM, product, prefix, and variance paths all remain
green. No capability or limit changes, so coverage stays `204/40/168/13`.

## Contraction lowering module

The direct logical contraction, tiled CMAC, depthwise/grouped convolution,
NHWC convolution, and general NCHW spatial-convolution lowerers now live in
`renderer/rockchip/contract.py`. This places recognition, physical packing,
CBUF tiling, epilogue composition, and `RKCMACTask`/`RKConvTask` construction
behind one engine-family boundary while leaving emission independent.

The public renderer/compiler entry is now 487 physical lines, down from 1,291
and below the 500-line organization goal. No handwritten code moved into
`autogen/`. Frozen images, the full compiler/image/telemetry suite, Mypy, Ruff,
and representative CMAC/CONV hardware paths remain green. Coverage is unchanged
at the authoritative `204/40/168/13`.

## Rejected wide planar PPU packing

Two preserved WIP commits prove that the `(32,2,11,28)` `test_max_pool2d`
family can be recognized as one 64-plane sliding pooling operation, but cannot
be legalized efficiently with the current selector-CMAC reformatter. The
logical input has 19,712 contiguous NCHW elements; PPU requires HWC8 atoms, and
the resulting transpose exceeds the existing planner bounds. The local RK3588
reference performs the same conversion on the CPU, which this backend does not
adopt. Active legality is restored unchanged: future work needs a native
NCHW-to-HWC8 conversion path rather than a higher 400-task ceiling.

## PPU asymmetric MAX padding hardware contract

The raw sliding-PPU probe now covers independent four-sided padding. Three
FP16 MAX geometries using 5x5 and 3x2 kernels, nonuniform strides, and asymmetric
top/bottom/left/right padding are bit-exact on RK3588 with zero mismatches over
624 outputs. The PPU padding register alone is sufficient for these MAX cases.

This is hardware-contract evidence, not a new TestOps pass. Large padded NCHW
inputs still require a native conversion to the PPU's HWC8 external layout;
the current selector schedule would require 826 tasks. The compiler therefore
continues to reject that layout before submission instead of raising resource
ceilings or using CPU packing.

## Native padded sliding MAX

`RKPool` now carries four explicit padding sides. The sliding-pool lowerer
derives those sides from the affine guarded load and proves every coordinate:
real input points must select the unique FP16 source load, while padding points
must select negative infinity. Emission validates the three-bit fields and
writes the characterized PPU padding register directly.

The official `test_max_pool2d_padding` method passes in focused native-only
testing. A representative plan uses one PPU task inside a 150-task bounded
selector schedule, so it remains classified as a correctness fallback until a
direct NCHW-to-HWC8 conversion exists. No CPU packing, tolerance relaxation, or
resource-limit increase is used. Expected coverage is `205/40/167/13`; the
last full-census result remains authoritative at `204/40/168/13`.

## One-task GEMM: compute is proven, packing is the blocker

The broad GEMM schedules in `rockchip_addmul`, `rockchip-2607`,
`allbilly/npu`, and `allbilly/rk3588` were compared with the clean emitter.
They use the same physical contract already understood here: an aligned CNA
activation surface, a weight stream blocked as
`[output/16,input/32,output_lane,input_lane]`, and the gapped FP16 CMAC output
surface. The clean logical legalizer now accepts that complete already-packed
contract and emits one `RKCMACTask`.

`probe_cmac_width.py` verifies actual one-task GEMMs at 16x16x16, 32x32x32,
64x64x32, 16x16x64, and 8x16x32. Every output is bit-exact against FP32 matrix
multiplication rounded once to FP16. Probe-side NumPy packing is hardware ABI
characterization only; it is never invoked by the compiler or runtime and is
not native coverage.

Standard dynamic tinygrad matmul still needs an NPU-resident transform from
row-major RHS to the blocked weight stream and a physical-output compaction.
The existing scalar strided-DPU experiment does not provide that transform:
unaligned column offsets are atom-aligned and tested notch values do not change
the observed eight-lane stride. The backend therefore keeps selector-CMAC as a
bounded correctness fallback and does not raise resource limits or copy the
references' CPU packing.

Within that fallback, dense unmasked contractions up to 64 physical output
channels use the already-proven 64-output compact selector. This reduces the
64x64 plan from 305 to 217 tasks and `1x64 @ 64x40` from 83 to 43. Conditional
and padded contractions stay at 32 outputs to preserve their characterized
rounding, while N=99 keeps the cheaper existing candidate. Official simple,
batched, and batched-vector matmul tests remain green with no coverage or limit
change.

The same physical ABI is now proven at `M=1,N=128,K=128`: one broad CMAC task
is bit-exact. Enabling the corresponding ordinary row-major `test_matvec`
graph is still not clean, because current selector packing expands the dynamic
rhs transform to 1,074 tasks. The logical K/source fences and 400-task ceiling
remain active. This separates the next requirement precisely: implement a
direct native weight-layout transform rather than split the GEMM or admit a
larger selector schedule.

## One-row FP32 CMAC writeback is already compact

The direct CMAC width probe now places a canary after a 32-value FP32 result.
With the production `SURFACE_ADD=0x40` setting, one physical task writes the
32 FP32 channels contiguously and leaves every following canary word untouched.
The existing `(1,64)` layout declaration is therefore conservative physical
allocation metadata, not an observed 64-word output footprint.

The FP16 compact setting `SURFACE_ADD=0x20` must not be reused for FP32. It
writes channels 0--7 followed by channels 16--31 and leaves the final eight
words untouched. This is a precision-dependent address-generator contract,
not a generic byte-stride control.

This result creates a possible direct destination for 32-output FP32 CMAC
tiles, including weighted interpolation. It does not yet enable a compiler
path: the legalizer must first construct those tiles without overlapping
writes and prove the official FP32 accumulation contract. Coverage remains
`204/40/168/13`.

## Rejected auxiliary FP32-to-FP16 cast

The fused lerp datapath proves that BRDMA can supply an FP32 operand to BS even
though direct FP32 MRDMA input times out. Reusing that path as
`(0 - x) * -1 + 0` converts finite values, infinity, and NaN correctly, but it
turns negative zero into positive zero. Supplying negative-zero addends does
not repair the sign.

The alternative BS multiply route was also tested behind
`ROCKCHIP_UNSAFE_BS_MUL=1`. External multiplier source zero emits all zeros;
source one times out. The ordinary wide FP16 fill suite passes after runtime
recovery. A generic cast therefore remains unsupported: the working auxiliary
path is valid for the specific fused lerp expression, not a bit-complete FP32
input conversion contract.

## K=128 boundary census

The complete uncached strict census at `68c4d8272` confirms that the one-task
prepacked `M=1,N=128,K=128` characterization changes no existing method
behavior: 204 methods pass natively, 40 are frontend-only, 168 fail with typed
native rejects, and 13 retain upstream skips. All 168 failures retain a
method-level first reject; no fallback execution, numerical mismatch, timeout,
invalid submission, reset failure, process abort, or unclassified failure
occurred during the 3,488.50-second run.

The exact first-reject distribution is 53 unsupported-output-dtype, 35
plan-stage-limit, 26 unsupported-input-dtype, 18 unsupported-layout, 16
numerical-contract, 12 unsupported-ALU, five requires-reformat, two
unaligned-row, and one LUT-domain-unproven. This makes the next matvec milestone
unambiguous: preserve the proven one-task broad CMAC compute and replace the
1,074-task generic row-major-to-weight transform with a bounded device-native
block pack. No resource ceiling or numerical tolerance is increased.
