# RK3588 fixed-function NPU backend

This is the active coverage/research branch. It targets the complete forward
`test_ops.py` inventory through RK3588 engine tasks, with frontend-only methods
reported separately and unsupported kernels rejected before submission.
The frozen `rockchip-pr`, `rockchip-2608`, and `rockchip-2607` branches remain
minimal, architectural, and behavioral/register-programming references.

The current authoritative uncached strict census at `02ae2f927` is 168 native,
40 frontend-only, 204 failed, and 13 upstream-skipped methods across the exact
425-method inventory. It completed without an NPU timeout, reset failure,
invalid submission, or process abort. Relative to `61288f302`, only
`test_grouped_conv2d` and `test_max_pool2d_unit_stride` change from failure to
native pass and there is no regression. Coverage details and durable artifact
hashes are recorded in `coverage.md`.

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
       RKContract
       RKSpatialConv
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
The current serialized device contract passes 90 tests plus 58 subtests in
731.94 seconds with fallback disabled.

Lowering uses twenty-one named ordered strategies grouped into elementwise,
movement/reformat, sum/product/MAX reduction, and contraction families. Every
strategy returns exactly one of `NATIVE`, `NOT_APPLICABLE`, or `UNSUPPORTED`: unrelated passes
cannot overwrite a useful reject, while applicable failures are ranked by
specificity before telemetry is emitted.

The compiler is split by responsibility under `renderer/rockchip/`:
`ir.py` owns the typed UOp-free plans, `expr.py` owns math recipes and UOp
canonicalization, `affine.py` owns affine maps and reject fingerprints,
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

Channel bias and optional ReLU now use a typed `RKEpilogue` on `RKContract`.
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
surfaces, one typed `RKSpatialConv` task runs per batch, and selectors unpack
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

Wide int32 and FP32 constant fills consume an FP16 DPU input even though WDMA
writes a wider public dtype. Legalization now requires that constant to
round-trip through FP16 exactly. Values such as `INT_MAX` and FP32 `0.1`
therefore return `NUMERICAL_CONTRACT` before image emission, while proven exact
fills remain native. This removes the raw FP16-packing `OverflowError` exposed
by `test_maximum` without claiming general integer or FP32 arithmetic.

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
