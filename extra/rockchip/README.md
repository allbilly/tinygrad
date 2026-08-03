# RK3588 fixed-function NPU backend

This is the active coverage/research branch. It targets the complete forward
`test_ops.py` inventory through RK3588 engine tasks, with frontend-only methods
reported separately and unsupported kernels rejected before submission.
The frozen `rockchip-pr`, `rockchip-2608`, and `rockchip-2607` branches remain
minimal, architectural, and behavioral/register-programming references.

The current authoritative uncached strict census at `2bf8c337b` is 154 native,
40 frontend-only, 218 failed, and 13 upstream-skipped methods across the exact
425-method inventory. Raw pytest reports 206 passed, 255 failed, 13 skipped,
and 77 passing subtests in 2,255.75 seconds. It completed without an NPU
timeout, reset failure, invalid submission, or process abort. Coverage details
and durable artifact hashes are recorded in `coverage.md`.

Relative to the SIN census, only `test_mulacc_with_zero_strides` changes from
failure to native pass. Every remaining failure is now a typed native reject;
there are no device or unclassified frontend failures in the authoritative
inventory.

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
The richer candidate ordering preserves the complete strict result: 81 tests
plus 56 subtests pass in 750.28 seconds with fallback disabled.

Lowering uses seventeen named ordered strategies grouped into elementwise,
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
contains 28,957, so `MAX_LINE_COUNT=25000 python sz.py` fails by 3,957 lines.
The exact 3,989-line delta is 3,980 counted Rockchip backend lines, the
five-line generic native-program hook, and four generic correctness lines.
Generated
register definitions, LUT payloads, and reproducible command data belong under
`runtime/autogen`; handwritten legality, layout, scheduling, and emission logic
remain counted. The generic hook can be reviewed separately, while the backend
needs real upstream line budget or independently useful in-tree reductions
before submission.
