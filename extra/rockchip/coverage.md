# Rockchip forward TestOps coverage

The completion target is every non-skipped forward case in the 425-method
`test/backend/test_ops.py` inventory. Focused tests are recorded as milestone
evidence but do not replace a complete uncached census.

All authoritative runs use `ROCKCHIP_FALLBACK=0`. The optional `RKPY` research
envelope invokes tinygrad's Python UOps emulator over mapped GEM buffers and is
therefore CPU semantic execution; it is excluded from pass counts. Likewise,
the 425-green `rockchip-2607` result cannot be treated as all-NPU proof because
that branch dispatches many families through NumPy-backed `_run_host_*` tasks.

## Current strict census

The complete uncached run at `95c0cc501` contains exactly 425 method records:
153 `PASS_NATIVE`, 40 `PASS_FRONTEND`, 219 `FAIL`, and 13 `SKIP_UPSTREAM`.
`ROCKCHIP_FALLBACK=0`, `CACHELEVEL=0`, and `SCACHE=0` were set throughout.
Raw pytest reports 256 failed methods/subtests, 205 passed, 77 passing subtests,
and 13 skipped in 2,250.74 seconds. No NPU timeout, invalid submission, reset
failure, or process abort occurred.

Relative to the `d237777da` census, only `test_sin` changes from failure to
native pass. The 219 failed methods first classify as 65
unsupported-output-dtype, 49 plan-stage-limit, 35 unsupported-layout, 23
unsupported-input-dtype, 18
requires-reformat, 16 numerical-contract, nine unsupported-ALU, and three
unsupported-reduction. Schema version 2 classifies 218 as `NATIVE_REJECT`;
only `test_mulacc_with_zero_strides` is a pre-device Clang failure without a
native reject. The post-census cumsum guard now records its structural
multi-tile numerical reject. `test_avg_pool2d_padding` reaches stage limit
before its former numerical contract in this run; neither method changes
outcome. The durable telemetry is
`~/rk2608_backups/census-sin-95c0cc501-20260803/test_ops_coverage.json`
(SHA-256 `0d6c92f01df86185e158077d6ce693253d1a947793b05250e736ec5015d9d57e`);
the JUnit XML SHA-256 is
`549009e30fa00aedca50fcfe3abcbee759271c83e9877441b0b7889857ec68a4`.

RKImage v3 removed the unused dependency/read/write masks and the artificial
64-stage image limit. The runtime already executes stages serially, so the
image now encodes that actual contract directly. The stale matching planner
fence is also gone; a 65-stage FP32 fill round-trips through the image codec
and passes strict RK3588 execution. A subsequent contiguous FP16 global-MAX
tree was rejected after hardware characterization: DPU source addresses are
effectively 16-byte atom aligned, so sub-atom relocation addends are ignored.
An extent sweep passed only 9/17/33/65 by accidentally copying an aligned odd
tail; 2/3/4/5/7/8/15/16/31/32/63/64/127/128/135 were wrong. The exact failed
experiment is archived as `0058-WIP-global-max-DPU-tree-unaligned-atom-failure.patch`
with SHA-256 `a1142aa6bb21b0e1e534ebd0044567a2e1a7089f05e2a28afd11757988950333`.
This confirms that reductions require a native physical reformat/layout stage
or a directly legal PPU surface; relocation offsets are not a valid gather.

The first directly legal reduction surface is now proven without host packing:
a dense FP16 HWC8 tensor with 4–256 spatial positions lowers to one typed
`RKReduce`, then one PPU global-MAX task. Hardware boundary tests cover
2x2x8, 4x4x8, and 16x16x8 with `ROCKCHIP_FALLBACK=0`. Input and output are
ordinary mapped Rockchip GEM buffers; no NumPy runner, gather, reformat, or
fallback lane participates. The PPU task descriptor uses the independently
verified `(op_idx=1, enable_mask=0x60, int_mask=0xc00)` contract. Shapes that
are not already dense HWC8 still reject until a native reformat stage exists.

Dense FP16 row sums with 4–16 rows and exactly 32 values per row now lower to
the existing typed `RKContract`. The image embeds one 64-byte FP16 ones vector
as an immutable constant, while the user input remains a directly addressed
`(N,32)` GEM surface. A strict 8x32 RK3588 test matches FP32 accumulation
rounded to FP16. Neither compilation nor execution performs host packing or
host arithmetic; unsupported row widths continue to reject.

Contiguous FP16 global sums now use a typed ordered `RKProgram`. The compiler splits
an arbitrary extent into descending power-of-two blocks. DPU pairwise stages
reduce each block only while the second half begins at a 16-byte address; the
remaining aligned runs are copied into one scratch surface and a masked K32
CMAC produces the scalar. The four repeated CMAC weight rows are immutable
`[1...1,0...0]` image constants, so no invocation-time packing occurs. Plans
whose block decomposition needs more than 32 final terms reject explicitly.
For extents 33–256, a stronger path pads the feature surface to K32 alignment
with one DPU copy and emits one variable-K CMAC. Its generated weights are
swizzled for the hardware layout and contain ones only for logical lanes, so
the accumulation remains FP32 until the final FP16 store. The permanent
seed-zero 135-element case matches the TestOps input distribution.

The initial all-DPU tree was an important rejected probe: both values of a
two-element input were read from lane zero, producing `2*x[0]`. Disabling both
disk and schedule caches reproduced the result. It proves that DPU
`EW_BASE_ADDR` cannot select a sub-16-byte lane through a relocation addend;
ordinary tensor buffer views that start at an offset are a different ABI case.
The committed planner consequently never emits such a relocation. Strict
hardware tests cover lengths 2, 16, 60, 135, 720, and 16,384. With the actual
TestOps selector (`DEV=ROCKCHIP`) and persistent compilation disabled
(`CACHELEVEL=0`), `test_sum_full` passes. Static affine multi-output ADD
reductions now enumerate their source selector matrix at compile time, copy
the input once into an aligned NPU scratch surface, and execute sparse CMAC
weight tiles. Together with the global path, this completes `test_sum_simple`,
`test_sum_full`, `test_sum_relu`, `test_sum_tiny`, `test_sum`, `test_mean`, and
`test_mean_axis`; at that milestone nested sums and explicit FP32 accumulation
remained unsupported. The later ordered nested-reduction milestone below
supersedes the nested-sum limitation without removing its rounding boundary.
Global `test_sum_relu` now passes through a DPU MAX-zero prepass and the direct
K64 CMAC; the final ReLU is removed only after proving all reduced lanes are
nonnegative. Global `test_mean` also passes: its compile-time reciprocal is
folded into generated K384 CMAC weights, preserving one FP32 accumulation and
one final FP16 conversion. Earlier claims for other methods came from mistakenly
using `DEVICE=ROCKCHIP`, which left TestOps on the default CPU backend; they
are superseded here. No tolerance or skip was changed.

A first attempt issued one CMAC task with 32–128 physical output channels.
Hardware returned valid results only for logical channels 0–15; channel 16
was the exact failure boundary across output sizes 18, 24, 60, and 90. The
failed experiment is preserved as
`wip-affine-cmac-wide-channel-boundary.patch` with SHA-256
`31bfecbb7cd9c80c957eb03de8aac9b1fa72962a720830afa61aaf70b4a100b9`.
The committed planner keeps the proven 32-channel physical CMAC command but
tiles at most 16 logical outputs per task. Tasks write sequentially at
16-element-aligned output offsets, so each later task overwrites only padding
from its predecessor. A strict `(3,4,5,6)` hardware matrix covers axes `3`,
`(1,3)`, `(0,2)`, `(1,2)`, and `1`, plus mean and a `(4,2,2)` tiny reduction.
The full explicit-device suite passes 44 tests plus 22 subtests in 328.63
seconds with `CACHELEVEL=0` after this milestone.

The native reformat path handles static affine movements without host gather.
The compiler enumerates the complete static selector map. Contiguous aligned
runs still coalesce into ordinary DPU ADD-zero atom copies. A map that breaks
those atoms now reuses the generated sparse-CMAC `RKProgram`: one DPU task
zeroes the complete aligned scratch surface, a second copies the logical input,
then sequential CMAC tasks select at most 16 logical outputs each. Making the
padding explicit avoids order-dependent accumulation from stale scratch bits.
The planner accepts at most 4,096 outputs and an 8 MiB generated-weight budget.
It does not recognize transpose, permute, flip, expand, slice, or unfold by
name.

Strict hardware cases cover 9- and 27-element transposes, a 360-element
permute, a 432-element multi-axis flip, and an 864-element expand. All match
exactly with `ROCKCHIP_FALLBACK=0`. Focused official methods now pass for
`test_transpose`, `test_permute`, `test_flip`, `test_expand`,
`test_stack_slice`, `test_unfold`, `test_slice_stride_gt_one`,
`test_double_slice`, and `test_diagonal`. The complete explicit-device suite
passes 45 tests plus 27 subtests in 340.26 seconds with `CACHELEVEL=0`.
The strict census and padding stabilization are recorded below.

Earlier slice tests appeared to show that DPU `SRC_BASE_ADDR` honors FP16
sub-atom relocation addends. The reduction probe separates two cases: a GEM
buffer view bound at an offset works, while adding two bytes to an already
bound DMA address reads lane zero. The DPU atom-copy planner therefore still
requires aligned task addresses; the sparse-CMAC fallback handles small
unaligned selector maps after one aligned full-surface copy instead of emitting
an illegal addend. Real offset buffer views still pass lengths 1, 2, 3, and 8
on hardware. Enabling the separate `ERDMA_NONALIGN` bit was also rejected: it
caused ordinary two-input DPU arithmetic to time out and is not part of the
committed path.

An independent unaligned-destination probe also timed out on the first
official flip shape. Its exact planner diff and failure are preserved as
`0063-WIP-unaligned-DPU-destination-timeout.patch`; the active compiler never
restores that register path. Sparse CMAC outputs start on proven 16-element
tile boundaries and later tiles overwrite only the preceding tile's padding.

The next numerical milestone ports 2607's genuine NPU-only POW8 algorithm into
the typed expression planner. The generated two-level LUT recipe passes 513
points over `[-4.1, 4.1]` plus both infinities and NaN on RK3588 with
`ROCKCHIP_FALLBACK=0`. The exact official `test_pow_const` invocation now
passes its `x**8.0` subcase and advances to the independent `x**5.5` mismatch
(117/2,925 lanes, maximum relative error 0.001953). The method is therefore
still `FAIL` and no method-level census gain is claimed yet.

`python sz.py` at this milestone reports 1,686 counted lines in the Rockchip
renderer and 26,819 repository lines overall. The two immutable generated LUT
payloads add 154 physical lines under `autogen`, which `sz.py` correctly
excludes; their auditable source is the 40-line generator change. The 44-line
typed compiler recipe remains counted rather than being hidden as generated
code.

Positive scalar `x**5.5` now uses three generated fixed-point ranges inside a
90-stage typed plan. An exhaustive strict-hardware sweep passes all 17,410
finite positive FP16 encodings in `[0,4]`, plus invalid-domain and special
values. The official method passes this subcase and advances to `x**-5.5`,
which still misses 379/2,925 lanes (maximum relative error 0.003021). No
method-level census gain is claimed. At this milestone `sz.py` reports 1,720
counted Rockchip-renderer lines and 26,853 repository lines overall; generated
payload rows remain excluded, while their generator and compiler recipes are
counted.

Negative scalar `x**-5.5` now bypasses the rounded reciprocal through shifted
Q10/Q15/Q15 generated ranges and explicit overflow-boundary arithmetic. All
18,434 positive FP16 bases in `[0,8]` pass exhaustive RK3588 comparison, and
invalid finite negative bases remain NaN. The official method now passes both
scalar exponents `+5.5` and `-5.5`, then reaches the independent constant-base
`5.5**x` failure (1,756/2,925 lanes in the current generic path). The method
remains `FAIL`; no census gain is claimed. `sz.py` reports 1,770 counted lines
in the renderer and 26,903 repository lines overall at this milestone.

Constant-base `5.5**x` now uses two generated Q15 ranges selected by generic
DPU masks, with the ordinary native EXP2 recipe outside `[-2,2]`. All 32,770
finite FP16 encodings in that interval plus infinities and NaN pass exhaustive
RK3588 comparison at the official tolerance with `ROCKCHIP_FALLBACK=0`. The
official `test_pow_const` invocation passes this subcase and reaches the
independent negative-base `(-5.5)**x` compiler rejection. The method therefore
remains `FAIL`; no method-level census gain is claimed. `sz.py` reports 1,781
counted lines in the renderer and 26,914 repository lines overall; the 154
new generated payload lines under `autogen` are excluded, while the 11-line
net compiler growth remains counted.

Negative-base `(-5.5)**x` now derives truncation, integer validity, and parity
with two native roundoff-LUT tasks plus generic DPU masks and arithmetic. An
exhaustive strict-hardware sweep passes all 32,770 finite FP16 exponents in
`[-2,2]`. The official chained test passes this subcase and reaches the
independent `8.0**x` numerical mismatch (1,975/2,925 lanes, maximum relative
error 15 in the generic path). The method remains `FAIL`; no census gain is
claimed. `sz.py` reports 1,807 counted renderer lines and 26,940 repository
lines overall at this milestone.

Constant-base `8.0**x` now uses four generated Q15 output-scale bands and a
generic native EXP2 fallback. All 32,770 finite FP16 encodings in `[-2,2]`
plus infinities and NaN pass exhaustive RK3588 comparison. The official
method passes this subcase and every following square/base-two tensor and
scalar case, then reaches only the final `0**x` float-input special case. The
method remains `FAIL`; no census gain is claimed. `sz.py` reports 1,823
counted renderer lines and 26,956 repository lines overall. The 306 new
autogen payload lines are excluded; net counted compiler growth is 16 lines.

The final zero-base subcase currently supplies an explicit FP32 input despite
`DEFAULT_FLOAT=HALF`. The 2607 runtime did not execute that input natively: it
serialized `fp32_inputs` metadata and used NumPy in `ops_rockchip.py` to narrow
the buffer to FP16 before NPU submission. That path is prohibited here.

A direct hardware probe tested the apparent alternative from the RK3588
register enum: set DPU `DATA_FORMAT.in_precision` to `precision_float32` and
perform FP32-to-FP16 ADD-zero entirely in the submitted task. The submission
timed out with `Errno 110`; restoring FP16 input precision recovered normal
execution. The exact failed typed-stage/emitter probe is preserved as
`wip-native-fp32-dpu-input-timeout.patch` with SHA-256
`d40b01ca5cd11e247ed295c3257f15d2fa11ac8756481ceda8799f3a4810ea25`.
Therefore `0**x`, `0.7**x`, and `(-2)**x` with explicit FP32 input remain
honest native rejections until another RK3588 engine or a native reformat path
is proven; CPU narrowing will not be used to make this method green.

The uncached strict census at `05dbc7570` confirms 425 methods with 83
`PASS_NATIVE`, 40 `PASS_FRONTEND`, 289 `FAIL`, and 13 `SKIP_UPSTREAM` in
492.27 seconds. The three power milestones advance successive subcases inside
`test_pow_const` but do not complete its explicit-FP32 tail, so they correctly
claim no method-level gain. The machine-readable report is preserved at
`~/rk2608_backups/census-20260802-131000/test_ops_coverage.json`.

The same hardware boundary appears at public bool output. 2607 wrote an
atom-padded FP16 mask and used NumPy after submission to pack it into byte-bool.
A strict replacement probe configured only the final mask stage for DPU int8
WDMA output; submission timed out with `Errno 110`. Normal FP16 DPU execution
recovered after restoring the proven format. The exact failed probe is archived
as `wip-native-int8-bool-wdma-timeout.patch` with SHA-256
`4b671b65e46c984305dc4751e54c456c21ecf2686cbd778fa03966fda9444e49`.
Comparison masks remain native when consumed inside FP16 arithmetic, but public
byte-bool output remains an honest rejection without CPU packing.

Run the census serially on RK3588:

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
PYTHONPATH=/home/orangepi/rk_2608/test/rockchip:/home/orangepi/rk_upstream \
CACHELEVEL=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
python -m pytest test/backend/test_ops.py -p conftest_rockchip -q --tb=no
```

Add a persistent machine-readable report without changing execution behavior:

```sh
ROCKCHIP_TELEMETRY=~/rk2608_backups/test_ops_coverage.json \
PYTHONPATH=/home/orangepi/rk_2608/test/rockchip:/home/orangepi/rk_upstream \
CACHELEVEL=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
python -m pytest test/backend/test_ops.py -p conftest_rockchip -q --tb=no
```

The version-1 JSON records the commit, environment, RK3588/driver identity,
method result, exact unittest subcase parameters and result, every executed
kernel lane, physical compiler signature, engine counts, RKImage version,
stage count, scratch bytes, constants bytes, duration, and every native reject.
The method summary uses `PASS_FRONTEND`, `PASS_NATIVE`, `PASS_MIXED`,
`PASS_FALLBACK`, `SKIP_UPSTREAM`, and `FAIL`. `PASS_NATIVE` is assigned only
when every realized kernel in every subcase executed through an RK engine.
Typed rejects include the exact legality detail, offending op, and a normalized
UOp fingerprint containing graph structure, dtype/shape families, constant
categories, affine index maps, and reduction descriptors while omitting buffer
slots.

At `da09c1fd9`, the strict uncached census records exactly 425 methods:

| Coverage outcome | Methods |
|---|---:|
| `PASS_NATIVE` | 90 |
| `PASS_FRONTEND` | 40 |
| `FAIL` | 282 |
| `SKIP_UPSTREAM` | 13 |

This is a seven-method native gain from the 83/289 baseline: `test_sum_simple`,
`test_sum_full`, `test_sum_relu`, `test_sum_tiny`, `test_sum`, `test_mean`, and
`test_mean_axis`. Pytest's mixed parent/subtest summary is `151 passed, 387
failed, 13 skipped` in 498.79 seconds; it is not the method census. The run had
no NPU timeout or reset failure. Its 256 typed reject events are:

| Typed reject kind | Events |
|---|---:|
| `unsupported_layout` | 63 |
| `unsupported_output_dtype` | 52 |
| `unsupported_reduction` | 46 |
| `requires_reformat` | 36 |
| `unsupported_contraction` | 27 |
| `unsupported_input_dtype` | 25 |
| `unsupported_alu` | 7 |

The largest remaining exact detail families are noncontiguous outputs (46
methods), unsupported ADD reductions (28), contraction output reformat (20),
and non-direct contraction epilogues (19). The durable JSON is
`~/rk2608_backups/census-affine-reduce-da09c1fd9/test_ops_coverage.json`
(SHA-256 `58541b3554192fd8550809bd546d2e00396489876a9893cf841acaa95cc71c09`);
the matching JUnit XML has SHA-256
`d44a26ac3cdf6887c3f2ca16d6cf19ea95a94d67c7743d776381152e4cd0b393`.

At `52f34b131`, the next strict census reaches 101 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 271 `FAIL`, and 13 `SKIP_UPSTREAM` in 544.95 seconds. The 11
exact gains are `test_diagonal`, `test_double_slice`, `test_flip`,
`test_meshgrid`, `test_permute`, `test_slice_ellipsis`,
`test_slice_one_endpoint_out_of_bounds`, `test_slice_stride_gt_one`,
`test_stack_slice`, `test_transpose`, and `test_unfold`; no prior pass regressed.
Typed rejects fall from 256 to 244 events, and noncontiguous-output rejects
fall from 46 to 37 methods.

That census also exposed an execution-order bug not seen in focused testing:
the one-element subcase of `test_expand` sometimes failed after a long LUT
program. Zeroing the aligned scratch surface on the NPU was a necessary layout
invariant, but it did not fix the ordered failure. The strict census at
`1d6d2781d` remained 101 `PASS_NATIVE`, 40 `PASS_FRONTEND`, 271 `FAIL`, and 13
`SKIP_UPSTREAM` in 552.52 seconds. Its JSON is
`~/rk2608_backups/census-zero-padded-sparse-1d6d2781d/test_ops_coverage.json`
(SHA-256 `56796ba75080fb30791316222dca7db9b6f06fd764df365f5102ba2e8bb932d1`);
the JUnit XML has SHA-256
`ac080cab53659dfc2b58060dfe73656cca900e6d2527370b6ca2052c8e9d19ee`.

The minimal reproducer is `test_exp2_log2_zero_times_negative` immediately
followed by `test_expand`. A read-only scratch trace showed that the NPU zero
task correctly wrote the complete 32-lane scratch tile, but the following
one-element DPU copy overwrote its first atom with `0x3240` followed by seven
stale `0x7c01` NaNs. The generic DPU emitter had hardcoded eight input and
output channels even for a one-element stage. CMAC zero weights cannot suppress
those lanes because IEEE multiplication preserves `0 * NaN` as NaN.

The native fix programs the logical channel count for FP16 stages smaller than
one eight-lane atom. The DPU then copies only the scalar into the NPU-zeroed
scratch surface, leaving its padding defined. The exact two-method reproducer
passes, and the complete strict hardware suite passes 46 tests plus 27 subtests
in 358.10 seconds with `ROCKCHIP_FALLBACK=0`. No host initialization or semantic
fallback is involved. The rejected register-reset experiment and scratch trace
are preserved in `rockchip-upstream-patches/wip-cmac-register-reset-scratch-trace.patch`
(SHA-256 `15d2882943eb0ed7c2193fa626f7202b172a3165fffb1e88ca4939e602f355b2`).

At committed head `7cf01ac95`, the complete uncached strict census reaches 102
`PASS_NATIVE`, 40 `PASS_FRONTEND`, 270 `FAIL`, and 13 `SKIP_UPSTREAM` in 557.23
seconds. Pytest's mixed method/subtest summary is 163 passed, 375 failed, and 13
skipped. The only method-level delta from `1d6d2781d` is `test_expand` moving
from `FAIL` to `PASS_NATIVE`; no prior pass regressed. The 244 typed reject
events remain distributed as 52 output dtype, 50 layout, 46 reduction, 37
reformat, 27 contraction, 25 input dtype, and 7 ALU rejects. The durable JSON is
`~/rk2608_backups/census-subatom-7cf01ac95/test_ops_coverage.json` (SHA-256
`1b02b4206d6a12690cb7e937d58c6a86ad02f8492fda2dc68d902c162c7eb2c3`);
the JUnit XML has SHA-256
`c83baff01bd3e96c6e1b6ce8cff4376f066058f349d41ad6b8db3970a9d7d421`.

The following architecture milestone replaces the fixed CMAC-prefix/DPU/main-
CMAC/CMAC-suffix container with a generic ordered `RKProgram`. Program-scope
scratch is validated once, while typed DPU, CMAC, and PPU steps contribute
commands, constants, and relocations through one composition path. This costs
seven counted compiler lines. Sixty-two compiler/image/native-program tests
pass, Ruff and mypy are clean, and the complete strict hardware suite remains
46 tests plus 27 subtests in 358.01 seconds. No lowering capability or command
semantics changed in this representation milestone.

The next compiler milestone replaces the nested lowering-selection expression
with six named ordered passes. `RKLowerResult` now has explicit `NATIVE`,
`NOT_APPLICABLE`, and `UNSUPPORTED` states. A pass returns not-applicable when
the graph is outside its family—for example, affine reformat no longer claims
a SIN expression merely because its value is not a direct index. Applicable
rejects use a declared specificity order before fingerprinted telemetry is
recorded. Sixty-three compiler/image/native-program tests pass, Ruff and mypy
are clean, and the complete strict hardware suite remains 46 tests plus 27
subtests in 358.94 seconds. This clarity costs 21 counted compiler lines.

The following mechanical milestone replaces the 2,275-physical-line renderer
module with a package organized around enforceable compiler boundaries. The
entry/lowering module is 519 physical lines; typed IR, expression recipes,
affine analysis, UOp-free register emission, and the image codec live in
separate modules. The emitter imports only typed Rockchip plans and generic
`Ops`, never UOps. Frozen command-image tests and all 60 compiler/image tests
remain green, Ruff and mypy are clean, and the complete strict hardware suite
remains 46 tests plus 27 subtests in 358.21 seconds. `sz.py` counts 2,134
compiler lines and 27,267 repository lines after the readability refactor; the
38-line increase is explicit module interfaces, not new hardware behavior.

Dense FP16 global extrema now use one generic ordered plan. DPU MAX reduces
atom-aligned blocks to eight lane maxima, four small CMAC selector tasks place
those lanes in PPU channel zero over a 2x4 surface, and PPU produces the scalar
maximum; input/output sign scales also cover the standard MIN decomposition.
The first hardware experiment tried to finish the tree with sub-atom DPU EW
addresses and returned lane zero (`128` instead of `134` for an increasing
135-element input). Width sweeps proved that EW_BASE_ADDR ignores low atom bits,
so that rejected design was not retained. Hardware boundary cases 2, 8, 9,
and 135 plus scaled MAX and MIN pass with no host semantics. The official FP16
global and scaled extrema subcases advance natively; their parent methods remain
failed until the later int/bool subcases have honest device support. All 66
host compiler/image/fallback/telemetry tests pass, Ruff and mypy are clean, and
the expanded strict hardware suite passes 47 tests plus 31 subtests in 366.90
seconds. The capability adds 62 counted compiler lines.

The complete uncached census at `a27a52212` confirms 102 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 270 `FAIL`, and 13 `SKIP_UPSTREAM` across exactly 425 methods;
pytest reports the same 163 passed, 375 failed, and 13 skipped method/subtest
mix in 565.90 seconds. Method totals do not change because later subcases still
reject, but `test_max` now records three passing native mixed-engine kernels
(two 135-element global/scaled reductions and one four-element reduction), and
`test_min` records its three FP16 global/scaled kernels before reaching the
integer-dtype case. There are no device timeouts. The 244 method-level reject
events are 67 layout, 63 reformat, 58 output dtype, 24 ALU, 21 input dtype, 7
plan-stage limit, and 4 reduction rejects. The durable JSON is
`~/rk2608_backups/census-extrema-a27a52212/test_ops_coverage.json` (SHA-256
`aa3e39c0d8f17139dccbf3f8cfc8f1a22c398cb33b52ef2663a3d6a9dc5473a5`);
JUnit XML SHA-256 is
`35b6f55181155b7172c50db687c39b3275586084670cc8e5840cd20950b980e7`.

Static affine FP16 MAX now batches up to eight logical outputs into PPU
channels. One DPU preparation packs the source, cost-bounded CMAC selector
tasks materialize a reusable HWC8 surface, and one PPU task per batch reduces
the spatial axes. Padding repeats a real reduction element, so it cannot alter
the maximum. The `(3,4,5,6).max(axis=1)` TestOps graph produces 90 outputs as
12 batches (2 DPU, 24 CMAC, and 12 PPU tasks total) and matches RK3588 hardware
exactly. Plans above 128 outputs, 512 inputs, 256 reduction elements, or 8 MiB
of selector constants reject explicitly. All 67 host tests, Ruff, and mypy are
clean; the expanded strict hardware suite passes 48 tests plus 31 subtests in
370.77 seconds. This generic capability adds 73 counted compiler lines.

One static affine FP16 input can now be materialized before a generic DPU
expression. The lowerer proves a dense output map, enumerates the differing
input map, rejects plans above 4,096 outputs, 512 source elements, or 8 MiB of
selector constants, then emits DPU packing, CMAC selector tiles, and the normal
ALU/LUT/mask schedule. It is not test-name-specific and also covers one
permuted operand. The full `test_broadcasted_add`, `test_broadcasted_add_2`,
and `test_broadcast_simple` methods pass natively on their 45x65 shapes in
61.45 seconds, establishing at least 105 native methods before the next full
census. All 68 host tests, Ruff, and mypy are clean; the strict hardware suite
passes 49 tests plus 31 subtests in 371.75 seconds. The capability adds 88
counted compiler lines.

The complete uncached census at `7c11a8f32` confirms 105 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 267 `FAIL`, and 13 `SKIP_UPSTREAM` across exactly 425 methods.
The only method deltas from `a27a52212` are `test_broadcast_simple`,
`test_broadcasted_add`, and `test_broadcasted_add_2` moving from failure to
native pass; no prior native method regresses. Pytest reports 166 passed, 360
failed, 13 skipped, and 12 passing subtests in 751.59 seconds. The 241 remaining
method-level reject events are 64 layout, 60 output dtype, 60 reformat, 22 input
dtype, 17 ALU, 14 plan-stage limit, and 4 reduction rejects. The durable JSON is
`~/rk2608_backups/census-broadcast-7c11a8f32/test_ops_coverage.json` (SHA-256
`415f41a98db200a4723ea8fdeff69453325f7ad67d8b78ab25ed5ff9bb2e66fb`);
JUnit XML SHA-256 is
`45bddd5fdf96f640f066bd06ae82575a89948f8d597db5ef21393cef1fe9d7c6`.

The pre-fix census JSON is
`~/rk2608_backups/census-affine-reformat-52f34b131/test_ops_coverage.json`
(SHA-256 `636e8c745dc5044bba7e055ce23ab5a7764585a9858e4a8a747831a548a789a7`);
its JUnit XML has SHA-256
`2548cbbb7c90ef9afda5541307f4266a2186d36a927e893ac271585ed20abc0e`.

## Current exact method baseline

At `0946a6da1`, after restoring operation-specific int32/FP32 wide constant
fills and adding the global RK3588 test-session lock, the uncached 2026-08-02
census completed without NPU timeouts or reset failures. Parsing the 425 JUnit
method records, rather than subtracting pytest's subtest totals, gives:

| Method status | Count |
|---|---:|
| Fully PASS | 123 |
| At least one FAIL | 289 |
| SKIP | 13 |
| Total | 425 |

Pytest's raw summary is `144 passed, 394 failed, 13 skipped` in 440.30 seconds.
Those are 551 outcomes: 425 collected methods plus 126 additional unittest
subtest outcomes. A method containing subtests contributes a base pass even
when one of its subtests fails, so the raw 144/394 totals are not method
coverage. The legacy JUnit format also does not retain enough subtest identity
to split all 126 outcomes reliably; the kernel/subcase telemetry milestone
will replace it with explicit records.

Using the first failure record for each of the 289 failing methods, the current
primary failure families are:

| Primary failure | Methods |
|---|---:|
| `RKPLAN_REJECT:unsupported_contraction` | 137 |
| `RKPLAN_REJECT:unsupported_dtype` | 97 |
| `RKPLAN_REJECT:unsupported_layout` | 43 |
| `RKPLAN_REJECT:unsupported_op` | 7 |
| Numerical/frontend exception | 3 |
| `OverflowError` | 1 |
| Subprocess failure | 1 |

The exact XML is preserved at
`~/rk2608_backups/research-wide-fill-census-20260802-094110/test_ops.xml`.
The restored wide fills are a fill-only capability and do not claim general
FP32 or integer arithmetic support.

At `e133b0a5b`, the first complete version-1 telemetry census reproduced the
same raw pytest result in 441.61 seconds and classified all 425 methods:

| Coverage outcome | Methods |
|---|---:|
| `PASS_NATIVE` | 83 |
| `PASS_FRONTEND` | 40 |
| `FAIL` | 289 |
| `SKIP_UPSTREAM` | 13 |

All 126 explicitly reported unittest subcases currently fail. The run executed
225 kernels, all successfully through `RK_DPU`, and observed 389 compiler
rejects: 213 contraction, 97 dtype, 72 layout, and 7 operation rejects. The
maximum native image has 64 stages. The durable report is
`~/rk2608_backups/research-full-telemetry-fixed-20260802-101118/test_ops_coverage.json`
(SHA-256 `ca5e3517a10065929d9dd174b680fcdfea2abef8fff2b129278ba0849c560d04`);
the matching JUnit XML has SHA-256
`fb40a6ed08aa4a29d78bad38f9c8b1f3fa284e67a1565448572765679b1e55f8`.

At `1002b1b02`, typed lowering and normalized fingerprints reproduced the same
425-method result in 442.92 seconds. The 389 reject events split into:

| Typed reject kind | Events |
|---|---:|
| `unsupported_reduction` | 149 |
| `unsupported_layout` | 86 |
| `unsupported_output_dtype` | 53 |
| `unsupported_contraction` | 35 |
| `requires_reformat` | 34 |
| `unsupported_input_dtype` | 25 |
| `unsupported_alu` | 7 |

The largest detail families are output/layout legalization (49 methods), MAX
reduction (39), ADD reduction (38), contraction output reformat (26), and
non-direct reduction epilogues (22). There are 358 exact normalized graph
digests; the leading digest covers six related cross-entropy/NLL MAX-reduction
methods. This confirms the native implementation order after hybrid coverage:
layout/reformat, generic reduction, then generalized contraction.

The durable typed report is
`~/rk2608_backups/research-typed-census-20260802-102720/test_ops_coverage.json`
(SHA-256 `e0be6147127d0058ad7fc869b936521e68c3377a163f7b68c5fefbd1239cb528`);
the matching JUnit XML has SHA-256
`47e0f3d37cf8cc4d9954b8c4e95b3ed2ebda03f5b0bd8bbeb9737115a743eba9`.

## Historical pytest-summary censuses

The older tables below predate method-aware JUnit parsing. They are preserved
as milestone evidence, but their PASS/FAIL columns were derived from pytest's
mixed method/subtest summary and must not be read as exact method coverage.

At `aab408cec`, the uncached 2026-08-01 census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 88 |
| FAIL | 324 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 450 failures because 126 failing unittest subtests are counted
in addition to their failed parent methods. Runtime was 65.15 seconds.

At `40c74406c`, after scalar and native wide fills plus the RKImage v2 width
fixes, the uncached 2026-08-02 census again completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 96 |
| FAIL | 316 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 442 failures after adding the same 126 failing subtests. Runtime
was 72.31 seconds. This is an exact eight-method gain from the baseline.

At `fd317872f`, after FP16 extrema canonicalization and composed predicates,
the uncached 2026-08-03 census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 104 |
| FAIL | 308 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 434 failures including the 126 failing subtests. Runtime was
74.26 seconds, another exact eight-method gain.

At `6ddda80b0`, after infinity-safe selection and generated integral rounding,
the uncached 2026-08-02 census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 110 |
| FAIL | 302 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 428 failures including the 126 failing subtests. Runtime was
160.17 seconds. A follow-up parser guard restores clean rejection for unrelated
nonnumeric constants discovered by this census.

At `015e735b1`, after EXP2 special-value preservation and the generated
two-level EXP LUT, the uncached 2026-08-02 census completed without NPU
timeouts:

| Status | Methods |
|---|---:|
| PASS | 112 |
| FAIL | 300 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 426 failures including the 126 failing subtests. Runtime was
204.28 seconds. The two new method passes are `test_exp2` and `test_exp`.

At `9db9a4d33`, after two-level sigmoid, IEEE infinite-numerator division,
refined SQRT/RSQRT, native subtraction, and range-normalized LOG2/LOG10, the
uncached 2026-08-02 census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 120 |
| FAIL | 292 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 418 failures including the 126 failing subtests. Runtime was
229.50 seconds. This is an exact eight-method gain from the preceding census;
the FP32-only `test_log` remains an intentional unsupported-dtype rejection.

At the generated EXPM1/tanh milestone, the uncached 2026-08-02 census again
completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 124 |
| FAIL | 288 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 414 failures including the same 126 failing subtests. Runtime
was 265.33 seconds. This is an exact four-method gain from `9db9a4d33`:
`test_hardsigmoid_extreme`, `test_quick_gelu_extreme`, `test_tanh`, and
`test_tanh_extreme`. EXPM1 materially reduces the CELU/ELU/SELU residuals but
does not yet turn those complete methods into passes.

At the generated inverse-trigonometric milestone, the uncached 2026-08-02
census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 127 |
| FAIL | 285 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 411 failures including the same 126 failing subtests. Runtime
was 299.24 seconds. The exact three-method gain is `test_asin`, `test_acos`,
and `test_atan`; all use generated math assets through the generic LUT stage.

After generic LOG2 multiplication and compact expression-input EXP2 special
value handling, the uncached 2026-08-02 census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 128 |
| FAIL | 284 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 410 failures including the same 126 failing subtests. Runtime
was 317.88 seconds. The exact one-method gain is
`test_exp2_log2_zero_times_negative`; the direct EXP2, LOG2, and LOG10 methods
remain passing.

At the generated inverse-hyperbolic-tangent research milestone, the uncached
2026-08-02 census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 129 |
| FAIL | 283 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 409 failures including the same 126 failing subtests. Runtime
was 327.70 seconds. The exact one-method gain is `test_atanh`; its broad and
edge tables remain research assets and are not part of the minimal upstream
contract.

At the generated multirange ASINH/ACOSH milestone, the uncached 2026-08-02
census again completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 131 |
| FAIL | 281 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 407 failures including the same 126 failing subtests. Runtime
was 358.37 seconds. The exact two-method gain is `test_asinh` and
`test_acosh`; both now replace oversized SQRT-plus-LOG compositions with
generated multirange assets selected by generic ALU and mask stages.

At the generated SINH/COSH milestone, the uncached 2026-08-02 census again
completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 133 |
| FAIL | 279 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 405 failures including the same 126 failing subtests. Runtime
was 372.28 seconds. The exact two-method gain is `test_sinh` and `test_cosh`;
recognition of their shared two-EXP decomposition replaces oversized plans
with generated math assets and generic overflow repair.

At the generated two-level ERF milestone, the uncached 2026-08-02 census
completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 134 |
| FAIL | 278 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 404 failures including the same 126 failing subtests. Runtime
was 388.00 seconds. The exact one-method gain is `test_erf`; its broad Q15 and
local Q16 generated assets plus a generic near-zero polynomial fit in 50 typed
stages and execute entirely on the NPU.

Hardware jobs must remain serial. A long-running pytest invocation may return
a terminal session id before it finishes; poll that same session rather than
starting another device command. Overlapping two RKNN submitters produced a
temporary cascade of `EINVAL` submits and repeated kernel soft resets during
this milestone. Killing the accidentally overlapping process restored normal
operation without a reboot. This is a test-orchestration failure signature,
not a numerical or compiler failure.

At the generated Softplus/LogSigmoid milestone, the uncached 2026-08-02
census completed without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 136 |
| FAIL | 276 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 402 failures including the same 126 failing subtests. Runtime
was 395.69 seconds. The exact two-method gain is `test_softplus` and
`test_logsigmoid`. The Softplus method includes beta 1, 3, and 1/3, positive
and negative magnitude-300 tails, and a scalar case. Beta 3 uses two generated
LUT tasks over the original input so the NPU does not expose an intermediate
FP16 `3*x` rounding point. Mish now lowers to 61 typed stages but remains a
strict numerical mismatch and is still counted as FAIL.

At the generated Mish milestone, the uncached 2026-08-02 census completed
without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 137 |
| FAIL | 275 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 401 failures including the same 126 failing subtests. Runtime
was 394.30 seconds. The exact one-method gain is `test_mish`; the generated
broad and midrange LUT assets plus a local polynomial replace the 61-stage
Softplus/Tanh composition with 34 generic typed stages and meet the strict
official comparison without host semantic work.

At the generated Hardswish milestone, the uncached 2026-08-02 census completed
without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 138 |
| FAIL | 274 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 400 failures including the same 126 failing subtests. Runtime
was 398.61 seconds. The exact one-method gain is `test_hardswish`; one generated
broad LUT plus generic local and positive-tail arithmetic replaces the
rounding-sensitive direct decomposition in 47 typed stages, entirely on the
NPU.

At the generated QuickGELU milestone, the uncached 2026-08-02 census completed
without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 139 |
| FAIL | 273 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 399 failures including the same 126 failing subtests. Runtime
was 410.51 seconds. The exact one-method gain is `test_quick_gelu`; two
generated ranges and a generic near-zero series meet the ordinary comparison
while retaining the already-passing magnitude-300 extreme method. The plan is
63 typed stages and performs no host semantic work.

At the generated GELU milestone, the uncached 2026-08-02 census completed
without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 141 |
| FAIL | 271 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 397 failures including the same 126 failing subtests. Runtime
was 434.68 seconds. The exact two-method gain is `test_gelu` and
`test_gelu_extreme`; both the tanh approximation and exact decomposition lower
to the same generic stage recipe with generated data selected by LUT identity.
Each plan has 53 typed stages, retains the ordinary TestOps tolerance, and
performs no host semantic work.

At the generated ELU-family milestone, the uncached 2026-08-02 census completed
without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 143 |
| FAIL | 269 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 395 failures including the same 126 failing subtests. Runtime
was 434.86 seconds. The exact two-method gain is `test_elu` and `test_selu`;
ELU alpha 1, ELU alpha 0.1, and SELU share one parameterized 35-stage recipe
and differ only by generated broad/local data identities. CELU remains a
numerical mismatch and is not claimed by this milestone.

At the generated CELU milestone, the uncached 2026-08-02 census completed
without NPU timeouts:

| Status | Methods |
|---|---:|
| PASS | 144 |
| FAIL | 268 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 394 failures including the same 126 failing subtests. Runtime
was 441.47 seconds. The exact one-method gain is `test_celu`, including all
matrix and scalar subcases for integer alpha 1 through 4. Alpha 1 reuses ELU;
alpha 2 through 4 share one generic typed recipe selected by six generated
payload identities.

At the FP16-contract cleanup milestone, the uncached 2026-08-02 census was:

| Status | Methods |
|---|---:|
| PASS | 140 |
| FAIL | 272 |
| SKIP | 13 |
| Total | 425 |

Pytest reports 398 failures including the same 126 failing subtests. Runtime
was 435.19 seconds, with no NPU timeout. Removing int32 and FP32 constant-fill
claims deliberately gives back `test_full`, `test_full_like`, `test_ones_like`,
and `test_zeros_like`; those methods mix dtypes outside the declared FP16
contract. `RKALUStage.out_dtype`, wide WDMA emission, and the extra advertised
dtypes are gone. The full census remains informational rather than the
hardware contract.

Two post-CELU probes were rejected without being committed:

- FP16 SIN/COS LUTs pass a focused half-precision sweep, but the official
  methods include FP32 inputs or scalar dtype checks. They provide no honest
  method gain without the prohibited runtime FP32 narrowing ABI. The exact WIP
  is preserved under `~/tinygrad/rockchip-upstream-patches/` as
  `wip-fp16-sin-cos-no-census-gain.patch` with SHA-256
  `ebd8016663393810c3dc87d804cc9d5d366303f2596be806489893f4a6522908`.
- Replaying a suffix vector with relocation addends makes the first 64 values
  of a 65-value row correct, then fails because DPU WDMA writes 16-byte atoms:
  65 FP16 lanes occupy a 72-lane physical write and the next 130-byte row base
  is unaligned. Native broadcast therefore needs an explicitly padded device
  layout or another engine, not host expansion. The exact WIP is preserved in
  the same directory as `wip-affine-suffix-broadcast-alignment-failure.patch`
  with SHA-256
  `12c2532b4ff92d16e71cef4c4e86b817a31d27f42be3c8dddc21f6679dab48ce`.

The general-GEMM reference in `allbilly/rk3588/conv_grok/gemm_npu.py` also
confirms that ordinary row-major operands must be packed and padded before the
CMAC task. Its NumPy host packers are useful as a layout oracle but cannot be
ported into the thin runtime. The current direct CMAC contract therefore stays
limited to already legal packed surfaces until device-native layout conversion
exists.

The following diagnostic-only milestone keeps the legacy 140/272/13 pytest
tally unchanged
but splits every pre-submission rejection into `unsupported_dtype`,
`unsupported_layout`, `unsupported_contraction`, or `unsupported_op`. Focused
compiler tests cover all four classes. This makes the informational census
useful without treating honest hardware exclusions as numerical or device
failures.

## Milestones after the baseline

| Capability | Focused official gain | Full census folded in? |
|---|---:|---|
| Rank-0 FP16 constant fills | `test_ones`, `test_zeros` | Yes (`40c74406c`) |
| Native tiled int32/FP32 constant fills (research-only, restored) | `test_full`, `test_full_like`, `test_ones_like`, `test_zeros_like` | Current (`52c6657e2`) |
| Global FP16 CMAC sums, scaled mean, and ReLU-sum | `test_sum_simple`, `test_sum_full`, `test_mean`, `test_sum_relu` | Yes (`da09c1fd9`) |
| Tiled sparse-CMAC affine FP16 reductions | `test_sum`, `test_sum_tiny`, `test_mean_axis` | Yes (`da09c1fd9`) |
| Static affine DPU/CMAC selector reformat | transpose, permute, flip, expand, stack-slice, unfold, strided/double slice, diagonal | Yes (`7cf01ac95`) |
| Scalar multi-axis affine MAX through CMAC/PPU | `test_max_pool2d_simple` | Yes (`6133442c2`) |
| Masked/windowed affine MAX through CMAC/PPU | ceil-mode, bigger-stride, dilation, and padding pool methods | Yes (`6133442c2`) |
| FP16 absolute value and finite ordered extrema | `test_abs`, `test_abs_exact`, exact ReLU variants, `test_clip` | Yes (`fd317872f`) |
| Composed FP16 predicates used inside arithmetic | `test_sign`, `test_sign_exact` | Yes (`fd317872f`) |
| Infinity-safe ordered threshold selection | `test_inf_where` | Yes (`6ddda80b0`) |
| Generated algorithm-23 round-to-nearest-even LUT | `test_round` | Yes (`6ddda80b0`) |
| Integral rounding composed from the roundoff LUT | `test_trunc`, `test_floor`, `test_ceil` | Yes (`6ddda80b0`) |
| IEEE special-value wrapper around EXP2 LUT | `test_exp2` | Yes (`015e735b1`) |
| Generated two-level EXP LUT with signed-factor recognition | `test_exp` | Yes (`015e735b1`) |
| Generated two-level sigmoid reused by generic MUL | `test_sigmoid`, `test_silu`, `test_swish` | Yes (`9db9a4d33`) |
| Mask-composed sign preservation for infinite division numerators | `test_div_naninf` | Yes (`9db9a4d33`) |
| Generated SQRT seed with three generic Newton refinements | `test_sqrt` | Yes (`9db9a4d33`) |
| Range-scaled generated RSQRT seed with generic Newton correction | `test_rsqrt` | Yes (`9db9a4d33`) |
| Range-normalized generated logarithm tables | `test_log2`, `test_log10` | Yes (`9db9a4d33`) |
| Stable generic clip for ReLU-difference saturation | `test_hardsigmoid_extreme` | Yes |
| Scaled-input reuse of the generated sigmoid LUT | `test_quick_gelu_extreme` | Yes |
| Generated tanh ranges with a stable local polynomial | `test_tanh`, `test_tanh_extreme` | Yes |
| Generated inverse-trigonometric tables and local/tail arithmetic | `test_asin`, `test_acos`, `test_atan` | Yes |
| Generic dynamic LOG2 multiplication and compact nested EXP2 special values | `test_exp2_log2_zero_times_negative` | Yes |
| Generated ATANH broad/edge tables with generic local arithmetic | `test_atanh` | Yes (research branch only) |
| Generated multirange inverse-hyperbolic tables and local arithmetic | `test_asinh`, `test_acosh` | Yes (research branch only) |
| Generated hyperbolic tables and generic overflow repair | `test_sinh`, `test_cosh` | Yes (research branch only) |
| Generated two-level ERF tables and generic local polynomial | `test_erf` | Yes (research branch only) |
| Generated Softplus tables reused by LogSigmoid | `test_softplus`, `test_logsigmoid` | Yes (research branch only) |
| Generated Mish ranges with generic local arithmetic | `test_mish` | Yes (research branch only) |
| Generated Hardswish broad range with generic local series | `test_hardswish` | Yes (research branch only) |
| Generated QuickGELU ranges with generic local series | `test_quick_gelu` | Yes (research branch only) |
| Generated tanh-approximate and exact GELU ranges | `test_gelu`, `test_gelu_extreme` | Yes (research branch only) |
| Parameterized generated ELU/SELU ranges | `test_elu`, `test_selu` | Yes (research branch only) |
| Parameterized generated CELU alpha 1–4 ranges | `test_celu` | Yes (research branch only) |
| Generated two-level POW8 range reduction | `test_pow_const` exponent-8 subcase; method still fails at exponent 5.5 | Not yet |
| Generated multirange positive POW5.5 | `test_pow_const` positive-5.5 subcase; method still fails at negative 5.5 | Not yet |
| Shifted multirange negative POW5.5 | `test_pow_const` negative-5.5 subcase; method still fails at constant-base 5.5 | Not yet |
| Split-range constant-base POW5.5 | `test_pow_const` constant-base 5.5 subcase; method still fails at negative base | Not yet |
| Native roundoff parity for negative-base POW5.5 | `test_pow_const` negative-base 5.5 subcase; method still fails at constant-base 8 | Not yet |
| Four generated Q15 bands for constant-base POW8 | `test_pow_const` constant-base 8 subcase; method still fails at zero-base float semantics | Not yet |

The historical wide-fill milestone wrote the requested dtype directly through
DPU WDMA; it did not use runtime narrowing or host semantic work. Its RKImage
dependency and relocation-width fixes remain useful, while the dtype-specific
fill path is no longer part of the declared backend contract.

The DPU MAX operation is not yet claimed IEEE-exact for all NaN operand
orders. Hardware testing showed that finite extrema are correct and FP16
absolute value preserves the expected infinity, signed-zero, and NaN results;
special-value MAX behavior remains outside this milestone.

The canonical DPU expression scheduler is now shared by contiguous and affine-
broadcast lowering. Topological ordering, use counts, scratch liveness, and
in-place reuse live behind one typed, UOp-free scheduling boundary rather than
two lowering-specific copies. This refactor removes 17 counted lines (27,473
total; 2,340 in the Rockchip renderer/compiler), with no coverage claim or
semantic change. Ruff, mypy, all 68 host compiler/image/telemetry tests, and
the strict RK3588 suite (49 tests plus 31 subtests) pass; the hardware suite
completed in 372.01 seconds with `ROCKCHIP_FALLBACK=0`.

Static pointwise FP16 expressions can now feed affine ADD reductions without a
host materialization. The shared DPU scheduler writes the expression to NPU
scratch, and the existing cost-bounded sparse CMAC selector reduces that
surface, including scalar outputs. Every input must have the same proven
affine map and extent; convolution's differently indexed operands therefore
continue to reject for the contraction lowerer. The complete four-subcase
`test_binary_crossentropy` method passes natively at its unchanged tolerance.
The `reduction="none"` BCE form still fails its existing LUT accuracy contract,
and positional weights still require a separate broadcast capability; neither
was hidden with a tolerance or fallback. Ruff, mypy, all 69 host tests, and the
strict RK3588 suite (50 tests plus 31 subtests) pass, the latter in 372.76
seconds. This capability adds 16 counted compiler lines (27,489 total; 2,356
in the Rockchip renderer/compiler) before the next full census.

The complete uncached census at `181dfc75a` confirms 109 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 263 `FAIL`, and 13 `SKIP_UPSTREAM` across exactly 425 methods.
The four exact deltas from `7c11a8f32` are `test_binary_crossentropy`,
`test_einsum_trace`, `test_simple_grouped_conv2d`, and
`test_sum_collapse_neg` moving from failure to native pass; no prior native
method regresses. Pytest reports 170 passed, 356 failed, 13 skipped, and 12
passing subtests in 850.41 seconds. There are no device timeouts or invalid
submissions. Of the 263 remaining failed methods, 257 have a typed native
reject: 68 layout, 60 output-dtype, 59 reformat, 24 ALU, 22 input-dtype, 20
plan-limit, and 4 reduction classifications. The six numerical/frontend
failures without a reject are BCE reductions, exact copysign, lerp, maximum,
zero-stride multiply-accumulate, and `pow_const`; they remain failures at the
official tolerances. The durable JSON is
`~/rk2608_backups/census-pointwise-181dfc75a/test_ops_coverage.json` (SHA-256
`31afa3697c726182bc7759b02051357e7701a8b88226a96086f3163f146446d5`);
JUnit XML SHA-256 is
`e66904bf774118f5e783fbc02a2adb937f44c8d9ad884475ae0430120de0b3f9`.

Bounded affine FP16 contractions now have a fully native dynamic pack/compute/
unpack path. Static selector CMAC tasks pack both argument surfaces into padded
activation and CMAC-weight layouts, a generalized multi-row CMAC task computes
M×N×K, and selector tasks compact its physical output into the graph's affine
destination. The implementation groups affine operand sequences into matrix
rows and columns rather than matching matmul test names. It currently accepts
M,N≤16, K≤64, source surfaces≤512 elements, and at most 8 MiB of generated
selector constants. Hardware established that useful FP16 CMAC rows retain a
128-byte physical stride (`align_out*4`), matching `conv_grok`; this is encoded
in the target layout and scratch allocation. Dynamic 4×4, 8×8, and 9×9 probes
pass at unchanged tolerances, as do complete `test_small_gemm`, `test_9_gemm`,
and `test_matmul_simple` methods. An 8×8 plan is intentionally a correctness
fallback—91 NPU tasks, 460,032 constant bytes, and 4,864 scratch bytes—so a
future cost model can prefer direct CNA/DMA packing without changing the
affine contraction IR. Ruff, mypy, all 70 host tests, and the strict RK3588
suite (51 tests plus 34 subtests) pass; the latter completed in 410.15 seconds.
The capability adds 80 counted compiler lines (27,569 total; 2,436 in the
Rockchip renderer/compiler) before the next full census.

The complete uncached census at `e18ce14c2` confirms 116 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 256 `FAIL`, and 13 `SKIP_UPSTREAM` across exactly 425 methods.
The seven exact deltas from `181dfc75a` are `test_9_gemm`,
`test_matmul_batched`, `test_matmul_batched_vector`, `test_matmul_simple`,
`test_small_gemm`, `test_strided_conv1d_simple`, and
`test_strided_conv2d_simple` moving from failure to native pass. The latter two
are generic affine-contraction matches; there is no convolution test-name
logic. No prior native method regresses. Pytest reports 177 passed, 346 failed,
13 skipped, and 15 passing subtests in 1,024.16 seconds. There are no device
timeouts or invalid submissions. Of the 256 remaining failed methods, 250
have a typed native reject: 69 layout, 60 output-dtype, 41 reformat, 30
plan-limit, 24 ALU, 22 input-dtype, and 4 reduction classifications. The
durable JSON is
`~/rk2608_backups/census-contract-e18ce14c2/test_ops_coverage.json` (SHA-256
`8e60df220935360a3755299cda4deff69c39dbe35c88fcad854eb7e0ad60ef54`);
JUnit XML SHA-256 is
`a0e133efd99abee2a2cd2768e999e6d8942f009ed3b8b592857d7286bbbfab65`.

Static coordinate predicates in affine FP16 reformats now lower to the same
bounded sparse-CMAC selector used by generic movement. The compiler evaluates
only a strict whitelist of scalar coordinate operations over known `RANGE`
values; it never evaluates tensor loads. False selector rows become native
zero rows, which covers the FP16 subcases of `tril` and `triu` for arbitrary
tested diagonals without a named-operation path. Explicit boolean inputs still
reject on their unsupported dtype, and `padding_add` still requires a separate
two-source affine composition capability, so this focused milestone makes no
new method-level census claim. Ruff and mypy pass, as do all 71 host tests and
the strict RK3588 suite (52 tests plus 37 subtests) in 411.96 seconds with
`ROCKCHIP_FALLBACK=0`. The change brings the research tree to 27,601 counted
lines, including 2,468 lines in the Rockchip renderer/compiler.

Zero-masked affine FP16 input surfaces can now be materialized before generic
DPU arithmetic. The lowerer enumerates only proven static coordinate maps,
groups 16-output selector tiles into bounded source windows, initializes NPU
scratch, and emits ordered DPU/CMAC tasks; it does not recognize padding by
name or evaluate tensor data on the host. Complete `test_padding_add` passes
natively at its unchanged tolerance in 41.74 seconds. Its 64x64 correctness
plan uses 362 NPU tasks, 1,002,752 constant bytes, and 8,352 scratch bytes, so
it remains an explicit cost-model target for a future CNA/DMA reformat path.
Ruff and mypy pass, as do all 70 host compiler/image/telemetry tests and the
strict RK3588 suite (53 tests plus 37 subtests) in 411.41 seconds with
`ROCKCHIP_FALLBACK=0`. The research tree now has 27,639 counted lines,
including 2,506 in the Rockchip renderer/compiler.

The complete uncached census at `d2af3b3d9` confirms 117 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 255 `FAIL`, and 13 `SKIP_UPSTREAM` across exactly 425 methods.
`test_padding_add` is the sole method transition from the `e18ce14c2` census;
no prior native method regresses. Pytest reports 178 passed, 345 failed, 13
skipped, and 15 passing subtests in 1,090.91 seconds. There are no device
timeouts or invalid submissions. First-reject classifications for the 255
failed methods are 66 unsupported-layout, 62 unsupported-output-dtype, 41
requires-reformat, 30 plan-limit, 24 unsupported-ALU, 22 unsupported-input-
dtype, 4 unsupported-reduction, and 6 numerical/frontend failures without a
native reject. The largest exact family remains 44 convolution-like graphs
whose affine CMAC input is not one FP16 pointwise surface. The durable JSON is
`~/rk2608_backups/census-masked-d2af3b3d9/test_ops_coverage.json` (SHA-256
`abf844358296731eee23b9aae1cea6eba7f1497ecca327de68a19c369caea78c`);
JUnit XML SHA-256 is
`c10b1b60eebf6564f1e8c8c87f843eb65b449eb6bde957c841b15f46e26b1c74`.

Affine contractions now flatten any number of static reduction axes into K,
and the proven CMAC row path accepts bounded taller M surfaces. This is generic
contraction analysis: a 2D convolution becomes M output positions, N output
channels, and K as the product of input-channel and kernel axes. All six
subcases of `test_conv2d_bs_1_cin_1` pass natively at unchanged tolerances in
76.68 seconds. An attempted 4-channel 3x3 case exposed that an 8.34 MiB single
constant BO cannot be mapped on this RK3588 runtime; it now rejects before
submission. The shared sparse-plan ceiling is therefore 2 MiB, consistent
with `conv_grok`'s bounded GEMM buffers and above the largest previously proven
native image (about 1.0 MiB). Ruff and mypy pass, as do all 71 host tests and
the strict RK3588 suite (54 tests plus 37 subtests) in 431.79 seconds with
`ROCKCHIP_FALLBACK=0`. This focused complete-method gain is not yet folded into
the 117-pass census. The research tree now has 27,641 counted lines, including
2,508 in the Rockchip renderer/compiler.

Zero-masked affine contraction operands now use the same compile-time
coordinate proof as masked movement: an out-of-bounds point becomes an empty
hardware selector row, while every selected point must resolve to a valid GEM
buffer index. This is not a convolution recognizer and no tensor value is read
by the compiler or runtime. Complete `test_asymmetric_padding_conv1d` passes
natively, including all three official padding subtests, in 55.42 seconds at
unchanged tolerances. The larger `test_simple_padding_conv1d` graph remains a
typed plan-limit reject because its selectors would require 6.01 MiB, above
the proven 2 MiB constant-BO ceiling. Ruff and mypy pass, as do all 72 host
tests and the strict RK3588 suite (55 tests plus 37 subtests) in 448.19 seconds
with `ROCKCHIP_FALLBACK=0`. This focused complete-method gain is not yet folded
into the 117-pass census. The research tree now has 27,660 counted lines,
including 2,527 in the Rockchip renderer/compiler.

The first post-contraction census attempt exposed an unbounded compiler-cost
path and is intentionally not recorded as coverage: `test_large_input_conv2d`
spent the 300-second watchdog enumerating affine coordinates before reaching
existing surface limits, aborting pytest at 29%. Tiled contraction now checks
K, source extents, and a 65,536-point affine-visit budget before enumeration.
The same graph rejects precisely (`out=90720,lhs=262144,rhs=960,K=160`) in
3.67 seconds without submitting a device task. Accepted plans are unchanged.
Ruff and mypy pass, as do all 72 host tests and the strict RK3588 suite (55
tests plus 37 subtests) in 448.85 seconds with `ROCKCHIP_FALLBACK=0`. The
research tree now has 27,664 counted lines, including 2,531 in the Rockchip
renderer/compiler. A complete census must be restarted from this bounded head.

The complete uncached census at `ff4a13dd7` confirms 122 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 250 `FAIL`, and 13 `SKIP_UPSTREAM` across exactly 425 methods.
Five methods move from failure to native pass: `test_asymmetric_padding_conv1d`,
`test_asymmetric_padding_conv2d`, `test_conv2d_bs_1_cin_1`,
`test_negative_padding_conv2d`, and `test_small_gemm_padded`; no prior native
method regresses. Pytest reports 180 passed, 324 failed, 13 skipped, and 34
passing subtests in 1,374.25 seconds. There are no device timeouts, allocation
failures, or invalid submissions. The former 44-instance exact reject
`affine CMAC input is not one FP16 pointwise surface` falls to four. First-
reject classifications for failed methods are now 62 unsupported-output-dtype,
51 plan-limit, 44 unsupported-layout, 37 requires-reformat, 24 unsupported-ALU,
22 unsupported-input-dtype, 4 unsupported-reduction, and 6 numerical/frontend
failures without a native reject. The most expensive passing kernels remain
correctness fallbacks: 364 stages/0.85 MiB/38.9 seconds for
`test_broadcasted_add_2` and 362 stages/1.00 MiB/38.7 seconds for
`test_padding_add`. The durable JSON is
`~/rk2608_backups/census-contract-bounded-ff4a13dd7/test_ops_coverage.json`
(SHA-256 `f2dc9a3c75da906fbd1365ab835b1161fdc1009455c37b6f6fffae10e2cd66b3`);
JUnit XML SHA-256 is
`429581e362977ac1983c07f328ce4757f9d4736d4edb13f3997c05bf1886cafe`.

Static affine MAX can now produce one scalar output when the graph has more
than one reduction axis. The compiler enumerates the bounded affine selector,
packs the selected values into NPU scratch with CMAC, and uses PPU spatial MAX;
it does not recognize pooling by name. Complete `test_max_pool2d_simple` passes
natively at its unchanged exact comparison. A single-axis scalar MAX remains
on the established padded DPU-tree path, preventing sparse-CMAC interception
of dense global reductions; count 135 is covered by the full hardware suite.
Ruff and mypy pass, as do all 75 host compiler/image/telemetry/fallback tests
and the strict RK3588 suite (56 tests plus 37 subtests) in 448.91 seconds with
`ROCKCHIP_FALLBACK=0`. This focused complete-method gain is not yet folded into
the 122-pass census. Counted source remains 27,664 lines, including 2,531 in
the Rockchip renderer/compiler.

Large affine ADD reductions can now be split into greedy, atom-aligned source
windows instead of generating one dense selector over the complete input.
Each window spans at most 512 FP16 values, produces at most 16 outputs, and is
materialized by NPU zero/copy plus CMAC. Identical immutable CMAC payloads are
deduplicated when ordered images are composed. The planner accepts only scales
that are exactly representable in FP16, at most 65,536 affine visits, 2 MiB of
unique constants, and 400 hardware stages. A 2x2 average over a 1,232-element
surface is hardware-conformant with 54 tasks and exact FP16 output; this is a
generic reduction capability but does not yet make a new complete TestOps
method pass. Ruff and mypy pass, as do all 76 host tests and the strict RK3588
suite (57 tests plus 37 subtests) in 455.32 seconds.

Three rejected probes define the current boundary. A two-term FP16 expansion
for 1/6 reduced average-pool mismatches from roughly 35% to 7.7%, but CMAC
accumulation order still differed by one ULP, so non-exact scales reject. A
direct K=640 CMAC global-average plan wrote only its first output and is outside
the proven K<=512 contract. Resetting once per ordered program made ordinary
DPU/CMAC/PPU probes pass but caused the sensitive LUT-to-CMAC sequence to time
out; reset-per-stage remains required. The exact experiments are preserved as
`wip-two-term-cmac-average-one-ulp.patch` and
`wip-dynamic-k320-row-cmac-timeout.patch` in the persistent patch archive.
The research tree now has 27,709 counted lines, including 2,576 in the
Rockchip renderer/compiler.

Static affine MAX now follows compile-time `WHERE` predicates around one FP16
input surface. Coordinates that select padding are represented by an NPU-filled
negative-infinity sentinel lane; selected coordinates must still resolve to a
valid affine GEM index. This is a generic masked-reduction rule, not a pooling
name or shape recognizer. Complete `test_max_pool2d_ceil_mode` and
`test_max_pool2d_ceil_mode_output_size_reduce_by_one` pass natively, including
all three official kernel-size subtests, at exact comparison and unchanged
tolerances. The strict hardware suite passes 58 tests plus 37 subtests in
457.09 seconds with `ROCKCHIP_FALLBACK=0`. Together with the earlier focused
`test_max_pool2d_simple` gain, the inferred native method total is 125 after
the last 122-pass census; these three methods still require a complete census
before becoming an authoritative total. The research tree now has 27,731
counted lines, including 2,598 in the Rockchip renderer/compiler.

Affine MAX surfaces larger than the direct 512-input/128-output pack now split
each eight-output PPU batch into its own atom-aligned source window. Every
window is independently NPU-zeroed and copied, uses at most 512 FP16 lanes,
then feeds the same CMAC selector and PPU MAX stages. The compiler counts the
emitted DPU/CMAC/PPU stages and unique immutable payloads before acceptance;
the existing 400-stage, 65,536-visit, and 2 MiB limits remain hard rejects.
Complete `test_max_pool2d_bigger_stride` and
`test_max_pool2d_bigger_stride_dilation` pass natively with all nine official
subtests at exact comparison. The strict hardware suite passes 59 tests plus
37 subtests in 466.16 seconds with `ROCKCHIP_FALLBACK=0`. The inferred native
method total is 127 after the last 122-pass census, pending a new complete
census. The research tree now has 27,748 counted lines, including 2,615 in the
Rockchip renderer/compiler.

Consecutive affine-MAX output batches now greedily share one source pack while
their combined physical span stays within a 192-lane cost target; a single
batch may still use the proven 512-lane hardware ceiling. This trades a modest
increase in immutable selector size for far fewer DPU zero/copy tasks. All nine
`test_max_pool2d_padding` combinations become native: accepted plans contain
246–336 stages and 0.39–1.44 MiB of unique constants. The first 128-lane
configuration left one plan at 400 stages; running the complete method aborted
inside the driver's reset ioctl after several long submissions. That
configuration was rejected. The 192-lane configuration passes the complete
method in one process in 288.25 seconds, then passes the strict hardware suite
(59 tests plus 37 subtests) in 473.36 seconds without a timeout or reset error.
The inferred native method total is 128 after the last 122-pass census, pending
a new complete census. The research tree now has 27,760 counted lines,
including 2,627 in the Rockchip renderer/compiler.

The complete uncached census at `6133442c2` confirms 128 `PASS_NATIVE`, 40
`PASS_FRONTEND`, 244 `FAIL`, and 13 `SKIP_UPSTREAM` across exactly 425 methods.
The six focused MAX gains are the only method transitions from `ff4a13dd7`:
`test_max_pool2d_simple`, both ceil-mode methods, bigger-stride,
bigger-stride+dilation, and padding; no prior native method regresses. Pytest's
raw accounting is 182 passed, 299 failed, 13 skipped, and 57 passing subtests
in 1,963.04 seconds. There are no NPU timeouts, reset failures, invalid
submissions, or process aborts.

First-reject classifications for the 244 failed methods are 64 unsupported-
output-dtype, 49 plan-limit, 41 unsupported-layout, 32 requires-reformat, 25
unsupported-ALU, 22 unsupported-input-dtype, and 4 unsupported-reduction. The
seven failures without a native reject are numerical or frontend failures:
binary-crossentropy reductions, exact copysign, lerp, maximum, zero-stride
mulacc, constant power, and sum-collapse. The largest passing MAX plans are
336 stages/1.38 MiB/35.87 seconds for padding and 244 stages/0.99 MiB/26.07
seconds for bigger-stride. The durable JSON is
`~/rk2608_backups/census-affine-max-6133442c2/test_ops_coverage.json`
(SHA-256 `b06e60965a0851c89788e187c18845f53a0bce5650d9a81535a02bdfaf69abae`);
JUnit XML SHA-256 is
`eab5ba761799359a29202220355545cf81da5201bdb5b14364b2267e67e22caa`.

Unmasked affine-MAX CMAC groups now address atom-aligned subviews of a shared
NPU-packed scratch window. Each selector therefore uses its compact local K
span even when several output groups share one larger input copy; masked
groups retain the established common negative-infinity sentinel layout. This
is a target-plan address/layout optimization, not a pooling recognizer. All
four `test_max_pool2d_smaller_stride` subtests pass natively at exact
comparison. The two formerly rejected plans shrink from 292 stages/2.72 MiB
and 402 stages/3.45 MiB to 274 stages/1.78 MiB and 386 stages/1.97 MiB. All 80
host tests, Ruff, and mypy pass, as does the strict hardware suite (59 tests
plus 37 subtests) in 472.86 seconds with `ROCKCHIP_FALLBACK=0`. The inferred
native total is 129 after the current 128-pass census, pending the next full
census. The research tree now has 27,765 counted lines, including 2,632 in the
Rockchip renderer/compiler.

An affine-MAX output atom whose eight logical results span more than the proven
K512 source window now lowers through scalar PPU atoms. Each logical result is
packed and reduced independently on DPU/CMAC/PPU, stored in one aligned scratch
atom, then the established sparse-CMAC reformatter gathers lane zero from all
atoms into dense output. This is a generic target-plan fallback for wide
source topology; it does not inspect dilation or pooling names. Complete
`test_max_pool2d_dilation` passes natively at exact comparison. Its four plans
contain 48–195 stages and at most 1.0 MiB of unique constants. All 81 host
tests, Ruff, and mypy pass, as does the strict hardware suite (60 tests plus
37 subtests) in 494.88 seconds with `ROCKCHIP_FALLBACK=0`. The inferred native
total is 130 after the current 128-pass census. The research tree now has
27,817 counted lines, including 2,684 in the Rockchip renderer/compiler.

Sparse affine contractions now accept block-diagonal output/input pair sets and
charge the completed target program for actual deduplicated constant payloads,
rather than rejecting against a theoretical non-deduplicated Cartesian cost.
The NPU may calculate the harmless Cartesian superset, while the final static
selector materializes only outputs proved by the affine graph. This makes all
seven `test_conv2d` subtests native without a convolution-name path or CPU
packing. The grouped plan uses 167 NPU stages, 1,517,904 constant bytes, and
16,896 scratch bytes; the larger dense plan uses 175 stages, 729,792 constant
bytes, and 8,448 scratch bytes.

The first dense hardware probe timed out near stage 166 while packing the RHS
selector. Its 2,048 logical FP16 lanes had allocated exactly 4,096 bytes, but
each 16-lane logical CMAC tile physically writes 32 lanes. The final tile
therefore needs one additional 16-lane tail. Explicitly allocating 4,128 bytes
fixes the timeout; the compiler test records this physical requirement. The
temporary per-stage runtime diagnostic is preserved as
`rockchip-upstream-patches/wip-stage-trace-20260803-010650.patch` (SHA-256
`99489edee4309361f3427b1f70ca290db4dc3a527a0f749260c283451fd6a93e`) and is
not present in the production runtime. All 80 host tests, Ruff, and mypy pass,
as does the strict hardware suite (61 tests plus 39 subtests) in 532.62 seconds
with `ROCKCHIP_FALLBACK=0`. The inferred native total is 131 after the current
128-pass census. The research tree now has 27,818 counted lines, including
2,685 in the Rockchip renderer/compiler.

Selector planning now compares typed stage, unique-constant, and scratch costs
for full sparse CMAC and bounded source-window candidates. Fully empty static
selector tiles are represented by one NPU zero-fill and skipped rather than
consuming CMAC payloads. Nonfinal window boundaries must end on an eight-FP16
destination atom; the first hardware candidate violated this rule at output
element eight and shifted the remainder of the channel, so it was rejected.

When no direct candidate fits, a generic two-level selector first gathers
source-local groups into padded atom-aligned NPU scratch, then compacts that
scratch into dense destination order with a second set of windowed CMAC tasks.
There is no CPU packing or tensor-semantic path. This turns
`test_simple_padding_conv1d` native: its plan contains 395 stages, 856,464
constant bytes, and 15,072 scratch bytes, versus the former 5,236,736-byte
reject. The same planner reduces the large `test_conv2d` regression from 175
stages/729,792 bytes to 100 stages/407,344 bytes; all seven official subtests
pass in 72.17 seconds.

All 81 host tests, Ruff, and mypy pass, as does the strict hardware suite (62
tests plus 39 subtests) in 522.05 seconds with `ROCKCHIP_FALLBACK=0`. The
inferred native total is 132 after the current 128-pass census. The research
tree now has 27,871 counted lines, including 2,738 in the Rockchip
renderer/compiler. The pre-change compiler and tests are preserved under
`~/rk2608_backups/windowed-empty-selectors-before-16a2cbb7e-20260803-012547`.

The affine contraction M bound now follows the existing physical resource
contract: `M<=128` and `M*aligned_K<=4096`, instead of an independent M64
software cap. The unchanged 400-stage and 2 MiB completed-plan limits remain
authoritative, so an M128 probe still rejects on cost while M129 rejects on
surface size. The M81/K4/N4 plan is legal at 374 stages, 2,005,248 constant
bytes, and 20,192 scratch bytes. Complete `test_simple_conv2d_1x1` passes
natively in 44.43 seconds at unchanged tolerance.

All 82 host tests, Ruff, and mypy pass, as does the strict hardware suite (63
tests plus 39 subtests) in 565.46 seconds with `ROCKCHIP_FALLBACK=0`. The
inferred native total is 133 after the current 128-pass census. This capability
changes one counted condition only, so the research tree remains at 27,871
counted lines with 2,738 in the Rockchip renderer/compiler. The pre-change
compiler and tests are preserved under
`~/rk2608_backups/tall-cmac-m128-before-cc1348718-20260803-014900`.

Windowed selectors now read an atom-aligned typed source subview directly when
its complete aligned K span is inside the declared source extent. Selector
weights are zero for unused physical lanes, so no scratch zero/copy is needed;
unsafe tail windows retain the established NPU pack path. This is a layout
legality decision shared by contractions and affine reductions, not a shape or
operation-name shortcut.

The formerly rejected batch-8 K2 convolution plans fall from 540 stages to 263
and 219 stages. Existing padded Conv1D falls from 395 to 174 stages, M81/K4
Conv2D from 374 to 180 stages, and the large affine-average compiler plan from
54 stages to 18 direct CMAC windows. Complete `test_simple_conv2d`, whose old
selector estimate was 8.34 MiB, now uses 242 stages, 331,312 constant bytes,
and 17,632 scratch bytes and passes natively in 29.87 seconds.

All 83 host tests, Ruff, and mypy pass, as does the strict hardware suite (64
tests plus 42 subtests) in 583.71 seconds with `ROCKCHIP_FALLBACK=0`. The
inferred native total is 134 after the current 128-pass census. The research
tree now has 27,877 counted lines, including 2,744 in the Rockchip
renderer/compiler. The pre-change compiler and tests are preserved under
`~/rk2608_backups/direct-cmac-windows-before-880171ff4-20260803-020414`.

Affine contraction compute now tiles M into ordered tasks satisfying
`tile_M*aligned_K<=4096`; packed A and physical Mx64 output scratch remain one
typed program resource. Separately, pointwise graphs with two or more static
affine FP16 broadcasts materialize each surface through the generic selector
planner and then schedule the ordinary DPU expression. This covers the K=1
form where early simplification removes the reduction, without a convolution
recognizer or host work.

The formerly rejected grouped M168/K5/N6 plan uses two compute tiles and totals
260 stages/625,328 constant bytes. K=1/cin=1 batch-1 and batch-8 plans use 15
and 71 stages. Together these complete all 14 `test_conv1d` subtests natively
in 180.73 seconds at unchanged tolerance. All 84 host tests, Ruff, and mypy
pass, as does the strict hardware suite (65 tests plus 44 subtests) in 616.00
seconds with `ROCKCHIP_FALLBACK=0`. The inferred native total is 135 after the
current 128-pass census. The research tree now has 27,932 counted lines,
including 2,799 in the Rockchip renderer/compiler. The pre-change compiler and
tests are preserved under
`~/rk2608_backups/tiled-cmac-m-before-04b9585b3-20260803-022155`.

The complete uncached strict census at `bc5c4353a` confirms 137
`PASS_NATIVE`, 40 `PASS_FRONTEND`, 235 `FAIL`, and 13 `SKIP_UPSTREAM` across
exactly 425 methods. The run used `ROCKCHIP_FALLBACK=0` and completed in
2,234.48 seconds without an NPU timeout, reset failure, invalid submission, or
process abort. No method that was native in the `6133442c2` census regressed.

Nine methods transition from `FAIL` to `PASS_NATIVE`: `test_conv1d`,
`test_conv2d`, `test_copysign`, `test_max_pool2d_dilation`,
`test_max_pool2d_smaller_stride`, `test_medium_grouped_conv2d`,
`test_simple_conv2d`, `test_simple_conv2d_1x1`, and
`test_simple_padding_conv1d`. The two unanticipated transitions confirm that
the layout/contraction work generalizes beyond its focused probes.

The 235 failed methods first reject as 64 unsupported-output-dtype, 48
plan-stage-limit, 35 unsupported-layout, 30 requires-reformat, 25
unsupported-ALU, 23 unsupported-input-dtype, and 4 unsupported-reduction.
Six failures have no native reject and therefore remain numerical or frontend
contract investigations. The durable JSON is
`~/rk2608_backups/census-conv-bc5c4353a-20260803/test_ops_coverage.json`
(SHA-256 `010527a2a5af8efc43ecc09a8b160978677c3299c46ca2a7866d57f086375cb4`);
the JUnit XML SHA-256 is
`85b082338e041c16c4243eafeaaf127830dccfcd7721e71b146e6cbb7a3043c0`.

Reduction epilogues now compose through generic ordered target plans. A tiled
contraction writes its dense result to padded NPU scratch; every non-output
affine FP16 operand is expanded through the existing selector planner, and the
parsed pointwise expression is scheduled as a final DPU program. Completed
program cost remains subject to the shared 400-stage and 2 MiB ceilings, and
no UOp reaches the target plan or emitter.

Complete `test_simple_conv2d_bias` passes natively in 29.52 seconds at
unchanged tolerance. Its plan has 258 stages (44 DPU and 214 CMAC), 347,696
constant bytes, and 18,592 scratch bytes. `test_bias_conv_transpose2d`,
`test_nested_conv2d`, and `test_output_padded_conv_transpose2d` now progress
past epilogue legality and reject on the existing surface/K/selector resource
bounds rather than pretending support. All 82 host compiler/image tests, Ruff,
and mypy pass; the strict hardware suite passes 65 tests plus 44 subtests in
615.29 seconds with `ROCKCHIP_FALLBACK=0`. The inferred native method total is
138 after the current 137-pass census, pending the next complete run. The
research tree has 27,985 counted lines.

Conditional affine ADD reductions now recognize a single FP16 indexed surface
guarded by a compile-time coordinate predicate. Rejected predicate points are
represented by absent CMAC selector weights, so masking is performed by the
NPU's generated immutable matrix rather than host materialization or a named
prefix-sum path. The same static-affine and input-bound proofs apply before any
selector is emitted.

Complete `test_small_cumsum` passes natively. Its plan contains two DPU stages,
one CMAC task, 2,144 constant bytes, and 64 scratch bytes. The larger 2D prefix
surface remains a typed plan-cost rejection, and the 512/1022-element frontend
uses FP32 accumulation outside the current native dtype contract. All 83 host
tests, Ruff, and mypy pass; the expanded strict hardware suite passes 66 tests
plus 44 subtests in 617.39 seconds with `ROCKCHIP_FALLBACK=0`. The inferred
native total is 139 after the current 137-pass census, pending the next complete
run. The research tree has 27,995 counted lines.

Infinite threshold selection now has a dedicated generic expression recipe.
For `WHERE(x<threshold, +/-inf, x)` and its reversed comparison, the finite
branch is first clamped to a finite threshold. The active comparison mask then
forms signed infinity through `sign*mask/(1-mask)`, which is zero on the finite
branch and infinite on the selected branch. This avoids both `infinity*0` and
`infinity-infinity`, while hardware probes cover positive/negative infinity,
NaN, and both comparison directions.

Complete `test_masked_fill` passes natively at unchanged tolerance. All 84 host
tests, Ruff, and mypy pass; the expanded strict hardware suite passes 67 tests
plus 44 subtests in 618.87 seconds with `ROCKCHIP_FALLBACK=0`. The inferred
native total is 140 after the current 137-pass census, pending the next complete
run. The research tree has 28,008 counted lines. The unsuccessful attempt to
admit 648-element contraction sources without a more efficient pack engine is
preserved as `wip-wide-source-contraction-still-stage-limited.patch` (SHA-256
`04eef8eaa739e249c4ee8f5bb93d7005dce8812639e276d2400a35c2aa7dc605`).

Nested scalar ADD reductions now lower as two ordered native selector plans.
The compiler first proves that the complete affine index map is a dense
bijection over one FP16 input. It then reduces the innermost axes into padded
FP16 NPU scratch and submits a second reduction over that materialized result.
The intermediate store is semantically required: an initially tested
single-reduction collapse was rejected after RK3588 showed the expected
one-ULP difference from removing the intermediate FP16 rounding boundary.

All three axis variants in complete `test_sum_twice` pass natively at unchanged
tolerance. The focused hardware test also covers all three variants. All 89
host compiler/image/telemetry/fallback tests, repository-wide Ruff, and mypy
over 225 tinygrad modules pass; the expanded strict hardware suite passes 68
tests plus 47 subtests in 619.03 seconds with `ROCKCHIP_FALLBACK=0`. The
inferred native total is 141 after the current 137-pass census, pending the next
complete run. The research tree has 28,064 counted lines. Pre-change sources
are preserved under
`~/rk2608_backups/nested-sum-before-2d2c0410c-20260803`.

Short scalar FP16 MUL reductions now use explicit physical lane
materialization. The source is padded once in NPU scratch; each logical lane
is selected by CMAC into a separate addressable atom, and DPU MUL stages fold
those atoms in source order. This avoids the disproven assumption that a
sub-16-byte relocation addend can select a lane. Extents 2–32 are admitted;
larger or non-dense products remain typed rejects until a logarithmic native
scan/reformat plan is available.

Complete `test_const_reduce` now passes natively: its fill, SUM, MUL, and MAX
kernels all execute through RKImage tasks. The nine-lane product plan contains
21 stages (9 CMAC and 12 DPU), 16,560 constant bytes, and 816 scratch bytes.
Focused hardware coverage also checks three- and nine-lane products, including
a nonconstant finite sequence. All 90 host tests, repository-wide Ruff, and
mypy over 225 tinygrad modules pass; the strict hardware suite passes 69 tests
plus 50 subtests in 625.15 seconds with `ROCKCHIP_FALLBACK=0`. The inferred
native total is 142 after the current 137-pass census, pending the next complete
run. The research tree has 28,108 counted lines. Pre-change sources are
preserved under
`~/rk2608_backups/scalar-product-before-fe251d06d-20260803`.

Affine FP16 MUL reductions now transpose the reduction into a small number of
full logical term surfaces. Each reduction coordinate is materialized through
the generic selector planner, then ordinary vector DPU MUL stages fold those
surfaces in source order. Work therefore scales with the short reduction width
rather than gathering every output scalar independently. The same affine
partition, source-bound, 65,536-visit, 400-stage, and 2 MiB proofs remain in
force.

Complete `test_prod` passes natively, including scalar input, axis 1, axis 3,
and keepdim cases; `test_prod_dtype_arg` remains green. The `(3,4,5,6)` axis-1
plan uses 29 stages, 82,016 constant bytes, and 1,344 scratch bytes; axis 3 uses
41 stages, 74,064 constant bytes, and 2,624 scratch bytes. All 91 host tests,
repository-wide Ruff, and mypy over 225 tinygrad modules pass; the strict
hardware suite passes 70 tests plus 52 subtests in 632.61 seconds with
`ROCKCHIP_FALLBACK=0`. The inferred native total is 143 after the current
137-pass census, pending the next complete run. The research tree has 28,178
counted lines. Pre-change sources are preserved under
`~/rk2608_backups/affine-product-before-1887a2fff-20260803`.

Masked affine MUL reductions now materialize multiplicative identity without
host data synthesis. The compiler follows each static WHERE branch at every
affine coordinate. Active points select the indexed FP16 value; inactive
points select lane zero from an NPU-filled ones atom. DPU ADD combines those
two disjoint surfaces, then the established vector product fold executes in
source order. Dynamic predicates and non-one inactive arms reject.

Complete `test_small_cumprod` passes natively at unchanged tolerance. Its
ten-element prefix plan has 69 stages, 39,168 constant bytes, 2,560 scratch
bytes, and no UOps after target lowering. All 92 host tests, repository-wide
Ruff, and mypy over 225 tinygrad modules pass; the strict hardware suite passes
71 tests plus 52 subtests in 643.21 seconds with `ROCKCHIP_FALLBACK=0`. The
inferred native total is 144 after the current 137-pass census, pending the next
complete run. The research tree has 28,267 counted lines. Pre-change sources
are preserved under
`~/rk2608_backups/masked-product-before-35f390287-20260803`.

Wide dense affine SUM now uses the existing two-level selector when one
windowed CMAC cannot keep both source width and destination alignment legal.
For a 256x256 surface, the first level reduces two K256 rows per aligned
intermediate group and the second compacts padded scalar atoms into the dense
output. The plan has 145 stages, 36,864 unique constant bytes, and 2,080
scratch bytes, remaining below all shared limits.

Hardware validation exposed a separate producer boundary: one 32,768-element
FP16 constant fill succeeds, while an untiled 65,528-element probe timed out.
FP16 fills now tile at the proven 32,768-lane DPU width; a 65,536-element fill
uses two address-aligned tasks and writes every value. Complete
`test_sum_collapse` consequently passes natively in 16.98 seconds. All 94 host
tests, repository-wide Ruff, and mypy over 225 tinygrad modules pass; the
strict hardware suite passes 73 tests plus 52 subtests in 659.40 seconds with
`ROCKCHIP_FALLBACK=0`. The inferred native total is 147 after the current
146-pass census, pending the next complete run. The research tree has 28,269
counted lines. Pre-change sources are preserved under
`~/rk2608_backups/two-level-affine-sum-before-6f7872696-20260803`.

Static affine ADD reductions may now select from multiple FP16 input surfaces.
The compiler evaluates every static `WHERE` branch at each affine coordinate,
builds one selector plan per source, and combines the source-local partial
sums with ordinary DPU ADD stages. Coordinates selecting no input must reduce
to zero when all indexed leaves are replaced by zero; dynamic predicates and
nonzero host-synthesized values remain typed rejects. The final composed plan
continues to obey the shared 400-stage and 2 MiB constant limits.

Complete `test_sum_cat_collapse` now passes natively at unchanged tolerance.
Its `(256,256)` plus `(256,64)` concatenated row-sum plan contains 178 stages,
71,680 unique constant bytes, and 3,168 scratch bytes. A focused nonconstant
hardware test independently exercises the same CMAC-per-source plus DPU-combine
path. All 95 host tests, repository-wide Ruff, and mypy over 225 tinygrad
modules pass; the strict hardware suite passes 74 tests plus 52 subtests in
691.56 seconds with `ROCKCHIP_FALLBACK=0`. The inferred native total is 148
after the current 146-pass census, pending the next complete run. The research
tree has 28,349 counted lines. Pre-change sources are preserved under
`~/rk2608_backups/multi-source-sum-before-a0546884d-20260803`.

The first current-head census attempt exposed a numerical-contract regression
in the new two-level selector. `test_avg_pool2d_padding` exceeded its 300-second
method watchdog and was aborted while issuing a stage reset. Isolating the
method with a 900-second diagnostic watchdog showed that the driver was not
wedged: all nine subcases completed in 358.49 seconds. The three K2 cases
passed, while every K3 case returned the unscaled sum—exactly 9x or 6x the
reference. The fallback two-level planner had omitted `scale` and therefore
encoded one in its second-level compaction weights.

Two-level selection now applies the reduction scale in the second CMAC level.
An exactly representable 0.25 scale passes the wide nonconstant RK3588 test;
scales such as 1/9 and 1/6 that the current FP16 weight contract cannot encode
exactly return `NUMERICAL_CONTRACT` before image execution instead of silently
becoming one. With
the original 300-second watchdog, `test_avg_pool2d_padding` now completes in
124.75 seconds with three native subcase passes and six typed plan rejects.
All 96 host tests, repository-wide Ruff, and mypy over 225 modules pass; the
strict hardware suite passes 74 tests plus 54 subtests in 709.02 seconds. No
new complete-method pass is claimed, and the authoritative census remains the
146-pass artifact until a new uninterrupted run finishes. Counted source is
28,351 lines. Pre-change sources are preserved under
`~/rk2608_backups/two-level-scale-before-862202e58-20260803`.

The corrected complete uncached census at `85eeab7f5` finishes normally in
2,526.78 seconds with exactly 148 native, 40 frontend-only, 224 failed, and 13
upstream-skipped methods. `test_sum_collapse` and `test_sum_cat_collapse` are
the only transitions from `c94d0d24c`; no native method regresses. Raw pytest
reports 260 failed, 200 passed, 13 skipped, and 78 passing subtests. There is no
timeout, invalid submission, reset failure, or process abort.

The fresh first-reject Pareto is 65 unsupported-output-dtype, 53
plan-stage-limit, 35 unsupported-layout, 23 unsupported-input-dtype, 18
requires-reformat, 10 unsupported-ALU, nine numerical-contract, and three
unsupported-reduction; eight failed methods had no method-level first reject.
The focused audit above separates accepted numerical errors from later rejects
and frontend/compiler failures. Telemetry and JUnit artifacts are stored
under `~/rk2608_backups/census-scale-85eeab7f5-20260803/` with SHA-256
`ad2acf33274bd9db8023931e65598f6e2e95c47205bb51b0dfcb28bfbd434605`
and `1a1430b4a2a11e21aa31720f92afe6bd359d5e339adbc74f11a3e74aa263a908`
respectively.

The post-census non-exact-scale investigation proved that DPU BN can multiply
the flying CMAC FP32 accumulator before output conversion. A closest-product
two-FP16 factorization of 1/9 (`0.06744384765625 * 1.6474609375`) differs from
the mathematical reciprocal by only `6.6227383022088304e-09`. It passed small
ramp and random row-sum probes, but the complete official padded-pooling method
still failed strict `rtol=1e-5`: 3x3 subcases mismatched 2--8 of 1,920--2,560
outputs and 3x2 subcases mismatched 62--92 of 2,880--3,840 outputs, always by
one FP16 ULP. A single FP16 BN reciprocal was substantially worse (890/2,560
and 1,308/3,840 mismatches).

Two follow-up register probes close misleading alternatives. Programming the
output converter as integer `29127 >> 18` with FP32-to-FP16 conversion disabled
returned all zeros, confirming that this mode is not a higher-precision FP16
reciprocal. Enabling DPU RDMA and EW FDIV in the same CMAC task timed out after
six seconds; the ordinary CMAC row-sum recovery test passed immediately after
reset. No tolerance, skip, or CPU retry was added. The active compiler retains
the pre-submit `NUMERICAL_CONTRACT` rejection, and the WIP implementation plus
hardware notes are archived as `0134-WIP-fused-BN-two-factor-scale.patch` with
SHA-256 `a472dcbfb2a610e0aa0afe86f0318ad7e293d97417cf11067a6e94473054c22c`.

The first accepted-native numerical repair handles the exact lowered identity
`WHERE(x != 0, EXP2(-inf*x), 1)` used by zero-base power. Arithmetic selection
cannot eagerly evaluate the inactive exponent at zero because `-inf*0` becomes
NaN and `NaN*0` remains NaN. The compiler now replaces inactive zero with one
before EXP2 and applies the predicate afterward. The strict RK3588 regression
returns `[inf, inf, 1, 0, 0, 0]` for exponents `[-2,-1,0,1,2,3]`; the official
`test_pow_const` proceeds past this subcase and next exposes the independent
arbitrary-base `0.7**x` LUT accuracy contract. No complete-method gain is
claimed yet.

The biased-convolution failure was not a raw CMAC error: an un-biased 1x1 K8
probe is bit-identical to PyTorch's FP32 accumulation, while the old separate
DPU bias task rounded the accumulator to FP16 first. The first biased/ReLU
convolution already differed in 29 of 200 FP16 encodings, and the second
convolution amplified that boundary to 38 outputs outside `rtol=1e-3`.

`RKContract` now carries a typed channel-bias/optional-ReLU `RKEpilogue`. The
compiler gathers FP16 channel bias into aligned four-lane atoms, converts it to
a padded FP32 bias surface using DPU tasks, and the CMAC command enables BRDMA
addition before FP16 output conversion. ReLU uses the same flying DPU stage.
The official `test_biased_conv2d` method now passes unchanged; its two kernels
are 87 stages each rather than 92 and 91. `test_simple_conv2d_bias` remains
green, all 88 host compiler tests pass, and the strict hardware suite passes 76
tests plus 54 subtests in 729.98 seconds. This is one focused inferred native
gain; the authoritative complete census remains 148 until the next uncached
425-method run.

The rest of the accepted-numerical audit initially failed closed at the exact
legality boundary. Structural guards returned `NUMERICAL_CONTRACT` before
RKImage execution for `x + (y-x)*z` until it had a fused FP32-intermediate task;
signed-zero `copysign` reconstructed through `x<0 OR reciprocal(x)<0` because
RK3588 FDIV loses the required negative-zero sign; unreduced probability BCE
whose staged LUT composition misses 38/320 outputs; and root scaled-EXP2 with
an uncharacterized factor such as `log2(0.7)`. Whole-expression activation
canonicalizers remain legal, so TANH, Mish, QuickGELU, and both GELU variants
retain their generated-table paths.

Focused official runs now report those four precise typed rejects instead of
AssertionError, and all 89 host compiler tests pass. The ordinary sampled
`test_copysign` method will also reject until signed-zero semantics are truly
implemented; that deliberate temporary coverage regression prevents a sampled
pass from claiming an invalid full-domain kernel. Combined with the fused bias
repair, the post-census inferred native total remains 148 pending a complete
uncached run, with no known accepted-wrong path left in the audited bucket.

The lerp guard has now been replaced by a hardware-proven generic fused
arithmetic path. Raw probes establish the operand contract: MRDMA converts the
main FP16 `y` input to FP32, BRDMA supplies FP32 `x` to BS subtraction, NRDMA
supplies FP16 `z` to BN multiplication, and ERDMA supplies FP16 `x` to EW
addition. BRDMA repeats its channel operand across width, so each fused task is
an eight-channel atom. CMAC materializes dense FP32 `x` directly in 32-lane
identity tiles; this avoids the rejected padded-half conversion design, whose
grouped DPU converter timed out above 17 groups and would exceed the 400-stage
contract when tiled.

The official 45x35 tensor lerp is 247 tasks (50 CMAC plus 197 DPU), 2,048
constant bytes, and 6,528 scratch bytes. It is bit-exact against the required
FP32-intermediate calculation. The scalar-weight subcase also passes, and the
unchanged official `test_lerp` completes in 27.47 seconds with
`ROCKCHIP_FALLBACK=0`. The focused device regressions pass, and all 90 compiler
tests plus three subtests pass. This is one inferred complete native-method
gain, raising the post-census estimate from 148 to 149 pending the next full
uncached census. The remaining pre-submit numerical guards cover signed-zero
copysign, unreduced probability BCE, and uncharacterized scaled EXP2.
The complete strict device suite passes 77 tests plus 54 subtests in 731.46
seconds with fallback disabled and no concurrent NPU process.

Plan telemetry now separates native correctness evidence from efficient native
plans. `RKPlanCost` includes exact emitted task, reset, command-word, constant,
and scratch counts plus logical read/write and MAC estimates. Selector candidates
use the richer tuple instead of comparing only stages, constants, and scratch.
Kernel telemetry retains `PASS_NATIVE` semantics but adds `native_quality`,
`task_count`, `command_words`, and `reset_count`; plans above 64 tasks or 1 MiB
of constants are labeled `CORRECTNESS_FALLBACK`. This deliberately identifies
the 178--395-task passes as direct-path debt rather than hiding them inside the
native total. The 400-task compiler ceiling is unchanged.
All 91 compiler tests plus three subtests pass, and the complete strict device
suite remains green at 77 tests plus 54 subtests in 729.59 seconds.

A direct dense-row PPU investigation rejected an apparent shortcut before it
entered the compiler. Global MAX over tightly packed HWC1, HWC2, and HWC4
surfaces submits successfully but returns incomplete channel-stepped maxima;
the same command is exact only for HWC8. Setting `PPU_MISC_CTRL.NONALIGN` for
HWC1 times out, and three DPU MAX stages with two-byte source addends do not
reduce within an eight-lane atom. A known-good HWC8 PPU recovery passes after
the timeout. The safe probes and the opt-in timeout case are retained in
`extra/rockchip/probe_ppu_channels.py`. Consequently the compiler keeps the
typed stage-limit rejection rather than accepting a corrupt one-channel layout.

Direct CMAC width characterization establishes that a task can produce 32
logical FP16 channels. The apparent zero outputs above channel 15 were a layout
mistake: WDMA stores the second channel block in physical lanes 32--47. Hardware
probes are exact for N=16, 20, 24, 28, and 32. The tiled contraction lowerer now
accepts N<=32 and maps the second physical atom during output compaction. A
dynamic 4x9 by 9x24 contraction passes unchanged on hardware; its current cost
is 83 tasks, 3,762 command words, 388,160 constant bytes, 3,456 scratch bytes,
and is therefore correctly marked `CORRECTNESS_FALLBACK`.

The first three `test_einsum_ellipsis` subcases now proceed through strict native
execution, but the method's final 32x7x24x24x24 case honestly remains rejected:
its contraction surfaces contain 3,096,576 values per operand and K=13,824.
No complete-method coverage gain is claimed from this milestone.
All 102 compiler/image/fallback/telemetry tests plus three subtests pass. The
complete strict device suite passes 78 tests plus 54 subtests in 740.71 seconds.

Wide CMAC characterization now extends the same physical rule through N=64,
96, and 128, all bit-exact. The lowerer chooses `align_out` in 32-channel groups,
requires `align_in >= align_out`, sizes physical output rows at twice the logical
channel group, and maps each 16-channel block through its 32-lane WDMA atom.
A dynamic 4x9 by 9x40 graph and the larger 1x64 by 64x40 graph are bit-exact on
hardware; the latter uses 331 tasks without changing the 400-task ceiling.

The old 512-element RHS pre-reject is replaced by cost-bounded planning. Before
constructing selector payloads, the compiler checks both the sparse constant
lower bound and the minimum number of 16-output packing/compute/unpack tasks.
The official 64x99 matmul shape rejects in about 45 ms with a precise 408-task
lower bound, rather than spending seconds constructing an inevitably illegal
plan. Passing it now requires a more direct dynamic-weight reformat engine, not
a larger task limit.

All 103 compiler/image/fallback/telemetry tests plus three subtests pass. The
complete strict device suite passes 79 tests plus 56 subtests in 769.17 seconds
with no concurrent NPU process.

One-row CMAC output compaction and a wider proven selector window remove the
64x99 matmul stage blocker without changing the 400-task or 2 MiB ceilings.
Hardware establishes that `DPU_SURFACE_ADD=0x20` stores every probed one-row
FP16 output from N=16 through N=128 contiguously. The same setting does not
remove multi-row physical row padding, so `RKContract.compact_output` is a
typed M=1-only emission contract. A final DPU copy writes only the logical
output extent and therefore never writes CMAC padding beyond the user buffer.

A single selector CMAC exactly gathers the stride-99 positions
`[0,99,...,1485]` through a 1,504-lane input window. This proven allowance is
scoped to tiled-contraction RHS packing; all ordinary affine selector plans
retain their former 512-lane ceiling and cost profiles. Generated selector
payloads deduplicate to 770,048 bytes. Input tails are bounded by the
Rockchip allocator's explicit 4-KiB GEM rounding and can only feed zero-weight
padding lanes. Direct contiguous LHS use eliminates two packing tasks.

The resulting `1x64 @ 64x99` plan is 399 tasks, 18,326 command words, 399
resets, 803,024 constant bytes, 33,056 scratch bytes, and 9,459,811 estimated
MACs. A random nonconstant hardware run is bit-exact against FP32 accumulation
followed by FP16, and unchanged strict `TestOps.test_matmul` passes in 48.05
seconds with `ROCKCHIP_FALLBACK=0`. The complete uncached census confirms the
method as `PASS_NATIVE`.

The rejected DPU reformat experiment is retained rather than hidden. Scalar
height rows are written eight half-lanes apart, source line-notch values zero,
four, and five do not change the gathered addresses, and DPU NONALIGN times
out. The ordinary DPU recovery test passes after reset. All 103 unit tests plus
three subtests pass, mypy is clean across 225 modules, touched-file Ruff is
clean, and the serialized strict device suite passes 79 tests plus 56 subtests
in 728.41 seconds. Repository-wide Ruff still reports 13 pre-existing
Python-3.12 nested-f-string syntax findings in `extra/rockchip/gen_lut.py` under
its Python-3.11 parser target.

The complete `d1437ad58` census finishes in 2,484.47 seconds with 150 native,
40 frontend-only, 222 failed, and 13 upstream-skipped methods. Raw pytest is
202 passed, 258 failed, 13 skipped, and 78 passing subtests. Relative to
`85eeab7f5`, biased Conv2D, lerp, and matmul become native while copysign is the
one intentional pass-to-reject transition described above. There is no NPU
timeout, invalid submission, reset failure, or process abort.

The telemetry JSON and JUnit XML are stored under
`~/rk2608_backups/census-wide-matmul-d1437ad58-20260803/`. Their SHA-256 hashes
are `8b2abaca04b8200a16ae92120a9aa98df25e31fce6bdca9715b3a335c7f70d92`
and `b85ebee1091c93ee11e4d8dc478be122230224e826b75746a7d457e23d265799`
respectively. The first-reject Pareto is 65 unsupported-output-dtype, 47
plan-stage-limit, 34 unsupported-layout, 23 unsupported-input-dtype, 18
requires-reformat, 10 unsupported-ALU, and 10 numerical-contract, with 15
partial/front-end failures lacking one method-level first reject.

## Direct spatial convolution milestone

`test_simple_conv2d_batched` was the closest stage-limit reject at 421 tasks,
only 21 above the unchanged ceiling. Plan inspection showed that merely raising
the limit would preserve the wrong architecture: 310 tasks performed an
im2col-like affine input expansion, 12 packed weights, only two performed the
contraction, and roughly 81 restored NCHW output order. A new exact affine
matcher identifies dense stride-one NCHW/OIHW spatial contractions from their
complete coefficient maps and lowers the proven channel-4 family to typed
`RKSpatialConv` steps. All input, weight, and output layout conversion remains
in NPU selector tasks; the runtime executes no packing or tensor semantics.

The direct batched plan is 92 tasks, 4,208 command words, 92 resets, 700,384
constant bytes, 5,344 scratch bytes, and 294,632 estimated MACs. The 400-task
and 2 MiB limits are unchanged. An initial four-lane CNA alignment assumption
submitted but returned uniformly tiny values. Register-by-register comparison
with the preserved `ref/rk3588/examples/conv_simple.py` oracle isolated the
actual eight-lane input/weight contract, NHWC conversion field, DMA strides,
and padded weight format. After correction, the emitted register values match
the known-good direct-convolution task exactly.

The official strict method passes in 10.97 seconds at unchanged tolerance, and
a randomized device test confirms both numerical parity and two telemetry-
visible `CONV` tasks. The full serialized device suite passes 79 tests plus 56
subtests in 728.68 seconds; 104 host tests plus three subtests, full tinygrad
mypy, and touched-file Ruff also pass. The complete census below confirms this
failure-to-native transition.

### Channel-16 direct layout

The direct matcher now also accepts simplified batch-one graphs and the
separately proven 16-input-channel CNA contract. Activations are packed on the
NPU into C1/H/W/C2 eight-lane atoms, while weights are packed into KH/KW/OC/IC
order with sixteen physical input lanes. The typed emitter selects the matching
non-NHWC conversion, CBUF, and DMA-stride fields. Its register values are
identical to the known-good channel-16 `conv_simple.py` task.

`test_simple_conv2d_m4` changes from an early affine-surface plan-limit reject
to a 275-task program: 12,652 command words, 275 resets, 899,072 constant bytes,
8,864 scratch bytes, and 1,341,344 estimated MACs. The unchanged official test
passes in 30.99 seconds, and a randomized device test confirms parity plus one
telemetry-visible `CONV` task. The complete expanded hardware suite passes 81
tests plus 56 subtests in 750.28 seconds; 105 host tests plus three subtests,
mypy, and touched-file Ruff also pass. Together the two direct-convolution
milestones account for the two failure-to-native transitions in the complete
census.

## Complete direct-convolution census

The complete uncached strict run at `c65396da1` finishes in 2,498.83 seconds:

```
152  PASS_NATIVE
 40  PASS_FRONTEND
220  FAIL
 13  SKIP_UPSTREAM
425  total
```

Raw pytest reports 204 passed, 256 failed, 13 skipped, and 78 passing subtests.
Relative to the `d1437ad58` census, exactly
`TestOps.test_simple_conv2d_batched` and `TestOps.test_simple_conv2d_m4`
transition from failure to native pass. No accepted method regresses. The run
completes without an NPU timeout, invalid submission, reset failure, or process
abort.

The telemetry JSON and JUnit XML are stored under
`~/rk2608_backups/census-direct-conv-c65396da1-20260803/`. Their SHA-256 hashes
are `7dd09a9152e43af0fbc726ce71e4972d2bd3eff3a96c17d26c0d7d3f47a3a369`
and `427a54dc6a4b9fac478b0ab8b548c80740a78b5483c14d121801ca602b758867`
respectively.

The version-1 census artifact leaves failed subtest rejects under each subcase
rather than promoting one to a method-level field. Reading both locations
resolves 12 of the 15 apparent method-level gaps. The aggregate first-reject Pareto is 65
unsupported-output-dtype, 50 plan-stage-limit, 35 unsupported-layout, 23
unsupported-input-dtype, 18 requires-reformat, 13 numerical-contract, 10
unsupported-ALU, and three unsupported-reduction. Only `test_cumprod`,
`test_maximum`, and `test_mulacc_with_zero_strides` have no compiler reject at
any level; they require numerical/runtime/frontend failure classification, not
a fabricated reject.

Telemetry schema version 2 fixes the reporting gap. The method record now
promotes the earliest reject belonging to a failed subcase into
`first_reject`; rejects from passing subcases are deliberately excluded. Each
subcase also carries its own `first_reject`, `failure_kind`, and failure class
and message. Method failures distinguish `NATIVE_REJECT`, `DEVICE_FAILURE`,
`POST_EXECUTION_FAILURE`, and `NON_DEVICE_FAILURE`. Consequently `cumprod` is
recorded as a post-execution numerical failure after its native program,
`maximum` retains its FP16-packing `OverflowError`, and
`mulacc_with_zero_strides` retains its pre-device Clang failure instead of any
of them being counted as an unspecified compiler reject.

### Masked-product numerical guard

The census's accepted 20-element cumulative-product program returned a
corrupted result after earlier mixed DPU/CMAC/PPU/CONV workloads. Repeating the
same seeded input in isolation was bit-exact, so the failure is state-sensitive
rather than a deterministic expression-recognition error. A separate
10-element hardware regression remains bit-exact and uses one physical CMAC
output tile.

The clean contract is consequently narrowed to at most sixteen masked-product
outputs. Wider plans return `NUMERICAL_CONTRACT` before device submission; the
400-task ceiling and all tolerances remain unchanged. The compiler regression
checks that 20 outputs reject, the proven 10-output hardware test passes in
8.18 seconds, and the unchanged official method now reaches a precise native
reject in 2.36 seconds rather than exposing intermittent wrong output. Wider
native cumulative products remain blocked until mixed-engine stress testing
proves the hardware state and multi-tile layout reliable.

### Exact wide-fill input contract

The remaining backend-origin non-reject failure came from `test_maximum`'s
integer `INT_MAX` case. Lowering accepted a typed int32 fill, but the emitter
then attempted to materialize `2147483647` as the DPU's FP16 input and leaked a
Python `OverflowError`. The wide public WDMA dtype does not change the scalar
input precision.

Legalization now proves exact FP16 round-trip representability for every int32
or FP32 fill constant. `INT_MAX` and FP32 `0.1` reject with
`NUMERICAL_CONTRACT`; exact constants such as four retain the existing native
wide-fill path. The unchanged focused `test_maximum` reaches the typed reject
in 2.73 seconds after its earlier native subcases pass. No emitter exception,
host conversion, or relaxed comparison is involved.

## First-class native reformat plan

The standalone affine movement lowerer previously returned either a raw
`RKDPUProgram` or an opaque `RKProgram` of selector contracts. That exposed the
implementation topology but lost the semantic fact that both represented the
same physical transform.

It now returns typed `RKReformat` with logical `src` and `out` tensor refs, the
complete static output-to-source mapping, and `RKReformatKind.COALESCED_DPU` or
`RKReformatKind.SELECTOR_CMAC`. The selected implementation remains an ordered
tuple of UOp-free `RKDPUProgram`/`RKContract` steps with an explicit scratch
table. `emit_reformat` cannot inspect UOps or re-plan the transform; it only
serializes those chosen steps through the existing emitters.

This milestone is intentionally command-byte and coverage neutral. It creates
the comparison boundary required for a later direct reformat hierarchy without
prematurely widening every mixed `RKProgram` step type. All 102 focused
compiler/image tests plus three subtests pass, mypy and Ruff are clean, and the
unchanged strict transpose, permute, and flip methods pass on RK3588 in 28.31
seconds. The full host selection passes 110 tests plus three subtests, and the
complete serialized hardware suite passes 81 tests plus 56 subtests in 746.38
seconds without timeout, reset failure, or invalid submission.

## Compact 32-output selector tiles

The windowed selector planner still limited each CMAC task to sixteen logical
outputs even though the hardware probe proves compact one-row writes for every
N from 16 through 128. It now builds up to 32 selector rows per task, uses one
physical 32-channel weight/output surface, and sets the typed
`RKContract.compact_output` contract. A partial final tile writes only into
scratch-backed padding; logical copies never expose it to a user buffer.

This does not raise either global plan limit. It reduces batched direct
convolution from 92 tasks/4,208 command words/9.82 seconds to 49 tasks/2,230
command words/5.23 seconds. The channel-16 case falls from 275 tasks/12,652
command words/29.37 seconds to 139 tasks/6,396 command words/14.87 seconds.
Their official outputs remain bit-exact. Movement, the bounded 399-task matmul,
and existing max-pool methods also remain green.

All 110 Rockchip host tests plus three subtests, mypy, Ruff, and diff checks
pass. The complete serialized device suite passes 81 tests plus 56 subtests in
700.77 seconds with no timeout, reset failure, or invalid submission. The next
complete census must determine which previously stage-limited method families
cross below the unchanged 400-task ceiling.

## Complete compact-selector census

The complete uncached strict run at `d237777da` finishes normally in 2,345.47
seconds with 152 `PASS_NATIVE`, 40 `PASS_FRONTEND`, 220 `FAIL`, and 13
`SKIP_UPSTREAM`. Raw pytest reports 204 passed, 256 failed, 13 skipped, and 78
passing subtests. No NPU timeout, invalid submission, reset failure, or process
abort occurs. No method changes outcome relative to the preceding direct-CONV
census.

The performance improvement is nevertheless visible in full-run telemetry.
`test_simple_conv2d_batched` falls from 92 to 49 tasks and 9.82 to 5.24
seconds, while `test_simple_conv2d_m4` falls from 275 to 139 tasks and 29.37 to
14.85 seconds. The complete run is 153.36 seconds faster. The bounded
399-task matmul remains a 42.69-second `CORRECTNESS_FALLBACK`, so packing and
native reformat remain the dominant architectural target rather than a larger
task limit.

The schema-version-2 Pareto contains 48 stage-limit methods. Two graphs that
previously stopped at that limit now proceed far enough to expose a more
specific numerical contract, but neither becomes a native method pass. The
only accepted native numerical failure was a 20-element cumulative sum whose
output became state-sensitive after the mixed-engine workload. The follow-up
prefix-scan guard below removes it from the accepted contract.

### Masked-sum numerical guard

The 20-element cumulative sum used two CMAC output tiles and returned the same
corrupt value in all lanes after the complete mixed-engine workload. The
existing ten-element hardware regression remains bit-exact and uses one output
tile. This is the additive counterpart of the already guarded masked-product
second-tile instability.

Legalization now identifies the structural prefix selector—row `i` contains
exactly source lanes `0..i`—and returns `NUMERICAL_CONTRACT` when it has more
than sixteen outputs. No test name or sampled value participates. The proven
ten-output hardware test still passes, while unchanged official `test_cumsum`
reaches the typed 20-output reject in 2.46 seconds instead of returning wrong
data. Wider prefix scans remain blocked until alternating-engine stress proves
the multi-tile state contract.

## Compact 64-output reformat tiles

One-row compact CMAC writes were already hardware-characterized through N=128,
but the reusable windowed selector stopped at 32 outputs and inferred its input
width from payload size. It now carries `align_in` explicitly and may emit a
64-output physical selector tile. The standalone typed `RKReformat` lowerer
also compares the sparse and windowed candidates instead of unconditionally
choosing the sparse implementation.

An 8x8 transpose consequently changes from two DPU preparation tasks plus four
16-output CMAC tasks to one direct 64-output CMAC task. The 432-element flip
falls from 27 selector contracts to seven. Direct-CONV packing uses the same
proven width: `test_simple_conv2d_batched` drops from 49 to 28 tasks and 5.24
to 2.99 seconds, while `test_simple_conv2d_m4` drops from 139 to 80 tasks and
14.85 to 8.55 seconds. Their unchanged official outputs remain exact.

The first unrestricted generic-contraction probe exposed an important
numerical boundary. `test_padded_conv_transpose2d` fell from a 415-stage reject
to a 245-task program, but 101 of 504 outputs exceeded the official tolerance
(maximum absolute error 0.02344). Generic tiled-contraction packing therefore
retains its previously characterized 32-output selector boundary; the same
graph again rejects at 415 stages before submission. This is not treated as a
coverage gain or fixed by relaxing tolerance.

All 112 Rockchip host tests plus three subtests, mypy, and Ruff pass. Exact
RK3588 movement and focused official methods pass, and the complete serialized
device suite passes 81 tests plus 56 subtests in 680.95 seconds with no timeout,
reset failure, or invalid submission. The authoritative method census remains
152 native, 40 frontend-only, 220 failed, and 13 upstream-skipped pending the
next complete run.

## Proven two-level periodic SIN

The current branch previously rejected direct `Ops.SIN` as unsupported ALU.
The frozen rockchip-2608 branch supplies a genuine NPU implementation: native
ROUNDOFF plus split FP16 periodic reduction, followed by broad and amplified
local LUT tasks. The clean port appends `SIN=73` and `SIN_LOCAL=74` without
renumbering any existing generated asset. Their payload digests match the
reference byte-for-byte.

The target plan contains 56 DPU tasks, seven reusable scratch surfaces, and no
UOps. The unchanged official forward-only `TestOps.test_sin` passes in 19.67
seconds with fallback disabled, including scalar, NaN/infinity, and magnitude
`10..1e6` subcases. A permanent serial hardware regression covering regional
boundaries and magnitudes through 10,000 passes in 13.06 seconds. No tolerance,
task ceiling, runtime conversion, or CPU semantic path changed.

The complete serialized hardware suite passes 82 tests plus 56 subtests in
693.48 seconds. Its first run exposed only a stale research-test assumption:
the explicit fallback-coherence test still used SIN to force the Python lane,
so the now-native kernel correctly reported DPU instead. That test now uses
still-unsupported TAN and independently proves DPU-to-Python-to-DPU mapped
buffer coherence; the fallback remains disabled for every official coverage
run. The clean rerun has no device or numerical failures.

The complete uncached census confirms this as the sole method transition from
`unsupported_alu` to `PASS_NATIVE`, producing the authoritative
153/40/219/13 result. COS is not claimed because the census plugin promotes
its public method to FP32, and TAN remains rejected because the frozen native
probes missed strict near-pole comparisons.
