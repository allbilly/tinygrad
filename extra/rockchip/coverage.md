# Rockchip forward TestOps coverage

The completion target is every non-skipped forward case in the 425-method
`test/backend/test_ops.py` inventory. Focused tests are recorded as milestone
evidence but do not replace a complete uncached census.

All authoritative runs use `ROCKCHIP_FALLBACK=0`. The optional `RKPY` research
envelope invokes tinygrad's Python UOps emulator over mapped GEM buffers and is
therefore CPU semantic execution; it is excluded from pass counts. Likewise,
the 425-green `rockchip-2607` result cannot be treated as all-NPU proof because
that branch dispatches many families through NumPy-backed `_run_host_*` tasks.

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

Contiguous FP16 global sums now use a typed `RKSumProgram`. The compiler splits
an arbitrary extent into descending power-of-two blocks. DPU pairwise stages
reduce each block only while the second half begins at a 16-byte address; the
remaining aligned runs are copied into one scratch surface and a masked K32
CMAC produces the scalar. The four repeated CMAC weight rows are immutable
`[1...1,0...0]` image constants, so no invocation-time packing occurs. Plans
whose block decomposition needs more than 32 final terms reject explicitly.

The initial all-DPU tree was an important rejected probe: both values of a
two-element input were read from lane zero, producing `2*x[0]`. Disabling both
disk and schedule caches reproduced the result. It proves that DPU
`EW_BASE_ADDR` cannot select a sub-16-byte lane through a relocation addend;
ordinary tensor buffer views that start at an offset are a different ABI case.
The committed planner consequently never emits such a relocation. Strict
hardware tests cover lengths 2, 16, 60, 135, 720, and 16,384. The unchanged
official `test_sum_simple`, `test_sum_full`, `test_sum`, `test_sum_relu`,
`test_sum_tiny`, `test_mean`, and `test_mean_axis` methods pass with
`ROCKCHIP_FALLBACK=0`. `test_sum_twice` reaches native execution but remains a
numerical failure: its first subcase differs from Torch by one FP16 ULP near
0.194, outside the default relative tolerance. No tolerance was changed.
The uncached strict device suite passes 41 tests plus 17 subtests after this
generalization.

The first native reformat path handles static affine movements at the proven
16-byte DPU atom granularity. The compiler enumerates only static index maps,
coalesces adjacent aligned atoms, and emits ordinary DPU ADD-zero copy tasks;
runtime tensor values never visit the CPU. Strict hardware tests cover HWC8
permute, expand, and flip. A movement rejects when an output atom maps to
strided source elements or crosses a source-run boundary (for example an 8x8
scalar transpose), rather than silently using host gather.

Earlier slice tests appeared to show that DPU `SRC_BASE_ADDR` honors FP16
sub-atom relocation addends. The reduction probe separates two cases: a GEM
buffer view bound at an offset works, while adding two bytes to an already
bound DMA address reads lane zero. The reformat planner now requires aligned
source and destination atoms and rejects the latter representation. Real
offset buffer views still pass lengths 1, 2, 3, and 8 on hardware. Enabling the
separate `ERDMA_NONALIGN` bit was also rejected: it caused ordinary two-input
DPU arithmetic to time out and is not part of the committed path.

An independent unaligned-destination probe also timed out on the first
official flip shape. Its exact planner diff and failure are preserved as
`0063-WIP-unaligned-DPU-destination-timeout.patch`; the active compiler keeps
the destination-atom legality check.

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
