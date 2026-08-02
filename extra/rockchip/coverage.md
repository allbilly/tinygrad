# Rockchip forward TestOps coverage

The completion target is every non-skipped forward case in the 425-method
`test/backend/test_ops.py` inventory. Focused tests are recorded as milestone
evidence but do not replace a complete uncached census.

Run the census serially on RK3588:

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
PYTHONPATH=/home/orangepi/rk_2608/test/rockchip:/home/orangepi/rk_upstream \
CACHELEVEL=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
python -m pytest test/backend/test_ops.py -p conftest_rockchip -q --tb=no
```

## Exact censuses

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

The following diagnostic-only milestone keeps the 140/272/13 tally unchanged
but splits every pre-submission rejection into `unsupported_dtype`,
`unsupported_layout`, `unsupported_contraction`, or `unsupported_op`. Focused
compiler tests cover all four classes. This makes the informational census
useful without treating honest hardware exclusions as numerical or device
failures.

## Milestones after the baseline

| Capability | Focused official gain | Full census folded in? |
|---|---:|---|
| Rank-0 FP16 constant fills | `test_ones`, `test_zeros` | Yes (`40c74406c`) |
| Native tiled int32/FP32 constant fills (research-only, now retracted) | `test_full`, `test_full_like`, `test_ones_like`, `test_zeros_like` | Historical (`40c74406c`) |
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

The historical wide-fill milestone wrote the requested dtype directly through
DPU WDMA; it did not use runtime narrowing or host semantic work. Its RKImage
dependency and relocation-width fixes remain useful, while the dtype-specific
fill path is no longer part of the declared backend contract.

The DPU MAX operation is not yet claimed IEEE-exact for all NaN operand
orders. Hardware testing showed that finite extrema are correct and FP16
absolute value preserves the expected infinity, signed-zero, and NaN results;
special-value MAX behavior remains outside this milestone.
