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

## Milestones after the baseline

| Capability | Focused official gain | Full census folded in? |
|---|---:|---|
| Rank-0 FP16 constant fills | `test_ones`, `test_zeros` | Yes (`40c74406c`) |
| Native tiled int32/FP32 constant fills | `test_full`, `test_full_like`, `test_ones_like`, `test_zeros_like` | Yes (`40c74406c`) |
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
| Stable generic clip for ReLU-difference saturation | `test_hardsigmoid_extreme` | No |
| Scaled-input reuse of the generated sigmoid LUT | `test_quick_gelu_extreme` | No |
| Generated tanh ranges with a stable local polynomial | `test_tanh`, `test_tanh_extreme` | No |
| Generated inverse-trigonometric tables and local/tail arithmetic | `test_asin`, `test_acos`, `test_atan` | Yes |
| Generic dynamic LOG2 multiplication and compact nested EXP2 special values | `test_exp2_log2_zero_times_negative` | Yes |

The wide-fill milestone writes the requested dtype directly through DPU WDMA;
there is no runtime narrowing or host semantic work. It also upgrades RKImage
dependencies to 64 bits and relocation indices to 32 bits, removing the image
serialization limits exposed by tiled fills and large constant payloads. These
focused passes are included in the 96-pass census above.

The DPU MAX operation is not yet claimed IEEE-exact for all NaN operand
orders. Hardware testing showed that finite extrema are correct and FP16
absolute value preserves the expected infinity, signed-zero, and NaN results;
special-value MAX behavior remains outside this milestone.
