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
| IEEE special-value wrapper around EXP2 LUT | `test_exp2` | No |

The wide-fill milestone writes the requested dtype directly through DPU WDMA;
there is no runtime narrowing or host semantic work. It also upgrades RKImage
dependencies to 64 bits and relocation indices to 32 bits, removing the image
serialization limits exposed by tiled fills and large constant payloads. These
focused passes are included in the 96-pass census above.

The DPU MAX operation is not yet claimed IEEE-exact for all NaN operand
orders. Hardware testing showed that finite extrema are correct and FP16
absolute value preserves the expected infinity, signed-zero, and NaN results;
special-value MAX behavior remains outside this milestone.
