# Rockchip forward TestOps status

Last updated: 2026-08-01

## Meaning of this file

This status separates the frozen broad prototype from the `rockchip-2608` research backend. A rejection is not counted as a pass. The runtime has no CPU semantic fallback, so unsupported cases fail during compilation with `RKPLAN_REJECT:unsupported_graph`. Declared representation-only ABI conversions exist for Sqrt/RSqrt and experimental FP32 Log; all function semantics remain NPU task graphs. These FP32 experiments are not part of the planned minimal upstream FP16 branch.

Use:

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF ...
```

Gradients are intentionally out of scope until forward coverage is stable.

## Frozen prototype final result

The final `rockchip-2607` forward-only run at commit `1eb757cad` completed with
zero failures:

| Status | Methods |
|---|---:|
| PASS | 405 |
| FAIL | 0 |
| SKIP | 13 |
| Collected | 418 |

It additionally recorded 126 passing unittest subtests. The 13 skips were five
explicit manual-gradient methods outside `FORWARD_ONLY=1` and eight upstream
skips. This is the behavioral oracle target.

The current master version of `test_ops.py` collects 425 methods. The only
inventory addition relative to the 424-method final `rockchip-2607` tree is
`TestOps.test_softmin`; the older complete run collected 418 because six more
methods were added to `test_ops.py` between its base and current master.

## Historical prototype baseline

The earlier July 31 baseline, before the complete forward milestone, was:

| Status | Count |
|---|---:|
| PASS | 129 |
| FAIL | 287 |
| SKIP | 8 |
| Total | 424 |

The supplied root-cause table classified 176 failures: 94 unsupported ops, 22 unsupported dtypes, 42 numerical assertions, 6 timeouts, 6 uint8 overflow errors, 2 missing expected errors, 2 `NotImplementedError`, and 2 unsupported layouts. It did not account for the remaining 111 failed cases, so that table must not be presented as a complete partition.

That 129/287 tally is historical and must not be presented as the final
`rockchip-2607` result.

## Clean branch full baseline

Command:

```sh
CACHELEVEL=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
  python -m pytest test/backend/test_ops.py \
  -p test.rockchip.conftest_rockchip -q --tb=no
```

At `a9dd8e0da` plus the current-master-compatible forward plugin:

| Status | Methods |
|---|---:|
| PASS | 79 |
| FAIL | 333 |
| SKIP | 13 |
| Collected | 425 |

Pytest reports `459 failed` because the 333 failed methods contain 126 failing
unittest subtests. Runtime was 71.82 seconds. This is the honest clean-rewrite
parity baseline; the 18 host compiler tests and six focused hardware tests are
smoke/contract tests, not a replacement for this census.

## Clean branch current exact census

Latest complete uncached census after the logarithm milestone:

| Status | Methods |
|---|---:|
| PASS | 131 |
| FAIL | 281 |
| SKIP | 13 |
| Collected | 425 |

Pytest reports `407 failed` because 126 failing unittest subtests are counted
in addition to their failed parent methods. Runtime was 292.18 seconds. This
exact run includes EXP2 special values, sigmoid/SiLU/Swish, QuickGELU, both
GELU forms, Erf, ELU/SELU, Mish, LogSigmoid, Softplus, Sinh/Cosh, Sqrt, RSqrt,
natural Exp, CELU α=1–4, and Log2/Log/Log10.

## Focused verified matrix

| Group | Host compile checks | RK3588 checks | Status |
|---|---:|---:|---|
| Native hook | 1 | — | PASS |
| RKImage codec/validation/relocation, including 64-bit dependencies | 6 | — | PASS |
| DPU ADD/MUL/MAX/DIV/copy/fill/ABS/WHERE/composed masks and structural liveness | included in compiler suite | included | PASS |
| Generated variable-width EXP2 LUT plus IEEE epilogue | exhaustive 32,770 FP16 encodings in domain | 4 sizes and special values | PASS |
| Two-level HardSwish LUT | typed 36-stage plan | 1 dense sweep | PASS |
| Two-level tanh LUT and saturation | typed 35-stage plan | 1 dense sweep | PASS |
| Shared two-level sigmoid/SiLU/Swish | typed 24/25-stage plans | 1 dense local sweep | PASS |
| Dedicated two-level QuickGELU | typed 58-stage plan | normal/extreme official methods and 1 broad sweep | PASS |
| Dedicated tanh/exact GELU | two typed 51-stage plans | normal/extreme official methods and 2 broad sweeps | PASS |
| Dedicated two-level Erf | typed 44-stage plan | official normal/scalar/extremes and 1 strict dense sweep | PASS |
| Parameter-specialized ELU/SELU | three typed 35-stage plans | official methods and 3 dense sweeps | PASS |
| Asymmetric two-level Mish | typed 38-stage plan | strict official method and 1 ideal-curve sweep | PASS |
| Broad/tail LogSigmoid | typed 15-stage plan | official method and strict `[-12,12]` sweep | PASS |
| Parameterized Softplus | typed 27/27/8-stage plans | all official β/extreme/scalar cases and dense sweep | PASS |
| Direct Sinh/Cosh | typed 30/11-stage plans | normal and ±extreme official cases plus dense sweep | PASS |
| Refined Sqrt | typed 25-stage plan plus declared FP32 ABI input | official normal/zero/scalar cases, dense sweep, and IEEE specials | PASS |
| Refined RSqrt | typed 42-stage plan plus declared FP32 ABI input | official normal/zero/scalar cases, geometric sweep, and IEEE specials | PASS |
| Natural Exp | typed 36-stage broad/local plan | official normal/scalar/IEEE cases and dense sweep | PASS |
| CELU α=1–4 | typed 35/30-stage final-output plans | all official tensor/scalar cases and dense sweeps | PASS |
| Log2/Log/Log10 | typed 57-stage scale-specific plans; FP32 Log 61 stages | all official methods, measured dense sweep, and FP32 boundary | PASS |
| Direct affine CMAC matmul | included in compiler suite | 1 | PASS |
| Constant-backed CMAC row sum | included in compiler suite | 1 | PASS |
| Explicit-layout PPU global max | included in compiler suite | 1 | PASS |
| Clean image/compiler suite total | 46 | 26 (plus 6 subtests) | PASS |

The host total is the collected total across `test/null/test_native_program.py`, `test/unit/test_rockchip_image.py`, and `test/unit/test_rockchip_compiler.py`. The device total is `test/device/test_rockchip.py`, run serially.

## Supported contracts

- dtype: FP16 expression graphs, tiled native int32/FP32 constant fills, bounded FP32 Sqrt/RSqrt input conversion, and experimental FP32 Log two-plane ABI;
- mode: forward only;
- static shapes;
- DPU contiguous storage and one output;
- direct and composed EXP2 use one variable-width task, but only the finite `[-2,2]` domain is correct;
- HardSwish uses two generated LUT tasks with arithmetic/mask selection and no host fallback;
- tanh uses two generated LUT tasks with near-zero correction and exact `-1/+1` saturation tails;
- Sqrt uses one generated seed LUT, three NPU Newton refinements, and device special-value masks;
- RSqrt uses exact range scaling, one generated seed LUT, an NPU Newton refinement, and device special-value masks;
- natural Exp uses generated broad/local LUTs and device special-value masks;
- CELU α=1–4 uses ELU1 or direct final-output broad/local tables plus near-zero correction;
- Log2/Log/Log10 use scale-specific broad/local tables, exact power-of-16 normalization, and device special-value masks;
- CMAC K=32 with directly legal memory, no host gather or pack;
- CMAC output width in the proven 4–16 range used by current tests/recognizer;
- PPU global max only for explicit `(K,8)` HWC-compatible storage and legal kernel split.

## Expected rejects

- FP32 expression/input graphs, integer arithmetic/input graphs, uint8, bool, and gradients;
- noncontiguous elementwise indexing;
- user-visible bool comparisons and WHERE graphs needing bool/int inputs or non-FP16 outputs;
- composed EXP2 graphs and values requiring overflow/underflow or NaN policy outside `[-2,2]`;
- unpacked/general matmul, batched matmul, arbitrary K, and host-required gather;
- spatial NCHW convolution without a device layout stage;
- windowed NCHW pooling and unsupported PPU kernel shapes;
- generic unsupported `Ops.*` graphs.

## Next low-hanging group

The FP16 comparison mask and WHERE graph are implemented. The next low-hanging
boundary is device-native bool/int representation:

- emit byte-wide user-visible comparison results without CPU packing;
- ingest bool conditions and int32 operands through an explicit NPU layout stage;
- reuse those typed boundaries for the leading `test_where`, comparison,
  `maximum`/`minimum`, and `*_like` variants;
- reject dynamic casts until packing is proven, as native int32 WDMA consumes
  eight FP16 lanes while producing four int32 lanes.

Spatial convolution is not low-hanging because it first needs a device-visible layout contract.

## Rules for future tally updates

1. Record branch SHA, test command, and environment.
2. Separate PASS, honest compile rejection, numerical mismatch, device timeout, and harness error.
3. Ensure root-cause counts sum to the total failed count.
4. Never turn a reject into a pass with runtime NumPy or `run_host` unless upstream backend policy explicitly allows semantic CPU fallback. This clean branch intentionally does not.
5. Run device tests serially and rerun any timeout in isolation after reset before classifying it.
