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

Latest complete uncached census after infinity-safe WHERE, signed infinite division, prefix-repeat broadcast, and exact copysign:

The subsequent typed-cast milestone passes focused official `test_cast`; this gain is not folded into the census table until the next complete run.

| Status | Methods |
|---|---:|
| PASS | 167 |
| FAIL | 245 |
| SKIP | 13 |
| Collected | 425 |

Pytest reports `358 failed` because 113 failing unittest subtests are counted
in addition to their failed parent methods; 13 subtests pass. Runtime was 1087.02 seconds. This
exact run includes EXP2 special values, sigmoid/SiLU/Swish, QuickGELU, both
GELU forms, Erf, ELU/SELU, Mish, LogSigmoid, Softplus, Sinh/Cosh, Sqrt, RSqrt,
natural Exp, CELU α=1–4, Log2/Log/Log10, round-to-nearest-even, trunc, floor,
ceil, ASIN, ACOS, ATAN, SIN, ATANH, ASINH, ACOSH, IEEE predicates, all five mixed-dtype comparison methods, scalar `isclose`, logical-not,
dynamic and infinity-safe `WHERE`, square int32 WHERE transpose, exact maximum/minimum and copysign, zero-axis bool constants, signed infinite
division, and both suffix-tile and prefix-repeat broadcast ADD.

All five mixed-dtype `test_cmp_*` methods now pass FP16, exact signed int32, bool, same-shape/suffix-broadcast, scalar, reverse, and infinity cases.
Int32 values are never narrowed: four byte planes preserve all 32 bits and the sign-biased high byte makes unsigned lexicographic comparison signed-correct.

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
| Native round-to-nearest-even | typed 20-stage algorithm-23 plan | official method, exact dense sweep, ties, signed zero, infinity, and NaN | PASS |
| Integral rounding composition | one shared roundoff asset plus masks per plan | all three official trunc/floor/ceil methods and exact dense sweeps | PASS |
| Two-level ASIN | typed 43-stage broad/detail plan | official method, strict 4,097-point domain sweep, invalid inputs, and NaN | PASS |
| Regional ACOS | typed 47-stage asymmetric broad/coarse/fine endpoint plan | official method, strict 4,097-point sweep, invalid inputs, and NaN | PASS |
| Reciprocal-folded ATAN | typed 42-stage broad/detail plan | official method and strict 4,097-point `[-16,16]` sweep | PASS |
| FP16 SIN/COS | typed 56/59-stage broad/local plans | official FP16 SIN and seeded FP16 hardware SIN/COS contract | PASS; FP32 COS rejects |
| Bounded ATANH | typed 47-stage broad/detail plan | official method, strict 4,097-point domain sweep, endpoint infinities, invalid inputs, and NaN | PASS |
| Ranged ASINH | typed 46-stage core/range plan | official method and strict 4,097-point `[-32,32]` sweep | PASS |
| Endpoint-aware ACOSH | typed 43-stage core/range plan | official method, strict 4,097-point `[1,32]` sweep, exact endpoint, invalid inputs, and NaN | PASS |
| Typed bool output and IEEE predicates | versioned slot declaration plus native FP16 masks | official `isnan`, `isinf` directional modes, `isfinite`, and direct ABI pack checks | PASS |
| Generic FP16 comparisons | typed 31/32-stage plans with native NaN/infinity classification | all six relations over finite values, equal/opposite infinities, and NaNs | PASS |
| Lossless bool-input ABI | versioned input slot plus byte-to-FP16 0/1 widening | official bool/FP16 logical-not and direct odd-size ABI check | PASS |
| Exact int32 comparisons | four byte planes plus suffix-tile input declaration | all five official mixed-dtype methods and full-range int32 boundary vector | PASS |
| Exact int32 WHERE output | four NPU-written byte planes plus lossless reassembly | official `test_where` and full-range constant-arm boundary vector | PASS |
| Exact square int32 transpose | raw input byte planes, four NPU copy stages, and lossless reassembly | official `test_where_permute` and full-bit-pattern 5x5 transpose | PASS |
| Exact int32 extrema and mixed dtype | shared signed comparison DAG, raw selected byte planes, typed numeric widening | official `test_maximum`/`test_minimum` and full-range boundary vectors | PASS |
| Infinity-safe WHERE | threshold clamping plus reciprocal-generated signed infinity | official `test_inf_where`, `test_masked_fill`, and selected/unselected infinity vectors | PASS |
| Signed infinite division | infinite numerator becomes device multiply on finite nonzero denominator domain | official `test_div_naninf` and signed hardware vector | PASS |
| Prefix broadcast and exact copysign | distinct repeat/tile metadata plus typed FP16 sign-bit transport | both official copysign methods, two broadcast ADD methods, and layout/sign vectors | PASS |
| Typed casts | FP32 widening, numeric int input/output, bool widening/packing, and NPU truncation/predicate stages | all five official `test_cast` subcases and boundary vectors | PASS |
| Direct affine CMAC matmul | included in compiler suite | 1 | PASS |
| Constant-backed CMAC row sum | included in compiler suite | 1 | PASS |
| Explicit-layout PPU global max | included in compiler suite | 1 | PASS |
| Clean image/compiler suite total | 66 | 46 (plus 6 subtests) | PASS |

The host total is the collected total across `test/null/test_native_program.py`, `test/unit/test_rockchip_image.py`, and `test/unit/test_rockchip_compiler.py`. The device total is `test/device/test_rockchip.py`, run serially.

## Supported contracts

- dtype: FP16 expression graphs, lossless contiguous bool inputs/typed bool outputs, exact int32 comparisons/extrema/fill/copy, constant-arm WHERE
  outputs and square transpose, plus declared int32-to-FP16 mixed-extrema widening,
  tiled native int32/FP32 constant fills, bounded FP32 Sqrt/RSqrt input conversion, and experimental FP32 Log two-plane ABI;
- comparison: all six public FP16 relations preserve IEEE NaN and infinity semantics; plans exceeding 64 stages reject before image encoding;
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
- round-to-nearest-even uses the native algorithm-23 LUT and device-side sign/special-value masks;
- trunc/floor/ceil reuse round-to-nearest-even and correct direction with primitive device masks;
- ASIN uses broad and detail generated assets, with the second table sharing negative coordinates for center precision and positive endpoint distance for the singular region;
- ACOS uses an asymmetric signed broad table plus coarse and fine endpoint-distance tables; direct `pi/2-ASIN` composition is not part of the contract;
- ATAN folds `|x|>1` through reciprocal on-device, then uses broad/detail tables and reconstructs the sign with FP16 masks;
- ACOSH uses `x-1` endpoint coordinates, separate core/range tables, and device masks to preserve exact `acosh(1)=0` and return NaN below one;
- CMAC K=32 with directly legal memory, no host gather or pack;
- CMAC output width in the proven 4–16 range used by current tests/recognizer;
- PPU global max only for explicit `(K,8)` HWC-compatible storage and legal kernel split.

## Expected rejects

- FP32 expression/input graphs, integer arithmetic/outputs outside the declared extrema/fill/copy/WHERE contracts, uint8, boolean
  reductions/unsupported layouts, and gradients;
- noncontiguous elementwise indexing;
- WHERE graphs needing dynamic int32 arms and comparison graphs with non-suffix layouts; constant-arm int32 WHERE, square transpose/copy, and exact comparisons are supported;
- composed EXP2 graphs and values requiring overflow/underflow or NaN policy outside `[-2,2]`;
- unpacked/general matmul, batched matmul, arbitrary K, and host-required gather;
- spatial NCHW convolution without a device layout stage;
- windowed NCHW pooling and unsupported PPU kernel shapes;
- generic unsupported `Ops.*` graphs.

## Research freeze and next direction

The coverage branch is frozen at `815003d78`. Native round-to-nearest-even,
trunc, floor, ceil, IEEE predicates, mixed-dtype comparisons/extrema, bool
logical-not, constant-arm int32 WHERE, square int32 transpose, and typed casts
are retained here as hardware research. They are not the contract of the first
upstream-oriented branch.

Further raw TestOps recovery on this branch would next require dynamic int32
data flow and general movement, but that work is intentionally deferred. The
merge-oriented rewrite starts from current master with only contiguous FP16
arithmetic, mask/WHERE, one generic LUT stage and EXP2, plus a useful direct
contraction only if its layout contract remains small. FP32/int/bool ABI
experiments and the activation catalog remain on this frozen branch.

TAN is explicitly not in this category. The frozen branch computes it on the
host; clean native two-table experiments fit the image limit but fail strict
near-pole accuracy after FP16 range reduction. The retained WIP patch documents
the measured 78/60-stage SIN-COS quotient and 64-stage regional designs.

Two direct hardware probes of DPU FP16-mask to int8 output timed out: one kept
the proven eight-lane cube/WDMA geometry, and one used a 16-lane int8 atom.
Therefore direct byte-wide NPU bool output remains an honest unimplemented
boundary; the current research ABI only packs already-computed FP16 masks after
submission, and the TRM precision enum alone is not a valid hardware contract.

Scaled-log reuse for ATANH remains rejected: its first 61-stage form saturated
ratios above four, while correct two-band high-range normalization required
73 stages and violated the typed image's dependency limit. ATANH now passes
through distinct bounded broad/detail assets instead of reviving that path.

Spatial convolution is not low-hanging because it first needs a device-visible layout contract.

## Rules for future tally updates

1. Record branch SHA, test command, and environment.
2. Separate PASS, honest compile rejection, numerical mismatch, device timeout, and harness error.
3. Ensure root-cause counts sum to the total failed count.
4. Never turn a reject into a pass with runtime NumPy or `run_host` unless upstream backend policy explicitly allows semantic CPU fallback. This clean branch intentionally does not.
5. Run device tests serially and rerun any timeout in isolation after reset before classifying it.
