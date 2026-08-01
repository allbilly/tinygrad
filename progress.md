# Rockchip clean rewrite progress

Last updated: 2026-08-01

## TestOps parity correction

The size/architecture rewrite is complete, but behavioral parity is not. The
earlier 129-pass/287-fail table was an intermediate `rockchip-2607` baseline,
not its final state. Oracle commit `1eb757cad` records zero forward-only
failures: 405 passed and 13 intentional skips across its then-current 418
collected methods, plus 126 passing subtests.

Current master collects 425 methods (it adds `test_softmin` relative to the
424-method oracle inventory). The first uncached clean-branch census with the
ported forward contract was 79 passed, 333 failed, and 13 skipped. After the
typed extrema, WHERE mask, division, ABS, copy, scalar-fill, and native wide-fill
milestones through exact int32 extrema and typed mixed-dtype widening, the 2026-08-01 census is
**159 passed, 253 failed, and 13 skipped**. Pytest prints `375 failed` because
it separately counts 122 failing subtests; four subtests now pass.

The clean branch must preserve its `<5000` handwritten-line target
while recovering the remaining native forward coverage. Focused 64-host/44-NPU
tests prove only the implemented compiler contracts and must not be described
as full TestOps completion.

## Branch and recovery points

- Oracle branch: `rockchip-2607`
- Frozen oracle tag: `rockchip-2607-frozen-20260801`
- Frozen oracle commit: `51b4f919e`
- Clean branch: `rockchip-2608`
- Clean worktree: `/home/orangepi/rk_2608`
- Clean base: `277433259eb71b5fc3d6d5cc33c5a1be1458e9fa` (`master` and `upstream/master` when the branch was created)
- Python environment: `. /home/orangepi/tinygrad/.venv/bin/activate`
- Required test context: `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF`

The old branch is an oracle only. No old Rockchip WIP was deleted or rewritten. Each clean milestone is also archived as a mail-formatted patch in `/home/orangepi/tinygrad`.

## Research branch versus upstream scope

`rockchip-2608` is now explicitly the research/coverage branch, not a proposed
single upstream PR. Its activation catalog, experimental FP32 two-plane ABI,
int/FP32 fills, PPU probe, progress files, and full TestOps plugin are retained
to characterize RK3588 and recover the frozen oracle without losing work.

A future merge-oriented branch must start from current master and carry only a
small honest FP16 contract. Its target stage IR should use generic `Ops` for ALU
semantics, a distinct mask stage, and a generic LUT stage whose generated asset
identity is data rather than one of many pseudo-opcodes. It should reject FP32,
integer, bool, unsupported layouts, and unimplemented contractions; remove the
runtime NumPy ABI experiment; include only hardware-native features required by
a useful end-to-end workload; and satisfy the repository's 25,000-line cap.
The 425-method census remains informational and must not dictate that upstream IR.

## Completed milestones

| Commit | Capability | Patch archive |
|---|---|---|
| `d72ef73b2` | Versioned typed RKImage ABI and default-no-op native renderer hook | `0159-rockchip-add-typed-image-ABI-and-native-hook.patch` |
| `7daaad820` | UOp-free typed DPU expression DAG, liveness, scratch reuse | `0160-rockchip-lower-typed-DPU-expression-graphs.patch` |
| `578b10b51` | Thin RK3588 DRM/GEM runtime and blocking task submission | `0161-rockchip-add-thin-RK3588-runtime.patch` |
| `9a6325415` | Direct legal-layout FP16 CMAC contraction | `0162-rockchip-lower-direct-affine-CMAC-contracts.patch` |
| `65f13e9f5` | Explicit-layout PPU global max | `0163-rockchip-lower-explicit-layout-PPU-max.patch` |
| `93dc9e35f` | Offline-generated EXP2 LUT and one generic LUT stage | `0164-rockchip-add-generated-EXP2-LUT-stage.patch` |
| `1d3acac82` | Row sums lowered to constant-backed CMAC contracts | `0165-rockchip-lower-row-sums-as-CMAC-contracts.patch` |
| `65acc1858` | Exhaustive FP16-domain LUT simulation and error metadata | `0166-rockchip-verify-generated-LUT-error-bounds.patch` |
| `a9dd8e0da` | Clean rewrite/LUT design documentation | `0167-rockchip-document-clean-rewrite-and-LUT-tuning.patch` |
| `70f621d04` | Full forward TestOps census contract | `0168-rockchip-restore-full-forward-TestOps-census.patch` |
| `5b9a088cf` | 32-bit RKImage constant relocation indices | `0169-rockchip-widen-RKImage-constant-relocations.patch` |
| `9b4479640` | Ordered WHERE extrema normalization | `0170-rockchip-normalize-ordered-WHERE-extrema.patch` |
| `04f53c58a` | Scalar DPU fills | `0171-rockchip-lower-scalar-DPU-fills.patch` |
| `3183a307b` | Generic FP16 WHERE masks | `0172-rockchip-lower-generic-half-WHERE-masks.patch` |
| `5829aaf48` | Fused typed division and ADD-zero copy | `0173-rockchip-fuse-typed-DPU-division.patch` |
| `fa22d6764` | Native ABS canonicalization | `0174-rockchip-canonicalize-native-absolute-value.patch` |
| `89a5cffbc` | Tiled native int32 fills and 64-bit image dependencies | `0175-rockchip-tile-native-int32-fills.patch` |
| `f49f99acf` | Tiled native FP32 constant fills | `0176-rockchip-tile-native-FP32-fills.patch` |
| `3f763ea7b` | Composed FP16 predicates and structural DAG liveness | `0177-rockchip-compose-native-FP16-predicates.patch` |
| `9278751cd` | Ordered clamp canonicalization for stable saturation | `0178-rockchip-canonicalize-stable-ordered-clamp.patch` |
| `b34d5f283` | Partial and multi-tile EXP2 LUT launches | `0179-rockchip-tile-generated-EXP2-LUT.patch` |
| `d7fce428a` | Variable-width LUT tasks and two-level HardSwish | `0180-rockchip-add-two-level-HardSwish-LUT.patch` |
| `9baa14d7d` | Two-level tanh with device saturation | `0181-rockchip-add-two-level-tanh-LUT.patch` |
| `228e2b51a` | Direct EXP2 IEEE special-value epilogue | `0182-rockchip-handle-EXP2-special-values.patch` |
| `a1a966fe8` | Shared two-level sigmoid for sigmoid/SiLU/Swish | `0183-rockchip-add-two-level-sigmoid-LUT.patch` |
| `91f57be47` | Dedicated two-level QuickGELU with bounded tails | `0184-rockchip-add-two-level-QuickGELU-LUT.patch` |
| `1a1e069f3` | Dedicated two-level tanh/exact GELU | `0185-rockchip-add-two-level-GELU-LUTs.patch` |
| `d93ef27e4` | Dedicated two-level Erf with exact signed tails | `0186-rockchip-add-two-level-Erf-LUT.patch` |
| `30f2bf666` | Parameter-specialized two-level ELU/SELU | `0187-rockchip-add-two-level-ELU-LUTs.patch` |
| `bf260860a` | Asymmetric two-level Mish | `0188-rockchip-add-two-level-Mish-LUT.patch` |
| `865e1201f` | Broad plus amplified-tail LogSigmoid | `0189-rockchip-add-two-level-LogSigmoid-LUT.patch` |
| `6157fcb05` | Parameterized Softplus with LUT post-scaling | `0190-rockchip-add-Softplus-LUTs.patch` |
| `b304b9204` | Direct Sinh/Cosh LUTs with infinite tails | `0191-rockchip-add-Sinh-Cosh-LUTs.patch` |
| `5d0a9360a` | Exact 124-pass TestOps census | `0192-rockchip-record-124-pass-TestOps-census.patch` |
| `705877b63` | Refined Sqrt LUT and bounded FP32 input ABI | `0193-rockchip-add-refined-Sqrt-LUT.patch` |
| `a1c8090c4` | Range-scaled refined RSqrt LUT | `0194-rockchip-add-refined-RSqrt-LUT.patch` |
| `41ba2345c` | Broad/local natural Exp LUTs | `0195-rockchip-add-natural-Exp-LUTs.patch` |
| `ddb9d0733` | Rejected int fill probe retained for reference | `0196-rockchip-record-rejected-int-fill-probe.patch` |
| `bb115b4a3` | Direct final-output CELU LUTs | `0197-rockchip-add-CELU-LUTs.patch` |
| `730efd61e` | Scale-specific Log2/Log/Log10 LUTs and experimental typed FP32 ABI | `0198-rockchip-add-logarithm-LUTs.patch` |
| `1f4cc5da8` | Native round-to-nearest-even algorithm-23 LUT | `0199-rockchip-add-native-roundoff-LUT.patch` |
| `e6a88f6c1` | Separate hardware stage operations from generated LUT identities | `0200-rockchip-separate-LUT-assets-from-stage-ops.patch` |
| `29559149c` | Compose trunc/floor/ceil from native roundoff and masks | `0201-rockchip-compose-native-integral-rounding.patch` |
| `4d174db23` | Preserve rejected scaled-log high-range normalization probe | `0202-rockchip-record-rejected-scaled-log-probe.patch` |
| `a2fc51580` | Native two-level ASIN LUT with device-only regional composition | `0203-rockchip-add-two-level-ASIN-LUT.patch` |
| `81bd7960c` | Regional ACOS LUTs with fine endpoint interpolation | `0204-rockchip-add-regional-ACOS-LUTs.patch` |
| `26951cf4e` | Reciprocal-folded two-level ATAN LUT | `0205-rockchip-add-reciprocal-folded-ATAN-LUT.patch` |
| `b5a71b29f` | FP16 SIN/COS regional LUTs with split periodic reduction | `0206-rockchip-add-FP16-SIN-COS-LUTs.patch` |
| `76e4131f1` | Bounded broad/detail ATANH with device special-value handling | `0207-rockchip-add-bounded-ATANH-LUTs.patch` |
| `3b51f1b65` | Ranged two-table FP16 ASINH | `0208-rockchip-add-ranged-ASINH-LUTs.patch` |
| `2da53e486` | Endpoint-aware two-table FP16 ACOSH | `0209-rockchip-add-endpoint-ACOSH-LUTs.patch` |
| `2651579d8` | Preserve and characterize rejected native TAN designs | `0210-rockchip-record-native-TAN-limits.patch` |
| `091df1bd8` | Typed bool-output ABI and native IEEE predicates | `0211-rockchip-add-typed-bool-output-ABI.patch` |
| `bd1b8009a` | IEEE-correct generic FP16 comparison roots | `0212-rockchip-lower-IEEE-FP16-comparisons.patch` |
| `25df366fb` | Lossless typed bool-input ABI widening | `0213-rockchip-add-lossless-bool-input-ABI.patch` |
| `4e2931c2c` | Exact int32 comparison byte planes and suffix broadcast | `0214-rockchip-add-exact-int32-comparisons.patch` |
| `297b6fdc0` | Exact int32 WHERE output planes and general suffix tiling | `0215-rockchip-add-exact-int32-WHERE-output.patch` |
| `04dd21e23` | Exact square int32 transpose/copy through raw byte planes | `0216-rockchip-add-exact-int32-transpose.patch` |
| `8e75af4a8` | Exact int32 extrema, copy/fill, and typed mixed-dtype widening | `0217-rockchip-add-exact-int32-extrema.patch` |
| `f02b8c970` | Infinity-safe threshold WHERE and masked fill | `0218-rockchip-lower-infinity-safe-WHERE.patch` |
| current milestone | Signed infinite-numerator division | `0219-rockchip-preserve-infinite-division-sign.patch` |

## Architecture now implemented

The compiler boundary is:

```text
post-early_simplify UOps
  -> semantic recognition and affine analysis
  -> immutable typed Rockchip plans (last point that can contain UOps is before this arrow)
  -> engine command emission and relocations
  -> deterministic RKImage bytes
  -> runtime decode, allocation, patch, reset, submit, wait
```

Important invariants:

- `RKDPUProgram`, `RKContract`, `RKPool`, `RKStage`, and `RKImage` retain no `UOp`.
- `RKDPUOp` contains only eight hardware stage operations; the 67 generated table identities live separately in `RKLUT` and are data on one `LUT` stage.
- RKImage version 9 contains only target/version data, command stages, dependencies, relocations, scratch declarations, constants, and typed ABI slot declarations.
- Runtime NumPy is restricted to declared ABI element-format conversion and never executes tensor semantics on the CPU.
- Unsupported dtype, layout, or graph combinations reject before device submission.
- Scratch allocation is liveness-aware and can reuse a dead intermediate in place.
- LUT fitting and exhaustive verification live in `extra/rockchip`; runtime imports only generated immutable metadata/data.
- Hardware tests are serialized because parallel NPU reset/submission pollutes shared device state.

Implemented forward-only subset:

- contiguous copy and fill;
- ADD, MUL, MAX, DIV, ABS, ordered extrema, and generic FP16 WHERE expression DAGs;
- CMPLT, CMPNE, OR, and AND composition into internal FP16 masks, with structural common-expression liveness;
- stable ordered-clamp recovery from `relu(x+0.5)-relu(x-0.5)`, avoiding large-value FP16 cancellation;
- scalar operands materialized as declared RKImage constants;
- EXP2 in one variable-width DPU LUT task over the generated `[-2, 2]` domain;
- direct EXP2 device masks/divisions restore `+inf`, `0`, and `NaN` for `+inf`, `-inf`, and `NaN` inputs;
- HardSwish using the oracle-proven Q14 broad LUT, arithmetic outer fallback, and Q15 near-zero LUT correction, all selected on the NPU;
- tanh using the oracle-proven Q15 broad/local LUTs, identity correction near zero, and exact device-side saturation outside `[-4,4]`;
- sigmoid using Q15 broad/local LUTs and device saturation, reused directly by SiLU and Swish;
- QuickGELU using dedicated Q14 broad/Q15 negative-local LUTs, a near-zero polynomial, and bounded shared-sigmoid tails;
- tanh and exact GELU using separate Q15 broad/local LUTs, near-zero polynomials, and exact zero/x tails;
- Erf using Q15 broad/local LUTs, a near-zero linear correction, and exact signed tails;
- ELU alpha 1/0.1 and SELU using six generated negative-branch tables and one reusable typed schedule;
- Mish using an asymmetric broad table, central local table, near-zero polynomial, and exact tails;
- LogSigmoid using a broad correction and 32x amplified positive-tail table;
- Softplus beta 1, 3, and 1/3 using generated corrections, amplified tails, and Q15 LUT post-scaling;
- Sinh/Cosh using direct central LUTs, local Sinh correction, and device-generated infinite tails;
- Sqrt using a Q14 seed, three NPU Newton steps, and device masks for zero, infinity, negative input, and NaN;
- bounded FP32 Sqrt/RSqrt inputs narrowed to FP16 only at the runtime ABI boundary, with all function semantics remaining in the NPU program;
- RSqrt using exact power-of-16 input scaling, a dedicated Q13 seed, one NPU Newton step, output rescaling, and IEEE masks;
- natural Exp using asymmetric broad and direct local tables plus device-generated IEEE special values;
- CELU α=1–4 using ELU1 or direct final-output broad/local tables and a near-zero polynomial;
- Log2/natural Log/Log10 using scale-specific broad/local LUTs, power-of-16 normalization, near-one correction, and IEEE masks;
- experimental FP32 natural Log using atom-aligned `hi/lo` FP16 input planes and declared FP16-to-FP32 output widening;
- round-to-nearest-even using the RK3588 algorithm-23 LUT, with NPU masks preserving sign, infinity, and NaN;
- trunc, floor, and ceil composed from the same roundoff asset plus primitive DPU comparison masks;
- ASIN on the declared FP16 `[-1,1]` domain using one broad and one shared center/endpoint detail LUT, with device masks for sign and invalid inputs;
- ACOS on `[-1,1]` using asymmetric broad, endpoint-distance, and fine-endpoint LUT assets; `pi/2-ASIN` was measured and rejected rather than hidden by relaxed tolerance;
- ATAN using device-side reciprocal magnitude folding and broad/detail LUT selection over a bounded `[0,1]` coordinate;
- FP16 SIN/COS using split periodic reduction, broad/local LUTs, and device-generated NaN for non-finite inputs; plugin-forced FP32 COS remains rejected;
- ATANH using distinct broad/detail assets, endpoint-distance addressing, device infinity at `±1`, and NaN outside the domain;
- ASINH using two physical tables shared across center, broad, middle, and large magnitude regions with device sign reconstruction;
- ACOSH using endpoint-coordinate core/range tables, device regional composition, exact one handling, and NaN synthesis below the domain;
- `isnan`, `isinf` (including directional modes), and `isfinite` using native FP16 masks, with a declared bool-output slot converted from the
  atom-padded FP16 NPU representation to the public byte ABI after submission;
- all six generic FP16 comparison roots with explicit NaN and signed-infinity classification; boolean inversion is lowered structurally rather
  than assuming that `not (x<y)` is IEEE `x>=y` in the presence of NaN;
- lossless public byte-bool input widening to atom-padded FP16 0/1 DMA temporaries, with logical-not evaluated by the NPU and packed through the
  typed bool-output ABI;
- exact signed-int32 comparison through four unsigned-byte FP16 planes with a sign-biased most-significant byte, plus a narrow declared suffix-tile
  layout for `(N,...,K)` versus `(K,)` comparison broadcasts;
- exact dynamic int32 constant-arm WHERE output through four NPU-written byte planes and lossless runtime reassembly; scalar/suffix FP16 inputs can
  be tiled for ordinary FP16 outputs as well as bool comparison outputs;
- exact square int32 transpose/copy through a versioned RKImage layout declaration: the runtime performs only representation/layout conversion
  before raw byte-plane encoding, four NPU ADD-copy stages write the output planes, and runtime reassembly preserves every bit;
- exact signed-int32 maximum/minimum through one shared lexicographic comparison DAG and four raw selected output planes, plus exact raw byte-plane
  fill/copy, typed scalar tiling, bool-to-int output, and declared int32-to-FP16 numeric ABI widening for mixed-dtype extrema;
- infinity-safe threshold WHERE: device min/max clamping and reciprocal-generated signed infinity avoid `0*inf` in both selected and unselected arms;
- infinite-numerator division lowered to device multiplication on the tested finite nonzero domain, preserving the denominator sign lost by RK3588 DIV;
- directly legal `A @ packed_B.T`, currently `A=(1,32)` and `packed_B=(N,32)` for proven output widths;
- row sum for `(N,32)`, implemented as the same CMAC contract with an image-owned FP16 ones vector;
- global MAX over explicitly HWC-compatible `(K,8)` input layouts supported by the PPU kernel constraints.
- native int32 constant fills, split into the proven 64-output DPU tile with typed destination offsets.
- native FP32 constant fills, split into the proven four-output/64-byte source tile.

## Size result

`python sz.py` reports:

```text
tinygrad/renderer/rockchip.py  1899
tinygrad/runtime/ops_rockchip.py  174
handwritten Rockchip total  2073
```

This meets the requested `<5000` research-backend goal with 2,927 lines of headroom. The generated register and LUT modules are mechanically generated and are excluded by `sz.py`.

Compared with the frozen implementation, the runtime is thin and the UOp-free
plan/image boundary is preserved, but this research branch has again accumulated
an activation catalog. Approximate current physical source distribution is:

- target types and DPU expression recipes: renderer lines 18–484;
- RKImage validation, codec, and relocation: renderer lines 485–564;
- UOp canonicalization, affine analysis, and typed lowering: renderer lines 565–1156;
- register emission: renderer lines 1157–1413;
- renderer integration: renderer lines 1414 onward;
- allocation/submission/runtime ABI experiments: 174 `sz.py` lines.

This distribution is acceptable only for the frozen research/coverage branch.
The future upstream branch must replace the catalog with generic `Ops` ALU,
mask, and LUT stages and include only the minimal assets required by its declared
FP16 workload.

The whole repository is 27,049 `sz.py` lines, a `+2,081` delta from the 24,968-line base. Therefore `MAX_LINE_COUNT=25000 python sz.py` fails globally by 2,049 lines even though the research backend itself is below 5,000. This is an explicit blocker for upstream submission and must be resolved by constructing a minimal branch, not hidden through unrelated compression or generated files.

## Exact validation commands

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest \
  test/null/test_native_program.py \
  test/unit/test_rockchip_image.py \
  test/unit/test_rockchip_compiler.py -q -n12
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest test/device/test_rockchip.py -q
python -m ruff check .
python -m mypy tinygrad/
python sz.py
```

Host compiler tests may use `-n12`. Device tests must remain serial.

## Debugging method

Use this order when adding or diagnosing a path:

1. Reduce the failure to one forward-only FP16 expression with a static shape.
2. Capture the post-`early_simplify` sink and inspect RANGE, INDEX, REDUCE, STORE, dtype, slots, and affine coefficients.
3. Decide legality before emission. Reject layouts requiring gather, packing, channel padding, or dtype conversion unless an explicit typed stage or bounded ABI conversion exists.
4. Assert the typed plan recursively contains no `UOp`.
5. Compare normalized command words with the frozen oracle. Addresses must be zero placeholders at this point.
6. Validate every relocation by stage, command-word index, buffer kind/index, addend, source mask, and destination field shift.
7. Round-trip `encode_image -> decode_image -> encode_image` and demand byte equality.
8. Run the single hardware case serially. Reset before each stage and use blocking submit.
9. Only after the isolated case passes, run the complete device file repeatedly to detect stale state.
10. Run Ruff, mypy, host tests, device tests, `git diff --check`, and `sz.py` before committing.

Useful failure interpretations:

- `RKPLAN_REJECT:unsupported_graph`: semantic/layout contract did not match; inspect UOps and affine coefficients, not the runtime.
- numerical mismatch with a completed submission: compare command order, dimensions, strides, conversion fields, and relocation patching.
- timeout/error 110: reset and rerun one test serially; if reproducible, bisect register order and task metadata.
- wrong address bits: decode the 64-bit command as target/value/register and inspect relocation `shift`, `mask`, then `field_shift` in that order.
- correct first run, wrong later run: stale lanes or state pollution; verify per-stage reset and output tile coverage.
- LUT mismatch: regenerate, verify payload SHA, run exhaustive simulator, then run a hardware sweep inside the declared domain.

Recent precision probes that must not be rediscovered as final fixes:

- Do not recognize `isinf`/`isfinite` by scanning an expression for infinity constants: `isclose` contains the same constants and was falsely
  accepted as `isinf`. Match the exact nested `CMPNE(..., true)`/`OR` structure and require every atom to share the same direct FP16 index.
- The raw DPU `MASK(rhs-lhs)` comparison reports true for `inf-inf` and NaN differences. Public comparison roots must classify NaN, positive
  infinity, negative infinity, and finite values on-device; in particular, IEEE `x>=y` is not `not (x<y)` when either operand is NaN.
- DPU `out_precision=0` int8 output from the proven FP16 mask task timed out with both eight-lane and 16-lane WDMA/data-cube layouts. The TRM advertises int8 output, but FP16-to-int8 conversion is not yet a proven byte-wide bool ABI; do not enable it from the format field alone.
- Arbitrary scaled LOG2 made atanh compile in 61 stages, but the `(1+x)/(1-x)` ratio reaches 199 and saturated above 4. Symmetric `>4`/`>64` power-of-16 normalization fixed the range but expanded direct log/atanh to 69/73 stages, beyond RKImage's 64-bit dependency contract. Fuse stages or introduce a justified wide-domain asset; do not relax the image invariant.
- Frozen `rockchip-2607` does not provide a native TAN oracle: its final path is `_run_host_tan`, which calls NumPy. The clean shared SIN/COS quotient first required 78 stages; common reduction and hierarchical selection reached 60 stages but still missed 73/2,925 values at strict tolerance because the two LUT results round independently.
- A two-task `TAN_CORE`/`TAN_EDGE` regional design reached 64 stages and reduced the first `[-1.5,1.5]` method to five boundary misses. Amplified signed pole-distance variants passed that first method, but the `[-5,5]` method retained 18–22 near-pole misses; storing the reduced angle in FP16 loses the distance to odd multiples of `pi/2`.
- A stable `d/tan(d)` correction avoids table quantization at the reciprocal pole, but direct pole-distance reduction made 752/2,925 strict misses away from poles, and combining it with the regional core reached 65 stages before optimization and remained inaccurate. The complete experimental diff is preserved at `/home/orangepi/tinygrad/tan-native-wip-20260801.patch`; no failed TAN path is enabled in the green branch.
- LogAddExp cannot be recovered merely by removing the obsolete plugin FP32 promotion. A native FP16 Softplus composition used 32 stages but missed
  196 comparisons; the numerically preferable absolute-difference form still missed 110. A dedicated finer two-LUT correction missed 113 because
  subtracting close FP16 inputs loses information before the fused reference operation. The complete probe is preserved at
  `/home/orangepi/tinygrad/logaddexp-native-wip-20260801.patch`; no inaccurate path is enabled.

- `hardswish` baseline (`(x*clamp)*half(1/6)`) mismatched 34/2925 values;
- one-task BS pre-scaling reduced that to 9/2925, but still missed the official tolerance;
- `x*(clamp/6)` reduced it to 4/2925; `(x*clamp)/6` produced 13/2925;
- the adjacent upper FP16 BS coefficient produced 92/2925 mismatches; Q16 and Q12 `OUT_CVT` scale probes respectively overflowed and divided by `MINUS_EXP` without applying the assumed numerator;
- the final passing path uses `rockchip-2607`'s Q14 broad/Q15 local two-LUT policy, reduced to 36 typed stages by variable-width LUT tasks;
- raw EXP2 LUT overflow handling mapped `[+inf,-inf,nan]` to `[8,0.25,8]`; the committed device-mask epilogue now restores `[inf,0,nan]`.
- reusing the sigmoid LUT for QuickGELU fixed extreme inputs but left 116/2,925 normal mismatches; dedicated 2607-proven Q14/Q15 tables reduced the official normal and extreme methods to zero failures.

No-host-semantic-fallback audit:

```sh
rg -n '_HOST_|run_host|host.*layout' tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py
```

Allocator `copyin`/`copyout`, the declared FP32-to-FP16 Sqrt/RSqrt input conversion, the experimental Log `hi/lo` plane encode/output widening,
lossless byte-bool input widening, exact int32 byte-plane encoding/reassembly, declared square-transpose/scalar-tiling layout conversion,
int32-to-FP16 numeric widening for mixed extrema, and packing
NPU-computed FP16 masks into public bool bytes are ABI transport, not semantic CPU execution. The runtime may use NumPy only for these
representation conversions; it never evaluates a tensor function or predicate on the host. These experiments are research-only and are excluded
from the planned minimal upstream branch.

## Explicitly pending

- Spatial direct convolution. Current tinygrad NCHW storage does not match the proven NPU atom/HWC surface. The frozen branch used host materialization for broad contraction/conv cases, so it is not a valid clean implementation source. A future path needs a typed `RKConv` plus an explicit device layout/weight stage. A 1x1 `1x32 -> 8` convolution canonicalizes to the already-supported affine CMAC contract and cannot be distinguished semantically after simplification.
- Wider affine/batched contractions and CMAC tiling.
- General int32 arithmetic/dynamic WHERE arms, non-square int32 movement, non-suffix broadcast layouts, and broader boolean reductions. Int32
  comparisons, extrema, constant-arm WHERE outputs, fills, copies, and square transpose are exact; contiguous
  byte-bool inputs are losslessly widened, and generic FP16 comparison/IEEE predicate outputs use the public bool-output ABI.
- LOG2, reciprocal, SIN, and additional generated LUT identifiers.
- Windowed pooling until layout/channel padding is expressed on device.
- General FP32 expressions/inputs, boolean reductions/layouts, general integer inputs/arithmetic/casts, and gradient support. Native FP32/int32 constant fills,
  bounded FP32 Sqrt/RSqrt input conversion, experimental FP32 Log two-plane ABI, and typed IEEE-predicate bool outputs are the only current exceptions.
- Multicore/program-chain submission beyond the stable reset-separated single-core task sequence.
- Broad TestOps parity. Unsupported graphs deliberately reject rather than run on the CPU.

The next feature should be selected by hardware-native leverage, not raw failed-test count. FP16 masks, IEEE-correct comparison roots, and a
representation-only public bool ABI, exact int32 comparison inputs, constant-arm int32 WHERE outputs, and square transpose/copy now exist.
`test_where_permute`, `test_maximum`, and `test_minimum` pass. The next boundary is dynamic int32 WHERE arms and general int32 movement for further integer operators. Direct device
byte output remains unproven; spatial
convolution should wait for an explicit device layout design.
