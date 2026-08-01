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
milestones through dedicated two-level GELU variants, the 2026-08-01 census is
**116 passed, 296 failed, and 13 skipped**. Pytest prints `422 failed` because
it separately counts 126 failing subtests.

Standalone Erf, ELU/SELU, and Mish are additionally green, giving a current validated delta of
**120 passed, 292 failed, and 13 skipped** without relabeling it as a new full
census. The clean branch must preserve its `<5000` handwritten-line target
while recovering the remaining native forward coverage. Focused 37-host/18-NPU
tests prove only the implemented compiler contracts and must not be described
as full TestOps completion.

## Branch and recovery points

- Oracle branch: `rockchip-2607`
- Frozen oracle tag: `rockchip-2607-frozen-20260801`
- Frozen oracle commit: `51b4f919e`
- Clean branch: `rockchip-2608`
- Clean worktree: `/tmp/rk_2608`
- Clean base: `277433259eb71b5fc3d6d5cc33c5a1be1458e9fa` (`master` and `upstream/master` when the branch was created)
- Python environment: `. /home/orangepi/tinygrad/.venv/bin/activate`
- Required test context: `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF`

The old branch is an oracle only. No old Rockchip WIP was deleted or rewritten. Each clean milestone is also archived as a mail-formatted patch in `/home/orangepi/tinygrad`.

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
| current milestone | Asymmetric two-level Mish | `0188-rockchip-add-two-level-Mish-LUT.patch` |

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
- RKImage contains only target/version data, command stages, dependencies, relocations, scratch declarations, and constants.
- Runtime code does not import NumPy and does not execute tensor semantics on the CPU.
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
- directly legal `A @ packed_B.T`, currently `A=(1,32)` and `packed_B=(N,32)` for proven output widths;
- row sum for `(N,32)`, implemented as the same CMAC contract with an image-owned FP16 ones vector;
- global MAX over explicitly HWC-compatible `(K,8)` input layouts supported by the PPU kernel constraints.
- native int32 constant fills, split into the proven 64-output DPU tile with typed destination offsets.
- native FP32 constant fills, split into the proven four-output/64-byte source tile.

## Size result

`python sz.py` reports:

```text
tinygrad/renderer/rockchip.py  950
tinygrad/runtime/ops_rockchip.py  74
handwritten Rockchip total  1024
```

This meets the requested `<5000` backend goal with 3,976 lines of headroom. The generated register and LUT modules are mechanically generated and are excluded by `sz.py`.

Compared with the frozen implementation, the dominant 77.5% task/graph-lowering catalog was replaced by three bounded recognizers and one primitive DAG scheduler. Approximate physical source distribution is now:

- RKImage types, codec, validation, relocation: renderer lines 18–192;
- semantic analysis and typed lowering: renderer lines 193–334;
- register emission: renderer lines 335–475;
- renderer integration: renderer lines 476–485;
- allocation/submission runtime: 85 physical lines, 74 `sz.py` lines.

The whole repository is 26,000 `sz.py` lines, a `+1,032` delta from the 24,968-line base. Therefore `MAX_LINE_COUNT=25000 python sz.py` still fails globally by 1,000 lines even though the backend itself is well below 5,000. Fixing that would require an upstream cap decision or unrelated repository reductions; no unrelated master code was compressed to disguise this backend cost.

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
3. Decide legality before emission. Reject layouts requiring gather, packing, channel padding, or dtype conversion unless an explicit device stage exists.
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

- `hardswish` baseline (`(x*clamp)*half(1/6)`) mismatched 34/2925 values;
- one-task BS pre-scaling reduced that to 9/2925, but still missed the official tolerance;
- `x*(clamp/6)` reduced it to 4/2925; `(x*clamp)/6` produced 13/2925;
- the adjacent upper FP16 BS coefficient produced 92/2925 mismatches; Q16 and Q12 `OUT_CVT` scale probes respectively overflowed and divided by `MINUS_EXP` without applying the assumed numerator;
- the final passing path uses `rockchip-2607`'s Q14 broad/Q15 local two-LUT policy, reduced to 36 typed stages by variable-width LUT tasks;
- raw EXP2 LUT overflow handling mapped `[+inf,-inf,nan]` to `[8,0.25,8]`; the committed device-mask epilogue now restores `[inf,0,nan]`.
- reusing the sigmoid LUT for QuickGELU fixed extreme inputs but left 116/2,925 normal mismatches; dedicated 2607-proven Q14/Q15 tables reduced the official normal and extreme methods to zero failures.

No-host-fallback audit:

```sh
rg -n 'numpy|_HOST_|run_host|host.*layout' tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py
```

Allocator `copyin`/`copyout` is normal buffer transport and is not semantic CPU execution.

## Explicitly pending

- Spatial direct convolution. Current tinygrad NCHW storage does not match the proven NPU atom/HWC surface. The frozen branch used host materialization for broad contraction/conv cases, so it is not a valid clean implementation source. A future path needs a typed `RKConv` plus an explicit device layout/weight stage. A 1x1 `1x32 -> 8` convolution canonicalizes to the already-supported affine CMAC contract and cannot be distinguished semantically after simplification.
- Wider affine/batched contractions and CMAC tiling.
- User-visible bool packing and general comparison outputs. FP16 masks used inside WHERE are native already.
- LOG2, reciprocal, SIN, and additional generated LUT identifiers.
- Windowed pooling until layout/channel padding is expressed on device.
- FP32 expressions/inputs, bool, general integer arithmetic/casts, and gradient support. Native FP32/int32 constant fills are bounded output contracts.
- Multicore/program-chain submission beyond the stable reset-separated single-core task sequence.
- Broad TestOps parity. Unsupported graphs deliberately reject rather than run on the CPU.

The next feature should be selected by hardware-native leverage, not raw failed-test count. The FP16 mask primitive now exists; the next boundary is device-native bool output packing and int/bool inputs so comparison and the leading `test_where` variants can complete. Spatial convolution should wait for an explicit device layout design.
