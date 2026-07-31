# Rockchip clean rewrite progress

Last updated: 2026-08-01

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

Implemented forward-only FP16 subset:

- contiguous copy and fill;
- ADD, MUL, and MAX expression DAGs, including multiple reset-separated stages;
- scalar operands materialized as declared RKImage constants;
- EXP2 on one proven 128-element DPU tile over the generated `[-2, 2]` LUT domain;
- directly legal `A @ packed_B.T`, currently `A=(1,32)` and `packed_B=(N,32)` for proven output widths;
- row sum for `(N,32)`, implemented as the same CMAC contract with an image-owned FP16 ones vector;
- global MAX over explicitly HWC-compatible `(K,8)` input layouts supported by the PPU kernel constraints.

## Size result

`python sz.py` reports:

```text
tinygrad/renderer/rockchip.py  449
tinygrad/runtime/ops_rockchip.py  74
handwritten Rockchip total  523
```

This meets the requested `<5000` backend goal with 4,477 lines of headroom. The generated register and LUT modules are mechanically generated and are excluded by `sz.py`.

Compared with the frozen implementation, the dominant 77.5% task/graph-lowering catalog was replaced by three bounded recognizers and one primitive DAG scheduler. Approximate physical source distribution is now:

- RKImage types, codec, validation, relocation: renderer lines 18–192;
- semantic analysis and typed lowering: renderer lines 193–334;
- register emission: renderer lines 335–475;
- renderer integration: renderer lines 476–485;
- allocation/submission runtime: 85 physical lines, 74 `sz.py` lines.

The whole repository is 25,499 `sz.py` lines, a `+531` delta from the 24,968-line base. Therefore `MAX_LINE_COUNT=25000 python sz.py` still fails globally by 499 lines even though the backend itself is well below 5,000. Fixing that would require an upstream cap decision or unrelated repository reductions; no unrelated master code was compressed to disguise this backend cost.

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

No-host-fallback audit:

```sh
rg -n 'numpy|_HOST_|run_host|host.*layout' tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py
```

Allocator `copyin`/`copyout` is normal buffer transport and is not semantic CPU execution.

## Explicitly pending

- Spatial direct convolution. Current tinygrad NCHW storage does not match the proven NPU atom/HWC surface. The frozen branch used host materialization for broad contraction/conv cases, so it is not a valid clean implementation source. A future path needs a typed `RKConv` plus an explicit device layout/weight stage. A 1x1 `1x32 -> 8` convolution canonicalizes to the already-supported affine CMAC contract and cannot be distinguished semantically after simplification.
- Wider affine/batched contractions and CMAC tiling.
- WHERE/comparison/mask stages.
- LOG2, reciprocal, SIN, and additional generated LUT identifiers.
- Windowed pooling until layout/channel padding is expressed on device.
- FP32, bool, integer, and gradient support. Per user direction this branch is forward-only first.
- Multicore/program-chain submission beyond the stable reset-separated single-core task sequence.
- Broad TestOps parity. Unsupported graphs deliberately reject rather than run on the CPU.

The next feature should be selected by hardware-native leverage, not raw failed-test count. A typed comparison/mask stage is the best next DPU primitive; a spatial convolution should wait for an explicit device layout design.
