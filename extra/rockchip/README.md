# RK3588 fixed-function NPU backend

This is the active coverage/research branch. It targets the complete forward
`test_ops.py` inventory through RK3588 engine tasks, with frontend-only methods
reported separately and unsupported kernels rejected before submission.
The frozen `rockchip-pr`, `rockchip-2608`, and `rockchip-2607` branches remain
minimal, architectural, and behavioral/register-programming references.

The compiler boundary is:

```text
post-early_simplify UOps
  -> Rockchip legality and affine analysis
  -> UOp-free typed plans
       RKALUStage(op: Ops)
       RKMaskStage
       RKLUTStage(lut: RKLUTId)
       RKContract
       RKProgram(ordered typed steps)
  -> RKImage commands and relocations
  -> DRM allocation, patch, submit, and wait
```

The native runtime does not import NumPy, narrow FP32 buffers, or execute tensor
semantics on the host. Native lowering returns a typed plan or a typed reject
at the legality decision, including detail, offending op, and a normalized
slot-independent graph fingerprint. `ROCKCHIP_FALLBACK=0` is the authoritative
mode for coverage and development. The explicit `RKPY` experiment is retained
only as a historical/debugging reference: it executes generic UOps on the CPU
and must never contribute to Rockchip pass totals.

Mixed-engine work is represented by a generic ordered `RKProgram`. Scratch is
declared once at program scope, and each typed DPU, CMAC, or PPU step is emitted
in order with constants and relocations remapped centrally. The former fixed
CMAC-prefix/DPU/main-CMAC/CMAC-suffix pipeline no longer constrains composition.

Lowering uses six named ordered passes for DPU arithmetic, affine reformat,
global sum, affine reduction, PPU reduction, and contraction. Every pass returns
exactly one of `NATIVE`, `NOT_APPLICABLE`, or `UNSUPPORTED`: unrelated passes
cannot overwrite a useful reject, while applicable failures are ranked by
specificity before telemetry is emitted.

The compiler is split by responsibility under `renderer/rockchip/`:
`ir.py` owns the typed UOp-free plans, `expr.py` owns math recipes and UOp
canonicalization, `affine.py` owns affine maps and reject fingerprints,
`emit.py` owns DPU/CMAC/PPU register emission, and `image.py` owns the versioned
image codec and relocations. The package entry contains resource planning,
ordered legalization, and renderer integration. Register emission imports no
UOp definitions and cannot recover source-graph semantics.

The frozen `rockchip-2607` branch is a behavioral and register oracle, not
evidence that all of its 425 passing methods ran on the NPU. Its later runtime
contains NumPy-backed `_run_host_*` implementations for generic elementwise and
reductions, variance, interpolation, losses, tangent, and large einsum. Only
paths that reach `DRM_IOCTL_RKNPU_SUBMIT` without host semantic preprocessing
are eligible to be ported as native capabilities.

## Declared contract

- static, contiguous FP16 elementwise ADD, SUB, MUL, MAX, and division;
- scalar operands and FP16 fills through the same ALU stages;
- FP16 `WHERE` with a directly representable less-than mask and finite arms;
- generated EXP2, EXP, sigmoid, refined SQRT/RSQRT, logarithm, inverse
  trigonometric, and inverse-hyperbolic LUT assets with declared domains;
- direct FP16 CMAC for `M=1`, `K=32`, and `4 <= N <= 16` when the right-hand
  input is already stored as `(N, 32)`;
- direct FP16 sums of 4–16 dense rows of length 32 using an image-owned ones
  vector and the same CMAC contract;
- contiguous FP16 global sums whose power-of-two block decomposition fits a
  32-term aligned DPU/CMAC plan;
- direct aligned K64–K512 CMAC sums or scaled sums with generated
  hardware-packed weights and a single FP32 accumulation;
- global ReLU-sum through a DPU MAX-zero prepass and the same direct CMAC;
- static affine FP16 ADD reductions with at most 512 input and 128 output
  elements, using NPU zero-fill/copy preparation and sequential sparse CMAC
  tiles of at most 16 logical outputs; constant mean scaling is folded into
  the weights;
- static affine FP16 movements with at most 512 source and 4,096 output
  elements: aligned runs use DPU atom copies, while arbitrary selector maps up
  to an 8 MiB generated-weight budget use the same sparse CMAC pipeline;
- sub-atom FP16 DPU stages program their logical channel count, so NPU-zeroed
  physical padding remains defined when a scalar or short tail feeds CMAC;
- one demonstrated two-kernel workload: direct `(1,32) @ (8,32).T`, followed
  by bounded sigmoid using generic ALU stages and two sigmoid LUT assets.

Native arithmetic is FP16. Int32 and FP32 are currently admitted only for
operation-specific constant fills; this does not claim general arithmetic for
either dtype. User-visible bool outputs, noncontiguous elementwise layouts,
reductions outside the proven static affine bounds, general contractions,
fused CMAC epilogues, convolution, pooling, and gradients remain outside the
native contract. A fused CMAC epilogue is rejected rather than silently
dropped.

## EXP2 generation and characterization

Run `python extra/rockchip/gen_lut.py` to regenerate
`tinygrad/runtime/autogen/rockchip_lut.py`. The generated file records its
schema, payload SHA-256 digest, domain, interpolation parameters, exhaustive
simulator population, and simulator error bounds.

Every one of the 32,770 FP16 encodings in `[-2, 2]` is checked in the simulator
and on RK3588 hardware. The 2026-08-01 hardware sweep measured:

- maximum absolute error: `0.0010883808135986328` at `1.9560546875`;
- maximum relative error: `0.0008962760912254453` at `-1.9326171875`;
- maximum error versus correctly rounded FP16: one ULP;
- monotonic output over the complete declared domain;
- exact `exp2(0) == 1`.

Inputs outside the declared domain are not claimed by this first LUT contract.
The EXP implementation uses two sequential NPU LUT tasks: a broad table over
`[-2, 2]` and a higher-resolution local table over `[-0.25, 0.25]`. See
`extra/rockchip/lut.md` for the table scaling, selection, and tuning procedure.

## Validation

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest \
  test/null/test_native_program.py test/unit/test_rockchip_image.py \
  test/unit/test_rockchip_compiler.py -q -n12
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest test/device/test_rockchip.py -q
python -m ruff check .
python -m mypy tinygrad/
python extra/rockchip/gen_lut.py
```

The hardware suite is serial. Anything included in it is a promised capability
and has no skip on an RK3588 host.

The current committed strict census at `7cf01ac95` contains 102 native passes,
40 frontend-only passes, 270 failures, and 13 upstream skips across exactly 425
methods. It ran uncached with `ROCKCHIP_FALLBACK=0`; no prior native pass
regressed, and the latest gain is the scalar `test_expand` subcase through
sub-atom DPU preparation followed by sparse CMAC.

## Current upstream blocker

The base master contains 24,968 counted lines. This research branch currently
contains 27,267, so `MAX_LINE_COUNT=25000 python sz.py` fails by 2,267 lines.
The exact 2,299-line delta is 2,294 counted Rockchip backend lines (2,134
renderer/compiler, 111 runtime, 33 historical Python-fallback adapter, and 16
telemetry support) plus the five-line generic native-program hook. Generated
register definitions, LUT payloads, and reproducible command data belong under
`runtime/autogen`; handwritten legality, layout, scheduling, and emission logic
remain counted. The generic hook can be reviewed separately, while the backend
needs real upstream line budget or independently useful in-tree reductions
before submission.
