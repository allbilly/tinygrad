# RK3588 fixed-function NPU backend

This branch is the merge-oriented reduction of the `rockchip-2608` research
backend. It deliberately implements a small FP16 inference contract instead of
using the full `test_ops.py` inventory as an operator catalog.

The compiler boundary is:

```text
post-early_simplify UOps
  -> Rockchip legality and affine analysis
  -> UOp-free typed plans
       RKALUStage(op: Ops)
       RKMaskStage
       RKLUTStage(lut: RKLUTId)
       RKContract
  -> RKImage commands and relocations
  -> DRM allocation, patch, submit, and wait
```

The runtime does not import NumPy, narrow FP32 buffers, execute tensor
semantics on the host, or provide a CPU fallback. Unsupported graphs reject
before submission with `RKPLAN_REJECT:unsupported_graph`.

## Declared contract

- static, contiguous FP16 elementwise ADD, SUB, MUL, MAX, and division;
- scalar operands and FP16 fills through the same ALU stages;
- FP16 `WHERE` with a directly representable less-than mask and finite arms;
- generated EXP2, EXP, sigmoid, refined SQRT/RSQRT, and logarithm LUT assets
  with declared domains;
- direct FP16 CMAC for `M=1`, `K=32`, and `4 <= N <= 16` when the right-hand
  input is already stored as `(N, 32)`;
- one demonstrated two-kernel workload: direct `(1,32) @ (8,32).T`, followed
  by bounded sigmoid using generic ALU stages and two sigmoid LUT assets.

The renderer advertises only `dtypes.half`. FP32, integer and user-visible bool
outputs, noncontiguous elementwise layouts, general contractions, fused CMAC
epilogues, convolution, pooling, and gradients are outside this contract. A
fused CMAC epilogue is rejected rather than silently dropped.

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

## Current upstream blocker

The base master contains 24,968 counted lines. This backend branch currently
contains 25,889, so `MAX_LINE_COUNT=25000 python sz.py` fails by 889 lines. The
handwritten backend is 916 counted lines (843 renderer/compiler and 73
runtime). The code must not be hidden under `runtime/autogen` or moved out of
tree to evade this limit. The generic native-program hook is an independent
five-line counted change and can be reviewed separately; the backend needs
real upstream line budget or an independently useful in-tree reduction before
it can be submitted.
