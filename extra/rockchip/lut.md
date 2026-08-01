# RK3588 LUT generation and tuning

Rockchip LUTs are generated assets selected by `RKLUTId`; they are not target
opcodes. `RKLUTStage` always describes the same hardware stage kind, while the
generated catalog supplies the payload, index multiplier, output exponent,
domain, digest, and measured simulator bounds.

Regenerate the catalog with:

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
python extra/rockchip/gen_lut.py
```

The output is deterministic and lives in
`tinygrad/runtime/autogen/rockchip_lut.py`. Review a change by comparing the
payload digest and metadata as well as the generated values.

## Table format

Each table has two 513-entry signed-int16 halves. The negative and positive
halves each provide 512 interpolation intervals plus the terminal endpoint.
The DPU uses `BN_MUL` to map an FP16 input to an interval and linearly
interpolates adjacent entries. `MINUS_EXP` controls the output conversion
scale. A table generator must clamp quantized payloads to int16 and simulate
the same interpolation and final FP16 rounding used by hardware.

## Two-level EXP

A single table cannot provide both adequate accuracy near zero and the dynamic
range needed for positive `exp(x)`. EXP therefore composes two NPU tasks:

- `EXP` covers `[-2, 2]`; positive values store `exp(x)/8`, then generic ALU
  stages restore the factor of eight.
- `EXP_LOCAL` covers `[-0.25, 0.25]` at higher index resolution.
- Generic mask and ALU stages select the local result for the open local
  interval and the broad result elsewhere.
- The compiler recognizes both `log2(e)` and `-log2(e)` factors in lowered
  EXP2 expressions, preserving the sign for `exp(-x)` and sigmoid.
- Generic mask/division stages restore `+inf`, `-inf`, and NaN behavior.

This is two LUT NPU submissions inside one typed plan; it does not evaluate
tensor semantics on the host.

## Tuning procedure

1. Choose a declared input domain from the workload and hardware range.
2. Sweep index scale and output exponent while rejecting int16 saturation.
3. Simulate every relevant FP16 encoding with hardware-equivalent indexing,
   interpolation, and output rounding.
4. Record maximum absolute and relative error; also check ULP error,
   monotonicity, endpoints, signed zero, infinities, NaNs, and saturation.
5. Run a representative and boundary-heavy RK3588 hardware sweep.
6. Use the smallest test tolerance justified by those measurements. Do not
   globally relax FP16 tolerances to make unrelated operations pass.

The generated two-level EXP simulator currently covers all 32,770 finite FP16
encodings in `[-2, 2]` and records its exact worst-case bounds in the catalog.
The RK3588 test additionally samples 4,097 evenly spaced values and checks the
special values. Inputs outside the declared domain are not part of this LUT's
accuracy contract.

## Two-level sigmoid

Sigmoid is the single activation-specific asset included for the demonstrated
CMAC-to-activation workload. Its broad Q15 table covers `[-8, 8]`; its local
Q15 table covers `[-2, 2]` at four times the index resolution. Generic mask
and ALU stages select the local result, saturate finite tails, and restore
infinity and NaN behavior. The same typed expression is reused by SiLU and
Swish through `Ops.MUL`; those functions do not have separate LUT identities.

The generated simulator checks all 36,866 finite FP16 encodings in `[-8, 8]`
and records absolute and relative bounds. Relative error is expected to be
less informative in the negative tail where the reference approaches zero,
so conformance uses the normal-domain strict relative tolerance plus absolute,
boundary, and special-value checks.

Early simplification may fold a positive sigmoid input multiplier into the
EXP2 coefficient. The lowerer recovers that scale and executes it as generic
`Ops.MUL` before the same sigmoid LUT stages. A hardware sweep of the scaled
composition records at most `1e-3` absolute error and `3.1e-3` relative error
away from zero. Saturated QuickGELU tails meet the strict TestOps comparison;
the normal-range QuickGELU case remains a numerical mismatch and is not
claimed as passing.

## Refined SQRT

`SQRT` provides an initial estimate over `[0, 4]` in Q14. Three Newton steps,
`y = (y + x/y)/2`, execute as ordinary ALU/division stages on the NPU. The
epilogue uses masks and division to restore zero, positive infinity, negative
input NaN, and input NaN semantics. No host correction is involved.

The generated simulator exhaustively checks 10,241 FP16 encodings from
`2^-8` through `4`, including FP16 rounding after every refinement operation.
The RK3588 sweep extends through `16` to verify convergence beyond the seed
table's direct domain and separately checks IEEE special values. Smaller
positive subnormals are not yet part of the accuracy claim.

## Range-scaled RSQRT

`RSQRT` stores a bounded `[0.5, 4]` seed over the positive part of `[-4, 4]`.
Two mask-controlled factors of 16 move small inputs into the seed range; one
Newton correction refines the estimate, and matching factors of four restore
the output scale. The IEEE epilogue creates positive infinity for zero and NaN
for negative/NaN inputs while mapping positive infinity to zero.

The generated simulator checks the same 10,241 FP16 encodings from `2^-8`
through `4`, including FP16 rounding at each stage. The hardware suite uses a
2,049-point geometric sweep plus special values. Values below `2^-8` are not
yet part of the declared accuracy range.

## Range-normalized logarithms

LOG2 and LOG10 each use a broad table over `[0.25, 4]` plus a high-resolution
table around one. Two powers-of-16 masks normalize inputs down to `2^-8`; the
corresponding exponent offset is applied with generic arithmetic. A quadratic
near-one expression avoids relative-error amplification as the result tends
to zero. Zero, infinity, negative inputs, and NaN are repaired on the NPU.

The complete 10,241-value FP16 range from `2^-8` through `4` passes the strict
hardware comparison for both functions. Native `Ops.SUB` is material here:
using MUL-by-minus-one plus ADD exceeded RKImage's 64-stage dependency limit.
FP32 logarithms and smaller positive values are outside this contract.

Arbitrary multiplication of a LOG2 result now falls through to the generic
ALU path; only the constant `log10(2)` scale selects LOG10 tables. EXP2 repairs
special values for expression inputs as well as direct buffers. The compact
repair divides by the positive-infinity validity mask and multiplies by the
negative-infinity validity mask, so the LOG2-zero times negative composition
fits exactly in the 64-stage image while retaining infinity and NaN behavior.

## Cancellation-resistant EXPM1

The generic EXPM1 recognizer replaces both `exp(x)-1` and `1-exp(x)` after EXP
has decomposed to EXP2. A broad table covers `[-2, 2]`; a Q17 local table
covers `[-0.25, 0.25]` so subtraction near zero does not lose the significant
bits before the DPU writes FP16. Positive payloads are range-scaled and restored
with generic masks and arithmetic. This materially narrows CELU, ELU, and SELU
error, but those complete methods remain numerical mismatches and are not yet
claimed as passing.

## Tanh ranges and local polynomial

Tanh is recognized from tinygrad's `2*sigmoid(2*x)-1` decomposition. A broad
Q15 table handles `[-2, 2]`, a Q16 mid table resolves `[-0.5, 0.5]`, and the
clamped local interval uses `x-x^3/3`. The polynomial avoids the LUT output
quantization that remained visible near zero and is clamped before evaluation
so unselected extreme inputs cannot overflow and contaminate the result.
Generic masks saturate finite tails and preserve NaN behavior. Both strict
`test_tanh` methods pass on RK3588 without a tanh DPU opcode or host fallback.

## Inverse trigonometric functions

Tinygrad decomposes `acos(x)` through `asin(x)` and `atan(x)` through
`asin(x/sqrt(1+x*x))`. The compiler recognizes those decompositions before
they consume the 64-stage dependency mask. ASIN and ACOS use independent broad
generated tables because subtracting a Q14 ASIN result from pi/2 loses ACOS
precision. They share a high-resolution edge table indexed by `1-abs(x)`;
ASIN additionally uses a local odd polynomial near zero. The edge payload
encodes mathematical zero as the smallest nonzero value and a generic nonzero
mask restores exact zero, avoiding the RK3588 LUT engine's zero-entry quirk.

ATAN uses one broad table over `[-8, 8]`, an odd local polynomial through the
fifth power, and the generic asymptotic tail `sign(x)*pi/2-1/x`. ASIN, ACOS,
and ATAN all pass their complete forward-only TestOps methods, including the
declared invalid-domain and large-magnitude cases. These are LUT asset
identities consumed by `RKLUTStage`, not target opcodes.

## Inverse hyperbolic tangent

ATANH is recognized from tinygrad's `log((1+x)/(1-x))/2` decomposition. A
broad Q13 table covers `[-0.875, 0.875]`, the odd fifth-order series handles
the local interval, and a Q12 edge table is indexed by `1-abs(x)` to retain
resolution near the singularities. Generic masks turn exact `-1` and `1` into
signed infinities and values outside the domain into NaN. The complete strict
forward TestOps method and a dense RK3588 sweep pass without host fallback.

## Inverse hyperbolic sine and cosine

ASINH and ACOSH are recognized from tinygrad's
`log(x+sqrt(x*x+offset))` decompositions before their generic SQRT and LOG
recipes exceed the 64-stage image limit. ASINH uses a Q12 table over
`[-512, 512]`, a Q13 table over `[-8, 8]`, a Q14 near table over `[-2, 2]`,
and an odd fifth-order series near zero. ACOSH uses a Q12 table over
`[1, 512]`, a Q13 table indexed by `x-1`
through `9`, and a Q16 edge table over `[1, 1.125]` to resolve its infinite
slope at one. Generic masks select ranges and produce NaN below the ACOSH
domain; the table identities remain data consumed by `RKLUTStage`.

The offline simulator exhaustively checks every finite FP16 encoding in each
declared range and records maximum absolute and relative error. Hardware tests
cover dense local/mid ranges, magnitude 300 tails, and invalid ACOSH inputs.

## Hyperbolic sine and cosine

Tinygrad decomposes SINH and COSH into two EXP graphs. Recognizing the shared
`(exp(x) +/- exp(-x))/2` structure avoids duplicating the already multistage
EXP implementation. Generated Q13 SINH and COSH tables cover `[-2, 2]`; SINH
uses its odd fifth-order series near zero to avoid relative-error amplification.
Generic masks and division create the signed or positive overflow tails on the
NPU. The simulator exhaustively covers the declared FP16 range, while hardware
tests add magnitude-300 overflow cases.

## Error function

The compiler recognizes tinygrad's Abramowitz-Stegun ERF expansion before its
polynomial and EXP graph exceeds the image stage limit. A generated Q15 table
covers `[-2, 2]`, a Q16 table covers `[-0.25, 0.25]`, and the odd fifth-order
Maclaurin series handles the immediate near-zero range. Generic masks saturate
finite tails to `-1` or `1`. The offline
simulator exhaustively checks the declared FP16 domain and the RK3588 suite
adds magnitude-300 saturation inputs.

The local Q16 LUT produces an undefined conversion result for its smallest
near-zero payload on RK3588. DPU stages are eager, so multiplying that dead
result by a zero selection mask still contaminates the final value. The
lowerer therefore shifts only the polynomial-selected local-LUT input away
from zero; that LUT output is dead by construction, while every live local-LUT
input and the near-zero polynomial remain unchanged. This is preferable to a
tolerance exception because exact zero and the complete dense hardware sweep
then follow the intended numerical path.

## Softplus and LogSigmoid

The compiler recognizes the stable `logaddexp(x, 0)` decomposition rather
than matching a Tensor method name. It evaluates
`softplus(x) = softplus(-abs(x)) + max(x, 0)`: the generated signed Q16 table
stores the cancellation-resistant residual `softplus(-abs(x)) - 0.5`, and
generic ALU stages restore the offset and positive part. Negating the same
recognized form implements LogSigmoid, so no separate LogSigmoid asset or
target opcode is required.

`beta=3` needs a two-task table over the original input. If `3*x` is first
written to FP16 scratch, its rounding differs from the fused Softplus
reference at the strict TestOps threshold. The near Q17 table covers original
inputs `[-5/6, 0]`; the far Q20 table covers `[-2, -5/6]`. Both directly encode
`softplus(3*x)/3`, and `max(x, 0)` supplies the positive branch. Their input
scales are powers of two (`16384` and `8192`), avoiding the hardware index
drift observed with the non-representable ideal scale `6553.6`.

The generator exhaustively simulates every finite FP16 encoding in the
declared ranges, and the hardware test densely sweeps the default and scaled
local domains plus magnitude-300 tails. Mish reuses the generic Softplus and
Tanh plans and now fits in 61 stages, but it remains a strict numerical
mismatch by up to one additional FP16 ULP and is not claimed by this milestone.

## Mish ranges

The follow-up Mish asset recognizes the existing
`x*tanh(softplus(x))` decomposition and replaces its 61-stage composition with
two generated data ranges plus generic local arithmetic. A broad Q14 table
covers `[-2, 2]`, a Q16 table covers `[-0.5, 0.5]`, and the immediate
`[-0.125, 0.125]` interval uses the fourth-order Horner series
`x*(0.6+x*(0.32+x*(-0.016+x*(-86/1875))))`. The polynomial avoids the output
quantization that exceeded strict relative tolerance near zero.

Both LUT inputs are moved away from zero only when their outputs are dead under
the local selection masks. This prevents the RK3588 zero-entry conversion
quirk from contaminating exact `mish(0)`. The compiler plan contains 34 typed
stages and no Mish hardware opcode; `MISH` and `MISH_MID` are generated asset
identities consumed by the generic LUT stage.

## Hardswish range and local series

The Hardswish recognizer recovers the input of tinygrad's
`x*relu6(x+3)/6` decomposition. One generated Q14 table covers `[-2, 2]`;
inside `[-0.125, 15/128]`, generic ALU stages evaluate the cancellation-safe
identity `x*x/6 + x/2`. The same identity handles the positive `(2, 3)` tail,
and values at or above three select the input directly. The original staged
formula remains the negative-tail fallback.

The generator exhaustively simulates all 32,770 finite FP16 encodings in the
table domain. Its maximum absolute error is 0.00051116943359375, and maximum
relative error for outputs above 0.01 is 0.0007416965546671742. A rejected
second-level LUT is retained as commented WIP in the generator: scaling the
input and output by 16 improved ordinary local values but exposed the RK3588
zero-entry quirk for FP16 subnormals. The final plan has 47 typed stages, one
generic LUT stage, and no Hardswish opcode or host semantic work.
