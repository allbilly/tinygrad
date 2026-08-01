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
