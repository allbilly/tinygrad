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
