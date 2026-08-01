# RK3588 DPU LUT generation and tuning

Last updated: 2026-08-01

## Design boundary

LUT fitting, quantization, search, simulation, and reports are offline compiler work. The renderer/runtime only consume a committed immutable generated artifact:

```text
extra/rockchip/gen_lut.py
  -> deterministic fitting/quantization and exhaustive simulator
  -> tinygrad/runtime/autogen/rockchip_lut.py
  -> renderer emits table upload and fixed DPU LUT registers
  -> runtime submits commands; it never fits or evaluates a LUT on the CPU
```

This keeps generated numerical bulk out of handwritten `sz.py` lines and keeps runtime execution deterministic.

## Current generated artifacts

| Field | Value |
|---|---|
| Identifier | `RKLUT.EXP2 = 1` |
| Schema | 4 |
| Domain | `[-2.0, 2.0]` |
| Tables | LE and LO, 513 signed int16 entries each |
| Knot spacing | `1/256` input units |
| Input/index scale | 8192 (`FP16 bits 0x7000`) |
| Stored output scale | 8192 |
| Output minus exponent | 13 |
| Payload SHA256 | `ad3ef028711e351f457bb1598b28abc2c21c3de99b21b12dd210e21680760dcc` |
| Exhaustive simulated encodings | 32,770 finite FP16 encodings in domain, including both signed zeros |
| Simulated max absolute error | `0.0010343084909396616` |
| Simulated max relative error | `0.0007054306848276658` |
| Proven hardware widths | 1, 128, 129, and 2,925 FP16 values in one task |

The simulator includes output rounding to FP16. It models table interpolation for the declared domain. Hardware tests remain required because register interpretation, internal rounding, saturation, and chip revision behavior can differ from the mathematical simulator.

Direct EXP2 appends a 13-stage typed program. Threshold masks distinguish `+inf`, `-inf`, and NaN: NaN activates both high and low masks, while each infinity activates only its matching mask. Device divisions synthesize positive infinity and NaN, and multiplication restores negative underflow to zero. This passes `[+inf,-inf,nan] -> [inf,0,nan]` without inspecting values on the host.

Regenerate and verify:

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
python extra/rockchip/gen_lut.py
git diff -- tinygrad/runtime/autogen/rockchip_lut.py
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest \
  test/unit/test_rockchip_compiler.py::TestDPUCompiler::test_exp2_uses_generated_lut -q
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF python -m pytest \
  test/device/test_rockchip.py::TestRockchip::test_generated_exp2_lut -q
```

Regeneration should produce no diff unless the specification intentionally changed.

HardSwish adds two immutable tables and follows the final `rockchip-2607` algorithm:

- `RKLUT.HARDSWISH = 2`: Q14 broad approximation on `[-2,2]`;
- ordinary DPU arithmetic computes the exact decomposed fallback outside `[-2,2]`;
- `RKLUT.HARDSWISH_LOCAL = 3`: Q15 approximation of `16*hardswish(x)` after an NPU `x*16` stage;
- the local result is scaled by `1/16` and selected on `[-0.125, 15/128]`;
- an NPU nonzero mask removes the one-count LUT-zero workaround at exact zero.

This is two NPU LUT tasks, not host evaluation. On the official 2,925-element case it lowers to 36 stages and five scratch buffers and passes `rtol=0.001, atol=1e-6`.

Tanh uses the same regional-correction structure:

- `RKLUT.TANH = 4`: signed Q15 tanh on `[-4,4]`;
- `RKLUT.TANH_LOCAL = 5`: Q15 `4*tanh(x)` addressed by `z=16*x` near zero;
- the local result is scaled by `1/4` on the NPU and selected on `[-0.25,0.25]`;
- a clamped identity replaces the LUT inside `[-0.04,0.04]`;
- device masks select exact `-1/+1` tails outside `[-4,4]`.

The official normal and `[-300,-297]` extreme methods both pass. The plan is 35 stages with five scratch buffers.

Sigmoid adds `RKLUT.SIGMOID = 6` over `[-8,8]` and `RKLUT.SIGMOID_LOCAL = 7` over `[-2,2]`, both Q15. NPU masks select the dense local table, saturate infinities, and preserve NaN. The 24-stage sigmoid expression is reused by SiLU/Swish with one final MUL, so all three official methods pass without duplicate runtime recipes. Dense strict-relative validation is guaranteed for the local domain; finite tails beyond the broad domain currently follow the frozen branch's saturation policy.

## Current command contract

One EXP2 task emits 1,064 commands:

- 1,028 table-upload commands: two access-config writes plus `2 x 513` data writes;
- 36 DPU/RDMA/PC configuration commands;
- destination and source relocations at command words 1,032 and 1,059.

The important fields are:

- DPU/RDMA `S_POINTER = 0x30`;
- FP16 data format `0x48000002`;
- width `ceil(count/8)-1`, channel `0x70007`, and a matching rounded surface stride;
- `BN_MUL_CFG = half(8192) << 16` for LUT indexing;
- `OUT_CVT_SCALE = 0x10001` and `OUT_CVT_SHIFT = 13 << 12`;
- hybrid/overflow LUT config `0x68` and index selection `0x50500`;
- range endpoints `LE_START=0xffffc000`, `LO_END=0x4000`;
- overflow slope scale `16434 << 16`, shift `13 << 5`;
- RDMA source/destination are the only tensor-address relocations;
- PC enable mask is `0x18`.

Keep command ordering stable. DPU LUT programming is stateful and order/reset changes can produce timeouts or stale table behavior.

## How to tune one LUT

1. Define the mathematical function, exact input domain, special-value policy, and required output dtype.
2. Confirm the hardware index equation, LE/LO table selection, interpolation fraction, endpoint behavior, and internal fixed/FP format from a known-good command stream.
3. Select index and output scales that use int16 range without saturating the required domain.
4. Generate floating knots at the exact hardware sample points.
5. Quantize to the actual signed table representation; never optimize an unquantized table and assume the result survives packing.
6. Simulate hardware table selection, interpolation, conversion shift/scale, saturation, and final FP16 rounding.
7. Sweep every finite FP16 encoding in-domain when practical. Record max absolute error, max relative error with a defined near-zero policy, worst inputs, saturation counts, and ULP distribution.
8. Optimize integer knots against the chosen metric. Re-evaluate after every scale/domain change.
9. Generate immutable data and provenance metadata; hash the packed little-endian payload, not its Python source formatting.
10. Compare emitted commands to the frozen oracle and run a serialized hardware sweep.
11. Test repeated execution and execution after another DPU engine task to expose state pollution.
12. Commit generator, generated artifact, metadata test, and hardware test together.

Useful tuning knobs:

- domain endpoints;
- knot spacing and table split;
- input/index scale (`BN_MUL_CFG`);
- stored output scale and `OUT_CVT_SHIFT`;
- output conversion scale;
- endpoint values;
- underflow/overflow slope scale and shift;
- integer knot corrections near worst-case inputs;
- input range reduction and output reconstruction stages.

Do not tune only random FP32 inputs. The device consumes FP16 here, so exhaustive representable inputs are cheap and reveal boundary/signed-zero cases.

## Diagnosing accuracy

Classify the error shape before changing knots:

- constant multiplicative bias: output scale/shift is wrong;
- error grows with input magnitude: input index scale or overflow slope is wrong;
- spikes exactly at zero/table boundary: LE/LO selection or duplicate zero knot differs;
- sawtooth within every interval: interpolation fraction or quantization model is wrong;
- isolated endpoint failures: start/end encoding or clamp policy is wrong;
- simulator passes but hardware fails everywhere: register order, format bits, relocation, or reset state is wrong;
- first run passes and later runs fail: table/state pollution, not fitting quality;
- error floor near half an ULP: final FP16 rounding is the limit and more knot tuning will not help.

Always save the worst input, simulated intermediate index/fraction, neighboring knots, expected output, hardware output, and full command-image hash.

## When a two-level LUT is justified

A two-level LUT means two NPU LUT tasks with declared scratch tensors and NPU-side composition. It is not a larger hidden host table and it is not runtime fitting. Two useful forms are a cascade and a regional correction:

The clean representation is:

```text
x -> LUT_A -> scratch FP16 z -> LUT_B -> y

x -> LUT_broad  --+
x -> LUT_local  --+-> masks/select -> y
```

This composition can help when:

- one uniform input grid wastes knots in flat regions;
- the first LUT implements a monotonic domain warp and the second approximates the function on the warped coordinate;
- the first LUT gives a coarse monotonic approximation and the second calibrates systematic output-domain error;
- one task cannot cover the needed dynamic range without poor index resolution.

A cascade is most suitable for monotonic functions such as EXP2/LOG2 because the intermediate can remain a single-valued monotonic coordinate. A regional correction supports nonmonotonic functions when masks explicitly preserve the region; HardSwish uses this second form.

Costs and risks:

- two table uploads and two reset-separated submissions;
- one scratch allocation and extra memory traffic;
- FP16 rounding between levels;
- compounded interpolation error;
- more stale-state exposure;
- composition may be slower and less accurate than one LUT plus one DPU MUL/ADD range-reconstruction stage.

Tune a two-level design jointly:

1. Choose a monotonic first-stage mapping `z = W(x)` and constrain it to a safe FP16/domain range.
2. Fit the second stage `y = G(z)` against the quantized, FP16-rounded output of the first stage, not ideal `W(x)`.
3. Search both quantized tables together or alternate optimization until the end-to-end metric stops improving.
4. Simulate both hardware stages, including the intermediate FP16 store/load.
5. Require a material improvement over the single-level artifact at the same declared domain.
6. Benchmark submission and memory cost. Reject the design if accuracy gain is below the application requirement or latency regresses too much.
7. Emit two ordinary typed `RKDPUStage(EXP2/LUT, ...)` records; do not add a special two-level runtime opcode.

For wide-range EXP2, first evaluate mathematical range reduction plus ordinary DPU arithmetic. A LUT-only composition can warp the domain, but preserving an integer exponent and a fractional residual generally needs more than one scalar intermediate or an additional arithmetic/mask stage. Two LUT tasks should be adopted only after a bit-accurate experiment proves they outperform that simpler schedule.

## Adding another LUT operation

- Add a stable `RKLUT` identifier; never reuse a tinygrad `Ops` numeric value.
- Add its declarative domain/scales and generation formula offline.
- Generate packed immutable payload and SHA metadata.
- Add exhaustive simulation and recorded bounds.
- Extend the generic typed LUT stage with metadata selection, not a named high-level activation recipe.
- Add exact command-image checks and a serialized RK3588 test.
- Document special values and out-of-domain behavior explicitly.

LOG2 is a sensible next primitive only after zero/negative/NaN/Inf policy is defined. High-level GELU, softplus, BCE, or loss functions should remain tinygrad decompositions over primitive arithmetic/LUT stages rather than new runtime semantic tags.
