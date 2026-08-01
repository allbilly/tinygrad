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

LUT identity is data, not a DPU opcode. `RKDPUOp` has one generic `LUT` stage;
its typed plan carries an `RKLUT` identifier that selects immutable payload,
scale, shift, and special algorithm metadata. Adding a generated table no
longer expands the hardware-operation enum or the emitter dispatch surface.

## Current generated artifacts

| Field | Value |
|---|---|
| Identifier | `RKLUT.EXP2 = 1` |
| Schema | 21 |
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

QuickGELU uses two additional generated NPU tasks rather than retuning sigmoid and risking regressions:

- `RKLUT.QUICK_GELU = 8` is Q14 over `[-2,2]`, with index scale 8192 and output scale 16384;
- five measured integer-knot corrections from the passing 2607 oracle are applied after quantization: `(LE,276,+4)`, `(LE,375,+1)`, `(LE,408,+1)`, `(LE,427,+1)`, and `(LO,49,+1)`;
- `RKLUT.QUICK_GELU_LOCAL = 9` is Q15 for `x in [-2,-1]`, addressed by `z=(x+1.5)*4`;
- local knots blend the ideal curve and PyTorch-style staged FP16 multiplication/sigmoid result equally, so fitting includes intermediate half rounding;
- a device polynomial `0.5*x + 0.4253*x^2` handles `[-0.16,0.16]`;
- the shared bounded sigmoid expression supplies tails outside `[-2,2]`, including exact zero/identity asymptotes for extreme negative/positive inputs.

The typed plan has 58 stages and six reusable scratch buffers. The official normal and both extreme subcases pass at `rtol=0.001, atol=1e-6`. A mathematically ideal dense float32 curve is not the tuning oracle: it differs from PyTorch's staged FP16 boundaries, which is why the sparse corrections and local blend must remain generator inputs rather than unexplained edits to generated data.

GELU has separate immutable artifacts because the tanh approximation and exact-erf form are observably different at FP16 boundaries:

- `RKLUT.GELU_TANH = 10` and `RKLUT.GELU_EXACT = 12` are asymmetric Q15 broad tables over `[-4,4]`;
- negative entries store GELU directly, while positive entries store GELU divided by four to fit Q15 and are multiplied by four with a device sign mask;
- `RKLUT.GELU_TANH_LOCAL = 11` and `RKLUT.GELU_EXACT_LOCAL = 13` store `2*GELU(x)` over `[-0.5,0.5]`, addressed by `z=8*x`, then scale by one-half on device;
- a shared near-zero series `0.5*x + x^2/sqrt(2*pi)` handles `[-0.04,0.04]`;
- device masks return zero for `x < -4` and identity for `x > 4`.

Each variant lowers to 51 stages and six scratch buffers. Both official normal forms and all four `[-400,-300]`/`[300,400]` extreme subcases pass at the strict TestOps tolerance. The supplemental exact-erf dense sweep allows `atol=2e-4` because the declared negative tail intentionally saturates to zero while ideal erf remains a sub-`1.3e-4` negative value near the boundary.

Standalone Erf uses `RKLUT.ERF = 14`, a direct Q15 table over `[-4,4]`, and `RKLUT.ERF_LOCAL = 15`, Q15 `3*erf(x)` over `[-0.25,0.25]` addressed by `z=16*x`. The local result is divided by three on device, while `2*x/sqrt(pi)` replaces both LUTs inside `[-0.04,0.04]`. Device masks select exact `-1/+1` tails outside the broad domain. The 44-stage typed plan passes the official normal, scalar, positive-extreme, and negative-extreme subcases plus a strict 2,049-point `[-8,8]` sweep.

ELU/SELU use three generated broad/local pairs: `ELU1 = 16/17`, `ELU01 = 18/19`, and `SELU = 20/21`. Broad tables cover the negative branch on `[-8,0]`; local tables address `x in [-0.5,0]` through `z=4*x`. Gains use available Q15 precision (1/2 for alpha 1, 8/16 for alpha 0.1, and 0.5/1 for SELU) and are inverted by device MUL stages. A second-order `scale*(x+x^2/2)` handles `[-0.03,0]`, exact negative saturation handles `x<-8`, and ordinary MAX/MUL handles the positive branch. All three share one 35-stage, six-scratch lowering recipe and pass their official methods plus dense `[-10,10]` sweeps.

Mish uses `RKLUT.MISH = 22`, an asymmetric Q15 table on `[-8,8]` whose positive half stores Mish divided by eight, and `RKLUT.MISH_LOCAL = 23`, direct Q15 Mish over `[-1,1]` addressed by `z=2*x`. The NPU restores the positive broad scale with a sign mask, uses `0.6*x+0.32*x^2` inside `[-0.08,0.08]`, and selects zero/identity tails outside the broad domain. Its typed plan has 38 stages and six scratch buffers. The official Torch method passes at `rtol=1e-3`; the supplemental ideal-float dense sweep uses `rtol=1e-2` because it does not model PyTorch's staged FP16 boundaries.

LogSigmoid uses `RKLUT.LOGSIGMOID = 24`, Q15 `-log1p(exp(-abs(x)))` over `[-8,8]`, and `RKLUT.LOGSIGMOID_TAIL = 25`, Q15 `32` times the same correction over `[-16,16]`. The NPU reconstructs `min(x,0)+correction`, selects the amplified table above `x=3.5`, and clamps the result nonpositive. The 15-stage plan uses four scratch buffers and passes the official method plus a strict dense `[-12,12]` sweep; farther positive tails may round sub-micro corrections to signed zero.

Softplus uses broad/tail pairs `SOFTPLUS1 = 26/27` and `SOFTPLUS3 = 28/29`, plus the Q13 wide `SOFTPLUS13 = 30` table. The β=3 pair stores full-Q15 corrections and sets `OUT_CVT_SCALE` to Q15 `1/3` with an integer shift of 15, applying scale before FP16 storage; baking `/3` into knots or using a later MUL both miss strict FP16 boundaries. Inputs are clamped to each table's declared domain before lookup, and masks enforce zero/identity asymptotes. β=1 and β=3 use 27 stages/four scratch buffers; β=1/3 uses eight stages/two buffers. All official normal, extreme, and scalar cases pass. The ideal dense sweep uses `atol=3e-6` for amplified-tail values that intentionally round to zero.

Hyperbolic functions use `SINH = 31`, Q13 on `[-2,2]`, `SINH_LOCAL = 32`, Q15 `4*sinh(x)` near zero, and `COSH = 33`, Q13 on `[-2,2]`. Sinh selects the local table inside `|x|<0.125` and identity inside `|x|<0.04`; Cosh needs only the broad table. Inputs are clamped before lookup, and division by a device mask creates signed Sinh infinity or positive Cosh infinity for `|x|>10`. The 30/11-stage plans pass normal and ±300 official cases plus strict central dense sweeps.

Sqrt uses `SQRT = 34`, a Q14 seed table over `[0,4]` with index scale 4090. Three Newton steps execute on the NPU as `y=(y+x/y)/2`, removing the linear interpolation curvature error. Device masks preserve exact signed zero, synthesize positive infinity above the FP16 finite range, and return NaN for negative or NaN input. The typed program has 25 stages and four reusable scratch buffers. FP32 explicit/scalar inputs are narrowed once at the declared runtime ABI boundary; the LUT, refinement, and all special-value semantics remain NPU operations. The official method and a dense 2,049-point `[0,16]` sweep pass.

Reciprocal Sqrt uses `RSQRT = 35`, a Q13 seed clamped to `[0.5,4]` over the same `[0,4]` address domain. Positive inputs below `1/16` and `1/256` are multiplied by exact powers of 16 before lookup; the result is multiplied by the corresponding powers of four. One NPU inverse-square-root Newton step, `y=y*(1.5-0.5*x*y*y)`, removes interpolation error without the extra FP16 rounding of `1/sqrt(x)`. Device masks restore positive-zero/infinity/negative/NaN semantics. The 42-stage, six-scratch program passes the official method and a dense geometric sweep from `2^-8` to 4. DPU division currently normalizes `-0`, and general finite inputs above 4 still need high-range reduction.

Natural Exp uses `EXP = 36`, an asymmetric Q15 broad table over `[-2,2]`: negative knots store `exp(x)` directly while positive knots store `exp(x)/8`, restored by an NPU mask. `EXP_LOCAL = 37` stores direct Q14 Exp over `[-0.25,0.25]` to avoid the broad table's zero-side scale discontinuity. Device masks synthesize positive infinity, zero, and NaN for nonfinite inputs. The 36-stage, four-scratch program passes every official subcase and a dense `[-2,2]` sweep.

CELU α=2/3/4 uses direct final-output broad/local pairs `CELU2 = 38/39`, `CELU3 = 40/41`, and `CELU4 = 42/43`. Broad tables cover `[-4,0]` in Q14 for α=2 and Q13 for α=3/4; Q15 local tables cover `[-0.5,0]`. A second-order `x+x²/(2α)` replaces the LUT inside `[-0.03,0]`, avoiding cancellation after `exp(x/α)-1`. α=1 reuses the ELU1 tables. The 35/30-stage plans pass all official tensor/scalar cases and dense per-α sweeps.

Logarithms use scale-specific broad/local pairs: `LOG2 = 44/45`, natural `LOG = 46/47`, and `LOG10 = 48/49`. Two power-of-16 masks normalize positive inputs into `[0.25,4]` and add the exact `-4` or `-8` exponent offset; this covers values down to `2^-10` with six fewer tasks than four power-of-four bands. Broad tables use Q13/Q14/Q15 according to each base's normalized output range. Q15 local tables store four times the result near one, and `x-x²/2` replaces them inside `|x-1|<0.02`. The natural-log negative local knots 77–78 have a recorded `+8` correction for RK3588 interpolation rounding.

Half inputs use 57 stages and eight reusable scratch buffers. FP32 natural-log inputs use a declared two-plane ABI representation: the runtime encodes `hi=fp16(x)` and `lo=fp16(x-hi)` into atom-aligned planes, the NPU adds the first-order `lo/hi` correction, and the NPU writes an FP16 result that the runtime widens to FP32. These are element-format encode/decode operations only; logarithm evaluation, range selection, correction, and IEEE zero/infinity/NaN semantics remain NPU tasks. The FP32 plan has 61 stages. All three official methods pass at their unchanged `rtol=1e-3`; the supplemental geometric `[2^-10,4]` characterization uses the measured `rtol=1.1e-3` envelope (observed maximum `0.001038`). A rejected `+16` Q14 correction at broad LO knots 295/296/311/312 fixed the two normalized low-input misses but regressed four direct positive inputs, so it is not part of the generated artifact.

Round-to-nearest-even uses `ROUNDOFF = 50`, the RK3588 algorithm-23 index mode rather than an ordinary function-sampled table. Both 513-entry banks alternate between `0` and Q14 `1`; the emitter uses index selector 14, the `0x44000000..0x44800000` endpoint contract, and the proven LE slope scale/shift `23107/22`. NPU arithmetic forms `abs(x)`, the LUT rounds its magnitude, and masks restore sign, signed infinities, and NaN. The 20-stage, six-scratch plan passes the full official method and an exact 4,097-point `[-16,16]` sweep plus half ties, infinities, signed zero, and NaN. Truncation reuses that result and subtracts/adds one only when the rounded value crossed zeroward; tinygrad's floor and ceil WHERE graphs then compose from truncation and the same primitive masks. All three official methods and dense sweeps pass. The rejected standalone `CVT_ROUND` int32 probe remains documented but is unnecessary.

ASIN uses two ordinary NPU LUT tasks selected and composed entirely with typed DPU arithmetic:

- `ASIN = 51` stores `0.5*asin(x)` over `[0,1]` at index scale 16,384 and Q15 output; SHA256 `e3bfd3ca769499c1818f3d05f5536a371cf06ef96d855d4d338f8dea96ee5ec9`;
- `ASIN_DETAIL = 52` uses the LE bank for `4*asin(abs(x))` near zero and the LO bank for `0.5*asin(1-x)` near the endpoint, at index scale 65,504 and Q15 output; SHA256 `6b2daeebe2393a0a0e4ddd6f298458a9cc50d0db6dae9b662c4d56813e16bf5d`;
- masks select identity inside `|x|<=0.04`, the detail center through `0.125`, broad interpolation through `0.875`, then endpoint-distance detail through one;
- sign and invalid-domain NaN are reconstructed on the NPU; no host semantic path is used.

The complete expression is 43 stages with eight scratch buffers and one instance of each asset. It passes the official ASIN method and strict FP16 comparison over a 4,097-point `[-1,1]` sweep plus `[-2,-1,-0,+0,+1,+2,nan]`.

ACOS cannot safely reuse ASIN through subtraction from π/2. That first probe compiled in 44 stages but missed 156/2,925 official outputs, with max absolute error `0.000977` and relative error `0.00812`. The committed design therefore uses three ordinary regional LUT assets:

- `ACOS = 53` stores `acos(x)/4` in the negative bank and `acos(x)/2` in the positive bank; SHA256 `6f23bcf3d689aa5641e4051aac740be8024c13deb76db202711a9528f48ec448`;
- `ACOS_ENDPOINT = 54` stores direct `acos(1-d)` for endpoint distance `d`; SHA256 `da0c3dc5469308b4d1ee30b237cc22c31dc92a311c558b55e920377613014bfc`;
- `ACOS_FINE_ENDPOINT = 55` stores `8*acos(1-d)` addressed with `64*d`, then the NPU decodes by `1/8`; SHA256 `1a42b99e01379b7c4c564298e940f3358ace5efeaeca2b7e1a4517f1982a5d13`.

The coarse/fine split is at `d=0.003`, while the endpoint region begins at `|x|>0.85`. One offline correction changes the negative bank's shared zero knot from positive-bank half scaling to negative-bank quarter scaling; without it, exactly three dense-sweep values immediately below zero were corrupted by the table discontinuity. The 47-stage, nine-scratch plan passes both the official method and strict 4,097-point domain/special-value hardware test without tolerance changes or host evaluation.

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
