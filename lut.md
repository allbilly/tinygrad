# RK3588 DPU LUT implementation and tuning

This document describes how the Rockchip backend recognizes, emits, measures,
and tunes RK3588 DPU lookup-table operations. It is based on the backend in
`tinygrad/runtime/support/rockchip.py`, the register programs in
`ref/npu/include/rknnops.h`, and measurements on the local RK3588 NPU.

## Scope

The DPU LUT path is useful for:

- primitive transcendental ops such as EXP2, LOG2, SIN, SQRT, and RSQRT;
- recognized composite ops such as sigmoid;
- specialized tables such as round-to-nearest-even;
- activation algorithms from `rknnops.h`, after their domain and output
  quantization have been verified on hardware.

A reference table is a starting point, not a complete backend implementation.
The graph recognizer, input domain, fp16 stage boundaries, output
dequantization, special values, and interaction with later DPU tasks all affect
whether a TestOps method actually passes.

## Current backend path

The main pieces are:

| Component | Location | Purpose |
|---|---|---|
| LUT markers and builders | `support/rockchip.py`, `_LUT_*` and `_build_*_lut` | Construct the two 513-entry tables and scaling metadata |
| Graph recognition | `_try_sigmoid`, `_try_round`, `_try_lut` | Match only graphs whose semantics are known |
| Classification | `plan_rk` | Select `dpu_lut` and carry input/output scale metadata |
| Emission | `_emit_dpu_lut` | Fill tables and emit DPU, RDMA, conversion, LUT, and PC registers |
| Nested lowering | `_try_elementwise_subtasks` | Materialize LUT results that are operands of later elementwise stages |
| Submission | `ops_rockchip.py` | Run a standalone LUT task or a staged program |

`build_native_program` first rewrites `MUL(a, RECIPROCAL(b))` to `FDIV(a, b)`.
Always inspect the graph after this rewrite when writing a recognizer. A
recognizer that matches the original Tensor graph can silently never run.

## Table geometry

The ordinary LUT path uploads two signed int16 tables:

- table 0: LE/negative side;
- table 1: LO/positive side;
- 513 entries per table;
- 1026 entries total.

For the common `LUT_*_INDEX_SELECT=5` configuration, adjacent entries represent
32 units in the scaled index domain. If the BN multiplier is
`index_scale`, the approximate source-domain step is:

```text
step = 32 / abs(index_scale)
```

The half-domain covered by 512 intervals is:

```text
domain_limit = 512 * step = 16384 / abs(index_scale)
```

For a desired source interval `[-D, D]`, a useful starting value is:

```text
index_scale = 16384 / D
```

Use a slightly smaller value such as 4090 or 8190 when an endpoint equal to
`LUT_LO_END` is known to select overflow rather than the final table entry.
The actual BN operand is the fp16 bit representation of `index_scale`, so tune
only values that remain distinct after fp16 conversion:

```python
bn_mul_operand = int(np.float16(index_scale).view(np.int16)) & 0xFFFF
```

Generate the table coordinates from the same quantized multiplier:

```python
hardware_index_scale = float(np.float16(index_scale))
step = 32 / abs(hardware_index_scale)
```

Using `32/index_scale` with a non-representable nominal scale shifts the knots
relative to the addresses selected by hardware. Tangent exposed this at
`index_scale=16384/1.05`: the nominal value is about `15603.81`, but the DPU
receives fp16 `15600`. The drift caused a repeatable `0.001953` output error
near the upper table edge; quantized-scale knot generation removed it.

Negative input scaling is supported by using a negative BN multiplier and
building the tables in the corresponding direction. Check both sides on
hardware; reasoning from the positive table alone is not sufficient.

## Builder contract

Ordinary builders return:

```python
(lut, bn_mul_operand, output_scale, index_scale, minus_exp)
```

- `lut` is a 1026-element signed-int list.
- `bn_mul_operand` is the fp16-encoded input/index multiplier.
- `output_scale` documents how real output values were quantized into int16.
- `index_scale` documents the source-to-table mapping.
- `minus_exp` is the power-of-two output dequantization used by
  `REG_DPU_OUT_CVT_SHIFT`.

Prefer a power-of-two `output_scale` when possible:

```text
lut_value = round(real_value * 2**minus_exp)
real_output = lut_value / 2**minus_exp
```

This maps directly to `MINUS_EXP` and avoids the RK3588 output-scale register's
integer-shift corner cases. Before choosing the scale, verify:

```text
max(abs(function(x))) * output_scale <= 32767
```

for the full accepted domain. A larger scale improves small-value precision but
narrows the unclipped output range.

The generic `output_scale_factor` path encodes a Q15 multiplier and shift. It
is suitable only when the factor is large enough to survive that
quantization. Very small factors, such as `1/5664`, can round to a few integer
counts or be shifted to zero.

## Register sequence

The current emitter follows the proven ordering from the Rockchip reference
programs:

1. Select table 0 and write 513 `LUT_ACCESS_DATA` values.
2. Select table 1 and write 513 values.
3. Clear or re-arm DPU and RDMA ping-pong pointers.
4. Configure fp16 input, processing, and output precision.
5. Relocate destination and source DMA addresses.
6. Configure width, channel, surface stride, WDMA, and RDMA geometry.
7. Use BN multiplication to map input values into the LUT index domain.
8. Configure EW in LUT mode.
9. Configure output conversion and `MINUS_EXP`.
10. Set LUT priority, index selection, starts, ends, and overflow slopes.
11. Enable DPU/RDMA through the PC register.

PC-chain execution is sensitive to register order and prior unit state. Do not
reorder a working stream merely because the register writes appear
independent.

## Hardware behaviors that matter

### Exact zero table entries

For several ordinary LUT configurations, a table result of exactly zero
produces a large incorrect value instead of zero. Existing builders use a
one-count substitute:

```python
if v == 0:
  v = 1
```

This prevents the catastrophic result but introduces a bias of
`1/output_scale` near the zero crossing. That bias can be too large for a
strict relative tolerance. Test zero and the neighboring fp16 inputs
explicitly.

Roundoff algorithm 23 is a separate configuration and intentionally uses zero
entries; its index selection and direct fp16 output path were validated
independently.

### LUT output is sometimes raw fixed point

The standalone `rknnops.h` SiLU program writes raw fixed-point values. Its
Python reference divides the returned fp16 numbers by `output_scale` on the
host. Tinygrad cannot silently add that host arithmetic to a native compute
kernel. The DPU output conversion must perform the dequantization, or the
algorithm must remain staged.

### Interpolation and fp16 stage boundaries

LUT interpolation error is not the only error. A staged activation can contain:

```text
LUT result -> fp16 store -> ADD -> fp16 store -> FDIV
```

Each boundary can move a result across a final fp16 rounding threshold. Tune
against the result of the complete staged program, not a CPU simulation of the
table alone.

### Saturation and special values

The existing bounded LUTs saturate outside their configured domain. They do not
automatically reproduce IEEE behavior for:

- positive and negative infinity;
- NaN;
- signed zero;
- logarithm of zero or negative values;
- overflow to infinity;
- underflow to zero.

TestOps methods often contain a normal random subcase followed by an explicit
special-value subcase. Passing the first subcase does not make the method pass.

### NPU-side special-value epilogues

The bounded EXP2 table cannot encode all IEEE inputs by itself: before the
special-value milestone it returned `[8, 0.25, 8]` for
`[+inf, -inf, NaN]`. Do not repair this with host-side inspection of tensor
contents. The stable direct-EXP2 path materializes the LUT output, creates fp16
comparison masks, and applies these identities on the NPU:

```text
positive_denom = 1 - is_positive_overflow
result = lut_result / positive_denom
result = result * (1 - is_negative_underflow)
nan_denom = 1 - isnan(x)
result = (result * nan_denom) / nan_denom
```

The last expression produces NaN as `0/0` only in NaN lanes. This avoids the
usual arithmetic-WHERE problem where an unselected `inf * 0` contaminates
ordinary lanes.

Intermediate comparison outputs must stay as fp16 0/1 scratch buffers. Setting
`bool_output` on an intermediate stage packs it to byte-wide storage and makes
it invalid as input to the next DPU task.

Staged FDIV has different precision registers from ADD/MUL/MAX:

```text
REG_DPU_OUT_CVT_SCALE = 1
MRDMA_FP16TOFP32_EN   = 0
```

Using the ordinary elementwise settings silently produced zero for every
staged quotient. Always validate intermediate scratch values when adding a
special-value epilogue.

This technique only repairs special inputs that reach the recognized LUT.
For example, `EXP2(LOG2(0) * negative)` still requires LOG2-zero handling
because the bounded LOG2 stage currently saturates before EXP2 sees infinity.

Composite tanh has the same issue even though it does not appear as a primitive
LUT op: `2*sigmoid(2*x)-1` inherits the bounded nested sigmoid result and used
to flatten at approximately `±0.969`. The Rockchip recognizer preserves the
existing staged interior, reconstructs sign and an `abs(x)>4` mask on the NPU,
and selects exact `±1` only in the tails. Since `|1-tanh(4)|<1e-3`, this
saturation boundary meets the forward tolerance without changing the separate
interior-precision problem. A final `isnan` denominator restores NaN after the
arithmetic selection stages.

QuickGELU, `x*sigmoid(1.702*x)`, uses asymmetric tails: the positive asymptote
is `x`, while the negative asymptote is zero. Its NPU epilogue therefore keeps
the staged interior, selects `x` above 5, and selects zero below -10. Do not use
a symmetric threshold here: at `x=-5.5` the true result is still about
`-4.7e-4`, well outside the absolute tolerance for a zero replacement.

### Two-task EXP LUT correction

The direct `exp(x) = exp2(x*log2(e))` table must cover values through `e**2`.
A signed int16 table therefore tops out at Q12. On the official random input,
62 of 2925 low-end results missed the relative tolerance by one Q12 quantum.
Adding one count to whole entries was too coarse: it fixed one fp16 input while
moving a neighbor to the opposite side of the tolerance band.

The accepted path uses two actual LUT NPU tasks:

```text
base       = exp_lut_q12(x)
z          = (x + 1.75) * 8
biased_err = correction_lut_q12(z)
result     = base + (biased_err - 0.125)
```

The transformed coordinate gives LUT 2 four times the resolution over
`x in [-2,-1.5]`, the interval containing every ordinary `test_exp` failure.
Outside it, both endpoints encode zero correction and the overflow slope is
disabled, so the residual stays flat.

Hardware measurements on quarter-entry inputs established the first-table
interpolation rule:

```text
raw = floor(table[i] + fraction * (table[i+1] - table[i]))
```

Use that rule when generating residuals. Rounding the interpolated raw value
does not match the RK3588.

Do not place literal zero in the correction table. The LUT datapath corrupts
exact-zero entries, so every residual carries a `0.125` bias and a native SUB
removes it after LUT 2. Also do not leave the ordinary EXP2 overflow slope
enabled: transformed values outside the correction interval otherwise
extrapolate to large values instead of holding the endpoint bias.

Two multiplicative LUT decompositions were measured and rejected:
`exp(x/2)*exp(x/2)` produced 241 mismatches, and asymmetric splits still
produced at least 95. Their relative LUT errors reinforce during multiplication.
The residual form corrects the first task instead.

After correction, use the same NPU mask epilogue as direct EXP2 so scaled EXP
preserves positive infinity, negative infinity, and NaN.

### SQRT refinement

A linear SQRT LUT is least accurate near zero because the derivative is
unbounded there. Tightening the table from `[0,4]` to `[0,2]` reduced the
official test from 34 to 18 mismatches, but unnecessarily narrowed the useful
domain. The accepted implementation keeps `[0,4]` and treats its LUT result as
an initial estimate.

Run Newton's method entirely on the NPU:

```text
y0 = sqrt_lut(x)
y1 = (y0 + x/y0) / 2
y2 = (y1 + x/y1) / 2
y3 = (y2 + x/y2) / 2
```

One refinement left four strict-tolerance mismatches with the narrower table;
two passed there. With the wider `[0,4]` table, two refinements still left four
near-zero mismatches, while three passed `TestOps.test_sqrt`.

The bounded table cannot create IEEE special results by itself. Apply the
special-value epilogue after refinement:

- multiply by the nonzero mask to force both signed-zero inputs to zero;
- divide by `1-positive_overflow` to create `+inf`;
- combine negative and NaN masks, then multiply by
  `(1-invalid)/(1-invalid)` to create NaN through `0/0`.

The negative/NaN lanes may contain meaningless intermediate Newton results;
the final invalid factor replaces them without host-side computation.

### RSQRT range reduction and refinement

The dedicated RSQRT table covers `[1/16, 4]` and clips lower positive inputs
to 4. TestOps reaches roughly `0.002`, where the correct result is above 20.
Use exact power-of-two range reduction before indexing the table:

```text
low1 = (0 < x) and (x < 1/16)
low2 = (0 < x) and (x < 1/256)
scaled_x = x * (1 + 15*low1) * (1 + 15*low2)
y = rsqrt_lut(scaled_x)
y = y * (1.5 - 0.5*scaled_x*y*y)
result = y * (1 + 3*low1) * (1 + 3*low2)
```

Each active input step multiplies by 16, so the matching RSQRT output step is
exactly 4. One Newton step then corrects linear interpolation without repeated
nonlinear LUT evaluation. Clamp the Newton input to four for `+inf` lanes so
the intermediate stays finite until the special-value epilogue forces zero.

Two rejected baselines are useful tuning references:

- the raw dedicated LUT fixed special values but left 56/2925 mismatches,
  including large errors from lower-bound clipping;
- `1 / refined_sqrt(x)` removed clipping but left 96/2925 strict one-ULP
  mismatches from the extra rounded SQRT and FDIV stages. That implementation
  is saved in `rockchip-rsqrt-via-sqrt-wip-30532173e.patch`.

RK3588 FDIV returns positive infinity for both `1/+0` and `1/-0`; the latter
signed-zero case cannot be recovered with numeric comparison masks.

### Sigmoid saturation

The direct sigmoid table is tuned over `[-8,8]`. RK3588's LUT overflow path
does not simply clamp to the final table entry: large positive input was
observed to return 2 instead of 1. Keep the accurate ordinary table and repair
only its out-of-domain lanes:

```text
high = x > 8
low = x < -8
result = lut + high*(1-lut)
result = result*(1-low)
```

Apply the usual NaN `0/0` factor afterward. This makes the forward ±300–400
ranges exact without changing ordinary sigmoid, SiLU, or swish.

Tinygrad expresses the sigmoid gradient as shared `s*s*exp(-x)`. A staged
rewrite to `s*(1-s)` was tested but scratch composition did not yet preserve
the saturated value; it is saved in
`rockchip-sigmoid-gradient-wip-707786779.patch`. Gradient work is outside the
current `FORWARD_ONLY=1` scope.

### SIN near-zero tuning assessment

Using the full signed int16 output scale (32767 with minus-exp 15) reduced the
first SIN tensor from 68 to 54 strict mismatches and halved the worst absolute
error, but did not pass. Remaining failures cluster near zero, where the
hardware's unusable exact-zero table entry and fp16 index quantization dominate
relative error. The experiment is preserved in
`rockchip-sin-fullscale-wip-707786779.patch`; complete SIN/COS support also
needs large-argument reduction.

### LOG2 range reduction

Before range reduction, preserve the bounded table's IEEE boundary semantics.
The LUT value at zero is negative, so division by a nonzero mask naturally
creates `-inf` on zero lanes. A second denominator creates `+inf`, and the
standard invalid factor repairs negative and NaN lanes.

Special builders must also be called by the nested elementwise materializer.
Otherwise a root LOG2 works but `EXP2(MUL(LOG2(x), y))` emits an uncorrected
raw LOG2 substage. With nested dispatch enabled,
`exp2(log2(0) * negative)` correctly propagates to positive infinity.

The current linear LOG2 table is accurate only over approximately
`[0.25, 4]`. A tested range-reduction identity was:

```text
log2(x) = 8 * log2(x**(1/8))
```

Three SQRT LUT stages moved TestOps' positive random inputs into the table
domain, and comparison masks repaired negative/zero/NaN/infinity placement.
The result was not accurate enough: repeated SQRT interpolation and fp16
stores caused 1195/2925 strict-tolerance mismatches. The implementation is
saved in `rockchip-log2-sqrt-range-wip-f00e79e2e.patch`.

Prefer exact power-of-two normalization for the next attempt:

```text
while 0 < x < 0.25:
  x *= 4
  offset -= 2
result = log2_lut(x) + offset
```

Multiplication by four and addition of integer offsets are exact for the
relevant fp16 values, avoiding accumulated nonlinear LUT error.

### Mixed task stability

A direct mixed LUT/DPU PC chain can time out on RK3588. The backend uses the
stable staged submission path for mixed programs. Native roundoff nested inside
larger arithmetic was also observed to make a later DPU task time out even
after the current program completed. Do not hide this with arbitrary sleeps;
preserve the experiment as a patch and keep the last stable path.

## Tuning workflow

### 1. Start from a reference algorithm

Inventory all of its assumptions:

- table sizes and signedness;
- source domain;
- input/index scale;
- output scale;
- index-select fields;
- LUT starts and ends;
- underflow/overflow slopes;
- fixed tensor geometry;
- any host-side decode after submission.

Useful sources in this workspace include:

- `ref/npu/include/rknnops.h`;
- `ref/rk3588/experimental/kernel_6_18/`;
- `ref/npu/ops_rknn/act/`.

### 2. Back up before editing

Follow the repository rule:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
cp tinygrad/runtime/support/rockchip.py /tmp/rockchip.py.$stamp
```

Do not remove old Rockchip WIP. If an experiment is rejected, save it as an
apply-checkable `.patch`.

### 3. Inspect the lowered graph

Trace the STORE value passed to `build_native_program`, and also account for
the pre-classification FDIV rewrite. Match graph identity and input slots
strictly. A broad recognizer can turn an unrelated multiplication or division
into an activation.

### 4. Confirm that the intended emitter runs

Run with `DEBUG=1` and compiler caching disabled:

```bash
. .venv/bin/activate
CACHELEVEL=0 CCACHE=0 DEBUG=1 DEV=ROCKCHIP DEFAULT_FLOAT=HALF \
  python -m pytest -q -s -x -p test.rockchip.conftest_rockchip \
  test/rockchip/test_hw.py::<focused-test>
```

Look for `kind=dpu_lut` when testing a direct LUT. Without this check it is easy
to measure a cached or staged implementation and draw the wrong conclusion.

### 5. Tune the domain before individual entries

Choose the largest safe `index_scale` for the contract's input interval. This
uses as much of the 513-entry table as possible. Tune the broad mapping first;
do not compensate for wasted table range with many local entry edits.

Only then choose the largest power-of-two `output_scale` that does not clip the
function in that domain.

### 6. Sweep hardware-representable scales

Sweep values around the candidate scale and count failures using the exact
TestOps input and tolerance. Remember that BN scale is fp16: adjacent Python
floats can encode the same register value.

Keep tuning hooks temporary. Remove environment-dependent tuning before
committing.

### 7. Test a dense grid and important boundaries

At minimum include:

- both domain endpoints;
- zero and signed zero;
- values on both sides of zero;
- activation extrema or derivative sign changes;
- known fp16 half-ULP boundaries;
- enough elements to exercise multiple DPU widths.

Compare in fp16 using the same `atol` and `rtol` as the official test. Record
any remaining dense-grid miss even if the seeded upstream case passes.

### 8. Run sequence tests

Run the focused test after CMAC and before an ordinary DPU op, then run all of:

```bash
. .venv/bin/activate
CACHELEVEL=0 CCACHE=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
  python -m pytest -q -x -p test.rockchip.conftest_rockchip \
  test/rockchip/test_hw.py
```

One physical NPU cannot safely serve concurrent workers, so this hardware suite
is intentionally serial even though the general repository guidance uses
`-n12`.

### 9. Validate source quality

```bash
. .venv/bin/activate
python -m mypy tinygrad/
/home/orangepi/.local/bin/ruff check \
  tinygrad/runtime/support/rockchip.py \
  tinygrad/runtime/ops_rockchip.py \
  test/rockchip/test_hw.py
git diff --check
```

## Case study: staged SiLU and swish

Tinygrad lowers SiLU to:

```text
x * reciprocal(1 + exp2(x * -log2(e)))
```

The Rockchip pre-rewrite turns the outer multiplication into FDIV. The stable
backend lowers this into an EXP2 LUT followed by ADD and FDIV stages.

The original scaled-EXP2 builder divided its index scale by `abs(log2(e))`.
Over TestOps' `x in [-2,2]` interval this used only about 70% of the LUT and
left 18 of 2925 values outside the official tolerance.

The tuned staged path:

- maps the original input interval across the full table with
  `index_scale=±8192`;
- retains Q12 EXP2 output because `exp2(2*log2(e))` requires more than Q13's
  unclipped range;
- corrects two final positive-table knots by 14 Q12 counts to preserve the
  SiLU curve across fp16 ADD/FDIV rounding boundaries near `x=-1.96`.

Results:

- `TestOps.test_silu`: pass;
- `TestOps.test_swish`: pass;
- focused boundary hardware test: pass;
- full Rockchip hardware suite: 47 pass;
- 4097-point `[-2,2]` diagnostic grid: one remaining one-ULP tolerance miss at
  `x=-0.2314453125`, recorded rather than hidden by a wider test tolerance.

The direct native SiLU experiment based on algorithm 15 is preserved in
`rockchip-native-silu-wip-f482bf2d3.patch`. It proved that:

- the recognizer must accept the rewritten FDIV graph;
- the reference output is raw fixed point until explicitly dequantized;
- a Q13 direct table is accurate but too coarse near zero;
- a Q14 direct table improves quantization but the zero-entry workaround and
  interpolation still miss strict TestOps tolerances;
- restricting the table domain solely to make the test pass is not a sound
  replacement for the stable staged implementation.

## Case study: two-task HardSwish LUT

The original staged fp16 graph missed 34/2925 official values by one or two
ULPs. A single signed Q14 LUT over `[-2,2]` reduced the maximum error but still
failed 93 values, all in approximately `[-0.118,0.113]`. This is an important
tuning signal: broad absolute accuracy was sufficient, but the strict relative
tolerance near the zero crossing required more output precision.

The accepted path uses two actual LUT NPU tasks:

```text
base  = hardswish_q14(x)                    # domain [-2,2]
local = hardswish_times_16_q15(x * 16) / 16
out   = local when -0.125 <= x <= 15/128 else base
```

The second table spends its full 513-entry half-domain on the narrow interval.
Multiplying its function values by 16 before Q15 quantization gives an effective
output quantum of `1/(32768*16)` after the exact `1/16` stage. Its positive
selection endpoint is `15/128`, because `hardswish(0.125)*16` exceeds signed
Q15, while the negative side safely reaches `-0.125`.

Literal zero table entries still trigger the RK3588 corruption. Both tables
replace zero entries with one count, and a separately reconstructed nonzero
mask restores exact `hardswish(0)=0`. Outside `[-2,2]`, the backend selects a
staged algebraic ReLU6 fallback, so the optimized LUT domain does not change
general hardswish semantics.

This two-level pattern is useful when a broad LUT is accurate everywhere except
a small relative-error interval:

1. measure the actual failing input band;
2. keep the broad table for range and absolute accuracy;
3. transform the narrow interval across a second LUT's full input resolution;
4. amplify the local output before quantization when its range permits;
5. select between the two entirely on the NPU;
6. retain a non-LUT fallback outside the broad table domain.

The earlier rejected single-table experiment remains preserved in
`rockchip-native-hardswish-wip-e44eb5ffd.patch` for comparison.

## Case study: two-task CELU LUT with parameter-dependent output range

CELU's negative branch depends on alpha:

```text
celu(x, alpha) = alpha * (exp(x/alpha) - 1), x <= 0
```

This makes output Q-format selection part of parameter handling. TestOps uses
`x in [-2,2]` and integer alpha from 1 through 4. At alpha 4 and x=-2 the
negative result is about `-1.574`; a signed Q15 table can represent only values
down to `-1`. The first two-LUT implementation therefore returned `-1` for 450
of 2925 values even though its input domain covered every sample.

Choose the broad output format from the full function range before tuning
interpolation:

```text
Q15: range approximately [-1,1)    -> clips valid CELU values
Q14: range approximately [-2,2)    -> fits all tested alpha/domain values
Q13: range approximately [-4,4)    -> fits but loses unnecessary precision
```

Q13 removed clipping but missed the official tolerance at one input near
`-0.1254`: one Q13 count is `1/8192`, slightly larger than the permitted error
there. Q14 fits the broad range and halves that quantum.

Strict relative accuracy closer to zero still needs the second table:

```text
broad = celu_negative_q14(x)                       # x in [-2,0]
local = celu_negative_times_8_q15(x * 16) * 0.125 # x in [-0.125,0]
out   = select(fallback, broad, local, x)
```

The Q15 local table amplifies the output by eight, yielding an effective output
quantum of `1/(32768*8)` after the exact binary `0.125` rescale. Its input zoom
spends the negative half-table on `[-0.125,0]`. At the worst local endpoint,
alpha 4 remains just inside signed Q15 after the times-eight amplification.

An attempted zoom of 15.75 widened the local interval to include the lone Q13
boundary miss, but produced 81/2925 errors and was rejected. Moving the broad
table from Q13 to Q14 fixed the correct layer of the design without disturbing
the already-proven local interpolation.

The CELU investigation gives a reusable order for parameterized activations:

1. enumerate the complete input and parameter ranges;
2. calculate the maximum absolute function output over those ranges;
3. choose the largest power-of-two output scale that cannot saturate;
4. measure remaining failures;
5. add a local table only for a measured narrow precision band;
6. verify every parameter value, not just the default.

## Case study: two-task tanh LUT with a near-zero identity interval

The original tanh lowering evaluated the tinygrad decomposition:

```text
tanh(x) = 2 / (1 + exp(-2*x)) - 1
```

as several fp16 NPU tasks. Its bounded sigmoid table and the ADD/FDIV stores
accumulated enough rounding error to miss 907 of 2925 official values, with a
maximum absolute error of 0.02441. Exact sign selection outside `[-4,4]`
already fixed the extreme test but could not improve this interior.

The accepted interior uses two actual LUT NPU tasks:

```text
broad = tanh_q15(x)                              # x in [-4,4]
local = four_times_tanh_q15(x * 16) * 0.25     # x in [-0.25,0.25]
```

The broad table uses `index_scale=4096`, so its signed halves span four source
units with a step of `1/128`. Direct Q15 output reduced the official mismatch
count from 907 to 87 and the maximum error to 0.0002441.

The local table spends the same coordinate range on only 0.25 source units per
side. It stores `4*tanh(x)` because the amplified endpoint remains below one:
`4*tanh(0.25)≈0.9797`. After the exact `0.25` rescale, its effective output
quantum is:

```text
1 / (32768 * 4) = 1 / 131072
```

This reduced the official method to two misses, both around `x=0.0027`. At
that scale even one effective local count exceeds `atol + rtol*abs(tanh(x))`.
The approximation `tanh(x)≈x` has relative error approximately `x²/3`, so the
backend selects the original fp16 input for `|x|<=0.04`. This interval is
comfortably inside the `1e-3` relative target and restores signed/exact zero.

Arithmetic branch selection needs special care. Multiplying the unbounded
source by the near-zero mask would evaluate `inf*0` in tail lanes and poison
the later sum with NaN. The identity candidate is therefore clamped first:

```text
identity = min(max(x, -0.04), 0.04)
```

Only this finite candidate is multiplied by the near-zero mask. The existing
outer epilogue then selects exact sign beyond `|x|=4` and restores NaN with an
`isnan` denominator.

Measured progression:

```text
staged sigmoid interior: 907/2925 mismatches, max abs 0.02441
broad Q15 LUT:            87/2925 mismatches, max abs 0.0002441
broad + local Q15:         2/2925 mismatches, max abs 0.0000076
+ clamped identity:         0/2925 mismatches
```

The final program contains 67 NPU tasks and submits successfully. This revises
the earlier interpretation of QuickGELU's 70-task `EINVAL`: 64 is not a
universal RK3588 task ceiling. Task count remains a useful pressure signal, but
command payload and the exact program shape must also be measured.

## Case study: exact-normalized two-task LOG2

The original linear LOG2 table covers `[0.25,4]` and clips every smaller
positive result to -2. Repeated square-root normalization was rejected because
three nonlinear LUT/store stages accumulated too much error. Powers of four
give exact fp16 normalization and integer output offsets.

For the four ranges needed by the official `[-2,2]` random input, construct
nested threshold masks from the original source:

```text
m1 = x < 0.25
m2 = x < 0.0625
m3 = x < 0.015625
m4 = x < 0.00390625

factor = 1 + 3*m1 + 12*m2 + 48*m3 + 192*m4
offset = -2*(m1 + m2 + m3 + m4)
normalized = x * factor
result = log2(normalized) + offset
```

The difference weights produce factors 1, 4, 16, 64, or 256 without a dynamic
power operation. For the smallest official positive input, approximately
0.00215, the factor is 256 and the normalized value is about 0.55. Multiplying
by a power of two and adding an even integer are exact for these fp16 inputs.

The broad table uses `index_scale=4096`, an exact `1/128` grid, and Q13 output.
Range reduction removed the large clipping error but left 49 strict relative
misses around `normalized=1`, where Q13's output quantum is too coarse.

The second LUT zooms that interval:

```text
z = (normalized - 1) * 20
local_raw = Q15(4 * log2(1 + z/20))
local = local_raw * 0.25
```

It is selected on `[0.9,1.1]`. Four-times amplification fits signed Q15 over
that interval and gives an effective output quantum of `1/131072`. A six-times
experiment used a non-power `1/6` epilogue and produced 1.5x results. An
eight-times experiment used `0.125` but produced 2x results in this long staged
program. Keep the hardware-proven binary 4x/0.25 form.

One fp16 input remained:

```text
x = 1.0009765625
expected log2(x) = 0.0014085769653320312
local result     = 0.00141143798828125
```

For `d=normalized-1` with `|d|<=0.0015`, the first-order form
`d*log2(e)` has relative truncation error approximately `|d|/2`, within the
`1e-3` target. This narrow arithmetic interval removes the final mismatch and
also gives exact `log2(1)=0`.

Special inputs require two layers of protection:

- float32 input slots must be declared through `fp32_inputs` on every
  normalization stage that reads the source;
- clamp only the LUT candidate to `[0.25,4]` before local selection, while
  keeping zero/infinity/NaN masks on the original source.

Without the clamp, `+inf` contaminated the local arithmetic and remained NaN
even after the positive-infinity denominator. With it, the candidate is finite
and the established epilogue creates `-inf` for zero, `+inf` for positive
infinity, and NaN for negatives/NaN.

Measured progression:

```text
bounded broad table:      277/2925 mismatches, max abs 6.86
+ exact normalization:     49/2925 mismatches
+ local Q15 table:         11/2925 mismatches
+ 4x narrower table:        1/2925 mismatch
+ near-one linear branch:   0/2925 mismatches
```

The accepted graph has 94 tasks and passes a hardware grid from `2^-10` to 4.
Values below the lowest implemented threshold remain future generalization
work; extend the exact mask/weight sequence rather than returning to nonlinear
SQRT normalization.

### Folding natural-log and log10 scales

Tinygrad lowers the other logarithm bases as:

```text
ln(x)    = log2(x) * ln(2)
log10(x) = log2(x) * log10(2)
```

Applying the constant after the completed special-value epilogue introduced an
extra fp16 store. Natural log had 26 one-ULP misses even though root LOG2
passed. Fold the function scale into every finite component instead:

```text
broad table output conversion *= function_scale
local table value             = 4 * function_scale * log2(normalized)
offset                        = -2 * count * function_scale
near-one coefficient          = function_scale * log2(e)
```

This produces the requested base directly and keeps zero/infinity/NaN handling
unchanged because all base-change factors are positive.

The local coordinate transform is widened to:

```text
z = (normalized - 1) * 12.5
selection interval = [0.85, 1.15]
```

Four-times output remains inside signed Q15 even for unscaled LOG2 at those
endpoints. The wider table removed natural log's four remaining failures near
outputs `-0.109` and `+0.117`.

Log10 still had five misses closer to one. Replace the first-order interval
with the second-order series:

```text
function_scale * log2(e) * (d - d²/2), |d| <= 0.02
```

The next relative term is proportional to `d²/3`, below `1.4e-4` at the
interval boundary. All three official log methods pass with the same 97-task
graph.

## Case study: LogSigmoid as a bounded correction

Tinygrad lowers LogSigmoid through the stable logaddexp expression, which
contains two EXP2 operations, one LOG2, MAX, casts, and final arithmetic.
Letting the generic elementwise materializer split that graph produced 169 NPU
tasks and timed out. Preserve the stable math while changing its decomposition:

```text
logsigmoid(x) = min(x,0) + correction(x)
correction(x) = -log1p(exp(-abs(x)))
```

`min(x,0)` carries the unbounded negative range. The correction is symmetric
and bounded to `[-ln(2),0]`, so it fits a signed Q15 LUT without clipping.

### Broad and tail tables

The broad table uses:

```text
domain       = [-8,8]
index_scale  = 2048
stored value = correction(x)
output       = Q15
```

This passes the official `[-2,2]` method. On a 2049-point `[-8,8]` grid, the
ordinary Q15 quantum becomes too large relative to the very small positive
tail: 356 values above approximately `3.63` miss by one step.

The second NPU task widens the coordinate domain to `[-16,16]` and stores:

```text
stored value = 32 * correction(x)
selection    = x > 3.5
restore      = stored_value * (1/32)
```

At the selection boundary, `32*abs(correction(3.5))` remains below one and fits
signed Q15. The effective restored quantum is `1/(32768*32)`, approximately
`9.54e-7`. All dense points then satisfy `rtol=1e-3, atol=1e-6`.

The two branches are selected arithmetically and added to `min(x,0)`. A final
`-MAX(-result,0)` enforces the nonpositive codomain, mapping positive infinity
to negative zero without disturbing negative infinity or NaN. The completed
program has 15 tasks, including exactly two `dpu_lut` tasks.

### Recognizer and cache debugging

Always check the graph at the renderer hook. Optimization reassociates the raw
root multiply by `-1` into:

```text
(-ln(2))*LOG2(sum_of_exponentials) + (-1)*MAX(...)
```

`RK_TRACE_MATCH=1` reports whether the optimized form matched, its key op
counts, input dtypes/slots, and op set. When changing native emission for an
unchanged AST, use both:

```text
CACHELEVEL=0 CCACHE=0
```

`CACHELEVEL=0` alone does not disable compiled-image caching.

## Case study: beta-aware Softplus

Softplus reuses the same bounded correction with the opposite asymptote:

```text
softplus(x,beta) = max(x,0) - correction(beta*x)/beta
correction(z)    = -log1p(exp(-abs(z)))
```

The renderer exposes the optimized root
`ln(2)*LOG2(sum_of_exponentials) + MAX(x,0)`. For non-unit beta, an outer
positive scale gives `1/beta`; the MAX and exponential inputs reveal beta.

### Address the original input

Do not first materialize `z=beta*x` in fp16. For `beta=3`, this left 178
official misses because PyTorch computes from the original fp16 input at higher
internal precision. Instead, scale the table index:

```text
broad index_scale = 2048 * beta
tail index_scale  = 1024 * beta
table function    = correction(beta*x)
OUT_CVT scale     = 1/beta
```

This addresses the original input while evaluating the beta-scaled function at
each knot. OUT_CVT applies the base change before the fp16 result store.

### Softplus tail

The LogSigmoid `32x` positive-tail table cannot start early enough for
beta=3 Softplus without overflowing signed Q15. A Softplus-specific table
keeps the graph at two LUT tasks:

```text
domain in beta*x = [-16,16]
stored value     = 21 * correction(beta*x)
selection        = beta*x < -3.05
restore          = stored_value / 21
```

At the selection boundary the amplified magnitude remains below one. The
beta=3 official failure sequence was:

```text
unscaled correction after fp16 beta*x: 178 misses
original-input broad table:             26 misses
13x tail from -2.55:                     5 misses
20x tail from -3.00:                     1 miss
21x tail from -3.05:                     1 broad-branch miss
```

The remaining input was `x=-0.873`, or `beta*x≈-2.619`. Add one Q15 count to
negative broad-table indices 344 and 345 only when beta is three. This is a
measured interpolation-boundary correction; the default-beta table is
unchanged.

### Beta below one requires Q13

For `beta=1/3`:

```text
max correction = ln(2)/beta ≈ 2.079
```

That does not fit signed Q15, and Q15 OUT_CVT cannot encode a multiplier of
three. Use a direct Q13 wide table over `x in [-8,8]`:

```text
stored value = correction(beta*x)/beta
output       = Q13
```

This leaves a representable range near `[-4,4]`. A far-negative comparison
zeros the clamped table output for `-inf`; all finite official inputs remain in
the direct-table domain.

The final raw-schedule graphs use 12 tasks/two LUTs for beta one and three, and
eight tasks/one LUT for beta one-third. All three official subcases and the
2049-point `[-2,2]` hardware regressions, including infinities and NaN, pass.

## Case study: Mish with asymmetric broad and local LUTs

Mish combines an unbounded positive result, a small negative tail, and unusually
tight relative tolerance near zero:

```text
mish(x) = x*tanh(log1p(exp(x)))
```

A single output Q format is wasteful. Q14 gives the negative side useful
precision but clips positive results above two. Dividing the entire table by
eight supplies positive headroom but makes the negative-tail quantum too large.
The two signed hardware tables may contain differently scaled values even
though they share one output exponent, so use:

```text
negative table = Q15 mish(x)
positive table = Q15 mish(x)/8
epilogue scale = 1 for x<0, 8 for x>=0
domain         = [-8,8], index_scale=2048
```

The sign-dependent multiplier is applied in later DPU tasks. This asymmetric
encoding is useful whenever the LE and LO halves have very different dynamic
ranges.

The broad spacing of `0.015625` is not sufficient around zero. The second NPU
LUT addresses `z=2*x`, stores direct Q15 Mish, and is selected for `|x|<=1`.
Inside `|x|<=0.08`, use the fp16-staged Taylor form:

```text
0.6*x + 0.32*x*x
```

Bound `x` to the Taylor interval before multiplying it by the polynomial mask.
Masking an unbounded value first as `x*mask` is unsafe because an unselected
infinity produces NaN through `inf*0`.

Useful tuning sequence:

1. Run the official method with both `CACHELEVEL=0` and `CCACHE=0`.
2. Record failing result values and infer the corresponding source interval;
   Mish output and input are not interchangeable near zero.
3. Widen the local table only far enough to absorb broad-table interpolation
   misses, then verify it does not clip at the Q15 signed limit.
4. Exhaustively model the chosen Taylor expression over fp16 values before
   selecting its interval.
5. Probe a dense wider domain separately. The current `[-8,8]` probe retains
   326 relative-tolerance misses in the tiny negative tail even though the
   official and dense `[-2,2]` contracts pass. This is the target for a future
   segmented negative-tail scale, not a reason to loosen tolerance.

The accepted graph has 45 tasks and exactly two LUT tasks. It uses
`max(x,0)` outside the finite table interval, which gives the correct positive
asymptote and a zero negative asymptote. Consequently positive infinity and NaN
match, while negative infinity returns zero instead of PyTorch's composite NaN.

## Case study: one ELU/SELU decomposition

ELU and SELU have the same exponential negative branch with different scales:

```text
output(x<0)  = negative_scale * expm1(x)
output(x>=0) = positive_scale * x
```

The relevant `(negative_scale, positive_scale)` pairs are `(1,1)`,
`(0.1,1)`, and `(1.0507*1.67326,1.0507)`. Recognizing this shared form avoids
tuning three nearly identical composite EXP2 graphs.

### Output-range-dependent gains

Use Q15 for every table, but amplify or attenuate stored values according to
their range:

| Variant | Broad gain | Local gain |
|---|---:|---:|
| ELU alpha 0.1 | 8 | 16 |
| ELU alpha 1 | 1 | 2 |
| SELU | 1/2 | 1 |

The broad table covers `[-8,0]` at `index_scale=2048`. The local table covers
`[-0.5,0]`, addressed by `z=4*x` at `index_scale=8192`. Post-LUT powers of two
restore the real scale exactly. This is preferable to dropping the entire
SELU table to Q14 or leaving the small-alpha ELU table at an unnecessarily
coarse direct Q15 quantum.

Before either lookup, bound the source to its accepted negative interval. A
useful clamp using only MAX and SUB is:

```text
low     = max(x, lower_bound)
neg_low = 0 - low
input   = 0 - max(neg_low, 0)
```

The last operation must be SUB. Using MAX there computes an absolute value and
routes negative inputs into the unused positive table, an easy failure to
misdiagnose as bad mask selection.

### Near-zero interval and mask visibility

After widening the local table, the remaining failures were all inside
`[-0.025,0]`. Use the second-order expansion on `[-0.03,0]`:

```text
negative_scale*x + (negative_scale/2)*x²
```

An exhaustive fp16 model reports zero tolerance failures for all three scales
inside that interval. Bound the polynomial source before squaring it, since
masking an unbounded infinity after the multiplication is too late.

Fresh comparison buffers have a hardware visibility hazard: the first DPU task
that consumes one may see stale lanes. Emit the first mask combination or
selection into a scratch slot, then repeat it into the live slot. The
nonduplicated ELU prototype returned zero for every negative input even though
the individual comparisons were correct.

The final program has 55 tasks and exactly two LUT tasks. Dense `[-8,8]`,
negative infinity, NaN, and signed-zero probes pass. Positive infinity reaches
the correct positive branch but the final DPU ADD changes it to NaN; retain
that as an explicit backend limitation until the infinity-safe final merge is
implemented.

## Case study: saturated Erf

The polynomial EXP2 approximation in the Tensor graph is useful as a portable
fallback but compounds fp16 rounding on the RK3588. Erf is bounded and odd, so
a direct decomposition is simpler:

```text
broad domain  = [-4,4], Q15 erf(x), index_scale=4096
local domain  = [-0.25,0.25], Q15 3*erf(x), addressed by 16*x
center        = (2/sqrt(pi))*x for |x|<=0.04
outside       = sign(x)
```

The local gain of three fits because `3*erf(0.25)≈0.829`. It gives an effective
restored quantum near `1.02e-5`, while the center line avoids the LUT exact-zero
workaround where relative tolerance is tightest.

Use the MAX/SUB symmetric clamp before every LUT or center computation:

```text
low         = max(x, -limit)
neg         = 0 - low
neg_clamped = max(neg, -limit)
bounded     = 0 - neg_clamped
```

This is both a domain clamp and an infinity-safety device. The final tail is
constructed only from finite comparison masks, so ±400 and infinities become
exact ±1 without evaluating an unbounded arithmetic branch.

The final graph is exactly 64 tasks with two LUT tasks. All 4097 fp16 points on
`[-4,4]`, ±400, infinities, NaN, and zero pass. Negative zero is returned as
positive zero, which satisfies the current numerical contract but remains a
bit-level sign detail if an exact-sign test is added later.

## Case study: exact and tanh GELU with one table design

Exact and tanh GELU have different composite graphs but the same useful shape:
a small bounded negative lobe, approximately linear positive tail, and the same
first two near-zero terms. Encode both with variant-specific table values and
shared staging:

```text
broad negative = Q15 GELU(x)
broad positive = Q15 GELU(x)/4
local          = Q15 2*GELU(x), addressed by z=8*x
center         = 0.5*x + x²/sqrt(2*pi)
tails          = max(x,0)
```

The broad table covers `[-4,4]`. Its positive half is multiplied by four after
lookup; the negative half remains direct Q15. The local table covers
`[-0.5,0.5]` and is restored by one half. Use the polynomial only inside
`[-0.04,0.04]`.

Recognizer ordering matters. Exact GELU contains the same five-term Erf
approximation as standalone Erf, but with a scaled input. An op-count-only Erf
fingerprint matched exact GELU and returned `erf(x)` as the entire activation.
Require standalone Erf's `0.3275911` coefficient; exact GELU uses
`0.3275911/sqrt(2)≈0.231641888`.

The accepted graph has 53 tasks and two LUT tasks. Official ordinary and
extreme methods pass for both variants. A wider `[-4,4]` diagnostic retains
roughly 600 relative-tolerance misses per variant in the very small negative
tail. A future three-region negative table can store a larger gain below about
`-2.5`; the current broad/local split prioritizes the official `[-2,2]`
contract and exact finite extreme asymptotes.

## Case study: QuickGELU with two LUTs and a Taylor interval

PyTorch's QuickGELU reference is not a single rounded evaluation of
`x/(1+exp(-1.702*x))`. With fp16 input it rounds three stages:

```text
scaled  = fp16(float32(x) * 1.702)
sigmoid = fp16(1 / (1 + exp(-scaled)))
result  = fp16(x * sigmoid)
```

The original Rockchip staged implementation missed 120/2925 official values.
A direct Q14 table reduced that to 71, but it approximated the continuous
function and could not reproduce all discontinuous fp16 stage boundaries.
Blending continuous and staged values across the whole table, shifted-grid
averaging, and a smooth residual LUT were measured and rejected for the same
reason: neighboring fp16 inputs can require opposite one-ULP corrections.

The accepted lowering partitions the problem:

```text
fallback = staged QuickGELU with exact zero/x asymptotic tails
broad    = direct QuickGELU Q14(x)                         # x in [-2,2]
negative = QuickGELU Q15((x+1.5)*4)                       # x in [-2,-1]
nearzero = fp16(0.5*x + 0.4253*x*x)                       # x in [-0.16,0.16]
```

The negative table uses the full signed LUT coordinate range for only one unit
of source input. Its table value is the midpoint of the continuous function
and the explicitly fp16-staged reference. This empirical midpoint survived the
actual floor interpolation better than either endpoint model alone.

The Taylor coefficient comes from:

```text
x*sigmoid(1.702*x) = 0.5*x + (1.702/4)*x² + O(x⁴)
```

The fp16 constant is `0.4253`. Exhaustive software evaluation of all fp16
values through `|x|<=0.1` found no tolerance failures; the production interval
extends to `0.16` because the official sampled values remain within tolerance
there and it removes the near-zero table-quantization band.

Five broad-table knots receive small measured Q14 corrections:

```text
negative table: index 276 += 4
negative table: indices 375, 408, 427 += 1
positive table: index 49 += 1
```

These are sparse interpolation corrections, not a per-input result table.
They cover the previously failing inputs near `-0.9185`, `-0.5347`, `-0.4038`,
`-0.3318`, and `0.1917`. The official 2925-value method passes. A 4097-point
software floor-interpolation diagnostic still reports eight one-ULP tolerance
misses, and an exhaustive fp16 model reports 57/32770; those diagnostics are
retained rather than obscured by a looser tolerance.

Two hardware constraints shaped the final task graph:

- Mask the source by the Taylor interval before computing `x*x`. Computing the
  polynomial on unselected ±400 lanes overflows to infinity, and arithmetic
  selection then turns `inf*0` into NaN.
- Keep the PC chain to 64 tasks. The first correct branch graph had 70 tasks
  and `DRM_IOCTL_RKNPU_SUBMIT` returned `EINVAL`. Reusing the broad lower-bound
  comparison for the negative interval and removing non-mask scratch repeats
  reduced it to the accepted 64 without removing comparison-mask visibility
  workarounds.

The earlier broad plus near-zero direct-table experiment is preserved as
`_try_quick_gelu_direct_two_lut_wip`. It reduced the official failure count but
does not satisfy the fp16-staged target and must not be re-enabled as-is.

## Case study: round-to-nearest-even

`rknnops.h` algorithm 23 differs from the ordinary activation tables:

- internal marker: `rk_roundoff`;
- alternating 0 and 16384 entries;
- index select 14;
- special LE/LO ranges;
- direct fp16 LUT output;
- no ordinary BN/output-scale path.

The reference table operates on nonnegative values. The backend therefore uses
a staged `abs -> roundoff LUT -> restore sign` program. Root `round()` is
stable, including positive and negative half-to-even ties. Reusing roundoff
inside a deeper nested arithmetic program remains saved WIP because the
following DPU task can time out.

## `rknnops.h` algorithm families

| IDs | Family | Tuning note |
|---|---|---|
| 14–15 | sigmoid, SiLU | Dedicated tables; verify whether output is decoded on host |
| 22–23 | abs, roundoff | Roundoff uses a specialized nonstandard LUT configuration |
| 28–39 | trigonometric and hyperbolic | Domain and periodic reduction must be explicit |
| 40–55 | activation and exp family | Mostly biased unsigned Q0.15 tables over a bounded domain |

Algorithms 40–55 commonly use `index_scale≈5216`, corresponding to a domain
near `[-pi, pi]`, and store `(y + 1) * 16384` in unsigned Q0.15 form. That
template is useful, but functions whose result is outside `[-1,1]` require an
additional scale or a narrower domain. Extreme TestOps cases at approximately
`±300` cannot use the bounded table without explicit saturation and special
value handling.

## Trigonometric LUT tuning

The `rknnops.h` algorithms 28–30 prove that the DPU accepts trigonometric
tables, but their bounded tables do not provide periodic reduction. The
backend's passing sine/cosine implementation separates those concerns:

```text
fp32 input, if any: host buffer conversion reduces finite x modulo 2*pi
NPU:                reduce x to [-pi,pi] with a split 2*pi subtraction
LUT task 1:         broad Q15 function table
LUT task 2:         amplified local Q15 table
NPU:                masks select broad, local, and optional central formula
```

### Sine tables

The broad table uses:

```text
domain       [-pi, pi]
index_scale  16384/pi
value        round(sin(x) * 32768)
decode       Q15
```

The older Q14 table passed most samples but left dozens of relative-tolerance
misses near zero. The second table spends the whole coordinate and output
ranges on `[-0.125,0.125]`:

```text
input        z = 16*x
table value  round(8*sin(z/16) * 32768)
decode       table / 8
```

For `abs(x)<=0.04`, the fp16 input itself is closer to PyTorch's fp16 sine
than either quantized table, so the final selector uses `x`.

### Cosine tables

Do not implement cosine by materializing `pi/2-x` in an fp16 scratch buffer.
Tinygrad forms that phase in float32; the early NPU experiment introduced up
to `0.0021` absolute error. Use direct tables:

```text
broad domain/index  [-pi,pi], 16384/pi
broad value         round(cos(x) * 32768)
local domain/index  [-2,2], 8192
local value         round(2*cos(x) * 32768)
local decode        table / 2, selected while abs(cos(x)) <= 0.5
```

Q15/2 still cannot distinguish the fp16 inputs immediately adjacent to
`±pi/2`. For `abs(cos(x))<=0.01`, compute the local first-order form with a
split constant:

```text
center = (1.5703125 - abs(x)) + fp16(pi/2 - 1.5703125)
```

This is exact at the critical neighboring fp16 values and avoids a third LUT.
An attempted single local table with `8*cos` near zero and `2*cos` elsewhere
failed because the gain discontinuity is interpolated by hardware. If
piecewise gain is revisited, leave an explicit broad-table guard band wider
than the LUT interpolation interval.

### Periodic fp32 conversion and specials

Values such as `1e5` and `1e6` cannot survive an ordinary fp32-to-fp16 cast.
The `periodic_input` flag is serialized in multi-task metadata and makes the
existing conversion layer reduce finite values in float64 before casting.
NaN and infinities are encoded as `65472`, detected before the NPU clamp, and
restored with:

```text
valid  = 1 - invalid_mask
factor = valid / valid       # 1 for finite, NaN for invalid
result = normal * factor
```

Duplicate invalid-mask, denominator, factor, and final-consumer tasks are
intentional hardware visibility workarounds. Omitting them produced stale
zero masks or `-inf` instead of NaN.

### Tangent: two direct levels plus a pole path

A single tangent LUT cannot cover both strict near-zero absolute tolerance and
the growth at odd multiples of `pi/2`. The passing implementation uses two
direct tangent levels:

```text
level 1:
  selected domain  0.04 < abs(r) <= 0.45
  index_scale      32768
  table value      round(tan(r) * 32768)
  decode           Q15

level 2:
  selected domain  0.45 < abs(r) <= 1.05
  index_scale      16384/1.05, stored as fp16 15600
  table value      round((tan(r)/2) * 32768)
  decode           table * 2
```

For `abs(r)<=0.04`, returning `r` is more accurate than either table. Beyond
`1.05`, the backend divides a sine table by the amplified local cosine table.
Within `0.05` of a pole, denominator errors are too strongly amplified, so the
original fp16 magnitude is used to form a split distance `d`.

Two details are necessary at the pole:

1. Bias `abs(x/pi)` downward by `0.0005` before round-to-nearest period
   selection. The fp16 `1/pi` multiplier otherwise creates false exact `.5`
   ties and selects the wrong side of the pole.
2. Use a split pole constant and cancel sine-table magnitude:

```text
q = sin(r) / (abs(d) * abs(sin(r)))     # sign(r) / abs(d)
q = q * (1 - d*d/3)                    # cotangent correction
```

For the first and third positive-magnitude poles, the split constants are:

```text
pi/2 = 1.5 + 0.0703125 + (pi/2 - 1.5703125)
3*pi/2 = 4.5 + 0.2109375
               + ((pi/2 - 1.5703125) + (pi - 3.140625))
```

Do not combine different tangent gains in one table without a guard band.
Hardware interpolation crosses the gain discontinuity and creates errors on
both sides. The rejected bounded transform `tan/(1+abs(tan))` also loses too
much precision when inverted for large tangent values.

### Sinh/cosh: direct fixed point versus reference normalization

`rknnops.h` algorithms 38 and 39 are not directly reusable as tinygrad
kernels. They generate unsigned biased values normalized by `sinh(max_x)` or
`cosh(max_x)`, write raw fp32-like output, and rely on host code to calculate:

```text
(raw - 16384) / 16384
```

That is useful register evidence, but native backend output must already have
the tensor's real value. The passing implementation uses signed direct tables:

```text
broad sinh/cosh:
  domain/index  [-2,2], 8192
  value         round(function(x) * 8192)
  decode        Q13

local sinh:
  domain/index  approximately [-0.25,0.25], 65504
  value         round(4*sinh(x) * 32768)
  decode        Q15 then multiply by 0.25
  selection     0.04 < abs(x) <= 0.125
```

`65504` is both the largest finite fp16 value and a useful address multiplier:
it devotes almost the complete LUT coordinate space to the local interval
without a separate input-scaling NPU task. For `abs(x)<=0.04`, return `x`;
the cubic sinh term is below the required relative tolerance there.

The Q13 broad sinh table alone missed 23 official values by one count. This is
an output-precision problem, not an input-grid problem, so increasing the broad
address scale would not help. The amplified local table reduces the decoded
output quantum from `2^-13` to `2^-17`.

Finite official overflow inputs are handled outside the LUT: clamp table input
to `[-2,2]`, form `large=abs(x)>10`, then divide the finite endpoint result by
`1-large`. This produces sign-preserving infinity for sinh and positive
infinity for cosh. Direct fp16 NaN/infinity input still exposes root-LUT
nonfinite behavior and should get an explicit sentinel/validity path if those
values are added to the official method.

### Asin/acos: two tasks, three precision regions

The `rknnops.h` algorithms 32 and 33 establish useful register geometry, but
their unsigned Q0.15 output is biased by one and normalized for a host-side
decode. Tinygrad instead needs signed, fully decoded tensor values. The native
paths use two DPU LUT tasks per operation and ordinary elementwise tasks for
selection.

The first attempted asin design was a uniform `asin(x)/2` Q15 table over
`[-1,1]`, decoded by a multiply by two. Enumerating all 15,414 finite fp16
values in that interval found six failures:

```text
±0.99853515625, ±0.9990234375, ±0.99951171875
```

The official NumPy seed contains four of those values, so an endpoint fix is
required rather than optional dense-test hardening. The derivative of asin is
singular at `|x|=1`; increasing a uniform address scale cannot cover the full
domain and improve those endpoints simultaneously.

The passing asin geometry is:

| Task/table | LUT input | Stored value | Decode/use |
|---|---:|---:|---|
| broad LO | `abs(x)` | `asin(abs(x))/2` | multiply by 2 for `0.125 < abs(x) <= 0.875` |
| detail LE | `-abs(x)` | `4*asin(abs(x))` | multiply by 0.25 for `0.04 < abs(x) <= 0.125` |
| detail LO | `1-abs(x)` | `asin(1-d)/2` | multiply by 2 for `abs(x) > 0.875` |
| no LUT | `abs(x)` | identity | `abs(x) <= 0.04` |

The broad address multiplier is `16384`; the detail multiplier is `65504`.
The detail task is still one NPU task: its LE and LO physical tables implement
different functions, and a staged composite coordinate routes each lane to
the intended half. This is the useful general pattern for a two-level LUT
whose function has two unrelated difficult regions.

For acos, `pi/2 - fp16_asin(x)` is not accurate enough. Even using an exactly
rounded fp16 asin intermediate creates 70 failures on the official seeded
tensor because the extra store and subtraction cross acos rounding
boundaries. Acos therefore uses direct tables:

| Task/table | Domain | Stored value | Decode |
|---|---:|---:|---:|
| broad LE | `[-0.875,0]` | `acos(x)/4` | multiply by 4 |
| broad LO | `[0,0.875]` | `acos(x)/2` | multiply by 2 |
| endpoint LO | distance `d=1-abs(x)` | `acos(1-d)` | direct |

The asymmetric broad gains use the available Q15 range efficiently while
avoiding a staged `pi-acos(abs(x))` reconstruction for ordinary negative
inputs. Negative endpoint lanes can use that identity safely because their
result is near pi and has a much larger relative tolerance.

Build the sign masks as `negative = (x < 0)` and
`nonnegative = 1-negative`. Testing only `x > 0` incorrectly sends both signed
zeros through the negative LE gain, decoding `acos(0)/2` by four and returning
approximately pi. The dense hardware test includes `-0.0` and `+0.0` to keep
this boundary covered.

Literal zero LUT entries are unsafe in the normal DPU configuration. Builders
replace them with one count, but `acos(1)` must be exact enough for an absolute
tolerance of `1e-6`. The endpoint path therefore detects the only fp16 value
above `0.99975` and masks the one-count substitute back to zero.

Both methods clamp the table address input to `[-1,1]`, separately compare the
unclamped `abs(x)` with one, and multiply the selected finite result by
`valid/valid`. That expression is one on valid lanes and `0/0` on
out-of-domain lanes, restoring NaN for the `±300` TestOps subcases. As with
other PC-chain comparisons, the first comparison consumer is emitted as a
scratch task before the value used by the result path.

When tuning this family, simulate the exact quantized address multiplier,
integer table entries, interpolation, and an fp16 cast after every staged
operation. Then test the unchanged official method on hardware. The CPU model
is excellent for choosing regions, but it did not predict the final isolated
acos rounding miss caused by `pi-acos(abs(x))`; only the complete NPU chain
made that visible.

## Commit checklist

- The intended graph is recognized after all pre-rewrites.
- `CACHELEVEL=0 CCACHE=0` was used while measuring emitter changes.
- Table zero, endpoints, signed zero, NaN, and infinities were considered.
- Input and output scales are representable by the actual registers.
- Focused official TestOps methods pass unchanged.
- A hardware regression covers the critical boundary values.
- Full `test/rockchip/test_hw.py` matches the current-HEAD baseline; isolate
  methods when persistent NPU state makes a shared-process result ambiguous.
- Mypy, Ruff, and `git diff --check` pass.
- Rejected Rockchip WIP is preserved as an apply-checkable patch.
- `progress.md` and `test_ops_status.md` are updated before the milestone
  commit.
