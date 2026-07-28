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

### LOG2 range reduction

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
DISKCACHE=0 DEBUG=1 DEV=ROCKCHIP DEFAULT_FLOAT=HALF \
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
DISKCACHE=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
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

## Case study: rejected signed-Q14 HardSwish

`rknnops.h` algorithm 51 uses the shared biased unsigned Q0.15 path, normalizes
the table by the maximum absolute output, and selects a different output
precision. It is not equivalent to loading signed values into the ordinary
fp16 LUT emitter.

An exact recognizer plus signed Q14 table over `[-2,2]` was measured as a
single `dpu_lut` task. It failed 98/2925 official values, compared with 34/2925
for the existing staged graph. Most additional failures were one Q14 count
near zero, where strict relative tolerance and the nonzero-center workaround
make the signed table unsuitable.

The experiment is preserved in
`rockchip-native-hardswish-wip-e44eb5ffd.patch`. Future work should port and
measure the complete biased-Q0.15 pipeline, including output precision,
debiasing, and restoration of the reference `max_abs` scale. Reusing only the
reference function samples is insufficient.

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

## Commit checklist

- The intended graph is recognized after all pre-rewrites.
- `DISKCACHE=0` was used while measuring emitter changes.
- Table zero, endpoints, signed zero, NaN, and infinities were considered.
- Input and output scales are representable by the actual registers.
- Focused official TestOps methods pass unchanged.
- A hardware regression covers the critical boundary values.
- Full `test/rockchip/test_hw.py` passes in one process.
- Mypy, Ruff, and `git diff --check` pass.
- Rejected Rockchip WIP is preserved as an apply-checkable patch.
- `progress.md` and `test_ops_status.md` are updated before the milestone
  commit.
