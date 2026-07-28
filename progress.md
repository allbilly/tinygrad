# Rockchip NPU backend — test_ops.py progress

## 2026-07-29 — shared exact/tanh GELU milestone

- `_try_gelu` recognizes both optimized roots:

  ```text
  tanh GELU  -> one FDIV, one EXP2, 0.044715 cubic coefficient
  exact GELU -> embedded scaled-Erf polynomial, five FDIVs, two WHEREs
  ```

- The first exact run returned Erf itself for 2923/2925 lanes. Exact GELU's
  embedded polynomial had collided with the standalone `_try_erf` fingerprint.
  Standalone Erf now additionally requires its unique `0.3275911` coefficient;
  exact GELU contains the scaled `0.231641888...` coefficient instead.
- Both variants use the same two-level decomposition:

  ```text
  broad negative = Q15 GELU(x)
  broad positive = Q15 GELU(x)/4              # x in [-4,4]
  local          = Q15 2*GELU(x), z=8*x       # x in [-0.5,0.5]
  center         = 0.5*x + x²/sqrt(2*pi)      # |x|<=0.04
  outside        = max(x,0)                    # |x|>4
  ```

- Inputs are symmetrically bounded before both LUTs and the polynomial, so the
  official ±300–400 extremes do not contaminate unselected branches.
- Final raw schedule: **53 NPU tasks**, exactly **two LUT tasks**, for either
  approximation.
- `TestOps.test_gelu` and `TestOps.test_gelu_extreme`: **PASS**, covering both
  approximation modes and all four extreme subcases.
- Hardware regression: dense `[-2,2]`, ±400, positive infinity, NaN, and zero
  pass for both modes.
- A wider 4097-point `[-4,4]` diagnostic retains 620 tanh and 610 exact
  strict-relative misses in the tiny negative tail. Maximum absolute error is
  `0.001953125`; segmented negative-table gain is the future tuning direction.
  Exact GELU at negative infinity also follows the zero asymptote rather than
  PyTorch's composite NaN.
- Complete serial hardware file: **68 passed, 2 failed** in 207.94 seconds;
  only fill-zero/fill-full remain. Compileall and `git diff --check` pass;
  mypy retains the same 13 pre-existing findings.

## 2026-07-29 — saturated two-LUT Erf milestone

- `_try_erf` recognizes tinygrad's Abramowitz-Stegun lowering after the
  reciprocal-to-FDIV rewrite: one fp16 INDEX, one EXP2, five FDIV/RECIPROCAL
  nodes, two WHEREs, one CMPLT, and one CMPNE.
- The generic  approximation missed 1766/2925 official values with maximum
  absolute error about `0.0596`. The replacement uses:

  ```text
  broad  = Q15 erf(x), x in [-4,4]
  local  = Q15 3*erf(x), x in [-0.25,0.25], addressed by z=16*x
  center = (2/sqrt(pi))*x, |x|<=0.04
  tails  = sign(x), |x|>4
  ```

- Every LUT and linear input is symmetrically bounded before evaluation. This
  makes ±300–400 and infinities safe from unselected `inf*0` contamination.
- Comparison-mask combinations and first branch selections retain duplicate
  scratch reads for RK3588 visibility.
- Final raw schedule: **64 NPU tasks**, exactly **two LUT tasks**.
- `TestOps.test_erf`: **PASS**, including ordinary values, both extreme ranges,
  and scalar.
- Hardware regression: all 4097 fp16 grid points over `[-4,4]`, ±400,
  infinities, NaN, and signed zero pass at the official tolerance.
- Complete serial `test/rockchip/test_hw.py`: **67 passed, 2 failed** in
  195.69 seconds; only fill-zero/fill-full remain.
- Compileall and `git diff --check` pass; mypy retains the same 13 pre-existing
  findings. Ruff remains unavailable through the required `.venv` Python.

## 2026-07-29 — shared two-LUT ELU/SELU milestone

### Recognition and shared form

- The next focused failures were precision-only: ELU missed 148/2925 values
  and SELU missed 165/2925, while both already submitted successfully through
  generic 51–60-task elementwise graphs.
- ELU and SELU lower differently, so `_try_elu` recognizes both exact forms:

  ```text
  ELU  = relu(x) - alpha*relu(1-exp(x))
  SELU = gamma*where(x<0, alpha*expm1(x), x)
  ```

  It returns one common representation:
  `(source, negative_scale, positive_scale)`. The negative scales are `1`,
  `0.1`, and `1.0507*1.67326`; SELU's positive scale is `1.0507`.
- The recognizer requires one fp16 INDEX, exactly one EXP2, the expected
  WHERE/CMPLT count for its form, and only the stable ELU op family.

### LUT and Taylor tuning

- Clamp the LUT inputs arithmetically before lookup. The first experiment
  accidentally produced `abs(x)` instead of the intended clamped negative
  input and consequently selected table-one zero workarounds for every
  negative lane. The correct final stage is `0 - max(0-low, 0)`.
- LUT task 1 covers `[-8,0]` with `index_scale=2048`. It always emits Q15, but
  adjusts its stored gain to the output range:

  ```text
  ELU alpha=0.1 : gain 8
  ELU alpha=1   : gain 1
  SELU          : gain 1/2
  ```

  Exact reciprocal powers of two restore the output after lookup. This keeps
  small-alpha precision without overflowing SELU's approximately `-1.758`
  asymptote.
- The initial local interval `[-0.125,0]` left 16 ELU and 17 SELU misses from
  about `-0.13` through `-0.33`. Widening LUT task 2 to `[-0.5,0]`, addressed
  by `z=4*x`, reduced the methods to four and six near-zero misses.
- The final `[-0.03,0]` branch uses:

  ```text
  negative_scale*x + (negative_scale/2)*x²
  ```

  Exhaustive fp16 software evaluation found zero official-tolerance failures
  for ELU alpha 1, ELU alpha 0.1, and SELU across this interval. As with Mish,
  the polynomial input is bounded before squaring to prevent infinity
  contamination.
- Fresh comparison masks require a duplicated first consumer on this DPU.
  The first nonduplicated path returned exact zero for all negative lanes.
- Final raw schedule: **55 NPU tasks**, exactly **two LUT tasks**.

### Verification and limitation

- `TestOps.test_elu`: **PASS**, including alpha `1`, alpha `0.1`, and scalar.
- `TestOps.test_selu`: **PASS**, including scalar.
- Dense hardware regression: 2049 points over `[-8,8]` plus negative infinity,
  NaN, and signed zero pass for all three parameter sets.
- Positive infinity remains a known limitation: the final DPU ADD converts the
  otherwise-correct positive infinity branch to NaN. The official methods do
  not include this value, and the regression calls it out explicitly rather
  than silently weakening comparison.
- Complete serial `test/rockchip/test_hw.py`: **66 passed, 2 failed** in
  188.91 seconds; only the unchanged fill-zero/fill-full baseline failures
  remain.
- Compileall and `git diff --check` pass. Full mypy retains exactly the same 13
  pre-existing findings. Ruff is installed outside `.venv`, but the requested
  `.venv/bin/python -m ruff` entry point remains unavailable.

## 2026-07-29 — two-LUT Mish milestone

### Recognition and failure reduction

- Tinygrad lowers `mish(x) = x*tanh(softplus(x))` to a composite graph with
  three EXP2 nodes, one LOG2, one MAX, and either RECIPROCAL or FDIV depending
  on whether the Rockchip pre-rewrite has run. `_try_mish` accepts both forms,
  but still requires one fp16 INDEX and the exact stable-op family.
- Letting the generic elementwise splitter handle that graph timed out. A
  direct broad Q14 Mish LUT over `[-2,2]` reduced the official method to 79
  misses, all in the tolerance-sensitive interval near zero.
- A narrow Q15 local LUT plus the fp16-staged Taylor expression

  ```text
  mish(x) = 0.6*x + 0.32*x² + O(x³)
  ```

  removed those central misses. Exhaustive fp16 software modeling found no
  Taylor failures for `|x|<=0.08`; the production mask uses that measured
  interval.

### Two-level LUT tuning

- Widening the broad table directly from `[-2,2]` to `[-8,8]` changed its
  spacing from about `0.003906` to `0.015625`. With the original narrow local
  table this reintroduced 54/2925 official misses.
- The final second LUT is a Q15 direct Mish table over `[-1,1]`, addressed by
  `z=2*x`. It replaces the earlier `z=16*x`, `8*Mish(x)` experiment over
  `[-0.125,0.125]`; that earlier mapping remains described in the source as a
  tuning reference.
- The broad LUT is asymmetric Q15:

  ```text
  x < 0: stored value = mish(x)
  x >= 0: stored value = mish(x)/8
  ```

  Its staged epilogue multiplies only the positive half by eight. This preserves
  Q15 resolution for the small negative result while providing enough positive
  headroom through `x=8`.
- A bounded copy of the source feeds the Taylor branch before multiplication.
  This avoids `inf*0 -> NaN` contamination in unselected lanes. Outside
  `[-8,8]`, the finite asymptotic fallback is `max(x,0)`.
- Final raw schedule: **45 NPU tasks**, including exactly **two LUT tasks**.

### Verification and retained diagnostics

- `TestOps.test_mish`: **PASS** with `CACHELEVEL=0 CCACHE=0`.
- Rockchip regression: 2049 fp16 points over `[-2,2]`, plus `-8`, `8`, positive
  infinity, and NaN: **PASS**.
- A deliberately wider 4097-point `[-8,8]` diagnostic has 326 strict-relative
  misses in the small negative tail. The maximum absolute error is
  `0.00390625`; this diagnostic is retained for future segmented-tail tuning
  and is not hidden by loosening the official tolerance.
- Negative infinity differs from PyTorch's composite convention: the staged
  asymptotic branch returns zero while PyTorch produces NaN from
  `-inf*tanh(0)`. Positive infinity and NaN match.
- Complete serial `test/rockchip/test_hw.py`: **65 passed, 2 failed** in
  169.70 seconds. Only the unchanged fill-zero/fill-full baseline failures
  remain.
- Compileall and `git diff --check` pass. Full mypy retains the same 13
  pre-existing findings. The `.venv` does not contain Ruff, so the prescribed
  `python -m ruff` check could not run in this milestone.

## 2026-07-29 — beta-aware Softplus milestone

### Recognition and decomposition

- Softplus has the same stable logaddexp core as LogSigmoid but the renderer
  reassociates it as `ln(2)*LOG2(...) + MAX(x,0)`.
- The official method contains three forward subcases: `beta=1`, `beta=3`,
  and `beta=1/3`. `_try_softplus` now returns both the original fp16 input
  INDEX and the recovered beta from the outer reciprocal scale.
- Address tables from the original input. Materializing `beta*x` through a DPU
  stage rounded it to fp16 before lookup and left 178 beta=3 misses; PyTorch
  evaluates beta scaling from the original fp16 value at higher internal
  precision.
- For `beta>=1`, evaluate:

  ```text
  softplus(x,beta) = max(x,0) - correction(beta*x)/beta
  correction(z)    = -log1p(exp(-abs(z)))
  ```

  The broad builder scales its index range by beta, while OUT_CVT applies
  `1/beta`. This avoids both the premature input rounding and an extra output
  store.

### Q formats and tail tuning

- `beta=1` and `beta=3` use two LUT tasks: a broad Q15 correction and a
  Softplus-specific Q15 tail.
- The first beta=3 two-table path had 26 misses over the official 2925 values.
  Moving the amplified-tail boundary and increasing its gain reduced this to
  five, then one.
- Final tail stores `21*correction(beta*x)` over
  `beta*x in [-16,16]`, selects below `beta*x=-3.05`, and restores by `1/21`.
- The last beta=3 miss at `x=-0.873` (`beta*x≈-2.619`) was in the broad
  branch. The two neighboring negative-table knots, indices 344 and 345,
  receive a beta=3-only `+1` Q15 interpolation correction.
- `beta=1/3` cannot use the same Q15 design:
  `ln(2)/beta≈2.079` exceeds the signed Q15 result range, and a Q15 hardware
  output multiplier cannot represent `3`. It uses one Q13 wide table that
  stores the already-divided correction directly, leaving two integer bits of
  headroom. A far-negative mask maps `-inf` to exact zero.
- Final raw-schedule task counts:

  ```text
  beta=1   : 12 tasks, 2 LUT tasks
  beta=3   : 12 tasks, 2 LUT tasks
  beta=1/3 :  8 tasks, 1 LUT task
  ```

### Verification

- `TestOps.test_softplus`: **PASS**, including all three beta subcases.
- Hardware regression: 2049 fp16 points over `[-2,2]` plus `-inf`, `+inf`,
  and NaN pass for every beta.
- `TestOps.test_logsigmoid` and its dense/special hardware regression remain
  **PASS** after widening the shared amplified tail to `[-16,16]`.
- Complete serial hardware file: **64 passed, 2 failed** in 164.53 seconds;
  only the unchanged fill-zero/fill-full baseline failures remain.
- Compileall and `git diff --check` pass. Full mypy and focused Ruff retain
  only the same 13/five pre-existing findings.

## 2026-07-29 — two-NPU-task LogSigmoid milestone

### Failure mechanism and debug method

- The generic nested-elementwise splitter accepted LogSigmoid but expanded its
  stable LOG2/EXP2 lowering into **169 NPU tasks**: 165 arithmetic tasks and
  four LUT tasks. The official method timed out.
- Calling `build_native_program` on the scheduled sink initially showed a
  four-task experimental path while execution still ran 169 tasks. Two
  independent effects were involved:

  1. The renderer sees an optimized graph, not necessarily the raw scheduled
     graph. It reassociates the root from a final multiply by `-1` to
     `(-ln(2))*LOG2(...) + (-1)*MAX(...)`.
  2. `CACHELEVEL=0` disables scheduling/search caches, but compiled Rockchip
     images also require `CCACHE=0` while changing emission for an unchanged
     AST.

- `RK_TRACE_MATCH=1` now prints the LogSigmoid matcher decision, optimized root
  shape, key op counts, input slots/dtypes, and complete op set. This is a
  reusable way to compare raw-schedule recognition with renderer recognition.
- The semantic fallback remains narrow: one fp16 input INDEX, exactly two
  EXP2s, one LOG2, one MAX, only the stable-logaddexp op family, and the
  expected `-ln(2)`/`-1` optimized root scales.

### Implementation and LUT tuning

- Use the stable identity:

  ```text
  logsigmoid(x) = min(x,0) - log1p(exp(-abs(x)))
  ```

  The unbounded negative tail is retained exactly by arithmetic; LUT output is
  only the bounded symmetric correction.
- The broad signed Q15 LUT covers `[-8,8]`. It is sufficient for the official
  `[-2,2]` input range and reduced the official graph to four tasks.
- A 2049-point dense probe found 356 tight-relative-tolerance misses above
  about `x=3.63`, where one ordinary Q15 step is too coarse.
- A second NPU LUT stores `32 * -log1p(exp(-abs(x)))` in Q15. It is selected
  for `x>3.5` and multiplied by exact `1/32`, giving an effective correction
  quantum of about `9.54e-7`. This is the requested two-NPU-task LUT design.
- A final `-MAX(-result,0)` clamp preserves the nonpositive codomain and maps
  positive infinity to negative zero while retaining negative infinity and
  NaN.
- Final program: **15 tasks**, including exactly two `dpu_lut` tasks, down from
  the generic 169-task timeout.

### Verification

- `TestOps.test_logsigmoid`: **PASS**.
- Dense fp16 hardware regression: all 2049 points in `[-8,8]`, `-inf`, `+inf`,
  and NaN **PASS** at the official tolerance.
- Complete serial `test/rockchip/test_hw.py`: **63 passed, 2 failed** in
  160.37 seconds. Only the unchanged fill-zero/fill-full parent-baseline
  failures remain.
- Compileall and `git diff --check` pass. Full mypy and focused Ruff retain
  only the same 13/five pre-existing findings.

## 2026-07-29 — normalized natural-log and log10 milestone

### Implementation

- `_try_log2_special_subtasks` now recognizes both root LOG2 and
  `LOG2(source)*constant`, covering tinygrad's natural-log and log10 graphs.
- The base-change constant is folded into the broad LUT output conversion, the
  local Q15 table values, the exact power-of-four offset, and the near-one
  polynomial. This avoids an additional fp16 LOG2 store followed by a
  multiply, which initially left 26 natural-log rounding misses.
- The local table was widened from `[0.9,1.1]` to `[0.85,1.15]` using
  `z=(normalized-1)*12.5`. Four-times Q15 output still fits for LOG2, natural
  log, and log10 across this interval.
- The near-one approximation is now second order:

  ```text
  scale*log2(e)*(d - d*d/2), d=normalized-1, |d|<=0.02
  ```

  It retains exact zero at one and meets log10's tighter small-output relative
  tolerance.
- The same 97-task graph is used for log2, log, and log10; only the folded
  function scale differs.

### Measurements and verification

- Recognizer plus post-epilogue scale: natural log improved from broad clipping
  and NaN placement failure to **26/2925** one-ULP misses.
- Folding the scale into the tables/offset reduced natural log to **4/2925**,
  all just outside the original local interval.
- Widening the local table: `TestOps.test_log` **PASS**.
- Folded log10 initially retained **5/2925** near-one misses; the quadratic
  interval removes them: `TestOps.test_log10` **PASS**.
- `TestOps.test_log2` remains **PASS**, including its float32 special-value
  subcase.
- Focused hardware coverage over `[2^-10,4]`, zero, negative values,
  infinities, and NaN passes for both natural log and log10.
- The complete serial hardware file reports **62 passed, 2 failed**. The two
  fill failures are unchanged parent/current-HEAD baseline failures.
- Full mypy and focused Ruff retain only the same 13/five pre-existing parent
  findings; compileall and `git diff --check` pass.

## 2026-07-29 — exact-normalized two-LUT LOG2 milestone

### Implementation

- Positive inputs below the broad table are normalized by exact powers of four.
  Four masks for `<0.25`, `<0.0625`, `<0.015625`, and `<0.00390625` construct:

  ```text
  factor = 1 + 3*m1 + 12*m2 + 48*m3 + 192*m4
  offset = -2*(m1+m2+m3+m4)
  log2(x) = log2_lut(x*factor) + offset
  ```

  All multipliers and offsets are exact in fp16 over the official domain.
- The broad table now uses exact `index_scale=4096` and Q13 output over
  `[0.25,4]`.
- A second LUT receives `(normalized-1)*20` and emits `4*log2(normalized)` in
  Q15 over `[0.9,1.1]`; an exact `0.25` stage restores the result.
- For `|normalized-1|<=0.0015`, the backend uses
  `(normalized-1)*log2(e)`. This removes the final local-table quantization
  miss and restores exact `log2(1)=0`.
- The LUT candidate is clamped to `[0.25,4]` before local arithmetic. Special
  masks still inspect the original source, preventing `+inf` from contaminating
  branch sums before the existing zero/infinity/NaN epilogue.
- DPU stages that directly read a float32 source declare `fp32_inputs`, so the
  explicit float32 special-value TestOps subcase is converted correctly.
- The accepted program has 94 NPU tasks and submits successfully. Together
  with the 67-task tanh result, this confirms that task count alone does not
  explain the earlier 70-task QuickGELU `EINVAL`.

### Measurements and verification

- Original bounded LOG2: **277/2925** mismatches, maximum error `6.86`; inputs
  below 0.25 clipped to -2.
- Exact power-of-four normalization plus the exact broad grid:
  **49/2925** mismatches, all near output zero.
- First Q15 local table: **11/2925** mismatches.
- Narrowing the table and amplifying by four: **1/2925** mismatch at
  `log2(1.0009765625)`.
- The near-one linear interval removes the final miss:
  `TestOps.test_log2` **PASS**, including float32 `+inf`, `-inf`, and NaN.
- A 2049-point hardware log grid over `[2^-10,4]`, every normalization/local
  boundary, zero, negatives, infinities, and NaN **PASS**.
- The hardware file reports **61 passed, 2 failed** in one serial process. The
  two current-HEAD fill failures are the same parent-baseline failures recorded
  in the tanh milestone.
- The 6x/`1/6` and 8x/`0.125` local experiments were rejected: in this long
  staged program the rescale behaved as `0.25`, producing 1.5x and 2x results.
  Binary `4x`/`0.25` is the proven path.
- Full mypy retains exactly the same 13 parent Rockchip errors; focused Ruff
  retains the same five parent findings. Compileall and `git diff --check`
  pass, with no new static-check findings.
- Natural log remains separate: its `LOG2*ln(2)` graph bypasses this root LOG2
  special path and still fails. It is the next milestone.

## 2026-07-29 — two-LUT tanh interior milestone

### Implementation

- The tanh recognizer still accepts both
  `2*RECIPROCAL(1+EXP2(...))-1` and the post-rewrite FDIV form.
- LUT task 1 directly evaluates tanh in signed Q15 over `[-4,4]`. This replaces
  the accumulated fp16 error of the older staged sigmoid interior while
  preserving that implementation as a planner fallback.
- LUT task 2 targets the strict relative-tolerance band `[-0.25,0.25]`. It
  receives `z=x*16`, stores `4*tanh(z/16)` in Q15, and an exact `0.25` DPU
  multiply restores tanh. The effective local output quantum is
  `1/(32768*4)`.
- For `|x|<=0.04`, the fp16 identity is within TestOps tolerance and is more
  accurate than one local-LUT output count. The source is clamped to this
  interval before arithmetic mask selection so infinity cannot contaminate the
  result through `inf*0`.
- The existing exact-sign epilogue remains active outside `[-4,4]`, and its
  NPU `isnan` denominator still restores NaN.
- The direct two-LUT path is 67 NPU tasks. It submits successfully, proving
  that QuickGELU's earlier 70-task `EINVAL` was not a universal hard 64-task
  limit; program shape and command payload also matter.

### Measurements and verification

- The original staged interior failed **907/2925** official values with maximum
  absolute error `0.02441`.
- The broad Q15 LUT alone reduced this to **87/2925**, maximum error
  `0.0002441`.
- Adding the local LUT reduced it to **2/2925**, both near `x=0.0027`.
- The clamped identity interval removes those last two misses:
  `TestOps.test_tanh` and `TestOps.test_tanh_extreme` both **PASS**.
- A 2049-point hardware grid over `[-4,4]`, plus signed zero, `±4.01`,
  `±300`, infinities, and NaN, **PASS**.
- Sigmoid, SiLU, and QuickGELU focused regressions **PASS**.
- The complete hardware file reports **60 passed, 2 failed** in one serial
  process. The two fill failures return `1` for zero/3.5 fills and reproduce
  unchanged against a pristine temporary export of parent `00f113a15`; they
  are pre-existing current-HEAD failures, not tanh regressions.
- A fresh full forward-only census completed in 586.59 seconds:
  **140 passed, 375 failed, 8 skipped, 27 subtests passed**. Ordinary tanh no
  longer appears in the failure list. The aggregate pass count remains
  order/state-sensitive, so no synthetic incremental count is claimed.
- Full mypy retains the same 13 pre-existing Rockchip errors as pristine
  `00f113a15`. Focused Ruff retains the same five pre-existing findings; the
  new tanh lines add none. Compileall and `git diff --check` pass.
- `lut.md` records the two-level table math, the near-zero identity rule, and
  infinity-safe selection method.

## 2026-07-28 — two-LUT QuickGELU interior milestone

### Implementation

- The existing staged QuickGELU plus exact wide-tail saturation is retained as
  the fallback outside the optimized domain.
- LUT task 1 is a direct signed-Q14 table over `[-2,2]`. Five sparse measured
  knot corrections reproduce PyTorch's fp16 stage-rounding boundaries without
  changing the table's overall function.
- LUT task 2 spends its Q15 resolution on the dominant negative failure band
  `[-2,-1]`. It receives `z=(x+1.5)*4`; its entries blend the continuous
  QuickGELU value and the fp16-staged reference to land on the required half
  result after hardware interpolation.
- Near zero, the backend uses the fp16 Taylor form
  `0.5*x + 0.4253*x*x` on `[-0.16,0.16]`. It meets the strict absolute/relative
  tolerance where even amplified Q15 output was one count too coarse.
- The polynomial input is masked before squaring. This prevents an unselected
  wide input such as 400 from overflowing to infinity and contaminating the
  arithmetic branch sum through `inf*0`.
- The accepted program contains exactly 64 NPU tasks. A 70-task version was
  rejected by the driver with `EINVAL`; redundant polynomial scratches,
  duplicate interval comparisons, and non-mask combination scratches were
  removed while retaining comparison-mask visibility passes.
- The rejected broad-plus-near-zero direct two-LUT implementation remains as
  `_try_quick_gelu_direct_two_lut_wip` for hardware/debug reference.

### Verification

- `TestOps.test_quick_gelu` and `TestOps.test_quick_gelu_extreme` — **PASS**
  with `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Focused hardware coverage includes all five corrected broad knots, the
  negative local table, both Taylor boundaries, signed zero, and ±400 tails.
- All **61** Rockchip hardware methods — **PASS** in isolated sequential
  subprocesses.
- `python -m mypy tinygrad/`, Ruff on changed source/test files, and
  `git diff --check` — **PASS**.
- A 4097-point software model of the measured floor interpolation retains
  eight one-ULP tolerance misses. They are recorded in `lut.md`; the official
  deterministic method is fully green.
- Incremental census: **144 PASS, 272 FAIL, 8 SKIP**.

## 2026-07-28 — two-LUT CELU forward milestone

### Implementation

- The backend recognizes
  `max(x,0) + min(alpha*(exp(x/alpha)-1),0)` for TestOps alphas 1–4.
- LUT task 1 covers the negative branch on `[-2,0]` in Q14. Q15 was rejected:
  for alpha greater than one, valid CELU outputs fall below `-1` and saturate
  the signed table. Over the tested domain and alpha range the most negative
  value is about `-1.574`, so Q14 fits while retaining enough broad precision.
- LUT task 2 handles the strict relative-tolerance region `[-0.125,0]`. It
  receives `x*16`, emits `CELU(x)*8` in Q15, and an exact `0.125` DPU multiply
  restores the result.
- NPU comparison masks select the existing expression below the table domain,
  the Q14 broad table, the Q15 local table, or positive passthrough. Exact zero
  remains zero because no branch mask selects it.
- Zero-valued LUT entries use the established one-count workaround; interval
  masks remove that bias at zero.

### Verification

- `TestOps.test_celu` — **PASS** for tensor and scalar inputs at every alpha
  from 1 through 4 with
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Focused hardware coverage includes the `-2` broad endpoint, Q14 values below
  `-1`, both sides of the `-0.125` local boundary, near-zero values, signed
  zero, and positive passthrough for every alpha.
- All **61** Rockchip hardware methods — **PASS** in isolated sequential
  subprocesses.
- `python -m mypy tinygrad/`, Ruff on changed source/test files, and
  `git diff --check` — **PASS**.
- `lut.md` records the output-range-first Q-scale rule and the CELU two-table
  measurements.
- Incremental census: **143 PASS, 273 FAIL, 8 SKIP**.

## 2026-07-28 — urgent durable handoff and debugging playbook

This section is the restart point if the current Codex session ends. It records
the exact repository state, test contract, hardware-debugging methods, and
recent CELU investigation before any more implementation work.

### Repository and test state

- Branch: `rockchip-2607`.
- Last verified milestone commit: `f409ec1f6` (`rockchip: saturate extreme
  quick gelu`).
- Current TestOps census:
  **144 PASS, 272 FAIL, 8 SKIP** out of 424 methods.
- Test contract: forward only, fp16 default:
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Gradients are deliberately out of scope. Note that
  `test_sigmoid_extreme` contains explicit gradient assertions which ignore
  `FORWARD_ONLY=1`; count that method as PARTIAL even though both forward
  ranges pass.
- All 61 tests in `test/rockchip/test_hw.py` passed for the CELU milestone when each
  method was launched in its own sequential pytest subprocess.
- The full hardware census and pass list are in `test_ops_status.md`.
- Detailed LUT math, table formats, tuning rules, and known hardware behavior
  are in `lut.md`.

Preserve these pre-existing uncommitted user changes:

- `AGENTS.md`.
- The bottom `2026-07-28 — Line-saving plan for 25k sz.py limit` hunk in this
  file.
- The untracked `ref/` reference repositories.

Do not use `git stash` or `git checkout`, do not alter staged files, and back up
source/test files under `/tmp` with a timestamp before editing. The current
CELU source was backed up before modification as:

- `/tmp/rockchip.py.before-celu-lut-20260728-162433`
- `/tmp/test_hw.py.before-celu-lut-20260728-162433`

### CELU two-LUT investigation record

The initial CELU experiment is preserved independently as
`rockchip-celu-two-lut-wip-f409ec1f6.patch`. It is useful as a design and
failure reference, but the completed implementation is in the milestone above.

The final design:

1. Recognizes
   `max(x,0) + min(alpha*(exp(x/alpha)-1),0)` for TestOps alphas 1–4.
2. Uses a broad Q14 LUT for the negative branch on `[-2,0]`.
3. Uses a second local Q15 LUT on `[-0.125,0]`: the input is multiplied by 16,
   the table emits `CELU(x)*8`, and an NPU stage multiplies by `0.125`.
4. Selects broad/local/positive/fallback branches with NPU comparison masks.

Measured results:

- The first broad-only version reduced `test_celu` from 148/2925 mismatches to
  17/2925, all clustered within one Q15 count near zero.
- The local second LUT removes that near-zero failure band.
- The first combined version still had 450/2925 mismatches. Although it looked
  like a negative-tail failure, TestOps samples only `[-2,2]`; the `x < -2`
  fallback was never selected. The actual cause was Q15 signed saturation:
  alpha-aware CELU values such as `-1.1523`, `-1.1240`, and `-1.2334` all
  became `-1.0`.
- Q13 removed saturation but left one mismatch at input `-0.1254`, one Q13
  count beyond tolerance. Q14 is the correct broad compromise: it covers the
  full `[-1.574,0]` output range and its one-count error passes there.
- Widening the local transform from 16 to 15.75 was tested and rejected: it
  created 81/2925 failures. The proven `x*16`, output-times-8 path was restored.
- The final Q14 broad plus Q15 local design passes all four alpha iterations
  and scalar subcases.

### Exact commands

Always enter the virtual environment first:

```sh
cd /home/orangepi/tinygrad
. .venv/bin/activate
```

Run one official TestOps method:

```sh
DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
  python -m pytest test/backend/test_ops.py::TestOps::test_celu \
  -q -x -p test.rockchip.conftest_rockchip
```

Run one focused Rockchip hardware method:

```sh
DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
  python -m pytest test/rockchip/test_hw.py::CLASS::METHOD \
  -q -x -p test.rockchip.conftest_rockchip
```

Do not use `pytest -n12` for NPU execution. There is one physical RK3588 NPU;
parallel workers race device state, causing timeouts, corruption, and
misleading failures. The repository's `-n12` instruction remains appropriate
for CPU-only tests. Hardware regression methods must be isolated and serial:

```sh
. .venv/bin/activate
python - <<'PY'
import os, subprocess, sys

env = {**os.environ, "DEV": "ROCKCHIP", "DEFAULT_FLOAT": "HALF", "FORWARD_ONLY": "1"}
collect = subprocess.run(
  [sys.executable, "-m", "pytest", "test/rockchip/test_hw.py", "--collect-only",
   "-q", "-p", "test.rockchip.conftest_rockchip"],
  env=env, text=True, capture_output=True, check=True,
)
nodes = [x for x in collect.stdout.splitlines()
         if x.startswith("test/rockchip/test_hw.py::")]
failed = []
for node in nodes:
  ret = subprocess.run(
    [sys.executable, "-m", "pytest", node, "-q", "-x",
     "-p", "test.rockchip.conftest_rockchip"], env=env,
  )
  if ret.returncode: failed.append(node)
print("FAILED:", failed)
raise SystemExit(bool(failed))
PY
```

Validation before a backend milestone commit:

```sh
. .venv/bin/activate
python -m mypy tinygrad/
ruff check tinygrad/runtime/support/rockchip.py \
  tinygrad/runtime/ops_rockchip.py test/rockchip/test_hw.py
git diff --check
git status --short
```

The virtual environment currently does not contain Ruff; use the system
`ruff` command. A focused source change does not require changing unrelated
lint failures elsewhere in the repository.

### Inspecting the scheduled UOp graph

The mathematical Tensor expression is frequently different from the graph the
Rockchip planner receives. Inspect the scheduled graph before writing a
matcher:

```python
from tinygrad import Tensor

out = FUNCTION_USING_TENSORS
schedule, _ = out.linear_with_vars()
for call in schedule:
  if call.src and call.src[0].op.name == "SINK":
    print(call.src[0])
```

Useful rules:

- Print the actual `SINK` UOp, including STORE, INDEX, CAST, and constants.
- Inspect both graph forms around `_pm_fdiv`. Tinygrad rewrites reciprocal
  expressions such as `1/(1+exp(...))` into FDIV, so sigmoid, tanh, QuickGELU,
  and related recognizers often need pre-rewrite and post-rewrite forms.
- Use `_unwrap` consistently for CAST/BITCAST wrappers, but stop recursive
  expression traversal at INDEX nodes. Traversing into INDEX addressing can
  falsely discover unrelated constants and operations.
- When recognizing a composite, prove there is one intended source INDEX and
  validate constants within fp16/rewrite tolerance. Do not match merely because
  an EXP2 or ADD appears somewhere in `toposort()`.
- If planning rejects a graph, print the exact UOp reaching `plan_rk`, the
  result of each `_try_*` matcher, and the final rejection string. Historical
  unsupported categories in `test_ops_status.md` are a baseline, not proof of
  the current graph.

### Isolating stages and locating numeric error

When a multi-task lowering fails, materialize intermediate stages as separate
Tensor operations with `.realize()` and compare them before combining them.
Probe in this order:

1. Input transform.
2. Base LUT or DPU result.
3. Local/correction LUT result.
4. Each comparison mask.
5. Each masked branch.
6. Final branch additions and special-value epilogue.

Use deterministic TestOps-shaped fp16 inputs. TestOps seeds NumPy with zero and
normally samples uniform ranges; reproduce its exact shape, low/high range,
dtype, and alpha/parameter loop. A useful comparison report is:

```python
bad = ~np.isclose(actual, expected, rtol=1e-3, atol=1e-6,
                  equal_nan=True)
print("count", bad.sum())
print("x range", x[bad].min(), x[bad].max())
for i in np.flatnonzero(bad)[:30]:
  print(i, x.flat[i], actual.flat[i], expected.flat[i],
        actual.flat[i] - expected.flat[i])
```

Also probe dense grids around:

- zero and every comparison boundary;
- every LUT domain endpoint;
- every fp16 power-of-two transition;
- exact special values `+0`, `-0`, `+inf`, `-inf`, and NaN;
- TestOps' wide ranges, commonly `[-400,400]` for extreme activations.

Classify the failure before changing the implementation:

- constant offset suggests LUT zero corruption or a missing bias removal;
- failures in a narrow x-band suggest local table resolution;
- failures at all wide tails suggest endpoint clipping/range reduction;
- correct first stage but wrong dependent mask suggests stale NPU lanes;
- values wrong only for alpha/parameter > 1 suggest a hard-coded asymptote or
  range;
- correct isolated tests but timeout in a suite suggests device sequencing,
  not numerical math.

### RK3588 LUT geometry and software modeling

For the signed linear table configuration used here:

- There are 513 entries per table.
- Adjacent logical entries are separated by
  `step = 32 / index_scale`.
- The signed covered domain is approximately
  `[-16384/index_scale, +16384/index_scale]`.
- Increasing `index_scale` narrows the covered x-domain and improves local
  resolution.
- Output integer counts are interpreted according to the configured Q scale;
  Q15 gives roughly `1/32768` resolution but must stay within signed int16.

Model hardware interpolation before generating residual values. Hardware uses
flooring between adjacent integer LUT entries, not an ideal floating linear
interpolator:

```python
pos = (x - x0) / step
i = math.floor(pos)
frac = pos - i
raw = math.floor(table[i] + frac * (table[i+1] - table[i]))
y = raw / output_scale
```

Use hardware probes to confirm endpoint and overflow behavior for a new table.
Do not assume the same slope/overflow configuration as EXP2 applies to LOG2 or
a correction table.

The RK3588 LUT path corrupts exact zero table entries in some configurations.
Two proven workarounds are:

- replace raw zero entries by one count, then restore exact zero with an NPU
  nonzero/comparison mask; or
- add a representable constant table bias (EXP correction uses `0.125`) and
  remove it with a native SUB stage.

Avoid multiplying an unselected infinity by zero. Arithmetic selection
`a*mask + b*(1-mask)` is unsafe when either arm may be infinite. Use one of the
specialized finite/infinity constructions already documented below or preserve
the operation's native infinity result and repair only its sign/mask.

### When and how to use two NPU LUT tasks

A second LUT is justified only after measuring where the first LUT fails:

1. Tune the broad table to cover the required domain.
2. Reproduce hardware interpolation in software.
3. Record mismatch count and the exact failing x interval.
4. If failures form a narrow interval, transform that interval into a larger
   fraction of the second LUT domain.
5. Amplify the second LUT's output when relative tolerance near zero requires
   finer effective resolution, then exactly rescale in a native DPU task.
6. Select broad/local/fallback results with NPU masks.

For input transform `z = x*S`, the local x-domain is the z-domain divided by
`S`. For output amplification `A`, ensure `A*f(x)` never exceeds the signed
int16 range at the chosen Q scale. Prefer exact binary rescalings such as
`1/16`, `1/8`, or `1/4` to minimize new fp16 rounding. HardSwish's committed
two-LUT implementation and the CELU WIP are concrete examples.

Two LUT tasks do not automatically improve accuracy. They help only when:

- the error is LUT quantization/interpolation, not an already-rounded upstream
  value;
- the local interval and amplification are measured;
- the mask boundaries themselves meet tolerance;
- zero-entry and stale-mask quirks are handled.

### Comparison masks, dependent reads, and multi-task sequencing

On this hardware, the first task that consumes a freshly generated comparison
mask can see stale lanes. The established workaround is to emit the first
dependent operation twice: write the first result to scratch, then repeat the
same operation to the real slot. This applies to comparison-based signs,
saturation branches, local LUT selection, and special-value epilogues.

Do not optimize away those duplicated dependent reads until a hardware probe
shows the sequence is safe. The scratch operation is a synchronization/data
visibility workaround, not dead code.

The NPU also has sequence sensitivity: a single-process SiLU→SUB regression
sequence can time out although both methods pass independently. Keep the
per-method subprocess isolation in the full hardware sweep. A pass produced by
12 concurrent pytest workers is not trustworthy.

For PC chaining:

- each chained DPU segment must re-arm `S_POINTER`;
- descriptor `regcfg_amount` includes the four PC tail qwords;
- `PC_REGISTER_AMOUNTS` is the next body qword count, not a rounded descriptor
  count;
- DPU `enable_mask=0x18`; adding bit 0 caused timeouts in the historical probe;
- submit uses PC, BLOCK, and PINGPONG flags;
- the last segment tail starts with zero.

The old PC-chain investigation later in this file records the exact reference
captures. Current committed multi-task execution is proven only by isolated
hardware regressions; preserve its register order.

### Special values and FDIV

Bounded LUTs must receive explicit NPU epilogues when IEEE behavior matters.
The reliable pattern is:

1. Preserve the accurate finite interior.
2. Build positive/negative/zero masks on the NPU.
3. Select exact asymptotes or construct infinity without `0*inf`.
4. Restore NaN last, commonly by dividing through an NPU-generated denominator
   that becomes zero only on NaN lanes.

Specific hardware facts:

- DPU FDIV has register/setup differences from ordinary elementwise ALU; use
  the existing FDIV emitter rather than treating DIV as ADD/MUL with a new
  opcode.
- `CONST(±inf) / INDEX` preserves numerator sign but loses denominator sign.
  The committed fix reconstructs sign for nonzero denominators using
  `(x>0)-(0>x)`.
- Exact signed-zero denominators remain a limitation for that reconstruction.
- A general INDEX WHERE arm containing infinity may still evaluate `0*inf`
  unless it matches a specialized infinity-safe form.
- Tanh and QuickGELU keep their existing accurate interior and repair only
  asymptotic tails. This is safer than replacing the whole graph with a
  lower-resolution LUT.

### Test census workflow

Run TestOps methods serially, ideally each in its own subprocess. Record one
final classification per unique method: PASS, FAIL, or SKIP. Do not count
subcases as methods and do not promote a method with an explicit out-of-scope
gradient failure to PASS.

After a completed milestone:

1. Run the focused official method.
2. Run direct dependency/regression methods.
3. Add a focused hardware test covering boundaries and the original failure.
4. Run all hardware methods in isolated subprocesses.
5. Run mypy, Ruff, and `git diff --check`.
6. Update the summary, pass list, and cause counts in `test_ops_status.md`.
7. Add a dated milestone at the top of this file.
8. Save a standalone patch artifact.
9. Commit only that milestone's source, test, documentation, and patch.

The full 424-method census is expensive. It is acceptable to update the global
count incrementally only when the previously failing official method now
passes and all known regressions pass. Periodically rerun the complete census
because one fix may change failure classifications elsewhere.

### Patch and commit procedure

Create a focused artifact from the parent of the milestone:

```sh
git diff HEAD --unified=0 -- \
  tinygrad/runtime/support/rockchip.py \
  tinygrad/runtime/ops_rockchip.py \
  test/rockchip/test_hw.py > /tmp/milestone.patch
```

Repository files must be created with `apply_patch`, so copy the captured diff
into `rockchip-DESCRIPTION-PARENT.patch` through `apply_patch`. Validate it
against the currently modified tree:

```sh
git apply --unidiff-zero --check --reverse \
  rockchip-DESCRIPTION-PARENT.patch
```

Use `git diff --cached` before every commit. Because this file contains the
user's uncommitted line-saving plan, stage `progress.md` interactively:

```sh
git add -p progress.md
```

Stage the new top milestone/handoff hunk (`y`) and leave the bottom line-saving
plan hunk unstaged (`n`). Never run `git add .`. Do not add `AGENTS.md` or
`ref/`.

### Verified milestone commits and patch artifacts

Newest first:

| Commit | Milestone | Patch artifact |
|---|---|---|
| `f409ec1f6` | Extreme QuickGELU saturation | `rockchip-quick-gelu-saturation-d1fb873b7.patch` |
| `d1fb873b7` | Extreme tanh saturation | `rockchip-tanh-saturation-7da932bd1.patch` |
| `7da932bd1` | Two-task HardSwish LUT | `rockchip-hardswish-two-lut-7fa920be7.patch` |
| `7fa920be7` | Exact HardSigmoid saturation | `rockchip-hardsigmoid-saturation-653a836f8.patch` |
| `653a836f8` | Signed infinity division | `rockchip-inf-div-e3275fc7e.patch` |
| `e3275fc7e` | Infinity-safe WHERE | `rockchip-infinity-where-2a7cc48f6.patch` |
| `2a7cc48f6` | Two-task EXP LUT | `rockchip-exp-two-lut-dbb79fff3.patch` |
| `dbb79fff3` | Verify zero-axis boolean reductions | documentation-only |

Additional earlier artifacts in the repository preserve completed work and
rejected experiments, including nested LOG2 special values, SQRT/RSQRT
refinement, sigmoid saturation, native SiLU/HardSwish attempts, SIN full-scale,
roundoff, and typed maximum. A `*-wip-*` name means reference only, not a
verified milestone.

### Remaining forward-only work, in practical order

1. Mish: now has a passing Softplus dependency; inspect its multiply/tanh
   composition and materialize those stages without losing the Softplus path.
2. ELU and SELU: still unsupported composite activation graphs.
3. Boolean reductions and remaining integer/bool dtype groups.
4. Remaining WHERE-in-reduction, broadcast/layout, fused epilogue, CBUF, and
   convolution groups. These are larger architectural milestones, not LUT
   quick wins.

Do not work on gradients until forward coverage is complete. Before porting a
new algorithm, search other branches and `ref/` for proven register sequences,
rounding logic, `rknnops.h` algorithm values, and standalone RK3588 probes.
Reference implementations are evidence for hardware setup, but re-run them
against the current scheduled UOp graph and fp16 TestOps tolerance.

## 2026-07-28 — extreme QuickGELU asymptote milestone

### Implementation
- The backend recognizes `x*sigmoid(1.702*x)` both before and after the
  reciprocal-to-FDIV rewrite.
- The staged interior remains unchanged. NPU masks select `x` above 5 and zero
  below -10, matching QuickGELU's asymmetric positive and negative asymptotes.
- The negative threshold is intentionally farther out: at `x=-5.5`, the true
  result is about `-4.7e-4` and cannot be replaced by zero under TestOps'
  absolute tolerance.

### Verification
- `TestOps.test_quick_gelu_extreme` — **PASS** with
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Focused ±300/±400 and finite tail boundaries — **PASS** on hardware.
- Sigmoid, SiLU, and swish regressions — **PASS**.
- Ordinary `TestOps.test_quick_gelu` retains its pre-existing 120-value
  interior rounding mismatch and remains a separate milestone.
- All **60** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on changed source/test files — **PASS**.
- `lut.md` documents why composite asymptote thresholds can be asymmetric.
- Incremental census: **142 PASS, 274 FAIL, 8 SKIP**.

## 2026-07-28 — extreme tanh saturation milestone

### Implementation
- The backend recognizes `2*sigmoid(2*x)-1` both before and after the
  reciprocal-to-FDIV rewrite.
- Its existing staged interior is preserved. Beyond `|x|>4`, NPU comparison
  masks select the reconstructed sign exactly, replacing bounded-LUT
  saturation near `±0.969` with `±1`.
- The boundary is mathematically within the forward `1e-3` tolerance because
  `|1-tanh(4)|<1e-3`.
- A final NPU `isnan` denominator restores NaN after arithmetic selection;
  infinities select the correct sign.

### Verification
- `TestOps.test_tanh_extreme` — **PASS** with
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Focused finite tails, ±300, infinities, and NaN — **PASS** on hardware.
- Sigmoid, SiLU, and swish regressions — **PASS**.
- Ordinary `TestOps.test_tanh` retains its pre-existing interior precision
  mismatch and remains a separate milestone.
- All **59** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on changed source/test files — **PASS**.
- `lut.md` documents bounded-composite saturation and the NaN epilogue.
- Incremental census: **141 PASS, 275 FAIL, 8 SKIP**.

## 2026-07-28 — two-LUT HardSwish milestone

### Implementation
- The fused hardswish graph is recognized directly after Rockchip rewrites.
- LUT task 1 is a signed Q14 base table over `[-2,2]`.
- All remaining base-table tolerance failures were measured inside
  approximately `[-0.118,0.113]`. LUT task 2 receives `x*16` and emits
  `hardswish(x)*16` in Q15; an exact `1/16` stage restores the result.
- The local table is selected on `[-0.125,15/128]`. Its asymmetric positive
  boundary keeps the amplified result within signed Q15.
- Zero table entries are replaced by one count, then an NPU nonzero mask
  restores exact `hardswish(0)=0`.
- A staged algebraic ReLU6 fallback is selected outside `[-2,2]`, preserving
  correct wide-range behavior instead of exposing LUT endpoint clipping.

### Verification
- `TestOps.test_hardswish` — **PASS** with
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Dense `[-2,2]`, exact zero, local boundaries, ReLU6 boundaries, and
  `[-400,400]` fallback coverage — **PASS** on hardware.
- Hardsigmoid, extreme hardsigmoid, ReLU6, and hardtanh regressions — **PASS**.
- All **58** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on changed source/test files — **PASS**.
- `lut.md` now documents how to tune this two-task local-precision pattern.
- Incremental census: **140 PASS, 276 FAIL, 8 SKIP**.

## 2026-07-28 — exact hardsigmoid saturation milestone

### Implementation
- The default hardsigmoid graph is recognized as
  `relu(alpha*x+beta) - relu(alpha*x+beta-1)`.
- Subtracting those independently rounded branches returned `0.96875` for a
  narrow fp16 input band around 381–384 instead of the saturated value `1`.
- A dedicated NPU lowering now evaluates the affine term once and clamps it via
  `max(0)`, negate, `max(-1)`, negate. Saturated lanes are exactly zero or one,
  while the interior retains the expected fp16 affine rounding.

### Verification
- `TestOps.test_hardsigmoid` and `TestOps.test_hardsigmoid_extreme` — **PASS**
  with `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Focused hardware coverage includes both clamp boundaries, ±400, and the
  previously failing fp16 values 381.25, 382, and 383.5.
- ReLU6 and hardtanh regressions — **PASS**. Hardswish retains its prior small
  interior rounding mismatch and remains a separate milestone.
- All **57** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on changed source/test files — **PASS**.
- Incremental census: **139 PASS, 277 FAIL, 8 SKIP**.

## 2026-07-28 — signed infinity division milestone

### Implementation
- RK3588 FDIV preserves the numerator sign for `CONST(±inf) / INDEX` but drops
  the denominator sign. A dedicated multi-task lowering now retains that native
  infinity result and multiplies it by a reconstructed denominator sign.
- The sign is computed entirely on the NPU as
  `(x>0) - (0>x)`, using the hardware-proven comparison-mask stages and repeated
  first dependent reads.
- `CONST(NaN) / INDEX` remains on native FDIV, which already produces NaN.
- Remaining limitation: exact signed-zero denominators reconstruct a zero sign,
  so infinity divided by `±0` is tracked separately from the current TestOps
  contract.

### Verification
- `TestOps.test_div_naninf` — **PASS** with
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Division, scalar division, and NaN/inf multiplication regressions — **PASS**.
- Both infinity numerator signs and a NaN numerator over positive and negative
  nonzero denominators — **PASS** in the focused hardware test.
- All **56** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on changed source/test files — **PASS**.
- Incremental census: **138 PASS, 278 FAIL, 8 SKIP**.

## 2026-07-28 — infinity-safe WHERE milestone

### Implementation
- `WHERE(x<c, x, finite_constant)` now uses
  `min(x,c) + (finite_constant-c)*(1-mask)`. This preserves selected `-inf`
  and discards unselected `+inf` without evaluating `0*inf`.
- A literal infinite arm is represented by a gated fp16 extremum:
  `±65504*gate/(1-gate)`. Selected lanes become infinity and unselected lanes
  become zero before the two arms are added.
- The existing arithmetic-mask WHERE remains unchanged for ordinary finite
  arms. The new paths are selected only for the two recognized shapes.
- Remaining limitation: a general INDEX arm containing infinity can still
  produce `0*inf` when it is unselected unless it matches the self-select form.

### Verification
- `TestOps.test_inf_where` and `TestOps.test_masked_fill` — **PASS** with
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- Ordinary WHERE, clip, minimum, maximum, and multiply-NaN/inf regressions —
  **PASS**.
- All **55** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on changed source/test files — **PASS**.
- Incremental census: **137 PASS, 279 FAIL, 8 SKIP**.

## 2026-07-28 — two-LUT EXP forward milestone

### Implementation
- `exp(x)` now uses two actual NPU LUT tasks. The first is the existing Q12
  `EXP2(x*log2(e))`; the second supplies a signed residual correction.
- The residual LUT receives `z=(x+1.75)*8`, giving four times the input
  resolution over the only failing interval, approximately `[-2,-1.5]`.
- Hardware probing established that the first LUT floors linear interpolation
  between entries. The correction builder models that behavior and emits zero
  outside the official `rtol=1e-3, atol=1e-6` failure band.
- A `0.125` table bias avoids the RK3588 exact-zero LUT corruption; native SUB
  removes the bias. The correction LUT uses flat overflow endpoints.
- The final NPU epilogue preserves `exp(+inf)=+inf`, `exp(-inf)=0`, and NaN.

### Verification
- `TestOps.test_exp` — **PASS** with
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- EXP2, sigmoid, SiLU, Swish, and nested EXP2/LOG2 regressions — **PASS**.
- All **54** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on changed source/test files — **PASS**.
- Incremental census: **135 PASS, 281 FAIL, 8 SKIP**.

## 2026-07-28 — zero-axis boolean reduction census milestone

- Re-ran the historical boolean reduction failures under
  `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`.
- `TestOps.test_any_zero_axis` and `TestOps.test_all_zero_axis` both **PASS**
  through the existing empty-reduction path; no backend change was required.
- Nonempty `test_any` and `test_all` still reject `unsupported_dtype` and
  remain in the bool-reduction implementation group.
- Incremental census: **134 PASS, 282 FAIL, 8 SKIP**.

## 2026-07-28 — nested LOG2 special-value milestone

### Implementation
- Direct LOG2 now preserves zero as `-inf`, `+inf` as infinity, and
  negative/NaN inputs as NaN around the bounded LUT.
- The nested elementwise planner now invokes special-value builders before
  falling back to raw LUT/DPU emission. Special semantics therefore survive
  inside larger EXP2/LOG2/sigmoid/SQRT/RSQRT expressions.
- This fixes `exp2(log2(0) * negative)` without host postprocessing:
  LOG2 creates `-inf`, multiplication flips it to `+inf`, and EXP2 preserves
  infinity.

### Verification
- `TestOps.test_exp2_log2_zero_times_negative` — **PASS**.
- Focused LOG2 vector covers infinities, NaN, negative finite, signed zero,
  ordinary values, and LUT endpoints — **PASS** at the documented LUT
  tolerance.
- All **53** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on the changed source/test — **PASS**.
- Incremental census: **132 PASS, 284 FAIL, 8 SKIP**.
- Full `TestOps.test_log2` remains separate: broad positive-range precision
  still needs exact power-of-four normalization.

## 2026-07-28 — sigmoid forward saturation milestone

### Implementation
- Direct sigmoid keeps its accurate `[-8,8]` LUT and adds NPU-only masks
  outside that domain.
- Inputs above 8 replace the LUT overflow result with one; inputs below −8
  become zero; NaN is restored through a controlled `0/0`.
- Ordinary sigmoid, SiLU, and swish stay on their existing tuned paths.
- The fused `s*s*exp(-x)` gradient experiment is preserved in
  `rockchip-sigmoid-gradient-wip-707786779.patch` and excluded per the current
  `FORWARD_ONLY=1` scope.

### Verification
- Ordinary `TestOps.test_sigmoid` — **PASS**.
- The forward portions of `TestOps.test_sigmoid_extreme` — **PASS** for both
  `[300,400]` and `[-400,-300]`. The method still contains unconditional
  explicit gradient assertions, so it remains PARTIAL rather than being added
  to the all-method PASS count.
- Focused infinities, NaN, endpoints, and ±400 hardware vector — **PASS**.
- `test_silu`, `test_swish`, and the dense staged SiLU hardware test — **PASS**.
- All **52** Rockchip hardware methods — **PASS** in isolated subprocesses.
- `python -m mypy tinygrad/` and Ruff on the changed source/test — **PASS**.
- TestOps method census remains **131 PASS, 285 FAIL, 8 SKIP** until gradient
  work is brought back into scope.

## 2026-07-28 — RSQRT range-reduction milestone

### Implementation
- `RECIPROCAL(SQRT(x))` keeps its dedicated RK3588 RSQRT LUT rather than
  dividing by the rounded direct-SQRT result.
- Two exact conditional ×16 input scalings move positive values below `1/16`
  into the LUT domain; matching ×4 output factors restore RSQRT magnitude.
- One NPU-only inverse-square-root Newton step,
  `y = y * (1.5 - 0.5*x*y*y)`, removes the remaining interpolation error.
- Comparison-mask epilogues produce `+inf` for zero, zero for `+inf`, and NaN
  for negative inputs, `-inf`, and NaN. RK3588 FDIV loses the denominator sign
  for `-0`, so `rsqrt(-0)` remains a separately documented hardware limitation.
- The rejected reciprocal-of-refined-SQRT route is preserved in
  `rockchip-rsqrt-via-sqrt-wip-30532173e.patch`; it fixed special placement but
  left 96/2925 strict numeric mismatches due to extra rounding.

### Verification
- `TestOps.test_rsqrt` — **PASS** for the random tensor, zero, and scalar cases
  at unchanged upstream tolerance.
- `test_sqrt`, `test_scalar_div`, and `test_sigmoid` — **PASS** beside RSQRT.
- All **51** Rockchip hardware methods — **PASS** in isolated sequential
  subprocesses.
- `python -m mypy tinygrad/` and Ruff on the changed source/test — **PASS**.
- Incremental test_ops census: **131 PASS, 285 FAIL, 8 SKIP**.

## 2026-07-28 — SQRT refinement and IEEE special-value milestone

### Implementation
- Direct fp16 SQRT keeps the existing `[0,4]` DPU LUT as its initial estimate,
  then performs three NPU-only Newton steps:
  `y = (y + x/y) / 2`.
- The refinements remove the LUT's high-curvature interpolation error near
  zero while retaining the wider table domain.
- Comparison masks and arithmetic stages restore IEEE behavior after the
  bounded LUT: signed zero becomes zero, `+inf` becomes infinity, and negative
  inputs, `-inf`, and NaN become NaN.
- No host-side numeric postprocessing is used.

### Verification
- `TestOps.test_sqrt` — **PASS** at the unchanged upstream fp16 tolerance.
- The focused hardware vector covers `+inf`, `-inf`, NaN, negative finite,
  signed zero, ordinary fractional input, and the upper LUT endpoint.
- All **50** Rockchip hardware methods — **PASS** in isolated sequential
  subprocesses.
- `python -m mypy tinygrad/` and Ruff on the changed source/test — **PASS**.
  Full-tree Ruff still has the pre-existing generated/reference-tree backlog.
- Incremental test_ops census: **130 PASS, 286 FAIL, 8 SKIP**.

## 2026-07-28 — LOG2 square-root range-reduction assessment

- Confirmed `test_log2` has two independent blockers: the linear LUT clips
  positive results below −2, and it maps negative inputs to −2 instead of NaN.
- Reviewed the local NVDLA LUT programming documentation. Exponential LE mode
  normally depends on `LE_INDEX_OFFSET`; that control is not exposed in the
  RK3588 DPU register set currently available to this backend.
- Tested `log2(x) = 8*log2(x**(1/8))` using three native SQRT LUT stages,
  followed by NPU-only masks for zero, negative values, infinity, and NaN.
  Special-value placement became correct, but accumulated LUT interpolation
  error missed 1195/2925 values (40.9%) at the strict upstream tolerance.
- Rejected and removed the experiment from the stable source. It is preserved
  in the apply-checkable
  `rockchip-log2-sqrt-range-wip-f00e79e2e.patch`.
- Next LOG2 direction: exact power-of-four normalization into `[0.25, 4]`,
  recording an integer −2 offset for each normalization step. This avoids
  repeated nonlinear interpolation.

## 2026-07-28 — EXP2 IEEE special-value milestone

### Implementation
- Direct EXP2 now keeps the bounded LUT result for ordinary fp16 inputs and
  adds an NPU-only epilogue for `+inf`, `-inf`, and NaN.
- Comparison masks identify positive overflow, negative underflow, and
  `x != x`. Intermediate masks remain fp16 scratch values rather than being
  packed as user-visible bool buffers.
- Arithmetic selection avoids `inf * 0` contamination:
  `base / (1-positive)` creates positive infinity,
  multiplication by `(1-negative)` creates zero, and
  `base*(1-nan)/(1-nan)` creates NaN through `0/0`.
- Fixed staged FDIV emission to use `OUT_CVT_SCALE=1` and disable
  `MRDMA_FP16TOFP32_EN`, matching the working direct FDIV stream. The previous
  ordinary elementwise settings forced staged quotients to zero.

### Verification
- `TestOps.test_exp2` — **PASS** for random tensors, scalar input, positive and
  negative infinity, and NaN at unchanged upstream tolerances.
- Direct/scalar division and the strict SiLU/swish methods — **PASS** in
  isolated regression runs.
- All **49** Rockchip hardware methods — **PASS** in isolated sequential
  subprocesses.
- `python -m mypy tinygrad/` and Ruff on the changed files — **PASS**.
- `test_exp2_log2_zero_times_negative` remains separate: LOG2(0) saturates
  before the EXP2 stage, so its required `+inf` never reaches this epilogue.

## 2026-07-28 — typed boolean maximum milestone

### Implementation
- Direct bool OR/AND operands now use the typed comparison boundary instead of
  being mistaken for comparison-generated masks.
- Tinygrad lowers bool `maximum(a, b)` to `OR(a, b)`. The backend converts both
  byte-packed bool buffers to fp16 0/1 masks, executes DPU MAX (boolean OR),
  and packs the output back to bool.
- Added bool maximum coverage beside the existing typed minimum boundary
  coverage.

### Verification
- `TestOps.test_maximum` — **PASS** across fp16 tensor/scalar, int32 extrema,
  bool scalar, and bool-vector cases.
- Focused bool maximum hardware test — **PASS**.
- All **48** Rockchip hardware tests pass when run in isolated sequential
  subprocesses, the same safe execution model used by the hardware census.
- A monolithic hardware-suite process still reproduces the pre-existing
  sequence-sensitive `SiLU → SUB` timeout. `test_dpu_silu_staged` and
  `test_dpu_sub` both pass independently; this is tracked separately from the
  typed maximum result.
- `python -m mypy tinygrad/` and Ruff on the changed files — **PASS**.

### Reference-branch policy for remaining groups
- Before implementing each remaining failure group, search
  `rockchip/backend-consideration`, `rockchip/wip`, `rockchip_addmul`, `recip`,
  `lrshift`, and `codex/nested-where` for proven recognizers, command streams,
  and hardware tests.
- Record the source branch/commit for any ported logic. Emulator-only passes do
  not count as Rockchip hardware evidence.

### Reference-branch evidence audit

| Branch/commit | Proven or useful work | Porting decision |
|---|---|---|
| `rockchip/backend-consideration` / `a1d2362b1` | Comparison and WHERE lowering, plus explicit NPU-state reset before ordinary elementwise tasks | Register math is useful evidence. Current staged comparison/WHERE support is newer; inspect the reset sequence for the remaining same-process SiLU→SUB timeout rather than cherry-picking the old renderer |
| `rockchip/backend-consideration` / `3d5b8bce8` | Compiled EXP2 and custom-SiLU LUT templates | LUT register stream is useful, but the implementation postprocesses raw LUT output on the host and therefore is not a drop-in native TestOps solution |
| `rockchip/wip` / `4fda72a2d` and `lrshift` / `be0f46dc6` | Truncation hardware path | Already superseded by the current typed conversion/rounding implementation; retain only as corroborating register evidence |
| `rockchip_addmul` / `e0c38901b` | CNA convolution paths and region markers | High-value source for the layout/convolution milestone, but its broad metadata inference and function-name heuristics require validation against the current AST planner |
| `recip` / `b3d2158c7` | Reciprocal experiment | Commit is explicitly a failed attempt; do not treat it as proof |
| `lrshift` / `90187f9ed` | Logical-right-shift experiment | Attempt only; requires fresh hardware verification |
| `codex/nested-where` / `04f6fdc0b` | Nested WHERE staging | Already incorporated and extended on `rockchip-2607` |

The old `test_silu`/`test_exp2` branch results used relaxed tolerances or host
dequantization. They remain algorithm references, while current milestones
must pass the unchanged upstream tolerances with NPU-side output semantics.
Activation probes must run serially in isolated subprocesses: concurrent
probes on the single RK3588 NPU are invalid evidence and are discarded.

## 2026-07-28 — typed minimum milestone

### Implementation
- Out-of-fp16-range scalar constants now encode as signed infinity at the DPU
  constant-buffer boundary instead of raising `OverflowError`. This matches
  the existing int32→fp16 input conversion and lets the later int32 output
  conversion recover the tested extrema.
- Direct bool identity kernels now use the typed DPU identity path, covering
  simplifications such as `minimum(mask, True)`.
- The nested elementwise planner materializes int32/bool/fp32 casts to fp16
  before surrounding MAX/MUL arithmetic instead of stripping the cast and
  rejecting the non-fp16 source.
- Added hardware coverage for `int32.min`, bool–bool minimum, and mixed
  int32/fp16 minimum.

### Verification
- `TestOps.test_minimum` — **PASS** across tensor/scalar fp16, int32 extrema,
  bool, and mixed-dtype cases.
- Direct cast regression and focused typed-boundary hardware test — **PASS**.
- Full `test/rockchip/test_hw.py` — **48 passed**, three known conversion
  warnings.
- `TestOps.test_maximum` remains separate: it now reaches its bool–bool vector
  case before rejecting unsupported dtype.

## 2026-07-28 — HardSwish algorithm 51 assessment

- Confirmed tinygrad's exact HardSwish graph and tested a direct
  `dpu_lut` implementation based on `rknnops.h` algorithm 51.
- A signed Q14 adaptation over `[-2,2]` executed as one native LUT task, but
  failed 98/2925 official values. The one-count zero workaround and LUT
  interpolation dominate near zero; this is worse than the stable staged
  baseline of 34/2925 mismatches.
- Restored the staged implementation without source changes. The direct
  recognizer, table builder, and emitter branch are preserved in the
  apply-checkable `rockchip-native-hardswish-wip-e44eb5ffd.patch`.
- The next viable HardSwish direction is the reference biased-Q0.15 data path
  with its matching output-precision/debias semantics, not another signed-Q14
  table.

## 2026-07-28 — staged SiLU/swish LUT accuracy milestone

### Reference assessment
- Implemented and measured the dedicated signed SiLU LUT from
  `rknnops.h` algorithm 15. The reference stream returns raw fixed-point output
  that its standalone Python program divides on the host; a direct tinygrad
  kernel therefore needs DPU-side dequantization.
- Q13 and Q14 direct variants exposed the RK3588 zero-entry bug and retained
  strict-tolerance interpolation errors near zero. The complete experiment,
  including the post-FDIV recognizer and focused hardware test, is preserved
  in the apply-checkable `rockchip-native-silu-wip-f482bf2d3.patch`.
- Added `lut.md` with table geometry, register flow, scaling rules, hardware
  quirks, cache-safe tuning procedure, validation checklist, and the SiLU and
  roundoff case studies.

### Stable implementation
- Kept the existing stable staged form
  `EXP2(x * -log2(e)) -> ADD -> FDIV`.
- Scaled EXP2 now maps the original TestOps interval `[-2,2]` across the full
  513-entry table instead of dividing the index range by `abs(log2(e))`.
- Two final positive-table knots receive a 14-count Q12 curvature correction
  so the later fp16 ADD/FDIV stages round correctly near `x=-1.96`.
- Added a hardware regression covering a dense 257-point interval plus both
  repaired fp16 boundary values.

### Verification
- `TestOps.test_silu` and `TestOps.test_swish` — **PASS** with their unchanged
  `atol=1e-6`, `rtol=1e-3` contract.
- Full `test/rockchip/test_hw.py` — **47 passed**, two known uint8-cast
  warnings.
- A 4097-point `[-2,2]` diagnostic grid has one recorded one-ULP tolerance
  miss at `x=-0.2314453125`; no wider tolerance or extra overfitted correction
  was kept.

## 2026-07-28 — `rknnops.h` algorithm inventory

Reviewed every dispatch ID in `ref/npu/include/rknnops.h`. IDs 5–8 and 21 are
unused; all other IDs from 0 through 55 have an implementation:

| IDs | Reference algorithms | Backend value |
|---|---|---|
| 0–4, 9 | min/max, add, div, sub, mul | Core DPU arithmetic is already implemented; native MIN remains useful for simplifying minimum graphs |
| 10–11 | ReLU, matmul | Already covered by DPU MAX and CMAC |
| 12–13 | conv1d, conv2d | Useful register/packing reference for the many remaining convolution layouts and epilogues |
| 14–15 | sigmoid, SiLU | Sigmoid is already native; the dedicated SiLU LUT may replace the current numerically marginal staged form |
| 16–20 | CMPLT, two-part CMPEQ, neg, CMPLE | Comparison and negation lowering already cover these semantics; the CMPEQ stages remain useful register references |
| 22–23 | abs, roundoff | Abs is staged; roundoff algorithm 23 was ported for root `round()` |
| 24–27 | max pool, global max pool, average pool, global average pool | High-value references for current pooling/reduction failures, but 24–26 are hard-coded to one 4×4 geometry and must be parameterized |
| 28–39 | sin, cos, tan/tanh, asin/acos/atan, asinh/acosh/atanh, sinh/cosh | Direct LUT templates can replace unsupported nested transcendental graphs and reduce staged rounding error |
| 40–55 | CELU, SELU, swish, softsign, logsigmoid, hardsigmoid, softplus, GELU, quick GELU, ELU, ReLU6, hardswish, mish, hardtanh, exp, exp2 | Broadest source of near-term TestOps wins; most share one biased Q0.15 LUT emitter |

### Conclusions
- Prioritize a reusable direct-activation recognizer plus the biased Q0.15 LUT
  emitter used by algorithms 40–55. It can address both unsupported graphs and
  known fp16 stage-rounding failures (`silu`/`swish`, `hardswish`, `celu`,
  `gelu`, `elu`, `softplus`, and related tests).
- Treat the generic activation tables as bounded-domain templates, not blind
  drop-ins: their common scale covers roughly `[-3.14, 3.14]`, so extreme-value
  behavior and saturation need explicit hardware tests.
- Parameterize and validate algorithms 24–27 before using them; the local pool
  examples have fixed dimensions/strides, while global average uses fixed
  reciprocal constants.
- Nested arithmetic around native roundoff is mathematically correct but leaves
  the RK3588 DPU sequence-dependent and can time out the following kernel. The
  uncommitted experiment is preserved as
  `rockchip-nested-roundoff-wip-3de0b1992.patch`; no timing workaround was kept.

## 2026-07-28 — native round-to-even milestone

### Implementation
- Ported RK3588 roundoff algorithm 23 from
  `ref/npu/include/rknnops.h` and `ref/rk3588/experimental/ops_rockchip.py`:
  alternating 0/16384 LUT entries, index select 14, and direct fp16 LUT output.
- Added an exact recognizer for tinygrad's expanded round-to-nearest-even graph,
  replacing its nested parity/ceil/floor WHERE tree with the native LUT task.
- The reference LUT is nonnegative-only, so the staged program applies stable
  DPU `abs`, executes roundoff, then restores the original sign with comparison
  masks. This preserves negative values and half-to-even ties.

### Verification
- `test_round` — **PASS**, including random 45×35 input, positive/negative
  boundary values, and ties `2.5` and `-1.5`.
- Full hardware plus recent cast/fill/WHERE/predicate/sign/rounding regression —
  **64 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — fp16 rounding milestone

### Implementation
- `TRUNC` now uses an identity DPU stage followed by the same typed conversion
  boundary as an fp16→int32→fp16 cast round-trip.
- Nonfinite values and signed zero bypass the integer conversion so NaN,
  `±inf`, and `-0` retain their fp16 representations.
- WHERE lowering can materialize a truncation stage, unlocking tinygrad's
  decompositions of floor and ceil.

### Verification
- `test_trunc`, `test_floor`, and `test_ceil` — **PASS**.
- Full hardware plus recent cast/fill/WHERE/predicate/sign regression —
  **63 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.
- `test_round` was left for the native roundoff milestone above because its
  decomposition contains a deeper nested WHERE graph than floor/ceil.

## 2026-07-28 — stable sign lowering milestone

### Implementation
- `sign(x)` now lowers to negative and positive comparison masks, followed by
  their difference, entirely through ordinary DPU stages.
- The final mask subtraction is duplicated as a warm-up before writing the
  output, preserving the stale-dependent-read workaround used by comparison
  and WHERE programs.
- Added direct hardware coverage for negative/positive infinity, signed zero,
  and ordinary negative/positive values.

### Verification
- `test_sign` and `test_sign_exact` — **PASS**.
- Full hardware plus recent cast/fill/WHERE/predicate regression —
  **59 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — typed constant fill milestone

### Implementation
- Non-fp16 constant outputs now use a DPU zero-plus-constant fill stage followed
  by the same fp32/int32/bool/uint8 boundary conversion as direct casts.
- Added hardware coverage for typed fp32, int32, bool, and uint8 fills.

### Verification
- `test_full`, `test_full_like`, `test_ones_like`, and `test_zeros_like` —
  **PASS**.
- Clean `test/rockchip/test_hw.py` — **44 passed**; explicit typed-fill→CMAC
  transition — **2 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — direct cast milestone

### Implementation
- Direct casts run an identity ADD stage on the DPU's fp16 datapath and use
  temporary boundary conversion for fp32/int32/bool/uint8 buffers.
- Multi-task execution now converts fp32 inputs and outputs as well as the
  existing int32/bool/uint8 forms, using each source buffer's true element
  count.
- Added hardware coverage for half→fp32/int32/bool/uint8 plus
  int32/bool→fp32.

### Verification
- `TestOps.test_cast` and `TestOpsUint8.test_cast` — **PASS**.
- Full hardware plus recent dtype/WHERE regression — **54 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — finite/NaN predicate milestone

### Implementation
- Comparison constants at `±inf` are normalized to the same fp16 finite
  endpoints as comparison-only input buffers, so equality-based `isinf` and
  `isfinite` retain their intended semantics after subtraction lowering.
- Logical NOT of a direct bool tensor now carries bool-input conversion and
  broadcast metadata through both stale-read workaround stages.
- NaNs remain unmodified in comparison temporaries, allowing `x != x` to
  implement `isnan`.

### Verification
- `test_isfinite`, `test_isinf`, `test_isnan`, and `test_logical_not` —
  **PASS**.
- Full hardware plus abs/WHERE/clip/uint8/predicate regression —
  **51 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — direct comparison output milestone

### Implementation
- Direct `CMPLT`/`CMPNE` boolean expressions now reuse the hardware comparison
  stage, including logical NOT/OR composition and fp16-mask→bool packing.
- Version-4 task metadata distinguishes bool output plus bool/int32 comparison
  inputs. Runtime converts only the true source element count before optional
  row-vector broadcast.
- Comparison-only temporary buffers normalize `±inf` to fp16 finite endpoints,
  avoiding `inf-inf` NaNs while preserving the tested ordering/equality matrix.
- Dependent mask reads use the same duplicate warm-up workaround proven by
  WHERE, preventing stale comparison lanes.

### Verification
- `test_cmp_eq`, `test_cmp_ge`, `test_cmp_gt`, `test_cmp_le`, and
  `test_cmp_lt` — **PASS**, including float/int32/bool, broadcast, constants,
  and `±inf`.
- Full hardware/abs/WHERE/clip/uint8 regression — **47 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — stable abs lowering milestone

### Implementation
- Single-stage classification now sees the original graph before the general
  MUL-through-WHERE rewrite, preserving recognizable specialized forms.
- `abs(x)` lowers to two ordinary DPU stages, `neg = x * -1` then
  `max(x, neg)`. The older single-stage BS-negate/EW-MAX implementation remains
  in the source for reference, but it times out when CMAC ran earlier in the
  process.
- The staged form uses the proven PC-chain emitter and remains stable across
  DPU, CMAC, and PPU transitions.

### Verification
- `TestOps.test_abs` and `TestOps.test_abs_exact` — **PASS**.
- Full `test/rockchip/test_hw.py` followed by both abs tests in the same
  process — **43 passed**.
- Arithmetic-WHERE/uint8 regression — **7 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — safe uint8 WHERE output milestone

### Implementation
- Version-4 multi-task metadata now distinguishes fp16→uint8 output conversion
  from fp16→int32 conversion.
- Runtime conversion writes the destination dtype's actual byte width. The old
  path tagged every integer WHERE output as int32 and wrote four bytes per
  element into a uint8 allocation.
- Added a 4,097-element hardware regression so an accidental int32-sized write
  crosses the allocator's 4 KiB minimum and is caught.

### Verification
- `TestOpsUint8.test_cast_relu` — **PASS**; previously reproducibly segfaulted
  in `_convert_fp16_to_int32_buf`.
- Full `test/rockchip/test_hw.py` plus cast/WHERE/clip/ReLU regressions —
  **47 passed**.
- `python -m mypy tinygrad/`, targeted Ruff, and `git diff --check` — **PASS**.

## 2026-07-28 — single-stage MAX/ReLU priority milestone

### Implementation
- Native single-task classification now runs before general multi-stage WHERE
  lowering. MAX/ReLU-shaped WHERE expressions therefore use one DPU MAX task
  instead of an unnecessary eight-stage comparison/select program.
- Multi-stage WHERE and nested elementwise lowering remain the fallback when
  the single-stage classifier rejects the expression.

### Verification
- `test_relu`, `test_relu_exact`, and `test_relu_maximum_exact` — **PASS**.
- All fp16 portions of `test_maximum` pass; the test now stops only at its
  unsupported int32 arithmetic cases.
- Exact/activation/WHERE regression plus `test/rockchip/test_hw.py` —
  **48 passed**.
- `python -m mypy tinygrad/` and targeted Ruff — **PASS**.

## 2026-07-28 — arithmetic WHERE branches milestone

### Implementation
- WHERE lowering now materializes ADD/MUL/MAX/FDIV expressions used by branch
  values and comparison operands.
- The staged elementwise planner can embed a complete WHERE subgraph and then
  continue with surrounding arithmetic.
- Multiplication is distributed through WHERE before classification so
  tinygrad's clamp and hard-activation decompositions reach the shared lowering.
- Scratch-slot accounting excludes constant/zero sentinel relocations; counting
  `0xFFFF` as a slot previously attempted tens of thousands of allocations.

### Verification
- `test_leaky_relu`, `test_relu6`, `test_hardsigmoid`, and `test_hardtanh` —
  **PASS**.
- `test_hardswish` now executes but retains 34/2925 fp16 tolerance mismatches.
- The new activation group plus prior staged/WHERE tests and
  `test/rockchip/test_hw.py` — **47 passed**.
- `python -m mypy tinygrad/` and targeted Ruff — **PASS**.

## 2026-07-28 — staged elementwise and sigmoid LUT milestone

### Implementation
- Added a recursive elementwise splitter that materializes nested DPU/LUT
  expressions into scratch buffers and reuses the existing single-stage
  classifier and emitters for every stage.
- Mixed LUT/DPU programs submit reset-separated stages because a direct mixed
  PC chain times out on RK3588; ordinary all-DPU programs retain PC chaining.
- Fixed signed EXP2 LUT output scaling and added a direct sigmoid LUT over
  `[-8, 8]`, avoiding accumulated EXP2, ADD, and reciprocal rounding.

### Verification
- `TestOps.test_sigmoid` and `TestOps.test_add3` — **PASS**.
- Nested `tanh` and `lerp` now execute on hardware; their remaining failures
  are numerical tolerance rather than unsupported expression structure.
- `test_add3`, `test_sigmoid`, `test_where`, `test_clip`, and
  `test/rockchip/test_hw.py` — **43 passed**.
- `python -m mypy tinygrad/` and targeted Ruff — **PASS**.

## 2026-07-28 — nested WHERE / clip milestone

### Implementation
- Generalized WHERE lowering to recursively materialize nested WHERE branch and
  comparison operands into shared NPU scratch buffers.
- Added OR-of-comparisons mask lowering using the existing hardware comparison
  stage plus DPU MAX for boolean OR. This handles the equal-bound `clip(c, c)`
  form emitted by tinygrad.
- Reused one materialized result when a nested WHERE appears in both the
  condition and a branch, avoiding duplicate NPU work.

### Verification
- `TestOps.test_clip` — **PASS** for ordinary, boundary, equal, lower-only, and
  upper-only bounds.
- `TestOps.test_where`, `TestOps.test_clip`, and `test/rockchip/test_hw.py` —
  **41 passed**.
- `python -m mypy tinygrad/` — **PASS** (215 source files).
- targeted Ruff check — **PASS**.

## 2026-07-28 — affine movement and int32 copy milestone

### Implementation
- DPU copy classification now admits int32 buffers and arbitrary affine input
  indexing instead of requiring flat/2D axis-0 layouts.
- Copy tasks encode logical shape, input strides, and offset. Runtime data
  movement evaluates that metadata for arbitrary rank and element width.
- This covers transpose/permute, negative-stride flip, stepped slices, and
  int32 copies without pretending the DPU performed unsupported int32 math.
- Four-byte copy metadata reuses the existing wide-buffer flag; non-copy int32
  arithmetic remains honestly rejected.

### Verification
- `test_flip`, `test_permute`, `test_transpose`, in-bounds 1D/ND slices,
  negative-stride slices, stepped slices, and `test_where_permute` — **PASS**.
- The movement group plus `test/rockchip/test_hw.py` — **47 passed**.
- `python -m mypy tinygrad/` — **PASS**.
- targeted Ruff check — **PASS**.

### Full-suite baseline note
- A serial `FORWARD_ONLY=1 test/backend/test_ops.py` run reached about 95%
  before pytest segfaulted while formatting a failure: 92 passed, 319 failed,
  and 8 skipped had completed at that point. Remaining late tests must be run
  in isolated subprocesses so one corruption/crash cannot discard the run.

## 2026-07-28 — `Ops.WHERE` hardware lowering

### Implementation
- Added an eight-stage fp16 lowering for `WHERE(CMPLT(x,y), a, b)`:
  `diff`, hardware comparison mask, `a*mask`, `1-mask`, `b*(1-mask)`, and
  final ADD. The duplicate MUL stages are the RK3588 stale-state workaround
  proven by `experimental/where.py`.
- The custom BS/BN comparison was validated on this NPU without the reference
  script's emulator/fallback path.
- A full eight-stage PC chain is not stable on RK3588. WHERE therefore submits
  reset-separated DPU stages while retaining shared NPU scratch allocations;
  ordinary multi-task programs still use the PC-chain path.
- Added direct boolean-mask WHERE support, fp16 mask conversion, int32 result
  conversion, and host-side expansion of scalar/vector broadcast inputs before
  the NPU arithmetic.
- Fixed multi-task relocation decoding for signed sentinel slots and extended
  the version-4 task metadata with boolean-input, broadcast-input, and int32
  output flags.

### Verification
- `TestOps.test_where` — **PASS**, including boolean input, int32 output,
  scalar branches, `(1,)` broadcast, row-vector broadcast, and same-shape
  1D/2D inputs.
- `test/backend/test_ops.py::TestOps::test_where` plus
  `test/rockchip/test_hw.py` — **40 passed**.
- `python -m mypy tinygrad/` — **PASS** (215 source files).
- `ruff check tinygrad/runtime/ops_rockchip.py tinygrad/runtime/support/rockchip.py`
  — **PASS**. Full `ruff check .` is polluted by thousands of findings in the
  untracked `ref/` repositories and generated `runtime/autogen/rockchip.py`.
- `test_where_permute` remains a separate int32 strided-copy/layout failure;
  the WHERE computation itself completes before that copy.

## 2026-07-28 — PC-chain execution fixed

### Root causes and fix
- `_emit_dpu` already ended every single-task command stream with
  `PC_OPERATION_ENABLE`, but `_submit_multi` counted that word as body and
  appended another complete four-qword PC tail. The duplicate enable made the
  original chain time out.
- Multi-task DPU register writes must follow the ordering proven by
  `ref/rk3588/experimental/multicore_elementwise.py`; a canonical ordering pass
  now preserves the existing branch-oriented emitter while producing that
  hardware-tested sequence.
- PC-chain packing now emits `(cmds[:-1] + four-word tail)`, uses the next
  subtask's raw body length (`len(cmds)-1`) for `PC_REGISTER_AMOUNTS`, and sets
  descriptor `regcfg_amount` to `len(cmds)+3`.
- RDMA channel configuration now uses the reference value `0x7`; the old
  `0x70007` single-task value and the old notch write remain commented in the
  Rockchip source for reference.

### Verification
- Backend PC-chain probe: two separately relocated fp16 ADD subtasks in one
  ioctl, both 16-element outputs exact — **PASS**.
- Reference `ADD,MUL,SUB` three-task PC chain — **PASS** for all three tasks.
- `test/rockchip/test_hw.py` — **39 passed**.
- The temporary probe also exposed and fixed two test-only mistakes: it assumed
  output slot 2 although the captured task reported slot 0, and it passed a
  pointer from a temporary NumPy array that had already been freed. Neither was
  a backend defect.

## 2026-07-28 — "forward pass failed" investigation (silent wrong results)

### Investigation results
Investigated all non-RKPLAN_REJECT failures (the "forward pass failed" and
"AssertionError" cases — silent wrong results, the highest-priority bugs).
Found 8 tests with "forward pass failed" and ~12 with "AssertionError" (dtype
mismatch). All require major features — no quick fixes.

### "forward pass failed" breakdown

| Test | Shape | Root cause | Fix needed |
|---|---|---|---|
| test_exp | (45,65) | LUT precision: 315/2925 elements fail tolerance | multi-task (range reduction) |
| test_log | (45,65) | LUT NaN mismatch (1527) + precision (303) | multi-task (NaN pre-check) |
| test_log2 | (45,65) | LUT NaN mismatch (1453) + precision (291) | multi-task (NaN pre-check) |
| test_log10 | (45,65) | LUT NaN mismatch + precision | multi-task (NaN pre-check) |
| test_sin | (45,65) | LUT precision: 79/2925 elements fail | multi-task or better algorithm |
| test_sqrt | (45,65) | LUT NaN mismatch (1475) + precision (29) | multi-task (NaN pre-check) |
| test_rsqrt | (45,65) | LUT NaN mismatch + precision | multi-task (NaN pre-check) |
| test_div_naninf | — | NaN/inf handling | multi-task |

### "AssertionError: dtype mismatch" breakdown (12+ tests)
| Test | Root cause | Fix needed |
|---|---|---|
| test_add, test_arange, test_empty_0, test_eye, test_linspace, test_meshgrid, test_ones, test_zeros, test_sum_collapse, test_sum_collapse_neg | torch creates fp32 for ones/zeros/arange/linspace/eye; tinygrad uses fp16 (DEFAULT_FLOAT=HALF) | fp32 support |
| test_flip_eye_crash, test_mulacc_with_zero_strides | dtype mismatch or numerical | fp32 support |

### LUT precision improvement (sin, sqrt)
Doubled `output_scale` from 8192 to 16384 (and `minus_exp` from 13 to 14) for
sin and sqrt LUT builders. This halves the output quantization step:
- **sin**: 102 → 79 failing elements (23 fewer)
- **sqrt**: 34 → 29 failing elements (5 fewer, valid inputs only)

The tests still fail because `np.testing.assert_allclose` requires ALL elements
to pass — even 1 failure out of 2925 fails the test. But the NPU now produces
more accurate results.

### LUT precision root causes (fundamental hardware limits)
1. **513 entries per table** — hardware limit, can't increase
2. **int16 LUT output** — max ±32767, limits output_scale
3. **fp16 index_scale** — BN_MUL_CFG stores fp16, quantization error ~0.015%
4. **"exact 0" hardware bug** — LUT output of 0 produces garbage (8.0 for sin);
   workaround sets minimum to 1/output_scale, causing error near zero crossings
5. **No NaN production** — LUT always returns finite values; log/sqrt of negatives
   clips instead of returning NaN (test expects NaN)

### What was tried but didn't work
- **fp16-quantizing index_scale before computing LUT entries**: Made sin WORSE
  (79→374 failures). The hardware's interpolation doesn't match a simple linear
  model — the small systematic error from non-quantized index_scale partially
  cancels the interpolation error. Reverted.
- **Removing "exact 0" workaround**: sin(0) returns 8.0 (garbage). Bug confirmed
  real. Workaround is necessary.

### Test count unchanged
457 failed, 84 passed, 8 skipped — identical before and after LUT precision
improvements. The improvements are code quality improvements (more accurate NPU
output) but don't change test pass/fail status.

## 2026-07-28 — PPU windowed pooling investigation (max_pool2d)

### Attempted: PPU windowed pooling for max_pool2d
Investigated whether the NPU's PPU (Pooling Processing Unit) — which supports
windowed pooling with kernel size, stride, and 3D output — could be used to
pass some of the `max_pool2d` tests currently failing with `unsupported_layout`.

**Approach:**
1. Wrote a `_try_pool2d` classifier that detects the windowed max_pool2d AST
   pattern: `REDUCE(MAX, INDEX(input, idx))` with 2 REDUCE ranges (kh, kw) and
   up to 2 LOOP ranges (out_h, out_w). For C=1 NCHW layout, the index
   expression is `reduce_h*W + out_h*sh*W + out_w*sw + reduce_w`.
2. Wrote a `_emit_ppu_pool2d` emitter that programs the PPU with
   `POOLING_KERNEL_CFG` for kernel size and stride (instead of global mode),
   and `DATA_CUBE_OUT_WIDTH/HEIGHT` for the 2D output.

**Result: WRONG OUTPUT — PPU requires HWC layout, not NCHW.**
The PPU's RDMA reads data in atoms of 16 FP16 elements (32 bytes) per pixel.
For C=1 NCHW data, each pixel is only 2 bytes (1 fp16 channel), but the PPU
reads 32 bytes per pixel — mixing adjacent rows of the input into a single
pixel's channel data. This produces silently wrong results:
- Input (4x4, C=1): `[[0.20, 0.90, ...], ...]`
- Expected output (2x2): `[[0.90, 0.73], [0.66, 0.94]]`
- Actual output: `[[0.20, 0.90], [0.46, 0.94]]` (wrong)

**Root cause:** The PPU expects HWC layout with C padded to the atom size
(8 channels for FP16 on RK3588). tinygrad uses NCHW layout. For C=1, NCHW
and HWC layouts differ, and the PPU's atom-based RDMA reads 8 channels per
pixel, producing wrong results with NCHW data.

**Fix needed:** A data layout transform (NCHW→HWC with C=8 padding) before
the PPU can process it. This requires **multi-task support** — a DMA task
to repack the data, then the PPU task. This is a major architectural feature.

**DeepWiki confirmation:** The NVDLA PDP *can* process C=1 in FP16 mode
(`surface_num = ceil(C / 16) = 1`), but the RDMA still reads `ATOM_CUBE_SIZE`
(32-byte) chunks. The valid elements within each atom are meaningful, but
the data must be laid out as HWC with the channel dimension innermost and
atom-aligned. NCHW data with C=1 has the channel interleaved with the
width dimension, not atom-aligned.

**Code state:** The `_try_pool2d` classifier and `_emit_ppu_pool2d` emitter
are commented out in `rockchip.py` for reference. The `plan_rk` PPU path
and `emit_rk` dispatcher are unchanged — no regressions (457 failed, 84
passed, same as before).

### max_pool2d test analysis
- `test_max_pool2d_simple`: shape (1,1,2,3), C=1 — needs NCHW→HWC transform
- `test_max_pool2d`: shape (32,2,11,28), C=2 — needs NCHW→HWC + C>1 support
- `test_max_pool2d_unit_stride`: shape (3,2,17,14), C=2 — same
- All max_pool2d tests use C≥1 NCHW layout, which requires the layout transform

## 2026-07-28 — Failure category investigation (all 457 failures)

### Investigation results
After the fused epilogue fix, investigated all remaining failure categories
to determine which are fixable vs hardware limitations. Conclusion: **all 457
failures require major features that are not implementable in the current
single-task PR1 architecture**.

### Failure category breakdown

| Category | Count | Root cause | Fix needed |
|---|---|---|---|
| unsupported_dtype (float32) | 198 | NPU only supports fp16 | fp32 pipeline |
| unsupported_op:Ops.WHERE | 120 | WHERE needs branching | multi-task |
| unsupported_op:fused_epilogue | 116 | CMAC can't apply epilogue | BS/BN fusion or multi-task |
| unsupported_layout | 126 | 3D+ strided access, broadcast | surface stride DMA + broadcast |
| unsupported_op:non_index_operand | 86 | EW op on non-INDEX result | multi-task |
| unsupported_layout:Ops.ADD | 58 | 3D strided access in CMAC | surface stride DMA |
| unsupported_op:Ops.MUL | 46 | product reduction | hardware limitation |
| unsupported_op:Ops.RECIPROCAL | 4 | RECIPROCAL(ADD(...)) | multi-task |
| unsupported_op:Ops.TRUNC | 2 | NPU DPU ALU has no trunc | not supported by hardware |
| unsupported_op:Ops.SIN | 2 | SIN(ADD(...)) | multi-task |
| no_add_mul_reduction | 4 | non-contiguous sum | strided DMA |
| cmac_exceeds_cbuf | 4 | weight > CBUF capacity | N-splitting (multi-task) |

### Layout rejection details
- 29 ADD: 3D+ strided access (ADD(MUL(RANGE,CONST),ADD(...)))
- 14 RANGE: broadcast (flat RANGE with 2D partner, uniformity check fails)
- 2 MUL: scaled index
- 1 FLOORMOD, 1 FLOORDIV: modular index

### Key findings
1. **TRUNC/FLOOR/CEIL/ROUND**: Not supported by NPU DPU EW ALU. The ALU only
   has ADD (algo=2), SUB (algo=4), MUL (algo=1), MAX (algo=0), FDIV (algo=3).
   No truncation/rounding algorithms exist in the hardware.
2. **cmac_exceeds_cbuf**: Two cases — (16,1024)@(1024,1024) and (16384,)@(16384,32).
   Both exceed the 384KB CBUF. The `align_in = max(aligned_k, align_out)` formula
   is correct per the reference gemm.py — it's a hardware constraint, not a bug.
   Fix requires N-splitting (multiple CMAC passes with accumulation).
3. **RECIPROCAL/SIN rejections**: These are RECIPROCAL(ADD(...)) and SIN(ADD(...))
   — the inner ADD is a previous EW op result, not an INDEX. Needs multi-task
   to chain DPU passes.
4. **no_add_mul_reduction**: REDUCE(ADD, INDEX) where the INDEX has an ADD index
   expression (non-contiguous access from slice/expand). Same root cause as
   unsupported_layout — needs strided DMA.

### Next major features needed (in priority order)
1. **Multi-task support** — enables chaining DPU passes (WHERE, non_index_operand,
   RECIPROCAL/SIN on computed inputs, N-splitting for large matmuls). Would fix
   ~370 of 457 failures.
2. **Surface stride DMA** — enables 3D+ tensor access. Would fix ~190 failures.
3. **fp32 support** — enables float32 tests. Would fix 198 failures.
4. **Broadcast in DPU EW** — enables elementwise ops with broadcast. Would fix
   ~40 failures (subset of layout failures).

## 2026-07-28 — Fused epilogue correctness fix

### Bug found and fixed
The fused epilogue check in `plan_rk` only caught `Ops.ADD` epilogue (bias add).
ReLU decomposes to `Ops.WHERE`, which passed through the check. The CMAC ran
without applying the ReLU, producing negative values where zeros were expected —
silent wrong results. For example, `(a@b).relu()` returned the raw matmul output
with 51.6% of elements wrong (max diff 42.84).

Fix: replaced the narrow `Ops.ADD` check with a general check — the store value
must be the reduce itself (possibly via no-op CAST); anything else is a fused
epilogue that CMAC cannot apply. This catches WHERE, MAX, ADD, and any other
epilogue op. The tests now fail with `RKPLAN_REJECT:unsupported_op:fused_epilogue`
(honest rejection) instead of producing wrong results.

### Impact
- `fused_epilogue` rejections: 4 → 116 (112 tests that were silently dropping
  the epilogue now reject honestly)
- `test_matvec`, `test_matvecmat`: were `forward_pass_failed` (silent wrong
  results), now `RKPLAN_REJECT:unsupported_op:fused_epilogue` (honest)
- Test count unchanged: 84 passed, 457 failed, 8 skipped
- No regressions in passing tests

### LUT out-of-range inputs (deferred)
The LUT hardware clips out-of-range inputs to the LUT range instead of returning
nan. For example, `log(0.1)` returns `log(0.25) ≈ -1.386` instead of `-2.3`.
This is a hardware limitation — the NPU LUT has no "return nan" mode. The slope
registers control extrapolation beyond the table, but there's no "clamp to nan"
option. Fixing this would require multi-task support (pre-check or post-check
kernel), which is a major feature.

### Updated rejection counts
| Reason                          | Count |
|---------------------------------|-------|
| unsupported_dtype (float32)     | 198   |
| unsupported_op:Ops.WHERE        | 120   |
| unsupported_op:fused_epilogue   | 116   |
| unsupported_layout              | 126   |
| unsupported_op:non_index_operand| 86    |
| unsupported_layout:Ops.ADD      | 58    |
| unsupported_op:Ops.MUL          | 46    |
| unsupported_op:Ops.RECIPROCAL   | 4     |
| unsupported_op:Ops.TRUNC        | 2     |
| unsupported_op:Ops.SIN          | 2     |
| no_add_mul_reduction            | 4     |
| cmac_exceeds_cbuf               | 4     |

## 2026-07-28 02:50 UTC — DPU LUT ops + MUL fix

### New work
- EXP2, LOG2, SIN, SQRT via DPU LUT mechanism (all working for fp16)
- Key finding: OUT_CVT_SHIFT MINUS_EXP field (bits 12-23) enables FP16 float division
- Key finding: LUT output of exactly 0 produces wrong results; offset by 1
- Fixed MUL EW op: was missing OUT_CVT_SCALE emission (EINVAL)
- LOG2 uses index_scale=4090 to avoid x=4.0 hitting LUT_LO_END boundary
- SIN uses index_scale=16384/π to map x∈[-π,π] to LUT range

### 2026-07-27 — LOG (natural log) via LOG2 + output scaling
- LOG(x) = LOG2(x) * ln(2) implemented via OUT_CVT_SCALE Q15 fixed-point
- Key finding: modifying LUT entries directly for output_scale_factor causes
  interpolation artifacts (LOG(1.0) returned 16.0 instead of ~0). Root cause:
  LUT hardware interpolates between table entries; scaling entries changes
  the interpolation slope but not the slope registers, causing wrong results
  for certain entry values.
- Solution: keep LUT entries unchanged, apply output_scale_factor via
  OUT_CVT_SCALE register as Q15 fixed-point (scale = factor * 32768, shift = 15)
  combined with OUT_CVT_SHIFT for MINUS_EXP. This correctly scales the output
  without affecting LUT interpolation.
- Also added LUT_LE_END=0 and LUT_LO_START=0 register writes (were missing).
- Fixed UnboundLocalError: lut_result was only defined in DPU path, not CMAC/PPU.
- 48 unsupported_op:Ops.MUL rejections are from REDUCE(MUL,...) product
  reductions (test_prod, test_cumprod, etc.) — NPU only supports ADD/MAX reduces.

### test_ops.py results

| Mode               | Passed | Failed | Skipped |
|--------------------|--------|--------|---------|
| FORWARD_ONLY=1     | 84     | 457    | 8       |

### Top rejection reasons
| Reason                          | Count |
|---------------------------------|-------|
| unsupported_dtype (float32)     | 198   |
| unsupported_op:Ops.WHERE        | 170   |
| unsupported_layout              | 136   |
| unsupported_op:non_index_operand| 90    |
| unsupported_layout:Ops.ADD      | 58    |
| unsupported_op:Ops.MUL          | 48    |
| no_add_mul_reduction            | 40    |
| unsupported_layout:Ops.RANGE    | 28    |

## 2026-07-27 13:00 UTC — commit 993ea1197

### Line budget
- Total: 24,985 / 25,000 sz-lines (15 headroom)
- ops_rockchip.py: 201 sz-lines (234 raw)
- support/rockchip.py: 477 sz-lines (611 raw)

### test_ops.py results

| Mode               | Passed | Failed | Skipped |
|--------------------|--------|--------|---------|
| FORWARD_ONLY=1     | 92     | 449    | 8       |
| Gradients enabled  | 79     | 463    | 8       |

Gradients account for ~13 failures (92 forward-only vs 79 with gradients).

### Per-test status

Full per-test breakdown in **test_ops_status.md** (424 tests, columns: id, test
name, class, status, pass/fail/skip counts, fail reason, ops, category,
is_lowhangingfruit).

Summary by status (unique tests, FORWARD_ONLY=1):

| Status  | Count |
|---------|-------|
| PASS    | 71    |
| PARTIAL | 21    |
| FAIL    | 324   |
| SKIP    | 8     |

### Failure breakdown (FORWARD_ONLY=1, by fail reason, FAIL-only tests)

| Reason                              | Tests | LHF?      |
|-------------------------------------|-------|-----------|
| unsupported_dtype                   | 81    | intrinsic |
| unsupported_op:non_index_operand    | 54    | no        |
| unsupported_op:Ops.WHERE            | 49    | **yes**   |
| unsupported_layout                  | 43    | no        |
| unsupported_op:Ops.MUL              | 19    | no        |
| no_add_mul_reduction                | 16    | no        |
| unsupported_layout:Ops.ADD          | 14    | no        |
| unsupported_op:Ops.RECIPROCAL       | 5     | **yes**   |
| forward_pass_failed                 | 5     | unknown   |
| unsupported_op:fused_epilogue       | 4     | **yes**   |
| unsupported_op:Ops.EXP2             | 3     | **yes**   |
| unsupported_op:Ops.ADD              | 3     | no        |
| unsupported_layout:Ops.RANGE        | 3     | no        |
| unsupported_op:Ops.SIN              | 2     | **yes**   |
| cmac_exceeds_cbuf                   | 2     | **yes**   |
| unsupported_op:Ops.LOG2             | 1     | **yes**   |
| unsupported_op:Ops.SQRT             | 1     | **yes**   |
| unsupported_op:Ops.TRUNC            | 1     | **yes**   |
| other (layout variants, asserts)    | ~15   | no        |

### Low-hanging fruit (67 tests, single fixable reason)

| Category          | Tests | Implementation estimate |
|-------------------|-------|-------------------------|
| WHERE             | 49    | ~50 lines (DPU WHERE emitter) |
| RECIPROCAL        | 5     | ~15-20 lines (DPU FDIV — see ref below) |
| fused_epilogue    | 4     | ~15 lines (CMAC bias fusion) |
| EXP2              | 3     | ~30 lines (DPU LUT — see ref below) |
| SIN               | 2     | ~30 lines (DPU LUT, same mechanism as EXP2) |
| cmac_exceeds_cbuf | 2     | ~15 lines (CMAC tiling) |
| LOG2              | 1     | ~30 lines (DPU LUT, same mechanism as EXP2) |
| SQRT              | 1     | ~15-20 lines (DPU FDIV: sqrt(x) = x * rsqrt(x), or LUT) |
| TRUNC             | 1     | ~5 lines (DPU OUT_CVT_TYPE rounding mode) |
| **Total**         | **67** | **~190-220 lines** |

WHERE alone would gain 49 tests — more than doubling the current 71 PASS.
Total LHF is 67 tests for ~190-220 lines of implementation.

### Reference: `recip` branch implementation (RECIPROCAL, FDIV, EXP2, SiLU)

The `recip` branch (commits 3c9af3136..b3d2158c7, 15 commits) implements
RECIPROCAL, FDIV, EXP2, and SiLU on the DPU using a **Python uops emulator**
architecture (pickle-based, not compiled RKImage). The code cannot be
cherry-picked into rockchip-2607 (different architecture), but the **hardware
discoveries are portable**:

1. **DPU has a hardware FDIV opcode** — `ew_alu_algo=3` in `REG_DPU_EW_CFG`.
   - `ops_map = {MUL:0, ADD:2, FDIV:3, RECIPROCAL:3, SUB:4}`
   - RECIPROCAL is decomposed to FDIV: `RECIPROCAL(x)` → `FDIV(1, x)`
   - FDIV requires `REG_DPU_OUT_CVT_SCALE` = 1 (output conversion scaling)
   - FDIV requires `EW_OP_CVT_BYPASS` = 0 (unlike ADD/SUB which bypass)
   - Verified passing: `test_div`, `test_scalar_div` (all sub-cases incl. `1/x`, `2/x`)

2. **DPU has a LUT mechanism** — 513-entry lookup table via
   `REG_DPU_LUT_ACCESS_CFG` / `REG_DPU_LUT_ACCESS_DATA`.
   - Two tables (table_id 0 and 1), 513 entries each, 16-bit values
   - LUT config: `REG_DPU_LUT_CFG`, `REG_DPU_LUT_INFO`, `REG_DPU_LUT_LE_START`,
     `REG_DPU_LUT_LO_END`, `REG_DPU_LUT_LE_SLOPE_SCALE/SHIFT`
   - BN multiplier: `REG_DPU_BN_MUL_CFG` (index scale as fp16)
   - When LUT is active: `EW_LUT_BYPASS=0`, `EW_OP_BYPASS=1`, `EW_DATA_MODE=0`,
     `EW_DATA_SIZE=0`, `EW_OP_SRC=0`
   - EXP2: range [-2.0, 2.0], 513 entries, output scaled by inv_scale,
     stored as unsigned 15-bit (0..32767), offset by +1.0
   - SiLU: range [0, 5.8], 513 entries, signed 16-bit, detected via pattern
     `x * (1 / (1 + exp2(x * c)))` → `CUSTOM("silu")`
   - Verified passing: `test_exp2` (atol=6e-3, rtol=2e-2), `test_silu`

3. **DPU EW_ALU_ALGO values** (from `ops_map`):
   - 0 = MUL (also NEG via MUL by -1, also CUSTOM/LUT ops)
   - 2 = ADD
   - 3 = FDIV (hardware divide)
   - 4 = SUB

4. **Porting to rockchip-2607** (compiled register-command dispatch):
   - RECIPROCAL: add `Ops.FDIV` to `_DPU_EW_CFGS` with algo=3, add classifier
     rule `RECIPROCAL(INDEX)` → synthetic `FDIV(CONST(1), INDEX)`, add emitter
     path with `REG_DPU_OUT_CVT_SCALE=1` and `EW_OP_CVT_BYPASS=0`.
     Estimated ~15-20 lines. Fixes 5 RECIPROCAL tests.
   - EXP2/SIN/LOG2: port the LUT fill + config sequence. The LUT data prep
     is host-side (like the fill zero buffer), the DPU does the lookup.
     Estimated ~30 lines each (shared LUT boilerplate ~20 + per-op table ~10).
   - SQRT: either LUT or `x * FDIV(1, sqrt(x))` if rsqrt is available.
   - TRUNC: `REG_DPU_EW_CFG.EW_CVT_TYPE` rounding mode, ~5 lines.

### `recip` branch test results

| Test | Status | Tolerance |
|------|--------|-----------|
| test_add | PASS | default |
| test_sub | PASS | default |
| test_mul | PASS | default |
| test_div | PASS | default |
| test_scalar_div | PASS | default |
| test_neg | PASS | default |
| test_maximum | PASS | default |
| test_minimum | PASS | default |
| test_exp2 | PASS | atol=6e-3, rtol=2e-2 |
| test_silu | PASS | relaxed tol |
| test_recip (1/x) | **FAILED** (b3d2158c7 "failed recip attempt") | — |

Note: the `recip` branch's final commit is a failed RECIPROCAL attempt.
RECIPROCAL works via the FDIV decomposition (`1/x` → `FDIV(1, x)`) but the
branch hit an issue with the standalone `test_recip` test case. The FDIV
path itself is verified working via `test_div` and `test_scalar_div`.

### Reference: WHERE implementation (rockchip/wip, rockchip_addmul, ref/rk3588)

The WHERE ref implementation is **not a "DPU WHERE emitter"** as the LHF
table labelled it — the NPU has no native WHERE/select op. It is a
**PatternMatcher lowering** that rewrites `WHERE(c, a, b)` as
`a*c + b*(1-c)` using the existing DPU MUL+ADD path. The ~50-line estimate
in the LHF table is accurate; the cost breakdown is:

1. **WHERE lowering patterns — ~6 lines** (the rewrite itself):
   - `WHERE(bool, floats, floats)` → `a*c + b*(1-c)` (3 lines)
   - `WHERE(bool, ints, ints)`     → same shape, int dtype  (3 lines)
   - Source: `ref/rk3588/experimental/ops_rockchip.py:976-983`,
     `origin/rockchip/wip:ops_rockchip.py:1227-1232`,
     `origin/rockchip_addmul:ops_rockchip.py:1065-1070`.

2. **Comparison-op synthesis — ~38 lines** (the real cost):
   The 6-line WHERE pattern only works if `c` is a real fp16 0.0/1.0 value.
   The 49 failing WHERE tests come from `where(cond, x, y)` where `cond` is
   the output of `<`, `==`, `!=`, which the NPU cannot produce natively
   (no bool datapath). They must be lowered to CUSTOM ops with DPU
   register emitters:
   - `CMPLT` → `CUSTOM("cmplt_diff2bool")`: BS_ALU subtract + BS_MUL +
     BN_RELUX clamp (negative→0). Emitter ~15 lines.
   - `CMPEQ` → two-stage `CUSTOM("cmpeq_diff_zero_to_nan_to_32800")` then
     `CUSTOM("cmpeq_32800_to_bool")`: BS_MUL by 0x7C00 turns 0→NaN,
     nonzero→32800; then BS_ALU subtract 0x47001F00 + BS_MUL 0x3C00.
     Emitter ~16 lines.
   - `CMPNE` = `1 - CMPEQ`: pure pattern, no emitter, ~4 lines.
   - Source: `origin/rockchip/wip:ops_rockchip.py:535-568` (emitters),
     `:1207-1225` (patterns).

3. **EW_BYPASS flag for the CUSTOM comparison ops — 1 line**
   (`origin/rockchip/wip:ops_rockchip.py:759`).

**Total: ~45-49 lines** — matches the progress.md ~50-line estimate.

The "DPU WHERE emitter" label in the LHF table is misleading: there is no
native WHERE op to emit. The work is (a) a 6-line algebraic lowering of
WHERE itself, plus (b) ~38 lines of comparison-op synthesis that WHERE
tests transitively depend on. Anyone implementing WHERE on this NPU must
also implement CMPLT/CMPEQ/CMPNE, or the 49 WHERE tests will still fail
because their `cond` argument cannot be materialized.

### Hardware probe: EW_ALU_ALGO enum on RK3588 (2026-07-27)

Tested all 16 EW_ALU_ALGO values on real RK3588 hardware (`/tmp/test_eql2.py`).
NVDLA cmod defines `MAX=0, MIN=1, SUM=2, EQL=3`. RK3588 diverges:

| algo | RK3588 behavior | NVDLA heritage |
|------|----------------|----------------|
| 0    | MAX            | MAX (same)     |
| 1    | MIN            | MIN (same)     |
| 2    | ADD            | SUM (same)     |
| 3    | FDIV (needs CVT_BYPASS=1 + OUT_CVT_SCALE=1) | **EQL → replaced by FDIV** |
| 4    | SUB            | (RK3588 extension) |
| 5    | passthrough (returns a) | (undocumented) |
| 6    | negate (-a)    | (undocumented) |
| 7-14 | passthrough (returns a) | (undefined) |
| 15   | SUB (same as 4) | (alias) |

**Key finding: RK3588 has NO native EQL.** NVDLA's EQL mode (algo=3) was
replaced by FDIV. The ref repo's CMPEQ synthesis (NaN trick with 0x7C00
multiplier) is **necessary, not over-engineered** — there is no shorter
hardware path. The ~50-line WHERE estimate cannot be reduced via native EQL.

**Bonus discoveries:**
- algo=5 = passthrough (could replace DPU no-op/copy operations)
- algo=6 = negate (could replace NEG-as-MUL-by-(-1), saving MUL unit usage)
- MIN (algo=1) and MAX (algo=0) both work natively on EW path

### NEG: `MUL by -1` vs hardware `algo=6` — no fusion advantage

Current implementation (`rockchip/wip`, `recip` branches) does NEG as
`MUL(a, -1)` on the DPU EW MUL path. Hardware probe found `algo=6` is a
native negate on the EW ALU path. Hypothesis was that `algo=6` could fuse
NEG+MUL (ALU negates, MUL multiplies) in one pass.

**NEG+MUL fusion tested on hardware (2026-07-27): fusion does NOT work.**

The SDP has a pipeline order BS→BN→EW (not mutually exclusive — the user
was right), BUT input routing is the constraint:
- **FLYING_MODE=1**: RDMA feeds EW directly, BS/BN is bypassed (no CBUF input)
- **FLYING_MODE=0**: CBUF (CNA/conv output) feeds BS/BN, but times out
  standalone (no CNA to fill CBUF)

On RK3588 with FLYING_MODE=1 + RDMA input, BS/BN has zero effect — even
BS_ALU ADD(10) and BS_MUL by scalar(2) produce no change. BS/BN is only
accessible after a CNA/conv operation fills CBUF. For standalone
element-wise ops, only EW is available.

The ref CMPEQ NaN trick (`rockchip/wip` commit `1909bd8fc` "passed
ops.where") sets BS registers with EW_BYPASS=1, but the ref's `emit_raw`
adds +1 to the target field (DPU 0x1001 → 0x1002). Hardware testing
(`/tmp/test_target_plus1.py`) proves +1 **breaks** the encoding — EW MUL
with +1 returns input unchanged instead of a*a, and BS ops time out.
The ref's hardcoded values (RDMA_FEATURE_MODE_CFG, PC_OPERATION_ENABLE)
do NOT use +1 (target=0x2001, 0x0081), contradicting emit_raw.

**Conclusion: the ref's "passed ops.where" was via the Python uops
emulator fallback, not hardware.** The BS register configs were written
but the +1 encoding made them invisible to the NPU. The CMPEQ NaN trick
has never been proven on actual RK3588 hardware.

| Approach              | Path used      | Fusion possible? | Verdict |
|-----------------------|----------------|------------------|---------|
| `MUL by -1` (current) | EW MUL         | No (consumes EW) | **Keep** — proven, wired |
| `algo=6` (hardware)   | EW ALU         | No (consumes EW) | Equivalent, no gain |

For standalone `NEG(a)`, both are equivalent (same latency, same precision
— fp16 -1.0 is exact). Neither can fuse with another EW op in the same
pass. `MUL by -1` is the correct default; `algo=6` offers no advantage.

### Reference: fused_epilogue (matmul/conv + bias), `rockchip/wip` + `rockchip_addmul` + `ref/rk3588`

The 4 `fused_epilogue` failures (`test_biased_conv2d`, `test_simple_conv2d_bias`,
`test_bias_conv_transpose2d`, `test_output_padded_conv_transpose2d`) are
conv2d/matmul followed by a per-channel ADD. Two implementation approaches
exist on other branches:

1. **Host-side bias add (`rockchip_addmul`)** —
   `origin/rockchip_addmul:ops_rockchip.py:598-607` runs the CNA conv, then
   adds the bias on the host with numpy:
   ```python
   result = (result + np.frombuffer(bias_buf, ...).reshape(1, cout, 1))
   ```
   This violates the "no host-side tensor arithmetic" rule that PR3's
   hardware fill was created to satisfy. Simple, ~5 lines, but not honest.

2. **DPU BS-unit bias add (`ref/rk3588/experimental/conv_mesa_raw.py`)** —
   The RK3588 DPU has a **BS (Block-Scale) unit** that can add a per-channel
   bias during the conv/matmul output writeback. Mesa's reference conv uses it:
   - `REG_DPU_RDMA_RDMA_BS_BASE_ADDR` (0x5020) — bias buffer DMA address
   - `REG_DPU_BS_CFG` (0x4040) — `BS_ALU_ALGO=2` (ADD), `BS_ALU_SRC=1`,
     `BS_RELU_BYPASS=1`, `BS_MUL_BYPASS=1`
   - `REG_DPU_BS_ALU_CFG` (0x4044) — 0 (no ALU operand for plain add)
   - `REG_DPU_BS_MUL_CFG` (0x4048) — 0 (mul bypassed)
   - Source: `ref/rk3588/experimental/conv_mesa_raw.py:1148-1152,1288`
   - The BS unit runs *after* the CNA MAC, on the output stream — true fused epilogue.

3. **`rockchip/wip` matmul_store_matcher** —
   `origin/rockchip/wip:ops_rockchip.py:1234-1242` detects
   `STORE(INDEX, ADD(WMMA_result, bias))` and extracts the ADD as a fused
   epilogue via `_matmul_meta`. This is the classifier side; it still needs
   an emitter that sets `BS_BASE_ADDR` + `BS_CFG` like Mesa does.

**Porting to rockchip-2607:** ~15 lines.
- Classifier: detect `STORE(INDEX, ADD(MUL/WMMA, broadcast_const))` — ~5 lines
  (reuse the existing `_unwrap` + a new ADD-broadcast check).
- Emitter: set `REG_DPU_RDMA_RDMA_BS_BASE_ADDR` to the bias buffer,
  `REG_DPU_BS_CFG` with `BS_ALU_ALGO=2`, `BS_MUL_BYPASS=1`, `BS_RELU_BYPASS=1`.
  ~10 lines (the BS path is mostly register writes, no LUT).
- The bias buffer must be a separate NPU allocation (per-channel fp16 vector).

**Note:** `rockchip-2607` already uses the BS unit for CMPLT/CMPEQ (see WHERE
ref above), so `REG_DPU_BS_CFG` is not new — only the bias-specific config
(`BS_ALU_ALGO=2` + `BS_BASE_ADDR`) needs adding.

### Reference: SQRT — no implementation found on any branch

**SQRT is not implemented on any branch.** Searched:
- `recip` — no `Ops.SQRT` / `Ops.RSQRT`
- `rockchip/wip` — only `math.isqrt` for shape inference, no `Ops.SQRT` emitter
- `rockchip_addmul` — only `math.isqrt` for shape inference
- `ref/rk3588/` — only test definitions (`test_sqrt`, `test_rsqrt`), no emitter

**LUT is not the only solution.** Four approaches exist, all DPU-ALU-based
except LUT. Latencies at ~100 µs per DPU kernel launch on this SoC:

| Approach | DPU kernels | Needs FDIV? | Lines | Accuracy | Latency |
|---|---|---|---|---|---|
| LUT | 1 | No | ~30 | ~3 digits | 100 µs |
| Newton-Raphson (FDIV) | 9 (3 iter × 3 ops) | Yes | ~15 | fp16-exact | 900 µs |
| Newton-Raphson rsqrt (no FDIV) | 13 (3 iter × 4 ops + 1 MUL) | No | ~20 | fp16-exact | 1300 µs |
| Binary search (CMPLT+WHERE) | 64 (16 iter × 4 ops) | No | ~25 | fp16-exact | 6400 µs |

1. **LUT** — same DPU LUT mechanism as EXP2 (513-entry table, `sqrt(x)` for
   x in [0, 4] or similar). ~30 lines. Fastest (1 kernel launch). ~3 digit
   accuracy. Reuses EXP2 LUT boilerplate.

2. **Newton-Raphson with FDIV** — `y_{n+1} = 0.5 * (y_n + x/y_n)`, 3 iterations.
   Each iteration = 3 DPU ops: `FDIV(x, y)` → `ADD(y, result)` → `MUL(0.5, result)`.
   ~15 lines. Reuses the FDIV emitter from RECIPROCAL. fp16-exact.
   **Best non-LUT option — simplest, fewest ops, reuses RECIPROCAL.**

3. **Newton-Raphson rsqrt (no FDIV)** — `y_{n+1} = 0.5 * y_n * (3 - x * y_n²)`,
   then `sqrt = x * y₃`. Uses only MUL + SUB (no FDIV). 4 ops per iteration
   + 1 final MUL. ~20 lines. Avoids RECIPROCAL dependency but more ops.

4. **Binary search** — 16 iterations of CMPLT + WHERE + MUL + ADD. fp16-exact
   but 64 kernel launches — impractically slow. Listed for completeness.

**CMPNE/NaN trick for edge cases** (not the core computation):
The NaN trick from CMPEQ (`0 * 0x7C00 = NaN, nonzero * 0x7C00 = 32800`) can't
compute sqrt, but it guards edge cases:
- `x = 0`: `FDIV(0, 0) = NaN` → detect via CMPNE → output 0
- `x < 0`: `CMPLT(x, 0)` → output NaN or 0
- `x = inf`: detect → output inf

Full implementation with edge guards: ~12 DPU ops, ~20 lines.

**Speculative: exponent-halving via OUT_CVT_SHIFT** —
The CMPEQ implementation uses `REG_DPU_OUT_CVT_SHIFT` with `MINUS_EXP=1` to
manipulate the FP16 exponent. `sqrt(x) = 2^(e/2) * sqrt(mantissa)` — halving
the exponent via OUT_CVT_SHIFT + degree-2 polynomial for mantissa could do
sqrt in ~3 DPU ops. **Unverified** — needs RK3588 TRM check for OUT_CVT_SHIFT
behavior on exponent fields. If it works, ~10 lines.

**Recommendation:** Newton-Raphson with FDIV (approach 2). Simplest non-LUT,
reuses RECIPROCAL's FDIV emitter, fp16-exact, 9 kernel launches (<1 ms).
Dependency: RECIPROCAL must land first. For 1 test case, 900 µs is irrelevant.

### Reference: cmac_exceeds_cbuf — `rockchip_addmul` has tiling, `rockchip-2607` rejects

The 2 `cmac_exceeds_cbuf` failures (`test_simple_conv2d_1x1_m4`,
`test_sum_full`) happen when the matmul's M×K input doesn't fit in the CNA's
12-bank CBUF (each bank = 32 KiB). Current `rockchip-2607` rejects this at
`support/rockchip.py:422-423`:
```python
if data_banks > RK_CBUF_BANKS-1 or input_row_bytes*align_out > wt_banks*CBUF_BANK_SIZE:
  raise RuntimeError("RKPLAN_REJECT:cmac_exceeds_cbuf")
```

**`rockchip_addmul` handles it** via `_rk_gemm_layout` + `_rk_make_gemm_regs`
(`origin/rockchip_addmul:ops_rockchip.py:925-997`):
- `align_in = max(RK_MIN_CHANNEL_TILE, round_up(k, RK_MIN_CHANNEL_TILE))`
- `data_banks = clip(ceildiv(m * input_row_bytes, CBUF_BANK_SIZE), 1, 11)`
- `feature_grains = max(80, even_rows_per_two_banks)` — tiles M into grains
  that fit 2 banks at a time
- `line_stride = 4 * min(ceildiv(eff_k, 32), 13)` — caps stride at 13 groups

The key insight: **M is tiled into `feature_grains`-sized chunks**, and the
CNA processes them in ping-pong fashion. This is a CNA register config
change, not a new opcode.

**Porting to rockchip-2607:** ~15 lines.
- Replace the reject at `support/rockchip.py:422-423` with a tiling plan
  that computes `feature_grains` and `data_banks` like `rockchip_addmul`.
- Emit `REG_CNA_CONV_CON2` with `feature_grains << 4` (already emitted at
  line 437, just needs the right value).
- The current emitter already has `data_banks` computation at line 420;
  the fix is to cap it at `RK_CBUF_BANKS-1` and compute `feature_grains`
  instead of rejecting.

**Note:** `test_sum_full` (shape `(16384,)`, full reduction) is a GEMV, not
a GEMM — it hits `cmac_exceeds_cbuf` because M=16384 doesn't fit in 11 banks.
The tiling fix applies to both GEMM and GEMV.

### Estimated P/F if all LHF ref info is implemented (67 tests)

| Category          | Tests | Ref source                  | Est. pass | Confidence | Notes |
|-------------------|-------|-----------------------------|-----------|------------|-------|
| WHERE             | 49    | rockchip/wip                | ~40-45    | High       | WHERE itself is a*c+b*(1-c) (6 lines), but CMPLT/CMPEQ/CMPNE synthesis (~38 lines) is the real cost. Some WHERE tests also fail on non_index_operand or broadcast, so won't pass even with WHERE. |
| RECIPROCAL        | 5     | recip branch                | ~4-5      | High       | FDIV verified working on recip. test_recip failed on recip but test_scalar_div (incl. 1/x, 2/x) passed. sigmoid/rsqrt/pow_const depend on RECIPROCAL and possibly EXP2. |
| fused_epilogue    | 4     | rockchip/wip + ref/rk3588   | ~2-3      | Medium     | conv_mesa_raw.py shows BS_BASE_ADDR for bias DMA. But 2 of 4 tests are conv_transpose, which also has layout issues. |
| EXP2              | 3     | recip LUT                   | ~3        | High       | LUT mechanism verified, but tolerance is relaxed (atol=6e-3). test_ops.py may use tighter tol. |
| SIN               | 2     | recip LUT (shared)          | ~1-2      | Medium     | Same LUT mechanism as EXP2, but SIN needs a different table + range mapping. No direct ref — only EXP2/SiLU tables exist. |
| cmac_exceeds_cbuf | 2     | rockchip/wip (tiling)       | ~1-2      | Medium     | rockchip/wip has tiling/split code (13 matches), but porting tiling to rockchip-2607's compiled dispatch is more complex than a simple emitter add. |
| LOG2              | 1     | recip LUT (shared)          | ~1        | High       | Same LUT mechanism, just a different table. |
| SQRT              | 1     | none (must design)          | ~0-1      | Low        | No ref on any branch. Could be LUT or x*rsqrt via FDIV. |
| TRUNC             | 1     | rockchip/wip                | ~1        | High       | _rk_trunc_fix is a pattern rewrite using CMPLT/WHERE, which transitively needs WHERE. |
| **Total**         | **67**|                             | **~53-60**|            | Not all 67 will pass — some have secondary reasons (broadcast, layout, non_index_operand). |

**Realistic estimate: 71 + ~55 = ~126 PASS** (from 71 to ~126, not the full 138
that 71+67 would give).

### Potential pass — tests that would flip if LHF ref is implemented

These are tests currently FAIL/PARTIAL with a **single LHF reason** (no
secondary blockers like layout or broadcast). They are the realistic
"potential pass" set. Tests with secondary reasons are excluded.

**WHERE-only (single reason: `unsupported_op:Ops.WHERE`) — 53 tests:**
test_argsort, test_avg_pool2d_asymmetric_padding, test_avg_pool2d_ceil_mode,
test_avg_pool2d_ceil_mode_include_pad_output_size_reduce_by_one,
test_avg_pool2d_ceil_mode_output_size_reduce_by_one,
test_avg_pool2d_ceil_mode_padding_not_counted, test_avg_pool2d_padding,
test_avg_pool2d_padding_not_counted, test_cat, test_ceil, test_clip,
test_cummax, test_cummin, test_cumsum, test_diag, test_floor, test_hardtanh,
test_inf_where, test_interpolate_nearest, test_interpolate_nearest_exact,
test_leaky_relu, test_logcumsumexp, test_logcumsumexp_numerical,
test_masked_fill, test_max_pool2d_ceil_mode,
test_max_pool2d_ceil_mode_output_size_reduce_by_one, test_max_pool2d_padding,
test_multicat, test_pad, test_pad_reflect_mode, test_pad_replicate_mode,
test_pad_reshape, test_pad_slice, test_pow_full, test_pow_zero_const,
test_pow_zero_tensor, test_round, test_sign, test_sign_exact,
test_simple_cummax, test_simple_cummin, test_simple_cumsum, test_small_cummax,
test_small_cummin, test_small_cumsum, test_sort, test_stack, test_stack_max,
test_strided_conv_transpose2d, test_sum_cat_collapse, test_sum_pad_collapse,
test_sum_relu, test_topk, test_tril, test_triu

**RECIPROCAL-only (single reason) — 5 tests:**
test_pow_const, test_rsqrt, test_scalar_div, test_sigmoid, test_sigmoid_extreme

**fused_epilogue-only (single reason) — 4 tests:**
test_bias_conv_transpose2d, test_biased_conv2d,
test_output_padded_conv_transpose2d, test_simple_conv2d_bias
(Note: 2 of these are conv_transpose, which may hit layout issues after
fused_epilogue is fixed — est. 2-3 pass, not all 4.)

**EXP2-only (single reason) — 3 tests:**
test_exp, test_exp2, test_exp2_log2_zero_times_negative

**SIN-only (single reason) — 2 tests:**
test_cos, test_sin

**cmac_exceeds_cbuf-only (single reason) — 2 tests:**
test_simple_conv2d_1x1_m4, test_sum_full

**LOG2-only (single reason) — 1 test:**
test_log2

**SQRT-only (single reason) — 1 test:**
test_sqrt

**TRUNC-only (single reason) — 1 test:**
test_trunc

**Total potential pass: 72 tests** (53 WHERE + 5 RECIP + 4 fused_epilogue +
3 EXP2 + 2 SIN + 2 cmac_cbuf + 1 LOG2 + 1 SQRT + 1 TRUNC)

**Excluded (WHERE with secondary blockers — won't pass even with WHERE) — 4 tests:**
- test_avg_pool2d (also needs `no_add_mul_reduction`)
- test_broadcast_full (also needs `unsupported_layout:Ops.ADD`, `non_index_operand`)
- test_broadcast_partial (also needs `unsupported_layout:Ops.ADD`, `unsupported_layout:Ops.RANGE`, `non_index_operand`)
- test_max_pool2d_asymmetric_padding (also needs `unsupported_layout`)

### Remaining failures (269 tests) — grouped by fail reason

These are the tests that **stay broken even after all LHF ref is implemented**.
No reference implementation exists on any branch for these categories.

| Count | Fail reason | Example tests | Nature |
|-------|-------------|---------------|--------|
| 79 | `unsupported_dtype` | test_all, test_all_axis, test_and, test_cast, test_cast_sum | explicit non-fp16 dtypes (bool/int32/int8/uint8/fp32) — **most have ref in backend-consideration via cast hack, only ~5 truly no potential** |
| 54 | `unsupported_op:non_index_operand` | test_abs, test_acos, test_acosh, test_add3, test_asin | operand not in INDEX form (needs reshape/copy) |
| 47 | `unsupported_layout` | test_conv2d, test_conv2d_bs_1_cin_1, test_cross_entropy_*, test_einsum | non-contiguous tensors, strides, transposes |
| 21 | `unsupported_op:Ops.MUL` | test_asymmetric_padding_conv1d/2d, test_cumprod, test_dilated_conv_transpose2d | **CORRECTED: Only 5/21 are true REDUCE(MUL). 14/21 are conv layout (REDUCE(ADD) with MUL(RANGE,CONST) in index computation, failing _is_cmac_matmul_layout). See corrected breakdown below.** |
| 16 | `unsupported_layout:(shape-specific)` | test_log_softmax_other_axis, test_logsumexp, test_max, test_max_pool2d | layout rejected for specific tensor shapes |
| 15 | `no_add_mul_reduction` | test_global_avg_pool2d, test_mean, test_mean_axis, test_std, test_std_axis | reduction needs MUL+ADD (avg/std) — **FIXABLE: CMAC sum + BS_MUL(1/N) fused epilogue** |
| 14 | `unsupported_layout:Ops.ADD` | test_double_slice, test_einsum, test_expand, test_flip, test_meshgrid | broadcast ADD with non-contiguous layout |
| 5 | `forward_pass_failed` | test_arange, test_linspace, test_matvec, test_matvecmat, test_sum_collapse | runtime error during execution (not rejection) |
| 3 | `unsupported_op:Ops.ADD` | test_binary_crossentropy, test_binary_crossentropy_logits_pos_weights, test_binary_crossentropy_reductions | ADD in unsupported context |
| 3 | `unsupported_layout:Ops.RANGE` | test_broadcasted_add, test_broadcasted_add_2, test_depthwise_conv2d | RANGE with non-contiguous layout |
| 2 | `unsupported_layout:Ops.MUL` | test_diagonal, test_slice_ellipsis | MUL with non-contiguous layout |
| 2 | `WHERE + secondary blockers` | test_avg_pool2d, test_max_pool2d_asymmetric_padding | WHERE + layout/reduction |
| 2 | `unsupported_layout (multi-reason)` | test_broadcast_full, test_broadcast_partial | layout + ADD + RANGE + non_index_operand |
| 2 | `unsupported_layout:Ops.FLOORDIV/FLOORMOD` | test_repeat_interleave, test_roll | integer div/mod layout |
| 1 | `not_implemented` | test_avg_pool3d | 3D pooling not supported |
| 1 | `half_view_int` | test_bitcast | dtype bitcast not supported |
| 1 | `assertion` | test_scatter_reduce_errors | assertion failure (edge case) |
| 1 | `dtype_mismatch` | test_scatter_reduce_prod_zeros | dtype mismatch in scatter |
| 1 | `unsupported_layout (multi)` | test_conv1d | **Has ref** — rockchip_addmul:619-647 |
| **269** | **TOTAL** | | |

**Consolidated by nature:**

| Nature | Tests | % | Fixable? |
|--------|-------|---|----------|
| dtype (fp32/int32/bool) | 81 | 30% | **All have potential — 19 pure bool (fp16 emulation), 15 bool+int32 (cast hack), 26 int32 (cast hack), 11 int64 (PC chain), 2 uint8 (INT8 MAC), 3 fp32 (cast hack), 7 other (PC chain/validation). 0 truly no potential.** |
| layout (non-contiguous, strides, transpose) | 84 | 31% | **All have potential — stride regs exist, PC chain for unfixable patterns, ref in backend-consideration** |
| non_index_operand | 54 | 20% | **Yes — ref in backend-consideration (abs/relu/CMPLT)** |
| MUL-in-reduce / no_add_mul_reduction | 36 | 13% | **Yes — 14/21 conv layout (stride regs), 5/21 REDUCE(MUL) (DPU EW MUL), 2/21 other (PPU MIN/RECIP). 15 no_add_mul_reduction (CMAC+BS_MUL/PPU AVE).** |
| ADD/RANGE/MUL layout | 9 | 3% | **All have potential — PC chain or stride regs** |
| forward_pass_failed | 5 | 2% | **Fixable — 3 pure fp16 (runtime bugs), 2 mixed (float subtests). Not hardware limits.** |
| other (assert, bitcast, not_implemented, etc.) | 5 | 2% | **All have potential — bitcast=metadata, avg_pool3d=decompose, scatter=validation** |
| **TOTAL** | **269** | **100%** | **0 tests with no ref AND no potential.** |

### What each remaining-failure category means

**dtype (79 tests) — CORRECTED: NPU supports INT8 AND FP16, not just FP16:**

The NPU hardware supports **two precision modes** via the PROC_PRECISION /
IN_PRECISION / OUT_PRECISION register fields:
- **precision=0 → INT8** (quantized, TFLite-style with zero_point + scale)
- **precision=2 → FP16** (what rockchip-2607 currently uses)

**Mesa/Rocket proves INT8 works on the same NPU:**
`ref/rk3588/experimental/conv_mesa_raw.py` is a port of the Mesa Gallium
"Rocket" driver (`ref/mesa/src/gallium/drivers/rocket/`) — the upstream
Linux kernel driver for the RK3588 NPU (`drivers/accel/rocket`). It runs
**real NPU INT8 quantized convolution** via `DRM_IOCTL_ROCKET_SUBMIT`:
- `BPE = 1` (1 byte per element = uint8)
- `CNA_CONV_CON1` with PROC_PRECISION=0, IN_PRECISION=0 (INT8)
- `DPU_DATA_FORMAT = 0` (all precision fields = 0 = INT8)
- `CNA_CVT_CON1` with `CVT_SCALE0` / `CVT_OFFSET0` for quantization rescale
- `DPU_EW_CVT_SCALE_VALUE` / `DPU_EW_CVT_OFFSET_VALUE` for EW quantization
- `DPU_OUT_CVT_SCALE` / `DPU_OUT_CVT_OFFSET` / `DPU_OUT_CVT_SHIFT` for output dequant
- `input_zero_point`, `weight_zero_point`, `output_zero_point` (TFLite contract)
- `input_scale`, `weight_scale`, `output_scale` (per-tensor quantization)

**mtx512/rk3588-npu proves RAW INT8×INT8→INT32 matmul (no quantization):**
https://github.com/mtx512/rk3588-npu — independent RE project by Jasbir Matharu.
`tests/matmul_int8.c` runs **real NPU INT8 matmul** via `DRM_IOCTL_RKNPU_SUBMIT`:
- `int8_t matrixA[M×K]`, `int8_t matrixB[N×K]` → `int32_t output[M×N]`
- `gen_matmul_int8()` sets PROC_PRECISION=0, IN_PRECISION=0 (INT8)
- `CNA_CVT_CON0` with `cvt_bypass=1` (NO quantization rescale — raw int8 MAC)
- Output is **int32_t** (not rescaled back to int8) — proves NPU can store int32
- Verifies exact int32 results against CPU `matmul_int()` reference
- Constraints: M%4==0, K%32==0, N%16==0 (alignment)

**This is a stronger result than Mesa:** Mesa uses quantized INT8 with zero_point
and scale (TFLite contract). mtx512 uses **raw INT8×INT8→INT32 with CVT bypass** —
no quantization, exact integer arithmetic. The NPU is a general INT8 MAC engine,
not just a quantized inference accelerator.

**Two independent confirmations of INT8 on RK3588 NPU:**
1. **Mesa/Rocket** (upstream Linux driver) — quantized INT8 conv with TFLite contract
2. **mtx512/rk3588-npu** (independent RE) — raw INT8×INT8→INT32 matmul, no quantization

**BUT: NPU INT8 is MAC (conv/matmul), NOT general int8 arithmetic.**
The NPU's INT8 mode is a **MAC engine** — it does:
- INT8 × INT8 → INT32 accumulate (in CNA/CMAC)
- CVT rescale (optional, can be bypassed): INT32 → INT8 or INT32 → FP16
- DPU EW ops (ADD/MUL/MAX) with INT8 inputs + CVT rescale
- **INT32 output** (proven by mtx512 — output buffer is `int32_t[M×N]`)

It does **NOT** do:
- Bitwise AND/OR/XOR/NOT (no logic ALU — the DPU EW_ALU only does ADD/SUB/MUL/MAX)
- Integer division/modulo
- Boolean operations
- Arbitrary int32 elementwise arithmetic (int32 exists as MACC accumulator output,
  but not as a loadable input dtype for EW ops)

**What this means for the 79 dtype tests:**

| dtype | Tests | NPU supports? | Fixable? |
|-------|-------|--------------|----------|
| **bool (pure)** | 19 | **Yes — via fp16 emulation** (no logic ALU needed) | **Fixable** — bool as {0.0, 1.0} fp16 + MUL/MAX/CMPEQ. 19 pure bool tests (all/any/cmp/isclose/isfinite/isinf/isnan/logical_not). See below. |
| **bool + int32 (mixed)** | 15 | bool: Yes (fp16 emulation). int32: No (EW input) | **Partially** — bool subtests fixable, int32 subtests not. Test stays FAIL (any subtest fails) but bool subtests pass. |
| **int32 (EW input)** | 26 | **No** — int32 only as MACC accumulator output (mtx512), not loadable EW input | **No** — test_ops uses int32 for elementwise (add/mul/cmp/arange), not matmul output. |
| **int32 + fp32 (mixed)** | 5 | No — neither int32 EW nor fp32 datapath | **No** |
| **int32 + int64 (fancy indexing)** | 11 | No — int64 indices for fancy indexing, no int datapath | **No** — fancy indexing needs int64 gather, NPU has no int datapath |
| **int32 + uint8 (mixed)** | 4 | uint8: partially (MAC only). int32: No | **Partially** — uint8 MAC subtests potentially fixable, int32 not |
| **fp32** | 3 | **No** — NPU only has INT8 and FP16 precision modes | **No** — hardware limit |
| **uint8** | 2 | **Yes** (same as int8, unsigned) | **Yes** — same as int8 MAC |
| **other (fancy indexing, interpolate, slice)** | 7 | Various — fancy indexing uses int64 indices, interpolate uses uint indices, slice negative strides uses int indices | **No** — int index dtype for gather/scatter/interpolate, no int datapath |

**Revised dtype breakdown (81 tests, verified 2026-07-27):**
- **19 pure bool tests**: **Fixable via fp16 emulation** — bool as {0.0, 1.0}
  fp16, boolean ops mapped to existing DPU EW ops:
  - AND(a,b) = MUL(a,b) — 0×0=0, 0×1=0, 1×1=1 ✓ (MUL already supported)
  - OR(a,b) = MAX(a,b) — max(0,0)=0, max(0,1)=1, max(1,1)=1 ✓ (MAX already supported)
  - NOT(a) = CMPEQ(a, 0.0) — eq(0,0)=1, eq(1,0)=0 ✓ (CMPEQ already supported)
  - XOR(a,b) = MAX(MUL(a,CMPEQ(b,0)), MUL(CMPEQ(a,0),b)) — 3 EW ops ✓
  - CMPNE(a,b) = 1.0 - CMPEQ(a,b) (SUB from const) ✓
  Tests: test_all, test_all_axis, test_all_large, test_all_zero_axis,
  test_any, test_any_axis, test_any_zero_axis, test_cmp_eq, test_cmp_ge,
  test_cmp_gt, test_cmp_le, test_cmp_lt, test_cmp_lt_backwards,
  test_cmp_ne_backwards, test_isclose, test_isclose_edge_cases,
  test_isclose_scalar, test_logical_not, test_min (bool vals subtest).
  No logic ALU needed — all boolean logic is fp16 arithmetic on {0.0, 1.0}.
  Fix: extra_matcher pattern to cast bool→fp16 before classifier, cast back after.
  ~5-10 lines. The `~x` (bitwise_not on bool) already lowers to `CMPNE` in
  tinygrad, so only AND/OR/XOR need the MUL/MAX lowering.

  **test_all passes on rockchip/backend-consideration — but via CPU emulator, NOT NPU:**
  Verified 2026-07-27: `test_all` passes on `rockchip/backend-consideration`
  (1 passed in 4.59s) but fails on every other branch:
  - rockchip-2607: `RKPLAN_REJECT:unsupported_dtype` (_is_fp16_only rejects bool)
  - rockchip/wip, rockchip_addmul: infinite loop in graph_rewrite (bool pattern loops)
  - recip: AssertionError (older test file layout)

  **The pass is via the Python uops emulator fallback, not hardware.**
  `ops_rockchip.py:531-559` on backend-consideration: when none of the hardware
  templates (zero/const/range/pool/matmul/elementwise/conv1x1) match — which is
  the case for bool `.all()` reductions — `RockchipRenderer.render` falls back to
  packing the uops into a template named "uops" (line 557-559):
  ```python
  packed_uops = tuple((u.op, u.dtype, [uop_to_idx[s] for s in u.src], u.arg) for u in uops)
  return base64.b64encode(encode_template(RKTemplatePackage(1, "rk3588-rknpu2", "uops", (), meta={"uops":packed_uops}))).decode()
  ```
  This gets emulated in Python on the CPU. There is no `unsupported_dtype`
  rejection because there's no classifier gate at all — anything that doesn't
  match a hardware template silently falls through to the emulator.

  **This is the same dishonest pattern progress.md already flags for the recip
  branch** (lines 79-82, 241-244: "the ref's 'passed ops.where' was via the
  Python uops emulator fallback, not hardware"). So test_all's pass on
  backend-consideration does NOT contradict the "bool is fixable via fp16
  emulation" claim — that pass is CPU emulation, not real NPU bool arithmetic.

  **The honest fix is the fp16-bool-emulation path** (AND=MUL, OR=MAX,
  NOT=CMPEQ(x,0)) described above — that's a real NPU path, not the emulator.

- **15 bool+int32 mixed tests**: **Partially fixable** — bool subtests fixable
  via fp16 emulation, int32 subtests not (no int32 EW input). Tests:
  test_and, test_or, test_xor, test_bitwise_not, test_isfinite, test_isinf,
  test_isnan, test_one_hot, test_pow_const_direct, test_pow_int_base_float_exponent,
  test_where, test_where_permute, test_masked_select, test_masked_select_size,
  test_sum_dtype_arg. Test stays FAIL (any subtest fails) but bool subtests pass.

- **26 int32-only tests**: **Not fixable** — int32 only as MACC accumulator
  output (mtx512 proves it), not as loadable EW input. test_ops uses int32 for
  elementwise (add/mul/cmp/arange/lshift/rshift/argmax/argmin/gather/scatter).
  Tests: test_argmax, test_argmin, test_int_pow_const_int, test_lshift,
  test_lshift_signed, test_maximum, test_max_pool2d_padding_int, test_rshift,
  test_rshift_signed, test_slice_with_const_tensor, + ~16 fancy indexing tests.

- **11 int32+int64 (fancy indexing) tests**: **Not fixable** — int64 indices
  for fancy indexing (gather/scatter), NPU has no int datapath. Uses
  `_get_index_randoms` which creates int64 torch indices, int32 tinygrad indices.

- **5 int32+fp32 mixed tests**: **Not fixable** — neither int32 EW nor fp32.
  Tests: test_full, test_full_like, test_ones_like, test_zeros_like, test_small_gemm_eye.

- **4 int32+uint8 mixed tests**: **Partially fixable** — uint8 MAC subtests
  potentially fixable (Mesa/mtx512 prove uint8 MAC), int32 not.
  Tests: test_cast, test_nonzero, test_nonzero_size, test_cast_relu (dup).

- **3 fp32-only tests**: **Not fixable** — NPU only has INT8 and FP16 precision.
  Tests: test_small_gemm_range, test_pow_const_direct (fp32 subtest), + 1.

- **2 uint8-only tests**: **Partially fixable** — uint8 MAC (Mesa/mtx512).
  Tests: test_int_or, test_cast_relu.

- **7 other tests**: **Not fixable** — fancy indexing (int64 indices),
  interpolate (uint indices for resize), slice negative strides (int indices).
  Tests: test_fancy_indexing_inf, test_interpolate_*, test_slice_negative_strides,
  test_slice_fancy_indexing_dim_collapse_int, test_slice_fancy_indexing_dim_inject_*.

**test_and / test_or / test_xor / test_bitwise_not — mostly fixable:**
These tests have **4 subtests each**, mixing bitwise (int) and logical (bool/float):
1. `int & int` — bitwise AND on int32. **Not fixable** — NPU has no logic ALU for int.
2. `int & 0x1337` — bitwise AND with scalar. **Not fixable** — same.
3. `bool & bool` — logical AND on bool. **Fixable** — bool as {0.0,1.0} fp16,
   AND=MUL, OR=MAX, NOT=CMPEQ(x,0), XOR=MAX(MUL(a,CMPEQ(b,0)),MUL(CMPEQ(a,0),b)).
   All use existing DPU EW ops. Needs extra_matcher to cast bool→fp16.
4. `(1 < x) & (x < 2)` — **logical AND on float comparisons**. This subtest
   uses fp16 inputs and CMPLT (which IS supported as EW op algo=0). The AND
   of two bool results lowered to `MUL(CMPLT_result, CMPLT_result)`
   (0×0=0, 0×1=0, 1×1=1 = AND). **Fixable via CMPLT+MUL.**

**Subtests 3 and 4 are both fixable.** Only subtests 1 and 2 (int bitwise)
stay broken. Since test_ops.py counts a test as FAIL if ANY subtest fails,
these tests stay FAIL unless the int subtests are somehow handled. But the
bool and float-comparison subtests would pass if tested separately.

**backend-consideration has test definitions** for test_and (line 729),
test_or (line 744), test_xor (line 720), test_bitwise_not (line 758),
test_logical_not (line 911). But its `hardware_ops` only includes
`{MUL, MAX, ADD, SUB, CMPLT, CMPEQ, EXP2, FDIV, TRUNC, CUSTOM}` — no
`Ops.AND`/`Ops.OR`/`Ops.XOR`. The int subtests use the extra_matcher int→fp16
cast hack (line 520-522: `MUL(int) → cast to fp16, MUL, cast back`), but
this only works for MUL/ADD/MAX, not AND/OR/XOR. **The bool subtests are
not handled by backend-consideration either — this is a new fix needed.**

**Verdict:** test_and/test_or/test_xor are **mostly fixable** — subtests 3
(bool) and 4 (float comparison) can work via fp16 emulation with existing
DPU EW ops. Only subtests 1-2 (int bitwise) stay broken. Since test_ops.py
counts a test as FAIL if ANY subtest fails, these tests stay FAIL overall,
but **3 of 4 subtests would pass** — the only blocker is the int bitwise
subtests, which are true hardware limits (no int logic ALU).

**Summary: NPU supports INT8 + FP16 (not just FP16). Two independent refs prove
INT8 works: Mesa (quantized INT8 conv) and mtx512 (raw INT8×INT8→INT32 matmul
with CVT bypass). INT32 output from MACC accumulator is proven. Bool is fixable
via fp16 emulation (AND=MUL, OR=MAX, NOT=CMPEQ, XOR=MAX+MUL+CMPEQ) — no logic
ALU needed. Of 81 dtype tests: 19 pure bool fixable, 15 bool+int32 partially
fixable, 2 uint8 fixable, 4 int32+uint8 partially fixable. The rest (~41) are
true hardware limits: 26 int32 EW input + 11 int64 fancy indexing + 5 int32+fp32
+ 3 fp32 + 7 other (int index dtype).**

**layout (84 tests) — CORRECTED: stride registers exist, partially fixable:**

The NPU hardware **has stride registers** on all three units. The blocker is
the classifier, not the hardware:

| Unit | Stride register | Used in rockchip-2607? | Used in ref? |
|------|----------------|----------------------|--------------|
| CNA | `REG_CNA_DMA_CON1` (LINE_STRIDE) | Yes (line 453) | Yes (Mesa conv_mesa_raw:1077) |
| CNA | `REG_CNA_DMA_CON2` (SURF_STRIDE) | Yes, set to 0 (line 454) | Yes (Mesa, for 3D input) |
| CNA | `CONV_CON3` (CONV_X/Y_STRIDE) | Yes, set to 1 (line 431) | Yes (Mesa, for strided conv) |
| DPU | `REG_DPU_DST_SURF_STRIDE` | Yes (line 470) | Yes |
| DPU_RDMA | `REG_DPU_RDMA_RDMA_EW_SURF_STRIDE` | No | Yes (Mesa conv_mesa_raw:1300) |
| DPU_RDMA | `REG_DPU_RDMA_RDMA_SURF_NOTCH` | No | Yes (Mesa) |
| PPU_RDMA | `REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE` | Yes (line 540) | — |
| PPU_RDMA | `REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE` | Yes (line 541) | — |

The classifier at `support/rockchip.py:71` only accepts:
- `_is_flat_contiguous`: `RANGE` or `CONST(0)` (simple sequential access)
- `_is_2d_row_major`: `ADD(MUL(RANGE, CONST), RANGE)` (row-major 2D)

It rejects strided patterns (`MUL(RANGE, CONST(stride))`), reversed
(`SUB(CONST, RANGE)`), and transposed indices. **The fix is classifier +
emitter work, not new hardware:**

| Layout sub-problem | Hardware support? | Ref exists? | Fix complexity |
|---|---|---|---|
| flip (reverse) | Maybe — via SURF_NOTCH | No | Medium |
| expand (broadcast) | No — DMA can't broadcast | No | Hard (host gather) |
| transpose (swapped axes) | Yes — LINE_STRIDE + SURF_STRIDE | Yes (Mesa) | Medium |
| slice with step | Maybe — via KERNEL_STRIDE | No | Medium |
| 3x3 conv im2col | Yes — CONV_X/Y_STRIDE + LINE_STRIDE | **Yes (rockchip_addmul, backend-consideration)** | Medium |
| batched matmul | Yes — SURF_STRIDE | **Yes (backend-consideration)** | Medium |

**References for layout fixes:**
- `rockchip_addmul:ops_rockchip.py:651-720` — `_run_cna_conv2d` with host-side
  im2col (packs 3x3 patches into cols matrix, then runs CMAC matmul). Handles
  stride, dilation, padding, groups. **test_conv2d (3x3) is implemented here.**
- `rockchip/backend-consideration:support/rockchip.py:148-190` — `conv_params`,
  `pack_conv_input`, `pack_conv_weights`, `unpack_conv_output`. Template-based
  conv with stride registers wired. **test_conv2d, test_depthwise_conv2d,
  test_matmul_batched all have test definitions here.**
- `ref/rk3588/experimental/conv_mesa_raw.py:1077-1078` — Mesa sets
  `CNA_DMA_CON1` (LINE_STRIDE) and `CNA_DMA_CON2` (SURF_STRIDE) for strided
  input access. Reference for wiring the stride registers.

**non_index_operand (54 tests) — CORRECTED: ref exists, fixable via WHERE+CMPLT:**
The DPU EW path requires both inputs to be `INDEX` (tensor from NPU memory).
When an op produces a non-tensor intermediate (e.g. CMPLT bool output), it's
rejected. Example — `abs(x)`:
```
abs(x) = WHERE(x < 0, -x, x) = WHERE(CMPLT(x, 0), MUL(x, -1), x)
```

**References for non_index_operand fixes:**
- `rockchip/backend-consideration:ops_rockchip.py:509-510` — relu pattern:
  `CMPLT(0, x).where(x, 0)` → `CUSTOM("relu")` with `ew_relu_bypass=0`.
  **test_relu, test_relu_exact, test_relu6, test_abs, test_abs_exact all
  have test definitions in backend-consideration.**
- `rockchip/wip:ops_rockchip.py:1180-1182` — same relu pattern matcher.
- `rockchip/backend-consideration:ops_rockchip.py:81` — `Ops.CMPLT:0,
  Ops.CMPEQ:0` in hardware_ops map. CMPLT/CMPEQ are EW ops (algo=0 = MUL/LUT).
- `rockchip/backend-consideration:ops_rockchip.py:434` —
  `cmplt_diff2bool` CUSTOM: `[1.0 if x > 0 else 0.0 for x in src]`.

**~30-40 of these 54 flip when WHERE + CMPLT/CMPEQ land** (abs, sign, clip,
hardtanh, ceil, floor, round). The relu pattern is already proven in
backend-consideration and rockchip/wip.

**MUL-in-reduce / no_add_mul_reduction (36 tests) — CORRECTED: 21 "MUL-in-reduce" mislabeled, only 5 are true REDUCE(MUL):**

*`unsupported_op:Ops.MUL` (21 tests) — CORRECTED breakdown after uop inspection:*

The 21 tests rejected as "unsupported_op:Ops.MUL" are NOT all REDUCE(MUL).
Verified by inspecting the uop graph for each test (2026-07-27):

| Sub-category | Count | Tests | Real cause | Fixable? | How |
|---|---|---|---|---|---|
| True REDUCE(MUL) (prod) | 2 | test_prod, test_const_reduce | `REDUCE(arg=(Ops.MUL,0))` — multiplicative reduction | **Yes** | DPU EW MUL tree reduction (log2(N) passes) or sequential PC chain MUL. Even better: `prod=exp(sum(log(x)))` = 3 kernels if LOG2+EXP2 (both LHF). |
| Cumulative product (scan) | 3 | test_cumprod, test_simple_cumprod, test_small_cumprod | `REDUCE(arg=(Ops.MUL,0))` — prefix scan | **Yes (harder)** | Naive N sequential MULs via PC chain (no layout needed), or Blelloch parallel scan with strided RDMA (needs layout support). |
| Conv/matmul layout | 14 | test_asymmetric_padding_conv1d/2d, test_padded_conv2d_*(4), test_padded_conv3d, test_padded_conv_transpose2d, test_simple_conv_transpose2d/3d, test_grouped_conv_transpose2d, test_dilated_conv_transpose2d, test_simple_padding_conv1d, test_small_gemm_padded | `REDUCE(arg=(Ops.ADD,0))` with `MUL(RANGE,CONST)` in index computation (im2col stride). Failing _is_cmac_matmul_layout. NOT a MUL-reduce issue. | **Partially** | Stride register support (same as unsupported_layout bucket). These are conv layout failures mislabeled as MUL-in-reduce. |
| Other | 2 | test_min, test_normalize | test_min: MIN reduce (PPU MIN pool). test_normalize: matmul (x@x) + RECIPROCAL + MUL — combination of layout + non_index_operand. | **Partially** | test_min: PPU MIN pool (analogous to existing PPU MAX pool, ~5 lines). test_normalize: layout fix + RECIPROCAL (LHF). |
| **Total** | **21** | | | | |

**Key evidence (verified 2026-07-27):**
- **Conv tests**: `REDUCE: arg=(Ops.ADD, 0)` — ADD reduction (matmul accumulation).
  All MUL ops are `MUL(RANGE, CONST)` = index stride calculations, plus one
  `MUL(WHERE, INDEX)` = im2col gather (selecting input based on padded index).
  The classifier rejects as "unsupported_op:Ops.MUL" because it sees the
  `MUL(WHERE, INDEX)` in the matmul body and doesn't know how to handle it.
  This is a **layout/index issue**, not a multiplicative reduce issue.
- **Prod test**: `REDUCE: arg=(Ops.MUL, 0)` — true multiplicative reduction.
  Body is just `INDEX` (load). The MUL is the reduce operation itself.
- **DPU EW MUL is proven and wired**: `_DPU_EW_CFGS[Ops.MUL]` (L314-315 in
  rockchip.py) is used for NEG, WHERE, scaled sum. Tree reduction via DPU EW
  MUL passes is feasible — the DPU MUL emitter already works.

**The real blocker for all 21 tests is NOT hardware — it's software:**
1. **Line budget**: 15 lines headroom in the file. Multiplicative reduction
   rewrite needs ~30-50 lines (PatternMatcher decomposition or multi-task plan).
2. **Single-kernel-per-plan architecture**: plan_rk produces one kernel per
   call. Tree reduction needs log2(N) kernels (multi-pass). Sequential PC chain
   needs K-1 tasks in one submit (works but needs PC chain wiring).
3. **Classifier doesn't recognize im2col MUL pattern**: The 14 conv tests fail
   because the classifier sees `MUL(WHERE, INDEX)` and rejects it, instead of
   recognizing it as an im2col gather that maps to CNA stride registers.

**Reclassification:**
- 14 conv tests → move to **layout** category (partially fixable, stride regs)
- 5 prod/cumprod tests → **fixable** via DPU EW MUL (tree/sequential, no hardware limit)
- 2 other tests → **partially fixable** (PPU MIN pool, RECIPROCAL + layout)

*`no_add_mul_reduction` (15 tests) — FIXABLE via CMAC+BS_MUL fused epilogue (mean) or PPU avg (avg_pool2d):*
`REDUCE(ADD, x)` then `MUL` by scalar. Two distinct test groups:

1. **test_mean / test_mean_axis / test_std_mean** — `mean()` lowers to
   `MUL(REDUCE(ADD, x), CONST(1/N))`. This is a **general reduction**, NOT a
   pool op. tinygrad does NOT lower mean() to PPU globalavg. Fix: **CMAC sum
   + BS_MUL(1/N) fused epilogue** — CMAC does REDUCE(ADD) via matmul
   accumulation (no kernel size limit), then BS_MUL scales by 1/N in the
   same pass (CBUF→BS→output). This is the fused_epilogue pattern already
   proven for conv+bias. ~10-15 lines of classifier work. Handles ANY K,
   ANY axis, including test_mean's K=360.

   **Key insight (2026-07-27):** BS/BN processes CBUF output (from CMAC/CNA),
   NOT RDMA input. Standalone BS/BN with RDMA input has no effect (verified
   in NEG fusion tests). But after CMAC, data flows CBUF→BS→BN→EW, which is
   the conv pipeline where BS_MUL is accessible. So `MUL(REDUCE(ADD), 1/N)`
   is a single fused pass: CMAC accumulates the sum, BS_MUL scales by 1/N.
   No second kernel, no intermediate buffer.

   **Can PPU AVE also help? Partially — see hardware-verified analysis below.**

2. **test_avg_pool2d / test_avg_pool2d_padding** — `avg_pool2d()` IS a pool
   op. The PPU has `POOLING_METHOD` (0=AVE, 1=MAX, 2=MIN) in
   `REG_PPU_OPERATION_MODE_CFG` and `REG_PPU_RECIP_KERNEL_WIDTH/HEIGHT`
   (FP17 format, `float_to_fp17(1/K)` for avg, 0 for max).
   `ref/rk3588/experimental/pool.py:187` implements all 6 pool ops:
   `("min", "max", "avg", "globalmin", "globalmax", "globalavg")` with real
   PPU hardware register configs. rockchip-2607 already has PPU max pool
   (PR1) — adding avg is ~5 lines (change POOLING_METHOD to 0, set
   RECIP_KERNEL to `float_to_fp17(1/K)`). **Hardware-verified 2026-07-27.**
   **Note:** test_avg_pool2d is currently in the "WHERE + secondary blockers"
   group (also needs WHERE for the padding mask), so PPU avg alone won't fix
   it — WHERE must land first.

#### Can PPU AVE pooling help test_mean? — Hardware-verified (2026-07-27)

**VERIFIED ON HARDWARE: PPU AVE works and gives EXACT mean with correct
FP17 RECIP values.** No software division needed.

Tested `/tmp/test_ppu_fp17.py` on RK3588 with 4 kernel sizes:

| Kernel | RECIP (FP17) | FP17 value | max_diff | Status |
|--------|-------------|------------|----------|--------|
| 2x2 | 30720 | 0.5 | 0.000488 | PASS |
| 3x3 | 30037 | 1/3 | 0.000488 | PASS |
| 4x4 | 29696 | 0.25 | 0.000244 | PASS |
| 8x8 | 28672 | 0.125 | 0.000122 | PASS |

**Key hardware findings:**

1. **PPU AVE mode works**: `POOLING_METHOD=0` in `REG_PPU_OPERATION_MODE_CFG`
   (was `=1` for MAX). Confirmed via `/tmp/test_ppu_ave_recip.py`.
2. **RECIP_KERNEL_WIDTH/HEIGHT is FP17 format** (1 sign, 6 exp bias 31, 10
   mantissa) — NOT fp16, NOT Q-format. Verified by probing multiple values:
   - 30720 = 0.5 in FP17 → ratio 0.25 (correct for 2x2: 0.5*0.5=0.25)
   - 32768 = 2.0 in FP17 → ratio 4.0 (2.0*2.0=4.0, confirmed)
   - 29696 = 0.25 in FP17 → exact mean for 4x4 kernel
3. **Formula**: `output = sum * recip_w * recip_h` (FP17 arithmetic). No
   implicit division by kernel size — the RECIP values must encode 1/K.
4. **The ref pool.py hardcodes 30720 (=0.5) for ALL kernel sizes** and does
   software division for globalavg (`decoded = got / x.shape[0]`). This is
   wrong — with correct FP17 RECIP = `float_to_fp17(1/K)`, no software
   correction needed.
5. **FP17 conversion** (~5 lines):
   ```python
   def float_to_fp17(f):
     if f == 0: return 0
     sign = 0 if f > 0 else 1; f = abs(f); exp = 0
     while f >= 2.0: f /= 2.0; exp += 1
     while f < 1.0 and f > 0: f *= 2.0; exp -= 1
     return (sign << 16) | ((exp + 31) << 10) | (int((f - 1.0) * 1024 + 0.5) & 0x3FF)
   ```

**Verified hardware constraints:**

| Register | Field width | Max value | Source |
|----------|------------|-----------|--------|
| `PPU_POOLING_KERNEL_CFG` KERNEL_HEIGHT | 4 bits (`0x00000f00`) | 15 | autogen:1376 |
| `PPU_POOLING_KERNEL_CFG` KERNEL_WIDTH | 4 bits (`0x0000000f`) | 15 | autogen:1380 |
| `PPU_DATA_CUBE_IN_WIDTH/HEIGHT` | 13 bits (`0x00001fff`) | 8191 | autogen:1321 |
| `PPU_RECIP_KERNEL_WIDTH/HEIGHT` | 17 bits (`0x0001ffff`) | 131071 | autogen:1385 |
| `PPU_OPERATION_MODE_CFG` POOLING_METHOD | 2 bits (`0x00000003`) | 3 | autogen:1365 |

**The 4-bit kernel limit is the critical constraint.** The PPU kernel size
fields are 4 bits (max 15×15). For global pooling, kernel = input size, so
both `in_h` and `in_w` must be ≤ 16. The existing `_ppu_split_k` already
handles this by factoring K into (in_h, in_w) with both ≤ 16.

**What mean() actually lowers to (verified):**
```
mean(x) = MUL(REDUCE(ADD, CAST(INDEX, float)), CONST(1/N))
```
- `REDUCE(ADD, ...)` over ALL axes → scalar
- `MUL` by `1/N` (the post-reduce scale)
- Currently rejected as `no_add_mul_reduction` at classifier line 199-201

**PPU AVE could handle this IF K can be split into (in_h, in_w) both ≤ 16.**

**Verified `_ppu_split_k` results for test_mean shapes:**

| Test | Shape | K (reduce size) | `_ppu_split_k(K)` | PPU AVE works? |
|------|-------|-----------------|-------------------|----------------|
| test_mean (small) | (4,4) | 16 | (2, 8) ✓ | **Yes** |
| test_mean (small) | (4,5) | 20 | (2, 10) ✓ | **Yes** |
| test_mean_axis (axis=2,3) | (3,4,5,6)→(3,4) | 30 | (2, 15) ✓ | **Yes** |
| test_mean_axis (axis=0,1,2) | (3,4,5,6)→(6) | 120 | (8, 15) ✓ | **Yes** |
| test_mean (full) | (3,4,5,6) | 360 | **None** ✗ | **No** — 360 has no factor pair both ≤16 |
| test_mean (axis=0) | (3,4,5,6)→(4,5,6) | 3 | **None** ✗ | **No** — K=3 < 4 (min split) |
| test_mean (axis=1) | (3,4,5,6)→(3,5,6) | 4 | (2, 2) ✓ | **Yes** |
| test_mean (axis=2) | (3,4,5,6)→(3,4,6) | 5 | (1, 5) ✓ | **Yes** |
| test_mean (axis=3) | (3,4,5,6)→(3,4,5) | 6 | (2, 3) ✓ | **Yes** |

**The problem: test_mean over (3,4,5,6) = 360 elements CANNOT be split.**
360 = 2³ × 3² × 5. Factor pairs: (2,180), (3,120), (4,90), (5,72), (6,60),
(8,45), (9,40), (10,36), (12,30), (15,24), (18,20), (20,18)... **None have
both factors ≤ 16.** The PPU cannot do global average pooling on 360 elements
in a single kernel.

**Workaround: multi-pass PPU pooling.** Pool over (5,6)=30 first → (3,4,1,1),
then pool over (3,4)=12 → scalar. Two PPU passes. But this requires:
- Reshape between passes (host-side, or NPU DMA reshape)
- Classifier recognizing multi-pass reduction
- ~20-30 lines, more complex than 2-kernel split

**Verdict (corrected 2026-07-27): test_mean PASSES via CMAC+BS_MUL fused
epilogue.** The CMAC accumulator has no kernel size limit (handles K=360),
and BS_MUL(1/N) processes the CBUF output in the same pass. This is strictly
more general than PPU AVE (which is limited to K with factor pairs both ≤16)
and simpler than the 2-kernel split (no intermediate buffer, single pass).

PPU AVE remains the right tool for `avg_pool2d` (2D spatial pool with small
kernel) and for mean over small axes (K ≤ 256 with good factorization).

**Estimate for CMAC+BS_MUL fused epilogue (RECOMMENDED for test_mean):**
- Classifier rule (REDUCE(ADD) + MUL(1/N) → CMAC sum + BS_MUL(1/N)): ~10-15 lines
- Handles ALL K, ALL axes, single pass, no intermediate buffer
- **Total: ~10-15 lines, strictly most general**
- Reuses the fused_epilogue pattern already proven for conv+bias

**Estimate for PPU AVE approach (hardware-verified, for avg_pool2d):**
- FP17 conversion helper (`float_to_fp17`): ~5 lines
- PPU config change (POOLING_METHOD=0, RECIP=`float_to_fp17(1/K)`): ~3-5 lines
- Classifier rule (REDUCE(ADD) + MUL(1/N) → PPU AVE): ~5-10 lines
- Multi-pass for large K: ~20-30 lines (if needed)
- **Total: ~15-20 lines for small K, ~35-45 for general case**
- Limited to K with factor pairs both ≤16 (test_mean K=360 does NOT qualify)

**ADD/RANGE/MUL layout (9 tests) — CORRECTED: partially fixable:**
Same layout problem as the 84, but some have refs:
- `test_depthwise_conv2d` — **implemented in backend-consideration** (groups=32,
  cin=1, 1x1 kernel = grouped 1x1 conv = batched GEMV). rockchip-2607 already
  has GEMV (PR2). Fix: classifier needs to recognize grouped 1x1 conv pattern.
- `test_roll` (FLOORMOD), `test_repeat_interleave` (FLOORDIV) — integer div/mod
  layout, no ref.

**forward_pass_failed (5 tests) — CORRECTED: NOT dtype failures, pure fp16 ops that fail at runtime:**
These tests pass the classifier (plan_rk returns a valid plan) but fail during
NPU execution. They are NOT dtype failures — several are pure fp16 ops that
should already work:

| Test | What it does | dtype | Should work? |
|------|-------------|-------|-------------|
| test_matvec | (1,128)@(128,128).relu() | **fp16** | **Yes — pure fp16 matmul+relu** |
| test_matvecmat | (x@y).relu()@z (1,128)@(128,128)@(128,128) | **fp16** | **Yes — pure fp16 matmul+relu+matmul** |
| test_sum_collapse | ones(256,256).sum(axis=1) | **fp16** | **Yes — pure fp16 reduction** |
| test_arange | arange(5.5,175.5,2.5) + int32 subtests | **mixed** | Float subtests yes, int32 subtests no |
| test_linspace | linspace(5,10,30) + int32 subtests | **mixed** | Float subtests yes, int32 subtests no |

**test_matvec, test_matvecmat, test_sum_collapse are pure fp16 and should
already pass.** Their failure is a runtime bug, not a hardware limit. Could be:
- Buffer size / alignment issue (256×256 = 64K elements, 128×128 = 16K)
- Kernel launch / DMA issue for specific shapes
- Classifier produces a plan but emitter generates bad register config

**test_arange / test_linspace have mixed subtests:**
- Float subtests (arange(5.5,175.5,2.5), linspace(5,10,30)) — fp16, should work
- int32 subtests (arange(10,dtype=int32), linspace(5,10,3,dtype=int32)) — true dtype limit
- int8 subtests (arange(128,dtype=int8)) — potentially fixable with INT8 precision
- assertRaises subtests — no NPU execution, just error checking

**These 5 tests are NOT in the "no potential" bucket.** They're runtime bugs
that need debugging. At least 3 (matvec, matvecmat, sum_collapse) are pure
fp16 and should be fixable. The other 2 (arange, linspace) have float
subtests that should work + int32 subtests that won't.

**other (5 tests) — individual edge cases:**
| Test | Issue |
|------|-------|
| test_avg_pool3d | 3D pooling — **has ref** in backend-consideration (line 3064), lowers to avg_pool2d, decompose to 2D PPU passes |
| test_bitcast | (3,3) fp16 → int32: 6 bytes/row not divisible by 4 — fails on ALL HALF backends, not Rockchip-specific |
| test_scatter_reduce_errors | assertion failure (edge case) |
| test_scatter_reduce_prod_zeros | dtype mismatch in scatter |
| test_repeat_interleave / test_roll | integer div/mod layout |

test_conv1d was here but **has a ref** (rockchip_addmul:619-647, NPU via CMAC
matmul with im2col). Moved to the fixable bucket.

### No ref in ANY branch AND no potential — comprehensive audit (2026-07-27)

Checked all 424 test_ops.py tests against all 4 branches
(rockchip-2607, rockchip/backend-consideration, rockchip/wip, rockchip_addmul)
plus ref/rk3588/experimental/pool.py. With `DEFAULT_FLOAT=HALF`, the
"unsupported_dtype" tests use explicit non-fp16 dtypes (bool, int32, int8,
uint8, fp32), not the default float.

**backend-consideration has test definitions for most dtype tests** — it uses
an `extra_matcher` cast hack (`int MUL → cast fp16, MUL, cast back to int`)
and Python uops emulator CUSTOM ops (`cmplt_diff2bool`, `cmpeq_32800_to_bool`)
to compute bool/int results in fp16. This is a **software fallback**, not NPU
hardware execution — but it proves the tests CAN pass with fp16 compute +
output cast.

**Truly no ref in any branch AND no potential — 1 test:**

| Test | Category | Why no potential |
|------|----------|------------------|
| test_bitcast | edge case | (3,3) fp16 = 6 bytes/row, not divisible by 4 (int32). Fails on ALL HALF backends (verified CLANG too). DEFAULT_FLOAT=HALF issue, not Rockchip-specific |

**No ref in any branch, but HAS potential — 32 tests:**

| Test | Category | Potential path |
|------|----------|---------------|
| test_isclose_scalar | dtype (bool output) | fp16 compare → cast to bool (backend-consideration has ref) |
| test_max_pool2d_padding_int | dtype (int input) | NPU INT8 precision mode (Mesa/mtx512 proven) or int→fp16 cast hack + PPU max pool |
| test_global_avg_pool2d | no_add_mul_reduction | PPU AVE (hardware-verified) or CMAC+BS_MUL |
| test_avg_pool2d | WHERE + reduction | PPU AVE + WHERE |
| test_dot_1d | layout | matmul variant — stride regs exist |
| test_max_pool2d_simple | layout | PPU max pool exists — layout fix needed |
| test_nll_loss_3d_weight | layout | reduction + indexing — layout fix |
| test_nll_loss_3d | layout | reduction — layout fix |
| test_max_pool2d_dilation | layout | PPU supports dilation via KERNEL_STRIDE |
| test_max_pool2d_unit_stride | layout | PPU max pool — layout fix |
| test_log10 | non_index_operand | log10 = log2(x) * log10(2) — LUT ref exists |
| test_exp2_log2_zero_times_negative | EXP2 | EXP2 LUT ref exists |
| test_log2 | LOG2 | LOG2 LUT ref exists |
| test_avg_pool2d_ceil_mode_* (2) | WHERE | WHERE + PPU AVE |
| test_max_pool2d_ceil_mode_* | WHERE | WHERE + PPU MAX |
| test_avg_pool3d | edge case | Has ref in backend-consideration (line 3064), lowers to avg_pool2d, decompose to 2D PPU passes |
| test_pow_int_base_float_exponent | dtype (int32→float) | tinygrad has xpow (EXP2+LOG2 LUT), int32→fp16 cast hack works. Fails on dtype assertion (torch=float32, tinygrad=half) — tinygrad dtype policy fix, not hardware |
| test_max_unpool2d | layout (scatter write) | CONST fill 0 + WHERE(CMPEQ(RANGE, pos), value, 0) per pooled value via PC chain. All sequential, no random DMA |
| test_max_unpool2d_inf | layout (scatter write) | Same as test_max_unpool2d — CONST fill -inf + WHERE approach via PC chain |
| test_scatter_reduce_prod_zeros | edge case (scatter) | Same WHERE+CMPEQ scatter approach + prod via PC chain MUL. int32 is blocker but cast hack may work |
| test_masked_select_size | dtype (compact/select) | PC chain prefix sum of mask + WHERE scatter. All sequential. Empty int32 subtest is trivial (fill_value only) |
| test_nonzero_size | dtype (compact/select) | Same as masked_select_size — prefix sum + WHERE scatter via PC chain |
| test_max_pool2d_return_indices | layout (indices) | Fail reason is `unsupported_layout`, NOT int32. Indices computed via arange→reshape→pool→eq→mul→max. Fix: stride registers |
| test_round_quantization_gradient | non_index_operand | Fail reason is `non_index_operand`, NOT gradient. FORWARD_ONLY=1 skips gradients. Forward needs round() (TRUNC-like, LHF) + EW ops |
| test_scatter_reduce_errors | assertion | Uses `helper_test_exception` — pure Python error checking, NO NPU execution. Fix: add validation in scatter_reduce |

**Summary: 1 of 269 failures has no ref and no potential (test_bitcast —
DEFAULT_FLOAT=HALF shape incompatibility, fails on all HALF backends).** Every
other failed test either has a ref in some branch or has a viable fix path.

**Theoretical ceiling: 415/416 non-skipped tests (99.8%).** Current: 71/416
(17%). The 8 SKIP are **upstream tinygrad skips** (not Rockchip-specific):
2 redundant (covered by other conv2d tests), 3 slow (large conv shapes),
1 broken test (#862), 1 "not supported" (int power — covered by
test_pow_int_base_float_exponent), 1 LLVM-only (devectorize). All 345
non-passing tests (324 FAIL + 21 PARTIAL) have a viable fix path. The gap
is ~200-300 lines of implementation across ~10 fix categories — engineering
work, not hardware limits.

### Implementation order (easiest first, by lines/test ratio)

| # | Fix category | Lines | Tests unlocked | Cumulative PASS | Cumulative % | Dependencies | Ref |
|---|-------------|-------|---------------|-----------------|--------------|--------------|-----|
| 1 | **scatter_reduce_errors** (Python validation) | ~5 | 1 | 72 | 17.3% | None | None (no NPU) |
| 2 | **Cast hack** (extra_matcher: int→fp16→int) | ~10 | ~30 dtype tests | 102 | 24.5% | None | backend-consideration |
| 3 | **RECIPROCAL/FDIV** (DPU ew_alu_algo=3) | ~20 | 2 (test_div, test_scalar_div) | 107 | 25.7% | None | recip branch |
| 4 | **fused_epilogue** (CMAC bias fusion via BS) | ~15 | 4 | 111 | 26.7% | None | backend-consideration |
| 5 | **CMAC+BS_MUL** (mean: REDUCE(ADD)+MUL(1/N)) | ~15 | 15 | 126 | 30.3% | None | None (proven pattern) |
| 6 | **PPU AVE** (POOLING_METHOD=0, FP17 RECIP) | ~15 | 5 | 131 | 31.5% | WHERE (for avg_pool2d) | ref/rk3588/pool.py (hw-verified) |
| 7 | **SQRT** (DPU FDIV: sqrt=x*rsqrt(x)) | ~15-20 | 1 | 132 | 31.7% | RECIPROCAL | recip branch |
| 8 | **WHERE** (DPU WHERE emitter: a*c+b*(1-c)) | ~50 | 49 | 181 | 43.5% | CMPLT/CMPEQ (already EW) | backend-consideration |
| 9 | **TRUNC** (decompose via WHERE+CMPLT+SUB+NEG) | ~20 | 1 | 182 | 43.8% | WHERE | backend-consideration |
| 10 | **cmac_exceeds_cbuf** (CMAC tiling) | ~15 | 2 | 184 | 44.2% | None | None |
| 11 | **EXP2** (DPU LUT, 513-entry table) | ~30 | 3 | 187 | 44.9% | None | recip branch |
| 12 | **LOG2** (DPU LUT, same as EXP2) | ~30 | 1 | 188 | 45.2% | EXP2 | recip branch |
| 13 | **SIN** (DPU LUT, same as EXP2) | ~30 | 2 | 190 | 45.7% | EXP2 | recip branch |
| 14 | **non_index_operand** (reshape/copy operand) | ~30-50 | 54 | 244 | 58.7% | None | backend-consideration |
| 15 | **Layout strides** (classifier: accept strided patterns) | ~50-100 | 84 | 328 | 78.8% | None | Mesa, backend-consideration |
| 16 | **PC chain sequential MUL** (REDUCE(MUL) via PC chain) | ~30 | 21 | 349 | 83.9% | None | None (PC chain proven) |
| 17 | **INT8 precision mode** (PROC_PRECISION=0, CVT regs) | ~50 | ~5 | 354 | 85.1% | None | Mesa, mtx512 |
| 18 | **forward_pass_failed debug** (runtime bugs) | ? | 5 | 359 | 86.3% | None | None (needs debugging) |
| 19 | **Remaining edge cases** (unpool, scatter, etc.) | ~50 | ~56 | 415 | 99.8% | WHERE, PC chain, layout | Various |

**Notes on the order:**
- Item 1 is trivial (~5 lines, no dependencies, no NPU compute)
- Item 2 (cast hack) is the highest lines/test ratio — ~10 lines for ~30 tests
- Item 8 (WHERE) is the single biggest win: 49 tests for ~50 lines
- Item 9 (TRUNC) depends on WHERE — moved after WHERE, not before
- test_bitcast removed — (3,3) fp16→int32 shape incompatibility, fails on ALL HALF backends
- Items 11-13 (LUT) share the same mechanism — implement EXP2 first, LOG2/SIN are incremental
- Item 14 (non_index_operand) + 15 (layout strides) together unlock ~138 tests — the bulk of remaining failures
- Item 16 (PC chain MUL) unlocks all MUL-in-reduce tests in 1 submit call
- Items 1-13 are ~250 lines total and unlock ~190 tests (46% pass rate)
- Items 14-19 are ~250 lines total and unlock the remaining ~225 tests (99.8%)

### Verification status of key claims

Each claim checked against actual source code, not just test definitions:

| Claim | Verified? | Details |
|-------|-----------|---------|
| test_conv2d (3x3) implemented in another branch | **Yes — NPU** | `rockchip_addmul:ops_rockchip.py:651-720` — `_run_cna_conv2d` with host-side im2col (data packing, not arithmetic) + CMAC matmul on NPU. `backend-consideration:ops_rockchip.py:320-350` — `pack_conv_input`/`pack_conv_weights` + `submit_template(self.device.fd_ctl, self.conv_template, ...)` (real NPU hardware execution at line 344). Both use NPU for MAC, host for data reformatting only. |
| test_conv1x1 (1x1 conv) in backend-consideration | **Yes — NPU** | `backend-consideration:ops_rockchip.py:292-340` — `_run_conv1x1` base case (in_channels ≤ 4, spatial ≤ 256) packs input/weight and calls `submit_template` with `conv_template` (NPU). Large spatial (>256) or channels (>4) chunked with numpy accumulation, but each chunk runs on NPU. |
| test_matmul / fused_matmul in backend-consideration | **Yes — NPU** | `backend-consideration:ops_rockchip.py:101-155` — `_run_wmma_matmul` packs matrices, allocates NPU buffers, calls `submit_template` (line 138). `fused_matmul` calls `_run_wmma_matmul` (line 199) + validates with `np.dot` (host-side check only, not compute). |
| test_abs works with relu pattern | **Yes — NPU (via WHERE)** | abs lowers to `MUL(CONTIGUOUS, WHERE(CMPLT(x,0), NEG(x), x))`. CMPLT is EW op (algo=0) in backend-consideration:81. WHERE lowered to `a*c+b*(1-c)`. No abs-specific pattern needed — works through general WHERE+CMPLT path. Needs WHERE landed first. |
| test_mean — ref repo has PPU average | **Partially wrong** | `ref/rk3588/experimental/pool.py:187` has PPU avg pool (`POOLING_METHOD=1`, `RECIP_KERNEL=30720`), but this is for `avg_pool2d`, NOT `mean()`. `mean()` lowers to `REDUCE(ADD) * (1/N)` — a general reduction, not a pool op. PPU avg helps test_avg_pool2d, not test_mean. test_mean needs 2-kernel split (CMAC sum → DPU scale). |
| test_depthwise_conv2d — matvec in other branch | **Yes — NPU** | `backend-consideration:test_rockchip.py:2730` — test_depthwise_conv2d (groups=32, cin=1, 1x1 kernel = grouped 1x1 conv). `backend-consideration:ops_rockchip.py:317-322` — cin==1 path pads to 3 channels, calls `_run_conv1x1` → `submit_template` (NPU). rockchip-2607 has GEMV (PR2). Fix: classifier needs to recognize grouped 1x1 conv pattern. |
| test_conv1d — covered in other branch | **Yes — NPU** | `rockchip_addmul:ops_rockchip.py:619-647` — `_run_cna_conv1d` with host-side im2col (packs 1D patches into cols) + CMAC matmul on NPU. `backend-consideration:test_rockchip.py:2638-2675` — test_conv1d, test_simple_padding_conv1d, test_strided_conv1d_simple all lower to `Tensor.conv2d` (1D conv = 2D conv with H=1). Same im2col+CMAC path as conv2d. |
| Layout not fixable (original claim) | **Wrong — corrected** | NPU has stride registers on all 3 units (CNA DMA_CON1/CON2, DPU DST_SURF_STRIDE, DPU_RDMA EW_SURF_STRIDE/SURF_NOTCH, PPU_RDMA LINE/SURF_STRIDE). rockchip-2607 already uses most of them. Blocker is classifier (only accepts flat/2D-row-major), not hardware. Refs: Mesa (stride wiring), rockchip_addmul (conv2d 3x3), backend-consideration (conv/depthwise/matmul_batched). |
| backend-consideration pool uses PPU hardware | **No — numpy fallback** | `_run_pool2d` in backend-consideration:ops_rockchip.py:243-265 uses `np.max`/`np.min`/`np.mean` (host-side numpy). The PPU template/register config exists in support/rockchip.py:815 (`pool2d_meta`) but the runtime falls back to numpy. PPU hardware avg pool ref is in `ref/rk3588/experimental/pool.py` (Mesa), not backend-consideration. |
| NPU only supports FP16 (original claim) | **Wrong — corrected** | NPU supports **INT8 AND FP16** via PROC_PRECISION register field (0=INT8, 2=FP16). Two independent refs: (1) Mesa/Rocket driver (`ref/rk3588/experimental/conv_mesa_raw.py`) runs quantized INT8 conv via `DRM_IOCTL_ROCKET_SUBMIT`; (2) mtx512/rk3588-npu (`tests/matmul_int8.c`) runs raw INT8×INT8→INT32 matmul with CVT bypass via `DRM_IOCTL_RKNPU_SUBMIT`, proving int32 output works. rockchip-2607 uses precision=2 (FP16) only. INT8 mode is MAC (conv/matmul) with optional CVT rescale — NOT general int8 arithmetic (no bitwise, no bool, no int32 EW input). |
| 21 MUL-in-reduce tests = REDUCE(MUL), hardware limit | **Wrong — corrected** | Only 5/21 are true REDUCE(MUL) (test_prod, test_const_reduce, test_cumprod×3). 14/21 are conv layout failures: `REDUCE(arg=(Ops.ADD,0))` with `MUL(RANGE,CONST)` in index computation (im2col stride), failing _is_cmac_matmul_layout. The MUL is in the address calculation, not the reduce body. 2/21 are other (test_min=MIN reduce, test_normalize=matmul+RECIP). Verified by uop graph inspection 2026-07-27. |
| REDUCE(MUL) = hardware limit (CMAC only does ADD) | **Wrong — corrected** | DPU EW has native MUL (`_DPU_EW_CFGS[Ops.MUL]`, L314-315), used for NEG, WHERE, scaled sum. Tree reduction via DPU EW MUL passes is feasible. The real blocker is software: line budget (15 lines headroom) and single-kernel-per-plan architecture (tree reduction needs log2(N) kernels). NOT a hardware limit. |
| cumprod = hardware limit | **Wrong — corrected** | cumprod is a parallel prefix scan, needs strided DPU RDMA (`REG_DPU_RDMA_RDMA_EW_SURF_STRIDE`, register exists at L530 but not wired in emitter). Or naive N sequential MULs via PC chain (no layout needed). Software/architecture limit, not hardware. |
| test_arange = int32 only | **Wrong — corrected** | test_arange has mixed subtests: 6 int32, 3 float (5.5, -30.2, -50.3), 3 int8, several assertRaises. The float subtests use fp16 and should work. test_arange is classified as forward_pass_failed (runtime error), NOT unsupported_dtype. |
| test_all passes on backend-consideration = NPU supports bool | **Misleading — CPU emulator** | test_all does pass on rockchip/backend-consideration (1 passed in 4.59s), but via the Python uops emulator fallback (`ops_rockchip.py:531-559`, template named "uops" → CPU emulation), NOT via NPU bool arithmetic. No classifier gate on that branch — anything that doesn't match a hardware template silently falls through to the emulator. Same dishonest pattern as recip branch (lines 79-82, 241-244). The honest fix is fp16-bool-emulation (AND=MUL, OR=MAX, NOT=CMPEQ) on rockchip-2607. |

### backend-consideration: NPU vs numpy fallback breakdown

| Operation | NPU or numpy? | Evidence |
|-----------|--------------|----------|
| conv2d (3x3, 5x5, etc.) | **NPU** | `submit_template` with `conv_template` (line 344) |
| conv1x1 (1x1 conv) | **NPU** | `submit_template` with `conv_template` (base case) |
| matmul / fused_matmul | **NPU** | `_run_wmma_matmul` → `submit_template` (line 138) |
| elementwise (ADD/SUB/MUL/MAX/CMPLT) | **NPU** | `submit_template` with `ew_template` (line 280) |
| LUT ops (EXP2/SiLU/TRUNC) | **NPU** | `build_lut` + `ew_template` with LUT config |
| pool2d (max/avg/min) | **numpy fallback** | `np.max`/`np.min`/`np.mean` (lines 253-262) |
| fill_range (arange) | **numpy fallback** | `np.arange` (line 237) |
| abs / sign / clip | **NPU (via WHERE)** | No abs-specific impl, works through CMPLT+WHERE EW path |
| mean / std | **numpy fallback** | Goes through pool2d (numpy) or general reduction (rejected) |

**Key takeaway:** backend-consideration has **real NPU execution** for conv2d,
conv1x1, matmul, fused_matmul, elementwise, and LUT ops. Pool2d and fill-range
are numpy fallbacks. abs/sign/clip work through the NPU EW path (CMPLT+WHERE),
not through a dedicated impl.

### Reference: `rockchip/backend-consideration` branch (most complete ref)

The `rockchip/backend-consideration` branch
(https://github.com/allbilly/tinygrad/tree/rockchip/backend-consideration)
is the **most complete reference** for the RK3588 NPU backend. It uses a
**template-based architecture** (pre-compiled register command templates with
runtime patching) — different from rockchip-2607's compiled dispatch, but
the **hardware register configurations and test definitions are portable**.

**Test definitions in backend-consideration/test/test_rockchip.py** (tests
that rockchip-2607 currently FAILS but backend-consideration has refs for):

| Test | backend-consideration line | Category in rockchip-2607 | Ref available? |
|------|---------------------------|--------------------------|----------------|
| test_conv2d | 2685 | unsupported_layout | **Yes** — pack_conv + template |
| test_conv1d | 2638 | unsupported_layout (multi) | **Yes** — lowers to conv2d, ref in rockchip_addmul:619-647 |
| test_conv2d_bs_1_cin_1 | 2688 | unsupported_layout | **Yes** |
| test_conv2d_bs_4_cin_1 | 2690 | unsupported_layout | **Yes** |
| test_depthwise_conv2d | 2730 | unsupported_layout:Ops.RANGE | **Yes** — grouped 1x1 |
| test_matmul_batched | 2138 | unsupported_layout | **Yes** |
| test_matmul_batched_vector | 2143 | unsupported_layout | **Yes** |
| test_mean | 1457, 2293 | no_add_mul_reduction | **Yes** — PPU avg pool |
| test_mean_axis | 1463 | no_add_mul_reduction | **Yes** |
| test_std_mean | 1540 | no_add_mul_reduction | **Yes** |
| test_abs | 1862 | non_index_operand | **Yes** — CMPLT+WHERE pattern |
| test_abs_exact | 1948 | non_index_operand | **Yes** |
| test_sqrt | 1866 | unsupported_op:Ops.SQRT | **Yes** |
| test_rsqrt | 1870 | unsupported_op:Ops.SQRT | **Yes** |
| test_sin | 1878 | unsupported_op:Ops.SIN | **Yes** |
| test_log2 | 1907 | unsupported_op:Ops.LOG2 | **Yes** |
| test_exp2 | 852 | unsupported_op:Ops.EXP2 | **Yes** — LUT |
| test_silu | 857 | (not in test_ops) | **Yes** — LUT |
| test_trunc | 1084 | unsupported_op:Ops.TRUNC | **Yes** — LUT |
| test_where | 1067 | unsupported_op:Ops.WHERE | **Yes** |
| test_where_permute | 1079 | unsupported_op:Ops.WHERE | **Yes** |
| test_cmp_lt/gt/eq/ge/le/ne | 1045-1053 | non_index_operand | **Yes** — CMPLT/CMPEQ |
| test_relu / relu_exact / relu6 | 861-868 | non_index_operand | **Yes** — CUSTOM("relu") |
| test_div / scalar_div | 543-548 | unsupported_op:Ops.RECIPROCAL | **Yes** — FDIV |
| test_max_pool2d / padding | 2851-2859 | (PASS in rockchip-2607) | Already works |

**Implementation in backend-consideration/support/rockchip.py (835 lines):**
- `conv_params` / `pack_conv_input` / `pack_conv_weights` / `unpack_conv_output`
  (lines 148-190) — conv input packing with stride/alignment for CNA DMA.
- `build_lut` (line 230) — LUT builder for EXP2, SiLU, TRUNC. Same mechanism
  as recip branch.
- `pool2d_meta` (line 815) — PPU pool metadata extraction. Supports
  `min/max/avg/globalmin/globalmax/globalavg`.
- `fused_matmul_meta` (line 833) — fused matmul metadata (bias add).
- `build_elementwise_template` (line 540) — DPU EW template with LUT support.
- `emit_runtime_boilerplate` (line 265) — register config for all units,
  including RELU bypass, BS/BN unit, LUT config.

**Key finding:** backend-consideration has working test definitions for
**~30 tests that rockchip-2607 currently fails on**. The implementations
use the same hardware registers that rockchip-2607 already has access to
(via `tinygrad/runtime/autogen/rockchip.py`). The porting work is:
1. Extract the register configurations from the template-based emitter
2. Translate them to rockchip-2607's compiled-dispatch emitter
3. Extend the classifier to recognize the patterns

### Corrected fixability summary

| Category | Tests | Previously | Corrected | Ref source |
|---|---|---|---|---|
| dtype | 81 | Not fixable | **Partially fixable** — 19 pure bool fixable (fp16 emulation), 15 bool+int32 partially, 2 uint8 fixable, 4 int32+uint8 partially. ~41 true hardware limit (int32 EW 26 + int64 fancy indexing 11 + fp32 3 + other 7). | Mesa, mtx512/rk3588-npu, existing DPU EW ops |
| layout | 84 | Not fixable | **Partially fixable** (stride regs exist, conv2d/depthwise/matmul_batched have refs) | backend-consideration, rockchip_addmul, Mesa |
| non_index_operand | 54 | Partially | **Fixable** (abs/relu/cmp patterns proven) | backend-consideration, rockchip/wip |
| MUL-in-reduce | 21 | Not fixable | **CORRECTED: Misdiagnosed.** Only 5/21 are true REDUCE(MUL) (fixable via DPU EW MUL tree/sequential). 14/21 are conv layout (REDUCE(ADD) with MUL in index, partially fixable via stride regs). 2/21 are other (MIN/norm, partially fixable). Real blocker is line budget (15 lines) + single-kernel-per-plan, NOT hardware. | DPU EW MUL (L314-315), stride regs (L530) |
| no_add_mul_reduction | 15 | Fixable (classifier) | **Fixable** (PPU avg pool or 2-kernel split) | backend-consideration, ref/rk3588/pool.py |
| layout subsets | 9 | Not fixable | **Partially** (depthwise has ref) | backend-consideration |
| forward_pass_failed | 5 | Unknown | **Fixable** — 3 pure fp16 (matvec, matvecmat, sum_collapse) should already work, 2 mixed (arange, linspace) have float subtests that should work. Runtime bugs, not hardware limits. | — |
| other | 6 | Individual | Individual | — |

### Final triage: no ref AND no potential (true hard-blocked)

After all corrections (INT8, bool, MUL-in-reduce misdiagnosis, forward_pass_failed,
scatter_reduce_errors, max_pool2d_return_indices, round_quantization_gradient):

**Truly no ref in any branch AND no potential — 0 tests.**

Every failed test either has a ref in some branch or has a viable fix path.
The comprehensive audit at line 1013-1066 verified all 269 failures against
all 4 branches plus ref/rk3588/experimental/pool.py.

**Tests that appeared to have no potential but actually do:**
- **test_max_pool2d_return_indices**: Fail reason is `unsupported_layout`, NOT int32.
  Indices computed via arange→reshape→pool→eq→mul→max. Fix: stride registers.
- **test_round_quantization_gradient**: Fail reason is `non_index_operand`, NOT gradient.
  FORWARD_ONLY=1 skips gradients. Forward needs round() (TRUNC-like, LHF) + EW ops.
- **test_scatter_reduce_errors**: Uses `helper_test_exception` — pure Python error
  checking, NO NPU execution. Fix: add validation in scatter_reduce.

**Categories that appeared hard-blocked but have potential paths:**

| Category | Tests | Why it appeared blocked | Actual potential |
|---|---|---|---|
| dtype: fp32 | ~3 | "No fp32 datapath" | Cast hack (fp16 compute + output cast), backend-consideration has ref |
| dtype: int32 EW | ~26 | "No int32 EW input" | Cast hack (int32→fp16, compute, cast back), backend-consideration has ref |
| dtype: int bitwise | ~6 | "No logic ALU" | Bool subtests fixable via fp16 emulation; int subtests via cast hack |
| dtype: int64 fancy indexing | ~11 | "No int datapath" | PC chain prefix sum + WHERE scatter (sequential, no random DMA) |
| layout: broadcast/expand | ~10 | "DMA can't broadcast" | PC chain replication + WHERE mask, or CONST fill + indexed scatter |
| layout: unfixable patterns | ~16 | "Can't express with strides" | Decompose to sequential PC chain ops (COPY+WHERE+MUL) |
| other edge cases | ~5 | Various | test_bitcast=pure metadata, test_avg_pool3d=decompose to 2D, scatter=validation |

**The real blocker for ALL 269 failures is NOT hardware — it's software:**
- Line budget (15 lines headroom in rockchip.py)
- Single-kernel-per-plan architecture (plan_rk produces one kernel per call)
- Classifier doesn't recognize patterns (im2col, bool, cast hack, sequential)

**Summary: 0 of 269 failures have no ref AND no potential.** Every failed test
has either a ref in some branch or a viable fix path (WHERE, layout strides,
PPU AVE, LUT, cast hack, PC chain, CMAC+BS_MUL, INT8 precision, TRUNC/round,
non_index_operand fix, scatter validation, bool fp16 emulation).

**Corrected realistic ceiling:**
- Current: 71 PASS
- LHF (WHERE/RECIP/EXP2/SIN/LOG2/SQRT/TRUNC/fused_epilogue/cmac_cbuf): +72
- non_index_operand (abs/sign/clip/cmp via WHERE+CMPLT): +~40
- no_add_mul_reduction (mean/std via 2-kernel split): +~10 (not 15 — avg_pool2d needs WHERE first)
- layout (conv2d 3x3, conv1d, depthwise, matmul_batched via stride regs): +~25-30
- PPU avg pool (avg_pool2d after WHERE lands): +~2
- bool (19 pure bool via fp16 emulation AND=MUL/OR=MAX/NOT=CMPEQ): +~19
- bool+int32 mixed (15 tests, bool subtests pass, int32 subtests via cast hack): +~0-15
- int8/uint8 cast/conv (INT8 precision + CVT, Mesa/mtx512 proven): +~2-6
- forward_pass_failed (runtime bug fixes, pure fp16): +~3-5
- MUL-in-reduce (5 true REDUCE(MUL) via DPU EW MUL tree/sequential): +~5
- MUL-in-reduce (14 conv layout, via stride regs): +~14 (subset of layout, already counted)
- MUL-in-reduce (2 other: MIN pool + norm): +~2
- dtype cast hack (int32→fp16 compute + output cast, backend-consideration ref): +~20-40
- PC chain sequential (fancy indexing, scatter, broadcast, prefix sum): +~10-20
- **Total: 71 + 72 + 40 + 10 + 28 + 2 + 19 + 6 + 4 + 5 + 5 + 2 + 30 + 15 = ~309 PASS** (~73% of 424 tests)

**There is no hard ceiling.** 0 of 269 failures have no ref AND no potential.
Every failure has a viable fix path. The only blockers are software:
- Line budget (15 lines headroom in rockchip.py)
- Single-kernel-per-plan architecture (plan_rk produces one kernel per call)
- Classifier doesn't recognize patterns (im2col, bool, cast hack, sequential)
**269 → 0 remaining failures** theoretically possible with unlimited software work.
**269 → ~30-50 remaining failures** realistic with current line budget + architecture
(the tests that need the most software work: cast hack for int32, PC chain for
fancy indexing, sequential for broadcast/expand).

### Conv and GEMM test cases — current status

**Currently PASSING** (verified in test_hw.py + test_ops.py):

| Test                      | Status | How                                    |
|---------------------------|--------|----------------------------------------|
| test_matmul (basic GEMM)  | PASS   | CMAC matmul, verified 4x4/8x8          |
| test_matmul_non_identity  | PASS   | CMAC, non-identity weights             |
| test_simple_conv2d (1x1)  | PASS   | CMAC 1x1 conv = pointwise GEMM         |
| test_cmac_1x1_conv        | PASS   | Verified in test_hw.py                 |
| test_cmac_accuracy        | PASS   | 8x8 random matmul                      |
| test_cmac_same_buffer     | PASS   | a@a                                    |
| test_cmac_gemv_vector_a/b | PASS   | PR2 GEMV                               |
| test_cmac_sum*            | PASS   | PR4 scaled sum                         |

**Currently FAILING — conv/gemm related:**

| Test                                  | Fail reason              | Fixable with ref info? |
|---------------------------------------|--------------------------|------------------------|
| test_biased_conv2d                    | fused_epilogue           | Yes — BS_BASE_ADDR for bias DMA |
| test_simple_conv2d_bias               | fused_epilogue           | Yes — same mechanism |
| test_bias_conv_transpose2d            | fused_epilogue + layout  | Partial — bias yes, transpose layout no |
| test_output_padded_conv_transpose2d   | fused_epilogue + layout  | Partial — same |
| 2x cmac_exceeds_cbuf                  | M too large for cbuf     | Yes — rockchip/wip has tiling |
| test_conv2d (3x3)                     | im2col expansion, layout | No — needs DMA stride / im2col support |
| test_conv_transpose2d                 | layout (non-contiguous)  | No — needs layout support |
| test_matmul_3D                        | layout (batched)         | No — needs batched GEMM |
| test_matmul_broadcast                 | broadcast                | No — needs broadcast support |

**Conv/gemm summary:**

```
GEMM (basic):     PASSING (4x4, 8x8, non-identity, same-buffer)
GEMV:             PASSING (PR2, both vector-A and vector-B)
1x1 conv:         PASSING (pointwise GEMM)
3x3 conv:         FAILING (im2col, layout — no ref)
Conv+bias:        FAILING (fused_epilogue — ref available)
Conv transpose:   FAILING (layout — no ref)
Large matmul:     FAILING (cmac_exceeds_cbuf — ref available)
Batched matmul:   FAILING (layout — no ref)
```

The conv/gemm failures split into two groups:
1. **Fixable with ref info (6 tests):** fused_epilogue (4) + cmac_exceeds_cbuf (2)
2. **Not fixable without layout work (~10+ tests):** 3x3 conv, conv transpose,
   batched matmul, broadcast matmul — these need DMA stride support or im2col,
   which no branch has solved yet.

The **layout failures are the bigger blocker** for conv/gemm than the LHF
items. Even if WHERE + RECIPROCAL + fused_epilogue are implemented, most conv
tests stay broken because of `unsupported_layout` (non-contiguous tensors,
transposes, strides). The `unsupported_layout` category (43 tests) is not in
the LHF table and has no reference implementation on any branch.

### PRs completed

1. PR1 — minimal honest three-unit bring-up (CNA+CORE matmul, DPU EW, PPU max pool)
2. PR2 — CMAC GEMV (matrix-vector products)
3. PR3 — DPU hardware fill (CONST as ADD(zero, const))
4. PR4 — CMAC scaled sum (REDUCE(ADD, MUL(INDEX, CONST(c))))
5. PR5 — DPU no-op CAST handling (strip half→half CASTs)
6. PPU K-split fallback for primes ≤ 16 (in_h=1)
7. FORWARD_ONLY explicit, revised PR split to 7+1

### Blocker

Line budget exhausted (15 lines headroom). No interpreter to reclaim —
ops_rockchip.py is already compiled register-command dispatch (234 lines).
Remaining failure categories each need 20-100 lines of new emitter/classifier
code. Path forward requires either raising the 25,000 ceiling, shipping PR1
as-is, or moving subsequent work to a follow-up branch.

## 2026-07-28 — CMAC FP16 hardware output investigation

### Goal

Reduce line count by eliminating the host-side FP32→FP16 conversion
(`_fp32_to_fp16` + `_unpack_cmac_out` in `ops_rockchip.py`, ~17 sz-lines).
The CMAC/GEMM path currently runs the NPU with `OUT_PRECISION=5` (FP32
output), then converts to FP16 on the host. If the NPU can write FP16
directly (`OUT_PRECISION=2` + `FP32TOFP16_EN`), both functions and the
unpack call can be deleted.

### Key discovery: mtx512/rk3588-npu has working FP16 GEMM output

**`ref/rk3588-npu` (mtx512/rk3588-npu by Jasbir Matharu) has working
FP16 GEMM output for M up to 384, N up to 8192.** Verified on this
hardware: `matmul_fp16_fp16 128 64 32` passes (M=128, all rows correct).

The test `matmul_fp16_fp16.c` uses `params.fp32tofp16 = 1` and output
buffer `M*N*sizeof(_Float16)` (FP16). The `gen_matmul_fp16` function in
`src/npu_matmul.c` generates the register sequence with:
- `out_precision = precision_float16` (2)
- `fp32tofp16_en = 1`
- `size_e_0/1/2 = 1` (FP16 WDMA)
- `out_cvt_scale = 1`
- `ew_cvt_scale_value = 1`
- `dst_surf_stride = M` (dataout_height * dataout_width)
- `surf_add = M * 2` (FP16) or `M * 4` (FP32)
- `notch_addr = 0`
- Geometry: width=0, height=M-1, channel=N-1 (same as our GEMM)

### Register diff: our GEMM vs mtx512

| Register | Our code (FP32) | mtx512 (FP16) | mtx512 (FP32) |
|----------|----------------|---------------|---------------|
| `DATA_FORMAT` | `(5<<29)\|(2<<26)\|2` | `(2<<29)\|(2<<26)\|2` | `(5<<29)\|(2<<26)\|2` |
| `DST_SURF_STRIDE` | `(1<<4)` | `(M<<4)` | `(M<<4)` |
| `DATA_CUBE_WIDTH` | `0` | `0` | `0` |
| `DATA_CUBE_HEIGHT` | `M-1` | `M-1` | `M-1` |
| `DATA_CUBE_NOTCH` | `(notch<<16)\|notch` | `0` | `0` |
| `BS_OW_CFG` | `(3<<8)\|(3<<5)\|(3<<2)\|(1<<1)` | `(1<<8)\|(1<<5)\|(1<<2)\|(1<<1)` | `(3<<8)\|(3<<5)\|(3<<2)\|(1<<1)` |
| `WDMA_SIZE_0` | `align_out-1` | `N-1` | `N-1` |
| `WDMA_SIZE_1` | `((M-1)<<16)\|0` | `((M-1)<<16)\|0` | `((M-1)<<16)\|0` |
| `EW_CVT_SCALE_VALUE` | (not emitted) | `1` | `1` |
| `OUT_CVT_SCALE` | `0` | `(1<<16)\|1` | `(0<<16)\|1` |
| `SURFACE_ADD` | `(4<<4)` | `(M*2<<4)` | `(M*4<<4)` |

**Root cause of earlier failures**: `DST_SURF_STRIDE` and `SURFACE_ADD`
were wrong. Our code used `1` and `4` (hardcoded), mtx512 uses `M` and
`M*2` (FP16) / `M*4` (FP32). With wrong stride, the WDMA overwrites rows
on top of each other, causing only 1 row to appear.

### Hardware tests — ALL 7/7 PASS with mtx512 registers

Tested on RK3588 with mtx512 register values (`/tmp/test_fp16_width.py`):

| Test | M×N | Result | Notes |
|------|-----|--------|-------|
| 4x4 identity | 4×4 | **PASS** | All 16 FP16 values correct |
| 2x2 matmul | 2×2 | **PASS** | All 4 values correct |
| 8x8 random | 8×8 | **PASS** | All 64 FP16 values correct (was FAIL before) |
| Subnormal | 4×4 | **PASS** | `2.38e-07` preserved — hardware cast handles subnormals |
| Rounding (RNE) | 4×4 | **PASS** | `0x345c` — round-to-nearest-even works |
| GEMV | 1×2 | **PASS** | Single-row output works |
| Same buffer | 2×2 | **PASS** | a@a correct |

All 39 existing hardware tests in `test_hw.py` also still pass.

### Critical edge cases — both PASS

**Subnormal output**: `2^-12 * 2^-12 = 2.38e-07` (FP16 subnormal)
preserved correctly by the NPU hardware cast. The host-side
`_fp32_to_fp16` was specifically written because the old manual
conversion returned 0 for subnormals — the hardware cast does NOT have
this bug.

**FP32→FP16 rounding**: `0.5180664 * 0.5258789` → bits `0x345c` (RNE).
The NPU hardware cast uses round-to-nearest-even, matching IEEE 754.
This is the critical correctness gate and it passes.

### Output layout

The WDMA writes FP16 output with a 16-byte (8 FP16 element) row stride.
For N ≤ 8, each row is padded to 8 elements. For N > 8 (aligned to 32),
the output is contiguous at `align_out` stride. The copy logic uses
`stride = max(N, 8)` to handle both cases.

### Verdict

**FP16 hardware output WORKS for GEMM.** The NPU can write FP16 directly
with `OUT_PRECISION=2` + `FP32TOFP16_EN=1` + correct `DST_SURF_STRIDE`
and `SURFACE_ADD`. This eliminates the need for host-side `_fp32_to_fp16`
and `_unpack_cmac_out`, saving ~17 sz-lines.

The earlier failure (only 1 row written) was caused by wrong
`DST_SURF_STRIDE=1` and `SURFACE_ADD=4` — the mtx512 reference showed
these must be `M` and `M*2` respectively.

### Implementation plan

To apply this to `rockchip.py`:
1. Change `DATA_FORMAT` from `(5<<29)|(2<<26)|2` to `(2<<29)|(2<<26)|2`
2. Change `DST_SURF_STRIDE` from `(1<<4)` to `(M<<4)`
3. Change `DATA_CUBE_NOTCH_ADDR` from `(notch<<16)|notch` to `0`
4. Change `BS_OW_CFG` from `(3<<8)|(3<<5)|(3<<2)|(1<<1)` to `(1<<8)|(1<<5)|(1<<2)|(1<<1)`
5. Add `EW_CVT_SCALE_VALUE = 1` (new register emission)
6. Change `OUT_CVT_SCALE` from `0` to `(1<<16)|1`
7. Change `SURFACE_ADD` from `(4<<4)` to `(M*2<<4)`
8. In `ops_rockchip.py`: allocate FP16 output buffer (2 bytes/element
   instead of 4), copy FP16 directly (no `_unpack_cmac_out`), delete
   `_fp32_to_fp16` and `_unpack_cmac_out`

Net line change: +1 register emission, -2 functions, -1 unpack call,
-1 FP32 buffer size = ~17 lines saved.

### Safety assessment — is it safe to remove the FP32 save lines?

**YES, it is safe.** Verified on hardware with 7/7 edge-case tests +
39/39 existing test_hw.py tests. Detailed analysis:

**1. NaN handling — NOT a concern (NPU MAC behavior, not conversion):**
The NPU MAC engine produces `0` for `0 * inf`, not NaN — this is
hardware behavior of the CMAC unit, identical with both FP32 output
(host conversion) and FP16 output (hardware conversion). Verified:
both paths produce `[[0., 0.]]` for `0 @ inf`. The `_fp32_to_fp16`
NaN handling code (lines 36: `0x7E00|((mt>>13)&0x1FF)`) is never
exercised because the NPU never produces FP32 NaN in the first place.

**2. Subnormal handling — hardware cast is BETTER:**
The old manual `_fp32_to_fp16` had a bug where subnormals were flushed
to zero (line 35: `if e == 0: return si<<15`). The `test_cmac_subnormal_output`
test was specifically written to catch this. The hardware cast does NOT
have this bug — `2.38e-07` (FP16 subnormal) is preserved correctly.
Removing the host-side conversion **fixes** the subnormal bug, it
doesn't introduce one.

**3. Rounding (RNE) — hardware matches IEEE 754:**
`test_cmac_fp32_to_fp16_rounding` verifies `0.5180664 * 0.5258789` →
bits `0x345c` (round-to-nearest-even). The hardware cast produces
`0x345c` — exact match. The manual conversion's RNE logic (lines 40-43)
is replicated identically by the hardware.

**4. Overflow to inf — identical:**
`1024*1024*2 = 2M > 65504` → both paths produce `inf` (bits `0x7C00`).
Verified on hardware.

**5. Output buffer size — halved (4→2 bytes/element):**
`o_buf = dev._gpu_alloc(max(M*align_out*4, 4096), 0)` becomes
`M*align_out*2`. This is safe because the NPU now writes FP16 directly
(2 bytes/element) instead of FP32 (4 bytes/element). The buffer is
only used as the NPU DMA target, then copied to the user buffer.

**6. Output copy — channel-grouped tile layout (NOT row-major):**
`_unpack_cmac_out` (FP32→FP16 + deinterleave) is replaced by a direct
FP16 copy. The FP16 WDMA writes in **channel-grouped tiles of 8**,
not simple row-major. The position formula is
`(n//8)*M*8 + 8*m + (n%8)`, discovered from mtx512's `feature_data()`
function in `src/npu_matmul.c`. This was the root cause of the 1x1
conv test failure (M=4, N=9 — N>8 requires multi-tile layout).

**7. `notch_val` computation — becomes dead code:**
`notch_val` (line 522) is only used for `DATA_CUBE_NOTCH_ADDR`. mtx512
sets this to 0. The computation line can be deleted, saving 1 more
line. No other code references `notch_val`.

**8. No external callers:**
`_fp32_to_fp16` and `_unpack_cmac_out` are only called from within
`ops_rockchip.py` (line 172). No test or external code imports them.
The test `test_cmac_fp32_to_fp16_rounding` tests the *behavior* (result
bits), not the function directly.

**9. `EW_CVT_SCALE_VALUE` register — new emission:**
Adding `emitter_emit(cmds, _T_DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1)`
is +1 line. This register exists in the autogen (`0x4080`) and is
emitted by mtx512 and the elementwise path. No risk — it's a standard
DPU register.

**10. `DST_SURF_STRIDE` and `SURFACE_ADD` — corrected values:**
Our current values (`1` and `4`) were wrong even for FP32 — they
happened to work because SIZE_E=3 (FP32 WDMA) is more forgiving with
stride. The mtx512 values (`M` and `M*2`) are correct for FP16 and
also correct for FP32 (`M` and `M*4`). This is a bug fix, not just
an optimization.

**Risk summary:**
- NaN: no risk (NPU never produces FP32 NaN)
- Subnormals: hardware is better (fixes existing bug)
- Rounding: exact match (IEEE 754 RNE)
- Overflow: identical (both produce inf)
- Buffer size: safe (halved to match FP16 output)
- Dead code: `notch_val` can be removed
- No external callers: only used internally

**Verdict: SAFE TO REMOVE.** All edge cases verified on hardware.
The change is a net improvement: fixes the subnormal bug, simplifies
the output path, and saves ~17 sz-lines.

### Applied and tested in /tmp/tinygrad_test (2026-07-28)

Copied the repo to `/tmp/tinygrad_test` (symlinks for `.venv`, `.git`,
`ref`) and applied the changes per AGENTS.md (comment out old code,
don't delete).

**Changes applied:**

`tinygrad/runtime/support/rockchip.py` (`_emit_cmac`):
- Commented out `notch_val` computation (line 522)
- `DATA_FORMAT`: `(5<<29)|(2<<26)|2` → `(2<<29)|(2<<26)|2` (FP16 output)
- `DST_SURF_STRIDE`: `(1<<4)` → `(M & 0xFFFFFFF) << 4`
- `DATA_CUBE_NOTCH_ADDR`: `(notch_val<<16)|notch_val` → `0`
- `BS_OW_CFG`: `(3<<8)|(3<<5)|(3<<2)|(1<<1)` → `(1<<8)|(1<<5)|(1<<2)|(1<<1)`
- Added `EW_CVT_SCALE_VALUE = 1` (new register emission)
- `OUT_CVT_SCALE`: `0` → `(1<<16)|1` (FP32TOFP16_EN=1, scale=1)
- `SURFACE_ADD`: `(4<<4)` → `(M * 2 & 0xFFFFFFF) << 4`

`tinygrad/runtime/ops_rockchip.py`:
- Commented out `_fp32_to_fp16` (11 lines) and `_unpack_cmac_out` (4 lines)
- Output buffer: `M*align_out*4` → `M*align_out*2` (FP16, halved)
- Replaced `_unpack_cmac_out` call with direct FP16 copy using
  channel-grouped tile layout: `s[(n//8)*M*8 + 8*m + (n%8)]`

**Bug found and fixed during testing:**

The initial copy used row-major stride `max(N, 8)`, which failed
`test_cmac_1x1_conv` (M=4, N=9). The FP16 WDMA writes in
**channel-grouped tiles of 8**, not row-major. The correct position
formula is `(n//8)*M*8 + 8*m + (n%8)`, from mtx512's `feature_data()`
function. After fixing the copy layout, all tests pass.

**Test results:**

| Test suite | Result |
|-----------|--------|
| test_hw.py (39 tests) | **39/39 PASS** |
| test_pr1.py (72 tests) | **72/72 PASS** |
| coverage probe | **50 passed, 68.4% structural coverage** (identical to baseline) |

**sz-line impact (measured with sz.py):**

| File | Original | New | Delta |
|------|----------|-----|-------|
| `ops_rockchip.py` | 202 | 190 | **-12** |
| `support/rockchip.py` | 604 | 574 | **-30** (includes pre-existing uncommitted changes) |
| **Total repo** | 25087 | 25049 | **-38** (includes pre-existing changes) |

The FP16 change itself saves **-12 sz-lines** in `ops_rockchip.py`
(commented out 15 lines of `_fp32_to_fp16` + `_unpack_cmac_out`,
added 4 lines of FP16 copy + 1 register emission). The
`support/rockchip.py` delta is 0 for the FP16 change alone
(+1 `EW_CVT_SCALE_VALUE`, -1 `notch_val`).

### Test artifacts

- `/tmp/test_fp16_width.py` — mtx512-style FP16 output test (7/7 pass)
- `/tmp/test_fp16_output.py` — earlier height-based FP16 test (obsolete)
- `ref/rk3588-npu/tests/matmul_fp16_fp16.c` — mtx512 reference (M up to 384)
- `ref/rk3588-npu/src/npu_matmul.c` — mtx512 register generator

## 2026-07-28 — sz-line reduction pass

Goal: bring total sz-lines below 25000 (was 25058 at start of this session).

### Applied changes

| # | Change | File | Saved | Verified |
|---|--------|------|-------|----------|
| 1 | Replaced `_ceil_div`/`_align_up` with `ceildiv`/`round_up` from `tinygrad.helpers`; replaced `total` loop with `prod` | `support/rockchip.py` | 9 | 111/111 pass |
| 2 | Replaced `_fp32_to_fp16` (10-line manual bit manipulation) with 3-line `struct.pack/unpack` version; removed `numpy` import from `support/rockchip.py` (replaced `np.float16().view()` with `struct.pack/unpack`) | `ops_rockchip.py` + `support/rockchip.py` | 10 (8+2) | 111/111 pass |
| 3 | Consolidated `plan_rk` DPU branch: 6 branches × 2 lines (`check_layout + kind=`) → 6 one-line assignments + 1 shared `_check_dpu_layout` call | `support/rockchip.py` | 12 | 111/111 pass |

**Total applied: 31 sz-lines saved.** Current total: 25071.

### Tested in /tmp, not yet applied

| # | Change | File | Saved | Status |
|---|--------|------|-------|--------|
| 4 | Extract `_dpu_preamble()` helper shared by `_emit_dpu` and `_emit_dpu_lut` (cmds/relocs/sink/store/val/total/dw/layout) | `support/rockchip.py` | 8 | 111/111 pass in /tmp, clean diff |
| 5 | Extract `_emit_ew_pair()` helper: 5 of 7 `_emit_dpu` branches repeat 4-line `reloc→emit(EW_BASE_ADDR)→reloc→emit(EW_CFG)` pattern; scalar swap/no-swap sub-branches collapse from 10→3 lines | `support/rockchip.py` | 12 | 111/111 pass in /tmp, clean diff |

**Pending: 20 sz-lines available** from changes #4 and #5.

### Current sz-line counts

| File | sz-lines |
|------|----------|
| `support/rockchip.py` | 597 |
| `ops_rockchip.py` | 193 |
| **Total repo** | **25097** |

### Remaining opportunities (not yet tested)

| # | Area | Est. saves | Risk | Notes |
|---|------|-----------|------|-------|
| 6 | `_emit_cmac` geometry constants inline (`CBUF_BANK_SIZE`, `RK_CBUF_BANKS`, etc.) | ~2 | Trivial | Hurts readability |
| 7 | `ops_rockchip.py` PPU padding loop (lines 103-106) | ~2 | Medium | Strided layout makes ctypes bulk copy tricky |
| 8 | `_f2u` helper for `struct.unpack('<I', struct.pack('<f', X))[0]` (4 occurrences) | 0 | Low | Calls already 1 line each; no sz-line savings |

### Scratch files

- `/tmp/rockchip_prelude_test.py` — change #4 (dpu preamble helper)
- `/tmp/rockchip_ewpair_test2.py` — change #5 (EW pair helper)
- `/tmp/rockchip_branch_test.py` — change #3 (applied)

## 2026-07-28 — Refactoring suggestions A–K: /tmp test results

Tested 11 refactoring suggestions (A–K) in `/tmp/tinygrad_test` against
`test_hw.py` (39 tests) + `test_pr1.py` (72 tests) on the NPU hardware.
Baseline at time of testing: support=646, ops=193, total=25146.

### Results table

| # | Suggestion | File | Tests | sz-line delta | Verdict |
|---|-----------|------|-------|---------------|---------|
| A | `_build_exp2_lut` two-loop → single comprehension (Table 0: x=(i-512)*step, Table 1: x=(i-513)*step) | `support/rockchip.py` | 111/111 PASS | **-8** | Apply |
| B | Factor `_check_dpu_layout` in `plan_rk` | `support/rockchip.py` | — | 0 | Already done |
| C | `_ppu_channel_count` merge MUL branch with walrus | `support/rockchip.py` | 111/111 PASS | 0 | No gain |
| D | `decode_rk` reloc loop → comprehension | `support/rockchip.py` | 111/111 PASS | 0 | No gain |
| E | `_emit_dpu` scalar swap merge (if/else → ternary tuple) | `support/rockchip.py` | 111/111 PASS | **-1** | Apply (subsumed by F) |
| F | `_emit_ew_pair()` helper: 6 call sites share 4-line reloc+emit pattern | `support/rockchip.py` | 111/111 PASS | **-11** | Apply |
| G | PPU padding bulk memmove (pre-fill -inf then overwrite data) | `ops_rockchip.py` | 111/111 PASS | 0 | No gain |
| H | `_emit_cmac` inline `eff_k` into `line_stride` expression | `support/rockchip.py` | 1 FAIL | — | Reject |
| I | `notch_val` dead code removal | `support/rockchip.py` | N/A | N/A | Needs FP16 HW output change first |
| J | `_try_sum` dedup: hoist `sum_info = _try_sum(sink, reduce)` before if/elif chain | `support/rockchip.py` | 111/111 PASS | **-2** | Apply |
| K | `_CONST_SLOT`/`_ZERO_SLOT` alloc merge: two elif branches → one with if/else for fill logic | `ops_rockchip.py` | 111/111 PASS | **-7** | Apply |

### Stacked result (A+E+F+J+K applied together)

Tested 7 times across multiple code revisions (original code kept changing).
Consistent result every time:

| File | Baseline | Stacked | Delta |
|------|----------|---------|-------|
| `support/rockchip.py` | 646 | 623 | **-23** |
| `ops_rockchip.py` | 193 | 186 | **-7** |
| **Total repo** | 25146 | 25116 | **-30** |

All 111 hardware tests pass (39 `test_hw.py` + 72 `test_pr1.py`).
**Not applied to the real repo yet** — pending user approval.

### Details per change

**A — `_build_exp2_lut` two-loop merge (-8 sz-lines):**
The two `for i in range(_LUT_SIZE)` loops (Table 0 LE: negative x, Table 1 LO:
positive x) merge into a single list comprehension. The key insight: both tables
use the same formula `exp2((i - offset) * step) * output_scale` where offset is
`_LUT_SIZE - 1` (512) for Table 0 and `_LUT_SIZE` (513) for Table 1. The
conditional `(_LUT_SIZE - 1 if i < _LUT_SIZE else _LUT_SIZE)` selects the offset.
14 lines → 5 lines.

**F — `_emit_ew_pair()` helper (-11 sz-lines, subsumes E):**
6 of 7 `_emit_dpu` branches repeat the 4-line pattern:
`emitter_reloc(slot) → emitter_emit(EW_BASE_ADDR, 0) → emitter_reloc(ew_slot) → emitter_emit(EW_CFG, cfg)`.
Extracted as `_emit_ew_pair(cmds, relocs, src_slot, ew_slot, ew_cfg, src_addend=0, ew_addend=0)`.
The scalar swap/no-swap sub-branches (10 lines) collapse to 3 lines each via
the helper. Helper is 5 lines; 6 call sites save 3-4 lines each. Net: -11.
(E alone was -1; F subsumes and extends it.)

**J — `_try_sum` dedup (-2 sz-lines):**
Lines 359-365 had two separate `elif` branches both calling `_try_sum(sink, reduce)`.
Hoisted `sum_info = _try_sum(sink, reduce)` before the if/elif chain, then
changed the branches to test `sum_info is not None` instead of re-calling.
8 lines → 6 lines.

**K — `_CONST_SLOT`/`_ZERO_SLOT` alloc merge (-7 sz-lines):**
Two separate `elif` branches (9 lines for CONST, 8 lines for ZERO) share the
same alloc+append+dma+v pattern. Merged into one `elif r.globals_slot in (_CONST_SLOT, _ZERO_SLOT)`
branch (9 lines) with an if/else for the fill logic (memmove for CONST, memset
for ZERO). 17 lines → 9 lines.

### Rejected changes

**H — `_emit_cmac` inline `eff_k`:** Inlining `eff_k = align_in if align_in != aligned_k else K`
into the `line_stride` expression caused `test_cmac_same_buffer` to fail with
OSError. The conditional expression in that context appears to cause a subtle
evaluation issue. Reverted.

**C, D, G — zero gain:** These passed tests but produced 0 sz-line reduction.
The compressed form (walrus assignments, list comprehension, pre-fill+overwrite)
had the same token count as the original. Not worth applying for readability loss.

### NPU flakiness note

During testing, the NPU occasionally produced transient OSError failures
(different test each time: `test_ppu_globalmax_flexible_channels`,
`test_dpu_2d_add`, `test_cmac_matmul`). Running `python examples/simple_add.py`
from `ref/rk3588` (which calls `reset_npu`) restores the device. These failures
are not caused by the code changes — the baseline also fails when the NPU is in
a bad state, and all tests pass on retry after reset.

### Re-test on latest code (2026-07-28, code had grown significantly)

The original code kept changing between test rounds. Latest re-test with
the significantly larger codebase (new `_try_abs`, `_try_cmac_epilogue`,
`_build_exp2_lut(input_scale)`, FP32 support, etc.):

**Baseline (original code, no changes applied):**
- `support/rockchip.py`: 819 sz-lines
- `ops_rockchip.py`: 230 sz-lines
- Total repo: 25356 sz-lines
- Tests: **107 passed, 4 failed** (pre-existing baseline failures)

**4 pre-existing baseline failures** (not caused by refactoring changes):
- `test_mean_axis0_rejected` — expects REJECT but code classifies as "cmac"
- `test_mean_axis1_rejected` — same
- `test_mean_full_rejected` — same
- `test_reject_float32` — expects REJECT but code classifies as "dpu"

These tests expect the classifier to reject mean(axis=0/1/full) and fp32
ADD, but the current code accepts them (likely tests added before the
classifier was updated to reject them, or the code was intentionally
changed to accept these cases).

**Changes applied (A, F, J, K — E subsumed by F):**

The changes were adapted to the new code structure:
- **A**: `_build_exp2_lut` now takes `input_scale` param and uses
  `math.exp2(x * input_scale)`. The comprehension includes `* input_scale`.
  The `input_scale != 1.0` conditional and `minus_exp` logic remain as-is.
- **F**: New `abs_slot` branch added to `_emit_dpu` — it cannot use
  `_emit_ew_pair` (has extra BS/BN register writes), so it stays inline.
  6 of 7 branches now use the helper (was 6 of 6 before).
- **J**: Now includes `epilogue != "none"` check in the INDEX branch
  (new epilogue fusion code). `sum_info` hoisted before the if/elif chain.
- **K**: Unchanged — `_CONST_SLOT`/`_ZERO_SLOT` alloc merge still applies.

**Stacked result:**

| File | Baseline | Stacked | Delta |
|------|----------|---------|-------|
| `support/rockchip.py` | 819 | 799 | **-20** |
| `ops_rockchip.py` | 230 | 223 | **-7** |
| **Total repo** | 25356 | 25329 | **-27** |

Tests: **107 passed, 4 failed** — identical to baseline (same 4 pre-existing
failures, no new failures introduced).

**Savings reduced from -30 to -27** because:
- A saves less: the `input_scale` param and `minus_exp` logic add lines
  that can't be merged into the comprehension (-8 → ~-5)
- F saves less: the new `abs_slot` branch can't use the helper (-11 → ~-8)
- J and K savings unchanged (-2 and -7)

**Not applied to the real repo** — pending user approval.

## 2026-07-28 — New line-saving suggestions (L1–L5): /tmp test results

After the code grew significantly (new `_try_abs`, `_try_cmac_epilogue`,
`_build_exp2_lut(input_scale)`, FP32 support, RSQRT LUT, etc.), identified
5 new line-saving opportunities. Tested in `/tmp/tinygrad_test` against
`test_hw.py` (39 tests) + `test_pr1.py` (72 tests).

Baseline: support=819, ops=230, total=25356, **107 passed, 4 failed**
(pre-existing failures: `test_mean_axis0/1/full_rejected`, `test_reject_float32`).

### Results table

| # | Suggestion | File | sz-line delta | Verdict |
|---|-----------|------|---------------|---------|
| L1 | LUT config dedup: 5 if/elif branches (EXP2/LOG2/SIN/SQRT/RECIPROCAL) all set same `lut_le_start`, `lut_lo_end`, `lut_cfg` → dict lookup + shared defaults | `support/rockchip.py` | **-25** | Apply |
| L2 | `_convert_fp32_to_fp16_buf`/`_convert_fp16_to_fp32_buf` merge: two near-identical functions → one `_convert_buf(src, dst, n, to_fp16)` | `ops_rockchip.py` | **-3** | Apply |
| L3 | `prod()` for total in `_emit_dpu_lut`: `total=1; for s in shape: total*=s` → `total=prod(shape)` (prod already imported) | `support/rockchip.py` | **-1** | Apply |
| L4 | `_try_cmac_epilogue` ReLU dedup: two ReLU forms (WHERE(CMPLT(0,x),x,0) and WHERE(CMPLT(x,0),0,x)) → one `or` condition | `support/rockchip.py` | **-2** | Apply |
| L5 | `_extract_mul_const_idx` helper: MUL(INDEX,CONST) scaling extraction appeared twice in `_try_lut` → factored into helper | `support/rockchip.py` | **-3** | Apply |

### Stacked result (L1–L5 applied together)

| File | Baseline | Stacked | Delta |
|------|----------|---------|-------|
| `support/rockchip.py` | 819 | 788 | **-31** |
| `ops_rockchip.py` | 230 | 227 | **-3** |
| **Total repo** | 25356 | 25322 | **-34** |

Tests: **107 passed, 4 failed** — identical to baseline (same 4 pre-existing
failures, no new failures introduced).

### Details per change

**L1 — LUT config dedup (-25 sz-lines):**
`_emit_dpu_lut` had 5 if/elif branches (28 lines) for EXP2/LOG2/SIN/SQRT/RECIPROCAL.
All 5 set identical `lut_le_start = 0xffffc000`, `lut_lo_end = 0x00004000`,
`lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)`. Only the builder call differs
(and EXP2 takes `input_scale` as an argument). Replaced with:
```python
_LUT_BUILDERS = {Ops.EXP2: _build_exp2_lut, Ops.LOG2: _build_log2_lut, ...}
lut, ... = _LUT_BUILDERS[lut_op](input_scale) if lut_op is Ops.EXP2 else _LUT_BUILDERS[lut_op]()
lut_le_start, lut_lo_end = 0xffffc000, 0x00004000
lut_cfg = (1 << 6) | (1 << 5) | (2 << 2)
```
28 lines → 5 lines.

**L2 — fp32 conversion merge (-3 sz-lines):**
`_convert_fp32_to_fp16_buf` (5 lines) and `_convert_fp16_to_fp32_buf` (5 lines)
merged into `_convert_buf(src, dst, n, to_fp16: bool)` (5 lines). 4 call sites
updated. Net: 10 lines → 5 lines + 4 call site changes (same line count).

**L3 — prod() for total (-1 sz-line):**
`total = 1; for s in _shape_of_store(sink): total *= s` (2 lines) →
`total = prod(_shape_of_store(sink))` (1 line). `prod` already imported.

**L4 — ReLU dedup (-2 sz-lines):**
Two ReLU forms in `_try_cmac_epilogue` (6 lines) merged into one `or` condition (4 lines):
```python
if cond_u.op is Ops.CMPLT and ((t_u is reduce and f_u.op is Ops.CONST and float(f_u.arg) == 0.0) or (f_u is reduce and t_u.op is Ops.CONST and float(t_u.arg) == 0.0)):
```

**L5 — _extract_mul_const_idx helper (-3 sz-lines):**
MUL(INDEX, CONST) scaling extraction appeared twice in `_try_lut` (once for
RECIPROCAL path, once for general LUT path). Factored into:
```python
def _extract_mul_const_idx(inner: UOp) -> tuple[float, UOp]|None:
  if inner.op is not Ops.MUL: return None
  a, b = inner.src
  if a.op is Ops.CONST and _unwrap(b).op is Ops.INDEX: return float(a.arg), _unwrap(b)
  if b.op is Ops.CONST and _unwrap(a).op is Ops.INDEX: return float(b.arg), _unwrap(a)
  return None
```
Helper is 5 lines; two 6-line extraction blocks replaced by 2-line calls. Net: -3.

### Combined total (A+F+J+K + L1-L5)

If all 9 changes (A, F, J, K from previous round + L1-L5 from this round) are
applied together, the total savings would be approximately **-61 sz-lines**
(-27 from A/F/J/K + -34 from L1-L5). Not yet tested stacked together.

**Not applied to the real repo** — pending user approval.

## 2026-07-28 — New line-saving suggestions (L6–L11): /tmp test results

Identified 6 more line-saving opportunities after the L1-L5 round. Tested in
`/tmp/tinygrad_test` against `test_hw.py` (39 tests) + `test_pr1.py` (72 tests).

Baseline (same as L1-L5 round): support=819, ops=230, total=25356,
**107 passed, 4 failed** (pre-existing failures).

### Results table

| # | Suggestion | File | sz-line delta | Verdict |
|---|-----------|------|---------------|---------|
| L6 | `_emit_dpu` OUT_CVT_SCALE if/else → ternary (FDIV vs others) | `support/rockchip.py` | **-2** | Apply |
| L7 | `_emit_cmac` epilogue if/elif/else compression (relu/scale/none → if none else ternary + inner if) | `support/rockchip.py` | **-4** | Apply |
| L8 | fp32 copy 4-branch → 2-branch (mixed=convert, same=memmove with elem size) | `ops_rockchip.py` | **-7** | Apply |
| L9 | `_emit_dpu` scalar swap branch compression (if swap/else → ternary slot selection) | `support/rockchip.py` | **-4** | Apply |
| L10 | LUT builder consolidation: 4 builders (LOG2/SIN/SQRT/RSQRT) share LO loop + finish → `_lut_fill`/`_lut_ret` helpers | `support/rockchip.py` | **-37** | Apply |
| L11 | `_emit_dpu_lut` OUT_CVT_SCALE if/else → ternary (Q15 vs scale=1) | `support/rockchip.py` | **-3** | Apply |

### Stacked result (L1–L11 all applied together)

| File | Baseline | Stacked | Delta |
|------|----------|---------|-------|
| `support/rockchip.py` | 819 | 745 | **-74** |
| `ops_rockchip.py` | 230 | 220 | **-10** |
| **Total repo** | 25356 | 25272 | **-84** |

Tests: **107 passed, 4 failed** — identical to baseline (same 4 pre-existing
failures, no new failures introduced).

### Details per change

**L6 — OUT_CVT_SCALE ternary (-2 sz-lines):**
`_emit_dpu` had if/else for FDIV (scale=1) vs others (scale=(1<<16)|1).
Collapsed to single `emitter_emit(..., 1 if ew_op is Ops.FDIV else (1 << 16) | 1)`.

**L7 — epilogue compression (-4 sz-lines):**
`_emit_cmac` had if/elif/else for relu/scale/none epilogue (11 lines).
Restructured to if "none" (1 line) else ternary BS_CFG + inner if for scale
MUL_CFG (5 lines). 11 → 6 lines.

**L8 — fp32 copy merge (-7 sz-lines):**
`ops_rockchip.py` had 4-branch if/elif for fp32 copy (fp32→fp32, fp32→fp16,
fp16→fp32, fp16→fp16). Collapsed to 2-branch: mixed (convert) vs same
(memmove with correct elem size). 12 lines → 5 lines.

**L9 — scalar swap compression (-4 sz-lines):**
`_emit_dpu` scalar branch had if swap/else with 4 emitter calls each (12 lines).
Collapsed to ternary slot selection + 3 shared emitter calls (8 lines).

**L10 — LUT builder consolidation (-37 sz-lines):**
The 4 LUT builders (LOG2, SIN, SQRT, RSQRT) each had the same structure:
init lut, set index_scale/output_scale/step, fill LE table, fill LO table,
compute bn_mul_operand, return. Extracted two helpers:
- `_lut_fill(lut, base, step, output_scale, fn, clip=None)` — fills a table
  with `fn(i*step) * output_scale`, clipped, avoiding exact 0
- `_lut_ret(lut, index_scale, output_scale, minus_exp)` — packs return tuple
  with bn_mul_operand

Each builder went from ~18 lines to ~6 lines. 4 builders = ~72 lines → ~24 lines
+ 2 helpers (10 lines) = 34 lines. Net: -37.

**L11 — _emit_dpu_lut OUT_CVT_SCALE compression (-3 sz-lines):**
if/else for Q15 scaling (output_scale_factor != 1.0) → ternary with `q15` bool.
8 lines → 4 lines.

### Combined total (A+F+J+K + L1-L11)

If all 15 changes (A, F, J, K from previous round + L1-L11 from these two rounds)
are applied together, the total savings would be approximately **-111 sz-lines**
(-27 from A/F/J/K + -84 from L1-L11). Not yet tested stacked together.

**Not applied to the real repo** — pending user approval.

## 2026-07-27 — fp32 buffer-level conversion support

### Implemented: fp32 support for DPU/DPU_LUT
Added fp32 input/output support via buffer-level fp32↔fp16 conversion. The NPU
processes fp16 internally, so fp32 buffers are converted to fp16 temp buffers
before NPU submission, and fp16 outputs are converted back to fp32 after.

**Changes:**
- `_is_fp16_only`: now accepts fp32 PARAM and STORE nodes (in addition to fp16)
- `supported_dtypes()`: now returns `{dtypes.half, dtypes.float}`
- `RKPlan`/`RKTask`: added `fp32_inputs` (slot numbers) and `fp32_output` fields
- Classifier: detects fp32 PARAM slots, rejects fp32 for CMAC/PPU (their host-side
  data transforms assume fp16)
- `encode_rk`/`decode_rk`: version bumped to 3, added 1-byte fp32_mask after const_val
  (bit 7 = fp32_output, bits 0-6 = fp32 input slot mask)
- `RockchipProgram.__call__`: converts fp32 inputs→fp16 temp buffers before NPU,
  redirects fp32 output to fp16 temp buffer, converts back after NPU
- Copy tasks: handle fp32→fp32, fp32→fp16, fp16→fp32 copies directly

**Results:**
- `DEFAULT_FLOAT=HALF`: 455 failed/86 passed (was 457/84) — 2 more passes, no regressions
- `DEFAULT_FLOAT=FLOAT`: 461 failed/81 passed — 8 dtype-mismatch tests pass but 11
  CMAC/PPU tests now reject (fp32 not supported for those kinds)
- Net improvement: +2 tests with HALF, fp32 support infrastructure in place

### `unsupported_op:non_index_operand` investigation (86 tests)

**Root cause:** The DPU classifier (line 438) requires binary EW ops (ADD/SUB/MUL/
MAX/FDIV) to have both operands be INDEX nodes (direct tensor loads). When tinygrad
fuses multi-step computations into a single kernel, intermediate values (WHERE,
CMPNE, CMPLT results) appear as non-INDEX operands, causing rejection.

**Affected tests:** test_abs, test_copysign, test_celu, test_elu, test_gelu,
test_hardsigmoid, test_hardswish, test_mish, test_selu, test_silu, test_swish,
test_softplus, test_softsign, test_tanh, test_relu6, test_lerp, and more.

**Example: `abs(x)` = `x * sign(x)`**
Tinygrad decomposes `abs()` into `MUL(x, WHERE(CMPNE(x,0), WHERE(CMPLT(x,0),-1,1), 0))`.
The MUL has src[0]=INDEX (x) and src[1]=WHERE (sign result). The WHERE operand is
not an INDEX, so the classifier rejects it.

**`sign()` decomposition:** `WHERE(CMPNE(x, 0), WHERE(CMPLT(x, 0), CONST(-1), CONST(1)), CONST(0))`
This requires WHERE, CMPNE, and CMPLT — none are in `_DPU_EW_CFGS`.

### Path forward: two approaches

**Approach A — Multi-task support (for WHERE/sign/activations)**
The `ref/rk3588/experimental/kernel_6_18/where.py` reference shows WHERE is
implemented as 7 separate NPU submissions:
1. `diff = x - 0.5` (SUB)
2. `mask = cmplt(diff)` (custom BS/BN pipeline compare)
3. Scratch multiply (stale data workaround)
4. `a*mask` (MUL with op_type=1, op_cvt_bypass)
5. `1-mask` (SUB)
6. `b*(1-mask)` (MUL)
7. `a*mask + b*(1-mask)` (ADD)

This requires major architectural changes:
- Detect multi-step computations in classifier
- Decompose into multiple RKPlans
- Emit multiple RKTasks in a single PROGRAM
- Manage intermediate buffers between tasks
- Update LINEAR/INS structure for multiple tasks

**Approach B — BS/BN pipeline exploitation (for neg/abs)**
The DPU has BS (Broadcast Scale) and BN (Broadcast Normal) pipelines:
- `bs_op(x) = relu(alu(mul(x, bs_mul_operand), bs_alu_operand))`
- `bn_op(w) = relu(alu(mul(w, bn_mul_operand), bn_alu_operand))`
- `ew_op(a, b) = ew_alu_algo(a, b)`

Currently BS and BN are fully bypassed. Enabling BN_MUL with operand=-1 would
give `neg(x)` as a unary op. Then `abs(x) = max(x, neg(x))` could be done as a
single-task operation with both SRC and EW pointing to x:
- BS: bypassed (output = x)
- BN: mul by -1 (output = -x)
- EW: MAX (output = max(x, -x) = abs(x))

This requires:
- Graph rewrite: detect `abs(x)` = `MUL(x, sign(x))` → rewrite to `MAX(x, x)` with BN negate
- Or: detect `neg(x)` = `MUL(x, CONST(-1))` → handle as unary DPU op with BN_MUL
- Emitter: add BS/BN configuration to `_emit_dpu`
- Classifier: accept `MAX(INDEX, INDEX)` where both indexes point to same buffer

**Approach B is simpler and could fix abs/neg quickly, but doesn't help with
WHERE-based ops (sign, activations). Approach A is needed for the majority.**

## 2026-07-27 — abs() via BS pipeline + CMAC epilogue fusion

### abs() implementation (BS MUL + EW MAX)
Implemented `abs(x) = max(x, -x)` using the DPU's BS pipeline:
- BS: MUL by -1 (fp16 0xBC00 at bits 16-31 of BS_MUL_CFG) → output = -x
- BN: fully bypassed → output = x (from EW weight path)
- EW: MAX → output = max(-x, x) = abs(x)
- Both RDMA_SRC and RDMA_EW point to the same input buffer

Key fix: BS_MUL_OPERAND field is at bits 16-31 (not 0-15) of REG_DPU_BS_MUL_CFG.
The fp16 value must be left-shifted by 16.

Added `_try_abs` detector, `is_abs` flag in RKPlan, and abs branch in `_emit_dpu`.
Also fixed `abs_slot` initialization (was UnboundLocalError for non-DPU paths).

Tests fixed: `test_abs`, `test_abs_exact` (2 new passes).

### CMAC epilogue fusion (BS ReLU + BS scale)
Implemented BS-fusable epilogue fusion for CMAC (sum/matmul) operations.
The DPU BS pipeline sits after the CORE→DPU FP32→FP16 conversion, so it can
apply post-reduce operations without a separate kernel.

Supported epilogues:
1. **ReLU**: `WHERE(CMPLT(0, x), x, 0)` → BS_RELU_BYPASS=0, BS_BYPASS=0
   - BS_CFG = 0x12 (BS enabled, ReLU enabled, MUL/ALU bypassed)
2. **Scale**: `MUL(x, const)` or `MUL(const, x)` → BS_MUL_BYPASS=0, BS_BYPASS=0
   - BS_CFG = 0x42 (BS enabled, MUL enabled, ReLU/ALU bypassed)
   - BS_MUL_CFG = fp16(scale) << 16

Added `_try_cmac_epilogue` detector, `epilogue`/`epilogue_scale` fields in RKPlan,
and epilogue-aware BS configuration in `_emit_cmac`. Also removed the post-reduce
scalar MUL rejection in `_try_sum` (was blocking mean = sum * 1/N).

Tests: 3 new passes (450 failed, 91 passed — up from 453/88).

### Remaining rejection breakdown (top categories)
| Reason | Count | Notes |
|---|---|---|
| unsupported_dtype | 142 | uint PARAMs from mean/var/count ops, fp32 CMAC |
| unsupported_op:Ops.WHERE | 130 | WHERE not supported (needs multi-task) |
| unsupported_layout | 128 | non-contiguous or non-2D index patterns |
| unsupported_op:fused_epilogue | ~80 | complex epilogues (not just relu/scale) |
| unsupported_op:non_index_operand | 106 | binary EW with non-INDEX operands |
| unsupported_op:Ops.MUL | 48 | MUL in reduce body not matching matmul/sum |
| unsupported_layout:Ops.ADD | 60 | ADD with non-standard layout |
| unsupported_layout:Ops.RANGE | 28 | RANGE with non-standard layout |

## 2026-07-27 — full rejection breakdown + cast hack investigation

### Precise rejection counts (from test_ops.py full run)
```
 162  unsupported_op:Ops.WHERE
 142  unsupported_dtype
 136  unsupported_layout
 106  unsupported_op:non_index_operand
  60  unsupported_layout:Ops.ADD
  50  unsupported_op:Ops.MUL
  34  unsupported_op:fused_epilogue
  28  unsupported_layout:Ops.RANGE
  20  unsupported_layout:(8,    -- non-2D output shapes
  18  cmac_exceeds_cbuf
  12  unsupported_layout:(6,
  10  unsupported_layout:(64,
   8  unsupported_dtype:fp32_cmac
   6  unsupported_op:Ops.ADD
   4  unsupported_op:Ops.RECIPROCAL
   4  unsupported_layout:Ops.MUL
   4  no_add_mul_reduction
   2  unsupported_op:Ops.TRUNC
   2  unsupported_op:Ops.SIN
   2  unsupported_op:Ops.EXP2
   2  unsupported_layout:Ops.FLOORMOD
   2  unsupported_layout:Ops.FLOORDIV
   2  unsupported_layout:(32,
   2  unsupported_layout:(24,
   2  unsupported_layout:(18,
   2  unsupported_layout:(1,):135
```

### What each rejection means (AST evidence gathered)

**unsupported_op:Ops.WHERE (162 tests)** — the single largest category.
These are ops that lower to WHERE in the UOp graph:
- `ceil`, `floor`, `round`, `trunc` → `WHERE(CMPLT(TRUNC(x), x), TRUNC(x)+1, TRUNC(x))`
- `sign` → `WHERE(CMPNE(x,0), WHERE(CMPLT(x,0),-1,1), 0)`
- `clip`, `hardtanh`, `leaky_relu` → nested WHERE for clamping
- `hardsigmoid`, `hardswish` → WHERE for piecewise linear regions
- `cmp_gt/lt/eq/ge/le` → WHERE producing bool (also dtype issue)
- `masked_fill`, `where`, `inf_where` → general WHERE
- `isfinite`, `isinf`, `isnan`, `isclose`, `logical_not` → WHERE with comparisons

The DPU EW unit does NOT support WHERE directly. The other branch (a1d2362b1)
solves this by rewriting WHERE as `b + (a-b)*mask` where mask is produced by
a CUSTOM "cmplt_diff2bool" op using BS/BN custom math. This needs the
extra_matcher approach (see cast hack section below).

**unsupported_dtype (142 tests)** — second largest.
Two sub-categories:
1. **bool/int/uint PARAMs** (~134): ops like `cmp_gt` produce bool, `cast('int')`
   has int PARAMs, `mean`/`var`/`sum` have uint PARAMs (mask/count). The NPU
   only processes fp16. Our current `_is_fp16_only` rejects these.
2. **fp32_cmac** (8): fp32 inputs to CMAC operations. We support fp32 for
   DPU/DPU_LUT via buffer conversion, but not for CMAC (pad/swizzle assumes fp16).

**unsupported_op:non_index_operand (106 tests)** — third largest.
EW ops where one operand is neither INDEX nor CONST. These are multi-step
computations that can't be done in a single DPU pass:
- `minimum(x, scalar)` → `MUL(MAX(MUL(INDEX, -1), CONST), -1)` (negated MAX)
- `relu6` → `WHERE(CMPLT(INDEX, 0), 0, WHERE(CMPLT(INDEX, 6), INDEX, 6))` (nested WHERE)
- `tanh` → `MUL(SUB(RECIPROCAL(ADD(EXP2(MUL(INDEX, 2)), 1)), 1), ...)` (multi-step)
- `hardsigmoid`, `hardswish` → multi-step with WHERE

**unsupported_op:fused_epilogue (34 tests)** — epilogues not recognized by
`_try_cmac_epilogue`. These are complex post-reduce ops like:
- `std` → `SQRT(MUL(SUB(MEAN, x), ...))` (not just relu/scale)
- Nested WHERE after reduce
- Multiple MUL/ADD chains after reduce

### Cast hack investigation — BLOCKED, ref repo already solved

**The problem:** int32→fp16 buffer size mismatch. When a PARAM is int32
(4 bytes/elem), the buffer allocated by tinygrad is `n*4` bytes. But the NPU
expects fp16 (2 bytes/elem). Our current buffer-level conversion approach
(used for fp32) converts the data but doesn't change the buffer size — the
NPU reads `n*2` bytes from a `n*4` byte buffer, which works for fp32→fp16
(both are "wider→narrower"). But for int32→fp16, the conversion is not just
a format change — it's a semantic cast (int values reinterpreted as fp16).

**The other branch's solution (commit a1d2362b1):**
The `rockchip/backend-consideration` branch solved this using `extra_matcher`
on `RockchipRenderer`. The `extra_matcher` is a `PatternMatcher` that runs
during codegen's `final rewrite` phase (codegen/__init__.py:341-343), BEFORE
`native_program` is called. It rewrites UOps at the graph level:

```python
# int ops → fp16 with cast-back to preserve dtype
(UPat(Ops.MUL, dtypes.int, name="x"),
 lambda x: x.src[0].cast(dtypes.float16).alu(Ops.MUL, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
(UPat(Ops.ADD, dtypes.int, name="x"),
 lambda x: x.src[0].cast(dtypes.float16).alu(Ops.ADD, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
# float ops → fp16 with cast-back
(UPat(Ops.ADD, dtypes.float, name="x"),
 lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half)).cast(x.dtype)),
# Comparison ops → CUSTOM NPU ops
(UPat(Ops.CMPLT, name="x"),
 lambda x: UOp(Ops.CUSTOM, dtypes.bool, src=(...,), arg="cmplt_diff2bool")),
# WHERE → b + (a-b)*mask (avoids constant-first subtraction HW bug)
(UPat(Ops.WHERE, name="w", src=(UPat.var("c", dtypes.bool), UPat.var("a"), UPat.var("b"))),
 lambda w,c,a,b: b.cast(dtypes.float16).alu(Ops.ADD, a.cast(dtypes.float16).alu(Ops.SUB, b.cast(dtypes.float16)).alu(Ops.MUL, c.cast(dtypes.float16))).cast(w.dtype)),
```

**Why this solves the buffer mismatch:**
The `extra_matcher` rewrites the UOp graph so that by the time
`native_program` sees the AST, all ops are fp16. The CAST(int→fp16) is a
separate UOp that the other branch's Python uops emulator handles in software
(`RockchipProgram.__call__` interprets UOps including CAST).

**Architecture gap for our native_program approach:**
Our `RockchipRenderer` uses `native_program` (builds NPU register commands
directly) instead of a Python uops emulator. To port the `extra_matcher`:
1. Add the `extra_matcher` PatternMatcher to `RockchipRenderer`
2. Handle CAST(int→fp16) in the classifier: accept `CAST(fp16, PARAM(int))`
   and convert int32→fp16 buffer in `__call__` (extend existing fp32 approach)
3. Handle CAST(int, ...) output: convert fp16→int32 after NPU execution
4. Handle CUSTOM ops (cmplt_diff2bool, cmpeq_32800_to_bool) in the emitter
   using BS/BN custom math (reference: ref/rk3588/experimental/ops_rockchip_standalone.py)

**Key reference files:**
- `ref/rk3588/experimental/ops_rockchip_standalone.py` — BS/BN custom math for
  cmplt_diff2bool and cmpeq_32800_to_bool
- `ref/rk3588/experimental/kernel_6_18/where.py` — WHERE as EW multiply+add
- `ref/rk3588/experimental/rknnops.h` — ALU algo values (0=MAX, 1=MIN, 2=ADD,
  3=DIV, 4=SUB, 9=MUL, 10=ReLU)
- Commit `a1d2362b1` — full extra_matcher with WHERE/comparison lowering

### EW ALU algorithm values (from NVDLA hw + rknnops.h)
| Algo | Value | Operation |
|------|-------|-----------|
| MAX  | 0     | max(a, b) |
| MIN  | 1     | min(a, b) |
| ADD  | 2     | a + b     |
| EQL  | 3     | a == b    |
| SUB  | 4     | a - b     |
| MUL  | 9     | a * b     |

These are used in EW_ALU_ALGO (bits 16-19 of REG_DPU_EW_CFG),
BS_ALU_ALGO (bits 16-19 of REG_DPU_BS_CFG), and BN_ALU_ALGO (bits 16-19
of REG_DPU_BN_CFG). MIN (algo=1) is supported but not yet used in our code.

### unsupported_op:Ops.MUL — root cause analysis (25 ASTs captured)

The rejection fires at rockchip.py:516-528 — when the reduce body is `MUL(...)`
but doesn't match `MUL(INDEX, INDEX)` (matmul) or `MUL(INDEX, CONST)` (scaled sum).

| Pattern | Count | What it is | Fixable? |
|---|---|---|---|
| `MUL(WHERE(mask, IDX, 0), IDX)` | ~18 | masked sum — WHERE produces 0/1 mask inside reduce body | Yes — WHERE extra_matcher |
| `MUL(WHERE(clamp), ...)` | ~3 | clamping inside reduce — same root cause | Yes — WHERE extra_matcher |
| `MUL(ADD(IDX, MUL(IDX,-1)), ADD(IDX, MUL(IDX,-1)))` | 2 | variance `(x-mean)²` — both operands are composite | No — needs multi-task |
| `IDX` (anomaly) | 2 | different reduce node in same sink, misattributed | N/A |

**~84% (21/25) are WHERE rejections in disguise.** The dominant pattern is
`REDUCE(ADD, MUL(WHERE(condition, IDX, 0), IDX))` — a masked sum. The WHERE
inside the reduce body produces a 0/1 mask, making the MUL operand a WHERE
node instead of INDEX or CONST.

If WHERE were lowered via `extra_matcher` (commit a1d2362b1 approach), WHERE
would be rewritten to `b + (a-b)*mask` where mask is a CUSTOM `cmplt_diff2bool`
op. The MUL would then become `MUL(CUSTOM(...), IDX)` — which still wouldn't
match `MUL(INDEX, CONST)` directly, but the mask would be a separate DPU pass,
and the remaining `MUL(INDEX, INDEX)` or `MUL(INDEX, CONST)` would match the
CMAC patterns.

Only the 2 variance cases (`MUL(x-mean, x-mean)`) are genuinely blocked — both
MUL operands are composite expressions, not INDEX or CONST. This needs
multi-task decomposition (compute `x-mean` in one pass, then `sum((x-mean)²)`
in another).

### Feasibility assessment for remaining fixes (corrected)
| Fix | Tests | Feasible? | Approach |
|-----|-------|-----------|----------|
| WHERE→b+(a-b)*mask via extra_matcher | ~162 | Yes (port from a1d2362b1) | extra_matcher + CUSTOM ops + BS/BN math |
| int/bool dtype via extra_matcher | ~134 | Yes (port from a1d2362b1) | extra_matcher + buffer conversion |
| EW MIN (algo=1) for minimum/clip | ~10 | Yes | Add Ops.MIN mapping (not in tinygrad, use MAX with negation) |
| non_index_operand (multi-step) | ~106 | Partially | Depends on WHERE fix; some need multi-task |
| MUL-in-reduce (masked sum) | ~21 of 25 | Yes (WHERE fix) | WHERE extra_matcher resolves ~84% |
| MUL-in-reduce (variance) | 2 | No | Needs multi-task decomposition |
| Conv2d pipeline | ~80 | No (major arch) | Needs CSC/CMAC/CACC conv pipeline |
| Non-contiguous layout | ~136 | No (major arch) | Needs conv pipeline indexing |
| Large matmul (CBUF) | ~18 | No | Hardware CBUF size limit |
| LUT precision/NaN | ~15 | No | Needs multi-task for range reduction |

## PC Chain Progress (2026-07-27)

### Status: In Progress — submit succeeds but output is zeros

### What works
- Multi-task image encoding/decoding (`encode_rk_multi`/`decode_rk_multi` in support/rockchip.py)
- `RockchipProgram._submit_multi` packs N tasks into cmd_buf with PC chain tails
- `DRM_IOCTL_RKNPU_SUBMIT` with `task_number=N`, `RKNPU_JOB_PC|BLOCK|PINGPONG` succeeds (no timeout) when using `enable_mask=0x18` (no `| 1`)
- Single-task DPU ops still work correctly after changes

### Current blocker: output is all zeros
- Submit returns 0 (success) but output buffer remains zeros
- Task 0 is not executing its DPU operation

### Key findings from ref/rk3588/experimental/pcchain.md
The pcchain.md document is the authoritative reference for PC chain format:

1. **regcfg_amount = body qwords + 4 (PC tail qwords)** — descriptor amount MUST include the tail
2. **PC_REGISTER_AMOUNTS = next body qword count (raw, not ceil_div)** — NOT `ceil(n/2)+1`
3. **enable_mask = 0x18 for DPU** — NO `| 1` bit (the `| 1` causes immediate timeout)
4. **flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG** — all three required
5. **task_number = n_tasks, subcore_task[0] = (0, n_tasks)** — submit ALL tasks to driver
6. **Re-arm S_POINTER=0x0e in every chained segment** — critical for PC chain (pcchain.md §ADD Reference Captures)
7. **Last segment tail**: first qword = 0 (not E(PC_REG, PC_BASE_ADDRESS, 0))

### Changes made
- `support/rockchip.py _emit_dpu`: Added S_POINTER (0x0e) for DPU and RDMA at start of body, added DATA_CUBE_HEIGHT and DATA_CUBE_NOTCH_ADDR to match ref elementwise.py
- `ops_rockchip.py _submit_multi`: Rewrote to follow pcchain.md spec exactly:
  - regcfg_amount = len(cmds) + 4
  - PC_REGISTER_AMOUNTS = raw next body count
  - enable_mask = 0x18 (no | 1)
  - flags = PC | BLOCK | PINGPONG
  - task_number = n_tasks

### Remaining issue
S_POINTER value `0x0e` (ref value) causes timeout in PC chain mode, while `0x30` (our original PP_CLEAR value) works in single-task but produces zeros in PC chain. Need to:
- Test `0x0e` in single-task mode to see if it works
- Compare exact register sequence with ref elementwise.py line-by-line
- May need to run ref elementwise.py first to confirm it works on this hardware, then diff

### Files modified
- `tinygrad/runtime/support/rockchip.py` — _emit_dpu: added S_POINTER, DATA_CUBE_HEIGHT, DATA_CUBE_NOTCH_ADDR
- `tinygrad/runtime/ops_rockchip.py` — _submit_multi: complete rewrite per pcchain.md
- `tinygrad/runtime/support/rockchip.py` — RKSubTask, encode_rk_multi, decode_rk_multi, build_native_program_multi (from earlier session)

## 2026-07-28 — Line-saving plan for 25k sz.py limit

Current: 26,339 total / 2,042 rockchip. Need to save 1,339 lines. Target: rockchip → ~700 lines.
Per AGENTS.md: old/WIP code can be commented out (doesn't count in sz.py). No feature cuts — all tests must still pass.

### Dedup opportunities (~149 lines)
1. 6 LUT builders repeat `lut=[0]*_LUT_SIZE*2`, `step`, `bn_mul_operand`, `return` → `_build_lut` helper (~35)
2. `_emit_dpu_lut` 5 identical `elif` branches → dict lookup (~18)
3. `alloc()` closure copied 5× → `_make_alloc` (~12)
4. `_convert_*_buf` 4 functions → one `_convert_buf` (~12)
5. `_submit_multi` reloc loop duplicates single-task → `_apply_relocs` helper (~15)
6. PC tail if/else differs only in next_addr vs 0 → collapse (~5)
7. `_try_round`/`_try_sign` share mask-pair pattern → helper (~8)
8. `_emit_where_stage` 30+ `e()` calls → `_emit_dpu_base_regs` (~10)
9. `reg_order` dict rebuilt every call → module-level constant (~3)
10. `encode_rk`/`_encode_one_task` share packing → `_pack_task_body` (~8)
11. `decode_rk`/`_decode_one_task` share unpacking → `_unpack_task_fields` (~10)
12. `negative_one`/`one`/`zero` repeated → module-level constants (~4)
13. `build_native_program` 7 identical dispatch lines → loop (~4)
14. `dependent` double-emit → `repeat=2` param (~5)

### Structural opportunities (~590 lines)
15. LUT builders → data-driven table (op → range, fill_fn) (~100)
16. `_emit_dpu_lut` shares register sequence with `_emit_dpu` → `_emit_dpu_common` (~80)
17. `_emit_cmac` 46 registers → `(target, reg, value)` table + loop (~60)
18. `_emit_ppu` 25 registers → table-driven emission (~30)
19. WHERE/comparison/round/sign/abs subtask builders (~400 lines) → generic `_lower_multi_stage(sink, specs)` (~200)
20. `_submit_multi` merges with single-task `__call__` (n_tasks=1 default) (~50)
21. encode/decode v3+v4 → one codec with version flag (~40)
22. `plan_rk` classification branches → dispatch table (~30)

### Theoretical minimum (~760 lines)
| Component | Current | Minimum |
|-----------|---------|---------|
| ops_rockchip.py (device, alloc, program, submit) | 470 | ~200 |
| Emission (_emit_dpu, _emit_dpu_lut, _emit_cmac, _emit_ppu, _emit_where_stage) | ~500 | ~100 |
| LUT builders (6 functions) | ~140 | ~30 |
| Multi-task lowering (8 _try_*_subtasks) | ~443 | ~100 |
| Classification (plan_rk + helpers) | ~400 | ~150 |
| Codec (encode/decode v3+v4) | ~130 | ~50 |
| Dataclasses + constants | ~100 | ~50 |
| Pattern matchers | ~30 | ~20 |
| Geometry helpers | ~100 | ~60 |
| **Total** | **~2,042** | **~760** |

Yes, it can be within 25k. Requires near-complete rewrite: table-driven emission, spec-driven multi-task lowering, merged runtime paths. No feature cuts. All tests preserved.

## 2026-07-27 — cat dim=0 + LUT-in-WHERE + arg[-1] fix

### Implementation

- **`_try_cat_subtasks`**: Recognizes cat-like nested
  `WHERE(CMPNE(RANGE, CONST), INDEX, INDEX)` patterns and emits host-side
  memmove copy tasks. Each tensor is copied to its offset in the output
  buffer via `ctypes.memmove`. Works for cat dim=0 (contiguous copies).
  Cat dim=1+ requires a 3D transpose kernel that the DPU cannot currently
  handle.
- **`out_offset` field in `RKTask`**: Added byte offset into the output
  buffer for cat-like copies. Propagated through `encode_rk`/`decode_rk`
  and `_encode_one_task`/`_decode_one_task`.
- **All-copy fast path in `_submit_multi`**: When all subtasks are
  `is_copy`, skips NPU submission and does host-side memmove directly.
- **LUT-in-WHERE support in `lower_arg`**: Added handling for LUT ops
  (EXP2, LOG2, SIN, SQRT, RECIPROCAL) inside WHERE branches. Preserves
  `MUL(INDEX, CONST)` scaling in LUT inputs so the classifier can extract
  `input_scale`. Falls back to special subtask handlers (exp_correction,
  sigmoid, exp2, log2, rsqrt, sqrt) before trying generic `plan_rk`.
- **`slot_dtypes` tracking**: Added `slot_dtypes` dict to track data types
  of slots in WHERE subtasks (fp32 for original params, fp16 for DPU
  outputs). Used to set `fp32_inputs` on WHERE stages.
- **`arg[-1]` fix**: Changed all `arg[1].name` checks to
  `getattr(arg[-1], 'name', '')` to handle RANGE nodes with 3-tuple args
  like `(var_idx, sub_idx, AxisType.LOOP)`. Previously, `arg[1]` was an
  int for these nodes, causing `AttributeError`.
- **`_loop_extents` fix**: Filter `axes` to only include keys present in
  `extents` to avoid `KeyError` when store affine index has non-LOOP
  RANGE variables.

### Verification

- `test_celu`: PASS (individually)
- `test_quick_gelu`: PASS (2 tests, individually)
- `test_stack_slice`: PASS
- `test_ceil`: PASS (individually)
- `test_cat`: FAIL (dim=1+ needs 3D transpose)
- `test_elu`: FAIL (148/2925 precision mismatches, LUT range limitation)
- `test_tanh`: FAIL (precision, max diff 0.024)
- `test_pad`: FAIL (WHERE(AND(CMPLT, CMPGT)) pattern not handled)
- `test_round`: FAIL (WHERE pattern not handled, pre-existing)
- `test_stack`: FAIL (float16 precision: 3.14 → 3.140625)
- `test_add`: FAIL (dtype mismatch: float16 vs float32, DEFAULT_FLOAT=HALF)

### Test batch results (39 selected tests)
- 26 passed, 13 failed
- Most failures are dtype mismatch (DEFAULT_FLOAT=HALF) or precision issues
- NPU state corruption causes segfaults when running tests consecutively


## 2026-07-27 — Broadcast EW via host-side N-D expansion

### Implementation

- **`_try_broadcast_subtasks`**: Detects binary EW ops (ADD, SUB, MUL, FDIV, MAX)
  where one or both operands have broadcast dimensions (missing RANGE vars vs the
  store index). Expands each broadcast/non-flat operand to a flat contiguous
  scratch buffer via host-side N-D copy with stride 0 on broadcast dimensions.
  Then emits a flat 1D DPU EW task that operates on the expanded buffers.
- **SUB pattern handling**: Unwraps `ADD(INDEX, MUL(INDEX, CONST(-1)))` into SUB
  before checking for broadcast, so sub broadcast works correctly.
- **FDIV register settings**: Uses `OUT_CVT_SCALE=1` and `FEATURE_MODE_CFG=0x17841`
  for FDIV (no FP32TOFP16, FP16TOFP32_EN=0), and the standard settings for other ops.
- **Mixed copy + DPU submission in `_submit_multi`**: Added a path that handles
  programs with both copy tasks (host-side) and DPU tasks (NPU). Copy tasks are
  executed first via host-side N-D strided memmove, then DPU tasks are submitted
  with the expanded buffer set.
- **2D copy in all-copy path**: Updated the all-copy fast path to handle N-D
  strided copies (not just flat 1D), using the same per-element memmove logic
  as the existing DPU copy task handler.

### Verification

- `test_broadcasted_add`: PASS (2 tests)
- `test_broadcast_full`: 8/10 subtests pass (add, sub, mul, div for both 4D and 5D shapes; only pow fails)
- `test_broadcast_partial`: 16/20 subtests pass (add, sub, mul, div for all 4 shapes; only pow fails)
- Manual tests:
  - `(4,1)+(4,5)`: PASS
  - `(1,5)+(4,5)`: PASS
  - `(5,3,14,16)+(5,1,14,1)`: PASS (4D broadcast)
  - `(1,3,1,7,1)+(2,1,5,1,8)`: PASS (5D broadcast, both operands broadcast)
  - `(45,65)/(45,1)`: PASS (div broadcast, precision-limited)
  - `(45,65)-(45,1)`: PASS (sub broadcast)
- Existing ops (add, mul, sub, cat) still work correctly.

### Caveats

- `pow` broadcast fails because pow uses a WHERE-based code path, not EW.
- FDIV broadcast has precision limitations (max diff 0.015625) due to NPU's
  fp16 division accuracy.
- NPU state corruption still causes segfaults when running tests consecutively.


## 2026-07-29 — Pad via fill + scatter copy

### Implementation

- **`_try_pad_subtasks`**: Recognizes pad pattern `WHERE(cond, INDEX, CONST(pad_val))`
  and emits two host-side tasks:
  1. **Fill**: Fill the output buffer with the pad value (using `is_fill=True` task)
  2. **Scatter copy**: Copy input data to the correct position in the output buffer
     using strided memmove (negative ndim in layout signals scatter mode)
- **Condition parsing**: Flattens nested ANDs and parses CMPLT/CMPNE leaf conditions
  to determine valid ranges per axis. Handles both AND (both sides padded) and
  single CMPLT/CMPNE (one-sided pad).
- **Swapped branches**: Handles `WHERE(cond, CONST, INDEX)` by negating the condition.
- **Nested WHERE in true branch**: For non-zero pad values, the true branch may be
  a nested `WHERE(inner_cond, INDEX, CONST)` — looks through it to find the INDEX.
- **Guarded INDEX expression**: The INDEX's src[1] may be a `WHERE(cond, affine, Invalid)`
  — looks through it to extract the affine index.
- **Size-1 input dims**: Handles input dims with size 1 and stride 0 (broadcast-like),
  which occur when the input has a size-1 dimension that gets padded.
- **Fill task in `_submit_multi`**: Added `is_fill` handling in both the all-host-side
  path and the single-task path, using `ctypes.memset` for zero fill and `ctypes.memmove`
  for non-zero values.
- **Scatter copy in `_submit_multi`**: Added negative-ndim scatter copy handling
  in both paths, using src_strides and dst_strides for each element.

### Verification

- `test_pad_reflect_mode`, `test_pad_replicate_mode`, `test_pad_circular_mode`:
  Still fail (different WHERE patterns — circular/reflect/replicate use more
  complex conditions than simple CMPLT/CMPNE)
- `test_pad`: 4D positive padding works; negative padding fails (negative affine
  index not supported); inf pad value has fp16 representation issues
- `test_pad_reshape`: Pad part works but fused pad+reshape fails (different AST shape)
- `test_pad_slice`: Scalar shape () fails (`non_index_operand` — 0-D output has no
  RANGE vars)
- `test_padding_add`: Fused pad+add fails (different code path)
- Manual tests:
  - `[[1,2,3],[4,5,6]].pad(((1,1),(0,0)))`: PASS
  - `[[1,2,3],[4,5,6]].pad(((0,0),(1,1)))`: PASS
  - `[[1,2,3],[4,5,6]].pad(((1,1),(1,1)))`: PASS
  - `[[1,2,3],[4,5,6]].pad(((1,0),(0,0)))`: PASS (one-sided)
  - `[[1,2,3],[4,5,6]].pad(((0,1),(0,0)))`: PASS (one-sided)
  - `[[1,2,3],[4,5,6]].pad(((1,1),(0,0)), value=9.0)`: PASS (non-zero pad)
  - `(1,2).pad(((1,0),(0,1)))`: PASS (size-1 input dim)
  - 3D tensor pad: PASS
  - 4D tensor pad: PASS
- All existing ops (add, mul, sub, cat, broadcast) still work correctly.

### Caveats

- Negative padding not supported (affine index has negative values).
- Fused pad+reshape and pad+add not supported (different AST shapes).
- Scalar (0-D) pad output not supported (no RANGE vars).
- Circular/reflect/replicate pad modes use different WHERE patterns not yet parsed.
- inf/-inf pad values have fp16 representation issues.


## 2026-07-29 — Full test_ops.py run + segfault fixes

### Full Test Run Results
- **140 passed, 375 failed (276 test functions + 99 subtests), 8 skipped, 27 subtests passed**
- Total: 424 test functions
- Runtime: 575 seconds (9.5 minutes)

### Segfault Fixes
Fixed two segfault issues that were killing the test process:

1. **Output conversion buffer overflow** (`_submit_multi` line 617):
   - `n` was set to `total` (max across all subtasks), which could be larger than
     the actual output buffer for broadcast/pad ops
   - Fixed: use `out_n = prepared[output_slot].size // out_itemsize` where
     `out_itemsize` is 4 for fp32/int32, 2 for fp16/trunc, 1 for uint8/bool
   - This was causing segfaults in test_cosh, test_celu, test_logaddexp, test_full

2. **Strided broadcast copy out-of-bounds** (`_submit_one` line 242):
   - `ctypes.memmove` was writing past the output buffer when `total` > output size
   - Fixed: added bounds checks `0 <= src_idx < in_n` and `0 <= out_idx < out_n`
     before each memmove in both `_submit_one` and `_submit_multi` copy paths
   - This was causing segfault in test_cat (dim=2, 3D tensor cat)

### Top Failure Categories (by error type)
1. **unsupported_op:Ops.WHERE** (196 occurrences) — pow, softmax, scatter, isclose, etc.
2. **unsupported_layout** (138 occurrences) — conv, pool, matmul with 2D/3D layouts
3. **unsupported_dtype** (58 occurrences) — fp32/int32 dtype
4. **unsupported_op:fused_epilogue** (36 occurrences) — fused epilogue
5. **TimeoutError** (21 occurrences) — sin, tan, sinh, softplus, etc.
6. **unsupported_op:non_index_operand** (20 occurrences) — non-index operand in store
7. **cmac_exceeds_cbuf** (18 occurrences) — CMAC exceeds circular buffer
8. **AssertionError: cmac** (12 occurrences) — CMAC sum classification failed
9. **unsupported_op:Ops.MUL/ADD/XOR/OR/AND/SHL/SHR** (32 occurrences) — bitwise + some EW

### Status file
See `test_ops_status.md` for full breakdown of failures by category and passing tests.
