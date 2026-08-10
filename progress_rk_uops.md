# Rockchip UOp executor progress

## Architecture rules

- UOps are the semantic program. Do not add `_lower_<tensor operation>()` handlers.
- `RKValue` is the physical ABI: argument/scratch location, semantic dtype, lane count, and canonical layout.
- One UOp may emit several `RKEWOp` stages, but stages remain in one `RKImage`/PC chain where the runtime permits.
- Memory materialization is separate from numeric semantics. Host fallbacks may calculate addresses and copy raw data only.
- The old lowering catalog is a temporary correctness oracle and is used only when the generic executor declines a program.

## Milestones

### 1. Typed elementwise executor — complete

- Added canonical `RKLayout` and `RKValue` types.
- Added `RKContext`, with typed values, scratch ownership, constants, static vectors, and static gathers.
- Added true-arity handlers for FP16 and INT16 arithmetic, unary negation/reciprocal, FP16 comparisons, boolean mask algebra, and ternary `WHERE`.
- Added explicit bool-mask conversion at the output boundary.
- Routed the generic executor ahead of the legacy catalog while retaining safe fallback for unsupported graphs.
- Added host-independent unit coverage for the ABI, dependency lowering, ternary arity, static gathers, bool output conversion, and native INT16 layout.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 10 passed at this milestone.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 2. Math UOp recipes — complete

- Moved SQRT, EXP2, LOG2, and SIN recipe ownership into the generic executor.
- Each semantic math UOp now expands directly to its existing DPU arithmetic recipe and physical stages.
- Added one generic recipe test covering all four math UOps.

### 3. Structural RANGE/reduction execution — complete

- Added one static structural interpreter for `REDUCE`/`RANGE` with ADD, MAX, and MUL.
- Added the symmetric local LOAD/STORE accumulator interpreter for the same operators.
- Structural execution produces ordinary semantic UOps and reuses `RKContext`; it does not recover a tensor operation name.
- Added coverage for direct reductions and mutable local accumulators across all three operators.

### 4. Symmetric host address materialization — complete

- Added serialized `RKHostAddress` records and explicit `NATIVE`/`HOST_ADDRESS` image classification.
- Added opt-in direct dynamic gather and last-writer scatter through `ROCKCHIP_HOST_GATHER=1`.
- Dynamic gathers run before EW stages; dynamic scatters run after EW/post-gather stages.
- The runtime only reads indices, bounds-checks addresses, and copies raw lanes. It performs no numeric or reduction semantics.
- Scatter-reduce is intentionally not part of `RKHostAddress`.

### 5. First legacy deletion census — complete

- Deleted `_lower_native_int16_ew` and its operation-pattern rewrite catalog, including the sign, ReLU6, leaky-ReLU, ABS/NEG, MIN, and masked-select graph recovery rules.
- Added a canonical `BOOL_INT16` layout so INT16 comparisons, boolean compositions, and ternary `WHERE` compose without rediscovering mask representation.
- Added an INT16 `XOR -1` UOp recipe. Tinygrad's portable `~max(~x, ~y)` minimum now executes as its ordinary XOR/MAX UOps instead of being recognized as a tensor minimum.
- Made the INT16 `WHERE` recipe saturation-safe across the complete signed range and materialized static selectors in the consumer's bool layout.
- Added an explicit INT16-to-INT32 output-boundary conversion without introducing general INT32 arithmetic semantics.
- Kept legacy task-count assertions behind `ROCKCHIP_UOPS=0`; the generic correctness executor promises one submission, not an optimized stage count.
- Removed 222 renderer lines while adding 84 generic renderer lines and 39 focused unit-test lines in this milestone.
- The remaining operation-specific catalog is still the correctness oracle for UOp families not yet migrated.
- Preserve `Ops.WMMA -> CMAC` as the only planned graph-level fast path.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 14 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipInt16EWOps test/backend/test_rockchip2.py::TestRockchipInt16EWOps -x -q -n0`: 42 passed.

### 6. Accurate generic ADD/reduction execution — complete

- Added a generic three-half physical recipe for `REDUCE(ADD, MUL)` and expanded FP32 ADD/MUL expressions stored at an FP16 boundary.
- Made ordinary half `ADD` own the same recipe when its UOp dependency tree contains multiple MUL terms; no matmul or convolution shape is recognized.
- Fused an immediately consumed FP32-to-FP16 reduction boundary into the following ADD so bias participates before the final physical rounding.
- Tagged physical recipe ADDs only to prevent recursive recipe expansion.
- Added UOp use-count registration and in-place-safe scratch reuse. The large convolution scratch request fell from 1.88 GB to a bounded allocation.
- The scoped completion census is the 445 tests in `test/backend/test_rockchip.py`; `test_rockchip2.py` is not part of the requested gate.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 17 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchip::test_big_gemm -x -q -n0`: passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipConvOps -x -q -n0`: 42 passed, 6 skipped, 37 subtests passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 7. Nonfinite WHERE/MAX semantics — complete

- Added a true-arity `WHERE` recipe for one infinite constant arm using division-generated infinity, avoiding the invalid `0 * inf` intermediate.
- Made MAX materialization use finite `-65504` for statically gated negative-infinity fill lanes, avoiding the RK3588 raw-MAX NaN behavior.
- Kept these rules attached to `WHERE`, `LOAD`, and `MAX` UOps; no padding or pooling graph is recognized.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 19 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipMaxPoolOps -x -q -n0`: 13 passed, 33 subtests passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.
