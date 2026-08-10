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

### 8. Bounded structural expansion — complete

- Replaced repeated general-purpose UOp substitution during static RANGE execution with a cached DAG rewrite specialized for static range values.
- Added explicit iteration and expanded-node budgets before materializing a structural reduction or local accumulator.
- Avoided scanning for direct `REDUCE` nodes when a program contains none.
- Decline unsupported float `WHERE` local accumulators before expansion; the generic streaming executor remains the next milestone.
- Added a unit regression proving oversized structural programs return to the correctness oracle without constructing the expanded graph.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 20 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 9. Whole-recipe physical liveness and nonfinite selection — complete

- Expanded SQRT, EXP2, LOG2, and SIN handler recipes before physical allocation so their complete UOp dependency graph is visible to `RKContext`.
- Kept scratch ownership conservative and used the existing use-counted in-place reuse rule; a direct LOG2 recipe remains bounded at 88 physical scratch slots.
- Tagged arithmetic internal to math recipes so the accurate reduction ADD recipe is not recursively applied to polynomial evaluation.
- Made positive-mask state visible before emitting a mixed math chain, matching the DPU's stateful execution requirement.
- Added safe `WHERE` recipes for selected NaN, expanded absolute value, and `EXP2(infinity * x)` domains; `0**x` now preserves infinity, zero, one, and NaN lanes without an operation-level power lowerer.
- Generic lowering is attempted exactly once on the original Tinygrad program. The legacy oracle no longer re-enters the generic executor with an already rewritten physical recipe.
- Composite nonfinite LOG2 selectors are explicitly declined until their value-class ABI is represented generically.
- Updated the integer-power submission assertion to distinguish the generic one-image correctness contract from the legacy fusion count.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 23 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_exp2 test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_log2 test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_exp2_log2_zero_times_negative test/backend/test_rockchip.py::TestRockchipSqrtOps -x -q -n0`: 5 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipTensorPowerOps::test_pow_const test/backend/test_rockchip.py::TestRockchipTensorPowerOps::test_pow_full test/backend/test_rockchip.py::TestRockchipTensorPowerOps::test_pow_neg_inf_frac_exponent -x -q -n0`: 3 passed.

### 10. Whole-program accurate ADD expansion — complete

- Moved the accurate multi-product ADD recipe into the same pre-allocation expansion pass as math recipes.
- Prevented dynamically introduced precise reduction graphs from invalidating physical scratch lifetime assumptions.
- Retained conservative allocation for constants, gathers, and multi-use ALU values while preserving in-place reuse for single-use values.
- Restored the full convolution census after the allocator change; no convolution or matmul graph recognizer was added.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 23 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipConvOps -x -q -n0`: 42 passed, 6 skipped, 37 subtests passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py test/backend/test_rockchip.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 11. Iterative large-program preparation — complete

- Replaced recursive precise-recipe tagging with a topological UOp rewrite.
- Added topological pre-lowering for large straight-line arithmetic programs so Python recursion depth is not a semantic limit.
- Classified static UOps iteratively once per `RKContext` instead of recursively rescanning deep arithmetic dependencies.
- Applied the generic expanded-node budget after physical recipe expansion as well as before it; oversized correctness recipes decline cleanly instead of constructing an unbounded image.
- The remaining `test_dot_1d` blocker is numerical: the 65-term generic and legacy FP16 paths agree with each other but miss one near-zero Torch lane by 0.0073. A stronger generic accumulation recipe is still required.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 23 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipConvOps::test_biased_conv2d test/backend/test_rockchip.py::TestRockchipTranscendentalOps::test_log2 -x -q -n0`: 2 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py test/backend/test_rockchip.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.
