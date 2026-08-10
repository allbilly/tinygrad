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

### 12. Vectorized dependent RANGE execution — complete

- Fixed generic structural execution so it enumerates only the selected reduction axes; parent output axes remain vector lanes.
- A 45-by-65 matrix-vector dot now produces one 65-term physical reduction recipe instead of expanding into 2,925 scalar products.
- Kept the rule structural: the executor distinguishes selected `RANGE` ownership and does not recognize dot, matrix-vector, or tensor shapes.
- Added a regression proving that changing the output lane count from 1 to 45 does not change the physical EW stage count.
- The previously failing near-zero lane in `TestRockchipDotOps::test_dot_1d` now meets the Torch tolerance through the generic accurate reduction path.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 24 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipDotOps::test_dot_1d -x -q -n0`: passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 13. Bounded 128-term accurate reduction — complete

- Raised the post-recipe generic graph cap from 8,192 to 16,384 nodes so the composable accurate ADD recipe can execute 128-term reductions.
- Measured the representative 128-output recipe at 8,821 UOp nodes, 7,951 EW stages, 3,871 scratch slots, and 990,976 scratch bytes.
- Kept the cap finite and added a host-independent regression for a vectorized 128-by-128 reduction.
- Updated the zero-stride reduction submission assertion: three generic UOp programs produce three submissions, while the legacy oracle retains its four-submission expectation.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 24 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipDotOps -x -q -n0`: 7 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py test/backend/test_rockchip.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 14. Canonical selected extrema neutral — complete

- Added a MAX-boundary physical rule that maps selected/gated negative-infinity padding to the canonical finite FP16 neutral, `-65504`.
- Kept standalone nonfinite `WHERE` semantics unchanged; the conversion applies only when the UOp program's result is a MAX composition and the constant is used as a `LOAD` default or `WHERE` arm.
- Removed invalid DPU raw-MAX interactions between generated `-inf` selector values and finite data without recognizing cumulative-max tensor structure.
- Added a regression proving the MAX composition no longer emits division barriers for its neutral selector.
- Verified NPU health independently after the original NaN failure; all reference elementwise cases passed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 25 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipCumulativeExtremaOps -x -q -n0`: 8 passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 15. Exact static WHERE lane routing — complete

- Added exact raw-lane routing for root `WHERE` UOps whose predicates are static in the output index.
- Flattened nested static `WHERE` trees into leaf masks and post-gathers, so selected FP16/INT16 values are copied bit-for-bit instead of recomputed through arithmetic mask blending.
- Added a local physical peephole for expanded `-max(-x, -y)` only inside finite static selections, mapping it to the DPU raw MIN stage used by the exact path.
- Materialized generated negative-infinity padding as the canonical finite neutral when a smaller gated source is packed into a larger static-selection buffer.
- Kept nonfinite extrema neutral conversion directional: selected `-inf` candidates under MAX are canonicalized, while max-pool's false padding arms retain their existing safe lowering.
- No sort operation or bitonic network is recognized; all rules are attached to `WHERE`, local MIN spelling, and memory routing UOps.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 26 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipSortValueOps test/backend/test_rockchip.py::TestRockchipMaxPoolOps -x -q -n0`: 14 passed, 33 subtests passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipCumulativeExtremaOps::test_simple_cummax -x -q -n0`: passed.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py`: pass.

### 16. Vectorized generic math reductions — complete

- Moved repeated scalar ADD structure into the generic UOp path: one representative semantic term is executed over the complete lane domain, then the materialized lanes are reduced physically.
- Extended structural INDEX materialization to periodic affine addresses such as `0..9, 0..9, ...`; tiled broadcast inputs no longer force scalar graph expansion.
- Reduced the 320-term BCE-with-logits program from 15,054 input UOps and the legacy oracle's 243,840 EW stages to one 320-lane map and 865 EW stages.
- Normalized semantic SQRT/EXP2/LOG2/SIN recipes once before physical allocation and prevented already-expanded recipes from re-entering the accurate-ADD expansion.
- Added the missing tagged SIGN semantic as its own four-stage physical recipe instead of treating it as ordinary `SUB(x, x)`.
- Added composite inverse-hyperbolic/atan recipe ownership to the generic math expansion, while conservatively leaving exact COPYSIGN raw-bit graphs with the legacy oracle for now.
- Added a serialized-image count guard so an oversized generic candidate declines before any 16-bit RKImage count can overflow.
- No BCE, cosine, dot, or sign tensor-operation recognizer was added; the new rules are repeated ADD execution, periodic INDEX, math UOps, and SIGN semantics.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipTranscendentalOps -x -q -n0`: 45 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/unit/test_rockchip_uops.py test/backend/test_rockchip.py::TestRockchipDotOps test/backend/test_rockchip.py::TestRockchipSignOps -x -q -n0`: 39 passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 28 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 17. Bounded product-residual reductions — complete

- Added one generic physical path for 256-or-more repeated FP16 `MUL` terms followed by `ADD`, retaining each rounded
  product's residual and compensating the final physical reduction.
- Kept the rule semantic and composable: the path parses `MUL`, `ADD`, optional scalar/buffer bias, and the ordinary
  `WHERE(CMPLT(0, x), x, 0)` UOps emitted for ReLU. It does not recognize GEMM or convolution shapes.
- Materialized both operand streams through ordinary static gathers and bounded the complete scratch request before
  accepting a program.
- Restored the nested-convolution accuracy lost when milestone 11 capped the full expanded recipe. The old accurate path
  required about 17,800 EW stages per kernel; the bounded path uses about 4,600 stages for each 288-term nested-convolution
  reduction.
- Retained enough product precision for the 256-by-256 GEMM tolerance without converting one semantic UOp into one ioctl.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 29 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchip::test_big_gemm test/backend/test_rockchip.py::TestRockchipConvOps::test_nested_conv2d -x -q -n0`: 2 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchip test/backend/test_rockchip.py::TestRockchipConvOps -x -q -n0`: 73 passed, 6 skipped, 37 subtests passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 18. Bounded semantic math reductions and submission groups — complete

- Bounded static `RANGE` environment construction before allocation; oversized Cartesian products now decline with
  `RKPLAN_REJECT:static_index_budget` instead of exhausting the Python process.
- Generalized the repeated-ADD UOp rule from scalar output to batched output. Repeated EXP2/SQRT/LOG2/SIN results are
  first materialized over their physical lane matrix, then reduced, preserving the semantic result boundary between a
  math UOp recipe and its consumer.
- Lowered generic mapped local ADD loops through the typed UOp executor rather than requiring the legacy EW walker.
- Extended the residual-preserving generic `MUL+ADD` rule to 64-term reductions only when the result spans more than one
  hardware FP16 tile. This covers large ordinary UOp dots without capturing small convolution programs or recognizing
  attention/matmul graph dialects.
- Partitioned stateful physical dependency groups in the runtime: large equal-lane groups are spatially tiled, while
  small-lane groups longer than 48 stages are submitted sequentially with an explicit state initialization at each
  boundary. One semantic UOp program still owns the complete image; submission boundaries are a physical runtime rule.
- Restored all four scaled-dot-product-attention methods through the generic paths. The causal softmax denominator now
  matches the independently materialized EXP2 sum, and its 64-term QK reduction retains product residuals.
- Preserved the existing one-ioctl contract for ordinary core and convolution images.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 31 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipAttentionOps -q -n0`:
  4 passed.
- Attention plus complete transcendental class: 49 passed.
- Core plus complete convolution class: 73 passed, 6 skipped, 37 subtests passed.
- `TestRockchipReductionOps::test_std_mean`: passed without process abort.
- Three consecutive isolated `TestRockchipTranscendentalOps::test_logsumexp` runs: passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
