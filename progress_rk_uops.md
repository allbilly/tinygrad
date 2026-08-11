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

### 19. Nonfinite-safe absolute UOp composition — complete

- Folded the ordinary `WHERE(x < 0, -x, x)` UOp spelling to native physical ABS before a consuming LOG2/EXP2 recipe
  expands it into arithmetic. This prevents an inactive `0 * inf` arm from contaminating the selected value with NaN.
- Accepted equivalent negation spellings owned by ordinary UOps: unary NEG, multiplication by `-1`, and matched FDIV
  numerators such as `WHERE((1/x) < 0, -1/x, 1/x)`.
- Fixed both remaining FP16 power boundary cases through composable WHERE/ABS/LOG2/MUL/EXP2 semantics. No POW handler or
  tensor-power graph lowerer was added to the generic executor.
- Made execution of an empty physical EW sequence an explicit runtime no-op. Zero-stage constant/size images no longer
  enter submission grouping with an empty operation tuple.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 33 passed.
- Complete Rockchip integer and tensor power families: 9 passed.
- Complete transcendental and power families plus masked-select/nonzero size regressions: 56 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py::TestRockchipAttentionOps -q -n0`:
  4 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 20. IEEE predicate composition and bounded program resources — complete

- Lowered inverted and composed FP16 comparisons through one IEEE boolean recipe, preserving unordered NaN semantics for
  `>=`, `<=`, equality, inequality, and `isclose` instead of treating hardware masks as ordinary finite arithmetic.
- Added generic nonfinite-safe physical recipes for threshold `WHERE`, infinite-numerator FDIV, and shifted ReLU caps.
  These rules fix infinity selection, signed division, and extreme hard-sigmoid inputs without recognizing tensor methods.
- Initialized constant scratch from the declared physical slot size, so gather-only static `WHERE` routes preserve every
  nonzero constant lane through padding and slicing.
- Added a bounded LRU for persistent Rockchip program resources. Evicted programs release GEM scratch, command/task
  buffers, and cached PC-chain bodies, then rehydrate them transparently if the global compiler cache invokes them again.
- Reduced the authoritative full-census failures from 52 to 4 while increasing passes from 381 to 429. The old dense
  timeout/bad-address/memory-error cascade is gone; the four remaining failures are two submission timeouts and two
  unsupported `NEG` graphs.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 38 passed.
- Complete comparison class plus `isclose`, infinity-WHERE, and padding regressions: 11 passed.
- Fresh `test_div_naninf`, `test_hardsigmoid_extreme`, and `test_cross_entropy_smoothing` runs: passed.
- Forced `ROCKCHIP_PROGRAM_CACHE=1` mixed semantic and attention regressions: 8 passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -q -n0`:
  429 passed, 4 failed, 12 skipped, 154 subtests passed in 19m28s.
- `.venv/bin/python -m ruff check tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py test/unit/test_rockchip_uops.py`: pass.
- `.venv/bin/python -m mypy tinygrad/renderer/rockchip.py tinygrad/runtime/ops_rockchip.py`: pass.

### 21. Complete upstream-port census — complete

- Lowered root `WHERE(bool, int_constant, int_constant)` directly in the typed UOp executor. The predicate remains a
  canonical FP16 mask, the exact integer selection is computed as a physical EW recipe, and one explicit boundary stage
  converts the result to the canonical INT32 output layout.
- Removed the final fallback dependency on the legacy integer-WHERE adapter for permuted integer selection and NLL
  indexing. Both previously surfaced as unsupported unary `NEG` nodes only after the original ternary UOp was lost.
- Added one bounded runtime recovery for transient blocking-submit timeouts: reset the NPU and resubmit the same physical
  PC chain once. A second timeout remains an error. This prevents one driver timeout from poisoning every later program
  while leaving semantic execution on the NPU.
- Added host-independent regressions for the INT32 `WHERE` boundary and the exactly-once timeout recovery policy.
- Reached zero failures in the complete Rockchip port of upstream `test_ops`; `test_rockchip2.py` was not used.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 40 passed.
- Fresh `TestRockchipWhereOps::test_where_permute` and `TestRockchipLossOps::test_nll_loss_reductions`: 2 passed.
- Sustained core/INT16/comparison/attention/transcendental/loss/WHERE sequence: 106 passed in 5m42s.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -q -n0`:
  433 passed, 12 skipped, 154 subtests passed in 18m27s.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 22. Strict typed-address and local-control milestone — complete

- Stopped applying the legacy whole-program FP16 rewrite before typed lowering. Math UOps now expand inside the generic
  executor, and semantic `WHERE` nodes remain explicit phase barriers instead of being prematurely converted into
  arithmetic blends. The complete tensor-power class consequently lowers through ordinary comparison, selection,
  `LOG2`, multiply, and `EXP2` semantics.
- Added exact generic INT32 comparison by splitting canonical INT32 values into four raw byte lanes. Equality compares
  all bytes and signed less-than uses the existing lexicographic byte primitive; resulting masks use the canonical
  `BOOL_INT16` physical ABI.
- Added a bounded render-time interpreter for constant `RANGE` plus scalar local `LOAD`/`STORE` programs used only by
  global addresses. It follows UOp loop/local semantics, memoizes nested accumulator results by their semantic free
  ranges, and produces static gather offsets without reading or evaluating tensor values on the host.
- Added packed boolean `LOAD` as a typed memory rule: raw bytes are gathered into zero-extended canonical INT16 lanes,
  after which ordinary `CAST`, comparison, and arithmetic handlers compose normally.
- Generalized dynamic raw gather from fixed 16-bit values to 1/2/4-byte typed values. Direct runtime INT32 loads now
  select and repack all four bytes, while the existing equality-mask machinery remains dtype-independent.
- Moved `WHERE(output_lane < sum(bool), LOAD(dynamic_index), fill)` into the generic structural/memory path and renamed
  it around that UOp form. This is the physical gate required by bounded dynamic loads, not a tensor-operation lowerer.
- Added regressions for nested static local address programs, packed bool loads, full raw INT32 dynamic loads, nested
  semantic `WHERE` around math, exact INT32 comparison, and phased raw selection.
- Removed temporary renderer trace-print hooks. Rewrite inspection continues through Tinygrad's normal VIZ/DEBUG tools.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 47 passed.
- Strict `TestRockchipTensorPowerOps`: 7 passed.
- Strict `TestRockchipOneHotOps::test_one_hot` and `TestRockchipGatherOps::test_gather`: passed.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_UOPS_ONLY=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py::TestRockchipMaskedSelectOps -q -n0`: 2 passed in 48.56s.
- A diagnostic strict census before this milestone completed 193 of 445 tests: 148 passed, 39 failed, 6 skipped, and
  120 subtests passed in 17m56s. It was interrupted in a slow cumulative test; this is a baseline, not a final census.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 23. Strict predicate-prefix and coordinate execution — complete

- Routed direct unrolled FP16 predicate totals/prefixes, packed-bool prefixes, and typed integer nonzero prefixes through
  the generic executor. These are blocked physical implementations of `LOAD`, comparison, `CAST`, and `ADD` UOps;
  they are not tensor-operation handlers.
- Renamed the fixed nonzero catalog entries around their actual bounded predicate-count plus coordinate-selection UOp
  semantics, and made both FP16 and integer forms available to strict typed dispatch.
- Routed exact INT32 equality histograms, occurrence sums, prefix sums, and bounded lookup through strict dispatch. The
  blocked byte implementation prevents large valid `CMPNE`/`WHERE`/`ADD` programs from overflowing RKImage's 16-bit
  stage fields while preserving those UOps as the semantic program.
- Replaced the normalized-prefix sign compare with exact bounded `MAX(delta, 0)` then `MIN(..., 1)` arithmetic. This
  avoids an unnecessary standalone compare/reset cycle after the INT32-to-FP16 boundary.
- Reduced the randomized nonzero prefix kernel from 7,691 EW stages and 770 compare submissions to the existing blocked
  1,069-stage recipe with 12 compares. The final 4,367-UOp coordinate histogram now compresses to 778 byte-exact EW
  stages instead of being rejected by strict mode.
- Verified NPU health after the diagnostic driver abort using the independent RK3588 elementwise reference; every
  ADD/MUL/SUB/MAX/NEG/FDIV size through 131,072 lanes passed.
- Added host-independent regressions for strict FP16 predicate-prefix dispatch and compare-free normalized INT prefixes.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_UOPS_ONLY=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py::TestRockchipNonzeroOps -x -q -n0`: 2 passed in 225.13s.
- Deterministic 32x10 integer-coordinate nonzero pipeline: strict result matched NumPy, 118x2 output.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n0`: 49 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 24. Strict typed dynamic selection and scatter execution — complete

- Generalized the physical dynamic gather implementation and its direct/multi-index entry points around raw typed
  values rather than a fixed 16-bit tensor operation. Strict dispatch now materializes half, INT16, and INT32 dynamic
  loads through the same address-selection machinery.
- Kept dynamic INT32 address expressions in the canonical `RKLayout.INT32` ABI even when the program result is FP16.
  Expanded fancy-index selectors can consequently execute their ordinary `LOAD`, comparison, boolean, and `WHERE`
  UOps without recovering a tensor-level fancy-index dialect.
- Added true ternary boolean `WHERE` lowering and canonical `BOOL_MASK` to `BOOL_INT16` coercion. Mixed FP16 and INT32
  predicates now compose in one program, including the nested predicate selectors emitted by scatter and scatter-reduce.
- Selected nonfinite FP16 constants as raw bits when the selector uses `BOOL_INT16`; this preserves infinity exactly
  without requiring an FP16 mask or multiplying a nonfinite value by zero.
- Routed root INT32 bounds predicates and masked dynamic typed loads through strict physical memory materialization.
- Renamed the remaining dynamic-selection helpers around their UOp/memory semantics. The strict scatter tests execute
  Tinygrad's expanded selector/reduction UOps; the legacy scatter-specific lowerers are not routed by strict mode.
- Added a host-independent regression for boolean `WHERE` over exact INT32 comparisons and the canonical packed-bool
  output boundary.

Verification:

- Strict `TestRockchipFancyIndexOps` plus `TestRockchipScatterOps`: 15 passed in 75.13s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 50 passed.
- A pre-milestone strict census completed the first 193 of 445 tests before interruption in a long cumulative kernel:
  161 passed, 26 failed, 6 skipped, and 120 subtests passed in 22m34s. This was 13 fewer failures than the preceding
  148-pass/39-failure prefix; the four scatter failures reported by that census are now fixed by this milestone.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 25. Strict INT32 bitwise and expanded power semantics — complete

- Routed root INT32 `AND`, `OR`, and `XOR` UOps to the existing exact raw-byte physical recipe. Each canonical INT32
  lane is split into byte bit-planes, the semantic boolean operation is evaluated with native INT16 arithmetic, and the
  four result bytes are repacked at the output boundary.
- Routed `SHL` and signed/unsigned `SHR` UOps to the exact five-stage byte-plane barrel recipe. Constant and runtime
  shift counts share the same physical implementation; this is a UOp handler, not recognition of a tensor operation.
- Added conservative `CMOD` range semantics. A remainder by a known nonzero constant is now proven bounded even when
  its dividend is data-dependent, allowing later small `ADD`/`WHERE` nodes to choose an exact physical layout.
- Allowed integer comparisons to consume the canonical `INT_FP16` ABI directly. Integers produced by truncating FP16
  values retain exact ordering in those lanes, so Tinygrad's expanded power parity/sign graph composes without a POW
  graph lowerer.
- Added host-independent regressions for byte-plane INT32 logic, the barrel-shift recipe, and expanded `CMOD` parity
  arithmetic.

Verification:

- Strict `TestRockchipBitwiseOps`: 10 passed in 6.22s.
- Strict `TestRockchipTensorPowerOps`: 7 passed in 94.30s.
- Strict broadcast full/partial coverage: 2 passed and 30 subtests passed in 46.72s, including all six POW subfailures
  from the latest census.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 53 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 26. Large strict attention through bounded address plans and scratch lifetimes — complete

- Extended the FP32-to-FP16 storage boundary to preserve semantic `EXP2`, `LOG2`, `SQRT`, and `SIN` UOps. Their
  physical recipes remain owned by the corresponding UOp handler after the conversion boundary.
- Added a compact divided-affine memory rule for addresses containing `RANGE // constant`. The GQA key/value broadcast
  is represented by two gather axes instead of enumerating 1,048,576 static offsets.
- Proved large contiguous output indices symbolically before attempting fallback enumeration, and materialized static
  expressions that contain no `RANGE` exactly once. A million-lane constant expression no longer allocates a million
  compile-time environments.
- Added physical scratch lifetime coloring for the linear pre-gather plus EW schedule. UOps and all 2,494 physical GQA
  stages remain unchanged, but dead intermediates reuse storage: the GQA image fell from 2,421 virtual scratch slots and
  5,077,204,992 bytes to 89 physical slots and 186,646,528 bytes.
- Added host-independent regressions for FP32 semantic math at a half storage boundary, compact divided-range address
  plans, range-independent million-lane static values, and a 128-stage chain reusing dead `RKValue` storage.

Verification:

- Strict `TestRockchipAttentionOps`: 4 passed in 20.99s; strict GQA alone passed in 9.85s with the compacted image.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 57 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 27. FP32 reduction semantics at the FP16 storage boundary — complete

- Preserved the complete FP32 expression when a half output store consumes an FP32 accumulator. The generic storage
  rule now lowers that semantic expression as one compensated multi-half recipe instead of independently narrowing
  every `ADD` to a naive FP16 chain.
- This is a typed storage-boundary rule shared by reductions and cumulative programs; it does not recognize einsum,
  cumsum, or another tensor operation.
- Fixed scalar einsum accumulation and both cumulative-sum precision cases. Together with the attention GQA fix in
  milestone 26, all 26 failures from the latest measured prefix now have focused passing evidence; a new full census is
  still required for the 252 tests that were not reached by that run.
- Added a host-independent 64-term pure FP32 addition regression proving the half storage boundary emits a compensated
  physical expansion rather than one EW stage per naive addition.

Verification:

- Strict `TestRockchipEinsumOps` plus `TestRockchipCumulativeOps`: 12 passed in 99.42s.
- Strict `test_simple_cumsum`: 1 passed in 77.92s; strict `test_cumsum`: 1 passed in 5.97s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 58 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 28. Canonical INT32 inputs under bounded integer roots — complete

- Made a dynamic INT32 `LOAD` select the canonical `RKLayout.INT32` ABI even when range analysis proves the final root
  is bounded enough for FP16 or INT16. Physical input representation now takes precedence over result-range storage.
- Prevented the bounded comparison shortcut from converting a real INT32 input load through `_int_fp16_expr`. Exact
  comparisons split the canonical four-byte value, execute byte-wise masks, and widen the bounded result only at the
  output boundary.
- This directly composes Tinygrad's one-hot `LOAD`, coordinate arithmetic, `CMPNE`, and ternary `WHERE` UOps without a
  one-hot lowerer.
- Added a host-independent nested-range regression matching the 3-by-6 one-hot UOp program.

Verification:

- Strict `TestRockchipOneHotOps::test_one_hot`: 1 passed in 2.81s.
- The fresh full strict census reached the first previously unknown gap after 111 passed, 6 skipped, and 96 subtests
  passed in 11m57s; that sole failure was the one-hot case fixed by this milestone.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 59 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 29. Exact composable FP16 comparison and larger structural RANGE execution — complete

- Replaced reset-heavy FP16 `CMPEQ`/`CMPNE` and `CMPLT` stages with exact raw-byte UOp recipes. Equality canonicalizes
  signed zero and rejects NaNs; ordering maps canonical FP16 bytes to an unsigned lexical key and masks unordered NaNs.
- Cached each physical FP16 value's byte split, NaN classification, and sortable representation in `RKContext`, so
  adjacent comparison and boolean UOps reuse the same physical `RKValue` components instead of rediscovering layout.
- Kept boolean composition as boolean UOps and made `CAST(bool -> half/int)` an explicit BOOL_INT16 boundary recipe.
  This prevents the generic storage path from recovering the former FP16 positive-mask graph dialect.
- Raised the generic static RANGE budget to 1,024 iterations and the expanded-node budget to the RKImage 16-bit limit.
  The 1,022-lane cumulative-index loop is now interpreted by the same local LOAD/STORE/MAX structural executor.
- The 512-lane cumulative index image fell from 13,834 stages with 2,050 standalone compare stages to 25,624 native
  INT16-composable stages with zero compare resets. The 1,022-lane image has 51,128 stages, also with zero compare
  resets and below the RKImage limit.
- Added/updated host-independent comparison, boolean-store, typed-`WHERE`, and nonfinite-selection regressions for the
  canonical BOOL_INT16 ABI.

Verification:

- Strict `TestRockchipComparisonOps`: 5 passed in 30.16s.
- Strict `TestRockchipCumulativeExtremaOps::test_simple_cummax`: 1 passed in 94.81s.
- Strict `TestRockchipCumulativeExtremaOps`: 8 passed in 205.35s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n0`: 60 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 30. Dependent scalar extrema through generic structural UOps — complete

- Generalized scalar local definitions from one `RANGE` to an ordered tuple of nested ranges. The same structural
  executor now interprets multiple dependent local buffers, recursively materializes their dependencies, and leaves
  mixed-use local loads for semantic execution instead of forcing them through the address-only evaluator.
- Added a bounded vectorization rule for two dependent scalar `MAX` accumulators. It validates the actual local
  `LOAD`/`STORE`, comparison, cast, coordinate, and affine-output UOps, materializes the candidate expression through
  the ordinary typed executor, and then performs the two physical reductions. It does not identify softmax, argmax,
  or another tensor operation.
- Preserved true UOp arity and dependency semantics across the nested 45-by-65 and flat 2,925-iteration forms emitted
  by Tinygrad. Both forms stay under RKImage's 16-bit stage fields without expanding the two local loops into a
  44,000-node expression and a 400,000-stage image.
- Gave one-lane physical reductions a 64-byte-spaced scratch ABI. The DPU writes an aligned scalar footprint even when
  the semantic stage count is one, so reducing contiguous two-byte lanes corrupts neighbors. Static gathers now space
  the candidates before the FP16 maximum and the INT16 coordinate maximum.
- Made embedded `CAST(bool -> int)` widen through the canonical `BOOL_INT16` ABI, and implemented embedded INT32
  bitwise-not as four exact raw-byte `255 - byte` stages followed by explicit repacking. This covers the ordinary UOps
  used by arg-extrema without relying on saturating signed INT32 subtraction at `INT32_MIN`.
- Reset the NPU once when a Rockchip device is opened. Precision state survives process exit, so initializing only the
  software mode flag produced nondeterministic first-program results even though the hardware health check passed.
- Added host-independent regressions for embedded exact INT32 not and the dependent scalar-local extrema structure.

Verification:

- Strict `TestRockchipArgExtremaOps`: 3 passed in 7.45s.
- Strict `TestRockchipArgExtremaOps::test_softmax_argmax`: 1 passed in 35.43s, covering both softmax axes.
- The verified strict prefix through collection index 193 is now 188 passed and 6 skipped; 251 collected tests remain
  to be censused in strict mode.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 62 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 31. Exact raw-bit sign UOps with an explicit gradient — complete

- Removed the generic executor's remaining `copysign` graph rejection. Tinygrad now states floating-point `copysign`
  directly as `BITCAST`, integer `AND`/`OR`, and `BITCAST` UOps, so the semantic program preserves signed zero and NaN
  payload bits without recovering a tensor-operation dialect in the renderer.
- Wrapped that value-producing UOp body in a Tinygrad `FUNCTION` with an explicit derivative. The forward program stays
  raw-bit exact while ordinary CPU autograd retains the magnitude gradient and the sign input receives zero gradient.
- Added the physical half/INT16 raw-byte ABI to `RKContext`. Half-to-INT16 and INT16-to-half `BITCAST` are representation
  changes over the same `RKValue`; mask-specific integer `AND`, disjoint-mask `OR`, and the general INT16 bitwise fallback
  operate on explicit raw bytes and repack at the storage boundary.
- Made a root raw `BITCAST` use an exact two-byte post-gather. A native FP16 copy canonicalizes NaNs and therefore cannot
  implement a raw-bit storage boundary.
- The unmodified generic path also passed sort, argsort, top-k, and elementwise extrema. The verified strict prefix now
  extends through collection index 204: 199 passed, 6 skipped, and 240 collected tests remain to be censused.
- Added a host-independent raw half/INT16 mask round-trip regression.

Verification:

- Strict `TestRockchipSortValueOps`, `TestRockchipSortIndexOps`, `TestRockchipTopKOps`, and
  `TestRockchipElementwiseExtremaOps`: 5 passed.
- Strict `TestRockchipSignOps`: 6 passed in 3.40s.
- CPU `TestOps::test_copysign` and `TestOps::test_copysign_exact`: 2 passed in 15.58s, including backward checks.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 63 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 32. Composable FP32 math operands at the FP16 physical boundary — complete

- Made the FP32-to-FP16 storage boundary recursively lower semantic math UOps before committing the physical half
  representation. A nested `SIN`, `EXP2`, `LOG2`, or `SQRT` therefore remains owned by its UOp recipe while surrounding
  half arithmetic reuses Tinygrad's ordinary symbolic algebra.
- Preserved additive FP32 phase operands for `SIN` by reducing each physical term independently and carrying constant,
  addition, and split-period rounding residuals as an FP16 high/correction pair. This executes Tinygrad's expanded
  cosine and tangent programs literally; no cosine, tangent, softmax, or sigmoid tensor-operation lowerer is used.
- Split compensated arithmetic into addition-only and product-aware forms. Periodic reduction can retain the small
  residual needed beside tangent poles without applying Dekker multiplication to a large period multiple and
  overflowing its FP16 splitter.
- Bounded storage algebra simplification to small expressions. Large softmax sums retain their physical precision tags
  and compile in about one second instead of recursively rediscovering accurate-add expansions and exhausting memory.
- Added host-independent regressions for additive FP32 SIN phase materialization and nested storage algebra reuse.
- The complete strict transcendental class now passes. The verified strict prefix extends through collection index 249:
  244 passed, 6 skipped, and 195 collected tests remain to be censused.

Verification:

- Strict `TestRockchipTranscendentalOps`: 45 passed in 143.24s.
- Strict `TestRockchipTranscendentalOps::test_tan`: 1 passed in 31.51s, including both near-pole vectors and large angles.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 65 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 33. Composable loss reductions and indexed storage — complete

- Generalized static `INDEX` materialization so non-affine offsets become an explicit `RKGather`, while affine repeated
  inputs retain their compact mapped-address representation. Loss expressions now consume the same physical `RKValue`
  ABI as ordinary elementwise programs instead of requiring cross-entropy or NLL graph recognition.
- Kept runtime-addressed loads literal through the UOp executor and added a symmetric, opt-in `HOST_ADDRESS` ABI for
  affine dynamic gather addresses. Host fallback is limited to index calculation and raw layout materialization; it does
  not evaluate arithmetic, comparison, reduction, or transcendental semantics.
- Added bounded integer-to-half conversion at the physical storage boundary. Sparse loss label counts and denominators
  now use the generic typed `CAST` recipe without rediscovering a categorical-loss graph dialect.
- Added product-error materialization and compensated scalar addition for medium mixed-sign FP32 `MUL`/`ADD` reductions.
  This preserves the semantic UOps while accounting for FP16 product rounding at the physical boundary.
- Removed the unsafe matrix regrouping shortcut. Non-affine class-probability loss reductions now execute their ordinary
  UOp reduction structure, trading speed for deterministic correctness.
- Added host-independent regressions for affine dynamic host-address encoding and non-affine scalar product reductions.
- The complete strict loss class now passes. The verified strict prefix extends through collection index 263:
  258 passed, 6 skipped, and 181 collected tests remain to be censused.

Verification:

- Strict `TestRockchipLossOps`: 14 passed in 228.35s.
- Strict direct NLL variants: 6 passed in 62.69s.
- Strict `test_cross_entropy_reductions`: three consecutive focused passes.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 67 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 34. IEEE inverted comparisons and exact half-value predicates — complete

- Made boolean `CMPNE(CMPLT(half, half), true)` preserve IEEE unordered semantics compositionally. The handler reuses
  the canonical raw FP16 classification cached in each operand's `RKValue`, computes `(1 - less) * numeric`, and keeps
  NaN lanes false for `>=` and `<=` without a compare ioctl or tensor-operation lowerer.
- Strengthened the host-independent inverted-comparison regression to require that final native INT16 numeric mask.
- Added exact equality as a semantic short circuit in Tinygrad's `isclose` UOp construction. With HALF as the default,
  Tinygrad's fused algebra can reduce `x - (x + epsilon)` before the half value is materialized; accepting the ordinary
  equality UOps restores the mathematically valid result on CPU and Rockchip without adding an `isclose` renderer path.
- Updated the `WHERE` submit-count audit for the generic executor. Two permuted typed selections now use four submits
  rather than the legacy catalog's 24; the old expectation remains active under `ROCKCHIP_UOPS=0`.
- Cast, bitcast, classification, comparison, contract skips, logical predicates, rounding, modulo, division rounding,
  boolean reductions, `WHERE`, and interpolation all pass in the contiguous strict census. The verified prefix extends
  through collection index 310: 300 passed, 11 skipped, and 134 collected tests remain to be censused.

Verification:

- Strict collection indices 264–310: 42 passed and 5 contract skips across the complete class runs.
- Strict `TestRockchipComparisonOps::test_cmp_ge`: 1 passed in 3.23s.
- Strict `TestRockchipLogicalPredicateOps`: 4 passed in 5.14s.
- Strict `TestRockchipBooleanReductionOps` plus `TestRockchipWhereOps`: 10 passed in 11.49s.
- Strict `TestRockchipInterpolateOps`: 8 passed in 7.99s.
- CPU `DEFAULT_FLOAT=HALF TestOps::test_isclose`: 1 passed in 13.79s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 67 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 35. Generic layout fills and opaque-source math recipes — complete

- Made dynamic-address probes check a typed CONST for zero without coercing it through Python `int`. Guarded FP16
  loads with `+inf`, `-inf`, NaN, or another nonzero fill now fall through to ordinary typed LOAD materialization
  instead of raising before the generic executor can handle them.
- Added a host-independent guarded `+inf` fill regression that verifies the exact `0x7c00` FP16 gather fill bits.
- Bounded optional compensated-add and composite-math peepholes to 4,096-node graphs. Literal UOp execution remains
  available up to the RKImage 65,535-stage limit, so a large correctness graph no longer incurs quadratic rewrite work.
- Made each SQRT/EXP2/LOG2/SIN handler temporarily replace its source dependency with an opaque typed PARAM while its
  small physical recipe is rewritten. The original UOp source is substituted back afterward, preventing one math UOp
  from sending a 52,000-node reduction dependency through the whole legacy graph-rewrite catalog again.
- The formerly aborting 52,498-node `std(axis=...)` program now compiles and passes through literal ADD/MUL/SQRT UOps.
- Movement, triangular, concat, and padding classes pass in the contiguous strict census. The verified prefix extends
  through collection index 357: 346 passed, 12 skipped, and 87 collected tests remain to be censused.

Verification:

- Strict `TestRockchipMovementOps`: 31 passed and 1 skipped in 6.88s.
- Strict triangular, concat, and padding classes: 15 passed and 28 subtests passed in 10.92s.
- Strict `TestRockchipReductionOps::test_std_axis`: 1 passed in 87.89s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 68 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 36. Single-owner FP32 storage precision and complete generic reductions — complete

- Made the FP32-to-FP16 storage boundary the single owner of its compensated ADD recipe. Generic math expansion still
  lowers any nested SQRT/EXP2/LOG2/SIN UOps, but it no longer treats already-physical storage ADDs as fresh expressions
  and recursively compensates them a second time.
- Disabled `RKContext`'s optional accurate-ADD discovery after a storage recipe has been materialized. Each semantic
  boundary is therefore lowered once, while ordinary non-storage UOp graphs retain the existing precision discovery.
- A six-term row sum fell from 18,662 physical EW stages to 108 and now matches the FP16 reference. The 64-term unit
  regression caps its physical recipe below 2,000 stages so recursive precision expansion cannot silently return.
- The complete reduction class now passes through ordinary CONST/LOAD/CAST, ADD/MUL/MAX, structural reduction, and
  math-handler composition. No sum, mean, variance, std, product, or normalization lowerer was added in this milestone.
- The verified strict prefix extends through collection index 393: 382 passed, 12 skipped, and the final 51 collected
  IncrementalOps tests remain to be censused.

Verification:

- Strict first half of `TestRockchipReductionOps`: 18 passed and 18 deselected in 111.89s.
- Strict sum/variance half of `TestRockchipReductionOps`: 18 passed and 18 deselected in 89.79s.
- Strict `TestRockchipReductionOps::test_sum`: 1 passed in 3.82s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 68 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 37. Complete per-UOp census and stable physical submission — complete

- Completed the authoritative strict census in one process. All 445 collected Rockchip cases are now accounted for:
  433 passed and 12 explicit contract skips, with another 154 parameterized subtests passing.
- Kept FP32 storage precision attached to UOp structure. Pure root ADD trees are committed once; large root product-ADD
  trees and product-ADD boundaries nested under later FP16 arithmetic retain the bounded accurate-ADD pass. This covers
  ordinary reductions, small and large dot products, biased convolutions, and causal attention without recognizing any
  tensor operation.
- Added a host-independent nested storage regression for `ADD(CAST(FP32 product sum -> FP16), FP16 bias)`. The handler
  discovers only CAST, ADD, and MUL semantics and preserves true UOp arity.
- Terminated PC chains with `REGISTER_AMOUNTS=0` while pointing the speculative terminal fetch into the existing mapped,
  zero-filled guard page. This removed the reproducible address-zero IOMMU fault without splitting physical stages into
  one ioctl per UOp.
- Made transient driver-start timeout recovery bounded and configurable. The production defaults use a 6-second submit
  timeout with at most four reset-and-retry attempts; `ROCKCHIP_SUBMIT_TIMEOUT_MS` and `ROCKCHIP_SUBMIT_RETRIES` can
  tighten the policy for diagnostics.
- The old operation-specific catalog remains present as an oracle, so the renderer is still 9,892 executable lines.
  Passing the complete census is the prerequisite for deleting those superseded paths in the next milestones; no LOC
  saving is claimed yet.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -x -q -n0`:
  433 passed, 12 skipped, and 154 subtests passed in 1,467.42s.
- Strict `TestRockchip` plus `TestRockchipAttentionOps`: 35 passed in 137.94s.
- Strict convolution/attention/reduction precision regressions: 5 passed in 9.99s.
- Strict masked-select and nonzero programs after guarded PC termination: 4 passed in 81.71s, with no new address-zero
  IOMMU fault.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -x -q -n12`: 69 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 38. Default host-address ABI and whole-schedule scratch reuse — complete

- Made raw host address materialization the correctness default for runtime-dependent LOAD and STORE addresses. It
  remains explicitly classified as `HOST_ADDRESS`, performs only index/address calculation and raw byte movement, and
  can be disabled with `ROCKCHIP_HOST_GATHER=0` for native-only audits.
- Fixed the strict UOp-only NLL program without adding an NLL lowerer: its runtime integer index now feeds the generic
  host gather, after which ordinary LOAD/ADD/MUL and reduction UOps execute normally.
- Extended physical scratch lifetime coloring across initial gathers, host gathers, EW stages, mid-program gathers,
  post-gathers, and host scatter. Mid-gather destinations remain pinned because partial materializations carry state
  across phases; all ordinary expression values can reuse the surrounding arena.
- Removed the incorrect 65,535 limit on EW stage count. RKImage already stores that count as 32-bit; only scratch,
  combined gather, and host-address counts use 16-bit header fields.
- The 20,642-UOp NLL3D program previously rejected with 111,474 virtual scratch values and 103,682 EW stages. It now
  fits the physical ABI and passes by literal UOp execution. No tensor-operation recognizer was introduced.
- A fresh strict census reached 249 passed, 6 skipped, and 126 passing subtests before exposing NLL3D. Its focused fix
  passes; the remaining cases still need a resumed strict census before legacy deletion begins.

Verification:

- Strict `TestRockchipLossOps::test_nll_loss`: 1 passed in 4.14s.
- Strict `TestRockchipLossOps::test_nll_loss_3d`: 1 passed in 171.29s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 70 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 39. Typed FP32 boundaries and resumed strict interpolation census — complete

- Kept FP32 storage precision attached to semantic boundaries. Independent nested CAST boundaries and FP32 numerator /
  denominator sums now retain their high/low half expansions through later arithmetic instead of being flattened into
  one prematurely rounded expression. Weighted NLL3D passes without an NLL-specific renderer path.
- Extended generic typed LOAD materialization to raw FP32 arguments. Static remaps first gather 32-bit lanes, DPU ADD
  converts each four-lane group into the canonical FP16 physical ABI, and one mid-program gather compacts the aligned
  groups. A raw FP32 zero operand avoids the secondary EW input's non-FP32 interpretation; no host numeric conversion
  is used.
- Bounded FP32-output PC chains to 16 stages and reset between chunks. Hardware probes established that 17 four-lane
  conversion tasks pass while 18 in one chain time out; a 3,348-element conversion now completes correctly in 6.7s.
- Preserved complete static float subgraphs until their single FP16 materialization boundary. Interpolation coordinates
  are therefore cancelled in FP32 before rounding the fractional weight, rather than rounding the large coordinate and
  its floor separately. The focused one-axis maximum error fell from 0.02783 to 0.001953.
- Added composable terminal CAST/BITCAST coverage, periodic dynamic host-gather materialization, semantic TRUNC,
  grouped boolean structural reduction, and the second generic attempt over Tinygrad's ordinary FP16 rewrite. These
  additions solve UOp semantics and physical layouts; no tensor-operation lowerer was added.
- The resumed strict UOp-only census now again extends through collection index 310: 300 passed, 11 explicit contract
  skips, and 134 collected tests remain. There is no current failing case at that boundary.
- The legacy oracle is still present, so line saving has not started: `sz.py` reports 10,051 executable renderer lines
  and 488 runtime lines. The expected large reduction remains gated on completing the remaining strict census first.

Verification:

- Strict resumed loss/cast/bitcast/classification/comparison/rounding/modulo/division/boolean/WHERE prefix: 300 passed
  and 11 skipped cumulatively through collection index 310.
- Strict `TestRockchipInterpolateOps`: 8 passed in 56.05s.
- Strict `TestRockchipInterpolateOps::test_interpolate_bilinear`: 1 passed in 19.92s.
- Native FP16→FP32 3,348-element stress conversion: exact pass in 6.7s.
- Independent RK3588 health check: all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 lanes passed.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 78 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 40. Generic memory composition and complete structural-reduction replay — complete

- Completed strict UOps-only replay of movement, triangular, concat, padding, and all 36 reduction methods. The
  authoritative prefix now extends through collection index 393: 382 passed, 12 explicit contract skips, and the final
  51 IncrementalOps cases remain. There is no failing case in the completed prefix.
- Made static LOAD materialization composable when its default is another typed value. The default is copied first and
  the gated source is overlaid as a partial gather, so concat and padding require no graph or tensor-operation lowerer.
- Extended the mapped ADD-loop executor to materialize one reduction and then run arbitrary ordinary post UOps. CAST,
  MUL, SQRT, WHERE, and other handlers now compose after the accumulator rather than forcing literal loop expansion.
- Added generic sequential output-store execution and generic independent scalar-local reduction staging. Multiple
  reductions are materialized into canonical scratch values, then their shared output UOps execute normally; stacked
  std/mean passes without a std/mean recognizer.
- Made RANGE discovery semantic by excluding AFTER/END ordering sources attached to range nodes. Structural execution
  still respects program order, while index enumeration no longer mistakes completed reduction loops for output axes.
- Routed plain ADD/MAX/MUL local loops through the generic reduction arena before mapped materialization. A 16,384-lane
  full sum now gathers directly into the arena and completes in one ioctl as required by the backend contract.
- Kept large math UOps owned by their handlers. Programs above the eager-recipe threshold defer expansion, and the
  handler registers one tagged physical recipe. A correction-equals-count std edge fell from a 230,619-node eager graph
  to 373 nodes and from minutes of compilation to a few seconds.
- Replaced quadratic scratch lifetime coloring with a heap-based allocator. It preserves pinned mid-gather state and
  interval safety while avoiding a full physical-slot scan for every virtual UOp value.
- The legacy correctness catalog is still present, so no line saving is claimed yet: `sz.py` reports 10,217 executable
  renderer lines and 488 runtime lines. The final 51-case replay remains the gate before deleting superseded paths.

Verification:

- Strict movement/triangular/concat/padding continuation: 46 passed and 1 skipped cumulatively.
- Strict `TestRockchipReductionOps`: all 36 methods passed across resumed segments; the final 11-method segment passed
  in 46.02s, and `test_sum_full` separately passed in 4.20s with exactly one submit.
- Strict `TestRockchipReductionOps::test_std_mean`: 1 passed in 19.13s.
- Strict `TestRockchipReductionOps::test_std_one_in_axis`: 1 passed in 4.76s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 83 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 41. Complete IncrementalOps UOp-only replay — complete

- Completed the final 51 collected IncrementalOps methods under `ROCKCHIP_UOPS_ONLY=1`. The segmented strict census
  now accounts for all 445 collected cases: 433 passed, 12 explicit contract skips, and zero failures.
- Added the missing physical boundary for a root FP32 CONST. The semantic value is represented in the canonical FP16
  RKValue layout and the existing terminal FP32 conversion owns the 32-bit output, matching FP32 LOAD behavior without
  host numeric conversion.
- `full_like` with an FP32 destination now passes through CONST, RANGE, STORE, and the typed output boundary. The other
  50 IncrementalOps methods needed no renderer change, which is the intended per-UOp architecture metric.
- The legacy catalog remains in place until a single-process full census confirms that segmented replay did not hide
  state or ordering interactions. Renderer size therefore remains 10,217 executable lines and runtime size remains 488.

Verification:

- Strict `TestRockchipIncrementalOps`: 51 passed in 13.72s.
- Strict `TestRockchipIncrementalOps::test_full_like`: 1 passed in 5.21s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 42. Clean single-process UOp replay and generic product-reduction precision — complete

- Completed the authoritative single-process replay under `ROCKCHIP_UOPS_ONLY=1`. All 445 collected cases are now
  accounted for together: 433 passed, 12 explicit backend-contract skips, zero failures, and 154 parameterized
  subtests passed. There are no remaining unreplayed cases.
- Used the combined replay to expose deterministic FP16 accumulation gaps that segmented execution had hidden. An
  eight-term product sum now uses the existing two-product physical expansion with Kahan accumulation, and a static
  broadcast bias is materialized through the ordinary gather ABI. The biased two-layer convolution passes without a
  convolution lowerer.
- Routed structural ADD loops whose semantic term is an ordinary MUL through the generic dot reducer before the plain
  scalar reducer. Large loop dots expand their bounded product UOps into the existing precise product/add recipe;
  `broadcastdot` now passes without recognizing matmul.
- Tightened the plain scalar reduction rule so the accumulator update owns ADD/MUL/MAX semantics and only direct LOAD
  terms are reduced by that path. Transformed terms such as ABS and nonzero predicates fall through to the generic
  mapped-term executor. All normalization variants, including `p=1` and `p=0`, now compose correctly.
- The legacy catalog is still present and deletion has not started. `sz.py` reports 10,233 executable renderer lines
  and 488 runtime lines, so no line saving is claimed yet. The clean 445-case replay is the safety gate for the next
  milestone: remove superseded operation-specific paths and measure the actual reduction.

Verification:

- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP RUN_SLOW=1 ROCKCHIP_UOPS_ONLY=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py -q -n0 -x -rs`: 433 passed, 12 skipped, and 154 subtests passed in 1,400.50s.
- Strict `TestRockchipConvOps::test_biased_conv2d`, `TestRockchipConvOps::test_conv1d`, and
  `TestRockchipDotOps::test_broadcastdot`: 3 passed and 14 subtests passed in 8.70s.
- Strict `TestRockchipDotOps` plus `TestRockchipEinsumOps`: 13 passed in 56.34s.
- Strict `TestRockchipReductionOps`: all 36 methods passed in 100.41s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 43. Delete the superseded operation catalog and make UOp lowering permanent — complete

- Removed the renderer's legacy fallback and the `ROCKCHIP_UOPS`/`ROCKCHIP_UOPS_ONLY` selection boundary. `render()`
  now always lowers the supplied UOp program, retries only after the ordinary per-UOp FP16 math rewrite, and rejects a
  program if neither generic pass can represent it.
- Deleted the operation-specific `lower_ew()` dispatch wall and 94 unreachable lowerers/helpers for tensor power,
  fancy indexing, scatter, cumulative operations, arg-extrema, sort/top-k, pooling indices, boolean/integer reductions,
  and related graph-dialect recovery. The retained `lower_ew()` is only the shared physical EW graph emitter used by
  composable UOp and reduction handlers.
- Recomputed a conservative module call graph rooted at the renderer/runtime API, exported unit-test ABI, module
  pattern matchers, decorators, defaults, and class bases. All 253 remaining top-level definitions are reachable; this
  milestone therefore stops at the safe dead-code boundary instead of deleting still-used typed/structural machinery.
- Reduced `tinygrad/renderer/rockchip.py` from 10,233 to 6,658 executable lines: 3,575 lines removed, or 35.0%.
  The physical diff is 39 insertions and 3,884 deletions. Runtime remains 488 executable lines.
- The remaining 6,658 lines are not the final size target. Much of the generic path still consists of reachable INT32
  byte-layout recipes, structural reduction materializers, dynamic-address handling, and bounded-stage-count helpers.
  The next deletion milestone must consolidate those behind canonical `RKValue` conversions, generic RANGE/local-state
  execution, symmetric gather/scatter, and shared typed compare/WHERE primitives before their callers can be removed.

Verification:

- Focused strict gather/fancy-index/scatter/cumulative/arg-extrema/sort/top-k/reduction/dot replay:
  83 passed in 482.66s.
- `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP RUN_SLOW=1 .venv/bin/python -m pytest
  test/backend/test_rockchip.py -q -n0 -x -rs`: 433 passed, 12 skipped, and 154 subtests passed in 1,422.38s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 44. One physical executor, compact local codegen, and profiled deletion — complete

- Routed every internally composed EW/reduction subprogram back through `_lower_uop_program(...,
  vectorize_reductions=False)` and deleted the old `lower_ew()` physical emitter plus its five private helpers. There is
  now one typed physical expression executor, `RKContext`, rather than a generic path that recursively called an older
  binary-only emitter.
- Made the FP16 output boundary accept the canonical FP16-backed `BOOL_MASK` and `INT_FP16` layouts. This closed the one
  physical-ABI gap exposed when BOOL-to-INT output conversion moved from `lower_ew()` to `RKContext`.
- Enabled compact local/RANGE codegen on the default renderer with a 16-byte local-state limit and removed the separate
  `RockchipBoolRenderer` mode. Added `SPECIAL` to generic static materialization so stacked multi-local outputs such as
  `std_mean` compose through their ordinary lane-id UOp.
- Profiled the historical slowest case, `test_nll_loss_3d`, with cProfile and Tinygrad PROFILE events. The old default
  spent 55.54s in its largest `do_to_program`, including 21.72s in Rockchip rendering, then executed 149,903 physical
  EW stages. Compact local codegen reduced the test call from 83.72s to 42.26s, a 49.5% improvement.
- Deleted the 195-line vectorized-unrolled ADD reduction path. Direct unrolled UOp programs now pass through the same
  executor, and the unit contracts check accepted/serializable images instead of requiring that removed optimization's
  gather layout.
- Tested deletion of the remaining 108-line vectorized MUL+ADD residual reducer. A biased two-layer convolution exposed
  a real 0.0127 precision regression, so that path and its strict unit contract were restored. It remains the next
  precision recipe to absorb into the generic RANGE reducer; no tolerance was weakened.
- Removed obsolete `ROCKCHIP_UOPS=0` branches and task-count assertions from the authoritative tests. Submission counts
  now describe only the permanent generic renderer.
- Reduced the renderer from 6,658 to 6,159 executable lines in this milestone, another 499 lines. From the 10,233-line
  pre-deletion baseline, 4,074 executable lines are gone (39.8%). Runtime remains 488 executable lines. The physical
  renderer diff for this milestone is 17 insertions and 526 deletions.
- Stopped repeating the 23-minute full replay at the user's request. The attempted run was clean through 114 passed,
  6 expected skips, and 96 subtests before interruption; focused numerical gates below cover the changed executor,
  local reductions, product precision, and convolution surfaces. A full `/445` replay remains required at the next
  large checkpoint.

Verification:

- Profiled strict `TestRockchipLossOps::test_nll_loss_3d`: 1 passed in 42.26s call time after the change, versus 83.72s.
- Strict `TestRockchipReductionOps`: 36 passed in 134.98s.
- Strict `TestRockchipConvOps`: 42 passed, 6 skipped, and 37 subtests passed in 311.48s.
- Strict dot/einsum plus NLL3D/std-mean/variance precision gate: 16 passed in 115.18s.
- Strict RKContext-only dot/reduction/cast/classification/comparison/logical/bool/WHERE gate: 67 passed in 145.29s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 45. Reject non-index DAGs early and delete dynamic accumulation recovery — complete

- Profiled the new slowest case, `test_nested_conv2d`. Of 186.48s call time under cProfile, 172.51s was spent in
  `_flatten_binary`: the dynamic-index accumulation recognizer recursively visited an ordinary shared convolution ADD
  DAG 97.55 million times before discovering that no runtime index tensor existed.
- Added a linear-time semantic precondition before flattening: the candidate must contain both a `WHERE` and a runtime
  INT `LOAD`. This is a guard on the address-selection rule, not a convolution fast path. `nested_conv2d` fell from
  150.42s to 5.47s call time (96.4% faster), and `large_input_conv2d` fell from 100.05s to 67.15s.
- Removed the dynamic-index accumulation entry points entirely after the performance diagnosis. Deleted the unrolled
  selector recovery, the local-loop selector recovery, affine-load parsing, and three custom accumulation images.
  Direct runtime addresses continue through host gather; expanded selector graphs execute their ordinary UOps.
- The pool/unpool, fancy-index, cumulative-extrema, arg-extrema, sort, and top-k hardware gate passes without the custom
  accumulation path. This confirms that the deleted graph dialect was an optimization/catalog layer, not required
  semantics.
- Reduced the renderer from 6,159 to 5,918 executable lines, another 241 lines. From the 10,233-line pre-deletion
  baseline, 4,315 executable lines are gone (42.2%). The physical renderer diff is 1 insertion and 260 deletions;
  runtime remains 488 executable lines.

Verification:

- Strict `TestRockchipConvOps::test_nested_conv2d`: 1 passed in 5.47s call time, versus 150.42s before the guard.
- Strict `TestRockchipConvOps::test_large_input_conv2d`: 1 passed in 67.15s call time, versus 100.05s.
- Strict pool/unpool/gather/fancy-index/cumulative-extrema/arg-extrema/sort/top-k gate:
  40 passed and 33 subtests passed in 424.82s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 46. Delete superseded unrolled integer selector catalog — complete

- Removed seven unrolled integer/bool recovery handlers and their private byte-equality matrix representation. Generic
  UOp replay now owns bool/int prefix counts and occurrence histograms instead of rediscovering those tensor programs.
- Kept only the two structural address materializers still required by `nonzero`: bounded INT32 lookup and bounded
  integer predicate coordinates. Removing either was tested independently and rejected by the corresponding strict
  `nonzero` case; this is the current semantic boundary rather than dead catalog retention.
- The deletion passed arg-extrema, sort, top-k, INT16 EW, cumulative ADD/MUL/extrema, WHERE, and both nonzero hardware
  gates. No host numeric evaluator or NumPy fallback was added: host participation remains address calculation and raw
  gather/scatter copies only.
- Reduced the renderer from 5,918 to 5,638 executable lines, another 280 lines. From the 10,233-line pre-deletion
  baseline, 4,595 executable lines are gone (44.9%). The physical renderer diff is 2 insertions and 300 deletions;
  runtime remains 488 executable lines.
- The focused duration report identifies cumulative extrema as the next compile/execution bottleneck: simple cummin
  took 90.52s and simple cummax 69.43s. The next milestone will profile that generic structural path before deleting
  another catalog block.

Verification:

- Strict arg-extrema/sort/top-k gate: 6 passed in 39.76s.
- Strict `TestRockchipNonzeroOps`: 2 passed in 49.07s.
- Strict INT16 EW/cumulative ADD/MUL/extrema/WHERE gate: 25 passed in 233.02s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 84 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 47. Vectorize static UOp materialization — complete

- Profiled the slowest known case, `test_max_unpool2d`, under cProfile. Of 190.52s total instrumented time, 171.80s
  was `_static_values`; scalar static interpretation made 47.71 million cast calls and 73.92 million recursive UOp
  evaluations. The hardware path itself issued 2,407 ioctls and spent only about 2s inside `ioctl`.
- Reused the existing typed vector UOp evaluator for compile-time RANGE/SPECIAL/index expressions. Static destination
  and value DAGs are now evaluated once over all lanes with shared subexpression caching, then validated and reordered
  exactly as before. This is address/layout planning only; runtime tensor arithmetic remains on the NPU.
- Added a unit contract covering multidimensional RANGE evaluation, WHERE, integer arithmetic, and nontrivial output
  reordering. The wider gather/fancy-index/nonzero/arg-extrema/sort/top-k/WHERE hardware gate also passes.
- `test_max_unpool2d` fell from about 108s to 20.58s (81% faster). The same generic change reduced simple cummin from
  90.52s to 60.42s and simple cummax from 69.43s to 46.84s, both about one third faster.
- This performance milestone adds three executable renderer lines, taking the renderer from 5,638 to 5,641 lines.
  The next milestone resumes deletion from this baseline.

Verification:

- Strict `TestRockchipMaxUnpoolOps::test_max_unpool2d`: 1 passed in 20.58s call time.
- Strict max-unpool plus simple cummin/cummax gate: 4 passed in 131.12s.
- Strict gather/nonzero/fancy-index/arg-extrema/sort/top-k/WHERE gate: 22 passed in 138.67s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 85 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 48. Delete prefix and occurrence graph recovery — complete

- Disconnected and replayed the seven remaining prefix/count/histogram dispatch entries. Generic UOp execution passed
  cumulative ADD/MUL/extrema, one-hot, masked-select, nonzero, arg-extrema, sort, and top-k without those recognizers.
- Deleted the complete now-unreachable 16-function dependency component: scalar local-loop recovery, unrolled FP16 and
  INT32 prefix builders, predicate masks, prefix row materializers, and the INT32 occurrence image. This removes another
  tensor-graph dialect rather than preserving it as an optimization catalog.
- Closed the one generic gap exposed by a unit-only normalized INT32 prefix: comparisons now normalize weak integer
  constants to typed integer operands. The program then composes through ordinary INT32 compare/WHERE UOps. Updated
  the two unit contracts to require accepted, serializable generic images instead of obsolete physical stage counts.
- No CPU numeric semantics were introduced. Compile-time static address/layout evaluation remains the only vectorized
  host calculation; dynamic arithmetic, comparison, WHERE, prefix, and reduction semantics execute as NPU stages.
- Reduced the renderer from 5,641 to 5,254 executable lines, another 387 lines. From the 10,233-line pre-deletion
  baseline, 4,979 executable lines are gone (48.7%). The physical renderer diff is 5 insertions and 420 deletions;
  runtime remains 488 executable lines.
- The generic path was also faster for several formerly recovered programs: simple cumprod fell from 23.32s to 12.12s
  and logcumsumexp from 19.15s to 5.74s in the focused duration reports.

Verification:

- Strict cumulative ADD/MUL/extrema gate: 18 passed in 147.45s.
- Strict one-hot/masked-select/nonzero/arg-extrema/sort/top-k gate: 11 passed in 102.52s.
- Post-gap-fix strict cumsum/simple-cumsum/masked-select gate: 3 passed in 31.42s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 85 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 49. Delete scalar loop boolean reduction recovery — complete

- Disconnected both pre-generic boolean reduction handlers. Ordinary scalar/axis `any` and `all` passed through generic
  RANGE/local UOps, proving the scalar loop recognizer and its integer-predicate image were superseded.
- Retained the grouped boolean structural executor after `test_all_large` demonstrated its distinct SPECIAL/BARRIER/IF
  program is not yet accepted generically. This preserves a proven semantic boundary instead of deleting required code.
- Removed the stale `ROCKCHIP:BOOL` test selector left after milestone 44 deleted the alternate renderer. The large test
  now exercises the sole canonical Rockchip renderer and passes through the retained grouped executor.
- Reduced the renderer from 5,254 to 5,208 executable lines, another 46 lines. From the 10,233-line pre-deletion
  baseline, 5,025 executable lines are gone (49.1%). The physical renderer diff is 2 insertions and 50 deletions;
  runtime remains 488 executable lines.

Verification:

- Strict logical-predicate and boolean-reduction gate: 11 passed in 13.54s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 85 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 50. Share static gather planning across reduction operands — complete

- Profiled the slowest recorded remaining case, `test_large_input_conv2d`. Of 112.08s instrumented time, 93.64s was
  the residual-preserving MUL+ADD reducer and 87.42s was `_gather_offsets`. Its 320 non-affine plans rebuilt the same
  90,720-lane output RANGE environment, causing 174.18 million generator calls.
- Added one lazy `RKStaticIndexEvaluator` per related gather-plan set. It materializes static output RANGE vectors and
  destination lanes once, while each operand retains a fresh expression cache to bound memory. The evaluator only
  handles compile-time INDEX/gate expressions; it never reads or computes runtime tensor values.
- Reused the evaluator for residual MUL+ADD operands and optional bias. Added a unit contract proving two distinct
  gather rows invoke RANGE materialization exactly once and retain correct nontrivial output ordering.
- `test_large_input_conv2d` fell from 67.15s to 18.34–18.62s, about 72% faster. Big GEMM, nested convolution, and biased
  convolution also retain strict numerical output through the same residual-preserving path.
- This performance milestone adds 14 executable renderer lines, taking the renderer from 5,208 to 5,222 lines. The
  next milestone resumes catalog deletion from this baseline.

Verification:

- Strict large-input convolution: 1 passed in 18.62s call time.
- Strict big-GEMM/nested/biased/large-input reduction gate: 4 passed in 31.69s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 86 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 51. Delete bool-total dynamic-load recovery — complete

- Tested deletion of the 147-line dependent scalar-local extrema recognizer. All 27 arg-extrema/sort/top-k/max-pool/
  cumulative-extrema methods remained numerically correct, but `softmax_argmax` regressed from about 3.4s to 156.11s.
  Restored that handler as a measured temporary physical optimization until generic multi-local execution is bounded.
- Independently disconnected all three dynamic-load graph recognizers. Generic `RKHostAddress` handled ordinary typed
  runtime loads, but strict SPECIAL-lane negative-normalized and multi-axis fancy-index programs proved the direct and
  multi-index structural parsers are still required; both were restored.
- Masked-select and nonzero passed without the bool-total-gated dynamic-load recognizer. Deleted that handler and its
  private bool-count parser. Those programs now compose through generic UOps/address materialization instead of a
  specialized `lane < sum(bool)` graph dialect.
- No CPU numeric evaluator was introduced. Host address execution remains limited to reading indices, calculating
  addresses, and copying raw typed bytes; predicate totals and selection semantics remain NPU UOps.
- Reduced the renderer from 5,222 to 5,140 executable lines, another 82 lines. From the 10,233-line pre-deletion
  baseline, 5,093 executable lines are gone (49.8%). The physical renderer diff is 88 deletions; runtime remains 488
  executable lines.

Verification:

- Trial strict arg-extrema/sort/top-k/max-pool/cumulative-extrema gate: 27 passed and 33 subtests passed in 312.10s.
- Final strict one-hot/gather/masked-select/nonzero/fancy-index/scatter gate: 21 passed in 137.16s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 86 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 52. Materialize masked loads before neutral rewrites — complete

- Profiled the current slowest simple cumulative-extrema case, `test_simple_cummin`. The instrumented 101.71s run
  spent 42.79s executing 66,231 ioctls; 10,738 synchronized gather calls consumed 12.68s, while static-value recovery
  consumed another 14.36s. The value and index images contained 2,044 and 1,537 synchronization points respectively.
- Moved generic masked-LOAD materialization ahead of the nonfinite reduction-neutral rewrite, and taught that generic
  fold to commute a finite nonzero scalar multiply through the mask. This preserves the LOAD default directly instead
  of expanding the selector into thousands of alternating gather/EW stages.
- The 512-lane value image fell from 8,177 to 1,030 EW stages and from 2,044 to 2 synchronization points. Strict
  `test_simple_cummin` fell from 58.38s to 42.16s (28% faster), while `test_simple_cummax` fell from 47.36s to 40.24s.
- A trial that forced large FP16 equality chains entirely onto the DPU removed another 4,098 materializations but took
  longer than two minutes, so it was reverted. The retained change addresses the measured bottleneck without trading
  synchronization for a slower physical recipe.
- No CPU numeric semantics were introduced. The host still performs only static index/default materialization; the
  multiply, comparison, selection, and cumulative extrema remain NPU execution.
- This performance milestone adds 16 executable renderer lines, taking the renderer from 5,140 to 5,156 lines. The
  following milestone resumes deletion from this measured baseline; runtime remains 488 executable lines.

Verification:

- Strict simple cumulative minimum: 1 passed in 42.16s call time.
- Strict simple cumulative maximum: 1 passed in 40.24s call time.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 87 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 53. Delete scalar loop reduction recovery — complete

- Disconnected the multi-scalar-local reduction recognizer first. `std_mean` then expanded a 13,125-iteration local
  expression and hit Python recursion depth, so that bounded multi-accumulator structural executor remains required.
- Also trialed deletion of the exact INT32 bounds-mask executor. Thirteen gather/fancy-index/validation cases passed,
  but the extended gate stopped at cold `masked_select(size=4)`, so it was restored conservatively. Later diagnosis
  showed that rejection was the independent masked-LOAD rewrite issue below; bounds-mask deletion remains a candidate.
- Disconnected the ordinary scalar ADD/MUL/MAX loop recognizer. The generic mapped/local UOp paths passed the complete
  reduction census numerically. The sole initial regression was `test_sum_full` using two submit segments instead of
  the required one for its 16,384-lane NOOPT sum.
- Added a physical scheduling peephole that composes a leading idempotent vector materialization with its dependent
  lane gathers. It proves the removed vector is dead before redirecting those gathers to the original operand, so the
  scalar EW chain starts without a synchronization split. This is a physical RKImage rule, not a tensor-op recognizer.
- Deleted the scalar loop parser/executor's 40-line lowering body, its private 42-line reduction-image builder, and the
  now-unreferenced spaced-gather and post-reduction sqrt/cuberoot helpers. SQRT/cuberoot semantics remain owned by the
  generic UOp math recipes.
- A cold-cache masked-select replay exposed a pre-existing rewrite issue hidden by cached images: an outer predicate
  total was being merged into a dynamic LOAD gate. Generic masked-LOAD folding now declines when the outer predicate
  has additional runtime LOAD dependencies, preserving that total as ordinary NPU WHERE/comparison execution.
- No CPU numeric semantics were introduced. The new peephole changes only static gather scheduling, and runtime host
  address work remains limited to reading indices, calculating bounded addresses, and copying raw typed lanes.
- Reduced the renderer from 5,156 to 5,062 executable lines, another 94 lines. From the 10,233-line pre-deletion
  baseline, 5,171 executable lines are gone (50.5%). The physical renderer diff is 32 insertions and 134 deletions;
  runtime remains 488 executable lines.

Verification:

- Cold full `TestRockchipReductionOps`: 36 passed in 121.55s.
- Cold one-hot/masked-select/nonzero gate: 5 passed in 73.60s.
- Cold `test_sum_full`: one NPU submit contract restored; passed in 2.54s call time.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 88 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 54. Narrow the residual loop matcher to dot semantics — complete

- Retried the 60-line INT32 bounds-mask deletion after fixing cold masked-LOAD folding. Cold masked-select/nonzero and
  all renderer units passed without it, but multi-axis integer fancy indexing still rejected while consuming its
  boolean bounds buffer. Restored the exact DPU byte-mask executor; replacing it with host comparison would violate
  the no-CPU-numeric boundary.
- Milestone 53 left `RKLoopReduction` with only one caller: the dot-product physical path. Removed the now-dead scalar
  reduction fields and parsing for generic nodes/environments, post-scale, sqrt, reciprocal, cuberoot, and clamping.
  The matcher now accepts only a plain FP16 local accumulator result and returns the six values dot lowering consumes.
- This is dead-state deletion rather than a new graph dialect. Non-dot reductions fall directly through to the generic
  mapped/local UOp executor, while dot retains its existing product-precision recipe.
- No CPU numeric semantics or host data evaluation were introduced.
- Reduced the renderer from 5,062 to 5,041 executable lines, another 21 lines. From the 10,233-line pre-deletion
  baseline, 5,192 executable lines are gone (50.7%). The physical renderer diff is 8 insertions and 29 deletions;
  runtime remains 488 executable lines.

Verification:

- Cold complete dot/einsum gate: 13 passed in 24.24s.
- Cold std/normalize/full-sum fallthrough gate: 3 passed in 36.46s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 88 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 55. Separate direct-NPU and generic masked-load folding — complete

- A final cold-cache replay after milestone 54 found that the safety rule for generic masked LOADs also disabled the
  existing all-DPU direct multi-index recognizer. Multi-axis integer fancy indexing therefore rejected even though its
  exact INT32 bounds-mask producer had correctly been retained.
- Made additional runtime gate dependencies an explicit opt-in used only by the direct typed-load recognizer. That
  recognizer executes the merged equality/bounds selection on DPU; the generic host-address path continues to keep
  predicate-total gates as ordinary NPU WHERE/comparison UOps.
- Cold multi-axis fancy indexing and cold masked-select now pass together, proving the two ownership paths no longer
  depend on cached compiler images.
- No CPU numeric semantics were introduced. This correctness follow-up adds one executable renderer line, taking the
  renderer from 5,041 to 5,042 lines; runtime remains 488 executable lines.

Verification:

- Cold complete `TestRockchipFancyIndexOps`: 10 passed in 66.17s.
- Cold conflicting fancy-index/masked-select pair: 2 passed in 8.49s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 88 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 56. Keep comparison-derived integer sums in native INT16 — complete

- Profiled the slowest current cold test, `TestRockchipNonzeroOps::test_nonzero`. Its instrumented 80.06s call made
  69.39 million function calls: compilation consumed 38.68s, Rockchip rendering 20.46s, and runtime 35.04s. The
  generated programs contained 9,070 synchronized gather points, 46,213 ioctls, and 36,784 buffer synchronizations.
- Split all nine fixtures and inspected every serialized image. The 32x10 and 10x5x3 cases dominated at 34.24s and
  21.41s in the instrumented diagnostic. Their unrolled prefix images repeatedly converted native `BOOL_INT16`
  comparison results into `INT_FP16` through raw-byte WHERE recipes before adding them.
- Made the physical ABI compositional at the program boundary: when a bounded integer result contains native FP16
  comparisons, choose canonical INT16 scratch instead of INT_FP16. Boolean casts then remain their existing exact
  0/1 INT16 lanes, ADD stays native INT16 DPU execution, and only the terminal output widens to INT32.
- The first two prefix images fell from 158 total synchronization points to 42 and from 438 mid-gathers to 84.
  Cold `test_nonzero` fell from 47.93s to 35.00s, about 27% faster.
- Trialed a more aggressive per-UOp INT16 choice inside mixed INT32 programs. A large coordinate ADD correctly exposed
  that its bounded WHERE operands had already materialized as INT32, so the over-broad experiment was removed rather
  than inserting implicit reinterpretation or host arithmetic.
- No CPU numeric semantics were introduced. Comparison, cast, prefix ADD, coordinate selection, and terminal widening
  all execute on DPU. This performance milestone adds one executable renderer line, taking it from 5,042 to 5,043;
  runtime remains 488 executable lines.

Verification:

- Cold full `TestRockchipNonzeroOps::test_nonzero`: 1 passed in 35.00s call time.
- Cold nonzero-size/masked-select/one-hot gate: 4 passed in 22.16s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 89 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 57. Delete direct integer-to-FP32 cast recovery — complete

- Disconnected the 36-line direct INT32/bool-to-FP32 image. The typed executor already owned INT32-to-FP16 narrowing
  and FP16-to-FP32 output conversion; composing those two physical value boundaries passed the cold cast census.
- Added generic CAST ownership for INT32 values and native BOOL_INT16 values. Boolean conversion remains an ordinary
  DPU WHERE selecting FP16 zero or one, followed by the same terminal FP32 conversion; no host value inspection is
  involved.
- Extended unit coverage to reversed/remapped INT32 and boolean loads, proving the generic path is not limited to a
  contiguous direct-load spelling. The generic INT32 image uses four EW stages where the deleted image used six.
- Removed the complete now-unreachable direct cast function and dispatch. Reduced the renderer from 5,043 to 5,014
  executable lines, another 29 lines. From the 10,233-line pre-deletion baseline, 5,219 executable lines are gone
  (51.0%). The physical renderer diff is 6 insertions and 38 deletions; runtime remains 488 executable lines.
- No CPU numeric semantics were introduced.

Verification:

- Cold complete `TestRockchipCastOps`: 2 passed in 5.84s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 90 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 58. Delete direct FP16-to-INT32 cast recovery — complete

- Disconnected the remaining seven-line direct FP16-to-INT32 wrapper. The first generic replay exposed that a root
  cast from FP16 has no statically provable integer range, so the context selected INT32 before reaching its existing
  half-valued truncation recipe.
- Made the physical ABI rule explicit: an integer program rooted in an embedded FP16 cast uses INT_FP16 scratch unless
  a true dynamic INT32 load already requires INT32. Generic CAST then executes the ordinary DPU truncation recipe and
  the existing terminal INT32 widening.
- Deleted the wrapper and dispatch after cold cast, integral rounding, modulo, and division-rounding tests passed.
  Reduced the renderer from 5,014 to 5,009 executable lines, another 5 lines; runtime remains 488 executable lines.
- No CPU numeric semantics were introduced.

Verification:

- Cold complete `TestRockchipCastOps`: 2 passed in 5.24s.
- Cold integral-rounding/modulo/division-rounding gate: 9 passed in 29.31s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 90 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 59. Group synchronized gathers once per execution point — complete

- Continued the `TestRockchipNonzeroOps::test_nonzero` profile from milestone 56. Runtime line-level profiling found
  9.60s spent rebuilding each synchronized-gather batch by rescanning every mid-gather for every one of its 9,070
  execution points.
- Replaced that quadratic scheduling pass with one stable grouping pass keyed by execution point. Gather order within
  every point, DPU stage boundaries, synchronization, and raw memory movement are unchanged.
- Cold `test_nonzero` fell from 35.00s after the physical-ABI improvement to 28.34s. Relative to its original 47.93s
  cold baseline, the two profiled fixes reduce call time by about 41%.
- The gather-heavy 512/1022-element cumulative-minimum fixture passes cold, exercising the multi-point scheduler over
  large prefix programs.
- No CPU numeric semantics were introduced. This scheduling optimization adds one executable runtime line, taking it
  from 488 to 489; the renderer remains 5,009 executable lines.

Verification:

- Cold full `TestRockchipNonzeroOps::test_nonzero`: 1 passed in 28.34s call time.
- Cold `TestRockchipCumulativeExtremaOps::test_simple_cummin`: 1 passed in 207.61s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 90 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 60. Delete the uint8 cast image catalog — complete

- Disconnected the mapped ADD-loop recognizer first. Small mean/variance/std axis cases passed through the literal
  executor, but full 15x25x35 variance and std reached a roughly 13k-deep expanded accumulator and hit Python's
  recursion limit. Restored that block unchanged: it remains the generic large static ADD-loop executor, not dead
  operation-specific code.
- Moved uint8 into the physical value ABI instead. A half-to-uint8 `CAST` now emits its truncation/modulo recipe through
  ordinary typed UOps, materializes canonical INT16 scratch, and exposes the low byte at the terminal STORE boundary.
  A uint8 `WHERE` composes the same INT16 values and canonical constants without recognizing the surrounding graph.
- Deleted both `_lower_fp16_uint8_cast` and `_typed_int16_byte_image`, plus their early dispatch. Renderer executable
  size fell from 5,009 to 4,984 lines, another 25 lines. From the
  10,233-line pre-deletion baseline, 5,249 executable lines are gone (51.3%); runtime remains 489 lines.
- No CPU numeric semantics were introduced. Truncation, modulo, WHERE, and FP16-to-INT16 conversion all execute on DPU;
  the terminal gather only materializes the selected low byte.

Verification:

- Cold complete `TestRockchipCastOps`: 2 passed in 6.10s.
- Cold upstream `TestOpsUint8::test_cast`: 1 passed in 2.74s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 61. Delete the boolean cast image wrapper — complete

- Moved FP16-to-bool `CAST` into the typed UOp handler. It now composes the existing native ABS and positive-mask
  recipes for any FP16 expression, producing the canonical BOOL_MASK physical value before the normal terminal bool
  conversion.
- Deleted `_typed_int_image` and the early direct-load boolean dispatch. The old wrapper rebuilt the output store and
  retargeted the terminal image; the generic context already owns both scratch placement and bool output conversion.
- Added coverage for a reversed, composed `(load + 1).cast(bool)` program, not only the direct-load spelling. The same
  expression passes on hardware for zeros, signs, infinities, and NaN.
- Renderer executable size fell from 4,984 to 4,971 lines, another 13 lines. From the 10,233-line pre-deletion
  baseline, 5,262 executable lines are gone (51.4%); runtime remains 489 lines.
- No CPU numeric semantics were introduced. ABS, comparison-mask generation, and bool output conversion execute on
  DPU.

Verification:

- Cold `TestRockchipCastOps::test_cast`: 1 passed in 6.29s.
- Composed FP16-expression-to-bool hardware boundary check: pass.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 62. Restore FP16 state without resetting after native INT16 chains — complete

- Profiled the current slowest case, the 512-element half of `test_simple_cummin`. Its instrumented 89.69s execution
  spent 55.03s in ioctl and 54.01s specifically in 513 NPU resets. The cumulative-index image alternates long native
  INT16 byte/mask chains with one ordinary FP16 MAX at each structural gather point.
- Replaced reset-on-INT16-to-FP16 transition with a fully stateful FP16 stage. The emitted DPU operation, dependency
  order, gathers, submits, and tasks are unchanged; the stateful stage explicitly restores the ordinary FP16 register
  configuration that the reset previously supplied.
- A cold standalone 512-element cumulative minimum still executes exactly 2,052 submits and 24,158 DPU tasks, but now
  completes in 12.65s. The complete 512/1,022-element `test_simple_cummin` fell from 207.61s to 28.77s, about 86% faster.
- The full cumulative-extrema, comparison, and cast hardware gate passes, covering native INT16, ordinary FP16,
  comparison isolation, terminal integer conversion, and subsequent-program state recovery.
- No CPU numeric semantics were introduced. This runtime scheduling change adds two executable lines, taking runtime
  from 489 to 491; renderer remains 4,971 executable lines.

Verification:

- Cold `TestRockchipCumulativeExtremaOps::test_simple_cummin`: 1 passed in 28.77s.
- Cold cumulative-extrema/comparison/cast gate: 15 passed in 80.51s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 63. Delete the dot-loop graph recognizer — complete

- Disconnected the remaining dot-loop fast path so dot, matvec, multidot, broadcast-dot, and einsum programs had to
  use the ordinary mapped static ADD reduction. The complete cold dot/einsum gate passed before deletion, proving the
  private graph parser was no longer required for correctness or precision.
- Deleted `_loop_reduction_shape`, `RKLoopReduction`, `_loop_reduction_match`, `_lower_dot_loop_reduction`, and the now
  unreachable `_lower_composed_uops` wrapper. This removes tensor-program recovery of MUL+ADD loops; the surviving
  structural executor consumes RANGE/local LOAD/STORE/ADD semantics generically.
- Renderer executable size fell from 4,971 to 4,910 lines, another 61 lines. From the 10,233-line pre-deletion
  baseline, 5,323 executable lines are gone (52.0%). The physical renderer diff is 70 deletions; runtime remains 491
  executable lines.
- No CPU numeric semantics were introduced. Products and reductions remain DPU EW stages, and no WMMA/matmul graph
  recognition replaced the deleted path.

Verification:

- Cold complete `TestRockchipDotOps` and `TestRockchipEinsumOps`: 13 passed in 22.34s.
- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 64. Delete stale physical-value and rejection scaffolding — complete

- Removed the unused `RKMultiGather` and `RKStatic` placeholder types and their `RKLeaf` aliases. No live lowering,
  serialization, runtime, or test path referenced them; physical values are represented directly by the current typed
  `RKValue`/`RKArg` machinery.
- Removed two diagnostic-only `reject` closures from the generic mapped and multi-local reduction executors. Their
  reason strings were never observed, so each branch now returns the same `None` result directly. Also deleted two
  unreferenced EW/pool constants.
- Renderer executable size fell from 4,910 to 4,900 lines. From the 10,233-line pre-deletion baseline, 5,333
  executable lines are gone (52.1%); runtime remains 491 executable lines.
- No CPU numeric semantics or lowering behavior were introduced. This milestone is semantics-neutral dead-code
  removal, so it does not require the currently unavailable NPU for behavioral validation.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 65. Delete dormant reduction-helper modes — complete

- Audited all repository callers of the private row and arena reducers, equality-mask builder, scalar evaluator, and
  output-store parser. Deleted configuration state that no caller supplied: INT32 row reduction, FP32/INT16 arena
  output, per-level arena barriers, configurable equality barriers, scalar-evaluator cache injection, and explicit
  REDUCE rejection.
- Kept the only live physical behavior unchanged: FP16/INT16 balanced rows, optional per-operation barriers for
  precision reducers, and the fixed SUB/ABS/compare equality sequence. Removed the final pool-index constant that
  became unreferenced with the stale radix in milestone 64.
- Renderer executable size fell from 4,900 to 4,891 lines. From the 10,233-line pre-deletion baseline, 5,342
  executable lines are gone (52.2%); runtime remains 491 executable lines.
- No CPU numeric semantics or lowering behavior were introduced. This is a private dead-mode deletion validated by
  serialized-image unit contracts and static checks; hardware remains unavailable pending reboot.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 66. Delete the unreachable fill ABI — complete

- Audited every `RKImage` producer across the repository. No renderer path constructed `RKFill`; only the decoder
  could recreate it, leaving its serializer flag, scratch-lifetime handling, append guards, and runtime host memcpy
  branch unreachable from any compiled UOp program.
- Deleted `RKFill` end to end and reserved its former header flag. Bumped `RKIMAGE_VERSION` from 31 to 32 so any stale
  serialized blob is rejected explicitly rather than interpreted under the narrower physical ABI.
- Renderer executable size fell from 4,891 to 4,880 lines and runtime fell from 491 to 486 lines. From the 10,233-line
  pre-deletion renderer baseline, 5,353 executable lines are gone (52.3%).
- No CPU numeric semantics or generated execution behavior changed. Compile-time constants continue through ordinary
  gather materialization; this removes a host numeric-fill path that the UOp renderer never emitted.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed, including all RKImage round trips.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 67. Remove fixed target, version, and alignment state — complete

- Audited every image and scratch producer. All images target RK3588, every decoder rejects versions other than the
  current one, and every scratch allocation uses 4 KiB alignment. These values were carried as per-object state even
  though no producer could vary them and neither target nor version affected runtime execution.
- Removed `RKTarget`, `RKImage.target`, `RKImage.version`, and `RKScratch.alignment`. The serializer now writes its
  compile-time image version directly, scratch records contain only their variable size, and the runtime applies the
  sole 4 KiB scratch-alignment contract directly. Bumped the serialized ABI to version 33.
- Renderer executable size fell from 4,880 to 4,878 lines; runtime remains 486 lines. From the 10,233-line renderer
  baseline, 5,355 executable lines are gone (52.3%). More importantly, 37 physical lines and three false ABI degrees
  of freedom were removed.
- No CPU numeric semantics or generated DPU task changed. Serialized-image round trips cover the narrowed records.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 68. Delete the duplicate image gather split — complete

- Every mid-program gather already serializes its own physical `after` stage, but `RKImage.gather_after` retained a
  second fallback split in the image header, scratch coloring, image composition, and runtime scheduler. Audited all
  producers and made the few legacy mid-gather constructors write their existing split directly into `RKGather.after`.
- Deleted `RKImage.gather_after` and the corresponding header/runtime fallback logic. Image composition now offsets
  each explicit point, and decode validates every mid-gather point directly. Bumped the serialized ABI to version 34.
- One unit contract had asserted a `gather_after` value even though its image contained no mid-gathers; replacing that
  meaningless state assertion with the actual emitted EW/output contract confirmed the field was not semantic.
- Renderer executable size fell from 4,878 to 4,876 lines; runtime remains 486 lines. From the 10,233-line renderer
  baseline, 5,357 executable lines are gone (52.3%). No CPU numeric semantics or DPU operation changed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 69. Delete the unconstructible host negative-index flag — complete

- Audited every `RKHostAddress` constructor. None enabled `normalize_negative`; negative-index normalization that is
  semantically present in Tinygrad is either executed by the native typed-load path or already represented in the
  address UOps before host materialization. The serialized flag and runtime NumPy branch were therefore unreachable.
- Removed `RKHostAddress.normalize_negative`, its image-record word, decoder validation, and runtime branch. Bumped the
  serialized ABI to version 35.
- Renderer executable size fell from 4,876 to 4,874 lines and runtime fell from 486 to 485 lines. From the 10,233-line
  renderer baseline, 5,359 executable lines are gone (52.4%).
- No CPU numeric semantics were added; this deletes an unused host-side semantic option while retaining bounded raw
  address calculation and movement.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.

### 70. Merge terminal gathers into the physical stage timeline — complete

- `post_gathers` were not a distinct hardware operation: they were raw gathers executed after the final EW stage,
  exactly the point already represented by `RKGather.after == len(ew_ops)`. Moved every specialized and generic image
  producer onto that explicit point and deleted the separate `RKImage` phase.
- Removed the post-gather header count, decoder partition, scratch-coloring pass, image-composition branches, serialized
  field, and runtime submission branch. Existing terminal mid-gathers and former post-gathers now retain their original
  order in one synchronized batch. Bumped the serialized ABI to version 36.
- The unit audit exposed one obsolete distinction: a raw-bitcast image already had two ordinary scratch gathers at the
  final point before its separate output gather. The contract now verifies the terminal ARG movement rather than the
  deleted phase label.
- Renderer executable size fell from 4,874 to 4,869 lines and runtime fell from 485 to 484 lines. From the 10,233-line
  renderer baseline, 5,364 executable lines are gone (52.4%). No CPU numeric semantics or DPU arithmetic changed.

Verification:

- `.venv/bin/python -m pytest test/unit/test_rockchip_uops.py -q -n12`: 91 passed, including all image round trips and
  terminal raw-bit layout contracts.
- `.venv/bin/python -m ruff check .`: pass.
- `.venv/bin/python -m mypy tinygrad/`: 216 source files passed.
