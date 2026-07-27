# Rockchip NPU backend — test_ops.py progress

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
| RECIPROCAL        | 5     | ~10 lines (route as scalar MUL) |
| fused_epilogue    | 4     | ~15 lines (CMAC bias fusion) |
| EXP2              | 3     | ~10 lines (LUT or DPU) |
| SIN               | 2     | ~10 lines (LUT or DPU) |
| cmac_exceeds_cbuf | 2     | ~15 lines (CMAC tiling) |
| LOG2              | 1     | ~5 lines |
| SQRT              | 1     | ~5 lines |
| TRUNC             | 1     | ~5 lines |
| **Total**         | **67** | **~125 lines** |

WHERE alone would gain 49 tests — more than doubling the current 71 PASS.
Total LHF is 67 tests for ~125 lines of implementation.

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
