# test_ops.py Status — Rockchip NPU Backend

**Last run:** 2026-07-29 (parent `00f113a15` + two-LUT tanh working tree)
**Command:** `PYTHONPATH=. DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 .venv/bin/python -m pytest test/backend/test_ops.py -q --tb=line`

The run completed in 586.59 seconds. Ordinary and extreme tanh pass in focused
and full-suite execution. The aggregate count is unchanged from the prior
single-process census because NPU state/order changes which otherwise-green
method fails; use the method list, not only the aggregate, when measuring a
milestone.

**Post-census milestone:** `TestOps.test_log2` now passes in isolated execution,
including its float32 infinity/NaN subcase. The summary table below remains the
last complete census rather than mixing an incremental result into it.
`TestOps.test_log` and `TestOps.test_log10` also pass after folding their
base-change scales into the normalized two-LUT path.
`TestOps.test_logsigmoid` now passes through a dedicated 15-task/two-LUT path;
the complete hardware file is at 63 passed with only the two baseline fill
failures.
`TestOps.test_softplus` now passes for beta `1`, `3`, and `1/3`; the complete
hardware file advances to 64 passing methods with the same two fill failures.
`TestOps.test_mish` now passes through a dedicated 45-task/two-LUT path. Its
hardware regression covers a dense `[-2,2]` grid, finite `±8`, positive
infinity, and NaN. The complete hardware file advances to **65 passed,
2 failed** in 169.70 seconds; only fill-zero/fill-full remain.
`TestOps.test_elu` and `TestOps.test_selu` now pass through one shared
55-task/two-LUT path. The complete hardware file advances to **66 passed,
2 failed** in 188.91 seconds. Dense finite inputs, negative infinity, NaN, and
signed zero pass; positive infinity is separately tracked as a final-DPU-ADD
limitation.
`TestOps.test_erf` now passes through a saturated 64-task/two-LUT path,
including both official extreme ranges and scalar. Its 4097-point dense/special
hardware regression passes, advancing the complete hardware file to
**67 passed, 2 failed** in 195.69 seconds.
Both `TestOps.test_gelu` methods now pass for exact and tanh approximations,
including every ±300–400 extreme subcase. Their shared 53-task/two-LUT path
advances the complete hardware file to **68 passed, 2 failed** in 207.94
seconds.

## Summary

| Status | Count |
|--------|-------|
| **Passed** | 140 |
| **Failed** | 276 test functions + 99 subtests = 375 total |
| **Skipped** | 8 |
| **Subtests passed** | 27 |
| **Total test functions** | 424 |

## Failure Breakdown by Error Type

| Error | Count | Description |
|-------|-------|-------------|
| `RKPLAN_REJECT:unsupported_op:Ops.WHERE` | 196 | WHERE op not supported (pow, softmax, scatter, etc.) |
| `RKPLAN_REJECT:unsupported_layout` | 138 | 2D/3D layouts not supported (conv, pool, matmul) |
| `RKPLAN_REJECT:unsupported_dtype` | 58 | fp32/int32 dtype not supported |
| `RKPLAN_REJECT:unsupported_op:fused_epilogue` | 36 | Fused epilogue not supported |
| `TimeoutError` | 21 | Historical census: NPU timeout (sin, tan, sinh, etc.; Softplus and Mish were fixed post-census) |
| `RKPLAN_REJECT:unsupported_op:non_index_operand` | 20 | Non-index operand in store |
| `RKPLAN_REJECT:cmac_exceeds_cbuf` | 18 | CMAC exceeds circular buffer |
| `AssertionError: dtype` | 15 | dtype mismatch (fp16 vs fp32) |
| `RKPLAN_REJECT:unsupported_op:Ops.XOR/OR/AND/SHL/SHR` | 14 | Bitwise ops not supported |
| `AssertionError: cmac` | 12 | CMAC sum classification failed |
| `RKPLAN_REJECT:unsupported_op:Ops.MUL` | 12 | `REDUCE(MUL, ...)` not supported — product/cumprod/argmax/argmin (line 919 only handles ADD and MAX) |
| `RKPLAN_REJECT:unsupported_op:Ops.ADD` | 6 | `REDUCE(ADD, ADD(...))` — reduce body is ADD not MUL/INDEX — cross_entropy/binary_crossentropy/nll_loss (line 901) |

## Failure Categories

### 1. Conv/Pool/MatMul (unsupported_layout) — ~80 tests
All conv2d, conv3d, conv_transpose, pool2d, matmul_batched tests fail with `unsupported_layout`.
These need DPU conv/channel layout support.

**Tests:** test_simple_conv2d, test_padded_conv2d_*, test_strided_conv*, test_max_pool2d_*, test_avg_pool2d*, test_matmul_batched, test_dot, test_einsum, etc.

### 2. WHERE-based ops — ~60 tests
Pow, softmax, scatter, isclose, isinf, isnan, etc. use WHERE which is not supported as a DPU op.

**Tests:** test_pow*, test_softmax*, test_scatter*, test_isclose*, test_isinf, test_isnan, test_nonzero, test_one_hot, etc.

### 3. Reduction ops (sum/max/min/mean/std/var) — ~35 tests
Sum, max, min, mean, std, var, prod, cumsum, cummax, cummin, cumprod all fail.

**Tests:** test_sum*, test_max, test_min, test_mean*, test_std*, test_var*, test_prod, test_cum*, test_argmax, test_argmin, etc.

### 4. Transcendental ops — ~12 tests
Exp, sin, cos, tan, sinh, cosh, sigmoid_extreme.

**Tests:** test_exp, test_sin, test_cos, test_tan, test_sinh, test_cosh, test_sigmoid*

### 5. Bitwise ops — ~8 tests
AND, OR, XOR, SHL, SHR, bitwise_not.

**Tests:** test_and, test_or, test_xor, test_lshift*, test_rshift*, test_bitwise_not, test_int_or

### 6. Interpolate/upsample — ~10 tests
All interpolate variants fail.

**Tests:** test_interpolate_*

### 7. Pad edge cases — ~7 tests
Negative padding, fused pad+reshape, pad modes (circular/reflect/replicate).

**Tests:** test_pad, test_pad_reshape, test_pad_slice, test_padding_add, test_pad_circular_mode, test_pad_reflect_mode, test_pad_replicate_mode

### 8. Cat/multicat — ~2 tests
test_cat (segfault fixed, now fails with wrong results for dim=1+), test_multicat

### 9. Dtype issues — ~15 tests
fp32 output dtype mismatch, int operations.

**Tests:** test_ones, test_zeros, test_arange, test_linspace, test_eye, test_full, test_bitcast

### 10. Other — ~20 tests
test_repeat, test_roll, test_flip_eye_crash, test_diag, test_meshgrid, test_stack, test_sort, test_topk, test_tril, test_triu, test_nonzero, test_gather, test_fancy_indexing*, etc.

## Low-Hanging Fruit (ordered by effort/impact)

| Error | Count | Effort | Impact | Notes |
|-------|-------|--------|--------|-------|
| Bitwise (AND/OR/XOR/SHL/SHR) | 14 | **Low** | Low | Simple EW ops, just need register config like other EW |
| `non_index_operand` | 20 | **Low** | Med | Store pattern recognition — handle more cases like pad did |
| `fused_epilogue` | 36 | **Low-Med** | Med | Split fused ops into separate tasks |
| `unsupported_dtype` | 58 | **Med** | High | Add fp32/int32 conversion paths (already have fp16↔fp32 infra) |
| `TimeoutError` | 21 | **Med** | Low | NPU hangs from wrong register config for transcendentals |
| `cmac_exceeds_cbuf` | 18 | **Med** | Low | Needs tiling/segmentation of large CMAC |
| `cmac classification` | 12 | **Med** | Low | CMAC sum classification logic |
| `Ops.MUL` (REDUCE(MUL)) | 12 | **Med** | Low | Product reduction — needs PPU/CMAC MUL reduce support (line 919 only handles ADD/MAX). Tests: prod, cumprod, argmax, argmin, argsort |
| `Ops.ADD` (REDUCE(ADD,ADD)) | 6 | **Med** | Low | Nested ADD in reduce body — cross_entropy/binary_crossentropy/nll_loss. Needs flattening or multi-stage CMAC (line 901) |
| `Ops.WHERE` | 196 | **High** | **Highest** | Used everywhere — host-side WHERE like pad, or DPU WHERE op |
| `unsupported_layout` | 138 | **High** | High | Conv/pool/matmul 2D/3D DPU layouts — core NPU feature |

**Suggested order:** Bitwise → non_index_operand → fused_epilogue → unsupported_dtype → WHERE → unsupported_layout

The first 3 are quick wins (~70 errors). Then dtype (58 more). WHERE+layout are the big ones but hardest.

## What Works (140 passing)
- Basic EW ops: add, sub, mul, div (1D/2D, same shape)
- Broadcast EW: N-D broadcast (4D/5D) for add/sub/mul/div/max
- Simple pad: 1D/2D/3D/4D positive padding with zero/non-zero values
- Cat dim=0: host-side memmove
- SQRT, EXP2, quick_gelu (fp32 fixes)
- tanh, including ordinary interior precision and extreme/special values
- log2, including exact power-of-four normalization and near-one precision
- natural log and log10, including range reduction and IEEE special values
- LogSigmoid, including dense `[-8,8]` coverage and IEEE special values
- Softplus for beta `1`, `3`, and `1/3`, including dense/special coverage
- Mish on the official method and dense `[-2,2]` interval, using two LUT tasks
- ELU (alpha 1 and 0.1) and SELU, including dense finite/negative-special coverage
- Erf, including dense `[-4,4]`, ±400, infinities, NaN, and zero
- Exact and tanh GELU, including all official extreme ranges
- abs, sign, round, inf_div (fp32 inputs)
- ceil, floor, clip (WHERE CMPNE + fp32)
- relu, sigmoid (basic)
- matmul (simple 2D cases)
- reshape, permute, transpose, slice (basic)
- contiguous, realize
