# test_ops.py per-test status (FORWARD_ONLY=1)

## Current hardware census — 2026-07-28

Run configuration: `DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1` with
`test.rockchip.conftest_rockchip`, on the RK3588 NPU. The 424 collected methods
were run serially in 20-test subprocess batches because one physical NPU cannot
safely serve 12 concurrent pytest workers. The batch containing methods 400–419
segfaulted, so those methods were rerun individually.

**Current summary: 137 PASS, 279 FAIL, 8 SKIP (424 unique tests).**

This is the previous complete hardware census plus independently rerun LUT
milestones. SQRT, RSQRT, and EXP moved from numeric mismatch to PASS.

Forward-only follow-up: both forward ranges in `test_sigmoid_extreme` now pass,
but the method has two explicit gradient assertions that run even with
`FORWARD_ONLY=1`. Per current scope those assertions are not counted as a
backend regression, and the method remains PARTIAL rather than increasing the
PASS total.

The census originally found one reproducible crash in
`TestOpsUint8::test_cast_relu`. It is now fixed: version-4 task metadata
distinguishes uint8 from int32 output, and the conversion writes one byte per
uint8 element instead of overrunning the allocation with four-byte writes.

| Result group | Count | Main current causes |
|---|---:|---|
| PASS | 137 | Core fp16 arithmetic/casts/fills/rounding, comparisons/predicates/sign, WHERE/clip/abs/minimum/maximum, affine copies, GEMM subsets, selected reductions/activations/LUT special values, and uint8 ReLU cast |
| FAIL: unsupported WHERE | 72 | Remaining WHERE graphs include reductions, padding/index generation, or unsupported operands/layouts |
| FAIL: unsupported dtype | 33 | Remaining bool, fp32, int/uint, and dtype-changing kernels |
| FAIL: unsupported layout | 47 | Broadcast/RANGE, convolution, pooling, batched matmul, and reduction layouts |
| FAIL: numeric mismatch | 25 | Remaining LUT/activation precision, fp16 accumulation/rounding, and special values |
| FAIL: non-index operand | 13 | Elementwise graphs still outside the staged planner |
| FAIL: fused epilogue | 13 | Convolution/reduction output stages |
| FAIL: dtype mismatch | 12 | Incorrect result dtype or special-value representation |
| FAIL: CBUF limit | 9 | Large reductions/variance and one convolution |
| FAIL: other | 55 | Other unsupported ops, assertions, layouts, and framework-side failures |
| SKIP | 8 | Upstream slow/redundant/broken/platform-specific skips |

The 137 passing methods are:

`test_9_gemm`, `test_abs`, `test_abs_exact`, `test_add`, `test_add3`, `test_all_zero_axis`, `test_any_zero_axis`,
`test_arange_4096`, `test_arange_big`,
`test_big_gemm`, `test_broadcastdot`, `test_cast`, `test_ceil`, `test_chunk`, `test_clip`,
`test_cmp_eq`, `test_cmp_ge`, `test_cmp_gt`, `test_cmp_le`, `test_cmp_lt`,
`test_conv2d_errors`, `test_cummax_zero_axis`, `test_cummin_zero_axis`,
`test_cumprod_zero_axis`, `test_cumsum_zero_axis`, `test_detach`,
`test_diagonal`, `test_div`, `test_double_slice`, `test_einsum_arity_check1`,
`test_einsum_arity_check2`, `test_einsum_shape_check`, `test_empty_0`,
`test_expand`, `test_exp`, `test_exp2`, `test_exp2_log2_zero_times_negative`, `test_eye`, `test_flatten`, `test_flip`, `test_flip_eye_crash`, `test_floor`,
`test_full`, `test_full_like`,
`test_gemm`, `test_gemm_fp16`, `test_gemm_with_zeros_shape`,
`test_hardsigmoid`, `test_hardtanh`, `test_idiv_shift_rewrite_negative`,
`test_inf_where`, `test_isfinite`, `test_isinf`, `test_isnan`, `test_leaky_relu`,
`test_logical_not`, `test_matmul`, `test_matmul_simple`, `test_matvec`,
`test_matvecmat`, `test_mean`, `test_mean_zero_axis`, `test_meshgrid`,
`test_masked_fill`, `test_maximum`, `test_minimum`, `test_mul`, `test_mul_naninf`, `test_neg`, `test_negative_dims`,
`test_negative_dims_eye`, `test_negative_dims_full`,
`test_negative_dims_kaiming`, `test_ones`, `test_ones_like`, `test_permute`,
`test_prod_dtype_arg`, `test_relu`, `test_relu6`, `test_relu_exact`,
`test_relu_maximum_exact`, `test_reshape`, `test_round`, `test_rsqrt`, `test_scalar_div`,
`test_scalar_mul`, `test_scalar_rsub`, `test_scalar_sub`,
`test_scaled_dot_product_attention_gqa_errors`,
`test_scatter_no_reduce_tensor_src`, `test_sigmoid`,
`test_sign`, `test_sign_exact`,
`test_simple_conv2d_1x1`, `test_slice_both_endpoints_out_of_bounds`,
`test_slice_ellipsis`, `test_slice_errors`, `test_slice_in_bounds_1dim`,
`test_slice_in_bounds_multidim`, `test_slice_int_indexing`,
`test_slice_negative_strides`, `test_slice_on_0dim_tensor`,
`test_slice_one_endpoint_out_of_bounds`, `test_slice_start_gt_end`,
`test_slice_stride_gt_one`, `test_slice_with_none`, `test_slice_zero_in_shape`,
`test_silu`, `test_small_gemm`, `test_split`, `test_squeeze`, `test_stack_slice`,
`test_sqrt`,
`test_std_mean_loaded_nan`, `test_std_zero_in_axis`, `test_sub`,
`test_sum_collapse_neg`, `test_sum_fake`, `test_sum_simple`,
`test_sum_with_zeros_shape`, `test_tiny_add`, `test_tiny_mul`,
`test_topo_sort`, `test_transpose`, `test_trunc`, `test_unflatten`, `test_unfold`,
`test_unsqueeze`, `test_var_zero_in_axis`, `test_view`, `test_where`,
`test_where_permute`, `test_swish`, `test_zeros`, `test_zeros_like`, `TestOpsUint8::test_cast`, and
`TestOpsUint8::test_cast_relu`.

Current nested-LOG2 milestone regression: all **53 hardware tests pass in
isolated sequential subprocesses**, including EXP2, LOG2, sigmoid, SQRT, and
RSQRT special-value assertions. A single-process run retains the
sequence-sensitive SiLU→SUB timeout; both tests pass in isolation. `lut.md`
records the LUT tuning, range reduction, Newton refinement, and special-value
procedures plus the remaining SiLU one-ULP dense-grid diagnostic.

## Historical detailed baseline — 2026-07-27, commit 993ea1197

The per-test table below is retained for failure-shape and subcase reference.
Its aggregate is the pre-milestone baseline, not the current census.

**Historical summary:** 71 PASS, 21 PARTIAL, 324 FAIL, 8 SKIP.

**Low-hanging fruit:** 67 tests with single fixable reason

| # | ID | Test | Class | Status | P/F/S | Reason | Ops | Category | LHF |
|---|---|------|-------|--------|-------|--------|-----|----------|-----|
| 1 | TestOps::test_9_gemm | test_9_gemm | TestOps | PASS | 1/0/0 |  |  |  |  |
| 2 | TestOps::test_abs | test_abs | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 3 | TestOps::test_abs_exact | test_abs_exact | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 4 | TestOps::test_acos | test_acos | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 5 | TestOps::test_acosh | test_acosh | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 6 | TestOps::test_add | test_add | TestOps | PASS | 1/0/0 |  |  |  |  |
| 7 | TestOps::test_add3 | test_add3 | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 8 | TestOps::test_all | test_all | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 9 | TestOps::test_all_axis | test_all_axis | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 10 | TestOps::test_all_large | test_all_large | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 11 | TestOps::test_all_zero_axis | test_all_zero_axis | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 12 | TestOps::test_and | test_and | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 13 | TestOps::test_any | test_any | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 14 | TestOps::test_any_axis | test_any_axis | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 15 | TestOps::test_any_zero_axis | test_any_zero_axis | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 16 | TestOps::test_arange | test_arange | TestOps | FAIL | 0/1/0 | forward_pass_failed |  | forward_pass_failed |  |
| 17 | TestOps::test_arange_4096 | test_arange_4096 | TestOps | PASS | 1/0/0 |  |  |  |  |
| 18 | TestOps::test_arange_big | test_arange_big | TestOps | PASS | 1/0/0 |  |  |  |  |
| 19 | TestOps::test_argmax | test_argmax | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 20 | TestOps::test_argmin | test_argmin | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 21 | TestOps::test_argsort | test_argsort | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 22 | TestOps::test_asin | test_asin | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 23 | TestOps::test_asinh | test_asinh | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 24 | TestOps::test_asymmetric_padding_conv1d | test_asymmetric_padding_conv1d | TestOps | PARTIAL | 1/3/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 25 | TestOps::test_asymmetric_padding_conv2d | test_asymmetric_padding_conv2d | TestOps | PARTIAL | 1/3/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 26 | TestOps::test_atan | test_atan | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 27 | TestOps::test_atanh | test_atanh | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 28 | TestOps::test_avg_pool2d | test_avg_pool2d | TestOps | FAIL | 0/6/0 | no_add_mul_reduction,unsupported_op:Ops.WHERE | WHERE | WHERE,no_add_mul_reduction |  |
| 29 | TestOps::test_avg_pool2d_asymmetric_padding | test_avg_pool2d_asymmetric_padding | TestOps | PARTIAL | 1/3/0 | unsupported_op:Ops.WHERE | WHERE | WHERE |  |
| 30 | TestOps::test_avg_pool2d_ceil_mode | test_avg_pool2d_ceil_mode | TestOps | PARTIAL | 1/3/0 | unsupported_op:Ops.WHERE | WHERE | WHERE |  |
| 31 | TestOps::test_avg_pool2d_ceil_mode_include_pad_output_size_reduce_by_one | test_avg_pool2d_ceil_mode_include_pad_output_size_reduce_by_one | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 32 | TestOps::test_avg_pool2d_ceil_mode_output_size_reduce_by_one | test_avg_pool2d_ceil_mode_output_size_reduce_by_one | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 33 | TestOps::test_avg_pool2d_ceil_mode_padding_not_counted | test_avg_pool2d_ceil_mode_padding_not_counted | TestOps | PARTIAL | 1/3/0 | unsupported_op:Ops.WHERE | WHERE | WHERE |  |
| 34 | TestOps::test_avg_pool2d_padding | test_avg_pool2d_padding | TestOps | PARTIAL | 1/9/0 | unsupported_op:Ops.WHERE | WHERE | WHERE |  |
| 35 | TestOps::test_avg_pool2d_padding_not_counted | test_avg_pool2d_padding_not_counted | TestOps | PARTIAL | 1/3/0 | unsupported_op:Ops.WHERE | WHERE | WHERE |  |
| 36 | TestOps::test_avg_pool3d | test_avg_pool3d | TestOps | FAIL | 0/1/0 | not_implemented |  | not_implemented |  |
| 37 | TestOps::test_bias_conv_transpose2d | test_bias_conv_transpose2d | TestOps | FAIL | 0/1/0 | unsupported_op:fused_epilogue |  | fused_epilogue | Y |
| 38 | TestOps::test_biased_conv2d | test_biased_conv2d | TestOps | FAIL | 0/1/0 | unsupported_op:fused_epilogue |  | fused_epilogue | Y |
| 39 | TestOps::test_big_gemm | test_big_gemm | TestOps | PASS | 1/0/0 |  |  |  |  |
| 40 | TestOps::test_binary_crossentropy | test_binary_crossentropy | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.ADD | ADD | ADD_in_reduce |  |
| 41 | TestOps::test_binary_crossentropy_logits_pos_weights | test_binary_crossentropy_logits_pos_weights | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.ADD | ADD | ADD_in_reduce |  |
| 42 | TestOps::test_binary_crossentropy_reductions | test_binary_crossentropy_reductions | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.ADD | ADD | ADD_in_reduce |  |
| 43 | TestOps::test_bitcast | test_bitcast | TestOps | FAIL | 0/1/0 | half_view_int |  | half_view_int |  |
| 44 | TestOps::test_bitwise_not | test_bitwise_not | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 45 | TestOps::test_broadcast_full | test_broadcast_full | TestOps | PARTIAL | 1/10/0 | unsupported_layout:Ops.ADD,unsupported_op:Ops.WHERE,unsupported_op:non_index_operand | ADD,WHERE | WHERE,broadcast,nested_EW |  |
| 46 | TestOps::test_broadcast_partial | test_broadcast_partial | TestOps | PARTIAL | 1/20/0 | unsupported_layout:Ops.ADD,unsupported_layout:Ops.RANGE,unsupported_op:Ops.WHERE,unsupported_op:non_index_operand | ADD,RANGE,WHERE | WHERE,broadcast,nested_EW |  |
| 47 | TestOps::test_broadcast_simple | test_broadcast_simple | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 48 | TestOps::test_broadcastdot | test_broadcastdot | TestOps | PASS | 1/0/0 |  |  |  |  |
| 49 | TestOps::test_broadcasted_add | test_broadcasted_add | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.RANGE | RANGE | broadcast |  |
| 50 | TestOps::test_broadcasted_add_2 | test_broadcasted_add_2 | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.RANGE | RANGE | broadcast |  |
| 51 | TestOps::test_cast | test_cast | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 52 | TestOps::test_cat | test_cat | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 53 | TestOps::test_ceil | test_ceil | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 54 | TestOps::test_celu | test_celu | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 55 | TestOps::test_chunk | test_chunk | TestOps | PASS | 1/0/0 |  |  |  |  |
| 56 | TestOps::test_clip | test_clip | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 57 | TestOps::test_cmp_eq | test_cmp_eq | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 58 | TestOps::test_cmp_ge | test_cmp_ge | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 59 | TestOps::test_cmp_gt | test_cmp_gt | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 60 | TestOps::test_cmp_le | test_cmp_le | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 61 | TestOps::test_cmp_lt | test_cmp_lt | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 62 | TestOps::test_cmp_lt_backwards | test_cmp_lt_backwards | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 63 | TestOps::test_cmp_ne_backwards | test_cmp_ne_backwards | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 64 | TestOps::test_const_reduce | test_const_reduce | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 65 | TestOps::test_conv1d | test_conv1d | TestOps | PARTIAL | 2/13/0 | unsupported_layout,unsupported_layout:Ops.ADD,unsupported_layout:Ops.RANGE | ADD,RANGE | broadcast,layout |  |
| 66 | TestOps::test_conv2d | test_conv2d | TestOps | PARTIAL | 1/7/0 | unsupported_layout |  | layout |  |
| 67 | TestOps::test_conv2d_bs_1_cin_1 | test_conv2d_bs_1_cin_1 | TestOps | PARTIAL | 1/6/0 | unsupported_layout |  | layout |  |
| 68 | TestOps::test_conv2d_bs_4_cin_1 | test_conv2d_bs_4_cin_1 | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 69 | TestOps::test_conv2d_bs_4_cin_3 | test_conv2d_bs_4_cin_3 | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 70 | TestOps::test_conv2d_errors | test_conv2d_errors | TestOps | PASS | 1/0/0 |  |  |  |  |
| 71 | TestOps::test_copysign | test_copysign | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 72 | TestOps::test_copysign_exact | test_copysign_exact | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 73 | TestOps::test_cos | test_cos | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.SIN | SIN | SIN | Y |
| 74 | TestOps::test_cosh | test_cosh | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 75 | TestOps::test_cross_entropy_class_indices | test_cross_entropy_class_indices | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 76 | TestOps::test_cross_entropy_class_probabilities | test_cross_entropy_class_probabilities | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 77 | TestOps::test_cross_entropy_reductions | test_cross_entropy_reductions | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 78 | TestOps::test_cross_entropy_smoothing | test_cross_entropy_smoothing | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 79 | TestOps::test_cummax | test_cummax | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 80 | TestOps::test_cummax_zero_axis | test_cummax_zero_axis | TestOps | PASS | 1/0/0 |  |  |  |  |
| 81 | TestOps::test_cummin | test_cummin | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 82 | TestOps::test_cummin_zero_axis | test_cummin_zero_axis | TestOps | PASS | 1/0/0 |  |  |  |  |
| 83 | TestOps::test_cumprod | test_cumprod | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 84 | TestOps::test_cumprod_zero_axis | test_cumprod_zero_axis | TestOps | PASS | 1/0/0 |  |  |  |  |
| 85 | TestOps::test_cumsum | test_cumsum | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 86 | TestOps::test_cumsum_zero_axis | test_cumsum_zero_axis | TestOps | PASS | 1/0/0 |  |  |  |  |
| 87 | TestOps::test_depthwise_conv2d | test_depthwise_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.RANGE | RANGE | broadcast |  |
| 88 | TestOps::test_detach | test_detach | TestOps | PASS | 1/0/0 |  |  |  |  |
| 89 | TestOps::test_diag | test_diag | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 90 | TestOps::test_diagonal | test_diagonal | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.MUL | MUL | broadcast |  |
| 91 | TestOps::test_dilated_conv2d | test_dilated_conv2d | TestOps | PARTIAL | 1/2/0 | unsupported_layout |  | layout |  |
| 92 | TestOps::test_dilated_conv_transpose2d | test_dilated_conv_transpose2d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 93 | TestOps::test_div | test_div | TestOps | PASS | 1/0/0 |  |  | nested_EW |  |
| 94 | TestOps::test_div_int | test_div_int | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 95 | TestOps::test_div_naninf | test_div_naninf | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 96 | TestOps::test_div_rounding_mode | test_div_rounding_mode | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 97 | TestOps::test_dot | test_dot | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 98 | TestOps::test_dot_1d | test_dot_1d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 99 | TestOps::test_double_slice | test_double_slice | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 100 | TestOps::test_einsum | test_einsum | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 101 | TestOps::test_einsum_arity_check1 | test_einsum_arity_check1 | TestOps | PASS | 1/0/0 |  |  |  |  |
| 102 | TestOps::test_einsum_arity_check2 | test_einsum_arity_check2 | TestOps | PASS | 1/0/0 |  |  |  |  |
| 103 | TestOps::test_einsum_ellipsis | test_einsum_ellipsis | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 104 | TestOps::test_einsum_shape_check | test_einsum_shape_check | TestOps | PASS | 1/0/0 |  |  |  |  |
| 105 | TestOps::test_einsum_trace | test_einsum_trace | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 106 | TestOps::test_elu | test_elu | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 107 | TestOps::test_empty_0 | test_empty_0 | TestOps | PASS | 1/0/0 |  |  |  |  |
| 108 | TestOps::test_erf | test_erf | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 109 | TestOps::test_exp | test_exp | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.EXP2 | EXP2 | EXP2 | Y |
| 110 | TestOps::test_exp2 | test_exp2 | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.EXP2 | EXP2 | EXP2 | Y |
| 111 | TestOps::test_exp2_log2_zero_times_negative | test_exp2_log2_zero_times_negative | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.EXP2 | EXP2 | EXP2 | Y |
| 112 | TestOps::test_expand | test_expand | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 113 | TestOps::test_eye | test_eye | TestOps | PASS | 1/0/0 |  |  |  |  |
| 114 | TestOps::test_fancy_conv2d | test_fancy_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 115 | TestOps::test_fancy_indexing_inf | test_fancy_indexing_inf | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 116 | TestOps::test_flatten | test_flatten | TestOps | PASS | 1/0/0 |  |  |  |  |
| 117 | TestOps::test_flip | test_flip | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 118 | TestOps::test_flip_eye_crash | test_flip_eye_crash | TestOps | PASS | 1/0/0 |  |  |  |  |
| 119 | TestOps::test_floor | test_floor | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 120 | TestOps::test_fmod | test_fmod | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 121 | TestOps::test_full | test_full | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 122 | TestOps::test_full_like | test_full_like | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 123 | TestOps::test_gather | test_gather | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 124 | TestOps::test_gelu | test_gelu | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 125 | TestOps::test_gelu_extreme | test_gelu_extreme | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 126 | TestOps::test_gemm | test_gemm | TestOps | PASS | 1/0/0 |  |  |  |  |
| 127 | TestOps::test_gemm_fp16 | test_gemm_fp16 | TestOps | PASS | 1/0/0 |  |  |  |  |
| 128 | TestOps::test_gemm_with_zeros_shape | test_gemm_with_zeros_shape | TestOps | PASS | 1/0/0 |  |  |  |  |
| 129 | TestOps::test_global_avg_pool2d | test_global_avg_pool2d | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 130 | TestOps::test_grouped_conv2d | test_grouped_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 131 | TestOps::test_grouped_conv_transpose2d | test_grouped_conv_transpose2d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 132 | TestOps::test_hardsigmoid | test_hardsigmoid | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 133 | TestOps::test_hardsigmoid_extreme | test_hardsigmoid_extreme | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 134 | TestOps::test_hardswish | test_hardswish | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 135 | TestOps::test_hardtanh | test_hardtanh | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 136 | TestOps::test_idiv_shift_rewrite_negative | test_idiv_shift_rewrite_negative | TestOps | PASS | 1/0/0 |  |  |  |  |
| 137 | TestOps::test_inf_where | test_inf_where | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 138 | TestOps::test_int_or | test_int_or | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 139 | TestOps::test_int_pow_const_int | test_int_pow_const_int | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 140 | TestOps::test_interpolate_bilinear | test_interpolate_bilinear | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 141 | TestOps::test_interpolate_bilinear_corners_aligned | test_interpolate_bilinear_corners_aligned | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 142 | TestOps::test_interpolate_linear | test_interpolate_linear | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 143 | TestOps::test_interpolate_linear_corners_aligned | test_interpolate_linear_corners_aligned | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 144 | TestOps::test_interpolate_nearest | test_interpolate_nearest | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 145 | TestOps::test_interpolate_nearest_exact | test_interpolate_nearest_exact | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 146 | TestOps::test_interpolate_trilinear | test_interpolate_trilinear | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 147 | TestOps::test_interpolate_trilinear_corners_aligned | test_interpolate_trilinear_corners_aligned | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 148 | TestOps::test_isclose | test_isclose | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 149 | TestOps::test_isclose_edge_cases | test_isclose_edge_cases | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 150 | TestOps::test_isclose_scalar | test_isclose_scalar | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 151 | TestOps::test_isfinite | test_isfinite | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 152 | TestOps::test_isinf | test_isinf | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 153 | TestOps::test_isnan | test_isnan | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 154 | TestOps::test_large_bs_conv | test_large_bs_conv | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 155 | TestOps::test_large_ic_conv | test_large_ic_conv | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 156 | TestOps::test_large_input_conv2d | test_large_input_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 157 | TestOps::test_leaky_relu | test_leaky_relu | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 158 | TestOps::test_lerp | test_lerp | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 159 | TestOps::test_linspace | test_linspace | TestOps | FAIL | 0/1/0 | forward_pass_failed |  | forward_pass_failed |  |
| 160 | TestOps::test_log | test_log | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 161 | TestOps::test_log10 | test_log10 | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 162 | TestOps::test_log2 | test_log2 | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.LOG2 | LOG2 | LOG2 | Y |
| 163 | TestOps::test_log_softmax | test_log_softmax | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 164 | TestOps::test_log_softmax_other_axis | test_log_softmax_other_axis | TestOps | FAIL | 0/1/0 | unsupported_layout:(100,):10 |  | layout |  |
| 165 | TestOps::test_logaddexp | test_logaddexp | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 166 | TestOps::test_logcumsumexp | test_logcumsumexp | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 167 | TestOps::test_logcumsumexp_numerical | test_logcumsumexp_numerical | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 168 | TestOps::test_logical_not | test_logical_not | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 169 | TestOps::test_logsigmoid | test_logsigmoid | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 170 | TestOps::test_logsumexp | test_logsumexp | TestOps | FAIL | 0/1/0 | unsupported_layout:(65,):45 |  | layout |  |
| 171 | TestOps::test_lshift | test_lshift | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 172 | TestOps::test_lshift_signed | test_lshift_signed | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 173 | TestOps::test_masked_fill | test_masked_fill | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 174 | TestOps::test_masked_select | test_masked_select | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 175 | TestOps::test_masked_select_size | test_masked_select_size | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 176 | TestOps::test_matmul | test_matmul | TestOps | PASS | 1/0/0 |  |  |  |  |
| 177 | TestOps::test_matmul_batched | test_matmul_batched | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 178 | TestOps::test_matmul_batched_vector | test_matmul_batched_vector | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 179 | TestOps::test_matmul_simple | test_matmul_simple | TestOps | PASS | 1/0/0 |  |  |  |  |
| 180 | TestOps::test_matvec | test_matvec | TestOps | FAIL | 0/1/0 | forward_pass_failed |  | forward_pass_failed |  |
| 181 | TestOps::test_matvecmat | test_matvecmat | TestOps | FAIL | 0/1/0 | forward_pass_failed |  | forward_pass_failed |  |
| 182 | TestOps::test_max | test_max | TestOps | FAIL | 0/1/0 | unsupported_layout:(1,):135 |  | layout |  |
| 183 | TestOps::test_max_dont_collapse | test_max_dont_collapse | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 184 | TestOps::test_max_nan | test_max_nan | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 185 | TestOps::test_max_pool2d | test_max_pool2d | TestOps | PARTIAL | 1/5/0 | unsupported_layout:(64, |  | layout |  |
| 186 | TestOps::test_max_pool2d_asymmetric_padding | test_max_pool2d_asymmetric_padding | TestOps | PARTIAL | 1/3/0 | unsupported_layout:(8,,unsupported_op:Ops.WHERE | WHERE | WHERE,layout |  |
| 187 | TestOps::test_max_pool2d_bigger_stride | test_max_pool2d_bigger_stride | TestOps | PARTIAL | 1/4/0 | unsupported_layout:(8, |  | layout |  |
| 188 | TestOps::test_max_pool2d_bigger_stride_dilation | test_max_pool2d_bigger_stride_dilation | TestOps | PARTIAL | 1/5/0 | unsupported_layout:(8, |  | layout |  |
| 189 | TestOps::test_max_pool2d_ceil_mode | test_max_pool2d_ceil_mode | TestOps | PARTIAL | 1/3/0 | unsupported_op:Ops.WHERE | WHERE | WHERE |  |
| 190 | TestOps::test_max_pool2d_ceil_mode_output_size_reduce_by_one | test_max_pool2d_ceil_mode_output_size_reduce_by_one | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 191 | TestOps::test_max_pool2d_dilation | test_max_pool2d_dilation | TestOps | FAIL | 0/1/0 | unsupported_layout:(6, |  | layout |  |
| 192 | TestOps::test_max_pool2d_padding | test_max_pool2d_padding | TestOps | PARTIAL | 1/9/0 | unsupported_op:Ops.WHERE | WHERE | WHERE |  |
| 193 | TestOps::test_max_pool2d_padding_int | test_max_pool2d_padding_int | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 194 | TestOps::test_max_pool2d_return_indices | test_max_pool2d_return_indices | TestOps | FAIL | 0/1/0 | unsupported_layout:(18, |  | layout |  |
| 195 | TestOps::test_max_pool2d_simple | test_max_pool2d_simple | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 196 | TestOps::test_max_pool2d_smaller_stride | test_max_pool2d_smaller_stride | TestOps | PARTIAL | 1/4/0 | unsupported_layout:(6, |  | layout |  |
| 197 | TestOps::test_max_pool2d_unit_stride | test_max_pool2d_unit_stride | TestOps | FAIL | 0/1/0 | unsupported_layout:(6, |  | layout |  |
| 198 | TestOps::test_max_unpool2d | test_max_unpool2d | TestOps | FAIL | 0/1/0 | unsupported_layout:(24, |  | layout |  |
| 199 | TestOps::test_max_unpool2d_inf | test_max_unpool2d_inf | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 200 | TestOps::test_maximum | test_maximum | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 201 | TestOps::test_mean | test_mean | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 202 | TestOps::test_mean_axis | test_mean_axis | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 203 | TestOps::test_mean_zero_axis | test_mean_zero_axis | TestOps | PASS | 1/0/0 |  |  |  |  |
| 204 | TestOps::test_medium_grouped_conv2d | test_medium_grouped_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 205 | TestOps::test_meshgrid | test_meshgrid | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 206 | TestOps::test_min | test_min | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 207 | TestOps::test_minimum | test_minimum | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 208 | TestOps::test_mish | test_mish | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 209 | TestOps::test_mod | test_mod | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 210 | TestOps::test_mul | test_mul | TestOps | PASS | 1/0/0 |  |  |  |  |
| 211 | TestOps::test_mul_naninf | test_mul_naninf | TestOps | PASS | 1/0/0 |  |  |  |  |
| 212 | TestOps::test_mulacc_with_zero_strides | test_mulacc_with_zero_strides | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 213 | TestOps::test_multicat | test_multicat | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 214 | TestOps::test_multidot | test_multidot | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 215 | TestOps::test_neg | test_neg | TestOps | PASS | 1/0/0 |  |  |  |  |
| 216 | TestOps::test_negative_dims | test_negative_dims | TestOps | PASS | 1/0/0 |  |  |  |  |
| 217 | TestOps::test_negative_dims_eye | test_negative_dims_eye | TestOps | PASS | 1/0/0 |  |  |  |  |
| 218 | TestOps::test_negative_dims_full | test_negative_dims_full | TestOps | PASS | 1/0/0 |  |  |  |  |
| 219 | TestOps::test_negative_dims_kaiming | test_negative_dims_kaiming | TestOps | PASS | 1/0/0 |  |  |  |  |
| 220 | TestOps::test_negative_padding_conv2d | test_negative_padding_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 221 | TestOps::test_nested_conv2d | test_nested_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 222 | TestOps::test_nll_loss | test_nll_loss | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 223 | TestOps::test_nll_loss_3d | test_nll_loss_3d | TestOps | FAIL | 0/1/0 | unsupported_layout:(32, |  | layout |  |
| 224 | TestOps::test_nll_loss_3d_weight | test_nll_loss_3d_weight | TestOps | FAIL | 0/1/0 | unsupported_layout:(16, |  | layout |  |
| 225 | TestOps::test_nll_loss_ignore_index | test_nll_loss_ignore_index | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 226 | TestOps::test_nll_loss_reductions | test_nll_loss_reductions | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 227 | TestOps::test_nll_loss_weight | test_nll_loss_weight | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 228 | TestOps::test_nonzero | test_nonzero | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 229 | TestOps::test_nonzero_size | test_nonzero_size | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 230 | TestOps::test_normalize | test_normalize | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 231 | TestOps::test_one_hot | test_one_hot | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 232 | TestOps::test_ones | test_ones | TestOps | PASS | 1/0/0 |  |  |  |  |
| 233 | TestOps::test_ones_like | test_ones_like | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 234 | TestOps::test_or | test_or | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 235 | TestOps::test_output_padded_conv_transpose2d | test_output_padded_conv_transpose2d | TestOps | FAIL | 0/1/0 | unsupported_op:fused_epilogue |  | fused_epilogue | Y |
| 236 | TestOps::test_pad | test_pad | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 237 | TestOps::test_pad_circular_mode | test_pad_circular_mode | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 238 | TestOps::test_pad_reflect_mode | test_pad_reflect_mode | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 239 | TestOps::test_pad_replicate_mode | test_pad_replicate_mode | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 240 | TestOps::test_pad_reshape | test_pad_reshape | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 241 | TestOps::test_pad_slice | test_pad_slice | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 242 | TestOps::test_padded_conv2d_1x1 | test_padded_conv2d_1x1 | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 243 | TestOps::test_padded_conv2d_bs1 | test_padded_conv2d_bs1 | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 244 | TestOps::test_padded_conv2d_p21 | test_padded_conv2d_p21 | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 245 | TestOps::test_padded_conv2d_p22 | test_padded_conv2d_p22 | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 246 | TestOps::test_padded_conv3d | test_padded_conv3d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 247 | TestOps::test_padded_conv_transpose2d | test_padded_conv_transpose2d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 248 | TestOps::test_padding_add | test_padding_add | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 249 | TestOps::test_permute | test_permute | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 250 | TestOps::test_pow | test_pow | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 251 | TestOps::test_pow_const | test_pow_const | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.RECIPROCAL | RECIPROCAL | RECIPROCAL | Y |
| 252 | TestOps::test_pow_const_direct | test_pow_const_direct | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 253 | TestOps::test_pow_full | test_pow_full | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 254 | TestOps::test_pow_int | test_pow_int | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 255 | TestOps::test_pow_int_base_float_exponent | test_pow_int_base_float_exponent | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 256 | TestOps::test_pow_zero_const | test_pow_zero_const | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 257 | TestOps::test_pow_zero_tensor | test_pow_zero_tensor | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 258 | TestOps::test_prod | test_prod | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 259 | TestOps::test_prod_dtype_arg | test_prod_dtype_arg | TestOps | PASS | 1/0/0 |  |  |  |  |
| 260 | TestOps::test_quick_gelu | test_quick_gelu | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 261 | TestOps::test_quick_gelu_extreme | test_quick_gelu_extreme | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 262 | TestOps::test_relu | test_relu | TestOps | PASS | 1/0/0 |  |  |  |  |
| 263 | TestOps::test_relu6 | test_relu6 | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 264 | TestOps::test_relu_exact | test_relu_exact | TestOps | PASS | 1/0/0 |  |  |  |  |
| 265 | TestOps::test_relu_maximum_exact | test_relu_maximum_exact | TestOps | PASS | 1/0/0 |  |  |  |  |
| 266 | TestOps::test_repeat | test_repeat | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 267 | TestOps::test_repeat_interleave | test_repeat_interleave | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.FLOORDIV | FLOORDIV | layout_floordiv_mod |  |
| 268 | TestOps::test_reshape | test_reshape | TestOps | PASS | 1/0/0 |  |  |  |  |
| 269 | TestOps::test_roll | test_roll | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.FLOORMOD | FLOORMOD | layout_floordiv_mod |  |
| 270 | TestOps::test_round | test_round | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 271 | TestOps::test_round_quantization_gradient | test_round_quantization_gradient | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 272 | TestOps::test_rshift | test_rshift | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 273 | TestOps::test_rshift_signed | test_rshift_signed | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 274 | TestOps::test_rsqrt | test_rsqrt | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.RECIPROCAL | RECIPROCAL | RECIPROCAL | Y |
| 275 | TestOps::test_scalar_div | test_scalar_div | TestOps | PASS | 1/0/0 |  |  | RECIPROCAL | Y |
| 276 | TestOps::test_scalar_mul | test_scalar_mul | TestOps | PASS | 1/0/0 |  |  |  |  |
| 277 | TestOps::test_scalar_rsub | test_scalar_rsub | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 278 | TestOps::test_scalar_sub | test_scalar_sub | TestOps | PASS | 1/0/0 |  |  |  |  |
| 279 | TestOps::test_scaled_dot_product_attention | test_scaled_dot_product_attention | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 280 | TestOps::test_scaled_dot_product_attention_causal | test_scaled_dot_product_attention_causal | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 281 | TestOps::test_scaled_dot_product_attention_gqa | test_scaled_dot_product_attention_gqa | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 282 | TestOps::test_scaled_dot_product_attention_gqa_errors | test_scaled_dot_product_attention_gqa_errors | TestOps | PASS | 1/0/0 |  |  |  |  |
| 283 | TestOps::test_scaled_dot_product_attention_mismatch_ls | test_scaled_dot_product_attention_mismatch_ls | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 284 | TestOps::test_scatter | test_scatter | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 285 | TestOps::test_scatter_add | test_scatter_add | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 286 | TestOps::test_scatter_mul | test_scatter_mul | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 287 | TestOps::test_scatter_no_reduce_tensor_src | test_scatter_no_reduce_tensor_src | TestOps | PASS | 1/0/0 |  |  |  |  |
| 288 | TestOps::test_scatter_reduce | test_scatter_reduce | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 289 | TestOps::test_scatter_reduce_errors | test_scatter_reduce_errors | TestOps | PASS | 1/0/0 |  |  | assertion |  |
| 290 | TestOps::test_scatter_reduce_prod_zeros | test_scatter_reduce_prod_zeros | TestOps | FAIL | 0/1/0 | dtype_mismatch |  | dtype_mismatch |  |
| 291 | TestOps::test_sd_big_conv | test_sd_big_conv | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 292 | TestOps::test_selu | test_selu | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 293 | TestOps::test_sigmoid | test_sigmoid | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.RECIPROCAL | RECIPROCAL | RECIPROCAL | Y |
| 294 | TestOps::test_sigmoid_alt_extreme | test_sigmoid_alt_extreme | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 295 | TestOps::test_sigmoid_extreme | test_sigmoid_extreme | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.RECIPROCAL | RECIPROCAL | RECIPROCAL | Y |
| 296 | TestOps::test_sign | test_sign | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 297 | TestOps::test_sign_exact | test_sign_exact | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 298 | TestOps::test_silu | test_silu | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 299 | TestOps::test_simple_conv2d | test_simple_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 300 | TestOps::test_simple_conv2d_1x1 | test_simple_conv2d_1x1 | TestOps | PASS | 1/0/0 |  |  |  |  |
| 301 | TestOps::test_simple_conv2d_1x1_m4 | test_simple_conv2d_1x1_m4 | TestOps | FAIL | 0/1/0 | cmac_exceeds_cbuf |  | cmac_exceeds_cbuf | Y |
| 302 | TestOps::test_simple_conv2d_batched | test_simple_conv2d_batched | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 303 | TestOps::test_simple_conv2d_bias | test_simple_conv2d_bias | TestOps | FAIL | 0/1/0 | unsupported_op:fused_epilogue |  | fused_epilogue | Y |
| 304 | TestOps::test_simple_conv2d_m4 | test_simple_conv2d_m4 | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 305 | TestOps::test_simple_conv2d_nhwc | test_simple_conv2d_nhwc | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 306 | TestOps::test_simple_conv3d | test_simple_conv3d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 307 | TestOps::test_simple_conv_transpose2d | test_simple_conv_transpose2d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 308 | TestOps::test_simple_conv_transpose3d | test_simple_conv_transpose3d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 309 | TestOps::test_simple_cummax | test_simple_cummax | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 310 | TestOps::test_simple_cummin | test_simple_cummin | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 311 | TestOps::test_simple_cumprod | test_simple_cumprod | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 312 | TestOps::test_simple_cumsum | test_simple_cumsum | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 313 | TestOps::test_simple_grouped_conv2d | test_simple_grouped_conv2d | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 314 | TestOps::test_simple_padding_conv1d | test_simple_padding_conv1d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 315 | TestOps::test_simple_padding_conv2d | test_simple_padding_conv2d | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 316 | TestOps::test_simple_repeat | test_simple_repeat | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 317 | TestOps::test_sin | test_sin | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.SIN | SIN | SIN | Y |
| 318 | TestOps::test_sinh | test_sinh | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 319 | TestOps::test_slice_both_endpoints_out_of_bounds | test_slice_both_endpoints_out_of_bounds | TestOps | PASS | 1/0/0 |  |  |  |  |
| 320 | TestOps::test_slice_ellipsis | test_slice_ellipsis | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.MUL | MUL | broadcast |  |
| 321 | TestOps::test_slice_errors | test_slice_errors | TestOps | PASS | 1/0/0 |  |  |  |  |
| 322 | TestOps::test_slice_fancy_indexing_dim_collapse_int | test_slice_fancy_indexing_dim_collapse_int | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 323 | TestOps::test_slice_fancy_indexing_dim_inject_and_collapse | test_slice_fancy_indexing_dim_inject_and_collapse | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 324 | TestOps::test_slice_fancy_indexing_dim_inject_none | test_slice_fancy_indexing_dim_inject_none | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 325 | TestOps::test_slice_fancy_indexing_errors | test_slice_fancy_indexing_errors | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 326 | TestOps::test_slice_fancy_indexing_list_indices | test_slice_fancy_indexing_list_indices | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 327 | TestOps::test_slice_fancy_indexing_list_with_tensors | test_slice_fancy_indexing_list_with_tensors | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 328 | TestOps::test_slice_fancy_indexing_no_dim_collapse | test_slice_fancy_indexing_no_dim_collapse | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 329 | TestOps::test_slice_fancy_indexing_tuple_indices | test_slice_fancy_indexing_tuple_indices | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 330 | TestOps::test_slice_fancy_indexing_with_tensors | test_slice_fancy_indexing_with_tensors | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 331 | TestOps::test_slice_in_bounds_1dim | test_slice_in_bounds_1dim | TestOps | PASS | 1/0/0 |  |  |  |  |
| 332 | TestOps::test_slice_in_bounds_multidim | test_slice_in_bounds_multidim | TestOps | PASS | 1/0/0 |  |  |  |  |
| 333 | TestOps::test_slice_int_indexing | test_slice_int_indexing | TestOps | PASS | 1/0/0 |  |  |  |  |
| 334 | TestOps::test_slice_negative_strides | test_slice_negative_strides | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 335 | TestOps::test_slice_on_0dim_tensor | test_slice_on_0dim_tensor | TestOps | PASS | 1/0/0 |  |  |  |  |
| 336 | TestOps::test_slice_one_endpoint_out_of_bounds | test_slice_one_endpoint_out_of_bounds | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 337 | TestOps::test_slice_start_gt_end | test_slice_start_gt_end | TestOps | PASS | 1/0/0 |  |  |  |  |
| 338 | TestOps::test_slice_stride_gt_one | test_slice_stride_gt_one | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 339 | TestOps::test_slice_with_const_tensor | test_slice_with_const_tensor | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 340 | TestOps::test_slice_with_none | test_slice_with_none | TestOps | PASS | 1/0/0 |  |  |  |  |
| 341 | TestOps::test_slice_zero_in_shape | test_slice_zero_in_shape | TestOps | PASS | 1/0/0 |  |  |  |  |
| 342 | TestOps::test_small_cummax | test_small_cummax | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 343 | TestOps::test_small_cummin | test_small_cummin | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 344 | TestOps::test_small_cumprod | test_small_cumprod | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 345 | TestOps::test_small_cumsum | test_small_cumsum | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 346 | TestOps::test_small_gemm | test_small_gemm | TestOps | PASS | 1/0/0 |  |  |  |  |
| 347 | TestOps::test_small_gemm_eye | test_small_gemm_eye | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 348 | TestOps::test_small_gemm_padded | test_small_gemm_padded | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.MUL | MUL | MUL_in_reduce |  |
| 349 | TestOps::test_small_gemm_range | test_small_gemm_range | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 350 | TestOps::test_softmax | test_softmax | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 351 | TestOps::test_softmax_argmax | test_softmax_argmax | TestOps | FAIL | 0/1/0 | unsupported_layout:(65,):45 |  | layout |  |
| 352 | TestOps::test_softmax_other_axis | test_softmax_other_axis | TestOps | FAIL | 0/1/0 | unsupported_layout:(100,):10 |  | layout |  |
| 353 | TestOps::test_softplus | test_softplus | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 354 | TestOps::test_softsign | test_softsign | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 355 | TestOps::test_softsign_exact | test_softsign_exact | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 356 | TestOps::test_sort | test_sort | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 357 | TestOps::test_sparse_categorical_crossentropy | test_sparse_categorical_crossentropy | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 358 | TestOps::test_sparse_categorical_crossentropy_ignore_index | test_sparse_categorical_crossentropy_ignore_index | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 359 | TestOps::test_sparse_categorical_crossentropy_label_smoothing | test_sparse_categorical_crossentropy_label_smoothing | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 360 | TestOps::test_sparse_categorical_crossentropy_reductions | test_sparse_categorical_crossentropy_reductions | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 361 | TestOps::test_split | test_split | TestOps | PASS | 1/0/0 |  |  |  |  |
| 362 | TestOps::test_sqrt | test_sqrt | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.SQRT | SQRT | SQRT | Y |
| 363 | TestOps::test_squeeze | test_squeeze | TestOps | PASS | 1/0/0 |  |  |  |  |
| 364 | TestOps::test_stack | test_stack | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 365 | TestOps::test_stack_max | test_stack_max | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 366 | TestOps::test_stack_slice | test_stack_slice | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 367 | TestOps::test_std | test_std | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 368 | TestOps::test_std_axis | test_std_axis | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 369 | TestOps::test_std_keepdim | test_std_keepdim | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 370 | TestOps::test_std_mean | test_std_mean | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 371 | TestOps::test_std_mean_loaded_nan | test_std_mean_loaded_nan | TestOps | PASS | 1/0/0 |  |  |  |  |
| 372 | TestOps::test_std_one_in_axis | test_std_one_in_axis | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 373 | TestOps::test_std_zero_in_axis | test_std_zero_in_axis | TestOps | PASS | 1/0/0 |  |  |  |  |
| 374 | TestOps::test_strided_conv1d_simple | test_strided_conv1d_simple | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 375 | TestOps::test_strided_conv2d | test_strided_conv2d | TestOps | PARTIAL | 1/2/0 | unsupported_layout |  | layout |  |
| 376 | TestOps::test_strided_conv2d_simple | test_strided_conv2d_simple | TestOps | FAIL | 0/1/0 | unsupported_layout |  | layout |  |
| 377 | TestOps::test_strided_conv2d_simple_vec | test_strided_conv2d_simple_vec | TestOps | SKIP | 0/0/1 |  |  |  |  |
| 378 | TestOps::test_strided_conv_transpose2d | test_strided_conv_transpose2d | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 379 | TestOps::test_sub | test_sub | TestOps | PASS | 1/0/0 |  |  |  |  |
| 380 | TestOps::test_sum | test_sum | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 381 | TestOps::test_sum_cat_collapse | test_sum_cat_collapse | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 382 | TestOps::test_sum_collapse | test_sum_collapse | TestOps | FAIL | 0/1/0 | forward_pass_failed |  | forward_pass_failed |  |
| 383 | TestOps::test_sum_collapse_neg | test_sum_collapse_neg | TestOps | PASS | 1/0/0 |  |  |  |  |
| 384 | TestOps::test_sum_dtype_arg | test_sum_dtype_arg | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 385 | TestOps::test_sum_fake | test_sum_fake | TestOps | PASS | 1/0/0 |  |  |  |  |
| 386 | TestOps::test_sum_full | test_sum_full | TestOps | FAIL | 0/1/0 | cmac_exceeds_cbuf |  | cmac_exceeds_cbuf | Y |
| 387 | TestOps::test_sum_pad_collapse | test_sum_pad_collapse | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 388 | TestOps::test_sum_relu | test_sum_relu | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 389 | TestOps::test_sum_simple | test_sum_simple | TestOps | PASS | 1/0/0 |  |  |  |  |
| 390 | TestOps::test_sum_tiny | test_sum_tiny | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 391 | TestOps::test_sum_twice | test_sum_twice | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 392 | TestOps::test_sum_with_zeros_shape | test_sum_with_zeros_shape | TestOps | PASS | 1/0/0 |  |  |  |  |
| 393 | TestOps::test_swish | test_swish | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 394 | TestOps::test_tan | test_tan | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 395 | TestOps::test_tanh | test_tanh | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 396 | TestOps::test_tanh_extreme | test_tanh_extreme | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 397 | TestOps::test_tiny_add | test_tiny_add | TestOps | PASS | 1/0/0 |  |  |  |  |
| 398 | TestOps::test_tiny_mul | test_tiny_mul | TestOps | PASS | 1/0/0 |  |  |  |  |
| 399 | TestOps::test_topk | test_topk | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 400 | TestOps::test_topo_sort | test_topo_sort | TestOps | FAIL | 0/1/0 | unsupported_op:non_index_operand |  | nested_EW |  |
| 401 | TestOps::test_transpose | test_transpose | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 402 | TestOps::test_tril | test_tril | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 403 | TestOps::test_triu | test_triu | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.WHERE | WHERE | WHERE | Y |
| 404 | TestOps::test_trunc | test_trunc | TestOps | FAIL | 0/1/0 | unsupported_op:Ops.TRUNC | TRUNC | TRUNC | Y |
| 405 | TestOps::test_unflatten | test_unflatten | TestOps | PASS | 1/0/0 |  |  |  |  |
| 406 | TestOps::test_unfold | test_unfold | TestOps | FAIL | 0/1/0 | unsupported_layout:Ops.ADD | ADD | broadcast |  |
| 407 | TestOps::test_unsqueeze | test_unsqueeze | TestOps | PASS | 1/0/0 |  |  |  |  |
| 408 | TestOps::test_var | test_var | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 409 | TestOps::test_var_axis | test_var_axis | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 410 | TestOps::test_var_keepdim | test_var_keepdim | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 411 | TestOps::test_var_one_in_axis | test_var_one_in_axis | TestOps | FAIL | 0/1/0 | no_add_mul_reduction |  | no_add_mul_reduction |  |
| 412 | TestOps::test_var_zero_in_axis | test_var_zero_in_axis | TestOps | PASS | 1/0/0 |  |  |  |  |
| 413 | TestOps::test_view | test_view | TestOps | PASS | 1/0/0 |  |  |  |  |
| 414 | TestOps::test_where | test_where | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 415 | TestOps::test_where_permute | test_where_permute | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 416 | TestOps::test_xor | test_xor | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 417 | TestOps::test_zeros | test_zeros | TestOps | PASS | 1/0/0 |  |  |  |  |
| 418 | TestOps::test_zeros_like | test_zeros_like | TestOps | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 419 | TestOpsUint8::test_cast | test_cast | TestOpsUint8 | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 420 | TestOpsUint8::test_cast_relu | test_cast_relu | TestOpsUint8 | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 421 | TestOpsUint8::test_interpolate_bilinear | test_interpolate_bilinear | TestOpsUint8 | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 422 | TestOpsUint8::test_interpolate_nearest | test_interpolate_nearest | TestOpsUint8 | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 423 | TestOpsUint8::test_interpolate_nearest_exact | test_interpolate_nearest_exact | TestOpsUint8 | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
| 424 | TestOpsUint8::test_min | test_min | TestOpsUint8 | FAIL | 0/1/0 | unsupported_dtype |  | dtype |  |
