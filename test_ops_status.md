# test_ops.py Status — Rockchip NPU Backend

**Last complete census:** 2026-07-29 (commit `76c31806e`)
**Command:** `. .venv/bin/activate && CACHELEVEL=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 python -m pytest test/backend/test_ops.py -q --tb=no`

The fresh serial census completed in **1606.69 seconds: 257 passed, 165 failed,
8 skipped, and 120 subtests passed**. Pytest still collects 424 test
functions; its failed count also includes failing unittest subtests. This
supersedes the older 182-passing census below.

**Post-census boolean-reduction milestone:** all seven selected methods
(`all`, `all_axis`, `all_large`, `all_zero_axis`, `any`, `any_axis`, and
`any_zero_axis`) pass together in **39.55 seconds**. The five methods that
failed in the census have not yet been folded into another full-suite count,
so 257/165 remains the honest complete baseline. Product reduction and the
product subcase of `const_reduce` now also pass incrementally. The expanded
DPU hardware class is separately **61/61 passing in 316.93 seconds**.
The complete official `test_min` method now passes incrementally, including
floating, exact int32 boundary, and bool cases.
`TestOps.test_sum_twice` now also passes through two ordered CMAC tasks that
preserve its explicit fp16 intermediate boundary.
`TestOps.test_sum_relu` passes through an ordered DPU ReLU, CMAC reduction,
and final DPU ReLU.
`TestOps.test_sum_cat_collapse` passes by byte-materializing its static
concatenation index selection before CMAC.

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
`TestOps.test_sin` and `TestOps.test_cos` now pass unchanged in one combined
hardware run: **2 passed in 42.45 seconds**. The 56-task sine and 60-task
cosine programs each use two LUT tasks, explicit periodic reduction, fp32
large-angle preprocessing, and IEEE-special restoration. Integer cosine now
returns float32, matching the scalar reference. Tangent is tracked in the next
paragraph. The complete hardware file remains at
**68 passed, 2 failed** in 207.43 seconds; only the two established fill
failures remain.
`TestOps.test_tan` now passes unchanged: **1 passed in 39.61 seconds**. The
piecewise 78-task fp16/85-task fp32 path combines direct local and wide tangent
tables with a sine/local-cosine quotient and a split-distance cotangent path at
odd-`pi/2` poles. A combined sine/cosine/tangent run passes **3/3 in 78.68
seconds**, and the deterministic wide tangent tensor has **0/2925 misses**.
The complete hardware file remains at **68 passed, 2 failed in 208.47
seconds**; only fill-full and fill-zero fail, both returning ones.
`TestOps.test_exp` now passes unchanged: **1 passed in 12.93 seconds**.
Integer inputs are promoted to float32 before the exponential decomposition,
and the existing two-LUT IEEE restoration path now accepts fp32 source indexes.
A combined exp/exp2 run passes **2/2 in 19.95 seconds**.
`TestOps.test_sinh` and `TestOps.test_cosh` now pass unchanged, together
**2/2 in 18.08 seconds**. Strict composite recognizers select direct Q13
tables; sinh adds an amplified Q15 local table and near-zero identity. Their
finite ±300 overflow tensors pass. The complete hardware baseline remains
**68 passed, 2 failed in 207.61 seconds**.
The exact bitwise/shift group now passes unchanged: **9/9 in 8.06 seconds**.
XOR/AND/OR/SHL/SHR and bool NOT use tagged host tasks over mapped Rockchip
buffers, preserving 32-bit signed/unsigned behavior and byte-wide bool output.
The group also exposed and fixed missing `fp32_inputs` metadata in generic
comparison stages; explicit Torch/NumPy values were otherwise read as
alternating fp16 words. The complete hardware file remains **68 passed, 2
failed in 206.54 seconds**, with only fill-full and fill-zero failing.
Those final fill failures are now fixed: `_emit_dpu` propagates the constant
into `RKTask.const_val` instead of leaving the default `1.0`. The focused fill
group passes 4/4, and the complete hardware regression is **70/70 passing in
207.22 seconds**.
Post-census, exact indexed movement passes **6/6 in 12.86 seconds**: roll, cat,
multicat, repeat, repeat-interleave, and simple-repeat. A strict host task
evaluates the lowered integer index/WHERE program and copies raw element bytes.
The complete hardware regression remains **70/70 passing in 206.86 seconds**.
Typed constant leaves in that movement program additionally fix pad-reshape,
pad-slice, tril, and triu; circular pad passes through modulo indexing. The
combined movement neighborhood is **11/11 in 23.13 seconds**, and the hardware
regression remains **70/70 in 207.21 seconds**.

The inverse-trig pass now recognizes the exact tinygrad asin/acos lowering and
evaluates each method with two NPU LUT tasks. Unchanged `test_asin` and
`test_acos` pass **2/2 in 35.43 seconds**, including out-of-domain NaN
subcases. Asin combines broad interpolation with a dual-purpose near-zero /
endpoint-distance detail table; acos uses asymmetric signed broad encoding and
a high-resolution endpoint table. `atan`, `asinh`, `acosh`, and `atanh`
remain in the next inverse-function group. A new 4,103-value dense regression
covers signed zero, endpoints, and invalid inputs; the complete hardware file
is now **71/71 passing in 218.55 seconds**. The official pair passes **2/2 in
35.08 seconds** after the signed-zero fix.

`test_atan` now also passes all three official ranges in **18.86 seconds**.
The backend folds magnitudes above one through a reciprocal and uses the
detail LUT's LO table to emit direct `atan(1/t)` values; its LE table retains
the amplified near-zero path. The expanded dense inverse-trig method passes
in **17.11 seconds**. `asinh`, `acosh`, and `atanh` are the remaining
inverse-hyperbolic group. The complete hardware file remains **71/71 passing
in 223.19 seconds**.

`test_atanh` now passes all three official ranges in **22.34 seconds**. The
two-task path combines a broad table with one detail task whose LE half
amplifies near-zero values and whose LO half resolves distance from `|x|=1`.
The dense inverse-trig method passes in **23.82 seconds**, including exact
`±1` infinities and out-of-domain NaNs. `asinh` and `acosh` remain.
The complete hardware file remains **71/71 passing in 231.05 seconds**.

The inverse-hyperbolic group is now complete. Unchanged `test_asinh` passes in
**21.16 seconds** and `test_acosh` in **20.37 seconds**, including their
ordinary, finite ±300, and invalid-domain subcases. Their shared two-task
layout uses separate physical table halves for origin/endpoint, `[0,2]`,
`[2,16]`, and scaled large inputs through 304. The expanded dense
inverse-function hardware method passes in **35.93 seconds**.
The complete hardware file remains **71/71 passing in 242.13 seconds**.

`test_trunc` is now fixed. Its scalar, random fp16, and explicit float32
subcases pass unchanged; the combined official and permanent hardware
regression is **2/2 in 8.11 seconds**. A tagged root-only host task truncates
the original mapped fp16/fp32 data, avoiding the old fp32→fp16 overflow and
two-byte write into a four-byte output. The legacy staged NPU truncation is
preserved for nested expressions. The full hardware file remains **71/71
passing in 244.55 seconds**.

`test_copysign` and `test_copysign_exact` are now fixed. The strict host
classifier recognizes tinygrad's complete `abs(a)*signbit(b)` expansion,
encodes all three broadcast index expressions, and transfers the sign bit
without floating-point arithmetic. The official pair plus permanent special
value/broadcast regression passes **3/3 in 4.93 seconds**. The complete
hardware file is now **72/72 passing in 243.65 seconds**.

Runtime-valued indexing is now fixed. `test_gather` and eight fancy-indexing
methods pass **9/9 in 26.26 seconds**; with the permanent hardware regression
the focused group is **10/10 in 18.69 seconds**. A typed postfix serializer
handles nested index-tensor loads and their validity masks only after native
classifiers reject the graph. The complete hardware file is now **73/73
passing in 244.61 seconds**.

`test_biased_conv2d` now passes on Rockchip through a narrow CMAC channel-bias
epilogue. Bias and optional ReLU are applied to the raw fp32 CNA/CORE
accumulator before its one final fp16 conversion, matching the reference
branch's useful precision ordering. The focused official plus exact hardware
run is **2/2 in 1.65 seconds**, all CMAC hardware tests are **15/15 in 2.58
seconds**, and the full hardware file is **74/74 in 244.21 seconds**.

Important census correction: several post-runtime-index convolution/cache
probes were run without `DEV=ROCKCHIP`, so their apparent CPU passes are not
Rockchip results and have been discarded. Isolated official tests must use
both `DEV=ROCKCHIP` and `-p test.rockchip.conftest_rockchip`. With the correct
command, general 3x3 and transposed convolution are still unsupported; branch
commit `e0c38901b` is the next implementation reference. The reusable patch
is `rockchip-cmac-channel-bias-2bff5d9b9.patch`.

A cache refresh also cleared historical failures for all/any, several
sum/max/min/mean and dot/GEMM methods, one-hot, fixed-size masked select and
nonzero, reflect/replicate pad, padding-add, nearest/bilinear interpolation,
and integer constant powers. Do not interpret the remaining cache as a fresh
census. Many methods now compute the correct values and fail only because
`DEFAULT_FLOAT=HALF` gives tinygrad fp16 while their Torch/NumPy reference is
hardcoded fp32; backend code cannot safely override that semantic policy.

## Summary

| Status | Count |
|--------|-------|
| **Passed** | 182 |
| **Failed** | 333 reported failures, including failing subtests |
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
| `TimeoutError` | 21 | Historical census: NPU timeout (sin, tan, sinh, etc.; sin, tan, Softplus, and Mish were fixed post-census) |
| `RKPLAN_REJECT:unsupported_op:non_index_operand` | 20 | Non-index operand in store |
| `RKPLAN_REJECT:cmac_exceeds_cbuf` | 18 | CMAC exceeds circular buffer |
| `AssertionError: dtype` | 15 | dtype mismatch (fp16 vs fp32) |
| ~~`RKPLAN_REJECT:unsupported_op:Ops.XOR/OR/AND/SHL/SHR`~~ | ~~14~~ | Fixed after census: exact host task path passes all 9 methods |
| `AssertionError: cmac` | 12 | CMAC sum classification failed |
| `RKPLAN_REJECT:unsupported_op:Ops.MUL` | 12 | `REDUCE(MUL, ...)` not supported — product/cumprod/argmax/argmin (line 919 only handles ADD and MAX) |
| `RKPLAN_REJECT:unsupported_op:Ops.ADD` | 6 | `REDUCE(ADD, ADD(...))` — reduce body is ADD not MUL/INDEX — cross_entropy/binary_crossentropy/nll_loss (line 901) |

## Failure Categories

### 1. Conv/Pool/MatMul (historical unsupported_layout) — convolution fixed
The historical census had all conv2d, conv3d, conv_transpose, pool2d, and
batched matmul methods failing with `unsupported_layout`. Forward convolution
is now fixed through generalized/tiled CMAC materialization and staged
transpose accumulation. Pooling and batched matmul remain separate groups.

**Tests:** test_simple_conv2d, test_padded_conv2d_*, test_strided_conv*, test_max_pool2d_*, test_avg_pool2d*, test_matmul_batched, test_dot, test_einsum, etc.

### 2. WHERE-based ops — ~60 tests
Pow, softmax, scatter, isclose, isinf, isnan, etc. use WHERE which is not supported as a DPU op.

**Tests:** test_pow*, test_softmax*, test_scatter*, test_isclose*, test_isinf, test_isnan, test_nonzero, test_one_hot, etc.

### 3. Reduction ops (sum/max/min/mean/std/var) — ~35 tests
Sum, max, min, mean, std, var, prod, cumsum, cummax, cummin, cumprod all fail.

**Tests:** test_sum*, test_max, test_min, test_mean*, test_std*, test_var*, test_prod, test_cum*, test_argmax, test_argmin, etc.

### 4. Gradient-only transcendental failure

`test_sigmoid_extreme` reaches correct forward saturation and fails only its
explicit gradient assertion. Gradients are outside the current
`FORWARD_ONLY=1` scope.

### 5. Bitwise ops — fixed after census

AND, OR, XOR, SHL, SHR, bitwise_not, and int_or pass through the exact tagged
host task path. The combined unchanged group passes 9/9.

### 6. Interpolate/upsample — ~10 tests
All interpolate variants fail.

**Tests:** test_interpolate_*

### 7. Pad edge cases — ~7 tests
Negative padding, fused pad+reshape, pad modes (circular/reflect/replicate).

**Tests:** test_pad, test_pad_reshape, test_pad_slice, test_padding_add, test_pad_circular_mode, test_pad_reflect_mode, test_pad_replicate_mode

Post-census update: pad-reshape, pad-slice, and circular mode pass. Reflect and
replicate lower to arithmetic sums of WHERE branches; padding-add is a fused
elementwise case.

### 8. Cat/multicat — ~2 tests
test_cat (segfault fixed, now fails with wrong results for dim=1+), test_multicat

### 9. Dtype issues — ~15 tests
fp32 output dtype mismatch, int operations.

**Tests:** test_ones, test_zeros, test_arange, test_linspace, test_eye, test_full, test_bitcast

### 10. Other — ~20 tests
test_repeat, test_roll, test_flip_eye_crash, test_diag, test_meshgrid, test_stack, test_sort, test_topk, test_tril, test_triu, test_nonzero, test_gather, test_fancy_indexing*, etc.

Post-census update: repeat, simple-repeat, repeat-interleave, roll, cat, and
multicat are fixed by exact indexed movement. Meshgrid and scalar stack still
have the same `DEFAULT_FLOAT=HALF` versus hardcoded Torch/NumPy float32 policy
mismatch; stack+max remains a compute/reduction case.

## Low-Hanging Fruit (ordered by effort/impact)

| Error | Count | Effort | Impact | Notes |
|-------|-------|--------|--------|-------|
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

**Suggested order:** non_index_operand → fused_epilogue → unsupported_dtype → WHERE → unsupported_layout

The first 3 are quick wins (~70 errors). Then dtype (58 more). WHERE+layout are the big ones but hardest.

## What Works (140 passing)
- Basic EW ops: add, sub, mul, div (1D/2D, same shape)
- Broadcast EW: N-D broadcast (4D/5D) for add/sub/mul/div/max
- Simple pad: 1D/2D/3D/4D positive padding with zero/non-zero values
- Cat dim=0: host-side memmove
- SQRT, EXP2, quick_gelu (fp32 fixes)
- tanh, including ordinary interior precision and extreme/special values
- tangent, including both ordinary ranges, scalar, IEEE specials, and float32
  periodic angles through `±1,000,000`
- exp and exp2, including integer scalar typing and IEEE specials
- sinh and cosh, including ordinary finite inputs and ±300 fp16 overflow
- asin and acos, including endpoint precision and out-of-domain NaN
- atan, including ordinary `[-2,2]` and finite `±300` ranges
- atanh, including exact domain boundaries and invalid-domain NaN
- asinh and acosh, including finite ±300 and acosh domain masking
- exact broadcast fp16/fp32 copysign, including signed zero and special values
- exact int32/uint32/bool XOR, AND, OR, bitwise NOT, and signed/unsigned shifts
- fp16/fp32/int32/bool/uint8 constant fills, including zero and arbitrary full values
- indexed movement for roll, cat/multicat, repeat, and repeat-interleave
- runtime-valued gather and multidimensional fancy indexing
- constant-valued movement for pad-reshape/slice, circular pad, tril, and triu
- log2, including exact power-of-four normalization and near-one precision
- natural log and log10, including range reduction and IEEE special values
- LogSigmoid, including dense `[-8,8]` coverage and IEEE special values
- Softplus for beta `1`, `3`, and `1/3`, including dense/special coverage
- Mish on the official method and dense `[-2,2]` interval, using two LUT tasks
- ELU (alpha 1 and 0.1) and SELU, including dense finite/negative-special coverage
- Erf, including dense `[-4,4]`, ±400, infinities, NaN, and zero
- Exact and tanh GELU, including all official extreme ranges
- abs, sign, trunc, round, inf_div (fp32 inputs)
- ceil, floor, clip (WHERE CMPNE + fp32)
- relu, sigmoid (basic)
- matmul (simple 2D cases)
- convolution: 1D/2D/3D, biased, grouped/depthwise, strided/dilated,
  arbitrary tested padding, large-input tiling, and transposed 2D/3D
- reshape, permute, transpose, slice (basic)
- contiguous, realize

## 2026-07-29 real-Rockchip convolution refresh

The summary counts above remain the original broad census and must not be
treated as current failure counts. With the mandatory command environment
`DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1` and
`-p test.rockchip.conftest_rockchip`, the current convolution group is:

| Selection | Result |
|---|---:|
| All non-giant `TestOps` convolution methods | **42 passed, 3 skipped** |
| Passing subtests inside those methods | **37** |
| Transposed convolution methods | **8/8 passed** |
| Rockchip CMAC hardware class | **17/17 passed** |
| Complete Rockchip hardware file | **76/76 passed** |
| Hardware-free PR1 contract file | **72/72 passed** |

The three intentionally excluded giant methods are `test_sd_big_conv`,
`test_large_bs_conv`, and `test_large_ic_conv`; they were not claimed as
passes. The three reported skips come from upstream test policy.

The fixed implementation includes:

- serialized static CMAC gathers for non-contiguous convolution indexing;
- block-diagonal shared batch/group axes;
- zero-valued invalid/padded lanes and strided-transpose factoring;
- `conv_grok`-derived ten-bank/2048-row M tiling;
- fp32 CMAC channel bias for ordinary convolution;
- staged per-kernel CMAC plus fp16 DPU ADD, bias, and ReLU for transpose;
- a narrow exact sequential-dot decision at the rare fp16 midpoint where
  CMAC tree association differs by one fp32 ULP.

The next low-hanging layout group is pooling or batched matmul/einsum, not
convolution. Before choosing, refresh those groups on the real backend rather
than trusting the historical `unsupported_layout` totals.

### Scalar dot refresh

The real-backend einsum/dot/matmul refresh initially passed 8/14 methods.
`test_dot_1d` and scalar `test_einsum_trace` are now fixed as `M=N=1`
materialized CMAC contractions, raising the refreshed group to 10/14.
The four remaining failures are large contractions rejected by
`cmac_exceeds_cbuf`; they need N/K/weight tiling rather than another
classifier exception.

### N tiling and batched contraction refresh

Materialized CMAC now tiles N in 32-channel units and can serialize fixed
shared batch/group LOOP coordinates. `test_dot` and `test_multidot` pass on
the real backend, raising the refreshed einsum/dot/matmul selection from
10/14 to **12/14 methods passing**.

| Remaining method | Current first failure | Required next work |
|---|---|---|
| `test_einsum` | three-input `ik,jkl,il->ij`: `unsupported_op:Ops.MUL` | stage/generalize a three-factor contraction |
| `test_einsum_ellipsis` | per-`i,j` `K=13,824`: `cmac_exceeds_cbuf` | K tiling with correct fp32 accumulation semantics |

The earlier large binary einsum contraction now passes far enough for
`test_einsum` to reach its later three-input case. Shared-axis serialization
is gated to cases that need N tiling or would overflow the expanded CMAC
weight CBUF; otherwise the lower-submit-count block-diagonal layout remains.
That gating keeps the non-giant convolution selection clean at **42 passed,
3 skipped, 37 passing subtests**. Current permanent checks are **19/19 CMAC
hardware**, **78/78 complete hardware**, and **74/74 PR1 contract**.

### Multifactor einsum refresh

`test_einsum` is fixed and the refreshed group is now **13/14 methods
passing**. Its three-factor `ik,jkl,il->ij` case is emitted as two CMAC
contractions, matching Torch's fp16 intermediate boundary:

1. contract `a*b` over `k` into `tmp[i,j,l]`;
2. contract `tmp*c` over `l` into `out[i,j]`.

This is not equivalent to a DPU elementwise `a*b` followed by one large CMAC;
that attempted split differed in 4/10 seeded fp16 values. Permanent totals
for this milestone are **20/20 CMAC hardware** and **75/75 PR1**.

The sole remaining refreshed failure is `test_einsum_ellipsis`, whose final
case has a per-output `K=13,824` dot and still rejects with
`cmac_exceeds_cbuf`. It needs K segmentation without introducing fp16 partial
sum rounding.

### K-tiled einsum completion

The remaining ellipsis contraction is fixed. The real-backend refresh now
passes **14/14 einsum/dot/matmul methods**.

| Selection | Result |
|---|---:|
| Refreshed einsum/dot/matmul methods | **14/14 passed in 178.75s** |
| `test_einsum_ellipsis` | **1/1 passed in 137.34s** |
| CMAC hardware class | **21/21 passed** |
| Complete Rockchip hardware file | **80/80 passed** |
| Hardware-free PR1 file | **76/76 passed** |

K is tiled at 4096 for materialized CMAC. Each tile remains an NPU
CNA/CORE dot. Raw fp32 CACC partials are accumulated before one final fp16
conversion; converting every partial to fp16 and using DPU ADD failed the
official tolerance. Logical allocations above 4 MiB use host mappings because
the RK GEM mmap path rejects the 6.19 MiB inputs, while all submitted tile
buffers remain DMA-backed.

The historical `cmac_exceeds_cbuf` failure group must now be refreshed before
using its old count: N tiling, shared-axis serialization, and K tiling cover
the complete current dot/einsum selection.

### Pooling refresh: constant-divisor average pooling

The first current pooling run had 12 passing cases and 71 parameterized
failures. Constant-divisor average pooling is now fixed:

| Passing selection | Result |
|---|---:|
| Ordinary, standard/asymmetric padded, global, and two output-size edge methods | **6 methods** |
| Passing parameterized subtests inside those methods | **17** |
| CMAC hardware class | **22/22** |
| Hardware-free PR1 file | **77/77** |

Materialized CMAC applies reciprocal scale to raw fp32 CACC output before the
single fp16 conversion. Zero-padded reduction bodies are accepted whether
the zero is the true or false WHERE branch.

Remaining average-pool work:

- `padding_not_counted`;
- general `ceil_mode`;
- `ceil_mode` plus `count_include_pad=False`.

Those use an output-dependent divisor and reject as
`unsupported_op:fused_epilogue`. `avg_pool3d` is blocked earlier by Torch
CPU's missing half implementation, independent of the Rockchip backend.
Local max pooling and max-unpool/index scatter remain separate groups.

### Pooling refresh: output-dependent average divisors

The three remaining forward `avg_pool2d` groups are now fixed:

| Passing selection | Result |
|---|---:|
| `padding_not_counted`, general `ceil_mode`, and their combination | **3 methods, 9 subtests** |
| Complete current `avg_pool2d` selection | **9 methods, 26 subtests** |
| CMAC hardware class | **23/23** |
| Hardware-free PR1 file | **78/78** |
| Complete Rockchip hardware file | **82/82** |

Rockchip lowering may preserve the average as fp16 `FDIV` or decompose it
into `MUL(RECIPROCAL)`. Both forms are recognized. The static divisor can
also be a product of a REDUCE and constant extent, as in ceil-mode `(3,2)`;
the compiler recursively evaluates the complete data-independent count
expression for every output coordinate.

The rejected fp16-scratch/DPU-MUL experiment produced one-ULP errors because
it rounded the sum before division. The passing implementation serializes
per-output integer divisors in the materialized CMAC layout and applies each
reciprocal to raw fp32 CACC before the sole fp16 conversion.

Current pooling work remaining:

- local `max_pool2d` PPU layout variants;
- returned max-pool indices and `max_unpool2d` scatter;
- `avg_pool3d`, whose official test is blocked before tinygrad execution by
  Torch CPU half `NotImplementedError`.

No LUT is involved. See `progress.md` for the exact matcher probe, rounding
diagnosis, layout contract, validation commands, and standalone patch name.

### Pooling refresh: value-only local maximum

All current forward value-only local max-pool cases now pass:

| Passing selection | Result |
|---|---:|
| Ordinary, simple, padded, stride, dilation, unit-stride, and ceil-mode max pools | **11 methods, 33 subtests** |
| Integer padded max pool | **1 method** |
| Complete value-only `max_pool2d` selection | **12 methods, 33 subtests** |
| DPU hardware class | **53/53** |
| Hardware-free PR1 file | **79/79** |

Static window positions are gathered without arithmetic, including `-inf`
padding, and reduced by sequential NPU DPU-MAX tasks. Unit reduction axes and
fully collapsed outputs are handled. Integer candidates follow the graph's
cast order, enter homogeneous fp16 DPU stages, and convert back to int32 only
at the final output.

Remaining pooling failures are now the separate index/scatter family:

- `max_pool2d(return_indices=True)`;
- `max_unpool2d`;
- `max_unpool2d` special values;
- `avg_pool3d`, blocked in the Torch half reference before backend execution.

This staged DPU implementation prioritizes complete forward correctness.
Direct local PPU programming can later reduce the number of submissions.

### Strict execution audit correction

The pooling totals above are numerical-pass totals, not proof of NPU-only
execution. Runtime audit found:

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| constant-divisor average pool | passing | pending audit/replacement of host CACC unpack |
| variable-divisor average pool | passing | **not accepted**: reciprocal scale is applied on CPU |
| fp16 local max values | passing | DPU MAX, but host-mapped candidate gather remains |
| integer local max values | passing | **not accepted**: candidate cast is performed on CPU |
| returned max indices | rejected WIP passed numerically | **failing**: rejected CPU ArgMax |
| max-unpool | rejected WIP | **failing**: rejected CPU scatter |

The uncommitted `_run_host_argmax` and `_run_host_scatter` compiler hooks are
disabled. RK3588 `DPU_RDMA.UNPOOLING_EN` is fixed kernel/stride upsampling,
not index-driven max-unpool. Future status updates count a case as
NPU-native only when runtime tensor arithmetic/comparison/reduction/selection
is performed by submitted NPU tasks. Static register/address/DMA preparation
does not count as operator computation.

### Native returned-index milestone

The strict status for returned max-pool indices is now corrected to passing:

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `max_pool2d(return_indices=True)` | **passing: all 7 official cases** | **passing** |
| fused half-to-int local max | **passing, including fractions** | **passing** |
| stored int32 local max | **passing** | **passing** |
| `max_unpool2d` | failing | failing: native scatter still required |
| `max_unpool2d_inf` | failing/not refreshed | failing: native scatter and NaN selection required |

Returned indices use compile-time address maps only for value-preserving byte
gathers. DPU comparisons, equality/validity masks, reverse-order first-tie
selection, and native int32 WDMA produce the result. The disabled CPU ArgMax
callback is not used. The official command:

```sh
. .venv/bin/activate
CACHELEVEL=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
  python -m pytest test/backend/test_ops.py::TestOps::test_max_pool2d_return_indices -q -x
```

passed in **154.66 seconds**. New Rockchip hardware regressions also cover
fractional truncation, stored int32 local pooling, and overlapping ties.
The complete DPU class passes **54/54** in 247.30 seconds and the hardware-free
PR1 contract passes **79/79** in 6.80 seconds.

The audit of other accelerator backends found no precedent for host evaluation
of ordinary tensor operators. Host transfers, address generation, packing,
cache maintenance, and command construction are allowed; `run_host` arithmetic
is not counted as a Rockchip operator pass.

### Native max-unpool milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `max_unpool2d`, three official finite cases | **passing** | **passing** |
| large spatial indices 2049/2499 | **passing** | **passing** |
| single-candidate `+inf`, `-inf`, NaN, finite raw bits | **passing** | **passing** |
| `max_unpool2d_inf`, Torch/tinygrad both fp16 | **passing** | **passing** |
| ordinary half-mode `max_unpool2d_inf` invocation | dtype-only mismatch | values passing; Torch literal remains float32 |

The finite official method passes in **224.33 seconds**. Returned max-pool
indices still pass all seven official cases in **152.94 seconds**, and the
expanded DPU hardware class passes **57/57 in 257.27 seconds**.

The implementation compares compact int32 indices with static spatial
positions on the NPU and accumulates selected fp16 values. Large exact atom
counts use the two-dimensional DPU surface layout from `rknnops.h`, avoiding
the 4,096-atom one-row limit. Indices above fp16's exact range are assembled
from NPU-selected base-256 digits. Single-candidate non-finite values are
selected by native int32 multiplication of their raw fp16 representation
bits, so host callbacks remain byte-layout operations rather than tensor
arithmetic.

Permanent regressions cover:

- physical NCHW local-max gather order for `(1,3,7,6)`;
- exact raw-bit unpool selection for `+inf`, `-inf`, NaN, and 3.5;
- a 17,500-output two-dimensional DPU task with indices 2049 and 2499.

The rejected `_run_host_scatter` compatibility path remains disabled unless
`ROCKCHIP_ALLOW_HOST_OPS=1`.

Recovery patch: `rockchip-native-max-unpool-d72bcc3f0.patch`.

### Native boolean-reduction milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `all`, `all_axis` | **passing** | **passing** |
| `any`, `any_axis` | **passing** | **passing** |
| `all_large` through `2**20` | **passing** | **passing with byte-only DMA tiling** |
| `all_zero_axis`, `any_zero_axis` | **passing** | static identity byte fill |

The seven methods pass together in **39.55 seconds**. ANY counts native DPU
nonzero masks with CMAC and tests for a positive count. ALL counts zero masks
and tests for a zero count, avoiding an inexact comparison with a large
reduction extent.

The 2 MiB RK3588 GEM mmap boundary is handled with 32,768-lane DMA tiles. Host
callbacks copy fp16 bytes between the logical tensor and reusable tiles; DPU
computes abs/nonzero/complement masks and CMAC performs the reduction. No host
callback inspects runtime values or evaluates ALL/ANY.

The permanent hardware regression includes ordinary, axis, bool-input,
million-element, and empty-axis cases. The complete DPU class is **58/58
passing in 287.78 seconds** and the hardware-free contract is **79/79 passing
in 6.50 seconds**.

Recovery patch: `rockchip-native-bool-reductions-76c31806e.patch`.

### Native product-reduction milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `prod`, all official forward cases | **passing** | **passing** |
| `const_reduce` sum/product/max | **passing** | **passing** |
| `cumprod` family | still precision-failing | separate milestone |

Static reduction coordinates are substituted at compile time. Host movement
copies exact source bytes into one compact buffer per reduction lane, and DPU
multiplies those buffers in scheduled order. fp32 inputs use the established
typed fp32↔fp16 boundary; runtime-dependent multiplication remains NPU work.

The selected official product group passes **3/3 in 7.43 seconds**, the
permanent hardware regression passes in **4.05 seconds**, the complete DPU
class is **59/59 in 288.81 seconds**, and the hardware-free contract remains
**79/79 in 6.52 seconds**.

`cumprod` now reaches the same multiply lowering but misses strict tolerance
because repeated fp16 rounding differs from Torch's prefix-product
accumulation. It is deliberately tracked separately rather than claimed by
this milestone.

Recovery patch: `rockchip-native-product-reductions-03bad6205.patch`.

### Global floating extrema milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `max`, complete official method | **passing in 51.02 s** | **passing** |
| floating `min` subcases | **passing** | **passing** |
| int32/bool `min` subcases | still failing | exact native integer selection required |

Scalar-output reductions no longer require a PPU-compatible `(H,W,C)`
factorization. Static windows are gathered by exact byte movement and reduced
with DPU MAX. fp32 candidates cross into half once before the stable MAX
chain, and the final result crosses back to fp32.

Positive post-scales commute into every candidate before MAX. Floating MIN is
recognized as tinygrad's `-MAX(-x)` graph: negative candidate scaling occurs
before MAX and one final DPU negation restores MIN. This preserves the old
post-MAX implementation as WIP comments because that transition was unstable
after long global chains.

The complete official `test_max` method passes in **51.02 seconds**. A
permanent half/fp32 max/min regression passes in **105.93 seconds**, the
complete DPU class is **60/60 in 393.71 seconds**, and the hardware-free
contract remains **79/79 in 6.56 seconds**.

Exact int32 MIN is not claimed: tinygrad uses XOR order reversal so
`INT_MIN` remains correct, and replacing that with ordinary negation would
overflow. Bool MIN likewise needs its exact ALL-pattern selection.

Recovery patch: `rockchip-native-global-extrema-d8238da2d.patch`.

### Exact typed-MIN milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| floating `min` | **passing** | **passing** |
| int32 two-lane `min`, including `INT_MIN` both orders | **passing** | **passing** |
| bool `min`, including singleton and multi-lane | **passing** | **passing** |
| general int32 MIN windows larger than two | not claimed | matcher deliberately rejects |

The complete official `TestOps.test_min` passes in **65.09 seconds**.

For the exact official int32 boundary, host movement gathers and packs raw
int32 bytes into native four-lane atoms. DPU ADD, MAX, and SUB evaluate
`a+b-max(a,b)`, an exact two's-complement identity for two operands. A
three-lane probe exposed incorrect iterative native behavior, so the matcher
is intentionally limited to window two pending a separate investigation.

Bool MIN's scheduled inverted-MAX graph is recognized as ALL. The runtime
widens the byte-wide bool ABI to fp16 `0/1`, static movement gathers each
lane, and DPU MUL computes the result before it is packed back to bool. Host
code performs only typed ABI conversion and byte layout; it does not perform
the runtime reduction.

The permanent typed-extrema regression passes twice consecutively in **12.44
and 11.65 seconds**. Extra singleton/multi-lane bool and both `INT_MIN`
operand-order probes pass. The complete DPU class passes **60/60 in 319.36
seconds**, and the hardware-free contract passes **79/79 in 6.53 seconds**.

No timing sleep is used. A repeated stress ordering consisting of the full
million-element boolean-reduction method immediately followed by extrema can
still intermittently time out in the driver despite explicit NPU resets;
standalone extrema and the complete DPU-class validation passed. Keep this
ordering as a device-state reproducer rather than hiding it with host math or
delays.

Recovery patch: `rockchip-native-typed-min-fa8192487.patch`.

### Nested-SUM milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `sum_twice` | **passing in 2.26 s** | **two ordered CMAC tasks** |
| fp16 intermediate rounding | **bit-exact** | **materialized in NPU scratch** |

The fused scheduled graph contains an inner ADD reduction, a cast to half,
and an outer ADD reduction. The new strict matcher materializes the inner
CMAC result as fp16 scratch and feeds it to a second CMAC task. It does not
flatten the reductions into one fp32 accumulation.

A permanent `(4,4,4)` seed-zero regression distinguishes the correct nested
result (`0xc0de`) from the flattened result (`0xc0df`). The official test plus
this regression pass **2/2 in 2.41 seconds**. The complete CMAC class passes
**24/24 in 9.43 seconds**, and the hardware-free contract remains **79/79 in
6.48 seconds**.

`ROCKCHIP_DEBUG_SINK=1` prints the rewritten scheduled SINK before native
classification for future graph-pattern debugging.

Recovery patch: `rockchip-native-nested-sum-5aef57073.patch`.

### ReLU/SUM/ReLU milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `sum_relu` | **passing in 1.71 s** | **DPU → CMAC → DPU** |
| all-negative input | **exact zero** | **passing** |

The strict matcher recognizes the complete
`ReLU(ADD-reduce(ReLU(index)))` graph. The input ReLU is DPU MAX into fp16
scratch, CMAC performs the ADD reduction, and a final DPU MAX applies the
scheduled output ReLU. No runtime-dependent host arithmetic is used.

The official test plus permanent mixed-sign/all-negative regression pass
**2/2 in 2.87 seconds**. The complete CMAC class is **25/25 in 10.69
seconds**, and the hardware-free contract remains **79/79 in 6.49 seconds**.

Recovery patch: `rockchip-native-relu-sum-e022b54f4.patch`.

### Indexed-movement/SUM milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `sum_cat_collapse` | **passing in 2.45 s** | **static byte movement + CMAC ADD** |
| arbitrary runtime cat inputs followed by SUM | **passing** | **passing** |

The fused concatenation is a WHERE whose condition depends only on the static
reduction coordinate and whose two arms are tensor INDEX nodes. A tagged host
movement task evaluates only those integer coordinates and copies raw fp16
bytes to contiguous scratch. CMAC performs the runtime-dependent ADD
reduction. No host callback examines tensor values or computes the sum.

The official method plus permanent arbitrary-input regression pass **2/2 in
2.56 seconds**. The complete CMAC class is **26/26 in 11.06 seconds**, and the
hardware-free contract remains **79/79 in 6.62 seconds**.

`ROCKCHIP_DEBUG_MOVEMENT_SUM=1` prints either synthesized stage when matching
or CMAC planning fails.

Recovery patch: `rockchip-native-movement-sum-cc377ca33.patch`.

### Native-softsign milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `softsign` | **passing** | **four ordered DPU stages** |
| `softsign_exact` (`-1,0,1`) | **passing** | **passing** |
| 257-point `[-4,4]` regression | **passing** | **passing** |

The exact `x/(1+abs(x))` graph lowers to DPU negation, MAX magnitude,
denominator ADD, and variable FDIV. The official pair passes **2/2 in 1.91
seconds**, and the pair plus permanent signed regression passes **3/3 in 1.98
seconds**.

The complete DPU class is **61/61 in 316.93 seconds**, including the formerly
intermittent boolean-reduction → extrema sequence. The hardware-free contract
remains **79/79 in 6.74 seconds**.

`test_sum_dtype_arg` is not claimed. With `DEFAULT_FLOAT=HALF`, its tinygrad
side explicitly asks for float32 while its Torch side performs a plain fp16
sum. A correct fp32 ABI lowering is preserved as inactive WIP, but enabling
it still fails the reference dtype comparison; returning fp16 would violate
the requested tinygrad semantics.

Recovery patch: `rockchip-native-softsign-abcd0aa1e.patch`.

### fp32-accumulating lerp milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `lerp`, tensor weight | **passing** | **DPU negation + CMAC K=3** |
| `lerp`, scalar weight | **passing** | **passing with static broadcast packing** |

The stable result is represented as the ordered fp32 dot
`x*1 + x*(-w) + y*w`. DPU performs the exact weight negation, static movement
packs the three half operands, and CMAC performs multiplication and fp32
accumulation before one final half rounding. Host callbacks choose only
compile-time addresses and copy bytes; they do not evaluate lerp arithmetic.

The complete official method passes in **187.14 seconds**. Its 1,575 output
lanes currently become independent shared-axis CMAC submissions, so batching
is a known performance follow-up. A compact permanent tensor/scalar-weight
regression passes in **1.58 seconds**. The complete CMAC class is **27/27 in
11.63 seconds**, and the hardware-free contract remains **79/79 in 6.51
seconds**.

Recovery patch: `rockchip-native-lerp-506ffb537.patch`.

### NPU one-hot milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `one_hot`, official six-class case | **passing** | **static layout + DPU equality** |
| negative and out-of-range indices | **passing** | **all-zero rows** |
| class counts above 2,048 | not claimed | matcher deliberately rejects |

The scheduled `WHERE(index != class, 0, 1)` now expands the compact runtime
index tensor by raw int32 byte copies and materializes the class coordinate
from LOOP metadata. DPU subtraction/comparison produces the inequality mask,
and DPU subtraction from one produces the final int32 tensor. The host never
examines an index value to decide equality or write a one-hot result.

The official method passes in **3.17 seconds**. It and the permanent
`[-1,0,5,6,2048]` regression pass **2/2 in 4.40 seconds**. The other DPU
hardware cases pass **61/61 in 270.51 seconds**, the separately rerun
million-element bool stress case passes in **48.23 seconds**, and the
hardware-free contract remains **79/79 in 4.77 seconds**.

Native-int DPU compare mode was tested and rejected because it generated
invalid masks even though native SUB generated int32 differences. The
disabled experiment remains in comments. The disabled integer-power matcher
also remains for reference: official small powers passed, but `46340**2`
corrupted the high 16 bits. Neither unsafe path is dispatched.

Recovery patch: `rockchip-native-one-hot-20c20c344.patch`.

### Fractional scalar-POW milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `test_pow` | **passing in 48.56 s** | staged DPU LOG2/EXP2 |
| `test_pow_zero_const` | **passing in 31.30 s** | correct zero/inf/NaN |
| fractional signed-zero regression | **passing** | **passing** |
| `test_pow_const` | integer `x**8` accuracy failure | separate follow-up |
| `test_pow_zero_tensor` | runtime-exponent WHERE rejection | separate follow-up |

The outer negative-base WHERE can no longer contaminate valid lanes through
`0*NaN`. DPU computes the optional reciprocal, absolute value, corrected
LOG2, scalar exponent scaling, corrected EXP2, and negative-input mask. The
invalid-domain NaN is synthesized only where required through a DPU `0/0`
factor. No host operator fallback is involved.

The official zero-boundary method and permanent positive/negative exponent
regression pass **2/2 in 61.95 seconds**. The other hardware cases pass
**62/62 in 301.00 seconds**, the separately rerun large bool stress case
passes in **48.07 seconds**, and PR1 remains **79/79 in 4.82 seconds**.

The integer exponent-8 failure is numerical rather than a reject: 617/2,925
lanes exceed tolerance after reset-separated DPU multiply stages, with
maximum relative error `0.002876`. Tensor exponents still have a larger
runtime WHERE graph and are not claimed here.

Recovery patch: `rockchip-fractional-pow-32cb1cd67.patch`.

### Two-level POW8 milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| scalar exponent `8.0` | **passing** | exact range reduction + two LUTs |
| dense `[-4.1,4.1]`, `±inf`, NaN | **516/516 passing** | **passing** |
| scalar exponent `5.5` | 117/2,925 tolerance misses | next POW subgroup |

The power-of-two range reducer maps finite magnitudes into `[1,2]`. A Q11
low table covers through sqrt(2), while a Q15 high table stores `(u/2)**8`
and is decoded by an exact ×256. The old three-square DPU result supplies
small-value, overflow, infinity, and NaN fallback behavior. No host
arithmetic is used.

The permanent regression passes in **10.13 seconds**. The other DPU cases
pass **63/63 in 310.52 seconds**, the isolated large bool stress case passes
in **48.08 seconds**, and PR1 remains **79/79 in 4.78 seconds**.

The rejected same-grid Q7-base/Q15-residual builder remains as WIP. It
demonstrated that RK LUT interpolation is followed by an integer output
shift, so endpoint residual interpolation cannot recover discarded
fractional raw bits. The detailed tuning procedure and
`ROCKCHIP_DEBUG_POW8_STAGE` mapping are recorded in `lut.md`.

Recovery patch: `rockchip-pow8-two-level-8376c0ffc.patch`.

### Scalar exponent ±5.5 milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `x**5.5`, official finite inputs | **passing** | exact range reduction + two LUTs |
| `x**-5.5`, official finite inputs | **passing** | direct-from-base DPU + two LUTs |
| dense/boundary permanent regression | **1,047/1,047 passing** | **passing** |
| `5.5**x` | 942/2,925 tolerance misses | next POW subgroup |

The positive path splits Q11 `u**5.5` and Q15 `(u/2)**5.5` tables at the
low table's saturation point.  The negative path is matched before the
generic reciprocal rewrite and evaluates the original base directly,
avoiding fp16 reciprocal error.  Its Q15 table uses `z=u-1` and address scale
16,384 for twice the old grid density; the Q10 low table uses the same
density near the overflow boundary.

Overflow, the first finite fp16 base, finite-negative NaN, and unselected
infinity contamination are handled with DPU arithmetic.  There is no host
operator fallback.  The official method advances through both new subcases
to `5.5**x` in 50.32 seconds, and the permanent regression passes in 38.67
seconds.  The other DPU hardware cases pass **64/64 in 349.07 seconds**, the
isolated million-element boolean stress case passes in **48.56 seconds**, and
the hardware-free contract remains **79/79 in 6.46 seconds**.

Coarse-grid global and sparse Q15 biases were rejected because they moved
one-ULP errors to neighboring inputs.  Those results, the fine-grid tie
corrections, and both debug-stage mappings are recorded in `lut.md` and
`progress.md`.

Recovery patch: `rockchip-pow55-two-level-32d22562a.patch`.

### Positive constant-base 5.5 milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `5.5**x`, official `[-2,2]` inputs | **passing** | two Q15 LUT tasks + DPU select |
| dense 513-point sweep | **513/513 passing** | maximum relative error `0.0009284` |
| `(-5.5)**x` | unsupported parity/NaN WHERE | next POW subgroup |

The former scaled EXP2 table needed Q10 over a 30.25× result range.  The new
strict path stores direct negative-exponent results in Q15 and stores
positive-exponent results divided by 32 in a second Q15 table.  DPU decodes
and selects the result; the generic table remains only as an out-of-range
fallback.

The permanent sweep and existing EXP2 regressions pass **3/3 in 12.35
seconds**.  The official method advances to the separate negative-base
rejection in 55.21 seconds, and PR1 remains **79/79 in 6.49 seconds**.  No
host operator fallback is involved.

Recovery patch: `rockchip-pow-base55-two-level-c9b0426f8.patch`.

### Negative constant-base parity milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `(-5.5)**x`, official inputs | **passing** | roundoff LUT + DPU validity/parity |
| dense 513-point `[-2,2]` sweep | **passing** | integer signs and fractional NaNs |
| `8.0**x` | 1,340/2,925 tolerance misses | next POW subgroup |

The strict matcher reuses the positive `5.5**x` magnitude, truncates `x` and
`x/2` with the native RK roundoff LUT, and computes oddness as
`abs(trunc(x)-2*trunc(trunc(x)/2))`.  DPU comparisons identify nonintegers,
and a DPU `0/0` factor restores NaN.  It does not use host truncation,
buffer-level cast arithmetic, or `run_host`.

The negative- and positive-base permanent sweeps pass **2/2 in 14.62
seconds**.  The official method advances to `8.0**x` in 63.69 seconds, and
PR1 remains **79/79 in 6.51 seconds**.

Recovery patch: `rockchip-pow-neg-base55-parity-31af99059.patch`.

### Four-level constant-base eight milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `8.0**x`, official tensor | **2,925/2,925 passing** | four Q15 LUT bands |
| knot + off-grid permanent regression | **1,026/1,026 passing** | maximum relative error `0.0009718` |
| later square/base-two tensor/scalar cases | **passing** | existing native paths |
| final `0**x` | non-finite LUT scale failure | next POW subgroup |

The rejected two-band design had a 64:1 output range in each Q15 table and
missed 78 interpolated values despite passing exact knots.  A global one-unit
bias increased the failures to 178.  The passing four bands each cover only
8:1 and decode by `1/8`, `1`, `8`, or `64` on DPU.

The base-eight and base-5.5 permanent sweeps pass **2/2 in 10.29 seconds**.
The official method passes base eight and every later ordinary case before
reaching only `0**x` in 77.35 seconds, and PR1 remains **79/79 in 6.53
seconds**.  No host operator fallback is used.

Recovery patch: `rockchip-pow-base8-four-level-52089b962.patch`.

### Zero-base POW milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `0**x`, official values | **passing** | DPU sign/zero/NaN masks |
| infinities, signed zero, NaN regression | **10/10 passing** | **passing** |
| final `0.7**x` | exponent 3 outside generic LUT range | next POW subgroup |

The strict lowering avoids `EXP2(x*-inf)` entirely.  DPU masks select zero
or one, FDIV synthesizes positive infinity for negative exponents, and a
comparison-derived `0/0` restores NaN.  The permanent regression passes in
4.29 seconds and the official method advances to its final 0.7-base case in
77.93 seconds.  PR1 remains **79/79 in 6.55 seconds**.  No host operator
fallback is used.

Recovery patch: `rockchip-zero-base-pow-4804f8bc5.patch`.
