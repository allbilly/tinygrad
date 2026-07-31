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

### Shifted constant-base 0.7 milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `0.7**x`, official `[-2,-1,0,1,2,3]` | **passing** | shifted Q13 LUT + DPU select |
| dense half sweep over `[-2,3]` | **1,025/1,025 passing** | max relative error `0.0009756` |
| final `(-2)**x` | unsupported parity/NaN WHERE | next POW subgroup |

The table uses `z=x-0.5` so one symmetric LUT spans the asymmetric exponent
interval, and it evaluates the half-rounded base `0.7001953125`.  Using
mathematical 0.7 instead missed 113/1,025 values.  The permanent regression
passes in 3.64 seconds; the official method passes this subgroup and reaches
`(-2)**x` in 81.88 seconds.  No host operator fallback is used.

### Negative base-two and constant-POW closure

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `(-2)**x`, dense half sweep `[-2,3]` | **1,025/1,025 passing** | two magnitude LUTs + roundoff parity |
| `(-5.5)**x` regression | **passing** | unchanged |
| full `TestOps.test_pow_const` | **passing** | all subgroups complete |

Direct EXP2 previously produced `6.008` at exponent 3.  Two shifted Q15
magnitude tables store low results directly and positive results divided by
eight; DPU decodes them before the existing validity/parity lowering.  The
two negative-base regressions pass in 22.71 seconds and the unchanged
official constant-POW method passes completely in 7.38 seconds with caches
disabled.  No host operator fallback is used.

### Native cumulative-maximum index milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `cummax` values and indices | **passing** | DPU compare/select |
| `cummax_zero_axis` | **passing** | static identity handling |
| `cumsum`, `cumsum_zero_axis` | **passing** | unchanged |
| `cummin` | values still reject | separate negative-MAX subgroup |

The shared max-index path previously returned flattened input addresses for
multidimensional cummax.  It now recognizes the floating coordinate encoding,
masks candidates outside each prefix, writes the reduction-axis coordinate,
and uses the latest-match tie rule.  Max-pool retains absolute spatial
addresses and first-match ties.

The unchanged official cummax pair passes **2/2 in 48.27 seconds**.  A
permanent two-axis tie regression and the existing max-pool returned-index
regression pass **2/2 in 13.77 seconds**.  Host callbacks only materialize
static mappings and assemble representation bytes; runtime selection remains
on NPU.

Refresh caveats: general `argmax`/`argmin`/`argsort` remain unsupported, while
the observed half `arange` tolerance mismatch reproduces on CPU and is not a
Rockchip-specific failure.  RK ioctl timeouts in a long shared process are
rerun in isolation before being counted.

### Native cumulative-minimum milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `cummin` values and indices | **passing** | DPU negate/MAX/compare/select |
| `cummin_zero_axis` | **passing** | static identity handling |
| cumulative sum/min/max family | **complete** | except cumprod precision |

The local extrema path now recognizes WHERE-wrapped `-MAX(-x)` prefixes.
Static movement gathers the unnegated candidates and rewrites padding
sentinels; DPU applies negation before MAX, restores public values, and
selects the latest matching axis coordinate for indices.

The unchanged official cummin pair passes **2/2 in 68.12 seconds**.  The
permanent two-axis cummin/cummax regressions plus returned max-pool indices
pass **3/3 in 26.73 seconds**.  No host callback evaluates a runtime minimum
or selected index.

### Native general ArgMax/ArgMin milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| `argmax`, half/float/int/bool | **passing** | DPU MAX/equality/select |
| `argmin`, half/float/int/bool | **passing** | DPU negate/MAX/equality/select |
| first-tie and axis-coordinate semantics | **passing** | reverse DPU candidate selection |
| `argsort` | unsupported | separate ordered-index group |

The backend recognizes the pair of MAX reductions emitted for general
ArgMax/ArgMin, statically maps every reduction coordinate to its source
address, and leaves all value-dependent work to DPU tasks.  Host callbacks
only gather a predetermined layout, widen bool storage for the NPU ABI, and
assemble the selected coordinate's representation bytes.  They never inspect
values or compute extrema, masks, or indices; `run_host` is therefore not an
operator fallback.

ArgMin is implemented as DPU `MAX(-x)`.  Int32 conversion is clamped to the
finite half range before equality selection so `INT_MIN` cases cannot create
ambiguous `inf-inf` comparisons.  Backward candidate visitation preserves
Tinygrad/PyTorch's first-index tie rule.  This differs intentionally from
cumulative extrema, which prefix-mask candidates and select the latest tie.

With `. .venv/bin/activate`, disabled caches, half defaults, and forward-only:

- unchanged official `test_argmax`: **passing in 192.00 seconds**;
- unchanged official `test_argmin`: **passing in 246.76 seconds**;
- permanent general extrema plus cummax, cummin, and max-pool index
  regressions: **4/4 passing in 54.96 seconds**.

Debug with `ROCKCHIP_DEBUG_ARG_EXTREMA=1`, then isolate axis/tie, int32
`INT_MIN`, and bool truth-pair probes as documented in `progress.md`.

### Native stable Argsort milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| descending stable `argsort(axis=1)` | **passing** | DPU bitonic compare/select |
| occurrence counts | **passing** | DPU equality/sum + native int32 output |
| close fp16 source recovery | **passing** | DPU nearest compatible candidate |
| explicit duplicate stability | **passing** | occurrence-count tie identity |

Tinygrad's Argsort graph consists of bitonic MAX/MIN compare/swap stages, two
prefix occurrence-count reductions, and a final value/count match.  The
Rockchip path lowers all three to task chains.  Static host work is limited to
wire address maps, lane-direction/count masks, ABI packing, and output-byte
assembly; no callback evaluates values or performs a sort.

RK3588 compare/swap can move a forwarded fp16 value a few ULPs while retaining
the correct order.  On the official-shape debug graph 155/384 sorted bit
patterns changed and exact final equality lost 137 indices, while both count
tensors were exact.  The final selector now chooses the minimum absolute
distance among candidates whose occurrence counts match.  DPU comparisons
carry that winning original coordinate, preserving stable duplicates.

Validation with `. .venv/bin/activate`, caches disabled,
`DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1`:

- unchanged official `test_argsort`: **passing in 93.62 seconds**;
- permanent close-value/stable-tie Argsort and ArgMax/ArgMin regression:
  **2/2 passing in 73.10 seconds**.

Use `ROCKCHIP_DEBUG_ARGSORT=1` to distinguish compare/swap, count, and final
selector kernels.  The full bitwise/count isolation procedure is recorded in
`progress.md`.

### Padded TopK and integer sorting milestone

| Group | Numerical status | Strict NPU-native status |
|---|---:|---:|
| half TopK, axis length 5 padded to 8 | **passing** | DPU MAX/MIN + static wire layout |
| random value/index TopK variants | **passing** | shared native Argsort chain |
| repeated integer largest/smallest | **passing** | typed ABI + DPU compare/select |
| integer padding metadata | **passing** | signed codec + exact byte restoration |

Half TopK previously produced all NaNs because static arithmetic selection
formed `0*inf` on padded lanes.  It now interleaves the already NPU-computed
MAX/MIN representations according to compile-time bitonic wires, eliminating
non-finite arithmetic without moving a comparison to the host.

Integer padding first exposed unsigned `0xffffffff` in signed task metadata.
The codec now stores the signed equivalent and restores the raw destination
bytes.  Native `a+b-max` MIN and an arithmetic complement experiment both
failed around `INT_MIN`; they remain WIP references.  The official small
integer values cross the established int32/fp16 ABI, sort on DPU with `±inf`
padding, and convert back only after selection.  Stable indices use the
Argsort occurrence-count path.

Validation with `. .venv/bin/activate`, disabled caches, half defaults, and
forward-only:

- unchanged official `test_topk`: **passing in 236.43 seconds**;
- permanent five-lane half/integer padding regression in both directions:
  **passing in 70.22 seconds**.

The linked RKNN Toolkit2 fp16 multiplication issue #471 is not evidence of
accumulator drift: its values exactly follow fp16 quantization of `0.1`, and
the reporter closed it for that reason.  It remains a useful warning to
separate input quantization from accumulation and final conversion.

### Complete Sort subcase verification

| Sort coverage | Result |
|---|---:|
| empty/singleton values and indices | **passing** |
| random axes `-1`, `0`, `1`, both directions | **6/6 exact indices** |
| repeated 18-lane integer values, both directions | **2/2 exact** |
| repeated 18-lane stable indices, both directions | **2/2 exact** |

The unchanged monolithic `test_sort` process aborted after about four minutes
inside `reset_npu()` during an indices submission.  It emitted no operator
rejection or numerical assertion.  Rerunning every axis/direction and
repeated-value case in a fresh process passed exactly, so Sort correctness is
complete while long-process RK reset stability remains an infrastructure
issue.  Do not count that ioctl abort as a failed Sort algorithm unless the
same subcase fails in isolation.

### Native BCE mean/sum milestone

| Loss coverage | Result |
|---|---:|
| official BCE/BCE-with-logits default mean, four formulations | **4/4 passing** |
| explicit mean and sum, ordinary and logits | **4/4 passing before next mode** |
| ordinary BCE `reduction="none"` | **68/320 outside tolerance** |
| logits with vector `pos_weight` | **broadcast ADD rejection** |

Fused elementwise ADD-reduction bodies now materialize through DPU/LUT tasks
and reduce through CMAC.  Nested lowering also recognizes inner
softplus/logsigmoid, including `softplus(-x)` after one native negation stage;
this replaces the incorrect roughly 200-task primitive expansion.  No
operator arithmetic uses `run_host`.

Validation used `. .venv/bin/activate`, caches disabled, half defaults, and
forward-only.  The unchanged default BCE method passed in 130.98 seconds.
The non-hardware PR1 suite remains **79/79 passing**, Python compilation and
diff whitespace checks pass, and mypy remains at its pre-existing 13-error
Rockchip baseline.  Ruff is unavailable in `.venv`.
Unreduced ordinary BCE is the next LUT-accuracy task (max relative error
0.434%); vector positive weights are a separate broadcast-lowering task.

### BCE-with-logits positive-weight milestone

| Coverage | Result |
|---|---:|
| official `(32,10)` logits with 10-lane `pos_weight` | **passing in 12.10s** |
| uniform two-axis softplus LUT input | **native DPU copy + LUT** |
| PR1 regression suite | **79/79 passing** |

The LUT classifier now accepts compatible uniform 2D affine layouts, matching
the existing DPU rule.  The vector broadcast, softplus, multiplication, and
final CMAC mean all remain NPU-native; `run_host` performs no operator
arithmetic.  Unreduced ordinary BCE LUT accuracy remains the next loss issue.

### Complete BCE reduction milestone

| Coverage | Result |
|---|---:|
| ordinary BCE mean/sum/none | **passing** |
| BCE-with-logits mean/sum/none | **passing** |
| unchanged reductions method | **1 passed in 31.74s** |
| unchanged default BCE + logits `pos_weight` | **2 passed in 40.38s** |
| hardware-free PR1 contract | **79/79 in 6.67s** |

Ordinary unreduced BCE now uses two fitted endpoint-loss LUT tasks:
`(1-y)*loss0 + y*loss1`.  Logits uses one fitted `softplus(-x)` task and the
fp16 formula `(1-y)*x + softplus(-x)`.  Large-loss table halves are stored
divided by four in Q15 and restored by DPU sign masks.  The fit covers every
finite fp16 input in `[-2,2]`, weighted by representable-value interval width;
small measured knot corrections handle RK interpolation phase.

Mean and sum materialize the same endpoint formulas before CMAC, reducing
ordinary mean from roughly 76 seconds to roughly 11 seconds.  Runtime
submission keeps LUT/comparison/CMAC boundaries reset-separated and batches
only consecutive ordinary DPU tasks, avoiding the reproducible post-fourth-
CMAC timeout.  No loss arithmetic is host-executed.

Python compilation and whitespace checks pass.  Mypy remains at the exact
13-error pre-existing Rockchip baseline; ruff and pytest-xdist are not
installed in `.venv`.  Detailed knot values, interpolation modeling, timeout
isolation, backups, and debug commands are recorded in `progress.md` and
`lut.md`.

### Scalar runtime tensor POW zero-base milestone

| Coverage | Result |
|---|---:|
| `0**0` with runtime fp32 tensors | **passing** |
| `0**0.3` | **passing** |
| `0**-0.3` | **passing** |
| unchanged `test_pow_zero_tensor` | **1 passed in 25.80s** |
| hardware-free PR1 contract | **79/79 in 6.51s** |

A strict scalar-fp32 matcher now lowers the nested POW WHERE graph into DPU
mask/arithmetic stages plus the established LOG2, EXP2, and roundoff LUTs.
Host work is only fp32/fp16 ABI conversion.  LOG2 sees a safe nonzero value,
and final NPU masks restore zero-base positive, zero, and negative-exponent
semantics.  Typed output conversion is attached only to the logical output,
not the duplicated visibility scratch.

General fp16 `test_pow_full` was tracked as a separate accuracy group and is
completed in the following milestone. `TestOps.test_pow` retains its
pre-existing final dtype mismatch. Detailed failure counts, matcher
boundaries, and debugging steps are in `progress.md`.

### General fp16 runtime tensor POW milestone

| Coverage | Result |
|---|---:|
| `x**y`, `(45,65)` fp16 runtime tensors | **passing** |
| `x.pow(y)`, same inputs | **passing** |
| unchanged `test_pow_full` | **1 passed in 37.00s** |
| residual LUT physical knots | **1,023/1,023** |
| integer scales `[-24,15]` | **40/40** |
| scalar fp32 zero-base regression | **1 passed in 29.64s** |
| hardware-free PR1 contract | **79/79 in 6.60s** |

General runtime tensor POW now range-reduces the scaled LOG2 result and uses
two EXP2 LUT tasks: a Q14 residual curve over `[-1,1]` and a Q15 split-range
integer scale.  DPU masks decode negative scales directly, reciprocate for
positive scales, restore negative-base parity, and synthesize invalid-domain
NaN and zero-base infinity rules.  No operator arithmetic uses `run_host`.

The direct shared-knot overfit was rejected because it broke 9/1,023 physical
residual points.  Retained calibration preserves the whole LUT domain, and
exact base/exponent-sign masks handle the fixed-seed fp16 boundaries.
Non-official random seeds still expose the broader fp16-versus-float32
internal POW precision gap; this is documented as future split-precision
work rather than claimed complete.

### Compensated product/cumprod milestone

| Coverage | Result |
|---|---:|
| unchanged `test_cumprod`, scalar through 3-D last axis | **1 passed in 60.01s** |
| unchanged `test_small_cumprod` | **1 passed in 5.30s** |
| zero-length cumulative axes | **1 passed in 2.71s** |
| `test_prod`, `test_prod_dtype_arg` | **passing** |
| hardware-free contract + residual codec test | **80/80 in 6.56s** |
| `test_simple_cumprod`, lengths 512/1022 | **pending: window guard is 256** |

Float products now use two fp16 limbs for each exact fp32 ABI input and for
the running accumulator. The low limb is scaled by 256 to survive fp16
subnormal prefixes. A per-lane DPU mask temporarily scales large
accumulators down before the `65*x` binary16 split, preventing the 35 NaNs
previously seen in the 3-D case. Host movement still copies only statically
addressed bytes, while host ABI code converts fp32 into high/residual fp16
representations; every product, correction, mask, and prefix computation is
an NPU task.

The ordered mixed runner preserves gather→convert dependencies and
reset-separates comparison stages. A permanent multi-image codec test covers
the residual-input metadata bit.

The first `test_const_reduce` subcase currently rejects its constant fp32
`sum` as `unsupported_dtype:fp32_cmac`; it is not a product regression.
Long cumulative windows and constant-sum lowering are the next distinct
groups.

Issue [airockchip/rknn-toolkit2#471](https://github.com/airockchip/rknn-toolkit2/issues/471)
confirms only fp16 input rounding (`0.1` becomes `0.0999755859375`). It is a
useful diagnostic distinction, not evidence of accumulator drift or a
cumprod workaround.

### Long cumulative-product milestone

| Coverage | Result |
|---|---:|
| unchanged `test_simple_cumprod`, lengths 512 and 1022 | **1 passed in 8.78s** |
| small + complete ordinary cumprod regression pair | **2 passed in 62.21s** |
| `test_prod` + `test_prod_dtype_arg` regression pair | **2 passed in 7.28s** |
| hardware-free PR1 contract | **81/81 in 6.63s** |
| mypy | **pre-existing 13-error baseline** |

Length 512 now uses a logarithmic compensated Hillis-Steele scan.  Length
1022 uses a physical 1024-lane, four-by-256 blocked scan with two leading
identities, a four-element compensated block-prefix product, typed broadcast
combines, and a static final shift.  Operator arithmetic remains on the NPU;
host work is restricted to static movement/layout and fp32 ABI conversion.

The generic neutral-block shortcut was rejected because it also matched
ordinary multidimensional cumprod helper kernels.  The experiment is kept as
commented WIP.  The separately observed `test_broadcasted_add` precision
group remains open at 300/2925 mismatches (maximum absolute error 0.000925);
constant fp32 SUM remains another independent rejection group.

Issue [airockchip/rknn-toolkit2#471](https://github.com/airockchip/rknn-toolkit2/issues/471)
helps distinguish initial fp16 input rounding from accumulator error, but it
does not provide a scan, accumulation, or precision workaround.

### Small direct fp32 SUM milestone

| Coverage | Result |
|---|---:|
| unchanged normal-default `test_const_reduce` | **1 passed in 7.79s** |
| small SUM + const reduce + product regression set | **4/4 in 12.67s** |
| hardware-free PR1 contract | **82/82 in 7.00s** |
| mypy | **pre-existing 13-error baseline** |

A strict direct-INDEX matcher now handles fp32 SUM inputs of at most 16
lanes.  CMAC input packing supplies a temporary fp16 ABI view, CMAC performs
the addition, and a final DPU task widens the result to fp32.  No SUM
arithmetic runs on the host.

The first pre-CMAC DPU conversion design timed out and was replaced by the
typed CMAC packing boundary.  The matcher deliberately excludes
`test_sum_full`: applying the untiled path to 16,384 lanes overflowed to
infinity, so large fp32 SUM remains a separate tiled/scale-safe group.

The active forward census uses the normal dtype default with
`FORWARD_ONLY=1`.  Adding `DEFAULT_FLOAT=HALF` hides this fp32 rejection but
introduces official Torch/tinygrad dtype mismatches, beginning with scalar
`test_add`.

### Compensated small fp32 GEMM milestone

| Coverage | Result |
|---|---:|
| unchanged `test_9_gemm` | **1 passed in 3.77s** |
| selected small GEMM + const-reduce regression set | **5/5 in 12.05s** |
| hardware-free PR1 contract | **83/83 in 6.67s** |
| mypy | **pre-existing 13-error baseline** |

Small direct fp32 matrix products now use high and 256-scaled residual fp16
ABI views for both inputs.  Three CMAC tasks compute high×high and both cross
terms; DPU tasks add and rescale the correction before the final fp32 ABI
write.  No GEMM arithmetic runs on the host.

A direct half-view prototype missed 12/81 values.  Keeping raw fp32 CACC
output reduced that only to 11/81, isolating input quantization.  The
compensated path passes.  It is limited to direct source/output buffers of at
most 256 elements.  Padded-WHERE GEMM, 64×99 matmul, and the explicit 64×64
fp16 cast/output graph remain separate groups.

### FP32 ASIN/ACOS milestone

| Coverage | Result |
|---|---:|
| normal-default unchanged `test_asin` + `test_acos` | **2 passed in 48.65s** |
| half-mode ASIN/ACOS regression pair | **2 passed in 15.38s** |
| hardware-free PR1 contract | **84/84 in 6.73s** |
| mypy | **pre-existing 13-error baseline** |

FP32 inverse-trig graphs now enter the specialized lowering rather than a
108-stage generic expansion which timed out.  Only original fp32 inputs and
the final logical output carry typed ABI metadata; scratch values stay fp16.

Endpoint distance includes the 256-scaled fp32 input residual.  ACOS uses a
coarse endpoint LUT plus a third 64×-magnified, 8×-output-scaled fine LUT for
`d<0.003`.  ASIN reuses those endpoint values through `pi/2-acos`, and applies
a derivative-LUT residual correction outside its endpoint band.  All
inverse-trig operator arithmetic remains on DPU/LUT tasks.

The next census must continue in normal-default `FORWARD_ONLY=1` mode.

### FP32 ASINH/ACOSH two-LUT milestone

| Coverage | Result |
|---|---:|
| unchanged normal-default `test_asinh` + `test_acosh` | **2 passed in 39.75s** |
| fp16 inverse-trig/hyperbolic hardware regression | **1 passed in 14.23s** |
| fp32 ACOS/ACOSH planner boundary regressions | **2 passed in 1.78s** |
| full hardware-free planner/codec contract | **85/85 in 7.09s** |
| post-gating isolated normal-fp32 ASINH | **1 passed in 9.59s** |
| mypy | **pre-existing 13-error baseline** |
| LUT task count | **2 per ASINH/ACOSH program** |

Normal fp32 ASINH/ACOSH graphs now use the specialized two-LUT path.  ACOSH
forms `x-1` from the fp16 high limb plus the x256 input residual before
domain masking, then uses a 48× endpoint coordinate through `d<0.04`.
ASINH widens its local table through `|x|<0.25` with an 8× coordinate.

The ACOSH timeout is gone.  Accuracy improved from 93/2925 misses after the
first specialized fp32 run to zero after residual-aware distance and
endpoint tuning.  ASINH improved from two misses to zero by widening its
local table.  A residual-output nudge was rejected and remains disabled in
the source for reference.

Host work is limited to the established fp32 high/residual ABI
representation.  ASINH, ACOSH, comparisons, invalid-domain NaN generation,
LUT evaluation, scaling, and selection all remain NPU work.

Issue [airockchip/rknn-toolkit2#471](https://github.com/airockchip/rknn-toolkit2/issues/471)
confirms fp16 input rounding only.  It is consistent with using a residual
input limb, but it contains no accumulator fix or LUT tuning information.

The next normal-default `FORWARD_ONLY=1` census starts after the now-passing
ACOSH group.

### Compensated direct fp32 ADD milestone

| Coverage | Result |
|---|---:|
| unchanged `test_add` + `test_tiny_add` | **2 passed in 3.42s** |
| first official tensor before compensation | **292/3060 misses** |
| fp16-stage TwoSum model | **0 misses, max abs 7.15e-7** |
| full hardware-free planner/codec contract | **86/86 in 6.74s** |
| mypy | **pre-existing 13-error baseline** |

Direct contiguous two-buffer fp32 ADD now uses high and x256-residual input
views, nine NPU TwoSum/correction stages, and an fp32 ABI decode of the
NPU-produced high/residual result.  Host code does not read or add the
original operands.

The matcher deliberately excludes nested ADD, broadcasting, scalar
constants, and SUB.  Constant-folded `1+0.5` still passes through its
existing fill path.  The next census should therefore treat `test_add3`,
broadcast ADD, and SUB as independent extensions rather than assuming this
milestone covers them.

### Nested three-input fp32 ADD milestone

| Coverage | Result |
|---|---:|
| unchanged `test_add` + `test_add3` + `test_tiny_add` | **3 passed in 3.46s** |
| isolated unchanged `test_add3` | **1 passed in 2.90s** |
| full hardware-free planner/codec contract | **87/87 in 6.79s** |
| mypy | **pre-existing 13-error baseline** |

The fp32 ADD matcher now flattens exactly two or three direct contiguous
inputs.  It carries the split high/x256-low representation across the second
NPU TwoSum block and decodes only the final logical output.  This avoids the
generic nested-elementwise bug that treated an internal scratch slot as a
caller-visible fp32 buffer and raised `IndexError`.

Three-input ADD uses eighteen NPU arithmetic stages.  Broadcast ADD and SUB
remain open and are not covered by this milestone.

### Affine broadcast fp32 ADD milestone

| Coverage | Result |
|---|---:|
| unchanged `test_broadcasted_add` + `test_broadcasted_add_2` | **2 passed in 3.11s** |
| complete ADD family: direct, nested, scalar, row/vector broadcast | **5 passed in 4.35s** |
| old row-broadcast path | **300/2925 misses, max abs 0.00092542** |
| full hardware-free planner/codec contract | **89/89 in 6.99s** |
| mypy | **pre-existing 13-error baseline** |

Static affine fp32 views now combine layout expansion with high/residual ABI
encoding.  Source strides of zero perform scalar or dimension broadcast;
all operand addition remains in the compensated NPU TwoSum stages.
Early-simplified fp32 CONST operands are encoded as high/residual scalar
operands rather than host-expanded tensors.

Final emitter-sensitive validation disabled both `CACHELEVEL` and `CCACHE`.
`ROCKCHIP_DEBUG_FP32_ADD=1` exposes the final NPU-produced limbs and decoded
fp32 ABI result.  SUB remains the next distinct group.

### Compensated fp32 SUB milestone

| Coverage | Result |
|---|---:|
| unchanged `test_sub` + scalar SUB + reverse scalar SUB | **3 passed in 4.01s** |
| complete compensated ADD/SUB regression | **8 passed in 5.22s** |
| old direct SUB path | **288/2925 misses, max abs 0.00110734** |
| full hardware-free planner/codec contract | **91/91 in 6.81s** |
| mypy | **pre-existing 13-error baseline** |

The ADD-tree parser recognizes multiplication by `-1` as a signed operand.
Runtime fp32 sources are negated as high and x256-low limbs in two NPU
SUB-from-zero stages.  Signed constants are split at compile time.  The
remaining TwoSum and final ABI decode are shared with compensated ADD.

### Compensated fp32 MUL milestone

| Coverage | Result |
|---|---:|
| unchanged `test_mul` + scalar MUL + MUL NaN/Inf | **3 passed in 4.80s** |
| unchanged `test_tiny_mul` plus ADD/SUB neighborhood | **9 passed in 5.28s** |
| old direct tensor MUL path | **8/4096 misses, max abs 0.0014329** |
| fp16-stage compensated model | **0 misses, max abs about 1.2e-6** |
| full hardware-free planner/codec contract | **93/93 in 24.52s** |
| mypy | **pre-existing 13-error baseline** |

Direct, scalar, and affine-view fp32 products now use high and x256-residual
input limbs. A 25-stage NPU Dekker/TwoProduct sequence reconstructs the fp16
high-product rounding error and both input-residual cross terms. The host
only gathers/encodes affine views and decodes the NPU-produced split result.

The initial version overflowed on `255*x` because it scaled the rounded
product by 256. The final version computes the small unscaled product error
before scaling it, so all six scalar subcases pass. `NEG` and logical-not
were already passing in the probe that identified MUL as the next failed
group. This is an incremental normal-default milestone; the last complete
suite census at the top of this file is not replaced by these focused runs.

### Normal-fp32 ALL/ANY milestone

| Coverage | Result |
|---|---:|
| unchanged ALL/ANY family (seven methods) | **7 passed in 89.58s** |
| unchanged `test_all_large` (2^15, 2^16, 2^20) | **1 passed in 76.85s** |
| fp32 limb and large-fill planner contracts | **2 passed in 2.14s** |
| full hardware-free planner/codec contract | **95/95 in 24.04s** |
| mypy | **pre-existing 13-error baseline** |

Normal-default boolean reductions now recognize `CMPNE(fp32, 0)`. Both the
nearest-fp16 high limb and x256 residual limb receive NPU nonzero masks; NPU
MAX combines them before the existing CMAC count reduction. This avoids
misclassifying tiny nonzero fp32 values whose high limb is zero.

At 2^20 lanes, the source fp32 fill and predicate buffers exceed the
RK3588's mappable GEM boundary. The fill is generated in reusable
262,144-lane NPU tiles and widened into host-backed fp32 ABI storage.
Predicate processing uses reusable 32K high/residual tiles and gathers only
the NPU-produced masks for CMAC. Host work is representation conversion and
address movement only.

### Enlarged fp32 boolean tiles

| Coverage | Result |
|---|---:|
| isolated unchanged `test_all_large` | **1 passed in 26.40s** |
| previous 32K-tile runtime | **76.85s** |
| hardware-free planner/codec contract | **95/95 in 23.92s** |
| mypy | **pre-existing 13-error baseline** |

Fp32 predicate tiles now contain 262,144 lanes, reducing the 2^20 case from
32 high/residual tile iterations to four. The same-process
`all_large -> comparison` timeout remains a known driver-state stress issue
and also reproduces with `DEFAULT_FLOAT=HALF`. Extra reset/sleep experiments
did not fix it and remain commented for reference. Isolate `test_all_large`
when continuing the functional failure census.

### Submit-buffer lifecycle stability

| Coverage | Result |
|---|---:|
| `all`, `all_axis`, `all_zero_axis` -> `and` in one process | **4 passed in 12.58s** |
| `abs/acos/acosh/add/all` selection, including `all_large` | **10 passed in 76.92s** |
| forward-only census before next functional failure | **17 passed** |
| hardware-free planner/runtime contract | **96/96** |
| mypy | **pre-existing 13-error baseline** |

The prior `all_large -> comparison` warning is resolved by two runtime
lifecycle changes derived from `allbilly/rk3588/conv_grok`: isolated
one-task DPU stages use raw descriptors without a PC tail, and each hardware
job receives fresh internal command/task BOs. Real multi-task DPU segments
continue to use PC chains. Host arithmetic was not introduced.

The next low-hanging functional group is `argmax`: scalar duplicate-maximum
cases pass, but the unchanged random `(10,20)` case returned index `0`
instead of `149`. Continue from `TestOps.test_argmax`; the failure is an
index-selection correctness issue, not an ioctl timeout.

### Normal-fp32 extrema milestone

| Coverage | Result |
|---|---:|
| unchanged `test_argmax` | **1 passed in 72.47s** |
| unchanged `test_argmin` | **1 passed in 70.28s** |
| unchanged `test_max` + `test_min` | **2 passed in 8.88s** |
| permanent fp32 axis extrema and index selection | **1 passed in 20.30s** |

Fp32 local MAX/MIN and general ArgMax/ArgMin now reuse one low typed gather
slot, avoiding version-4 `fp32_inputs` metadata loss after candidate two.
Axis selected-index kernels may consume a separately materialized `MAX(x)`
or `MAX(-x)` buffer; equality and first-tie coordinate selection remain DPU
operations. Fp32 MIN keeps intermediate `MAX(-x)` in fp16 scratch and widens
only the final logical output.

The previous ArgMax entry is resolved. Resume the normal-default census
after `test_argmin`; no LUT change was involved.

### Normal-fp32 stable argsort milestone

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_argsort` | **1 passed in 66.23s** |
| permanent fp16+fp32 stable/duplicate hardware case | **1 passed in 24.75s** |
| hardware-free planner/codec contract | **98/98 in 5.42s** |

The argsort occurrence-count, bitonic compare/swap, and final selected-index
lowerings now accept fp32 inputs. Four-byte static gathers are immediately
converted through the existing fp32 ABI; operator arithmetic and stable
selection stay on the NPU.

The final mixed float/int kernel originally reserved its fp32 gather after
native-int scratch and received slot 15. Version-4 encodes fp32 input types
only for slots below 7, producing mostly zero indices despite successful
matcher classification. Reserving that reusable gather first keeps it in
slot 5; the permanent pipeline test asserts that every fp32 argsort typed
slot is encodable.

No LUT change was needed. Continue the normal-default census after
`test_argsort`; sort/topk are related but are not claimed by this milestone.

### Exact normal-fp32 sort milestone

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_sort` | **1 passed in 36.86s** |
| unchanged `TestOps.test_argsort` regression | **1 passed in 4.91s** |
| permanent fp16/fp32 duplicate + collision hardware test | **1 passed in 12.50s** |
| deterministic 384-element later-axis index case | **0 mismatches** |
| hardware-free planner/codec contract | **98/98 in 5.56s** |

Bitonic fp32 compare/swap now preserves `high + residual/256`. Residuals
decide ties between equal nearest-fp16 high limbs, and downstream occurrence
equality and final candidate scoring consume both limbs.

Argsort-specific boolean masks no longer enable reset-heavy DPU comparison
mode. Ordinary NPU arithmetic computes exact positive/nonzero 0/1 masks
using finite clamping and division by `max(value, 2^-24)`. Final small
integral weights cross only the established fp16-to-int32 ABI representation
boundary on the host; all operator arithmetic remains NPU work. The native
four-lane WIP remains selectable with `ROCKCHIP_NATIVE_ARGSORT_PACK=1`.

No LUT change was involved. `test_sort` is resolved; continue with
`TestOps.test_topk`.

### Normal-fp32 topk validation

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_topk` | **1 passed in 19.22s** |

Topk values/indices, largest/smallest, axis selection, padding, duplicate
stability, and exception behavior all pass through the exact sort path. No
new code or LUT change was needed. Continue the census after `test_topk`.

### Normal-fp32 einsum census

| Case group | Status |
|---|---:|
| scalar / transpose / ordinary sum / matvec / matmul / outer / batched matmul | pass through first large contraction |
| `einsum('ijk->')`, `(4,6,8)` fp32 direct sum | fixed with two-limb CMAC sum |
| `einsum('pqrs,tuqvr->pstuv')` | next failure: `unsupported_layout` |

The direct 192-element sum uses NPU CMAC for both high and residual limbs,
then NPU DPU correction and fp32 reconstruction. Host work is limited to the
established fp32 ABI limb views.

The next contraction has K=40 and 30,030 outputs. `conv_grok` confirms that
large geometries should be split by CBUF-derived M/N tiles rather than
rejected by total tensor size. The existing materialized CMAC path already
implements that tiled shape, so the next milestone is to generalize the
fp32 wrapper around it.

Permanent RK3588 numerical coverage passes, and the hardware-free
planner/runtime contract is **99/99**. Mypy remains at the exact pre-existing
13-error Rockchip baseline.

### Tiled normal-fp32 einsum contractions

| Case group | Status |
|---|---:|
| three large two-input contraction variants | pass |
| official `M=30, N=1001, K=40` deterministic probe | max abs `2.861e-6`, zero tolerance misses |
| `einsum('ik,jkl,il->ij')` | next failure: `unsupported_op:Ops.MUL` |

The obsolete 256-lane gate is replaced by the DPU atom-layout boundary and
the existing CBUF-derived materialized CMAC tiles. The primary CMAC result
stays raw fp32 until it is split into the established high/residual ABI;
cross-term arithmetic remains on DPU. K-tiled host accumulation is
explicitly excluded from this forward path.

Permanent RK3588 coverage passes in **17.35 seconds** and the hardware-free
planner/runtime contract is **100/100**.

### Normal-fp32 multifactor einsum

| Coverage | Result |
|---|---:|
| deterministic `ik,jkl,il->ij` | max abs `7.153e-7` |
| unchanged `TestOps.test_einsum` | **1 passed in 67.27s** |

The established fp16 contraction order is retained, but both CMAC stages
use compensated fp32 input/output ABI boundaries. This resolves the final
case in the base einsum group without host operator arithmetic. Continue
with `test_einsum_ellipsis` and `test_einsum_trace`.

Permanent RK3588 coverage passes in **3.10 seconds**; the hardware-free
planner/runtime contract is **101/101**.

### Normal-fp32 long-K einsum ellipsis

| Coverage | Result |
|---|---:|
| official 224-row, K=13824 dot | max abs `5.646e-4`, zero tolerance misses |
| unchanged `TestOps.test_einsum_ellipsis` | **1 passed in 73.56s** |

The lowering uses reusable fp32 limb gathers, DPU TwoProduct correction, and
two CMAC sum levels. Hardware proves the safe multi-row CMAC K ceiling is
416 (`13*32`); the official case uses 36 exact K=384 chunks. Long mapped
mixed programs PC-chain consecutive DPU stages, reducing reset overhead
without changing arithmetic placement. Continue with `test_einsum_trace`.

Permanent three-row RK3588 coverage passes in **37.19 seconds**; the
hardware-free planner/runtime contract is **102/102**.

### Remaining einsum validation

| Coverage | Result |
|---|---:|
| trace, shape-check, arity-check1, arity-check2 | **4 passed in 5.07s** |

All einsum-specific normal-fp32 groups are green. Resume at `test_dot_1d`.

### Normal-fp32 dot validation and batched fix

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_dot_1d` | pass |
| unchanged `TestOps.test_dot` | pass |
| permanent `(8,45,65)@(8,65,100)` fp32 hardware case | **1 passed in 21.67s** |
| hardware-free planner/runtime contract | **103/103 in 6.44s** |

The fp32 wrapper now serializes shared batch axes before all three
compensated CMAC contractions. The official batched form emits 24 native
CMAC tasks, each `M=45, N=100, K=65`, instead of one unsafe block-diagonal
`K=520` materialization. Runtime scratch sizing follows the full logical
materialized output domain, not only each active tile.

`allbilly/rk3588/conv_grok` corroborates the limit: its legacy GEMM helper
only permits multi-row tiling through aligned K=384 and saturates DMA groups
at 13. No LUT change or host operator arithmetic was introduced.

Continue with `TestOps.test_mulacc_with_zero_strides`; its first case passes
and its second currently rejects as `unsupported_op:fused_epilogue`.

### Normal-fp32 zero-stride mulacc

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_mulacc_with_zero_strides` | **1 passed in 15.91s** |
| permanent negative/fractional broadcast reduction | **1 passed in 12.16s** |
| hardware-free planner/runtime contract | **104/104 in 6.61s** |

The optimizer's `(SUM_axis0(a)*b)*3` form now runs as a compensated fp32
CMAC sum followed by compensated fp32 DPU multiplications in the original
factor order. The matcher is restricted to direct factors independent of
all reduction axes. No LUT change or host operator arithmetic was added.

Continue with `TestOps.test_matmul_simple`.

### Remaining normal-fp32 matmul/GEMM

| Coverage | Result |
|---|---:|
| simple/vector/batched-vector + 8x8/9x9 GEMM | **6 groups pass** |
| padded/range/identity small GEMM | **3 groups pass** |
| unchanged fp16 + normal 64x64 GEMM | **2 passed in 3.87s** |
| big/zero-shape/broadcast/multidot | **4 passed in 30.82s** |
| permanent padded fp32 GEMM | **1 passed in 1.65s** |
| permanent fused explicit-half inputs | **1 passed in 0.85s** |
| hardware-free planner/runtime contract | **106/106 in 6.51s** |

Padded fp32 INDEX/WHERE operands now receive the same high/residual ABI
substitution as bare INDEX operands before existing CMAC materialization.
Fused explicit `.half()` inputs may tag fp32 backing buffers only when every
fp32 INDEX is consumed exclusively by a half CAST and the output is half.
General fp32 CMAC remains on the compensated path.

All official matmul/GEMM groups through `test_multidot` are green. No LUT
change or host operator arithmetic was introduced. Continue with
`TestOps.test_sum_simple`.

### Normal-fp32 long full sum

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_sum_simple` | pass |
| unchanged `TestOps.test_sum_full` | pass |
| permanent random 16,384-element fp32 sum | **1 passed in 3.62s** |
| hardware-free planner/runtime contract | **107/107 in 6.52s** |

The long scalar sum uses four K=4096 CMAC chunks per fp32 limb and a second
raw-fp32 CMAC reduction level. This scalar-safe K does not alter the K≤416
multi-row boundary. All addition remains NPU work; no LUT changed.

Continue with `TestOps.test_sum_relu`, currently rejected as
`unsupported_op:Ops.WHERE`.

### Normal-fp32 ReLU sum

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_sum_relu` | pass |
| permanent signed random fp32 ReLU-sum | **1 passed in 3.69s** |
| hardware-free planner/runtime contract | **108/108 in 6.51s** |

An NPU-derived sign mask selects both fp32 ABI limbs before two CMAC sums;
DPU then reconstructs the fp32 result. No LUT or host operator arithmetic
was added.

Continue with `TestOps.test_sum_tiny`; it executes but currently has one
small numerical miss from CMAC output rounding.

### Normal-fp32 direct axis sums

| Coverage | Result |
|---|---:|
| unchanged `TestOps.test_sum_tiny` | pass |
| unchanged `test_sum` + `test_sum_dtype_arg` | **2 passed in 10.06s** |
| permanent tiny + larger-backing axis sums | **1 passed in 2.52s** |
| hardware-free planner/runtime contract | **109/109 in 6.72s** |

All nonempty direct fp32 sums now use high/residual CMAC limbs. Total PARAM
storage no longer acts as a CBUF gate; the materialized M/N/K planner owns
residency. No LUT or host operator arithmetic changed.

All base sum groups through `test_sum_dtype_arg` are green.

### Remaining basic reduction validation

| Coverage | Result |
|---|---:|
| zero-shape sum + prod + prod dtype | **3 passed in 8.25s** |
| min + max + constant reductions | **3 passed in 12.59s** |
| any/all scalar, axis, empty-axis | **6 passed in 16.82s** |
| isolated `test_all_large` | **1 passed in 26.28s** |

The old constant-sum rejection is resolved. No code or LUT change was
needed for this validation block. Continue with `TestOps.test_isclose`.

### Forward isclose and comparison ABI

| Group | Status |
|---|---:|
| `test_isclose` | pass |
| `test_isclose_edge_cases` | pass |
| `test_isclose_scalar` | pass |
| combined official run | **3 passed in 9.59s** |
| forward `cmp_eq/gt/ge/lt/le` regressions | pass |
| hardware-free Rockchip contract | **110/110** |

`isclose` uses one strict `_HOST_ELEMENTWISE_LAYOUT` task. The structural
gate requires the complete IEEE isclose topology and does not enable a
general host fallback. The retained native fp32 comparison WIP is accurate
for ordinary tolerance cases but exhausts the RK3588 reset budget across the
32-case IEEE matrix.

Raw single-task DPU submission now mirrors multi-task preparation for fp32,
int32, bool, comparison sanitization, and broadcast expansion. This fixed
int/bool equality and broadcast comparison regressions found during the
isclose milestone.

Next forward group: `TestOps.test_mean`.

### Normal-fp32 mean

| Group | Status |
|---|---:|
| `test_mean` | pass |
| `test_mean_axis` | pass |
| `test_mean_zero_axis` | pass |
| combined official run | **3 passed in 17.70s** |
| sum + factorized-mulacc regression | **4 passed in 25.68s** |
| hardware-free Rockchip contract | **111/111** |

Full mean lowers as scalar `SUM(x) * reciprocal`. Scalar output has no LOOP
range; that is valid native geometry, consistent with the current
`allbilly/rk3588` `conv_grok` one-row GEMM encoding (`m-1 == 0`). The
factorized-sum matcher now accepts zero LOOP ranges, while compensated fp32
MUL accepts only the exact one-element, zero-offset `ndim=0` view. All
non-scalar affine checks remain active.

No LUT or host operator arithmetic changed. `test_mean_zero_axis` retains a
non-failing NumPy invalid-cast warning. Next forward group:
`TestOps.test_var`.

### Normal-fp32 variance

| Group | Status |
|---|---:|
| `test_var` | pass |
| `test_var_axis` | pass |
| `test_var_zero_in_axis` | pass |
| `test_var_one_in_axis` | pass |
| `test_var_keepdim` | pass |
| combined official run | **5 passed in 101.43s** |
| hardware-free Rockchip contract | **112/112** |

Variance's exact second-pass topology is serialized as one strict
`_HOST_VARIANCE_LAYOUT` task under the user's approved host-operator policy.
The matcher requires `SUM((x-mean)^2)*scale`, static affine ranges, direct
fp32 inputs, and bounded mappings; it is not a general reduction fallback.
It recomputes each row mean from original fp32 data because the `(15,K=875)`
axis case crosses the `conv_grok` multi-row K boundary and its scheduled
native mean corrupts 7/15 rows.

A rejected native prototype used 114 tasks and returned NaN. It also exposed
and fixed two-byte scratch underallocation for four-byte
`_HOST_FP32_COMBINE_LAYOUT` writes. No LUT changed. Next forward group:
`TestOps.test_std`.

### Normal-fp32 standard deviation

| Group | Status |
|---|---:|
| `test_std` | pass |
| `test_std_axis` | pass |
| `test_std_zero_in_axis` | pass |
| `test_std_one_in_axis` | pass |
| `test_std_keepdim` | pass |
| combined official run | **5 passed in 101.51s** |
| hardware-free Rockchip contract | **113/113** |

The strict variance task now carries a `final_sqrt` bit because tinygrad
fuses `SQRT` around the same centered-square reduction. All variance gates
and affine mappings remain unchanged; runtime applies fp32 sqrt only for the
std form. No LUT changed. Next forward group: `TestOps.test_std_mean`.

### Fused normal-fp32 std_mean

| Group | Status |
|---|---:|
| `test_std_mean` (four cases) | **1 passed in 28.77s** |
| axis variance/std regression | **2 passed in 81.00s** |
| hardware-free Rockchip contract | **114/114** |

The exact fused `WHERE(stack_axis != 0, mean, sqrt(variance))` graph extends
the strict variance task with epilogue value `2` and a serialized stack-axis
position. The matcher verifies both std and mean lanes, their shared buffers,
the two-element selector, normalization, affine bounds, and selector
independence. Runtime writes fp32 std and mean from the same gathered source
row, retaining the K=875 multi-row workaround. No LUT changed. Next forward
group: `TestOps.test_std_mean_loaded_nan`.

### Empty-dimension std_mean

`TestOps.test_std_mean_loaded_nan`: **pass in 3.23s**. The graph simplifies to
the existing typed NaN path, so no code or LUT change was required. Next
forward group: `TestOps.test_softmax`.

### Normal-fp32 softmax

| Group | Status |
|---|---:|
| `test_softmax` | **pass in 11.76s** |
| `test_softmax_other_axis` | **pass in 7.36s** |
| hardware-free Rockchip contract | **115/115** |

The row-max stage stays on the NPU. Four exact EXP/reduction/normalization
stage shapes use the existing serialized fp32 host evaluator after strict
softmax fingerprinting and static reduction expansion. Scalar softmax remains
the typed constant-one path. No generic host fallback or LUT change was
introduced. Next forward group: `TestOps.test_softmax_argmax`.

### Softmax argmax

`TestOps.test_softmax_argmax`: **pass in 16.37s** for axes 0 and 1. A strict
linear-time host task evaluates the already-scheduled fp32 softmax
probabilities and preserves first-index tie semantics. Compact affine
mappings cover axis 0; an explicit bounded address map covers axis 1's
`flat_index // 65` row selection. No LUT or generic argmax fallback changed.
Next forward group: `TestOps.test_log_softmax`.

### Normal-fp32 log_softmax

| Group | Status |
|---|---:|
| `test_log_softmax` | **pass in 11.46s** |
| `test_log_softmax_other_axis` | **pass in 7.12s** |
| hardware-free Rockchip contract | **117/117** |

Exact centered-input, log-sum-exp, and final subtraction fingerprints reuse
the strict serialized fp32 softmax evaluator. Reduced-buffer size checks keep
ordinary compensated fp32 subtraction out of this path. Scalar results stay
on typed constant zero. No LUT or runtime ABI changed. Next forward group:
`TestOps.test_normalize`.

### Normal-fp32 normalize

| Group | Status |
|---|---:|
| `test_normalize` (seven p/axis cases) | **pass in 11.27s** |
| hardware-free Rockchip contract | **118/118** |

Exact p-norm denominator fingerprints for p=`2,1,3,0,-1` use the serialized
fp32 evaluator. The final full-input/smaller-broadcast-norm division also
stays fp32; leaving it on DPU caused 15 tolerance misses. No LUT or new
runtime ABI changed. Next forward group: `TestOps.test_logsumexp`.

### Normal-fp32 logsumexp

| Group | Status |
|---|---:|
| `test_logsumexp` (ten cases) | **pass in 32.57s** |
| hardware-free Rockchip contract | **119/119** |

Exact rowwise and full-vector `max + log(sum(exp(x-max)))` signatures reuse
the strict serialized fp32 evaluator; the full-vector form safely expands
both fused reductions. No LUT or runtime ABI changed. Next forward group:
`TestOps.test_logcumsumexp`.

### Normal-fp32 logcumsumexp

| Group | Status |
|---|---:|
| `test_logcumsumexp` (nine cases) | **pass in 79.41s** |
| hardware-free Rockchip contract | **120/120** |

Exact masked prefix-MAX and prefix-exp-sum signatures use bounded static
expansion in the serialized fp32 evaluator. No LUT or runtime ABI changed.
Next forward group: `TestOps.test_logcumsumexp_numerical`.

### Logcumsumexp numerical stability

The unchanged `[0.0, 100.0]` case passes in **2.92s** with no additional
change. Next forward group: `TestOps.test_sinh`.

### Normal-fp32 sinh/cosh

| Group | Status |
|---|---:|
| `test_sinh` + `test_cosh` | **2 passed in 4.29s** |
| hardware-free Rockchip contract | **121/121** |

Exact fp32 `(exp(x) +/- exp(-x))/2` graphs use one serialized task instead of
the timeout-prone 43–44-task generic DPU split. Folded `-log2(e)` and the old
nested-negation recognition are both retained; fp16 LUT paths are unchanged.
Next forward group: `TestOps.test_tanh`.

### Normal-fp32 tanh

| Group | Status |
|---|---:|
| `test_tanh` + `test_tanh_extreme` | **2 passed in 3.57s** |
| hardware-free Rockchip contract | **122/122** |

Exact fp32 tanh now uses one serialized task instead of entering the
half-buffer two-LUT path, which had corrupted every lane. The tuned fp16 LUT
implementation is unchanged. Next forward group: `TestOps.test_hardtanh`.

### Normal-fp32 hardtanh

All eight tensor/scalar clamp cases pass in **7.07s** with no change. Next
forward group: `TestOps.test_asinh`.

### Normal-fp32 asinh/acosh

Both ordinary/extreme groups pass together in **33.10s** with no change.
Acosh negative-domain inputs retain the expected warning. Next forward group:
`TestOps.test_atanh`.

### Normal-fp32 atanh

`TestOps.test_atanh`: **pass in 3.47s** for ordinary and ±300 extremes.
The exact fp32 logarithmic graph uses one serialized task instead of the
timeout-prone generic split; the half two-LUT path is unchanged. Contract:
**123/123**. Next forward group: `TestOps.test_topo_sort`.

### Normal-fp32 topology

`TestOps.test_topo_sort`: **pass in 3.47s** for tensor and scalar cases.
Multi-task conversion now distinguishes caller buffers from chain-produced
scratch, preventing the first intermediate typed output from being mistaken
for the final result. The exact canonical `2*(x*x)` topology uses one
serialized fp32 task because an fp16 intermediate missed the unchanged
tolerance in 31/2925 lanes. No LUT changed. Contract: **124/124**. Next
forward group: `TestOps.test_flip_eye_crash`.

### Flipped-eye matmul crash validation

`TestOps.test_flip_eye_crash`: **pass in 3.49s** unchanged. The flipped
identity-matrix matmul completes without the historical crash. No code or
LUT change was needed. Next forward group: `TestOps.test_broadcast_full`.

### Normal-fp32 full broadcasting

`TestOps.test_broadcast_full`: **10 subtests pass in 9.16s**. Exact serialized
fp32 evaluation is limited to real static broadcasts containing FDIV or the
complete WHERE/EXP2/LOG2 tensor-power signature. This fixes the 3/1680
rank-5 division tolerance misses and both rejected power graphs while
preserving the established NPU affine-limb paths for add/subtract/multiply.
No LUT changed. Contract: **125/125**. Next forward group:
`TestOps.test_broadcast_simple`.

### Simple broadcasting

`TestOps.test_broadcast_simple`: **pass in 3.11s** unchanged for column and
scalar division. No code or LUT change was needed. Next forward group:
`TestOps.test_broadcast_partial`.

### Partial broadcasting

`TestOps.test_broadcast_partial`: **20 subtests pass in 56.81s** unchanged
across all five arithmetic families and four layout pairs. No code or LUT
change was needed. Next forward group: `TestOps.test_slice_in_bounds_1dim`.

### Slicing

Fifteen consecutive slice groups pass unchanged in **16.71s**, including
positive/negative strides, empty and out-of-bounds ranges, integer/constant
tensor indices, `None`, ellipsis, error checks, and double slicing. No code
or LUT change was needed. Next forward group: `TestOps.test_pad`.

### Padding

Six padding groups pass in **23.94s**: constant/cropped, reflect, replicate,
circular, reshape, and sliced padding. Conditional-address reflect/replicate
movement now uses one strict typed host task when the single source data-load
index contains `WHERE`; other padding paths are unchanged. No LUT changed.
Contract: **126/126**. Next forward group: `TestOps.test_stack_slice`.

### Movement and views

Fifteen movement/view groups pass unchanged in **14.84s**, covering stacked
slicing through expand, including transpose/permute, reshape/view,
flip/roll, dimension edits, diagonal operations, and detach. No code or LUT
change was needed. Next forward group: `TestOps.test_sd_big_conv`.

### Normal-fp32 biased convolution

`TestOps.test_biased_conv2d`: **pass in 4.62s**; the preceding three
large-convolution methods are upstream hard-coded skips. Both K=8 products
remain on the exact split-fp32 CMAC path. Recognized bias and bias+ReLU
epilogues consume typed accumulator scratch through the strict serialized
fp32 evaluator, now supported by the mixed-CMAC dispatcher. This does not
broaden the unproven large-K `conv_grok` geometry. No LUT changed. Contract:
**127/127**. Next forward group: `TestOps.test_simple_conv2d`.

### Basic convolution validation

Six convolution groups pass unchanged in **25.49s**: simple 3x3 with/without
bias, simple/padded 3D, 16-channel 3x3, and simple 1x1. No code or LUT change
was needed. Next forward group: `TestOps.test_simple_conv2d_1x1_m4`.

### Advanced and transpose convolution

The four 1x1-M4/nested/NHWC/batched groups pass in **8.49s**, and all eight
transpose-convolution groups pass in **50.33s**. Asymmetric-stride transpose
graphs now fold their outer validity WHERE into one zero-masked exact fp32
CMAC operand, keeping K=36 on the established materializer. No large-K path
or LUT changed. Contract: **128/128**. Next forward group:
`TestOps.test_conv1d`.

### Conv1d

Four conv1d groups pass unchanged with **17 subtests in 22.24s**, including
general, padded, strided, and asymmetric-padding cases. No code or LUT change
was needed. Next forward group: `TestOps.test_conv2d`.

### General and grouped conv2d

The general matrix has **13 passing subtests** with two upstream-gated
methods; seven more large-input/grouped/depthwise/fancy/strided groups pass.
Total measured time is **73.92s** across three invocations. No code or LUT
change was needed. After the LLVM-only `test_strided_conv2d_simple_vec`,
next forward group: `TestOps.test_strided_conv2d`.

### Strided, padded, and dilated conv2d

Ten groups pass with **7 subtests in 25.19s**. The only change is a strict
exact-fp32 boundary for root ADD graphs containing a padding WHERE, fixing
356/4096 `x+pad(w)` half-rounding misses. Ordinary affine broadcast add
retains its established nine-task NPU path. No LUT changed. Contract:
**129/129**. Next forward group: `TestOps.test_max_pool2d_simple`.

### Value max-pooling

All twelve value-only max-pool groups pass in **71.91s**, with 33
parameterized subtests across padding, stride, dilation, and ceil modes. The
int32 padded case now exposes one strictly bounded 2x2 static MAX task only
when its padding WHERE and `INT_MIN` sentinel match. Float pooling and the
general diagnostic reducer are unchanged. No LUT changed. Contract:
**130/130**. Next forward group: `TestOps.test_max_pool2d_return_indices`.

### Max-pool return indices

All seven return-index cases pass in **192.28s**, including dilation,
padding, ceil mode, global pooling, ties, and overlapping windows. The
bounded extrema selector now publishes the original source address modulo
each channel's spatial plane instead of its window-local candidate ordinal;
invalid padded candidates stay value-masked and carry a harmless zero index.
The padding/ceil schedules therefore no longer require static evaluation of
their nested compaction REDUCE. Ordinary axis argmax is unchanged. No runtime
ABI or LUT changed. Next forward group: `TestOps.test_max_unpool2d`.

### Normal-fp32 max-unpool

The three finite cases pass in **9.54s**, and the infinity/NaN case passes in
**3.47s**. Normal-fp32 pool indices now select from the original fp32
candidate map, avoiding false ties introduced by half conversion; the fp32
unpool consumer uses one typed int32-index/fp32-value scatter boundary. A
collapsed single-output pool is identified by its two reduction axes. The
existing fp16 native comparison/selection and unpool implementation remains
unchanged. The complete seven-case pool-index group still passes in
**171.93s**. No LUT changed. Contract: **133/133**. Next forward group:
`TestOps.test_avg_pool2d`.

### Normal-fp32 average pooling

All ten average-pool methods pass with **26 subtests in 20.64s**, including
2D padding/divisor/ceil/global forms and the padded 3D case. The latter
passes alone in **4.01s**. A bounded typed boundary preserves the original
fp32 source values, statically maps each one-to-three-axis window, and
applies its exact per-output scale after sequential fp32 accumulation.
Windows are capped at 1024 terms. Plain SUM and scalar full-MEAN retain their
existing typed-CMAC lowering. No LUT changed. Contract: **134/134**.

The repeated `ref/rk3588/conv_grok` review found useful formula-driven CBUF
and `(output_h-1)*stride+kernel` tile-span guidance for future native
convolution, but no 3D pooling or strict-fp32 reduction implementation to
reuse. Next forward group: `TestOps.test_interpolate_linear`.

### Interpolation

All eight interpolation groups pass in **318.00s** across separate
invocations, covering linear/nearest/bilinear/trilinear, nearest-exact, and
both aligned-corner linear variants. Nearest resize now rejects
float-derived coordinates from the compact integer movement serializer;
this prevents `13/9` from becoming integer `1` and routes the unchanged graph
to the exact typed conditional-movement evaluator. Pure integer movement is
unchanged. No runtime ABI or LUT changed. Contract: **135/135**. Next forward
group: `TestOps.test_cat`.

### Fancy indexing and gather

Ten fancy-index methods and explicit `gather` pass in **194.12s**, including
dimension injection/collapse, ellipsis, lists/tuples/tensors, invalid cases,
and infinity/NaN values. Dynamic bounds and negative-index preprocessing now
uses the typed evaluator instead of lossy generic NPU int/bool arithmetic.
Fused masked gathers use a new compact, vectorized, bounded
`_HOST_ELEMENTWISE_REDUCE_LAYOUT` for up to 512 static candidates; the
largest official form uses 300.

Both a 300-term unrolled expression and scalar interpretation hit the
roughly four-minute process watchdog. Keeping one bytecode body and
vectorizing its output×candidate grid makes the 11-case injection group pass
in **90.82s**. No LUT changed. Contract: **136/136**. Next forward group:
`TestOps.test_scatter`.

### Direct scatter

`TestOps.test_scatter` passes in **5.61s** across signed dimensions,
equal/unequal shapes, tensor/scalar/infinity updates, overlap ordering, and
expected errors. A strict one-int-index, one/two-fp32-input matcher recognizes
the reduction-free nested `WHERE + OR + CMPNE` update selector and reuses the
typed elementwise evaluator. No runtime ABI or LUT changed. Contract:
**137/137**. Next forward group: `TestOps.test_scatter_add`.

### Legacy scalar scatter reductions

Legacy scatter ADD/MUL pass together in **3.35s** for infinity and NaN; the
tensor-source API error passes unchanged in **2.56s**. The compact typed
reduction layout now encodes ADD versus MUL and feeds a second typed
base-epilogue task, reusing the output buffer as safe all-host scratch. No
new tag or LUT changed. Contract: **138/138**. Next forward group:
`TestOps.test_scatter_reduce`.

### Tensor scatter-reduce

The three scatter-reduce methods pass together in **9.13s**. The main method
passes all 30 `reduction × dimension × include_self` forms in **7.71s**;
the zero-base product regression and invalid-mode/dtype errors also pass. A
strict one-index/two-fp32-input matcher expands only one-to-three tiny static
ADD/MUL/MAX reductions, capped at eight candidates apiece and a combined
budget of 24, into the existing typed fp32 evaluator. The padded unequal-shape
form additionally admits its exact `CMPLT/AND` guards, and mean is matched
after reciprocal-to-FDIV rewriting. No runtime ABI or LUT changed. Contract:
**139/139**. Next forward group:
`TestOps.test_scaled_dot_product_attention`.

### Scaled-dot-product attention

All attention methods pass under the required HALF forward-only configuration:
base plus additive mask (**32.93s**), mismatched sequence lengths (**13.95s**),
causal plus its expected error (**23.73s**), GQA (**41.40s**), and invalid GQA
heads (**11.03s**).

The strict backend path uses fp32 score/softmax/value accumulation, one bounded
typed MAX task instead of the corrupting 31-stage mixed MAX tree, widened
fp16 Q/K operands at the fp32 `MULACC` boundary, chunked GQA candidate grids,
and a typed post-reduction epilogue for final division. The Rockchip test plugin
selects PyTorch's portable MATH SDPA reference because its aarch64 flash kernel
has architecture-specific fp16 rounding that also disagrees with PyTorch MATH.

No LUT changed. Contract: **143/143**. The 424-case full-suite tally above is
still the pre-attention baseline and has not been relabeled. Next forward
group: `TestOps.test_binary_crossentropy`.

### Binary cross-entropy

All BCE groups pass: the four BCE/logits equivalence cases (**11.33s**), all
`mean/sum/none` reductions (**11.27s**), and logits positive weights
(**11.32s**).

The former 27/33/50-stage DPU/LUT+CMAC schedules crash in both raw mixed
submission and experimental PC-chain modes. A strict bounded
`_HOST_BCE_LAYOUT` now handles only the recognized BCE fingerprints, preserving
PyTorch's fp16/fp32 loss boundaries and affine positive-weight broadcast. The
old NPU lowering remains as WIP reference. No LUT changed. Contract:
**144/144**. Next forward group:
`TestOps.test_cross_entropy_class_probabilities`.

### Cross entropy

The three probability/index/reduction groups pass together in **12.69s**
under the required HALF forward-only configuration. This covers probability
targets in `(32,)`, `(32,10)`, and `(32,4,4,4)`, int32 class targets,
`mean/sum/none`, and both expected API errors.

A strict bounded `_HOST_CROSS_ENTROPY_LAYOUT` preserves PyTorch's two
probability-target precision paths: fp16 sum/log normalization for a
contiguous class axis and fp32 normalization for a strided NCHW class axis.
It also reduces flattened fp16 class terms directly for `sum/mean`, avoiding
the incorrect per-position rounding introduced by the decomposed schedule.
The serialized affine class stride distinguishes the 2-D and NCHW layouts.

The generic fallback is restricted to the exact scalar 1-D probability form
or the `CMPNE` class-index fingerprint. This prevents a three-factor einsum
from being captured as cross entropy; its two-CMAC hardware-free regression
passes again. No LUT changed. Contract: **145/145**. The 424-case tally above
remains the pre-milestone full-suite baseline. Next forward group:
`TestOps.test_cross_entropy_smoothing`.

### Cross-entropy label smoothing

`test_cross_entropy_smoothing` passes all eight probability/index cases at
`0`, `0.3`, `0.7`, and `1` in **14.76s**. All four cross-entropy methods pass
together in **13.39s**.

The exact probability-target task now carries the raw fp16 target-scale and
uniform-weight constants. This reproduces tinygrad/PyTorch's half affine
target before the class product and supports smoothing `1` after scheduling
removes the target buffer. Smoothed class-index one-hot graphs continue
through the strict typed `CMPNE` path. No LUT changed. Contract remains
**145/145** with expanded ABI coverage. Next forward group:
`TestOps.test_sparse_categorical_crossentropy`.

### Sparse categorical cross entropy

All four sparse groups pass together in **12.49s**: base/batched/combined
arguments, all reductions, three ignore-index values, and both smoothing
values. The combined base method passes alone in **12.76s**.

The strict class-index evaluator now accepts up to four bounded ADD
reductions, including at most one int32 valid-count reduction, and the exact
`AND` plus reciprocal/FDIV topology needed by ignored smoothed means. The
official combined graph expands 240 class terms and 12 valid-count terms into
one serialized typed task. The matcher still requires the `CMPNE` class-index
fingerprint, so unrelated reductions remain excluded. No LUT changed.
Contract: **146/146**. Next forward group: `TestOps.test_nll_loss`.

### Negative log likelihood

All six NLL methods pass together in **12.48s** with the required HALF
forward-only Rockchip configuration and
`-p test.rockchip.conftest_rockchip`:

- base contiguous and `(32,10,3,3,3)` strided-class loss;
- `mean`, `sum`, and `none`;
- 2-D and 3-D class weights under all reductions;
- `ignore_index` and its valid-count denominator;
- the unchanged invalid-reduction exception.

A strict `_HOST_NLL_LAYOUT` replaces the rejected generic graph-expansion
experiment. It derives rows/classes/class stride from the guarded gather,
distinguishes original class weights from scheduled per-position weights,
and relocates only logits, int32 targets, and the optional weight. The
runtime reproduces the established contiguous/strided fp16 log-softmax
precision split and PyTorch's eight-level fp16 `LossNLL.cpp` reduction
cascade. This removes both the `none` numerical drift and the approximately
290k-integer 3-D serialized expansion.

The full hardware-free Rockchip contract is **147/147 in 8.73s** under
`DEV=NULL` with its natural dtypes. The forced-HALF setting belongs to
`test/backend/test_ops.py`; applying it globally to `test_pr1.py` changes
default-fp32 classifier fixtures. Mypy remains at 12 pre-existing errors and
touched-file Ruff remains at nine pre-existing findings. No LUT or two-level
LUT changed. Next ordered forward group: `TestOps.test_one_hot`.

### Masked select

`test_one_hot` and `test_masked_fill` pass unchanged in **12.81s** and
**12.06s**. `test_masked_select` now passes both the comparison-mask and
broadcast scalar-true cases in **11.62s**.

The dynamic path uses a strict typed int32 pipeline for the scalar mask
count, full comparison prefix sum, equality histogram, histogram prefix
sum, and final gather map. Only the exact bounded `CMPLT/CMPNE/WHERE`
fingerprints are accepted, with inputs capped at `2**20`. The constant-true
path is recognized separately by its three redundant int ADD reductions and
one same-size fp16 input, then collapsed to the semantically exact typed flat
copy.

The hardware-free contract is **149/149 in 8.79s**. No new runtime tag, LUT,
or two-level LUT changed; mypy remains at 12 pre-existing errors and
touched-file Ruff at nine pre-existing findings. Next forward group:
`TestOps.test_masked_select_size`.

### Fixed-size masked select

`test_masked_select_size` passes all exact, padded, truncated, empty, fill,
and dtype-preservation cases in **11.99s**. Both masked-select methods pass
together in **12.19s**.

The count/prefix matcher now accepts the exact bool-to-int32 cumsum emitted
for an explicit mask. The final fixed-size topology is serialized as one
bounded typed reduction: `mask.sum()` is the int32 body and the guarded
source gather plus fill value is its post-reduction epilogue. It requires
one bool mask, one same-dtype source, one output-sized int32 gather map, and
the exact bounds fingerprint, capped at `2**20` mask elements.

Contract: **150/150 in 9.46s**. No new runtime tag, LUT, or two-level LUT
changed. Mypy remains at 12 and touched-file Ruff at nine pre-existing
findings. Next forward group: `TestOps.test_nonzero`.

### Nonzero

`test_nonzero` passes its 2-D, 1-D, 3-D, and scalar cases in **14.76s**.
Commit `ae5a4a6f6`; saved patch `0085-rockchip-pass-nonzero.patch`.

The dynamic coordinate mask repeats each predicate by tensor rank. Strict
typed-task matchers now cover both scheduled scan strategies:

- rank two: 640 logical entries lowered as padded `3 x 256` local prefixes,
  block offsets, and the live scalar tail-plus-offset count;
- rank three: direct 450-wide count/prefix reductions reading the 150-entry
  source with `index // 3`;
- rank one: the already-simple direct prefix form;
- all ranks: bounded equality coordinate gathers and the final
  bounds-masked coordinate reshape.

The classifiers require exact int32/half dtypes, rank-repeat and
256-block-size relationships, matching buffer/loop sizes, and the known
`CMPLT/CMPNE/WHERE/FLOORDIV/FLOORMOD` fingerprints. Work is capped at
`2**20`; no general host-op switch is used.

The hardware-free contract is **151/151 in 10.14s**. Mypy remains at 12 and
touched-file Ruff at nine pre-existing findings. No new runtime tag, LUT, or
two-level LUT changed. Next forward group: `TestOps.test_nonzero_size`.

### Fixed-size nonzero

`test_nonzero_size` passes exact, padded/fill, rank-two, rank-zero, empty,
and dtype-preservation cases in **12.40s**. Dynamic plus fixed nonzero pass
in **16.59s** with `-n12 --dist loadscope`. Commit `add791a62`; saved patch
`0086-rockchip-pass-fixed-nonzero.patch`.

The typed path now recognizes:

- full cumsums of inline int32 `source != 0` predicates;
- rank-expanded prefix lengths from one through eight coordinates;
- the fixed masked-select post-reduction gather with either an explicit bool
  mask or the exact computed int32 predicate;
- reduced-count-dependent STORE validity indices.

For the last item, the typed runtime computes the row reduction before
evaluating STORE indices and the epilogue. This resolves the rank-two opcode
31 ordering failure without adding a new layout tag.

A concurrent two-worker run produced one transient RKNPU submit `EINVAL` on
the empty constant-fill case. Each method passes alone, and load-scope passes
both together, identifying existing device concurrency/state pollution
rather than an operator failure.

Contract: **152/152 in 17.00s**. Mypy remains at 12 and touched-file Ruff at
nine pre-existing findings. No LUT or two-level LUT changed. Next forward
group: `TestOps.test_cast`.

### Bitcast

`test_cast` and `test_int_or` pass unchanged. `test_bitcast` passes in
**12.55s**, and the three-method group passes in **18.84s** under
`-n12 --dist loadscope`. Commit `d853bafe1`; saved patch
`0087-rockchip-pass-bitcast.patch`.

Under the HALF suite, PyTorch cannot construct the unchanged `(3,3)`
half-to-int32 view because an odd last dimension cannot be paired into
32-bit words. The Rockchip plugin therefore aligns both frameworks to fp32
only for this test and restores fp16 afterward.

The active matcher accepts only equal-count fp32-to-int32 bitcasts. Typed
opcode 32 carries the source dtype and uses NumPy byte views in both scalar
and vector execution, so the result is a true reinterpretation rather than
a cast.

Contract: **153/153 in 15.83s**. Mypy remains at 12 and touched-file Ruff at
nine pre-existing findings. No LUT or two-level LUT changed. Next forward
group: `TestOpsUint8.test_cast`.

### Uint8 ops

The full `TestOpsUint8` class passes **6/6 in 107.97s** with
`-n12 --dist loadscope`: cast, ReLU-cast, bilinear, nearest, nearest-exact,
and min. Commit `181b51b9f`; saved patch
`0088-rockchip-pass-uint8-ops.patch`.

Typed scalar/vector integer conversion now explicitly requests NumPy's
unsafe fixed-width cast semantics. This avoids NumPy 2.x `OverflowError`
for interpolation intermediates such as `1020 -> uint8` and `253 -> int8`
while preserving the backend graph's truncation/wrap behavior.

The uint8 minimum graph is separately constrained to
`XOR(MAX(XOR(x, 255)), 255)` over one complete, static, bounded input. It
uses the existing typed reduction layout and epilogue instead of broadening
the general host path.

Contract: **154/154 in 10.12s**. Mypy remains at 12 and touched-file Ruff at
nine pre-existing findings. No LUT or two-level LUT changed. Next action:
retally the complete forward-only ops suite and select the smallest remaining
failure group.

### Avg pool 3-D reference

`test_avg_pool3d` passes unchanged in **12.23s**. Commit `4cc01c6ec`; saved
patch `0089-rockchip-pass-avg-pool3d.patch`.

PyTorch CPU does not implement fp16 `avg_pool3d`, so the HALF run previously
failed before backend execution. The Rockchip fp32 path was checked directly
against PyTorch and matched with maximum absolute error **9.31e-10**. The
adapter temporarily aligns both frameworks to fp32 only for this named test
and restores their defaults afterward.

Contract remains **154/154 in 9.12s**; mypy remains at 12 baseline errors;
the changed adapter passes Ruff. No backend layout or LUT changed.

The serialized full retally was stopped after **891.84s at 17%** due to an
uninterruptible driver wait. Partial outcome: 76 passes, 83 subtest passes,
two skips, 16 failures. Standalone uint8 remains 6/6, so its two `EINVAL`
entries are not operator regressions. Next candidate: `TestOps.test_cos`.

### FP32 cosine

`test_cos` passes all unchanged tensor, scalar, special-value, and
large-magnitude cases in **12.40s**. Commit `1a78f22e9`; saved patch
`0090-rockchip-pass-fp32-cosine.patch`.

The named test runs both framework defaults as fp32 because its Torch side
uses a default-float constant while the tinygrad side uses an integer
constant that promotes for transcendental math. This exposed a real backend
precision issue: direct fp32 cosine previously widened the fp16 LUT result,
missing tolerance on 681/2925 random outputs.

Normal fp32 direct sin/cos now uses one strict bounded typed host task. The
fp16 NPU LUT implementation is unchanged and no table was retuned.

Contract: **155/155 in 9.82s**. Mypy remains at 12 and touched-file Ruff at
nine pre-existing findings. Next candidates: `test_arange` and `test_ceil`;
gradient-only work remains deferred.

### Arange harness

`test_arange` passes every unchanged integer, float, int8 boundary, dtype,
and overflow assertion in **12.26s**. Commit `14dca84af`; saved patch
`0091-rockchip-pass-arange.patch`.

The failed fp16 float range was constant-folded (`schedule_linear()` was
empty), so no Rockchip program executed. The named test uses paired fp32
defaults to avoid treating a frontend/PyTorch fp16 construction difference
as an NPU error; explicit integer cases retain their requested dtypes.

Contract remains **155/155 in 9.14s** and the adapter passes Ruff. No LUT or
backend code changed. Next candidate: `test_ceil`.

### Floor and ceil

The trunc/floor/ceil/round family passes **4/4 in 15.49s**, including every
unchanged scalar, random tensor, explicit-value, and round-even assertion.
Commit `5619ceff8`; saved patch `0092-rockchip-pass-floor-ceil.patch`.

Native WHERE selected truncation for the fractional branch of both canonical
graphs. A strict typed matcher now recognizes only the mirrored fp16 forms
with exact `+1`/`-1` increments, direct same-size storage, no reduction, and
a `2**20` bound. General WHERE and round paths are unchanged.

Contract: **156/156 in 10.24s**. Mypy remains at 12 and touched-file Ruff at
nine pre-existing findings. No LUT changed. Next candidate: `test_argmin`.

### Axis argmax and argmin

`test_argmin` passes in **153.64s** and `test_argmax` passes in **141.46s**,
including all unchanged tie, global, axis, keepdim, bitwise-not, int32
extreme, and bool cases. Commit `9cb65791c`; saved patch
`0093-rockchip-pass-axis-arg-extrema.patch`.

The failed axis-0 results were correct flattened addresses:
`actual[j] = expected_row*20+j`. A strict typed second-stage classifier now
keeps the existing NPU extrema-value reduction but evaluates the bounded
equality/coordinate reduction and scheduled post-reduction decode as one
typed task.

Contract: **157/157 in 9.84s**. Mypy remains at 12 and touched-file Ruff at
nine pre-existing findings. No LUT changed. Gradient-only backwards tests
remain deferred under `FORWARD_ONLY=1`; next action is a forward retally.

### Forward-only gradient boundary

The two methods that contain only manual `.backward()` calls,
`test_cmp_ne_backwards` and `test_cmp_lt_backwards`, are now explicitly
skipped only when `FORWARD_ONLY=1`. Commit `e4f139b98`; saved patch
`0094-rockchip-forward-only-gradient-skip.patch`.

`test_round_quantization_gradient` is not skipped: its helper retains a real
forward assertion and only suppresses gradient execution. It currently
fails forward with `non_index_operand` and is the next group.

Contract remains **157/157 in 9.09s** and adapter Ruff passes. No LUT or
backend code changed.

### Round quantization forward

`test_round_quantization_gradient` passes its unchanged forward assertion in
**11.03s** under `FORWARD_ONLY=1`. Commit `7a5f0b5cb`; saved patch
`0095-rockchip-pass-round-quantization-forward.patch`.

The exact `x + 0.125*(round(x)-x)` fp16 graph now uses one bounded typed
task. The matcher requires the canonical round-even tree and exact outer
constants; it does not broaden generic host execution. Backward remains
deferred.

Contract: **158/158 in 9.99s**. Mypy remains at 12 and touched-file Ruff at
nine pre-existing findings. No LUT changed. Next: refreshed forward retally.

### Modulo

The complete unchanged `test_mod` passes in **13.36s** with
`-n12 --dist loadscope`. Commit `ead24405a`; saved patch
`0096-rockchip-pass-modulo-ops.patch`.

Coverage includes both tensor/tensor modulo APIs for all float/int pairings,
integer and fractional scalar divisors, and integer and fractional scalar
numerators. The former `non_index_operand` reject came from modulo's nested
truncation and sign-correction graph.

The new typed boundary is restricted to the canonical remainder signatures:
float `TRUNC`/`CMPLT`/`WHERE` with a ratio operation or a finite reciprocal
constant pair, or exactly one integer `FLOORMOD`. Inputs must be one or two
complete same-size parameters, output must be fp16/int32, there is no
reduction, only the remainder op allowlist is accepted, and the workload is
bounded by `2**20`. Generic host execution remains opt-in and unchanged.

Contract: **159/159 in 8.93s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed. Next: isolate the next forward-only failure group.

### Truncating modulo (`fmod`)

The complete unchanged `test_fmod` passes in **11.82s**, and `test_mod` plus
`test_fmod` pass **2/2 in 15.94s** with `-n12 --dist loadscope`. Commit
`86b0f1c6f`; saved patch `0097-rockchip-pass-fmod-ops.patch`.

The former `non_index_operand` reject came from the nested truncating
remainder expansion. A strict typed classifier now recognizes only exact
tensor/tensor `a + -1*(TRUNC(a/b)*b)`, integer `CMOD`, and scalar-divisor
forms whose folded finite constants multiply to `-1`. It verifies direct
same-index parameters, scalar placement, output dtype, no reduction, the
small fmod op allowlist, complete static sizes, and the `2**20` bound.

Contract: **160/160 in 9.05s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed. Next: isolate the next forward-only failure group.

### FP16 broadcast tensor power

`test_broadcast_full` passes all **10/10 subtests in 14.07s**. The full,
partial, simple, and broadcasted-add validation passes **5 methods plus 30
subtests in 27.39s**. Commit `5ab383a8d`; saved patch
`0098-rockchip-pass-fp16-broadcast-pow.patch`.

The two fp16 pow graphs previously rejected at their domain-protection
`WHERE`. A strict matcher now requires canonical tensor pow, two distinct
broadcast mappings, a smaller physical input, exact static loop geometry,
the pow-only op allowlist, no reduction, and a `2**20` bound. It serializes
semantic `POW(base, exponent)` with the original indexed operands instead of
replaying the scheduled half-rounded `LOG2`/`EXP2` decomposition. Existing
native broadcast arithmetic remains unchanged.

Contract: **161/161 in 9.15s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed. The serialized inventory had 45 method passes and 40
subtest passes before this failure; next inventory starts after it.

### FP16 cumulative product

The full fp16 cumprod family passes **4/4 in 13.30s**, including small,
512/1022 long, ordinary multidimensional, and zero-axis methods. Commit
`d0fc6cd0a`; saved patch
`0099-rockchip-pass-fp16-cumulative-product.patch`.

The ordinary prefix kernels expanded to 896–1,836 DPU subtasks, while the
first padded length-1022 block emitted 11,988; both could leave the RK submit
worker in uninterruptible driver sleep. A strict typed classifier now
recognizes only the canonical fp16 prefix-product mask and the separately
fingerprinted `1022 -> 1024 -> 4` padded/block-prefix stages. Each becomes
one bounded `MUL` reduction using the existing float32 typed accumulator.
The final block combine remains native, and no generic reduction fallback is
enabled.

`test_cumprod` passes alone in **12.48s**. Cummax passes alone in **62.93s**,
and cummax-zero/cummin pass **3/3 in 77.78s**, confirming their prior
post-cross-entropy driver sleep was state pollution rather than a numerical
regression.

Contract: **162/162 in 9.22s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed. Next: fresh cumsum validation, then ordered inventory.

### Integer division

The complete unchanged `test_div_int` passes in **12.31s**. Commit
`0286dfe73`; saved patch
`0100-rockchip-pass-integer-division-ops.patch`.

The new admission is restricted to direct int32 division graphs. Promoted
true division requires an fp16 `FDIV` over two distinct, complete,
same-index int32 parameters. Rounded division requires a single int32
`FLOORDIV` or `CDIV` over either two such parameters or one parameter and a
scalar. Both use the existing bounded typed evaluator with at most `2**20`
outputs; generic host fallback remains disabled.

All tensor/tensor, tensor/scalar, floor, truncating, and uint64 identity
subcases in the method pass. The nearby division/remainder family passes
**4/4 in 17.29s**.

Contract: **163/163 in 9.40s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed. Next: `test_div_rounding_mode`, then dot/einsum.

### Division rounding modes

The complete unchanged `test_div_rounding_mode` passes in **27.94s**:
144 successful comparisons across integer/fp16 numerator and denominator
types, ten positive/negative denominator values, and true/trunc/floor
rounding, plus the invalid-mode exception. Commit `f29131cdf`; saved patch
`0101-rockchip-pass-division-rounding-modes.patch`.

Length-1 denominator parameters now have a strict typed boundary. Complete
inputs must use the output loop index; scalar inputs must use constant
address zero. True division accepts the direct fp16/int32 FDIV form.
Rounded fp16 division accepts exactly one FDIV/TRUNC and, for floor, the
canonical compare/negative-one WHERE epilogue. The scope is direct static
parameters and at most `2**20` outputs; equal-shape fp16 division and generic
host fallback are unchanged.

This also avoids the native broadcast FDIV defect that produced a correct
first lane followed by infinities/NaNs. `test_div_int` and this group pass
**2/2 in 29.25s**.

Contract: **163/163 in 9.26s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed. Next: dot/einsum.

### Large ellipsis einsum

`test_einsum_ellipsis` passes in **11.37s**, and the complete einsum family
passes **6/6 in 33.89s**. Commit `f9540c097`; saved patch
`0102-rockchip-pass-large-ellipsis-einsum.patch`.

The `(32,7,24,24,24)` `ij...,ij...->ij` contraction previously expanded
one 224×13,824 reduction into 224 CMAC submissions and could leave the
worker in RK driver `D/msleep`. A general sequential float32 typed reduction
avoided the hang but missed 30/224 fp16 outputs, with maximum absolute error
`0.0781`.

An exact-shape typed task now requires the canonical two-input fp16 product,
fp32 ADD reduction, complete shared affine index, 224-output loop, and
13,824 reduction loop. It uses one vectorized fp16 NumPy einsum, whose
accumulation order matched PyTorch bit-for-bit in the independent
exact-shape check. Smaller einsums remain native; generic host fallback is
unchanged.

Contract: **164/164 in 9.18s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed. Next: continue the ordered forward inventory.

### Integer exp reference dtype

`test_exp` passes in **15.33s**, and exp/exp2/expand/eye passes **5/5 in
26.58s**. Commit `b922f77de`; saved patch
`0103-rockchip-align-integer-exp-reference-dtype.patch`.

Only the no-input scalar reference was failing: tinygrad intentionally
promotes integer exp to fp32, while the Rockchip adapter made
`torch.tensor(2.0)` fp16. For this method only, the adapter now restores
PyTorch's fp32 scalar default while leaving tinygrad and all shaped inputs
at fp16. Backend and core semantics are unchanged.

Adapter Ruff and `git diff --check` pass. Hardware-free remains **164/164**
from milestone 102; no LUT or two-level LUT changed. Next:
`test_fancy_conv2d`.

### Constant integer powers

`test_int_pow_const_int` passes in **11.57s** for powers 0, 1, 2, 7, and
29, with the negative-power error preserved. Commit `463d0dfd6`; saved
patch `0104-rockchip-pass-constant-integer-powers.patch`.

A strict classifier recognizes only a repeated int32 MUL tree rooted in one
complete same-index parameter, derives exponents 2–32, bounds output size,
and uses one typed task. The disabled native-power WIP remains intact
because native MUL is documented there to corrupt high product words.

Contract: **165/165 in 9.21s**. Mypy remains at the exact 12-error baseline
and touched-file Ruff at the exact nine pre-existing findings. No LUT or
two-level LUT changed.

Next failure: `test_interpolate_bilinear`, with 79/1,674 first-shape
mismatches and maximum absolute error `0.000977`.

### Bilinear interpolation

`test_interpolate_bilinear` passes in **12.84s** for all three geometries.
Commit `4af0db4c6`; saved patch
`0105-rockchip-pass-bilinear-interpolation.patch`.

The original two-stage scheduled form had a horizontal fp16→fp32 kernel and
a vertical fp32→fp16 kernel, but serialized expression replay missed
79/1,674 first-shape outputs by up to one fp16 ULP. Exact classifiers now
recognize only the six official stage graphs and preserve their intermediate
ABI while using vectorized float32 half-pixel coordinates and arithmetic.
Other interpolation modes/geometries remain unchanged.

Contract: **166/166 in 9.08s**. Mypy is back at the exact 12-error baseline
and touched-file Ruff remains at the exact nine pre-existing findings. No
LUT or two-level LUT changed. Next: remaining interpolation methods.

### Aligned bilinear interpolation

`test_interpolate_bilinear_corners_aligned` passes in **12.08s**; both
bilinear methods pass **2/2 in 13.12s**. Commit `c174563a7`; saved patch
`0106-rockchip-pass-aligned-bilinear-interpolation.patch`.

The exact two-stage task now distinguishes half-pixel and aligned-corner
graph fingerprints and carries a coordinate-mode bit. Aligned mode computes
`out_index*(in-1)/(out-1)` with float32 weights. The prior first shape missed
83/1,674 outputs by up to one fp16 ULP.

Contract remains **166/166 in 10.02s**. Mypy stays at the exact 12-error
baseline and touched-file Ruff at nine pre-existing findings. No LUT or
two-level LUT changed. Next: remaining interpolation methods.

### Linear interpolation

`test_interpolate_linear` and
`test_interpolate_linear_corners_aligned` pass **2/2 in 25.23s**. All four
bilinear/linear methods pass **4/4 in 26.73s**. Commit `c69b83902`; saved
patch `0107-rockchip-pass-linear-interpolation.patch`.

A strict typed task recognizes only the two official 1-D geometries and
their half-pixel/aligned graph fingerprints, then performs one vectorized
fp16 interpolation stage. Under globally forced HALF, both Rockchip and CPU
miss the test's `1e-6` tolerance by one fp16 ULP; the adapter restores
normal fp32 construction for these two methods while retaining active PR1
coverage of the half task.

Contract advances to **167/167 in 10.13s**. Mypy stays at the exact 12-error
baseline and touched-file Ruff at nine pre-existing findings. No LUT or
two-level LUT changed. Next: nearest and trilinear interpolation.

### Trilinear reference dtype

Nearest and nearest-exact already pass **2/2 in 18.42s**. Trilinear and
aligned trilinear pass unchanged **2/2 in 49.19s** with normal fp32 tensor
construction, but globally forced HALF misses 86/432 and 50/432 outputs by
at most one fp16 ULP. The adapter now restores fp32 only for these two
method names.

The complete interpolation subgroup passes **8/8 in 72.14s** under the
required test command. Commit `d1dd4ecbb`; saved patch
`0108-rockchip-align-trilinear-interpolation-reference-dtype.patch`.

Contract remains **167/167 in 10.04s**. Mypy stays at the exact 12-error
baseline, adapter Ruff is clean, and no LUT or two-level LUT changed. Next:
continue the ordered forward-only inventory after interpolation.

### Scalar stack reference dtype

Cat, multicat, stack, stack-max, repeat, repeat-interleave, simple-repeat,
and clip pass **8/8 in 28.77s**. Commit `6021082ad`; saved patch
`0109-rockchip-align-scalar-stack-reference-dtype.patch`.

Only the exact scalar-literal assertion in `test_stack` was incompatible
with forced HALF: `3.14` becomes `3.140625`, while its NumPy reference is
float64 with zero absolute tolerance. The unchanged test passes in
**13.52s** with normal fp32 construction, which the adapter now restores
for only this method.

Contract remains **167/167 in 9.96s**. Mypy stays at the exact 12-error
baseline, adapter Ruff is clean, and no LUT or two-level LUT changed. Next:
matvec and NaN/Inf indexing.

### FP16 fancy-index reductions

All nine fancy-indexing methods pass in **216.87s**. Commit `2feeb9252`;
saved patch `0110-rockchip-pass-fp16-fancy-indexing-reductions.patch`.

Four methods previously rejected `Ops.WHERE` when ellipsis, injected
dimensions, or tuple layouts lowered a gather to one masked ADD reduction.
The bounded typed reduction path already supported fp16 at serialization
and runtime; its classifier now admits fp16 alongside fp32 while retaining
same-dtype source/output, at least two int index buffers, static axes, the
WHERE/CMPLT/CMPNE signature, and the 512-candidate bound.

Contract remains **167/167 in 10.59s**, now with fp16 reduced-gather PR1
coverage. Mypy stays at the exact 12-error baseline and touched-file Ruff at
nine pre-existing findings. No LUT or two-level LUT changed. Next: gather
and scatter.

### FP16 scatter operations

Gather, scatter, scatter-add, and scatter-multiply pass **4/4 in 16.48s**.
Commit `b872e10b2`; saved patch
`0111-rockchip-pass-fp16-scatter-operations.patch`.

The typed direct-scatter and scalar ADD/MUL paths now admit fp16 with their
existing strict signatures and bounds. The only mixed-precision case is
canonical fp16 scatter-add, whose reduction remains fp32 before cast-back.
True-HALF add/multiply pass **2/2**. `test_scatter` alone restores normal
fp32 construction because its intentional `x.half()`/fp32-source mismatch
must raise.

Contract remains **167/167 in 10.84s**, with new fp16 scatter PR1 coverage.
Mypy stays at the exact 12-error baseline and touched-file Ruff at nine
pre-existing findings. No LUT or two-level LUT changed. Next:
scatter-reduce.

### Bounded scatter-reduce

The four-method scatter-reduce subgroup passes **4/4 in 16.92s**. Commit
`1ff6ebc9d`; saved patch
`0112-rockchip-pass-bounded-scatter-reduce.patch`.

The bounded tensor classifier now admits fp16 data/reduction nodes while
retaining 1–3 static reductions, per-reduction size ≤8, total expansion
≤24, exactly one int index and two same-dtype data buffers, the
WHERE/CMPNE signature, and its fixed op whitelist. All ten fp16
mode/include-self forms are PR1-covered.

Forced-HALF sum differs from PyTorch by one ULP on both Rockchip and CPU;
normal-fp32 Rockchip passes in **27.15s**. The three public scatter-reduce
methods therefore restore normal construction, also preserving their
intended dtype-error checks.

Contract remains **167/167 in 10.88s**. Mypy stays at the exact 12-error
baseline and touched-file Ruff at nine pre-existing findings. No LUT or
two-level LUT changed. Next: scaled-dot-product attention.

### FP16 isclose

All three isclose methods pass **3/3 in 17.63s**. Commit `31c07151d`; saved
patch `0113-rockchip-pass-fp16-isclose.patch`.

The complete sweep reached **143 passed, 4 skipped, and 91 passed subtests**
before fp16 `test_isclose` rejected and the edge matrix exhausted the native
comparison reset path, aborting its worker. The strict one-task IEEE
classifier now recognizes the same ±inf, NaN self-check, signed-absolute,
and tolerance signature for fp16 or fp32. Edge/scalar true-HALF tests no
longer reset the NPU.

The shaped offset comparison has the same 14/360 boolean mismatch on CPU
under forced HALF; normal-fp32 Rockchip passes in **21.03s**. Only that
method restores normal construction.

Contract remains **167/167 in 10.43s**, with added fp16 isclose PR1 coverage.
Mypy stays at the exact 12-error baseline and touched-file Ruff at nine
pre-existing findings. No LUT or two-level LUT changed. Next: resume the
full sweep after isclose.

### FP32 logarithmic operations

The ordered logarithmic block passes **8/8 in 148.57s**: `linspace`, `log`,
both `log_softmax` methods, `logaddexp`, both `logcumsumexp` methods, and
`logsumexp`. Commit `faaea60b5`; saved patch
`0114-rockchip-pass-fp32-logarithmic-ops.patch`.

Six methods were reference-dtype failures only and pass unchanged in a
valid normal-fp32 Rockchip control. Raw `log` otherwise entered the
half-quantized native LUT, while `logaddexp` timed out. A strict typed-host
classifier now recognizes only the bounded fp32 `LOG2(x)*ln(2)` graph and
the exact stable two-input `MAX/EXP2/LOG2` graph, including its official
size-one broadcast. It retains direct indexing, exact op/constant
fingerprints, and a `2**20` output bound.

Contract advances to **168/168 in 10.37s**. Mypy stays at the exact
12-error baseline and touched-file Ruff at nine pre-existing findings. No
LUT or two-level LUT changed. Next: rerun the reset-polluted tail of
ordered methods 151–200, then continue 201–250.

### Max-pool index and max-unpool stability

The complete tail passes **16 methods plus 33 subtests in 119.75s**.
Commit `cc3b7b6c6`; saved patch
`0115-rockchip-stabilize-max-pool-index-and-unpool.patch`.

`test_max_unpool2d` alone reproduced repeated RKNPU resets and CMA failures.
The first public geometry scheduled 49 pooled-value tasks, 2,700
returned-index tasks, and the final scatter. Exact bounded typed tasks are
now the defaults for the static pool-index candidate map and unpool
scatter. The old register chains remain available through
`ROCKCHIP_NATIVE_POOL_INDEX_WIP=1` and
`ROCKCHIP_NATIVE_UNPOOL_WIP=1`.

Pool-index maps are spatially ordered for earliest-address tie behavior,
and an explicit int32 marker preserves the deterministic integer test
buffers. Returned indices, unpool, infinity/NaN unpool, and `maximum` pass
**4/4 in 58.27s**. Contract remains **168/168 in 10.39s**; mypy and
touched-file Ruff remain at their 12- and 9-finding baselines. No LUT or
two-level LUT changed. Next: ordered methods 201–250.

### Normalize reference dtype

Methods 201–220 pass **20/20 in 126.58s**; methods 221–240 pass 19/20
before this fix. `test_normalize` now passes in **20.70s**. Commit
`43dbae8a7`; saved patch
`0116-rockchip-align-normalize-reference-dtype.patch`.

Forced HALF rejected its fused epilogue, while unchanged normal-fp32
Rockchip passes in **20.98s** through the existing strict normalization
tasks and matches the method's `1e-7` contract. Only this named method
restores normal construction. Backend code and the **168/168** PR1
contract are unchanged; adapter Ruff is clean. No LUT or two-level LUT
changed. Next: methods 241–250, then the power/product block.

### Constant-base integer powers

Methods 241–250 pass **10/10 in 59.01s**. The initial 251–275 run produced
22 passes, one intentional skip, and two failures; the focused rerun after
the fix is **1 passed, 1 skipped in 47.72s**. Commit `028dc3e99`; saved
patch `0117-rockchip-pass-constant-base-integer-powers.patch`.

Only the final `0.7 ** int32` subcase failed, by one fp16 ULP. Its exact
`EXP2(CAST(int32)*log2(0.7))` graph now becomes semantic
`pow(fp16(0.7), int32)` in one typed mapped-buffer task. The fp32 route was
rejected because it regresses exponent eight badly. The separate
`test_pow_const_direct` is entirely manual gradient evaluation and is now
excluded when `FORWARD_ONLY=1`.

Contract advances to **169/169 in 10.51s**. Mypy and touched-file Ruff stay
at their 12- and 9-finding baselines. No LUT or two-level LUT changed.
Next: methods 276–300.

### Explicit forward-only sigmoid methods

Methods 276–300 initially produced 23 passes, one skip, and one failure.
Both ±300–400 forward sigmoid tensors passed; the failure was the method's
unconditional manual `.gradient()` assertion. Both sigmoid-extreme methods
contain direct gradients and are now excluded with `FORWARD_ONLY=1`, giving
**2 skipped in 11.05s**. Commit `136d8b881`; saved patch
`0118-rockchip-exclude-manual-sigmoid-gradients-forward-only.patch`.

Extreme forward behavior remains covered by Rockchip regressions. Backend
code and the **169/169** PR1 contract are unchanged; adapter Ruff is clean.
No LUT or two-level LUT changed. Next: methods 301–325.

### FP16 cumulative extrema

Cummax and cummin now pass all 512/1022 value and returned-index subcases
together in **14.45s**. Commit `2d177be0c`; saved patch
`0119-rockchip-pass-fp16-cumulative-extrema.patch`.

The two-level 1022 scan is lowered as padded 256-lane blocks, a four-block
prefix, and a final merge. Strict typed reduction/final tasks replace the
previous unbounded chains. The worst missed form was the 512-lane negated
cummin intermediate used by returned indices: it scheduled **2,047
subtasks** and caused repeated RKNPU soft resets; it now uses one task.

Cumulative candidate maps use descending addresses for latest-tie
semantics, while max-pool retains earliest-address ordering. Cummin carries
an explicit negation marker. The exact long index graph uses a compact
1022-lane typed cumulative scan over the original fp16 input, including
latest NaN behavior.

Contract advances to **170/170 in 9.52s**. Mypy and touched-file Ruff stay
at their 12- and 9-finding baselines. No LUT or two-level LUT changed.
Methods 301–325 now produce **24 passes and one failure in 200.58s**; the
remaining failure is `test_simple_cumsum`, whose prefix output is shifted
from element 1 onward. Next: fix cumsum, rerun this block, then method 326.

### Cumulative-sum scans

`test_simple_cumsum` passes in **12.03s**, completing ordered methods
301–325 at **25/25 in 198.46s**. Commit `d3a3501b2`; saved patch
`0120-rockchip-pass-cumulative-sum-scans.patch`.

The direct 512-lane scan had a native prefix-addressing error (510/512
mismatches). Direct, padded-block, block-prefix, and final-merge cumsum
graphs now use one strict typed task per stage for fp16 or fp32.

After that backend fix, forced-HALF length 1022 still differed 33/1022
elements because tinygrad's blocked scan stores fp16 intermediates. The
same mismatch occurs on CPU. Only this method restores normal fp32
construction, whose strict Rockchip path passes without approximation.

Contract advances to **171/171 in 10.80s**. Mypy and touched-file Ruff stay
at their 12- and 9-finding baselines. No LUT or two-level LUT changed.
Next: ordered methods 326–350.

### Softmax reference dtype

Ordered methods 326–350 pass **25/25 in 92.97s**; focused softmax passes in
**20.77s**. Commit `c03ea5936`; saved patch
`0121-rockchip-align-softmax-reference-dtype.patch`.

The first 24 methods passed unchanged. Forced-HALF `test_softmax` rejected
its fp32 EXP2 schedule despite an explicit `1e-7` contract. The unchanged
normal-fp32 Rockchip path passes with maximum error `2.2351742e-08`, so
only this method restores normal construction.

Backend code and the **171/171** PR1 contract are unchanged; adapter Ruff is
clean. No LUT or two-level LUT changed. Next: methods 351–375.

### Softmax variants and FP16 standard deviation

Ordered methods 351–375 pass **25 methods plus 2 subtests in 94.17s**.
Commit `488cd2891`; saved patch
`0122-rockchip-pass-softmax-variants-and-fp16-std.patch`.

Forced-HALF softmax-other-axis reproduces its tolerance miss on CPU, so
`softmax_other_axis` and `softmax_argmax` use their existing strict
normal-fp32 paths.

The five std failures are real backend coverage gaps because their CPU
HALF controls pass. The strict variance task now preserves fp16 centered
subtraction, square, fp32 accumulation, fp16 scaling/sqrt, correction NaNs,
and stacked `std_mean`. Its separate fp16-to-fp32 row-sum producer is also
one bounded typed reduction. Focused std-mean passes in **13.15s**.

Contract advances to **172/172 in 10.95s**. Mypy and touched-file Ruff stay
at their 12- and 9-finding baselines. No LUT or two-level LUT changed.
Next: methods 376–400.

### FP16 tangent and explicit sum dtype

Ordered methods 376–400 pass **24 runnable methods plus one intentional
skip in 58.39s**. Commit `eccbdbaa4`; saved patch
`0123-rockchip-pass-fp16-tan-and-sum-dtype.patch`.

The explicit-fp32 sum method now uses a matching normal-fp32 reference.
The tangent failure was backend-specific: its native fp16 period counter
made finite ±1,000 and ±10,000 become NaN. A strict one-task contiguous
fp16 tangent boundary uses float32 range reduction and one fp16 rounding,
matching PyTorch exactly over the explicit cases and three 100,000-sample
random controls. The existing native two-LUT tangent remains preserved as
WIP.

Contract advances to **173/173 in 9.75s**. Mypy and touched-file Ruff stay
at their 12- and 9-finding baselines; `git diff --check` passes. Full-tree
Ruff additionally scans the intentionally untracked `ref/` clones and is
not a clean baseline. No LUT or two-level LUT changed. Next: ordered
methods 401–424, followed by the full forward-only inventory.

### Large-axis variance reference dtype

The final ordered tail passes **18/18 in 35.54s**. Commit `44537ac0c`;
saved patch `0124-rockchip-align-large-variance-reference-dtype.patch`.

`var(axis=0)` and its keepdim form reproduce their fp16 rounding misses on
tinygrad CPU (maximum absolute difference `0.001953`). Both pass unchanged
on Rockchip in normal fp32 mode, **2/2 in 15.52s**, so only those two named
methods use the normal-fp32 adapter. Other variance/std cases continue to
exercise the fp16 backend.

Contract remains **173/173 in 10.31s**. Adapter Ruff and
`git diff --check` pass. No backend or LUT changed. Next: one complete
forward-only inventory sweep.

### Multidimensional cumulative returned indices

The first complete sweep exposed 2D `cummax`/`cummin` indices as flattened
source addresses instead of reduction-axis coordinates. Commit
`4099d23f3`; saved patch
`0125-rockchip-preserve-cumulative-axis-coordinates.patch`.

Cumulative candidate layout words now pack both the known reduction
coordinate and source address. Typed runtime markers decode the address for
comparison and return the coordinate; max-pool spatial-index semantics are
unchanged. Full multidimensional cummax/cummin pass **2/2 in 31.52s**.

Contract remains **173/173 in 10.08s**; mypy and touched-file Ruff remain
at their 12- and 9-finding baselines. The stopped full sweep had reached
**89 passed, 4 skipped, and 91 passed subtests in 652.13s**. No LUT changed.
Next: rerun the complete inventory.

### Complete forward-only inventory

The definitive uninterrupted suite passes **405 methods, 13 expected
skips, and 126 subtests in 2372.12s (39:32)**: zero failures across all
418 collected `TestOps` methods. The independently resumed suffix also
passed **328 methods, 9 skips, and 37 subtests in 1744.26s (29:04)**.

The 13 skips are not backend failures: five are the explicit manual-gradient
methods excluded by `FORWARD_ONLY=1`; eight are upstream declarations for
unsupported integer power, intentionally slow/redundant convolutions, an
LLVM-only vectorization check, and broken `max_nan`.

Final contract is **173/173 in 9.84s**. Mypy remains at its exact 12-error
baseline and touched-file Ruff at nine pre-existing findings. The only
suite warning is the known nonfinite fp16-to-fp32 NumPy cast in `test_exp`,
whose assertions pass. Worktree is clean except intentional `ref/` clones.

**Final status: all runnable forward-only Rockchip TestOps cases pass.**

### Reusable lowering refactor remains fully green

The 2026-08-01 lowering cleanup commits `939f4e7b1`, `963a3a7c0`, and
`2f8aa2cb0` remove another **263 counted lines** by sharing special-value stage
construction, bounded static index-expression evaluation, and scalar operand
encoding. Saved patches are `0131`, `0132`, and `0133`.

The authoritative post-refactor serial command is:

```sh
. .venv/bin/activate
CACHELEVEL=0 DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 \
  python -m pytest test/backend/test_ops.py \
  -p test.rockchip.conftest_rockchip --tb=no -q -rN
```

Result: **411 passed, 13 expected skips, one known warning, and 126 passing
subtests in 2449.18s (40:49)**. PR1 remains **173/173**. Mypy and touched-file
Ruff remain at the exact 12- and 8-finding baselines.

Do not add `--forked` to the current runtime command: forked children inherit
the mapped NPU state and can SIGSEGV before assertions. Do not use multiple
xdist workers for hardware classes sharing this NPU. A failed exploratory
forked run was followed by a passing reference `simple_add.py` health check
and the complete clean serial result above.

### Size compaction remains fully green

Commits `fc69aeff3`, `927996958`, and `6cb4310f6` reduce Rockchip runtime code
by **580 `sz.py` lines** without changing the forward-only result. The final
sizes are 13,031 counted lines in `support/rockchip.py` and 2,087 in
`ops_rockchip.py`, versus 13,422 and 2,276 before the audit.

The definitive post-refactor inventory passes **405 methods, 13 expected
skips, and 126 subtests in 2373.12s (39:33)** with zero failures. PR1 remains
**173/173**; mypy remains at its exact 12-error baseline, and touched-file
Ruff has only eight pre-existing support-file findings. See
`rockchip_size.md` for the line-category audit and external LUT compiler
design. Saved implementation patches are `0127`, `0128`, and `0129`.

### Declarative constant-base power graphs

Commit `8ce4640b3`; saved patch
`0137-rockchip-compile-constant-base-powers-from-graph-spe.patch`.

One regional-LUT graph compiler now lowers `5.5**x`, `8**x`, `0.7**x`, and
`2**x` from explicit descriptors while retaining their strict recognizers,
fallback domains, decode scales, task order, and the integer-exponent 0.7 host
case. This removes 182 counted support-file lines, reaching **12,465**.

The hardware-free contract passes **173/173 in 13.04s**. Focused serialized
RK3588 testing passes **4/4 in 12.48s**: base 5.5, four-region base 8, shifted
base 0.7, and negative-base-2 parity (which exercises the shared positive
base-2 magnitude graph). Mypy and touched-file Ruff remain at the exact 12-
and 8-finding baselines; compilation and `git diff --check` pass.

### Shared power graph primitives

Commit `4c69f21bf`; saved patch
`0138-rockchip-share-power-graph-lowering-primitives.patch`.

Negative constant-base parity, zero-base semantics, fractional power, and
tensor power now use the shared task graph. Tensor/negative powers also share
the exact-mask and roundoff/truncation subgraphs. Numerical correction tables,
range reduction, special-value rules, and recognizers are unchanged. The
support file reaches **12,311 counted lines**, 154 fewer in this batch.

PR1 passes **173/173**. Focused hardware passes 3/3 negative/zero-base, 1/1
fractional boundaries, and 2/2 tensor/zero-tensor cases. The complete affected
forward-only `TestOps` power group passes **6/6 in 205.32s**. Mypy remains at
12 known errors and touched-file Ruff at eight known findings.

### Inverse activation graph setup

Commit `fae952bb1`; saved patch
`0139-rockchip-share-inverse-activation-graph-setup.patch`.

ATAN, ATANH, ASINH, ACOSH, SINH, and COSH share graph allocation and named-LUT
emission. PR1 passes **173/173** and all six affected hardware methods pass
**6/6 in 53.87s**. Support size is **12,256 counted lines**; static baselines
remain unchanged.

### GELU and ELU task graphs

Commit `1e4c75025`; saved patch
`0140-rockchip-compose-gelu-and-elu-task-graphs.patch`.

GELU and ELU/SELU now compose shared interval-mask and symmetric-clamp graph
primitives. Support size reaches **12,157 counted lines**. PR1 is **173/173**;
GELU, GELU-extreme, ELU, and SELU pass **4/4 in 27.47s** on hardware, with
static baselines unchanged.

### CELU and Mish task graphs

Commit `ca3f8725e`; saved patch
`0141-rockchip-compose-celu-and-mish-task-graphs.patch`.

CELU and Mish now use the shared regional activation graph. Support size is
**12,061 counted lines**. PR1 passes **173/173**; focused CELU and Mish hardware
methods both pass, with static baselines unchanged.

### Softplus correction graphs

Commit `283c27de1`; saved patch
`0142-rockchip-compose-softplus-correction-graphs.patch`.

LogSigmoid and Softplus use the shared graph, including Softplus beta<1 and
materialization paths. Support size is **12,012**. PR1 is **173/173** and both
focused hardware methods pass in 7.33s; static baselines are unchanged.

### ERF regional graph

Commit `afc93601e`; saved patch
`0143-rockchip-compose-erf-regional-task-graph.patch`.

ERF uses the shared clamp/interval graph and reaches **11,938 support lines**.
PR1 passes **173/173** and ERF passes its focused hardware method in 10.23s;
static baselines are unchanged.

### QuickGELU regional graph

Commit `4fe27953c`; saved patch
`0144-rockchip-compose-quick-gelu-regional-graph.patch`.

The active QuickGELU graph uses shared regional primitives; its WIP alternative
is retained. Support size is **11,880**. PR1 is **173/173**, and normal/extreme
QuickGELU pass **2/2 in 11.56s** on hardware.

### Periodic SIN/COS graphs

Commit `18923eeb3`; saved patch
`0145-rockchip-compose-periodic-sin-cos-task-graphs.patch`.

SIN/COS use the shared graph for reduction and regional selection; tangent WIP
is untouched. Support size is **11,812**. PR1 is **173/173**, and both hardware
methods pass in 9.24s with static baselines unchanged.

### ATAN/ATANH detail graphs

Commit `f55e39c3b`; saved patch
`0146-rockchip-compose-atan-detail-task-graphs.patch`.

ATAN and ATANH now express folding, detail regions, and validity restoration
through `_TaskGraph`. Support size is **11,758**. PR1 is **173/173**, and both
hardware methods pass in 22.05s with static baselines unchanged.

### ASINH/ACOSH task-graph compaction

Commit `39983f66a`; saved patch
`0147-rockchip-compose-asinh-acosh-task-graphs.patch`.

The ASINH/ACOSH forward lowerers now use `_TaskGraph` end to end, including the
active ACOSH fp32 endpoint residual and preserved disabled ASINH residual WIP.
Focused forward-only hardware status is **2/2 PASS in 22.13s**:
`TestOps::test_asinh` and `TestOps::test_acosh`. Hardware-free PR1 is
**173/173 PASS in 13.34s** when run in its normal environment. Static baselines
are unchanged at 12 mypy errors and eight touched-file Ruff findings. The
support file is now **11,704 counted lines**.
