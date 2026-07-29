# test_ops.py Status — Rockchip NPU Backend

**Last complete census:** 2026-07-29 (commit `a864e9519`)
**Command:** `PYTHONPATH=. DEV=ROCKCHIP DEFAULT_FLOAT=HALF FORWARD_ONLY=1 .venv/bin/python -m pytest test/backend/test_ops.py -q --tb=line`

The fresh serial census completed in **918.42 seconds: 182 passed, 333 failed,
8 skipped, and 27 subtests passed**. Pytest still collects 424 test functions;
its failed count also includes failing unittest subtests. This supersedes the
older 140-passing census below. The dedicated Rockchip hardware regression is
separately **70/70 passing in 207.22 seconds**.

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
