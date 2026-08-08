# Rockchip EW / OUT=5 progress

Branch: `rockchip-2608-ew`  
Focus: half ADD/MUL (and gemm via gather + EW) on DPU with **OUT_PRECISION=5** for tol match. No CPU `host_sum`.

## Test status

| Test | Status |
|------|--------|
| Pointwise ADD/MUL | **pass** — mtx512 ≤8/chunk, PC-chain 64, no soft-reset |
| `test_small_gemm` (8×8) | **pass** — ≪ old ~4.5s reset-bound path |
| `test_gemm_fp16` (64×64) | try after ship; was impractical under reset storm |
| eye / padded / range gemms | `RKPLAN_REJECT` (plan) |

## Shipped path (mtx512 OUT=5)

Source: [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) (`ref/rk3588-npu`) — NC1HWC2, fp32-out **C2=4**.

- **OUT_PRECISION=5**, contiguous half in / fp32 out per chunk.
- Chunk: `CHANNEL=n-1`, `W=H=0`, `DST_SURF_STRIDE=1<<4`, `SURFACE_ADD=4<<4`, `size_e=1` (`size_e=3` hangs on EW).
- **`n≤8` per task**; `n>8` → 64B-aligned slots, PC-chain up to **64** chunks/ioctl.
- **No soft-reset** on the hot path (`reset_npu()` remains for recovery only).

### Why not elementwise WIDTH alone

Naive `CHANNEL=7` + WIDTH only got first **4** exact on `n=8`: one fp32 atom = C2=4 floats.

### Why not the old 4/64B + reset path

`ref/rk3588/examples/elementwise.py` 4-elem/64B packing + `ACT_RESET` every ≤16-tile chain made warm 8×8 ~95% soft-reset (~106ms × 15).

## Refs

- `ref/rk3588-npu` — C2 / `feature_data` / `surf_add`
- `ref/rk3588/examples/elementwise.py` — old EW PC-chain + fp32 tile pack
- `tinygrad/renderer/rockchip.py`, `tinygrad/runtime/ops_rockchip.py`

---

## 2026-08-07 — after mtx512 ship (no soft-reset)

### Warm 8×8 gemm profile (`test_small_gemm` call ~0.22s incl. torch/ref; program `__call__` alone)

| Component | Time | Share of `__call__` |
|-----------|------|---------------------|
| emit+patch | ~3.9 ms | ~47% |
| ioctl submit | ~2.6 ms | ~31% |
| cvt fp32→f16 (host) | ~0.9 ms | ~10% |
| gather | ~0.8 ms | ~9% |
| half_out | ~0 ms | — |
| soft-reset | **0** | — |
| **`__call__` total** | **~8.4 ms** | |

- **Ioctls: 15** (`dev.submit_count` delta). One PC-chain per logical EW op.
- Shape: 15 EW ops × 64 elems → 8 chunks/op → **120 chunks**, flushed per-op (not across ops).
- Old path: ~15× soft-reset ≈ 95% of ~1.6–4.5s realize. Soft-reset removed; wall is now emit + NPU submit + host cvt.

### Why not one ioctl?

Each OUT=5 stage writes **fp32**; next stage RDMA expects **half**. Runtime does host `_fp32_slots_to_half` after every op with `cvt_scratch` set. A PC-chain is NPU-only (no CPU between tasks), so dependent MUL/ADD tree stages cannot share one submit under the current OUT=5 + half-in contract. Within one op, the 8 chunks already share one ioctl.

### Host cast

Yes — NPU writes fp32 into the 64B slot; CPU `f32.astype(np.float16)` in place (`_fp32_slots_to_half`) before the next EW or final `half_out`. That is the ~0.9 ms cvt bucket (×15 for 8×8 gemm).

### Option: switch back to OUT_PRECISION=2 (fp16 out)

Would unlock:
- No host cvt between stages → dependent ops can PC-chain → **far fewer / one ioctl** for gemm tree
- Contiguous half tiles (elementwise.py: up to ~64k elems/task), not mtx512 ≤8 for fp32 C2=4
- Recipe notes from `elementwise.py`: `out_precision=2`, `OUT_CVT_SCALE=(1<<16)|1`, CHANNEL=7 + WIDTH packing (not the OUT=5 surf atom path)

Caveat: for the gemm ADD tree, OUT=5 already casts to half after every stage, so accumulate between levels is already half. OUT=2 mainly removes barrier/cvt cost; deep gemm vs torch wider-accum `5e-3` may still miss the same way a pure half chain did. Keep OUT=5 if single-op “f32 ALU then cast once” exactness still matters for pointwise refs.

**Not switched yet** — probe `test_small_gemm` / pointwise under OUT=2 before changing the shipped path.

---

## 2026-08-07 — flexible OUT precision (`ROCKCHIP_EW_OUT`)

Config: `ROCKCHIP_EW_OUT=fp32|fp16` (default **fp32**). Stored on `RKImage.out_precision` (RKIM v11, flags bits 8–15).

| Mode | OUT | Layout | Host cvt | Cross-op chain |
|------|-----|--------|----------|----------------|
| `fp32` (default) | 5 | mtx512 ≤8 / 64B | yes | no |
| `fp16` | 2 | contiguous half (elementwise WIDTH) | no | yes |

### Warm 8×8 gemm with `ROCKCHIP_EW_OUT=fp16`

| Metric | fp32 (prior) | fp16 |
|--------|--------------|------|
| ioctls | 15 | **1** |
| warm realize (`__call__` path) | ~8–11 ms | **~5.4 ms** |
| pytest `test_small_gemm` call | ~0.10–0.22 s | **~0.08 s** |
| maxabs vs f32/torch matmul | (passes 5e-3) | ~0.0078 (passes 5e-3) |

Pointwise ADD/MUL also pass under fp16 (submit counts drop: e.g. `test_add` 6→1).

---

## 2026-08-07 — PC-chain buffers + medium gemm sweep

### Bigger regcmd/task floors (not a hard 23-task HW cap)

Symptom: gemm N=12 → **23** EW ops in one ioctl → `TimeoutError` errno 110; chains of 22 and 24 worked. Suspected 4096B scratch — **not** a hard max of 23 tasks (`struct_rknpu_task`×23 ≈ 920B; 23× fp16 EW regcmd ≈ 4048B still ≤4096, while 24 needs two pages and already succeeded after realloc).

Fix (match allbilly `elementwise.py`): raise floors to **regcmd 64KiB**, **task 16KiB**. After that, chain length 23 and **12×12 gemm** both pass.

Refs: `ref/allbilly-rk3588` / `tinygrad/ref/rk3588` — `experimental/pcchain.md` (EW: `AMOUNTS = next body qword count`); no documented “max 23 tasks”.

### `test_medium_gemm` advance (`ROCKCHIP_EW_OUT=fp16`, +1 N, 30s wall cap)

| N | Result |
|---|--------|
| 9–11 | ok (before bigger buffers; 12 timed out on 23-task chain) |
| 12–17 | ok after 64KiB/16KiB floors (~70–140 ms, 1 ioctl) |
| **18** | **tol fail** vs f32 matmul ref (maxabs≈0.0105 > 5e-3) — stopped |

At this point `test_medium_gemm` was `(17,17)@(17,17)`. `test_small_gemm` remains `(8,8)@(8,8)`.

---

## 2026-08-07 — EW `IN_PRECISION=5` (fp32-in) HW probe

TRM (`docs/rk3588_trm.md` / allbilly) lists `in_precision=3'd5` = fp32 as a valid encoding. allbilly/npu working recipes never use it for EW: always `IN=2` + `MRDMA_FP16TOFP32_EN(1)`, even when `OUT=5`.

### What we tried (single EW ADD, n=8, recover via `simple_add.py` each trial)

| Config | Result |
|--------|--------|
| `OUT=5 IN=2` mtx (shipped) | **OK** exact |
| `OUT=2 IN=2` WIDTH (shipped) | **OK** exact (on clean NPU) |
| `IN=5 OUT=5 PROC∈{2,5}` × `erdma_ds∈{0,1,2,3}` × mtx/WIDTH × fp32 buf | **TimeoutError** every time |
| `IN=5 OUT=2 PROC=5` mtx | **Timeout** |
| `IN=5` + `MRDMA_FP16TOFP32_EN` / half buf | **Timeout** |
| Control: `IN=2 OUT=5 PROC=5` + fp16to32 half | **completes** (not a timeout) |

### Verdict

**No usable EW fp32→fp32 (or fp32→fp16) RDMA path found.** Register space allows `IN=5`; submitting it hangs the job (errno 110). Stay on **half-in** (`IN=2` + widen for ALU). OUT=5 still needs host `_fp32_slots_to_half` between dependent stages; gemm accumulate remains a half reduce tree either way. Prefer `ROCKCHIP_EW_OUT=fp16` for chain/throughput; keep OUT=5 when single-op f32-store then cast matters.

---

## 2026-08-07 — corrected medium GEMM boundary + compensated DPU EW

All results here use DPU EW only: `DEFAULT_FLOAT=HALF`, `ROCKCHIP_EW_OUT=fp16`, FP16 inputs and outputs, no CMAC/CNA and no host sum.

### Correction: 18×18 was not a failure

The earlier sweep stopped at N=18 because `maxabs > 5e-3` was treated as failure. The test actually uses
`abs(error) <= atol + rtol*abs(reference)` with `atol=rtol=5e-3`.

| Sequential reduction | Result | Worst tolerance ratio |
|---|---:|---:|
| 18×18 | pass | 0.747 |
| 19×19 | pass | 0.769 |
| 20×20 | pass | 0.815 |
| **21×21** | **fail** (1 element) | **1.096** |

Thus the stock sequential FP16 EW limit for the deterministic sweep data is **20×20**, not 17×17.

### `ROCKCHIP_EW_REDUCE=kahan`

For a fully unrolled ADD tree of MUL terms, the Rockchip lowering can replace the sequential sum with a Kahan-style compensated sum after symbolic simplification. The compensation therefore remains in the emitted program instead of simplifying back to zero. It uses only FP16 DPU EW MUL/ADD operations and represents the running error as another FP16 value.

- EW stages for reduction length K: `K + 7*(K-1) = 8K-7`.
- With the current 64-task software chain cap: N=21 uses 161 stages / 3 ioctls; N=32 uses 249 stages / 4 ioctls.
- Every square size N=21–32 passes the same 5e-3/5e-3 tolerance.
- N=33 currently reaches `RKPLAN_REJECT:unsupported_graph`: the optimizer only fully unrolls arbitrary reductions through K=32. This is now the next compiler ceiling, not a measured DPU accuracy ceiling.
- The Kahan sweep reaches N=32 under the same deterministic inputs.

The shared compensated DAG also required memoizing `lower_ew.visit`; without it, recursively revisiting shared nodes grew exponentially during lowering.

### 1×1 through 32×32 wall profile

Command configuration: `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=kahan`.
Inputs were realized before timing. `cold` is the first output `realize()` for the shape; `warm` is the median of the next three. Torch reference generation is excluded. All rows passed.

| N | cold ms | warm median ms | ioctls |
|---:|---:|---:|---:|
| 1 | 86.461 | 2.026 | 1 |
| 2 | 59.332 | 2.502 | 1 |
| 3 | 36.222 | 2.773 | 1 |
| 4 | 30.944 | 3.051 | 1 |
| 5 | 45.860 | 3.753 | 1 |
| 6 | 37.601 | 4.098 | 1 |
| 7 | 41.426 | 4.439 | 1 |
| 8 | 45.651 | 5.064 | 1 |
| 9 | 50.097 | 5.625 | 1 |
| 10 | 55.050 | 6.269 | 2 |
| 11 | 60.003 | 7.073 | 2 |
| 12 | 65.084 | 7.911 | 2 |
| 13 | 71.397 | 8.643 | 2 |
| 14 | 77.842 | 9.527 | 2 |
| 15 | 84.717 | 10.488 | 2 |
| 16 | 92.066 | 11.719 | 2 |
| 17 | 119.975 | 13.010 | 2 |
| 18 | 314.339 | 14.451 | 3 |
| 19 | 118.008 | 15.753 | 3 |
| 20 | 127.791 | 17.362 | 3 |
| 21 | 138.563 | 19.086 | 3 |
| 22 | 149.768 | 21.145 | 3 |
| 23 | 160.815 | 22.994 | 3 |
| 24 | 174.269 | 25.370 | 3 |
| 25 | 187.882 | 27.323 | 3 |
| 26 | 202.478 | 29.707 | 4 |
| 27 | 218.993 | 32.358 | 4 |
| 28 | 268.390 | 35.380 | 4 |
| 29 | 256.566 | 38.584 | 4 |
| 30 | 274.921 | 41.661 | 4 |
| 31 | 294.521 | 45.336 | 4 |
| **32** | **312.460** | **49.019** | **4** |

The N=18 cold result is an outlier; warm scaling is monotonic. These are end-to-end output-realization wall times, not isolated ioctl hardware time.

---

## 2026-08-07 — EW PC-chain cap raised from 64 to 256

The prior `_EW_CHAIN=64` was a conservative software cap, not a measured hardware limit. The compensated 32×32 GEMM has 249 dependent FP16 EW tasks, making it a direct chain test.

| Software cap | ioctls | Median wall (20 inputs) | Result vs cap 64 |
|---:|---:|---:|---:|
| 64 | 4 | 50.581 ms | baseline |
| **256** | **1** | **50.915 ms** | bit-exact 20/20 |

Additional single-ioctl dependent ADD-chain probes passed through 256 tasks. The real compensated GEMM also produced identical outputs at caps 64, 128, and 256; random-seed tolerance misses were identical across caps and are numerical, not chain corruption.

Set the OUT=2 `_EW_CHAIN=256`. The 32×32 compensated GEMM now fits in one ioctl. Wall time does not improve materially because emitting/patching 249 register bodies and the EW work dominate this small workload; the change removes submit boundaries rather than arithmetic.

This does **not** apply to OUT=5 mtx512: a 256-task `test_add` chain timed out. Keep `_EW_CHAIN_FP32=64`; the raised cap is specifically for the contiguous FP16-output EW recipe used by compensated GEMM.

---

## 2026-08-07 — compensated EW advanced one-by-one to 37×37

Rockchip's optimizer now permits full unroll through K=64. Both the heuristic threshold and the generic `UNROLL <= 32` guard had to be raised specifically for the ROCKCHIP renderer; before that, N=33 produced three stores and was rejected before submission.

Each size was run separately with a hard 30-second timeout under `ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=kahan`:

| N | Result | Worst tolerance ratio | ioctls | first-run wall |
|---:|---:|---:|---:|---:|
| 33 | pass | — | 2 | <1 s |
| 34 | pass | 0.829 | 2 | 666 ms |
| 35 | pass | 0.807 | 2 | 875 ms |
| 36 | pass | 0.852 | 2 | 934 ms |
| **37** | **pass** | **0.862** | **2** | **877 ms** |
| 38 | **fail** (1 element) | 1.188 | 2 | 1054 ms |

The Kahan sweep reaches 37×37. N=38 is a numerical limit: Kahan corrects ADD-rounding but cannot recover the product bits lost when each FP16×FP16 result is stored as FP16.

### N=37 merged into one PC chain

Kahan N=37 emits `8*37-7 = 289` dependent FP16 EW tasks, which crossed the previous 256-task cap. Raising the OUT=2 cap to 512 was bit-exact against cap 256 for 10/10 inputs:

| FP16 chain cap | ioctls | warm median wall |
|---:|---:|---:|
| 256 | 2 | 70.564 ms |
| **512** | **1** | **70.013 ms** |

Set `_EW_CHAIN=512` for OUT=2. OUT=5 mtx512 remains separately capped at 64.

---

## 2026-08-07 — FP16 TwoProduct EW reaches 64×64

Config: `ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=twoproduct`. DPU EW only, no CMAC/CNA and no host arithmetic.

Kahan cannot recover the low product bits after FP16 store. TwoProduct uses a Dekker split (`splitter=65`) to represent every FP16×FP16 product as two FP16 values: rounded product plus exact-ish residual. A general TwoSum then accumulates all product highs followed by residuals in a two-half `(high, low)` accumulator. The graph is constructed after symbolic simplification so the error terms remain as physical EW MUL/ADD tasks.

The mixed TwoProduct graph timed out with 512 tasks in one chain but passes with 256, so runtime uses `_EW_CHAIN_TWOPRODUCT=256` while ordinary OUT=2 Kahan retains 512.

Every size was run separately with a hard 30-second timeout. All N=38 through N=64 passed:

| N | ratio | ioctls | first-run wall ms | N | ratio | ioctls | first-run wall ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 38 | 0.117 | 7 | 2192 | 52 | 0.126 | 9 | 5608 |
| 39 | 0.115 | 7 | 2576 | 53 | 0.127 | 10 | 5853 |
| 40 | 0.093 | 7 | 2738 | 54 | 0.173 | 10 | 6194 |
| 41 | 0.169 | 8 | 3464 | 55 | 0.148 | 10 | 6502 |
| 42 | 0.106 | 8 | 3125 | 56 | 0.145 | 10 | 6846 |
| 43 | 0.074 | 8 | 3336 | 57 | 0.152 | 10 | 7182 |
| 44 | 0.117 | 8 | 3549 | 58 | 0.136 | 10 | 7717 |
| 45 | 0.120 | 8 | 3895 | 59 | 0.180 | 11 | 7980 |
| 46 | 0.078 | 8 | 4005 | 60 | 0.158 | 11 | 8350 |
| 47 | 0.149 | 9 | 4215 | 61 | 0.134 | 11 | 8849 |
| 48 | 0.105 | 9 | 4494 | 62 | 0.155 | 11 | 9554 |
| 49 | 0.160 | 9 | 4854 | 63 | 0.144 | 11 | 9888 |
| 50 | 0.151 | 9 | 5113 | **64** | **0.145** | **11** | **10106** |
| 51 | 0.150 | 9 | 5281 | | | | |

The authoritative GEMM milestone is now `test_gemm_fp16` at 64×64. It passes in TwoProduct mode below the 30-second wall cap with substantial numerical margin. The redundant incremental `test_medium_gemm` was removed; its one-by-one census remains documented above.

### `test_gemm_fp16` cold wall decomposition

Profile config matches the passing test: `DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_OUT=fp16 ROCKCHIP_EW_REDUCE=twoproduct`. The named pytest passes in **18.62 s** with `-n12`; a directly instrumented cold realization takes **7.207 s**, so approximately **11.4 s** is pytest/xdist worker startup and imports.

| Cold realization component | Wall | Share |
|---|---:|---:|
| render + `lower_ew` | 6293 ms | 87.3% |
| program `__call__` | 404 ms | 5.6% |
| scheduler/program setup outside those regions | 509 ms | 7.1% |
| **output `realize()`** | **7207 ms** | **100%** |

The TwoProduct DAG itself is cheap to construct (17.9 ms). Most of the 6.24 s inside `lower_ew` is plan bookkeeping over the 2806-op graph: dependency/use counting, scratch assignment, and gather-offset construction. This is the dominant next optimization target, not ioctl count.

Program-call decomposition:

| Runtime component | Wall |
|---|---:|
| gather/setup/final-output work outside EW runner | 276.4 ms |
| EW runner total | 128.1 ms |
| ├ command emit (2806 bodies) | 40.6 ms |
| ├ relocation patch | 17.9 ms |
| ├ submit CPU construction | 29.1 ms |
| ├ 11 blocking ioctls | 14.0 ms |
| └ loop/other | 26.5 ms |

Other excluded measurements: input Tensor construction 3.6 ms, Torch reference 0.9 ms, and final `numpy()` copyout 2.7 ms.

### Cold-lowering optimization

The 7.207 s direct cold realization was Python compiler overhead rather than DPU execution. `lower_ew` calculated each logical input gather four times because repeated leaf UOps were only deduplicated after offset expansion, and its dependency-use count scanned the whole graph once per node.

Caching lowered leaf operands and replacing the quadratic use-count comprehension with one linear edge walk preserves the output (worst tolerance ratio remains 0.145) and changes the cold profile as follows:

| Component | Before | After | Change |
|---|---:|---:|---:|
| direct output `realize()` | 7.207 s | 2.732 s | 2.64× faster |
| `lower_ew` | 6.306 s | 1.762 s | 3.58× faster |
| gather-offset construction | 4.170 s / 512 calls | 1.054 s / 128 calls | four duplicate calls removed |
| other lowering work | 2.119 s | 0.691 s | 3.07× faster |
| named pytest (`-n12`) | 18.62 s | 14.39 s | 4.23 s less wall time |

`test_gemm_fp16` still passes under its hard 30-second timeout. The remaining roughly 1.05 s in gather-offset construction is now the largest measured lowering component.

---

## 2026-08-07 — FP16-only cleanup and CPU-execution audit

The Rockchip backend now exposes only the hardware path under test: contiguous FP16 DPU EW output. The legacy mtx512 `OUT=5` path was removed because it converted FP32 results back to FP16 on the host after every operation. FP32 argument casting, host pack/unpack helpers, and the intermediate-result copyout path were removed with it.

`sz.py` result for executable Rockchip code:

| File | Before | After | Reduction |
|---|---:|---:|---:|
| `tinygrad/renderer/rockchip.py` | 438 | 325 | 113 |
| `tinygrad/runtime/ops_rockchip.py` | 241 | 131 | 110 |
| **combined** | **679** | **456** | **223 (32.8%)** |

Comments and docstrings were retained because `sz.py` does not count them. Repository total lines fell from 25,730 to 25,507.

### Execution audit

- There is no CMAC/CNA path.
- There is no NumPy import, FP32 temporary, `astype`, host sum, or host ADD/MUL in the Rockchip renderer/runtime.
- Contiguous FP16 arguments are bound directly as DPU sources; they are no longer copied into scratch.
- The final DPU EW operation writes directly to the destination argument; there is no host result copy.
- Constants and constant-only fills are materialized as FP16 bytes on the host. This is data initialization, not tensor arithmetic.
- GEMM still needs host gather/layout preparation because DPU EW only consumes contiguous pairwise vectors and cannot apply tinygrad's arbitrary load indexes. Masked padding is represented by `-1` gather entries and zero-filled. Every GEMM MUL/ADD, including TwoProduct residual recovery, executes on DPU EW.
- The safe PC-chain limit is stored in the compiled image (512 for ordinary/Kahan graphs and 256 for mixed TwoProduct graphs), removing the previous runtime dependence on the current environment variable.
- RKImage version validation is now strict, and malformed/nonzero masked-load defaults are rejected instead of silently miscomputed.

This is closer to other tinygrad hardware backends: allocator methods move buffers, `Program.__call__` binds arguments/builds commands/submits, and the device performs tensor arithmetic. Unlike general GPU/DSP backends, Rockchip still preprocesses indexed operands on the host and uses blocking ioctls rather than an asynchronous hardware queue, so it remains a focused DPU-EW capability harness rather than a general backend.

Verification after cleanup:

- `test_gemm_fp16`: **1 passed in 13.30 s** with `-n12` and a hard 30 s timeout.
- Direct 64×64 cold realization: **1.975 s**, 11 ioctls, within the 5e-3/5e-3 tolerance.
- Non-slow Rockchip census: **18 passed, 5 skipped in 12.16 s**. Four skips are slow tests; `test_mul_naninf` records that RK3588 FP16 DPU EW MUL returns NaN for infinity operands rather than hiding it with a CPU fallback.
- Full `ruff check .`: pass.
- Full `mypy tinygrad/`: pass (216 files).

---

## 2026-08-07 — 256×256 affine-gather profile and optimization

With a 180-second cap, Rockchip's FP16-tolerance `test_big_gemm` first passed in 110.55 s. The original `TestOps.test_big_gemm` completed in 136.96 s but missed its stricter 1e-4/1e-3 tolerance on 35 of 65,536 elements (maximum absolute difference 0.000977).

The initial direct cold profile took 108.738 s and submitted 88 ioctls:

| Component | Before | Share |
|---|---:|---:|
| render/lower | 86.511 s | 79.6% |
| compile-time gather-offset expansion | 83.235 s | 76.5% |
| runtime host gather/layout | 13.551 s | 12.5% |
| other scheduler/program setup | 7.384 s | 6.8% |
| DPU EW runner | 1.292 s | 1.2% |
| 88 blocking ioctls | 0.614 s | 0.6% |

The kernel has 11,254 logical EW operations over 65,536 output elements. The tested 64,000-element tile cap splits every operation in two, producing 22,508 DPU tasks. At the TwoProduct 256-task chain cap this is `ceil(22508/256) = 88` submits.

Raising the tile cap is unsafe. A 65,536-element one-ioctl ADD (`WIDTH=8191`) completed with 99.8% wrong values, while 65,528 (`WIDTH=8190`) timed out in the ioctl. The NPU was reset after the probe and `_MAX_EW_ELEMS_FP16` remains 64,000.

Large unmasked gathers are now encoded as a base plus affine `(destination divisor, range limit, source stride)` axes rather than a complete offset tuple. Masked/padded operations keep the general offset fallback. Runtime materializes both forms with bulk uint16 indexing; this performs only layout movement and integer address generation on the host, while every floating-point MUL/ADD remains on DPU EW.

After the change:

| Component | Before | After |
|---|---:|---:|
| direct cold realization | 108.738 s | **4.908 s** |
| render/lower | 86.511 s | **0.229 s** |
| gather-plan construction | 83.235 s | **0.005 s** |
| runtime host gather/layout | 13.551 s | **1.664 s** |
| serialized RKImage | 128 MiB of raw offsets | **475,838 bytes** |
| DPU tasks / ioctls | 22,508 / 88 | **22,508 / 88** |

The named Rockchip `test_big_gemm` now passes in 16.93 s including fresh pytest/xdist startup. The non-slow Rockchip census remains green (18 passed, 5 skipped), and full Ruff/mypy pass.

---

## 2026-08-07 — convolution census begins with 1x1

The generic `TestOps.test_simple_conv2d_1x1` case, input `(1,4,9,9)` and weight `(4,4,1,1)`, passes on Rockchip with FP16 input and DPU EW-only execution. It is now mirrored in `test/backend/test_rockchip.py` so the backend census uses the same `5e-3/5e-3` tolerance as `test_gemm_fp16` and asserts the hardware submit count.

Direct cold realization took **0.193 s**, used **1 ioctl submit**, and had maximum absolute error **1.52588e-05** against the Torch FP16 reference. The generic pytest case passed in **11.34 s** including fresh xdist worker startup.

The next representative 3x3 sweep also passes:

| Case | Output | Cold realization | Ioctl submits | Maximum absolute error |
|---|---:|---:|---:|---:|
| unpadded | `(1,4,7,7)` | 0.449 s | 7 | 0.00195312 |
| batch 2 | `(2,4,7,7)` | 0.293 s | 7 | 0 |
| padding 1 | `(1,4,9,9)` | 0.491 s | 7 | 1.52588e-05 |
| stride 2 | `(1,4,4,4)` | 0.295 s | 7 | 0 |
| depthwise, 4 groups | `(1,4,7,7)` | 0.102 s | 2 | 0 |

These cases remain FP16-input, DPU EW-only tests. Torch is used only to construct the test oracle; backend execution performs no host floating-point arithmetic.

The six-test convolution subset passes in **12.21 s** and the complete Rockchip census passes in **18.08 s** (`28 passed, 1 skipped`), both with `-n12` and a hard 30-second timeout. Full Ruff and mypy checks pass.

### Convolution reduction width and layout expansion

The convolution census now also covers both sides of K=64, the larger M4 cases from `test_ops`, grouped convolution, and dilation:

- 3x3 with 7 input channels (K=63): 11 submits, 0.629 s direct cold realization.
- 3x3 with 8 input channels (K=72): 13 submits, 0.676 s; this proves 64 reduction terms is not the current backend limit.
- 3x3 with 16 input/output channels (K=144): 25 submits, 1.195 s.
- 1x1 with 16 input/output channels at 32x32: 3 submits, 0.169 s.
- Two-group 3x3: 7 submits, 0.471 s.
- Dilated 3x3: 7 submits, 0.279 s.

All twelve Rockchip convolution tests pass in **13.42 s**. The complete backend census is now **34 passed, 1 skipped in 17.58 s**, still under the hard 30-second timeout.

### Convolution index and masked-layout fixes

Rockchip now evaluates `CDIV`, `CMOD`, `FLOORDIV`, and `FLOORMOD` while constructing integer gather indices. This is host address/layout calculation only; no floating-point operation moved off DPU EW. Output-padded transposed convolution no longer rejects its index expression and passes the allowed FP16 tolerance (1 submit, 1.806 s direct cold realization).

The optimizer no longer upcasts masked Rockchip axes into several scalar stores, preserving the renderer's single contiguous-store contract. Asymmetric 1D and 2D padding and padded 3D convolution now pass their original `test_ops` checks. A zero-select `WHERE` around a masked load is folded into that load's gather gate, so simple explicit padding also passes without CPU floating-point execution. The asymmetric 2D backend census case realizes in 0.469 s with 1 submit and exact output in the measured probe.

### Complete convolution census and native ReLU

The Rockchip census now mirrors every one of the 48 `test*conv*` methods from `test/backend/test_ops.py` and applies exactly the `test_gemm_fp16` ceiling (`atol=5e-3`, `rtol=5e-3`, including gradient tolerances). It stays synchronized with future convolution additions while the existing explicit Rockchip cases continue to assert exact ioctl-submit counts.

Native DPU ReLU is emitted by leaving `EW_RELU_BYPASS` clear after an EW pass-through. Convolution ADD trees may include a bias term, and the TwoProduct reduction now carries a three-half expansion so wider Conv2D reductions and nested bias/ReLU graphs remain within the FP16 contract. The resulting submit counts are recorded in the explicit backend tests rather than hidden behind a CPU path.

The formerly opaque EW words are now composed from the RK3588 register fields named by the hardware reference: FP16 data mode and element size, ALU algorithm, ReLU/LUT bypass, operand source/conversion, and operation type. These definitions reproduce the verified words exactly:

| Operation | `DPU_EW_CFG` |
|---|---:|
| ADD | `0x108202c0` |
| MUL | `0x108003c4` |
| native ReLU | `0x108000c0` |

Two cases use an FP32-accumulated Torch golden cast back to FP16: biased ConvTranspose2D and ConvTranspose3D. PyTorch CPU's FP16 reduction is less accurate for these cases; for example, the biased 2D outlier is `-0.60840` from CPU FP16 versus an exact FP16-input sum of `-0.61804`, while DPU EW returns `-0.61816`. Against the FP32 golden, the complete biased case has maximum absolute error `0.001953`, and ConvTranspose3D has zero tolerance violations with maximum absolute error `0.015625`. Torch remains test-oracle code only.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- All mirrored convolution cases: **42 passed, 6 intentionally skipped, 37 subtests passed in 42.51 s**.
- Complete `test/backend/test_rockchip.py`: **78 passed, 7 skipped, 37 subtests passed in 51.06 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No CMAC/CNA execution, host floating-point fallback, or relaxed tolerance was added. Host work remains integer gather/layout preparation; all convolution MUL/ADD/ReLU arithmetic executes through DPU EW with FP16 buffers.

---

## 2026-08-07 — DPU EW binary MAX and FP16 MaxPool2D census

The Rockchip renderer now distinguishes the two uses of the EW MAX datapath. Binary MAX sets `EW_RELU_BYPASS`, producing the register word `0x108002c0` verified by `~/rk3588/examples/elementwise.py` and the upstream research renderer. Native unary ReLU keeps that bypass clear and remains `0x108000c0`. Both words are assembled from named RK3588 register fields rather than embedded hex literals.

Pooling remains a DPU-EW limit experiment. The upstream pooling examples often switch to PPU or CMAC for layout/reformat and large reductions; none of those paths were imported. MaxPool windows are expanded into binary FP16 EW MAX operations, and the existing host gather performs integer address/layout preparation only.

Padded pooling required preserving the FP16 fill value in gather metadata. RKImage v14 stores the raw 16-bit fill pattern, allowing masked lanes to be initialized to `-inf` without host floating-point tensor arithmetic. The rewrite only moves a `WHERE` mask into a load when it proves that the outer condition implies the load condition, including commuted `AND` expressions. MAX-tree masks are folded only when every leaf has the same or exact complementary condition. This covers symmetric padding, asymmetric padding, and ceil-mode edge windows without a general unsafe `WHERE` rewrite.

Every numeric FP16 `test_max_pool2d*` method from `test/backend/test_ops.py` now passes at exactly the `test_gemm_fp16` tolerance ceiling (`atol=rtol=grad_atol=grad_rtol=5e-3`):

- **11 methods passed, 33 subtests passed** in 14.24 s on the NPU.
- `test_max_pool2d_padding_int` is explicitly skipped because Rockchip accepts FP16 inputs only.
- `test_max_pool2d_return_indices` is explicitly skipped because DPU EW does not produce the required integer index tensor.
- A direct FP16 binary-maximum regression and a MaxPool submit-count regression were added to `test/backend/test_rockchip.py`.

Direct cold output-realization profiles all used one `DRM_IOCTL_RKNPU_SUBMIT`:

| Case | Wall time | Submits | Maximum absolute error |
|---|---:|---:|---:|
| `(1,1,2,3)`, 2x2 | 0.110 s | 1 | 0 |
| `(3,2,17,14)`, 5x5 stride 1 | 0.141 s | 1 | 0 |
| `(4,2,111,28)`, asymmetric-padded 5x5 | 0.474 s | 1 | 0 |

Complete verification after the change:

- `test/backend/test_rockchip.py`: **91 passed, 9 skipped, 70 subtests passed in 44.94 s** with `-n12`.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 445 executable lines, runtime 142; comments and docstrings retained.
- CPU-execution audit: runtime NumPy use is limited to `uint16` buffer views and integer gather-index construction. There is no host MAX, ADD, MUL, reduction, FP32 conversion, CMAC, or CNA path.

AvgPool2D is the next pooling boundary. Five initial cases already execute, while the remaining forms expose FP32 `MUL/CAST` lowering and, for `count_include_pad=False`, dynamic reciprocal/mask expressions. Those are separate ADD/MUL lowering work and were not hidden by a CPU fallback in this MAX milestone.

---

## 2026-08-07 — FP16 AvgPool through DPU EW, including valid-count divisors

All `test_avg_pool*` cases from `test/backend/test_ops.py` now run through the FP16 DPU EW backend. No PPU, CMAC, CNA, or CPU tensor fallback was added. The PPU average-pooling recipes in `~/rk3588` and `rockchip-upstream-research` were used only to confirm hardware semantics; this branch deliberately continues to test the DPU EW ADD/MUL limit.

Fixed-divisor AvgPool lowers the FP32-shaped tinygrad epilogue to FP16 ADD/MUL and folds compile-time casts of scalar constants. The reduction sum and final scale multiply execute on DPU EW. Across the unpadded and padded 2D sweep, all 14 kernel/padding combinations used one ioctl and the largest measured absolute error was `0.0009765625`, below the unchanged `5e-3/5e-3` FP16 tolerance ceiling.

`count_include_pad=False` requires a divisor that varies at edges. This is still not host tensor arithmetic: the divisor depends only on static output geometry, never on input values. The compiler evaluates the geometry-only RANGE/WHERE expression and stores its raw FP16 reciprocal bits in RKImage v15. At program launch, the runtime copies those immutable `uint16` constants into scratch. It performs no floating-point reciprocal, sum, multiply, mean, or conversion. Every input-dependent ADD and MUL is submitted to DPU EW. A dedicated regression asserts that the padded valid-count path performs one NPU ioctl.

Average pooling also exposed the previous 64-term compiler unroll ceiling. Rockchip now permits scaled ADD reductions through 512 terms, while GEMM/convolution and other reductions keep their proven 64-term heuristic bound. This enables the 308-term global 11x28 AvgPool and the 512-term AvgPool3D case without changing convolution lowering. The reciprocal-leaf detector is restricted to static `RECIPROCAL` roots, avoiding a quadratic graph walk in convolution compilation.

Representative direct cold profiles, excluding Torch reference generation:

| Case | Wall time | Submits | Maximum absolute error |
|---|---:|---:|---:|
| 3x3 fixed divisor, `(32,2,11,28)` | 0.217 s | 1 | 0.0004883 |
| 3x3 valid-count padding, `(32,2,11,28)` | 0.473 s | 1 | 0.0004883 |
| global 11x28, `(32,2,11,28)` | 1.032 s | 2 | 0.0003052 |
| AvgPool3D 8x8x8, `(1,1,16,16,16)` | 7.981 s | 3 | 0.0003357 |

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- AvgPool census: **10 passed, 26 subtests passed in 20.58 s**.
- Convolution regression census after the unroll change: **42 passed, 6 skipped, 37 subtests passed in 45.66 s**.
- Complete Rockchip census: **102 passed, 9 skipped, 96 subtests passed in 60.12 s**.
- Full Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 498 executable lines, runtime 144; comments and docstrings retained.
- Runtime CPU audit: NumPy is used only for raw `uint16` views and integer indexing/copies. There is no host floating-point arithmetic or conversion.

PyTorch CPU does not implement FP16 AvgPool3D, so that one test constructs its oracle in FP32 and casts the oracle to FP16. This affects test-reference generation only; Rockchip execution remains FP16 DPU EW.

---

## 2026-08-08 — First post-518 milestone: nearest interpolation through DPU EW

Development restarted from the known-good `518487a5c24c7390e5b40cdb85f91d6e158c4383` baseline. The first newly admitted upstream method is `TestOps.test_interpolate_nearest`, covering six 1D, 2D, and 3D resize shapes. It passes at the unchanged `test_gemm_fp16` tolerance ceiling (`atol=rtol=grad_atol=grad_rtol=5e-3`). A direct `(2,3,13) -> (2,3,9)` regression additionally asserts exactly one NPU ioctl.

Nearest interpolation is a gather, but the output is not accepted as a host-produced result. Rockchip prepares only the static integer gather offsets and mask, then sends the gathered FP16 values through a DPU EW `ADD 0` pass-through stage. Typed compile-time evaluation now preserves integer, boolean, FP16, and FP32 cast semantics in layout expressions. No input-dependent floating-point arithmetic, CMAC, CNA, PPU, or CPU numeric fallback was added.

Focused verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Direct submit-count regression: **1 passed in 2.83 s**, exactly 1 ioctl.
- Full upstream `test_interpolate_nearest`: **1 passed in 3.96 s**, covering all six resize shapes.
- Gather regression group (nearest, MaxPool, AvgPool, output-padded ConvTranspose2D): **4 passed in 8.13 s**.
- Complete Rockchip census, run as 113 separate sequential pytest processes to limit retained program buffers: **104 passed, 9 intentionally skipped**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 501 executable lines, runtime 144; comments and docstrings retained.

The current `518` runtime still requests contiguous GEM buffers against an 8 MiB CMA pool. Several large existing cases logged recoverable CMA allocation failures even with one pytest node per process, although this run produced no RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. Fixing this allocation policy safely is the next prerequisite before broadening the interpolation census.

---

## 2026-08-08 — Page-backed RKNPU buffers remove the 8 MiB CMA limit

All Rockchip GEM objects now use the allocation policy captured from the vendor RKNN runtime: ordinary data, scratch, and command buffers use `NON_CONTIGUOUS|CACHEABLE|IOMMU_LIMIT_IOVA_ALIGNMENT` (`0x403`), while task buffers additionally use `KERNEL_MAPPING` (`0x40b`). This selects the driver's page-backed IOMMU allocation path directly instead of first exhausting the board's 8 MiB CMA pool and entering the driver's broken contiguous-to-noncontiguous fallback.

Cacheable mappings require explicit ownership transfer. Copy-in, copy-out, command/task construction, constants, gathers, and fills now issue the corresponding `MEM_SYNC` ioctl. Before each program, every argument is synchronized from and then back to the device, matching the vendor behavior. A narrower gather-only synchronization experiment produced stale zero cache lines in `test_add3` after allocator reuse, so the broader handoff is required for correctness on this driver.

Allocation is transactional: a failed create is reported as `MemoryError`, while a failed map/mmap destroys the newly created GEM before propagating the allocation failure. Program-owned scratch and command/task lifetime, PC-chain construction, and the passing `518` submission path remain unchanged.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Tiny ADD health probe: **1 passed in 2.83 s**, exactly 1 ioctl.
- Large GEMM + nearest gather + chained ADD: **3 passed in 6.89 s**.
- Complete monolithic Rockchip census: **104 passed, 9 skipped, 96 subtests passed in 169.07 s**.
- Post-coherency regression (large GEMM, nearest, chained ADD, output-padded ConvTranspose2D): **4 passed in 11.38 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 501 executable lines, runtime 165; comments and docstrings retained.

Across the complete monolithic run, `CmaFree` stayed at 6144 KiB and the kernel logged no CMA allocation failure, RKNPU timeout, invalid IRQ, IOMMU fault, or oops. The allocator safety prerequisite is therefore cleared. The next performance target is the fresh-process `test_simple_conv_transpose3d` node, previously measured at 65.21 s.

---

## 2026-08-08 — Vectorized static gather planning removes ConvTranspose3D compile stall

`TestRockchipConvOps.test_simple_conv_transpose3d` previously passed but took **65.21 s** in a fresh pytest process. `DEBUG=2` separated the device work from compilation: scheduling took 182.05 ms, two uploads took 0.25 ms total, and the DPU program took 271.54 ms. Almost the entire remaining minute was Rockchip renderer CPU time.

A CPU profile measured 112.88 s under profiling in `lower_ew`. The 108 masked convolution gathers called the scalar index interpreter 58.6 million times; `_eval_expr` consumed 108.43 s. Per-output expression memoization reduced the ordinary test from 65.21 s to 45.16 s, confirming repeated static-index evaluation as the cause but not removing enough Python iteration.

Rockchip now evaluates complete static gather-index vectors with NumPy. RANGE values, typed casts, comparisons, masks, integer division/modulo, and affine output addresses are computed as vectorized layout metadata. This remains host integer address preparation only: no input-dependent FP16 value, convolution product, sum, or activation is computed on CPU. The existing runtime gather copies FP16 bit patterns according to those offsets, and all numeric convolution work still runs through DPU EW.

Results after vectorization:

- ConvTranspose3D pytest time: **6.32 s** (down from 65.21 s, 10.3x faster).
- Shell wall time including imports: **8.604 s**.
- Scheduling: 170.26 ms; NPU program: 274.85 ms.
- Complete monolithic Rockchip census: **104 passed, 9 skipped, 96 subtests passed in 67.19 s** (down from 169.07 s after cache-sync enablement).
- Focused nearest/AvgPool/ConvTranspose regressions: **4 passed in 7.38 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 552 executable lines, runtime 165; comments and docstrings retained.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. Every individual Rockchip node is again below 30 seconds.

---

## 2026-08-08 — `test_interpolate_nearest_exact`

The next upstream method was added individually to the Rockchip census. `TestOps.test_interpolate_nearest_exact` reuses the six 1D, 2D, and 3D shapes from nearest interpolation with PyTorch's `nearest-exact` coordinate rule. All static index casts and masks are prepared by the vectorized Rockchip gather planner, and every FP16 result passes through DPU EW before reaching the output.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused upstream method: **1 passed in 3.41 s**; shell wall time 5.574 s.
- Complete Rockchip census: **105 passed, 9 skipped, 96 subtests passed in 67.96 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 552 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next upstream interpolation case is `test_interpolate_linear`.

---

## 2026-08-08 — `test_interpolate_linear`

The upstream 1D linear interpolation method now passes both resize directions, `(2,3,52) -> (2,3,29)` and `(2,3,29) -> (2,3,52)`, at the unchanged FP16 tolerance. A direct regression asserts that the first direction uses exactly one ioctl.

Interpolation coordinates and fractional weights depend only on shape. Rockchip retains those FP32 cast/WHERE/TRUNC expressions as static geometry and uploads their final FP16 bit patterns. It no longer rewrites geometry-only FP32 ADD/MUL into dynamic nodes. The gathered low/high FP16 input values and every input-dependent lerp MUL/ADD execute through DPU EW; there is no CPU interpolation or input-dependent arithmetic.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused direct and upstream methods: **2 passed in 3.06 s**; direct case exactly 1 ioctl.
- Complete Rockchip census: **107 passed, 9 skipped, 96 subtests passed in 101.45 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 553 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next upstream interpolation case is `test_interpolate_linear_corners_aligned`.

---

## 2026-08-08 — `test_interpolate_linear_corners_aligned`

The second upstream 1D linear interpolation method passes both resize directions with `align_corners=True`. This changes only the static coordinate/weight formula; gathered FP16 values and lerp arithmetic continue through the already verified DPU EW path.

- Focused upstream method: **1 passed in 3.03 s**.
- Complete admitted interpolation group: **4 passed in 4.38 s**.
- Rockchip collection: 117 nodes (the previous full census was 107 passed / 9 skipped; this new node passed focused validation).
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next upstream interpolation case is `test_interpolate_bilinear`.

---

## 2026-08-08 — `test_interpolate_bilinear` without shared-core changes

The upstream bilinear method now passes all three 2D resize pairs. A direct `(2,3,12,20) -> (2,3,9,31)` regression asserts exactly two ioctls, one DPU EW program per separable axis.

Tinygrad's generic interpolation keeps the first-axis result in an internal FP32-sized buffer. Rockchip still advertises only FP16 as a supported dtype and performs no FP32 arithmetic or conversion. For Rockchip programs, that oversized internal buffer now carries a contiguous FP16 payload produced by the first DPU pass; the second Rockchip pass reads the same FP16 payload. This preserves the device's actual FP16 storage/arithmetic contract without changing `tinygrad/mixin/op.py` or any other shared core file.

The host continues to prepare static integer gathers and geometry-only FP16 weights. Every input-dependent low/high interpolation MUL/ADD executes on DPU EW. No CPU interpolation, CMAC, CNA, or PPU path was added.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused direct and upstream methods: **2 passed in 4.29 s**; direct case exactly 2 ioctls.
- Complete Rockchip census: **110 passed, 9 skipped, 96 subtests passed in 101.59 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 553 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next upstream interpolation case is `test_interpolate_bilinear_corners_aligned`.

---

## 2026-08-08 — `test_interpolate_bilinear_corners_aligned`

The aligned-corners bilinear method passes all three upstream 2D resize pairs through the same two-pass FP16 DPU EW path.

- Focused upstream method: **1 passed in 4.04 s**.
- Complete admitted interpolation group: **6 passed in 6.92 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next upstream interpolation case is `test_interpolate_trilinear`.

---

## 2026-08-08 — `test_interpolate_trilinear`

The upstream trilinear method passes `(2,3,5,2,8) -> (2,3,3,6,4)`. A smaller direct regression confirms exactly three ioctls, one FP16 DPU EW lerp pass for each spatial axis. Both internal FP32-sized separable buffers carry only contiguous FP16 NPU-produced payloads.

- Focused direct and upstream methods: **2 passed in 3.61 s**; direct case exactly 3 ioctls.
- Complete admitted interpolation methods plus four submit-count regressions: **11 passed in 7.94 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The final floating-point interpolation method is `test_interpolate_trilinear_corners_aligned`.

---

## 2026-08-08 — Complete FP16 interpolation census

`test_interpolate_trilinear_corners_aligned` passes its upstream 3D resize case, completing all eight floating-point `test_interpolate*` methods from `TestOps`: nearest, nearest-exact, 1D linear, aligned 1D linear, bilinear, aligned bilinear, trilinear, and aligned trilinear.

All nearest values and linear low/high operands are gathered as FP16 bit patterns. Static geometry supplies only indices and FP16 weights. Input-dependent interpolation arithmetic executes on DPU EW in one, two, or three passes according to spatial rank. Rockchip remains FP16-only, performs no CPU interpolation, and changes no shared tinygrad core file.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- Focused aligned-trilinear method: **1 passed in 3.24 s**.
- Complete Rockchip census: **114 passed, 9 skipped, 96 subtests passed in 104.06 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 553 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The floating-point interpolation group is complete; uint8 interpolation remains outside the FP16 input contract.

---

## 2026-08-08 — `test_full_like` and `test_full`: typed raw constant fill

The remaining `TestOps` census is now advancing in source order. `test_full_like` passes its explicit FP32 and int32 outputs, and `test_full` passes its integer output. These require no numeric NPU operation: each tensor is a compile-time scalar repeated into storage.

RKImage v16 records the fill element width in the previously reserved fill byte. The compiler serializes the scalar using the destination dtype's exact byte format, and the runtime copies those immutable bytes across the destination before synchronizing the cacheable buffer to the device. Dynamic Rockchip arithmetic remains FP16-only; this is raw constant initialization, not CPU input-dependent arithmetic or FP32 emulation.

Verification with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP ROCKCHIP_EW_REDUCE=twoproduct`:

- `test_full_like`: **1 passed in 2.66 s**.
- `test_full`: **1 passed in 2.61 s**.
- Complete Rockchip census: **116 passed, 9 skipped, 96 subtests passed in 109.58 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- `sz.py`: renderer 556 executable lines, runtime 165.

The complete run left `CmaFree` at 6144 KiB and produced no CMA failure, RKNPU timeout, invalid IRQ, IOMMU fault, or kernel oops. The next source-order methods are the negative-dimension exception tests.

---

## 2026-08-08 — Negative-dimension validation methods

Four source-order `TestOps` methods were admitted and run individually: `test_negative_dims`, `test_negative_dims_full`, `test_negative_dims_eye`, and `test_negative_dims_kaiming`. Every invalid shape raises before device allocation or submission.

- Individual times: 2.59 s, 2.58 s, 2.57 s, and 2.51 s.
- Incremental creation/validation group: **6 passed in 2.82 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).

The next source-order method is `test_zeros`.

---

## 2026-08-08 — Creation methods through `test_eye`

Six more source-order methods were admitted and run individually: `test_zeros`, `test_zeros_like`, `test_empty_0`, `test_ones`, `test_ones_like`, and `test_eye`. Uniform values use typed raw constant fill. `test_empty_0` folds to a constant NaN fill. Eye matrices are geometry-only FP16 static vectors followed by DPU EW pass-through, not CPU-generated numeric outputs.

- Individual methods: all pass in 2.66–3.03 s.
- Complete incremental creation/validation group: **12 passed in 3.15 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next source-order method is `test_split`.

---

## 2026-08-08 — Movement and range construction through `test_linspace`

Eight more source-order methods were admitted and run individually: `test_split`, `test_chunk`, `test_unfold`, `test_meshgrid`, `test_arange`, `test_arange_big`, `test_arange_4096`, and `test_linspace`.

Split/chunk remain views where possible; materialized FP16 movement uses the existing gather plus DPU pass-through path. Range/linspace values depend only on their scalar construction arguments and are materialized as static typed data. This includes FP16, int8, int32, and int64 output checks and does not introduce input-dependent CPU arithmetic.

- Individual method times: 2.92–5.72 s.
- Complete incremental source-order group: **20 passed in 4.94 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next source-order method is `test_sum_fake`.

---

## 2026-08-08 — Source-order sum/max reductions

Seven more methods were admitted and run individually: `test_sum_fake`, `test_sum_collapse`, `test_sum_collapse_neg`, `test_sum_pad_collapse`, `test_sum_twice`, `test_sum_cat_collapse`, and `test_max_dont_collapse`.

Geometry/constant reductions fold where legal. `test_sum_twice` reduces random FP16 input across two stages and therefore exercises the existing DPU EW ADD lowering rather than a host arithmetic path. No tolerance or backend code change was needed.

- Individual method times: 2.64–4.05 s.
- Complete incremental source-order group: **27 passed in 9.08 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

The next source-order method is `test_where`.

---

## 2026-08-08 — Lerp, broadcast ADD, SUB, and negation

Seven FP16 arithmetic methods were admitted and run individually: `test_lerp`, `test_broadcasted_add`, `test_broadcasted_add_2`, `test_sub`, `test_scalar_sub`, `test_scalar_rsub`, and `test_neg`. They reuse DPU EW ADD/MUL and broadcast gathers; no renderer/runtime change was necessary.

- Individual method times: 2.70–3.07 s.
- Complete incremental source-order group: **34 passed in 9.58 s**.
- Ruff: pass.
- `mypy tinygrad/`: pass (216 source files).
- No new kernel error signature.

`test_where` and the boolean tail of `test_tril` remain unadmitted because their packed-boolean input/int32 output cannot be executed by the FP16-only DPU without a CPU value-selection fallback. They are not skipped or counted as passing.

---

## 2026-08-08 — Exact upstream ADD and MUL methods

Six additional upstream `TestOps` methods were admitted: `test_tiny_add`, `test_tiny_mul`, `test_add`, `test_add3`, `test_mul`, and `test_scalar_mul`. Each method first passed in its own fresh pytest process, then the complete incremental class passed sequentially in one persistent process.

- Individual method times: 2.62–3.07 s.
- Complete incremental source-order group: **40 passed in 9.75 s**.
- No new RKNPU, IOMMU, CMA, or kernel-oops message during the grouped run.

The exact upstream `test_div` method is deliberately not counted yet. Its first dynamic graph is represented as `MUL(RECIPROCAL(x))`, and the Rockchip planner rejects `RECIPROCAL` before any ioctl. Native FP16 DPU division will be developed as a separate hardware milestone, beginning with a tiny one-submit regression before admitting the complete upstream method.

---

## 2026-08-08 — Native FP16 DPU division

Dynamic half division now uses RK3588's native DPU EW FDIV algorithm. Advertising typed `Ops.FDIV` lets tinygrad fuse `x * reciprocal(y)` before Rockchip lowering. The emitted stage uses named FDIV configuration bits, the FDIV output conversion scale, and FDIV's RDMA operand mode. No reciprocal or division is evaluated on the CPU, no CMAC path is used, and no shared tinygrad core file changed.

The hardware rollout was deliberately sequential:

- Three-element mixed-sign regression: **1 passed in 2.84 s**, exactly **1 ioctl**.
- Exact upstream `TestOps.test_div`: **1 passed in 3.18 s**.
- Complete incremental source-order group: **41 passed in 9.76 s**.
- Complete Rockchip census: **156 passed, 9 skipped, 96 subtests passed in 111.21 s**.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU, IOMMU, CMA, or kernel-oops message. Division-by-zero/non-finite semantics and rounding-mode division remain separate cases; this milestone covers the exact ordinary floating-point `test_div` method only.

---

## 2026-08-08 — Scalar FP16 division

The exact upstream `TestOps.test_scalar_div` method passes all seven tensor/scalar and scalar/tensor forms through native DPU FDIV, including two rank-zero cases. It passed individually in **3.07 s** without a renderer/runtime change.

Non-finite numerator handling is covered by the following separate milestone.

### Non-finite division sign

The first `-inf/x` sign reconstruction (`MUL → MAX → FDIV → MUL → FDIV`) timed out at task 4/5. That version was never committed. The replacement rewrites `±inf/x` as `(±1/x)/0`, producing two adjacent FDIV stages with no intervening pipeline transition. The vendor reference elementwise health check subsequently passed all ADD/MUL/SUB/MAX/NEG/FDIV sizes through 131,072 elements, allowing the replacement to be verified on the same boot.

- Separate `+inf/x` and `-inf/x` probes: bit-exact pass.
- Direct sign regression: **1 passed in 2.64 s**, exactly one ioctl for each sign.
- Exact upstream `TestOps.test_div_naninf`: **1 passed in 2.75 s**.
- Complete incremental source-order group: **43 passed in 9.97 s**.
- Complete Rockchip census: **159 passed, 9 skipped, 96 subtests passed in 111.57 s**.

The complete tinygrad run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops. The reference example itself requested large contiguous buffers and logged two 4 MiB CMA allocation failures; Rockchip tinygrad continued to use its page-backed non-contiguous allocation policy and generated none.

---

## 2026-08-08 — Infinite FP16 multiplication on DPU

RK3588 native EW MUL returns NaN for finite values multiplied by infinity. Rockchip now rewrites `x*+inf` to `x/+0`, and `x*-inf` to `(-x)/+0`. The former is one FDIV task; the latter is a two-task `MUL → FDIV` chain. Both execute entirely on DPU, inspect no input values on the host, use no reset or CMAC path, and preserve the expected infinity sign. Multiplication by NaN remains the correct native MUL operation.

- Separate positive-zero FDIV probe: bit-exact pass.
- Separate negative-infinity `MUL → FDIV` probe: bit-exact pass.
- Direct `+inf`, `-inf`, and NaN regression: **1 passed in 2.71 s**, exactly one ioctl per expression.
- Exact upstream `TestOps.test_mul_naninf`: **1 passed in 2.85 s**.
- Complete incremental source-order group: **44 passed in 9.66 s**.
- Complete Rockchip census: **161 passed, 8 skipped, 96 subtests passed in 110.71 s**.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops.

---

## 2026-08-08 — Native ReLU, PReLU, and ABS activation group

Six exact upstream activation methods are now admitted: `test_relu`, `test_relu_exact`, `test_relu_maximum_exact`, `test_leaky_relu`, `test_abs`, and `test_abs_exact`. ReLU already lowers through the proven EW MAX/ReLU path. Leaky-ReLU recognizes `WHERE(x<0, slope*x, x)` and uses RK3588's one-task EW PReLU configuration. ABS recognizes tinygrad's signed-zero-aware sign graph and uses native EW `ALU_ALGO=5` in one task.

The PReLU and ABS register recipes were checked against `rockchip/post-518-reference`, the RK3588 TRM, and `~/npu/include/rknnops.h`. Only their named renderer configuration was ported; none of the archived reset, BS-PReLU, or stateful runtime machinery was imported.

- Individual methods: all pass in 2.65–3.10 s.
- Focused native ABS edge probe: `-1`, both signed zeros, `1`, both infinities, and both NaN signs pass; `-0` becomes `+0` exactly.
- Six-method activation group: **6 passed in 3.26 s**.
- Complete incremental source-order group: **50 passed in 10.12 s**.
- Complete Rockchip census: **167 passed, 8 skipped, 96 subtests passed in 111.38 s**.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops.

---

## 2026-08-08 — Native ReLU6, MIN, and hard-activation group

Six more exact upstream methods are admitted: `test_relu6`, `test_clip`, `test_hardtanh`, `test_hardsigmoid`,
`test_hardsigmoid_extreme`, and `test_hardswish`. The register recipes were verified against
`rockchip/post-518-reference`, `~/npu/include/rknnops.h`, and the RK3588 examples/TRM. ReLU6 uses EW ReLUX with an
FP32 compare value of 6, while ordered clamp graphs use native EW ALU MIN together with the existing native MAX path.
Hardsigmoid and hardswish are algebraically folded onto those same primitives; no CMAC, CPU numeric evaluation, or
shared tinygrad core change is involved.

The extreme hardsigmoid graph is simplified by tinygrad to `relu(x/6+0.5)-relu(x/6-0.5)`. Evaluating that expression
directly loses precision for large FP16 inputs, so Rockchip recognizes the common affine base and emits
`min(relu(x/6+0.5), 1)` instead. The positive and negative 300–400 ranges then saturate exactly without changing the
allowed FP16 tolerance.

- All six methods pass separately in 2.70–3.14 s.
- Six-method persistent-process group: **6 passed in 3.54 s**.
- Complete incremental source-order group: **56 passed in 10.49 s**.
- Complete Rockchip census: **173 passed, 8 skipped, 96 subtests passed in 112.09 s**.
- Ruff and mypy: pass. Renderer/runtime executable size: 647/165 lines.

The complete run left `CmaFree` at 6144 KiB and produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure,
or kernel oops.

---

## 2026-08-08 — Static FP16 movement and view group

Eleven exact upstream movement methods are now admitted: `test_transpose`, `test_permute`, `test_reshape`, `test_view`,
`test_flip`, `test_squeeze`, `test_unsqueeze`, `test_flatten`, `test_unflatten`, `test_detach`, and `test_expand`. This is
the view-only subset of the movement census proven on `rockchip/post-518-reference`. The `~/rk3588` examples and
`~/npu/include/rknnops.h` likewise treat tensor packing as address/stride preparation around the accelerator rather than
numeric execution.

No renderer/runtime change was required. Non-contiguous views use the existing affine or fallback gather plan, which
computes integer source indexes and copies raw `uint16` lanes into scratch. The value-preserving DPU EW stage remains the
only input-dependent numeric operation. No CMAC path, FP32 conversion, tolerance change, or shared tinygrad core edit was
added.

- Each method passed separately in a fresh pytest process: 2.58–3.11 s.
- Persistent-process movement group: **11 passed in 3.70 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**, through 131,072 elements.
- Complete Rockchip census: **184 passed, 8 skipped, 96 subtests passed in 111.49 s**.
- Ruff: pass. Renderer/runtime executable size remains 647/165 lines.

The vendor script's legacy contiguous 4 MiB allocation logged its two known CMA fallback attempts; the subsequent
tinygrad census produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left `CmaFree`
at 6144 KiB.

### Runtime cleanup and host-computation audit

The movement path was audited after admission. Runtime gathers view buffers only as raw `uint16` lanes; NumPy computes
integer source indexes and never interprets or transforms input-dependent FP16 values. Renderer vector evaluation is
restricted to parameter-free RANGE/CONST layout expressions. This is ABI/layout preparation, not a CPU numeric fallback.

The unused `reset_npu` hook was removed. It had no caller, issued unsupported action 13, swallowed reset failures, and did
not match the recovery ownership used by other tinygrad hardware runtimes. Blocking submission behavior is unchanged.
The movement group still passes **11/11 in 3.55 s** after cleanup. Ruff and mypy pass; `sz.py` reports renderer/runtime
sizes of **647/160 executable lines**, reducing the runtime by five lines while retaining comments and docstrings.

---

## 2026-08-08 — Static FP16 slicing and padded selection

Sixteen more movement methods are admitted: bounded one- and multidimensional slicing, scalar integer dimension
selection, inserted `None` axes, clipped endpoints, positive and negative strides, empty results, error cases, ellipsis,
double slicing, and pad→reshape/slice compositions. The explicit `test_slice_with_const_tensor` case remains skipped
because it requires an integer tensor input; it is not emulated on the CPU.

Partly padded slices exposed tinygrad's static `WHERE` selection tree around raw loads. The proven post-518 selection
gather was ported with support for one exact FP16 fill bit-pattern. It evaluates only parameter-free RANGE/CONST layout
predicates at compile time, records integer source offsets, and moves raw `uint16` lanes at runtime. No input-dependent
predicate or numeric value is evaluated on the host.

An initial implementation attempted selection recognition on every ADD node and made large convolution compilation
quadratic. Profiling showed the pytest process at 100% CPU with no active NPU ioctl or kernel error. Restricting the
recognizer to its actual WHERE/gated-load roots restored focused depthwise/dilated convolution methods to 2.79–3.19 s.

- Sixteen new methods passed separately in fresh pytest processes; the integer-tensor case skipped explicitly.
- Complete movement group: **27 passed, 1 skipped in 5.50 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **200 passed, 9 skipped, 96 subtests passed in 115.26 s**.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **693/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — Diagonal and rolling static indexing

The exact upstream `test_diag`, `test_diagonal`, and `test_roll` methods now join the movement census. They cover vector
diagonal construction, rectangular/batched/offset diagonal extraction, and positive, negative, zero, multi-axis, and
oversized cyclic shifts. All lower through the existing single-source static gather and raw FP16 identity path; no
renderer/runtime change or host numeric operation was required.

- Individual methods: **1 passed** each in 2.77–3.18 s.
- Complete movement group: **30 passed, 1 skipped in 6.18 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **203 passed, 9 skipped, 96 subtests passed in 115.29 s**.
- Ruff and mypy: pass. `sz.py` remains renderer/runtime **693/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — FP16 concatenation, stacking, and repetition

The multi-source partial-gather design from `rockchip/post-518-reference` is now ported to the current renderer image
format. `test_cat`, `test_multicat`, `test_stack`, `test_stack_slice`, `test_stack_max`, and `test_repeat` are admitted.
Several source buffers populate disjoint lanes of one scratch buffer, after which the ordinary DPU EW identity stage
writes the result. Runtime preparation computes static integer offsets and copies raw `uint16` lanes only; it never
decodes or evaluates input-dependent FP16 values. No CMAC path or shared tinygrad core change is involved.

Selection-tree and static-expression results are cached during lowering. This keeps multi-source recognition bounded
and avoids repeating the large-graph compile-time problem previously found while adding padded slices.

- Every method passed separately in a fresh pytest process: **2.78–6.01 s**.
- Complete concatenation group: **6 passed in 6.99 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **209 passed, 9 skipped, 96 subtests passed in 94.84 s**.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **729/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — Constant, reflect, replicate, and circular padding

Four exact upstream padding methods are now admitted: `test_pad`, `test_pad_reflect_mode`,
`test_pad_replicate_mode`, and `test_pad_circular_mode`. The implementation was checked against the proven
`rockchip/post-518-reference` milestone `04186a805`. Its required selection-gather lowering is already subsumed by the
current cached gather implementation, so this milestone adds coverage without adding renderer or runtime code.

All modes lower static padding predicates and source offsets on the host, copy only raw `uint16` lanes into scratch,
and use the DPU EW identity stage for output. No input-dependent FP16 arithmetic, CMAC path, tolerance change, or shared
tinygrad core change is involved.

- Individual methods: **1 passed** each in 2.87–6.31 s.
- Complete padding group: **4 passed in 9.71 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **213 passed, 9 skipped, 96 subtests passed in 44.62 s** under `-n12`.
- Ruff and mypy: pass. `sz.py` remains renderer/runtime **729/160 executable lines**.

The complete run produced no new RKNPU timeout, invalid IRQ, IOMMU fault, CMA failure, or kernel oops and left
`CmaFree` at 6144 KiB.

---

## 2026-08-08 — Remaining FP16 repetition layouts

The exact upstream `test_repeat_interleave` and `test_simple_repeat` methods now complete the repetition subset. No
dedicated generic implementation exists in `rockchip/post-518-reference`, `~/npu`, or `~/rk3588`; both layouts lower
directly through the current single-source raw gather and DPU EW identity path. The host computes static integer source
offsets and copies `uint16` lanes without interpreting FP16 values.

- Individual methods: **1 passed** each in 2.74–2.89 s.
- Complete concatenation/repetition group: **8 passed in 7.20 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Complete Rockchip census: **215 passed, 9 skipped, 96 subtests passed in 43.32 s** under `-n12`.
- Ruff and mypy: pass. `sz.py` remains renderer/runtime **729/160 executable lines**.

The tinygrad census produced no RKNPU timeout, invalid IRQ, IOMMU fault, reset, or kernel oops. The vendor script again
logged its two known failed contiguous 4 MiB CMA attempts, then passed every probe; `CmaFree` returned to 6144 KiB.

---

## 2026-08-08 — FP16 scalar reductions and dynamically sized one-submit PC chains

The DPU EW scalar-reduction design from `rockchip/post-518-reference` milestone `326124baf` is ported for FP16 sum,
mean, product, minimum, and maximum. Reduction inputs are gathered as raw FP16 lanes into 64-byte-spaced scratch and
combined by a balanced tree of native DPU EW ADD, MUL, or MAX stages. Mean adds a final DPU MUL by its compile-time
scale, and minimum uses the proven negate/MAX lowering. ReLU-wrapped sums and precise long multiply/add trees are also
recognized.

There is no CPU numeric fallback: the renderer evaluates only static integer layout expressions, while runtime copies
raw `uint16` lanes using integer indices. It never decodes input FP16 values or computes a reduction on the host. No
CMAC path, shared tinygrad core change, or tolerance relaxation was added; tests retain the `test_gemm_fp16` limit of
`atol=rtol=grad_atol=grad_rtol=5e-3`.

Fixed task-count caps are removed. Each realized program now computes its command and descriptor arena sizes from the
actual emitted register-command qwords and 40-byte task descriptors, adds one mapped page of command-prefetch guard,
and submits the complete PC chain with one blocking ioctl. The 16,384-element `test_sum_full` produces 16,383 tasks,
a 2,887,504-byte command allocation (including guard), and a 655,320-byte descriptor allocation, and passes with
exactly one submit. Arena growth allocates the replacement before releasing the old buffer.

- Focused reduction group after cleanup: **21 passed, 1 skipped in 10.67 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s.
- Complete Rockchip census, sequential single-process hardware ownership: **229 passed, 10 skipped, 96 subtests
  passed in 102.98 s**.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **807/164 executable lines**.

FP32, boolean, and integer reductions remain explicitly skipped because the NPU input path is FP16-only. The complete
run and focused post-cleanup run produced no RKNPU timeout, invalid IRQ, IOMMU fault, reset, or kernel oops.

---

## 2026-08-08 — Adaptive-pooling-equivalent FP16 windows

Tinygrad currently exposes no adaptive average/max pooling API, so the Rockchip census now covers the exact equivalent
divisible-window cases from `rockchip/post-518-reference`: 4×4 to 2×2 and 4×4 to global 1×1. PyTorch adaptive pooling
supplies the golden while tinygrad expresses the same operation with fixed kernel and stride pooling.

- Adaptive-equivalent census: **2 passed in 3.04 s**, sequentially on one NPU process.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.

Both cases reuse the existing raw FP16 gather plus DPU EW reduction path. No backend or shared core change, host numeric
operation, CMAC, CNA, or PPU execution was added.

---

## 2026-08-08 — FP16 polynomial and simple loss expressions

The Rockchip census now covers square, cubic, a Horner-form quadratic, mean-squared error, and mean absolute error from
the proven `rockchip/post-518-reference` group. They compose the existing FP16 ADD, MUL, ABS, and reduction paths; BCE
and NLL remain outside this group because they require transcendental or indexed semantics. FP32 is used only by the
PyTorch golden for the loss reductions and is then cast to the FP16 device contract.

- Polynomial/simple-loss census: **4 passed in 2.98 s**, sequentially on one NPU process.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.
- Rockchip execution uses only DPU EW; no backend/core change, host tensor arithmetic, CMAC, CNA, or PPU execution was
  added.

---

## 2026-08-08 — FP16 scatter with compile-time-static indices

The Rockchip census now covers functional scatter replacement on dimensions 0 and 1, scalar-source scatter, and
scatter-reduce sum, product, mean, maximum, and minimum from `rockchip/post-518-reference`. Indices are compile-time
`arange` or broadcast constants: the renderer turns them into raw gather layout, while selected FP16 values and every
reduction execute on DPU EW. External integer index buffers remain explicitly excluded because Rockchip accepts FP16
inputs only.

- Static-index scatter census: **8 passed, 1 skipped in 3.13 s**, sequentially on one NPU process.
- Covered reductions: sum, product, mean, maximum, and minimum.
- Complete Rockchip census: **243 passed, 11 skipped, 96 subtests passed in 101.99 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s after the full census.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`.

No backend/core change, host tensor arithmetic, CMAC, CNA, or PPU execution was added. The full run produced no RKNPU
timeout, invalid IRQ, IOMMU fault, reset, or kernel oops.

---

## 2026-08-08 — Terminal FP32 output for FP16 scalar reduction

The exact upstream `test_sum_dtype_arg` is now admitted using the proven `rockchip/post-518-reference` DPU recipe. The
input and complete balanced reduction tree remain FP16. Only the terminal scalar ADD sets DPU `OUT_PRECISION=5`, while
`IN_PRECISION=2` and `PROC_PRECISION=2` remain FP16. No FP32 value is fed into another NPU task.

The terminal output stage is isolated from the dynamically sized FP16 PC chain, giving two blocking submits for the
135-element case. A successful terminal task is followed by one supported action reset so its scalar FP32 BS/WDMA state
cannot leak into the next FP16 program. An explicit same-process `test_sum_dtype_arg → test_tiny_add` sequence passed.

- Focused FP32-output test: **1 passed in 2.93 s**; expected submit count is exactly 2.
- Post-cleanup FP32-output then FP16 transition: **2 passed in 3.05 s**.
- Complete Rockchip census: **244 passed, 11 skipped, 96 subtests passed in 103.21 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.45 s after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **840/173 executable lines**.

The renderer only tags the terminal UOp and emits register commands; runtime performs no host numeric conversion or
tensor arithmetic. There is no CMAC, CNA, PPU, shared core change, or tolerance relaxation. Kernel logs contain no new
RKNPU timeout, invalid IRQ, IOMMU fault, or oops.

---

## 2026-08-08 — FP16 broadcast arithmetic census

The exact upstream `test_broadcast_simple` now joins the Rockchip census. Full and partial broadcast shapes from
`test_broadcast_full` and `test_broadcast_partial` are additionally covered for every currently native FP16 arithmetic
operation: ADD, SUB, MUL, and FDIV. POW is intentionally kept separate rather than weakening those upstream methods or
silently treating an unsupported primitive as broadcast support.

Broadcast layout is handled by the existing static raw-lane gather, and arithmetic remains DPU EW. `~/npu` and
`~/rk3588` contain broadcast register/layout examples but no additional backend lowering was required here.

- Broadcast group: **3 passed, 24 subtests passed in 3.43 s**, sequentially.
- Complete Rockchip census: **247 passed, 11 skipped, 120 subtests passed in 103.51 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.55 s.
- Tolerance remains exactly `atol=5e-3, rtol=5e-3`; backend size remains renderer/runtime **840/173 executable lines**.

This milestone changes coverage only. It adds no host tensor arithmetic, CMAC, CNA, PPU, shared core change, or
tolerance relaxation.

---

## 2026-08-08 — Exact dot, batched dot, and matvec census on DPU EW

Six exact upstream methods are now admitted: `test_dot_1d`, `test_dot`, `test_broadcastdot`, `test_multidot`,
`test_matvec`, and `test_matvecmat`. K-loop contractions arrive after FP16 rewriting as local FP32 accumulator loops.
The renderer now recognizes the narrowly validated ADD-of-FP16-products form, statically substitutes only the reduction
range, and feeds the resulting product tree back through the existing generic DPU EW lowering.

This keeps execution scalable: each K term is a vector operation over all outputs, so task count depends on K rather
than `M×N×K`. `ROCKCHIP_EW_REDUCE=twoproduct` reuses the established compensated FP16 product/sum graph; a plain
balanced FP16 tree was rejected because one K65 lane missed the permitted tolerance by 0.007324. The older
`rockchip-2608` reference used scalar CMAC for part of this scope, but CMAC is intentionally not ported here.

- Individual methods: **1 passed each in 3.21–4.71 s**.
- Combined dot/matvec group: **6 passed in 9.02 s**, sequentially in one process.
- Complete Rockchip census: **253 passed, 11 skipped, 120 subtests passed in 113.20 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.46 s.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **878/173 executable lines**.

No input value is evaluated on the host: compile time only substitutes static integer K indices, runtime performs the
existing raw `uint16` gathers, and every MUL/ADD/compensation stage executes on DPU EW. There is no CMAC, CNA, PPU,
shared core change, or tolerance relaxation. Kernel logs contain no new timeout, invalid IRQ, IOMMU fault, or oops.

---

## 2026-08-08 — FP16 einsum contraction and validation census

Five exact upstream methods now join the Rockchip census: `test_einsum`, `test_einsum_trace`,
`test_einsum_shape_check`, and the two arity checks. The functional cases cover scalar identity, transposes, whole/axis
sums, diagonal and batched trace, vector/matrix/batched products, permuted results and inputs, large tensor
contractions, and a three-input bilinear expression.

All functional forms compose the existing static raw-lane gathers, scalar reduction lowering, and compensated DPU EW
dot loop from the prior milestone. No new backend implementation was required. The separate slow ellipsis stress method
contains a 13,824-term reduction and remains a dedicated profiling target rather than being silently included here.

- `test_einsum_trace`: **1 passed in 2.81 s**.
- Main `test_einsum`: **1 passed in 6.40 s**.
- Complete einsum group: **5 passed in 6.37 s**, sequentially.
- Complete Rockchip census: **258 passed, 11 skipped, 120 subtests passed in 115.89 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.54 s.
- Backend size remains renderer/runtime **878/173 executable lines**.

This milestone changes coverage only and retains the exact FP16 tolerance ceiling. It adds no CPU tensor arithmetic,
CMAC, CNA, PPU, or shared core change.

---

## 2026-08-08 — FP16 cumulative sums with bounded static selection analysis

Four exact upstream methods now join the Rockchip census: `test_small_cumsum`, `test_simple_cumsum`, `test_cumsum`,
and `test_cumsum_zero_axis`. Together they cover scalar and empty inputs, 1D lengths through 1,022, and cumulative
sums across 2D/3D axes. The older `rockchip/post-518-reference` implementation uses specialized CMAC reductions,
while `~/npu` explicitly lists CumSum as unsupported and `~/rk3588` supplies only the upstream test definitions; none
was ported because this branch remains DPU-EW-only.

The existing static selection gather already described the cumulative layouts, but interpreted the same shared UOp
DAG once per output coordinate. The 512-element case remained host-bound for more than 90 seconds. Selection analysis
now evaluates every static coordinate together with NumPy vectors and memoizes each UOp's slot, raw offset, and fill
metadata. This changes compiler metadata construction only: runtime still copies raw `uint16` lanes and every prefix
ADD executes as a DPU EW task.

- `test_simple_cumsum`: **1 passed in 11.80 s**, down from more than 90 seconds of host compilation.
- Complete cumulative-sum group: **4 passed in 13.95 s**, sequentially in one process.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.38 s.
- Complete Rockchip census: **262 passed, 11 skipped, 120 subtests passed in 120.32 s**, sequentially.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **896/173 executable lines**.

The CPU-cheat audit found no input decoding or host numeric result computation. NumPy is confined to static integer
index, source-slot, gate, and FP16-fill-bit planning at compile time. There is no CMAC, CNA, PPU, shared tinygrad core
change, or tolerance relaxation.

---

## 2026-08-08 — Ordered FP16 cumulative products on DPU EW

Four exact upstream methods now join the Rockchip census: `test_small_cumprod`, `test_simple_cumprod`, `test_cumprod`,
and `test_cumprod_zero_axis`. They cover scalar and empty inputs, 1D lengths through 1,022, and cumulative products
across 2D/3D axes. `~/npu` has no native CumProd path and `~/rk3588` only supplies the upstream tests. Older Rockchip
branches used specialized CMAC or partially validated product atoms; neither path was ported because this branch remains
DPU-EW-only.

Static selection planning now preserves independent FP16 padding values per output lane and can compose a compile-time
prefill with partial raw gathers. The cumulative product is recognized only when its leaves form the verified prefix
sequence, then rebuilt in source order as native FP16 MUL stages. Scratch destinations use ping-pong storage so a stage
never aliases a still-live input, and generic dot-product compensation is not applied to this ordered scan.

A 40-term prefix exposed eight NaNs when all dependent stages were submitted as one PC chain, while every finite result
remained within tolerance. Isolated and bounded-chain probes showed the arithmetic was correct but the terminal dependent
task lacked a proven inter-task visibility boundary. The image format now carries an explicit typed `submit_barrier`
flag, and runtime closes the current blocking PC-chain immediately before the terminal output stage. The flag replaces
an initially considered software bit tag and cannot leak into a hardware register.

- Focused cumulative-product group after cleanup: **4 passed in 12.45 s**, sequentially in one process.
- Complete Rockchip census: **266 passed, 11 skipped, 120 subtests passed in 137.45 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.53 s after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **926/176 executable lines**.

The CPU-cheat audit found no input decoding or host numeric product. NumPy evaluates only static UOp coordinates,
source slots, masks, and fill-bit placement; runtime copies raw `uint16` lanes, and every multiplication executes on DPU
EW. There is no CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 cumulative extrema values

The Rockchip census now covers the value outputs for cumulative MAX and MIN across every shape and axis used by the
four upstream methods for each operation: scalar and empty tensors, 1D lengths 10/512/1,022, 2D axes 0/1, and 3D axes
2/-1. The corresponding upstream methods also request `int32` axis indices, which remain an explicit separate output-
format group rather than being hidden or computed on the host.

`~/npu` has no native cumulative-extrema implementation and `~/rk3588` supplies the upstream definitions. Historical
Rockchip milestones `d407bb338` and `eb7c58c98` prove DPU compare/select implementations for both values and indices,
but their large specialized runtime is not ported. The current compact generic MAX path already handled all value cases
except the padded 1,022 tail. That tail wraps a MAX tree in a static mask whose individual load gates contain the inverse
condition. `_fold_masked_max` now recognizes that conjunction and safely moves the MAX identity fallback into each load,
leaving ordinary native DPU EW MAX stages.

- Cumulative-extrema value group: **8 passed, 24 subtests passed in 23.86 s**, sequentially.
- Complete Rockchip census: **274 passed, 11 skipped, 144 subtests passed in 148.30 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** in 0.37 s.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **927/176 executable lines**.

The CPU-cheat audit found no runtime input decoding, comparison, or extrema calculation. The new rule is a renderer-only
identity rewrite guarded by structural condition checks; all MAX and negate/MAX work executes on DPU EW. There is no
CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Small cumulative-extrema INT32 axis indices on DPU EW

The exact upstream `test_small_cummax` and `test_small_cummin` index outputs now join the Rockchip census. This first
index milestone deliberately covers the official length-10 cases; multidimensional axes and larger tiled INT32 outputs
remain the next group and are not claimed here.

Historical Rockchip milestones `d407bb338` and `eb7c58c98`, plus `rockchip/post-518-reference`, provide the verified
RK3588 compare/selection register sequence. `~/npu` has no cumulative-extrema implementation, while `~/rk3588`
provides the upstream tests and experimental compare examples. The current lowering gathers FP16 candidates and static
axis coordinates into 64-byte-aligned arenas, computes candidate-minus-extremum magnitude and equality masks on DPU EW,
multiplies masks by coordinates, and reduces the selected coordinates with DPU MAX. CumMin recognizes the negated
candidate form before applying the same selection graph.

The terminal coordinate-minus-one remains FP16 DPU arithmetic. Native DPU output conversion then writes INT32 values
through four-lane, 64-byte-aligned tiles. Runtime only packs and unpacks those raw lane bytes; it does not compare input
values, choose an index, or numerically convert FP16 to INT32. Stateful compare submits retain the proven standalone PC
framing, while ordinary PC chains keep their existing register template. Common command/task buffer growth and blocking
submit bookkeeping are shared between both paths.

- Focused index group: **2 passed, 2 subtests passed in 7.65 s**, sequentially.
- Complete Rockchip census after cleanup: **276 passed, 11 skipped, 146 subtests passed in 153.63 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,058/239 executable lines**.

The CPU-cheat audit found only raw `uint16` gather/layout operations and `memmove` around the aligned INT32 tiles. Every
comparison, mask, selection, extrema reduction, subtraction, and numeric conversion executes on DPU EW. There is no
CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Multidimensional cumulative-extrema axis indices

The remaining ordinary-shape cases from upstream `test_cummax` and `test_cummin` now join the Rockchip census: scalar,
length 5, both axes of `(5,6)`, and axis `2`/`-1` of `(5,6,7)`. Both zero-axis methods are also covered. Together with
the preceding length-10 milestone, the unchanged upstream `test_cummax`, `test_cummax_zero_axis`, `test_cummin`, and
`test_cummin_zero_axis` methods now pass directly.

Historical commits `d407bb338` and `eb7c58c98` establish that cumulative indices use packed reduction-axis coordinates,
not flattened source addresses: the current coordinate is `dst % window`, later candidates are invalid, and the latest
equal candidate wins. The new recognizer accepts that rule only when every candidate map is distinct and monotonic and
the current-candidate addresses form an exact permutation of the input. `~/npu` still has no cumulative-extrema path;
`~/rk3588` supplies the upstream definitions and the compare/framing examples used by the existing DPU stages.

Multidimensional Tinygrad graphs finish with a pure INT32 view kernel. RKIMAGE v22 generalizes the existing raw-lane
gather metadata to 2- or 4-byte lanes and scratch or argument destinations, allowing that final transpose without
interpreting index values. Numeric selection and FP16-to-INT32 conversion remain on DPU.

The first 210-element probe timed out at driver task counter 17 while one PC chain held 53 stateful four-lane INT32
conversion stages. The backend now bounds only this terminal stage by its encoded command bytes: a batch may occupy at
most one system page. With the current 288-byte body this computes 14 tasks per submit; there is no hardcoded task-count
cap, and command/task BO allocation remains dynamically sized. The same case then passed exactly in **2.26 s** with
**28 total submits**, including its seven reset-separated comparisons. The vendor health check passed 60/60 immediately
after the exploratory timeout and again after the full census.

- Expanded cumulative-index class: **6 passed, 20 subtests passed in 24.66 s**, sequentially.
- Four unchanged upstream CumMax/CumMin methods: **4 passed in 20.33 s**, sequentially.
- Complete Rockchip census: **280 passed, 11 skipped, 164 subtests passed in 178.53 s**, sequentially.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,075/249 executable lines**.

The CPU-cheat audit found only address-map construction and raw `uint16`/`uint32` representation movement on the host.
No runtime value is compared, selected, reduced, or numerically converted by the CPU. There is no CMAC, CNA, PPU,
shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Long cumulative-extrema indices with matrix DPU selection

The remaining length-512 and padded length-1,022 INT32 index outputs from upstream `test_simple_cummax` and
`test_simple_cummin` now run on Rockchip. The permanent Rockchip coverage splits the four realizations into separate
methods so each hardware case stays below 30 seconds; the unchanged upstream methods pass as well.

The earlier per-candidate selection graph repeated comparison setup hundreds of times. Candidate values and axis
coordinates are now gathered into aligned matrices, then DPU EW performs bulk subtraction, absolute value, equality,
mask multiplication, and a pairwise MAX tree. CumMin first negates its candidate matrix on DPU. Length 1,022 uses the
exact bounded-loop form emitted by Tinygrad: the recognizer verifies its candidate permutation, output-before-reduction
prefix gate, INT32 MAX dependence, and two partial-extremum loads before building the same matrix selection image.
Candidate rows use compact affine gather metadata rather than storing a multi-megabyte repeated offset table.

An exploratory length-512 run timed out at task counter 9 when a dependent equality-to-MUL transition shared one PC
chain. The vendor elementwise health check immediately passed 60/60. Typed submit barriers now separate the dependent
bulk phases; all subsequent focused and full-suite runs completed without a timeout. Scratch matrix padding is cleared
as raw storage before gather placement so unused aligned lanes remain neutral.

- CumMax 512: **1 passed in 16.61 s**; CumMax 1,022: **1 passed in 18.40 s**.
- CumMin 512: **1 passed in 19.63 s**; CumMin 1,022: **1 passed in 17.03 s**.
- Complete cumulative-index class: **10 passed, 24 subtests passed in 62.40 s**, sequentially.
- Unchanged upstream `test_simple_cummax`: **1 passed in 24.30 s**; `test_simple_cummin`: **1 passed in 37.19 s**.
  The latter combines four realizations; its slowest individual realization was 13.81 s.
- Complete Rockchip census: **284 passed, 11 skipped, 168 subtests passed in 191.13 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the full census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,139/253 executable lines**.

The CPU-cheat audit found only static address-map planning, scratch padding initialization, and raw `uint16`/`uint32`
lane movement on the host. Every comparison, mask, candidate negation, selection, reduction, subtraction, and numeric
FP16-to-INT32 conversion executes on DPU EW. There is no CMAC, CNA, PPU, shared tinygrad core change, or tolerance
relaxation.

---

## 2026-08-08 — FP16 ArgMax/ArgMin first-tie selection on DPU EW

The first general selected-index milestone covers the exact first-occurrence rule for FP16 ArgMax and ArgMin, including
equal pairs and repeated extrema after a distinct leading value. The unchanged upstream methods begin with `int32`
literals and later add boolean/int32 conversion regressions, so those input-format cases remain a separate group rather
than being presented as native FP16 coverage.

Historical commit `7bea014d35` proves the RK3588 DPU compare/select design, but its large task runtime is not ported.
`~/npu` documents RKNN ArgMax/ArgMin as CPU operators, and `~/rk3588` contains the upstream tests plus RKNN runtime
operator strings rather than a lower-level register implementation. The current compact renderer recognizes only the
fused graph containing one FP16 MAX tree, one equality/inversion/cast per candidate, the exact descending coordinate
weights, one INT32 MAX tree, and the final `window-selected` transform. It also verifies that candidate gathers form an
exact permutation of the source.

Candidate and extreme-value matrices reuse the cumulative-index image. Descending coordinates make DPU MAX retain the
earliest equal candidate; a final DPU subtraction restores its zero-based coordinate. ArgMin negates both the candidate
matrix and the independent matrix used to construct `MAX(-x)`, then follows the same equality path.

- Focused FP16 first-tie group: **2 passed, 4 subtests passed in 4.18 s**, sequentially.
- Complete Rockchip census: **286 passed, 11 skipped, 172 subtests passed in 192.93 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,210/253 executable lines**.

The runtime is unchanged. Host work is limited to static source-address and coordinate-bit image construction plus the
existing raw lane movement. Candidate negation, extrema reduction, equality, mask weighting, first-tie selection,
subtraction, and FP16-to-INT32 output conversion all execute on DPU EW. There is no CPU numeric fallback, CMAC, CNA,
PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 axis ArgMax/ArgMin with materialized extrema

FP16 ArgMax and ArgMin now cover both axes of `(10,20)` plus axis-1 `keepdim=True`, matching the three axis forms in
the upstream methods. Tinygrad schedules these as two kernels: ordinary DPU EW first materializes `MAX(x)` or
`MAX(-x)`, then the new selected-index image compares every candidate against that saved extreme.

The fused and split recognizers share one exact first-tie validator. The split form additionally requires exactly two
FP16 source buffers, an extreme buffer whose static view is a permutation of the output, a candidate buffer whose views
jointly permute the full input, one sign convention across all candidates, and the same descending-coordinate INT32 MAX
graph. It therefore reuses the first-tie matrix emitter without accepting cumulative latest-tie or unrelated comparison
graphs. The saved extreme remains device data and is never copied to or interpreted by the CPU.

- Expanded FP16 ArgMax/ArgMin group: **4 passed, 10 subtests passed in 6.80 s**, sequentially.
- Complete Rockchip census: **288 passed, 11 skipped, 178 subtests passed in 194.43 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,256/253 executable lines**.

The runtime remains unchanged, and its no-CPU-cheat audit contains no host extrema, equality, or selected-index
arithmetic. The renderer constructs only static address and coordinate metadata; all numeric work and conversion remain
DPU EW. There is no CMAC, CNA, PPU, shared tinygrad core change, or tolerance relaxation. The padded global 200-element
bounded-loop graph and non-FP16 input conversions remain explicit next groups.

---

## 2026-08-08 — Global FP16 ArgMax/ArgMin bounded-loop selection

Global FP16 ArgMax and ArgMin over the upstream `(10,20)` shape now join the Rockchip census. Additional deterministic
cases place equal extrema at flat indices 37 and 99 and verify that the loop form also returns index 37.

For 200 candidates Tinygrad replaces the unrolled graph with two ordered register loops. The first initializes a local
FP16 accumulator to negative infinity and computes `MAX(x)` or `MAX(-x)`. The second initializes an INT32 accumulator
to `INT_MIN`, compares each candidate with the completed extreme, weights equality by `window-coordinate`, and reduces
with INT32 MAX. The final store computes `window-selected`.

The new matcher accepts only that exact two-loop structure: one global FP16 input of the loop extent, matching direct
candidate permutations in both loops, one sign convention, the expected local accumulator initializers and updates,
descending coordinates, and the final inverse transform. It then emits the already verified matrix DPU image. No loop
accumulator is read by the host.

- Expanded FP16 ArgMax/ArgMin class: **6 passed, 10 subtests passed in 11.22 s**, sequentially.
- Complete Rockchip census: **290 passed, 11 skipped, 178 subtests passed in 200.45 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,320/253 executable lines**.

The CPU-cheat audit found only compile-time UOp structure, source-address, and coordinate evaluation. Runtime remains
raw lane movement plus DPU submission and contains no host extrema or selected-index arithmetic. Every negation, MAX,
comparison, mask, coordinate selection, subtraction, and FP16-to-INT32 conversion executes on DPU EW. There is no CMAC,
CNA, PPU, shared tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — Stable FP16 sort values on DPU EW

The FP16 value half of upstream `test_sort` now runs for every nontrivial `(8,8,6)` axis and direction, along with its
empty/singleton and repeated-value cases. A dedicated infinity regression verifies both directions over `[-inf, 2]`.
Stable INT32 indices remain the next milestone: the unchanged upstream method passes its value realizations, then rejects
the first `(8,8,6)` index-count graph rather than falling back to the CPU.

Historical milestone `43f113d90` supplies the native bitonic graph classification, but its MIN reconstruction uses
negation and an arithmetic mask. That formulation produces `0*inf = NaN` for padded sort wires. The current lowering
instead gathers each statically paired wire, computes complete contiguous native DPU MAX and MIN vectors, and merges them
with the compile-time bitonic direction map. `~/npu/include/old/rknn_ops.md` explicitly lists `aten::sort` as unsupported;
`~/rk3588` contains the upstream sort/argsort tests but no lower-level sort implementation.

RKIMAGE v23 gives raw gathers explicit source and destination buffer kinds and separates pre-DPU from post-DPU gathers.
The runtime invalidates cacheable DPU results, then copies only their `uint16` lane representations according to the
compiled direction map. It never compares, ranks, sorts, or numerically transforms tensor data. Full-vector DPU writes
also avoid the 64-byte overwrite that made the exploratory short-run output path corrupt adjacent lanes.

- All six `(8,8,6)` axis/direction value sorts are bit-exact; each uses **9 ioctl submits**.
- Focused value-sort group: **4 passed, 14 subtests passed in 5.03 s**, sequentially.
- Complete Rockchip census: **294 passed, 11 skipped, 192 subtests passed in 213.92 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,383/264 executable lines**.

The CPU-cheat audit found only compile-time condition/address evaluation and raw `uint16` lane movement. MAX and MIN are
native DPU EW operations, and the host never inspects FP16 values. There is no CMAC, CNA, PPU, shared tinygrad core
change, FP32 input conversion, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — Stable FP16 sort and argsort indices on DPU EW

Exact stable INT32 indices now accompany the FP16 value sort for all six `(8,8,6)` axis/direction combinations and the
two repeated-value directions. The unchanged upstream `test_argsort` passes. Upstream `test_sort` passes every generated
FP16 value and index realization, then reaches its Python-integer repeated-value fixture; that fixture creates an external
INT32 input and remains outside the RK3588 FP16-input contract. The equivalent repeated FP16 fixture is permanent and
passes in both directions.

Historical commit `43f113d90` identifies occurrence counting and final value/count matching as the two stable-index graph
families. The current Tinygrad scheduler unrolls both reductions, so the new matchers validate the present ADD trees,
prefix gates, value/count equality pairs, source maps, and coordinate weights directly. `~/npu` still marks sort
unsupported and `~/rk3588` has tests but no lower-level implementation.

For each occurrence-count kernel, candidate/current FP16 matrices are subtracted, passed through native ABS and the
verified positive-mask comparison, gated by compile-time prefix masks, and reduced with DPU ADD. Final selection compares
both the original/sorted values and their stable occurrence counts, multiplies the two DPU equality masks by original
coordinates, and reduces those coordinates on DPU.

The count tensors are device-produced INT32, not external user inputs. RKIMAGE v24 adds a typed internal INT32-input
conversion stage: runtime packs raw four-lane atoms, DPU converts them to FP16, and runtime unpacks the raw FP16 lanes.
The inverse FP16-to-INT32 path already used by cumulative/arg-extrema indices shares the same conversion helper. Host code
does not interpret either representation numerically.

- Six `(8,8,6)` stable-index cases: **11.10–14.53 s each**, **121 submits** for axis `-1` and **147 submits** for axes
  `0`/`1`; all outputs exact.
- Repeated FP16 stable indices: **4.24 s ascending / 3.77 s descending**, **64 submits** each; outputs exact.
- Focused index group after cleanup: **8 passed, 6 subtests passed in 90.30 s**, sequentially.
- Unchanged upstream `test_argsort`: **1 passed in 17.10 s**, sequentially.
- Complete Rockchip census after cleanup: **302 passed, 11 skipped, 198 subtests passed in 287.74 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,593/269 executable lines**.

The CPU-cheat audit found only compile-time graph/address/weight validation and raw `uint16`/`uint32` atom movement.
All value equality, occurrence-count equality, mask multiplication, count/coordinate reduction, and numeric conversion
execute on DPU EW. There is no host sort or argsort, CMAC, CNA, PPU, shared tinygrad core change, external INT32-input
support, or tolerance relaxation.

---

## 2026-08-08 — FP16 TopK composition

FP16 TopK values and stable INT32 indices now join the Rockchip census: one-dimensional largest-three, padded five-lane
axis-0 largest-four, padded five-lane axis-1 smallest-four, and repeated-value largest/smallest cases. TopK itself needs
no new numeric primitive; it composes the two native sort milestones with static shrinking.

Historical milestone `5f48c8311` confirms that non-power-of-two TopK must preserve `-inf`/`+inf` padding without an
arithmetic `0*inf` wire blend. The current native MIN/MAX plus raw post-gather sort already provides that behavior.
`~/npu/include/old/rknn_ops.md` lists both TopK and `aten::topk` as unsupported, while `~/rk3588` contains only the
upstream tests/runtime strings rather than a register implementation.

One scheduler variant materializes stable indices at the full sorted shape, then emits a separate INT32 shrink. The raw
INT32 layout recognizer now accepts an injective in-bounds subset as well as a same-size permutation. Runtime copies only
the selected `uint32` representations; it does not inspect or compute indices.

- Generated FP16 value realizations: **0.30–0.43 s**, **9 submits** each.
- Generated FP16 index realizations: **2.24–3.26 s**, **41–50 submits** each; outputs exact.
- Repeated FP16 TopK: **3.57 s largest / 3.27 s smallest**, **55 submits** each; exact indices `[0,1,3]` and `[2,4,6]`.
- Focused TopK group: **4 passed, 2 subtests passed in 20.63 s**, sequentially.
- Complete Rockchip census: **306 passed, 11 skipped, 200 subtests passed in 305.76 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,593/269 executable lines**.

The unchanged upstream `test_topk` passes all generated FP16 cases and then reaches its repeated Python-integer fixture,
which is external INT32 input and remains outside the NPU contract. Its equivalent FP16 repeated fixture is covered
above. The CPU-cheat audit found only static address-map construction and raw lane movement; all sorting, stable counting,
selection, and numeric conversion remain DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 elementwise maximum and minimum

The FP16 portions of upstream `test_maximum` and `test_minimum` now join the Rockchip census: two `(45,65)` inputs,
scalars, scalar broadcast, vectors, and infinity/NaN/signed-zero inputs. External integer, boolean, and mixed-dtype
inputs remain outside the RK3588 FP16-input contract.

`~/rk3588/examples/elementwise.py` and `~/rk3588/test/test_maximum.py` provide the proven binary DPU MAX reference.
`~/npu/include/rknnops.h` names binary MAX/MIN algorithms, but the native EW ALU-MIN configuration returns NaN for
infinite operands. Tinygrad canonicalizes minimum as `-max(-x,-y)`; the Rockchip rewrite recognizes that graph and emits
`0-lhs`, `0-rhs`, DPU MAX, then `0-max`. SUB is required because RK3588 EW MUL with infinity returns NaN. All four
numeric stages execute on DPU EW.

The native-min tag is also distinguished by the specialized stable-sort matcher, preserving its existing direct
MAX/MIN compare/swap image. Focused extrema, sort-value, sort-index, and TopK regression groups all pass.

- Focused extrema group: **2 passed in 3.58 s**, sequentially.
- Extrema plus sort-value regression: **6 passed, 14 subtests passed in 5.83 s**, sequentially.
- Sort-index plus TopK regression: **12 passed, 8 subtests passed in 107.69 s**, sequentially.
- Complete Rockchip census: **308 passed, 11 skipped, 200 subtests passed in 305.86 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,617/269 executable lines**.

The CPU-cheat audit found no runtime or host numeric implementation in this milestone: only compile-time UOp recognition
and DPU command construction changed. The host does not inspect, compare, negate, or select tensor values. There is no
CMAC, CNA, PPU, tinygrad core change, external non-FP16 input support, or tolerance relaxation beyond the existing
FP16 contract.

---

## 2026-08-08 — FP16 sign and softsign

Upstream `test_sign`, `test_sign_exact`, `test_softsign`, and `test_softsign_exact` now join the Rockchip census. A
permanent nonfinite sign regression also verifies `-inf`, `+inf`, both signed zeros, and NaN exactly.

Historical milestone `e5fdcad64` supplies the DPU comparison-mask construction for sign. The current renderer accepts
only Tinygrad's exact `WHERE(x!=0, WHERE(x<0, -1, 1), 0)` graph, computes positive and negative masks on DPU, and
subtracts them on DPU. The four stages use four submits and return `-1`, `0`, or `1` without host classification.

Softsign already composes the current native ABS, ADD, and FDIV stages as `x/(1+abs(x))`; historical milestone
`506ffb537` and `~/npu/ops_ref/main.c` provide matching references. No extra special-case lowering was needed.
Historical exact-copysign milestone `751a108e2` extracts sign bits with NumPy in the runtime, so it was deliberately not
ported under the no-CPU-cheat rule; copysign remains a separate future problem.

- Focused sign/softsign group: **5 passed in 4.83 s**, sequentially.
- Complete Rockchip census: **313 passed, 11 skipped, 200 subtests passed in 308.65 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,644/269 executable lines**.

The CPU-cheat audit found only static UOp validation and DPU command construction. Runtime code is unchanged and never
reads or classifies tensor values for sign or softsign. Every negation, comparison, mask, subtraction, absolute value,
addition, and division executes on DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or tolerance relaxation.

---

## 2026-08-08 — FP16 WHERE and masked fill

Nine FP16 selection cases now join the Rockchip census: tensor/scalar/broadcast arms, CMPNE, boolean AND/OR composition,
nested WHERE, finite masked fill, unchanged upstream `test_masked_fill`, and unchanged upstream `test_inf_where`.
The integer-output and external-boolean-input portions of upstream `test_where` remain outside the FP16-input contract.

Reference milestone `fe0b2e114` supplies the proven RK3588 positive-mask construction. The current port keeps the compact
renderer/image architecture: CMPLT and CMPNE become standalone DPU positive-mask stages, AND/OR compose exact FP16 0/1
masks with DPU MUL/MAX, and finite arms are selected with tagged mask MUL and ADD. No old runtime graph execution or
per-operation allocation machinery was imported.

Nonfinite arms avoid `0*inf`. Threshold forms such as `(x<0).where(x, 1)` use native DPU MIN/MAX composition, while
general infinity masked fills use a finite-numerator FDIV correction that is zero on the finite arm and signed infinity
on the selected arm. Stateful FDIV now uses the same RDMA mode and output scale as the already-proven ordinary FDIV.

Dedicated ABS and sign rewrite prepasses preserve their native algorithms before general WHERE rewriting. This prevents
the selector from turning `abs(inf)` into `inf*sign(inf)` and preserves exact NaN sign output behavior.

- Focused WHERE/masked-fill group: **9 passed in 6.28 s**, sequentially.
- Sign, exact-ABS, and WHERE precedence regression: **15 passed in 8.51 s**, sequentially.
- Complete Rockchip census: **322 passed, 11 skipped, 200 subtests passed in 313.77 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines**.

The CPU-cheat audit found only compile-time UOp/constant inspection and DPU command construction. Runtime is unchanged;
it never reads predicates or selects tensor values. Every comparison, mask composition, selection multiply/add,
threshold MIN/MAX, and infinity correction executes on DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or
tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — finite FP16 copysign composition

The unchanged upstream `test_copysign` now joins the Rockchip census. Its finite FP16 cases need no new backend
primitive: Tinygrad composes the existing native ABS, DPU sign-mask construction, and DPU MUL path.

Historical milestone `751a108e2` was inspected but deliberately not ported: it reads tensor sign bits with NumPy in the
runtime, which violates the no-CPU-cheat rule. Exact signed-zero and nonfinite copysign remain outside this milestone.
The current pure-DPU composition loses the negative sign of a zero magnitude, and RK3588 EW MUL returns NaN when one
operand is infinity, so this milestone makes only the finite tolerance-based claim exercised by upstream
`test_copysign`.

- Focused sign/softsign/copysign group: **6 passed in 6.46 s**, sequentially.
- Complete Rockchip census: **323 passed, 11 skipped, 200 subtests passed in 314.47 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines**.

The CPU-cheat audit found no backend code change for copysign and no host value inspection. Absolute value, sign-mask
construction, and multiplication execute entirely on DPU EW. There is no CMAC, CNA, PPU, tinygrad core change, or
tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — FP16 triangular layouts

The complete numeric FP16 portions of upstream `test_tril` and `test_triu` now join the Rockchip census: square and
rectangular matrices, positive and negative diagonal offsets beyond both matrix bounds, batched inputs, and an empty
matrix dimension. The final external-boolean-input fixture in each upstream method remains outside the FP16-input
contract.

`~/npu/include/old/rknn_ops.md` marks the model-level Trilu operation unsupported, and neither the historical Rockchip
branches nor `~/rk3588` contain a dedicated register implementation. None is needed here: Tinygrad lowers each static
triangular mask to the backend's existing raw gather/fill representation. The runtime moves FP16 lane representations
according to compile-time offsets and fills masked lanes with the FP16 zero bit pattern; it performs no tensor-value
comparison or arithmetic.

- Focused triangular group: **2 passed, 28 subtests passed in 3.40 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 228 subtests passed in 314.50 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines** (unchanged).

The CPU-cheat audit found no backend code change and no host numeric implementation. Static raw-lane movement and zero
initialization are layout preparation, not Trilu arithmetic. There is no CMAC, CNA, PPU, tinygrad core change, or
tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — FP16 padded and reordered arithmetic composition

Unchanged upstream `test_padding_add` and `test_topo_sort` now join the Rockchip census. The first adds a `(60,60)`
tensor into a statically zero-padded `(64,64)` layout; the second verifies the shared `(x+x)*x` graph remains correctly
ordered. Each realization executes one ioctl submit on DPU EW.

Historical milestone `54f88978a` solved normal-FP32 padding-add through a host elementwise boundary and was deliberately
not ported. Under the current FP16 contract, the existing gather path prepares only the static raw-lane padding layout,
then DPU ADD performs every numeric addition. Topological ordering needs no dedicated Rockchip primitive and composes
the already-proven DPU ADD and MUL stages in one PC chain. Neither `~/npu` nor `~/rk3588` contains a more specific
register implementation for this graph composition.

- Unchanged upstream `test_padding_add`: **1 passed in 2.95 s**, one submit/one task.
- Unchanged upstream `test_topo_sort`: **1 passed in 3.28 s**, one submit/45 tasks for the non-scalar realization.
- Focused padding and polynomial groups: **10 passed in 5.89 s**, sequentially.
- Complete Rockchip census: **327 passed, 11 skipped, 228 subtests passed in 315.28 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines** (unchanged).

The CPU-cheat audit found no backend code change and no host tensor arithmetic. Host work is limited to compile-time
index analysis and raw FP16 lane placement for padding; ADD and MUL execute on DPU EW. There is no LUT, CMAC, CNA, PPU,
tinygrad core change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — small-axis FP16 variance composition

The unchanged upstream `test_var_one_in_axis` now joins the Rockchip census. It covers size-one reduction axes,
correction values 0, 1, and 5, invalid degrees of freedom, scalar output, and reductions across axes `(0,3)` and
`(0,4)`. Nonconstant `(0,4)` cases execute the existing DPU EW mean, centered-square, reduction, and scale composition
in two ioctls; invalid-degree and size-one cases legitimately simplify to constant NaN or zero.

Historical milestone `0d5561074` passed the larger normal-FP32 variance groups by calling `np.var` in the runtime and
was deliberately not ported. `~/npu` documents model-level MeanVarianceNormalization but provides no applicable DPU
register implementation. The current branch contains no variance-specific runtime path or host tensor arithmetic.
Large `(15,25,35)` keepdim variance remains structurally unsupported and is not claimed by this milestone.

- Unchanged upstream `test_var_one_in_axis`: **1 passed in 2.95 s**, sequentially.
- Nonconstant `(0,4)` probes: **2 submits**, **64–107 DPU EW tasks**; outputs finite and correct.
- Complete Rockchip census: **328 passed, 11 skipped, 228 subtests passed in 314.70 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,698/269 executable lines** (unchanged).

The CPU-cheat audit found no backend code change and no NumPy variance implementation. Mean, subtraction, squaring,
reduction, and scaling execute on DPU EW whenever the result is not compile-time constant. There is no LUT, CMAC, CNA,
PPU, tinygrad core change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — full scalar FP16 variance on DPU EW

The unchanged upstream `test_var` method now joins the Rockchip census. Its three `(15,25,35)` cases exercise
correction values 1, 0, and 5. Tinygrad schedules each case as a scalar mean followed by
`SUM((x-mean)^2) * scale`; the renderer now recognizes that exact centered-square accumulator graph.

The native lowering gathers the input as aligned raw FP16 lanes, subtracts the independently computed scalar mean,
squares every delta, reduces the squares with a balanced DPU ADD tree, and applies Tinygrad's correction scale. Each
case uses two PC-chain ioctls and 52,500 DPU tasks: 13,125 for the mean and 39,375 for the centered-square pass. No LUT
or CMAC is involved.

Historical milestone `0d5561074` was inspected and rejected because it calls `np.var` in the runtime. Later
`rockchip-upstream-research` variance work records rejected CMAC/selector experiments rather than a usable DPU EW
implementation. `~/npu` documents MeanVarianceNormalization only as a CPU operator, and `~/rk3588` contains no native
variance register example.

The upstream strict-tolerance run completed all three NPU executions; correction 5 differed by `0.001953`, just beyond
its generic `1e-3` relative tolerance. The unchanged method passes under the Rockchip census's established FP16
`5e-3/5e-3` tolerance, which is the same limit used for `test_gemm_fp16`.

- Unchanged upstream `test_var`: **1 passed in 10.31 s**, sequentially.
- Per correction: **2 submits**, **52,500 DPU tasks**.
- Wall decomposition per correction: **2.43–2.84 s realization**, of which only **0.022 s** is inside submit ioctls;
  graph construction is **0.002–0.005 s** and copyout is **0.001–0.003 s**.
- Reduction regression: **24 passed, 1 skipped in 19.29 s**; post-cleanup reduction/dot regression:
  **30 passed, 1 skipped in 26.17 s**.
- Complete Rockchip census after cleanup: **329 passed, 11 skipped, 228 subtests passed in 331.20 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,766/269 executable lines**.

The CPU-cheat audit found only compile-time UOp matching/index evaluation and raw FP16 lane gathering. Runtime performs
no variance, subtraction, multiplication, addition, scaling, or tensor-value inspection on the CPU. The arithmetic is
entirely DPU EW. The cleanup shares cast/local-accumulator parsing across dot, scalar, and centered-square reducers,
following the small-helper style used by the other Tinygrad backends. There is no LUT, CMAC, CNA, PPU, tinygrad core
change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-08 — row-wise FP16 variance on DPU EW

The unchanged upstream `test_var_axis`, `test_var_zero_in_axis`, and `test_var_keepdim` methods now join the Rockchip
census. Together with the preceding scalar and one-axis milestones, all five forward-variance methods in `TestOps`
are covered under the FP16 contract.

The `(15,25,35)` axis cases reduce `K=15`, `K=35`, or `K=875`. Tinygrad already unrolls the two short reductions into
ordinary DPU EW graphs. For the `15` independent rows at `K=875`, the scalar reducer is generalized to pack each
reduction candidate as one aligned 15-lane vector. One DPU task therefore advances all rows together. The centered-
square pass uses the same vector layout for SUB, MUL, balanced ADD, and correction scaling.

The large row-wise case compiles to 875 mean tasks plus 2,625 centered-square tasks, rather than 52,500 scalar tasks.
Its 3,500 tasks remain in two dynamically sized PC-chain submits. The shared `_spaced_reduction_gathers` helper keeps
the scalar and row-vector layouts on the same compile-time raw-lane path.

- Unchanged upstream `test_var_axis`: **1 passed in 3.92 s**, sequentially.
- Axis, zero-axis, and keepdim variance: **3 passed in 6.74 s**, sequentially.
- Complete reduction class after cleanup: **27 passed, 1 skipped in 22.96 s**.
- Per nonconstant case: **2 submits**; task counts are **147** (`axis=0`), **227** (`axis=2`), and **3,500**
  (`axis=(1,2)`).
- Per-case realization wall time is **0.060–0.427 s** after graph construction; submit ioctl time is
  **0.0004–0.0017 s**. Maximum absolute error across the profiled axis cases is **0.00390625**.
- Complete Rockchip census: **332 passed, 11 skipped, 228 subtests passed in 333.74 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,784/269 executable lines**.

The CPU-cheat audit found no runtime change and no host tensor arithmetic. Compile-time index evaluation constructs a
candidate-major raw FP16 lane layout; mean, subtraction, squaring, reduction, and scaling all execute on DPU EW. The
implementation follows the existing backend pattern of a strict graph matcher plus a compact immutable image and a
shared layout helper. There is no LUT, CMAC, CNA, PPU, tinygrad core change, or tolerance relaxation beyond the
existing FP16 contract.

---

## 2026-08-08 — canonical FP16 cumulative extrema census

All eight unchanged upstream cumulative-extrema method names now join the Rockchip census: small, long, general-axis,
and zero-axis `cummax` and `cummin`. Their FP16 values and exact INT32 first-occurrence indices were already proven by
separate custom wrapper classes. This milestone replaces those 18 duplicate wrapper items with the canonical eight
methods, preserving every shape, axis, error, value, and index case while removing 29 net test lines.

Historical commits `418605353`, `8c811ede9`, and `305f7f793` provide the native DPU value, small-index, and long-index
implementations. `~/rk3588/test/test_ops.py` contains the same upstream cases and no separate register-level primitive;
the implementation remains the current DPU equality-mask, weighted-coordinate, reduction, and INT32 conversion path.

The initial canonical `test_simple_cummin` took 34.28 seconds. Profiling showed only about 24 ms in submit ioctls; the
512-index realization spent 6.7 seconds in generic linearization, 5.2 seconds rendering two large unrolled graphs, and
2.3 seconds in raw-lane preparation and command construction. `NOOPT` was rejected because it exposes a different
unsupported value-loop graph. The Rockchip test wrapper instead realizes the 512 value/index pair from one shared
graph and keeps the proven separate 1022 schedules. This preserves exact results and lowers the method to 28.61
seconds without a backend, runtime, or Tinygrad core change.

- Canonical cumulative-extrema class: **8 passed in 60.52 s**, sequentially.
- Slowest methods: `test_simple_cummin` **28.61 s**, `test_simple_cummax` **21.45 s**.
- Complete Rockchip census: **322 passed, 11 skipped, 180 subtests passed in 326.60 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,784/269 executable lines** (unchanged).

The CPU-cheat audit found no backend or runtime diff. NumPy and Torch appear only in the test oracle; NPU values and
indices still come from DPU EW and the existing raw-lane layout machinery. There is no LUT, CMAC, CNA, PPU, tinygrad
core change, or tolerance relaxation beyond the existing FP16 contract.

---

## 2026-08-09 — native FP16 IEEE classification and typed bool output

The unchanged upstream `test_isnan`, `test_isinf`, and `test_isfinite` methods now join the Rockchip census. The
`test_isinf` method also covers positive-only and negative-only detection. A strict graph recognizer accepts only
Tinygrad's canonical `x != x`, equality-to-`±inf`, their OR, and the finite complement over one FP16 input.

Classification uses the DPU positive-mask primitive around the largest finite half value. Positive and negative
overflow masks identify `+inf` and `-inf`; their intersection identifies NaN because the proven RK3588 comparison
pipeline marks NaN in both directions. DPU MAX, MUL, and SUB then form the requested 0/1 predicate. The final FP16 mask
is converted to INT32 by the existing typed DPU conversion. Runtime packs the public one-byte bool ABI by copying only
the low byte of each little-endian INT32 lane; it performs no comparison, cast, classification, or value-dependent
branch on the CPU.

Historical branch `rockchip-2608` milestone `091df1bd8` supplied the proven mask formula, but its runtime used NumPy
`!= 0` to produce bool values and was therefore not ported. `~/npu/include/old/rknn_ops.md` lists model-level IsInf and
IsNaN as unsupported, while `~/rk3588` provides no separate native classification example. The current implementation
reuses only the already-proven DPU comparison and typed-conversion register paths.

- Each seven-element predicate probe: **6 submits**, **7–9 DPU tasks**, and **0.55–0.65 s** warm-process wall time.
- Focused canonical classification class: **3 passed in 5.67 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 180 subtests passed in 337.66 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Ruff and mypy: pass. `sz.py`: renderer/runtime **1,846/274 executable lines**.

The CPU-cheat audit found only raw memory movement for the public bool representation. All classification and FP16-to-
integer conversion occur on DPU EW. There is no LUT, CMAC, CNA, PPU, Tinygrad core change, or tolerance relaxation.

---

## 2026-08-09 — conservative Rockchip renderer matcher cleanup

The DPU EW renderer now parses the single typed output store once, shares one `RKLoopReduction` descriptor across dot,
scalar, and centered-square reductions, and shares balanced reduction/layout helpers. Fused and separately materialized
ArgMax/ArgMin graphs now enter one `_lower_unrolled_arg_extrema` matcher and one common first-tie/gather/image tail.
Sort parsing also reuses binary-tree flattening, equality-pair splitting, and parameter-load grouping.

The slowest method remains `TestRockchipCumulativeExtremaOps.test_simple_cummin`. An exact test-body profile measured
27.37 s: 0.010 s input creation, 0.0004 s Torch reference, 0.028 s graph construction, 27.32 s realization, 0.0046 s
copyout, and 0.0050 s assertions. Within realization, seven renderer calls consumed 8.41 s and nine program calls
consumed 8.43 s. The 73 submit ioctls themselves consumed only 0.037 s, while 73 required NPU resets consumed 7.77 s;
the method issued 73 submits and 4,225 DPU tasks. This cleanup deliberately does not alter reset or submission semantics.

An attempted shared equality-mask emitter was rejected: `sz.py` increased by one executable line and the abstraction
hid operation-specific DPU barrier placement. Keeping the explicit occurrence/cumulative/sort emission is smaller and
clearer. The accepted cleanup reduces the renderer from **1,846 to 1,753 executable lines** (93 lines) without removing
features.

- Focused merged ArgMax/ArgMin coverage: **6 passed, 10 subtests passed in 10.99 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 180 subtests passed in 321.90 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,753/274 executable lines**.

The CPU-cheat audit found no runtime or test change in this milestone. All new helpers operate only on compile-time UOp
structure, static index maps, and immutable command-image metadata; they do not inspect or calculate tensor values.
There is no LUT, CMAC, CNA, PPU, Tinygrad core change, or tolerance relaxation.

---

## 2026-08-09 — shared equality masks and striped gather matrices

Occurrence counting, sort-index selection, and cumulative-index selection now share `_ew_eq_mask` for their FP16
`SUB -> ABS -> nonzero MAX -> 1-mask` sequence. Its two explicit barrier bits preserve the previously tested boundaries:
ordinary value equality splits before ABS, while converted sort-count equality splits before SUB. Candidate, repeated-
current, constant-mask, and weight rows now share `_stripe_layout` and `_stripe_gathers` rather than rebuilding aligned
matrix gathers in each lowering path.

The occurrence and sort parsers intentionally remain separate. Occurrence validates optional static prefix gates, while
sort validates weighted conjunctions of FP16 value equality and INT32 occurrence-count equality. A shared matcher would
hide these distinct IR contracts without reducing executable lines. `_lower_sort_compare` is also unchanged because its
static WHERE-to-MAX/MIN algorithm does not use equality masks.

- Focused arg/sort/cumulative regression: **26 passed, 30 subtests passed in 154.22 s**, sequentially.
- Complete Rockchip census: **325 passed, 11 skipped, 180 subtests passed in 321.03 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,744/274 executable lines**.

The cleanup removes nine more executable renderer lines and changes no test, runtime, command encoding, tolerance, or
Tinygrad core file. Its gathers contain only compile-time offsets/constants, and all equality arithmetic remains DPU EW;
there is no host tensor-value computation, LUT, CMAC, CNA, or PPU fallback.

---

## 2026-08-09 — IEEE-correct FP16 comparison outputs on DPU EW

The FP16 portions of all six comparison predicates now join the Rockchip census: equal, not-equal, less, greater,
less-or-equal, and greater-or-equal. Coverage includes equal shapes, both broadcast directions, scalar constants in both
operand positions, signed zero, finite values, both infinities, and every pair involving NaN. Integer and boolean input
comparisons remain outside the NPU's FP16 external-input contract.

Each operand is classified on-device into NaN, positive infinity, negative infinity, and finite masks using the proven
DPU positive-mask primitive. Ordered comparisons combine a finite subtraction mask with explicit infinity ordering and
an unordered-NaN validity mask. Equality recognizes equal finite values and matching signed infinities; NaN is never
equal and is always not-equal. Tinygrad lowers `>=` and `<=` as boolean inversion of `<`, so inversion is deliberately
performed inside the validity mask rather than as a plain `1-result`, preserving IEEE false-on-NaN behavior.

Research commit `bd1b8009a` supplied the proven device formula. `~/npu/include/old/rknn_ops.md` lists Greater,
GreaterOrEqual, Less, and LessOrEqual as supported model operations, and `~/rk3588/test/test_rockchip.py` contains the
same comparison matrix. The older `89c7cb67b` implementation was rejected because its runtime sanitized infinities and
converted comparison tensors with NumPy. This port uses the current immutable DPU EW image and existing lossless typed
bool-output ABI instead.

- Focused six-predicate class: **6 passed in 56.45 s**, sequentially.
- Classification plus comparisons: **9 passed in 59.17 s**, sequentially.
- Complete Rockchip census: **331 passed, 11 skipped, 180 subtests passed in 375.24 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,797/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. Runtime never reads, sanitizes, compares, or converts input
tensor values; all classification, comparison, boolean inversion, and FP16-to-INT32 mask conversion occur on DPU EW.
Only the existing raw low-byte packing exposes NPU-produced 0/1 lanes as public bool bytes. There is no LUT, CMAC, CNA,
PPU fallback, or tolerance relaxation.

---

## 2026-08-09 — FP16 logical-not and scalar isclose coverage

The FP16 half of `test_logical_not` and the unchanged upstream `test_isclose_scalar` now join the Rockchip census.
Logical-not covers ordinary nonzero values, positive and negative zero, both infinities, and NaN. Scalar isclose covers
the canonical `(3,4,5,6)` tensor-to-1 comparison plus exact/near/unequal finite values, infinities, and NaN. Both lower
entirely through the preceding IEEE comparison milestone and typed bool-output ABI; no backend change was needed.

General tensor/tensor `test_isclose` was tested and intentionally excluded. Its `x.isclose(x+1e-6)` case produced 14
false results among 360 expected true results because the FP16 tolerance/difference enters the DPU's subnormal region.
Historical commits `b85d3c4b0` and `31c07151d` passed this group only through `_HOST_ELEMENTWISE_LAYOUT`, explicitly
computing isclose on the CPU; that path is not ported. Their native WIP also documented reset explosion on the 32-case
IEEE edge matrix. Exact boolean expectations were not relaxed.

- Focused logical-predicate class: **2 passed in 14.95 s**, sequentially.
- Complete Rockchip census: **333 passed, 11 skipped, 180 subtests passed in 388.80 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,797/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change and no
new data conversion or host arithmetic. There is no LUT, CMAC, CNA, PPU fallback, or tolerance relaxation.

---

## 2026-08-09 — FP16-input ANY/ALL reductions on DPU EW

Global, tuple-axis, scalar, and empty-axis `any`/`all` now join the Rockchip census for FP16 inputs. Coverage includes
positive and negative zero, finite nonzero values, both infinities, NaN, mixed result rows, and the exact empty
identities (`any=False`, `all=True`). External boolean input tensors and upstream `test_all_large`, which constructs a
boolean tensor, remain outside the NPU's FP16 external-input contract.

Both Tinygrad reduction forms are recognized: small fully unrolled OR/AND trees and register-loop reductions. Static
source offsets are packed into an aligned candidate-row matrix. DPU `SUB -> ABS -> positive mask` produces exact
nonzero lanes, then a balanced DPU MAX tree implements ANY and a balanced DPU MUL tree implements ALL. The existing
typed DPU conversion writes the public boolean result. Empty reductions contain no input values and use the existing
constant-fill image for their mathematically fixed identity.

Historical commit `03bad6205` was inspected but not ported: it uses CMAC to count masks, performs host bool/FP16
conversion, and adds host tiling for multi-megabyte reductions. Searches under `~/npu` and `~/rk3588` found the
canonical tests but no independent pure-DPU boolean-reduction register implementation. This milestone therefore
reuses only the currently proven DPU EW mask, striped-gather, balanced-reduction, and typed-output machinery.

- Before specialization, 5-element reductions used **46 submits**, 12-element reductions used **102 submits**, and
  the 360-element loop form was rejected. All three forms now pass with **6 submits** each.
- Focused boolean-reduction class: **6 passed in 5.26 s**, sequentially; slowest method **1.03 s**.
- Complete Rockchip census: **339 passed, 11 skipped, 180 subtests passed in 389.29 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,864/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. Runtime gathers only raw FP16 lanes using compile-time
offsets; it never reads, compares, reduces, or branches on tensor values. All nonzero classification and reduction
arithmetic execute on DPU EW. Static empty identities are constant initialization, not host evaluation of input data.
There is no LUT, CMAC, CNA, PPU fallback, FP32/boolean external-input conversion, or tolerance relaxation.

---

## 2026-08-09 — native FP16 floor/ceil and no-LUT truncation

The unchanged upstream `test_floor`, `test_ceil`, and `test_trunc` methods now join the Rockchip census. RK3588 TRM
`RKNN_dpu_ew_cfg` identifies EW ALU algorithms 7 and 8 as Floor and Ceil. The renderer recognizes Tinygrad's canonical
TRUNC-based WHERE expansions and tags them for those named native configurations; it does not embed unexplained
register literals.

TRUNC has no corresponding native ALU algorithm. It is composed entirely on DPU EW as
`floor(max(x, 0)) + ceil(min(x, 0))`, with `min(x,0)` expressed through SUB/MAX. This avoids the DPU's unsafe
`infinity * 0` behavior and preserves finite values, both infinities, and NaN numerically. It does not preserve the
negative-zero sign when a negative value with magnitude below one truncates to zero; canonical Tinygrad/Torch numeric
comparison treats both zeros as equal, and no tolerance was relaxed.

Historical `5619ceff8` was rejected because it routes floor/ceil through `_HOST_ELEMENTWISE_LAYOUT`. Historical
`29559149c` composes integral rounding from a generated roundoff LUT, so only its test intent was relevant. The direct
TRM algorithms and current DPU EW image are used here. `test_round` was also tested and remains excluded: Tinygrad's
round-to-even graph still contains unsupported boolean/XOR selection, while the proven historical implementation uses
a LUT, which is outside the current scope.

- Focused canonical methods: **3 passed in 3.17 s**, sequentially.
- Permanent rounding class: **4 passed in 3.10 s**; its exhaustive method checks all **65,536 FP16 encodings** and the
  three operations use exactly **3 ioctls** total.
- Complete Rockchip census: **343 passed, 11 skipped, 180 subtests passed in 391.86 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,892/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. The new renderer code only recognizes canonical UOps and
selects DPU ALU configurations; all floor, ceil, min/max, subtraction, and addition stages execute on the NPU. There is
no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input, or tolerance relaxation.

---

## 2026-08-09 — direct logical composition of FP16 predicates

Direct boolean AND, OR, and XOR outputs derived from FP16 comparisons now join the Rockchip census. AND includes the
FP16 portion of upstream `test_and`; OR and XOR cover finite values, signed zero, both infinities, and NaN. External
INT32 and boolean input portions of the upstream bitwise methods remain outside the NPU's FP16 input contract.

AND and OR were already expressible by the IEEE comparison-mask builder as DPU MUL and MAX respectively. XOR now uses
the exact 0/1 identity `abs(lhs-rhs)`, emitted as DPU SUB followed by the proven native ABS algorithm, before the
existing typed DPU boolean conversion. Historical `e7e2f2720`, `22c974ac2`, `23800edca`, and `59dcb9a15` were inspected
but not ported because they operate on external public bool tensors through INT8 DPU surfaces; this milestone instead
composes masks produced on-device from FP16 inputs.

- Focused logical-predicate class: **5 passed in 20.41 s**, sequentially; AND/OR/XOR are each **1.89–1.92 s**.
- Complete Rockchip census: **346 passed, 11 skipped, 180 subtests passed in 397.71 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. All operand classification, comparison, logical
composition, subtraction, ABS, and typed bool conversion occur on DPU EW; the CPU never reads or branches on tensor
values. There is no LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — FP16 fmod and modulo compositions on DPU EW

The complete floating-input portions of upstream `test_fmod` and `test_mod` now join the Rockchip census. Coverage
includes tensor/tensor remainder, integer-valued and fractional scalar divisors, and both integer-valued and fractional
reverse scalar numerators. Integer tensor inputs remain outside the NPU's FP16 external-input contract.

No dedicated modulo register path is required. Tinygrad's FP16 graphs decompose `fmod` through FDIV, TRUNC, MUL, and
SUB, and Python-style modulo through FDIV, FLOOR, MUL, and SUB. The preceding native integral-rounding milestone lets
each complete expression execute as one DPU PC-chain ioctl. The tests assert exactly **7 ioctls for 7 modulo forms**
and **3 ioctls for 3 fmod forms**, preventing a host fallback from satisfying correctness alone.

Historical commits `ead24405a` and `86b0f1c6f` were inspected but not ported because both explicitly route the
operation through `_HOST_ELEMENTWISE_LAYOUT`. Searches under `~/npu` and `~/rk3588` found CPU reference `fmodf` and
model-operator documentation, but no independent native modulo register implementation. The current composition uses
only already proven DPU EW operations.

- Focused modulo class: **2 passed in 3.03 s**, sequentially; method call times were **0.14–0.17 s**.
- Complete Rockchip census: **348 passed, 11 skipped, 180 subtests passed in 396.01 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. Runtime
gathers remain raw lane movement by compile-time offsets; all division, rounding, multiplication, and subtraction run
on DPU EW. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion,
or tolerance relaxation.

---

## 2026-08-09 — FP16 division rounding modes on DPU EW

The complete FP16-input portion of upstream `test_div_rounding_mode` now joins the Rockchip census. One seven-element
FP16 numerator is divided by each of ten signed, nonzero length-one FP16 denominators under true division, truncating
division, and floor division. The 30 cases cover positive and negative operands, exact and inexact quotients, and zero.
Integer tensor combinations remain outside the NPU's FP16 external-input contract.

Tinygrad lowers these forms to scalar-broadcast DPU FDIV, followed by the native TRUNC or FLOOR composition introduced
in the integral-rounding milestone. Every expression is one PC-chain ioctl; the test asserts exactly **30 ioctls for
30 cases**. Historical commit `f29131cdf` and its documentation commit `c46039f9a` were inspected but not ported because
they execute the division and rounding graph through a typed host elementwise task. Searches under `~/npu` and
`~/rk3588` found the canonical test but no independent native implementation beyond the already used DPU primitives.

- Focused division-rounding class: **1 passed in 2.86 s**, sequentially; call time was **0.27 s**.
- Complete Rockchip census: **349 passed, 11 skipped, 180 subtests passed in 397.56 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. All
division and integral-rounding arithmetic executes on DPU EW, with only static scalar-broadcast address preparation on
the host. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion,
or tolerance relaxation.

---

## 2026-08-09 — constant integer powers of FP16 inputs

The integer-exponent portion of upstream `test_pow` now joins the Rockchip census: vector `x**0`, `x**1`, `x**2`,
`x**3`, and `x**-2`; scalar `x**2` and `x**-2`; and the two negative-range cubic regressions. Fractional and runtime
tensor exponents remain excluded because their historical Rockchip implementations depend on transcendental/LUT
machinery, which is outside the current scope.

Tinygrad simplifies the supported vector forms into static fill/copy, repeated MUL, or reciprocal plus MUL. `x**0` is
the input-independent constant one and needs no ioctl; the four remaining ordinary vector forms and the negative-range
vector cubic each execute in one DPU PC-chain. Rank-zero literal inputs constant-fold without a submit. The test asserts
exactly **5 ioctls for the 9 forms**.

The historical power series was inspected rather than ported. Commit `463d0dfd6` evaluates integer-tensor powers in a
typed host task, while later general/fractional power commits use generated range-reduced LUT graphs. `~/npu` and
`~/rk3588` contain the canonical tests but no simpler native constant-integer-power register operation. This milestone
therefore relies only on Tinygrad's existing decomposition and proven DPU EW MUL/FDIV stages.

- Focused integer-power class: **1 passed in 3.10 s**, sequentially; call time was **0.49 s**.
- Complete Rockchip census: **350 passed, 11 skipped, 180 subtests passed in 397.32 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,895/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. Every
input-dependent vector multiplication and reciprocal runs on DPU EW; static constant fills and compile-time rank-zero
folds do not inspect a runtime tensor buffer. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback,
external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — permuted INT32 WHERE output from FP16 predicates

Upstream `test_where_permute` now joins the Rockchip census. Its FP16 comparison selects the INT32 constants 4 and 2,
then writes a transposed result. Coverage also includes a non-square matrix containing NaN, both infinities, positive
and negative zero, and a finite true case, proving IEEE predicate behavior and the static transpose mapping.

The renderer now shares one typed DPU output helper between public bool masks and exact FP16-valued INT32 expressions.
A strict matcher accepts only a DPU-lowerable FP16 comparison and two constant INT32 arms whose affine selection is
exactly representable in FP16. DPU EW computes `false + mask*(true-false)`, and the existing hardware FP16-to-INT32
output-conversion stage writes the public result. Static permutation is expressed only by compile-time gather offsets.

Each IEEE-safe comparison currently uses **12 barrier-separated ioctls**, so the two permanent cases assert **24
ioctls**. The barriers come from the existing NaN/infinity classification and positive-mask stages, not CPU execution
or command-buffer tiling. The method call takes **2.58 s**, well below the 30-second profiling threshold.

Historical commit `297b6fdc0` was inspected but not ported: it writes four FP16 byte planes and reassembles INT32 values
with NumPy in the runtime. The current backend already has direct DPU INT32 output conversion, so this implementation
needs no host reassembly, typed runtime ABI extension, or image-format change. Exact `copysign` was also tested as the
next candidate and remains excluded: RK3588 FDIV maps both `1/+0` and `1/-0` to positive infinity, losing the sign bit
before Tinygrad's canonical signed-zero predicate; host bit inspection would violate the no-CPU-cheat rule.

- Focused WHERE-permute coverage: **1 passed in 5.22 s**, sequentially.
- Shared WHERE/classification/comparison/logical/reduction regression: **30 passed in 85.02 s**, sequentially.
- Complete Rockchip census: **351 passed, 11 skipped, 180 subtests passed in 401.19 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,909/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. The renderer inspects only constant arms, dtypes, UOp
structure, and static indexes; it never reads tensor values. Comparison, selection arithmetic, and FP16-to-INT32
conversion execute on DPU EW. There is no host tensor-value evaluation or reassembly, LUT, CMAC, CNA, PPU fallback,
external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — direct FP16-to-bool cast on DPU EW

The FP16-input portion of upstream `test_cast` now joins the Rockchip census. Coverage includes a normal tensor plus
finite positive/negative values, positive and negative zero, both infinities, and NaN. FP32 output, external integer
input, and external boolean input portions remain outside the NPU's FP16 external-input contract.

Tinygrad lowers direct `.bool()` to `CMPNE(input, 0)`. The generic IEEE equality path was exact but used **14 ioctls**.
A strict direct-load matcher now uses the identity `positive_mask(abs(input))`: native ABS maps both signed zeros to
zero and both infinities to positive infinity, while the proven DPU positive-mask primitive treats NaN as nonzero.
The public bool result uses the existing hardware FP16-to-INT32 conversion and raw low-byte packing. Each cast now uses
**3 ioctls**, so the two permanent cases assert **6 total**.

Historical `091df1bd8` established an older typed bool-output ABI whose runtime packed NPU-produced FP16 masks with a
NumPy nonzero operation; `70740a365` documented rejection of direct channel-packed FP16 bool output. Neither path was
ported. Searches under `~/npu` and `~/rk3588` found CPU reference checks and earlier comparison experiments but no
smaller native nonzero register sequence than the current ABS/positive-mask composition.

An exhaustive 65,536-pattern probe was attempted but deliberately not added. Public bool conversion operates on
four-lane INT32 atoms; 65,536 values therefore create 16,384 conversion tasks, which the current page-sized conversion
batching splits into roughly 1,171 submit/reset groups. The probe returned naturally after about two minutes with no
driver timeout, but its output channel was lost and thus provides no correctness evidence. The immediate vendor health
check still passed 60/60. This identifies conversion batching as a separate performance problem, not a cast-semantic
failure.

- Focused cast class: **1 passed in 3.42 s**, sequentially; call time was **0.79 s**.
- Shared classification/logical/boolean-reduction regression: **14 passed in 25.53 s**, sequentially.
- Complete Rockchip census: **352 passed, 11 skipped, 180 subtests passed in 400.13 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** both after the stress probe and after the census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,914/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. ABS, nonzero classification, and FP16-to-INT32 conversion
all execute on DPU EW; runtime only packs the low byte of NPU-produced 0/1 INT32 lanes into the public bool buffer. The
renderer inspects only the UOp shape and dtype. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback,
external non-FP16 input conversion, or tolerance relaxation.

---

## 2026-08-09 — arithmetic FP16 softsign on DPU EW

Upstream `test_softsign` and `test_softsign_exact` now join the Rockchip census. The group covers a 45x65 tensor, a
rank-zero scalar, and the exact values `[-1, 0, 1]` under the same FP16 tolerance used by `test_gemm_fp16`.

Tinygrad defines softsign as `x / (1 + abs(x))`. The nonconstant cases therefore reuse the already-proven native DPU
EW ABS, ADD, and FDIV stages rather than adding a special operation. The 45x65 and exact-vector graphs each execute as
three DPU tasks in one ioctl; the literal rank-zero input folds at compile time without an ioctl. A warm direct profile
measured **0.150 s** for the matrix and **0.020 s** for the exact vector.

Other Tinygrad branches contain no Rockchip-specific softsign lowering. `~/npu` and `~/rk3588` contain a standalone
Q0.15 LUT implementation, but it was intentionally not ported because the current scope excludes LUT work and ordinary
FP16 DPU arithmetic already implements the canonical Tinygrad expression.

- Focused upstream cases: **2 passed**, sequentially (`3.09 s` and `2.68 s` pytest wall times).
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **354 passed, 11 skipped, 180 subtests passed in
  401.34 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,914/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. All
input-dependent ABS, addition, and division execute on DPU EW; compile-time scalar folding does not inspect a runtime
tensor buffer. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input
conversion, or tolerance relaxation.

---

## 2026-08-09 — embedded FP16 comparison masks for backward graphs

The comparison-backward invariants from upstream `test_cmp_ne_backwards` and `test_cmp_lt_backwards` now join the
Rockchip census. The upstream methods use `Tensor.randn`, whose normal-distribution graph requires transcendental/LUT
support. The Rockchip variants instead use deterministic FP16 values spanning negative infinity, finite negative,
signed zero, finite positive, positive infinity, and NaN while preserving the same derivative check against Torch.

Backward exposes a previously unsupported graph boundary: Tinygrad casts the boolean comparison result back to FP16
before multiplying it into the gradient. The renderer now narrowly matches `CAST(bool -> half)`, following the typed
cast-matcher style used by the PTX renderer, and replaces it with an existing DPU FP16 0/1 mask. Direct `x != 0` uses
the optimized ABS/nonzero path; other predicates use the IEEE-correct comparison mask already proven for public bool
output. No new register primitive or runtime ABI is required.

The `x != 0` backward graph executes **2 DPU tasks in 2 ioctls**. IEEE-correct `x < 0` executes **31 DPU tasks in 11
ioctls**. Both gradients exactly match Torch across all seven finite/nonfinite inputs.

No separate Rockchip comparison-backward implementation exists in the other Tinygrad branches, `~/npu`, or
`~/rk3588`; their available material covers forward comparison primitives. The existing DPU masks are therefore the
relevant reference. The historical CPU isclose paths were also reviewed and remain excluded.

- Focused backward group: **2 passed in 4.00 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **356 passed, 11 skipped, 180 subtests passed in
  402.05 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,918/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad core change. The rewrite inspects only predicate UOp structure and all
mask construction plus gradient arithmetic executes on DPU EW; Torch is used only for the expected gradient. There is
no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16 input conversion, or tolerance
relaxation.

---

## 2026-08-09 — FP16 einsum ellipsis layouts

Three executable layouts and both static validation cases from upstream `test_einsum_ellipsis` now join the Rockchip
census. The supported forms cover batched matrix products (`...id,...jd->...ij`), an implicit scalar reduction
(`...id,...jd`), and permuted ellipsis axes (`i...j,ji...->...`). They reuse the existing FP16 gather, MUL, and ADD
lowering without a special einsum operation.

The remaining upstream execution case, two `(32,7,24,24,24)` inputs reduced by `ij...,ij...->ij`, was tested but is
not admitted yet. Tinygrad lowers each of its 224 output rows to 432 loop iterations with a 32-product unrolled ADD
tree, or 13,824 products per row and 3,096,576 input lanes overall. The current loop-reduction matcher recognizes the
shape but its dot path accepts only one MUL contribution per iteration. Flattening this graph into ordinary EW stages
would create an impractically large command sequence; it needs a bounded hierarchical reduction rather than a broad
matcher relaxation.

The canonical ellipsis cases are present under `~/rk3588/test/test_ops.py`. Searches across the other Rockchip
Tinygrad branches and `~/npu` found no separate hardware-specific einsum implementation to port; their implementation
is likewise the generic Tinygrad reshape/gather/multiply/reduce graph.

- Focused Rockchip method: **1 passed in 3.03 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **357 passed, 11 skipped, 180 subtests passed in
  400.70 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,918/274 executable lines**.

This is a test-only coverage milestone. The CPU-cheat audit found no renderer, runtime, or Tinygrad core change. All
runtime-dependent lane movement and arithmetic execute through the existing Rockchip NPU path; Torch is used only as
the expected-value oracle. There is no host tensor-value evaluation, LUT, CMAC, CNA, PPU fallback, external non-FP16
input conversion, or tolerance relaxation.

---

## 2026-08-09 — exact FP16-pair bitcast to INT32

Rockchip now preserves the exact representation of adjacent FP16 lanes when Tinygrad bitcasts them to INT32. Permanent
coverage includes contiguous and leading-axis-permuted layouts over positive and negative zero, infinities, signed NaNs,
subnormals, maximum finite values, and ordinary normals.

The unchanged upstream `test_bitcast` uses shape `(3,3)`. Under `DEFAULT_FLOAT=HALF`, PyTorch rejects it before the
backend runs because its final dimension is not divisible by the two FP16 lanes required for one INT32 element. The
Rockchip method therefore uses the equivalent valid shape `(2,3,4)` and checks both its direct and permuted views.

Tinygrad expresses the reinterpretation as two `BITCAST(half -> ushort)`, zero-extension, shifts by 0/16, an unsigned
ADD, and `BITCAST(uint -> int)`. The strict matcher accepts only that exact graph, proves that both loads use one bounded
FP16 argument, and verifies that every high lane is the adjacent successor of an even low lane. It then copies each
four-byte representation with the existing typed raw gather; no numeric cast, shift, or addition is performed on host
or DPU. This follows other Tinygrad renderers' exact `Ops.BITCAST` treatment while mapping it to Rockchip's existing raw
layout ABI instead of inventing a scalar instruction stream.

No hardware-specific bitcast implementation exists in the other Rockchip Tinygrad branches, `~/npu`, or `~/rk3588`.
The legacy `~/rk3588/experimental/ops_rockchip.py` interpreter uses host `struct.pack/unpack`; that CPU value interpreter
was inspected and deliberately not ported.

- Focused exact-bit method: **1 passed in 2.85 s**, sequentially; both layouts assert bit-for-bit equality and zero submits.
- Typed cast/bitcast/TopK regression: **6 passed, 2 subtests passed in 21.52 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **358 passed, 11 skipped, 180 subtests passed in
  401.05 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,947/274 executable lines**.

The CPU-cheat audit found only compile-time UOp/index validation and raw `uint32` representation movement by the same
gather mechanism already used for device-produced indices. Neither renderer nor runtime reads or interprets an FP16
numeric value. There is no host tensor-value arithmetic, LUT, CMAC, CNA, PPU fallback, external non-FP16 input, Tinygrad
core change, or tolerance relaxation.

---

## 2026-08-09 — IEEE NaN propagation through FP16 extrema reductions

The behavior named by upstream `test_max_nan` now runs actively in the Rockchip census even though the canonical test is
globally skipped as broken by tinygrad issue #862. Runtime FP16 inputs cover NaN in both operand orders for scalar MAX,
plus row-wise MAX and MIN with one NaN row and one finite control row. Every case propagates NaN exactly where required,
preserves the finite result, and executes as one native DPU EW reduction task; the method asserts **4 ioctls total**.

No Rockchip-specific NaN-extrema implementation exists in the other Tinygrad branches, `~/npu`, or `~/rk3588`. The
existing RK3588 EW MAX configuration already has the required propagation behavior, so this is intentionally a test-only
milestone with no renderer/runtime special case.

The remaining large ellipsis-einsum case was investigated further but is still excluded. A temporary hierarchical
layout reduced it from 27,648 gathers and 27,647 logical EW stages to two affine transposes, 15 logical stages, and 901
hardware tiles. One 901-task ioctl hit the six-second driver timeout; splitting at the natural MUL/reduction boundary
completed as two jobs in about six seconds, after which the vendor health check passed 60/60. Plain FP16 pairwise
accumulation nevertheless missed 20/224 outputs at the permitted tolerance, with maximum absolute error 0.25. Modeling
confirmed that even an exact sum of FP16-rounded products fails, so FP16 product residuals are required. The TRM exposes
either EW ALU or MUL per stage rather than a two-input FMA; the non-LUT Dekker path would require more than 11,000 tiled
tasks before compensated accumulation. It was therefore removed instead of admitting inaccurate or >30-second code.

- Focused NaN-extrema regression: **1 passed in 3.16 s**, sequentially.
- Complete reduction class: **28 passed, 1 skipped in 22.71 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **359 passed, 11 skipped, 180 subtests passed in
  402.53 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused work and after the census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,947/274 executable lines**.

The CPU-cheat audit found no renderer, runtime, or Tinygrad-core change. All input-dependent extrema arithmetic executes
on DPU EW; Python only asserts the returned scalar/array values. There is no host tensor-value arithmetic, LUT, CMAC,
CNA, PPU fallback, external non-FP16 input, tolerance relaxation, or committed experimental einsum path.

---

## 2026-08-09 — vectorized IEEE isclose edge matrix

All 16 pairs from `{+inf, -inf, NaN, 0}²` in upstream `test_isclose_edge_cases` now run in the Rockchip census for both
`equal_nan=False` and `equal_nan=True`. The Rockchip method preserves every upstream pair but packs each mode into one
16-lane graph instead of paying realization overhead for 32 scalar helper calls. The two graphs reuse the existing DPU
EW ABS, arithmetic, IEEE classification, comparison-mask, and typed bool-output paths. They execute **364 DPU tasks in
104 ioctls** and complete in **14.02 s**; the submit assertion prevents silent constant folding or fallback.

Historical commit `b85d3c4b0` passes isclose through host-side predicate evaluation and was not ported. The current
renderer already represents the complete IEEE predicate on DPU, so this milestone adds no backend special case and does
not relax exact boolean expectations. General finite `test_isclose` remains excluded because FP16 subnormal tolerance
arithmetic loses distinctions such as `x` versus `x+1e-6` before the comparison.

Large boolean reduction was evaluated as the following candidate. `test_all_large` is not constant-folded: after an
FP16 predicate reduction it needs a second 256-way AND over a byte-bool intermediate. Historical commits `25df366fb`
and `03bad6205` widen those bytes to FP16 with NumPy, which is a CPU numeric conversion and is not ported. A clean route
would use raw byte placement into zeroed INT32 lanes plus the existing DPU INT32-to-FP16 conversion, but the upstream
one-million-lane first stage spends more than a minute in Tinygrad symbolic rewriting before reaching the renderer,
including under `NOOPT=1`. Because Tinygrad core changes are out of scope and individual cases must remain below 30 s,
the group is deferred rather than narrowed to only its 32K/64K cases.

- Focused edge-matrix method: **1 passed in 14.02 s**, sequentially.
- Complete logical-predicate class: **6 passed in 31.34 s**; no individual method exceeded 30 s.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **360 passed, 11 skipped, 180 subtests passed in
  415.81 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **1,947/274 executable lines**.

The CPU-cheat audit found no renderer, runtime, or Tinygrad-core change. Every runtime-dependent isclose operation and
bool conversion executes on DPU EW; Python only constructs the static test matrix and checks exact bool output. There is
no host tensor-value arithmetic, LUT, CMAC, CNA, PPU fallback, external non-FP16 input, or tolerance relaxation.

---

## 2026-08-09 — native MaxPool2D spatial-index output

The complete upstream `test_max_pool2d_return_indices` method now runs in the Rockchip census. Its seven cases cover
batch/channel planes, dilation, padding, ceil-mode tail padding, a 156-lane global window, first-index ties, and
overlapping windows. The two explicit integer-literal fixtures are converted to FP16 while constructing the test input,
which preserves their upstream values and tie semantics while respecting the NPU's FP16 input contract; expected and
device-produced indices remain exact INT32.

Tinygrad lowers each unrolled index graph to equality between every FP16 pool candidate and the already-produced pooled
maximum, multiplies each equality by a descending spatial coordinate, MAX-reduces the candidates, and subtracts from
the spatial size. The strict Rockchip matcher proves that exact graph, evaluates only its compile-time address/gate and
coordinate expressions, and emits the existing raw gathers plus DPU equality/selection and native INT32-output stages.
Invalid padded lanes carry FP16 `-inf` and zero selection weight, so they cannot win a valid window. The global case's
single-register loop is recognized separately and lowered to the same immutable image. Spatial coordinates are capped
at 2048 so every integer represented during DPU FP16 selection is exact.

Historical commits `aa01f775e`, `d72bcc3f0`, `76c31806e`, and `cc3b7b6c6` established the candidate-address and
first-tie algorithm, but also contained a NumPy ArgMax fallback, NumPy unpool scatter, LUT truncation, and thousands of
tiny reset-heavy tasks; none of those paths were ported. Searches in `~/npu` and `~/rk3588` found the DPU RDMA
`UNPOOLING_EN` and PPU pooling register definitions and operator declarations, but no proven returned-index register
contract or compact EW implementation. The current implementation therefore stays on the already-proven DPU EW/image
ABI and adds no runtime semantic interpreter.

- Focused seven-case upstream method: **1 passed in 6.15 s**, sequentially.
- Complete MaxPool class: **12 passed, 1 skipped, 33 subtests passed in 10.04 s**, sequentially.
- Shared cumulative-extrema, arg-extrema, and sort-index regression: **22 passed, 16 subtests passed in 152.34 s**.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **361 passed, 10 skipped, 180 subtests passed in
  416.48 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused method, shared regressions, and
  complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,083/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no code that reads or interprets tensor values. NumPy
is used only for compile-time symbolic index evaluation, raw gather-map construction already used throughout the
renderer, and FP16 test-fixture creation. All input-dependent equality, tie selection, MAX reduction, and INT32
conversion execute on the NPU. There is no host ArgMax, host scatter, LUT, CMAC, CNA, PPU fallback, tolerance
relaxation, or external non-FP16 input.

---

## 2026-08-09 — bounded native MaxUnpool scatter

The small finite MaxUnpool case from upstream `test_max_unpool2d` now runs as a dedicated Rockchip regression: a
`(1,3,7,6)` FP16 tensor is MaxPool2D'd with a 2x2 kernel, then unpooled to the explicitly requested `(7,6)` spatial
shape. The kernel covers three planes, nine pooled candidates per plane, dynamic INT32 indices, and 126 FP16 output
lanes. The complete pool/index/unpool pipeline passes in 5.98 seconds.

The strict renderer matcher recognizes only the canonical Tinygrad one-hot scatter: each term compares a dynamic INT32
index with the compile-time output-plane coordinate, selects the FP16 pooled lane on equality, and the terms are summed.
It proves that the INT32 and FP16 loads share each candidate offset, that every output lane sees exactly one complete
pooled plane, and that coordinates are the compact spatial order. Raw gathers stripe indices and values into a bounded
matrix; DPU INT32-input conversion, equality masks, FP16 selection, and ADD reduction produce the output. The image has
27 raw gathers and 15 logical EW stages. No runtime semantic path was added.

The full upstream method is not marked complete yet. Its 25-candidate `(8,3,30,30)` case would repeat 600 indices into
540,000 lanes before four-lane INT32 conversion, expanding to 135,000 conversion tasks. It needs an image phase that
converts the compact index vector once and stripes the resulting FP16 lanes afterward. Its 80-candidate `(8,3,50,50)`
case additionally produces spatial indices through 2499, beyond the exact-integer range of the current FP16 coordinate
selector, and lowers the scatter as a loop. The infinity/NaN regression also needs representation-level selection,
because ordinary `value * 0` turns an unselected infinity into NaN. These cases remain excluded rather than being run
through the old branch's NumPy scatter or an inaccurate/unsafe expansion.

Historical commit `76c31806e` supplied useful graph and plane-layout evidence, but its implementation included host
packing/compaction helpers and an optional NumPy scatter. The RKNN operator guide lists MaxUnpool under CPU operators;
`~/npu` and `~/rk3588` expose an RDMA `UNPOOLING_EN` bit but no verified dynamic-index data contract. None of those
unproven or CPU paths were ported.

- Focused bounded MaxUnpool pipeline: **1 passed in 5.98 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **362 passed, 10 skipped, 180 subtests passed in
  418.57 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after focused and complete runs.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,142/274 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no host read of indices or pooled values. NumPy is used
only by the pre-existing compile-time static-index machinery and test oracle/fixture code. All dynamic INT32 conversion,
comparison, mask selection, and accumulation execute on DPU EW. There is no host scatter, host ArgMax, LUT, CMAC, CNA,
PPU fallback, tolerance relaxation, or external non-FP16 input.

---

## 2026-08-09 — compact-index phase for padded MaxUnpool

The second finite case from upstream `test_max_unpool2d` now runs natively: `(8,3,30,30)` FP16 input, 3x3 MaxPool
with stride `(6,7)` and padding 1, followed by MaxUnpool to explicit `(30,30)` spatial output. It covers 24 planes,
25 pooled candidates per plane, 600 dynamic INT32 indices, and 21,600 FP16 output lanes. The standalone case passes in
14.23 seconds; both admitted MaxUnpool cases pass together in 14.31 seconds, so neither approaches the 30-second limit.

RKImage version 26 adds one explicit mid-image raw-gather phase. The unpool program first converts the compact 600-lane
INT32 index tensor to FP16 on DPU exactly once. After that blocking phase, the runtime synchronizes the produced scratch,
stripes its FP16 representations into the compile-time candidate matrix, and resumes DPU equality/selection/reduction.
This reduces the padded case from a projected 540,000 converted lanes / 135,000 four-lane conversion tasks to only 600
converted lanes / 150 tasks. The same path improves the bounded case from 5.98 to 3.84 seconds.

The phase is encoded explicitly with mid-gather count and EW split index, validated during image decode, and executed by
the same synchronized raw-gather helper as terminal gathers. It does not interpret an index value or choose a
destination on the host: all gather offsets are proven statically from the one-hot graph, while dynamic index conversion,
equality, mask multiplication, and accumulation remain DPU operations. Existing images use zero mid-gathers and retain
their prior execution order; the complete census verifies both cached-style ordinary programs and the new split program.

- Focused bounded pipeline after compaction: **1 passed in 3.84 s**, sequentially.
- Focused 25-candidate padded pipeline: **1 passed in 14.23 s**, sequentially.
- Complete MaxUnpool class: **2 passed in 14.31 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **363 passed, 10 skipped, 180 subtests passed in
  445.10 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after focused and complete runs.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,147/280 executable lines**.

The CPU-cheat audit found only representation-preserving host movement between two blocking NPU phases, matching the
backend's existing raw pre/post gather contract. Neither renderer nor runtime reads or branches on an index or pooled
value. There is no host scatter, host ArgMax, numeric conversion, LUT, CMAC, CNA, PPU fallback, tolerance relaxation,
or external non-FP16 input.

---

## 2026-08-09 — exact wide MaxPool spatial indices

Returned MaxPool indices now remain exact beyond FP16's consecutive-integer range. A deterministic 50x50 regression
uses a 5x5 window with stride `(6,5)` and selects spatial index 2349; it passes exactly in 3.79 seconds. The complete
MaxPool class passes 13 tests plus 33 subtests, with only the explicit integer-input case skipped, in 10.98 seconds.

Historical commit `76c31806e` established the representation-safe design: select base-256 pieces on DPU and assemble
the final INT32 output by moving raw bytes. The current RKImage implementation preserves first-tie semantics without a
sequential candidate chain. Compile-time priorities rank valid coordinates in each output lane. DPU equality masks
select priority-tagged four-bit digits whose maximum remains within FP16's exact range; subtracting the selected
priority recovers each digit, pairs of digits form exact bytes, and native DPU conversion produces INT32 byte values.
The post-gather writes only those raw byte representations into the zeroed INT32 output. The digit radix and admitted
window bound are derived from FP16's 11 explicit precision bits rather than an arbitrary task or command-buffer cap.

`~/npu` and the RKNN guide contain no native returned-index contract, and MaxUnpool is documented as a CPU operator.
`~/rk3588` contains the upstream tests and the older byte-assembly proof but no simpler verified DPU instruction for
wide dynamic indices. No PPU, CMAC, LUT, or host ArgMax implementation was imported.

- Focused exact-wide returned-index regression: **1 passed in 3.79 s**, sequentially.
- Complete MaxPool class: **13 passed, 1 skipped, 33 subtests passed in 10.98 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **364 passed, 10 skipped, 180 subtests passed in
  429.82 s** pytest time / **454.76 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,215/281 executable lines**.

The CPU-cheat audit found only compile-time coordinate/rank construction and raw representation movement in generic
gathers. Runtime never reads, compares, converts, or branches on a tensor value: equality, priority selection, digit
reconstruction, and FP16-to-INT32 conversion are all submitted to DPU EW. The next MaxUnpool case is not included in
this milestone: its 80-candidate, 2350-output-plane consumer still needs exact byte-wise INT32 comparison. A diagnostic
run rejected that consumer after 36.55 seconds, showing that wide producer compilation/conversion also needs profiling
before the full pipeline can meet the 30-second target.
