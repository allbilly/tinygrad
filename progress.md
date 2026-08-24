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

---

## 2026-08-09 — exact wide MaxUnpool scatter

Finite MaxUnpool now supports dynamic spatial indices beyond FP16's consecutive-integer range. A direct regression
scatters distinct values to indices 2049 and 2499 in seven independent 50x50 planes and checks the complete FP16 output
bit-exactly. The first wide upstream pipeline also passes: `(8,3,50,50)` FP16 input, 5x5 MaxPool with stride `(6,5)`,
80 pooled candidates per plane, 1,920 dynamic INT32 indices, and 60,000 unpooled FP16 output lanes.

The wide image compares the raw little-endian bytes of each dynamic INT32 index. Generic representation-preserving
gathers expose each compact byte lane, DPU INT32-input conversion converts byte values to exact FP16 integers once, and
the existing mid-image gather phase stripes those lanes over the candidate matrix. DPU SUB/ABS/MIN/inversion produces
one exact equality mask per required base-256 byte; DPU multiplication combines the masks, selects the pooled values,
and ADD-reduces the candidates. The loop matcher proves the canonical Tinygrad MaxUnpool accumulator, shared value/index
addresses, plane and candidate dimensions, output-coordinate order, and affine source layout before emitting this image.
It stores affine gather plans instead of expanding millions of repeated offsets in the compiled image.

Profiling the complete pool/index/unpool pipeline attributed 3.61 seconds to rendering (0.01 pool values, 1.44 pool
indices, 2.16 unpool), 15.46 seconds to DPU EW execution, and 15.16 seconds to INT32 conversion, including 15.06 seconds
in the required conversion-boundary resets. Removing the page-derived conversion batch limit or its inter-batch reset
caused reproducible driver timeouts and was reverted; the final runtime is unchanged. With affine plans and conservative
conversion batches, the standalone wide method passes in **23.46 s** pytest time / **25.57 s** process wall time.

- Direct exact indices 2049/2499 regression: pass.
- Complete MaxUnpool class: **4 passed in 34.97 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **366 passed, 10 skipped, 180 subtests passed in
  454.99 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,323/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no host interpretation of an index or pooled value.
Python constructs only static/affine gather geometry and moves raw representations; all dynamic conversion, byte-wise
comparison, selection, and reduction execute on DPU EW. There is no host scatter, host ArgMax, LUT, CMAC, CNA, PPU
fallback, tolerance relaxation, or external non-FP16 input.

---

## 2026-08-09 — equality-row matcher cleanup

Occurrence-count and stable sort-index lowering now share one bounded equality-row parser. It identifies the common
load and ordered candidate loads, validates their dtype and source buffers, evaluates their compile-time gather maps,
and rejects every out-of-range lane. The two IR matchers remain separate because occurrence gates and weighted
value/count conjunctions are different graphs; `_lower_sort_compare` is unchanged because its static MIN/MAX algorithm
does not overlap equality-mask emission. Existing `_ew_eq_mask`, `_stripe_layout`, and `_stripe_gathers` continue to
provide the shared DPU emission and aligned-matrix packing layers.

- Sort-index and TopK regression: **12 passed, 8 subtests passed in 108.12 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Repository-wide Ruff and Tinygrad mypy: pass.
- `sz.py`: renderer/runtime **2,310/281 executable lines**, down from **2,323/281**.

This is a renderer-only structural refactor. It adds no tensor-value access, numeric host operation, runtime behavior,
new tolerance, LUT, CMAC, or fallback path; the generated gather and DPU EW algorithms are unchanged.

---

## 2026-08-09 — representation-safe non-finite MaxUnpool

The remaining FP16 `test_max_unpool2d_inf` pipeline now preserves its exact non-finite output representation. Its fused
schedule stores an internal negative pool-selection weight and applies the spatial-size correction in the MaxUnpool
consumer. The unrolled pool-index matcher now recognizes and proves that form. Because FP16 subtraction and
multiplication are not representation-safe for infinities (`inf-inf` and `inf*0` produce NaN), this narrow path compares
the two raw FP16 bytes of every candidate with the pooled extremum instead.

Generic raw gathers place each byte in the low byte of a zeroed INT32 lane without interpreting it. Native DPU
INT32-input conversion turns the 0..255 byte values into exactly representable FP16 numbers; DPU byte equality selects
the negative first-tie weight. The single-candidate MaxUnpool consumer uses the same mechanism for its pooled FP16
value, multiplies each finite byte value by the DPU-computed equality mask, converts the selected bytes to INT32 on DPU,
and writes their low bytes into the FP16 output representation. Thus selected `+inf`, `-inf`, and NaN bits survive while
every unselected lane becomes exact positive zero. A direct four-plane regression covers `+inf`, `-inf`, NaN, and 3.5;
the upstream pool-to-unpool case verifies the fused internal-index correction.

Historical commit `76c31806e` supplied the original raw-representation selection proof, but its optional NumPy scatter
and operation-specific host packing runtime were not ported. The current implementation uses only the existing generic
RKImage gather phases and DPU EW conversions. `~/npu` documents MaxUnpool as a CPU operator and `~/rk3588` contains no
simpler verified native dynamic-index contract.

Wall-time profiling of a fresh standalone pipeline measured 1.55 seconds of graph construction, 0.012 seconds of
renderer work, and 1.52 seconds for realization plus copyout. The three NPU programs consumed 1.28 seconds, issuing
29 ioctls / 34 tasks; ten INT32 conversions consumed 1.06 seconds and twelve required resets consumed 1.27 seconds
(these timings overlap the program total). A first cold pytest cache miss reached 28.30 seconds, while the cached rerun
reported a 2.12-second test body and 4.64 seconds total. The hardware work itself is therefore well below 30 seconds.

- Direct exact-bit non-finite MaxUnpool and upstream infinity pipeline: pass.
- Complete MaxPool + MaxUnpool regression: **19 passed, 1 skipped, 33 subtests passed in 45.88 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **368 passed, 10 skipped, 180 subtests passed in
  482.49 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,416/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host code only constructs static gather maps and moves raw
bytes without comparing, converting, branching on, or selecting a tensor value. All runtime-dependent equality,
first-tie choice, byte masking, and numeric conversion execute on DPU EW. There is no host scatter, host ArgMax, LUT,
CMAC, CNA, PPU fallback, tolerance relaxation, or external floating-point input wider than FP16.

---

## 2026-08-09 — exact full-range INT32 one-hot equality

The complete upstream `TestOps.test_one_hot` now runs on Rockchip. Its scheduled integer graph is the strict form
`WHERE(dynamic_index != static_class_coordinate, 0, 1)`. The matcher proves that form, the INT32 input buffer and its
static bounds, every input gather offset, every compile-time coordinate, and the signed INT32 coordinate range before
emitting an image.

Historical commit `32cb1cd67` on `rockchip-2607` supplied the original DPU one-hot proof. That version expanded the
dynamic INT32 indices, converted whole values to FP16, and deliberately capped the class extent at 2,048. Its notes
also recorded that native INT32 comparison mode produced invalid masks and identified byte-limb comparison as the
required full-range solution. The current implementation completes that solution: representation-preserving gathers
place each of the four little-endian input bytes in a zeroed INT32 lane, DPU conversion turns each byte into an exact
FP16 integer in 0..255, and a mid-image gather expands those compact byte vectors over the output layout. Four DPU
SUB/ABS/MIN/inversion equality masks are multiplied and converted by DPU to the final INT32 output. Class 2,049 in a
2,050-class output therefore passes exactly instead of depending on FP16 whole-integer precision.

The RKNN v2.3.2 operator guide lists OneHot in its CPU-operator chapter, while `~/npu/include/old/rknn_ops.md` marks it
unsupported. `~/rk3588` contains the upstream test and runtime strings but no verified native DPU OneHot instruction.
No CPU operator or undocumented instruction was imported.

- Complete upstream one-hot case, full-byte out-of-range regression, and class-2,049 regression: **3 passed in 9.31 s**,
  sequentially. The largest focused test passed in **7.35 s**.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **371 passed, 10 skipped, 180 subtests passed in
  460.02 s** pytest time / **485.79 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,467/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The host evaluates only static class coordinates and
moves dynamic bytes by statically proven addresses; it never numerically reads, compares, branches on, or selects from
an input index. All dynamic byte conversion, equality, mask conjunction, and INT32 output conversion execute on DPU
EW. There is no host OneHot, scatter, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than
FP16.

---

## 2026-08-09 — exact dynamic INT32 gather on DPU EW

The unchanged upstream `TestOps.test_gather` now passes all positive and negative axis aliases, seven three-dimensional
gathers, both shape-error cases, a one-dimensional gather, and its `-inf` representation case. A direct regression also
uses dynamic indices `0`, `1`, `2`, `256`, `65,536`, `2**24`, and `-1` against `+inf`, `-inf`, and NaN source lanes;
valid results match bit-for-bit and every out-of-range high-byte alias remains zero.

Tinygrad lowers Gather to one bounds-masked FP16 load whose source address contains a dynamic INT32 load. The strict
matcher proves the zero default, the exact nonnegative/upper-bound predicate, both statically sized source buffers,
the dynamic-index gather map, and every candidate-substituted source address. It then exposes the four raw index bytes
and both raw FP16 source bytes with representation-preserving gathers. DPU INT32-input conversion turns each byte into
an exact FP16 integer, four DPU equality masks select the bounded candidate, and DPU ADD reductions select each FP16
byte independently. DPU INT32-output conversion plus raw post-gathers reconstruct the FP16 result. This preserves NaN
and infinity representations without unsafe `inf*0` masking.

Historical commit `f62827791` implemented gather and fancy indexing with a typed NumPy evaluator, including dynamic
negative-index preprocessing and masked reductions on the host. That implementation violates the current no-CPU-cheat
contract and was not ported. The RKNN v2.3.2 guide lists Gather in its CPU-operator chapter, the older operator table
marks `aten::gather` unsupported, and `~/rk3588` contains no verified native DPU Gather instruction. The current path
therefore composes only proven DPU EW and raw movement primitives.

- Complete unchanged upstream gather: **1 passed in 23.47 s**, sequentially.
- Upstream and exact raw-byte regression together: **2 passed in 24.73 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **373 passed, 10 skipped, 180 subtests passed in
  479.44 s** pytest time / **505.58 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,555/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host evaluation substitutes only compile-time candidate
coordinates into address expressions and validates their static bounds. Runtime moves bytes at those fixed addresses;
it never interprets an index or source value, chooses a candidate, or performs a numeric gather. Dynamic comparison,
selection, reduction, and numeric conversion execute on DPU EW. There is no host Gather, NumPy evaluator, LUT, CMAC,
CNA, PPU fallback, tolerance beyond the established FP16 test tolerance, or floating input wider than FP16.

---

## 2026-08-09 — shared exact INT32 equality-matrix emission

OneHot and dynamic Gather now share one immutable `RKByteEquality` image fragment. It owns raw four-byte extraction,
DPU byte conversion, compact-to-striped mid-gathers, compile-time coordinate-byte packing, and the four DPU equality
masks plus conjunction. The operation-specific matchers remain separate: OneHot still proves its integer WHERE graph,
while Gather still proves its bounds-masked dynamic address. Their terminal emitters continue to differ because OneHot
writes one INT32 mask and Gather uses that mask to select and reconstruct two raw FP16 bytes.

This is the same parser-versus-emitter separation used by the existing `RKLoopReduction` and gather-plan helpers. It
removes duplicated construction without creating a combined IR mega-matcher or changing runtime behavior. `sz.py`
reports renderer/runtime **2,550/281 executable lines**, down from **2,555/281**. The small net reduction includes the
new typed fragment record; roughly sixty formerly duplicated construction lines now have one implementation.

- Complete OneHot plus Gather regression: **5 passed in 31.59 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**.
- Renderer mypy, Ruff, and `git diff --check`: pass.

The refactor adds no tensor-value access, runtime operation, tolerance change, LUT, CMAC, or fallback. Exact dynamic
comparison remains DPU EW work and host code remains limited to compile-time geometry plus raw representation movement.

---

## 2026-08-09 — exact negative single-index fancy indexing

The upstream `TestOps.test_fancy_indexing_inf` and a direct negative-index regression now pass bit-exactly. The direct
case selects `[-1, -2, -3]` from `[+inf, -inf, NaN]`, proving both Tinygrad's negative-index normalization and raw
non-finite representation preservation.

Tinygrad expresses a single fancy index as `WHERE(index < 0, index + extent, index)` inside the bounds-masked source
address. The dynamic-gather matcher now recognizes only that exact normalization. It emits two compile-time coordinate
spellings for every bounded source candidate: `0..extent-1` and `-extent..-1`. Each spelling maps to the same statically
proved source address after candidate substitution, while the shared four-byte DPU equality fragment decides which
runtime spelling matches. Invalid indices match no row and retain the masked-load zero default. No dynamic value is
normalized or bounds-checked on the host.

Historical commit `f62827791` covered broad fancy indexing by executing negative-index preprocessing and multi-index
masked reductions in a typed NumPy evaluator. That CPU implementation remains intentionally unported. This milestone
admits only one dynamic INT32 index and uses the DPU/raw-byte path proven by Gather; multi-index broadcasting and fused
candidate reductions remain separate future work.

- Upstream infinity and exact negative-index regression: **2 passed in 6.04 s**, sequentially.
- Complete OneHot, Gather, and single-index fancy-index regression: **7 passed in 34.28 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **375 passed, 10 skipped, 180 subtests passed in
  482.85 s** pytest time / **510.13 s** process wall time, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `sz.py`: renderer/runtime **2,563/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The host substitutes candidate constants only to prove
static source addresses; raw runtime indices are never read, normalized, compared, or used to choose a source on CPU.
All dynamic normalization equivalence, equality, byte selection, and reconstruction execute through DPU EW. There is
no NumPy evaluator, host gather, LUT, CMAC, CNA, PPU fallback, tolerance beyond the established FP16 test tolerance, or
floating input wider than FP16.

---

## 2026-08-09 — exact multi-index FP16 fancy indexing

The unchanged upstream `TestOps.test_slice_fancy_indexing_with_tensors` now passes all four two-axis broadcast forms,
including its signed case with independent negative indices on both axes. A direct representation regression selects
`+inf`, `-inf`, NaN, `1.0`, `-0.0`, and `+0.0` through two dynamic INT32 index tensors and verifies every FP16 bit.

Tinygrad expresses each dynamic axis as `WHERE(index < 0, index + extent, index)` and combines the normalized values in
one bounds-masked source address. The matcher proves every normalized root, its canonical lower/upper gate, the static
input sizes and broadcast maps, and the complete candidate-substituted source address. It enumerates only compile-time
coordinate tuples; runtime index values remain opaque. The shared equality fragment now emits one exact four-byte DPU
mask per dynamic axis and conjoins those masks on DPU before selecting the two raw FP16 source bytes.

Generalizing the image exposed one stale constant: the mid-gather split was fixed at six operations, exactly the four
byte conversions plus two source-byte conversions needed by one index. With two indices it ran before all eight index
byte conversions completed. The split is now derived as `len(equality.pre_ops) + len(raw_value)`, so it follows the
actual number of dynamic axes instead of encoding the single-index layout.

Historical commit `f62827791` handled broad fancy indexing through a typed NumPy evaluator that read and normalized
dynamic indices on the host. That implementation remains unported. The current runtime only moves raw bytes using
compile-time-proved maps; all dynamic byte conversion, equality, multi-axis conjunction, selection, reduction, and
numeric conversion execute through DPU EW. The local RKNN references still expose no verified native DPU Gather/fancy
index instruction.

- Complete fancy-index class: **4 passed in 24.60 s**, sequentially.
- Shared OneHot, Gather, and fancy-index regression: **9 passed in 52.75 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **377 passed, 10 skipped, 180 subtests passed in
  501.56 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before the complete census and after the final
  candidate-budget guard.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,621/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host evaluation rejects `PARAM` values and substitutes
only compile-time candidates to derive and validate static addresses. Runtime gathers preserve opaque 1/2/4-byte
representations and never interpret an index or floating value. There is no host fancy indexing, dynamic NumPy
evaluation, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — bounded dynamic INT32 Scatter with exact FP16 representations

Rockchip Scatter now accepts one external INT32 index buffer for bounded FP16 tensor-source updates. The exact-bit
regression covers unique destinations and an overlapping destination with last-source-wins semantics. Its base and
source lanes include `+inf`, `-inf`, signed zero, and distinct NaN payloads, and every result is compared as raw
`uint16` rather than through a relaxed floating tolerance.

Tinygrad lowers this Scatter form to a nested WHERE selector. The strict matcher proves one direct dynamic-index load
per candidate, consecutive static candidate positions, static output coordinates, statically bounded base/source
addresses, and the complete last-wins selector truth table. The shared four-byte INT32 equality fragment now accepts a
different statically proved index offset row for each candidate. DPU masks are traversed in reverse candidate order to
construct mutually exclusive effective masks, raw FP16 source/base bytes are selected independently, and DPU
INT32-output conversion plus raw post-gathers reconstruct the exact FP16 output.

Historical direct-Scatter commit `16ea0c339` routed this family through a typed NumPy evaluator that read and computed
tensor values on the host, so it was not ported. `~/npu/include/old/rknn_ops.md` marks ScatterElements unsupported;
`~/rk3588` contains the upstream tests and RKNN runtime strings but no verified native DPU Scatter instruction. This
milestone therefore composes only the already proven DPU EW equality, mask, reduction, and representation-movement
primitives. It intentionally admits only one index buffer with one to eight candidates; wider/general N-D Scatter
graphs remain unsupported until they have a bounded hardware plan.

- Exact dynamic Scatter: **1 passed, 2 subtests passed in 5.43 s**, sequentially.
- Shared OneHot, Gather, fancy-index, and Scatter regression: **18 passed, 2 subtests passed in 56.01 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **378 passed, 9 skipped, 182 subtests passed in
  505.83 s**, sequentially.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the complete census.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,747/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The parser evaluates only static coordinates and an
abstract Boolean selector truth table; it never reads a runtime index, base, or source value. Runtime NumPy code only
moves opaque bytes at compile-time-proved addresses. Dynamic INT32 conversion and equality, last-wins choice, FP16-byte
selection, and reductions all execute on DPU EW. There is no host Scatter, typed tensor evaluator, LUT, CMAC, CNA, PPU
fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — bounded exact FP16 Copysign and forward-only timing census

The complete seven-by-seven upstream Copysign value matrix now passes bit-exactly for finite values, signed zeros,
both infinities, and NaN. The Rockchip regression vectorizes the same 49 ordered pairs into one realization and checks
the raw `uint16` result, making it stricter and much faster than 49 scalar helper invocations.

Tinygrad's signed-zero-aware Copysign graph is recognized before the generic numeric sign rewrite. The strict matcher
proves its `WHERE(sign<0 OR 1/sign<0, -abs(magnitude), abs(magnitude))` structure, direct FP16 loads, static source
bounds, and output gather maps. Raw movement packs magnitude-low, magnitude-high, and sign-high bytes into one striped
matrix. DPU INT32-input conversion maps bytes to exact integers in 0..255, one combined DPU threshold comparison builds
both high-bit masks, and DPU arithmetic computes `magnitude_high - 128*magnitude_sign + 128*source_sign`. The unchanged
low byte and replaced high byte are converted and gathered back as raw FP16 output.

Historical exact-Copysign milestone `751a108e2` extracted tensor sign bits with NumPy in the runtime and was not ported.
`~/npu` and `~/rk3588` expose the ordinary DPU ABS/PReLU register primitives and upstream tests, but no native Copysign
or bitwise FP16 instruction. The current path therefore uses only raw representation movement and proven DPU EW. To
bound the reset-heavy four-lane conversion cost without an arbitrary element count, the exact path is admitted only
when its three-row conversion-tile arena fits one system page; larger finite tensors retain the established numeric
fast path.

Wall-time work removed two successive bottlenecks. The unchanged 49-scalar upstream method initially took **51.33 s**
at ten submits per pair. Packing all required bytes into shared matrices reduced it to **35.37 s**. Vectorizing the
same Cartesian coverage reduced the exact Rockchip test to **3.60 s** pytest time; exact plus the existing broad
Copysign test passed in **5.54 s**, and the complete sign/softsign class passed **7 tests in 7.25 s**.

The duration-enabled full census before removing two backward-only comparison methods passed **379 tests, 9 skipped,
182 subtests in 505.96 s**. Because this backend is explicitly forward-only, `test_cmp_ne_backwards` and
`test_cmp_lt_backwards` were then removed from the Rockchip class; collection now contains 386 nodes instead of 388.
The slowest forward calls were simple CumMin 26.58 s, Gather 20.79 s, wide MaxUnpool 19.98 s, simple CumMax 19.06 s,
and tensor fancy indexing 14.52 s.

An instrumented isolated simple CumMin run took **27.49 s**: renderer compilation 8.56 s, nine NPU program calls
8.43 s, and scheduling/PyTorch/reference overhead 10.31 s. Inside the device calls, 73 soft resets consumed **7.75 s**,
73 submit ioctls consumed only **0.023 s**, and gather/sync/command preparation consumed 0.65 s. BO initialization was
0.19 s and copyout 0.001 s. Future CumMin optimization should therefore reduce reset-separated comparison/conversion
stages; larger command buffers or faster arithmetic cannot materially improve this profile.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before the correctness census and after both full
  census/profile runs.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,813/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Host code recognizes static IR, proves addresses, and
moves opaque bytes; it never reads a sign, classifies a value, or performs Copysign. Byte conversion, high-bit masks,
sign removal/insertion, and output conversion execute on DPU EW. There is no NumPy tensor evaluator, host Copysign,
LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — bounded dynamic scalar Scatter add/multiply

The unchanged forward-only upstream `test_scatter_add` and `test_scatter_mul` now pass on Rockchip for external INT32
indices, including their infinity and NaN cases. A deterministic regression also covers repeated destinations with a
finite scalar, proving that add counts every hit and multiply applies every factor instead of collapsing the mask to a
single Boolean.

Tinygrad unrolls this Scatter family into one guarded equality term per candidate plus the FP16 base load. The strict
matcher proves one dynamic INT32 load per term, its static coordinate expression, its bounds-gated address, the exact
neutral result when unequal, one common scalar FP16 representation when equal, and the statically bounded base map.
The shared four-byte equality fragment builds every runtime mask on DPU. A compile-time validity matrix suppresses
lanes outside the index tensor's shape; DPU ADD or MUL then reduces the candidate rows and combines the result with the
base.

Non-finite scalars require avoiding `0*inf` on no-hit lanes. Conditional infinity is generated entirely on DPU from
finite inputs as `hit * 65504 * 2`. Conditional NaN uses the RK3588's verified `inf * 0` behavior as
`conditional_inf * (1-hit)`, producing NaN only when selected and exact zero otherwise. Add uses a reduced hit count;
multiply constructs one factor per finite candidate, while its infinity/NaN cases use the equivalent any-hit factor.
No host branch depends on a runtime index or FP16 value.

Historical milestone `1ff6ebc9d` admitted FP16 ScatterReduce by routing the complete graph through a typed host
elementwise evaluator, so that implementation was not ported. `~/npu/include/old/rknn_ops.md` marks ScatterElements
unsupported, and `~/rk3588` contains the upstream Scatter tests but no verified native DPU Scatter instruction. The
current implementation instead composes only the existing exact INT32 equality, static gather planning, and DPU EW
arithmetic primitives; it uses neither LUT nor CMAC.

- Repeated-index finite regression: **1 passed, 2 subtests passed in 3.93 s**, sequentially.
- Unchanged upstream Scatter add: **1 passed in 4.72 s**; Scatter multiply: **1 passed in 4.77 s**, sequentially.
- Complete Scatter class: **12 passed, 4 subtests passed in 10.91 s**, sequentially.
- Complete Rockchip census with `ROCKCHIP_EW_REDUCE=twoproduct`: **380 passed, 9 skipped, 184 subtests passed in
  508.16 s**, sequentially. Collection contains 389 forward-only nodes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused hardware tests.
- Repository-wide Ruff and Tinygrad mypy: pass. `git diff --check`: pass. `sz.py`: renderer/runtime **2,923/281
  executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The renderer encodes only compile-time FP16 constants;
static IR evaluation proves output coordinates, gates, and addresses and rejects dynamic parameters. Runtime gathers
move opaque representations according to those plans. Dynamic INT32 conversion, byte equality, validity masking,
hit/factor reduction, infinity/NaN construction, and final FP16 arithmetic all execute on DPU EW. There is no typed
tensor evaluator, host Scatter arithmetic, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider
than FP16.

---

## 2026-08-09 — flipped-eye static movement regression

The unchanged forward-only `TestOps.test_flip_eye_crash` is now part of the Rockchip movement census and passes
`eye(10) @ flip(eye(10), axis=0)` without reproducing its historical crash. Current Rockchip has no CMAC renderer or
runtime path: static Eye/Flip layouts are proved as raw gathers and the small contraction uses the established DPU EW
reduction.

Historical milestone `d134eb1b8` had already recorded this exact method passing without a backend change. `~/npu`
marks native `aten::flip` unsupported, while `~/rk3588` contains the same upstream method but no independent native
Flip instruction example. This confirms that the existing static gather path is the appropriate implementation rather
than adding a special opcode or host movement evaluator.

The adjacent upstream `test_mulacc_with_zero_strides` was investigated but deliberately not admitted. Its historical
fix in `ad990be16` is explicitly an FP32 CMAC plus compensated-DPU implementation, outside this branch's FP16-input,
no-CMAC scope; today its first constant-only case also reaches an unrelated CPU vector-cast compiler error. No
Tinygrad-core workaround or narrower replacement test was added.

- Unchanged flipped-eye method: **1 passed in 3.09 s**, sequentially.
- Complete movement class: **31 passed, 1 skipped in 6.03 s**, sequentially.
- The immediately preceding complete Rockchip census passed **380 tests, 9 skipped, 184 subtests in 508.16 s**; this
  test-only milestone adds the one focused passing node, for 390 collected nodes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the movement-class run.
- Repository-wide Tinygrad mypy and focused Ruff: pass. `git diff --check`: pass. `sz.py` remains renderer/runtime
  **2,923/281 executable lines**.

There is no backend/runtime/Tinygrad-core change and therefore no new CPU arithmetic or tensor-value access. The test
uses the existing compile-time address plan and DPU EW path with no LUT, CMAC, CNA, PPU fallback, tolerance beyond the
established FP16 cap, or floating input wider than FP16.

---

## 2026-08-09 — WIP bounded fixed FP16 masked select

Fixed-size `x.masked_select(x > 0, size=..., fill_value=...)` is lowered as four strict DPU-EW programs: an
IEEE-correct positive prefix count, an INT32 occurrence histogram, an INT32 histogram prefix, and an exact raw-FP16
guarded gather/fill. The matcher admits only a source-derived positive mask, proves every static prefix and address,
and bounds counts to the exact FP16 integer range. The exact regression includes NaN, both infinities, signed zero,
the smallest positive subnormal, and raw-bit checking of selected and fill values.

The initial correct path used 94 submits and 220 tasks for a representative six-input/four-output case. Reusing
whole-vector INT32 conversions and compact raw-source bytes reduced it to 50 submits and 87 tasks. Before the compact
raw-source change, the longest pad-and-truncate test took 7.39 s: 6.75 s was 64 reset ioctls, 0.024 s was 112 actual
submit ioctls, and about 0.61 s was host scheduling, rendering, marshalling, and reference work. Reset separation is
therefore the remaining wall-time bottleneck.

A reset-free ReLU/MIN validity experiment was rejected after it caused an IOMMU read fault and a six-second NPU job
timeout; the renderer was restored to the previously passing comparison path and no further NPU command was issued on
that boot. At this point the compact raw-source form still required focused hardware revalidation; the following
Nonzero milestone records that successful validation.

The cleanup pass follows the other hardware backends' compact immutable-plan style: runtime code remains unchanged,
FP16 representation encoding and INT32 conversion-tile sizing are centralized, an unused scratch constant was
removed, and matchers return only data consumed by their emitters. Host-only rendering produces four valid immutable
RKImages for both finite and special-value cases. Repository-wide Tinygrad mypy and Ruff pass, as does
`git diff --check`; `sz.py` reports renderer/runtime **3,224/281 executable lines**.

The CPU-cheat audit finds changes only in the renderer, tests, and this progress record. Compile-time code inspects
static IR and encodes constants but never reads tensor buffers or evaluates a dynamic FP16/INT32 value. Runtime numeric
semantics, Tinygrad core, tolerances, and the external FP16 contract are unchanged. There is no typed host evaluator,
LUT, CMAC, CNA, or PPU fallback.

---

## 2026-08-09 — bounded fixed FP16 Nonzero

Fixed-size FP16 `nonzero(size=..., fill_value=...)` now passes for rank-one truncation and padding, rank-two coordinate
selection, signed zeros, finite negative values, NaN, infinity, empty inputs, and scalar output shape. Tinygrad lowers
this operation through four programs: a repeated FP16 nonzero prefix count, the existing exact INT32 occurrence
histogram, its INT32 prefix, and a guarded coordinate selector. The new final matcher proves the complete source
nonzero count, dynamic prefix-index buffer, every static coordinate row, every validity threshold, and an exactly
FP16-representable integer fill before emitting an image.

The count and final selection remain DPU computations. Native ABS plus comparison creates the nonzero mask, DPU ADD
reduces it, the shared four-byte INT32 equality plan selects the runtime prefix indices exactly, and DPU arithmetic
combines static coordinate rows with the dynamic validity mask before exact INT32 output conversion. An initial
hardware run returned all-zero coordinates because the new image omitted the shared equality plan's FP16 `1`
constant; adding that immutable image constant fixed the result without a runtime or tolerance change.

Historical nonzero milestones `ae5a4a6f6` and `add791a62` evaluated complete tensor expressions with NumPy in the
runtime and were not ported. `~/npu/include/old/rknn_ops.md` marks native NonZero unsupported, and `~/rk3588` has no
verified native NonZero instruction. The current implementation instead composes the already proven prefix,
byte-equality, gather-plan, and DPU EW primitives.

- Fixed Nonzero class: **4 passed in 11.44 s**, sequentially; every individual method passed under 30 seconds.
- Adjacent fixed MaskedSelect regressions: **2 passed**, individually in **8.94 s** and **5.65 s**. This closes the
  post-reboot validation left pending by the preceding WIP milestone.
- One rank-one truncate realization: input realization 83 ms, graph construction 9 ms, scheduling 131 ms, rendering
  224 ms, four NPU programs 1.93 s, and copyout 3 ms. It used **51 submit ioctls / 69 DPU tasks**; the exact final
  selector was the largest program at 962 ms.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before focused work, after the initial numerical
  failure, and after the complete focused run. No new RKNPU timeout or IOMMU fault appeared in the kernel journal.
- Repository-wide Tinygrad mypy and Ruff, host compilation of every Nonzero program, and `git diff --check`: pass.
  `sz.py`: renderer/runtime **3,331/281 executable lines**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Static IR evaluation only proves addresses, coordinate
tables, and selector truth tables after replacing abstract dynamic nodes with validation constants; it never reads a
runtime source, prefix, or index buffer. Runtime gathers move opaque representations according to those plans.
Dynamic nonzero classification, INT32 equality, coordinate selection, validity masking, and output conversion all
execute on DPU EW. There is no typed host tensor evaluator, host Nonzero arithmetic, LUT, CMAC, CNA, PPU fallback,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — multidimensional dynamic FP16 Scatter

Bounded last-wins Scatter with an external INT32 index tensor and FP16 tensor source now covers every axis of a
three-dimensional `(2,3,4)` tensor. The raw-bit regression includes infinities, NaNs, and signed zero, deliberately
maps every candidate along the scatter axis to the same destination, and checks the final FP16 representations
exactly. This proves multidimensional index addressing, source addressing, collisions, and last-wins ordering rather
than only the earlier one-dimensional special-value cases.

The existing strict Scatter matcher assumed each unrolled dynamic index load addressed one globally constant index
position. Multidimensional codegen instead emits one candidate load whose offset varies across the non-scatter axes.
The matcher now retains the complete per-output offset row, sorts candidate rows lexicographically, proves their union
covers every external index element, and proves candidate offsets are strictly increasing for every output
lane. The unchanged exact Scatter image then applies four-byte INT32 equality, raw FP16 byte selection, and DPU
last-wins masks to those proven rows.

Historical Scatter milestones `16ea0c339`, `262622ff0`, and `1ff6ebc9d` used typed NumPy tensor evaluators and were not
ported. `~/npu/include/old/rknn_ops.md` marks native ScatterElements unsupported, while `~/rk3588` contains the upstream
tests but no independent native Scatter instruction example. The implementation therefore extends only the existing
immutable DPU plan instead of adding a host fallback.

- Three-dimensional dynamic tensor-source Scatter: **1 passed, 3 subtests passed in 7.22 s**, sequentially, covering
  axes 0, 1, and 2.
- Complete Scatter class: **13 passed, 7 subtests passed in 15.33 s**, sequentially.
- One representative axis-1 realization: input realization 100 ms, graph construction 3 ms, scheduling 106 ms,
  rendering 120 ms, one NPU program 1.29 s, and copyout 3 ms; **49 submit ioctls / 135 DPU tasks**.
- A discarded six-axis stress form completed axes 0/1/2/-1 but timed out on the redundant `-2` alias after five
  reset-heavy programs in one process. The retained test removes normalized-axis duplicates; it and the full class
  pass from fresh processes. Vendor `elementwise.py` passed **60/60** immediately after the timeout and after the
  retained class, confirming the driver remained usable.
- Focused Ruff and mypy, `git diff --check`, and host rendering for positive and normalized axes: pass. `sz.py`:
  renderer/runtime **3,330/281 executable lines**, one renderer line smaller than the preceding milestone.

The CPU-cheat audit found no runtime or Tinygrad-core change. Static IR evaluation proves only coordinate expressions,
buffer bounds, complete index coverage, and last-wins truth tables. It does not read an index, source, or base tensor.
Runtime gathers move opaque bytes; dynamic INT32 equality, collision masking, FP16 selection, and output construction
execute on DPU EW. There is no typed host Scatter, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating
input wider than FP16.

---

## 2026-08-09 — no-LUT FP16 round-to-even

The complete forward `test_round` group is now in the Rockchip census. Tinygrad's exact round-to-even graph is
recognized before its TRUNC/FLOOR/CEIL expansion and replaced with a DPU-only composition: native FLOOR obtains the
integer and fractional parts, native ABS and positive-mask stages detect `fraction > 0.5` and exact ties, the existing
native FLOOR/CEIL truncation composition determines parity, and DPU MUL/MAX/ADD applies the one-lane increment. This
avoids the historical ROUNDOFF LUT and the older typed host conversion fallback.

The first generic WHERE lowering passed ordinary ties but mapped infinities to NaN. A first specialized formula based
on `floor(x+0.5)` fixed nonfinite values but failed 1,025 FP16 encodings because the `+0.5` itself rounded before FLOOR,
including odd integers above 1,024. Computing `floor(x) + increment` from the fractional part instead is value-exact
over all 65,536 FP16 encodings. It preserves infinity signs and NaN locations; like the upstream value-level contract,
it does not promise raw NaN payloads or the sign bit of results numerically equal to zero.

The independent `~/rk3588/examples/elementwise_int.py` reference was also translated from its Rocket wrapper to this
machine's RKNPU ioctl ABI in a temporary probe. FP16 comparison to INT16 and INT16 ADD to INT32 both passed exactly.
Those modes were not added to the backend because a conservative implementation must retain the existing exact
byte/digit path for indices above signed INT16 range, so it would currently add machinery rather than remove it.

- Upstream `test_round`: **1 passed in 5.77 s**; adapted forward-only quantization expression: **1 passed in 3.59 s**.
- Exhaustive round regression: **all 65,536 FP16 encodings passed numerically**, **10 submit ioctls**, about **1.41 s**
  operator wall time and **4.10 s** under isolated pytest.
- Complete integral-rounding class: **7 passed in 7.20 s**, sequentially.
- Representative 45x35 round: input realization 96 ms, graph construction 1 ms, scheduling 44 ms, rendering 114 ms,
  DPU execution 642 ms, and copyout 3 ms; **7 submit ioctls**, 900 ms total.
- Rockchip collection is **400 tests**. Repository-wide Tinygrad mypy, Ruff, and `git diff --check` pass. `sz.py`:
  renderer/runtime **3,365/281 executable lines**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete focused run.

The CPU-cheat audit found no runtime or Tinygrad-core change. The renderer only matches static IR structure and emits
DPU EW stages; it never reads or numerically evaluates a tensor buffer. There is no LUT, CMAC, CNA, PPU fallback,
typed host rounding, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — dimension-injected dynamic FP16 fancy indexing

The three forward cases from `TestOps.test_slice_fancy_indexing_dim_inject_and_collapse` now run on Rockchip. They
combine dynamic INT32 tensor indices with collapsed integer dimensions, inserted `None` dimensions, and an ellipsis.
The original upstream method exceeded the 30-second per-test policy when all three cases shared one item, so the exact
expressions are retained as three independently bounded Rockchip methods.

Tinygrad emits two graph forms for this group. One is a complete Cartesian ADD of masked source loads; the new strict
matcher proves every canonical negative-normalization root, every equality predicate, the unique complete coordinate
product, all source/index buffer bounds, and every candidate address. The other form keeps one bounds-masked dynamic
load and is handled by the existing multi-index matcher. Positive coordinate equality and its equivalent negative
spelling are now ORed per axis on DPU before the axis masks are conjoined, so a 5x6 selection needs 30 candidate rows
instead of duplicating all four positive/negative sign combinations into 120 rows.

The first implementation was correct but too slow. A representative first case realized in **48.735 s**: INT32
conversion consumed **48.162 s**, including **45.668 s in 434 soft resets**; its 521 submit ioctls consumed only
0.186 s. Raw source bytes had been duplicated over every candidate row before conversion. Converting each statically
reachable source lane once and then striping the converted bytes through a mid-image raw gather reduced realization to
**4.619 s**, conversion to **3.858 s**, resets to 36, submits to 123, and tasks from 6,099 to 519.

Output-byte reconstruction exposed the same issue in the 3,000-lane ellipsis case. The decoded
`~/rk3588/examples/elementwise_int.py` FP16-to-INT16 recipe was translated to the current RKNPU ioctl ABI and extended
from comparison masks to neutral MAX conversion. Exact probes passed for 3, 8, and 3,000 FP16 integer lanes spanning
0 through 255. RKImage v27 therefore records a native INT16 output stage. DPU converts the selected numeric byte lanes
to contiguous INT16 in full-width tasks; runtime copies only their opaque low bytes into the FP16 result. A submit
boundary before each precision transition is required and covered by the nonfinite regression. This reduced the
ellipsis realization from **17.611 s** to **5.517 s**, resets from 156 to 48, submits from 201 to 93, and tasks from
2,161 to 663.

`~/npu/include/old/rknn_ops.md` marks advanced indexing and `aten::index` unsupported, and `~/rk3588` contains no native
Gather/fancy-index instruction example. Historical broad support used a typed NumPy evaluator and remains unported.
The current implementation instead extends the immutable equality/gather image shared by OneHot, Gather, Scatter,
MaskedSelect, and Nonzero.

- Complete fancy-index class: **7 passed in 23.07 s**, sequentially; every individual test is below 30 seconds.
- Complete Gather class: **2 passed in 14.46 s**. Exact full-byte OneHot and dynamic Scatter regressions also pass.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the final focused runs. No new RKNPU timeout,
  invalid IRQ, IOMMU fault, or kernel oops appeared.
- Rockchip collection: **403 tests**. Repository-wide Tinygrad mypy (216 files), Ruff, and `git diff --check`: pass.
  `sz.py`: renderer/runtime **3,463/282 executable lines**, total **28,800**.

The CPU-cheat audit found no Tinygrad-core change and only one runtime executable-line change, forwarding the immutable
INT16-output flag to command emission. Compile-time code proves UOp structure, integer coordinates, and raw addresses;
it never reads a dynamic index or FP16 value. Runtime gathers move opaque 1/2/4-byte representations. Dynamic INT32
conversion, positive/negative equality, axis conjunction, FP16-byte selection, reduction, and INT16 writeback all run
on DPU EW. There is no host fancy indexing or numeric conversion, LUT, CMAC, CNA, PPU fallback, tolerance relaxation,
or floating input wider than FP16.

---

## 2026-08-09 — native INT16 EW and integer-dimension fancy indexing

The remaining five forward cases from upstream `test_slice_fancy_indexing_dim_collapse_int` now run on Rockchip. They
cover a collapsed leading integer, a collapsed middle integer, three scalar dimensions, sparse dynamic axes, and a
mixed static-slice/dynamic-index form. Together with the preceding milestone, the complete Rockchip fancy-index class
now contains all twelve focused methods and every one passes individually under 30 seconds.

This milestone productizes the integer path proved by `~/rk3588/examples/elementwise_int.py` instead of adding more
FP16 integer emulation. RKImage v28 records native INT16 input and output independently; command emission configures
INT16 input/process/output precision, eight-lane geometry, INT16 RDMA precision, two-byte surface grouping, and the
integer converter mode. Runtime keeps INT16, INT32, and FP16 PC chains precision-homogeneous. An exact signed INT16 ADD
probe through Tinygrad's RKImage and RKNPU runtime passed eight lanes in one submit ioctl.

Arbitrary external INT32 indices are still compared without narrowing: each is decomposed as four opaque unsigned
bytes, and native INT16 SUB/ABS/MIN/MUL combines the exact byte equalities. Bounded-index masks, negative-normalized
linear coordinates, and raw FP16 byte selection now use this shared helper. Only final public INT32 storage is
regrouped from two-byte results into four-byte lanes; ROCKCHIP's public tensor dtype contract remains FP16.

- Five new integer-dimension cases: **5/5 passed individually**, from **3.39 s** to **12.99 s** pytest time.
- Complete twelve-method fancy-index class and both Gather regressions: every method passed individually; nonfinite
  outputs retain exact FP16 bit patterns.
- Representative largest case: **4.796 s** direct wall time, including **4.788 s** realization and 2.6 ms copyout;
  **3 submit ioctls / 4,115 DPU tasks / 0 soft resets**. The temporary native-INT32 implementation needed 9.252 s,
  391 submits, and 45 resets; the earlier FP16-emulation form needed 53.96 s, 1,013 submits, and 444 resets.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before the complete census and after each timeout
  investigation. Repository-wide Tinygrad mypy and Ruff plus `git diff --check` pass.
- Rockchip collection: **408 tests**. A complete serial census reached **151 passed / 7 skipped / 100 subtests passed**
  before the pre-existing multidimensional Scatter method timed out at `dim=1`. The same isolated method and subtest
  reproduce unchanged at baseline commit `dd6c694d1` after a passing vendor health check, proving it is not an INT16
  regression. Focused milestone tests do not time out.
- `sz.py`: renderer/runtime **3,682/307 executable lines**, total **29,044**. Runtime remains smaller than QCOM's 312
  executable lines; the renderer is net +219 lines over the preceding milestone for five cases and the reusable
  native INT16 pipeline.

The CPU-cheat audit found no Tinygrad-core change and no runtime tensor arithmetic. Compile-time code validates UOp
structure, static coordinates, and bounds. Runtime gathers move opaque bytes according to immutable plans; dynamic
byte equality, validity masks, weighted coordinates, selection, and reductions execute on DPU EW. There is no typed
host index/fancy-gather evaluator, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than
FP16.

---

## 2026-08-09 — no-collapse dynamic FP16 fancy indexing

All five forward cases from upstream `test_slice_fancy_indexing_no_dim_collapse` are now separate bounded Rockchip
methods. They cover five broadcast dynamic axes, static outer dimensions around three dynamic axes, trailing and
spanning ellipses, and a dynamic middle axis surrounded by static slices.

Four forms already lowered through the native-INT16 fancy-index image. The static-outer form exposed one obsolete
guard in `_lower_multi_fp16_fancy_index`: it rejected a complete candidate matrix over 64K lanes even though the
shared image builder now partitions candidates into independently bounded blocks. Removing that pre-blocking guard
lets the existing dynamic blocker calculate each safe matrix; runtime command and descriptor sizes remain derived
from the emitted bodies rather than a hardcoded candidate count.

- Five no-collapse cases: **5/5 passed individually**, from **6.15 s** to **17.87 s** pytest time.
- Largest direct profile: **9.206 s** total, including **9.198 s** realization and 2.6 ms copyout;
  **3 submit ioctls / 7,933 DPU tasks / 0 soft resets**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. `sz.py`: renderer/runtime
  **3,680/307 executable lines**, total **29,042**; this milestone removes two renderer lines while adding coverage.

The CPU-cheat audit found no runtime or Tinygrad-core change. The removed guard changes only static plan acceptance;
dynamic INT32 byte equality, negative normalization, raw FP16 selection, and reduction continue to execute through
native INT16 DPU EW. Runtime gathers only move opaque bytes according to compile-time address plans. There is no host
fancy indexing, LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — dimension-injected dynamic FP16 fancy indexing census

All eleven forward expressions from upstream `test_slice_fancy_indexing_dim_inject_none` are now independently
bounded Rockchip methods. They cover leading, trailing, internal, paired, and sparse `None` dimensions around as many
as four broadcast dynamic INT32 index tensors, plus the static all-`None` movement form.

Every case lowered through the existing native-INT16 fancy-index plan; no backend special case was needed. This is the
intended payoff from productizing INT16 as an RKImage/runtime precision rather than adding a matcher-specific probe.

- Dimension-injection group: **11/11 passed individually**, from **2.82 s** to **19.95 s** pytest time; no case
  exceeded the 30-second policy.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the complete focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **424 tests**.
- `sz.py` is unchanged at renderer/runtime **3,680/307 executable lines**, total **29,042**; tests and comments are not
  counted.

The CPU-cheat audit found no implementation change at all in this milestone. The tested path retains DPU-native INT16
byte equality, validity masks, raw FP16 selection, and reduction with opaque runtime movement only. There is no CPU
index evaluation, LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — mixed list/tensor fancy indexing

The four forward expressions from upstream `test_slice_fancy_indexing_list_with_tensors` are now independent Rockchip
methods. They cover one tensor wrapped in a Python index list, a tensor plus a collapsed scalar, a tensor plus a static
tuple index, and five broadcast tensor indices.

Three forms already passed. Tinygrad materializes the apparent static tuple `(1,1)` as a second external INT32 buffer,
so the fourth form exposed a real matcher gap: multi-index lowering accepted only negative-normalized axes. The matcher
now also accepts a direct positive-only INT32 load when the complete gate proves canonical nonnegative and strict upper
bounds. It still requires every data-index and gate load to be uniquely accounted for. Positive-only axes omit the
negative-coordinate alternative; normalized axes keep it.

- Mixed list/tensor group: **4/4 passed individually**, from **11.73 s** to **12.25 s** pytest time.
- The full five-negative-normalized-axis regression also passed after the matcher generalization.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **428 tests**.
- `sz.py`: renderer/runtime **3,693/307 executable lines**, total **29,055**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The second external INT32 buffer is never inspected on
host: runtime gathers its opaque bytes and native INT16 DPU EW performs exact four-byte equality, axis conjunction,
selection, and reduction. There is no host tuple/index evaluation, LUT, CMAC, tolerance change, or floating input
wider than FP16.

---

## 2026-08-09 — static-list and dynamic-tensor fancy indexing

All seven forward expressions from upstream `test_slice_fancy_indexing_list_indices` are now independent Rockchip
methods. They cover a static list alone, leading and trailing static lists mixed with dynamic tensors, broadcast list
shapes, negative static indices, column-shaped lists, and multiple static/dynamic axes.

Python lists and tuples become external INT32 buffers in Tinygrad's scheduled IR. Two strict matchers therefore needed
the same positive-only extension already used by the final FP16 selector. The bool bounds image now accepts direct
INT32 loads only when their canonical lower and upper gates cover every conjunction leaf. The flattened INT32 index
image accepts a direct contribution only when row-major compile-time strides prove a finite candidate extent and the
complete source index remains below signed INT16 range. Negative-normalized axes retain their signed alternatives.

- Static-list group: **7/7 passed individually**, from **4.15 s** to **15.77 s** pytest time.
- The preceding mixed tensor/tuple and five-negative-axis regressions pass after the shared matcher changes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **435 tests**.
- `sz.py`: renderer/runtime **3,710/307 executable lines**, total **29,072**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Static stride analysis enumerates only candidate values;
it never reads the list/tuple buffers. Runtime moves opaque bytes and native INT16 DPU EW computes exact byte equality,
bounds conjunction, weighted index reconstruction, FP16 selection, and reduction. There is no host fancy indexing,
LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — tuple fancy indexing

All six forward expressions from upstream `test_slice_fancy_indexing_tuple_indices` are now independent Rockchip
methods. They cover nested static tuples, positive and negative leading tuples mixed with dynamic tensors, trailing
tuples, column-shaped tuples, and tuple indices around an inserted `None` dimension.

Five forms reused the static-list milestone unchanged. The final form lowers to a 3,400-node unrolled selector because
Tinygrad materializes its tuples as external INT32 buffers. The strict unrolled matcher now recognizes direct
positive-only axes only when equality predicates prove a complete contiguous coordinate domain and the complete
Cartesian candidate product is present exactly once. Negative-normalized axes retain their signed alternatives.

- Tuple-index group: **6/6 passed individually**, from **2.94 s** to **8.41 s** pytest time.
- The original five-axis all-negative-normalized unrolled regression passes after the matcher extension.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **441 tests**.
- `sz.py`: renderer/runtime **3,724/307 executable lines**, total **29,086**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Direct tuple buffers remain opaque on host; compile-time
IR predicates prove only their allowed coordinate domains. Native INT16 DPU EW performs exact byte equality,
conjunction, FP16 selection, and reduction. There is no host tuple evaluation, LUT, CMAC, tolerance change, or floating
input wider than FP16.

---

## 2026-08-09 — public bounded INT16 DPU EW

The native INT16 precision introduced for index masks is now reachable from ordinary Tinygrad signed-INT16 elementwise
graphs. The implementation follows `~/rk3588/examples/elementwise_int.py`: DPU input, process, and output precision are
INT16, while ADD, SUB, MUL, MAX, MIN, ABS, and NEG use the decoded integer ALU encodings. Tinygrad's portable
`ADD(x, MUL(y, -1))`, XOR-based minimum, and WHERE-based absolute-value forms are folded back to native operations.
Static broadcasts reuse the same immutable raw-gather plans as FP16.

The path deliberately does not add INT16 to `supported_dtypes`: RK3588 saturates signed integer ADD/SUB/MUL/ABS/NEG,
whereas Tinygrad's general fixed-width integer contract wraps on overflow. Dedicated limit tests record the hardware
behavior exactly instead of falsely advertising a fully general INT16 backend.

- Five methods / **13 DPU programs passed individually**, including arithmetic, broadcasting, chaining, and saturation;
  every program asserts exactly **one submit ioctl**.
- Largest direct profile, 131,072-element ADD: **0.345 s total** = 242.2 ms device open + 8.7 ms input creation +
  0.1 ms graph construction + **91.4 ms realization** + 2.6 ms copyout; **1 ioctl / 3 DPU tasks**.
- FP16 broadcast and native-INT16 tuple-fancy regressions pass after sharing gather validation/cache helpers.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **446 tests**.
- `sz.py`: renderer/runtime **3,855/307 executable lines**, total **29,217**. Shared gather validation and immutable-plan
  retargeting remove duplicated FP16/INT16 bounds and cache plumbing.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time evaluates only static shapes, constants, and
gather addresses. Runtime copies opaque `uint16` lanes; all input-dependent signed arithmetic, extrema, absolute value,
negation, and saturation execute on DPU EW. There is no host numeric evaluator, LUT, CMAC, CNA, PPU fallback, tolerance
relaxation, or floating input wider than FP16.

---

## 2026-08-09 — zero-stride FP16 mulacc census

All three forward expressions from upstream `test_mulacc_with_zero_strides` now run on Rockchip: a scalar expanded over
`(2,4,3)` and reduced on the last axis, two `(2,4,3)` zero-stride broadcasts reduced over axes `(0,2)`, and the `1x2 @
2x3` matrix product. The upstream method's first and third expressions are entirely constant and therefore fold to a
CPU realization before reaching a device backend. The Rockchip method preserves the same shapes and algebra but binds
external FP16 buffers, then asserts exact results and the four resulting NPU submit ioctls.

Historical commit `ad990be16` solved the old normal-FP32 form with compensated CMAC; it was intentionally not ported.
The current FP16 backend already lowers these cases through raw broadcast gathers, DPU MUL/ADD reduction, and the
existing DPU-only dot path. `~/rk3588/examples/elementwise.py` independently proves the underlying MUL and ADD stages.

- Focused pytest: **1 passed in 2.95 s**, below the 30-second policy.
- Direct wall decomposition, all three cases: **0.317 s total**, including 58.6 ms input setup. Case realizations were
  **87.1 / 48.2 / 23.2 ms**, copyouts 2.4 / 0.7 / 0.6 ms, with **2 / 1 / 1 ioctls** and **2 / 4 / 3 tasks**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused run.
- Repository-wide Tinygrad mypy and Ruff plus `git diff --check`: pass. Rockchip collection: **447 tests**.
- `sz.py` remains renderer/runtime **3,855/307 executable lines**, total **29,217**; no backend implementation was added.

The CPU-cheat audit found that all six operands are realized external FP16 Rockchip buffers before timing/submission.
Runtime performs only raw layout gathers; every value-dependent multiplication and addition executes on DPU EW. There
is no CPU constant-fold result, CMAC, LUT, host reduction, tolerance relaxation, or Tinygrad-core change.

---

## 2026-08-09 — large boolean ALL reductions

The exact upstream `test_all_large` method now covers `2**15`, `2**16`, and `2**20` on Rockchip. Tinygrad's ordinary
non-local schedule expands the first `2**20` reduction into enough scalar UOps to consume more than five minutes and
about 2.1 GiB before rendering. A cache-distinct `ROCKCHIP:BOOL` renderer mode instead accepts Tinygrad's compact
16-lane grouped loop and proves its static launch coordinates form a complete source permutation before emitting DPU
work. The default Rockchip renderer remains non-local, so unrelated elementwise, pooling, convolution, and GEMM
schedules are unchanged.

Contiguous FP16 blocks use a direct in-place DPU tree: FP16 nonzero comparison produces the mask, MAX/MUL performs
ANY/ALL, and the existing DPU conversion exposes the exact bool byte. Tinygrad's second reduction reads an opaque bool
buffer; bytes are widened into zeroed INT16 lanes, reduced by native INT16 MAX/MUL, then copied back as raw bool bytes.
Non-contiguous grouped axes retain the proven striped-gather path. Historical `03bad6205` used CMAC plus host
bool/FP16 conversion, and `~/rk3588`/`~/npu` contain no pure-bool reference, so neither older approach was ported.

- Exact upstream focused pytest: **1 passed in 9.70 s** for all three sizes; the post-cleanup rerun passed in **11.84 s**.
- Largest direct profile (`2**20`): **5.179 s total** = 1.2 ms graph construction + **5.174 s realization** + 3.4 ms
  copyout; **26 ioctls / 3,195 DPU tasks**. Code generation alone is **0.372 s**, down from over five minutes.
- Existing scalar and non-contiguous-axis ANY/ALL regressions pass individually after the grouped matcher addition.
- Full serial Rockchip census: **439 passed, 9 skipped, 187 subtests passed** in **679.12 s**; collection is **448 tests**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the census. The first attempt exposed and led
  to termination of a leaked host-only scheduler experiment; it was not an NPU timeout or wedge.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **3,983/307 executable lines**, total **29,345**. Shared contiguous block-tree emission
  removes the duplicate FP16/INT16 reduction loop.

The CPU-cheat audit found only compile-time evaluation of scheduler coordinates and a bijection proof over integer
offsets. Runtime gathers never inspect values: they move opaque FP16 or bool bytes. Every input-dependent comparison,
AND/OR reduction, and output conversion executes on DPU EW. There is no host numeric reduction, CMAC, LUT, tolerance
change, floating input wider than FP16, or Tinygrad-core modification.

---

## 2026-08-09 — dynamic FP16 threshold MaskedSelect

The data-dependent half of upstream `test_masked_select` now passes at its exact `(32,10)` shape and `x > 0.5`
predicate. Scalar predicate totals and prefix scans accept one proven uniform FP16 threshold. DPU SUB shifts the
threshold to zero before the existing IEEE positive-mask stages; DPU ADD produces the dynamic count and prefix. Prefix
matrices are divided into blocks derived from `_MAX_EW_ELEMS_FP16`, compacted through immutable scratch gathers, and
converted to INT32 only at the terminal stage. Tinygrad's local-register histogram prefix is normalized back into the
existing verified unrolled prefix image, and the final raw-FP16 selector applies the same threshold before exact byte
selection.

The exact seeded case selects 118 of 320 values and passes in **12.83 s** under pytest. Direct wall decomposition is
**10.004 s total** = 3.4 ms input construction + **1.676 s dynamic shape/count/prefix** + **8.324 s final realization**
+ 1.1 ms copyout/compare; it uses **54 submit ioctls / 2,347 DPU tasks**. A scalar-`True` dynamic broadcast over 32
values also passes in **5.60 s**. The two existing fixed MaskedSelect methods and all four adjacent fixed Nonzero
methods pass individually, with the longest at **9.01 s**. Rockchip collection is now **450 tests**.

The remaining scalar-`True` half of the upstream method at 320 values is not claimed: constant folding fuses three
redundant 320-lane reductions, and generic `devectorize2` grows from 2,061 compact nodes beyond the 30-second/roughly
2-GiB host budget before the Rockchip renderer is called. Historical commit `797611d18` solved this with an early
native-program hook that replaced the topology with a typed flat copy. That hook changes Tinygrad core and is therefore
not ported under this branch's no-core-change rule. `~/rk3588` and `~/npu` provide the underlying EW primitives but no
standalone MaskedSelect instruction or backend-side pre-codegen hook.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after all focused NPU tests.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **4,081/307 executable lines**, total **29,443**. The cleanup shares threshold parsing,
  walks each prefix term once, and avoids reparsing the final 320-term predicate.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile-time logic reads only UOp structure, shapes,
constants, and statically proven gather addresses. Dynamic FP16 comparison, predicate count, prefix addition, INT32
histogram/index normalization, equality, and raw FP16 selection execute on DPU EW. Python reads only the DPU-computed
scalar required by Tinygrad's dynamic-output API; it does not compute a tensor value. There is no host numeric
evaluator, LUT, CMAC, CNA, PPU fallback, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT32 MaskedSelect through native INT16 EW

The exact upstream `test_masked_select_size` method now passes on Rockchip. This milestone builds on the public bounded
INT16 DPU EW path from `c359841b5`, rather than adding another FP16 exact-integer encoding. Following
`~/rk3588/examples/elementwise_int.py`, RKImage and the runtime now support native INT16 input/process followed by
terminal INT32 writeback. Exact regressions cover both an arithmetic result and a bare INT16-to-INT32 cast; each uses
one submit ioctl.

Tinygrad's external bool prefix is widened from opaque bytes into zeroed INT16 lanes and accumulated exactly on DPU.
The fixed selector compares each of the four bytes of arbitrary external INT32 indexes with native INT16 SUB/ABS/MIN,
reduces the equality matrix, applies the DPU-computed bool count for pad/truncate validity, and moves the four selected
value bytes back to the external INT32 output. Values do not need to fit INT16: only their individual unsigned bytes
enter the integer ALU.

- Exact upstream `test_masked_select_size`: **1 passed in 5.18 s**, covering output sizes 0/2/4/8, `-1` padding,
  truncation, empty INT32 fill, and float dtype preservation.
- All five Rockchip MaskedSelect methods pass individually in **5.18–12.74 s**; all four adjacent fixed Nonzero
  methods pass in **2.91–7.09 s**. No case crosses the 30-second profiling threshold.
- The native INT16 arithmetic/saturation/broadcast/writeback regression set passes, including exact direct cast.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **452 tests**.
- `sz.py`: renderer/runtime **4,194/325 executable lines**, total **29,574**.

The CPU-cheat audit found no runtime value inspection or Tinygrad-core change. Renderer logic proves only UOp structure,
static bounds, and gather addresses. Dynamic bool accumulation, INT32-index byte equality, selection, padding, and
INT16-to-INT32 conversion all execute on DPU EW. Runtime only submits stages and copies opaque bytes. There is no host
numeric evaluator, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT32 fixed-size Nonzero

The exact upstream `test_nonzero_size` method now passes on Rockchip for ordinary INT32 tensors. Each arbitrary INT32
source value is treated as four opaque bytes. Native INT16 MIN maps each byte to a 0/1 nonzero flag and native MAX
combines the four flags. DPU INT16 ADD then produces the prefix/count, exact byte equality selects the compact
coordinate, and the terminal INT16-to-INT32 stage writes the result. This covers negative values and values that cannot
be represented numerically in INT16 without narrowing the source tensor.

Historical `ae5a4a6f6`/`add791a62` nonzero implementations were inspected but deliberately not ported: they execute the
reduction with NumPy in the Rockchip runtime. The hardware emission instead follows the INT16 and regrouping behavior
proved by `~/rk3588/examples/elementwise_int.py`; `~/npu` has no standalone nonzero instruction. FP16 and INT32
fixed-Nonzero lowerers now share one strict graph-plan parser, matching the matcher/emitter separation used by other
Tinygrad hardware backends.

- Exact upstream `test_nonzero_size`: **1 passed in 5.50 s**, covering truncate/pad, rank two, scalar shape, empty
  input, and dtype preservation.
- A stronger raw-bit regression over `0`, `-1`, `256`, `-65536`, `INT_MAX`, and `INT_MIN` passes exactly in **3.79 s**.
- All six Rockchip Nonzero methods pass individually after cleanup in **2.97–7.06 s**; no case crosses 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **454 tests**.
- `sz.py`: renderer/runtime **4,308/325 executable lines**, total **29,688**. The shared fixed-Nonzero plan removes the
  duplicated FP16/INT32 count, compact-index, coordinate, fill, and bounds parsing.

The CPU-cheat audit found no runtime or Tinygrad-core change. Runtime gathers copy opaque bytes but never interpret a
tensor value. All input-dependent INT32 nonzero detection, prefix/count reduction, exact index equality, coordinate
selection, padding, and INT32 writeback execute on DPU EW. There is no host numeric evaluator, LUT, CMAC, tolerance
change, or floating input wider than FP16.

---

## 2026-08-09 — exact INT32 wraparound ADD through native INT16 EW

Bounded arbitrary INT32 ADD now executes exactly modulo `2**32` without an INT32 arithmetic assumption. Each external
INT32 operand is gathered as four opaque unsigned bytes into native INT16 lanes. DPU INT16 ADD accumulates each byte,
SUB/MAX/MIN constructs the carry, and the four result bytes are regrouped into the external INT32 output. This avoids
the RK3588 native INT32 saturating-ADD mismatch while reusing the public INT16 EW path introduced by `c359841b5`.

The FP16 predicate-prefix emitter was also separated from its original fixed-MaskedSelect parser. It now accepts a
statically proven arbitrary predicate/address matrix and blocks rows from `_MAX_EW_ELEMS_FP16`; the existing dynamic
threshold MaskedSelect regression still passes. This is emitter/matcher separation consistent with the other hardware
backends rather than another operation-specific arithmetic copy.

- Exact wrap regression includes `INT_MAX+1`, `INT_MIN-1`, multioperand carries, and a result crossing the sign bit;
  it passes in **2.82 s** and uses **one submit ioctl**.
- All seven native INT16 arithmetic/writeback methods pass in **3.16 s**. Dynamic-threshold MaskedSelect passes in
  **12.77 s**. All six fixed Nonzero methods pass individually in **2.85–7.10 s**.
- The slow upstream dynamic `test_nonzero` now completes its 320-value FP16 predicate prefix and exact three-endpoint
  INT32 ADD, then rejects the next distinct 640-by-236 dynamic INT32 coordinate histogram. That larger histogram is
  not claimed by this milestone; the focused run reaches the reject in **20.72 s**, without an NPU timeout.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **455 tests**.
- `sz.py`: renderer/runtime **4,371/325 executable lines**, total **29,751**.

The CPU-cheat audit found no runtime or Tinygrad-core change. Renderer evaluation is limited to UOp structure, static
bounds, and gather addresses. All input-dependent byte addition, carry generation, and regrouping run through DPU EW;
runtime gathers only copy opaque bytes. There is no host numeric evaluator, LUT, CMAC, tolerance change, or floating
input wider than FP16.

---

## 2026-08-09 — dynamic FP16 Nonzero with exact INT32 coordinates

All forward cases from upstream `TestOps.test_nonzero` now pass on Rockchip when run as independent hardware tests:
`(32,10)`, `(20,)`, `(10,5,3)`, and the six scalar values. They are split into four Rockchip tests so each physical-NPU
case has an independent wall-time result and a failure cannot obscure the next shape. The rank-two, rank-one, rank-three,
and scalar methods pass in **25.14 s**, **5.32 s**, **10.60 s**, and **3.02 s** respectively. Rockchip collection is now
**459 tests**.

The dynamic FP16 predicate count and prefix remain on the existing DPU FP16 path. Arbitrary dynamic INT32 coordinate
sums are gathered as opaque bytes and added exactly modulo `2**32` with the native INT16 carry emitter introduced in
`bdc9623bc`. Four byte-equality masks form the exact histogram, native INT16 reductions accumulate it, and a syntactically
proven `0 <= index < limit` gate selects static coordinate functions through exact INT32-byte equality. Register-lane
blocks derive from `_MAX_EW_ELEMS_FP16`; block transitions insert submit barriers without a hardcoded task-count or
command-buffer limit.

The original `(32,10)` profile was **32.34 s** = **12.01 s** dynamic count/prefix construction + **19.07 s** final
realization + about 1 ms copyout, using **109 submit ioctls / 4,507 DPU tasks**. The ioctls themselves totaled only about
44 ms. Renderer profiling found scalar static-layout proof was the avoidable cost: the 256-row prefix took **7.00 s** to
render, while the bounded coordinate selector separately scalar-evaluated hundreds of candidate expressions. A shared
vector static evaluator reduces the prefix render to **1.32 s** and brings the complete pytest case below 30 seconds;
NPU tasks and numeric behavior are unchanged.

The Rocket reference's FP16-compare-to-INT16 body was also tested as a possible reset reduction. A focused dynamic
threshold run timed out when that body was placed in the current RKNPU PC-chain transition, so the experiment was fully
reverted. The established reset-isolated FP16 compare path remains. The vendor `~/rk3588/examples/elementwise.py` health
probe passed **60/60** immediately after the revert and again after all final focused runs; no reboot was required.

- Exact INT32 wrap regression: pass; all seven native INT16 EW regressions: **7 passed in 3.02 s**.
- Fixed Nonzero regressions: **4 passed in 10.69 s**; exact arbitrary-INT32 bytes: **1 passed in 3.52 s**; upstream
  fixed-size Nonzero: **1 passed in 4.68 s**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **4,609/325 executable lines**, total **29,989**. Cleanup shares one local-add-loop parser,
  reuses the exact byte-sum emitter, and removes the unused loop-specific coordinate selector.

The CPU-cheat audit found no runtime or Tinygrad-core change. Renderer vector evaluation is limited to compile-time UOp
index expressions, constants, bounds, and candidate coordinate tables. Runtime gathers still move opaque bytes only.
All input-dependent FP16 predicates, dynamic counts/prefixes, INT32 byte carries/equality, histogram reductions,
coordinate selection, and INT32 writeback execute on DPU EW. There is no host numeric fallback, LUT, CMAC, tolerance
change, or floating input wider than FP16.

---

## 2026-08-09 — truncating FP16-to-INT32 cast on DPU

The FP16-input `.int()` portion of upstream `TestOps.test_cast` now passes exactly. A direct use of the existing DPU
FP16-to-INT32 converter was tested first and rejected: it rounds to nearest, producing `1` for positive fractions where
Tinygrad/PyTorch require truncation toward zero. The final implementation reuses the already hardware-tested
`_fold_trunc` composition, `floor(max(x,0)) + ceil(min(x,0))`, then routes the FP16 integer through the shared terminal
INT32 conversion arena. This also avoids multiplying sign masks by infinities.

Historical commit `815003d78` was inspected but not ported because its Rockchip runtime performs the numeric conversion
with NumPy `.astype(np.int32)`. Searches under `~/npu` and `~/rk3588` found conversion/truncation register fields but no
separate proven FP16-to-INT32 cast sequence. Reusing the current native FLOOR/CEIL and typed-output emitters is smaller
and keeps the hardware/host boundary consistent with the other Tinygrad backends.

- Upstream random `(3,3)` FP16-to-INT32 cast plus explicit positive/negative fractions: **1 passed in 3.24 s**.
- Complete focused Cast class: **2 passed in 3.77 s**; the two INT32 cases use **4 ioctls / 20 DPU tasks** total.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the cast runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **460 tests**.
- `sz.py`: renderer/runtime **4,615/325 executable lines**, total **29,995**.

The CPU-cheat audit found no runtime or Tinygrad-core change. The renderer only recognizes a direct FP16 load and builds
the native FLOOR/CEIL UOp composition. Fractional truncation and terminal INT32 conversion execute on DPU EW; runtime
only submits the existing typed stages and copies opaque output bytes. There is no NumPy numeric conversion, host tensor
inspection, LUT, CMAC, tolerance change, or external floating input wider than FP16.

---

## 2026-08-09 — exact FP16-to-FP32 widening on DPU

The FP16-input `.float()` portion of upstream `TestOps.test_cast` now runs on Rockchip. The DPU FP32 output converter
writes four dense FP32 lanes from one eight-lane FP16 atom; larger inputs are therefore gathered into aligned four-lane
groups and emitted as consecutive tasks in one PC-chain ioctl. This limit is derived from the FP16/FP32 element sizes,
not a task-count or command-buffer constant. Direct one- through four-lane inputs need no gather.

`~/rk3588`'s fused-pipeline probe established the FP16-to-FP32 register mode for four-lane, grouped, and wider inputs.
Focused hardware probes found that ADD-zero loses the sign of negative zero, while native `MAX(x,x)` pass-through
preserves every tested widened bit exactly: signed zeros, normal values, subnormals, infinities, and the FP16 NaN payload.
The standalone scalar FP32 emitter was folded into the shared stateful typed emitter, so scalar FP32 reductions and the
new vector cast use one register-construction path.

- One-, two-, three-, four-, eight-, and nine-element focused conversions pass; the permanent random `(3,3)` and exact
  encoding regression uses **one ioctl per realization**. The complete Cast class passes **3/3 in 3.89 s**.
- The existing FP32 scalar reduction regression passes with its unchanged **2-ioctl** contract.
- Full serial Rockchip census: **452 passed, 9 skipped, 187 subtests passed in 653.06 s**. The slowest case was
  `test_simple_cummin` at **27.99 s**, so no individual test crossed the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the census.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **461 tests**.
- `sz.py`: renderer/runtime **4,626/327 executable lines**, total **30,008**. Removing the separate scalar emitter offsets
  most of the new typed-cast matcher and grouped-layout code.

The CPU-cheat audit found no Tinygrad-core change or runtime numeric conversion. Renderer work is limited to proving
static FP16 addresses and preparing raw aligned lane movement; runtime gathers copy opaque `uint16` values. Widening and
FP32 writeback execute on DPU EW. There is no NumPy cast, host tensor inspection, LUT, CMAC, tolerance change, or FP32
input.

---

## 2026-08-09 — exact signed-INT16 comparisons on DPU

All six signed-INT16 comparisons now lower through the public native integer EW path: `<`, `>`, `<=`, `>=`, `==`, and
`!=`. RK3588 has no separately proven INT16 comparison opcode, so the implementation composes the operations established
by `~/rk3588/examples/elementwise_int.py`. Saturating `rhs-lhs` preserves the sign needed for ordered comparison across
the complete INT16 range; MAX with zero followed by MIN with one produces the strict-less mask. Equality uses saturated
SUB, ABS, and MIN-one, so every nonzero difference—including opposite endpoints—maps to one. Tinygrad's boolean
inversions are recovered as native `1-mask` SUB.

Historical exact-INT32 comparison commit `4e2931c2c` was inspected for its sign-safe mask composition, but its four
FP16 byte-plane ABI was not ported: direct INT16 inputs already have the correct signed representation. Boolean output
uses one shared affine low-byte gather also reused by the stored-boolean reduction paths. This is representation movement
after DPU computation, not host predicate evaluation.

- Full-range tensor/tensor ordering and equality, scalar operands in both directions, and a `(2,4)`/`(4,)` broadcast all
  pass. Each realization uses **one ioctl**. The three focused methods pass in **2.95 s**.
- A **131,072-element** comparison passes in one ioctl: **1.5 ms** tensor creation + **163.8 ms** realization +
  **2.6 ms** copyout = **167.8 ms** total. The complete native INT16 class passes **11/11 in 3.22 s**.
- Native INT16 plus the existing stored-boolean reduction group passes **17 tests in 12.76 s** before the large case was
  added; the affine low-byte cleanup therefore preserves both consumers.
- A serial full-census attempt reached **193 passed, 7 skipped, and 100 subtests** before one existing multidimensional
  scatter subcase hit a transient six-second driver timeout. Vendor health immediately passed **60/60**, and the exact
  scatter method then passed all three subtests in **7.26 s**. The parent milestone's complete census remains
  **452 passed, 9 skipped, 187 subtests**; no new matcher can accept that FP16-output scatter graph.
- Final vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed**. Repository-wide Tinygrad mypy (**216 files**),
  Ruff, and `git diff --check`: pass. Rockchip collection: **465 tests**.
- `sz.py`: renderer/runtime **4,644/327 executable lines**, total **30,026**. Sharing the affine INT16-low-byte transport
  reduced the initial comparison implementation by one executable line while removing several materialized offset tuples.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes comparison structure, static
broadcast addresses, and constants. Every input-dependent SUB, ABS, MAX, MIN, inversion, and boolean mask is computed by
DPU INT16 EW; runtime merely copies each NPU-produced low byte into the public bool buffer. There is no NumPy predicate,
host tensor inspection, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native INT16 comparison compositions on DPU

Boolean trees over signed-INT16 comparisons now remain in the native integer EW pipeline. The comparison matcher
recursively turns each predicate into its exact zero/one INT16 mask, then applies the same hardware-tested algebra used
by the FP16 logical path: MUL for AND, MAX for OR, ABS of SUB for XOR, and one-minus-mask for logical NOT. The DPU
operations are all independently covered by `~/rk3588/examples/elementwise_int.py`; no new register mode or runtime
special case was needed.

- Full-range `(x<y)&(x!=0)`, OR, XOR, and explicit `logical_not` all pass exactly, with **one ioctl per realization**;
  the focused method passes in **3.11 s**.
- The complete native INT16 and existing FP16 logical-predicate groups pass **18/18 in 32.13 s**. No individual test
  exceeds 30 seconds; the two slowest are the existing `isclose` methods at **11.07 s** and **10.69 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **466 tests**.
- `sz.py`: renderer/runtime **4,651/327 executable lines**, total **30,033**; the milestone adds seven renderer lines
  and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes the boolean tree and emits
native INT16 stages. Every input-dependent comparison and logical composition executes on DPU EW; runtime only performs
the existing opaque low-byte transport into the public bool buffer. There is no host predicate evaluation, LUT, CMAC,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native INT16 selection on DPU

Comparison-driven INT16 `where` now lowers through the same native integer EW graph as arithmetic and predicates. The
scheduled IR is a single typed WHERE over the already-supported comparison tree. The renderer rewrites that bounded form
to `mask*a + (1-mask)*b`; because the mask is exactly zero or one, one product is always zero and the saturating INT16 ADD
returns the selected branch exactly, including the signed endpoints. This avoids the unsafe `b + mask*(a-b)` form, whose
intermediate subtraction could saturate before selection.

The implementation uses only the INT16 SUB, MUL, and ADD behavior proven by `~/rk3588/examples/elementwise_int.py` and
adds no register mode or runtime special case. The native matcher declaration was moved below the comparison helper so
the renderer reads in dependency order.

- Full-range tensor/tensor selection and scalar fallback selection pass exactly; the method is **0.14 s** after startup.
- A composed AND predicate with broadcast inputs and arithmetic branches passes exactly in **one ioctl** and **0.14 s**.
- The complete native INT16 class passes **14/14 in 3.45 s**; no case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **468 tests**.
- `sz.py`: renderer/runtime **4,655/327 executable lines**, total **30,037**; selection adds four renderer lines and no
  runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only rewrites the typed selection graph.
Predicates, mask inversion, masked products, addition, and branch arithmetic all execute on DPU INT16 EW; runtime only
submits the existing stages. There is no host selection, NumPy tensor evaluation, LUT, CMAC, tolerance relaxation, or
floating input wider than FP16.

---

## 2026-08-09 — native INT16 clamp and ReLU on DPU

Signed-INT16 one- and two-sided clips plus ReLU now recover the canonical extrema hidden by Tinygrad's portable WHERE
form. For `where(lhs<rhs, lhs, rhs)` the renderer emits one native MIN stage; reversing the selected operands emits one
native MAX stage. Nested two-sided clip therefore needs exactly two DPU tasks instead of expanding each bound through a
comparison mask, inverse, two masked products, and ADD.

The scheduled UOps were inspected directly before adding the fold. Historical commit `c359841b5` supplies the native
INT16 pipeline, `~/rk3588/examples/elementwise_int.py` proves signed INT16 MIN/MAX register behavior, and the RKNN
operator references under `~/npu` list INT16 extrema support. The implementation is a structural renderer rewrite like
Tinygrad's other hardware-specific canonicalizations; the generic runtime remains unchanged.

- Full-range two-sided, lower-only, upper-only, and reversed-bound clips pass exactly. Task assertions prove **2, 1, 1,
  and 2 tasks**, respectively, all in one ioctl; the focused method takes **0.19 s** after startup.
- Full-range INT16 ReLU passes exactly as **one native MAX task / one ioctl** in **0.11 s** after startup.
- The complete native INT16 class passes **16/16 in 3.53 s**; no case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the hardware runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **470 tests**.
- `sz.py`: renderer/runtime **4,659/327 executable lines**, total **30,041**; extrema recovery adds four renderer lines
  and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes branch identity in a static
WHERE/CMPLT graph. Every input-dependent MIN/MAX and nested clamp executes on DPU INT16 EW. There is no host comparison,
selection, or clipping, no NumPy tensor evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native signed-INT16 sign on DPU

Signed-INT16 `sign()` now lowers as the exact identity `min(max(x,-1),1)`, using two native extrema tasks. Tinygrad's
scheduled form is `WHERE(x!=0, WHERE(x<0,-1,1), 0)`. A dedicated sign rewrite runs before the general INT16 WHERE pass,
matching the existing FP16 backend organization; otherwise the inner selection is expanded first and the opportunity to
recover the two-stage clamp is lost.

Historical commit `6b00f28ae` supplied the corresponding four-stage FP16 sign matcher, while
`~/rk3588/examples/elementwise_int.py` proves the INT16 MIN/MAX operations used by the smaller integer formulation.
There is no separate Tinygrad `signbit` method; its meaningful integer form, `x<0`, is already covered by the exact
signed-comparison milestone. Internal sign constants remain weak integers in scheduled IR, so the INT16 structural
constant helper now accepts both explicit INT16 and weak-integer constants inside this already typed graph.

- The first diagnostic run was numerically exact but exposed **13 tasks**, proving the general WHERE path was used. The
  dedicated pre-pass reduces the same full-range case to the asserted **2 tasks / 1 ioctl**; call time is **0.11 s**.
- With renderer caching disabled, the complete native INT16 class passes **17/17 in 3.53 s**. No case approaches 30 s.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused and class runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **471 tests**.
- `sz.py`: renderer/runtime **4,669/327 executable lines**, total **30,051**. The dedicated structural pass adds ten
  renderer lines and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time validates the exact sign tree and replaces it
with native extrema UOps. Every input-dependent sign result is computed by two DPU INT16 EW tasks; runtime only submits
the existing image. There is no host classification, comparison, or selection, no NumPy tensor evaluation, LUT, CMAC,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — native INT16 ReLU6 and hardtanh on DPU

INT16 ReLU6 and hardtanh now have explicit two-task contracts. Hardtanh naturally reuses the canonical clamp recovery.
Tinygrad expands ReLU6 as `relu(x)-relu(x-6)`, so the shared hard-activation pre-pass recognizes that exact graph before
the inner WHERE nodes expand and replaces it with `min(max(x,0),6)`.

Historical commit `6d6751002` provides the analogous FP16 cap fold and was used as the structural reference. The INT16
path deliberately does not reuse FP16 ReLUX configuration: `~/rk3588/examples/elementwise_int.py` proves integer MIN/MAX,
which are sufficient and keep the typed pipeline within known hardware behavior. The sign and ReLU6 rules share one
dedicated pre-pass, matching the staged rewrite organization used by the existing FP16 backend.

Non-integral leaky-ReLU slopes promote an INT16 tensor to a weak floating output and are outside this backend's external
FP16-only floating contract. Integral slopes retain INT16 and remain a separate bounded follow-on group.

- Full-range ReLU6 and custom `hardtanh(-3,4)` pass exactly, each as **2 tasks / 1 ioctl**; their combined call is
  **0.15 s** after startup.
- With renderer caching disabled, the complete native INT16 class passes **18/18 in 3.42 s**. No case approaches 30 s.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused and class runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **472 tests**.
- `sz.py`: renderer/runtime **4,684/327 executable lines**, total **30,066**. The bounded ReLU6 recognizer adds fifteen
  renderer lines and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes the exact hard-activation
graph and emits native extrema UOps. All bounds and activations execute on DPU INT16 EW. There is no host clipping,
comparison, or selection, no NumPy tensor evaluation, FP16 ReLUX assumption, LUT, CMAC, tolerance relaxation, or
floating input wider than FP16.

---

## 2026-08-09 — bounded integral INT16 leaky ReLU on DPU

Leaky ReLU with integral slopes from 2 through 32767 now stays in the native INT16 EW pipeline. The renderer recognizes
Tinygrad's exact `where(x<0, slope*x, x)` graph and uses `x + min(x,0)*(slope-1)`. Slope two needs only MIN plus ADD;
larger slopes add one MUL stage.

This identity remains exact under RK3588's saturating INT16 arithmetic. For nonnegative inputs the correction is zero.
For negative inputs, if `(slope-1)*x` saturates low, the final addition remains saturated low; otherwise the final sum is
exactly `slope*x` and saturates at the same boundary. The focused tests include `-32768`, values that overflow only after
scaling, zero, and `32767`.

Historical FP16 commit `ea8294006` and the TRM/RKNN references under `~/npu` document the PReLU mode, but that mode was
not assumed for integer data. The implementation uses only the signed INT16 MIN, MUL, and ADD behavior directly proven
by `~/rk3588/examples/elementwise_int.py`. Fractional slopes promote the output to floating point and remain outside this
bounded INT16 path.

- Slopes two and three pass every saturation endpoint exactly with asserted **2 and 3 tasks**, respectively, and one
  ioctl per realization; their combined call is **0.12 s** after startup.
- With renderer caching disabled, the complete native INT16 class passes **19/19 in 3.41 s**. No case approaches 30 s.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the focused and class runs.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **473 tests**.
- `sz.py`: renderer/runtime **4,696/327 executable lines**, total **30,078**. The bounded leaky-ReLU fold adds twelve
  renderer lines and no runtime lines.

The CPU-cheat audit found no runtime or Tinygrad-core change. Compile time only recognizes a bounded typed graph and
emits native MIN/MUL/ADD UOps. All input-dependent activation work executes on DPU INT16 EW. NumPy is used only for the
test oracle, never by the backend. There is no host classification or selection, unproven integer PReLU mode, LUT, CMAC,
tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT16 extrema loops on DPU

Signed-INT16 MAX/MIN reductions now continue past Tinygrad's 256-element unroll boundary. At 257 elements Tinygrad
switches to a local accumulator loop; the renderer recognizes the exact initialized MAX loop, including Tinygrad's
portable bitwise-complement form for signed MIN, materializes its static source layout, and emits a balanced native
INT16 extrema tree. The terminal DPU task writes the INT16 result directly.

The shared scalar-reduction arena no longer assumes that an output row fits in 64 bytes. Its block stride is derived as
`align64(rows*2)`, and gather spacing, task addresses, and scratch size all use that same value. The focused 40x257 case
therefore uses `257 * 128 = 32,896` scratch bytes instead of overlapping rows above the old implicit 32-row limit.

`~/rk3588/examples/elementwise_int.py` supplies the hardware reference for signed INT16 MAX/MIN. No branch contains a
more complete INT16 loop-reduction implementation. INT16 SUM is intentionally not claimed: Tinygrad promotes it to
INT32, whereas the native INT16 ADD path saturates, so an INT16 reduction tree would be observably wrong on overflow.
The existing exact byte-plane INT32 adder combines already materialized INT32 operands and is not a drop-in signed
INT16 reduction.

- Native 40x257 axis MAX and MIN pass exactly with asserted **256 DPU tasks / 1 ioctl each**; the focused method takes
  **3.09 s** including startup.
- The native INT16 class plus existing FP16 scalar min/max checks pass **22/22 in 4.71 s**.
- The complete physical-NPU Rockchip suite passes serially: **465 passed, 187 subtests passed, 9 skipped in 630.97 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** both before and after the complete suite.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **474 tests**.
- `sz.py`: renderer/runtime **4,738/327 executable lines**, total **30,120**. The native loop matcher plus the shared
  stride correction add 42 renderer lines and no runtime or Tinygrad-core lines.

The CPU-cheat audit found no runtime or core change. Compile time evaluates only static index expressions and constructs
gather plans; runtime gathers only rearrange tensor bytes. Every input-dependent comparison and reduction executes in
native DPU INT16 EW tasks. There is no host extrema calculation, NumPy tensor evaluation, LUT, CMAC, tolerance change,
or floating input wider than FP16.

---

## 2026-08-09 — exact INT16 cumulative extrema on DPU

One-dimensional signed-INT16 `cummax` and `cummin` values now lower through one shared native extrema image. Small scans
are fully unrolled by Tinygrad; the matcher recovers each prefix candidate from direct MAX or the portable
complemented-MAX/WHERE form. At 257 elements Tinygrad switches to a local accumulator loop, and the loop matcher now
accepts the exact prefix-gated layout in addition to ordinary scalar reductions.

Both forms prove their static gather plan before code emission: output lane `i` must contain exactly source offsets
`0..i`, with disabled candidates replaced by the signed MAX/MIN identity. Input-dependent values are then reduced by
native INT16 DPU tasks. Scalar and cumulative paths share `_int16_extrema_image`; arena strides are 64-byte aligned and
derived from the row count rather than a task-count constant.

The scheduled UOps were inspected at 8 and 257 elements. `~/rk3588/examples/elementwise_int.py` supplies the proven
signed INT16 MIN/MAX register behavior; no branch contains a more complete INT16 cumulative implementation. Cumulative
sum/product are not included because saturating INT16 ADD/MUL do not preserve Tinygrad's promoted or wrapped semantics.

- Non-monotonic 17- and 257-element cumulative MAX/MIN pass exactly with asserted **16 and 256 tasks**, respectively,
  and **one ioctl per realization**. The focused method takes **4.43 s** including startup.
- Aligned scalar plus cumulative focused methods pass **2/2 in 5.01 s**; the INT16 class plus FP16 extrema regressions
  pass **23/23 in 6.19 s**.
- Current Rockchip collection is **475 tests**. A serial duration run passed 193 tests before one transient timeout in an
  unchanged multidimensional scatter subtest. Vendor health remained **60/60**, that method then passed all three
  subtests alone in **4.47 s**, and the remaining suffix passed **272 tests, 86 subtests, 2 skips**. Thus every collected
  node was covered on current code without reboot; the immediately preceding complete run was 465/465 plus 9 skips.
- No completed method exceeded 30 seconds. Existing FP16 `test_simple_cummin` was slowest at **27.13 s**; the next were
  `test_dynamic_nonzero_rank_two` at **21.68 s** and FP16 `test_simple_cummax` at **19.37 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before testing, immediately after the transient
  timeout, and after the suite suffix.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **4,777/327 executable lines**, total **30,159**. Sharing scalar/cumulative emission limits
  this milestone to 39 renderer lines and no runtime or Tinygrad-core lines.

The CPU-cheat audit found no runtime or core change. Compile time evaluates only static prefix membership and source
offsets. Runtime gathers rearrange bytes and DPU INT16 MIN/MAX computes every cumulative value. There is no host extrema
or prefix calculation over tensor values, NumPy backend evaluation, LUT, CMAC, tolerance change, or floating input wider
than FP16.

---

## 2026-08-09 — exact signed-INT16 ArgMax/ArgMin on DPU

Signed-INT16 ArgMax and ArgMin now return exact first-tie INT32 indices for global, axis, and 257-element loop forms.
The existing FP16 parser and first-tie validator are typed rather than duplicated: INT16 candidates use direct loads for
ArgMax and Tinygrad's `x XOR -1` complement for ArgMin. The shared image performs native extrema, exact equality masks,
descending-coordinate selection, and terminal INT16-to-INT32 conversion on DPU.

Axis ArgMin materializes a complemented-minimum temporary. Its value kernel now recovers `MAX(~x...)` as
`~MIN(x...)`, implemented exactly as native MIN followed by `-1 - value`; this handles `-32768` without the saturation
error that a NEG-based complement would introduce. The index kernel then compares the same complemented domain.

The small fused layout keeps one ioctl and uses replicated extrema gathers. At 257 candidates, replication would require
`window**2 + 2*window` gathers and overflow RKImage's unsigned-16-bit gather count. That limit is derived from the image
field width, not a hardcoded candidate cap. The wide path reduces a second candidate arena once and mid-gathers the
compact extrema row, reducing gather growth from quadratic to linear. The terminal INT16-to-INT32 body is now appended
to an existing compatible PC chain, matching the prepare/emit/submit layering used by Tinygrad's other hardware
backends and avoiding an otherwise unnecessary ioctl.

Historical FP16 commits `2ec7c17ea`, `309896f58`, and `83bd88931` supplied the tie, axis, and global parser references.
`~/rk3588/examples/elementwise_int.py` proves the signed INT16 MIN/MAX, SUB, and conversion building blocks; neither it,
`~/npu`, nor another branch contains a native INT16 ArgMax implementation to copy.

- Full-range global first ties, axis 0/1 selections, and non-monotonic 257-element global ArgMax/ArgMin pass exactly.
  The final focused method takes **0.66 s call / 3.63 s including startup**.
- Asserted task counts are global-8 **22/31**, axis-1 **14/16**, axis-0 **12/14**, and global-257 **520/522** for
  Max/Min. Small global cases use **1 ioctl**; axis cases use one value plus one index program; the wide global case uses
  two phases because its compact extrema row must be mid-gathered.
- The INT16 and existing FP16 ArgExtrema regression groups passed **28/28 plus 10 subtests in 14.42 s**; final added
  axis-0 assertions also pass in the focused method. No case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after hardware testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **476 tests**.
- `sz.py`: renderer/runtime **4,829/327 executable lines**, total **30,211**. Sharing the existing parser/emitter limits
  the milestone to 52 renderer lines; runtime executable size is unchanged and Tinygrad core is untouched.

The CPU-cheat audit found no tensor-value read or host-side result calculation. Compile time checks only static offsets,
candidate coverage, and Tinygrad's first-tie IR. Runtime only chains command bodies and performs the existing raw
mid-gather. Every extrema, complement, equality, coordinate selection, and typed writeback operation executes on DPU.
There is no host ArgMax/ArgMin, NumPy backend evaluation, LUT, CMAC, tolerance change, or floating input wider than FP16.

---

## 2026-08-09 — exact signed-INT16 MaxPool values and indices on DPU

Rockchip now advertises signed INT16 as a native renderer dtype and lowers padded MaxPool values plus exact returned
INT32 indices. The existing static selection gather is typed instead of duplicated, so padding maps invalid lanes to
the signed `-32768` identity before native INT16 DPU MAX. The existing pool-index parser is likewise parameterized by
candidate dtype and reuses the native INT16 equality/descending-coordinate image added for ArgExtrema. First ties,
padding, batch/channel planes, and the global loop form remain validated from static UOps before emission.

`~/rk3588/examples/elementwise_int.py` supplies the proven signed INT16 MAX/MIN, equality-mask, and conversion register
behavior. The existing FP16 pool-index milestones supplied the spatial-coordinate and global-loop parsers. No new pool
parser, runtime tensor arithmetic, or CPU result path was added.

The first full MaxPool-class run exposed a precision-boundary hazard: native INT16 returned indices followed by the
existing wide FP16 index case timed out at task counter 1/raw status `0xc0000000`, although the wide test passed alone.
The vendor health probe remained 60/60. Runtime now records only whether the preceding EW image used native INT16 and
issues one device reset when leaving that mode. The exact INT16-then-wide sequence and the original full-class order
then pass; ordinary FP16 programs and tasks receive no extra resets.

- Full-range padded `(4,2,11,28)` INT16 MaxPool values, padded duplicate-max first ties, and global `(12,13)` returned
  indices pass bit-exactly. Returned-index cases use one value program plus one index program (**2 ioctls**).
- The complete MaxPool class passes **15 tests plus 33 subtests in 11.74 s**. The native INT16 class plus the new and
  wide pool regressions pass **25/25 in 8.30 s**. No case approaches the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** immediately after the diagnostic timeout and after
  final physical testing; no reboot was required.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **477 tests**.
- `sz.py`: renderer/runtime **4,850/331 executable lines**, total **30,236**. Sharing typed gather/equality/index emission
  limits the milestone to 21 renderer lines and four runtime state-transition lines; Tinygrad core is untouched.

The CPU-cheat audit found only compile-time evaluation of static gates, offsets, weights, and coordinate bounds.
Runtime gathers preserve raw INT16 bytes; every input-dependent maximum, equality mask, first-tie selection, and
INT16-to-INT32 writeback executes on DPU. There is no host MaxPool/index calculation, NumPy backend evaluation, LUT,
CMAC, tolerance relaxation, or Tinygrad-core change.

---

## 2026-08-09 — exact signed-INT16 MaxUnpool on DPU

Signed-INT16 MaxUnpool now scatters through exact dynamic INT32 indices and returns Tinygrad's promoted INT32 output.
The FP16 MaxUnpool parser is typed rather than copied. Its static plane/candidate validation produces the same gather
plans, while a compact native image compares all four index bytes on DPU, masks the INT16 candidate matrix, converts
selected lanes to INT32, and performs the balanced candidate sum in native INT32.

The INT32 reduction is required for exact Tinygrad semantics: duplicate destination indices are summed, and two valid
INT16 values can exceed the signed-INT16 range. A regression deliberately maps `30000` and `30000` to one destination
and verifies the exact result `60000`. The shared byte-mask helper now accepts either one repeated index-offset vector
or a statically validated row per candidate; all existing gather, fancy-index, masked-select, and scatter callers keep
the former path unchanged.

`~/rk3588/examples/elementwise_int.py` provides the native INT16 equality, MUL, INT16-to-INT32, and INT32 arithmetic
register evidence. Existing FP16 MaxUnpool supplied the UOp parser and static scatter-layout validation; no local
reference contains a more complete native integer MaxUnpool implementation.

- Direct full-range INT16 scatter passes exactly with **40 DPU tasks / 2 ioctls**; pool→index→unpool round-trip and the
  duplicate-index INT32 accumulation also pass. The focused method takes **3.17 s** including startup.
- The complete MaxUnpool class passes **7/7 in 37.35 s**. Duration decomposition is wide FP16 **20.12 s**, padded FP16
  **10.40 s**, and every other method at or below **1.41 s**; no individual case crosses the 30-second threshold.
- Four representative existing dynamic gather/fancy-index/scatter/masked-select regressions plus two subtests pass in
  **8.72 s**, covering the unchanged single-offset form of the shared byte-mask helper.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **478 tests**.
- `sz.py`: renderer/runtime **4,880/331 executable lines**, total **30,266**. The milestone adds 30 renderer lines,
  no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime change and no tensor-value evaluation in the renderer. Compile time validates only
static layouts and coordinates; runtime retains the existing raw gathers. Every dynamic index comparison, value mask,
typed conversion, duplicate accumulation, and output write executes on DPU. There is no host scatter/sum, NumPy backend
evaluation, LUT, CMAC, tolerance relaxation, or Tinygrad-core change.

---

## 2026-08-09 — exact dynamic INT16 gather and fancy indexing on DPU

Dynamic INT32 gather indices now select signed-INT16 tensors exactly, including out-of-range zero fill, Python-style
negative fancy indices, and multiple simultaneous fancy-index axes. The prior FP16 gather emitter was already a raw
two-byte selector: it compares all four dynamic index bytes with native INT16 DPU masks, selects each representation
byte, and post-gathers those bytes unchanged. It is now named and typed as a shared 16-bit image instead of being
duplicated for INT16.

All three existing parsers—one bounded dynamic index, fully unrolled multi-index selection, and bounded multi-index
loads—accept an explicit 16-bit value dtype. The generic native INT16 leaf parser now declines loads whose address or
gate depends on an external tensor, allowing those graphs to reach the specialized dynamic-index validators rather
than attempting to evaluate a dynamic address at compile time.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 gather implementation. Historical Rockchip commits
`75100be4a`, `85c85e7fe`, and `ea50b7c96` provide the dynamic/fancy FP16 parser and exact integer-mask structure;
`~/rk3588/examples/elementwise_int.py` proves the native INT16 equality, MUL, and ADD operations used by the shared
image.

- Full-range INT16 gather with positive, negative, and multi-byte out-of-range INT32 indices passes exactly in
  **2.95 s**, asserting **1 ioctl**. Negative row selection and two-axis fancy indexing pass in **3.05 s**.
- Four representative FP16 gather/fancy regressions pass **4/4 in 7.93 s**. No focused case approaches 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **480 tests**.
- `sz.py`: renderer/runtime **4,886/331 executable lines**, total **30,272**. Typing and renaming the shared parsers/image
  adds only six renderer lines, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime or core change. Compile time evaluates only static candidate offsets, bounds, and
coordinate domains; runtime performs the existing raw byte gathers. Every input-dependent index equality, candidate
mask, and selection reduction executes on DPU. There is no host indexing/result calculation, NumPy backend evaluation,
LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact dynamic INT16 scatter on DPU

Dynamic INT32 scatter indices now assign signed-INT16 values exactly with Tinygrad's last-wins semantics. The existing
FP16 scatter image already selects raw two-byte representations: it compares all four index bytes on DPU, constructs
reverse-order effective masks for duplicate destinations, selects source or preserved base bytes, and writes those
bytes unchanged. Its image, gather/output helpers, and parser are now named and typed for either FP16 or INT16 instead
of duplicating the algorithm.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 scatter implementation. Historical Rockchip
commits `826e67550` and `eef7cecce` provide the bounded and multidimensional FP16 scatter parser structure;
`~/rk3588/examples/elementwise_int.py` proves the native INT16 comparison, mask, and arithmetic operations used by the
shared image.

- A full-range signed-INT16 vector with repeated destinations passes exactly in **1.84 s**, proving that the later
  `32767` and `1234` writes replace earlier `-32768` and `-30000` writes. A multidimensional axis scatter passes in
  **1.59 s**.
- The complete Scatter class passes **15 tests plus 7 subtests in 18.79 s**. Existing FP16 nonfinite/raw-bit and
  multidimensional cases remain exact; the slowest individual method is **4.41 s**, well below 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **482 tests**.
- `sz.py`: renderer/runtime **4,887/331 executable lines**, total **30,273**. Sharing the existing raw 16-bit path adds
  one renderer dispatch line, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime or core change and no tensor-value read in the renderer. Compile time checks only
static scatter layout, candidate coverage, and bounds. Every dynamic equality, duplicate-destination priority mask,
source/base selection, and output write executes on DPU. There is no host scatter/result calculation, NumPy backend
evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact fixed-size INT16 masked select on DPU

Fixed-size `masked_select` now compacts signed-INT16 values under an external boolean mask, truncates or pads to the
requested output size, and preserves every selected/fill bit exactly. The existing INT32 selector already computes the
boolean count and compact-index equality on native INT16 DPU lanes before selecting each representation byte. It is now
typed by integer dtype and byte width, so INT16 uses the same parser and emitter with two bytes instead of four.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 masked-select implementation. The existing native
INT32 masked-select milestone provides the complete fixed-output UOp proof and byte-selection image, while
`~/rk3588/examples/elementwise_int.py` proves its INT16 count, equality, mask, and reduction operations.

- Full-range selected INT16 values, duplicate sign patterns, `-12345` fill, and two-element truncation pass exactly in
  **1.48 s** within the full-class run.
- The complete MaskedSelect class passes **6/6 in 27.19 s**. Existing dynamic FP16, exact nonfinite-bit FP16, padding,
  truncation, and INT32 coverage remain clean. The slowest individual method is **8.66 s**, below 30 seconds.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **483 tests**.
- `sz.py`: renderer/runtime **4,889/331 executable lines**, total **30,275**. Generalizing byte count adds only two
  renderer lines, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime or core change and no host value inspection. Compile time validates only the
fixed-output compact-index graph and buffer extents. Boolean counting, compact-index equality, validity masks, raw-byte
selection, and fill composition all execute on DPU. There is no host compaction/result calculation, NumPy backend
evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact fixed-size INT16 nonzero on DPU

Fixed-size `nonzero` now returns exact INT32 coordinates for signed-INT16 inputs, including rank-two layouts,
truncation, padding, and the `-32768` representation. The first compile probe exposed that fixed Nonzero consists of
two integer kernels: an unrolled predicate prefix and the final compact-coordinate selection. Both existing INT32
lowerers are now typed by source dtype and byte width. INT16 checks two opaque representation bytes while INT32 keeps
four; any nonzero byte becomes a native INT16 mask, then the shared prefix/equality/coordinate pipeline completes on
DPU.

No other branch, `~/npu`, or `~/rk3588` contains an integrated INT16 Nonzero implementation. Historical commits
`78a38f384` and `aa7cac490` provide the fixed FP16 and exact-byte INT32 graph proofs, while
`~/rk3588/examples/elementwise_int.py` proves the native INT16 MIN/MAX, reduction, and INT32 writeback operations used
by the shared image.

- Exact rank-one values `{0, -32768, 256, -256, 32767, -1}` and rank-two coordinate/fill selection pass in focused
  cold runs of **3.79 s** and **3.61 s**; they take **0.85 s** and **0.75 s** within the warm class run.
- The complete Nonzero class passes **12/12 in 49.15 s**. The slowest individual method is the pre-existing dynamic
  rank-two FP16 case at **22.15 s**, below the 30-second profiling threshold.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **485 tests**.
- `sz.py`: renderer/runtime **4,892/331 executable lines**, total **30,278**. Sharing the two integer stages adds only
  three renderer lines, no runtime lines, and no Tinygrad-core changes.

The CPU-cheat audit found no runtime/core change or input-value inspection. Compile time proves only the static prefix
coverage, coordinate layouts, and bounds. Runtime gathers opaque source bytes; byte nonzero masks, prefix counts,
compact-index equality, coordinate selection, validity/fill composition, and INT32 writeback execute on DPU. There is
no host Nonzero/result calculation, NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or floating input wider
than FP16.

---

## 2026-08-09 — exact INT16 raw layouts on the generic gather path

Signed-INT16 movement, concatenation, stacking, repetition, and constant padding are now covered bit-exactly. Compile
probes showed that pure raw layouts already lower through the renderer's common static gather planner; only a fused
layout-plus-negation probe was unsupported and is intentionally left for the arithmetic fusion group. The correct
milestone therefore adds coverage without a dtype-specific renderer path.

No branch, `~/npu`, or `~/rk3588` contains or needs a distinct INT16 layout implementation: these operations move raw
bytes and do not invoke arithmetic registers. The existing Tinygrad Rockchip static/partial gather path is the direct
reference, matching the separation used by other hardware backends between layout planning and numeric kernels.

- Chained permute/flip/slice, constant `-12345` two-dimensional padding, axis concat, internal-axis stack, and repeat
  all pass bit-exactly: **3/3 in 3.08 s**, with every test call at or below **0.14 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **488 tests**.
- `sz.py` is unchanged at renderer/runtime **4,892/331 executable lines**, total **30,278**; this is test-only coverage
  with no runtime, renderer, or Tinygrad-core modification.

The CPU-cheat audit found no implementation change and no host-dependent layout decision. Tinygrad computes only the
static address plan at compile time; runtime raw gathers perform every tensor movement on the device-visible buffers.
There is no host tensor read/result calculation, NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or numeric
conversion.

---

## 2026-08-09 — exact promoted INT16 sums in native INT32

Signed-INT16 sums now widen each gathered reduction row with the DPU INT16-to-INT32 output path and accumulate the
rows through native INT32 ADD. Both statically unrolled reductions and Tinygrad's register-loop reduction form are
covered. The logical image accepts the native 32-bit lane capacity; runtime atomizes INT16 conversion into eight-lane
tasks while leaving each 40-lane INT32 accumulation stage intact.

No Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated INT16 reduction implementation. The direct reference
is `~/rk3588/examples/elementwise_int.py`: it proves native signed-INT16 ADD, INT16-to-INT32 writeback, and native
INT32 ADD on the same DPU EW pipeline. The new image composes those proven operations with the existing balanced-row
reduction helper.

- Exact axis-one, axis-zero, and global unrolled sums pass, including `-32768`, `32767`, and results outside INT16.
- The full **40x257** register-loop reduction passes exactly with **1,542 DPU tasks / 2 submit ioctls**; its warm call
  takes **0.17 s** and the focused cold method takes **3.33 s**.
- The complete native INT16 EW class passes **24/24 in 6.22 s**; the existing INT16 max-unpool regression also passes.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **490 tests**.
- `sz.py`: renderer/runtime **4,949/331 executable lines**, total **30,335**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection in the renderer. Compile time validates
only static reduction shape, source coverage, masks, and buffer bounds. Runtime gathers opaque INT16 rows; widening,
balanced INT32 accumulation, and final INT32 writeback all execute on DPU. There is no host sum/result calculation,
NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — fused INT16 layouts and arithmetic

Native signed-INT16 arithmetic is now covered after multi-source and transformed layouts: concat followed by NEG,
stack followed by ABS, permute followed by ADD, constant pad followed by clip, and repeat followed by ADD. These were
the explicit remaining forms from the prior raw-layout milestone. Current compile probes showed that the generic
INT16 leaf/gather planner already composes all five, so no dtype- or operator-specific implementation was added.

`~/rk3588/examples/elementwise_int.py` directly proves signed-INT16 NEG, ABS, ADD, MAX, and MIN; no other Tinygrad
branch or `~/npu` provides a more integrated fused-layout path than the current generic Rockchip planner.

- All five exact cases pass in **3.09 s**, with calls from **0.02 s** to **0.17 s**. Concat, permute, and repeat use one
  DPU task; pad+clip uses two; stack+ABS composes two source-layout tasks and one ABS task in one ioctl.
- The complete native INT16 EW class passes **29/29 in 6.57 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **495 tests**.
- `sz.py` is unchanged at renderer/runtime **4,949/331 executable lines**, total **30,335**; this milestone is
  test-only and does not change Tinygrad core.

The CPU-cheat audit found no implementation change and no host-dependent arithmetic decision. The renderer derives
only static gather plans; opaque INT16 layout movement and every NEG/ABS/ADD/MAX/MIN stage remain on the device path.
There is no host result calculation, NumPy backend evaluation, LUT, CMAC, tolerance relaxation, or floating input
wider than FP16.

---

## 2026-08-09 — exact promoted INT16 cumulative sums

Signed-INT16 `cumsum` now stays exact in INT32 when Tinygrad lowers a large scan to a masked register-loop reduction.
The existing promoted-sum matcher already handled small unrolled scans; its loop coverage proof now recognizes
disjoint nested prefix chains rather than requiring each input lane exactly once. The same validator replaces the
older one-dimensional cumulative-extrema check, so scan safety has one shared static proof.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated INT16 cumulative-sum path.
`~/rk3588/examples/elementwise_int.py` proves the required signed INT16-to-INT32 writeback and ADD stages; this
milestone composes them through the existing balanced INT32 reduction image.

- Exact unrolled 17-element and loop-shaped 257-element scans pass independently. Batched **2x257 last-axis** and
  **257x2 first-axis** scans also pass, proving both contiguous and interleaved prefix layouts.
- Independent calls take **0.24 s**, **1.19 s**, **2.59 s**, and **2.73 s**; no case approaches the 30-second policy.
- The complete native INT16 EW class passes **33/33 in 12.55 s**. Focused cumulative-extrema, cumsum, and scalar-sum
  regressions pass **3/3 in 5.87 s** after sharing the validator.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **499 tests**.
- `sz.py`: renderer/runtime **4,963/331 executable lines**, total **30,349**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change or tensor-value inspection. The new helper examines only compile-time
source-offset sets and proves uniqueness, partitioning, and strict prefix nesting. Runtime gathers opaque INT16 lanes;
all widening, balanced ADD, and INT32 writeback execute on DPU. There is no host cumsum/result calculation, NumPy
backend evaluation, LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — ordered INT16 product reductions on DPU

Large signed-INT16 `prod` and `cumprod` register loops now lower to an ordered native-MUL chain. Masked cumulative lanes
use the exact multiplicative identity one. The chain is intentionally sequential rather than balanced because RK3588
INT16 MUL saturates; reassociation can change an overflowed result. Sum and product now share one typed local-loop
parser for accumulator initialization, source/load validation, static block construction, and scalar/prefix coverage.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated loop-product implementation.
`~/rk3588/examples/elementwise_int.py` directly proves signed-INT16 MUL and its saturating limits on this DPU EW path.

- A **40x257** axis product, a 257-element cumulative product, and **2x257 last-axis** plus **257x2 first-axis**
  cumulative products pass exactly. The sequence `300 * 300 * -1` proves ordered saturation: `32767`, then `-32767`.
- Independent calls take **0.22 s**, **0.91 s**, **1.88 s**, and **2.04 s**. First-axis cumprod includes one Tinygrad
  layout kernel and the 257-task product chain; the other cases use one submit.
- The complete native INT16 EW class passes **37/37 in 16.97 s**; no case approaches the 30-second policy.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **503 tests**.
- `sz.py`: renderer/runtime **4,991/331 executable lines**, total **30,377**. Sharing the existing sum parser keeps the
  implementation to a net 28 renderer lines; runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection. Compile time evaluates only static
addresses, masks, and reduction topology. Runtime gathers opaque INT16 lanes; every ordered saturating MUL and final
write executes on DPU. The NumPy routine exists only in the test oracle. There is no host product/result calculation,
LUT, CMAC, tolerance relaxation, or floating input wider than FP16.

---

## 2026-08-09 — exact INT16 any/all reductions on DPU

Large signed-INT16 `any` and `all` register loops now reduce exact nonzero predicates entirely on the DPU. Each opaque
INT16 lane is converted to a native 0/1 mask from both source bytes, preserving `-32768` as nonzero; `any` then uses
INT16 MAX and `all` uses INT16 MUL. The final low result byte is copied directly into Tinygrad's boolean output layout.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated INT16 any/all reduction. The local
`elementwise_int.py` reference proves the native integer MAX/MUL primitives; this milestone composes them with the
backend's existing exact integer-nonzero mask and shared row reducer.

- A 257-element global `any`, **2x257 last-axis** `any`, 257-element global `all`, and **257x2 first-axis** `all` pass
  exactly. The input includes `-32768` to prove that the high source byte participates in the nonzero predicate.
- Each operation uses **259 DPU tasks / one submit ioctl**. Independent test methods pass in **3.20 s** and **3.00 s**;
  warm calls in the full class take at most **0.12 s**, well below the 30-second policy.
- The complete native INT16 EW class passes **39/39 in 17.41 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. Rockchip collection: **505 tests**.
- `sz.py`: renderer/runtime **5,007/331 executable lines**, total **30,393**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change or tensor-value inspection. The renderer validates only static loop
offsets and topology. Runtime gathers opaque integer bytes; nonzero-mask construction, MAX/MUL reduction, and result
writeback execute on DPU. The post-gather is a raw low-byte layout copy, not host boolean computation. NumPy is used
only as the test oracle. There is no host reduction, LUT, CMAC, tolerance relaxation, or wider floating-point input.

---

## 2026-08-09 — exact INT16 nonzero counts and test-census split

Signed-INT16 `(x != 0).sum(...)` now produces exact INT32 counts for unrolled, register-loop, and matrix-axis graphs.
The existing byte-exact integer predicate mask feeds native INT16 ADD, followed by the DPU INT16-to-INT32 output
converter. The predicate emitter is shared with INT16 `any`/`all`; only the final reduction operation and output layout
differ. A shared local-ADD unroller also removes duplicate loop normalization from the existing FP16 count paths.

No checked Tinygrad branch, `~/npu`, or `~/rk3588` contains an integrated integer nonzero-count reduction.
`~/rk3588/examples/elementwise_int.py` proves the required native INT16 ADD-to-INT32 primitive on this hardware.

- Exact 7-element unrolled, 257-element loop, **2x257 last-axis**, and **257x2 first-axis** counts pass independently.
  Inputs include `-32768`, proving that both opaque source bytes participate in the nonzero test.
- The unrolled case uses **10 DPU tasks / one ioctl**; each 257-lane reduction uses **260 tasks / one ioctl**. Final
  focused pytest methods pass in **2.89 s**, **2.77 s**, and **2.92 s**.
- Per the census-layout rule, `test_rockchip.py` now contains only upstream `TestOps` method names: **327 collected**.
  The **181 Rockchip-only** hardware regressions live in `test_rockchip2.py`; combined collection remains **508**.
- After the split, the backend-only INT16 class passes **38/38 in 17.34 s** and its four upstream-derived siblings pass
  **4/4 in 2.94 s**. No individual case approaches the 30-second policy.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,041/331 executable lines**, total **30,427**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection. Compilation proves only static source
coverage, masks, and output topology. Runtime gathers opaque INT16 bytes; predicate construction, ADD reduction, and
INT32 writeback all execute on DPU. NumPy appears only in test oracles. There is no host result calculation, LUT, CMAC,
tolerance relaxation, or wider floating-point input. The unchanged upstream `test_ops.py` has 427 collected nodes and
has not been run as one direct ROCKCHIP census; the 100 nodes outside the 327-case selected census are unclassified,
not measured failures.

---

## 2026-08-09 — exact DPU bytewise complement

The unchanged upstream `test_bitwise_not` now passes for both INT32 and boolean inputs. INT32 values are treated as
four opaque bytes and complemented with native INT16 `255-byte`; boolean values use the same path as `1-byte`. A raw
post-gather packs the low byte of each INT16 lane back into the typed destination. This preserves every INT32 bit,
including both signed limits, without asking the DPU to represent the value numerically as FP16.

No checked Rockchip branch, `~/npu`, or `~/rk3588` contains an integrated Tinygrad bitwise-not lowerer.
`~/rk3588/examples/elementwise_int.py` proves native integer `EW_ALU_MINUS=4` for INT16, while
`~/rk3588/test/test_ops.py` carries the same upstream behavior test. The new work is the bounded RKImage integration,
static movement support, and exact byte packing.

- Direct unchanged upstream `TestOps.test_bitwise_not`: **1 passed in 2.93 s**. The selected Rockchip copy passes in
  **2.94 s**; the exact full-byte/movement regression passes in **2.74 s**.
- The exact regression covers `INT32_MIN`, `INT32_MAX`, byte/carry boundaries, a transpose, boolean flip, and asserts
  **two DPU tasks / two ioctls** for its two realizations.
- Per the census-layout rule, `test_rockchip.py` contains **328 upstream-only cases** and `test_rockchip2.py` contains
  **182 Rockchip-only cases**; combined collection is **510**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,066/331 executable lines**, total **30,452**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no tensor-value inspection. Compilation computes only
static byte offsets and output layout. Runtime gathers opaque bytes, DPU INT16 SUB performs the complement, and raw
post-gather repacks the result. NumPy is used only by test oracles. There is no host result calculation, LUT, CMAC,
tolerance relaxation, or floating-point widening.

The direct upstream failure census remains incomplete. On this working tree, `test_sort`, `test_cast`, and
`test_pow_zero_exponent` are directly confirmed failures, `test_masked_select` is unresolved after a compile-time
runaway, and the other **95** nodes outside the now-328-case selected census remain unclassified rather than measured
failures.

---

## 2026-08-10 — exact DPU binary bitwise operations

The unchanged upstream `test_and`, `test_or`, `test_xor`, and `test_int_or` now pass alongside bytewise complement.
INT32 inputs are gathered as four opaque byte lanes. Each byte is decomposed into eight exact 0/1 INT16 lanes by
descending threshold, clamp, scale, and subtract stages. Native MIN/MAX implement AND/OR; SUB followed by native ABS
implements XOR. Weighted INT16 ADD reconstructs each byte, and a raw post-gather writes the exact little-endian INT32
representation. Boolean buffer AND/OR use a one-task native mask path instead of the full decomposition.

The reference audit found no hardware bitwise primitive. `~/npu/include/old/rknn_ops.md` marks BitwiseAnd/Or/Xor as
unsupported, and `/home/orangepi/rk3588/experimental/ops_rockchip.py` explicitly fell back to CPU for these ops.
`~/rk3588/examples/elementwise_int.py` proves the native INT16 MIN/MAX/SUB/ABS/MUL/ADD building blocks used here;
other checked Rockchip branches only contain boolean-mask AND/OR composition. This milestone replaces the old CPU
fallback idea with an exact bounded DPU construction.

- Direct unchanged upstream methods pass independently: AND **4.74 s**, OR **3.04 s**, XOR **2.91 s**, and integer
  all-ones OR **2.96 s**. The selected five-method bitwise class passes serially in **4.90 s**.
- A transposed full-bit-pattern regression covering signed limits, alternating bits, and arbitrary hex patterns passes
  in **3.13 s**. AND/OR use **92 tasks** each and XOR uses **100 tasks**, one ioctl per realization; the regression
  asserts **284 tasks / three ioctls**.
- The former predicate-only custom `test_and/or/xor` methods moved to `test_rockchip2.py` as explicitly local tests and
  pass **3/3 in 8.24 s**. `test_rockchip.py` now collects **329 upstream-only cases**, `test_rockchip2.py` collects
  **186 backend-only cases**, and combined collection is **515**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,147/331 executable lines**, total **30,533**. A shared typed-load matcher removes
  duplicate INDEX/source/bounds parsing from byte addition, complement, boolean logic, and binary logic.

The CPU-cheat audit found no runtime or Tinygrad-core change and no tensor-value inspection. Compilation materializes
only static offsets, public scalar constants, and threshold rows. All data-dependent bit extraction, bit combination,
and reconstruction execute as native DPU INT16 tasks. NumPy appears only in test oracles. There is no host result
calculation, CPU fallback, LUT, CMAC, tolerance relaxation, or floating-point widening.

The direct upstream failure census remains incomplete. The selected census now covers 329 of 427 nodes; among the
remaining 98, `test_sort`, `test_cast`, and `test_pow_zero_exponent` are confirmed failures, `test_masked_select` is
unresolved after a compile-time runaway, and **94** remain unclassified.

---

## 2026-08-10 — batched upstream remainder census and promotion

The remainder workflow now operates by capability batch rather than one method per milestone. Fresh collection of
`TestOps` contains **421 method nodes**; the selected Rockchip census represented 306 unique upstream method names,
leaving 115 names for direct classification. A validation batch and a 14-method data/indexing batch ran without an
RKNPU timeout, followed by a clean 60/60 vendor health check.

Eight unchanged upstream methods were promoted together: `test_nonzero`, `test_max_unpool2d`,
`test_scatter_no_reduce_tensor_src`, all three missing fancy-indexing families, `test_slice_fancy_indexing_errors`, and
`test_scaled_dot_product_attention_gqa_errors`. `test_rockchip.py` now collects **337 upstream-only cases**;
`test_rockchip2.py` remains **186 backend-only cases**, for **523 combined**.

The same batch classified shared gaps rather than stopping at first failure:

- Full/partial broadcast passes ADD/SUB/MUL/DIV but fails only its dynamic POW subtests.
- Direct `argmax`/`argmin`, general scatter/reduce, and repeated-integer `topk` reach unsupported integer graphs or
  FP16 encoding of an INT32 sentinel. These form the next shared integer-lowering batch.
- `scatter_reduce_errors` and `scatter_reduce_prod_zeros` currently fail in the Torch reference before Rockchip numeric
  execution because the upstream FP16-default dtype differs from the fixture's FP32 tensors.
- Attention length mismatch reaches a large unsupported softmax graph. Transcendental softmax/POW cases remain outside
  the no-LUT scope rather than receiving CPU fallbacks.

Two passing methods exceeded the 30-second policy and were profiled together:

| Method | Wall | Renderer | Program calls | Submit ioctls | Tasks | Ioctl wall |
|---|---:|---:|---:|---:|---:|---:|
| `test_nonzero` | 33.439 s | 5.632 s / 16 | 18.049 s / 17 | 192 | 8,148 | 0.063 s |
| `test_max_unpool2d` | 31.186 s | 7.274 s / 9 | 21.735 s / 9 | 225 | 3,351 | 0.108 s |

Both are dominated by host render/command construction and precision-transition/reset handling, not NPU ioctl time.
The batch also measured `dim_inject_none` at 29.93 s. These profiles define a shared host-overhead optimization target.

The CPU-cheat audit is unchanged: this milestone modifies only the selected test census and documentation. All promoted
methods were executed directly from unchanged upstream bodies. There is no host result calculation, CPU fallback, LUT,
CMAC, tolerance change, or Tinygrad-core change. The batch health check passed **60/60** vendor probes.

---

## 2026-08-10 — batched exact INT32 shifts

One native INT16 barrel-shifter lowerer now covers the full upstream 32-bit shift family: signed and unsigned inputs,
left and right shifts, scalar constants, and tensor shift counts. Four opaque input bytes are decomposed into exact
0/1 INT16 bit planes. Five DPU stages conditionally shift by 1, 2, 4, 8, and 16 bits; arithmetic right shift replicates
the original sign bit. Weighted INT16 row reductions reconstruct the four raw output bytes without FP16 integer
encoding, CPU arithmetic, CMAC, or LUTs. The existing AND/OR/XOR lowerer now shares the byte decomposition helper.

- Unchanged upstream batch: `test_lshift`, `test_rshift`, `test_lshift_signed`, `test_rshift_signed`, and
  `test_idiv_shift_rewrite_negative` passed together, **5/5 in 3.85 s**.
- The promoted upstream bitwise/shift class plus a backend-only all-count regression passed **11/11 in 6.15 s**.
  The local regression checks every dynamic shift count from 0 through 31 on zero, signed limits, all-ones, and mixed
  bit patterns for signed left/right and unsigned right shift.
- Adjacent cases were classified in the same batch: `test_div_int` still needs general INT/FP cast and integer
  division, while integer/dynamic POW requires a separate multiply/polynomial capability. Broadcast ADD/SUB/MUL/DIV
  subtests pass; only POW remains unsupported. `test_isclose` reaches the existing FP16 tolerance boundary and `test_cast`
  still lacks general INT32/boolean-to-FP output conversion.
- Selected collection is now **342 upstream-only cases**; backend-only collection is **187**. The authoritative
  upstream-name remainder falls from 107 to **102** after this five-method promotion.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the physical batch.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,264/331 executable lines**, total **30,650**.

The CPU-cheat audit found no runtime or Tinygrad-core change and no input-value inspection. Initial and mid gathers only
move raw bytes and compiler-known constants/offsets; all data-dependent bit extraction, conditional selection, sign
extension, weighting, and reconstruction execute as native DPU INT16 EW tasks. No tolerance was changed.

---

## 2026-08-10 — batched arg-extrema, stable sort, and TopK

Four unchanged upstream methods were completed as one shared integer-selection milestone: `test_argmax`, `test_argmin`,
`test_sort`, and `test_topk`. Signed INT32 extrema and compare/swap now operate on four widened byte lanes with an exact
native-INT16 lexicographic comparator. Boolean extrema use the same running first-tie selection shape. Stable sort's
occurrence and final-index graphs share one weighted equality emitter for FP16 numeric values and exact INT32 counts.
FP16 equality canonicalizes signed zero, distinguishes signed infinities, and rejects NaNs without the reset-heavy FP16
comparison mode.

- Direct unchanged upstream batch: **4/4 passed in 18.31 s**, including axes, repeated values, booleans, bitwise-not,
  signed INT32 limits, stable integer sort, and TopK values/indices.
- The promoted test aliases plus backend-only arg/sort/topk regressions passed **23 tests / 36 subtests in 22.27 s**.
  New local coverage includes signed zero, both infinities, repeated signed limits, and ascending/descending stable order.
- The former slow representative 8x8x6 sort-index realization fell from **11.12 s to 0.53 s**. Its reset count fell
  from **99 to 1**, submit count from **130 to 20**, and measured program wall from **10.78 s to 0.14 s**. The complete
  upstream sort/topk pair fell from **97.04 s to below 30 s** as part of the four-method batch.
- `test_rockchip.py` collects **346 upstream-only cases** (323 unique upstream method names); `test_rockchip2.py` collects
  **189 backend-only cases**. The authoritative upstream-name remainder is now **98 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, `git diff --check`, and focused renderer mypy/Ruff: pass.
- `sz.py`: renderer/runtime **5,605/331 executable lines**, total **30,991**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime or Tinygrad-core modification and no tensor-value inspection. Compilation handles
only static graph topology, offsets, weights, and byte layout. Runtime gathers opaque bytes; all comparisons, equality
masks, first-tie updates, weighted reductions, and INT32 writeback execute as native DPU INT16 EW tasks. There is no
host result calculation, CPU fallback, LUT, CMAC, tolerance relaxation, or floating-point input wider than FP16.

---

## 2026-08-10 — batched integer/boolean casts and empty std cases

The unchanged upstream `test_cast` now passes by composing two existing DPU converters: INT32 input is converted to an
FP16 numeric tile, then widened into the required FP32 output layout. Boolean bytes are zero-extended into the same
INT32 tile format before conversion. Four-lane groups use the DPU's required 16-byte FP32 conversion stride. Existing
FP16-to-INT32, FP16-to-boolean, and FP16-to-FP32 paths remain unchanged.

The remainder batch also identified and promoted two already-correct empty-axis methods,
`test_std_mean_loaded_nan` and `test_std_zero_in_axis`. They exercise empty-shape/NaN semantics and do not claim native
SQRT support; all non-empty standard-deviation methods remain outside the current no-LUT hardware scope.

- The full upstream `test_cast`, both upstream empty-axis std methods, and a backend-only integer/boolean conversion
  regression pass together: **4/4 in 4.42 s**.
- The backend regression covers dynamic signed INT32 values and boolean values; all expected values are exactly
  representable through the NPU's FP16 intermediate.
- `test_rockchip.py` now collects **349 upstream-only cases** (326 unique upstream names), and `test_rockchip2.py`
  collects **190 backend-only cases**. The authoritative remainder falls from 98 to **95 of 421**.
- The same batch classified general scatter/scatter-reduce as one dynamic-index WHERE lowering gap, non-empty std as a
  SQRT gap, and masked-select as a CPU-bound compiler runaway rather than an NPU timeout.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,639/331 executable lines**, total **31,025**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime/core change and no tensor-value inspection. The renderer validates only CAST
structure and static offsets. Runtime movement zero-extends opaque boolean bytes; both numeric conversions and FP32
writeback execute on DPU. There is no CPU arithmetic, fallback, LUT, CMAC, tolerance change, or wider input contract.

---

## 2026-08-10 — batched dynamic Scatter and ScatterReduce

Dynamic tensor Scatter and the five tensor ScatterReduce modes now share exact native-INT16 byte equality for external
INT32 indices. Scatter performs last-wins raw 16-bit selection entirely in the integer EW pipeline. ScatterReduce packs
the native 0/1 equality result into an FP16 mask at one mid-gather boundary, then performs sum, product, mean, minimum,
or maximum on DPU EW. Both `include_self` modes are covered. This removes the former four INT32-to-FP16 byte conversions
and their device resets from every reduction realization.

- The unchanged upstream `test_scatter_reduce` passes all **30 reduction/dimension/include-self cases in 5.31 s** under
  the established `test_gemm_fp16` tolerance.
- Direct wall decomposition for those 30 cases: **2.253 s** method wall, **0.191 s / 30** renderer calls,
  **0.099 s / 30** program calls, **0.013 s / 138** submit ioctls, and **0 resets**. The pre-optimization run had already
  exceeded **38.58 s** before finishing the mean cases because every realization performed four conversion/reset paths.
- The unchanged upstream `test_scatter` numeric prefix fell from **39.03 s to 4.20 s**. Its final dtype-error assertion
  is not valid with `DEFAULT_FLOAT=HALF`, so the unchanged method is not claimed in `test_rockchip.py`; the same eight
  numeric cases live in `test_rockchip2.py`.
- The complete Scatter alias/regression batch passes **18 tests / 13 subtests in 11.81 s**.
- `test_rockchip.py` collects **350 cases** representing **327 unique upstream methods**; `test_rockchip2.py` collects
  **191 backend-only cases**. The authoritative upstream-name remainder is now **94 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,807/331 executable lines**, total **31,193**. Runtime and Tinygrad core are unchanged.

The CPU-cheat audit found no runtime or Tinygrad-core modification and no tensor-value inspection. The renderer uses
only static graph topology, offsets, and constants. Runtime gathers move opaque bytes; index comparison, last-wins
selection, reduction masks, arithmetic, and output reconstruction all execute on DPU EW. No LUT, CMAC, host result
calculation, tolerance relaxation beyond `test_gemm_fp16`, or input wider than FP16 was added.

---

## 2026-08-10 — batched exact signed INT32 division and remainder

One byte-restoring native-INT16 DPU lowerer now covers signed INT32 truncating division, floor division, C-style
remainder, and floor modulo. Four opaque input bytes are conditionally converted to magnitude, divided over 32 exact
bit steps, then sign- and rounding-corrected before raw INT32 writeback. A companion mixed INT32/FP16 path performs
device conversion followed by native FDIV, FLOOR/CEIL, MUL, and SUB stages. Direct use of the reference integer
`DIV` register was rejected: `~/rk3588/examples/elementwise_int.py` documents that it retains floating-point semantics
in integer precision and is therefore not exact signed integer division.

- Full unchanged upstream `test_mod`, `test_fmod`, `test_div_int`, and `test_div_rounding_mode`, plus a backend-only
  signed-limit regression, pass together: **5/5 in 33.43 s**. The local regression covers INT32 minimum/maximum,
  mixed signs, zero numerators, and all four rounding/remainder modes in **4 ioctls / 13,980 DPU tasks**.
- The only newly represented upstream method name is `test_div_int`; the other three replace earlier shortened local
  copies with their complete unchanged upstream bodies. `test_rockchip.py` now collects **351 upstream-only cases**
  representing **328 unique upstream method names**; `test_rockchip2.py` collects **192 backend-only cases**. The
  authoritative unselected remainder falls from 94 to **93 of 421**. These 93 are not all confirmed failures: most
  belong to shared transcendental-dependent families outside the current no-LUT scope.
- The former slowest case, unchanged `test_div_rounding_mode`, fell from **30.76 s to 26.34 s**. Before command reuse,
  a direct profile measured **28.093 s wall**, **0.106 s / 15 renderer calls**, **26.420 s / 144 program calls**, and
  **1.790 s / 320 submit ioctls**. Retaining the last immutable PC-chain bodies and prepatched scratch-only INT16
  bodies reduced the direct profile to **23.296 s wall**, **0.105 s renderer**, **21.599 s program**, and
  **0.334 s ioctl wall** without caching tensor values.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,117/352 executable lines**, total **31,524**.

The CPU-cheat audit found no Tinygrad-core change, runtime tensor-value inspection, or host numeric result calculation.
Compilation handles only graph topology, static offsets, constants, and register words. Runtime gathers opaque bytes;
all data-dependent sign detection, two's-complement conversion, comparison, subtraction, quotient/remainder formation,
rounding, and reconstruction execute on native DPU EW. The command cache retains immutable command qwords only. No
LUT, CMAC, tolerance relaxation, FP32 input emulation, or wider floating-point input was added.

---

## 2026-08-10 — exact native INT32 product and constant powers

A base-16 limb multiplier now evaluates arbitrary elementwise INT32 MUL trees exactly modulo 2^32 on native INT16
DPU EW. Each opaque input byte is split into two nibbles; schoolbook partial products stay below 256, and exact nibble
carry propagation keeps every intermediate within signed INT16 range. Common subexpressions are retained while walking
Tinygrad's exponentiation-by-squaring MUL graph. The older `rockchip-2607` implementation was deliberately not ported:
its `_try_int_power_host_subtasks` delegated the arithmetic to a typed host task, and its disabled native experiment
documented high-word corruption from direct INT32 MUL.

- Full unchanged upstream `test_int_pow_const_int` and a backend-only full-range product regression pass together:
  **2/2 in 4.03 s**. The upstream method covers powers 0, 1, 2, 7, and 29 plus the required negative-exponent error.
- Arbitrary two-input multiplication passes signed limits and **127 random full-range INT32 pairs** exactly. It uses
  **1 ioctl / 1,468 DPU tasks** independent of vector length; the repeated-square power paths use 1,364 tasks for
  exponent 2, 5,120 for exponent 7, and 8,876 for exponent 29.
- `test_rockchip.py` collects **352 upstream-only cases** representing **329 unique upstream method names**;
  `test_rockchip2.py` collects **193 backend-only cases**. The authoritative unselected remainder falls from 93 to
  **92 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,206/352 executable lines**, total **31,613**.

The CPU-cheat audit found no runtime or Tinygrad-core modification and no tensor-value inspection. Gathers move opaque
bytes and compiler-known constants; nibble extraction, partial products, carries, overflow truncation, and output
reconstruction all execute as native INT16 DPU stages. There is no host result calculation, CPU fallback, LUT, CMAC,
tolerance change, FP32 input emulation, or wider floating-point input.

---

## 2026-08-10 — exact default FP16 isclose and reset removal

Default FP16 `isclose` now bypasses subnormal tolerance arithmetic with a mathematically equivalent exact IEEE
equality path. At `rtol<=1e-5` and `atol<=1e-8`, even adjacent FP16 subnormals are farther apart than the tolerance.
Operands are realized in FP16 first, gathered as opaque bytes, and compared with native INT16 arithmetic. Signed zeros
are canonicalized, NaNs are classified from exponent/mantissa bytes, and `equal_nan` is preserved. Larger tolerances
retain the general device arithmetic path and OR exact equality into its result. The byte canonicalization emitter is
shared with stable sort equality instead of duplicated.

- Full unchanged upstream `test_isclose` passes in **17.02 s**, down from **65.35 s** after the first correctness fix.
  The intermediate exact-UOp implementation passed in 32.52 s but still paid reset-heavy FP16 comparisons and was
  replaced. Unchanged upstream `test_isclose_edge_cases` now passes in **3.27 s** and replaces its former packed custom
  copy in the upstream-only census.
- Final direct wall decomposition for all ten `test_isclose` realizations: **14.599 s wall**, **0.073 s / 10 renderer
  calls**, **13.666 s / 10 program calls**, **0.043 s / 140 submit ioctls**, and **13.443 s / 127 resets**. The remaining
  resets belong to the two explicit `rtol=0.01` general-tolerance graphs; every individual method is below 30 seconds.
- The full isclose/scalar/edge/local regression plus adjacent signed-zero/infinity sort regression passes **5 tests / 2
  subtests in 29.67 s**. The backend regression covers both signed zeros, adjacent subnormal/normal bit patterns,
  maximum finite values, both infinities, and NaNs under both `equal_nan` modes.
- `test_rockchip.py` collects **353 upstream-only cases** representing **330 unique upstream method names**;
  `test_rockchip2.py` collects **194 backend-only cases**. The authoritative unselected remainder falls from 92 to
  **91 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,302/352 executable lines**, total **31,709**.

Historical `b85d3c4b0` and `31c07151d` computed isclose predicates in host elementwise layouts and remain rejected. The
current runtime only moves raw bytes at explicit gather boundaries; FP16 expression realization, zero normalization,
NaN classification, equality, tolerance evaluation, and bool formation all execute on DPU EW. There is no host result
calculation, CPU fallback, LUT, CMAC, tolerance relaxation, FP32 input emulation, or wider floating-point input.

---

## 2026-08-10 — dynamic scalar-source scatter batch

The existing dynamic last-wins Scatter lowerer now accepts compiler-known FP16 source expressions as well as tensor
loads. Static source words are split into opaque bytes and fed into the same native INT16 equality-mask and raw-select
pipeline, so non-finite values never participate in FP16 mask arithmetic. This completes the unchanged upstream
`test_scatter` method, including its six dimensions, two additional shapes, four error fixtures, scalar `3`, scalar
infinity, and overlapping-index last-wins case. The unchanged validation-only `test_scatter_reduce_errors` method is
also admitted; a local exception helper preserves the upstream FP32/FP16 dtype-mismatch fixture while numeric Rockchip
inputs remain FP16.

- Both complete upstream methods pass together: **2 passed in 3.72 s**; wall time including pytest startup is
  **5.83 s**. A backend-only bit-exact dynamic infinity regression passes in **2.93 s**.
- `test_rockchip.py` collects **355 upstream-only cases** representing **332 unique upstream method names**;
  `test_rockchip2.py` collects **195 backend-only cases**. The authoritative unselected remainder falls from 91 to
  **89 of 421**.
- Of those 89, **85** belong to the deferred transcendental/LUT-dependent families. The four other exclusions are
  backward-only `test_cmp_lt_backwards`/`test_cmp_ne_backwards`, FP32-input `test_scatter_reduce_prod_zeros`, and the
  scalar-true half of `test_masked_select`, which runs away before the backend renderer without a Tinygrad-core hook.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. Repository-wide Tinygrad
  mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,313/352 executable lines**, total **31,720**.

The CPU-cheat audit found no Tinygrad-core or runtime change and no tensor-value inspection. Compilation evaluates only
static shape predicates, offsets, and source constants; runtime gathers opaque bytes. External index equality,
last-wins masking, reduction, base preservation, and output selection all execute on native INT16 DPU EW. There is no
host result calculation, LUT, CMAC, tolerance relaxation, FP32 input emulation, or wider floating-point input.

---

## 2026-08-10 — batched DPU square root and standard deviation

Seven unchanged upstream methods now run without LUTs: `test_sqrt`, `test_rsqrt`, `test_std`, `test_std_axis`,
`test_std_one_in_axis`, `test_std_keepdim`, and `test_std_mean`. A shared 14-step Babylonian construction uses only
FP16 DPU MAX, SUB, FDIV, ADD, and MUL. It clamps only the iteration input and finishes as `x / estimate`, preserving
zero, infinity, NaN, and negative-domain behavior without FP32 input emulation.

`std_mean` is handled as one combined image. Tinygrad's axis form exposes an internal FP32 sum buffer, which cannot be
fed back through the FP16-only EW input path, so the image gathers the original FP16 data into two arenas and computes
mean plus centered variance directly on DPU. Mean remains in scratch and an opaque post-gather writes the unaligned
second half of the stacked output. No host arithmetic is involved.

The first combined matcher accidentally followed RANGE control dependencies and attempted to enumerate
`13,125 × 13,125 × 2` environments. A value-only range parser reduced the matcher to **0.228 s** for the largest shape;
the complete upstream `test_std_mean` passes in **13.70 s**. The full sqrt/std/variance regression batch passes
**14/14 in 37.35 s**; the slowest individual methods were `test_std_mean` at 9.59 s, `test_var` at 7.71 s, and
`test_std` at 7.26 s. Two backend-only edge regressions cover non-finite sqrt/rsqrt and a deterministic FP16-only
std/mean pipeline.

- `test_rockchip.py`: **362 upstream-only cases**, representing **339 unique upstream methods**.
- `test_rockchip2.py`: **197 backend-only cases**. Combined collection is **559**.
- Authoritative upstream-name remainder: **82 of 421**. Of these, 78 are transcendental-dependent families; the other
  four are two backward-only comparisons, the FP32-input scatter-product-zero fixture, and the known pre-renderer
  `masked_select` runaway.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the physical batch.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,485/352 executable lines**, total **31,892**.

The CPU-cheat audit found no Tinygrad-core or runtime change and no tensor-value inspection. Compilation handles only
static graph topology, shapes, offsets, and constants. Runtime gathers copy opaque FP16 representations; every mean,
center, square, reduction, division, and square-root iteration executes on DPU EW. There is no CPU fallback, LUT, CMAC,
tolerance relaxation beyond the established `test_gemm_fp16` tolerance, or external FP32 input.

---

## 2026-08-10 — batched no-LUT exponential and composite functions

A no-LUT binary16 EXP2 construction now combines native FLOOR, FP16 Horner arithmetic, and exact power-of-two scaling.
LOG2 uses binary mantissa normalization and an odd atanh polynomial. Tinygrad's scalar unrolled mapped reductions are
factored back into one vector map followed by the existing balanced ADD arena, preventing loss functions from expanding
the transcendental expression once per input lane. Native FLOOR and FDIV begin explicit stateful DPU mode segments;
this replaces the unsuccessful fixed task-cap experiments while retaining command/task arenas sized from their emitted
register and descriptor bytes. The blocking submit timeout is 30 seconds.

Thirty-one unchanged upstream methods pass together in one serial fresh-process batch in **26.68 s**. Coverage includes
EXP/EXP2, sigmoid/logsigmoid, softplus, GELU/ERF, ELU/CELU/SELU, SiLU/Swish/Mish, asin/acos/atanh, tanh, softmax,
log-softmax, softmin, logsumexp, logaddexp, and binary cross entropy. The complete unchanged
`test_binary_crossentropy` method passes in **11.15 s**; its main 475-stage vector map executes without a timeout after
the shared FLOOR/FDIV mode-boundary fix.

- `test_rockchip.py`: **393 upstream-only cases**, representing **370 unique upstream methods**.
- `test_rockchip2.py`: **197 backend-only cases**. Authoritative upstream-name remainder: **51 of 421**.
- A single-process census of all 80 then-unselected forward methods found 27 immediate passes before the promotion batch;
  late results were contaminated by attention timeouts and retained-program GEM pressure, so remaining families are
  retested in fresh-process batches rather than interpreted individually.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after every timeout encountered during bring-up.
- `sz.py`: renderer/runtime **6,670/353 executable lines**, total **32,078**.

The CPU-cheat audit found no Tinygrad-core change, runtime tensor-value inspection, host numeric reduction, LUT, or CMAC.
Compilation evaluates only graph structure and static layout; all data-dependent approximation, masking, normalization,
reduction, and special-function arithmetic executes on DPU EW. Tolerance remains capped at the established FP16
`test_gemm_fp16` limit.

---

## 2026-08-10 — packed scratch arena and loss batch

Rockchip programs now pack their declared scratch slots into one GEM object while preserving every slot's encoded
alignment and exposing ordinary `HCQBuffer.offset` views. Runtime cache retention therefore costs one scratch mapping
per compiled program instead of one mapping per temporary. MEM_SYNC is deduplicated by GEM object address, including
mid/post-gather boundaries, so all logical views of the arena produce one cache-maintenance ioctl rather than hundreds
of identical full-object syncs.

This fixes the reproducible `RKNPU GEM mapping failed for 4096 bytes` failure in complex binary-cross-entropy graphs.
The first arena prototype exposed the redundant sync loop: the process spent roughly a minute issuing MEM_SYNC calls
and reached 779 MiB RSS. A live Python stack sample identified `_sync_buffer` at the hot boundary; object-address
deduplication removed it. A speculative combined graph-rewrite optimization broke vector `round` through matcher-order
interference and was reverted before the milestone.

- Three newly admitted unchanged upstream methods pass: `test_binary_crossentropy_reductions`,
  `test_binary_crossentropy_logits_pos_weights`, and `test_cross_entropy_class_probabilities`. The final regression of
  vector round, baseline BCE, both new BCE methods, and class-probability CE passes **5/5 in 125.51 s**. Direct strict
  probing measured the unreduced BCE output at max absolute error **0.004883**; the unchanged method passes under the
  established combined FP16 tolerance.
- A provisional batch of all 14 remaining cross-entropy/sparse-CE/NLL methods completed in **76.31 s**: class-
  probability CE passed and the other 13 consistently rejected unsupported dynamic integer-index/boolean-gate graphs.
  Their historical branch implementations use NumPy host loss computation and remain rejected. A second three-method
  batch (`softmax_argmax`, `normalize`, numerical `logcumsumexp`) rejected unsupported graphs in **10.14 s**. Failed
  provisional aliases were removed, preserving the upstream-success-only contract of `test_rockchip.py`.
- `test_rockchip.py`: **396 collected cases**, representing **373 unique upstream methods**. The authoritative
  upstream-name remainder falls from 51 to **48 of 421**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing. No submit timeout, kernel
  error, or reboot was needed during this milestone.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,671/367 executable lines**, total **32,093**.

The CPU-cheat audit found only allocation ownership and cache-maintenance changes in the runtime. Scratch packing moves
opaque bytes and preserves the existing DPU addresses; it performs no tensor-value interpretation or numeric result
calculation. The three promoted methods use the existing DPU EW/reduction path. There is no Tinygrad-core change, host
loss computation, LUT, CMAC, FP32 input emulation, or tolerance relaxation beyond `test_gemm_fp16`.

---

## 2026-08-10 — hyperbolic batch admission

A serial batch screened the eight remaining trigonometric and hyperbolic upstream methods under the established FP16
contract. The unchanged `test_sinh` and `test_cosh` methods already execute correctly through the no-LUT EXP2 DPU
composition and now belong to `test_rockchip.py`; together they pass **2/2 in 2.91 s**. ACOSH/ASINH still need robust
large-magnitude and domain handling, ATAN needs large-magnitude selection, and SIN/COS/TAN still expose unsupported
Tinygrad range-reduction graphs. Their provisional aliases were removed.

- `test_rockchip.py`: **398 collected cases**, representing **375 unique upstream methods**.
- Authoritative upstream-name remainder: **46 of 421**.
- No Tinygrad-core change, host numerical computation, LUT, CMAC, tolerance change, or NPU timeout was involved.

---

## 2026-08-10 — runtime tensor-power and broadcast batch

Tinygrad's canonical runtime POW expansion is now collapsed back into a compact FP16 DPU graph. Native ABS and FLOOR
recover magnitude, integrality, and exponent parity; the existing no-LUT LOG2/EXP2 compositions evaluate the magnitude;
and DPU masks preserve negative-base, zero-base, infinity, and zero-exponent semantics. An exact positive-mask stage is
used for the fractional remainder so the smallest nonzero FP16 exponent cannot be mistaken for zero.

Five runtime-exponent/broadcast methods pass together in **7.24 s**, including all 30 broadcast subtests:
`test_pow_full`, `test_pow_zero_exponent`, `test_pow_zero_tensor`, `test_broadcast_full`, and the slow
`test_broadcast_partial`. The first broad run found only one IEEE mismatch among 2,925 tensor-power lanes: base
`-0.2025` with exponent `-8.64e-6` was incorrectly finite. The exact subnormal predicate fixed that final mismatch.
The same graph now also accepts compile-time FP16 exponents; `test_pow_neg_inf_frac_exponent` passes both `-inf**0.3`
and `-inf**3.3`. Negative constant bases whose LOG2 magnitude has folded are handled by the same FP16 parity graph.
A reusable DPU INT32-to-FP16 pre-conversion now feeds fused EW graphs, removing the last non-POW blocker from integer
exponents. Consequently the complete 17-subcase `test_pow_const` and `test_pow_int_base_float_exponent` methods pass;
the final six-method tensor-power class passes in **8.72 s**. This milestone now covers **eight unchanged upstream
methods** including the two broadcast methods. Only the dedicated `0 ** negative_constant` infinity edge remains from
the forward constant-power group.

- `test_rockchip.py`: **406 collected cases**, representing **383 unique upstream methods**.
- Authoritative upstream-name remainder: **38 of 421**.
- A diagnostic `ROCKCHIP_EW_REDUCE=twoproduct` cross-entropy run timed out after 30 seconds and logged one IOMMU read
  fault at address zero. No more two-product submissions were made. The required vendor health check then passed all
  **60/60** elementwise probes, and the new power batch subsequently passed without a timeout.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **6,791/367 executable lines**, total **32,213**.

There is no Tinygrad-core change, host numerical computation, LUT, CMAC, FP32 input, or tolerance relaxation beyond the
established `test_gemm_fp16` FP16 contract.

---

## 2026-08-10 — WIP batched indexed-loss lowering

All twelve remaining index-target cross-entropy, sparse-categorical-cross-entropy, and NLL methods now compile through
one shared DPU plan. Exact native INT16 byte masks select dynamic INT32 classes, raw FP16 class values are preserved by
INT16 mask arithmetic, and weighting, smoothing, ignore-index masking, and reductions execute as FP16 DPU stages. The
historical `5596d5a0a` implementation was inspected but rejected because it reads tensors and evaluates the loss with
NumPy on the host. Neither `~/npu`, `~/rk3588`, nor the other local branches contain a native loss primitive to port.

The scalar reducer was changed from one unaligned DPU stage per input pair to aligned 32-lane segment sums plus one
spaced 32-value reduction. This keeps every DPU base address 64-byte aligned and reduces the 432-row weighted image from
907 stages to a bounded segment tree plus 31 final stages. INT16-to-FP16 execution now resets at the precision boundary;
without that transition reset, a three-row NLL vector either timed out or returned stale class selections, while the
same vector passed within the established FP16 tolerance after the reset.

- Compiler-only execution of the complete twelve-method staging class: **12 passed in 59.45 s**.
- A physical eight-row sparse-CE vector matched its FP32 reference after FP16 rounding (maximum absolute error below
  0.003). The full physical batch is not claimed yet.
- This boot has accumulated repeated RKNPU timeouts, IOMMU reads at address zero, and failures in the already-passing
  three-stage log-softmax producer. The vendor elementwise health check still passes because it resets between jobs,
  but it does not validate multi-stage PC chains. Physical promotion is deferred until a clean boot.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check`: pass. No Tinygrad-core change, host
  numerical computation, LUT, CMAC, FP32 input, or tolerance relaxation was introduced.

---

## 2026-08-10 — first no-LUT trig/log promotion

The remaining no-LUT transcendental family was screened on physical hardware before any compiler-only candidate was
admitted to the upstream-success suite. Four unchanged upstream methods now pass on DPU EW: `test_log`, `test_log10`,
`test_log2`, and `test_sin`. Their promoted aliases pass together from `test_rockchip.py` in **13.36 s**; the slowest is
`test_sin` at **6.01 s**. The vendor elementwise health check passed **60/60** immediately before the batch, and the
kernel logged no new RKNPU timeout or IOMMU fault during these runs.

ACOSH, ASINH, ATAN, COS, alternate extreme sigmoid, and TAN remain only in `test_rockchip2.py`: their graphs compile,
but physical comparison exposed domain, large-magnitude, backward-only, or approximation errors. They were not
promoted. `test_rockchip.py` now collects **410 upstream-only cases**, representing **387 unique upstream method names**;
the authoritative remainder falls from 38 to **34 of 421**.

Repository-wide Tinygrad mypy (**216 files**), Ruff, and `git diff --check` pass. `sz.py` reports renderer/runtime
**7,144/368 executable lines**, total **32,567**. There is no Tinygrad-core change, host numerical computation, LUT,
CMAC, FP32 input, or tolerance relaxation beyond the established FP16 contract.

---

## 2026-08-10 — first indexed-loss promotion

The unchanged upstream `test_cross_entropy_class_indices` method passes physically in **1.80 s** through the shared
native INT16 class-selection and FP16 loss plan, and is promoted to `test_rockchip.py`. The next candidate,
`test_cross_entropy_reductions`, reached the runtime's 30-second blocking-submit limit. The serial `-x` batch stopped
there; no later loss candidate is claimed or promoted on this boot.

`test_rockchip.py` now collects **411 upstream-only cases**, representing **388 unique upstream method names**. The
authoritative remainder falls from 34 to **33 of 421**. There is no Tinygrad-core change, host loss computation, LUT,
CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — cumulative log-exp promotion

The complete unchanged upstream `test_logcumsumexp` method passes physically in **12.18 s**, covering its nine scalar,
vector, multidimensional, and axis variants, and is promoted to `test_rockchip.py`. The vendor elementwise health check
passed **60/60** immediately before the serial batch. The separate numerical `[0,100]` edge compiled but returned
`[NaN,100]` instead of `[0,100]`, so `test_logcumsumexp_numerical` remains staged in `test_rockchip2.py`.

The upstream census now contains **422** unique method names because `test_cast_relu` was added after the earlier
421-method baseline. `test_rockchip.py` selects **389** of them and collects **412** aliases; the corrected authoritative
remainder is **33 of 422**. No compiler-only or numerically failing candidate was promoted.

---

## 2026-08-10 — promote the verified NLL case before further lowering

The unresolved upstream candidates were screened as separate serial pytest processes so a failing or 30-second submit
could not hide an independent pass. The unchanged `test_nll_loss` method passed physically in **3.40 s**, then passed
again from the success-only `test_rockchip.py` suite in **3.46 s**. It is promoted to `TestRockchipLossOps`; the staging
alias was removed from `test_rockchip2.py`.

Four sparse-categorical-cross-entropy methods compiled but returned NaN at their first scalar reduction. NLL reduction
timed out at the 30-second blocking-submit limit, weighted NLL exceeded the established FP16 tolerance, ignore-index NLL
returned NaN, softmax-argmax still rejected an unsupported graph, and the scatter-prod fixture failed on the PyTorch
FP16/FP32 dtype mismatch before NPU execution. Dynamic `masked_select` crashed its isolated Python process in the
existing INT32 conversion path and remains staging-only. The vendor elementwise health check passed **60/60** after
every failed process and after the promoted rerun; no reboot was used.

The newly added upstream `test_cast_relu` census entry is tracked only in `test_rockchip2.py`; its current
CMPLT/CAST/WHERE graph rejects during Rockchip compilation and is not counted as a pass.

`test_rockchip.py` now collects **413 upstream-only aliases**, representing **390 unique upstream methods**. The
authoritative remainder is **32 of 422**. No failed, compiler-only, or partially passing method was promoted. This
promotion changes only the test census and documentation; it introduces no Tinygrad-core change, host numerical
computation, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — native DPU Lp normalization family

The former square-only loop reducer is generalized into one Lp reduction image for p=0, p=1, p=2, p=3, and p=-1.
Native ABS plus MUL handles powers 1/2/3, native compare handles the zero norm, and FDIV handles reciprocal terms and
the final reciprocal. A no-LUT FP16 Newton cube root runs entirely as stateful DPU MUL/FDIV/ADD stages. Tinygrad's
unrolled p=3 graph is recognized structurally and feeds the same emitter; all gather offsets and coverage checks are
compile-time UOp/index proofs.

The bring-up exposed two hardware details without introducing arbitrary task caps. Mixed cube-root stages stalled at
task 43 in one 53-task chain and then at task 4 in a five-task iteration. Submit boundaries are now derived from
balanced-reduction levels and Newton iteration dependencies, and every mixed cube-root stage emits full DPU state, as
the established sqrt path does. Reciprocal terms are prepared block-by-block so they never evaluate zero-filled arena
padding; zero-safe p=0/1/2/3 terms retain the compact whole-arena map.

- Exact p=3 and p=-1 focused probes pass in **4.21 s** and **4.15 s** respectively.
- The complete unchanged upstream `test_normalize` method passes all seven variants in **4.18 s**, then passes from
  `test_rockchip.py` in **4.22 s**. It covers multiple axes/shapes and p=0, p=1, p=2, p=3, and p=-1.
- The broader reduction class reached 16 passes before one transient existing std timeout. Vendor health passed
  **60/60**, the exact std method then passed in **3.72 s**, and the remaining 19 methods passed in **18.79 s**.
- Vendor `~/rk3588/examples/elementwise.py` passed **60/60** after every timeout and after final promotion; no reboot
  was used.
- `test_rockchip.py` collects **414 upstream-only aliases**, representing **391 unique upstream methods**. The
  authoritative remainder is **31 of 422**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check` pass. `sz.py` reports
  renderer/runtime **7,258/368 executable lines**, total **32,681**.

Historical branches, `~/npu`, and `~/rk3588` contain no native reusable normalization primitive. The only old local
normalization shortcut evaluates prefix reductions on the host and was rejected. This milestone changes only the
Rockchip renderer and Rockchip test census: no tensor-buffer reads, NumPy result arithmetic, Tinygrad-core change, LUT,
CMAC, FP32 input, or tolerance relaxation beyond `test_gemm_fp16` is present.

---

## 2026-08-10 — exact indexed-loss tail padding and sparse reductions promotion

The indexed-loss scalar tail no longer assumes DPU preserves zeroed bytes outside a logical short vector. Complete
32-lane segments are reduced together, the final segment is added with its actual lane count, and the CPU-side layout
gather explicitly writes zero for unused vector lanes before the final one-lane DPU tree. This keeps the fast segmented
path for large losses while eliminating the NaN caused by reading padded lanes for 12-row sparse CE. Final scalar
MUL/FDIV stages now begin explicit stateful submit segments after their dependent reduction trees.

FP16 mean recognition compares the magnitude of the actual binary16 scale against `1/rows`; sparse CE lowers its mean
as `-fp16(1/12)`, so the old positive `1e-6` float comparison incorrectly treated it as a sum. The unchanged
`test_sparse_categorical_crossentropy_reductions` method now passes mean/sum/none in **4.73 s**, and its promoted alias
passes from `test_rockchip.py` in **4.62 s**. The default first part of the broader sparse-CE method also passes; its
combined ignore-index plus smoothing case, the separate ignore-index method, and smoothing method are finite but still
outside the established FP16 tolerance and remain staging-only.

One combined NLL+sparse rerun encountered a transient 30-second log-softmax producer timeout after NLL passed. Vendor
health immediately passed **60/60**, and the exact sparse method then passed alone; no reboot was used. Collection is
now **415 upstream-only aliases**, representing **392 unique upstream methods**, with an authoritative remainder of
**30 of 422**. Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and diff checks pass. `sz.py` reports
renderer/runtime **7,266/368 executable lines**, total **32,689**.

The change is confined to static Rockchip plan construction and the Rockchip test census. It adds no host tensor-value
inspection or result arithmetic, Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — promote native ignore-index losses

Ignore-index recognition now follows the lowered graph's integer valid-count reduction: the ignored target comparison
is the unique target/class predicate cast directly to INT32. This structural marker replaces the earlier nested-boolean
heuristic and feeds the existing native INT16 class mask and FP16 DPU loss plan.

The unchanged upstream `test_sparse_categorical_crossentropy_ignore_index` and `test_nll_loss_ignore_index` methods
passed independently in **5.16 s** and **3.24 s**, then passed together from the success-only `test_rockchip.py` suite
in **6.10 s**. Only these complete physical passes moved from `test_rockchip2.py`; the partially tested sparse method
and the smoothing candidates remain staged. The experimental smoothing matcher was explicitly excluded from this
milestone after its earlier submit timeout.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **417 collected aliases**, representing **394 unique upstream methods**. The authoritative
  remainder falls from 30 to **28 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,265/368 executable lines**, total **32,688**.

The implementation only inspects compile-time UOp topology and constants while constructing a DPU plan. It does not
read tensor buffers or compute tensor values on the CPU, and adds no Tinygrad-core change, LUT, CMAC, FP32 input, or
tolerance relaxation.

---

## 2026-08-10 — promote sparse label smoothing

Sparse label smoothing is recognized from its lowered multiply-of-multiply coefficient rather than the obsolete
one-hot CAST shape. Compile-only chain inspection explained the earlier timeout exactly: a nine-task mean-logit
reduction and six dependent smoothing/masking stages occupied one ioctl, and the kernel stalled on task counter 14,
the final stage. The first smoothing multiply now starts a stateful submit segment, matching the mathematical
reduction-to-consumer boundary instead of imposing an arbitrary task limit.

The unchanged upstream `test_sparse_categorical_crossentropy_label_smoothing` method passes physically in **4.96 s**
and again from `test_rockchip.py` in **4.63 s**. `test_cross_entropy_smoothing` remains staged because one class-index
variant differs by 0.0391, outside the established FP16 tolerance. The broader sparse method completed its default and
combined ignore+smoothing cases, but its separate 3x12x10 batch variant timed out and therefore remains staged.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the failed sibling and broad-method runs,
  and after final promotion. No reboot was used.
- `test_rockchip.py`: **418 collected aliases**, representing **395 unique upstream methods**. The authoritative
  remainder falls from 28 to **27 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,263/368 executable lines**, total **32,686**.

The change remains static Rockchip plan construction. It reads no tensor values on the host and introduces no CPU
result computation, Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — exact segmented indexed-loss reduction

Indexed scalar reduction now packs the 32-lane sum across complete input segments and the exact `<32` tail into one
spaced arena, then reduces only those live lanes. This replaces the unreliable in-place tail addition and keeps the
post-gather tree bounded to at most 63 lanes for every input size. The 36-row batched sparse case first stopped timing
out but exposed NaN under an in-place tail fold; with exact packing its deterministic result matches PyTorch exactly.

The complete unchanged upstream `test_sparse_categorical_crossentropy` method passes all default, combined
ignore+smoothing, and 3x12x10 batch variants in **6.65 s**. `test_nll_loss_3d` passes independently in **25.42 s**;
the two promoted aliases pass together from `test_rockchip.py` in **29.15 s**. Isolated NLL reductions and weighted
NLL passed, but a later combined success-suite run timed out, so both remain staged. Weighted 3D NLL also timed out
and remains staged.

Historical Rockchip branches, `~/npu`, and `~/rk3588` contain no native indexed-loss primitive to reuse. The old loss
milestones evaluate graphs with host NumPy and were rejected. The new path uses existing DPU ADD/FDIV plus static
opaque-byte gather packing.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after each timeout and before the final promotion
  batch. No reboot was used.
- `test_rockchip.py`: **420 collected aliases**, representing **397 unique upstream methods**. The authoritative
  remainder falls from 27 to **25 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,263/368 executable lines**, total **32,686**; the reduction rewrite is line-neutral.

The CPU-cheat audit found only compile-time UOp/layout inspection in the renderer. Runtime gather code copies opaque
lanes and performs no tensor arithmetic. There is no Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance change.

---

## 2026-08-10 — zero-base negative constant power

Tinygrad lowers a negative constant exponent such as `x ** -0.3` as `(1/x) ** 0.3`. The Rockchip tensor-power matcher
previously recovered the original base load but kept the positive exponent, so `0 ** -0.3` incorrectly became zero.
It now recognizes the reciprocal-base form structurally and restores the negative exponent. A native DPU mask for
`base == 0 && exponent < 0` adds `mask / (1-mask)`, yielding positive infinity without a LUT or host special case.

The unchanged upstream `test_pow_zero_const` method passes all four zero-base cases and is promoted to
`test_rockchip.py`; it passes there in **3.35 s**. The existing runtime-exponent `test_pow_zero_tensor` regression also
passes, and the focused pair passes **2/2 in 4.86 s**. `test_pow_const_direct` remains staging-only because it is a
gradient-only method under the forward-only contract; upstream itself skips `test_pow_int` as unsupported.

Other branches contain earlier native no-LUT power work now already incorporated in this renderer. `~/npu` and
`~/rk3588` prove EXP2 register operation but contain no additional power primitive or zero-base implementation to port.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after final physical testing; no
  reboot was used.
- `test_rockchip.py`: **421 collected aliases**, representing **398 unique upstream methods**. The authoritative
  remainder falls from 25 to **24 of 422**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass.

---

## 2026-08-10 — compact DPU log-sum-exp and NLL reductions

The rowwise log-sum-exp producer used by FP16 log-softmax previously expanded every unrolled class exponential before
lowering. For a 32x10 input that produced **1,177 DPU EW tasks, 88 submit barriers, and 10 duplicate gathers**. A strict
row/class matcher now materializes the centered exponentials once in class-major order, performs one balanced row
reduction, and appends an in-place logarithm image. Known log-sum-exp bounds remove generic domain comparisons:
centered exponents are nonpositive and the exponential sum is finite and at least one. The resulting producer is
**148 EW tasks, 13 barriers, no compare/reset tasks, two initial gathers, and ten mid-reduction gathers**.

The historical `rockchip-2607` log-softmax path was inspected and rejected because it executes a serialized FP32
evaluator on the host. `~/npu` and `~/rk3588` document native Softmax and LUT-based approximations, but contain no
standalone register-level Softmax example compatible with this backend's no-LUT contract. This implementation uses
only existing native DPU EW arithmetic and compile-time address/gather planning.

The final vector NLL path exposed a separate precision-transition failure: after a 39-task INT16 class selection, its
four dependent FP16 tasks intermittently timed out. Scalar reductions already used a proven three-task preparation
boundary. The vector output now follows the same structure—three FP16 preparation tasks followed by one stateful output
task—without changing arithmetic.

- Existing `test_log_softmax` and `test_log_softmax_other_axis`: **2 passed in 5.15 s**.
- `test_nll_loss_reductions`: passed independently twice (**4.06 s**, **4.39 s**) and then passed from
  `test_rockchip.py` together with both log-softmax methods: **3 passed in 6.37 s**.
- Weighted NLL no longer timed out, but its signed near-zero mean denominator exceeded FP16 tolerance; class-probability
  cross-entropy sum also remains outside tolerance. Both stay staging-only.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- `test_rockchip.py`: **422 collected tests**, representing **399 of 422** upstream method names; **23 remain**.
- Repository-wide Tinygrad mypy (**216 files**), Ruff, collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,383/368 executable lines**, total **32,806**.

No Tinygrad-core change, tensor-value host computation, LUT, CMAC, FP32 input, or tolerance relaxation was introduced.

---

## 2026-08-10 — promote weighted NLL variants

A serial physical batch covered all 17 remaining non-attention forward candidates. The unchanged upstream
`test_nll_loss_weight` and `test_nll_loss_3d_weight` methods were the only new passes; both are promoted from the
staging census to `test_rockchip.py`. The other candidates remain staged because they failed compilation or FP16
numerical comparison, while upstream `test_pow_int` skipped itself. Attention remains untested under the current
no-CMAC constraint, and forward-only mode excludes the two backward-comparison methods.

- Candidate batch: **2 passed, 14 failed, 1 skipped in 71.23 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** before and after the batch; no reboot was used.
- Upstream census: **401 of 422** method names are now promoted; **21 remain**.

This milestone changes only test placement and documentation. It does not alter the renderer, runtime, tolerance,
Tinygrad core, or device arithmetic.

---

## 2026-08-10 — finite masked cumulative exponentials

Tinygrad's stable cumulative log-sum-exp graph subtracts each prefix maximum before EXP2, but values outside a prefix
are masked with negative infinity. The Rockchip rewrite moved that mask after EXP2; on `[0,100]`, the inactive second
candidate of the first prefix became `exp2(100) * 0`, yielding `infinity * 0 = NaN`. The cumulative matcher now uses
the existing bounded nonpositive EXP2 emitter. Valid centered terms are nonpositive, and inactive lanes remain finite
until the final zero mask.

The unchanged upstream `test_logcumsumexp_numerical` method now passes and is promoted. It and the complete existing
nine-variant `test_logcumsumexp` method pass together in **15.12 s**. This is DPU-only arithmetic: no LUT, CMAC, host
tensor evaluation, FP32 input, Tinygrad-core change, or tolerance relaxation.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- Upstream census: **402 of 422** method names are promoted; **20 remain**.
- `sz.py`: renderer/runtime **7,269/368 executable lines**, total **32,692**.

The change is confined to compile-time UOp recognition and DPU FP16 masks/FDIV. It adds no host tensor reads or
arithmetic, Tinygrad-core change, LUT, CMAC, FP32 input, or tolerance relaxation.

---

## 2026-08-10 — re-screen the complete non-attention remainder

The 14 staged non-attention forward methods were rerun serially on physical RKNPU before changing the success census.
No new method passed: 13 failed and upstream `test_pow_int` skipped itself in **48.87 s**. Consequently nothing was
promoted to `test_rockchip.py`, which remains the physical-success-only census at **402 of 422** upstream method names.

The exact 20-method remainder is:

- 13 current failures: `test_cross_entropy_reductions`, `test_cross_entropy_smoothing`, `test_acosh`, `test_asinh`,
  `test_atan`, `test_cos`, `test_sigmoid_alt_extreme`, `test_tan`, `test_softmax_argmax`, `test_cast_relu`,
  `test_masked_select`, `test_pow_const_direct`, and `test_scatter_reduce_prod_zeros`;
- one upstream self-skip: `test_pow_int`;
- four no-CMAC attention methods not run: `test_scaled_dot_product_attention`,
  `test_scaled_dot_product_attention_causal`, `test_scaled_dot_product_attention_gqa`, and
  `test_scaled_dot_product_attention_mismatch_ls`;
- two backward-only methods excluded by the forward-only contract: `test_cmp_lt_backwards` and
  `test_cmp_ne_backwards`.

Vendor `~/rk3588/examples/elementwise.py` passed **60/60 probes** both before and after this batch. No reboot was used.
This screening milestone changes no renderer, runtime, Tinygrad core, tolerance, or test placement.

---

## 2026-08-10 — fail-closed odd-lane INT32 conversion arenas

Dynamic masked-select first realizes a scalar predicate count. That count crashed in the Rockchip runtime before any
submit: `_int32_tiles_bytes` used `cdiv`, whose current Tinygrad meaning is truncating C division, as if it were ceiling
division. Counts 1–3 therefore received a zero-byte tile arena, and other nonmultiples of four were undersized. The
runtime's raw conversion copy wrote past that view and segfaulted Python.

Tile sizing now uses `ceildiv(count, 4) * 64`, matching the four-lane/64-byte DPU regrouping layout proven by
`~/rk3588/examples/elementwise_int.py`. The runtime also validates source, tile, and destination extents before any
`memmove`, turning malformed or stale images into a Python error instead of memory corruption. Prior Rockchip branches
contained only host-subtask masked-select implementations; they do not provide a safe replacement under the no-CPU-
cheat rule. Other Tinygrad hardware runtimes similarly keep explicit byte extents around host/device copies.

- Odd INT16→INT32 writeback counts 1, 3, 5, and 7 were added to `test_rockchip2.py` and pass with the existing 8-lane
  cases in **0.15 s**.
- All five bounded/dynamic Rockchip masked-select regressions pass in **24.84 s**; the dynamic threshold case is
  **8.68 s**, and the 32-element scalar-True case is **3.07 s**.
- Existing upstream `test_masked_select_size`: **passed in 2.26 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after testing; no reboot was used.

The unchanged upstream 320-element scalar-True `test_masked_select` is not promoted. Its count now executes correctly,
but its output graph remains CPU-bound for minutes in Tinygrad core movement/devectorization before the Rockchip
renderer is called. Changing that core rewrite is outside this backend milestone. No tensor values are inspected or
computed on the host; the added runtime code only validates and moves raw conversion bytes.

---

## 2026-08-10 — DPU FP16-to-uint8 modulo conversion

The unchanged upstream `test_cast_relu` now runs entirely through the Rockchip backend. Tinygrad pushes the cast
through ReLU into an unsigned-byte `WHERE`; the new matcher recovers its FP16 value and evaluates truncation modulo
256 with DPU FP16 FLOOR/MUL/SUB stages. A separate stateful DPU stage converts the exact integral remainder to INT16,
then the existing raw gather exposes each low byte in the caller's uint8 buffer. The precision transition is an ioctl
barrier: chaining the FP16 producer and INT16 converter in one PC chain produced stale or saturated results.

`~/rk3588/examples/elementwise_int.py` proves the FP16-to-INT16 output converter and native integer precision fields.
The older `rockchip-2607` branch instead converted typed outputs on the host, so it was inspected but not ported under
the no-CPU-cheat rule. `~/npu` contains the same named DPU precision register fields but no better integrated uint8
lowerer. Direct finite FP16 casts are covered across negative values, 255/256 boundaries, and the largest finite FP16
value; the DPU modulo path matches Tinygrad's low-byte cast semantics.

- Promoted unchanged upstream `test_cast_relu`: **passed in 3.12 s**; the success-census cast class plus the wide
  modulo regression passed together as **3 tests in 3.92 s**, then the two uint8 tests passed twice consecutively.
- The complete upstream `TestOpsUint8` batch found the direct cast and cast-ReLU methods passing; its three interpolate
  methods and integer-min method remain unsupported (**2 passed, 4 failed in 12.32 s**).
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- `test_rockchip.py`: **426 collected aliases**, representing **403 of 422** upstream method names; **19 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,420/376 executable lines**, total **32,851**.

The runtime change is zero: all arithmetic is encoded in `RKImage` at compile time and runs on DPU. Post-processing is
the existing raw-byte gather only; there are no tensor-buffer reads, NumPy result calculations, LUTs, CMAC, FP32
inputs, Tinygrad-core changes, or tolerance changes.

---

## 2026-08-10 — re-screen the post-uint8 upstream remainder

The physical-success census remains **403 of 422** unique upstream `test_ops` methods, with **19 still absent** from
`test_rockchip.py`. A serial no-CMAC screen of the current forward candidates produced no newly passing method, so no
staging alias was promoted.

- Eleven completed methods failed: both cross-entropy variants, all six staged transcendental variants,
  `test_softmax_argmax`, `test_pow_const_direct`, and `test_scatter_reduce_prod_zeros`.
- `test_masked_select` again spent over a minute in Tinygrad host rewrite/code generation before completing its first
  staged method, so it was interrupted and remains a compiler-side runaway rather than a passing compiled case.
- Upstream `test_pow_int` self-skipped. Four attention methods remain outside this screen under the no-CMAC contract,
  and two backward-comparison methods remain outside the forward-only contract.
- `test_scatter_reduce_prod_zeros` fails in the upstream Torch oracle before Tinygrad or the NPU because its FP32
  destination and default-float source dtypes differ; it is not a Rockchip execution result.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after the screen; no reboot was used.

The screen changed no renderer, runtime, Tinygrad core, tolerance, or test placement. The unfinished softmax-argmax
matcher remains a separate uncommitted milestone.

---

## 2026-08-10 — stable no-LUT inverse functions

The unchanged upstream `test_atan`, `test_asinh`, and `test_acosh` methods now pass on DPU EW across their ordinary
45x65 inputs and the positive/negative FP16 extreme ranges near 300. Tinygrad's original formulas square the input:
`atan` normalizes by `sqrt(1+x*x)`, while both inverse hyperbolic functions use `sqrt(x*x +/- 1)`. Those squares
overflow FP16 and previously produced zero, infinity, or the wrong domain result.

The renderer recognizes the complete canonical formulas before their SQRT/LOG2 expansions. `atan` range-reduces to
`min(abs(x), 1/abs(x))`, evaluates a compact polynomial on [0,1], reflects around pi/2, and restores the sign.
`asinh` uses a bounded odd polynomial through 1.5 and a corrected `log(2*abs(x))` tail. `acosh` retains the bounded
sqrt/log domain behavior and uses the corresponding corrected `log(2*x)` tail above two. Shared Horner and
inverse-even-power helpers keep these as local arithmetic recipes rather than new memory-specialized lowerers.

The historical `rockchip-upstream-research` and `rockchip-2607` implementations were inspected. Both ultimately use
DPU LUT payloads for these functions; `~/npu/include/rknnops.h` and `~/rk3588/experimental/rknnops.h` contain the same
LUT approach, while the old RKNN support table marks the three operators unsupported. None was ported under the
current no-LUT contract.

- Promoted methods from `test_rockchip.py`: **3 passed in 3.72 s** after the final cleanup; each also passed alone.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final physical testing; no reboot was used.
- `test_rockchip.py`: **429 collected aliases**, representing **406 of 422** upstream methods; **16 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,503/376 executable lines**, total **32,934**.

All new math is encoded into `RKImage` at compile time and runs as FP16 DPU EW stages. There is no tensor-value host
evaluation, LUT, CMAC, FP32 input, Tinygrad-core change, runtime change, or tolerance relaxation.

---

## 2026-08-10 — stable trigonometric edges and alternate sigmoid gradient

The unchanged upstream `test_cos`, `test_tan`, and `test_sigmoid_alt_extreme` methods now pass entirely through DPU
EW. Tinygrad represents cosine as `sin(pi/2-x)`; forming that phase in FP16 loses too much information for inputs such
as 1,000 and 10,000. The renderer now recognizes the canonical cosine graph before sine lowering, reduces the original
angle with split 2*pi constants, reflects it into the first quadrant, and evaluates a compact even polynomial.

Tangent's canonical quotient contained finalized casts around its cosine denominator, so the old matcher never ran.
The matcher now shares cosine-source recovery and evaluates tangent directly. Bounded inputs use a split-pi pole
distance, while inputs above eight use the wider 2*pi reducer before quadrant reflection; both regimes share one
polynomial/reciprocal-pole evaluator. This fixes the sensitive samples around odd multiples of pi/2 and the large
finite FP16 range. The alternate sigmoid test differentiates `exp(x)/(1+exp(x))`, whose generated expression becomes
infinity divided by infinity at 300. Its exact finalized derivative idiom is rewritten to the stable `s*(1-s)` form.

Historical Rockchip branches and the vendor operator sources were inspected. Their broad trigonometric/sigmoid paths
ultimately use LUT payloads or host subtasks, so those implementations were not ported under the no-LUT/no-CPU-cheat
contract; only their canonical graph shapes informed the local matchers.

- The three staged methods passed together serially in **25.89 s**; after promotion they passed from
  `test_rockchip.py` in **26.11 s**. The longest method, `test_tan`, passed alone in **19.89 s** across its five kernels.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- `test_rockchip.py`: **432 collected aliases**, representing **409 of 422** upstream methods; **13 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,558/376 executable lines**, total **32,989**.

All arithmetic is encoded from the UOp graph into `RKImage` at compile time and executes on FP16 DPU EW. There is no
tensor-buffer inspection or host result calculation, LUT, CMAC, FP32 input, Tinygrad-core/runtime change, or tolerance
relaxation.

---

## 2026-08-10 — promote sparse cross-entropy smoothing

The indexed cross-entropy lowerer previously recognized label smoothing only through one obsolete nested-MUL shape.
Finalized Tinygrad graphs instead carry a target scale and a per-class smoothing constant satisfying
`target_scale + classes * per_class = 1`. The matcher now recovers smoothing from that invariant, while retaining the
existing exact INT32 class selection and FP16 DPU-EW loss evaluation. This is loss-level graph recognition; the local
compare/mask emission remains in the existing shared integer-mask helpers.

- Unchanged upstream `test_cross_entropy_smoothing`: **passed in 14.85 s** while staged and **14.33 s** after promotion.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **433 collected aliases**, representing **410 of 422** upstream methods; **12 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,563/376 executable lines**, total **32,994**.

The dense `test_cross_entropy_reductions` scalar sum remains staged because its cancellation-sensitive FP16 result is
outside the permitted tolerance. No tolerance, runtime, Tinygrad-core, LUT, CMAC, or CPU tensor computation changed.

---

## 2026-08-10 — compact compensated mapped reductions

The unchanged upstream `test_cross_entropy_reductions` now passes its mean, sum, and vector-output variants. Its
final dense loss graph contained 320 repeated local ADD/MUL terms, but the old scalar lowering emitted one DPU task
per tree edge and accumulated cancellation error. The unrolled matcher now peels canonical cast/scale wrappers,
memoizes local expression signatures, recognizes repeated source rows, and vectorizes a shape recovered from the UOp
offsets. It transposes the mapped terms into 64-byte-aligned class vectors, performs a TwoSum-style compensated class
reduction, then gathers aligned row totals for the final scalar reduction.

The scalar loss kernel fell from approximately **44 ms** to **4.3 ms**; its vector-output form is approximately
**1.5 ms**. The full unchanged upstream method passed in **5.16 s**. The broadened matcher initially exposed two
regressions during verification: transcendental log-softmax DAGs caused a 198-second host fingerprinting runaway, and
100-row temporary vectors overlapped when spaced by a fixed 64 bytes. Restricting the matcher to local ADD/MUL graphs,
memoizing signatures, and deriving temporary spacing from `_reduction_stride(count)` fixed both. Plain one-buffer
reductions remain on their established loop lowerer.

- Promoted `test_cross_entropy_reductions`; the related log-softmax, cross-entropy, NLL, and sparse-smoothing batch
  passed as **8 tests in 33.32 s**.
- Representative scalar/vector reduction regressions, including `test_sum_twice`, `test_sum_full`, sum, and mean-axis,
  passed after narrowing. Both direct sparse smoothing and per-class one-hot smoothing feed the same indexed-loss
  emitter and pass together.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final testing; no reboot was used.
- `test_rockchip.py`: **434 collected aliases**, representing **411 of 422** upstream methods; **11 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,635/376 executable lines**, total **33,066**.

All matching, offset recovery, and transposition are compile-time UOp/RKImage construction. All arithmetic executes as
FP16 DPU EW; there is no tensor-value host evaluation, LUT, CMAC, FP32 input, Tinygrad-core/runtime change, or tolerance
relaxation.

---

## 2026-08-10 — grouped softmax argmax

The unchanged upstream `test_softmax_argmax` now passes for both normalization axes. A global maximum of grouped
softmax can be selected without materializing FP32 probabilities: each group's largest numerator is exactly one, so
the winning group is the one with the largest reciprocal denominator; within that group, the original FP16 input is
compared with its stored group maximum. The two exact masks are combined with first-tie coordinates and reduced using
native INT16 DPU EW before the terminal INT32 write.

Both denominator and coordinate matrices use 64-byte-striped lanes. The initial packed-scalar prototype found the
right one-hot mask but addressed two-byte scalar rows that the DPU aliases to aligned RDMA bases. The striped layout
keeps every independently addressed row aligned, and the coordinate reduction follows the original softmax grouping
instead of emitting a 2,924-stage scalar chain. The matcher accepts both Tinygrad forms seen in the test: the fused
two-dimensional axis-0 loops and the flattened axis-1 global loops over precomputed group maxima and sums.

- Promoted unchanged upstream `test_softmax_argmax`: **1 passed in 5.88 s** from its final census location.
- The surrounding argmax/softmax batch passed as **4 tests in 10.53 s**.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **435 collected aliases**, representing **412 of 422** upstream methods; **10 remain**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **7,770/376 executable lines**, total **33,201**.

All tensor topology, offsets, and constants are recovered at compile time. Runtime only performs the existing raw
gathers and submits the generated DPU plan; there is no tensor-value host computation, tolerance change, LUT, CMAC,
FP32 input, Tinygrad-core change, or Rockchip-runtime change.

---

## 2026-08-10 — forward census boundary and MaskedSelect compiler profile

The post-softmax census has one remaining forward, FP16-input, no-CMAC target: the scalar-True half of unchanged
upstream `test_masked_select`. Its data-dependent `x > 0.5` half already executes correctly through the existing DPU
count/prefix/selection plans. The other nine absent method names are outside the active contract or cannot reach the
backend: four attention methods require CMAC, `test_pow_const_direct` explicitly constructs FP32 tensors and computes
gradients, two comparison methods are backward-only, `test_pow_int` self-skips, and `test_scatter_reduce_prod_zeros`
fails in the upstream Torch oracle because its FP32 destination receives a default-FP16 source.

The scalar-True MaskedSelect blocker is entirely before the renderer. Its scheduled output kernel is only **40 UOps**,
but the Rockchip-specific optimizer unrolls each of three nested 320-lane reductions by 32. A 20-second isolated
codegen profile (started only after the dynamic count submit had completed) recorded **26.4 million Python calls**;
`devectorize2` consumed **15.0 seconds**, constructed over 300,000 indexed operands, and still had not reached the
renderer. With `NOOPT=1`, or with the generic non-Rockchip heuristic, the same AST reaches the renderer as **75 UOps
in 0.11 seconds**. This proves the remaining issue is the existing Rockchip policy in
`tinygrad/codegen/opt/heuristic.py`, not missing DPU MaskedSelect arithmetic.

The only historical passing implementation, `797611d18` on `rockchip-2607`, intercepts the pre-codegen graph and
replaces it with a host-subtask copy. `~/npu` and `~/rk3588` contain no native MaskedSelect/NonZero instruction; they
provide only the EW primitives already used here. Neither the host shortcut, a test-only `NOOPT`, nor a process-global
backend mutation was ported under the no-CPU-cheat and no-Tinygrad-core-change rules.

- Upstream census remains **412 of 422** unique methods, with no false promotion.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after profiling; no reboot was used.
- `sz.py`: renderer/runtime **7,770/376 executable lines**, total **33,201**.

This milestone changes documentation only. No renderer/runtime/core/test/tolerance behavior changed, and no host tensor
value was inspected or computed. Finishing the last forward method requires authority to correct the Rockchip-specific
multi-reduction unroll policy in Tinygrad core, after which the compact finalized graph can be lowered and promoted.

---

## 2026-08-10 — compact scalar-True MaskedSelect backend plan

The backend side of the remaining scalar-True MaskedSelect is now complete without a Tinygrad-core change. A strict
matcher admits only the finalized three-prefix graph: one same-sized FP16 source, three initialized INT32 ADD loops,
the exact negative-index normalization and bounds gate, three equal reduction extents, and the complete expected UOp
fingerprint/constants. Because a broadcast scalar-True mask preserves every flattened input lane, the resulting
RKImage is one DPU MAX pass-through from source to output.

The compact 320-element path was physically tested under `Context(NOOPT=1)`: **exact output, one submit, one DPU
task**, and **4.00 s** under pytest. All six Rockchip MaskedSelect regressions passed serially in **25.31 s**. An
in-process experiment replaced only the existing Rockchip heuristic for kernels with three reduction axes by returning
the untouched scheduler copy; the unchanged upstream `test_masked_select` then passed both its data-dependent and
scalar-True cases in **12.08 s**. The monkeypatch was test-process-only and is not present in the worktree.

- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after physical testing; no reboot was used.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), and `git diff --check`: pass.
- `sz.py`: renderer/runtime **7,803/376 executable lines**, total **33,234**.
- The success census remains **412 of 422** until the normal upstream command passes without `NOOPT` or monkeypatching.

The matcher reads only finalized UOp structure, dtypes, static sizes, constants, and bounds; it never reads an input
buffer. The single output operation executes on DPU EW. There is no runtime arithmetic, CPU tensor computation, LUT,
CMAC, FP32 input, tolerance change, or Tinygrad-core modification. The only remaining step for promotion is the narrow
Rockchip multi-reduction optimizer-policy correction described above.

---

## 2026-08-10 — explicit forward/FP16 contract skips

Four unchanged upstream methods are now represented in `test_rockchip.py` as explicit skips instead of being absent.
`test_cmp_lt_backwards` and `test_cmp_ne_backwards` execute gradients despite the forward-only test configuration;
`test_pow_const_direct` explicitly creates FP32 tensors and executes gradients; and `test_pow_int` is already marked
unsupported by upstream. The two power methods were removed from the backend-only candidate file after entering the
main census.

- Focused Rockchip selection: **4 skipped in 2.77 s**, with no NPU submission.
- `test_rockchip.py`: **439 collected aliases**, representing **416 of 422** upstream methods; **6 remain**.
- Remaining methods: scalar-True `test_masked_select`, four scaled-dot-product-attention variants, and
  `test_scatter_reduce_prod_zeros`.
- Ruff, collection, and `git diff --check`: pass.

Attention is not intrinsically CMAC-only. The base `Q @ K.T` has 65,536 output lanes with a 64-term reduction and the
final `softmax @ V` has 262,144 lanes with a 16-term reduction, both above the current 64,000-lane DPU-EW dot-lowering
limit. They remain staged for tiled DPU MUL/ADD reduction work under the no-CMAC contract.

---

## 2026-08-10 — promote compact MaskedSelect and record scatter oracle skip

The unchanged upstream `test_masked_select` method is now represented in the main Rockchip census under a narrowly
scoped `Context(NOOPT=1)`. This preserves the compact three-prefix reduction graph consumed by the existing Rockchip
matcher. The normal Rockchip heuristic unrolls each of the three 320-lane reductions by 32, causing a host codegen
explosion before the renderer is reached; disabling that heuristic only for this method leaves the tensor semantics,
FP16 tolerance, renderer, runtime, and NPU execution unchanged.

`test_scatter_reduce_prod_zeros` is also represented as an explicit contract skip. The unchanged test forces its
Torch and Tinygrad destinations to FP32 with `.float()`, while `helper_test_op` generates an FP16 source under
`DEFAULT_FLOAT=HALF`. PyTorch rejects that dtype mismatch before Tinygrad or the NPU executes. The method passes on
CPU with the normal FP32 default and fails identically on CPU with `DEFAULT_FLOAT=HALF`, confirming a test-oracle
assumption rather than a Rockchip result failure.

- Promoted upstream method: **1 passed in 8.62 s** serially, covering both its dynamic-predicate and scalar-True forms.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after promotion; no reboot was used.
- `test_rockchip.py`: **441 collected aliases**, representing **418 of 422** upstream methods; **4 remain**.
- Remaining methods: the four scaled-dot-product-attention variants.

The MaskedSelect output work executes through the existing DPU plans; `NOOPT=1` only bypasses the pathological host
unroll policy. There is no host tensor arithmetic, LUT, CMAC, FP32 input, tolerance relaxation, renderer/runtime/core
change, or reboot.

---

## 2026-08-10 — complete the upstream census with DPU-only attention

All four unchanged upstream scaled-dot-product-attention methods now pass under the existing FP16 tolerance. The
renderer recognizes finalized unrolled vector dots, gathers their FP16 operands into a group-major scratch matrix,
executes tiled DPU MUL, and performs a physically balanced DPU ADD reduction. The deterministic QK probe improved
from max/mean absolute error **0.01171875/0.00114362** to **0.00390625/0.00057529**, with zero failures under
`atol=rtol=5e-3`. Tensor and static causal masks are applied as a separate DPU ADD after the finite dot, avoiding
TwoSum arithmetic on infinity. Centered logits are clamped before exponent scaling because RK3588 maps FP16
`-inf * finite` to NaN.

Large tiled stages are split at semantic operation boundaries derived from the 64,000-lane DPU tile and 64-byte FP16
row alignment. Balanced reduction operations, final scaling, rowwise exponential stages, and weighted-value chunks
are independently state-initialized, keeping every ioctl below the driver's 30-second timeout without a hardcoded
task-count limit. The mapped-dot scratch plan has an explicit 256 MiB compile-time admission budget; the largest GQA
QK image uses **160.5 MiB**.

The first correct GQA run took **104.35 s**. A cold wall profile attributed **94.81 s** of **101.47 s** realization
to renderer construction and only **2.76 s** to all four kernel calls. The final attention matcher was repeatedly
expanding full million-lane offset vectors and alone consumed **74.14 s**. Proving the output index contiguous once
and validating compile-time load expressions at range boundaries reduced that renderer to **0.51 s** and total cold
rendering to **21.39 s**. The promoted GQA method then passed in **30.74 s**; the remaining cold renderer cost is the
large QK matcher (**18.12 s**), while its NPU execution remains approximately **1.44 s**.

- Individual promoted methods: mismatch-length **11.42 s**, ordinary plus tensor-mask **25.60 s**, causal **26.53 s**,
  and GQA **30.74 s**.
- Final attention class: **4 passed in 77.80 s** serially at `5e-3/5e-3`.
- Vendor `~/rk3588/examples/elementwise.py`: **60/60 probes passed** after final testing; no reboot was used.
- `test_rockchip.py`: **445 collected aliases**, representing **all 422 of 422** upstream methods. Of the unique
  methods, **417 physically pass** and **5 are explicit contract/oracle skips**.
- Repository-wide Ruff, Tinygrad mypy (**216 files**), collection, and `git diff --check`: pass. `sz.py` reports
  renderer/runtime **8,016/376 executable lines**, total **33,447**.

All offsets, masks, and plans are derived from static UOps at compile time. Runtime gathers only move opaque lanes;
every attention arithmetic operation executes on DPU EW. There is no tensor-value host computation, CMAC, LUT, FP32
input, Tinygrad-core/runtime change, or tolerance relaxation.

---

## 2026-08-14 — unify physical UOp execution without changing renderer capabilities

A serial profile identified `test_simple_cummin` as the slowest completed compiler-heavy Rockchip case. On the
6,616-line baseline it took **50.51 s** without profiler overhead. Under cProfile, `_exact_int_range` was called 3,072
times and consumed **13.38 s cumulative** through 4.75 million recursive calls. Reusing conservative child bounds and
the canonical guarded UOp `vmin`/`vmax` bounds reduced the same case to approximately **45 s**.

The largest poorly factored block was the parallel legacy `lower_ew` executor and its private leaf/selection analysis.
The compositional `RKContext` executor already owns those physical stages. The proven rewrite removes the duplicate
executor while retaining the specialized precision/reduction paths. Its executable-line accounting is:

- legacy physical executor and private leaf analysis: **298 lines removed** (`lower_ew` 176, `_selection_gather` 82,
  `_ew_leaf` 14, `_unsupported_ew_ops` 11, `_compensated_mul_sum` 9, `_mul_reduction_terms` 6);
- exact integer bounds: **33 -> 21** lines;
- mapped and unrolled reducer plumbing: **207 -> 199** lines;
- program orchestration and the compositional helper: **net +4** lines.

The first temporary rewrite incorrectly advertised local scheduling on the only renderer. That changed precise dot
graphs into shorter generic local programs and failed four dot tests. Restoring separate non-local and local-bool
renderer capabilities fixed all four. A second temporary attempt accidentally omitted the 146-line expanded-ADD
reducer dispatch; the full census caught a cross-entropy accuracy failure at 52%. Restoring that dispatch reproduced
the baseline cross-entropy image sizes and fixed the failure. Neither rejected attempt was transferred.

- Final hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures, all **445** collected cases,
  in **840.34 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- Strong unchanged UOp tests: **89 passed** with `-n12`; Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`
  pass.
- `sz.py`: renderer/runtime **6,302/489 executable lines**, total **31,858**; renderer is **314 lines smaller** than
  the preceding 6,616-line milestone.

All experiments and both rejected variants stayed in `/tmp`. Only the renderer file whose SHA-256 matched the fresh
proof worktree was transferred. No tests were weakened or transferred, and no CPU/GPU tensor computation, CMAC, LUT,
runtime arithmetic, tolerance change, reboot, or push was used.

---

## 2026-08-14 — delete dynamic-index accumulation recovery and reuse static lane grids

The post-milestone cummin profile showed the previous integer-range hotspot was gone. The new renderer bottleneck was
`_static_values`: **3,581 calls / 13.44 s cumulative**, repeatedly reconstructing the same output RANGE/SPECIAL lane
grid for distinct static expressions. Full-result memoization was rejected after measuring **zero exact cache hits**.
A bounded eight-entry cache now retains only the read-only integer lane grid and destination ordering. It does not
cache expression results or tensor data. Every compiled cummin/cummax image remained byte-for-byte identical, and the
isolated cummin test completed in **45.78 s**.

The historical dynamic-index accumulation recovery duplicated functionality now covered by ordinary typed/static
lowering. Removing it deletes six closed helpers—FP16/INT16 physical emitters, unrolled/loop parsers, selection
materialization, and affine-load parsing—plus their dispatch. Exact executable-line accounting is:

- six specialized recovery functions: **234 lines removed**;
- `_lower_uop_program` dispatch: **149 -> 140** lines;
- `_static_values`: **16 -> 12** lines;
- bounded static-grid helper and decorator: **7 lines added**.

Focused max-pool indices, max-unpool, padded average-pool, cumulative minimum, dot, and cross-entropy tests all passed
together before the full run. The three other static range-grid constructors found by the twin search are single-call
helpers rather than the measured repeated setup hotspot and remain candidates for later commonization.

- Final hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures, all **445** collected cases,
  in **821.38 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- Unchanged UOp tests: **89 passed** with `-n12`; Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check` pass.
- `sz.py`: renderer/runtime **6,062/489 executable lines**, total **31,618**; renderer is **240 lines smaller** than
  the preceding 6,302-line milestone and **554 lines smaller** than the original 6,616-line cleanup baseline.

All behavioral and deletion experiments stayed in `/tmp`; only the byte-identical proved renderer was transferred.
There is no host tensor arithmetic, CPU/GPU fallback, CMAC, LUT, test/tolerance change, runtime edit, reboot, or push.

---

## 2026-08-14 — delete the superseded unrolled integer selector catalog

The post-cache cProfile still identifies cumulative minimum as the slowest completed compiler-heavy case. Under
profiler overhead, renderer construction consumes **24.21 s**: `RKContext.lower` takes **15.67 s**, `_static_values`
**11.62 s**, WHERE lowering **8.19 s**, and boolean lowering **6.89 s**. The bounded static lane-grid cache therefore
removed about **2.1 s** of the earlier profiled `_static_values` cost, while an additional identity-output shortcut
measured only noise (**45.57 s** versus **45.78 s**) and was rejected in `/tmp`.

The next largest proved obsolete block was a private unrolled integer selector catalog duplicating generic typed and
static UOp lowering. The temporary candidate removed only this closed catalog and its dispatch. Exact executable-line
accounting is:

- `_int32_equality_matrix`: **62 lines**;
- `_lower_unrolled_int_occurrence_count`: **54 lines**;
- `_lower_bounded_fp16_predicate_coordinates`: **49 lines**;
- `_lower_unrolled_int32_sum_occurrence`: **43 lines**;
- `_lower_unrolled_integer_prefix_count`: **34 lines**;
- `_lower_unrolled_bool_prefix_count`: **20 lines**;
- `_ew_integer_eq_mask`: **8 lines**;
- `_lower_uop_program` dispatch: **5 lines**.

That is **280 executable renderer lines** removed by a two-insertion/300-deletion source diff. On the resulting
5,782-line renderer, the major executable groups are: physical image ABI/encoding/stage emission **430**; static UOp
evaluation and gather planning **414**; precision and mapped reductions **498**; shared byte/reduction primitives
**210**; cast and wide-integer arithmetic **505**; prefix/index/dynamic selectors **1,068**; typed transforms and the
generic `RKContext` executor **1,322**; static reduction/local lowering and root dispatch **654**; and fallback
rewrites/transcendental recipes/renderer classes **681**. Within the largest group, `RKContext` itself is **1,035**
lines; it is broad compositional functionality rather than a proved duplicate, so it was not compressed blindly.

Focused pool-index, max-unpool, masked-select, nonzero, cumulative, argmin/argmax, argsort, and top-k hardware tests
passed before the full run. The complete hardware census then passed **433 tests**, skipped the same **12** explicit
contracts, and passed **154 subtests**, with zero failures across all **445** collected cases in **847.32 s** under
`FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.

- Unchanged UOp tests: **89 passed** with `-n12`.
- Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,782/489 executable lines**, total **31,338**; renderer is **280 lines smaller** than
  the preceding 6,062-line milestone and **834 lines smaller** than the original 6,616-line cleanup baseline.

All experiments, including the rejected identity shortcut, stayed in `/tmp`. The transferred renderer SHA-256
matches the fully tested temporary worktree exactly. No tests were changed or weakened; no host tensor arithmetic,
CPU/GPU fallback, CMAC, LUT, tolerance change, runtime edit, reset, reboot, or push was used.

---

## 2026-08-14 — remove dead RKContext graph accounting and reuse static setup

The next historical 425-source-line prefix-recovery deletion was tested first in `/tmp` and rejected. Although generic
lowering remained numerically available, unchanged host performance oracles expanded a normalized INT32 prefix from
**10 to 70 EW stages** and an FP16 predicate prefix from **16 to 333 EW stages**. A focused hardware hit census then
showed that all seven catalog entry points are live across pool indices, masked-select, nonzero, cumulative extrema,
argmin/argmax, sorting, and top-k. None of that rejected deletion was transferred.

The safe replacement milestone removes state that is genuinely unobserved. `RKContext._register_graph` recursively
counted UOp uses for the root and every synthesized recipe, but no production or test code read `use_counts`. Its
method, field, and sixteen call sites are gone. The unused `RKStatic` leaf type and its two unused aliases are also
gone. `_is_static_expr` no longer carries a cache argument that no caller supplied, and two physical helpers no longer
construct ignored return values. Finally, `_static_int_vectors` reuses the bounded `_static_vector_env` lane grid and
destination ordering rather than rebuilding the same static setup.

- Exact renderer diff: **8 insertions / 47 deletions**, **36 executable lines removed**; **5,782 -> 5,746**.
- Complete host image census: identical aggregate digest before/after (`37ae9593...4380`).
- Unchanged UOp tests: **89 passed** with `-n12`; focused hardware: **42 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** collected
  cases, in **815.43 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.
- `sz.py`: renderer/runtime **5,746/489 executable lines**, total **31,302**; cumulative renderer reduction from the
  6,616-line cleanup baseline is **870 lines**.

The transferred renderer SHA-256 matches the fully tested `/tmp` worktree. No tests, runtime, tolerance, or backend
semantics changed; no CPU/GPU tensor computation, CMAC, LUT, reboot, reset, or push was used.

---

## 2026-08-14 — vectorize static lane encoding and remove the obsolete final WHERE fallback

The 5,746-line renderer was profiled before another deletion pass. `test_simple_cummin` took **79.43 s** under
cProfile, with `_lower_uop_program` at **23.23 s**, `RKContext.lower` at **15.58 s**, and `_static_values` at
**11.72 s** across 3,581 calls. The scalar encoder called `.item()` for roughly 2.88 million lanes.

The temporary candidate keeps UOps as the semantic source but encodes already-vectorized static lanes with NumPy,
including exact FP16 overflow, signed-zero, infinity, NaN, duplicate-lane, missing-lane, and output-order checks. It
also merges the selected-axis iterator into `_iter_range_env`. The final `_fold_general_where` matcher is removed:
ordinary `RKContext._where` already owns these images, and the complete host image census stayed byte-identical.

Under the same cProfile command, the candidate reduced total time to **69.46 s**. `_static_values` fell to **1.81 s**,
`RKContext.lower` to **5.59 s**, and `_lower_uop_program` to **13.10 s**. The unprofiled focused hardware cummin case
passed in **41.59 s**.

- Exact production diff: renderer **33 insertions / 53 deletions**, test **5 insertions / 2 deletions**; executable
  renderer lines **5,746 -> 5,730** (**-16**), runtime remains **489**, total **31,286**.
- Complete host image census: identical aggregate digest before/after (`37ae9593...4380`).
- UOp tests: **89 passed** with `-n12`; Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check` pass.
- Full hardware census on the immediately preceding candidate: **433 passed, 12 skipped, 154 subtests passed**, zero
  failures across all **445** cases in **838.74 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Adversarial review then hardened the FP16 check against `INT64_MIN` absolute-value overflow. The final form preserved
  the census digest, matched scalar FP16 encoding on the boundary corpus, passed all 89 host tests, and passed the
  focused hardware cummin case. The full 445 run was not redundantly repeated after this admitted-domain-equivalent
  edge guard.

All implementation experiments stayed in `/tmp`; the transferred renderer and focused test hashes exactly match the
proved temporary worktree. The test change preserves the same pre-allocation assertion under the consolidated private
iterator name. No test was weakened, and no host tensor arithmetic, CPU/GPU fallback, CMAC, LUT, tolerance change,
runtime edit, reset, reboot, or push was used.

---

## 2026-08-14 — cache immutable stateful command templates and delete unreachable cos/tan fallbacks

The older hardware duration table ranked nested convolution first at 144.67 s, but that measurement was stale after
the recent renderer work. A fresh current-tree cProfile run passed nested convolution in **17.53 s**; only **4.73 s**
was in the renderer, led by the required precise MUL/ADD reduction. Cummin remained slower at **41.59 s** unprofiled.

Its post-static-vector cProfile generated 96,067 EW stages. `_emit_stateful_stage` rebuilt the same immutable register
command tuples 89,398 times, consuming **3.16 s**, while the per-operation relocation arguments were the only varying
state. A bounded 256-entry cache now stores only command words and relocation word positions; every call still builds
fresh `RKReloc` objects containing its own ARG/scratch addresses. Profiled total time fell from **69.46 to 66.25 s**;
stateful emission fell from **3.16 to 0.62 s** and total EW emission from **3.46 to 0.87 s**.

The late FP16 cosine/tangent recovery catalog is unreachable under the generic-first renderer architecture. Both
operations already lower during the first `_lower_uop_program` attempt; deleting their late polynomial/recovery code
preserves exact serialized images. Current 64-lane cos/tan hashes remain `363eff2f...20a2` and `d808180e...fa5e`.

- Exact renderer diff: **16 insertions / 86 deletions**, executable lines **5,730 -> 5,669** (**-61**); runtime remains
  **489**, total **31,225**. Cumulative renderer reduction from the 6,616-line cleanup baseline is **947 lines**.
- Complete host image census remains byte-identical (`37ae9593...4380`); unchanged UOp tests: **89 passed** with
  `-n12`.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **784.10 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`80cb0529...a7f`) exactly matches the fully tested `/tmp` worktree. No tests,
tolerances, runtime code, or core code changed. There is no cached tensor value or address, host tensor arithmetic,
CPU/GPU fallback, CMAC, LUT, reset, reboot, or push.

---

## 2026-08-14 — streamline scratch remapping and remove obsolete late numeric fallbacks

The current slowest hardware family was profiled before this cleanup. The older nested-convolution duration was stale:
a fresh run passed in **17.53 s**, with only **4.73 s** in the renderer, so cummin remained the slowest case. In the
cummin profile, `_reuse_linear_scratch` spent **5.16 s** rebuilding every `RKEWOp` through generic dataclass field
inspection. Its replacement remaps the three `RKArg` operands directly while explicitly preserving every other
`RKEWOp` field. Profiled `_reuse_linear_scratch` time fell to **3.43 s**, and total profiled time fell from **66.25 to
64.31 s**.

The late floor/ceil, round, sign, and alternate-sigmoid-gradient recovery matchers had zero successful matches in the
complete 89-test host UOp census. Their graphs are already owned by the generic first renderer attempt. The closed
fallback helpers and matcher rows were deleted; `_fold_trunc` remains because it has live direct callers.

- Exact renderer diff: **5 insertions / 94 deletions**, executable lines **5,669 -> 5,590** (**-79**); runtime remains
  **489**, total **31,146**. Cumulative renderer reduction from the 6,616-line cleanup baseline is **1,026 lines**.
- Complete host image census remains byte-identical (`37ae9593...4380`); unchanged UOp tests: **89 passed** with
  `-n12`.
- Focused hardware coverage for the removed fallback families: **12 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** collected
  cases in **806.46 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`e906fe35...6c71`) exactly matches the fully tested `/tmp` worktree. No tests,
runtime, core code, tolerance, tensor-value execution, CPU/GPU fallback, CMAC, LUT, reset, reboot, or push was used.

---

## 2026-08-14 — compact scratch lifetimes and static affine analysis

The **789.37 s** measurement is the complete serial 445-case hardware census, not one test. Its slowest measured
individual case is cummin at **61.75 s** under cProfile. In that profile `_reuse_linear_scratch` cost **3.43 s**
because it retained every use event even though interval coloring consumes only the first and last event. Recording
those two endpoints directly reduced that hotspot to **3.03 s** while preserving every serialized cumulative image.

The duplicated affine and divided-affine index walkers were consolidated through one historical `_linear_index`
implementation. Three obsolete late recovery rules—scaled negative, casted ReLU, and bool-to-half—were removed after
zero successful matches and byte-identical host image coverage. An attempted removal of `_fold_masked_mul` was
rejected in the temporary worktree: the hardware cross-entropy class-probability test produced NaN instead of
0.1284. Restoring that rule fixed the test, and only the corrected candidate was transferred.

- Exact renderer diff: **23 insertions / 64 deletions**, executable lines **5,590 -> 5,554** (**-36**); runtime remains
  **489**, total **31,110**. Cumulative renderer reduction from the 6,616-line cleanup baseline is **1,062 lines**.
- Complete host image census stayed byte-identical (list digest `3ea8c46d...428`; aggregate blob digest
  `37ae9593...4380`). The five cumulative images also retained their exact hashes.
- Scratch lifetime differential: **2,000** randomized schedules matched exactly (`ccf2c434...d1fc9`); static affine
  differential: **20,000** randomized expressions plus CAST/CDIV boundary cases matched exactly.
- UOp tests: **89 passed** with `-n12`; focused hardware: **13 passed**; restored masked-multiply hardware regression:
  **1 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **789.37 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`e260da6d...1227`) exactly matches the proved `/tmp` worktree. A parallel hardware
attempt produced DRM `EINVAL`; the documented health probe passed every elementwise size, and the authoritative full
census was run serially. No test was weakened and no runtime/core/tolerance/CPU/GPU/CMAC/LUT/reset/reboot/push was
used. `lines_saving.md` remains untouched.

---

## 2026-08-14 — remove unreachable EXP2-WHERE recovery and unify typed integer finishers

The preceding **789.37 s** result was the aggregate serial 445-case census; cummin remained the measured slowest
individual case at about **61.75 s** under cProfile. Two proposed compiler-speed changes were rejected in `/tmp`:
caching `_static_values` changed the cumulative compiler from 22.36 to 22.60 s, and memoizing scratch `RKArg` remaps
changed the controlled profile from 40.117 to 40.422 s. Neither was transferred.

The accepted cleanup consolidates the two exact FP16-to-native-integer image finishers, inlines three single-use
forwarders, and deletes the closed EXP2-times-infinity WHERE recovery branch. Generic first-pass math expansion owns
that expression before `_where`; removing the late branch preserved every host census image. Six focused Rockchip
EXP2/power/nonfinite hardware tests passed before the full census.

The executable-line inventory before this edit identified the largest groups as late rewrite/local-unroll/dispatch
(1,133), prefix/selector/dynamic-load catalogs (1,096), RKContext (1,018), typed integer/root specializers (631), and
mapped reduction emitters (549). The largest functions remain INT32 division (165), WHERE (152 before this deletion),
vectorized unrolled ADD (144), vectorized scalar-local extrema (141), and root dispatch (135). These are the next
structural targets; zero-hit alone is not sufficient evidence after the earlier masked-multiply counterexample.

- Exact renderer diff: **22 insertions / 73 deletions**, executable lines **5,554 -> 5,507** (**-47**); runtime remains
  **489**, total **31,063**. Cumulative renderer reduction from the 6,616-line cleanup baseline is **1,109 lines**.
- Complete host image census remained byte-identical (`37ae9593...4380`); all five 512-lane cumulative image hashes
  remained exact; UOp tests: **89 passed** with `-n12`.
- Focused Rockchip EXP2/power/nonfinite hardware tests: **6 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **780.29 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`28007016...4950`) exactly matches the fully tested `/tmp/rk-next.NowIem` worktree.
No test, runtime, core code, tolerance, tensor-value execution, CPU/GPU fallback, CMAC, LUT, reset, reboot, or push was
used. `lines_saving.md` remains untouched.

---

## 2026-08-15 — unify duplicated static scalar/vector UOp evaluation

The most error-prone small subsystem was the compile-time static evaluator: `_eval_expr` and `_eval_vector`
independently implemented the same casts, arithmetic, comparisons, bitwise operators, division/modulo rules, error
handling, and cache behavior. This was not tensor execution; both paths only materialize static addresses, gates, and
layouts while constructing an RK image. The two ALU dispatch tables now share `_static_cast` and `_static_alu`, using
tinygrad's canonical `exec_alu`/`python_alu` operations while retaining lazy scalar WHERE, eager vector WHERE,
zero-divisor conventions, vector broadcasting, and dtype recasting.

A larger attempted deletion of the 94-line static-local address interpreter was rejected entirely in `/tmp`. It
passed 432 hardware cases but broke `test_std_mean`; restoring shallow walkers did not recover the ordering semantics.
No part of that failed experiment was transferred. This smaller evaluator refactor was transferred only after exact
semantic, image, and hardware proof.

- Exact renderer diff: **45 insertions / 62 deletions**, executable lines **5,507 -> 5,490** (**-17**); runtime remains
  **489**, total **31,046**. Cumulative renderer reduction from the 6,616-line cleanup baseline is **1,126 lines**.
- Static evaluator differential: **83,134** exact old/new scalar and vector outcomes, covering NaN/Inf, integer
  boundaries, casts, comparisons, bitwise operations, both division families including zero, broadcasting, and lazy
  scalar branches.
- Ordered UOp image census: **107** lowering outcomes / **100** images, byte-identical list digest
  `a4ca3c5c...05fa`; UOp tests: **89 passed** with `-n12`.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **784.28 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`a209292d...166f`) exactly matches the fully tested
`/tmp/rk-static-eval.Ss3cwQ` worktree. No test, runtime, tolerance, tensor-value execution, CPU/GPU fallback, CMAC,
LUT, reset, reboot, or push was used.

---

## 2026-08-15 — make exact INT32 division compositional

The messiest remaining self-contained block was signed INT32 division. A 49-line root recognizer reverse-engineered
direct `CDIV`/`CMOD` plus Tinygrad's canonical floor-division wrapper, then a separate 165-line emitter duplicated
scratch allocation, constants, native INT16 byte arithmetic, sign handling, restoring division, and INT32 packing
outside `RKContext`. This made division dependent on one final graph spelling and prevented normal nesting or reuse.

The standalone 214-line recognizer/emitter and its early root dispatch are deleted. The same exact byte-restoring
algorithm now occupies 76 executable lines behind typed `RKContext` `CDIV`/`CMOD` lowering. It consumes ordinary
INT32 `RKValue`s, reuses the existing byte-plane and row-reduction primitives, packs the canonical INT32 layout, and
caches quotient/remainder components by semantic operand pair. Nested arithmetic now composes normally, and sibling
quotient/remainder expressions share one restoring core. Proven-bounded `INT_FP16` division/modulo keeps its prior
path; unsupported layouts still fail closed.

- Exact renderer diff: **90 insertions / 234 deletions**, executable lines **5,490 -> 5,353** (**-137**); runtime
  remains **489**, total **30,909**. Cumulative renderer reduction from the 6,616-line cleanup baseline is
  **1,263 lines**.
- A test-only physical integer image executor checked **100 deterministic random lanes plus signed extrema,
  divide-by-zero, mixed signs, and `INT_MIN / -1`**, for direct, nested, sibling, floor-division, and floor-modulo
  graphs. All results matched exact wrapped INT32 semantics; sibling `CDIV`/`CMOD` reused one core.
- At 100 lanes, the direct quotient image changed from **148,365 to about 140,390 bytes** and scratch allocation from
  **851,968 to 13,600 bytes**. At the 16,000-lane hardware limit the new image remains about **140 KB**, encodes and
  round-trips, and uses **2,176,000 bytes** of colored scratch.
- UOp tests: **92 passed** with `-n12`; focused hardware division/modulo coverage: **4 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **795.46 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`e24b5c91...1cf33`) exactly matches the fully tested
`/tmp/rk-int32-div.XExQbU` worktree. No runtime/core code, test tolerance, host tensor arithmetic, CPU/GPU fallback,
CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — compose INT32 bitwise UOps and unify raw-byte plumbing

INT32 `AND`/`OR`/`XOR` still bypassed the typed executor through a 50-line root-only emitter, even though
`RKContext._integer_bitwise` contained the same byte-plane algorithm for INT16. Raw representation plumbing was also
split across three caches and three nearly identical split/pack implementations for FP16, INT16, and INT32, while
FP16 and INT32 comparisons independently implemented the same ordered-byte loop.

The root emitter and dispatch are deleted. The typed bitwise executor now selects its two- or four-byte width from
the proven layout, so direct, constant, and nested INT32 operations use the same compositional UOp path. One
`_raw_parts` cache and one `_pack_raw` helper replace the five split/pack functions, and `_ordered_byte_less` owns the
shared lexical comparison. A direct root NOT keeps a one-stage raw-byte fast path because native wide subtraction
saturates `~INT_MIN` incorrectly; canonical boolean inversion is now one exact `1-x` stage.

- Exact renderer diff: **68 insertions / 149 deletions**, executable lines **5,353 -> 5,276** (**-77**); runtime
  remains **489**, total **30,832**. Cumulative renderer reduction from the 6,616-line cleanup baseline is
  **1,340 lines**.
- A physical integer-image oracle checked seeded full-range values and extrema for all three operators, direct and
  nested expressions, five constant masks, native/no-host execution, and encode/decode. The exact 4,000-element
  byte-lane limit encodes; 4,001 fails closed.
- Eight non-migrated raw consumers remained byte-identical: FP16 LT/NE/raw WHERE, INT16 XOR/mask packing, INT32 LT,
  nested NOT-plus-wide arithmetic, and exact INT32 division. The division digest remains `56483ae6...482f0`.
- A 64-lane AND keeps **92 EW stages** while its image falls from **14,062 to 12,254 bytes** and scratch from
  **48,640 to 14,336 bytes**. Direct INT32 NOT falls from **100 to 1 EW task**; boolean NOT falls from **2 to 1**.
- UOp tests: **92 passed** with `-n12`; focused Rockchip and Rockchip2 bitwise/shift hardware: **13 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **797.26 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`4b7c5a57...5243`) exactly matches the fully tested
`/tmp/rk-int32-bitwise.M4tnLP` worktree. No runtime/core code, hardware assertion, tolerance, host tensor arithmetic,
CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — compose signed and unsigned INT32 shifts

The next self-contained duplication was the 118-physical-line root-only INT32 barrel shifter. It separately parsed
loads and output casts, allocated two private scratch arenas, split values into bytes and bits, applied the five
`1/2/4/8/16` shift stages, reduced the result back into bytes, and packed four output gathers. Because this happened
before `RKContext`, it only recognized a final output shift and could not lower a shift nested inside another shift or
ordinary bitwise arithmetic.

That standalone emitter and its dispatch are deleted. The same exact bit-plane algorithm now consumes typed INT32
`RKValue`s inside `RKContext`, reusing `_raw_parts`, `_pack_raw`, `_stripe_layout`, `_int16_byte_bits`, the shared row
reducer, and the normal scratch allocator. Signed `SHR` retains arithmetic sign fill; unsigned `SHR` uses zero fill;
all amounts retain modulo-32 semantics. UINT32 load/static/constant values are admitted only in the canonical INT32
raw layout, and the existing raw UINT32-to-INT32 cast remains a representation alias.

- Exact renderer diff: **98 insertions / 129 deletions**, executable lines **5,276 -> 5,248** (**-28**); runtime
  remains **489**, total **30,804**. Cumulative renderer reduction from the 6,616-line cleanup baseline is
  **1,368 lines**.
- The test-only physical image executor checked signed and unsigned `SHL`/`SHR` at amounts
  **0, 7, 8, 15, 16, 31, and 32**, with `INT_MIN`, `INT_MAX`, negative values, tensor shift amounts, nested shifts,
  and a nested XOR. Every result matched exact wrapped 32-bit semantics, with native/no-host execution and exact
  encode/decode round trips.
- The compositional direct 8-lane shift uses **95 EW stages**, a **13,571-byte** image, and **57,568 bytes** of
  scratch versus the root-only image's **94 stages**, **13,391 bytes**, and **57,280 bytes**. This small direct cost
  buys ordinary nested composition and removes the parallel planner/emitter implementation.
- Three non-migrated wide-integer consumers remained byte-identical: INT32 AND, nested NOT-plus-ADD, and CDIV.
- UOp tests: **92 passed** with `-n12`; focused serial Rockchip/Rockchip2 bitwise and shift hardware: **13 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **797.40 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`dbf615db...f6d2`) exactly matches the fully tested
`/tmp/rk-int32-shift.MyfT5h` worktree. No runtime/core code, test tolerance, host tensor arithmetic, CPU/GPU fallback,
CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — delete obsolete semantic recovery catalogs

Two post-hoc graph recognizers remained between the semantic UOps and `RKContext`. `_isclose_match` walked a complete
boolean DAG to guess that it came from `Tensor.isclose`, then patched exact equality and NaN behavior back into the
comparison result. A separate bounded-lookup recognizer rebuilt `LOAD`/bounds/`WHERE` as a private candidate-selection
image. Both duplicated semantics already present in current UOps and made correctness depend on one historical graph
spelling.

The isclose recognizer and correction branch are deleted without replacement. The bounded lookup recognizer, its
private INT16 candidate image, and dispatch are also deleted; ordinary typed INT32 comparison, boolean, arithmetic,
WHERE, and load lowering now own the program. The only generic gap exposed was comparison against a constant
`weakint`; `_compare` now normalizes only those constants to typed INT32 before using its existing exact raw-byte
comparator.

- Exact renderer diff: **5 insertions / 98 deletions**, executable lines **5,248 -> 5,164** (**-84**); runtime remains
  **489**, total **30,720**. Cumulative renderer reduction from the 6,616-line cleanup baseline is **1,452 lines**.
- Default, `equal_nan=True`, and `rtol=.01` isclose images are byte-for-byte unchanged: **22,277 / 22,373 / 22,277
  bytes**, **497 / 499 / 497 EW stages**, with digests `e0e85163...20ba`, `5f5b9427...4abb`, and
  `71c20f0a...de1d`.
- A literal bounded lookup now executes as ordinary `NATIVE` UOps with zero host gathers/scatters, exact encode/decode,
  and simulated values `(0, 4, 10, 0)` for indices `(-1, 0, 2, 6)`. Its small synthetic image changes from
  **25 to 122 EW stages**, while scratch falls from **6,720 to 2,032 bytes** and gathers from **26 to 3**.
- The directly affected gather, masked-select, nonzero, and fancy-index NPU cluster passed **15/15** in **97.81 s**;
  the committed baseline took **97.61 s** on the same cluster.
- UOp tests: **93 passed** with `-n12`. Main-suite logical predicate/isclose hardware: **4 passed**. The separate
  Rockchip2 exact-bit values also matched, while its existing stale submit-count assertion fails identically at the
  baseline and candidate (`32 != 2`), so it was not represented as a new pass.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **879.09 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`2e0d54cf...f086`) exactly matches the fully tested
`/tmp/rk-obsolete-catalog.OlHjWc` worktree. No runtime/core code, hardware assertion, tolerance, host tensor arithmetic,
CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — unify Rockchip physical-builder and typed-dispatch plumbing

The messiest remaining code was not one of the long precision algorithms. It was the plumbing around them: six
physical image builders each recreated the same scratch allocator and gather/EW working lists, thirteen local closures
rebuilt the same scratch `RKArg`, and `RKContext.lower` branched the same CAST, comparison, and boolean semantics in
several places. These parallel mechanisms made small ABI changes require edits across unrelated lowering paths.

One `_physical_lists` constructor now owns scratch/list setup, and `_scratch_arg` owns scratch argument construction.
The six builders and thirteen call sites retain their original allocation and append order. `RKContext` now checks
comparison arity once, has one comparison dispatch and one boolean-ALU dispatch, shares equivalent CAST ABI branches,
and calls the accurate-add recipe directly instead of routing through a one-use wrapper.

- Exact renderer diff: **124 insertions / 169 deletions**, executable lines **5,164 -> 5,118** (**-46**); runtime
  remains **489**, total **30,674**. Cumulative renderer reduction from the 6,616-line cleanup baseline is
  **1,498 lines**.
- A baseline-versus-candidate lowering oracle compared **172 ordered outcomes**, including **164 RKImages** and
  **5,456,366 encoded bytes**. Every result was byte-for-byte identical; the aggregate digest is
  `eb1bf55e...52691`.
- UOp tests: **93 passed** with `-n12`; the Rockchip vendor health probe passed **60/60** elementwise cases.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **781.29 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`c2d28e18...31fee6d`) exactly matches the fully tested
`/tmp/rk-context-clean.Jok8In` worktree. No test/runtime/core code, assertion, tolerance, host tensor arithmetic,
CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — remove the parallel static-local address interpreter

The messiest remaining local-address path was a 97-line NumPy interpreter beside the semantic UOp executor. It
rediscovered scalar local definitions, free axes, dependency tables, output ordering, allocation limits, load gates,
and final offsets, then passed those offsets through a private `RKContext` side channel. The same local program still
had to be supported by the ordered semantic unroller, so fixes could diverge between two implementations.

The private interpreter and side channel are deleted. A direct rank-0 constant-True `masked_select(size=None)` is now
the exact generic algebraic identity `flatten()`, avoiding the only large program that needed the parallel evaluator.
Genuine local-address programs remain on the ordered semantic unroller. Its multi-buffer path now carries inherited
range environments, discovers semantic local dependencies transitively, preserves sequential updates, and ignores
`RANGE`/`AFTER` ordering-only loads. `RKContext._load` uses the same semantic traversal when deciding whether an
address is genuinely runtime-dependent.

- Exact renderer diff: **67 insertions / 137 deletions**, executable lines **5,118 -> 5,049** (**-69**); the generic
  masked-select identity adds one executable mixin line, runtime remains **489**, and total falls **30,674 -> 30,606**
  (**-68**). Cumulative renderer reduction from the 6,616-line cleanup baseline is **1,567 lines**.
- A baseline-versus-candidate lowering oracle compared **172 ordered outcomes**, including **164 RKImages**. Every
  outcome and serialized image was byte-for-byte identical; the aggregate digest is
  `e4c290e8305db1e24fb4075674c7dfe9b25c8444763fb2820dcbc325b600ae70`.
- Null Tensor/UOp identity tests cover matrix, empty, and scalar inputs. CPU masked-select value tests passed, and the
  Rockchip scalar-True hardware case now correctly performs **zero submits** while returning the exact flattened data.
- UOp tests: **93 passed** with `-n12`; focused scalar-True and `std_mean` hardware regressions: **2 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **797.22 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`fdfa6a39...0790bc`) exactly matches the fully tested
`/tmp/rk-static-local-clean.09aMA1` worktree. No runtime code, tolerance, host tensor arithmetic, CPU/GPU fallback,
CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — retire duplicate late math recovery and share storage rules

The messiest remaining fallback plumbing ran several composite/masked math recognizers twice: semantic UOps first
passed through `_expand_math_uops`, then the late `_fp16_rewrite` tried to recover sine, atan, inverse-hyperbolic,
masked-EXP2, and reciprocal-sqrt spellings again. The same region also copied nine identical storage-conversion rules
between `_pm_fp32_to_fp16` and `_pm_generic_storage_precision`.

Semantic math expansion is now the sole owner of those five recipes. Their implementations remain in place for that
owner; only the redundant late matcher objects and calls are deleted. The closed masked-EXP2 fallback and its private
nonpositive-EXP2 helper are deleted, while the still-observed EXP2, LOG2, SQRT, and ABS late guards remain. One
`_pm_storage_common` matcher now owns the nine shared storage rules without changing their order.

- Exact renderer diff: **4 insertions / 49 deletions**, executable lines **5,049 -> 5,010** (**-39**); runtime remains
  **489**, total falls **30,606 -> 30,567**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **1,606 lines**.
- The baseline-versus-candidate lowering oracle compared **172 ordered outcomes**, including **164 RKImages**; every
  outcome and serialized image remained byte-for-byte identical with aggregate digest
  `e4c290e8305db1e24fb4075674c7dfe9b25c8444763fb2820dcbc325b600ae70`.
- A separate 13-operation Tensor corpus covering sin, cos, tan, EXP2, LOG2, SQRT, reciprocal-sqrt, atan, asinh, acosh,
  atanh, ABS, and WHERE was byte-for-byte identical between baseline and candidate.
- UOp tests: **93 passed** with `-n12`; directly affected transcendental and attention hardware: **49 passed**.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **809.31 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass. An adversarial fable-judge audit
  found no weakened tests, dead references, scope creep, or alternate host execution path.

The transferred renderer SHA-256 (`28252db6...574d51`) exactly matches the fully tested
`/tmp/rk-fallback-clean.g1nbNz` worktree. No test/runtime/core code, assertion, tolerance, host tensor arithmetic,
CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — delete prefix, count, and occurrence graph recovery

Seven pre-generic dispatch entries still recognized historical masked-select/nonzero/cumulative graph spellings and
emitted private prefix, predicate-total, and INT32 occurrence images. Their 16-function dependency component rebuilt
static row matrices, IEEE masks, byte-carry sums, local ADD loops, and terminal conversions already expressible by the
typed UOp executor.

The seven entries and their now-closed helpers are deleted. FP16 predicates, INT32 prefix arithmetic, WHERE, local
updates, typed loads, and final widening now compose through ordinary UOps. The independent bounded-coordinate path
remains because it has distinct dynamic-rank/fill semantics and current hardware ownership.

- Exact production diff: **417 renderer lines deleted**, executable renderer size **5,010 -> 4,628** (**-382**);
  runtime remains **489**, total falls **30,567 -> 30,185**, and cumulative renderer reduction from the 6,616-line
  cleanup baseline is **1,988 lines**.
- Of **172 ordered unit outcomes / 164 RKImages**, **162 images remain byte-for-byte identical**. The two intentionally
  migrated synthetic prefixes remain `NATIVE`, use zero host gathers/scatters, terminate in INT32 widening, and
  encode/decode exactly. Their small image shapes change from **16 -> 333 EW / 1,085 -> 16,882 bytes** and
  **10 -> 70 EW / 855 -> 4,321 bytes**; the obsolete exact-stage assertions were replaced by those semantic/ABI
  contracts rather than hidden.
- Baseline/candidate hardware profiling identified `test_simple_cummin` as the slowest affected case
  (**35.46 / 35.60 s**, effectively unchanged). The 22-test cumulative/masked-select/nonzero gate passed in
  **152.79 / 154.27 s**. The 7-test one-hot/arg-extrema/sort/top-k gate passed in **42.81 / 35.98 s**; top-k improved
  **17.69 -> 4.88 s**, while sort changed **16.00 -> 22.00 s**. Combined affected time improved
  **195.60 -> 190.25 s**.
- A historical two-line native-INT16-to-FP16 state-restoration experiment was retested against the slowest case but
  did not improve today's schedule (**37.20 s**); it was rejected and fully reverted in `/tmp`.
- UOp tests: **93 passed** with `-n12`. Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero
  failures across all **445** cases in **814.71 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass. Fable-judge reports
  **VERIFIED WITH CAVEAT** solely for the two deliberate image-shape contract replacements above; no skipped numeric
  check, tolerance change, host execution path, dead reference, or scope creep was found.

The transferred renderer SHA-256 (`1a28d441...3deab`) and focused-test SHA-256 (`200850b4...79e03`) exactly match the
fully tested `/tmp/rk-prefix-clean.HwTr3p` worktree. No runtime/core code, hardware assertion, tolerance, host tensor
arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — move direct casts and scalar boolean reductions into typed UOp lowering

The messiest remaining renderer region is the front of `_lower_uop_program`: a legacy catalog parses particular
graph spellings and builds complete `RKImage`s before the composable `RKContext` UOp executor gets a chance to lower
the same semantics. The catalog duplicates graph traversal, static index evaluation, layout selection, scratch
allocation, and terminal conversion. It is also mixed with genuinely necessary precision and indexing fast paths,
so deleting the whole region is unsafe.

Direct half-to-INT32 and integer/bool-to-FP32 casts now lower through `RKContext` layouts and ordinary CAST/WHERE
composition. The private scalar loop boolean recognizer and integer-predicate image builder are deleted; the generic
executor owns those scalar reductions, while the distinct grouped boolean reducer remains. Three broader isolated
experiments were rejected rather than transferred: deleting scalar numeric reduction missed tolerance by **0.038**,
deleting the dot reducer changed **292 / 20,250** values with maximum error **0.02148**, and deleting the predicate-total
load path rejected a real collapsed fancy-index kernel. Those paths therefore remain explicit contracts, not presumed
dead code.

- Exact renderer diff: **7 insertions / 94 deletions**, executable lines **4,628 -> 4,549** (**-79**); runtime remains
  **489**, total falls **30,185 -> 30,106**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,067 lines**.
- The baseline/candidate keyed lowering census changed only the two expected direct-cast images. Remapped INT32 and
  bool-to-FP32 cases now explicitly exercise the typed executor and retain exact encode/decode round trips.
- UOp tests: **94 passed** with `-n12`; directly affected cast, rounding, modulo, division, predicate, and boolean
  reduction hardware tests: **22 passed**. The two exact reduction/dot regressions that rejected broader cuts also pass.
- Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445** cases in
  **822.36 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`343a5adc...33d9`) and focused-test SHA-256 (`e78336c9...ef7`) exactly match the
fully tested `/tmp/rk-cast-bool.vkKKFs` worktree. No runtime/core code, hardware assertion, tolerance, host tensor
arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — consolidate typed WHERE and raw-layout plumbing

The messiest remaining composable region was `RKContext._where`: boolean selection, INT16 selection, nonfinite FP16
selection, and raw FP16/INT32 byte selection each repeated mask arithmetic, arm lowering, scratch allocation, and
split/pack scheduling. That duplication sat beside generic `_raw_parts`/`_pack_raw` helpers which already owned the
same physical layout semantics.

`_raw_where` now owns lazy nonfinite handling and exact typed raw selection, while one `_masked_where` helper owns
the shared 0/1 mask arithmetic. `_raw_parts` and `_pack_raw` gained explicit cache/copy/destination controls so the
WHERE path reuses their physical ABI without changing scheduling. The same cleanup also merges the FP16/integer
nonzero-load recognizers, directly reuses the generic predicate-count proof, removes a dead static-vector helper and
two dead constants, collapses identical decode branches, and removes redundant scratch-reusability state. Two larger
deletions were rejected in `/tmp`: grouped boolean reduction is required by `all_large`, and multi-local reduction is
required by `std_mean`.

- Exact renderer diff: **83 insertions / 143 deletions**, executable lines **4,549 -> 4,493** (**-56**); runtime remains
  **489**, total falls **30,106 -> 30,050**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,123 lines**.
- The baseline/candidate UOp lowering census produced **165 RKImages / 161 unique image records**. Every serialized
  image, resource count, and occurrence count was byte-for-byte identical; the aggregate JSON-stream SHA-256 is
  `8a8b7a095be21f3cc3539636e7bee18dee5fe920c685addbfe77f9d69c3a45ff`.
- Focused hardware covering cumulative extrema, WHERE/nonfinite selection, grouped boolean reduction, and
  `std_mean`: **12 passed**. `test_simple_cummin` was **36.87 s -> 36.29 s**, treated as effectively unchanged rather
  than a claimed performance win.
- UOp tests: **94 passed** with `-n12`. Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero
  failures across all **445** cases in **832.86 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`d3afc4d4...7522b4`) exactly matches the fully tested
`/tmp/rk-hot-clean.vT6XjT` worktree. No tests, runtime/core code, hardware assertion, tolerance, host tensor
arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — share mapped-reduction precision and image plumbing

The largest messy renderer group is the mapped-reduction catalog. It is not dead: an instrumented full hardware
census observed 24 successful unrolled-ADD lowerings, 10 vectorized MUL+ADD lowerings, 19 mapped-loop ADD lowerings,
four scalar-extrema lowerings, and two multi-local lowerings. A wholesale deletion would therefore remove active
precision and performance contracts. The safe seam was their duplicated plumbing rather than their recognizers.

The exact 17-stage FP16 TwoProduct residual recipe now has one physical-stage constructor, and the two ADD-family
recognizers share one outer CAST/constant-scale parser. `_finish_mapped_add_reduction`, `_append_inplace_image`, and
linear scratch reuse now delegate complete argument relocation to `_map_image_args` instead of maintaining parallel
gather/EW/host/fill remappers. A duplicate RKContext destination allocator whose operands were unused is also removed.
All recognizers, Kahan/TwoProduct arithmetic, task ordering, barriers, scratch layouts, and native execution remain.

- Exact renderer diff: **53 insertions / 96 deletions**, executable lines **4,493 -> 4,448** (**-45**); runtime remains
  **489**, total falls **30,050 -> 30,005**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,168 lines**.
- The baseline/candidate UOp lowering census produced **165 RKImages / 161 unique image records**. Every serialized
  image and occurrence count remained byte-for-byte identical; the aggregate JSON-stream SHA-256 is
  `8a8b7a095be21f3cc3539636e7bee18dee5fe920c685addbfe77f9d69c3a45ff`.
- Focused hardware covering GEMM, biased convolution, attention, loss reductions, `std_mean`, and normalization:
  **7 passed** in **105.83 s**.
- UOp tests: **94 passed** with `-n12`. Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero
  failures across all **445** cases in **824.68 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`ad664554...df787`) exactly matches the fully tested
`/tmp/rk-reduction-clean.buWVb3` worktree. No tests, runtime/core code, hardware assertion, tolerance, host tensor
arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — centralize typed scratch materialization

`RKContext` repeated the same cache/materialization transaction for INT32 constants, static vectors, boolean loads,
ordinary typed gathers, dynamic candidate matrices, and raw FP32 input groups: construct a key, allocate scratch,
append a gather, store the value, and return it. The copies had already drifted into separate local variable names and
return conventions even though their insertion order and physical behavior were identical.

Two small typed helpers now own static-vector and gather-plan materialization while preserving every existing cache
key, layout, allocation size, gather, and insertion point. The same milestone reuses the established semantic-local
walker in the multi-local and scalar-extrema paths, and uses `_map_image_args` for the extrema child image instead of
another private gather/EW remapper. Ordering-only `AFTER` sources remain excluded from value dependency discovery.

- Exact renderer diff: **31 insertions / 63 deletions**, executable lines **4,448 -> 4,414** (**-34**); runtime remains
  **489**, total falls **30,005 -> 29,971**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,202 lines**.
- A serial baseline/candidate lowering census recorded **173 outcomes / 165 RKImages / 161 unique image hashes**.
  The complete ordered hash lists are identical, with SHA-256
  `d26e786c680022763fc2bbb5e002d4eb5f53c5e89531b2ba575eaa25df82115e`.
- Focused hardware covering dynamic gather, collapsed fancy indexing, FP32 casts, argmax/argmin, `std_mean`, and
  permuted WHERE: **7 passed** in **23.92 s**.
- UOp tests: **94 passed** with `-n12`. Full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero
  failures across all **445** cases in **819.94 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass.

The transferred renderer SHA-256 (`a83d2c8f...b7b7b6`) exactly matches the fully tested
`/tmp/rk-materialize.5TSRlj` worktree. No tests, runtime/core code, hardware assertion, tolerance, host tensor
arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — centralize post-reduction UOp replay

The mapped-reduction catalog remains the largest messy renderer group, but its active recognizers cannot be deleted:
the full hardware census exercises mapped-loop ADD, precision MUL+ADD, multi-local, unrolled-ADD, and scalar-extrema
paths. The duplicated post-reduction ownership was removable. Mapped-loop ADD replayed a general expression through
the typed UOp executor, vectorized MUL+ADD separately rebuilt bias and ReLU images by hand, and multi-local ADD carried
a third copy of output-range substitution and post-image construction.

`_lower_post_image` now owns output-range normalization and typed post-expression lowering.
`_append_reduction_post` materializes one completed reduction, aliases it back to the real output, canonically
orients commutative in-place stages, and appends the post image. Mapped-loop ADD and vectorized MUL+ADD both use this
path, while multi-local ADD reuses the same post-image builder. The TwoProduct stages, Kahan/compensated reduction,
scratch arena, reduction barriers, and fail-closed recognizers are unchanged.

- Exact renderer diff: **32 insertions / 53 deletions**, executable lines **4,414 -> 4,390** (**-24**); runtime remains
  **489**, total falls **29,971 -> 29,947**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,226 lines**.
- The serial baseline/candidate UOp census retained every one of the **173** baseline outcomes and every serialized
  image hash with identical occurrence counts. The candidate records one additional internal post-image lowering;
  no baseline image was removed or changed.
- Focused NPU coverage for large GEMM, biased convolution, GQA attention, cross-entropy, binary cross-entropy,
  `std_mean`, and normalization: **7 passed** in **60.15 s**.
- The required hardware census in `test/backend/test_rockchip.py` collected all **445** cases and completed with zero
  failures during a broader serial run using `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`. Every failure in that
  broader run belonged to the separate 199-case experimental `test_rockchip2.py` suite; its first failure was
  reproduced unchanged on the clean committed baseline. NPU health subsequently passed the RK3588 elementwise probe.
- UOp tests: **94 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`defad456...edcb`) exactly matches the fully tested
`/tmp/rk-post-reduction.OZhty2` worktree. No tests, runtime/core code, hardware assertion, tolerance, host tensor
arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — prune unreachable comparison recursion

The comparison island mixed two distinct responsibilities: one compact IEEE-aware numeric comparison recipe and a
recursive boolean-expression interpreter. `RKContext._compare` only invokes the recipe for a single `CMPLT` or
`CMPNE`; composed boolean expressions already lower through `RKContext._bool_binary`. The analogous recursive
AND/OR/XOR and boolean-inversion arms in `_native_int16_comparison` were likewise unreachable from its sole caller.

An intentionally broader experiment that deleted the IEEE recipe was rejected in `/tmp`: although its semantics and
unit tests passed, representative half-backed comparisons grew from 34 to 97 EW stages. The accepted rewrite keeps
the compact numeric atom construction and its exact allocation/order, removes only the unreachable recursive
admission, deletes `_ieee_bool`, and routes boolean composition directly to the typed boolean lowerer.

- Exact renderer diff: **17 insertions / 60 deletions**, executable lines **4,390 -> 4,348** (**-42**); runtime remains
  **489**, total falls **29,947 -> 29,905**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,268 lines**.
- A complete ordered unit lowering census compared **174 image records** and found every encoded byte string and
  occurrence count identical. Direct half-backed `CMPLT`, `CMPNE`, and `CMPEQ` retained their exact image hashes and
  resource counts: 34, 34, and 35 EW stages respectively.
- Focused hardware comparison, logical, classification, INT16, and argument-extrema coverage: **19 passed** in
  **11.62 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  cases in **798.55 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **94 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`9d3a23f5...0e16`) exactly matches the fully tested
`/tmp/rk-ieee-compact.qXt0BV` worktree. No tests, runtime/core code, hardware assertion, tolerance, host tensor
arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — reuse typed FP16 equality in scalar extrema

The scalar-extrema path had the messiest remaining comparison tail: after computing the maximum value, it manually
split both FP16 vectors into bytes, rebuilt signed-zero canonicalization and NaN classification, compared the bytes,
and multiplied the resulting mask by static coordinates. This duplicated the same typed raw-FP16 comparison already
owned by `RKContext`, including two global INT16/FP16 helper implementations used only by that context and tail.

The tail now describes its real semantics as ordinary UOps (`CMPEQ`, boolean-to-INT16 cast, coordinate multiply),
lowers that small image through `RKContext`, and composes it with the existing extrema prefix. Native INT16 equality is
one context method, and FP16 component classification uses those typed context primitives directly. The reduction,
first-index tie rule, coordinate transform, barriers, and terminal INT32 widening are unchanged.

An initial `/tmp` candidate exposed a real composition bug on hardware: argmax expected lane 149 but returned 200
because the final image accidentally restored the child constant table after appending the typed comparison image.
The accepted version retains `combined.constants`; the focused unit now asserts the six comparison constants and an
exact encode/decode round trip, so this failure no longer depends on hardware to detect.

- Exact production renderer diff: **42 insertions / 73 deletions**, executable lines **4,348 -> 4,321** (**-27**);
  runtime remains **489**, total falls **29,905 -> 29,878**, and cumulative renderer reduction from the 6,616-line
  cleanup baseline is **2,295 lines**.
- The scalar-extrema unit image retains **73 EW stages / 7 synchronized gathers** while shrinking from **11 to 5
  initial gathers** and **84 to 27 scratch slots**; its final encoded size is **3,698 bytes**. All non-extrema images
  in the ordered UOp census remained byte-for-byte identical.
- Focused hardware covering cumulative extrema, argmax/argmin, softmax-argmax, sort indices, top-k, and elementwise
  extrema: **15 passed** in **94.32 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  cases in **790.36 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **94 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`eb173fa5...16e0b`) and focused-test SHA-256 (`c42e5d7d...436b`) exactly match the
fully tested `/tmp/rk-extrema-refactor.1JUMHG` worktree. The sole test edit strengthens the constant-table contract;
no assertion was removed or weakened. No runtime/core code, host tensor arithmetic, CPU/GPU fallback, CMAC, reset,
reboot, or push was used.

---

## 2026-08-15 — unify ordered static-local execution

`_unroll_static_local` was two interpreters behind one dispatch: a single-buffer implementation rediscovered
initializers and updates from STOREs, while a separate recursive implementation expanded multiple dependent local
buffers. They implemented the same ordered local-accumulator semantics with different loop discovery, dependency,
dtype, and budget logic.

One recursive executor now handles both cases. `_static_local_defs` remains the single parser, falling back to the
term's REDUCE ranges only when an AFTER node does not carry explicit local loops. Updates retain literal STORE order
and use the rewritten term dtype; that detail preserves the physical FP16 accumulator created when storage rewriting
turns an FP32 local buffer into a half update. Cycles, nonconstant/oversized ranges, expansion budgets, and unsupported
local programs still reject to the unchanged fallback path.

The first `/tmp` prototype found three useful negative twins before acceptance. It initially missed simple local loops
whose range existed only in the update term; it then rebuilt rewritten half updates as FP32, expanding mapped images
from hundreds to thousands of EW stages; finally, filtering explicit AFTER loops to REDUCE axes rejected
softmax-argmax on hardware. The accepted parser keeps all explicit AFTER ranges, uses REDUCE-only term fallback, and
reconstructs updates with `definition.term.dtype`.

- Exact renderer diff: **36 insertions / 79 deletions**, executable lines **4,321 -> 4,278** (**-43**); runtime remains
  **489**, total falls **29,878 -> 29,835**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,338 lines**.
- A fresh committed-baseline/candidate unit census compared **167 RKImages / 163 unique images / 5,478,964 encoded
  bytes**. The complete hash/resource multiset is byte-for-byte identical with SHA-256
  `0cb9079af2ac4cbce246507a925f662e6f81d6dc004fa3c50d0f641bbbf5d55a`.
- Final focused hardware argument-extrema coverage: **3 passed** in **7.22 s**. The sole unit edit strengthens the
  dependent mapped-reduction contract with explicit **<200 / <300 EW-stage** ceilings.
- One first-pass full run saw a transient cross-entropy `NaN`. NPU health passed all elementwise probes; committed and
  candidate focused reruns both passed, and their ordered **15-image** SHA-256 lists were exactly identical. The clean
  required rerun then completed with **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  cases in **816.88 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **94 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`d5bc0736...b883cb`) and focused-test SHA-256 (`aeb35e10...bddd`) exactly match the
fully tested `/tmp/rk-next-messy.RRQfVm/tree` worktree. No assertion was removed or weakened. No runtime/core code,
host tensor arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — unify bounded dynamic typed-load parsing

Dynamic typed selection had two adjacent parsers for the same physical operation. The direct parser owned one INT32
axis plus an optional external bool gate; the multi-axis parser independently repeated output LOAD validation, data
and index PARAM checks, bounds-gate recognition, static candidate planning, source-bound validation, and raw-selector
dispatch. Only candidate construction and gate ownership differed.

`_lower_dynamic_typed_load` now owns that shared proof once. It first attempts the literal old direct contract, then
falls through to the old positive/negative-normalized multi-axis contract. Direct candidate order, external bool
gating, multi-axis Cartesian order, negative alternatives, source bounds, and the existing
`_dynamic_raw_gather_image` byte emitter are unchanged.

An initially smaller `/tmp` draft was rejected during adversarial review because it returned immediately when a
one-axis index WHERE was not canonical negative normalization. The old dispatch instead let that graph fall through
to the multi-axis parser. The corrected version preserves that ordering and also preserves the historical multi-path
2-byte raw default. A synthetic fallback graph is byte-identical before/after: SHA-256 `9719bc5a...a245`, **4,023
bytes / 29 gathers / 27 EW stages / 2 post-gathers**.

- Exact renderer diff: **54 insertions / 76 deletions**, executable lines **4,278 -> 4,259** (**-19**); runtime remains
  **489**, total falls **29,835 -> 29,816**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,357 lines**.
- The complete committed-baseline/candidate UOp census retained all **167 RKImages / 163 unique images / 5,478,964
  encoded bytes** byte-for-byte. Its hash/resource multiset SHA-256 remains
  `0cb9079af2ac4cbce246507a925f662e6f81d6dc004fa3c50d0f641bbbf5d55a`.
- Focused hardware gather and fancy-index coverage: **11 passed** in **54.37 s**, including direct, collapsed,
  negative-normalized, and multi-axis forms.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  cases in **827.58 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **94 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`32d10785...2ae8b0`) exactly matches the fully tested
`/tmp/rk-dynamic-parser.6XXzMN/tree` worktree. No test, runtime/core code, host tensor arithmetic, CPU/GPU fallback,
CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — centralize native comparison stages

The largest messy Rockchip area remains the specialized mapped-reduction catalog, but its recognizers are all live
in the full hardware census and their numeric/performance contracts differ. The safest messy slice was instead the
raw FP16/INT32 comparison block inside `RKContext`: four methods manually allocated scratch and appended native
INT16 EW stages even though `_i16`, `_i16_const`, `_i16_equal`, and `_i16_clamp_one` already expressed the same
typed operation and allocation order.

`_int32_compare`, `_fp16_equality`, `_fp16_ordered_values`, and `_fp16_less` now use those shared helpers. Constant,
scratch, and EW-stage ordering remains exact, including IEEE NaN gating and signed FP16 lexical ordering. Two broader
`/tmp` alternatives were rejected first: a grouped-bool merge saved only three executable lines, while an output-major
reducer saved 26 but changed a noncontiguous reduction from roughly five vector stages to about 90 scalar-output
stages. Neither was transferred.

- Exact renderer diff: **22 insertions / 45 deletions**, executable lines **4,259 -> 4,236** (**-23**); runtime remains
  **489**, total falls **29,816 -> 29,793**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,380 lines**.
- A fresh committed-baseline/candidate unit census compared all **167 RKImages / 163 unique images**. The complete
  encoded-image hash/resource multiset is byte-for-byte identical with SHA-256
  `0cb9079af2ac4cbce246507a925f662e6f81d6dc004fa3c50d0f641bbbf5d55a`.
- Focused hardware classification, comparison, logical-predicate, and argument-extrema coverage: **15 passed** in
  **11.49 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **833.05 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **94 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`fbc9f2c...d21d4`) exactly matches the fully tested
`/tmp/rk-compare-i16.0KGLFB` worktree. No test, runtime/core code, host tensor arithmetic, CPU/GPU fallback, CMAC,
reset, reboot, or push was used.

---

## 2026-08-15 — retire the legacy IEEE comparison recipe

`_ieee_comparison_mask` duplicated the typed raw comparator by rebuilding IEEE classes as a 31-line graph of FP16
mask arithmetic. That fallback was especially awkward for half-backed FP32 expressions: it cast them back to half,
constructed NaN/infinity masks, then re-entered `RKContext`, whose raw-byte comparator already implements the exact
same IEEE boundary.

`_half_backed_value` now proves that every dynamic LOAD really comes from FP16 storage, normalizes the expression once,
and sends `<`, `!=`, and `==` directly to `_fp16_less`/`_fp16_equality`. Native FP32 storage remains fail-closed. The
same normalization extends `_bool_binary`'s inverted-less guard, preserving unordered NaN semantics for `>=`, `<=`,
and `>`.

The first `/tmp` draft exposed that guard as a real negative twin: direct `<`, `!=`, and `==` passed, but `>=` returned
true for NaN because the guard only recognized syntactically-half operands. Nothing was transferred until the shared
normalization fixed all six comparison forms. The new unit regression requires a half-backed FP32 `>=` image to be
byte-identical to the direct FP16 image.

- Exact renderer diff: **16 insertions / 41 deletions**, executable lines **4,236 -> 4,211** (**-25**); runtime remains
  **489**, total falls **29,793 -> 29,768**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,405 lines**.
- The pre-existing **94-test / 167-RKImage / 163-unique-image** census remains byte-for-byte identical with SHA-256
  `0cb9079af2ac4cbce246507a925f662e6f81d6dc004fa3c50d0f641bbbf5d55a`.
- A focused NPU edge matrix passed `<`, `!=`, `==`, `>=`, `<=`, and `>` over signed zero, finite limits, infinities,
  and NaNs. Existing classification/comparison/logical-predicate/argument-extrema hardware coverage: **15 passed** in
  **11.24 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **788.72 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **95 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`c8a83f1f...77c86`) and focused-test SHA-256 (`287bbed7...1d248`) exactly match the
fully tested `/tmp/rk-half-compare.1JgDA7` worktree. No runtime/core code, host tensor arithmetic, CPU/GPU fallback,
CMAC, reset, reboot, or push was used.

---

## 2026-08-15 — unify repeated-tree matching and mapped aliases

`_lower_vectorized_unrolled_add_reduction` was the largest renderer function at 137 executable lines. Its admission
logic traversed every repeated term twice: first a recursive signature builder compared the complete trees, then a
second recursive walker paired the template's dynamic leaves with each term. The two traversals encoded overlapping
operation, dtype, argument, arity, and leaf-shape constraints.

One structural zipper now checks those invariants while collecting corresponding leaves. The existing per-leaf PARAM,
address, bounds, affine/periodic, and static-fallback proofs remain unchanged. Three nearby mapped-reduction paths also
replace one-off ARG-to-scratch callbacks with the existing `_alias_image_args` primitive: non-affine product residuals,
static fallback sources, and the scalar-extrema child image.

- Exact renderer diff: **10 insertions / 27 deletions**, executable lines **4,211 -> 4,194** (**-17**); runtime remains
  **489**, total falls **29,768 -> 29,751**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,422 lines**.
- A seeded differential generated **600** valid and intentionally malformed repeated trees. Committed and candidate
  admission matched in all 600 cases; all **360 accepted images** were byte-for-byte identical.
- The complete **95-test** UOp image census is byte-for-byte identical with SHA-256
  `a6f3f7b999818de32a4aa57cbe5778d16318e7ebb77d989ce289e582a0069eb1`.
- Focused hardware covering biased convolution, dot/matvec, cumulative extrema, argument extrema, logsumexp, and
  softmax: **10 passed** in **20.68 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **797.71 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **95 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`d75d66e8...e8a7f`) exactly matches the fully tested
`/tmp/rk-unrolled-zip.EEoWRh` worktree. No test, runtime/core code, host tensor arithmetic, CPU/GPU fallback, CMAC,
reset, reboot, or push was used.

---

## 2026-08-15 — centralize mapped-reduction infrastructure

The mapped-reduction catalog is the renderer's messiest live area. Its recognizers cover materially different UOp
forms—mapped local ADD, repeated unrolled ADD/product trees, vectorized MUL+ADD, multiple local accumulators, and
dependent scalar MAX—so deleting or merging their semantic admission rules would lose proven accuracy or performance.
The duplication was instead in their plumbing: separate semantic LOAD walkers, three copies of static-range flattening,
inline repeated-index evaluation, hand-appended INT16 stages, condition inversion parsing, and three handwritten Horner
polynomials.

Those mechanisms now use shared typed/structural helpers while every recognizer and fail-closed proof remains. A first
`/tmp` experiment that generalized the ADD finisher to MAX was rejected: it saved only five executable lines and changed
the scalar-extrema constant table. It was never transferred. The accepted candidate preserves every encoded program.

- Exact renderer diff: **86 insertions / 124 deletions**, executable lines **4,194 -> 4,154** (**-40**); runtime remains
  **489**, total falls **29,751 -> 29,711**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,462 lines**.
- Independent committed-baseline/current runs produced the same sorted **163 encoded-image/resource records** across
  all 95 UOp tests. The canonical JSON SHA-256 is
  `3c48cb16a8578ea523e9f1b7a773bc9bce878bae7e865970a818fb026d139039`.
- Helper differentials retained the former global/local LOAD traversal, exact range-linearization UOp keys, exact Horner
  UOp keys, typed/untyped condition inversion, and **1,000** randomized repeated-index tables plus fail-closed rejection
  of a lane-dependent delta.
- Focused hardware covering bitwise/INT16, argument extrema, transcendental polynomials, mapped reductions, matmul,
  GEMM, and convolution: **32 passed** in **100.51 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **784.62 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.
- UOp tests: **95 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass. Hardware tests ran serially because concurrent xdist workers on the single NPU are rejected
  by DRM with `EINVAL`; the post-attempt elementwise health check passed every size and operation.

The transferred renderer SHA-256 (`329aece7...e1c1812`) exactly matches the fully tested
`/tmp/rk-messy-clean.pqfYa1/tree` worktree. No test, runtime/core code, host tensor arithmetic, CPU/GPU fallback, CMAC,
reset, reboot, or push was used.

---

## 2026-08-15 — unify static evaluation and physical reduction plumbing

The semantic recognizers remain the renderer's largest live area, but another coherent layer of duplication sat below
them. Scalar and vector static-UOp evaluators separately implemented constants, ranges, casts, `WHERE`, and arithmetic;
row reductions separately rebuilt the balanced arena reducer; stateful and stateless EW emitters separately assembled
the same three relocation commands; and four precision paths separately walked a UOp graph to mark compensated ADDs.

One evaluator now handles scalar and vector compiler-known expressions while preserving scalar `WHERE` laziness and
vector eager selection. `_reduce_rows` delegates to the common balanced arena reducer with the original first-stage
barrier semantics, both EW emitters share one command finalizer, and precision recipes share one iterative ADD tagger.
Static and gather materialization also use one typed cache without changing allocation or gather order. Two unused
catalog symbols were removed. All semantic recognizers, resource limits, and fail-closed checks remain in place.

- Exact renderer diff: **75 insertions / 126 deletions**, executable lines **4,154 -> 4,099** (**-55**); runtime remains
  **489**, total falls **29,711 -> 29,656**, and cumulative renderer reduction from the 6,616-line cleanup baseline is
  **2,517 lines**.
- Independent committed-baseline/candidate runs produced the same sorted **163 encoded-image/resource records** across
  all 95 UOp tests. The canonical digest is
  `b6910d7dd0222365c9f8fb8445ed51a7d0b319ed305a2a0631db892511e08b7a` over **5,455,580** unique encoded bytes.
- A separate **166-record** mechanism differential covered scalar/vector static evaluation, compiler LOAD callbacks,
  range substitution, precise-product recipes, FP16/INT16/INT32 balanced reductions, and stateful/stateless EW command
  variants. Baseline and candidate both produced digest
  `4caa3066eb60c755d27540bbad35af55bb4448bf84d7a44c430af8f7eb941e9b`.
- Focused NPU coverage for bitwise, INT16 EW, dot, cumulative extrema, argument extrema, and fancy indexing:
  **42 passed** in **144.41 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **842.52 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- UOp tests: **95 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`4dbc5c8a...7f1567`) exactly matches the fully tested
`/tmp/tinygrad-rk-cleanup.36QsxN` worktree. No test, runtime/core code, host tensor arithmetic, CPU/GPU fallback, CMAC,
reset, reboot, or push was used.

---

## 2026-08-15 — reach the 4,000-line renderer milestone

The specialized mapped and dynamic recognizers remain the renderer's messiest live area, but the complete hardware
census proves they are not dead code: they preserve distinct precision, layout, and task-shape contracts. This
milestone therefore removed duplication around those algorithms instead of deleting their semantics. The serialized
EW record is described by one struct instead of three adjacent fragments; native INT16 stages share one flag bundle;
physical register constants are grouped by ABI role; finite MAX neutral rewriting uses one iterative graph walk; and
accurate ADD plus FP32 sine use the common iterative binary-tree traversal. Nonfinite raw `WHERE`, CAST routing,
stateful-stage finalization, and small typed forwarding helpers were also consolidated. The unused `RKMultiGather`
type and one-use matcher callbacks were removed.

All changes were developed and rejected or accepted in `/tmp/tinygrad-rk-4k.EFJ8M7` before transfer. A first image
census invocation accidentally imported the clean main checkout because the script itself lived in `/tmp`; the
corrected command pinned `PYTHONPATH` to the candidate. It then exposed and fixed a tested private-helper naming
regression before any NPU run. Only the corrected candidate was transferred.

- Exact renderer diff: **138 insertions / 241 deletions**, executable lines **4,099 -> 4,000** (**-99**); runtime
  remains **489**, total falls **29,656 -> 29,557**, and cumulative renderer reduction from the 6,616-line cleanup
  baseline is **2,616 lines**.
- Independent baseline/candidate runs produced the same sorted **169 RKImage emissions / 163 unique images / 5,455,580
  unique encoded bytes** across all 95 UOp tests. The complete encoded-image hash/resource multiset is byte-for-byte
  identical with SHA-256 `a6f3f7b999818de32a4aa57cbe5778d16318e7ebb77d989ce289e582a0069eb1`.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **835.65 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- UOp tests: **95 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass.

The transferred renderer SHA-256 (`12f395e1...c31f4`) exactly matches the fully tested isolated worktree. No test,
runtime/core code, host tensor arithmetic, CPU/GPU fallback, CMAC, reset, reboot, or push was used.

---

## 2026-08-16 — reach the 3,900-line renderer milestone

The remaining specialized recognizers still own distinct precision and layout contracts, but their physical image
assembly repeated scratch allocation, native INT16 EW emission, gather collection, and final `RKImage` construction.
One `_RKBuilder` now owns that mechanical layer for dynamic raw selection, exact byte equality/nonzero masks, bounded
coordinates, predicate-gated loads, and striped boolean reduction. The graph proofs and emitted stage order remain
unchanged. Static gather planning also reuses the existing vector environment, and the dependent scalar-extrema path
uses the shared image-alias and balanced-row helpers throughout.

The scalar loop reducer exposed a second closed seam. Its one-use `_reduction_image` and
`_spaced_reduction_gathers` wrappers retained sqrt, reciprocal, cube-root, whole-buffer preparation, custom fill, and
temporary-scratch modes which their sole caller could never request. The dispatcher already excluded every optional
post-operation before that call. Those unreachable branches and their private sqrt/cube-root emitters were deleted;
the live gather, negate, reduction, scale, scratch, and output sequence was folded directly into the caller. Unsupported
post forms now fail at the parser boundary and continue through the same later generic/mapped paths.

- Exact renderer diff: **201 insertions / 313 deletions**, executable lines **4,000 -> 3,900** (**-100**); runtime
  remains **489**, total falls **29,557 -> 29,457**, and cumulative renderer reduction from the 6,616-line cleanup
  baseline is **2,716 lines**.
- Independent baseline/candidate runs produced the same sorted **169 RKImage emissions / 163 unique images / 5,455,580
  unique encoded bytes** across all 95 UOp tests. The complete `IMAGE_CENSUS` payload is byte-for-byte identical with
  SHA-256 `37085a58c28ccff45c830d995231c70471073d441015e1c87ce0c810eb6e8155`.
- Focused NPU coverage for dynamic gather/masked-select/nonzero/fancy indexing, cumulative and argument extrema,
  grouped boolean reduction, dot, and scalar reductions: **76 passed** in **291.87 s**.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **778.59 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- UOp tests: **95 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass. Fable-judge verdict: **VERIFIED**; no weakened checks, scope creep, or debris was found.

The transferred renderer SHA-256 (`71fba87d...501f821d`) exactly matches the fully tested
`/tmp/tinygrad-rk-3p9-audit` worktree. No test, runtime/core code, host tensor arithmetic, CPU/GPU fallback, CMAC,
reset, reboot, or push was used.

---

## 2026-08-16 — reach the 3,800-line renderer milestone

The messiest remaining mechanical layer was the physical image pipeline beneath the live semantic recognizers.
Grouped boolean paths hand-built scratch, gather, and EW lists even though `_RKBuilder` already owned the same typed
operations; stateless and stateful DPU stages duplicated nearly the same register program; and scratch coloring walked
six physical phases separately. Those paths now share the builder, one stage template, and one ordered lifetime
schedule. Their UOp admission rules, stage ordering, allocation order, and serialized images remain unchanged.

Two obsolete wire features were also removed end to end. No renderer produced `RKFill` or host-address negative
normalization, and every current image encoded their reserved fields as zero. The encoder still writes zero and the
decoder now rejects nonzero legacy values, so the current ABI bytes are unchanged and unsupported inputs fail closed.
One-use scratch helpers, an unused reduction-barrier field, and the unreachable reciprocal branch of `_dpu_sqrt` were
removed. The final threshold gap was closed by reflowing single Python statements that already fit the repository's
line limit; no semicolon packing, recognizer deletion, comment deletion, or docstring deletion was used.

- Exact production diff: renderer **203 insertions / 317 deletions**, executable lines **3,900 -> 3,799** (**-101**);
  runtime **489 -> 483** (**-6**), total **29,457 -> 29,350**, and cumulative renderer reduction from the 6,616-line
  cleanup baseline is **2,817 lines**.
- Independent baseline/candidate runs produced byte-for-byte identical complete encoded-image/resource payloads across
  all **95** UOp tests, SHA-256 `a6f3f7b999818de32a4aa57cbe5778d16318e7ebb77d989ce289e582a0069eb1`.
  A separate **304-variant** DPU stage-template oracle was also exact, SHA-256
  `853f7cf9974e0dafa2f11e480f079782c45d6b9ef283b38d3690305ae11bc2b4`.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **1,722.15 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- UOp tests: **95 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass. Fable-judge verdict: **VERIFIED**; no weakened tests, CPU/GPU/CMAC path, scope creep,
  dependency change, or debris was found.

The transferred renderer SHA-256 (`897a515f...f80c85`) and runtime SHA-256 (`9c64f8ad...07998`) exactly match the
fully tested `/tmp/tinygrad-rk-3p8-audit` worktree. No reset, reboot, or push was used.

---

## 2026-08-16 — reach the 3,700-line renderer milestone

The largest poorly factored live block was dynamic candidate selection. Typed dynamic loads and bool-total gated loads
parsed nearly the same bounded INT32 axes, while a separate raw-gather emitter rebuilt candidate byte matrices,
negative-index alternatives, external gates, row reductions, channel repetition, block combination, and terminal
packing. One fail-closed `_lower_dynamic_typed_load` now owns that semantic and physical path. It selects raw
FP16/INT16/INT32 bytes exactly, including signed zero and NaN payloads, and handles multiple axes, negative
normalization, external boolean gates, repeated trailing channels, blocked candidate sets, and exact total/fill gates.
Candidate, allocation, ARG-slot, u16 image-field, and signed-i32 raw-address limits are checked before table allocation.

The unified selector uses compact affine candidate gathers when the active rectangle permits them, otherwise retaining
exact offset tables. This intentionally changes one direct dynamic-INT32 image from 77 gathers to 13 while preserving
its 55 EW stages and raw output semantics. All other pre-existing UOp images remain byte-for-byte identical. One-use
integer comparison/conversion wrappers, duplicate offset planning, and several mechanical forwarding paths were folded
into their callers. Large range-independent contiguous outputs retain a compact affine proof. No semicolon packing,
recognizer deletion, comment deletion, or docstring deletion was used.

- Exact production diff: renderer **222 insertions / 331 deletions**, executable lines **3,799 -> 3,700** (**-99**);
  runtime remains **483**, total falls **29,350 -> 29,251**, and cumulative renderer reduction from the 6,616-line
  cleanup baseline is **2,916 lines**. Tests add **177 lines** and delete none.
- Across the original 95 UOp tests, **168 of 169** emitted-image/resource records are byte-for-byte identical. The
  intended direct dynamic-INT32 image changes from SHA-256
  `25cd9bc8c2104d910731e5d00a255fbce6def3db1761520f174c43b3eddf4bdc`, **9,225 bytes / 77 gathers / 55 EW**, to
  `047361e9989a11cd182c64cbd8ea4a0cf86fc613ce0a00d43bc60e94320ec314`, **6,153 bytes / 13 gathers / 55 EW**.
  A test-only physical raw-byte executor proves FP16/INT16/INT32, negative, multi-axis, gated, repeated,
  1,001-candidate blocked, and total/fill semantics, with encode/decode round trips.
- Focused selector hardware: FancyIndex **10 passed** in **156.37 s**; MaskedSelect and Nonzero **4 passed** in
  **214.80 s**.
- Full required hardware census after reboot: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all
  **445** collected cases in **837.08 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- UOp tests: **103 passed** with `-n12`. Repository Ruff, `mypy tinygrad/` (**216 files**), and
  `git diff --check`: pass. Fable-judge verdict: **VERIFIED**; no weakened checks, host tensor arithmetic,
  CPU/GPU/CMAC path, scope creep, or debris was found.

The first full census reached **125 passed + 6 expected skips** with zero failures before the native driver crashed in
the next collected test and the required elementwise health probe reported an unusable device. An out-of-band soft
reset attempt did not recover it; the user rebooted the board. The reboot cleared the uncommitted `/tmp` worktree, so
the patch was recovered from the local session rollout and replayed into a fresh detached worktree. Its renderer,
test, and full-diff SHA-256 values exactly reproduced the pre-reboot values
`3c29cfcbb95ac851a171d9880e09e168b50edd5b5d163a39cdee6625c5f8f75d`,
`9d590c386f606126e1b19d2bdf49fa40d861d89149ebb8e6f6b14f0bd51721dd`, and
`ca59ea61c9629b6c1029feec13dcbf5ad664aa8e0de8487a5253a307394898f6`. The post-reboot elementwise health probe
passed every operation and size before the authoritative census. No runtime reset path, tolerance change, or push was
added or performed.

---

## 2026-08-16 — reach the 3,600-line renderer milestone

The obsolete late FP16 fallback had become a second catalog layered behind the composable typed-UOp renderer. Its
masked arithmetic/MAX, ABS, SQRT, and LOG2 rules no longer owned any admitted image in the host census. Those rules,
their retry plumbing, and their private condition helpers were removed. Hardware validation proved that two narrow
contracts remain live: dynamic fancy indexing needs outer/inner mask normalization, and causal attention needs a late
EXP2 expansion followed by the existing storage-precision rules. Mask normalization now lives directly in
`_lower_dynamic_typed_load`, and the fallback contains only EXP2 plus the shared storage matcher.

The physical cleanup also removes the two-use stripe-gather wrapper, simplifies exact typed-load/repeated-index
planning, shares square-plus-constant parsing across atan and inverse-hyperbolic recognition, unifies LOG2's two
mantissa-normalization loops, and deletes unreachable pre-ReLU configuration. Related wire dataclass declarations and
constant tables were compacted without changing field order, comments, docstrings, serialized layouts, or stage order.

- Exact production diff: renderer **112 insertions / 216 deletions**, executable lines **3,700 -> 3,600** (**-100**);
  runtime remains **483**, total falls **29,251 -> 29,151**, and cumulative renderer reduction from the 6,616-line
  cleanup baseline is **3,016 lines**. No tests were changed.
- All **103** UOp tests pass. Their complete **100-image** encoded/resource census remains byte-for-byte identical,
  with ordered-record SHA-256 `33ff8d379f93348dc0acd942a6dbc0da700ad30ce19bfb86aaeba3b7faa60cf0` and
  unique-image SHA-256 `ef248b698668cb718ed675892d0226a9da69ebe40c0864baa2f52f81359a29e7`.
- Full required hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **783.53 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` and serial `-n0`.
- Repository Ruff, `mypy tinygrad/` (**216 files**), and `git diff --check`: pass. The final renderer SHA-256 is
  `ad5901e3c17aa09585f28c764e8325ad3886ab47e5c015b4200dad60cb8ffab2`, exactly matching the fully tested isolated
  `/tmp/tinygrad-rk-3p6.07Z5Zn` worktree.

The first candidate census was stopped at 31% after six fancy-index failures and one causal-attention failure exposed
the two live contracts above. Only those contracts were restored; all seven focused cases then passed and the complete
census was restarted from zero. No test weakening, tolerance change, host tensor arithmetic, CPU/GPU/CMAC path,
runtime change, reset, or push was introduced.

---

## 2026-08-16 — reach the 3,500-line renderer milestone

Starting from committed `79b73acea` (renderer **3,540**, runtime **470**), this renderer-only pass reaches renderer
**3,500** with runtime unchanged at **470**. A typed load plan now feeds gather construction and exact-offset
consumers; reduction `_ew_ops` and precision paths share their mechanical emission; Horner polynomial construction is
reused; and wrapper, predicate, and probe plumbing is consolidated without changing the admitted semantics.

The initial GQA candidate failed because it eagerly materialized million-lane offset vectors. The final fix makes
offset materialization optional and uses `require_offsets` only for the exact-offset consumers, including the BOOL
path. No CPU/GPU/CMAC numeric cheat was used.

- `sz.py`: renderer/runtime **3,500/470**. Host UOp tests: **103 passed** with `-n12`; Ruff, `mypy tinygrad/`
  (**216 files**), and `git diff --check`: pass.
- Exact full hardware census: **433 passed, 12 skipped, 154 subtests passed**, zero failures across all **445**
  collected cases in **783.88 s** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`; log
  `/tmp/rk-main-445-census.txt`; recorded run SHA-256 `d280...`.
- Final renderer blob SHA-256: `2b014edd998b1f8773121e640e08f7ac99c92858`. The run was forward-only; no reset,
  reboot, or push was used.
---

## 2026-08-17 — reach the 3,360-line renderer milestone

Starting from committed 97ce7aafa (renderer 3,500, runtime 470), this
renderer/unit-test candidate reaches 3,360 executable renderer lines while
runtime remains 470. The pass consolidates the grouped BOOL path into the
typed local executor: the old block/grouped BOOL catalog is not restored.
Bounded indexed local BOOL accumulators and their bridge are admitted only
with one consistent AND/OR update, identity initialization, static positive
extents, complete source coverage, one store/load mapping, and valid worker
bounds. FP16 and stored-BOOL sources, barriers, raw BOOL-byte output,
partial-buffer identity fill, and malformed/non-BOOL carriers stay explicit
fail-closed cases. The all_large shape is therefore handled by the compact
native mapper without raising a generic EW or unroll budget.

The MaxUnpool2D regression exposed a separate range-dependency bug. Static-local
rewriting had stripped the source-bearing dependency from a WEAK RANGE before
later static planning. Preserving the original dependent RANGE identity avoids
the 80 * 24 * 2350 = 4,512,000-environment path against the unchanged
_MAX_STATIC_RANGE_ENVS = 262,144 limit; it is a dependency correction, not a
budget relaxation. The direct-substitution form emits the same normal-mode
programs as clean HEAD and the expanded-root comparison. The focused
test_max_unpool2d probe passes.

The NLL 3D aggregate correction is retained and documented at this boundary:
the image header's total EW-op field and each EW record count are u32 fields,
while _RKIMAGE_U16_MAX applies only to serialized scratch/gather/host resource
counts. The failing NLL image has 147,733 valid small EW stages, so treating
the aggregate as a u16 resource is incorrect. The generic-image regression
constructs more than _RKIMAGE_U16_MAX stages, round-trips encode/decode, and
keeps the per-stage _MAX_EW_ELEMS_FP16 limit intact. No NLL tolerance or
numeric reference path changed.

The compact direct-INDEX BOOL mapper normalizes rangeify INDEX nodes through
the existing INDEX.load path, recognizes direct FP16 nonzero predicates, and
maps BOOL MUL/MAX spellings to semantic AND/OR. Typed-load/layout checks,
active-lane bounds, malformed BOOL ADD, mixed integer carriers, and resource
over-bounds remain fail-closed. The final v2f bounds/sentinel correction
requires low >= 0 for affine/scalar plans, permits only the explicit -1
offset sentinel, rejects values below that sentinel, rejects negative source
indices on active lanes, and normalizes arbitrary negative inactive lanes
(including -31..-28) to the -1 fill sentinel before the final guard.

The accepted result is deliberately narrower than several investigated
alternatives. A dedicated grouped-BOOL restoration added 113 executable
renderer lines (3,348 -> 3,461) and was rejected in favor of the generic
typed mapper. Deleting the dot reducer changed 292 of 20,250 values with
maximum error 0.02148 and was rejected. The multi-local FP32 adapter exposed
a scratch access at offset 64 in a 64-byte allocation for n=2,3,7,32, while
the generic fallback changed ordered-FP32 cancellation; no adapter was
exported. A shared math-recipe helper grew the renderer by four executable
lines instead of reducing it. The first typed-load/offset draft eagerly
materialized million-lane GQA offsets; the final plan keeps materialization
optional and requires offsets only for exact-offset consumers.

- Exact production diff: tinygrad/renderer/rockchip.py 284 insertions / 426
  deletions; test/unit/test_rockchip_uops.py 136 insertions / 2 deletions;
  total 420 / 428. sz.py reports renderer/runtime 3,360/470, a net renderer
  reduction of 140 executable lines from 3,500.
- Host gates: test/unit/test_rockchip_uops.py 110 passed with -n12;
  repository Ruff, mypy tinygrad/ (216 source files), and git diff --check
  pass. No assertion was removed or weakened.
- Isolated final-artifact evidence includes the required elementwise health
  probe (the standalone run initially passed), focused MaxUnpool2D hardware 1
  passed, and focused all_large hardware 1 passed. The authoritative backend
  census collected and ran exactly 445 cases: 433 passed, 12 skipped, 0
  failed, and 154 subtests passed in pytest 857.52s (0:14:17), with wrapper
  wall time 918.752s, using
  FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest
  test/backend/test_rockchip.py -q -n0. The final log is
  /home/orangepi/rk-artifacts-current/full445-3360-20260817.log; its
  SHA-256 is 8d6962196f8ddeffb56a0228d0c64faf8ed91ac47ebea2b91f5910534e6c4940.
  Later standalone 4MiB retries can fail with ENXIO when fragmented 8MiB CMA
  cannot satisfy the allocation; this did not invalidate the successful
  census, which exited 0 with pre/post holder checks passing.

The current final source diff is limited to the renderer and unit test. No
input-dependent host numeric execution, CPU/GPU/CMAC path, tolerance
relaxation, runtime/core change, reset, reboot, or push was used. Current
blob IDs are renderer a9fb0db85893401ccd5eae7246d77522c96e5596, unit test
8fe3aa62b36d80f9ce137ba936a6eddb18b7bf76, and unchanged runtime
b9aa3842afb95edd2de7ea57bf2de2bfde1475df. The combined source diff SHA-256
(git diff --binary HEAD -- tinygrad/renderer/rockchip.py
test/unit/test_rockchip_uops.py) is
1bb927f9d1f953f6f0822a1aa8c0e81f47d8d1a790f2c2aa83777060f45683b8.

Proposed local commit subject:

    WIP rockchip: reach 3.4k renderer milestone

Commit remains pending; no commit or push was made.

---

## 2026-08-18 — reach the 3,299-line renderer milestone

Starting from committed `18cd6f60e` (renderer 3,360, runtime 470), the accepted
closed cleanup reaches exactly 3,299 executable renderer lines (**−61**) while
runtime remains 470. Byteplane materialization now uses one gather helper;
static-local child/reduction/extrema lowering shares one bounded child builder;
static gather/remap forwarding and typed storage matcher construction are
consolidated without changing the existing allocation, gather, barrier, or
serialized-image contracts. Argument slots outside the u16 wire range now
reject before encoding, with a focused unit regression for both overflow
values. No unrelated matcher deletion was included in this milestone.

- Exact production diff: `tinygrad/renderer/rockchip.py` **102 insertions / 161
  deletions**; `test/unit/test_rockchip_uops.py` **8 insertions / 0 deletions**;
  executable renderer **3,360 -> 3,299 (−61)**, runtime remains **470**. The
  renderer SHA-256 is
  `a1a2cda1f64c3130881be0dcb5e2e75b47a6dfa99ace9e5453ff834be6df1258`; the
  focused-test SHA-256 is
  `d975706dd7dd982f016a16b36ca29909f77cfc08f02e30ee9a06be43001f4f6e`; the
  combined source diff SHA-256 is
  `8e4f5b8f620015c0c7dda24b5f3e1018417c9fc279cbebf1b7a6ffc277ddb7d6`.
- Host gates: `test/unit/test_rockchip_uops.py -q -n12` **111 passed**;
  repository Ruff passed; `mypy tinygrad/` passed for **216 source files**;
  `git diff --check` passed; `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP
  .venv/bin/python -m pytest test/backend/test_rockchip.py --collect-only -q
  -n12` collected exactly **445** cases.
- The NPU census followed the corrected boot CMA configuration: the duplicate
  boot `extraargs` assignment was fixed so the intended 128 MiB CMA pool was
  effective (`CmaTotal: 131072 kB`), avoiding the prior fragmented 8 MiB pool
  and its 4 MiB RKNPU allocation failure chain. This was board configuration
  context, not a renderer/runtime workaround.
- Authoritative hardware command:

      FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -q -n0

  The pytest process reported **433 passed, 12 skipped, 154 subtests passed**,
  **0 failed**, across **445 collected** cases. The log is
  `/home/orangepi/rk-artifacts-current/full445-exact3299-post-cma-20260818.log`.
  Its wrapper metadata records a bookkeeping `CENSUS_WRAPPER_EXIT: 2` because
  the orchestration escaped `PIPESTATUS`; the pytest result itself completed
  normally with `CENSUS_PYTEST_EXIT: 0`. No wrapper exit is used to overclaim
  hardware success.

No test or tolerance was weakened, and no CPU/GPU/CMAC path, host tensor
numeric computation, or fallback was added. No runtime/core change was made;
all tensor arithmetic remains on the DPU EW path. Commit remains pending; no
push was made.

---

## 2026-08-18 — complete exact 3,200-line renderer milestone

The exact 3,200-line renderer is now hardware-validated after the post-reboot
repair. Starting from committed exact-3,299 HEAD `3814d0439d473e367e5b541b2361a97f2cf2cc21`,
the accepted renderer is exactly **3,200 executable lines** by `sz.py` (runtime
remains **470**) and has SHA-256
`fd730545a7b11ee52e880f7b81d4e4e0bd06b893dd0ec59888379405bbd850ba`. The
earlier nine-case screen was superseded by the focused post-repair check and
the authoritative full census below.

- Vendor `~/rk3588/examples/elementwise.py` health passed both before the
  focused rerun and after the authoritative census. The post-reboot and
  post-authoritative logs are
  `/home/orangepi/rk-artifacts-current/elementwise-post-reboot-exact3200-20260818.log`
  and
  `/home/orangepi/rk-artifacts-current/elementwise-post-authoritative-exact3200-20260818.log`;
  both exited 0.
- Focused repair check for the nine previously affected nodes: **9 passed in
  53.92 s**, exit 0. Log:
  `/home/orangepi/rk-artifacts-current/focused9-exact3200-post-reboot-20260818.log`.
- Host gates: `test/unit/test_rockchip_uops.py -q -n12` **111 passed**;
  repository Ruff passed; `mypy tinygrad/` passed for **216 source files**;
  `git diff --check` passed; backend collection gathered exactly **445**
  cases.
- Authoritative serial command:

      FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP .venv/bin/python -m pytest test/backend/test_rockchip.py -q -n0

  Pytest reported **433 passed, 12 skipped, 154 subtests passed, 0 failed**
  across exactly **445** collected cases in **839.57 s** (`0:13:59`), with
  exit code **0**. Log:
  `/home/orangepi/rk-artifacts-current/full445-repaired-exact3200-post-reboot-20260818.log`.
- Kernel dmesg retains a caveat from the run: **4 job timeouts** and **1,620
  soft resets** were recorded, but there was no IOMMU fault, kernel oops, or
  panic. Reset activity stopped, and the post-authoritative elementwise health
  check passed.

No CPU/GPU/CMAC path, host tensor numeric computation, fallback, or tolerance
relaxation was used; tensor arithmetic remains on the DPU EW path. The exact
3,200-line renderer milestone is complete and recorded locally; no push was
made.

---

## 2026-08-19 — pending reboot checkpoint

The shared checkout is at HEAD after WIP commit
`a6e9f7c3bd64b52c63edaadd8e1ab0c661ffa5dd` (`WIP rockchip: preserve 3.195k
renderer cuts`). The shared renderer is **3,195 executable lines** and host-
verified; no new hardware census has been run for it.

Offline composition records remain separate from the shared checkout:

- The exact-3,128 isolated tree is
  `/home/orangepi/tinygrad-compose-exact3133-to-3128-int32-shift-luna-max-20260818`.
  Its INT32-shift cut was verified offline only; no NPU or hardware execution
  was used.
- The exact-3,124 composition is
  `/home/orangepi/tinygrad-compose-shared3195-into-exact3128-luna-max-20260819`.
  `sz.py` reports renderer/runtime **3,124/470**. Recorded host gates are
  `test/unit/test_rockchip_uops.py` **111 passed** with `-n12`, repository Ruff
  passed, `mypy tinygrad/` passed for **216 files**, `py_compile` passed,
  `git diff --check` passed, and backend collection gathered exactly **445**
  cases. No hardware gate was run for this composition.
- `/home/orangepi/rk-artifacts-current/audit-next28-exact3128-luna-max-20260819.report.md`
  identifies a projected exact-3,100 chain (`3128 -> 3117 -> 3105 -> 3100`).
  That exact-3,100 chain is not implemented, judged, or hardware-tested.
- The rejected micro experiment report is
  `/home/orangepi/tinygrad-micro-cleanup-exact3213-luna-max-20260818/host-scatter-cleanup-exact3213-rejected-20260818.report.md`;
  it retained no patch.

The prior exact-3,200 result remains the last complete 445-case hardware
census: **433 passed, 12 skipped, 154 subtests passed, 0 failed**, recorded in
`/home/orangepi/rk-artifacts-current/full445-repaired-exact3200-post-reboot-20260818.log`.

Current health blocker: `simple_add.py` passed once after reboot then failed
on 4MiB CMA fallback `ENXIO`; dmesg proves built-in RKNPU0.9.8 missing
`num_pages` + refcount underflow. **Do not retry.** The durable driver patch is
`/home/orangepi/rknpu-driver-fix-20260818/rknpu-0.9.8-fallback-mmap-fix.patch`.
Pending next action is patched-kernel install/reboot. The elementwise
`NON_CONTIGUOUS` experiment was reverted; `simple_add.py` remains the health
gate.

This is a pending-reboot checkpoint: no new tests, hardware retries, or push
were performed.

---

## 2026-08-19 — exact 3,099 renderer gate, pending reboot

WIP code commits `6e036b997` and `54acddaa4` produce renderer/runtime
**3,099/470**. Host gates are UOps **114 passed** with `-n12`, repository Ruff
pass, `mypy tinygrad/` pass, backend collection **445**, and the bounds census
reported **zero OOB** images. The `source_matrix_lanes` repair and the
full-stride mapped-temporary scratch repair were independently verified.

The first exact-3,097 hardware census log reported **432 passed, 12 skipped,
1 failed, and 154 subtests**. Its rank-2 failure was nonzero scratch
underallocation introduced by inlining. The latest `simple_add.py` health
sequence had exactly one failure: the fourth 4 MiB allocation hit the CMA
fallback with `ENXIO`; there was no retry. Reboot is pending and the durable
driver patch remains uninstalled.

After reboot, run exactly one `.venv/bin/python ~/rk3588/examples/simple_add.py`
health check, then exactly one serial full-445 census with a fresh isolated
`CACHEDB` and `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`; do not run
`elementwise.py`.

The next local checkpoint is `WIP rockchip: save repaired 3.1k gate before
reboot`. No tests, hardware retries, or push are part of this checkpoint.

---

## 2026-08-19 — complete exact 3,099-line renderer milestone

The repaired exact-3.1k gate is complete in milestone commit
`30d3af7c2` (`rockchip: complete 3.1k renderer milestone`). Its code
checkpoint is `54acddaa49aa9ca84e4c6ced872d192c4677180b` (`WIP rockchip: fix
scratch bounds for 3.1k gate`); the earlier `194f94bc` commit was only the
pre-completion checkpoint. The renderer/runtime sizes are exactly **3,099/470**;
the renderer SHA-256 is
`fa07e25891d1c0ade44bbc802a2be1776ca696e9c8ca7a93e39e6bf3342be349`, and the
runtime is unchanged.

- Host gates are UOps **114 passed**, repository Ruff and `mypy tinygrad/`
  passed, backend collection gathered exactly **445** cases, and the bounds
  census found **833 unique bounds / 8,068 occurrences**, with **12/24
  nonzero** values and **zero OOB**.
- The prior exact-3,097 census was a failed preliminary run (**432 passed,
  12 skipped, 1 failed, 154 subtests**). Its rank-2 failure was diagnosed as
  nonzero scratch underallocation introduced by inlining; the full-stride
  mapped-temporary scratch repair fixed that case.
- Exactly one fresh-cache serial census was run with cache directory
  `cache-dwzkyM`. The authoritative log is
  `/home/orangepi/rk-artifacts-exact3099-final-20260819/final_hardware_gate_repaired.log`.
  The canonical pytest log proves **433 passed, 12 skipped, 0 failed, and
  154 subtests** across all **445** cases, with pytest exit code **0**.
- Separately, agent-captured evidence outside that pytest log records that the
  post-reboot `.venv/bin/python ~/rk3588/examples/simple_add.py` health check
  passed.
- Separately, agent-captured dmesg evidence outside that pytest log records a
  caveat of **1,369 resets**, **10 timeout messages**, **5 IOMMU-related
  strings**, and **0 kernel oops**; these are recorded observations, not
  census failures.

The exact 3,099-line renderer milestone is complete and recorded locally; no
push was made.

---

## 2026-08-19 — exact 2,998 renderer candidate, pending reboot

WIP code commit `e16b4ac18` (`WIP rockchip: stage 2.998k renderer candidate`)
deletes the dead `RKLoopReduction` family: `RKLoopReduction`, its matcher and
load helper, both loop-reduction lowerers, and the dead dispatcher branch. The
candidate is exactly **2,998/470** renderer/runtime executable lines. The
candidate renderer SHA-256 is
`ef50de4e8b2d97e6e2e6f85d1a936b431ec633cda3ba13e6becdd86461e141c6`; runtime
remains unchanged. The deletion is **111 physical lines / 101 executable
`sz.py` lines**.

- Host gates: UOps **114 passed** with `-n12`, repository Ruff passed,
  `mypy tinygrad/` passed for **216 source files**, backend collection gathered
  exactly **445** cases, and `git diff --check` passed.
- Independent baseline/candidate image capture returned exactly **1,499**
  lowerer results on each side (1,320 accepted images each). The complete
  return-sequence digest is byte-identical on both sides:
  `6e8f143fd417267f084645ab8a371eb9a7cb1e5c24ff7c26b7e99831404e5f2d`.
  The independent image differential reported no first mismatch; independent
  judges returned **VERIFIED**.
- The host-only bounds census covered **2,006 unique images / 2,770 image
  occurrences**, with **0 issues**, 0 lower failures, and 0 compile
  exceptions. It also checked 1,314 cache rows with 0 malformed rows; no
  hardware state was touched.

The health sequence made exactly one `.venv/bin/python
~/rk3588/examples/simple_add.py` attempt; it failed at the 4 MiB CMA fallback
with `ENXIO`. **No retry** was made. Reboot is pending, and no hardware census
was run for this candidate. After reboot, run exactly one `simple_add.py` health
check, then exactly one fresh-cache serial full-445 census; do not retry either
step.

The progress checkpoint is saved in commit subject

    WIP rockchip: save 2.998k candidate before reboot

The pre-existing untracked reject note remains untracked; no push was made.

---

## 2026-08-19 — exact 2,998 renderer experiment rejected

The exact-2,998 candidate from WIP code commit `e16b4ac18` and saved in
checkpoint commit `7c5836e3c` was run through the authoritative full backend
census. The run reported **427 passed, 12 skipped, 6 failed, and 154 subtests**
across 445 collected cases, with pytest exit code **1**. The canonical log is
`/home/orangepi/rk-artifacts-exact2998-final-20260819/final_hardware_gate.log`.
The six failed IDs were:

- `test/backend/test_rockchip.py::TestRockchipDotOps::test_broadcastdot`
- `test/backend/test_rockchip.py::TestRockchipDotOps::test_dot`
- `test/backend/test_rockchip.py::TestRockchipDotOps::test_dot_1d`
- `test/backend/test_rockchip.py::TestRockchipDotOps::test_multidot`
- `test/backend/test_rockchip.py::TestRockchipReductionOps::test_sum_dtype_arg`
- `test/backend/test_rockchip.py::TestRockchipReductionOps::test_sum_full`

The exact six-graph host-only differential proves these are not a
zero-acceptance case. Their pre-render schedules and AST/call UOps are equal,
while the base's exact old routes accept the graphs: `_loop_reduction_match` accepts
all six; `_lower_dot_loop_reduction` accepts the four dot graphs
(`broadcastdot`, `dot`, `dot_1d`, `multidot`); and
`_lower_scalar_loop_reduction` accepts the two reduction graphs
(`sum_dtype_arg`, `sum_full`). The serialized images materially change from
the old loop-reduction form to the generic mapped-add form (for example,
`broadcastdot` changes from 168,490 bytes / 4,011 EW ops to 5,769 bytes / 66
EW ops), despite the earlier broad 1,499-return corpus being byte-identical.
The route acceptance counts are `1/1` for `_loop_reduction_match` on every
graph, `1/1` for the dot lowerer on the four dot graphs and `0/1` on the two
reduction graphs, and `1/1` for the scalar lowerer on the two reduction graphs.
See the [six-failure host-only differential report](/home/orangepi/rk-artifacts-exact2998-final-20260819/six-failure-host-diff-20260819.md),
the [exact-2,998 integration report](/home/orangepi/rk-artifacts-exact2998-final-20260819/exact2998-deadroute-loop-reduction-removal-20260819.report.md),
and the [full hardware log](/home/orangepi/rk-artifacts-exact2998-final-20260819/final_hardware_gate.log).

The 2,998 candidate is **rejected** and the 3,000-line renderer milestone is
**NOT achieved**. Forward restore commit `9c2799c1b` (`rockchip: restore
precision reduction routes`) is current. The renderer is back at exact
**3,099/470** executable lines with SHA-256
`fa07e25891d1c0ade44bbc802a2be1776ca696e9c8ca7a93e39e6bf3342be349`, matching
the previously hardware-verified 3.1k source.

OpenNPU audit conclusion: **no line-negative transplant**. OpenNPU was used
only as an external reference; no OpenNPU code was transplanted into this
renderer. The retained [OpenNPU architecture report](/home/orangepi/opennpu-rk3588-audit-Qqy5uQ/repo/docs/ARCHITECTURE.md),
[operator-lowering report](/home/orangepi/opennpu-rk3588-audit-Qqy5uQ/repo/docs/OP_LOWERING.md),
and [register-command reference](/home/orangepi/opennpu-rk3588-audit-Qqy5uQ/repo/docs/NPU_REGCMD_REFERENCE.md)
are reference links only; they provide no line-negative transplant.

No further hardware retry or push was performed.

---

## 2026-08-21 — hardware-verify corrected 2,912-line renderer checkpoint

The prior exact-2,899 attempt is not a valid milestone. Its post-reboot
serial census was explicitly run with `ROCKCHIP_SUBMIT_RETRIES=0` and stopped
at a submit timeout in `TestRockchipLossOps.test_cross_entropy_smoothing`
(`247 passed, 6 skipped, 1 failed`, **254 nodes reached**, **126 subtests**,
exit 1). A separate retry-4 run then reached
`TestRockchipReductionOps.test_sum_dtype_arg` but failed numerically:
Rockchip returned `-2.8164062` versus the FP32 reference `-2.7783928`, with
absolute error `0.03801346` above the allowed `0.018891964` (`368 passed,
12 skipped, 1 failed`, **381 nodes reached**, **154 subtests**, exit 1).
Those runs are recorded in [the no-retry failure log](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-2899-postreboot/first-failure.txt)
and [the retry-4 failure log](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-2899-retry4/first-failure.txt),
with node accounting in [the no-retry count record](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-2899-postreboot/count-accounting.txt)
and [the retry-4 count record](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-2899-retry4/count-accounting.txt).

The corrected source is HEAD `0ee929a187a5562711e246ff4a5fb4eb8b3b7b5a`
(`WIP rockchip: restore precise scalar reductions`) and measures exactly
**2,912 executable renderer lines** with runtime **470** by `sz.py`. The
2,899 count therefore was not a valid completed milestone. This entry records
the corrected 2,912-line checkpoint; it does not establish any 2.9k-or-lower
milestone.

The root cause of the `sum_dtype_arg` failure was a duplicate normalized
`INDEX`/`LOAD` pair. The scalar parser admitted both the source `INDEX` and its
same-index `LOAD`; the lowerer then saw 270 flattened blocks instead of 135
unique lanes, rejected the scalar route, and fell back to the generic mapped
route. That route performed a sequential FP16 fold before the terminal FP32
conversion, producing the observed `-2.8164062`. The v2 repair requires the
canonical one-source `LOAD` / two-source `INDEX` / direct `PARAM` structure and
stably deduplicates normalized `INDEX.key` entries in first topological order.
It preserves rejection of gated or aliased loads, dynamic/out-of-bounds
indices, distinct physical index keys, and extent mismatches. The [v2 repair
report](/home/orangepi/rk-artifacts-exact2900-20260821/fix-sum-dtype/report-v2.md)
and reproducible [v2 patch](/home/orangepi/rk-artifacts-exact2900-20260821/fix-sum-dtype/fix-v2.patch)
contain the host-only contract and image evidence.

- Host gates on the corrected HEAD: `test/unit/test_rockchip_uops.py -q
  -n12` **115 passed**; Ruff passed; `mypy tinygrad/` reported no issues
  in **216 source files**; `compileall` and `git diff --check` passed; backend
  collection gathered **445 nodes**. The [v2 corrected source tree report](/home/orangepi/rk-artifacts-exact2900-20260821/fix-sum-dtype/report-v2.md)
  records the 115-pass, Ruff, mypy, compileall, and diff-check results; the
  [full-run metadata](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-fixed2912/run-metadata.txt)
  records the corrected HEAD and 445-node expectation.
- Focused hardware `TestRockchipReductionOps.test_sum_dtype_arg`: **1 pytest
  node passed**, exit 0. Its captured pytest log and
  `retry-markers.txt` contain **zero retry/timeout/reset log markers**; this
  does not establish that zero internal retry attempts occurred. Metadata
  records corrected HEAD, renderer 2,912, fresh cache, and explicit
  `ROCKCHIP_SUBMIT_TIMEOUT_MS=6000` / `ROCKCHIP_SUBMIT_RETRIES=4`; see [the
  focused pytest log](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-sum-dtype-fix/pytest.log),
  [retry markers](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-sum-dtype-fix/retry-markers.txt),
  and [metadata](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-sum-dtype-fix/metadata.txt).
- The fresh-cache serial command
  `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP
  ROCKCHIP_SUBMIT_TIMEOUT_MS=6000 ROCKCHIP_SUBMIT_RETRIES=4` with pytest
  `-n0` completed **433 passed, 12 skipped, 0 failed** across **445 collected
  pytest nodes**, with **154 subtests passed**, and pytest exit code **0**.
  The [full pytest log](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-fixed2912/pytest.log),
  [count accounting](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-fixed2912/count-accounting.txt),
  and [run metadata](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-fixed2912/run-metadata.txt)
  identify the exact current-HEAD run. The dedicated [retry/reset evidence](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-fixed2912/retry-reset-evidence.txt)
  records **zero retry/timeout/reset log markers in the captured census
  log**; this is not a claim of zero retry attempts.

No production CPU/GPU/CMAC fallback, runtime/core workaround, or tolerance
relaxation was added. The corrected scalar image is native DPU EW with no
host gathers/scatters. The unit regression's NumPy image simulator is
host-only test evidence, not a production execution path. This draft only
records the captured gates above; it did not modify the shared checkout or
run another NPU test.

---

## 2026-08-21 — exact 2,892 renderer milestone hardware-verified

Starting from clean `1ece167a2` (renderer/runtime **2,898/470** by `sz.py`),
current HEAD `e9afa79617bd66e928528ec3836dee7793a077f8`
(`WIP rockchip: readable cleanup to 2.892k`) measures
exactly **2,892/470** executable lines, with renderer SHA-256
`e4a0c4eece7970bf33686f9633b3488f72616134629ea706d4474d44af488cdb`. Since
**2,892 ≤ 2,900**, this is the exact **2.9k-or-lower** renderer milestone.
The v2 [milestone report](/home/orangepi/rk-artifacts-exact2900-20260821/milestone2892-progress/report.v2.md),
[progress patch](/home/orangepi/rk-artifacts-exact2900-20260821/milestone2892-progress/progress.v2.patch),
and [patch SHA](/home/orangepi/rk-artifacts-exact2900-20260821/milestone2892-progress/progress.v2.patch.sha256)
record the corrected provenance.

- The host-only UOps/Ruff/mypy/collection and scalar evidence were captured
  in an isolated reconstruction rooted at clean `1ece167a2`; its renderer
  content hash matches current HEAD
  `e9afa79617bd66e928528ec3836dee7793a077f8`; these are not separately
  captured git-e9afa worktrees. The gates are Rockchip UOps **115 passed**
  with `-n12`, Ruff passed, `mypy tinygrad/` reported no issues in **216
  source files**, and backend collection **445**. The canonical scalar
  half→float route preserves the exact 5,453-byte image SHA-256
  `b79d23fe6c1f99b3e28b2ed7758e02c89f1d23c6d838465123065e8eedae1f1c`; see
  the [isolated host verdict](/home/orangepi/rk-artifacts-exact2900-20260821/final-judge-exact2892/VERDICT.md)
  and [candidate capture manifest](/home/orangepi/rk-artifacts-exact2900-20260821/final-judge-exact2892/corpora/capture-candidate.json).
- Before reboot, the one current-head `simple_add.py` attempt failed while
  mapping the 4 MiB buffer with `OSError: [Errno 6] No such device or address`
  (`ENXIO`), exit 1; it was stopped with no retry, reset, or census. See the
  [pre-reboot health manifest](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-exact2892/health-gate.log).
- After reboot, the current-head `simple_add.py` health gate passed (exit 0);
  see its [post-reboot manifest](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-exact2892-postreboot/result.txt),
  [captured output](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-exact2892-postreboot/output.log),
  and [HEAD record](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-exact2892-postreboot/head.txt).
- The sole full current-head census used `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF
  DEV=ROCKCHIP`, serial pytest `-n0`, a fresh cache, and explicit
  `ROCKCHIP_SUBMIT_TIMEOUT_MS=6000 ROCKCHIP_SUBMIT_RETRIES=4`. It completed
  **433 passed + 12 skipped = 445** nodes, **154 subtests passed**, exit 0;
  see the [census command manifest](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-exact2892/command.txt),
  [run manifest](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-exact2892/run-metadata.txt),
  and [count manifest](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-exact2892/count-accounting.txt).
  The captured log contains zero retry/timeout/reset markers, as recorded in
  the [marker manifest](/home/orangepi/rk-artifacts-exact2900-20260821/hardware-full445-exact2892/retry-reset-evidence.txt);
  this is not a claim of zero retry attempts—four retries were configured.

No CPU/GPU/CMAC/host numeric production path, runtime/core workaround, or
tolerance relaxation was added. The unit NumPy image simulator remains
test-only evidence. This documentation pass ran no new NPU test and changed
only `progress.md` in the isolated checkout.

---

## 2026-08-21 — exact 2,860 renderer milestone hardware-verified

The exact-2,860 lineage is exact-2,870 `01e59d5d8` → current HEAD
`309da765e` (`WIP rockchip: readable cleanup to 2.860k`). `sz.py` reports
renderer/runtime **2,860/470**; the [exact2860 judge report](/home/orangepi/rk-artifacts-next2860-20260821/final-judge-exact2860/VERDICT.md)
and [size manifest](/home/orangepi/rk-artifacts-next2860-20260821/final-judge-exact2860/sz.log)
record the isolated one-file reconstruction and exact renderer image.
The [milestone report](/home/orangepi/rk-artifacts-next2860-20260821/milestone2860-progress/report.md)
and [progress patch](/home/orangepi/rk-artifacts-next2860-20260821/milestone2860-progress/progress.patch)
preserve this documentation delta.

- Host gates/evidence record Rockchip UOps **115** with `-n12`, b79d scalar
  oracle **1**, Ruff, mypy with no issues in **216 source files**,
  compileall/diff check, and backend collection **445**; these results are
  summarized in the [exact2860 judge report](/home/orangepi/rk-artifacts-next2860-20260821/final-judge-exact2860/VERDICT.md).
- The [persisted replay manifest](/home/orangepi/rk-artifacts-next2860-20260821/final-judge-exact2860/replay-exact2870-exact2860.json)
  records **2,073** normalized cases: 64 `native` records were cross-encoded
  by the reference and candidate renderers (the only candidate renderer
  encodings in this replay), while 1,000 `mapped` records were persisted-record
  normalization/digest comparisons, not candidate replays. The 1,500 full-
  capture and 1,226 compile-only checks are persisted JSON/digest equality
  checks, not exact-HEAD generator reruns; the 4 Kahan, 5 reduce, and 1,000
  stage groups in the manifest were not candidate-replayed.
- Before reboot, the one current-head `simple_add.py` health attempt failed
  with `OSError: [Errno 6]` (`ENXIO`) while mapping the 4 MiB buffer and was
  stopped without retry, reset, or census; see the [pre-reboot manifest](/home/orangepi/rk-artifacts-next2860-20260821/hardware-exact2860/simple_add-health-gate.txt).
  After reboot, current-head `simple_add.py` passed (exit 0); see the
  [post-reboot timestamps/HEAD manifest](/home/orangepi/rk-artifacts-next2860-20260821/hardware-exact2860-postreboot/timestamps.txt),
  [exit manifest](/home/orangepi/rk-artifacts-next2860-20260821/hardware-exact2860-postreboot/exit_code.txt),
  and [captured output](/home/orangepi/rk-artifacts-next2860-20260821/hardware-exact2860-postreboot/simple_add.log).
- The sole full census used `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`,
  serial pytest `-n0`, a fresh cache, and explicit
  `ROCKCHIP_SUBMIT_TIMEOUT_MS=6000 ROCKCHIP_SUBMIT_RETRIES=4`. It completed
  **433 passed + 12 skipped = 445** nodes, **154 subtests passed**, exit 0;
  see the [census command](/home/orangepi/rk-artifacts-next2860-20260821/hardware-full445-exact2860/command.txt),
  [counts](/home/orangepi/rk-artifacts-next2860-20260821/hardware-full445-exact2860/counts.txt),
  and [run metadata](/home/orangepi/rk-artifacts-next2860-20260821/hardware-full445-exact2860/run-metadata.txt).
  Only aggregate accounting is evidenced; no per-node manifest is persisted.
  The marker manifest records exactly zero captured
  `retry|timeout|resubmit|submit` markers while four retries were configured;
  see the [marker manifest](/home/orangepi/rk-artifacts-next2860-20260821/hardware-full445-exact2860/retry-markers.txt);
  this does not prove zero attempts.

No CPU/GPU/CMAC or production host-numeric path, runtime/core workaround, or
tolerance relaxation was added. This documentation pass changed only
`progress.md` in the isolated checkout and ran no additional NPU test.

---

## 2026-08-22 — exact 2,825 renderer milestone hardware-proven

The code milestone is commit `444561b3c30c37e2201fd512e5040fe9627e3368`
(`WIP rockchip: readable cleanup to 2.825k`), directly descended from
`4bdf6f46fa7671cda110c621e391f76356da8a5b` (the exact-2,860 documentation
commit). This is the `4bdf -> 444` source lineage. `sz.py` reports renderer/
runtime **2,825/470**. The code commit tree is
`4ca3fa9de047af393d12345269d728a4633fb6ce`; the renderer blob is
`725e025a2ad6257c165a055b33ab6520d890f750` (SHA-256
`9a82477963d4ea26fd26a4b599917bbfb2da599090f02bfadf955fa34dda799e`), and
the unchanged runtime blob is `b9aa3842afb95edd2de7ea57bf2de2bfde1475df`
(SHA-256 `da7ef1b98ac1658ddbd19b44b41c7f6d451e93acc43904b2bcd4523dd1ec683e`).
The source diff from `4bdf` is renderer-only (**57 insertions, 92 deletions,
net -35**). The tracked tree for the hardware artifacts was clean; five
pre-existing untracked files (`cut-oracle.json`, `loop-reduction-refactor-judge`,
`raw-cast-unification-3272-luna-max-20260818.reject.md`,
`rockchip-2911-ast-audit.reject.md`, and `rockchip-2911-horner-audit.reject.md`)
were preserved and are not part of this patch. The isolated combined-2825
provenance/application record has
final source commit `0ae93773786c223c2d6cb380a8f8dba5a8df36af`, the same final
tree/renderer blob, and union-patch SHA-256
`d4a174233133cb7adb09eb48cd895bea05c193181919a9b7b84ba359379930b7`;
see the [combined report](/home/orangepi/rk-artifacts-next2830-20260821/combined-2825/REPORT.md)
and [provenance verdict](/home/orangepi/rk-artifacts-next2830-20260821/combined-2825-judge-provenance/VERDICT.md).

- Independent host gates on the combined-2825 candidate are Rockchip UOps
  **115 passed** with `-q -x -n12`, Ruff pass, compileall pass,
  `mypy tinygrad/` with no issues in **216 source files**, `git diff --check`
  pass, and backend collection of exactly **445** tests. Candidate/baseline
  normalized semantic payloads are equal; raw b79d differences are metadata-only;
  their exact limits are b79d **11 graphs** (**10 compile successes** plus the
  intentional requested-double rejection), ABI/image/malformed **4 images,
  20 malformed blobs, 110 stages**, decode flags **256 cases (2 accepted,
  254 rejected)**, random ABI mutation/truncation **781 records**, and a
  function/class signature inventory of **268 entries**. The corrected
  top-level b79d normalization record is authoritative; the older intermediate
  `/home/orangepi/rk-artifacts-next2830-20260821/combined-2825/evidence/b79d-normalized.json`
  is not used for this result.
  The strict-v2 component evidence, persisted from the independently verified
  `16ac8bf` history rather than freshly replayed after the exact-444
  composition, adds selector **262,162 cases/0 mismatches**, shift **272
  cases/0 errors**, RKContext **2,755 records/0 exceptions**, and **six**
  exhaustive families with zero mismatches. These are host/provenance checks;
  no NPU execution is implied.
- The pre-reboot health artifact records a captured current-head
  `.venv/bin/python ~/rk3588/examples/simple_add.py` failure with
  `OSError: [Errno 6] No such device or address` (`ENXIO`) while mapping the
  4 MiB buffer and exit 1. Its artifact does not persist a reboot event or an
  invocation/retry counter; see the [pre-reboot health artifact](/home/orangepi/rk-artifacts-next2825-20260821/hardware-health/health-gate-metadata.txt)
  and its [captured stderr](/home/orangepi/rk-artifacts-next2825-20260821/hardware-health/simple_add.stderr.log).
  Following the user-reported reboot, the later recorded current-head
  invocation passed (exit 0), ending with `ADD ... PASS`; see the
  [post-reboot metadata](/home/orangepi/rk-artifacts-next2825-20260822/hardware-health-postreboot/health-gate-metadata.txt)
  and [captured output](/home/orangepi/rk-artifacts-next2825-20260822/hardware-health-postreboot/simple_add.stdout.log).
- The authoritative full census was one serial invocation with a fresh
  dedicated cache: `env -u SKIP_SLOW_TEST FORWARD_ONLY=1 DEFAULT_FLOAT=HALF
  DEV=ROCKCHIP ROCKCHIP_SUBMIT_TIMEOUT_MS=6000 ROCKCHIP_SUBMIT_RETRIES=4
  CACHEDB=/home/orangepi/rk-artifacts-next2825-20260822/hardware-full445/cache.db
  .venv/bin/python -m pytest test/backend/test_rockchip.py -q -x -rs -n0`.
  Collection and
  observation were both **445** (`collected_total=445`, `observed_total=445`);
  the result was **433 passed + 12 skipped = 445**, **154 subtests passed**,
  exit 0, in **998.06 s**. Execution accounting is aggregate, with the 445
  collection names persisted; no per-node outcome manifest is present. See
  the [census command](/home/orangepi/rk-artifacts-next2825-20260822/hardware-full445/command.txt),
  [counts](/home/orangepi/rk-artifacts-next2825-20260822/hardware-full445/counts.txt),
  and [run metadata](/home/orangepi/rk-artifacts-next2825-20260822/hardware-full445/run-metadata.txt).
  The captured log has zero `retry|timeout|resubmit|submit` markers while
  four retries were configured; marker absence does not prove zero attempts.
  See the [marker scan](/home/orangepi/rk-artifacts-next2825-20260822/hardware-full445/retry-markers.txt).

No CPU/GPU/CMAC/host production numeric fallback, runtime/core workaround, or
tolerance relaxation was added by the audited `4bdf`-to-`444` source diff,
and none was identified in the host/provenance/semantic evidence. This
documentation pass ran no `elementwise.py`, rerun, reset, or additional NPU
test (the captured health program output itself records `reset_npu ret=0`).
Commit `444561b3c` is the code milestone; this progress-only documentation
patch is pending.

---

## 2026-08-22 — exact 2,823 renderer / 465 runtime union-v2 milestone hardware-proven

The current shared code milestone is commit
`f3a0f590ab6809fccf7ca7dd77bf4544e714fa66` (`WIP rockchip: readable cleanup to
2.823k`), parent `eec67078bb5faf8f47b1d28e37e51ae0d4342fb5`, tree
`dcfe3ad7d10b755467100c64fd1771d12a5c061c`. Its renderer/runtime source blobs
are `f478b7be371b99d9e8b33814bba9a45954591006` and
`6d959bcea8b181faa67e74dadf20de50e91dcd3d`, with SHA-256
`86948c9a1302c0a657921413ea9646d5041f688a20e078cdfddb6082be033b74` and
`7aa9947a65c899132fc0e5fc0ce10b7c4ce4e5817792ca0bc7ab5f9ea5e28b85`.
`sz.py` reports renderer/runtime **2,823/465**. The exact isolated union
source candidate is commit `1318b3d1244fddd9a4cd43d75053aa1a2b9c4d52`,
parent `444561b3c30c37e2201fd512e5040fe9627e3368`, tree
`9c6e8f0097b112ed4cb39756fcccfdbce0d25404`; its two source blobs are
byte-identical to shared `f3a`. The union patch SHA-256 is
`d8dc9733e9651a1778d3ab5c1d0490f4e8894659d1abf9f567c091f46b77c23e`.
The component savings are WHERE-v2 **-2 renderer**, buffer-binding-v1
**-4 runtime**, and scheduler-v4 **-1 runtime**, for **-7 combined** from
base444 (renderer **-2**, runtime **-5**). The tracked shared tree is clean;
the five pre-existing untracked files listed in the preceding milestone were
preserved and are not part of this documentation patch.

- The [second independent union-v2 verdict](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4-independent-judge-v2/VERDICT-independent-judge-v2.md)
  records **VERIFIED WITH CAVEATS**: exact three-component provenance and
  replay, no test/config weakening, no added CPU/GPU/CMAC/host-numeric/LUT or
  other fallback route, and equal host semantic payloads. Its independent
  limits include WHERE **13,578** full cases, **978** adversarial cases,
  **216** coordinate/rebinding/lifetime cases, equal buffer exhaustive/state/
  sync probes, and equal scheduler v1/v2/v3/contract/order/oracle/fresh
  probes plus the 104-case cross oracle. Host gates are Rockchip UOps **115
  passed**, b79d **1 passed**, null UOps **62 passed/2 skipped/1 xfailed**,
  null stats **16 passed/11 skipped**, Ruff clean, mypy clean in **216 source
  files**, compileall clean, and backend collection **445**.
- The raw broad style monitor remains `NEEDS_CHANGES` with three heuristic
  findings, while the strict gate is PASS. The [manual style verdict](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4-independent-judge-v2/STYLE-VERDICT.md)
  classifies the two renderer findings as inherited standalone WHERE span
  attribution and the low scheduler `_run_ew_ops` finding as a lexical alias
  false positive: base/scheduler/final whole-function AST metrics are
  **22/36**, **21/35**, **21/35** BoolOps/joins, with `spatial` **3/8** in
  all three. This is a manual false-positive-aware **VERIFIED WITH CAVEATS**
  result, not a raw-monitor-clean PASS; a policy requiring raw broad PASS
  would reject it. The [provenance/no-cheat evidence](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4-independent-judge-v2/REPORT.md)
  records exact two-file scope, `git diff --check`, empty fallback findings,
  and no changed tests.
- The union manifest admitted only WHERE-v2, buffer-binding-v1, and
  scheduler-v4; dynamic-load-v2, math-v1, and LUT candidates were refused and
  are not part of this milestone.
- Hardware evidence reuses the single recorded post-reboot
  `simple_add.py` PASS at
  `/home/orangepi/rk-artifacts-next2790-20260822/health-post-reboot-wpi109`;
  `health/reused-from.txt` records `invocations_in_this_run=0`, so this
  documentation does not claim a health rerun. The one authoritative census
  artifact is
  `/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4/hardware-union-v2-1318b3d1244f-20260821T225646Z/`.
  Its fresh dedicated cache and serial command used
  `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`,
  `ROCKCHIP_SUBMIT_TIMEOUT_MS=6000`, `ROCKCHIP_SUBMIT_RETRIES=4`, `-n0`, and
  `-x`; collection and observation were **445**, with **433 passed + 12
  skipped = 445**, **154 subtests passed**, exit **0**, wrapper duration
  **1006.297 s**, and pytest duration **944.22 s**. Census stderr is empty.
  The marker scan emitted no `retry|timeout|resubmit|submit` text; four
  retries were configured, and marker absence does not prove zero attempts.
  Execution accounting is aggregate with collection names persisted; no
  per-node outcome manifest is present. See the [census result](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4/hardware-union-v2-1318b3d1244f-20260821T225646Z/RESULT.txt),
  [command](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4/hardware-union-v2-1318b3d1244f-20260821T225646Z/census/command.txt),
  [counts](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4/hardware-union-v2-1318b3d1244f-20260821T225646Z/census/counts.txt),
  [timings](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4/hardware-union-v2-1318b3d1244f-20260821T225646Z/census/timestamps.txt),
  and [marker scan](/home/orangepi/rk-artifacts-next2825-20260822/next-union-plan/union-v2-scheduler-v4/hardware-union-v2-1318b3d1244f-20260821T225646Z/census/retry-markers.txt).

No CPU/GPU/CMAC/host production numeric fallback or LUT route was added, and
no test was weakened. This documentation pass ran no tests, NPU command,
`elementwise.py`, rerun, reset, or push. Commit `f3a0f590a` is the shared code
milestone; this progress-only documentation patch is pending.

---

## 2026-08-22 — verified 2.820k Rockchip source milestone

The shared source checkpoint is commit `515cb6b2cf908e0a14b477dccb58db0f365c7026`,
parent `c473465726b932f06d420434d7a3340eefd3a7da`, tree
`24b76c436847edd859e0254c81a03203ad4d0fb3`.  The exact source reconstruction
is based on `f3a0f590ab6809fccf7ca7dd77bf4544e714fa66`; its combined patch is
`1f283e12f7abcb2273f2420490f5213d07a711ca7a124b40c5036c46334957ab`.
The reconstruction record is
`/home/orangepi/rk-artifacts-next2825-20260822/full445-2820-reconstruction.txt`.
Only `tinygrad/renderer/rockchip.py` and `tinygrad/runtime/ops_rockchip.py`
are source files in that patch.  The executable `sz.py` evidence is renderer
**2820**, runtime **459**, and repository total **28347** (the corresponding
size record is
`/home/orangepi/rk-artifacts-next2825-20260822/full445-2820-sz.txt`).  The
candidate source SHA-256 values are renderer
`d37bce7a51b95d78768e304a5e40c6c7b51236a0c5b9429d8c0f40fa396c39f4` and
runtime `870902e3e240689032cb5733165f235398c839422d189ec9e6e57f3813ad8e48`.

Host-only gates recorded by the corrected union-v2 report were Rockchip UOp
unit pack **115 passed**, focused b79d **1 passed**, PC-chain **32** normalized
cases (SHA-256
`6bc08cd68d6e5e2408e6937d501a8e7783cd5d7c40182f0108996e85de54bc47`), submit
oracle **144** cases with **0 mismatches**, and combined integration **12**
cases (SHA-256
`89256b63e1913453773f249105267f2418e82718de36445dc352946de7db16bf`), plus
Ruff, mypy, compileall, diff-check, and the exact-baseline ugliness check.
The authoritative host report and manifest are
`/home/orangepi/rk-artifacts-next2825-20260822/next-union-2820-salvage-rank13-rank14/REPORT-v2.md`
and
`/home/orangepi/rk-artifacts-next2825-20260822/next-union-2820-salvage-rank13-rank14/MANIFEST-v2.json`;
the report supersedes its stale-hash predecessor.

The one fresh serial hardware census is archived at
`/home/orangepi/rk-artifacts-next2825-20260822/full445-2820-union-v2-20260822T044657Z/`.
It ran `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` with a fresh cache,
`-n0 -x`, timeout 6000 ms, and four configured submit retries.  The result was
**433 passed + 12 skipped = 445**, **154 subtests passed**, exit **0**; pytest
reported **1006.32 s (0:16:46)** and the wrapper duration was **1062.885 s**.
The recorded boot ID was `3eb3630c-3147-4a2b-b726-e7e73167eea4`.  The marker
scan emitted no `retry`, `timeout`, `resubmit`, or `submit` markers; this is an
absence-of-marker observation, not a claim that zero retry attempts occurred.
The authoritative counts, timings, command, and marker files are
`census/summary.txt`, `census/timestamps.txt`, `census/command.txt`, and
`census/retry-markers.txt` below that directory.  The authoritative
`.venv/bin/python ~/rk3588/examples/simple_add.py` health result was reused
from the same boot and recorded `PASS`; it was not rerun for this census.

The provenance is one serial census, with no CPU/GPU/host numeric fallback,
CMAC shortcut, LUT route, weakened test, `elementwise.py` health run, reset,
or repeated census/duplicate experiment.  The device and source artifacts are
retained for audit; this progress entry does not change source or test files.

### Hardware LUT evidence, kept separate from the merge

The SiLU donor route is `/home/orangepi/rk3588/experimental/silu.py` (source
SHA-256 `ea5ddb90571470e19d4d9e74ef927e0fd1bc2e8d340d8424f847dc126c8a1f3f`)
and its independent evidence is in
`/home/orangepi/rk-artifacts-next2790-20260822/lut-v24-same-boot-20260822T112300Z/`.
The independent report is
`/home/orangepi/rk-artifacts-next2790-20260822/lut-v24-independent-judge-luna-max/REPORT.md`;
its SHA-256 is
`711aea7db73cf9103c8002d49553815b22702e565eeb00d86f5573d58f25a30c`.
It observed a real hardware LUT route with LUT bypass clear, 1026 signed
int16 table entries (513 per bank; table SHA-256
`2d75bf85cb5aa6df95b3a5b43ee13932f7a801e3bb80d4b8f2fa89d93445a932`), finite
lane-varying output, and donor-scaled maximum absolute error **0.0027546**.
The donor stream and runtime capture hashes were respectively
`4ecec9dc2ef116cf8ebb46c129ecddc50ec9c4e6141c30287eb1c490f865b696` and
`530ef5f217ae0ab45b3a840f33b1704dbf528928ec2f9524b0423756f8a4b2e0`.
The strict ideal raw-fixed-point oracle rejected the readback, so this is
hardware-proven donor-route evidence with that fixed-point model caveat, not
proof of Tinygrad integration.

The normalized EXP2 probe is archived at
`/home/orangepi/rk-artifacts-next2790-20260822/lut-exp2-v1131-hardware-20260822T122537/`.
Its command stream has SHA-256
`1b3aa43ac814f8ef570c230904e91483c47f8f341bc8fbd972dc7bdfa20e6c4d`, the LUT
table has SHA-256
`ad3ef028711e351f457bb1598b28abc2c21c3de99b21b12dd210e21680760dcc`, and the
validation record has SHA-256
`4334de26ef537b6f5f95defdd6110d8521cc81a98d117cfd338c582adb9c68f3`.
The probe recorded hardware-opened, LUT bypass clear, finite lane-varying
outputs within Tinygrad tolerance, one reset/submit, and the poisoned guard
check passing: bytes 256..4095 remained `0xa5` and changed guard bytes were
zero.  Its fail-closed human-review exit is not a hardware failure.

Neither LUT route is merged: the correct typed integration is line-positive
(the archival plan adds physical-stage machinery and does not remove the old
EW approximants).  The corrected future plan is
`/home/orangepi/rk-artifacts-next2825-20260822/lut-future-merge-v2-2820-459/worktree/FUTURE_LUT_MERGE.md`;
the corresponding patch is
`/home/orangepi/rk-artifacts-next2825-20260822/lut-future-merge-v2-2820-459/future-lut-merge-v2.patch`
(SHA-256 `d42abcb0f5e2ddc607a3a2bfed1724a66f05af557574952c81088acffa7f9e51`).
It is **ARCHIVAL_WIP_NOT_RUNTIME_INTEGRATED**, explicitly default-off, adds
997 physical lines (0 deletions; **+156 `sz.py` lines**), and has no
full-445/default-route/NPU integration proof.  It is pending an independent
verdict and is not merge-ready.  The v1
patch at
`/home/orangepi/rk-artifacts-next2825-20260822/lut-future-merge-2820/future-lut-merge.patch`
is archival/refuted merge metadata, not a ready patch.

### CMAC/WMMA research disposition

The read-only synthesis and provenance reports are in
`/home/orangepi/rk-artifacts-cmac-20260822/FINAL_CMAC_SYNTHESIS.md`,
`CMAC_AUDIT.md`, `cmac-git-archaeology.md`, `cmac-wmma-audit.md`,
`cmac-architecture-break-even.md`, `reduction-deletion-map.md`, and
`cmac-dedup-ledger.md`; their recorded hashes are in the synthesis report.
They inspect the existing `~/npu`, `~/rk3588`, and historical WMMA/CMAC
branches without repeating hardware or source experiments.  The conclusion
is **zero proven deletion** today.  A narrow future M1 contract is plausible:
`REDUCE(ADD, MUL(INDEX(A,k), INDEX(B,n*32+k)), k)` to the CNA → CORE/CMAC/CACC
→ DPU pipeline, with contiguous FP16 A, pre-packed FP16 B, **K=32**,
**N=4..16**, FP16 output, one reduction/store, affine zero-based indices, and
no casts, broadcasts, predicates, epilogues, K/N split, swizzle, multi-output,
or dynamic gather.  Broad reductions, generic WMMA, and the modular/OpenNPU
paths remain **NO-GO** because packing, CBUF layout, strides, output layout,
and precision semantics are not proven.  This is a future contract only; no
CMAC source, test, or device change is part of this milestone.

---

## 2026-08-22 — verified 2.811k renderer / 457 runtime source milestone

Source commit `be8bd962262664ca8a2de88836fe3e80d2a42d6f` (parent
`a9e1b162cab4b269a3ff3149999b8d1da96480a5`, tree
`fa2196271c55aa62fd982cd88177c183db877ec2`) applies only the canonical
two-file union patch (`1a87859a4606e51834b6c0145bb39488eaf02f9862e4cd7a83000ec8ba0d29ed`)
to base tree `612a59340eab4adf923c6eb170379c4ba989e4de`.  `sz.py` reports
renderer **2811**, runtime **457**, a **-11** combined-line delta from
2820/459; no test or runtime/core behavior was changed outside that patch.

The exact post-reboot hardware evidence is
`/home/orangepi/rk-artifacts-next2825-20260822/union-2811-457-postreboot-hardware-20260822/`:
`.venv/bin/python ~/rk3588/examples/simple_add.py` health **PASS** (exit 0),
and the `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP` census recorded
**433 passed + 12 skipped = 445**, **154 subtests**, exit **0**, in **916.87 s**.
The recorded marker scan observed zero `retry`, `timeout`, or `submit` lines
(and stderr was empty); this is an observed-marker statement, not a claim
that configured retry capacity was absent.  No CPU/GPU/host-numeric fallback,
CMAC shortcut, reset, or `elementwise.py` health run was used.

The hardware-proven LUT donor routes remain separate architectural evidence:
the typed integration is currently line-positive/default-off and is not merged.
CMAC archaeology still proves **zero deletion** today; only the narrow future
K=32, N=4..16 contiguous-FP16 contract is plausible, so no CMAC code is merged.
The raw-cast/byte-plane unification audit is **REJECTED**: numeric FP16
truncate/modulo and raw FP16 bitcast have different offset, lane, output-image,
and resource contracts, making the proposed shared route unsafe; no CAST change
is included.

## 2026-08-23 — CMAC experiment rejected after repeated post-health timeouts

The compact CMAC experiment is **REJECTED for integration**.  Multiple clean,
one-shot board gates produced the expected CMAC output (`lane 0 = 0x5000`) and
accepted the single blocking submit, but the required ordinary EW post-health
then timed out.  This repeated across the compact path, the sync-order A/B,
the v6-order lifecycle, and flags-1 variants; their external reports remain
archived under
`/home/orangepi/rk-cmac-lut-20260822/cmac-board-gate-luna-max/`.

The fresh-boot b62 gate is recorded in
`STRICT_GATE_B62FB25A_FAILURE_POSTHEALTH_AFTER_REBOOT_20260823.md`: boot ID
`8ec6177f-8f44-490c-8d66-f02a4be34ed7`, candidate
`b62fb25a6632f55cc2b35dcb0b7265a4b0a07bee`, pre-health **PASS**, exactly one
CMAC action-6/flags-5 submit with zero retries and lane `0x5000`, followed by
`simple_add.py` submit `TimeoutError: [Errno 110] Connection timed out`.
The preceding pre-health failure is retained in
`STRICT_GATE_B62FB25A_FAILURE_PREHEALTH_20260823.md`; no retry or recovery was
performed.  The production/test tree therefore remains the pre-CMAC
`910d3f9e5ff41957d8c3be3bf4cad7be6486ef8f` state.  CMAC commits, bundles,
reports, and LUT patches remain outside this source tree for provenance only;
no CPU/GPU numeric fallback or repair is admitted.

## 2026-08-23 — compact broad-CMAC line-cut candidate (hardware pending)

This isolated candidate starts from recovery parent `74093caf8` and is not a
shared-tree acceptance or a reversal of the timeout evidence above.  It
replaces the duplicated EW sum/dot specializers with one production
`to_program` CMAC route while retaining generic native EW fallback for
unsupported math, activations, extrema, predicates, dynamic addressing, and
mixed reductions.  Packing and output extraction use existing raw-lane
gathers; there is no CPU/GPU numeric reduction or matmul fallback.

The predeclared executable budget was at most **+206 additions**, at least
**-301 deletions**, and net at most **-95**.  Tokenized executable-line diff
against the parent is **+169/-319, net -150**: renderer **+150/-315** and
runtime **+19/-4**.  `sz.py` moves renderer **2820 -> 2655**, runtime
**457 -> 472**, and repository total **28345 -> 28195**.  The replacement
surface is `RKCMAC`, `_cmac_layout`, `_validate_cmac`, `emit_cmac_stage`,
`_lower_cmac_reduction`, and `RockchipProgram._run_cmac`.  It makes these nine
old functions obsolete and removes them:

- `_product_residual_ops`
- `_lower_dot_loop_reduction`
- `_lower_scalar_loop_reduction`
- `_finish_mapped_add_reduction`
- `_append_mapped_product_residual`
- `_lower_mapped_add_loop_reduction`
- `_lower_vectorized_unrolled_add_reduction`
- `_lower_vectorized_mul_add_reduction`
- `_lower_multi_scalar_local_reductions`

The matcher covers direct and loop `ADD` reductions, unrolled pure sums and
dots, dense matmul/matvec factorization, non-affine static K permutations,
paired batched dots through diagonal extraction, and FP16 or FP32 storage.
It emits one 45-qword CNA/CORE/DPU body with three relocations and a separately
owned four-qword PC tail.  It fails closed on invalid scratch extents, more
than one row above the donor's 384-channel CBUF boundary, excessive packing,
or unsupported semantics.  A CMAC timeout is never replayed.

Host evidence for this candidate:

- the emitted 45-qword body matched `/home/orangepi/rk3588/examples/gemm.py`
  exactly for `(M,N,K,out) = (1,4,32,fp16)`, `(3,4,5,fp16)`,
  `(8,16,32,fp16)`, `(1,128,128,fp32)`, and `(256,256,256,fp16)`;
- the production `to_program` path selects CMAC for a real `3x5 @ 5x4`
  Tensor matmul and a scalar half-to-FP32 Tensor sum;
- `python -m pytest test/unit/test_rockchip_uops.py -q -n12` reports
  **118 passed**;
- `python -m mypy tinygrad/` reports no issues in **216 files**;
- `python -m ruff check .` passes;
- the required Rockchip environment collects exactly **445 tests** without
  opening the device.

A private attempt to replay all collected cases through host compilation was
terminated when generic `logcumsumexp` expansion became pathological; it is
inconclusive and is not counted as a pass.  No hardware ioctl, census, or
`simple_add.py` health command was run because the board still requires a
fresh proven boot after the rejected CMAC poisoned-state evidence.  Acceptance
therefore remains pending a fresh-boot pre-health gate, the actual serial
445-case NPU census through production `to_program`, and post-health on the
same boot.  This milestone records a line-negative candidate and host proof,
not hardware correctness or merge readiness.

## 2026-08-23 — post-CMAC dead reduction cleanup

This isolated milestone follows `ff99ce6b0` and removes code made unreachable
by the compact CMAC replacement.  The predeclared and observed executable
budget is exactly **+2/-47, net -45**.  Renderer size moves **2655 -> 2610**
and repository total **28195 -> 28150**; runtime remains **472**.

The deleted private functions are `_eval_int`, `_reduction_store`,
`_reduction_input`, `_reduction_arena`, `_lower_mapped_uops`,
`_lower_post_image`, `_compensated_add`, and `_kahan_add`.  The unused
`_MAX_MAPPED_DOT_SCRATCH_BYTES` and
`_MIN_GENERIC_PRODUCT_RESIDUAL_TERMS` bounds are also removed.  No production
replacement was added: repository-wide symbol search found no caller for the
reduction helpers, and `_eval_int`'s sole unit-test use now calls `_eval_expr`
directly.  Generic native EW fallback and the production CMAC route are
unchanged.

Host verification reports **118 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, mypy success in **216 files**, Ruff
success, clean diff checks, and exactly **445 tests collected** in the required
Rockchip environment.  No device command or hardware claim is part of this
dead-code milestone; the fresh-boot CMAC acceptance gate above remains pending.

## 2026-08-23 — scaled pure sums through specialized CMAC parsing

This isolated milestone follows `367d539390c0d81001eeff23b5015a166708c4ae`.
Its predeclared executable budget was at most **+25 additions**, at least
**-35 deletions**, and net at most **-10**.  The observed renderer diff is
exactly **+25/-39, net -14**: renderer size moves **2610 -> 2596**, repository
total **28150 -> 28136**, and runtime remains **472**.

The general `RKLoopReduction`, `_loop_reduction_match`, and
`_reduction_post_parts` surfaces are removed.  Their CMAC-only replacement is
`_cmac_loop`, which extracts one additive local loop and its optional output
scale.  Pure sums may now use that scale as the existing packed FP16 CMAC B
weight when it is finite and represented exactly by FP16.  A scaled dot still
rejects CMAC and follows the existing native EW lowering; non-FP16-exact scales
also fail closed.  No generic reduction IR or CPU/GPU numeric fallback was
added.

The real production `to_program` path now routes an eight-element Tensor mean
to native CMAC `(M,N,K) = (1,1,8)` with constant weight bits `0x3000` (0.125),
no EW stage, and no host gather/scatter.  Direct comparison with the parent
shows byte-identical serialized images for the existing FP32 sum
(`8ffc4abe7b5d6ce966396b5a26094668fa7722d77687a7a0608fdbb2b274b2a9`)
and `3x5 @ 5x4` matmul
(`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`).

Host verification reports **120 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, census, reboot, or reset command was run: CMAC hardware acceptance is
still pending a proven fresh boot, pre-health, the serial production 445-case
census, and post-health on that same boot.

## 2026-08-24 — proven renderer bookkeeping cleanup reapplied

This isolated milestone follows `f2d6e56cf` and reapplies only the
still-relevant renderer subset of the historically hardware-proven
`be8bd9622` cleanup.  Current-state inspection found that its runtime changes
were already present, so the corrected pre-edit budget was exactly
**+11/-19, net -8** in the renderer alone.  The observed result matches that
budget: renderer size moves **2596 -> 2588**, repository total **28136 ->
28128**, and runtime remains **472**.

The manual dynamic-selector Cartesian-product and coordinate transpose are
replaced by `itertools.product` and `zip`; repeated zero/one mask gathers use
the existing `_RKBuilder.constant`; the two equivalent extrema affine
adjustments share one loop; and nonfinite `WHERE` arm selection no longer
keeps a redundant index temporary.  No matcher, execution class, numeric
fallback, tolerance, or runtime submit behavior changes.

Host verification reports **120 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, including raw execution checks for
multi-axis, external-gate, repeated-channel, 1001-candidate, and exact
total-fill dynamic selectors.  Mypy passes all **216 files**, repository-wide
Ruff passes, diff checks are clean, and the required Rockchip environment
collects exactly **445 tests**.  Production FP32-sum and `3x5 @ 5x4` CMAC
images remain byte-identical at SHA-256
`8ffc4abe7b5d6ce966396b5a26094668fa7722d77687a7a0608fdbb2b274b2a9`
and `064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.
No device command was run; the fresh-boot CMAC hardware bracket remains
pending.

## 2026-08-24 — broader CMAC scales and shared standalone submission

This isolated milestone follows `ebe5a518d`.  The predeclared renderer budget
was at most **+22/-26, net at most -4**, and the runtime budget was at most
**+8/-16, net -8**.  The observed executable diff is **+14/-19, net -5** in
the renderer and **+8/-16, net -8** in the runtime, for **+22/-35, net -13**
combined.  Renderer size moves **2588 -> 2583**, runtime **472 -> 464**, and
repository total **28128 -> 28115**.

`_cmac_scaled_root` replaces `_scaled_add_terms` and now gives local-loop,
direct `REDUCE`, and unrolled `ADD` pure sums one finite, FP16-exact constant
scale path, including nested outer scales.  Scaled dots continue to reject
CMAC and use native EW lowering.  `_run_cmac` now delegates buffer allocation,
descriptor setup, and submission to `_submit_standalone`; its 45-qword body,
four-qword tail, operation index, enable mask, reset, and no-retry behavior
remain distinct from the ordinary DPU standalone path.  No generic reduction
IR or CPU/GPU numeric fallback was added.

Host verification reports **121 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, including a nested scaled direct
reduction, scaled-dot rejection, and exact DPU/CMAC standalone descriptors and
tails.  Mypy passes all **216 files**, repository-wide Ruff passes, diff checks
are clean, and the required Rockchip environment collects exactly **445
tests**.  Production FP32-sum and `3x5 @ 5x4` CMAC images remain byte-identical
at SHA-256 `8ffc4abe7b5d6ce966396b5a26094668fa7722d77687a7a0608fdbb2b274b2a9`
and `064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.
No device, health, reset, reboot, or census command was run; the proven
fresh-boot hardware bracket remains pending.

## 2026-08-24 — native EW and CMAC normalization cleanup

This isolated milestone follows `80f6bb25d`.  Its predeclared and observed
renderer budget is exactly **+9/-16, net -7** executable lines.  Renderer size
moves **2583 -> 2576**, repository total **28115 -> 28108**, and runtime
remains **464**.

The one-use native EW register-field intermediates are replaced by the final
`_EW_CFG_COMMON`, special `_EW_CFG_*`, and ordinary `_EW_CFG` values.  A
separate load of the parent renderer confirms that all eleven resulting
register values are exactly identical.  `_cmac_scaled_root` now performs the
same repeated CAST/MUL normalization in one loop statement, while
`_cmac_loop` shares its short-circuit rejection guard.  No matcher admission,
wire value, CMAC shape, operation order, execution class, fallback, or runtime
behavior changes.

Host verification reports **123 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, including direct exact-register and
finite/nonfinite CMAC-scale contracts.  Mypy passes all **216 files**,
repository-wide Ruff passes, diff checks are clean, and the required Rockchip
environment collects exactly **445 tests**.  Production FP32-sum and
`3x5 @ 5x4` CMAC images remain byte-identical at SHA-256
`8ffc4abe7b5d6ce966396b5a26094668fa7722d77687a7a0608fdbb2b274b2a9`
and `064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.
No device, health, reset, reboot, or census command was run; the proven
fresh-boot hardware bracket remains pending.

## 2026-08-24 — immutable UOp analysis cleanup

This isolated milestone follows `23ab9311b`.  Its predeclared and observed
renderer budget is exactly **+13/-24, net -11** executable lines.  Renderer
size moves **2576 -> 2565**, repository total **28108 -> 28097**, and runtime
remains **464**.

`_index_ranges` replaces its mutable nested walker with the same first-seen
recursive range deduplication.  `_exact_int_range` now uses bounded memoization
on immutable UOps, making the per-`RKContext` `int_ranges` dictionary and its
threaded cache arguments obsolete.  A direct parent-versus-candidate oracle
matches all **18 range graphs** and **3002 generated integer graphs**, including
shared terms, casts, WHERE, complement, modulo, valid bounds, rejected ops, and
exception identity.  No admission, range bound, traversal order, image, wire
value, fallback, or runtime behavior changes.

Host verification reports **125 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, including first-seen range order,
RANGE-dependency exclusion, and exact integer-carrier bounds.  Mypy passes all
**216 files**, repository-wide Ruff passes, diff checks are clean, and the
required Rockchip environment collects exactly **445 tests**.  Production
FP32-sum and `3x5 @ 5x4` CMAC images remain byte-identical at SHA-256
`8ffc4abe7b5d6ce966396b5a26094668fa7722d77687a7a0608fdbb2b274b2a9`
and `064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.
No device, health, reset, reboot, or census command was run; the proven
fresh-boot hardware bracket remains pending.

## 2026-08-24 — shared static unrolling broadens CMAC coverage

This isolated milestone follows `0e7f6fe56`.  Its predeclared renderer budget
was at most **+18 additions**, at least **-30 deletions**, and net at most
**-12 executable source lines**.  The observed raw production diff is
**+19/-39, net -20**; one insertion/deletion pair is the retained docstring
update, leaving the executable source budget at **+18/-38**.  Authoritative
`sz.py` size moves renderer **2565 -> 2547** and repository total **28097 ->
28079** (**-18**); runtime remains **464**.

The existing bounded `_unroll_static_local` and `_unroll_static_reduces`
interpreters now expose structural ADD terms to `_lower_cmac_reduction` before
factorization.  CMAC requests `precise=False` only for this speculative parse;
an unsupported graph falls through and the generic EW path reruns the normal
precise reduction recipe.  This makes the bespoke one-axis `_cmac_loop` and
the separate axis-specific base/delta branch obsolete and removes them.  No
new reduction IR, runtime phase, codec field, command register, submit path,
or CPU/GPU numeric fallback was added.

The production `to_program` sweep covered **208** sum/mean schedules across
rank, axis, and `NOOPT` variants.  It found **14** intended new admissions:
multi-axis `NOOPT=1` sums and FP16-exact means now replace 5--22 EW stages with
one fixed CMAC.  For example, `(2,3,4).sum((0,2))` moves from nine EW stages to
CMAC `(M,N,K) = (3,1,8)`.  The other **194** schedules remain byte-identical.
A two-axis structural fixture proves both direct REDUCE and linked FP32 local
ADD forms produce the same `(2,1,6)` image and exact Cartesian source packing.
FP16 stateful locals, MAX/MUL reductions, activations, scaled dots, bias,
padding, dynamic addressing, and oversized graphs continue to use honest EW
fallback.

A separate parent-versus-candidate oracle kept all images byte-identical for
12 representative production programs (sum, mean, axis sum, dot, scaled-dot
fallback, ReLU fallback, matmul, einsum, plain/biased/padded convolution, and
avg-pool) and for the former one-axis direct/local CMAC fixtures.  Canonical
FP32-sum and `3x5 @ 5x4` hashes remain
`8ffc4abe7b5d6ce966396b5a26094668fa7722d77687a7a0608fdbb2b274b2a9` and
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.

Host verification reports **127 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, or hardware census command was run; fresh-boot CMAC
acceptance and its pre/post `simple_add.py` bracket remain pending.

## 2026-08-24 — remove one-caller scalar-local plumbing

This isolated milestone follows `995748f23`.  Its predeclared renderer budget
was at most **+19 additions**, at least **-36 deletions**, and net at most
**-17 executable source lines**.  The observed raw renderer diff is
**+16/-37, net -21**.  Authoritative `sz.py` size moves renderer **2547 ->
2530** and repository total **28079 -> 28062** (**-17**); runtime remains
**464**.

The fixed scalar-local extrema path no longer routes its sole call through
generic `_flat_static_ranges`, `_static_local_descriptor`, and
`_lower_static_local_child` wrappers.  Their exact constant-loop validation,
row-major flattening, child `to_program` lowering, host-address rejection, and
scratch materialization now remain visibly at the call site.  The sole
`_lower_composed_uops` caller likewise invokes `_lower_uop_program` directly
and preserves the same `RKPLAN_REJECT:composed_uops` failure.  No new IR,
admission rule, runtime path, wire field, CPU/GPU fallback, or test deletion
was introduced.

The first candidate also removed `_output_store`, but two admission-limit
tests intentionally use that helper as a direct production parser probe.  The
full corpus exposed the mistake; the helper and its production caller were
restored rather than changing those tests.  A whole-repository symbol audit
then confirmed that each of the four removed helpers had exactly one caller
and no test or runtime consumer.

A committed-parent/candidate lowering oracle observed **242 outcomes / 226
RKImages** across all Rockchip UOp tests.  Every admission result and encoded
image is byte-identical; both sides have aggregate SHA-256
`c62655755cab3cb37a04484aafc05aaf6e8a174b41d20bdf88bed9b5926d5a8b`.
Seven actual production `Tensor.schedule_linear -> to_program` programs
(argmax, bool cast, uint8 cast, sum, mean, multi-axis sum, and matmul) are also
byte-identical, with aggregate record SHA-256
`36bde9d0ded5569f6ac34882fe7a723b1994bcf34d613816f6db6a2f1a46def8`.

Host verification reports **127 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, or hardware census command was run; fresh-boot CMAC
acceptance and its pre/post `simple_add.py` bracket remain pending.

## 2026-08-24 — collapse single-owner dynamic planners

This isolated milestone follows `4d2153124`.  Its predeclared renderer budget
was at most **+42 additions**, at least **-52 deletions**, and net at most
**-10 executable source lines**.  The observed raw renderer diff is
**+40/-60, net -20**.  Authoritative `sz.py` size moves renderer **2530 ->
2516** and repository total **28062 -> 28048** (**-14**); runtime remains
**464**.

Three planner layers had exactly one production owner.  `_runtime_load_address`
now owns the former `_runtime_affine_index` proof before falling through to
the unchanged table-addressed contract.  `_lower_bounded_integer_predicate_coordinates`
now owns the fixed nonzero predicate, signed-INT16 encodability proof, and
native emitter formerly split through `_bounded_predicate_coordinate_plan`.
`_lower_dynamic_typed_load` now owns the exact bound/negative-normalization
proof formerly forwarded through `_bounded_dynamic_axes`.  These three
helpers are deleted; no generic IR, wire field, runtime path, CPU/GPU numeric
fallback, admission broadening, or test deletion was added.

The first two removals reached only **2521** lines, one short of the declared
target, because the inlined bodies shared more unchanged diff context than
predicted.  Rather than use formatting-only churn, the final candidate also
removed the third single-owner dynamic-axis helper named above.  Mypy then
exposed two inlined-local namespace collisions (`total` and `axis`); names
were separated without changing the graph or image, and the complete
verification was rerun.

A committed-parent/candidate lowering oracle again observed **242 outcomes /
226 RKImages** across all Rockchip UOp tests.  Every admission result and
encoded image remains byte-identical at aggregate SHA-256
`c62655755cab3cb37a04484aafc05aaf6e8a174b41d20bdf88bed9b5926d5a8b`.
Seven actual production `Tensor.schedule_linear -> to_program` programs
(argmax, bool cast, uint8 cast, sum, mean, multi-axis sum, and matmul) remain
byte-identical at aggregate record SHA-256
`36bde9d0ded5569f6ac34882fe7a723b1994bcf34d613816f6db6a2f1a46def8`.

Host verification reports **127 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, or hardware census command was run; fresh-boot CMAC
acceptance and its pre/post `simple_add.py` bracket remain pending.

## 2026-08-24 — specialize post-CMAC residual reductions at 2,500 lines

This isolated milestone follows `35018d59d`.  Its predeclared executable
split was **+15/-31, net -16**; the observed renderer diff is **+14/-30,
net -16** because the fixed batch-emitter replacement consumed one fewer
added and deleted executable line than estimated.  The net budget and target
remain exact: renderer size moves **2516 -> 2500**, repository total **28048
-> 28032**, and runtime remains **464**.

CMAC's existing ADD-reduction coverage had left `_reduce_arena` with one
owner and unused direct-output, FP32-output, INT32, callable-barrier, and
per-operation barrier controls.  `_reduce_rows` now owns its exact balanced
pairing and first dependent floating-stage barrier directly, while INT16
users retain the same flag-free ordering.  `_ew_ops` is again only the fixed
batch emitter its remaining six callers use.  The two scalar-extrema sites
own their fixed 64-byte-spaced MAX reductions directly, making
`_reduce_arena` and `_spaced_reduction` obsolete.

The final integer MAX reuses the original 256-byte spaced arena after its
floating maximum has been copied out, reducing that fixture from **27 to 26
scratch slots**.  An initial draft incorrectly tried to reuse the 64-byte
`best_values` buffer; a direct extent oracle caught the undersized four-row
destination before promotion.  The candidate was corrected to the dead
256-byte arena, and the unit test now asserts decoded gather/EW scratch
extents so that error cannot recur.

A committed-parent/candidate lowering oracle observed **242 outcomes / 226
RKImages** across all Rockchip UOp tests.  Exactly one image differs: the
intended scalar-extrema scratch reuse above, with the same five gathers, 73
EW stages, and seven mid-gathers.  The other **225 images are byte-identical**;
the combined parent/candidate record SHA-256 is
`773dfee70fa610ff6bb373132817110ca96e3c76f3be06cba98757fb07cefee4`.
A test-only raw physical executor checked **1,005** finite FP16 extrema cases,
including ties, negative values, and signed zero; parent and candidate both
return the same first-argmax result in every case.

Seven actual production `Tensor.schedule_linear -> to_program` programs
(argmax, bool cast, uint8 cast, sum, mean, multi-axis sum, and matmul) remain
byte-identical.  Their candidate aggregate SHA-256 is
`fa32f111b839151233799fdec4697b66f09fc3e06da0d495a19926f4039cda68`;
the `3x5 @ 5x4` CMAC image remains
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.

Host verification reports **127 passed** for
`test/unit/test_rockchip_uops.py -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No CMAC
admission, wire field, runtime submit behavior, CPU/GPU numeric fallback, or
test tolerance changed.  No device, health, reset, reboot, or hardware census
command was run; fresh-boot CMAC acceptance and its pre/post `simple_add.py`
bracket remain pending.

## 2026-08-24 — direct CMAC weight packing broadens reductions at 2,493 lines

This isolated milestone follows `3873e3e38`.  Its predeclared renderer budget
was **+24/-31 executable lines, net -7**.  The observed executable diff is
**+27/-34, net -7**: the multiple-constant semantic guard and final type
narrowing shifted three lines onto each side of the replacement without
changing the promised net reduction.  Authoritative `sz.py` size moves the
renderer **2500 -> 2493** and repository total **28032 -> 28025** (**-7**);
runtime remains **464**.

`_lower_cmac_reduction` now owns the former `_cmac_scaled_root` scale peeling
and parses one-source contraction terms with distinct compile-time weights.
FP16-exact finite weights are packed directly into the immutable CMAC B
surface through `RKGather.values`; pure sum and mean images therefore use
three scratch buffers and no constant scratch slot.  This deletes
`_cmac_scaled_root` and the uniform-weight constant materialization while
adding no generic IR, wire field, runtime path, or CPU/GPU numeric fallback.

Admission remains deliberately exact.  A one-source term may carry one
finite coefficient only when its value round-trips through FP16.  Multiple
constant factors, a per-term coefficient combined with an outer scale,
non-FP16-exact or nonfinite weights, and scaled two-dynamic-input dots all
retain the existing elementwise fallback.  This prevents reassociation such
as `(x*3)*(1/3)` from silently changing FP16 semantics.  Existing unscaled
dot and matmul contractions are unchanged.

The actual production `Tensor.schedule_linear -> to_program` path now lowers
`(source * arange(1, 6)).sum()` to one native `(1,1,5)` CMAC with packed
weights `1..5`, no EW stages, no host-address path, and no constants.  The
committed parent used 141 EW stages for that optimized graph; under
`NOOPT=1`, parent and candidate both retain the same 11-stage fallback.  A
64-case production sum/mean sweep reports **26 byte-identical programs** and
**38 packing-only changes**, with **zero new, lost, or other differences**.
All 38 changed cases materialize byte-identical CMAC A/B surfaces and output
maps; their aggregate physical-oracle SHA-256 is
`830861145bb6c1637d1908cad512e8a9e9cb58f746ad3a3b805b0edaf2cb544c`.

An instrumented committed-parent/candidate oracle observed **99 CMAC matcher
attempts** across the complete Rockchip UOp module: **88 byte-identical**, **9
packing-equivalent**, **2 intended new weighted admissions**, and **zero lost
or other outcomes**.  The new admissions are the direct weighted fixture and
the production weighted Tensor graph.  The unchanged `3x5 @ 5x4` CMAC image
remains SHA-256
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`;
the new direct-packed FP32 sum and weighted-sum images are respectively
`e3341d80e5b30aaf087a847a3da9ab1ec6288a563304b9693491624bde2a2788`
and `ed9102ae6d693c8b2f978efc0652ec114594fad8bd312669748cb75e855d088a`.

Host verification reports **128 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full census command was run
because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — collapse single-owner composite math plumbing at 2,478 lines

This isolated milestone follows `0f8d0e5ec`.  Its predeclared renderer budget
was **+25/-39 executable lines, net -14**.  The observed token-aware diff is
**+26/-41, net -15**: moving the recognizer boundary counted one additional
replacement on each side, while direct EXP2 scale ownership removed one more
old result line than estimated.  Authoritative `sz.py` size moves renderer
**2493 -> 2478** and repository total **28025 -> 28010** (**-15**); runtime
remains **464**.

Six helpers had exactly one production owner.  `_fold_atan` now owns the
former `_unit_ratio_source` normalization match.  `_fold_inverse_hyperbolic`
now owns the canonical LOG2 pattern, bounded asinh/acosh recipe, large-value
correction, and region selection formerly split among
`_hyperbolic_log_source`, `_hyperbolic_tail`, `_dpu_region`, and
`_dpu_inverse_hyperbolic`.  `_dpu_exp2` directly owns the exact exponent scale
formerly forwarded through `_dpu_pow2_integer`.  All six helpers are deleted;
their explanatory docstrings remain beside the replacement logic as comments.
No generic IR, new arithmetic recipe, admission change, wire/runtime path,
CPU/GPU numeric fallback, test change, or CMAC behavior change was introduced.

A committed-parent/candidate lowering oracle observed **242 outcomes / 226
RKImages** across the complete Rockchip UOp module.  Every admission result
and every encoded image is byte-identical; the ordered record SHA-256 is
`2f05709ca821f92667ada10d28698473d5516782e7b2a114c249e75f3c4aee95`.
Four actual production `Tensor.schedule_linear -> to_program` programs for
atan, asinh, acosh, and EXP2 retain their individual image hashes and aggregate
SHA-256 `9a4faa6a5eeda465059c23d5c612fe9b31d71bc05b9929050e0d45963ec6ac0e`.

Host verification reports **128 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full census command was run
because the board remains untrusted; fresh-boot acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — centralize RKContext physical records at 2,336 lines

This isolated milestone follows `c3ee8b34e`.  The predeclared renderer budget
was **+8/-15 executable lines, net -7**, including a line-neutral signed-scale
CMAC experiment.  A bit-exact arithmetic probe found that distributing a
negative binary outer scale changes signed zero at the accumulation boundary,
so that admission and its test edit were rejected before integration.  The
final behavior-preserving renderer budget is **+7/-14 executable lines, net
-7**; its raw diff is **+10/-14** because three explanatory comments are
retained.  Authoritative `sz.py` size moves renderer **2343 -> 2336** and
repository total **27855 -> 27848**; runtime remains **444**.

`RKContext.__init__` now groups the related typed empty containers which share
one per-context lifetime.  `RKContext._widen_int16` owns the existing INT32-
backed UINT-to-INT representation change, replacing its separate `lower`
branch.  `RKContext.lower` also shares INDEX/LOAD dispatch and uses immutable
record replacement for physical-only cast retyping.  Existing comments and
all semantic checks remain.  No CMAC admission, arithmetic recipe, image
field, wire encoding, runtime path, scheduler hook, CPU/GPU numeric fallback,
tolerance, or test changed.

A committed-parent/candidate lowering oracle observed **282 outcomes / 266
images / 41 CMAC images / 16 rejections** across all 146 Rockchip UOp tests.
Every admission, encoded image, and resource record is byte-identical; both
normalized records hash to
`5c6a6f9adcc2780a8d80fed7045dcfd3a12d5e880490462455b606e037800080`.
Actual production `Tensor.schedule_linear -> to_program` plain and ReLU matmul
images, plus their emitted 45-qword CMAC stages, remain byte-identical at
aggregate record SHA-256
`c285d5c531223c832f8925b8c768a484fa82ba57d2a589683533d21b9869edb7`.

Host verification reports **146 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  The Fable
Judge verdict is **VERIFIED WITH CAVEATS**, with the live hardware bracket as
its sole caveat.  The user has now authorized the serial 445-case run on the
new boot; its pre-health -> census -> post-health result is pending this commit.

## 2026-08-24 — immutable lowering records share physical ownership at 2,343 lines

This isolated milestone follows `023b6869f`.  Its predeclared and observed
production renderer budget is exactly **+13/-20 executable lines, net -7**.
Authoritative `sz.py` size moves renderer **2350 -> 2343** and repository
total **27862 -> 27855**; runtime remains **444**.

`_typed_load_plan` now makes `RKGather` the sole owner of an explicit offset
map when a consumer requires one.  This deletes `RKTypedLoadPlan.offsets` and
the BOOL load consumer's second base/axis/offset remap while retaining the same
bounded gather and fill bits.  `RKStage`, `RKTypedLoadPlan`, and
`_RKStaticLocalDef` are now typed immutable tuple records, making their three
frozen-dataclass wrappers obsolete without changing their named fields or
constructor call sites.  A complete function-call profile first confirmed all
three records and their producers remain live; repository search found no
external users of these internal types.  No admission, arithmetic recipe,
image field, wire encoding, runtime path, scheduler hook, CPU/GPU numeric
fallback, tolerance, or test changed.

A committed-parent/candidate lowering oracle observed **282 outcomes / 266
images / 41 CMAC images / 16 rejections** across all 146 pre-existing Rockchip
UOp tests.  Every admission, encoded image, and resource record is
byte-identical; both normalized records hash to
`5c6a6f9adcc2780a8d80fed7045dcfd3a12d5e880490462455b606e037800080`.
Actual production `Tensor.schedule_linear -> to_program` plain and ReLU matmul
images, plus their emitted 45-qword CMAC stages, are also byte-identical at
aggregate record SHA-256
`c285d5c531223c832f8925b8c768a484fa82ba57d2a589683533d21b9869edb7`.

Host verification reports **146 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  The final
Fable Judge verdict is **VERIFIED WITH CAVEATS**, with the prohibited live
hardware bracket as its sole caveat.  No device, health, reset, reboot,
hardware execution, or full hardware census command was run because the board
remains untrusted; fresh-boot acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — large affine contractions keep compact CMAC plans at 2,350 lines

This isolated milestone follows `c0791e753`.  Its predeclared and observed
production renderer budget is exactly **+12/-14 executable lines, net -2**.
The raw renderer diff is **+13/-15** because it replaces one retained comment;
comments remain excluded by `sz.py`.  Authoritative size moves renderer
**2352 -> 2350** and repository total **27864 -> 27862**; runtime remains
**444**.

`_lower_cmac_reduction` now evaluates affine `RKGather` axes only for the
physical CMAC lanes it selects.  This makes CMAC's unconditional
`require_offsets=True` full-output materialization and copied lane-remap plans
obsolete.  Below the existing selector budget, explicit-offset and reordered
fallback behavior is unchanged.  Above it, only ungated canonical affine
loads are eligible, and exact row/column divisibility proofs replace eager
enumeration; nonaffine, gated, and reordered cases remain generic fallback.
The replacement stays inside the existing CMAC owner and adds no generic IR,
wire field, runtime path, scheduler hook, CPU/GPU numeric fallback, tolerance,
or CMAC-then-EW sequence.

Four actual production `Tensor.schedule_linear -> to_program` matmuls now use
one CMAC, two input gathers, one output gather, and zero EW stages:
`256x256 @ 256x256`, `192x256 @ 256x160`, `64x384 @ 384x384`, and
`512x128 @ 128x128`.  Their candidate `(M,N,K)` records and image SHA-256
values are respectively `(256,256,256)` /
`8f0e3b2eb38c290929f01062471ea51bedc234a73e083b5050d2209fcac46159`,
`(192,160,256)` /
`8055dc41228e81a9943cf89d63ea957e63e3a4eded5130637f4b8df6cb301b13`,
`(64,384,384)` /
`5808af4615b16b3afcd89e226ae81c957773b3de711127aab6be94dd50031206`,
and `(512,128,128)` /
`b9b707ee230b6982178ecb21678df5491401eefa20ad073e0e1c1c395b79b3e6`.
The square parent's generic image had **15,853 EW stages, 512 gathers, and
33,554,432 gather cells**; the candidate has zero EW stages, two gathers, and
131,072 input-gather cells.  Its encoding is 786,629 bytes versus the parent's
665,472 because compact planning still serializes the two final physical
offset surfaces, so this is a CMAC coverage, compiler-memory, and stage-count
result rather than an encoded-size claim.  The parameterized production
regression checks exact A/B packing, image roundtrip, and FP32 contraction
equality for all four shapes.

An independent affine-stability oracle checked **129,975** exhaustive and
randomized predicate cases / **1,098,576 records** and passed at SHA-256
`2512d17f3eedfc88c19b88661bf177852660a38014840908d659e2617b08e42f`.
A committed-parent/candidate lowering oracle observed **278 outcomes / 262
images / 37 CMAC images / 16 rejections** across all 142 pre-existing Rockchip
UOp tests.  Every admission, encoded image, and resource record is
byte-identical; both normalized record sets hash to
`cebf86989b8f2055af6cfd7f51b3532bab904a726fb44ec85e68779533f6b256`.

Host verification reports **146 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  The final
Fable Judge verdict is **VERIFIED WITH CAVEATS**, with the prohibited live
hardware bracket as its sole caveat.  No device, health, reset, reboot,
hardware execution, or full hardware census command was run because the board
remains untrusted; fresh-boot CMAC acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — binary outer scales broaden weighted CMAC at 2,352 lines

This isolated milestone follows `55710b1d5`.  Its initial renderer budget was
**+4/-7 executable lines, net -3**.  The safety-adjusted budget, declared
before the final correction, is **+6/-9, net -3** because explicit scale
provenance replaces two previously unchanged lines; the observed production
diff matches that revised budget exactly.  Authoritative `sz.py` size moves
renderer **2355 -> 2352** and repository total **27867 -> 27864**; runtime
remains **444**.

`_lower_cmac_reduction` now folds an outer scale into one-source CMAC weights
only when every factor still visible in the scheduled UOps is positive and a
power of two, every cumulative peeled scale remains exactly representable in
FP16, and every final term weight remains finite and FP16-exact.  This admits
real weighted binary-scaled reductions without crossing a floating-point
rounding boundary.  The same owner now constructs its two physical gather
surfaces directly, deleting the one-use `gather_surface`; the typed-load plan
assignment and rejection also share one line.  Existing comments and
docstrings remain.  No generic reduction IR, wire field, runtime path,
CPU/GPU numeric fallback, tolerance, or hardware retry was added.

Negative and non-binary visible factors, an unsafe net-one composite chain,
extreme binary chains whose intermediate scale leaves the exact carrier,
scaled two-source dots, and non-scale post-reduction operations remain EW.
The stronger guard also removes a legacy unsafe admission for an unweighted
sum scaled by `0.75`.  These checks necessarily apply to the production
scheduled UOps; factor provenance already canonicalized away by the scheduler
cannot be recovered in the renderer.

An actual production `Tensor.schedule_linear -> to_program` weighted sum
scaled by `0.5` moves from **142 EW stages / five gathers / 6,225 bytes**,
SHA-256
`b7d32fa45e2d9278085fc43875792d73de52b22605be4bf3466f6812ed209f8d`,
to `(1,1,5)` CMAC with two input gathers and one post-gather / 2,377 bytes,
SHA-256
`5e40284edd9bc4e6677940cb4f62f101d94e9dd0b11fc1ce7d75fcc7740c4f81`.
The unsafe unweighted `0.75` case moves from CMAC / 2,377 bytes, SHA-256
`7cdd244283c8f2b06e04d6b5a620ac25a48e75fc2a66916d7975a2dc9e0b20ba`,
to **82 EW stages / 3,737 bytes**, SHA-256
`3320dbce9384ff0a8fbc4174a4a6d53bf2bc763f02f0613cc4532f1f1ac6ef09`.
Unscaled weighted CMAC, weighted non-binary and negative fallbacks, scaled-dot
fallback, and `3x5 @ 5x4` CMAC remain byte-identical; the seven-case candidate
production record hashes to
`f908a98f5176dc92b5daadd819d251421a0ae6955f085c294c0f19f1dec12d9e`.

The semantic/packing oracle checks six admitted scales over **427,680 vectors
per scale** plus **1,000 physical packed vectors per scale**, as well as four
safe visible factor chains; its image-record SHA-256 remains
`1fe2c6caff337a5f0cca9afc4ec2bd9680fdaeae0488d3cd410a7149d916b9b2`.
It proves why the rejected visible chains cannot be reassociated: `(0.1,5)`,
`(0.1,10)`, large-then-small binary, and small-then-large binary chains differ
on respectively **53,173**, **53,173**, **251,413**, and **6,405** vectors;
unweighted `0.75` differs on **28,224**.  A second randomized oracle covers
**64 valid cumulative-scale chains x 50,000 vectors** with chain-set SHA-256
`f383c9db6b183e0849a77f1ee3ac9fbb4664d4af4768925702c9f7643d5ce499`.
A committed-parent/candidate oracle observes **270 lowering outcomes / 254
RKImages** across the complete pre-existing Rockchip UOp module with zero
changes; both compact record sets hash to
`a7c2754f9c258e96cc80005c7ae4847911eb9e237a2383edbb26e8c46b6df7c3`.

Host verification reports **142 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  Two
adversarial refutations kept unsafe drafts out of the commit; the final Fable
Judge verdict is **VERIFIED WITH CAVEATS**, with the prohibited live hardware
bracket as its sole caveat.  No device, health, reset, reboot, hardware
execution, or full hardware census command was run because the board remains
untrusted; fresh-boot CMAC acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — one owner finalizes EW command chains at 444 runtime lines

This isolated milestone follows `a4e8802d8`.  Its predeclared and observed
production runtime budget is exactly **+14/-24 executable lines, net -10**.
Authoritative `sz.py` size moves runtime **454 -> 444** and repository total
**27877 -> 27867**; the CMAC-focused renderer remains **2355**.

`RockchipProgram._run_ew_ops` now gives its local `flush` owner the repeated
work of submitting a pending PC chain, optionally resetting after a precision
transition, clearing the submitted bodies, and returning precision state to
neutral.  This replaces eight open-coded finalization sequences across
barriers, terminal FP32 output, INT16/INT32 transitions, conversion, compare,
and function exit.  The existing submit -> optional reset -> clear ordering is
preserved.  No command recipe, chain boundary, reset point, renderer/image
format, numeric path, CPU/GPU fallback, retry, timeout, or tolerance changes.

A permanent mixed-boundary regression covers six sequences: submit barriers,
INT16 -> INT32 -> conversion -> plain transitions, native precision returning
to ordinary FP16, generic INT32 conversion, compare isolation, and 17 terminal
FP32 stages split at the 16-stage cap.  It records every submission's command
body lengths, conversion boundary, standalone compare body, and reset order.
The broader parent/candidate oracle hashes exact command bodies as well as
those event boundaries; both sides have SHA-256
`92a8a0ba76b1afb0d4ea5866fbde516ad1c90489e440b000506d29bb16039ece`.

Four actual production `Tensor.schedule_linear -> to_program` graphs then run
their decoded EW images through the fake physical runtime: ABS, minimum,
comparison, and native INT16 add.  Parent and candidate preserve their image
SHA-256 values respectively as
`3f3c1465a2e4970839a8a15675d53f56e2f58afa793b36d08ac556b3f8c85533`,
`7230b73ee7b39133536d38615b32f316fd80ad49aaee468de3678845baab446e`,
`82ab9212bd028c8599086bd2cd4bf76ec6c37d4ceee0fd9fabc993dd71f4b2a2`,
and
`b8ebfd858fc68a140d028899259b20e848051d7acb6db26134208ffcdb263186`.
Their combined encoded-image and submit/reset schedule record is byte-identical
at SHA-256
`5e0ac0badc8e3350128919355d8f6fd4861a402e1be9212ee7419336edd90b59`.

Host verification reports **142 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  Fable
Judge reports **VERIFIED WITH CAVEATS**: all host claims reproduce, the test
change only adds strict transition assertions, and no weakened check, scope
creep, or in-tree debris is present; the sole caveat is the prohibited live
hardware bracket.  No device, health, reset, reboot, hardware execution, or full
hardware census command was run because the board remains untrusted;
fresh-boot acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — filter physical CMAC candidates before ranking at 2,355 lines

This isolated milestone follows `80a1d95f1`.  Its predeclared and observed
production renderer budget is exactly **+8/-9 executable lines, net -1**.
Authoritative `sz.py` size moves renderer **2356 -> 2355** and repository
total **27878 -> 27877**; runtime remains **454**.

`_lower_cmac_reduction` now places factored and diagonal layouts in one
candidate set, applies every existing CMAC physical limit to each layout, and
only then ranks the survivors.  This deletes the separate diagonal fallback,
late physical validation, and diagonal-only output-offset branch.  Diagonal
physical cells are expressed through the same `outputs` permutation as every
other layout.  No generic reduction IR, new wire field, runtime phase,
CPU/GPU numeric fallback, tolerance, or existing assertion is added or
weakened.

The old ordering selected `(M,N,K) = (2,1,385)` for a production two-row
FP32-accumulating sum and then rejected it because multi-row CMAC input is
limited to 384 lanes, even though `(1,2,385)` was already a valid later
factorization.  The actual production `Tensor.schedule_linear -> to_program`
path now keeps that valid layout: it has zero EW stages, two gathers, output
offsets `(0,1)`, and SHA-256
`e9da143a258722242eb29492d30a6008d1602c476b2e5d3fb88ae9a01aaa3fb4`.
The parent fallback had 386 EW stages, 385 gathers, and SHA-256
`0e0e2104177840ca1a14a5b3ca94e3fb2a255ceb286a8dad187f2a96ea86525a`.
The CMAC encoding is 54,285 bytes versus the fallback's 35,903 bytes, so this
is a coverage and source-deletion result rather than a serialized-size claim.

The permanent regression materializes both raw CMAC surfaces and reconstructs
the physical output, proving exact FP32 sums before HALF storage.  The
384-lane boundary remains byte-identical as `(2,1,384)`, output offsets
`(0,64)`, 27,853 bytes, and SHA-256
`313ab439c5f1192afc73957f42a91f064b89cc741de82853d4770320c60a5a4d`.
A committed-parent/candidate oracle reran the original 140-test source and
observed **268 lowering outcomes / 252 RKImages / 16 honest rejections**;
every pre-existing record is identical, with compact ordered-record SHA-256
`3b6aa3c20d1460b98af7ecbf00b9dba70759db2483984edcdcd6054f4cebf3fd`.
Production plain and ReLU matmuls also remain byte-identical at respective
SHA-256 values
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`
and
`ed0f42d71e8049f79e621b28f6699c5950203025a5990fbfa8334b0418cf89ea`.

Host verification reports **141 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  Fable
Judge reports **VERIFIED WITH CAVEATS**: every host claim reproduces, the
focused test exercises the production compiler path, and no weakened check,
scope creep, or analogous post-ranking CMAC validation remains; the sole
caveat is the prohibited live hardware bracket.  No device, health, reset,
reboot, hardware execution, or full hardware census command was run because
the board remains untrusted; fresh-boot CMAC acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — explicit EW tiling ownership makes CMAC room at 454 runtime lines

This isolated milestone follows `54cbbdf13`.  Its predeclared and observed
runtime budget is exactly **+9/-17 executable lines, net -8**.  Authoritative
`sz.py` size moves runtime **462 -> 454** and repository total **27886 ->
27878**; the CMAC-focused renderer remains **2356**.

`RockchipProgram._tile` now owns only the common count/offset loop and always
returns address-patched command bodies.  Its four production callers directly
own their scratch INT16, INT16-to-INT32, same-precision INT16/INT32, and live
stage flags.  This deletes the weak-reference dereferences, five opaque mode
arguments (`bits`, `live`, `convert`, `scratch`, and `patch`), internal flag
matrix, caller-side patch loops, and temporary conversion-stage list.  The
remaining `weakref` import is retained for the program-resource cache.  No
renderer, image format, CMAC admission, command recipe, numeric path,
CPU/GPU fallback, tolerance, or existing assertion changed.

A committed-parent/candidate runtime oracle exercises seven physical routes:
scratch INT16, INT16-to-INT32, native INT16, native INT32, plain live,
stateful live, and INT16-output live.  Every submission boundary, qword count,
reset count, and command-body byte is identical; the ordered oracle SHA-256 is
`bfd51e8757bbcfcaf6788b7734bc77f687f091f41a4b06845dcca9071fd64d28`.
The permanent regression pins all seven parent bodies.

The complete parent/candidate UOp oracle observes **268 lowering outcomes /
252 RKImages / 16 honest rejections**.  Every record and encoded image is
byte-identical at ordered SHA-256
`7e5064151f5ce3e607f2e1d90d90de979a01eeb770dee94039f01b5a3b736c0c`.
The actual production `Tensor.schedule_linear -> to_program` plain and ReLU
matmuls remain one `(3,4,5)` CMAC with zero EW stages and respective image
SHA-256 values
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`
and
`ed0f42d71e8049f79e621b28f6699c5950203025a5990fbfa8334b0418cf89ea`.

Host verification reports **140 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  Fable
Judge reports **VERIFIED WITH CAVEATS**: all host claims reproduce and no
weakened check, scope creep, or debris is present; the sole caveat is the
prohibited live hardware bracket.  No device, health, reset, reboot, hardware
execution, or full hardware census command was run because the board remains
untrusted; fresh-boot CMAC acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — terminal ReLU joins the fixed CMAC body at 2,356 lines

This isolated milestone follows `f3e911238`.  Its predeclared and observed
production budget is exactly **+23/-24 executable renderer lines, net -1** and
**+1/-3 executable runtime lines, net -2**.  Authoritative `sz.py` size moves
the renderer **2357 -> 2356**, runtime **464 -> 462**, and repository total
**27889 -> 27886** (**-3**).

`_lower_cmac_reduction` now recognizes a terminal zero-clamped contraction at
either the FP32 accumulator boundary or the canonical HALF storage/`WHERE`
boundary.  The existing `RKCMAC` record uses one previously unused bit in its
output-format byte to request ReLU; `emit_cmac_stage` changes only
`REG_DPU_BS_CFG` from the bypass value `0x53` to the ReLU value `0x12`.  The
body remains exactly 45 qwords with the same three relocations and the runtime
retains the same four-qword CMAC tail, reset, and no-retry submission.  There
is no second stage, new command stream, generic reduction IR, CPU/GPU numeric
fallback, host arithmetic, or relaxed tolerance.

The first finished draft failed its adversarial review: an ordered `WHERE`
representing `minimum(contraction, 0)` folds to an `Ops.MAX` carrier tagged
`_NATIVE_MIN`, which the draft mistook for ReLU.  The promoted recognizer now
requires an untagged MAX.  A dedicated contraction-boundary regression proves
the tagged minimum returns `None` from CMAC admission and retains its terminal
EW store, while the corresponding untagged ReLU still routes to CMAC.

The broader `_relu_operand` owns the ordered-`WHERE` normalization formerly
duplicated inside `_fold_relu_cap`, so that nested helper is deleted.  The
stale `_NATIVE_RAW_MIN` marker and its unreachable layout branch are also
removed, and the two-line `_run_cmac` wrapper is inlined at its sole runtime
caller while retaining its contract comment.  Existing plain CMAC records
keep format bit zero and therefore preserve their wire bytes and commands.

The actual production `Tensor.schedule_linear -> to_program` path for
`(3x5 @ 5x4).relu()` moves from **65,763 EW stages / 2,637,908 encoded bytes**
at SHA-256
`3c5d63a358adc386aea6e4095521504b52837e6a8590692e2feff770893a6081`
to one native CMAC `(M,N,K) = (3,4,5)`, zero EW stages, a 45-qword body, and
4,725 encoded bytes at SHA-256
`ed0f42d71e8049f79e621b28f6699c5950203025a5990fbfa8334b0418cf89ea`.
The raw packing oracle materializes both CMAC surfaces and the output gather,
then checks the ReLU FP32 matrix result before HALF storage.  Ordinary
`3x5 @ 5x4` remains byte-identical at
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`;
its 45-qword stage hash remains
`4c9a492c1c921bbb8d63511088aa95a7d0dc471ebccf511cdd4fc008e796b1bd`.

A committed-parent/candidate oracle ran the same original 137-test source on
both revisions and captured all nested production lowerings.  Both sides
record exactly **266 calls** with identical input records.  Exactly one output
changes: the intended FP32 activation fixture moves from 34 EW stages to ReLU
CMAC `(2,2,2)`; the other **265 lowering outcomes are byte-identical**.  The
parent passes 137/137 assertions; the candidate passes 136 and reaches the
single expected legacy assertion that explicitly demanded the obsolete EW
fallback.  The updated suite replaces that assertion with the CMAC contract.

Host verification reports **139 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  The promoted
renderer, runtime, and unit-test SHA-256 values are respectively
`737d1f2479e18011a2b7420b052ae830c592d526489d30eb91f54d7d96e9c6bb`,
`1663c66ee99c858cbae9f7eceecfa77eecf8392031b0717bbdb9eac22a1baa24`,
and `86d1772da78c6872fd2abeb68c6715cf53871da129e1ecdaf9c1c386a9466e87`,
exactly matching the fully tested private prototype.  No device, health,
reset, reboot, hardware execution, or full census command was run because the
board remains untrusted; fresh-boot acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — retire obsolete expanded ABS/MINIMUM folds at 2,357 lines

This isolated milestone follows `eab77f7d0`.  The initially declared renderer
budget was **+2/-23 executable lines, net -21**.  Ruff then exposed that the
three-line `exact_static_selection` proof had no owner after `_fold_minimum`
was removed; the revised and observed budget is exactly **+2/-26, net -24**.
Authoritative `sz.py` size moves renderer **2381 -> 2357** and repository total
**27913 -> 27889**; runtime remains **464**.

`_fold_abs` and `_fold_minimum` recognized historical expanded MUL graphs.
Current production Tensor ABS and MINIMUM schedules already lower through
`RKContext._where`, `_raw_where`, and ordinary `_alu` stages, so both callbacks,
their storage-matcher entries, the guarded `_expand_math_uops` call, and its
now-dead static-selection proof are deleted.  The active `_fold_where_abs` and
`_fold_ordered_where` routes remain unchanged.  No new helper, IR, runtime
path, wire field, CPU/GPU numeric fallback, or test deletion is introduced.

A source-filename return profile across all **136** parent UOp tests invoked
each deleted matcher callback **451** times and observed zero non-`None`
returns.  The same run confirms the neighboring live callbacks are not dead:
`_fold_ordered_where` returns five replacements in 2,222 calls and
`_fold_where_abs` returns four in 1,219 calls.  A compile-only execution of the
actual census `test_minimum`, `test_abs`, and `test_abs_exact` methods builds
**15** Rockchip production images without touching the device; parent and
candidate images are byte-identical at aggregate SHA-256
`01b199e2a22f937d10a0f261c063d50cfdfbf5a0522b71906e4f8095b5797573`.

The new production regression pins the generic typed images directly.  ABS is
5,954 bytes / 123 EW stages / 16 mid-gathers at SHA-256
`3f3c1465a2e4970839a8a15675d53f56e2f58afa793b36d08ac556b3f8c85533`;
MINIMUM is 234 bytes / four EW stages at SHA-256
`7230b73ee7b39133536d38615b32f316fd80ad49aaee468de3678845baab446e`.
An immutable committed-parent/candidate oracle over the original full UOp
suite also preserves all **264 lowering outcomes / 248 RKImages** byte-for-byte
at aggregate SHA-256
`2c4e7c19af00c2c24da2f170a6a688385a2d79b75fd124f163371e810ac13c45`.
The promoted renderer and test SHA-256 values are
`d6f8301693ab480087294974e86339f3cfa9d252ffa054cad2a1ae92f525393e`
and `6ebd4277db4b36f2189160bef39a4be11bd434366aef26fbb01397b3a0217f97`,
exactly matching the fully tested isolated prototype.

Host verification reports **137 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full census command was run
because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — remove obsolete raw bitcast specialization at 2,381 lines

This isolated milestone follows `64bd2256e`.  Its predeclared and observed
renderer budget is exactly **+1/-18 executable lines, net -17**.  Authoritative
`sz.py` size moves renderer **2398 -> 2381** and repository total **27930 ->
27913**; runtime remains **464**.

The deleted `_lower_raw_fp16_bitcast` attempted to recover an emitted INT32
kernel that paired adjacent FP16 representation lanes.  Current production
`Tensor.bitcast` owns that operation as a zero-kernel movement, as required by
the existing hardware census test's zero-submit contract.  Raw FP16 sign and
payload arithmetic that does emit UOps already lowers through
`_lower_uop_program` and `RKContext._raw`.  The old two-specializer dispatch
loop is therefore replaced by a direct call to the still-live
`_lower_fp16_uint8_cast`; no replacement helper, IR, runtime path, wire field,
CPU/GPU numeric fallback, or test deletion is added.

A return-profile over the original complete Rockchip UOp suite called the raw
bitcast specializer **93** times and observed **zero** non-`None` admissions.
The actual production `Tensor.schedule_linear` probe for a `2x3x4` FP16 tensor
bitcast to INT32 observes zero scheduled kernels.  Its new regression sits
beside the existing emitted-UOp raw-sign/payload test, which continues through
the generic typed physical path.

An immutable committed-parent/candidate lowering oracle records every nested
`_lower_uop_program` return across the original suite: both sides produce
exactly **264 outcomes / 248 RKImages**, byte-identical at aggregate SHA-256
`2c4e7c19af00c2c24da2f170a6a688385a2d79b75fd124f163371e810ac13c45`.
The promoted renderer and test SHA-256 values
`45016665e16546a7a8575bee01702b19ea9366f22fbc4a4e6d4a4c35cf0e0173`
and `9105fda4fb013ff8b2d573aec57c8f71a38c5f01633c68f070941a14a35f911e`
exactly match the fully tested isolated prototype.

Host verification reports **136 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full census command was run
because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — nested weights and broad linear grids share CMAC planning at 2,398 lines

This isolated milestone follows `1590f3038`.  Its predeclared and observed
renderer budget is exactly **+9/-15 executable lines, net -6**.  Authoritative
`sz.py` size moves renderer **2404 -> 2398** and repository total **27936 ->
27930**; runtime remains **464**.

`_lower_cmac_reduction` now folds at most two finite FP16-exact constant
factors inside one FP32 linear term, replacing the former one-constant-only
rejection.  A term with no dynamic load is normalized as `1 * bias`, so one
literal additive constant shares the existing packed CMAC surfaces.  The
common M/N candidate search also replaces the all-linear `M=rows,N=1` special
path.  It explicitly prefers that old layout whenever it remains legal, but
may use another proved affine factorization when the old M or scratch bound
rejects.  No new helper, generic IR, image field, runtime phase, CMAC-to-EW
sequence, or CPU/GPU numeric fallback is added.

Admission remains conservative at observable rounding boundaries.  Nested
constants are limited to FP32 terms, each constant must round-trip through
FP16, an outer scale cannot also be distributed into a weighted term, and
scaled two-dynamic-input products remain generic.  More than two constants,
non-FP16-exact or nonfinite factors, and every nested HALF multiply retain the
EW fallback.  The HALF regression uses the concrete counterexample
`(-9.8109245e-05 * -53248) * -0.03955078125`, whose sequential result is
`-0.20654296875` rather than the folded `-0.2066650390625`.

The actual production `Tensor.schedule_linear -> to_program` path for a
`64x64` broadcast FP32 load plus literal bias `3`, stored as HALF, moves from
one EW stage to one native CMAC `(M,N,K) = (128,32,2)` with zero EW stages,
three gathers, and SHA-256
`48d3aeeae296d06ca6332be1ce5126d842b99960f617b666048c766771cb983a`.
Its encoding grows from 143 to 43,238 bytes, so this is a coverage and
source-deletion result rather than a serialized-size improvement.  A raw
surface oracle reconstructs all **4,096** outputs and exactly matches the
broadcast-plus-bias values.  A nested weighted sum moves from **113 EW stages
/ 4,948 bytes** to CMAC `(1,1,4)` / 2,377 bytes with SHA-256
`1557a5417de2b17922c4cfa065fda0a84187cfb720d23a55c31af2b1f9a72fff`.

An exhaustive-input generated oracle checked **512** deterministic valid
FP16 constant pairs against all **65,536** FP16 input bit patterns: all
**33,554,432** nested-FP32 results equal their folded CMAC weights, with
ordered result SHA-256
`e278d15eb7662c6e5bd1ced26c4e1f1d79f63bb36c610ff87fa2505dc903a322`.
An immutable committed-parent/candidate oracle observed **258** lowering
outcomes: **256 are byte-identical**, and exactly the nested-weight fixture
and broad literal-bias production graph are new CMAC admissions.  The
unchanged-record SHA-256 is
`87840d840a977f80564bb87d0b030d77d95b3156b2da886f9e1652bcb67d89c0`.
The existing production `3x5 @ 5x4` image remains byte-identical at
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.

Host verification reports **135 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full census command was run
because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — output-axis packing moves batched contractions to CMAC at 2,404 lines

This isolated milestone follows `3ff32b7cb`.  Its final predeclared and
observed renderer budget is exactly **+24/-32 executable lines, net -8**.
Authoritative `sz.py` size moves renderer **2412 -> 2404** and repository
total **27944 -> 27936**; runtime remains **464**.

`_lower_cmac_reduction.align` now accepts a physical lane order derived from
the existing `_affine_output_axes` proof.  When ordinary flattened-row
factoring fails, each proved output axis is considered as CMAC M while the
remaining axes form N.  Source offset plans are reordered only for packing,
and the inverse permutation is retained in the terminal post-gather so the
observable output remains in its original order.  Ordinary layouts have
explicit priority over reordered candidates, preserving all existing CMAC
images.  This adds no generic IR, semantic operation, runtime/wire path,
CMAC-to-EW sequence, or CPU/GPU numeric fallback.

The same closure removes stale genericity exposed by the new shared owner:
`_cmac_layout` no longer accepts its unused M argument,
`_static_vector_env` no longer accepts its unused count, and
`_typed_load_plan` drops the never-enabled `strip_cast` and
`require_default` branches.  Their shape/gather validation is retained in
the same functions with the same exceptions and bounds.

The actual production `Tensor.schedule_linear -> to_program` path for a
batch-2 `(2,4,9,9)` input and `(4,4,3,3)` weights with padding one changes
from **2,213 EW stages / 72 gathers / 175,688 encoded bytes** to one native
CMAC `(M,N,K) = (4,162,36)` with **zero EW stages / two gathers / 153,317
encoded bytes**.  The parent and candidate image SHA-256 values are
respectively
`6c1e9b38883f7f859ce9a194628bc272426cdcb5a55d8c7a723294c6f99a06a7`
and `caf362db3d071b2298166ce62e7686abd975927636cfd82c9d27faf9db95ab5a`.
The production regression materializes both packed surfaces, checks their
FP32 matrix product against direct padded convolution, verifies the inverse
NCHW post-gather map, and round-trips the native image codec.

A broader host-only oracle covers padding-one batches two, three, and four,
batch-three stride two, batch-two dilation two, and batch-two asymmetric
padding.  Their respective CMAC shapes are `(4,162,36)`, `(4,243,36)`,
`(4,324,36)`, `(4,75,36)`, `(4,162,36)`, and `(4,200,36)`, each with two
gathers and no EW stage.  All **150** deterministic small-integer physical
packing/mathematical cases pass; the ordered oracle SHA-256 is
`f25af3658dd7c97e2cb5e1d6a1f51395910d60e05b56f433d4021fc3e69abebc`.

A committed-parent/candidate oracle observed all **254 pre-existing lowering
outcomes**.  Every admission result and encoded image is byte-identical on
both sides at aggregate SHA-256
`c59e3fd1ddcbb592c0c28f390215a53c729ccaf89a6eb5fa09c09eb9d6e9b4ba`.

Host verification reports **134 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full hardware census command was
run because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — zero-gated padded contractions share CMAC typed loads at 2,412 lines

This isolated milestone follows `5f281daad`.  Its predeclared and observed
renderer budget is exactly **+10/-17 executable lines, net -7**.
Authoritative `sz.py` size moves renderer **2419 -> 2412** and repository
total **27951 -> 27944**; runtime remains **464**.

`_lower_cmac_reduction` now reuses `_typed_load_plan(...,
require_offsets=True)` for source typing, output-lane enumeration, gate
evaluation, argument ownership, and bounds validation.  This replaces its
private range environment, output-index evaluator, `CMACSource` record, LOAD
parser, offset evaluator, and manual source-bound check.  The selector-cell
cap is still charged before any offset table is materialized.  No generic IR,
wire field, runtime phase, retry/reset behavior, CPU/GPU numeric fallback, or
CMAC-to-EW sequence is added.

The shared plan admits only direct HALF loads or gated HALF loads whose
compile-time default is exact positive `+0.0`; `-0.0`, nonzero defaults,
unsupported gates, malformed loads, and out-of-bounds live lanes fail closed
to the existing EW path.  Zero-gated padding can therefore populate the
existing raw CMAC surfaces without a second address interpreter.

The actual production `Tensor.schedule_linear -> to_program` path for a
`(1,4,9,9)` input and `(4,4,3,3)` weights with padding one changes from
**2,213 EW stages / 72 gathers / 134,168 encoded bytes** to one native CMAC
`(M,N,K) = (4,81,36)` with **zero EW stages / two gathers / 39,893 encoded
bytes**.  The parent and candidate image SHA-256 values are respectively
`1a365110d969580267c10a5c7001684731de699eee956608009c15001c90c979`
and `1de6b9773377abed515eff87d7034cfd703877b73f03255fb4eb6a9ae40213f2`.
A physical packing oracle materializes both scratch surfaces and verifies
their FP32 matrix product exactly equals a direct padded convolution.

Host-only production classification also admits stride-two padding as
`(4,25,36)`, dilation-two padding as `(4,81,36)`, and asymmetric padding as
`(4,72,36)`, each with two gathers and no EW stages.  A **200-case** exact
small-integer packing/mathematical oracle covers those four families; its
ordered surface SHA-256 is
`d47cdcf88170866341df65d498d5f52a15983529990505284ec37f95faf4129d`.
Batched padded convolution remains an honest EW fallback.

A committed-parent/candidate oracle observed all **251 pre-existing lowering
outcomes**.  Every admission result and encoded image is byte-identical on
both sides at aggregate SHA-256
`c6b39952ebda64090a6a758db55ee039b0d0434b9715a1ff7caa1240e50ccae4`.
The new production regression separately checks raw packing arithmetic,
codec round-trip, native execution class, and `-0.0`/nonzero-default
rejection.

Host verification reports **133 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full hardware census command was
run because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — short FP32 contractions use compact CMAC records at 2,419 lines

This isolated milestone follows `7eeee5d97`.  Its predeclared and observed
renderer budget is exactly **+27/-34 executable lines, net -7**.
Authoritative `sz.py` size moves renderer **2426 -> 2419** and repository
total **27958 -> 27951**; runtime remains **464**.

`_lower_cmac_reduction` now keeps each term's source slots, already-bounded
output offsets, and coefficient in one record.  This replaces the parallel
`parsed`/`indexed`/`weights` state and the late all-terms offset expansion.
The selector-cell budget is charged immediately before each term's offsets
are materialized.  Fixed scratch-slot and one-use final image locals are also
removed.  No generic reduction IR, wire field, runtime phase, scheduler hook,
CPU/GPU numeric fallback, or CMAC-to-EW sequence is added.

A terminal FP32 ADD tree stored as either HALF or FLOAT is now recognized as
an additive boundary even when optimization has exposed only two or three
product terms.  The actual production
`Tensor.schedule_linear -> to_program` path therefore moves FP32-accumulating
K=2 and K=3 dots from **33** and **76** EW stages to one `(1,1,2)` and
`(1,1,3)` CMAC respectively for HALF output; direct FLOAT output moves from
**4** and **6** EW stages to the same CMAC shapes.  All four routes have zero
EW stages.  Their HALF-output candidate image SHA-256 values are
`73de5f6eece716c5db2c12fa6d518e4beb08c5c33a1d28f2e33560dfdc40bb93`
and
`ff73caac5f3d4a4af578bc04239b63e35729ddd244d3e84290afec624e6a1355`.
The FLOAT-output hashes are
`775a73b78e686d7a7fe6ce6cd412491073163dfd488c91335ef04c794ee2fdf4`
and
`fcf2ae38f30934cd755119d4889af4d211a461cca30ac99e82871859c13e7f21`.
All four fixed CMAC images are 4,425 bytes, versus respective parent encodings
of 1,586, 417, 3,446, and 579 bytes, so this is a coverage and stage-count
result rather than an encoded-size claim.  The existing production K=4 CMAC
remains byte-identical at SHA-256
`f5c8f042a3c11bbd2b52bcd460341da339485d3adfcfd39d06e3e2b478ec34ec`.

A raw packing/mathematical oracle checked **2,000** deterministic K=2/K=3
small-integer cases.  Every CMAC A/B surface contains the exact FP16 operands
and its FP32 matrix product equals the direct FP32 dot; the ordered surface
record SHA-256 is
`307e66cc4a7d47986d8c3af94827ffbbdeb47408dad110dc70ef0f8eaa9e3aaa`.
A committed-parent/candidate oracle observed **252 lowering outcomes / 236
image-bearing outcomes** across all pre-existing Rockchip UOp tests.  Exactly
the intended short FP32 fixture changes; the other **251 outcomes** are
byte-identical on both sides at ordered SHA-256
`68452a93903c2c6e594c1d749ac997ff2f9eacab6d5a912f3e06bfd914faf6e9`.
Scaled two-input dots, activations, nested post-reduction operations, invalid
weights, and ordinary HALF post-bias rounding boundaries remain honest EW
fallbacks.

Host verification reports **132 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full hardware census command was
run because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — broadcast linear terms share CMAC surfaces at 2,426 lines

This isolated milestone follows `7530041b5`.  Its final safety-adjusted
predeclared and observed renderer budget is exactly **+29/-36 executable
lines, net -7**.  Authoritative
`sz.py` size moves renderer **2433 -> 2426** and repository total **27965 ->
27958**; runtime remains **464**.

`_lower_cmac_reduction.align` and its nested `gather_surface` planner replace
the fixed one-source/two-source arity split, shared-affine-delta proof, and
separate constant-B versus dynamic-surface branches.  The matcher evaluates
each already-bounded compile-time output offset, then places row-stable terms
on CMAC A and column-stable terms on CMAC B.  Different source strides and
scalar, row, or column broadcasts therefore require no new IR.  A missing
operand denotes one finite FP16-exact coefficient packed into that surface, so
two-input products and one-input linear terms may share one terminal CMAC
accumulation.  Existing pure sums, weighted sums, dots, diagonal contractions,
matmuls, and multi-source surfaces retain their encoded images.
The selector-cell cap is checked before constructing the explicit offset
tuples as well as after physical gather expansion.

The actual production `Tensor.schedule_linear -> to_program` path now lowers
explicit FP32-accumulating `2x3 @ 3x2 + bias` graphs with column, row, and
scalar bias, followed by one final HALF cast, to native CMAC `(M,N,K) =
(2,2,4)` with four raw gathers, one output gather, and zero EW stages.  The
column-bias parent used 103 EW stages.  The respective CMAC image SHA-256 values
are `aeace0307420a55a9cd5bad87e1a9a19648e08e0b8b2b47896273e9c6fff00b3`,
`69daf96e9935d4fba475052c915e817c603f8f79afa6aeb99c3eecca82484c43`,
and `ff1bb777d89c5cb3c76b02a3d400b974920c77f7b77643290258d48562159379`.
Their encodings are 8,855, 6,935, and 6,935 bytes; the first is larger than the
parent's 4,715-byte encoding because composed surfaces retain explicit
raw-offset overlays, so this is a coverage and source-deletion result rather
than a serialized-size claim.  A raw packing oracle materializes all three
surface pairs and verifies that their FP32 matrix products exactly equal the
corresponding broadcast `lhs @ rhs + bias` values.

The ordinary HALF `matmul + bias` graph deliberately remains generic because
Tinygrad rounds the matmul before adding bias.  Parent and candidate retain the
same 10,728-stage image and SHA-256
`6782828d7c27dfea5e497794c31de7322a1cdbc435553bdbd048a23cb71d023e`;
no unsafe CMAC-then-EW sequence crosses that precision boundary.

A committed-parent/candidate lowering oracle observed **249 outcomes / 233
image-bearing outcomes** across all pre-existing Rockchip UOp tests.  Every
admission result, encoded image, and resource record is byte-identical at
aggregate SHA-256
`5427fa95d0050cb6efa84dc466e3e41b51ff1c6b537d4d0a4cfe20f4c6a8a7c9`.
The added production regression validates all three new admissions and exact
packed-surface arithmetic.

Host verification reports **131 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No runtime
or Tinygrad-core path, CPU/GPU numeric fallback, tolerance, or test contract
changed.  No device, health, reset, reboot, or hardware census command was run
because the board remains untrusted; fresh-boot CMAC acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — collapse immutable compiler plumbing at 2,433 lines

This isolated milestone follows `32425ff3c`.  Its predeclared executable
renderer budget was exactly **+45/-63, net -18**.  The observed raw renderer
diff is **+37/-55, net -18** because the common typed-output prefix and the
ready-node neutral rewrite each required fewer replacement lines on both sides
than budgeted.  Authoritative `sz.py` size moves renderer **2451 -> 2433** and
repository total **27983 -> 27965** (**-18**); runtime remains **464**.

Four existing owners now contain their remaining one-caller plumbing.
`_eval_expr` directly owns scalar and vector static ALU evaluation, making
`_static_alu` obsolete.  Immutable `_semantic_loads` uses one bounded cache,
deleting its optional caller-supplied cache and `_unroll_static_local`'s
threaded `local_cache`.  `_typed_half_image` shares the terminal FP16 result
prefix before its BOOL/INT32 and UINT8 ABI tails, while
`_finite_int_max_neutrals` assigns every ready node through one iterative
branch.  Existing comments and docstrings remain.  No admission, arithmetic
recipe, image field, wire encoding, runtime path, CMAC behavior, test, or
CPU/GPU numeric fallback changed.

A committed-parent/candidate oracle observed **249 lowering outcomes / 233
image-bearing outcomes** across the complete Rockchip UOp module.  Every
admission result, encoded image, and resource record is byte-identical; both
sides have ordered SHA-256
`a91ed527e713979a82329f0c993efb2d3c98bdcc5f0ca53b78bbc3ab15d3b276`.
A separate generated oracle covers all static division/modulo/bitwise modes
plus semantic-load order in **15 records**; parent and candidate both hash to
`7238953b0c3ea1a760cda45a855bda1473939d8c21a00ca436ed74b463d806ea`.

Six actual production `Tensor.schedule_linear -> to_program` kernels for
UINT8 conversion, BOOL conversion, weighted sum, maximum, mean, and matmul are
also byte-identical at aggregate SHA-256
`d78f750e6105ab18f509d8e6af802bc3f302cabc25bf7113e5b75704d03b4108`.
Their individual candidate image hashes are respectively
`bcf01b1015d95d27a817d78665d904f7aff232cfd5a23d062c2a2ee6a6b433c8`,
`fb55dbfb2151f7fdadca7d8af6d307d85c592f5f198344f4b4ddd202f44b7090`,
`ed9102ae6d693c8b2f978efc0652ec114594fad8bd312669748cb75e855d088a`,
`cbb8d1ad588865b34a0408f05b546ac58cbc60b2d2a80844cb445fbc8a21de5b`,
`9b2f550e74a6f09a06e6ea0ae1f6b15d0496e2d1d2301f8c7d0f917350c0022d`,
and `064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`.

Host verification reports **130 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full hardware census command was
run because the board remains untrusted; fresh-boot acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.

## 2026-08-24 — compose multi-source CMAC surfaces at 2,451 lines

This isolated milestone follows `be526ee1d`.  Its predeclared renderer budget
was exactly **+22/-29 executable lines, net -7**.  The final raw diff is
**+23/-30, net -7** because the post-draft safety audit replaced one existing
cell-cap line so every partial source surface is charged.  Authoritative
`sz.py` size moves renderer **2458 -> 2451** and repository total **27990 ->
27983** (**-7**); runtime remains **464**.

`_lower_cmac_reduction` now aligns contraction operands by their affine index
shape rather than requiring every term to use the first term's argument slots.
Each CMAC input surface may therefore be assembled from several argument
buffers: its first raw gather clears/fills the surface, and later gathers are
partial raw-lane overlays.  This makes the fixed `slots`/`params` admission and
the two single-source gather constructors obsolete.  The replacement remains
local to the existing CMAC owner; it adds no generic reduction IR, wire field,
runtime phase, scheduler hook, or CPU/GPU numeric fallback.

Admission remains tied to one arithmetic boundary.  Every term must have the
same one-source or two-source arity, all source-specific bounds are checked,
and the existing selector-cell cap now counts each materialized partial
surface.  Mixed product-plus-bias graphs retain their FP16 rounding boundary
and generic EW path; scaled two-dynamic-input dots, activations, and any
post-CMAC operation also remain fallback.  In particular, this change does not
introduce the unsafe CMAC-then-EW sequence implicated by the board timeout.

Two actual production `Tensor.schedule_linear -> to_program` graphs are new
admissions.  `(x.sum(float32)+y.sum(float32)).half()` moves from **286 EW
stages / 12,310 encoded bytes** to one `(1,1,16)` CMAC with three gathers /
2,538 bytes, SHA-256
`447a6625e44cab5790647245cdb5faed6156e57c76dff80021d5acbcdb02984d`.
The sum of two independent FP32 dot reductions moves from **973 EW stages /
40,608 bytes** to `(1,1,16)` CMAC with four gathers / 8,715 bytes, SHA-256
`60607911d9989916b00fd5056b320c2488a4e41e4818bd0b479bce52e3dbd061`.
A raw packing/mathematical oracle checked **200** exact small-integer cases
across those two routes.

A committed-parent/candidate oracle observed **244 lowering outcomes / 228
image-bearing outcomes** across the complete Rockchip UOp module: **226
existing images are byte-identical**, 16 rejections remain shared, and exactly
the two production fixtures above are new CMAC admissions.  The ordered record
SHA-256 is
`6ad70614f54e2be3d65f075f7e0d63f2c0d09da5127f58ac8aa83605b97a5256`.
The existing `3x5 @ 5x4` CMAC remains byte-identical at
`064e6704eb2a26182182aa86ef5ec63b18f986a8d4cf9a1df28682edddec2f8a`;
the mixed-bias fallback remains byte-identical at
`386294cf61807cc67ac73b5d375e8bac492b6f26b2a2dcf96f934d617cface8d`.

Host verification reports **130 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full hardware census command was
run because the board remains untrusted; fresh-boot CMAC acceptance and its
pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain
pending.

## 2026-08-24 — unify typed load ownership at 2,458 lines

This isolated milestone follows `1c8101a01`.  Its predeclared renderer budget
was **+63/-83 executable lines, net -20**.  The observed token-aware diff is
**+62/-82, net -20** because one preserved affine-address line aligned on
each side of the direct-owner rewrite.  Authoritative `sz.py` size moves
renderer **2478 -> 2458** and repository total **28010 -> 27990** (**-20**);
runtime remains **464**.

`RKContext._load` was the sole owner of five forwarding layers.  It now owns
nonconstant LOAD defaults and their partial gathers, affine/table runtime
address proof and host-address materialization, FP32 raw-load conversion, and
native BOOL/direct typed loads.  `_runtime_load_address`, `_load_default`,
`_load_runtime`, `_load_float`, and `_load_native` are deleted.  The runtime
address contract docstring remains beside its replacement logic as a comment.
No generic IR, new arithmetic, admission change, wire/runtime path, CPU/GPU
numeric fallback, test change, or CMAC behavior change was introduced.

A committed-parent/candidate lowering oracle observed **242 outcomes / 226
RKImages** across the complete Rockchip UOp module.  Every admission result
and every encoded image is byte-identical; the ordered record SHA-256 remains
`2f05709ca821f92667ada10d28698473d5516782e7b2a114c249e75f3c4aee95`.
Six actual production `Tensor.schedule_linear -> to_program` kernels covering
half ADD, FP32-to-half load conversion, INT32 ADD, BOOL NOT, BOOL WHERE, and a
dynamic indexed host gather are also byte-identical, with aggregate SHA-256
`50b38e142fdb904ab47ed258933172a132de9ed3d9fb19f0d5c89b72ca38f89c`.

Host verification reports **128 passed** for
`test/unit/test_rockchip_uops.py -x -q -n12`, mypy success in **216 files**,
repository-wide Ruff success, clean diff checks, and exactly **445 tests
collected** with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`.  No device,
health, reset, reboot, hardware execution, or full census command was run
because the board remains untrusted; fresh-boot acceptance and its pre/post
`.venv/bin/python ~/rk3588/examples/simple_add.py` bracket remain pending.
