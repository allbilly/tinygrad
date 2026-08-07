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

`test_medium_gemm` now `(17,17)@(17,17)`. `test_small_gemm` remains `(8,8)@(8,8)`.
