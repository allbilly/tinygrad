# Rockchip forward TestOps status

Last updated: 2026-08-01

## Meaning of this file

This status separates the frozen broad prototype from the clean `rockchip-2608` backend. A rejection is not counted as a pass. The clean runtime has no CPU semantic fallback, so unsupported cases fail during compilation with `RKPLAN_REJECT:unsupported_graph`.

Use:

```sh
. /home/orangepi/tinygrad/.venv/bin/activate
FORWARD_ONLY=1 DEFAULT_FLOAT=HALF ...
```

Gradients are intentionally out of scope until forward coverage is stable.

## Frozen prototype baseline

The last recorded broad run on the old implementation was:

| Status | Count |
|---|---:|
| PASS | 129 |
| FAIL | 287 |
| SKIP | 8 |
| Total | 424 |

The supplied root-cause table classified 176 failures: 94 unsupported ops, 22 unsupported dtypes, 42 numerical assertions, 6 timeouts, 6 uint8 overflow errors, 2 missing expected errors, 2 `NotImplementedError`, and 2 unsupported layouts. It did not account for the remaining 111 failed cases, so that table must not be presented as a complete partition.

That tally belongs to `rockchip-2607`; it is not evidence for the clean branch because the architecture and supported contract changed.

## Clean branch verified matrix

| Group | Host compile checks | RK3588 checks | Status |
|---|---:|---:|---|
| Native hook | 1 | — | PASS |
| RKImage codec/validation/relocation | 4 | — | PASS |
| DPU ADD/MUL/MAX/copy/fill and multistage liveness | included in compiler suite | 2 | PASS |
| Generated EXP2 LUT | exhaustive 32,770 FP16 encodings in domain | 1 | PASS |
| Direct affine CMAC matmul | included in compiler suite | 1 | PASS |
| Constant-backed CMAC row sum | included in compiler suite | 1 | PASS |
| Explicit-layout PPU global max | included in compiler suite | 1 | PASS |
| Clean compiler suite total | 18 after spatial-conv rejection test | 6 | PASS |

The host total is the collected total across `test/null/test_native_program.py`, `test/unit/test_rockchip_image.py`, and `test/unit/test_rockchip_compiler.py`. The device total is `test/device/test_rockchip.py`, run serially.

## Supported contracts

- dtype: FP16 only;
- mode: forward only;
- static shapes;
- DPU contiguous storage and one output;
- EXP2 exactly 128 elements for the first proven LUT task;
- CMAC K=32 with directly legal memory, no host gather or pack;
- CMAC output width in the proven 4–16 range used by current tests/recognizer;
- PPU global max only for explicit `(K,8)` HWC-compatible storage and legal kernel split.

## Expected rejects

- FP32, integer, uint8, bool, and gradient graphs;
- noncontiguous elementwise indexing;
- WHERE/comparison and fused epilogue families not expressible by current primitives;
- arbitrary EXP2 sizes or values requiring a different domain policy;
- unpacked/general matmul, batched matmul, arbitrary K, and host-required gather;
- spatial NCHW convolution without a device layout stage;
- windowed NCHW pooling and unsupported PPU kernel shapes;
- generic unsupported `Ops.*` graphs.

## Next low-hanging group

The best next group is typed DPU comparison/mask/WHERE support, not dtype emulation or spatial convolution:

- it reuses the existing expression DAG, scratch allocator, reset/dependency handling, and DPU emitter;
- it unlocks multiple high-level decompositions without adding named operator recipes;
- it does not require host layout transforms;
- it should be implemented as primitive compare/mask/select stages and verified against frozen command oracles.

After that, add one more generated primitive LUT such as LOG2 only if its exact hardware domain and error metadata are proven. Spatial convolution is not low-hanging because it first needs a device-visible layout contract.

## Rules for future tally updates

1. Record branch SHA, test command, and environment.
2. Separate PASS, honest compile rejection, numerical mismatch, device timeout, and harness error.
3. Ensure root-cause counts sum to the total failed count.
4. Never turn a reject into a pass with runtime NumPy or `run_host` unless upstream backend policy explicitly allows semantic CPU fallback. This clean branch intentionally does not.
5. Run device tests serially and rerun any timeout in isolation after reset before classifying it.
