# Problem: Rockchip cna runner silently drops the `+ bias` post-op

## Status
Fixed. Reproduced with `DEV=ROCKCHIP TC=1` on `test/trace_gemm.py` (tencore+1
shape: `(M, K, N) = (2, 33, 17)`, see `tinygrad/codegen/opt/tc.py:143`) and
fixed by inserting an explicit Rockchip CNA region marker into the UOp stream.
The normal UOp loop handles that marker, runs CNA into the TC temp register
surface, skips the covered WMMA staging region, then resumes on the post-WMMA
tail.

## Symptom
`a.matmul(b) + 1` with `TC=1` (default) on the rockchip device returns
`a.matmul(b)`, not `a.matmul(b) + 1`. The mismatch is silent — there is no
warning, no fallback, no debug log line. The existing tolerance-based tests
hide it because the gap is a constant `1.0` (or whatever the bias is), which
falls inside `atol` for small random tensors and is masked by fp16 noise for
larger ones.

Reproduce (no other source change needed):

```
DEBUG=2 DEV=ROCKCHIP FORWARD_ONLY=1 TC=1 python test/trace_gemm.py
```

Observed on this machine:

```
RKCNA_RUN m=2 n=17 k=33 batch=1 out=fp16
result:   [[20.41 20.45 20.52 ... 21.30]
           [50.13 50.28 50.47 ... 52.78]]
expected: [[21.40 21.46 21.51 ... 22.30]
           [51.13 51.30 51.46 ... 53.77]]
max_diff=1.0189, max_rel=0.0468  -- "PASS" only because of the loose rtol
```

`result` is `A @ B`; `expected` is `A @ B + 1`. The +1 is missing.

## Root cause
`RockchipRenderer._coalesce_wmma_to_cna` (`tinygrad/runtime/ops_rockchip.py:829`)
coalesces **any** program that contains a `rockchip_cmac` WMMA atom into a
`rkcna_v1` program, regardless of what sits after the WMMA in the uops list:

```python
meta = {
  "version": 1, "m": 1, "n": dims[0], "k": dims[2], "batch": 1,
  "a_dtype": dtype_in, "b_dtype": dtype_in, "c_dtype": dtype_out,
  "acc_dtype": dtype_out, "out_dtype": dtype_out,
  "post_op": None,                                # <-- always None
  "atoms": len(wmmas), "dims": dims,
}
```

`post_op` is never set. The full uops list (including the trailing `Ops.ADD`
for the +1) is pickled as `prg[2]` and never re-executed — it is only used
as a JIT cache key.

`RockchipProgram.__call__` dispatches to `_run_cna_group` for cna programs,
which calls `_run_cna_matmul` (ops_rockchip.py:570). That function does
exactly one thing: pack A/B per the meta, emit CNA/CORE/DPU regs via
`_rk_make_gemm_regs`, submit one blocking RKNPU job, sync output, unpack.
No elementwise pass. No bias add. The DPU `EW_CFG` it emits
(`ops_rockchip.py:749`) has `EW_ALU_ALGO=0` and `EW_OP_CVT_BYPASS=0`, so
the DPU's elementwise stage is effectively a no-op pass-through.

The `boilerplate` function (ops_rockchip.py:89) does know how to emit an
`EW_ALU_ALGO=2` (ADD) configuration — it is the same code path the standalone
`+ bias` test (`test_tiny_add` in `test/test_rockchip.py:107`) would use
when it falls through to the python emulator. That code is just not wired
into the cna runner.

## Fix
`Ops.WMMA` is not individually executed inside the `while` loop. Instead, the
renderer prepends an `Ops.NOOP` marker with `("rkcna_region", define_reg, end)`
for compatible Rockchip WMMA staging regions. `RockchipProgram.__call__` sees
that marker inside the existing `while` loop, runs the grouped CNA matmul, writes
the result into the same `DEFINE_REG` surface that the TC UOps would have
populated, seeds params/constants, and jumps to the UOp after the staging `END`.
This keeps post-matmul semantics generic: `+1`, bias, and other supported tail
ALU are handled by the existing loop instead of by a growing `post_op` table.

Pure CNA matmuls with no post-WMMA float ALU still take the direct CNA path.
The scalarized tail `ADD`s are batched when consecutive, so EW gets a list of
elements per submit rather than one submit per scalar add.

## Scope
- Same bug applies to any post-matmul elementwise: `+ scalar`, `+ bias`,
  `* scalar`, `relu`, etc. The cna runner ignores all of them.
- `_match_direct_to_cna` (ops_rockchip.py:857) has the same shape: it
  accepts a direct matmul with an ADD in the uops (the matmul accumulation
  itself) and still doesn't propagate a `post_op`.
- The python emulator path (TC=0, or when neither coalesce matches) is
  correct — it walks the uops one by one and applies the trailing ADD. That
  is why `test_tiny_add` and the TC=0 run of `test/trace_gemm.py` pass.

## Fix options
1. **Extend the cna meta with a real `post_op`.** The renderer should scan
   the WMMA sinks for a single trailing elementwise op that can be fused
   (constant add, constant mul, relu, etc.) and set
   `meta["post_op"] = ("add_imm", 1.0)` etc. The runner emits the matching
   DPU `EW_CFG` (the `boilerplate` code is already there for `Ops.ADD`).
2. **Refuse to coalesce when a post-op is present.** Treat any non-WMMA,
   non-load/store op after the WMMA as a coalesce-fallback condition. The
   python emulator will then handle the +1 correctly. Simplest, but loses
   the perf win of a fused DPU pass.
3. **Hybrid.** Coalesce only when the post-op is one of a known-fusible set
   (add_imm, mul_imm, relu) and emit the corresponding DPU config; otherwise
   fall back to option 2.

Option 1 is the right long-term answer and matches what `plan.md` already
calls out (`post_op` is a reserved meta field). The diagnostic for "did
the fusion actually run?" is missing in all three options — we should add
a `RKCNA_POST_OP=add_imm value=1.0` line at `DEBUG >= 3` so this can't
regress silently again.

## Related observations
- The debug=7 uops list (`RockchipProgram.__call__` at ops_rockchip.py:323)
  is only printed for the python-emulator path. When the cna path runs,
  `self.uops` (the fallback uops) is set but never iterated. So the
  "is there an `Ops.ADD` after the `Ops.WMMA` in the uops list?" check
  is invisible to a user running the test as-is. Either move the debug=7
  print before the cna dispatch, or duplicate it in `_run_cna_group`.
- `test/trace_gemm.py` with `TC=0` is the right vehicle for inspecting the
  uops list (raw MUL/ADD per K iteration, trailing `+1` ADD before STORE).
  With `TC=1` the same test goes through the cna path and never shows the
  uops. The current tolerance in `trace_gemm.py` (`atol=2.0, rtol=0.10`) is
  loose enough to mask the missing +1 — tighten to `atol=0.5, rtol=0.02`
  to make this bug a hard test failure until it is fixed.
- **The python emulator in `ops_rockchip.py` has no `ROCKCHIP` branch in
  its `Ops.WMMA` handler.** Look at `__call__` at ops_rockchip.py:406: the
  `elif u.op is Ops.WMMA:` block dispatches on `device = u.arg[4]` and has
  `elif device == "METAL":`, `elif device == "AMD" ...` (three AMD
  variants), `elif device == "CUDA":` (with four `(M, N, K)` sub-branches),
  and a final `else: raise NotImplementedError(f"unimplemented tensor
  core {u.arg}")`. There is no `elif device == "ROCKCHIP":` branch. The
  rockchip_cmac atom (`dims=(16,1,32)`, `u.arg[4]=="ROCKCHIP"`) is defined
  in `tc.py:143` and is happily emitted into the uops list by
  `RockchipRenderer`, but the only thing that can execute that WMMA is
  the cna coalesce at `ops_rockchip.py:829`. Every other device in this
  dispatch table handles its own WMMA in the for loop; rockchip is the
  odd one out. Concretely: if `_coalesce_wmma_to_cna` returns `None` for
  a rockchip WMMA (e.g. `dims != (16, 1, 32)`, or a `rkcna_only` env
  flag forces a fallback, or the WMMAs are mixed), the python emulator
  will hit the bare `raise NotImplementedError` at line 462 and the
  whole process aborts. There is no graceful fallback. The right design
  is one of:
    - add an `elif device == "ROCKCHIP":` branch to the WMMA dispatch
      (rockchip has `threads=1`, so the emulator just runs
      `generic_wmma_helper` for `dims=(16,1,32)` — the layout is
      straightforward since the CMAC atom is one M row, 16 N cols, 32 K
      reduction, with `elements_per_thread=(32, 512, 16)` already
      advertising that all the work is per-thread), or
    - refuse to emit a `ROCKCHIP` WMMA into the uops list at all when
      the cna path will not pick it up, and just emit MUL/ADD for the
      python emulator (matches what `TC=0` already does).
  Either way, the current state — WMMA in the uops list, no python-
  emulator handler for it, silent dependence on the coalesce — is a
  landmine. Fixing it is a prerequisite for option 2 of the fix
  options above (refuse to coalesce when a post-op is present), because
  that fallback path needs somewhere to land.
