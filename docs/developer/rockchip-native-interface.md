# Rockchip native operation interface

`RKNativeOp` is the option-C boundary for one physical CMAC or LUT operation.
It is a frozen, wire-first value embedded in `RKImage(version=32)`. The
encoded bytes, not a Python object identity or a resolver/store token, are the
cache identity.

## Producers

The future route modules expose one narrow shape:

```python
def try_cmac(uops: list[UOp]) -> RKNativeOp | None: ...
def try_lut(uops: list[UOp]) -> RKNativeOp | None: ...
```

`None` means that the exact route is not admitted and the normal renderer may
choose its existing fallback. A producer must return only immutable tuples,
exact `bytes` payloads, `RKArg` references, command qwords, and declarative
metadata. It must not return addresses, host numeric work, a mutable catalog,
or a second planner/segmenter.

Each relocation names a command word and its `(target, register)` pair and is
bound to one declared `RKArg`. Each asset carries its exact payload, SHA-256,
size, and upload ranges. Guards and repairs describe checks or exceptional
value policy; they do not execute those checks. `RKNativeTask`,
`RKNativeReset`, and `RKNativeSubmit` are the complete lifecycle controls.

## Runtime seam

`RockchipProgram` decodes and validates the complete image before allocating
scratch or touching an NPU buffer. Native dispatch currently performs a
host-only preflight (asset hashes, argument ownership, and relocation fields)
and then raises `RuntimeError("RK native execution effects are not implemented")`.
That deliberate fail-closed seam is the only place for the future
`physical_runtime_effects` implementation to attach.

The eventual effect order is: validate again immediately before effects,
upload embedded assets while the DPU is idle, reset, barrier-before, submit
the one task, barrier-after, and perform declared guard/readback checks. Any
unknown ownership, digest, relocation, control, or buffer bound must stop
before allocation, reset, upload, or submit. There is no EW fallback after a
native operation has been selected.

## Cache and mutation rule

`encode_image(decode_image(bytes))` is byte-exact. Native payloads contain all
asset bytes needed by the operation; a later resolver must not synthesize or
replace an omitted catalog entry. Equal immutable values remain valid, while a
payload, digest, qword, relocation, or descriptor mutation is rejected by
canonical validation before any device effect.
