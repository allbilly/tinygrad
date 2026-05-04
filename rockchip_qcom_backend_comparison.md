# Rockchip And QCOM Backend Comparison

This compares `tinygrad/runtime/ops_rockchip.py` and `tinygrad/runtime/ops_qcom.py` as two small hardware backends in tinygrad.

Current size:

```text
413 tinygrad/runtime/ops_rockchip.py
412 tinygrad/runtime/ops_qcom.py
```

The files are now similar in line count, but they solve different backend problems.

## High-Level Shape

| Area | Rockchip | QCOM |
| --- | --- | --- |
| Device | RK3588 RKNPU through DRM RKNPU ioctls | Adreno through `/dev/kgsl-3d0` KGSL ioctls |
| Base class | `Compiled` | `HCQCompiled` |
| Program artifact | `RKTemplatePackage` serialized behind `b"RKTP"` magic | Adreno binary or IR3/NIR package |
| Launch model | Copy a prebuilt RKNPU register template, patch DMA addresses, submit RKNPU task descriptors | Build an HCQ command queue, bind args state, emit CP packets and registers, submit KGSL command |
| Memory model | Stage-1 CPU-backed tinygrad buffers plus temporary RKNPU GEM buffers per launch | Resident `HCQBuffer` allocator backed by GPU objects or mapped external memory |
| Queue abstraction | No HCQ queue for compute launch yet | Full `QCOMComputeQueue` with `exec`, `wait`, `signal`, barriers, timestamps |
| Compiler ownership | Renderer recognizes supported patterns and emits Rockchip templates | Renderer/compiler emits Adreno shader image and metadata |
| Unsupported programs | Compile/runtime reject missing `RKTemplatePackage` | Unsupported behavior is mostly compiler/renderer side or resource validation |

## Runtime Boundary

### Rockchip

Rockchip is a template-submit backend. The runtime expects compilation to produce a structured Rockchip package:

- register command qwords;
- task descriptors;
- patch table entries;
- family metadata such as `elementwise`, `conv1x1`, or `fused_matmul`.

At launch, `RockchipProgram` decodes the package, prepares temporary RKNPU buffers, patches runtime DMA addresses into the register template, calls `submit_template`, copies results back, and frees launch temporaries.

This is deliberately different from the older runtime-register-emitter shape. Shape policy, register building, packers, and template validation now live in `tinygrad/runtime/support/rockchip.py`.

### QCOM

QCOM is an HCQ packet backend. `QCOMProgram` parses a real GPU program artifact, uploads the shader image to GPU memory, computes launch metadata, and then uses `QCOMComputeQueue.exec` to emit Adreno command packets for each dispatch.

The command queue owns the per-launch packet stream:

- CP packet headers;
- shader state registers;
- constant/texture/UAV state loading;
- cache flushes;
- timeline signaling and waiting;
- KGSL command submission.

The runtime has a more normal resident-device shape: buffers are GPU objects, and launches bind those resident buffers through `QCOMArgsState`.

## Program Representation

Rockchip programs are not general shaders. A supported tinygrad graph is lowered to one of a small set of RKNPU template families. The package is closer to a hardware schedule or task recipe than to a kernel binary.

QCOM programs are shader binaries with metadata. They can come from QCOM CL rendering or IR3/NIR paths. The backend then binds buffers and dispatch dimensions to that shader.

This difference explains why Rockchip still has family-specific launch helpers for elementwise, conv, and fused matmul, while QCOM has one generic compute queue launch path.

## Memory And Data Movement

Rockchip currently uses CPU-backed tinygrad buffers:

1. tinygrad buffers are ordinary `memoryview(bytearray(...))`;
2. launch code allocates RKNPU GEM buffers;
3. inputs/weights are copied and packed into those GEM buffers;
4. the RKNPU task runs;
5. outputs are copied/unpacked back to CPU memory.

This is simple and practical for bring-up, but it is not a fully resident accelerator memory model.

QCOM uses `HCQBuffer` as the normal allocator result. Buffers are GPU objects or mapped external pointers. Copies synchronize with the device, but the steady-state launch path binds GPU addresses directly.

## Queue And Synchronization

QCOM is integrated with HCQ:

- `QCOMComputeQueue` emits hardware queues;
- `QCOMSignal` supports timeline waits and timestamps;
- memory barriers and cache flushes are queue methods;
- `QCOMProgram.__call__` mostly delegates to `HCQProgram.__call__`.

Rockchip does direct blocking RKNPU submit through `submit_template`. It resets the NPU each launch and uses optional RKNPU memory sync. There is no Rockchip compute queue abstraction yet, so timeline/profiling behavior is much thinner than QCOM.

## Hardware Command Ownership

Rockchip command ownership is split:

- `ops_rockchip.py` owns runtime binding and launch staging;
- `support/rockchip.py` owns register-template generation, patching, validation, and submit helpers;
- renderer methods choose which template family can represent the uops.

QCOM command ownership is more centralized in the runtime file:

- queue packet emission is in `QCOMComputeQueue`;
- argument packing is in `QCOMArgsState`;
- binary parsing and shader upload are in `QCOMProgram`;
- allocation and KGSL setup are in `QCOMDevice`.

Both are acceptable tinygrad shapes, but QCOM is closer to the HCQ ideal because it has resident buffers and a generic queue ABI.

## Strengths

### Rockchip

- Small runtime after moving register/layout policy to support code.
- Explicit template ABI with magic/version rejection.
- Easy to compare generated register templates against RK3588 reference scripts and captures.
- PC-chain task submission can be represented by multi-task templates.
- Practical for a device whose programming model is closer to fixed-function task streams than generic compute shaders.

### QCOM

- Standard HCQ backend structure.
- Resident GPU buffers and reusable uploaded programs.
- Generic dispatch path across many kernels.
- Strong queue-level synchronization, timestamp, and cache-control integration.
- Better fit for normal tinygrad program launch semantics.

## Weaknesses

### Rockchip

- Still uses CPU-backed tinygrad buffers and temporary GEM buffers per launch.
- Launch path remains family-specific for data layout and output decode.
- No compute queue abstraction yet.
- Reset-each-launch behavior is conservative and expensive.
- Unsupported graphs fail instead of falling back inside the backend, which is correct but limits coverage.

### QCOM

- More tightly coupled to Adreno packet/register details in the runtime file.
- Supports only the specific KGSL/Adreno generation path encoded here.
- Has substantial command state complexity despite the small file size.
- Kernel image and NIR/CL metadata parsing are harder to inspect than Rockchip templates.

## Practical Takeaway

Line count alone now says both backends are in the same class. Architecturally, QCOM is the more mature tinygrad runtime: resident memory, HCQ queues, generic dispatch, and timeline support.

Rockchip is now small enough, but it is a different kind of backend. Its current best shape is a compiled RKNPU-template backend: keep hardware register construction in support/compiler code, keep runtime launch to patching and submitting templates, and grow toward resident buffers and queue semantics only after the supported template families are stable.

The next meaningful Rockchip step is not making `ops_rockchip.py` shorter. It is removing temporary CPU staging from normal launches and deciding whether RKNPU submits should become an HCQ-style queue abstraction like QCOM.
