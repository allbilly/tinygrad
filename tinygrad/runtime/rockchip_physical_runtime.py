"""Direct effects for one frozen ``RKNativeOp``.

The renderer owns the immutable wire record.  This module only binds its
canonical argument slots to caller buffers and uses the existing Rockchip
allocator, sync, reset, and submit primitives.  There is no resolver,
admission token, second planner, ioctl definition, or numeric fallback.
"""

from __future__ import annotations

import ctypes
import hashlib
import struct
from dataclasses import astuple, dataclass, field
from typing import NoReturn, Protocol, Sequence, cast

from tinygrad.renderer.rockchip import (
  RKArg,
  RKBufferKind,
  RKNativeAsset,
  RKNativeGuard,
  RKNativeKind,
  RKNativeOp,
  RKNativeRelocation,
  RKNativeRepair,
  RKNativeRepairKind,
  RKNativeSpan,
  RKNativeSpanKind,
  RKNativeReset,
  RKNativeSubmit,
  RKNativeTask,
  RK_EXP2_PHYSICAL_PROVENANCE,
  RK_EXP2_REPAIR_DEVICE_STAGE,
  RK_EXP2_REPAIR_METADATA,
)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.autogen import rockchip_physical as rkp
from tinygrad.runtime.support.hcq import HCQBuffer


_PAGE = 4096
_U32_MAX = (1 << 32) - 1
_TASK_DESCRIPTOR_BYTES = ctypes.sizeof(rk.struct_rknpu_task)
_EXP2_IDLE_UPLOAD_RANGES = ((0, 1026), (1026, 1026))
_CMAC_LOGICAL_LANES = 4

# Stable event IDs.  These are a finite vocabulary, not a second telemetry
# policy or an attempt counter encoded as an event name.
RK_PHYSICAL_TELEMETRY_IDS = {
  RKNativeKind.CMAC: (
    "cmac_v1.attempt",
    "cmac_v1.reset_action_6",
    "cmac_v1.barrier_before",
    "cmac_v1.submit_one",
    "cmac_v1.barrier_after",
    "cmac_v1.success",
  ),
  RKNativeKind.LUT: (
    "exp2_lut_v1.attempt",
    "exp2_lut_v1.asset_upload_idle",
    "exp2_lut_v1.reset_action_6",
    "exp2_lut_v1.barrier_before",
    "exp2_lut_v1.submit_one",
    "exp2_lut_v1.barrier_after",
    "exp2_lut_v1.success",
  ),
}


class PhysicalRuntimeReject(RuntimeError):
  """A frozen native operation or its call-time bindings failed preflight."""

  def __init__(self, reason: str, detail: str):
    super().__init__(f"{reason}: {detail}")
    self.reason, self.detail = reason, detail


class PhysicalOwnershipUnknown(RuntimeError):
  """The driver boundary was crossed without a provable completion result."""


class RKNativeOpLike(Protocol):
  kind: RKNativeKind
  commands: tuple[int, ...]
  relocs: tuple[object, ...]


class RKPhysicalDeviceLike(Protocol):
  native_reset_live_proven: bool

  def _gpu_alloc(self, size: int, flags: int = 0) -> HCQBuffer: ...
  def _gpu_free(self, buffer: HCQBuffer) -> None: ...
  def _sync_buffer(self, buffer: HCQBuffer, flags: int) -> None: ...
  def reset_npu(self) -> None: ...


class RKPhysicalProgramLike(Protocol):
  dev: RKPhysicalDeviceLike

  def _dma(self, buffer: HCQBuffer) -> int: ...
  def _submit_physical(self, cmd: HCQBuffer, task: HCQBuffer, contract: object) -> "PhysicalSubmitReceipt": ...


@dataclass(frozen=True)
class PhysicalSubmitReceipt:
  status: int
  ownership_known: bool
  submit_id: object | None


@dataclass
class PhysicalRuntimeTelemetry:
  route_id: str | None = None
  attempts: int = 0
  native_submit: int = 0
  native_submit_ids: list[object] = field(default_factory=list)
  host_numeric: int = 0
  gpu_numeric: int = 0
  asset_bytes: int = 0
  resets: int = 0
  pc_submits: int = 0
  reason_counts: dict[str, int] = field(default_factory=dict)
  events: list[str] = field(default_factory=list)

  def note(self, event: str) -> None:
    self.events.append(event)

  def reject(self, reason: str) -> None:
    self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1


def _read(buffer: HCQBuffer, size: int, offset: int = 0) -> bytes:
  allocation = getattr(getattr(buffer, "meta", None), "size", getattr(buffer, "size", None))
  if type(allocation) is not int or offset < 0 or size < 0 or offset + size > allocation:
    raise PhysicalRuntimeReject("readback", "read exceeds the declared allocation")
  return ctypes.string_at(int(buffer.va_addr) + offset, size)


def _write(buffer: HCQBuffer, data: bytes, offset: int = 0) -> None:
  allocation = getattr(getattr(buffer, "meta", None), "size", getattr(buffer, "size", None))
  if type(allocation) is not int or offset < 0 or offset + len(data) > allocation:
    raise PhysicalRuntimeReject("write", "write exceeds the declared allocation")
  ctypes.memmove(int(buffer.va_addr) + offset, data, len(data))


def _allocation_size(buffer: HCQBuffer) -> int:
  value = getattr(getattr(buffer, "meta", None), "size", getattr(buffer, "size", None))
  if type(value) is not int or value <= 0:
    raise PhysicalRuntimeReject("dma", "buffer allocation size is invalid")
  return value


def _dma(program: object, buffer: HCQBuffer) -> int:
  resolver = getattr(program, "_dma", None)
  if not callable(resolver):
    raise PhysicalRuntimeReject("dma", "program has no DMA resolver")
  value = resolver(buffer)
  if type(value) is not int or not 0 < value <= _U32_MAX or value & (_PAGE - 1):
    raise PhysicalRuntimeReject("dma", f"DMA address {value!r} is not an aligned uint32")
  if value + _allocation_size(buffer) > 1 << 32:
    raise PhysicalRuntimeReject("dma", "DMA allocation exceeds uint32")
  return value


def cmac_logical_output(raw_bytes: bytes, logical_lanes: int, swizzle: Sequence[int]) -> bytes:
  """Extract logical FP16 lanes from the raw 32-channel CMAC surface."""
  if len(raw_bytes) < rkp.CMAC_V1_OUTPUT_VIEW_BYTES or not 0 < logical_lanes <= 16:
    raise ValueError("invalid CMAC raw output image")
  indexes = tuple(swizzle)
  if len(indexes) != 32 or any(type(index) is not int or not 0 <= index < 64 for index in indexes):
    raise ValueError("invalid CMAC output swizzle")
  words = struct.unpack_from("<64H", raw_bytes)
  return struct.pack(f"<{logical_lanes}H", *(words[indexes[index]] for index in range(logical_lanes)))


class RockchipPhysicalEffects:
  """Execute exactly one frozen native operation with no retry or fallback."""

  def __init__(self, program: object, image: object, bufs: Sequence[HCQBuffer], native: RKNativeOp | None = None):
    self.program = cast(RKPhysicalProgramLike, program)
    self.image, self.bufs = image, tuple(bufs)
    candidate = native if native is not None else getattr(image, "native", None)
    if type(candidate) is not RKNativeOp:
      raise PhysicalRuntimeReject("native_schema", "RKImage has no RKNativeOp")
    self.native: RKNativeOp = candidate
    self._native_identity = self.native
    self.kind = self.native.kind
    self.telemetry = PhysicalRuntimeTelemetry("cmac_v1" if self.kind is RKNativeKind.CMAC else "exp2_lut_v1")
    self._cmd_buf: HCQBuffer | None = None
    self._task_buf: HCQBuffer | None = None
    self._resource_buffers: dict[str, HCQBuffer] = {}
    self._resource_args: dict[str, RKArg] = {}
    self._dma_by_arg: dict[RKArg, int] = {}
    self._binding_snapshot: tuple[tuple[RKArg, int, int, int, int, int], ...] = ()
    self._span_snapshot: tuple[RKNativeSpan, ...] = ()
    self._prepared = False
    self._reset = False
    self._in_flight = False
    self._unknown = False
    self._closed = False
    self._command_synced = False
    self.last_output: bytes | None = None
    self.last_logical_output: bytes | None = None
    self.last_reason: str | None = None

  @property
  def ownership_unknown(self) -> bool:
    return self._unknown

  @property
  def closed(self) -> bool:
    return self._closed

  @property
  def command_buffer(self) -> HCQBuffer | None:
    return self._cmd_buf

  @property
  def task_buffer(self) -> HCQBuffer | None:
    return self._task_buf

  def _event(self, suffix: str) -> None:
    self.telemetry.note(f"{'cmac_v1' if self.kind is RKNativeKind.CMAC else 'exp2_lut_v1'}.{suffix}")

  def _reject(self, reason: str, detail: str) -> NoReturn:
    self.last_reason = reason
    self.telemetry.reject(reason)
    raise PhysicalRuntimeReject(reason, detail)

  def _check_args(self) -> None:
    native = self.native
    expected_flags = 0 if self.kind is RKNativeKind.CMAC else rkp.LUT_V1_EXP2_REQUIRED_CONTROLS
    if type(native.flags) is not int or native.flags != expected_flags:
      self._reject("native_schema", "native controls differ from the route contract")
    if type(native.reads) is not tuple or type(native.writes) is not tuple or type(native.outputs) is not tuple:
      self._reject("resource_contract", "native resource lists are not immutable tuples")
    refs = native.reads + native.writes + native.outputs
    if any(
      type(arg) is not RKArg
      or type(arg.kind) is not RKBufferKind
      or arg.kind is not RKBufferKind.ARG
      or type(arg.index) is not int
      or not 0 <= arg.index < len(self.bufs)
      or type(arg.addend) is not int
      or arg.addend != 0
      for arg in refs
    ):
      self._reject("dma_binding", "native resource reference is not a bound zero-addend ARG")

  def _expected_contract(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if self.kind is RKNativeKind.CMAC:
      return rkp.CMAC_V1_COMMANDS, rkp.CMAC_V1_TAIL, rkp.CMAC_V1_TASK, rkp.CMAC_V1_SUBMIT, rkp.CMAC_V1_RESET
    if self.kind is RKNativeKind.LUT:
      return rkp.LUT_V1_EXP2_COMMANDS, (), rkp.LUT_V1_EXP2_TASK, rkp.LUT_V1_EXP2_SUBMIT, rkp.LUT_V1_EXP2_RESET
    self._reject("native_schema", "native kind is not CMAC or EXP2-LUT")

  def _check_command_contract(self) -> None:
    commands, tail, task, submit, reset = self._expected_contract()
    if type(self.native.commands) is not tuple or any(type(word) is not int or not 0 <= word < 1 << 64 for word in self.native.commands):
      self._reject("command_image", "native command words are not immutable uint64 values")
    if type(self.native.tail) is not tuple or any(type(word) is not int or not 0 <= word < 1 << 64 for word in self.native.tail):
      self._reject("command_image", "native command tail is not immutable uint64 values")
    if type(self.native.task) is not RKNativeTask or type(self.native.submit) is not RKNativeSubmit or type(self.native.reset) is not RKNativeReset:
      self._reject("native_schema", "native lifecycle records have foreign types")
    digest = rkp.CMAC_V1_BODY_SHA256 if self.kind is RKNativeKind.CMAC else rkp.LUT_V1_EXP2_COMMAND_SHA256
    if self.native.commands != commands or hashlib.sha256(struct.pack(f"<{len(commands)}Q", *self.native.commands)).digest() != bytes.fromhex(digest):
      self._reject("command_image", "native command bytes differ from the generated template")
    if self.native.tail != tail:
      self._reject("command_image", "native command tail differs from the generated contract")
    submit_fields = ("flags", "timeout_ms", "core_mask", "fence_fd", "task_count")
    if tuple(astuple(self.native.task)) != task:
      self._reject("task", "native task differs from generated descriptor")
    if tuple(getattr(self.native.submit, name) for name in submit_fields) != submit:
      self._reject("submit_contract", "native submit differs from generated contract")
    if (self.native.reset.flags, self.native.reset.value) != reset or self.native.reset.flags != rk.RKNPU_ACT_RESET:
      self._reject("reset_gate", "native reset is not the confirmed action-6 contract")

  def _check_relocations(self) -> None:
    expected = rkp.CMAC_V1_RELOCATIONS if self.kind is RKNativeKind.CMAC else rkp.LUT_V1_EXP2_RELOCATIONS
    expected_args = self.native.reads + self.native.outputs if self.kind is RKNativeKind.CMAC else self.native.outputs + self.native.reads
    if type(self.native.relocs) is not tuple or len(self.native.relocs) != len(expected):
      self._reject("relocations", "native relocation count is not canonical")
    for item, wanted, arg in zip(self.native.relocs, expected, expected_args):
      if type(item) is not RKNativeRelocation or (item.word_index, item.target, item.register, item.arg, item.shift, item.width) != (
        *wanted,
        arg,
        16,
        32,
      ):
        self._reject("relocations", "native relocation does not match the generated command")

  def _bind_resources(self) -> None:
    refs: tuple[RKArg, ...]
    roles: tuple[str, ...]
    sizes: tuple[int, ...]
    if self.kind is RKNativeKind.CMAC:
      if len(self.native.reads) != 2 or len(self.native.writes) != 1 or self.native.outputs != self.native.writes:
        self._reject("resource_contract", "CMAC must own two reads and one output")
      refs, roles, sizes = (
        self.native.reads + self.native.outputs,
        ("lhs", "rhs", "output"),
        (rkp.CMAC_V1_LHS_BYTES, rkp.CMAC_V1_RHS_BYTES, rkp.CMAC_V1_OUTPUT_VIEW_BYTES),
      )
    else:
      if len(self.native.reads) != 1 or len(self.native.writes) != 1 or self.native.outputs != self.native.writes:
        self._reject("resource_contract", "EXP2 must own one read and one output")
      refs, roles, sizes = (
        self.native.reads + self.native.outputs,
        ("input", "output"),
        (rkp.LUT_V1_EXP2_INPUT_ALLOCATION_BYTES, rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES),
      )
    seen: list[tuple[int, int]] = []
    for arg, role, required in zip(refs, roles, sizes):
      buffer = self.bufs[arg.index]
      if not isinstance(buffer, HCQBuffer):
        self._reject("buffer_binding", f"{role} is not an HCQBuffer")
      if buffer.size < required or _allocation_size(buffer) < required:
        self._reject("resource_bounds", f"{role} allocation is smaller than {required} bytes")
      dma, allocation = _dma(self.program, buffer), _allocation_size(buffer)
      if any(dma < old_dma + old_size and old_dma < dma + allocation for old_dma, old_size in seen):
        self._reject("dma_alias", f"{role} overlaps another native resource")
      seen.append((dma, allocation))
      self._resource_buffers[role] = buffer
      self._resource_args[role] = arg
      self._dma_by_arg[arg] = dma
    self._binding_snapshot = tuple(
      (arg, id(self.bufs[arg.index]), int(self.bufs[arg.index].va_addr), self.bufs[arg.index].size,
       _allocation_size(self.bufs[arg.index]), self._dma_by_arg[arg]) for arg in refs)
    self._span_snapshot = self.native.spans

  def _revalidate_bindings(self) -> None:
    if not self._binding_snapshot:
      self._reject("dma_binding", "native caller bindings were not snapshotted")
    if self.native is not self._native_identity or self.native.spans != self._span_snapshot:
      self._reject("span_contract", "native immutable plan or spans changed after preflight")
    current: list[tuple[RKArg, int, int, int, int, int]] = []
    ranges: list[tuple[int, int]] = []
    for arg, buffer_id, va_addr, requested, allocation, expected_dma in self._binding_snapshot:
      if not 0 <= arg.index < len(self.bufs):
        self._reject("dma_binding", "native argument index changed")
      buffer = self.bufs[arg.index]
      if not isinstance(buffer, HCQBuffer) or id(buffer) != buffer_id or int(buffer.va_addr) != va_addr or buffer.size != requested:
        self._reject("dma_binding", "caller buffer identity or span changed after preflight")
      actual_allocation, actual_dma = _allocation_size(buffer), _dma(self.program, buffer)
      if actual_allocation != allocation or actual_dma != expected_dma:
        self._reject("dma_binding", "caller DMA or allocation span changed after preflight")
      current.append((arg, id(buffer), int(buffer.va_addr), buffer.size, actual_allocation, actual_dma))
      ranges.append((actual_dma, actual_allocation))
    if any(a < c + d and c < a + b for index,(a,b) in enumerate(ranges) for c,d in ranges[index+1:]):
      self._reject("dma_alias", "caller DMA spans overlap before native patch or submit")
    for span in self.native.spans:
      if span.buffer not in self._dma_by_arg:
        self._reject("span_contract", "native span is not bound to a caller resource")
      allocation = next(item[4] for item in current if item[0] == span.buffer)
      if allocation != span.allocation or span.offset + span.size > allocation:
        self._reject("span_contract", "caller allocation no longer covers immutable native span")
    for role,arg in self._resource_args.items():
      if self._resource_buffers.get(role) is not self.bufs[arg.index]:
        self._reject("dma_binding", "caller resource role changed after preflight")
    self._dma_by_arg = {arg: dma for arg,_,_,_,_,dma in current}

  def _check_cmac_inputs(self) -> None:
    if (
      rkp.CMAC_V1_OUTPUT_SURFACE_BYTES != 128
      or rkp.CMAC_V1_OUTPUT_VIEW_BYTES != 256
      or rkp.CMAC_V1_OUTPUT_STRIDE_BYTES != 128
      or rkp.CMAC_V1_OUTPUT_SWIZZLE != tuple((channel // 16) * 32 + channel % 16 for channel in range(32))
      or len(rkp.CMAC_V1_OUTPUT_SWIZZLE) != rkp.CMAC_V1_PHYSICAL_CHANNELS
    ):
      self._reject("cmac_geometry", "CMAC output surface or byte swizzle differs from the immutable catalog")
    rhs = _read(self._resource_buffers["rhs"], rkp.CMAC_V1_RHS_BYTES)
    active = _CMAC_LOGICAL_LANES * rkp.CMAC_V1_RHS_KERNEL_STRIDE_BYTES
    if rhs[active:] != bytes(len(rhs) - active):
      self._reject("cmac_rhs", "inactive RHS lanes are not zero-filled")
    if (
      type(self.native.guards) is not tuple
      or type(self.native.repairs) is not tuple
      or type(self.native.assets) is not tuple
      or self.native.guards
      or self.native.repairs
      or self.native.assets
    ):
      self._reject("native_schema", "CMAC carries LUT-only metadata")

  def _check_exp2_metadata(self) -> None:
    if type(self.native.assets) is not tuple or len(self.native.assets) != 1 or type(self.native.assets[0]) is not RKNativeAsset:
      self._reject("asset", "EXP2 requires one immutable embedded asset")
    asset = self.native.assets[0]
    if (
      type(asset.payload) is not bytes
      or asset.asset_id != rkp.LUT_V1_EXP2_ASSET_ID
      or asset.size != rkp.LUT_V1_EXP2_TABLE_BYTES
      or len(asset.payload) != asset.size
      or asset.digest != bytes.fromhex(rkp.LUT_V1_EXP2_TABLE_SHA256)
      or asset.flags != 0
    ):
      self._reject("asset", "EXP2 asset bytes or digest differ from the generated table")
    if hashlib.sha256(asset.payload).digest() != asset.digest:
      self._reject("asset", "EXP2 embedded asset hash mismatch")
    if type(asset.ranges) is not tuple or any(
      type(interval) is not tuple or len(interval) != 2 or any(type(value) is not int for value in interval)
      for interval in asset.ranges
    ):
      self._reject("asset", "EXP2 asset ranges are not immutable integer pairs")
    if asset.ranges != _EXP2_IDLE_UPLOAD_RANGES or sum(count for _, count in asset.ranges) != asset.size or \
       any(start < 0 or count <= 0 or start + count > asset.size for start,count in asset.ranges):
      self._reject("asset", "EXP2 asset ranges are not the two canonical banks")
    indexes = (*range(1, 514), *range(515, 1028))
    encoded = struct.pack("<1026H", *((self.native.commands[index] >> 16) & 0xFFFF for index in indexes))
    if encoded != asset.payload:
      self._reject("asset", "EXP2 command table does not embed the immutable asset")
    if type(self.native.guards) is not tuple or len(self.native.guards) != 1 or type(self.native.guards[0]) is not RKNativeGuard:
      self._reject("output_guard", "EXP2 requires one output guard")
    guard = self.native.guards[0]
    if guard.buffer != self.native.outputs[0] or (guard.offset, guard.size, guard.fill) != (
      rkp.LUT_V1_EXP2_OUTPUT_BYTES,
      rkp.LUT_V1_EXP2_GUARD_BYTES,
      rkp.LUT_V1_EXP2_GUARD_FILL,
    ):
      self._reject("output_guard", "EXP2 output guard is not canonical")
    expected_spans = (
      RKNativeSpan(self.native.reads[0], RKNativeSpanKind.INPUT, 0, rkp.LUT_V1_EXP2_INPUT_BYTES,
                   rkp.LUT_V1_EXP2_INPUT_ALLOCATION_BYTES, provenance=RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(self.native.outputs[0], RKNativeSpanKind.OUTPUT_LOGICAL, 0, rkp.LUT_V1_EXP2_INPUT_BYTES,
                   rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, provenance=RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(self.native.outputs[0], RKNativeSpanKind.OUTPUT_PHYSICAL, 0, rkp.LUT_V1_EXP2_OUTPUT_BYTES,
                   rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, provenance=RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(self.native.outputs[0], RKNativeSpanKind.OUTPUT_PADDING, rkp.LUT_V1_EXP2_PADDING_OFFSET,
                   rkp.LUT_V1_EXP2_PADDING_BYTES, rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES,
                   rkp.LUT_V1_EXP2_PADDING_FILL, 2, RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(self.native.outputs[0], RKNativeSpanKind.OUTPUT_GUARD, rkp.LUT_V1_EXP2_OUTPUT_BYTES,
                   rkp.LUT_V1_EXP2_GUARD_BYTES, rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES,
                   rkp.LUT_V1_EXP2_GUARD_FILL, 1, RK_EXP2_PHYSICAL_PROVENANCE),
    )
    if type(self.native.spans) is not tuple or self.native.spans != expected_spans:
      self._reject("span_contract", "EXP2 geometry, padding, guard, or provenance is not immutable")
    if len(self.native.repairs) != len(RK_EXP2_REPAIR_METADATA) or any(type(rule) is not RKNativeRepair for rule in self.native.repairs):
      self._reject("exp2_repair", "EXP2 does not carry named device-side repair stages")
    expected_repairs = tuple(RKNativeRepair(RKNativeRepairKind.SPECIAL_VALUE, index + 1, index, index + 1, True,
      name, self.native.reads[0], self.native.outputs[0], RK_EXP2_PHYSICAL_PROVENANCE, RK_EXP2_REPAIR_DEVICE_STAGE)
      for index,name in enumerate(RK_EXP2_REPAIR_METADATA))
    if self.native.repairs != expected_repairs:
      self._reject("exp2_repair", "EXP2 repair provenance is not a device-side immutable contract")

  def _preflight(self) -> None:
    if type(self.native) is not RKNativeOp:
      self._reject("native_schema", "native object is foreign")
    self._check_args()
    self._check_command_contract()
    self._check_relocations()
    if self.kind is RKNativeKind.LUT:
      self._check_exp2_metadata()
    self._bind_resources()
    if self.kind is RKNativeKind.CMAC:
      self._check_cmac_inputs()
    if self.program.dev.native_reset_live_proven is not True:
      self._reject("reset_gate", "action-6 reset live gate is not proven by device owner")

  def _require_buffer(self, attr: str) -> HCQBuffer:
    buffer = getattr(self, attr)
    if buffer is None:
      self._reject("allocator", f"native {attr} buffer is missing")
    return buffer

  def _ensure_buffers(self) -> None:
    command_size = rkp.CMAC_V1_COMMAND_RESERVATION_BYTES if self.kind is RKNativeKind.CMAC else rkp.LUT_V1_EXP2_COMMAND_ALLOCATION_BYTES
    task_size = _PAGE if self.kind is RKNativeKind.CMAC else rkp.LUT_V1_EXP2_TASK_ALLOCATION_BYTES
    allocate = getattr(self.program.dev, "_gpu_alloc", None)
    if not callable(allocate):
      self._reject("allocator", "device has no Rockchip allocator")
    try:
      command_buffer = allocate(command_size)
      self._cmd_buf = command_buffer
      task_buffer = allocate(task_size, rk.RKNPU_MEM_KERNEL_MAPPING)
      self._task_buf = task_buffer
      command_buffer, task_buffer = self._require_buffer("_cmd_buf"), self._require_buffer("_task_buf")
      if _allocation_size(command_buffer) < command_size or _allocation_size(task_buffer) < task_size:
        self._reject("allocator", "native allocation is smaller than contract")
      ranges = [
        (_dma(self.program, command_buffer), _allocation_size(command_buffer)),
        (_dma(self.program, task_buffer), _allocation_size(task_buffer)),
      ]
      ranges += [(_dma(self.program, buffer), _allocation_size(buffer)) for buffer in self._resource_buffers.values()]
      if any(i < j and a < c + d and c < a + b for i, (a, b) in enumerate(ranges) for j, (c, d) in enumerate(ranges)):
        self._reject("dma_alias", "command/task memory aliases caller resource")
    except PhysicalRuntimeReject:
      raise
    except Exception as exc:
      self._reject("allocator", "native command/task allocation failed")
      raise AssertionError from exc

  def _patch_commands(self) -> tuple[int, ...]:
    commands = list(self.native.commands)
    for item in self.native.relocs:
      base = self._dma_by_arg.get(item.arg)
      if base is None:
        self._reject("relocations", "relocation argument was not bound")
      value = base + item.arg.addend
      if not 0 <= value <= _U32_MAX or value & 0xF:
        self._reject("relocations", "relocation DMA is not representable")
      if commands[item.word_index] >> 48 != item.target or commands[item.word_index] & 0xFFFF != item.register:
        self._reject("relocations", "relocation command word changed")
      commands[item.word_index] = (commands[item.word_index] & ~0xFFFFFFFF0000) | (value << 16)
    return tuple(commands)

  def _write_command_and_task(self) -> None:
    command_buffer, task_buffer = self._require_buffer("_cmd_buf"), self._require_buffer("_task_buf")
    self._revalidate_bindings()
    patched = self._patch_commands()
    command_image = struct.pack(f"<{len(patched)}Q", *patched) + struct.pack(f"<{len(self.native.tail)}Q", *self.native.tail)
    expected = rkp.CMAC_V1_COMMAND_IMAGE_BYTES if self.kind is RKNativeKind.CMAC else len(rkp.LUT_V1_EXP2_COMMANDS) * 8
    if len(command_image) != expected:
      self._reject("command_image", "native command image length differs from contract")
    ctypes.memset(int(command_buffer.va_addr), 0, _allocation_size(command_buffer))
    _write(command_buffer, command_image)
    task = self.native.task
    task_words = tuple(cast(tuple[int, ...], astuple(task)))
    if len(task_words) != 8:
      self._reject("task", "native task wire tuple must contain eight words")
    descriptor = rk.struct_rknpu_task(
      *task_words,
      _dma(self.program, command_buffer),
    )
    if _TASK_DESCRIPTOR_BYTES != rkp.CMAC_V1_TASK_DESCRIPTOR_BYTES:
      self._reject("task", "driver task descriptor size changed")
    ctypes.memset(int(task_buffer.va_addr), 0, _allocation_size(task_buffer))
    ctypes.memmove(int(task_buffer.va_addr), ctypes.addressof(descriptor), _TASK_DESCRIPTOR_BYTES)

  def _prepare_output(self) -> None:
    if self.kind is RKNativeKind.LUT:
      ctypes.memset(int(self._resource_buffers["output"].va_addr) + rkp.LUT_V1_EXP2_OUTPUT_BYTES, 0xA5, rkp.LUT_V1_EXP2_GUARD_BYTES)

  def _prepare(self) -> None:
    if self._prepared:
      return
    self._ensure_buffers()
    self._write_command_and_task()
    self._prepare_output()
    self._prepared = True
    self.telemetry.attempts += 1
    self._event("attempt")

  def _sync(self, buffer: HCQBuffer, flags: int) -> None:
    sync = getattr(self.program.dev, "_sync_buffer", None)
    if not callable(sync):
      self._reject("sync", "device has no sync primitive")
    sync(buffer, flags)

  def _cleanup(self) -> None:
    if self._unknown or self._closed:
      return
    free = getattr(self.program.dev, "_gpu_free", None)
    if not callable(free):
      self._unknown = True
      self.last_reason = "cleanup"
      self.telemetry.reject("cleanup_unknown")
      raise PhysicalOwnershipUnknown("native allocation cleanup primitive is unavailable")
    errors: list[Exception] = []
    for attr in ("_cmd_buf", "_task_buf"):
      buffer = getattr(self, attr)
      if buffer is not None:
        try:
          free(buffer)
        except Exception as exc:
          errors.append(exc)
        else:
          setattr(self, attr, None)
    if errors:
      self._unknown = True
      self.last_reason = "cleanup"
      self.telemetry.reject("cleanup_unknown")
      raise PhysicalOwnershipUnknown("native command/task cleanup ownership is unknown") from errors[0]
    self._closed = True

  def _upload_asset_while_idle(self) -> None:
    if self.kind is not RKNativeKind.LUT:
      return
    asset = self.native.assets[0]
    uploader = getattr(self.program.dev, "_upload_native_asset", None)
    try:
      if not callable(uploader):
        self._reject("asset", "device has no idle native-asset upload primitive")
      uploader(asset.payload, asset.ranges)
    except PhysicalRuntimeReject:
      raise
    except Exception as exc:
      self._reject("asset", f"native idle asset upload failed: {exc}")
    self.telemetry.asset_bytes += len(asset.payload)
    self._event("asset_upload_idle")

  def reset_before(self) -> None:
    if self._reset or self._in_flight:
      self._reject("lifecycle", "native reset is not first or one-shot")
    self._prepare()
    if self.kind is RKNativeKind.CMAC:
      self._sync(self._resource_buffers["lhs"], rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self._sync(self._resource_buffers["rhs"], rk.RKNPU_MEM_SYNC_TO_DEVICE)
    else:
      self._sync(self._resource_buffers["input"], rk.RKNPU_MEM_SYNC_TO_DEVICE)
    if not self._command_synced:
      self._sync(self._require_buffer("_cmd_buf"), rk.RKNPU_MEM_SYNC_TO_DEVICE)
    self._sync(self._require_buffer("_task_buf"), rk.RKNPU_MEM_SYNC_TO_DEVICE)
    reset = getattr(self.program.dev, "reset_npu", None)
    if not callable(reset):
      self._reject("reset", "device has no reset primitive")
    try:
      reset()
    except Exception:
      self._cleanup()
      raise
    self._reset = True
    self.telemetry.resets += 1
    self._event("reset_action_6")

  def barrier_before(self) -> None:
    if not self._reset or self._in_flight:
      self._reject("lifecycle", "native barrier-before is out of order")
    self._event("barrier_before")

  def submit_physical(self) -> PhysicalSubmitReceipt:
    if not self._reset or self._in_flight:
      self._reject("lifecycle", "native submit is out of order")
    submitter = getattr(self.program, "_submit_physical", None)
    if not callable(submitter):
      self._reject("submit", "program has no physical submit primitive")
    self._revalidate_bindings()
    self._in_flight = True
    self.telemetry.pc_submits += 1
    self._event("submit_one")
    try:
      result = submitter(self._require_buffer("_cmd_buf"), self._require_buffer("_task_buf"), self.native.submit)
    except Exception as exc:
      self._unknown = True
      self.telemetry.reject("submit_unknown")
      raise PhysicalOwnershipUnknown("physical submit crossed driver boundary") from exc
    if type(result) is not PhysicalSubmitReceipt:
      self._unknown = True
      self.telemetry.reject("submit_unknown")
      raise PhysicalOwnershipUnknown("physical submit returned a non-contract receipt")
    receipt = result
    if type(receipt.status) is not int or type(receipt.ownership_known) is not bool or not receipt.ownership_known or receipt.submit_id is None:
      self._unknown = True
      self.telemetry.reject("submit_unknown")
      raise PhysicalOwnershipUnknown("physical submit ownership is unknown")
    if receipt.status != 0:
      self._in_flight = False
      self._cleanup()
      raise PhysicalRuntimeReject("submit", f"physical submit returned {receipt.status}")
    self.telemetry.native_submit += 1
    self.telemetry.native_submit_ids.append(receipt.submit_id)
    return receipt

  def _validate_cmac_output(self) -> None:
    raw = _read(self._resource_buffers["output"], rkp.CMAC_V1_OUTPUT_VIEW_BYTES)
    self.last_output = raw
    self.last_logical_output = cmac_logical_output(raw, _CMAC_LOGICAL_LANES, rkp.CMAC_V1_OUTPUT_SWIZZLE)

  def _validate_exp2_output(self) -> None:
    output = self._resource_buffers["output"]
    raw = _read(output, rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES)
    if raw[rkp.LUT_V1_EXP2_OUTPUT_BYTES :] != bytes([rkp.LUT_V1_EXP2_GUARD_FILL]) * rkp.LUT_V1_EXP2_GUARD_BYTES:
      self._reject("output_guard", "EXP2 output guard was modified")
    padding = struct.pack("<H", rkp.LUT_V1_EXP2_PADDING_FILL) * (rkp.LUT_V1_EXP2_PADDING_BYTES // 2)
    if raw[rkp.LUT_V1_EXP2_INPUT_BYTES : rkp.LUT_V1_EXP2_OUTPUT_BYTES] != padding:
      self._reject("output_padding", "EXP2 physical padding was modified")
    self.last_output = raw[: rkp.LUT_V1_EXP2_OUTPUT_BYTES]

  def barrier_after(self) -> None:
    if not self._in_flight or self._unknown:
      self._reject("lifecycle", "native barrier-after is out of order")
    try:
      self._sync(self._resource_buffers["output"], rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      if self.kind is RKNativeKind.CMAC:
        self._validate_cmac_output()
      else:
        self._validate_exp2_output()
    except Exception as exc:
      self._unknown = True
      self.telemetry.reject("readback_unknown")
      raise PhysicalOwnershipUnknown("physical readback did not prove safe completion") from exc
    self._event("barrier_after")
    self._in_flight = False
    self._cleanup()

  def halt_unknown_ownership(self, reason: str) -> None:
    self._unknown = True
    self.last_reason = reason
    self.telemetry.reject("ownership_unknown")

  def execute(self, *, preflight: bool = True) -> PhysicalSubmitReceipt:
    try:
      if preflight:
        self._preflight()
      self._prepare()
      self._upload_asset_while_idle()
      self.reset_before()
      self.barrier_before()
      receipt = self.submit_physical()
      self.barrier_after()
      self._event("success")
      return receipt
    except PhysicalOwnershipUnknown:
      self.halt_unknown_ownership(self.last_reason or "physical ownership unknown")
      raise
    except Exception:
      if not self._in_flight and not self._unknown:
        self._cleanup()
      raise


__all__ = [
  "PhysicalOwnershipUnknown",
  "PhysicalRuntimeReject",
  "PhysicalRuntimeTelemetry",
  "PhysicalSubmitReceipt",
  "RKNativeOpLike",
  "RK_PHYSICAL_TELEMETRY_IDS",
  "RockchipPhysicalEffects",
  "cmac_logical_output",
]
