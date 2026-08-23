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
_U64_MAX = (1 << 64) - 1
_TASK_DESCRIPTOR_BYTES = ctypes.sizeof(rk.struct_rknpu_task)
_EXP2_IDLE_UPLOAD_RANGES = ((0, 1026), (1026, 1026))
_CMAC_LOGICAL_LANES = 4
_CMAC_GUARD_FILL = 0xA5
# These are runtime-owned allocation guards, not output guards.  In
# particular, CMAC's [128:256) host view tail is deliberately not a guard.
_CMAC_GUARD_SPANS = (
  ("_cmd_buf", rkp.CMAC_V1_COMMAND_IMAGE_BYTES, rkp.CMAC_V1_COMMAND_RESERVATION_BYTES - rkp.CMAC_V1_COMMAND_IMAGE_BYTES),
  ("_task_buf", rkp.CMAC_V1_TASK_DESCRIPTOR_BYTES, _PAGE - rkp.CMAC_V1_TASK_DESCRIPTOR_BYTES),
)
_BufferSnapshot = tuple[int, int, int, int, int, int, int, int, int, int, int]

# Stable event IDs.  These are a finite vocabulary, not a second telemetry
# policy or an attempt counter encoded as an event name.
RK_PHYSICAL_TELEMETRY_IDS = {
  RKNativeKind.CMAC: (
    "cmac_v1.attempt",
    "cmac_v1.asset_upload_idle",
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
  base = buffer.base
  allocation = getattr(getattr(base, "meta", None), "size", getattr(base, "size", None))
  if type(allocation) is not int or offset < 0 or size < 0 or offset + size > allocation:
    raise PhysicalRuntimeReject("readback", "read exceeds the declared allocation")
  return ctypes.string_at(int(buffer.va_addr) + offset, size)


def _write(buffer: HCQBuffer, data: bytes, offset: int = 0) -> None:
  base = buffer.base
  allocation = getattr(getattr(base, "meta", None), "size", getattr(base, "size", None))
  if type(allocation) is not int or offset < 0 or offset + len(data) > allocation:
    raise PhysicalRuntimeReject("write", "write exceeds the declared allocation")
  ctypes.memmove(int(buffer.va_addr) + offset, data, len(data))


def _allocation_size(buffer: HCQBuffer) -> int:
  base = buffer.base
  value = getattr(getattr(base, "meta", None), "size", getattr(base, "size", None))
  if type(value) is not int or value <= 0:
    raise PhysicalRuntimeReject("dma", "buffer allocation size is invalid")
  return value


def _obj_addr(buffer: HCQBuffer) -> int:
  base = buffer.base
  value = getattr(getattr(base, "meta", None), "obj_addr", None)
  if type(value) is not int or value < 0:
    raise PhysicalRuntimeReject("dma", "buffer object address is invalid")
  return value


def _handle(buffer: HCQBuffer) -> int:
  base = buffer.base
  value = getattr(getattr(base, "meta", None), "handle", None)
  if type(value) is not int or not 0 <= value <= _U32_MAX:
    raise PhysicalRuntimeReject("dma", "buffer GEM handle is invalid")
  return value


def _dma(program: object, buffer: HCQBuffer) -> int:
  resolver = getattr(program, "_dma", None)
  if not callable(resolver):
    raise PhysicalRuntimeReject("dma", "program has no DMA resolver")
  base = buffer.base
  try:
    base_dma, buffer_va, base_va = resolver(base), buffer.va_addr, base.va_addr
  except Exception as exc:
    raise PhysicalRuntimeReject("dma", "buffer DMA metadata is invalid") from exc
  if type(base_dma) is not int or type(buffer_va) is not int or type(base_va) is not int:
    raise PhysicalRuntimeReject("dma", "buffer DMA metadata is invalid")
  value = base_dma + buffer_va - base_va
  if type(value) is not int or not 0 < value <= _U32_MAX or value & (_PAGE - 1):
    raise PhysicalRuntimeReject("dma", f"DMA address {value!r} is not an aligned uint32")
  if value + _allocation_size(buffer) > 1 << 32:
    raise PhysicalRuntimeReject("dma", "DMA allocation exceeds uint32")
  return value


def _base_metadata(buffer: HCQBuffer) -> tuple[HCQBuffer, int, int]:
  try:
    base = buffer.base
    base_va, base_size = base.va_addr, base.size
  except Exception as exc:
    raise PhysicalRuntimeReject("dma", "buffer base metadata is invalid") from exc
  if not isinstance(base, HCQBuffer) or type(base_va) is not int or type(base_size) is not int or base_va < 0 or base_size <= 0:
    raise PhysicalRuntimeReject("dma", "buffer base metadata is invalid")
  if buffer.meta is not base.meta:
    raise PhysicalRuntimeReject("dma", "buffer view metadata is not identical to base metadata")
  allocation_size = _allocation_size(buffer)
  if base_size > allocation_size:
    raise PhysicalRuntimeReject("dma", "buffer base logical size exceeds allocation")
  if base_va > _U64_MAX or base_size > _U64_MAX - base_va:
    raise PhysicalRuntimeReject("dma", "buffer base range overflows address space")
  return base, base_va, base_size


def _buffer_snapshot(program: object, buffer: HCQBuffer) -> _BufferSnapshot:
  base, base_va, base_size = _base_metadata(buffer)
  buffer_va, buffer_size = buffer.va_addr, buffer.size
  if type(buffer_va) is not int or type(buffer_size) is not int or buffer_va < 0 or buffer_size < 0:
    raise PhysicalRuntimeReject("dma", "buffer metadata is invalid")
  if buffer_va > _U64_MAX or buffer_size > _U64_MAX - buffer_va:
    raise PhysicalRuntimeReject("dma", "buffer view range overflows address space")
  base_end, buffer_end = base_va + base_size, buffer_va + buffer_size
  if buffer_va < base_va or buffer_end > base_end:
    raise PhysicalRuntimeReject("dma", "buffer view range lies outside base")
  return (id(buffer), id(base), base_va, base_size, buffer_va, buffer_size, _allocation_size(buffer),
          _dma(program, buffer), _obj_addr(buffer), _handle(buffer), id(base.meta))


def _owned_buffer_snapshot(program: object, buffer: object, logical_size: int, role: str) -> _BufferSnapshot:
  if type(buffer) is not HCQBuffer:
    raise PhysicalRuntimeReject("allocator", f"native {role} allocation is not an exact HCQBuffer")
  if buffer.base is not buffer:
    raise PhysicalRuntimeReject("allocator", f"native {role} allocation is an HCQBuffer view")
  if type(buffer.size) is not int or buffer.size != logical_size:
    raise PhysicalRuntimeReject("allocator", f"native {role} logical size is not {logical_size}")
  if _allocation_size(buffer) < logical_size:
    raise PhysicalRuntimeReject("allocator", f"native {role} allocation is smaller than {logical_size}")
  return _buffer_snapshot(program, buffer)


def cmac_logical_output(raw_bytes: bytes, logical_lanes: int, swizzle: Sequence[int]) -> bytes:
  """Extract logical FP16 lanes from the raw 32-channel CMAC surface."""
  if len(raw_bytes) < rkp.CMAC_V1_OUTPUT_VIEW_BYTES or not 0 < logical_lanes <= 16:
    raise ValueError("invalid CMAC raw output image")
  indexes = tuple(swizzle)
  if len(indexes) != 32 or any(type(index) is not int or not 0 <= index < 64 for index in indexes):
    raise ValueError("invalid CMAC output swizzle")
  words = struct.unpack_from("<64H", raw_bytes)
  return struct.pack(f"<{logical_lanes}H", *(words[indexes[index]] for index in range(logical_lanes)))


def _native_fingerprint(native: RKNativeOp) -> bytes:
  """Hash every immutable command, asset, resource, and lifecycle field."""
  return hashlib.sha256(repr(astuple(native)).encode("utf-8")).digest()


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
    self._asset_buffers: dict[int, HCQBuffer] = {}
    self._resource_buffers: dict[str, HCQBuffer] = {}
    self._resource_args: dict[str, RKArg] = {}
    self._dma_by_arg: dict[RKArg, int] = {}
    self._asset_snapshot: dict[int, _BufferSnapshot] = {}
    self._command_snapshot: _BufferSnapshot | None = None
    self._task_snapshot: _BufferSnapshot | None = None
    self._binding_snapshot: tuple[tuple[RKArg, _BufferSnapshot], ...] = ()
    self._span_snapshot: tuple[RKNativeSpan, ...] = ()
    self._native_fingerprint: bytes | None = None
    self._prepared = False
    self._reset = False
    self._in_flight = False
    self._unknown = False
    self._closed = False
    self._command_synced = False
    self._assets_uploaded = False
    self._cmac_asset_mode = False
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
    has_asset_relocation = type(self.native.relocs) is tuple and any(
      type(item) is RKNativeRelocation and type(item.arg) is RKArg and item.arg.kind is RKBufferKind.ASSET
      for item in self.native.relocs)
    if self.kind is RKNativeKind.CMAC and has_asset_relocation:
      expected_args = self.native.reads + (RKArg(RKBufferKind.ASSET, 0),) + self.native.outputs
    else:
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
      if len(self.native.writes) != 1 or self.native.outputs != self.native.writes:
        self._reject("resource_contract", "CMAC must own one output write")
      self._cmac_asset_mode = any(item.arg.kind is RKBufferKind.ASSET for item in self.native.relocs)
      if self._cmac_asset_mode:
        if len(self.native.reads) != 1:
          self._reject("resource_contract", "CMAC asset route must own one LHS read")
        refs, roles, sizes = (
          self.native.reads + self.native.outputs,
          ("lhs", "output"),
          (rkp.CMAC_V1_LHS_BYTES, rkp.CMAC_V1_OUTPUT_VIEW_BYTES),
        )
      else:
        if len(self.native.reads) != 2:
          self._reject("resource_contract", "CMAC must own two reads for a dynamic RHS")
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
      _base_metadata(buffer)
      if buffer.size < required or _allocation_size(buffer) < required:
        self._reject("resource_bounds", f"{role} allocation is smaller than {required} bytes")
      dma, allocation = _dma(self.program, buffer), _allocation_size(buffer)
      if any(dma < old_dma + old_size and old_dma < dma + allocation for old_dma, old_size in seen):
        self._reject("dma_alias", f"{role} overlaps another native resource")
      seen.append((dma, allocation))
      self._resource_buffers[role] = buffer
      self._resource_args[role] = arg
      self._dma_by_arg[arg] = dma
    self._binding_snapshot = tuple((arg, _buffer_snapshot(self.program, self.bufs[arg.index])) for arg in refs)
    self._span_snapshot = self.native.spans

  def _revalidate_bindings(self) -> None:
    if not self._binding_snapshot:
      self._reject("dma_binding", "native caller bindings were not snapshotted")
    if self.native is not self._native_identity or self.native.spans != self._span_snapshot:
      self._reject("span_contract", "native immutable plan or spans changed after preflight")
    current: list[tuple[RKArg, _BufferSnapshot]] = []
    ranges: list[tuple[int, int]] = []
    for arg, expected in self._binding_snapshot:
      if not 0 <= arg.index < len(self.bufs):
        self._reject("dma_binding", "native argument index changed")
      buffer = self.bufs[arg.index]
      try:
        current_snapshot = _buffer_snapshot(self.program, buffer) if isinstance(buffer, HCQBuffer) else None
      except PhysicalRuntimeReject:
        current_snapshot = None
      if current_snapshot != expected:
        self._reject("dma_binding", "caller buffer identity or span changed after preflight")
      current.append((arg, expected))
      ranges.append((expected[7], expected[6]))
    if any(a < c + d and c < a + b for index,(a,b) in enumerate(ranges) for c,d in ranges[index+1:]):
      self._reject("dma_alias", "caller DMA spans overlap before native patch or submit")
    for span in self.native.spans:
      if span.buffer not in self._dma_by_arg:
        self._reject("span_contract", "native span is not bound to a caller resource")
      allocation = next(item[1][6] for item in current if item[0] == span.buffer)
      if allocation != span.allocation or span.offset + span.size > allocation:
        self._reject("span_contract", "caller allocation no longer covers immutable native span")
    for role,arg in self._resource_args.items():
      if self._resource_buffers.get(role) is not self.bufs[arg.index]:
        self._reject("dma_binding", "caller resource role changed after preflight")
    self._dma_by_arg = {arg: snapshot[7] for arg, snapshot in current}

  def _revalidate_assets(self) -> None:
    current = []
    for index, expected in sorted(self._asset_snapshot.items()):
      buffer = self._asset_buffers.get(index)
      if buffer is None or type(buffer) is not HCQBuffer or _buffer_snapshot(self.program, buffer) != expected:
        self._reject("asset", "immutable asset DMA span changed after allocation")
      current.append((index, expected))
    if dict(current) != self._asset_snapshot:
      self._reject("asset", "immutable asset allocation was not snapshotted")

  def _revalidate_owned_buffers(self) -> None:
    if self._command_snapshot is None or self._task_snapshot is None:
      self._reject("allocator", "native command/task allocations were not snapshotted")
    command_buffer, task_buffer = self._cmd_buf, self._task_buf
    if command_buffer is None or task_buffer is None or type(command_buffer) is not HCQBuffer or type(task_buffer) is not HCQBuffer:
      self._reject("allocator", "native command/task allocation type changed after allocation")
    if _buffer_snapshot(self.program, command_buffer) != self._command_snapshot:
      self._reject("allocator", "native command allocation metadata changed after allocation")
    if _buffer_snapshot(self.program, task_buffer) != self._task_snapshot:
      self._reject("allocator", "native task allocation metadata changed after allocation")
    self._revalidate_assets()

  def _revalidate_native_fingerprint(self) -> None:
    if self._native_fingerprint is None:
      self._reject("native_schema", "native fingerprint was not captured during preflight")
    if self.native is not self._native_identity:
      if type(self.native) is not RKNativeOp:
        self._reject("native_schema", "native immutable plan identity changed after preflight")
      if self.native.spans != self._span_snapshot:
        self._reject("span_contract", "native immutable spans changed after preflight")
      self._reject("native_schema", "native immutable plan identity changed after preflight")
    if _native_fingerprint(self.native) != self._native_fingerprint:
      self._reject("native_fingerprint", "native command, asset, or lifecycle bytes changed after preflight")

  def _check_cmac_asset(self) -> None:
    if not self._cmac_asset_mode:
      return
    if type(self.native.assets) is not tuple or len(self.native.assets) != 1 or type(self.native.assets[0]) is not RKNativeAsset:
      self._reject("asset", "CMAC asset route requires one immutable asset")
    asset = self.native.assets[0]
    if (asset.asset_id, asset.size, asset.digest, asset.ranges, asset.flags, asset.payload) != (
      rkp.CMAC_V1_RHS_ASSET_ID, rkp.CMAC_V1_RHS_ASSET_SIZE, bytes.fromhex(rkp.CMAC_V1_RHS_ASSET_SHA256),
      rkp.CMAC_V1_RHS_ASSET_RANGES, 0, rkp.CMAC_V1_RHS_ASSET_PAYLOAD):
      self._reject("asset", "CMAC RHS asset bytes, digest, flags, or ranges differ from the generated asset")
    if hashlib.sha256(asset.payload).digest() != asset.digest:
      self._reject("asset", "CMAC RHS asset hash mismatch")

  def _check_cmac_inputs(self) -> None:
    if (
      rkp.CMAC_V1_OUTPUT_SURFACE_BYTES != 128
      or rkp.CMAC_V1_OUTPUT_VIEW_BYTES != 256
      or rkp.CMAC_V1_OUTPUT_STRIDE_BYTES != 128
      or rkp.CMAC_V1_OUTPUT_SWIZZLE != tuple((channel // 16) * 32 + channel % 16 for channel in range(32))
      or len(rkp.CMAC_V1_OUTPUT_SWIZZLE) != rkp.CMAC_V1_PHYSICAL_CHANNELS
    ):
      self._reject("cmac_geometry", "CMAC output surface or byte swizzle differs from the immutable catalog")
    if self._cmac_asset_mode:
      self._check_cmac_asset()
      if type(self.native.guards) is not tuple or type(self.native.repairs) is not tuple or self.native.guards or self.native.repairs:
        self._reject("native_schema", "CMAC asset route carries unexpected guards or repairs")
    else:
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
    self._native_fingerprint = _native_fingerprint(self.native)

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
      if isinstance(command_buffer, HCQBuffer):
        self._command_snapshot = _buffer_snapshot(self.program, command_buffer)
      command_snapshot = _owned_buffer_snapshot(self.program, command_buffer, command_size, "command")
      task_buffer = allocate(task_size, rk.RKNPU_MEM_KERNEL_MAPPING)
      self._task_buf = task_buffer
      if isinstance(task_buffer, HCQBuffer):
        self._task_snapshot = _buffer_snapshot(self.program, task_buffer)
      task_snapshot = _owned_buffer_snapshot(self.program, task_buffer, task_size, "task")
      command_buffer, task_buffer = self._require_buffer("_cmd_buf"), self._require_buffer("_task_buf")
      self._command_snapshot, self._task_snapshot = command_snapshot, task_snapshot
      ranges = [
        (_dma(self.program, command_buffer), _allocation_size(command_buffer)),
        (_dma(self.program, task_buffer), _allocation_size(task_buffer)),
      ]
      ranges += [(_dma(self.program, buffer), _allocation_size(buffer)) for buffer in self._asset_buffers.values()]
      ranges += [(_dma(self.program, buffer), _allocation_size(buffer)) for buffer in self._resource_buffers.values()]
      if any(i < j and a < c + d and c < a + b for i, (a, b) in enumerate(ranges) for j, (c, d) in enumerate(ranges)):
        self._reject("dma_alias", "command/task memory aliases caller resource")
    except PhysicalRuntimeReject:
      raise
    except Exception as exc:
      self._reject("allocator", "native command/task allocation failed")
      raise AssertionError from exc

  def _ensure_asset_buffers(self) -> None:
    indexes = tuple(sorted({item.arg.index for item in self.native.relocs if item.arg.kind is RKBufferKind.ASSET}))
    if not indexes:
      return
    allocate = getattr(self.program.dev, "_gpu_alloc", None)
    if not callable(allocate):
      self._reject("asset", "device has no immutable asset allocator")
    if self._asset_buffers:
      if tuple(sorted(self._asset_buffers)) != indexes or not self._asset_snapshot:
        self._reject("asset", "immutable asset allocation snapshot is incomplete")
      self._revalidate_assets()
      return
    try:
      for index in indexes:
        if index in self._asset_buffers:
          continue
        if index < 0 or index >= len(self.native.assets):
          self._reject("asset", "asset relocation index is out of range")
        asset = self.native.assets[index]
        buffer = allocate(asset.size)
        self._asset_buffers[index] = buffer
        if isinstance(buffer, HCQBuffer):
          self._asset_snapshot[index] = _buffer_snapshot(self.program, buffer)
        snapshot = _owned_buffer_snapshot(self.program, buffer, asset.size, f"asset[{index}]")
        self._asset_snapshot[index] = snapshot
        allocation, dma = snapshot[6], snapshot[7]
        caller_ranges = [(_dma(self.program, resource), _allocation_size(resource)) for resource in self._resource_buffers.values()]
        prior_assets = [(_dma(self.program, resource), _allocation_size(resource))
                        for asset_index, resource in self._asset_buffers.items() if asset_index != index]
        if any(dma < old_dma + old_size and old_dma < dma + allocation for old_dma, old_size in (*caller_ranges, *prior_assets)):
          self._reject("dma_alias", "immutable asset DMA span overlaps a caller or asset resource")
        ctypes.memset(int(buffer.va_addr), 0, allocation)
        for start, count in asset.ranges:
          _write(buffer, asset.payload[start:start + count], start)
      self._asset_snapshot = {
        index: _owned_buffer_snapshot(self.program, buffer, self.native.assets[index].size, f"asset[{index}]")
        for index, buffer in sorted(self._asset_buffers.items())
      }
    except PhysicalRuntimeReject:
      raise
    except Exception as exc:
      self._reject("asset", "immutable asset allocation or write failed")
      raise AssertionError from exc

  def _patch_commands(self) -> tuple[int, ...]:
    commands = list(self.native.commands)
    for item in self.native.relocs:
      if item.arg.kind is RKBufferKind.ASSET:
        asset_buffer = self._asset_buffers.get(item.arg.index)
        base = None if asset_buffer is None else _dma(self.program, asset_buffer)
      else:
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
    self._revalidate_native_fingerprint()
    self._revalidate_owned_buffers()
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

  def _prepare_cmac_guards(self) -> None:
    if self.kind is not RKNativeKind.CMAC:
      return
    for attr, offset, size in _CMAC_GUARD_SPANS:
      buffer = self._require_buffer(attr)
      ctypes.memset(int(buffer.va_addr) + offset, _CMAC_GUARD_FILL, size)

  def _prepare(self) -> None:
    if self._prepared:
      return
    self._revalidate_native_fingerprint()
    self._ensure_asset_buffers()
    self._ensure_buffers()
    self._write_command_and_task()
    self._prepare_cmac_guards()
    self._prepare_output()
    self._prepared = True
    self.telemetry.attempts += 1
    self._event("attempt")

  def _sync(self, buffer: HCQBuffer, flags: int) -> None:
    sync = getattr(self.program.dev, "_sync_buffer", None)
    if not callable(sync):
      self._reject("sync", "device has no sync primitive")
    sync(buffer, flags)

  def _refuse_cleanup(self, detail: str) -> NoReturn:
    self._unknown = True
    self.last_reason = "cleanup"
    self.telemetry.reject("cleanup_unknown")
    raise PhysicalOwnershipUnknown(detail)

  def _validate_cleanup_snapshots(self) -> None:
    """Refuse every free if any owned metadata no longer names the allocation."""
    try:
      if self._cmd_buf is None:
        if self._command_snapshot is not None:
          self._refuse_cleanup("native command allocation disappeared before cleanup")
      elif self._command_snapshot is None or not isinstance(self._cmd_buf, HCQBuffer) or \
           _buffer_snapshot(self.program, self._cmd_buf) != self._command_snapshot:
        self._refuse_cleanup("native command metadata changed; refusing free")
      if self._task_buf is None:
        if self._task_snapshot is not None:
          self._refuse_cleanup("native task allocation disappeared before cleanup")
      elif self._task_snapshot is None or not isinstance(self._task_buf, HCQBuffer) or \
           _buffer_snapshot(self.program, self._task_buf) != self._task_snapshot:
        self._refuse_cleanup("native task metadata changed; refusing free")
      if set(self._asset_snapshot) != set(self._asset_buffers):
        self._refuse_cleanup("native asset snapshot coverage changed; refusing free")
      for index, buffer in self._asset_buffers.items():
        if not isinstance(buffer, HCQBuffer) or _buffer_snapshot(self.program, buffer) != self._asset_snapshot[index]:
          self._refuse_cleanup(f"native asset[{index}] metadata changed; refusing free")
    except PhysicalOwnershipUnknown:
      raise
    except Exception as exc:
      self._refuse_cleanup("native allocation metadata could not be verified before free")
      raise AssertionError from exc

  def _cleanup(self) -> None:
    if self._unknown or self._closed:
      return
    free = getattr(self.program.dev, "_gpu_free", None)
    if not callable(free):
      self._unknown = True
      self.last_reason = "cleanup"
      self.telemetry.reject("cleanup_unknown")
      raise PhysicalOwnershipUnknown("native allocation cleanup primitive is unavailable")
    self._validate_cleanup_snapshots()
    errors: list[Exception] = []
    buffers: list[tuple[str, HCQBuffer]] = [
      (attr, buffer) for attr in ("_cmd_buf", "_task_buf") if (buffer := getattr(self, attr)) is not None
    ]
    buffers.extend((f"asset[{index}]", buffer) for index, buffer in sorted(self._asset_buffers.items()))
    for name, buffer in buffers:
      try:
        free(buffer)
      except Exception as exc:
        errors.append(exc)
      else:
        if name.startswith("asset["):
          del self._asset_buffers[int(name[6:-1])]
        else:
          setattr(self, name, None)
    if errors:
      self._unknown = True
      self.last_reason = "cleanup"
      self.telemetry.reject("cleanup_unknown")
      raise PhysicalOwnershipUnknown("native command/task cleanup ownership is unknown") from errors[0]
    self._asset_snapshot = {}
    self._command_snapshot = self._task_snapshot = None
    self._assets_uploaded = False
    self._closed = True

  def _upload_asset_while_idle(self) -> None:
    if self._assets_uploaded:
      return
    self._revalidate_native_fingerprint()
    asset_indexes = tuple(sorted({item.arg.index for item in self.native.relocs if item.arg.kind is RKBufferKind.ASSET}))
    if asset_indexes:
      try:
        for index in asset_indexes:
          self._sync(self._asset_buffers[index], rk.RKNPU_MEM_SYNC_TO_DEVICE)
      except PhysicalRuntimeReject:
        raise
      except KeyError as exc:
        self._reject("asset", "immutable asset allocation is missing before idle upload")
        raise AssertionError from exc
      except Exception as exc:
        self._reject("asset", f"immutable asset sync failed: {exc}")
      self.telemetry.asset_bytes += sum(self.native.assets[index].size for index in asset_indexes)
      self._assets_uploaded = True
      self._event("asset_upload_idle")
      return
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
    self._assets_uploaded = True
    self._event("asset_upload_idle")

  def reset_before(self) -> None:
    if self._reset or self._in_flight:
      self._reject("lifecycle", "native reset is not first or one-shot")
    self._prepare()
    self._upload_asset_while_idle()
    if self.kind is RKNativeKind.CMAC:
      self._sync(self._resource_buffers["lhs"], rk.RKNPU_MEM_SYNC_TO_DEVICE)
      if not self._cmac_asset_mode:
        self._sync(self._resource_buffers["rhs"], rk.RKNPU_MEM_SYNC_TO_DEVICE)
    else:
      self._sync(self._resource_buffers["input"], rk.RKNPU_MEM_SYNC_TO_DEVICE)
      # The poison guard is host-initialized and must be handed to the device
      # before reset/submit just like the EXP2 input and command/task buffers.
      self._sync(self._resource_buffers["output"], rk.RKNPU_MEM_SYNC_TO_DEVICE)
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
    self._revalidate_native_fingerprint()
    self._revalidate_owned_buffers()
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
    if type(receipt.status) is not int or type(receipt.ownership_known) is not bool or not receipt.ownership_known:
      self._unknown = True
      self.telemetry.reject("submit_unknown")
      raise PhysicalOwnershipUnknown("physical submit ownership is unknown")
    if receipt.status != 0:
      self._in_flight = False
      self._cleanup()
      raise PhysicalRuntimeReject("submit", f"physical submit returned {receipt.status}")
    self.telemetry.native_submit += 1
    if receipt.submit_id is not None:
      self.telemetry.native_submit_ids.append(receipt.submit_id)
    return receipt

  def _validate_cmac_output(self) -> None:
    raw = _read(self._resource_buffers["output"], rkp.CMAC_V1_OUTPUT_VIEW_BYTES)
    self.last_output = raw
    self.last_logical_output = cmac_logical_output(raw, _CMAC_LOGICAL_LANES, rkp.CMAC_V1_OUTPUT_SWIZZLE)

  def _validate_cmac_guards(self) -> None:
    if self.kind is not RKNativeKind.CMAC:
      return
    expected = bytes([_CMAC_GUARD_FILL])
    for attr, offset, size in _CMAC_GUARD_SPANS:
      buffer = self._require_buffer(attr)
      if _read(buffer, size, offset) != expected * size:
        self._reject("cmac_guard", f"{attr} guard was modified")

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
      # The driver boundary has been crossed.  Any owned-buffer or caller
      # metadata mismatch is therefore terminal unknown ownership, and must
      # be caught before a FROM_DEVICE sync or output readback.
      self._revalidate_native_fingerprint()
      self._revalidate_owned_buffers()
      self._revalidate_bindings()
      if self.kind is RKNativeKind.CMAC:
        self._sync(self._require_buffer("_cmd_buf"), rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        self._sync(self._require_buffer("_task_buf"), rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      self._sync(self._resource_buffers["output"], rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      if self.kind is RKNativeKind.CMAC:
        self._validate_cmac_guards()
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
