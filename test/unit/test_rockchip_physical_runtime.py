"""Host-only lifecycle tests for direct frozen Rockchip native effects."""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass, replace

import pytest

from tinygrad.device import TinyELF
from tinygrad.helpers import Target
import tinygrad.renderer.rockchip as rk
import tinygrad.runtime.rockchip_physical_runtime as physical_runtime
from tinygrad.runtime.autogen import rockchip as driver
from tinygrad.runtime.autogen import rockchip_physical as rkp
from tinygrad.runtime.rockchip_physical_runtime import (
  PhysicalOwnershipUnknown,
  PhysicalRuntimeReject,
  PhysicalSubmitReceipt,
  RockchipPhysicalEffects,
  cmac_logical_output,
)
from tinygrad.runtime.ops_rockchip import RockchipProgram
from tinygrad.runtime.support.hcq import HCQBuffer


@dataclass
class _Meta:
  dma_addr: int
  obj_addr: int
  size: int
  handle: int = 1


class _SpyDevice:
  native_reset_live_proven = True

  def __init__(self, *, alias_dma: bool = False, corrupt_guard: bool = False, fail_free: bool = False,
               fail_second_alloc: bool = False):
    self.next_dma = 0x00100000
    self.alias_dma, self.corrupt_guard, self.fail_free, self.fail_second_alloc = alias_dma, corrupt_guard, fail_free, fail_second_alloc
    self.events: list[object] = []
    self.submit_count = self.task_count = self.timeout_retries = 0
    self.storage: list[ctypes.Array[ctypes.c_char]] = []
    self.freed: list[tuple[int, bytes]] = []

  def _touch_program(self, _program: object) -> None:
    self.events.append(("touch",))

  def _forget_program(self, _program: object) -> None:
    self.events.append(("forget",))

  def buffer(self, size: int, *, fill: bytes = b"") -> HCQBuffer:
    allocation = max(4096, (size + 4095) & -4096)
    storage = ctypes.create_string_buffer(allocation)
    if fill:
      ctypes.memmove(ctypes.addressof(storage), fill, min(len(fill), allocation))
    self.storage.append(storage)
    dma = 0x00100000 if self.alias_dma else self.next_dma
    self.next_dma += allocation + 0x1000
    return HCQBuffer(ctypes.addressof(storage), size, _Meta(dma, dma, allocation))

  def _gpu_alloc(self, size: int, flags: int = 0) -> HCQBuffer:
    if self.fail_second_alloc and flags == driver.RKNPU_MEM_KERNEL_MAPPING:
      self.events.append(("alloc_fail", size, flags))
      raise MemoryError("spy second allocation failure")
    result = self.buffer(size)
    self.events.append(("alloc", size, flags, result.meta.dma_addr))
    return result

  def _gpu_free(self, buffer: HCQBuffer) -> None:
    self.freed.append((buffer.size, ctypes.string_at(int(buffer.va_addr), min(buffer.size, 4608))))
    self.events.append(("free", buffer.size))
    if self.fail_free:
      raise OSError("spy free failure")

  def _sync_buffer(self, buffer: HCQBuffer, flags: int) -> None:
    self.events.append(("sync", buffer.meta.dma_addr, flags))

  def _upload_native_asset(self, payload: bytes, ranges: tuple[tuple[int, int], ...]) -> None:
    self.events.append(("asset", payload, ranges))

  def reset_npu(self) -> None:
    self.events.append(("reset",))


class _SpyProgram:
  def __init__(self, native: rk.RKNativeOp, *, fail_submit: bool = False, alias_dma: bool = False, corrupt_guard: bool = False,
               fail_second_alloc: bool = False):
    self.dev = _SpyDevice(alias_dma=alias_dma, corrupt_guard=corrupt_guard, fail_second_alloc=fail_second_alloc)
    self.image = rk.RKImage(rk.RKTarget.RK3588, version=32, native=native)
    self.native, self.fail_submit = native, fail_submit
    self.effects: RockchipPhysicalEffects | None = None
    self.submit_contracts: list[object] = []

  def _dma(self, buffer: HCQBuffer) -> int:
    return int(buffer.meta.dma_addr)

  def _submit_physical(self, command: HCQBuffer, task: HCQBuffer, contract: object) -> PhysicalSubmitReceipt:
    del command, task
    self.submit_contracts.append(contract)
    self.dev.events.append(("submit", contract))
    if self.fail_submit:
      raise TimeoutError("spy submit timeout")
    assert self.effects is not None
    output = self.effects._resource_buffers["output"]
    if self.effects.kind is rk.RKNativeKind.CMAC:
      ctypes.memmove(int(output.va_addr), struct.pack("<64H", *range(64)), rkp.CMAC_V1_OUTPUT_VIEW_BYTES)
    else:
      ctypes.memset(int(output.va_addr), 0, rkp.LUT_V1_EXP2_OUTPUT_BYTES)
      ctypes.memmove(int(output.va_addr) + rkp.LUT_V1_EXP2_PADDING_OFFSET,
                     struct.pack("<H", rkp.LUT_V1_EXP2_PADDING_FILL) * (rkp.LUT_V1_EXP2_PADDING_BYTES // 2),
                     rkp.LUT_V1_EXP2_PADDING_BYTES)
      if self.dev.corrupt_guard:
        ctypes.memset(int(output.va_addr) + rkp.LUT_V1_EXP2_OUTPUT_BYTES, 0x5A, 1)
    return PhysicalSubmitReceipt(0, True, 77)


def _cmac_native() -> rk.RKNativeOp:
  lhs, rhs, output = (rk.RKArg(rk.RKBufferKind.ARG, index) for index in range(3))
  relocs = tuple(
    rk.RKNativeRelocation(word, target, register, arg) for (word, target, register), arg in zip(rkp.CMAC_V1_RELOCATIONS, (lhs, rhs, output))
  )
  return rk.RKNativeOp(
    rk.RKNativeKind.CMAC,
    rkp.CMAC_V1_COMMANDS,
    relocs,
    (lhs, rhs),
    (output,),
    (output,),
    rkp.CMAC_V1_TAIL,
    task=rk.RKNativeTask(*rkp.CMAC_V1_TASK),
    submit=rk.RKNativeSubmit(*rkp.CMAC_V1_SUBMIT),
    reset=rk.RKNativeReset(*rkp.CMAC_V1_RESET),
  )


def _exp2_native() -> rk.RKNativeOp:
  input_arg, output = rk.RKArg(rk.RKBufferKind.ARG, 0), rk.RKArg(rk.RKBufferKind.ARG, 1)
  digest = bytes.fromhex(rkp.LUT_V1_EXP2_TABLE_SHA256)
  asset = rk.RKNativeAsset(rkp.LUT_V1_EXP2_ASSET_ID, digest, rkp.LUT_V1_EXP2_TABLE_BYTES, ((0, 1026), (1026, 1026)), payload=rkp.LUT_V1_EXP2_TABLE)
  relocs = tuple(
    rk.RKNativeRelocation(word, target, register, arg) for (word, target, register), arg in zip(rkp.LUT_V1_EXP2_RELOCATIONS, (output, input_arg))
  )
  repairs = tuple(rk.RKNativeRepair(rk.RKNativeRepairKind.SPECIAL_VALUE, index + 1, index, index + 1, True, name,
                                    input_arg, output, rk.RK_EXP2_PHYSICAL_PROVENANCE, rk.RK_EXP2_REPAIR_DEVICE_STAGE)
                  for index,name in enumerate(rk.RK_EXP2_REPAIR_METADATA))
  return rk.RKNativeOp(
    rk.RKNativeKind.LUT,
    rkp.LUT_V1_EXP2_COMMANDS,
    relocs,
    (input_arg,),
    (output,),
    (output,),
    assets=(asset,),
    guards=(rk.RKNativeGuard(output, rkp.LUT_V1_EXP2_OUTPUT_BYTES, rkp.LUT_V1_EXP2_GUARD_BYTES, rkp.LUT_V1_EXP2_GUARD_FILL),),
    repairs=repairs,
    task=rk.RKNativeTask(*rkp.LUT_V1_EXP2_TASK),
    submit=rk.RKNativeSubmit(*rkp.LUT_V1_EXP2_SUBMIT),
    reset=rk.RKNativeReset(*rkp.LUT_V1_EXP2_RESET),
    flags=rkp.LUT_V1_EXP2_REQUIRED_CONTROLS,
    spans=(
      rk.RKNativeSpan(input_arg, rk.RKNativeSpanKind.INPUT, 0, rkp.LUT_V1_EXP2_INPUT_BYTES,
                      rkp.LUT_V1_EXP2_INPUT_ALLOCATION_BYTES, provenance=rk.RK_EXP2_PHYSICAL_PROVENANCE),
      rk.RKNativeSpan(output, rk.RKNativeSpanKind.OUTPUT_LOGICAL, 0, rkp.LUT_V1_EXP2_INPUT_BYTES,
                      rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, provenance=rk.RK_EXP2_PHYSICAL_PROVENANCE),
      rk.RKNativeSpan(output, rk.RKNativeSpanKind.OUTPUT_PHYSICAL, 0, rkp.LUT_V1_EXP2_OUTPUT_BYTES,
                      rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, provenance=rk.RK_EXP2_PHYSICAL_PROVENANCE),
      rk.RKNativeSpan(output, rk.RKNativeSpanKind.OUTPUT_PADDING, rkp.LUT_V1_EXP2_PADDING_OFFSET,
                      rkp.LUT_V1_EXP2_PADDING_BYTES, rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES,
                      rkp.LUT_V1_EXP2_PADDING_FILL, 2, rk.RK_EXP2_PHYSICAL_PROVENANCE),
      rk.RKNativeSpan(output, rk.RKNativeSpanKind.OUTPUT_GUARD, rkp.LUT_V1_EXP2_OUTPUT_BYTES,
                      rkp.LUT_V1_EXP2_GUARD_BYTES, rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES,
                      rkp.LUT_V1_EXP2_GUARD_FILL, 1, rk.RK_EXP2_PHYSICAL_PROVENANCE),
    ),
  )


def _setup(native: rk.RKNativeOp, *, reset_live_proven: bool = True, fail_submit: bool = False, alias_dma: bool = False,
           corrupt_guard: bool = False, fail_free: bool = False):
  program = _SpyProgram(native, fail_submit=fail_submit, alias_dma=alias_dma, corrupt_guard=corrupt_guard)
  program.dev.fail_free = fail_free
  program.dev.native_reset_live_proven = reset_live_proven
  count = max((arg.index for arg in native.reads + native.outputs), default=-1) + 1
  size = 4096 if native.kind is rk.RKNativeKind.LUT else 2048
  bufs = tuple(program.dev.buffer(size) for _ in range(count))
  if native.kind is rk.RKNativeKind.CMAC:
    ctypes.memset(int(bufs[1].va_addr) + 256, 0, 1792)
  effects = RockchipPhysicalEffects(program, program.image, bufs, native)
  program.effects = effects
  return program, effects


def test_direct_effects_reject_old_plan_shapes_before_device_effects() -> None:
  with pytest.raises(PhysicalRuntimeReject, match="RKNativeOp"):
    RockchipPhysicalEffects(object(), rk.RKImage(rk.RKTarget.RK3588), (), None)


def test_cmac_patches_all_dma_sites_keeps_rhs_tail_and_swizzles_output() -> None:
  program, effects = _setup(_cmac_native())
  receipt = effects.execute()
  assert receipt == PhysicalSubmitReceipt(0, True, 77)
  assert effects.telemetry.events == [
    "cmac_v1.attempt",
    "cmac_v1.reset_action_6",
    "cmac_v1.barrier_before",
    "cmac_v1.submit_one",
    "cmac_v1.barrier_after",
    "cmac_v1.success",
  ]
  names = [event[0] for event in program.dev.events if isinstance(event, tuple)]
  assert names.count("reset") == 1 and names.count("submit") == 1 and len(program.dev.freed) == 2
  command = next(data for size, data in program.dev.freed if size == rkp.CMAC_V1_COMMAND_RESERVATION_BYTES)
  task = next(data for size, data in program.dev.freed if size == 4096)
  command_dma = next(event[3] for event in program.dev.events if event[0] == "alloc" and event[1] == rkp.CMAC_V1_COMMAND_RESERVATION_BYTES)
  assert struct.unpack_from("<8IQ", task) == (*rkp.CMAC_V1_TASK, command_dma)
  patched = struct.unpack_from("<46Q", command)
  assert all((patched[word] >> 16) & 0xFFFFFFFF != 0 for word in (18, 24, 31))
  assert effects.last_logical_output == struct.pack("<4H", 0, 1, 2, 3)


def test_real_rockchip_program_runs_frozen_cmac_with_spy_submit() -> None:
  native = _cmac_native()
  dev = _SpyDevice()
  image = rk.RKImage(rk.RKTarget.RK3588, version=32, native=native)
  program = RockchipProgram(dev, TinyELF(rk.encode_image(image), "cmac", Target(), ()))
  bufs = tuple(dev.buffer(size) for size in (64, 2048, 256))
  ctypes.memset(int(bufs[1].va_addr) + 256, 0, 1792)

  def submit(_command: HCQBuffer, _task: HCQBuffer, contract: object) -> PhysicalSubmitReceipt:
    dev.events.append(("submit", contract))
    ctypes.memmove(int(bufs[2].va_addr), struct.pack("<64H", *range(64)), rkp.CMAC_V1_OUTPUT_VIEW_BYTES)
    return PhysicalSubmitReceipt(0, True, 91)

  program._submit_physical = submit  # type: ignore[method-assign]
  assert program(*bufs, wait=True) is not None
  names = [event[0] for event in dev.events if isinstance(event, tuple)]
  assert names.count("reset") == 1 and names.count("submit") == 1 and names.count("free") == 2


def test_exp2_upload_repair_padding_guard_and_cleanup_order() -> None:
  program, effects = _setup(_exp2_native())
  receipt = effects.execute()
  assert receipt.submit_id == 77
  assert effects.telemetry.events == [
    "exp2_lut_v1.attempt",
    "exp2_lut_v1.asset_upload_idle",
    "exp2_lut_v1.reset_action_6",
    "exp2_lut_v1.barrier_before",
    "exp2_lut_v1.submit_one",
    "exp2_lut_v1.barrier_after",
    "exp2_lut_v1.success",
  ]
  names = [event[0] for event in program.dev.events if isinstance(event, tuple)]
  assert names.index("asset") < names.index("reset") < names.index("submit") < names.index("free")
  asset_event = next(event for event in program.dev.events if event[0] == "asset")
  assert asset_event[1] == rkp.LUT_V1_EXP2_TABLE and asset_event[2] == ((0, 1026), (1026, 1026))
  input_dma = effects._dma_by_arg[effects.native.reads[0]]
  input_sync = ("sync", input_dma, driver.RKNPU_MEM_SYNC_TO_DEVICE)
  assert input_sync in program.dev.events and program.dev.events.index(input_sync) < names.index("reset")


def test_exp2_readback_has_no_host_numeric_repair_or_input_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
  program, effects = _setup(_exp2_native())
  effects._preflight()
  input_buffer, output_buffer = effects._resource_buffers["input"], effects._resource_buffers["output"]
  original_read, original_write = physical_runtime._read, physical_runtime._write
  reads: list[tuple[HCQBuffer, int, int]] = []
  writes: list[tuple[HCQBuffer, bytes, int]] = []

  def read_spy(buffer: HCQBuffer, size: int, offset: int = 0) -> bytes:
    if buffer is input_buffer:
      raise AssertionError("EXP2 runtime inspected input bytes")
    reads.append((buffer, size, offset))
    return original_read(buffer, size, offset)

  def write_spy(buffer: HCQBuffer, data: bytes, offset: int = 0) -> None:
    writes.append((buffer, data, offset))
    original_write(buffer, data, offset)

  monkeypatch.setattr(physical_runtime, "_read", read_spy)
  monkeypatch.setattr(physical_runtime, "_write", write_spy)
  effects.execute(preflight=False)
  assert reads == [(output_buffer, rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, 0)]
  assert all(buffer is not output_buffer for buffer,_,_ in writes)
  assert effects.last_output == bytes(rkp.LUT_V1_EXP2_INPUT_BYTES) + \
    struct.pack("<H", rkp.LUT_V1_EXP2_PADDING_FILL) * (rkp.LUT_V1_EXP2_PADDING_BYTES // 2)


def test_exp2_missing_device_repair_provenance_is_zero_effect_reject() -> None:
  native = _exp2_native()
  bad_repairs = tuple(replace(rule, name="", provenance="", device_stage="") for rule in native.repairs)
  program, effects = _setup(replace(native, repairs=bad_repairs))
  with pytest.raises(PhysicalRuntimeReject, match="span_contract|exp2_repair"):
    effects.execute()
  assert effects.closed and program.dev.events == []


def test_exp2_asset_flags_are_zero_effect_reject() -> None:
  native = _exp2_native()
  program, effects = _setup(replace(native, assets=(replace(native.assets[0], flags=1),)))
  with pytest.raises(PhysicalRuntimeReject, match="asset"):
    effects.execute()
  assert effects.closed and program.dev.events == []


def test_second_native_allocation_is_tracked_and_freed_on_failure() -> None:
  program = _SpyProgram(_cmac_native(), fail_second_alloc=True)
  bufs = tuple(program.dev.buffer(size) for size in (64, 2048, 256))
  ctypes.memset(int(bufs[1].va_addr) + 256, 0, 1792)
  effects = RockchipPhysicalEffects(program, program.image, bufs, program.native)
  program.effects = effects
  with pytest.raises(PhysicalRuntimeReject, match="allocator"):
    effects.execute()
  assert effects.closed and not effects.ownership_unknown
  assert [event[0] for event in program.dev.events if isinstance(event, tuple)] == ["alloc", "alloc_fail", "free"]


def test_non_contract_submit_receipt_is_terminal_unknown() -> None:
  program, effects = _setup(_cmac_native())
  program._submit_physical = lambda _command, _task, _contract: object()  # type: ignore[method-assign]
  with pytest.raises(PhysicalOwnershipUnknown, match="non-contract receipt"):
    effects.execute()
  assert effects.ownership_unknown and not effects.closed and program.dev.freed == []


def test_dma_snapshot_is_revalidated_before_patch_and_submit() -> None:
  program, effects = _setup(_cmac_native())
  effects._preflight()
  effects._resource_buffers["lhs"].meta.dma_addr += 0x1000
  with pytest.raises(PhysicalRuntimeReject, match="dma_binding"):
    effects.execute(preflight=False)
  assert effects.closed and not effects.ownership_unknown
  names = [event[0] for event in program.dev.events if isinstance(event, tuple)]
  assert "reset" not in names and "submit" not in names


def test_native_plan_and_span_snapshot_is_revalidated_before_patch() -> None:
  program, effects = _setup(_exp2_native())
  effects._preflight()
  effects.native = replace(effects.native, spans=())
  with pytest.raises(PhysicalRuntimeReject, match="span_contract"):
    effects.execute(preflight=False)
  assert effects.closed and not effects.ownership_unknown
  assert all(event[0] not in ("reset", "submit", "asset") for event in program.dev.events)


def test_dma_snapshot_revalidation_runs_again_immediately_before_submit() -> None:
  program, effects = _setup(_cmac_native())
  calls = 0
  original = effects._revalidate_bindings

  def spy() -> None:
    nonlocal calls
    calls += 1
    original()

  effects._revalidate_bindings = spy  # type: ignore[method-assign]
  effects.execute()
  assert calls == 2


def test_submit_exception_is_unknown_and_holds_command_and_task() -> None:
  program, effects = _setup(_cmac_native(), fail_submit=True)
  with pytest.raises(PhysicalOwnershipUnknown):
    effects.execute()
  assert effects.ownership_unknown and not effects.closed and program.dev.freed == []
  assert [event[0] for event in program.dev.events if isinstance(event, tuple)].count("reset") == 1


def test_cleanup_failure_is_unknown_and_retains_unfreed_allocations() -> None:
  program, effects = _setup(_cmac_native(), fail_free=True)
  with pytest.raises(PhysicalOwnershipUnknown):
    effects.execute()
  assert effects.ownership_unknown and not effects.closed
  assert effects.command_buffer is not None and effects.task_buffer is not None


def test_unproven_reset_is_fail_closed_and_has_zero_effects() -> None:
  program, effects = _setup(_exp2_native(), reset_live_proven=False)
  with pytest.raises(RuntimeError, match="reset_gate"):
    effects.execute()
  assert not effects.ownership_unknown and effects.closed
  assert program.dev.freed == [] and not program.dev.events


def test_dma_alias_is_rejected_before_allocation_or_reset() -> None:
  program, effects = _setup(_cmac_native(), alias_dma=True)
  with pytest.raises(RuntimeError, match="dma_alias"):
    effects.execute()
  assert not effects.ownership_unknown and effects.closed and not program.dev.events


def test_mutated_embedded_asset_is_rejected_before_allocation_or_upload() -> None:
  native = _exp2_native()
  bad_asset = replace(native.assets[0], payload=b"x" * native.assets[0].size)
  bad = replace(native, assets=(bad_asset,))
  program, effects = _setup(bad)
  with pytest.raises(RuntimeError, match="asset"):
    effects.execute()
  assert effects.closed and program.dev.freed == [] and program.dev.events == []


def test_exp2_repair_policy_mutation_is_rejected_before_allocation() -> None:
  native = _exp2_native()
  bad_rule = replace(native.repairs[0], result_bits=0x7C00)
  program, effects = _setup(replace(native, repairs=(bad_rule,) + native.repairs[1:]))
  with pytest.raises(RuntimeError, match="exp2_repair"):
    effects.execute()
  assert effects.closed and program.dev.freed == [] and program.dev.events == []


def test_output_view_allocation_is_rejected_before_allocation() -> None:
  native = _cmac_native()
  program = _SpyProgram(native)
  bufs = (program.dev.buffer(2048), program.dev.buffer(2048), program.dev.buffer(128))
  effects = RockchipPhysicalEffects(program, program.image, bufs, native)
  with pytest.raises(RuntimeError, match="resource_bounds"):
    effects.execute()
  assert effects.closed and program.dev.events == []


def test_exp2_guard_corruption_is_unknown_after_submit() -> None:
  program, effects = _setup(_exp2_native(), corrupt_guard=True)
  with pytest.raises(PhysicalOwnershipUnknown):
    effects.execute()
  assert effects.ownership_unknown and program.dev.freed == []


def test_cmac_output_swizzle_is_raw_byte_layout_only() -> None:
  raw = struct.pack("<64H", *range(64)) + bytes(128)
  assert cmac_logical_output(raw, 4, tuple(range(16)) + tuple(range(32, 48))) == struct.pack("<4H", 0, 1, 2, 3)


def test_driver_contract_constants_are_action_six_and_one_submit() -> None:
  assert driver.RKNPU_ACT_RESET == 6
  assert rkp.CMAC_V1_RESET == (6, 0) and rkp.LUT_V1_EXP2_RESET == (6, 0)
  assert rkp.CMAC_V1_SUBMIT == rkp.LUT_V1_EXP2_SUBMIT == (0x5, 6000, 1, -1, 1)
