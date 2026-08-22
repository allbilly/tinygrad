import base64, hashlib, struct
from dataclasses import replace
from typing import cast

import pytest

from tinygrad.device import TinyELF
from tinygrad.helpers import Target
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKImage, RKNativeAsset, RKNativeGuard, RKNativeKind,
  RKNativeOp, RKNativeRelocation, RKNativeRepair, RKNativeRepairKind, RKNativeReset, RKNativeSpan, RKNativeSpanKind,
  RKNativeSubmit, RKNativeTask, RKTarget, RK_EXP2_PHYSICAL_PROVENANCE, RK_EXP2_REPAIR_DEVICE_STAGE,
  RK_EXP2_REPAIR_METADATA, RockchipCompiler, decode_image, encode_image)
from tinygrad.runtime.autogen import rockchip_physical as rkp
from tinygrad.runtime.ops_rockchip import RockchipDevice, RockchipProgram
from tinygrad.runtime.support.hcq import HCQBuffer


def _cmac_image() -> RKImage:
  activation, weights, output = (RKArg(RKBufferKind.ARG, index) for index in range(3))
  refs = (activation, weights, output)
  native = RKNativeOp(
    RKNativeKind.CMAC, rkp.CMAC_V1_COMMANDS,
    tuple(RKNativeRelocation(word, target, register, arg) for (word, target, register), arg in zip(rkp.CMAC_V1_RELOCATIONS, refs)),
    reads=refs[:2], writes=(output,), outputs=(output,), tail=rkp.CMAC_V1_TAIL,
    task=RKNativeTask(*rkp.CMAC_V1_TASK), submit=RKNativeSubmit(*rkp.CMAC_V1_SUBMIT), reset=RKNativeReset(*rkp.CMAC_V1_RESET))
  return RKImage(RKTarget.RK3588, version=32, native=native)


def _exp2_image() -> RKImage:
  input_, output = RKArg(RKBufferKind.ARG, 0), RKArg(RKBufferKind.ARG, 1)
  native = RKNativeOp(
    RKNativeKind.LUT, rkp.LUT_V1_EXP2_COMMANDS,
    (RKNativeRelocation(1032, 0x1001, 0x4020, output), RKNativeRelocation(1059, 0x2001, 0x5018, input_)),
    reads=(input_,), writes=(output,), outputs=(output,),
    assets=(RKNativeAsset(rkp.LUT_V1_EXP2_ASSET_ID, bytes.fromhex(rkp.LUT_V1_EXP2_TABLE_SHA256), rkp.LUT_V1_EXP2_TABLE_BYTES,
                          ((0, 1026), (1026, 1026)), payload=rkp.LUT_V1_EXP2_TABLE),),
    guards=(RKNativeGuard(output, rkp.LUT_V1_EXP2_OUTPUT_BYTES, rkp.LUT_V1_EXP2_GUARD_BYTES, rkp.LUT_V1_EXP2_GUARD_FILL),),
    repairs=tuple(RKNativeRepair(RKNativeRepairKind.SPECIAL_VALUE, index + 1, index, index + 1, True, name,
                                 input_, output, RK_EXP2_PHYSICAL_PROVENANCE, RK_EXP2_REPAIR_DEVICE_STAGE)
                   for index,name in enumerate(RK_EXP2_REPAIR_METADATA)),
    task=RKNativeTask(op_index=4, enable_mask=0x18, interrupt_mask=0x300, interrupt_clear=0x1FFFF,
                      interrupt_status=0, regcfg_amount=1064, regcfg_offset=0, reserved=0),
    submit=RKNativeSubmit(*rkp.LUT_V1_EXP2_SUBMIT), reset=RKNativeReset(*rkp.LUT_V1_EXP2_RESET),
    flags=rkp.LUT_V1_EXP2_REQUIRED_CONTROLS, spans=(
      RKNativeSpan(input_, RKNativeSpanKind.INPUT, 0, rkp.LUT_V1_EXP2_INPUT_BYTES,
                   rkp.LUT_V1_EXP2_INPUT_ALLOCATION_BYTES, provenance=RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(output, RKNativeSpanKind.OUTPUT_LOGICAL, 0, rkp.LUT_V1_EXP2_INPUT_BYTES,
                   rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, provenance=RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(output, RKNativeSpanKind.OUTPUT_PHYSICAL, 0, rkp.LUT_V1_EXP2_OUTPUT_BYTES,
                   rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, provenance=RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(output, RKNativeSpanKind.OUTPUT_PADDING, rkp.LUT_V1_EXP2_PADDING_OFFSET, rkp.LUT_V1_EXP2_PADDING_BYTES,
                   rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, rkp.LUT_V1_EXP2_PADDING_FILL, 2, RK_EXP2_PHYSICAL_PROVENANCE),
      RKNativeSpan(output, RKNativeSpanKind.OUTPUT_GUARD, rkp.LUT_V1_EXP2_OUTPUT_BYTES, rkp.LUT_V1_EXP2_GUARD_BYTES,
                   rkp.LUT_V1_EXP2_OUTPUT_ALLOCATION_BYTES, rkp.LUT_V1_EXP2_GUARD_FILL, 1, RK_EXP2_PHYSICAL_PROVENANCE),
    ))
  return RKImage(RKTarget.RK3588, version=32, native=native)


@pytest.mark.parametrize("factory", (_cmac_image, _exp2_image))
def test_native_image_is_exact_wire_cache_value(factory):
  image = factory()
  encoded = encode_image(image)
  decoded = decode_image(encoded)
  assert decoded == image
  assert encode_image(decoded) == encoded
  assert RockchipCompiler().compile(base64.b64encode(encoded).decode()) == encoded


def test_native_payload_and_relocation_mutation_are_rejected_before_encoding():
  image = _exp2_image()
  native = image.native
  assert native is not None
  asset = native.assets[0]
  bad_payload = replace(asset, payload=bytes((asset.payload[0] ^ 1,)) + asset.payload[1:])
  with pytest.raises(ValueError, match="payload"):
    encode_image(replace(image, native=replace(native, assets=(bad_payload,))))

  image = _cmac_image()
  native = image.native
  assert native is not None
  bad_reloc = replace(native.relocs[0], register=0x1110)
  with pytest.raises(ValueError, match="relocation"):
    encode_image(replace(image, native=replace(native, relocs=(bad_reloc,) + native.relocs[1:])))


def test_native_lut_cannot_omit_its_embedded_asset():
  image = _exp2_image()
  native = image.native
  assert native is not None
  with pytest.raises(ValueError, match="embedded asset"):
    encode_image(replace(image, native=replace(native, assets=())))


def test_native_decoder_rejects_trailing_and_truncated_bytes():
  encoded = encode_image(_cmac_image())
  with pytest.raises(ValueError): decode_image(encoded + b"trailing")
  with pytest.raises(ValueError): decode_image(encoded[:-1])


def test_native_wire_rejects_non_boolean_repair_fallback():
  encoded = encode_image(_exp2_image())
  marker = struct.pack("<HBBIII", 1, 1, 0, 1, 0, 1)
  offset = encoded.index(marker)
  malformed = bytearray(encoded)
  malformed[offset + 2] = 2
  with pytest.raises(ValueError, match="repair flags"):
    decode_image(bytes(malformed))


def test_native_canonical_validator_rejects_aliases_and_exact_type_violations():
  image = _cmac_image()
  native = image.native
  assert native is not None
  aliased_relocs = (native.relocs[0], replace(native.relocs[1], arg=native.outputs[0]), native.relocs[2])
  with pytest.raises(ValueError, match="duplicate native CMAC references"):
    encode_image(replace(image, native=replace(native, reads=(native.reads[0], native.outputs[0]), relocs=aliased_relocs)))
  with pytest.raises(ValueError, match="index"):
    encode_image(replace(image, native=replace(native, reads=(RKArg(RKBufferKind.ARG, True), native.reads[1]))))
  with pytest.raises(ValueError, match="lifecycle"):
    encode_image(replace(image, native=replace(native, submit=RKNativeSubmit(4, 6000, 1, -1, 1))))
  with pytest.raises(ValueError, match="lifecycle"):
    encode_image(replace(image, native=replace(native, reset=RKNativeReset(1, 0))))
  bad_commands = (native.commands[0] ^ 1,) + native.commands[1:]
  with pytest.raises(ValueError, match="command template"):
    encode_image(replace(image, native=replace(native, commands=bad_commands)))


def test_native_exp2_task_matches_detached_named_wire_contract():
  image = _exp2_image()
  native = image.native
  assert native is not None
  expected = RKNativeTask(op_index=4, enable_mask=0x18, interrupt_mask=0x300, interrupt_clear=0x1FFFF,
                          interrupt_status=0, regcfg_amount=1064, regcfg_offset=0, reserved=0)
  assert native.task == expected
  assert rkp.LUT_V1_EXP2_TASK == (4, 0x18, 0x300, 0x1FFFF, 0, 1064, 0, 0)
  assert decode_image(encode_image(image)) == image


def test_native_exp2_rejects_shifted_task_and_controls():
  image = _exp2_image()
  native = image.native
  assert native is not None
  shifted = RKNativeTask(0, 4, 0x18, 0x300, 0x1FFFF, 0, 1064, 0)
  with pytest.raises(ValueError, match="lifecycle"):
    encode_image(replace(image, native=replace(native, task=shifted)))
  with pytest.raises(ValueError, match="controls"):
    encode_image(replace(image, native=replace(native, flags=0)))


@pytest.mark.parametrize("ranges", (((0, 2052),), ((0, 1025), (1025, 1027))))
def test_native_exp2_requires_complete_two_bank_asset_coverage(ranges):
  image = _exp2_image()
  native = image.native
  assert native is not None
  asset = replace(native.assets[0], ranges=ranges)
  with pytest.raises(ValueError, match="asset contract"):
    encode_image(replace(image, native=replace(native, assets=(asset,))))


def test_native_exp2_rejects_tampered_asset_with_recomputed_digest_and_guard_metadata():
  image = _exp2_image()
  native = image.native
  assert native is not None
  asset = native.assets[0]
  payload = bytes((asset.payload[0] ^ 1,)) + asset.payload[1:]
  bad_asset = replace(asset, digest=hashlib.sha256(payload).digest(), payload=payload)
  with pytest.raises(ValueError, match="asset contract"):
    encode_image(replace(image, native=replace(native, assets=(bad_asset,))))
  bad_guard = replace(native.guards[0], offset=255)
  with pytest.raises(ValueError, match="output guard"):
    encode_image(replace(image, native=replace(native, guards=(bad_guard,))))


class _HostOnlyDevice:
  def __init__(self): self.events:list[str] = []
  def _touch_program(self, _program): self.events.append("touch")
  def _forget_program(self, _program): self.events.append("forget")
  def __getattr__(self, name):
    def unexpected(*_args, **_kwargs):
      self.events.append(name)
      raise AssertionError(f"unexpected native device effect: {name}")
    return unexpected


def test_native_runtime_preflights_then_fails_closed_before_device_effects():
  dev = _HostOnlyDevice()
  program = RockchipProgram(cast(RockchipDevice, dev), TinyELF(encode_image(_cmac_image()), "native", Target(), ()))
  with pytest.raises(RuntimeError, match="buffer_binding"):
    program(*(cast(HCQBuffer, object()) for _ in range(3)))
  assert dev.events == ["touch", "touch"]


def test_native_runtime_rechecks_asset_hash_and_canonical_metadata_before_effects():
  dev = _HostOnlyDevice()
  program = RockchipProgram(cast(RockchipDevice, dev), TinyELF(encode_image(_exp2_image()), "native", Target(), ()))
  assert program.image.native is not None
  object.__setattr__(program.image.native.assets[0], "payload", b"x" * 2052)
  with pytest.raises(RuntimeError, match="hash mismatch"):
    program(*(cast(HCQBuffer, object()) for _ in range(2)))
  assert dev.events == ["touch", "touch"]
