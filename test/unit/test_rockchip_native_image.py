import base64, hashlib
from dataclasses import replace

import pytest

from tinygrad.device import TinyELF
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKImage, RKNativeAsset, RKNativeGuard, RKNativeKind,
  RKNativeOp, RKNativeRelocation, RKNativeRepair, RKNativeRepairKind, RKNativeReset, RKNativeSubmit, RKNativeTask,
  RKTarget, RockchipCompiler, decode_image, encode_image)
from tinygrad.runtime.ops_rockchip import RockchipProgram


def _image() -> RKImage:
  source = RKArg(RKBufferKind.ARG, 0)
  output = RKArg(RKBufferKind.ARG, 1)
  payload = bytes(range(16))
  commands = ((0x1001 << 48) | (0x1234 << 16) | 0x1070, (0x1001 << 48) | (0x5678 << 16) | 0x1110)
  native = RKNativeOp(
    RKNativeKind.CMAC, commands, (RKNativeRelocation(0, 0x1001, 0x1070, source),),
    reads=(source,), writes=(output,), outputs=(output,), tail=(0x2001000000000000,),
    assets=(RKNativeAsset(7, hashlib.sha256(payload).digest(), len(payload), ((0, 8), (8, 8)), payload=payload),),
    guards=(RKNativeGuard(output, 0, 4, 0),),
    repairs=(RKNativeRepair(RKNativeRepairKind.SPECIAL_VALUE, 0xffff, 0x7c00, 0, True),),
    task=RKNativeTask(0, 4, 0x18, 0x300, 0x1ffff, len(commands), 0),
    submit=RKNativeSubmit(5, 6000, 1, -1, 1), reset=RKNativeReset(6, 0))
  return RKImage(RKTarget.RK3588, constants=b"constants", version=32, native=native)


def test_native_image_is_exact_wire_cache_value():
  image = _image()
  encoded = encode_image(image)
  decoded = decode_image(encoded)
  assert decoded == image
  assert encode_image(decoded) == encoded
  assert RockchipCompiler().compile(base64.b64encode(encoded).decode()) == encoded


def test_native_payload_and_relocation_mutation_are_rejected_before_encoding():
  image = _image()
  native = image.native
  assert native is not None
  bad_payload = replace(native.assets[0], payload=b"x" * native.assets[0].size)
  with pytest.raises(ValueError, match="payload"):
    encode_image(replace(image, native=replace(native, assets=(bad_payload,))))
  bad_reloc = replace(native.relocs[0], register=0x1110)
  with pytest.raises(ValueError, match="relocation"):
    encode_image(replace(image, native=replace(native, relocs=(bad_reloc,))))


def test_native_lut_cannot_omit_its_embedded_asset():
  image = _image()
  native = image.native
  assert native is not None
  with pytest.raises(ValueError, match="embedded asset"):
    encode_image(replace(image, native=replace(native, kind=RKNativeKind.LUT, assets=())))


def test_native_decoder_rejects_trailing_and_truncated_bytes():
  encoded = encode_image(_image())
  with pytest.raises(ValueError): decode_image(encoded + b"trailing")
  with pytest.raises(ValueError): decode_image(encoded[:-1])


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
  program = RockchipProgram(dev, TinyELF(encode_image(_image()), "native", None, ()))
  with pytest.raises(RuntimeError, match="native execution effects are not implemented"):
    program(object(), object())
  assert dev.events == ["touch", "touch"]


def test_native_runtime_rechecks_asset_hash_before_effects():
  dev = _HostOnlyDevice()
  program = RockchipProgram(dev, TinyELF(encode_image(_image()), "native", None, ()))
  assert program.image.native is not None
  object.__setattr__(program.image.native.assets[0], "payload", b"x" * 16)
  with pytest.raises(RuntimeError, match="hash mismatch"):
    program(object(), object())
  assert dev.events == ["touch", "touch"]
