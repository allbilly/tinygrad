import base64
from dataclasses import replace
from typing import cast

import pytest

from tinygrad.device import TinyELF
from tinygrad.helpers import Target
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKImage, RKNativeAsset, RKNativeKind, RKNativeOp,
  RKNativeRelocation, RKNativeReset, RKNativeSubmit, RKNativeTask, RKTarget, RockchipCompiler, decode_image, encode_image)
from tinygrad.runtime.autogen import rockchip_physical as rkp
import tinygrad.runtime.ops_rockchip as rockchip_runtime
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


def _cmac_asset_image() -> RKImage:
  activation, output = RKArg(RKBufferKind.ARG, 0), RKArg(RKBufferKind.ARG, 1)
  asset_ref = RKArg(RKBufferKind.ASSET, 0)
  asset = RKNativeAsset(rkp.CMAC_V1_RHS_ASSET_ID, bytes.fromhex(rkp.CMAC_V1_RHS_ASSET_SHA256), rkp.CMAC_V1_RHS_ASSET_SIZE,
                        rkp.CMAC_V1_RHS_ASSET_RANGES, payload=rkp.CMAC_V1_RHS_ASSET_PAYLOAD)
  native = RKNativeOp(
    RKNativeKind.CMAC, rkp.CMAC_V1_COMMANDS,
    tuple(RKNativeRelocation(word, target, register, arg) for (word, target, register), arg in zip(rkp.CMAC_V1_RELOCATIONS,
                                                                                                    (activation, asset_ref, output))),
    reads=(activation,), writes=(output,), outputs=(output,), tail=rkp.CMAC_V1_TAIL, assets=(asset,),
    task=RKNativeTask(*rkp.CMAC_V1_TASK), submit=RKNativeSubmit(*rkp.CMAC_V1_SUBMIT), reset=RKNativeReset(*rkp.CMAC_V1_RESET))
  return RKImage(RKTarget.RK3588, version=32, native=native)


@pytest.mark.parametrize("factory", (_cmac_image, _cmac_asset_image))
def test_native_image_is_exact_wire_cache_value(factory):
  image = factory()
  encoded = encode_image(image)
  decoded = decode_image(encoded)
  assert decoded == image
  assert encode_image(decoded) == encoded
  assert RockchipCompiler().compile(base64.b64encode(encoded).decode()) == encoded


def test_native_payload_and_relocation_mutation_are_rejected_before_encoding():
  image = _cmac_image()
  native = image.native
  assert native is not None
  bad_reloc = replace(native.relocs[0], register=0x1110)
  with pytest.raises(ValueError, match="relocation"):
    encode_image(replace(image, native=replace(native, relocs=(bad_reloc,) + native.relocs[1:])))


def test_native_decoder_rejects_trailing_and_truncated_bytes():
  encoded = encode_image(_cmac_image())
  with pytest.raises(ValueError): decode_image(encoded + b"trailing")
  with pytest.raises(ValueError): decode_image(encoded[:-1])


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


class _HostOnlyDevice:
  def __init__(self): self.events:list[str] = []
  def _touch_program(self, _program): self.events.append("touch")
  def _forget_program(self, _program): self.events.append("forget")
  def __getattr__(self, name):
    def unexpected(*_args, **_kwargs):
      self.events.append(name)
      raise AssertionError(f"unexpected native device effect: {name}")
    return unexpected


def test_rockchip_device_constructor_has_no_hidden_action6(monkeypatch: pytest.MonkeyPatch):
  reset_calls: list[object] = []

  class FakeFileIO:
    def __init__(self, *_args, **_kwargs): pass

  monkeypatch.setattr(rockchip_runtime, "FileIOInterface", FakeFileIO)
  monkeypatch.setattr(rockchip_runtime.rk, "DRM_IOCTL_RKNPU_ACTION",
                      lambda *args, **kwargs: reset_calls.append((args, kwargs)))
  dev = RockchipDevice("ROCKCHIP")
  assert reset_calls == [] and dev._ordinary_initialized is False


def test_ordinary_program_initialization_is_explicit_and_one_shot():
  class OrdinaryDevice:
    _ordinary_initialized = False
    resets = 0

    def reset_npu(self): self.resets += 1
    def _forget_program(self, _program): pass

  program = object.__new__(RockchipProgram)
  program.dev = OrdinaryDevice()
  program._ensure_ordinary_initialized()
  program._ensure_ordinary_initialized()
  assert program.dev.resets == 1 and program.dev._ordinary_initialized is True


def test_ordinary_v31_program_path_initializes_once():
  class OrdinaryDevice:
    _ordinary_initialized = False
    resets = 0

    def _touch_program(self, _program): pass
    def _forget_program(self, _program): pass
    def _sync_buffers(self, _buffers, _flags): pass
    def reset_npu(self): self.resets += 1

  dev = OrdinaryDevice()
  program = RockchipProgram(dev, TinyELF(encode_image(RKImage(RKTarget.RK3588, version=31)), "ordinary", Target(), ()))
  assert program(wait=True) is not None
  assert program(wait=True) is not None
  assert dev.resets == 1


def test_native_runtime_preflights_then_fails_closed_before_device_effects():
  dev = _HostOnlyDevice()
  program = RockchipProgram(cast(RockchipDevice, dev), TinyELF(encode_image(_cmac_image()), "native", Target(), ()))
  with pytest.raises(RuntimeError, match="buffer_binding"):
    program(*(cast(HCQBuffer, object()) for _ in range(3)))
  assert dev.events == ["touch", "touch"]


def test_native_asset_payload_digest_is_rechecked_before_encoding():
  image = _cmac_asset_image()
  native = image.native
  assert native is not None
  asset = replace(native.assets[0], payload=b"x" * rkp.CMAC_V1_RHS_ASSET_SIZE)
  with pytest.raises(ValueError, match="payload"):
    encode_image(replace(image, native=replace(native, assets=(asset,))))
