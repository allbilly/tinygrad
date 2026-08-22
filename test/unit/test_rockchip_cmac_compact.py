import base64, hashlib, struct
from dataclasses import astuple, replace

import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.rockchip import (RKBufferKind, RKNativeKind, RockchipRenderer, decode_image, encode_image)
from tinygrad.renderer.rockchip_cmac_uops import (CMACConstantAssetCertificate, CMACFallback, CMACReject, CMACRouteCounters, CMACUOpMatch,
  CMAC_RHS_ONE_N4_ASSET, CMAC_RHS_ONE_N4_ASSET_SHA256, CMAC_RHS_ONE_N4_PAYLOAD, is_cmac_physical_image, match_cmac_uops,
  trusted_constant_sum)
from tinygrad.runtime.autogen.rockchip_physical import (CMAC_V1_BODY_QWORDS, CMAC_V1_BODY_SHA256, CMAC_V1_COMMANDS,
  CMAC_V1_OUTPUT_VIEW_BYTES, CMAC_V1_PC_GUARD_BYTES, CMAC_V1_RELOCATIONS, CMAC_V1_RESET, CMAC_V1_SUBMIT, CMAC_V1_TAIL, CMAC_V1_TASK)
from tinygrad.uop.ops import AxisType, Ops


def _ast(*, trusted: bool = True):
  lhs = Tensor.empty((1, 32), dtype=dtypes.half, device="CPU")
  if trusted:
    view = trusted_constant_sum(lhs)
  else:
    output = Tensor.empty((1, 1), dtype=dtypes.half, device="CPU")
    output.assign(lhs.sum(axis=1))
    view = output
  return view.schedule_linear().src[0].src[0]


def _match(ast): return match_cmac_uops(ast.toposort())


def test_asset_oracle_is_fixed_and_exact():
  assert len(CMAC_RHS_ONE_N4_PAYLOAD) == 2048
  assert CMAC_RHS_ONE_N4_ASSET_SHA256 == "96a33b81830614e9b95b033117210b3933d7d971323992d35be7d901cb183c00"
  assert hashlib.sha256(CMAC_RHS_ONE_N4_PAYLOAD).hexdigest() == CMAC_RHS_ONE_N4_ASSET_SHA256
  assert CMAC_RHS_ONE_N4_ASSET.asset_id == 2 and CMAC_RHS_ONE_N4_ASSET.ranges == ((0, 2048),)
  for offset in range(0, 2048, 2):
    assert CMAC_RHS_ONE_N4_PAYLOAD[offset:offset + 2] == (b"\x00\x3c" if offset < 256 else b"\0\0")


def test_real_tensor_reduction_to_program_uses_asset_q24_and_exact_donor():
  to_program_cache.clear()
  renderer = RockchipRenderer(Target.parse("ROCKCHIP"))
  program = to_program(_ast(), renderer)
  blob = next(u for u in program.src if u.op is Ops.BINARY).arg
  image = decode_image(blob)
  assert image.native is not None and image.native.kind is RKNativeKind.CMAC and is_cmac_physical_image(image)
  native = image.native
  assert native.commands == CMAC_V1_COMMANDS and len(native.commands) == CMAC_V1_BODY_QWORDS == 46
  assert CMAC_V1_BODY_SHA256 == "e1a4fb0194156e87680375eab9594f22f9ae545b4be50776819fb8e83c5e4af1"
  assert hashlib.sha256(struct.pack("<46Q", *native.commands)).hexdigest() == CMAC_V1_BODY_SHA256
  assert tuple((r.word_index, r.target, r.register) for r in native.relocs) == CMAC_V1_RELOCATIONS
  assert native.relocs[1].arg == (native.relocs[1].arg.__class__)(RKBufferKind.ASSET, 0)
  assert native.reads == (native.relocs[0].arg,) and native.writes == native.outputs == (native.relocs[2].arg,)
  assert native.assets == (CMAC_RHS_ONE_N4_ASSET,)
  assert native.tail == CMAC_V1_TAIL and astuple(native.task) == CMAC_V1_TASK
  assert astuple(native.submit) == CMAC_V1_SUBMIT and astuple(native.reset) == CMAC_V1_RESET
  assert encode_image(image) == blob and decode_image(blob) == image
  assert len(blob) == 2674 and hashlib.sha256(blob).hexdigest() == "f6cb7e79bc97a31797379711cd3e5662e698cfdc7656abad99af91eb2b5a692b"
  assert renderer.cmac_counters == CMACRouteCounters(attempted=1, admitted=1, native=1)


def test_direct_sum_match_exposes_semantic_oracle_and_raw_view():
  result = _match(_ast())
  assert isinstance(result, CMACUOpMatch) and result.n == 4
  assert result.semantic_provenance.endswith("764c833fdcb22455f344812ab375867c0d8518fe")
  assert result.asset_certificate.raw_output == (128, 256, 128, 4096)
  assert result.output_span_provenance == (128, CMAC_V1_OUTPUT_VIEW_BYTES, 128, CMAC_V1_PC_GUARD_BYTES)
  assert result.native.relocs[1].arg.kind is RKBufferKind.ASSET


def test_ordinary_uninjected_sum_falls_back_before_to_program():
  result = _match(_ast(trusted=False))
  assert isinstance(result, CMACFallback) and result.reason is CMACReject.BOUNDS


def test_public_certificate_constructor_is_untrusted():
  ast = _ast()
  output = next(node for node in ast.toposort() if node.op is Ops.PARAM and node.arg.slot == 0)
  forged = output.replace(arg=replace(output.arg, layout_certificate=CMACConstantAssetCertificate()))
  result = _match(ast.substitute({output: forged}, walk=True))
  assert isinstance(result, CMACFallback) and result.reason is CMACReject.LAYOUT


@pytest.mark.parametrize("field,value", (
  ("asset_id", True), ("digest", b"x" * 32), ("size", True), ("upload_ranges", ((False, 2048),)),
  ("active_ranges", ((0, 255),)), ("tail_ranges", ((256, 1791),)), ("raw_output", (128, 256, 128, True)),
  ("producer", object()),
))
def test_asset_certificate_near_misses_are_fail_closed(field, value):
  good = _match(_ast())
  assert isinstance(good, CMACUOpMatch)
  bad = replace(good.asset_certificate, **{field: value})
  ast = _ast()
  output = next(node for node in ast.toposort() if node.op is Ops.PARAM and node.arg.slot == 0)
  bad_output = output.replace(arg=replace(output.arg, layout_certificate=bad))
  result = _match(ast.substitute({output: bad_output}, walk=True))
  assert result.reason is CMACReject.LAYOUT


def test_wrong_axis_and_reduction_metadata_fall_back():
  ast = _ast()
  axis = next(node for node in ast.toposort() if node.op is Ops.RANGE)
  assert _match(ast.substitute({axis: axis.replace(arg=(0, AxisType.WEAK))})).reason is CMACReject.AXES
  assert _match(ast.substitute({axis: axis.replace(arg=(True, AxisType.REDUCE))})).reason is CMACReject.AXES
  reduction = next(node for node in ast.toposort() if node.op is Ops.REDUCE)
  assert _match(ast.substitute({reduction: reduction.replace(arg=(Ops.ADD, 1))})).reason is CMACReject.FAMILY


def test_short_output_and_nonzero_output_offset_fall_back():
  assert _match(_ast(trusted=False)).reason in (CMACReject.BOUNDS, CMACReject.LAYOUT)
  ast = _ast()
  index = next(node for node in ast.toposort() if node.op is Ops.INDEX and node.src[0].op is Ops.PARAM and node.src[0].arg.slot == 0)
  assert _match(ast.substitute({index: index.replace(src=(index.src[0], index.src[1].replace(arg=1)))})).reason is CMACReject.MAP


def test_non_global_lhs_is_not_a_native_surface():
  ast = _ast()
  lhs = next(node for node in ast.toposort() if node.op is Ops.PARAM and node.arg.slot == 1)
  malformed = lhs.replace(arg=replace(lhs.arg, addrspace=AddrSpace.ALU, device="GPU"))
  result = _match(ast.substitute({lhs: malformed}, walk=True))
  assert isinstance(result, CMACFallback) and result.reason is CMACReject.MAP


def test_dynamic_matmul_never_uses_asset_route():
  lhs = Tensor.empty((1, 32), dtype=dtypes.half, device="CPU")
  rhs = Tensor.empty((32, 32), dtype=dtypes.half, device="CPU")
  output = Tensor.empty((1, 128), dtype=dtypes.half, device="CPU")
  output[:, :4].assign(lhs.matmul(rhs.T)[:, :4])
  result = _match(output.schedule_linear().src[0].src[0])
  assert not isinstance(result, CMACUOpMatch)


def test_fallback_bytes_match_forced_existing_ew(monkeypatch):
  x = Tensor.empty((16,), dtype=dtypes.half, device="CPU")
  ast = (x + 1).schedule_linear().src[0].src[0]
  renderer = RockchipRenderer(Target.parse("ROCKCHIP"))
  routed = renderer.render(list(ast.toposort()))
  monkeypatch.setattr("tinygrad.renderer.rockchip_cmac_uops.try_cmac", lambda *args, **kwargs: None)
  forced = RockchipRenderer(Target.parse("ROCKCHIP")).render(list(ast.toposort()))
  assert routed == forced
  image = decode_image(base64.b64decode(routed))
  assert image.version == 31 and image.native is None
  assert hashlib.sha256(base64.b64decode(routed)).hexdigest() == "b76663c8b53ba480f7bb4f3ac8e6c3766f48fd3608b3390ad8dc1cd8e9bde27f"
  assert renderer.cmac_counters.attempted == 1 and renderer.cmac_counters.admitted == 0
  assert renderer.cmac_counters.native == 0 and renderer.cmac_counters.fallback == 1


def test_untrusted_sum_fallback_bytes_match_existing_ew(monkeypatch):
  ast = _ast(trusted=False)
  renderer = RockchipRenderer(Target.parse("ROCKCHIP"))
  routed = renderer.render(list(ast.toposort()))
  monkeypatch.setattr("tinygrad.renderer.rockchip_cmac_uops.try_cmac", lambda *args, **kwargs: None)
  forced = RockchipRenderer(Target.parse("ROCKCHIP")).render(list(ast.toposort()))
  assert routed == forced
  blob = base64.b64decode(routed)
  assert decode_image(blob).version == 31 and hashlib.sha256(blob).hexdigest() == "cbe1806b7b392ad22919db73c25cb1ccfb9eec9623a079ea9923a499f1af5dfc"
  assert renderer.cmac_counters == CMACRouteCounters(attempted=1, fallback=1, reasons={CMACReject.BOUNDS: 1})
