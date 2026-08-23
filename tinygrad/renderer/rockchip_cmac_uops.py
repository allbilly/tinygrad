"""Strict immutable-asset CMAC route for the one-row K=32 sum."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.renderer.rockchip import (RKArg, RKBufferKind, RKImage, RKIMAGE_NATIVE_VERSION, RKNativeAsset,
  RKNativeKind, RKNativeOp, RKNativeRelocation, RKNativeReset, RKNativeSubmit, RKNativeTask, RKTarget, encode_image)
from tinygrad.runtime.autogen.rockchip_physical import (CMAC_V1_BODY_SHA256, CMAC_V1_COMMANDS, CMAC_V1_OUTPUT_VIEW_BYTES,
  CMAC_V1_PC_GUARD_BYTES, CMAC_V1_RELOCATIONS, CMAC_V1_RESET, CMAC_V1_SUBMIT, CMAC_V1_TAIL, CMAC_V1_TASK)
from tinygrad.uop.ops import AxisType, Ops, ParamArg, UOp

CMAC_UOP_SEMANTIC_PROVENANCE = "rk-cmac-semantic-v5:764c833fdcb22455f344812ab375867c0d8518fe"
CMAC_RHS_ONE_N4_ASSET_ID = 2
CMAC_RHS_ONE_N4_ASSET_SHA256 = "96a33b81830614e9b95b033117210b3933d7d971323992d35be7d901cb183c00"
# The physical RHS is [group=2, batch=1, row=16, K=32] FP16: four rows of
# group zero contain one, while every inactive row/lane is zero-filled.
CMAC_RHS_ONE_N4_PAYLOAD = b"\x00\x3c" * 128 + b"\0" * 1792
CMAC_RHS_ONE_N4_ASSET = RKNativeAsset(CMAC_RHS_ONE_N4_ASSET_ID, bytes.fromhex(CMAC_RHS_ONE_N4_ASSET_SHA256), 2048,
  ((0, 2048),), payload=CMAC_RHS_ONE_N4_PAYLOAD)
_TRUSTED_ASSET_PRODUCER = object()


@dataclass(frozen=True, eq=False)
class CMACConstantAssetCertificate:
  """Opaque proof minted only by the compiler-owned constant-sum rewrite."""
  asset_id: int = CMAC_RHS_ONE_N4_ASSET_ID
  digest: bytes = bytes.fromhex(CMAC_RHS_ONE_N4_ASSET_SHA256)
  size: int = 2048
  upload_ranges: tuple[tuple[int, int], ...] = ((0, 2048),)
  active_ranges: tuple[tuple[int, int], ...] = ((0, 256),)
  tail_ranges: tuple[tuple[int, int], ...] = ((256, 1792),)
  raw_output: tuple[int, int, int, int] = (128, 256, 128, 4096)
  producer: object|None = field(default=None, repr=False)

  # Certificates participate in UOp interning through ParamArg equality.  Keep
  # their identity opaque: Python's ordinary tuple equality would make a
  # forged ``False`` compare equal to the trusted integer ``0`` before the
  # matcher gets a chance to reject it.
  def __eq__(self, other: object) -> bool: return self is other
  __hash__ = object.__hash__

  @classmethod
  def _mint(cls) -> CMACConstantAssetCertificate:
    return cls(producer=_TRUSTED_ASSET_PRODUCER)


def _attach_asset_certificate(tensor: Any) -> Any:
  base = tensor._uop.base
  if base.op is not Ops.BUFFER or base.dtype.scalar() is not dtypes.half or base.max_numel() != 128 or \
     not isinstance(base.arg, ParamArg): raise ValueError("CMAC output must be a fresh 128-element FP16 buffer")
  cert = CMACConstantAssetCertificate._mint()
  return type(tensor)(tensor._uop.substitute({base: base.replace(arg=replace(base.arg, layout_certificate=cert))}))


def trusted_constant_sum(lhs: Any, *, n: int = 4) -> Any:
  """Rewrite the proven FP16 (1,32) sum to a physical N=4 lane-0 view."""
  if type(n) is not int or n != 4: raise ValueError("CMAC constant asset is fixed to N=4")
  if not hasattr(lhs, "shape") or lhs.shape != (1, 32) or lhs.dtype.scalar() is not dtypes.half:
    raise ValueError("CMAC constant sum needs a (1,32) FP16 input")
  from tinygrad import Tensor
  physical = _attach_asset_certificate(Tensor.empty((1, 128), dtype=dtypes.half, device=lhs.device))
  physical[:, :1].assign(lhs.sum(axis=1))
  return physical[:, :1]


def _is_exact_sum_uop(node: UOp) -> bool:
  if node.op is Ops.RESHAPE and node.shape == (1, 1) and len(node.src) == 2: node = node.src[0]
  if node.op is not Ops.CAST or node.dtype.scalar() is not dtypes.half or len(node.src) != 1: return False
  reduce = node.src[0]
  if reduce.op is not Ops.REDUCE or reduce.dtype.scalar() is not dtypes.float or type(reduce.arg) is not tuple or \
     len(reduce.arg) != 2 or reduce.arg[0] is not Ops.ADD or type(reduce.arg[1]) is not int or reduce.arg[1] != 1 or \
     len(reduce.src) != 1: return False
  permute = reduce.src[0]
  if permute.op is not Ops.PERMUTE or permute.shape != (32, 1) or type(permute.arg) is not tuple or len(permute.arg) != 2 or \
     type(permute.arg[0]) is not int or type(permute.arg[1]) is not int or permute.arg != (1, 0) or len(permute.src) != 1: return False
  source = permute.src[0]
  if source.op is not Ops.CAST or source.dtype.scalar() is not dtypes.float or len(source.src) != 1: return False
  reshape = source.src[0]
  if reshape.op is not Ops.RESHAPE or reshape.shape != (1, 32) or len(reshape.src) != 2: return False
  lhs = reshape.src[0]
  return lhs.op in (Ops.BUFFER, Ops.PARAM) and lhs.dtype.scalar() is dtypes.half and lhs.max_numel() == 32


class CMACReject(Enum):
  ARGUMENT = auto()
  ASSET = auto()
  AXES = auto()
  BOUNDS = auto()
  DTYPE = auto()
  DYNAMIC = auto()
  FAMILY = auto()
  LAYOUT = auto()
  MAP = auto()
  MULTI_OUTPUT = auto()
  N_RANGE = auto()


@dataclass(frozen=True)
class CMACFallback:
  reason: CMACReject
  detail: str


@dataclass
class CMACRouteCounters:
  attempted: int = 0
  admitted: int = 0
  native: int = 0
  fallback: int = 0
  reasons: dict[CMACReject, int] = field(default_factory=dict)
  @property
  def attempt(self) -> int: return self.attempted
  @property
  def selected(self) -> int: return self.admitted
  @property
  def native_submit(self) -> int: return self.native
  def reject(self, result: CMACFallback) -> None:
    self.fallback += 1
    self.reasons[result.reason] = self.reasons.get(result.reason, 0) + 1


@dataclass(frozen=True)
class CMACUOpMatch:
  native: RKNativeOp
  lhs: RKArg
  out: RKArg
  asset_certificate: CMACConstantAssetCertificate
  n: int = 4
  @property
  def semantic_provenance(self) -> str: return CMAC_UOP_SEMANTIC_PROVENANCE
  @property
  def body_hash(self) -> str: return CMAC_V1_BODY_SHA256
  @property
  def command_guard_bytes(self) -> int: return CMAC_V1_PC_GUARD_BYTES
  # (surface offset, host view bytes, row stride bytes, command guard bytes).
  @property
  def output_span_provenance(self) -> tuple[int, int, int, int]: return (128, CMAC_V1_OUTPUT_VIEW_BYTES, 128, CMAC_V1_PC_GUARD_BYTES)


def _reject(reason: CMACReject, detail: str) -> CMACFallback: return CMACFallback(reason, detail)


def _nodes(uops: Iterable[UOp]) -> tuple[UOp, ...]:
  result: dict[UOp, None] = {}
  for node in uops:
    for item in node.toposort() if node.op in (Ops.SINK, Ops.END, Ops.STORE, Ops.CALL, Ops.PROGRAM) else (node,): result[item] = None
  return tuple(result)


def _const_int(node: UOp) -> int | None: return int(node.arg) if node.op is Ops.CONST and type(node.arg) is int else None


def _param(node: UOp) -> tuple[int, int, DType, UOp] | None:
  if node.op not in (Ops.PARAM, Ops.BUFFER) or type(node.arg) is not ParamArg or len(node.src) != 1 or \
     node.arg.addrspace is not AddrSpace.GLOBAL or type(node.arg.slot) is not int or node.arg.slot < 0: return None
  count = _const_int(node.src[0])
  return None if count is None else (int(node.arg.slot), count, node.dtype.scalar(), node)


def _exact_int_tuple(value: Any, size: int) -> bool:
  return type(value) is tuple and len(value) == size and all(type(item) is int for item in value)


def _exact_ranges(value: Any, expected: tuple[tuple[int, int], ...]) -> bool:
  return type(value) is tuple and all(type(item) is tuple and len(item) == 2 and
    all(type(x) is int for x in item) for item in value) and value == expected


def _asset_certificate(node: UOp) -> CMACConstantAssetCertificate | None:
  cert = node.arg.layout_certificate if isinstance(node.arg, ParamArg) else None
  if type(cert) is not CMACConstantAssetCertificate or cert.producer is not _TRUSTED_ASSET_PRODUCER or \
     type(cert.digest) is not bytes or type(cert.asset_id) is not int or type(cert.size) is not int or \
     cert.asset_id != 2 or cert.digest != bytes.fromhex(CMAC_RHS_ONE_N4_ASSET_SHA256) or cert.size != 2048 or \
     not _exact_ranges(cert.upload_ranges, ((0, 2048),)) or not _exact_ranges(cert.active_ranges, ((0, 256),)) or \
     not _exact_ranges(cert.tail_ranges, ((256, 1792),)) or not _exact_int_tuple(cert.raw_output, 4) or \
     cert.raw_output != (128, 256, 128, 4096): return None
  return cert


def _store(nodes: tuple[UOp, ...]) -> tuple[UOp, tuple[int, int, DType, UOp]] | CMACFallback:
  stores = tuple(node for node in nodes if node.op is Ops.STORE)
  if len(stores) != 1: return _reject(CMACReject.MULTI_OUTPUT if stores else CMACReject.ARGUMENT, "CMAC sum needs one output store")
  store = stores[0]
  if len(store.src) != 2 or store.src[0].op is not Ops.INDEX or len(store.src[0].src) != 2:
    return _reject(CMACReject.ARGUMENT, "CMAC sum output must be a direct indexed store")
  output = _param(store.src[0].src[0])
  if output is None or output[2] is not dtypes.half: return _reject(CMACReject.DTYPE, "CMAC sum output must be FP16")
  if output[1] != 128: return _reject(CMACReject.BOUNDS, "CMAC sum output must be a fresh 128-element surface")
  if _const_int(store.src[0].src[1]) != 0: return _reject(CMACReject.MAP, "CMAC sum output must use zero-offset lane 0")
  if _asset_certificate(output[3]) is None: return _reject(CMACReject.LAYOUT, "CMAC sum output lacks immutable asset provenance")
  return store, output


def _sum_source(value: UOp, axis: UOp) -> tuple[int, int, DType, UOp] | CMACFallback:
  if value.op is not Ops.CAST or value.dtype.scalar() is not dtypes.half or len(value.src) != 1:
    return _reject(CMACReject.DTYPE, "CMAC sum must store FP16")
  reduce = value.src[0]
  if reduce.op is not Ops.REDUCE or reduce.dtype.scalar() is not dtypes.float or type(reduce.arg) is not tuple or \
     len(reduce.arg) != 2 or reduce.arg[0] is not Ops.ADD or type(reduce.arg[1]) is not int or reduce.arg[1] != 0 or len(reduce.src) != 2:
    return _reject(CMACReject.FAMILY, "CMAC sum needs FP32 ADD reduction metadata (0)")
  if reduce.src[1] is not axis: return _reject(CMACReject.AXES, "CMAC sum needs one K reduction axis")
  source = reduce.src[0]
  if source.op is not Ops.CAST or source.dtype.scalar() is not dtypes.float or len(source.src) != 1:
    return _reject(CMACReject.DTYPE, "CMAC sum needs FP32 input cast")
  if (load := source.src[0]).op is Ops.LOAD and load.dtype.scalar() is not dtypes.half: return _reject(CMACReject.DTYPE, "CMAC sum LOAD must be FP16")
  index = load.src[0] if load.op is Ops.LOAD and len(load.src) == 1 else load
  if index.op is not Ops.INDEX or len(index.src) != 2:
    return _reject(CMACReject.ARGUMENT, "CMAC sum needs one direct lhs index")
  lhs = _param(index.src[0])
  if lhs is None or lhs[1] != 32 or lhs[2] is not dtypes.half or index.src[1] is not axis:
    return _reject(CMACReject.MAP, "CMAC sum lhs must be contiguous K=32")
  return lhs


def _emit(lhs: tuple[int, int, DType, UOp], output: tuple[int, int, DType, UOp]) -> CMACUOpMatch | CMACFallback:
  cert = _asset_certificate(output[3])
  if cert is None: return _reject(CMACReject.LAYOUT, "CMAC sum output lacks immutable asset provenance")
  if lhs[2] is not dtypes.half or output[2] is not dtypes.half: return _reject(CMACReject.DTYPE, "CMAC sum surfaces must be FP16")
  native = _native(lhs[0], output[0])
  return CMACUOpMatch(native, RKArg(RKBufferKind.ARG, lhs[0]), RKArg(RKBufferKind.ARG, output[0]), cert)


def _match_constant_sum(nodes: tuple[UOp, ...]) -> CMACUOpMatch | CMACFallback:
  found = _store(nodes)
  if isinstance(found, CMACFallback): return found
  store, output = found
  ranges = tuple(node for node in nodes if node.op is Ops.RANGE)
  if len(ranges) != 1 or ranges[0].src[0].op is not Ops.CONST or _const_int(ranges[0].src[0]) != 32 or \
     type(ranges[0].arg) is not tuple or len(ranges[0].arg) != 2 or type(ranges[0].arg[0]) is not int or ranges[0].arg[0] != 0 or \
     ranges[0].arg[1] is not AxisType.REDUCE:
    return _reject(CMACReject.AXES, "CMAC sum needs one static K=32 REDUCE range")
  result = _sum_source(store.src[1], ranges[0])
  if isinstance(result, CMACFallback): return result
  params = tuple(node for node in nodes if node.op in (Ops.PARAM, Ops.BUFFER))
  if len(params) != 2 or any(_param(node) is None for node in params):
    return _reject(CMACReject.ARGUMENT, "CMAC sum may have only lhs and output parameters")
  if result[0] == output[0]: return _reject(CMACReject.ARGUMENT, "CMAC sum surfaces must not alias")
  if sum(node.op is Ops.LOAD for node in nodes) > 1: return _reject(CMACReject.ARGUMENT, "CMAC sum has multiple lhs loads")
  return _emit(result, output)


def _linear_terms(node: UOp) -> tuple[UOp, ...]:
  return _linear_terms(node.src[0]) + _linear_terms(node.src[1]) if node.op is Ops.ADD and len(node.src) == 2 else (node,)


def _match_linear_sum(nodes: tuple[UOp, ...]) -> CMACUOpMatch | CMACFallback:
  found = _store(nodes)
  if isinstance(found, CMACFallback): return found
  store, output = found
  value = store.src[1]
  if value.op is not Ops.CAST or value.dtype.scalar() is not dtypes.half or len(value.src) != 1:
    return _reject(CMACReject.DTYPE, "CMAC linear sum must store FP16")
  terms = _linear_terms(value.src[0])
  if len(terms) != 32: return _reject(CMACReject.N_RANGE, "CMAC linear sum needs exactly 32 terms")
  lhs: tuple[int, int, DType, UOp] | None = None
  for k, term in enumerate(terms):
    if term.op is not Ops.CAST or term.dtype.scalar() is not dtypes.float or len(term.src) != 1 or term.src[0].op is not Ops.LOAD:
      return _reject(CMACReject.FAMILY, "CMAC linear sum terms need FP32 casts of lhs loads")
    load = term.src[0]
    if load.dtype.scalar() is not dtypes.half or len(load.src) != 1 or load.src[0].op is not Ops.INDEX or \
       len(load.src[0].src) != 2 or _const_int(load.src[0].src[1]) != k:
      return _reject(CMACReject.MAP, "CMAC linear sum lhs indices must be K=0..31")
    current = _param(load.src[0].src[0])
    if current is None or current[1] != 32 or current[2] is not dtypes.half: return _reject(CMACReject.DTYPE, "CMAC linear sum lhs must be FP16 K=32")
    if lhs is None: lhs = current
    elif lhs[0] != current[0]: return _reject(CMACReject.ARGUMENT, "CMAC linear sum lhs parameter is unstable")
  params = tuple(node for node in nodes if node.op in (Ops.PARAM, Ops.BUFFER))
  if lhs is None or len(params) != 2: return _reject(CMACReject.ARGUMENT, "CMAC linear sum may have only lhs and output parameters")
  return _emit(lhs, output)


def _native(lhs_slot: int, out_slot: int) -> RKNativeOp:
  lhs, asset, out = RKArg(RKBufferKind.ARG, lhs_slot), RKArg(RKBufferKind.ASSET, 0), RKArg(RKBufferKind.ARG, out_slot)
  # The three command fields bind lhs, immutable RHS asset, and output in that order.
  relocs = tuple(RKNativeRelocation(word, target, register, arg) for (word, target, register), arg in zip(CMAC_V1_RELOCATIONS, (lhs, asset, out)))
  native = RKNativeOp(RKNativeKind.CMAC, tuple(CMAC_V1_COMMANDS), relocs, reads=(lhs,), writes=(out,), outputs=(out,), tail=tuple(CMAC_V1_TAIL),
    assets=(CMAC_RHS_ONE_N4_ASSET,), task=RKNativeTask(*CMAC_V1_TASK), submit=RKNativeSubmit(*CMAC_V1_SUBMIT), reset=RKNativeReset(*CMAC_V1_RESET))
  encode_image(RKImage(RKTarget.RK3588, version=RKIMAGE_NATIVE_VERSION, native=native))
  return native


def match_cmac_uops(uops: Iterable[UOp], *, counters: CMACRouteCounters | None = None) -> CMACUOpMatch | CMACFallback:
  if counters is not None: counters.attempted += 1
  try:
    nodes = _nodes(uops)
    result = _reject(CMACReject.DTYPE, "INDEX dtype") if any(node.op is Ops.INDEX and node.dtype is not dtypes.half for node in nodes) else (
      _match_constant_sum(nodes) if any(node.op is Ops.REDUCE for node in nodes) else _match_linear_sum(nodes))
  except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
    result = _reject(CMACReject.ARGUMENT, f"unrecognized UOp form: {exc}")
  if isinstance(result, CMACFallback):
    if counters is not None: counters.reject(result)
    return result
  if counters is not None: counters.admitted += 1
  return result


def try_cmac(uops: Iterable[UOp], *, counters: CMACRouteCounters | None = None) -> RKNativeOp | None:
  result = match_cmac_uops(uops, counters=counters)
  if isinstance(result, CMACUOpMatch):
    if counters is not None: counters.native += 1
    return result.native
  return None


def lower_cmac_uops(uops: Iterable[UOp], *, counters: CMACRouteCounters | None = None) -> RKImage | None:
  native = try_cmac(uops, counters=counters)
  return None if native is None else RKImage(RKTarget.RK3588, version=RKIMAGE_NATIVE_VERSION, native=native)


def is_cmac_physical_image(image: RKImage) -> bool:
  native = image.native
  return image.version == RKIMAGE_NATIVE_VERSION and native is not None and native.kind is RKNativeKind.CMAC and not any(
    (image.scratch, image.gathers, image.ew_ops, image.mid_gathers, image.post_gathers, image.host_gathers, image.host_scatters, image.constants))
