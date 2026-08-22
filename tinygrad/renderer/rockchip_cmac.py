"""Host-only CMAC matrixization oracle.

The planner describes static addresses and physical tiles; it never reads a
tensor buffer and deliberately has no runtime/device entry point.  Its limits
are the frozen FP16 CMAC donor: one output row, sixteen logical outputs, and a
32-term K tile.  Larger static matrices are represented as ordered tiles with
EW accumulation barriers.
"""
from __future__ import annotations
# ruff: noqa: E501,E702
import itertools, math, struct
from dataclasses import dataclass
from enum import Enum
from typing import Iterable
from tinygrad.renderer.rockchip import RKArg, RKBufferKind, RKStage

# The compact declarations mirror the existing RKImage/RKStage carrier style.
# ruff: noqa: E702

CMAC_TILE_N, CMAC_TILE_K, CMAC_TILE_M, CMAC_MAX_TILES = 16, 32, 1, 1 << 14

class CMACFamily(str, Enum):
  ADD = "add"
  SCALED_ADD = "scaled_add"
  MADD = "madd"
  SCALED_MADD = "scaled_madd"
  BIAS_MADD = "bias_madd"
  AFFINE_MADD = "affine_madd"
  LOCAL_ADD = "local_add"

MATRIX_FAMILIES = frozenset(CMACFamily(x) for x in ("add", "scaled_add", "madd", "scaled_madd", "bias_madd", "affine_madd"))

class CMACReject(str, Enum):
  EXTREMA = "extrema"
  INTEGER = "integer"
  BOOLEAN = "boolean"
  DYNAMIC = "dynamic"
  DTYPE = "dtype"
  FAMILY = "family"
  AXES = "axes"
  BOUNDS = "bounds"
  LIMIT = "limit"

@dataclass(frozen=True)
class CMACFallback:
  reason: CMACReject
  detail: str

@dataclass(frozen=True)
class StaticAxis:
  name: str
  extent: int
  def __post_init__(self):
    if not self.name or self.extent <= 0: raise ValueError("static axes need a name and positive extent")

@dataclass(frozen=True)
class CMACAxes:
  axes: tuple[StaticAxis, ...]; m: tuple[str, ...]; n: tuple[str, ...]; k: tuple[str, ...]
  batch: tuple[str, ...] = ()
  def validate(self) -> tuple[CMACFallback, dict[str, int]] | None:
    extents = {axis.name:axis.extent for axis in self.axes}
    groups = self.m + self.n + self.k + self.batch
    if len(extents) != len(self.axes) or len(groups) != len(set(groups)) or set(groups) != set(extents):
      return CMACFallback(CMACReject.AXES, "axes must be a disjoint M/N/K/batch partition"), extents
    return None
  def extent(self, group:tuple[str, ...]) -> int: return math.prod(dict((a.name, a.extent) for a in self.axes)[x] for x in group)

@dataclass(frozen=True)
class StaticIndex:
  """An affine index or an explicit static table over a subset of axes."""
  axes: tuple[str, ...] = (); strides: tuple[int, ...] = (); offset: int = 0; values: tuple[int, ...] = (); dynamic: bool = False
  def evaluate(self, coords:dict[str, int], extents:dict[str, int]) -> int:
    if self.dynamic: raise ValueError("dynamic index")
    if any(axis not in extents for axis in self.axes) or (self.values and self.strides): raise ValueError("invalid static index")
    if self.values:
      if len(self.values) != math.prod(extents[x] for x in self.axes): raise ValueError("static table extent mismatch")
      linear = 0
      for axis in self.axes: linear = linear*extents[axis] + coords[axis]
      return self.offset + self.values[linear]
    if len(self.axes) != len(self.strides): raise ValueError("static affine arity mismatch")
    return self.offset + sum(coords[x]*stride for x,stride in zip(self.axes, self.strides))

@dataclass(frozen=True)
class CMACShape:
  m: int; n: int; k: int; batches: int
  def __post_init__(self):
    if min(self.m, self.n, self.k, self.batches) <= 0: raise ValueError("CMAC shape must be positive")

@dataclass(frozen=True)
class CMACGather:
  source: RKArg; indices: tuple[int, ...]; packed: tuple[int, ...]; reuse: int

@dataclass(frozen=True)
class CMACTile:
  batch: int; m_start: int; n_start: int; k_start: int; m_count: int; n_count: int; k_count: int
  output: tuple[int, ...]; lhs: CMACGather; rhs: CMACGather | None; accumulator: int; barrier: bool; final: bool

@dataclass(frozen=True)
class CMACPlan:
  family: CMACFamily; shape: CMACShape; axes: CMACAxes; lhs: RKArg; rhs: RKArg | None; out: RKArg; bias: RKArg | None
  input_dtype: str; accumulator_dtype: str; output_dtype: str; final_round: bool; scale: float
  tiles: tuple[CMACTile, ...]; scratch_slots: int; barriers: tuple[int, ...]

def _product_values(group:tuple[str, ...], extents:dict[str, int]) -> Iterable[dict[str, int]]:
  return (dict(zip(group, values)) for values in itertools.product(*(range(extents[x]) for x in group)))

def _coords(axes:CMACAxes, extents:dict[str, int], batch:int, m:int=0, n:int=0, k:int=0) -> dict[str, int]:
  coords = dict(next(itertools.islice(_product_values(axes.batch, extents), batch, None), {}))
  for group, linear in ((axes.m,m), (axes.n,n), (axes.k,k)):
    for axis in reversed(group): linear, coords[axis] = divmod(linear, extents[axis])
  return coords

def _gather(source:RKArg, values:tuple[int, ...], packed:tuple[int, ...], reuse:int) -> CMACGather:
  return CMACGather(source, values, packed, reuse)

def _reject(reason:CMACReject, detail:str) -> CMACFallback: return CMACFallback(reason, detail)

def plan_cmac(*, family:CMACFamily, axes:CMACAxes, output_map:StaticIndex, lhs_map:StaticIndex,
              lhs:RKArg, out:RKArg, lhs_count:int, rhs_map:StaticIndex|None=None, rhs:RKArg|None=None,
              rhs_count:int=0, bias:RKArg|None=None, input_dtype:str="fp16", accumulator_dtype:str="fp32",
              output_dtype:str="fp16", operation:str="add", dynamic:bool=False, local:bool=False,
              scale:float=1.0) -> CMACPlan | CMACFallback:
  if operation.lower() in {"max", "min", "argmax", "argmin", "extrema"}: return _reject(CMACReject.EXTREMA, operation)
  if operation.lower() in {"int", "integer"}: return _reject(CMACReject.INTEGER, operation)
  if operation.lower() in {"bool", "boolean"}: return _reject(CMACReject.BOOLEAN, operation)
  if dynamic or output_map.dynamic or lhs_map.dynamic or rhs_map is not None and rhs_map.dynamic: return _reject(CMACReject.DYNAMIC, "dynamic index")
  if input_dtype != "fp16" or accumulator_dtype != "fp32" or output_dtype not in {"fp16", "fp32"}:
    return _reject(CMACReject.DTYPE, "CMAC contract is fp16 input/fp32 accumulation")
  if not math.isfinite(scale): return _reject(CMACReject.DTYPE, "non-finite scale")
  if family is CMACFamily.LOCAL_ADD and not local: return _reject(CMACReject.FAMILY, "LOCAL_ADD requires local=True")
  if family not in MATRIX_FAMILIES and family is not CMACFamily.LOCAL_ADD: return _reject(CMACReject.FAMILY, family.value)
  needs_rhs = family in {CMACFamily.MADD, CMACFamily.SCALED_MADD, CMACFamily.BIAS_MADD, CMACFamily.AFFINE_MADD}
  if needs_rhs and (rhs is None or rhs_map is None): return _reject(CMACReject.FAMILY, "MADD needs two static surfaces")
  if not needs_rhs and (rhs is not None or rhs_map is not None): return _reject(CMACReject.FAMILY, "ADD has no RHS surface")
  rhs_arg = rhs if rhs is not None else lhs
  if family is CMACFamily.BIAS_MADD and bias is None: return _reject(CMACReject.FAMILY, "bias surface is missing")
  if family in {CMACFamily.ADD, CMACFamily.SCALED_ADD, CMACFamily.LOCAL_ADD} and rhs is not None: return _reject(CMACReject.FAMILY, "ADD has no RHS surface")
  if lhs_count <= 0 or rhs_count < 0: return _reject(CMACReject.BOUNDS, "negative surface extent")
  try: valid = axes.validate()
  except ValueError as exc: return _reject(CMACReject.AXES, str(exc))
  if valid is not None: return valid[0]
  extents = {axis.name:axis.extent for axis in axes.axes}
  shape = CMACShape(axes.extent(axes.m), axes.extent(axes.n) or 1, axes.extent(axes.k), axes.extent(axes.batch) or 1)
  if shape.m * shape.n * shape.batches > CMAC_MAX_TILES * CMAC_TILE_N: return _reject(CMACReject.LIMIT, "too many output tiles")
  if family is CMACFamily.LOCAL_ADD and shape.m != 1: return _reject(CMACReject.LIMIT, "local ADD has one output row")
  tiles:list[CMACTile] = []; pack_ids:dict[tuple[RKArg, tuple[int, ...]], int] = {}; next_pack = 0
  try:
    for batch in range(shape.batches):
      for m_start in range(0, shape.m, CMAC_TILE_M):
        for n_start in range(0, shape.n, CMAC_TILE_N):
          n_count = min(CMAC_TILE_N, shape.n-n_start)
          for k_start in range(0, shape.k, CMAC_TILE_K):
            k_count = min(CMAC_TILE_K, shape.k-k_start)
            output = tuple(output_map.evaluate(_coords(axes, extents, batch, m_start+i, n_start+j, 0), extents) for i in range(CMAC_TILE_M) for j in range(n_count))
            lhs_values = tuple(lhs_map.evaluate(_coords(axes, extents, batch, m_start+i, 0, k_start+t), extents) if t < k_count else -1
                              for i in range(CMAC_TILE_M) for t in range(CMAC_TILE_K))
            rhs_values = None if rhs_map is None else tuple(rhs_map.evaluate(_coords(axes, extents, batch, 0, n_start+j, k_start+t), extents) if t < k_count else -1
                                                         for j in range(n_count) for t in range(CMAC_TILE_K))
            valid_lhs = tuple(value for i in range(CMAC_TILE_M) for value in lhs_values[i*CMAC_TILE_K:i*CMAC_TILE_K+k_count])
            valid_rhs = () if rhs_values is None else tuple(value for j in range(n_count) for value in rhs_values[j*CMAC_TILE_K:j*CMAC_TILE_K+k_count])
            if any(value < 0 or value >= lhs_count for value in valid_lhs): return _reject(CMACReject.BOUNDS, "LHS index out of bounds")
            if any(value < 0 or value >= rhs_count for value in valid_rhs): return _reject(CMACReject.BOUNDS, "RHS index out of bounds")
            for value in output:
              if value < 0 or value >= shape.m*shape.n*shape.batches: return _reject(CMACReject.BOUNDS, "output index out of bounds")
            lhs_key = (lhs, lhs_values); lhs_pack = pack_ids.setdefault(lhs_key, next_pack); next_pack += lhs_pack == next_pack
            rhs_pack = -1
            if rhs_values is not None:
              rhs_key = (rhs_arg, rhs_values); rhs_pack = pack_ids.setdefault(rhs_key, next_pack); next_pack += rhs_pack == next_pack
            tiles.append(CMACTile(batch, m_start, n_start, k_start, CMAC_TILE_M, n_count, k_count, output,
              _gather(lhs, lhs_values, tuple(i*CMAC_TILE_K+t for i in range(CMAC_TILE_M) for t in range(CMAC_TILE_K)), lhs_pack),
              None if rhs_values is None else _gather(rhs_arg, rhs_values,
                tuple(j*CMAC_TILE_K+t for j in range(n_count) for t in range(CMAC_TILE_K)), rhs_pack),
              0 if shape.k > CMAC_TILE_K else -1, k_start > 0, k_start+k_count == shape.k))
  except (KeyError, ValueError, StopIteration) as exc: return _reject(CMACReject.AXES, str(exc))
  plan = CMACPlan(family, shape, axes, lhs, rhs, out, bias, input_dtype, accumulator_dtype, output_dtype,
                  output_dtype == "fp16", scale, tuple(tiles), int(shape.k > CMAC_TILE_K),
                  tuple(i for i,tile in enumerate(tiles) if tile.barrier))
  try: validate_cmac_plan(plan, lhs_count, rhs_count)
  except ValueError as exc: return _reject(CMACReject.BOUNDS, str(exc))
  return plan

def validate_cmac_plan(plan:CMACPlan, lhs_count:int, rhs_count:int) -> None:
  """Exhaustively validate every static output, gather, tile, and precision invariant."""
  if (plan.input_dtype, plan.accumulator_dtype) != ("fp16", "fp32") or plan.output_dtype not in {"fp16", "fp32"}: raise ValueError("invalid precision")
  expected = set(range(plan.shape.m*plan.shape.n*plan.shape.batches)); outputs = {x for tile in plan.tiles if tile.final for x in tile.output}
  if outputs != expected: raise ValueError("output coverage is not dense")
  for index,tile in enumerate(plan.tiles):
    if not 0 <= tile.batch < plan.shape.batches or tile.m_start >= plan.shape.m or tile.n_start >= plan.shape.n: raise ValueError("tile coordinates")
    if tile.k_count <= 0 or tile.k_count > CMAC_TILE_K or tile.n_count <= 0 or tile.n_count > CMAC_TILE_N: raise ValueError("tile exceeds donor shape")
    if tile.k_start == 0 and tile.barrier or tile.k_start > 0 and not tile.barrier: raise ValueError("K barrier mismatch")
    lhs_valid = tuple(x for i in range(tile.m_count) for x in tile.lhs.indices[i*CMAC_TILE_K:i*CMAC_TILE_K+tile.k_count])
    rhs_valid = () if tile.rhs is None else tuple(x for j in range(tile.n_count) for x in tile.rhs.indices[j*CMAC_TILE_K:j*CMAC_TILE_K+tile.k_count])
    if any(x not in range(lhs_count) for x in lhs_valid): raise ValueError("LHS gather bounds")
    if any(x not in range(rhs_count) for x in rhs_valid): raise ValueError("RHS gather bounds")
    if tile.lhs.packed != tuple(i*CMAC_TILE_K+t for i in range(tile.m_count) for t in range(CMAC_TILE_K)): raise ValueError("LHS packing")
    if tile.rhs is not None and tile.rhs.packed != tuple(j*CMAC_TILE_K+t for j in range(tile.n_count) for t in range(CMAC_TILE_K)): raise ValueError("RHS packing")
  if plan.scratch_slots != int(plan.shape.k > CMAC_TILE_K) or len(plan.barriers) != sum(tile.barrier for tile in plan.tiles): raise ValueError("scratch/barrier accounting")

CMAC_TASK = (0, 0xD, 0x300)
_CMAC_COMMANDS = tuple(int(word, 16) for word in ("10010000000e4004 020120000120100c 0201000040001010 0201000000091014 0201000100011020 0201001f00201024 0201000000011028 020100000001102c 0201000008001030 0201000000401034 0201010100201038 0201000000b11040 0201000000011044 02010000000b104c 0201000100001050 0201000100001054 0201000100001058 020100010000105c 0201000000001070 0201000f000f1078 020100000004107c 0201000000001080 0201000100011084 0201000000201088 0201000000001110 0801000002013010 0801000000003014 08010000001f3018 0801000000003030 1001000001e4400c 1001480000024010 1001000000004020 1001000000104024 1001000000004030 1001000000004034 1001000700074038 1001001f001f403c 1001000000534040 1001000001264050 10010000001f4058 100100000000405c 1001000000534060 1001000003834070 1001000100014084 10010000004040c0 00810000000d0008").split())

def emit_cmac_stage(tile:CMACTile, lhs:RKArg|None=None, rhs:RKArg|None=None, out:RKArg|None=None) -> RKStage:
  """Return only the frozen donor command body; no submit or device access occurs."""
  if (rhs_source:=rhs or (tile.rhs.source if tile.rhs is not None else None)) is None: raise ValueError("CMAC ADD needs a materialized ones RHS")
  return RKStage(_CMAC_COMMANDS, ((18, lhs or tile.lhs.source), (24, rhs_source), (31, out or RKArg(RKBufferKind.ARG, 0))))

def serialize_cmac_stage(stage:RKStage) -> bytes:
  if stage.commands != _CMAC_COMMANDS or tuple(word for word,_ in stage.relocs) != (18,24,31): raise ValueError("invalid CMAC stage")
  return struct.pack("<46Q", *stage.commands)
