from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.rockchip.affine import rk_fingerprint
from tinygrad.renderer.rockchip.ir import (RKDPUProgram, RKCMACTask, RKConvTask, RKReduce, RKLegalizedReformat, RKProgram,
  RKRejectKind, RKReject, RKLowerKind, RKLowerResult)

RKNativePlan = RKDPUProgram|RKCMACTask|RKConvTask|RKReduce|RKLegalizedReformat|RKProgram

def native(plan:RKNativePlan) -> RKLowerResult: return RKLowerResult(RKLowerKind.NATIVE, plan=plan)
def not_applicable() -> RKLowerResult: return RKLowerResult(RKLowerKind.NOT_APPLICABLE)
def unsupported(kind:RKRejectKind, detail:str, node_op:Ops|None=None) -> RKLowerResult:
  return RKLowerResult(RKLowerKind.UNSUPPORTED, reject=RKReject(kind, detail, node_op))

@dataclass(frozen=True)
class RKLowerer:
  name: str
  applies: Callable[[tuple[UOp, ...]], bool]
  lower: Callable[[UOp], RKLowerResult]

def has_reduction(nodes:tuple[UOp, ...], op:Ops|None=None) -> bool:
  reductions = tuple(u for u in nodes if u.op is Ops.REDUCE)
  return bool(reductions) and (op is None or all(u.arg[0] is op for u in reductions))

_REJECT_PRIORITY = {
  RKRejectKind.NUMERICAL_CONTRACT:90, RKRejectKind.LUT_DOMAIN_UNPROVEN:85, RKRejectKind.PLAN_STAGE_LIMIT:80,
  RKRejectKind.UNSUPPORTED_INPUT_DTYPE:70, RKRejectKind.UNSUPPORTED_OUTPUT_DTYPE:70,
  RKRejectKind.UNALIGNED_ROW:60, RKRejectKind.REQUIRES_REFORMAT:60, RKRejectKind.UNSUPPORTED_DYNAMIC_PACK:60,
  RKRejectKind.UNSUPPORTED_LAYOUT:50, RKRejectKind.UNSUPPORTED_BROADCAST:50,
  RKRejectKind.UNSUPPORTED_REDUCTION:40, RKRejectKind.UNSUPPORTED_CONTRACTION:40, RKRejectKind.UNSUPPORTED_ALU:30,
}

def select_lowering(sink:UOp, lowerers:tuple[RKLowerer, ...]) -> RKLowerResult:
  nodes, rejects = tuple(sink.toposort()), []
  for lowerer in lowerers:
    result = lowerer.lower(sink) if lowerer.applies(nodes) else not_applicable()
    if result.kind is RKLowerKind.NATIVE: return result
    if result.kind is RKLowerKind.UNSUPPORTED:
      assert result.reject is not None
      rejects.append(result.reject)
  if not rejects: rejects.append(RKReject(RKRejectKind.UNSUPPORTED_ALU, "no Rockchip lowerer applies", sink.op))
  reject = max(enumerate(rejects), key=lambda item:(_REJECT_PRIORITY[item[1].kind], item[0]))[1]
  return RKLowerResult(RKLowerKind.UNSUPPORTED, reject=RKReject(reject.kind, reject.detail, reject.node_op, rk_fingerprint(sink)))
