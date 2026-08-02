from __future__ import annotations

from tinygrad.renderer.rockchip.expr import _ALUExpr, _Expr, _LUTExpr, _MaskExpr
from tinygrad.renderer.rockchip.ir import RKALUStage, RKArg, RKBufferKind, RKDPUProgram, RKDPUStage, RKLUTStage, RKMaskStage, RKScratch

_STAGED_EXPR = (_ALUExpr, _MaskExpr, _LUTExpr)

def schedule_expr(root:_Expr, output:RKArg, count:int, scratch:tuple[RKScratch, ...]=()) -> RKDPUProgram|None:
  """Schedule a canonical expression DAG, reusing dead temporary surfaces."""
  order:list[_Expr] = []
  def visit(expr:_Expr) -> None:
    for src in expr.src:
      if isinstance(src, _STAGED_EXPR) and src not in order: visit(src)
    if expr not in order: order.append(expr)
  visit(root)
  uses = {expr:sum(src == expr for node in order for src in node.src) for expr in order}
  values:dict[_Expr, RKArg] = {}
  free:list[int] = []
  resources, stages = list(scratch), list[RKDPUStage]()
  for expr in order:
    src = tuple(values[x] if isinstance(x, _STAGED_EXPR) else x for x in expr.src)
    if expr is root: dst = output
    elif isinstance(expr, _ALUExpr) and (reuse:=next((values[x] for x in expr.src if isinstance(x, _STAGED_EXPR) and
                                                     uses[x] == 1 and values[x].kind is RKBufferKind.SCRATCH), None)) is not None: dst = reuse
    else:
      slot = free.pop() if free else len(resources)
      if slot == len(resources): resources.append(RKScratch(((count+7)//8)*16))
      dst = RKArg(RKBufferKind.SCRATCH, slot)
    if isinstance(expr, _ALUExpr): stages.append(RKALUStage(expr.op, dst, src[0], src[1], count))
    elif isinstance(expr, _LUTExpr) and isinstance(src[0], RKArg): stages.append(RKLUTStage(expr.lut, dst, src[0], count))
    elif isinstance(src[0], RKArg): stages.append(RKMaskStage(dst, src[0], count))
    else: return None
    values[expr] = dst
    for source in expr.src:
      if isinstance(source, _STAGED_EXPR):
        uses[source] -= 1
        arg = values[source]
        if uses[source] == 0 and arg.kind is RKBufferKind.SCRATCH and arg != dst: free.append(arg.index)
  return RKDPUProgram(tuple(stages), tuple(resources))
