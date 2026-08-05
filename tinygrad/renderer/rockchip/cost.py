from __future__ import annotations
import math

from tinygrad.renderer.rockchip.emit import emit_program
from tinygrad.renderer.rockchip.image import RK_STAGE_RESET
from tinygrad.renderer.rockchip.ir import (RKALUStage, RKArg, RKCopyStage, RKCastStage, RKDPUProgram, RKFusedALUStage,
  RKStridedAtomGatherStage, RKLegalizedReformat, RKLUTStage, RKMaskStage, RKCMACTask, RKConvTask, RKPlanCost, RKPool, RKProgram, RKReduce)

def plan_cost(plan:RKProgram) -> RKPlanCost:
  """Estimate physical work after all semantic plans have been legalized."""
  image = emit_program(plan)
  reads = writes = macs = 0
  for step in plan.steps:
    if isinstance(step, RKDPUProgram):
      for stage in step.stages:
        if isinstance(stage, RKALUStage):
          reads += sum(stage.count*2 for operand in (stage.lhs,stage.rhs) if isinstance(operand,RKArg))
          writes += stage.count*stage.out_dtype.itemsize
          macs += stage.count
        elif isinstance(stage, RKFusedALUStage):
          reads += stage.count*(2+4+2) + (stage.count*2 if isinstance(stage.bn,RKArg) else 0)
          writes += stage.count*2
          macs += stage.count*3
        elif isinstance(stage, RKStridedAtomGatherStage):
          reads += stage.rows*8*2
          writes += stage.rows*8*2
        elif isinstance(stage, RKCopyStage):
          reads += stage.count*stage.dtype.itemsize
          writes += stage.count*stage.dtype.itemsize
        elif isinstance(stage, RKCastStage):
          reads += stage.count*stage.src_dtype.itemsize
          writes += stage.count*stage.dst_dtype.itemsize
        elif isinstance(stage, (RKMaskStage,RKLUTStage)):
          reads += stage.count*2
          writes += stage.count*2
          macs += stage.count
    elif isinstance(step, RKCMACTask):
      m, n, k = math.prod(step.lhs.layout.logical_shape[:-1]), step.rhs.layout.logical_shape[0], step.lhs.layout.logical_shape[-1]
      reads += math.prod(step.lhs.layout.logical_shape)*step.lhs.layout.dtype.itemsize
      reads += math.prod(step.rhs.layout.logical_shape)*step.rhs.layout.dtype.itemsize
      if step.epilogue is not None and step.epilogue.bias is not None:
        reads += math.prod(step.epilogue.bias.layout.logical_shape)*step.epilogue.bias.layout.dtype.itemsize
      writes += math.prod(step.out.layout.logical_shape)*step.out.layout.dtype.itemsize
      macs += m*n*k
    elif isinstance(step, RKConvTask):
      reads += math.prod(step.src.layout.logical_shape)*step.src.layout.dtype.itemsize
      reads += math.prod(step.weight.layout.logical_shape)*step.weight.layout.dtype.itemsize
      writes += step.out_channels*step.output_height*step.output_width*2
      macs += step.out_channels*step.output_height*step.output_width*step.in_channels*step.kernel_height*step.kernel_width
    elif isinstance(step, RKLegalizedReformat):
      nested = plan_cost(step.program)
      reads += nested.estimated_read_bytes
      writes += nested.estimated_write_bytes
      macs += nested.estimated_macs
    elif isinstance(step, RKPool):
      reads += math.prod(step.src.layout.logical_shape)*step.src.layout.dtype.itemsize
      writes += math.prod(step.out.layout.logical_shape)*step.out.layout.dtype.itemsize
      macs += math.prod(step.out.layout.logical_shape)*step.kernel_height*step.kernel_width
    elif isinstance(step, RKReduce):
      reads += math.prod(step.src.layout.logical_shape)*step.src.layout.dtype.itemsize
      writes += math.prod(step.out.layout.logical_shape)*step.out.layout.dtype.itemsize
      macs += math.prod(step.src.layout.logical_shape)
    else: raise TypeError(f"unsupported Rockchip cost step {type(step).__name__}")
  return RKPlanCost(len(image.stages), sum(len(stage.commands) for stage in image.stages),
                    sum(bool(stage.flags & RK_STAGE_RESET) for stage in image.stages), len(image.constants),
                    sum(resource.size for resource in plan.scratch), reads, writes, macs)
