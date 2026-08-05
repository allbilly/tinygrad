from __future__ import annotations
import os
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId as RKLUTId
from tinygrad.runtime.support.rockchip_telemetry import record as record_telemetry
from tinygrad.uop.ops import AddrSpace, Ops, ProgramInfo, UOp

from tinygrad.renderer.rockchip.ir import (RKTarget as RKTarget, RKEngine as RKEngine, RKBufferKind as RKBufferKind,
  RKLayoutKind as RKLayoutKind, RKReformatKind as RKReformatKind, RKArg as RKArg, RKALUStage as RKALUStage,
  RKFusedALUStage as RKFusedALUStage, RKFusedMulStage as RKFusedMulStage,
  RKStridedAtomGatherStage as RKStridedAtomGatherStage, RKCopyStage as RKCopyStage, RKCastStage as RKCastStage,
  RKMaskStage as RKMaskStage, RKLUTStage as RKLUTStage, RKDPUStage as RKDPUStage, RKScratch as RKScratch,
  RKDPUProgram, RKLayout as RKLayout, RKTensorRef as RKTensorRef, RKEpilogue as RKEpilogue,
  RKContractionPlan as RKContractionPlan, RKCMACTask, RKConvTask, RKDeconvTask as RKDeconvTask, RKConvPlan as RKConvPlan,
  RKConvSplit as RKConvSplit, RKConvTile as RKConvTile, RKConvTiling as RKConvTiling, RKReduce, RKPool,
  RKReformatPlan as RKReformatPlan, RKMultiSourceReformatPlan as RKMultiSourceReformatPlan, RKLegalizedReformat, RKProgram,
  RKPlanCost as RKPlanCost, RKRejectKind as RKRejectKind, RKReject as RKReject, RKLowerKind as RKLowerKind, RKLowerResult)
from tinygrad.renderer.rockchip.image import (RK_STAGE_RESET as RK_STAGE_RESET, RKReloc as RKReloc, RKStage as RKStage,
  RKImage as RKImage, encode_image, decode_image as decode_image, patch_image as patch_image, validate_image as validate_image)
from tinygrad.renderer.rockchip.affine import rk_fingerprint as rk_fingerprint
from tinygrad.renderer.rockchip.access import (RKAccessMap as RKAccessMap, RKIdentityMap as RKIdentityMap, RKAffineMap as RKAffineMap,
  RKPadMap as RKPadMap, RKPeriodicMap as RKPeriodicMap, RKAffineSegment as RKAffineSegment,
  RKPiecewiseAffineMap as RKPiecewiseAffineMap, RKStaticSelectorMap as RKStaticSelectorMap,
  RKMultiSourceAffineSegment as RKMultiSourceAffineSegment, RKMultiSourceAccessMap as RKMultiSourceAccessMap,
  RKMultiSourceAffineGridMap as RKMultiSourceAffineGridMap)
from tinygrad.renderer.rockchip.conv import plan_conv_cbuf as plan_conv_cbuf, legalize_conv_plan as legalize_conv_plan
from tinygrad.renderer.rockchip.contract import (legalize_contraction_plan as legalize_contraction_plan,
  lower_contract_result as lower_contract_result, lower_depthwise_spatial_contract_result as lower_depthwise_spatial_contract_result,
  lower_grouped_spatial_contract_result as lower_grouped_spatial_contract_result,
  lower_nhwc_spatial_contract_result as lower_nhwc_spatial_contract_result, lower_deconv_result as lower_deconv_result,
  lower_spatial_contract_result as lower_spatial_contract_result, lower_tiled_contract_result as lower_tiled_contract_result,
  lower_contract as lower_contract)
from tinygrad.renderer.rockchip.cost import plan_cost as plan_cost
from tinygrad.renderer.rockchip.lower import RKLowerer, has_reduction as _has_reduction, select_lowering
from tinygrad.renderer.rockchip.reformat import (lower_static_two_tap_result as lower_static_two_tap_result,
  lower_reformat_result as lower_reformat_result, lower_multi_source_reformat_result as lower_multi_source_reformat_result,
  lower_static_selector_reformat_result as lower_static_selector_reformat_result)
from tinygrad.renderer.rockchip.pool import (lower_reduce_result as lower_reduce_result, lower_global_max_result as lower_global_max_result,
  lower_sliding_max_result as lower_sliding_max_result, lower_dense_row_max_result as lower_dense_row_max_result,
  lower_affine_max_result as lower_affine_max_result)
from tinygrad.renderer.rockchip.reduce import (lower_add_reduce_result as lower_add_reduce_result,
  lower_affine_mean_result as lower_affine_mean_result, lower_nested_add_reduce_result as lower_nested_add_reduce_result,
  lower_scalar_mul_reduce_result as lower_scalar_mul_reduce_result, lower_affine_mul_reduce_result as lower_affine_mul_reduce_result,
  lower_masked_affine_mul_reduce_result as lower_masked_affine_mul_reduce_result,
  lower_multi_source_affine_reduce_result as lower_multi_source_affine_reduce_result,
  lower_pointwise_affine_reduce_result as lower_pointwise_affine_reduce_result,
  lower_affine_reduce_result as lower_affine_reduce_result, _finish_reduction_epilogue as _finish_reduction_epilogue)
from tinygrad.renderer.rockchip.elementwise import (lower_dpu_result as lower_dpu_result, lower_dpu as lower_dpu,
  lower_broadcast_alu_result as lower_broadcast_alu_result, lower_multi_broadcast_alu_result as lower_multi_broadcast_alu_result)
from tinygrad.renderer.rockchip.expr import _numerical_contract as _numerical_contract
from tinygrad.renderer.rockchip.emit import (emit_dpu as emit_dpu, emit_cmac_task as emit_cmac_task, emit_spatial_conv as emit_spatial_conv,
  emit_program as emit_program, emit_reduce as emit_reduce, emit_pool as emit_pool, emit_reformat as emit_reformat)

_LOWERERS = (
  RKLowerer("dpu", lambda nodes:not _has_reduction(nodes), lower_dpu_result),
  RKLowerer("multi_source_reformat", lambda nodes:not _has_reduction(nodes), lower_multi_source_reformat_result),
  RKLowerer("static_selector_reformat", lambda nodes:not _has_reduction(nodes), lower_static_selector_reformat_result),
  RKLowerer("static_two_tap", lambda nodes:not _has_reduction(nodes), lower_static_two_tap_result),
  RKLowerer("multi_broadcast_alu", lambda nodes:not _has_reduction(nodes), lower_multi_broadcast_alu_result),
  RKLowerer("broadcast_alu", lambda nodes:not _has_reduction(nodes), lower_broadcast_alu_result),
  RKLowerer("reformat", lambda nodes:not _has_reduction(nodes), lower_reformat_result),
  RKLowerer("affine_mean", lambda nodes:_has_reduction(nodes, Ops.ADD) and sum(u.op is Ops.REDUCE for u in nodes) == 2,
            lower_affine_mean_result),
  RKLowerer("nested_sum", lambda nodes:_has_reduction(nodes, Ops.ADD) and sum(u.op is Ops.REDUCE for u in nodes) > 1,
            lower_nested_add_reduce_result),
  RKLowerer("scalar_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_scalar_mul_reduce_result),
  RKLowerer("masked_affine_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_masked_affine_mul_reduce_result),
  RKLowerer("affine_mul", lambda nodes:_has_reduction(nodes, Ops.MUL), lower_affine_mul_reduce_result),
  RKLowerer("sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_add_reduce_result),
  RKLowerer("multi_source_sum", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_multi_source_affine_reduce_result),
  RKLowerer("pointwise_affine_reduce", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_pointwise_affine_reduce_result),
  RKLowerer("affine_reduce", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_affine_reduce_result),
  RKLowerer("multi_source_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_multi_source_affine_reduce_result),
  RKLowerer("ppu_reduce", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_reduce_result),
  RKLowerer("sliding_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_sliding_max_result),
  RKLowerer("dense_row_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_dense_row_max_result),
  RKLowerer("affine_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_affine_max_result),
  RKLowerer("global_max", lambda nodes:_has_reduction(nodes, Ops.MAX), lower_global_max_result),
  RKLowerer("depthwise_spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_depthwise_spatial_contract_result),
  RKLowerer("grouped_spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_grouped_spatial_contract_result),
  RKLowerer("nhwc_spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_nhwc_spatial_contract_result),
  RKLowerer("deconvolution", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_deconv_result),
  RKLowerer("spatial_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_spatial_contract_result),
  RKLowerer("tiled_contract", lambda nodes:_has_reduction(nodes, Ops.ADD), lower_tiled_contract_result),
  RKLowerer("contract", lambda nodes:_has_reduction(nodes) and not _has_reduction(nodes, Ops.MAX), lower_contract_result),
)

def lower_native(sink:UOp) -> RKLowerResult:
  return select_lowering(sink, _LOWERERS)

class RockchipRenderer(Renderer):
  has_local, has_shared, supports_float4 = False, False, False
  def __init__(self, target:Target): super().__init__(target)
  def supported_dtypes(self): return {dtypes.half, dtypes.int, dtypes.float}
  def native_program(self, ast:UOp) -> UOp|None:
    fallback = os.getenv("ROCKCHIP_FALLBACK", "0").upper()
    if fallback not in ("", "0", "PYTHON", "CLANG", "HOST"):
      raise RuntimeError(f"invalid ROCKCHIP_FALLBACK={fallback!r}")
    if fallback == "HOST":
      from tinygrad.runtime.rockchip_fallback import build_rkhc_program
      return build_rkhc_program(ast, self.target)
    info = ProgramInfo.from_sink(ast, self.target)
    params = tuple(sorted((u for u in ast.toposort() if u.op is Ops.PARAM and u.arg.slot >= 0), key=lambda u:u.arg.slot))
    result = lower_native(ast)
    if result.reject is not None:
      reject = result.reject
      record_telemetry("reject", lane="REJECT", program=info.name, reject_kind=reject.kind.value, detail=reject.detail,
        node_op=reject.node_op.name if reject.node_op is not None else None, fingerprint=reject.fingerprint,
        fingerprint_digest=dict(reject.fingerprint)["graph"],
        signature=[{"slot": u.arg.slot, "dtype": u.dtype.name,
                    "shape": [x if isinstance(x, int) else str(x) for x in u.shape]} for u in params])
      if fallback == "PYTHON":
        from tinygrad.runtime.rockchip_fallback import build_rkpy_program
        return build_rkpy_program(ast, self.target)
      if fallback == "CLANG":
        from tinygrad.runtime.rockchip_fallback import build_rkhc_program
        return build_rkhc_program(ast, self.target)
      raise RuntimeError(f"RKPLAN_REJECT:{reject.kind.value}:{reject.detail}")
    if isinstance(result.plan, RKDPUProgram): image = emit_dpu(result.plan)
    elif isinstance(result.plan, RKCMACTask): image = emit_cmac_task(result.plan)
    elif isinstance(result.plan, RKConvTask): image = emit_spatial_conv(result.plan)
    elif isinstance(result.plan, RKPool): image = emit_pool(result.plan)
    elif isinstance(result.plan, RKReduce): image = emit_reduce(result.plan)
    elif isinstance(result.plan, RKLegalizedReformat): image = emit_reformat(result.plan)
    elif isinstance(result.plan, RKProgram): image = emit_program(result.plan)
    else: raise RuntimeError("invalid Rockchip lowering result")
    # Rejected WIP: a blanket cost policy diverted legal CONV/deCONV plans to generic HALF host arithmetic, which missed
    # Torch in 19--20% of outputs. Task/constant cost classifies performance; it does not authorize semantic substitution.
    linear = UOp(Ops.LINEAR, src=tuple(u for u in params if u.addrspace is not AddrSpace.ALU))
    return UOp(Ops.PROGRAM, src=(ast, linear, UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=encode_image(image))), arg=info)
