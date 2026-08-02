from __future__ import annotations
import struct

from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import rockchip as rk, rockchip_lut as rklut
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId
from tinygrad.uop.ops import Ops

from tinygrad.renderer.rockchip.ir import (RKTarget, RKEngine, RKBufferKind, RKArg, RKALUStage, RKMaskStage, RKLUTStage,
  RKDPUProgram, RKLayoutKind, RKContract, RKReduce, RKProgram)
from tinygrad.renderer.rockchip.image import RK_STAGE_RESET, RKReloc, RKStage, RKImage

_TARGET_DPU, _TARGET_DPU_RDMA, _TARGET_PC = 0x1001, 0x2001, 0x81
_TARGET_CNA, _TARGET_CORE = 0x201, 0x801
_TARGET_PPU, _TARGET_PPU_RDMA = 0x4001, 0x8001
_EW_BASE = 0x108002c0
_ERDMA_FP16 = 0x40000008
_EW_CFG = {Ops.ADD:_EW_BASE | (2 << 16), Ops.MUL:_EW_BASE | (1 << 2) | (1 << 8), Ops.MAX:_EW_BASE,
           Ops.SUB:_EW_BASE | (4 << 16), Ops.FDIV:_EW_BASE | (3 << 16) | (1 << 8)}

def _command(target:int, reg:int, value:int) -> int:
  return ((target & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _emit_mask(stage_idx:int, plan:RKMaskStage) -> RKStage:
  width = (plan.count+7)//8-1
  # Rejected WIP: setting out_precision=int8 for a final public bool mask timed out on RK3588; exact probe is archived separately.
  regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (rk.REG_DPU_DATA_CUBE_WIDTH, width), (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
    (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53),
    (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0), (rk.REG_DPU_BS_OW_CFG, 2),
    (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width), (rk.REG_DPU_BN_MUL_CFG, 0),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0), (rk.REG_DPU_BS_CFG, 0x40040), (rk.REG_DPU_BS_ALU_CFG, 0x33800000),
    (rk.REG_DPU_BS_MUL_CFG, 0x40000000), (rk.REG_DPU_BN_CFG, 0x40082), (rk.REG_DPU_BN_MUL_CFG, 0x7c000000),
    (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0x3f800000), (rk.REG_DPU_EW_CFG, _EW_BASE|1), (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
    (rk.REG_DPU_OUT_CVT_OFFSET, 0), (rk.REG_DPU_OUT_CVT_SCALE, 0x10001), (rk.REG_DPU_OUT_CVT_SHIFT, 0),
    (rk.REG_DPU_SURFACE_ADD, 0x40))
  cmds = [_command(_TARGET_DPU, reg, value) for reg,value in regs]
  cmds += [_command(_TARGET_DPU_RDMA, reg, value) for reg,value in ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7), (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, _ERDMA_FP16))]
  relocs = []
  for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, plan.src),
                              (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, plan.src)):
    cmds.append(_command(target_id, reg, 0))
    relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
  cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849), _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  return RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET)

def _emit_roundoff(stage_idx:int, plan:RKLUTStage) -> RKStage:
  width, surf_stride, cmds = (plan.count+7)//8-1, ((plan.count+7)//8)*16, []
  for table_id in range(2):
    cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_CFG, (1 << 17) | (table_id << 16)))
    cmds.extend(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_DATA, value) for value in
                rklut.RK_LUT_ROUNDOFF[table_id*rklut.RK_LUT_ROUNDOFF_ENTRIES:(table_id+1)*rklut.RK_LUT_ROUNDOFF_ENTRIES])
  dpu = ((rk.REG_DPU_S_POINTER, 0x30), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (rk.REG_DPU_DST_SURF_STRIDE, surf_stride), (rk.REG_DPU_DATA_CUBE_WIDTH, width), (rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007),
    (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BS_OW_CFG, 2), (rk.REG_DPU_WDMA_SIZE_0, 7), (rk.REG_DPU_WDMA_SIZE_1, width),
    (rk.REG_DPU_BN_CFG, 0x53), (rk.REG_DPU_EW_CFG, 0x302), (rk.REG_DPU_SURFACE_ADD, 2*surf_stride), (0x40c4, 0),
    (rk.REG_DPU_LUT_CFG, 0x68), (rk.REG_DPU_LUT_INFO, 0xe0e00), (rk.REG_DPU_LUT_LE_START, 0),
    (rk.REG_DPU_LUT_LE_END, 0x44000000), (rk.REG_DPU_LUT_LO_START, 0x44000000), (rk.REG_DPU_LUT_LO_END, 0x44800000),
    (0x4120, 23107), (0x4124, 22))
  cmds.extend(_command(_TARGET_DPU, *x) for x in dpu[:3])
  dst_word = len(cmds)
  cmds.append(_command(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0))
  cmds.extend(_command(_TARGET_DPU, *x) for x in dpu[3:])
  rdma = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x30), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
    (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7), (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 1))
  cmds.extend(_command(_TARGET_DPU_RDMA, *x) for x in rdma)
  src_word = len(cmds)
  cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0))
  cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
           _command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101), _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
  relocs = (RKReloc(stage_idx, dst_word, plan.dst.kind, plan.dst.index, plan.dst.addend),
            RKReloc(stage_idx, src_word, plan.src.kind, plan.src.index, plan.src.addend))
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, RK_STAGE_RESET)

def _emit_lut(stage_idx:int, plan:RKLUTStage) -> RKStage:
  if plan.lut is RKLUTId.ROUNDOFF: return _emit_roundoff(stage_idx, plan)
  if plan.lut not in (RKLUTId.EXP2, RKLUTId.EXP, RKLUTId.EXP_LOCAL, RKLUTId.EXPM1, RKLUTId.EXPM1_LOCAL,
                      RKLUTId.SIGMOID, RKLUTId.SIGMOID_LOCAL, RKLUTId.TANH, RKLUTId.TANH_MID, RKLUTId.TANH_LOCAL,
                      RKLUTId.SQRT, RKLUTId.RSQRT, RKLUTId.LOG2, RKLUTId.LOG2_LOCAL, RKLUTId.LOG10, RKLUTId.LOG10_LOCAL,
                      RKLUTId.ASIN, RKLUTId.ASIN_LOCAL, RKLUTId.ASIN_EDGE, RKLUTId.ACOS, RKLUTId.ATAN,
                      RKLUTId.ATANH, RKLUTId.ATANH_EDGE, RKLUTId.ASINH, RKLUTId.ASINH_MID,
                      RKLUTId.ACOSH, RKLUTId.ACOSH_MID, RKLUTId.ACOSH_EDGE, RKLUTId.ASINH_NEAR,
                      RKLUTId.SINH, RKLUTId.COSH, RKLUTId.ERF, RKLUTId.ERF_LOCAL, RKLUTId.SOFTPLUS_NEG,
                      RKLUTId.SOFTPLUS_DIV3_NEAR, RKLUTId.SOFTPLUS_DIV3_FAR, RKLUTId.MISH, RKLUTId.MISH_MID,
                      RKLUTId.HARDSWISH, RKLUTId.QUICK_GELU, RKLUTId.QUICK_GELU_LOCAL, RKLUTId.GELU_TANH,
                      RKLUTId.GELU_TANH_LOCAL, RKLUTId.GELU_EXACT, RKLUTId.GELU_EXACT_LOCAL, RKLUTId.ELU1,
                      RKLUTId.ELU1_LOCAL, RKLUTId.ELU01, RKLUTId.ELU01_LOCAL, RKLUTId.SELU, RKLUTId.SELU_LOCAL,
                      RKLUTId.CELU2, RKLUTId.CELU2_LOCAL, RKLUTId.CELU3, RKLUTId.CELU3_LOCAL, RKLUTId.CELU4, RKLUTId.CELU4_LOCAL,
                      RKLUTId.POW8, RKLUTId.POW8_HIGH, RKLUTId.POW55, RKLUTId.POW55_LOCAL, RKLUTId.POW55_HIGH,
                      RKLUTId.POW_NEG55_LOW, RKLUTId.POW_NEG55_HIGH, RKLUTId.POW_NEG55_FAR,
                      RKLUTId.POW_BASE55_LOW, RKLUTId.POW_BASE55_HIGH, RKLUTId.POW_BASE8_FAR_LOW,
                      RKLUTId.POW_BASE8_LOW, RKLUTId.POW_BASE8_HIGH, RKLUTId.POW_BASE8_FAR_HIGH):
    raise ValueError(f"unimplemented Rockchip LUT {plan.lut}")
  name = plan.lut.name
  table, entries = getattr(rklut, f"RK_LUT_{name}"), getattr(rklut, f"RK_LUT_{name}_ENTRIES")
  bn_mul, minus_exp = getattr(rklut, f"RK_LUT_{name}_BN_MUL"), getattr(rklut, f"RK_LUT_{name}_MINUS_EXP")
  width, surf_stride, cmds = (plan.count+7)//8-1, ((plan.count+7)//8)*16, []
  for table_id in range(2):
    cmds.append(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_CFG, (1 << 17) | (table_id << 16)))
    cmds.extend(_command(_TARGET_DPU, rk.REG_DPU_LUT_ACCESS_DATA, value) for value in
                table[table_id*entries:(table_id+1)*entries])
  fixed = (
    (_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x30), (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x30),
    (_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5), (_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002),
    (_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, surf_stride), (_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, width),
    (_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x70007), (_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    (_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 2), (_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, 7),
    (_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, width), (_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x20040),
    (_TARGET_DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000), (_TARGET_DPU, rk.REG_DPU_BN_MUL_CFG, bn_mul << 16),
    (_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x302), (_TARGET_DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
    (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001), (_TARGET_DPU, rk.REG_DPU_OUT_CVT_SHIFT, minus_exp << 12),
    (_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 2*surf_stride), (_TARGET_DPU, 0x40c4, 0),
    (_TARGET_DPU, rk.REG_DPU_LUT_CFG, 0x68), (_TARGET_DPU, rk.REG_DPU_LUT_INFO, 0x50500),
    (_TARGET_DPU, rk.REG_DPU_LUT_LE_START, 0xffffc000), (_TARGET_DPU, rk.REG_DPU_LUT_LE_END, 0),
    (_TARGET_DPU, rk.REG_DPU_LUT_LO_START, 0), (_TARGET_DPU, rk.REG_DPU_LUT_LO_END, 0x4000),
    (_TARGET_DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 16434 << 16), (_TARGET_DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 13 << 5),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width), (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 1), (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17849),
    (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT, 0x01010101), (_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
  cmds.extend(_command(*x) for x in fixed[:4])
  dst_word = len(cmds)
  cmds.append(_command(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[4:30])
  src_word = len(cmds)
  cmds.append(_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0))
  cmds.extend(_command(*x) for x in fixed[30:])
  relocs = (RKReloc(stage_idx, dst_word, plan.dst.kind, plan.dst.index, plan.dst.addend),
            RKReloc(stage_idx, src_word, plan.src.kind, plan.src.index, plan.src.addend))
  return RKStage(RKEngine.DPU, tuple(cmds), relocs, RK_STAGE_RESET)

def emit_dpu(program:RKDPUProgram, target:RKTarget=RKTarget.RK3588) -> RKImage:
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  constants, constant_offsets, stages = bytearray(), {}, []
  def materialize(value:RKArg|float, count:int) -> RKArg:
    if isinstance(value, RKArg): return value
    bits, key = struct.pack("<e", value), (value, count)
    if key not in constant_offsets:
      constant_offsets[key] = len(constants)
      constants.extend(bits * (((count+7)//8)*8))
    return RKArg(RKBufferKind.CONSTANT, constant_offsets[key])
  for stage_idx, plan in enumerate(program.stages):
    if isinstance(plan, RKLUTStage):
      stages.append(_emit_lut(stage_idx, plan))
      continue
    if isinstance(plan, RKMaskStage):
      stages.append(_emit_mask(stage_idx, plan))
      continue
    if not isinstance(plan, RKALUStage): raise ValueError(f"unimplemented Rockchip stage {type(plan).__name__}")
    material_count = plan.count*2 if plan.out_dtype is dtypes.int else (32 if plan.out_dtype is dtypes.float else plan.count)
    lhs, rhs = materialize(plan.lhs, material_count), materialize(plan.rhs, material_count)
    width = ((plan.count*2 if plan.out_dtype is dtypes.int else plan.count)+7)//8-1
    wide_out = plan.out_dtype in (dtypes.int, dtypes.float)
    lanes = 8 if wide_out or plan.count >= 8 else plan.count
    out_precision = 4 if plan.out_dtype is dtypes.int else (5 if plan.out_dtype is dtypes.float else 2)
    dpu_regs = ((rk.REG_DPU_S_POINTER, 0xe), (rk.REG_DPU_FEATURE_MODE_CFG, 0x1e5),
      (rk.REG_DPU_DATA_FORMAT, (out_precision<<29)|(2<<26)|2), (rk.REG_DPU_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0),
      (rk.REG_DPU_DATA_CUBE_CHANNEL, ((lanes-1)<<16)|(lanes-1)),
      (rk.REG_DPU_BS_CFG, 0x53), (rk.REG_DPU_BN_CFG, 0x53), (rk.REG_DPU_BS_ALU_CFG, 0), (rk.REG_DPU_BS_MUL_CFG, 0),
      (rk.REG_DPU_BS_OW_CFG, 2), (rk.REG_DPU_WDMA_SIZE_0, 3 if wide_out else lanes-1), (rk.REG_DPU_WDMA_SIZE_1, width),
      (rk.REG_DPU_BN_MUL_CFG, 0), (rk.REG_DPU_BN_RELUX_CMP_VALUE, 0), (rk.REG_DPU_EW_CFG, _EW_CFG[plan.op]),
      (rk.REG_DPU_EW_CVT_SCALE_VALUE, 1), (rk.REG_DPU_OUT_CVT_OFFSET, 0),
      (rk.REG_DPU_OUT_CVT_SCALE, 0 if plan.out_dtype is dtypes.float else (1 if plan.op is Ops.FDIV or
       plan.out_dtype is dtypes.int else 0x10001)), (rk.REG_DPU_OUT_CVT_SHIFT, 0), (rk.REG_DPU_SURFACE_ADD, 0x40))
    rdma_regs = ((rk.REG_DPU_RDMA_RDMA_S_POINTER, 0xe), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
      (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0), (rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, lanes-1),
      (rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, _ERDMA_FP16))
    cmds = [_command(_TARGET_DPU, *x) for x in dpu_regs] + [_command(_TARGET_DPU_RDMA, *x) for x in rdma_regs]
    relocs = []
    for target_id, reg, arg in ((_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, plan.dst),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, lhs),
                                (_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, rhs)):
      cmds.append(_command(target_id, reg, 0))
      relocs.append(RKReloc(stage_idx, len(cmds)-1, arg.kind, arg.index, arg.addend))
    cmds += [_command(_TARGET_DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x17841 if plan.op is Ops.FDIV else 0x17849),
             _command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)]
    stages.append(RKStage(RKEngine.DPU, tuple(cmds), tuple(relocs), RK_STAGE_RESET))
  return RKImage(target, tuple(stages), program.scratch, bytes(constants))

def emit_contract(plan:RKContract, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Emit one direct FP16 CMAC task; all surfaces are already hardware-legal."""
  if target is not RKTarget.RK3588: raise ValueError(f"unsupported Rockchip target {target}")
  if plan.rhs.layout.kind is not RKLayoutKind.CMAC_WEIGHT: raise ValueError("CMAC RHS is not in weight layout")
  e, align_out, align_in = _command, plan.rhs.layout.physical_shape[0], plan.lhs.layout.physical_shape[-1]
  m = plan.lhs.layout.physical_shape[0]
  if align_in < 32 or align_in % 32: raise ValueError("CMAC K must be aligned to 32")
  if align_out != 32: raise ValueError("proven CMAC output tile is 32 physical channels")
  input_row_bytes = align_in*2
  feature_grains = max(80, (((2*256*128+input_row_bytes-1)//input_row_bytes)+1)&-2)
  data_banks = min(11, max(1, (m*input_row_bytes+32767)//32768))
  line_stride = 4*min(align_in//32, 13)
  notch = 8*min(align_out//32, 13)-1
  commands = (
    e(_TARGET_DPU, rk.REG_DPU_S_POINTER, 0x0e), e(_TARGET_CNA, rk.REG_CNA_CONV_CON1, 0x20000120),
    e(_TARGET_CNA, rk.REG_CNA_CONV_CON2, feature_grains<<4), e(_TARGET_CNA, rk.REG_CNA_CONV_CON3, 9),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE0, (1<<16)|m), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE1, ((align_in-1)<<16)|align_in),
    e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE2, 1), e(_TARGET_CNA, rk.REG_CNA_DATA_SIZE3, m),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE0, input_row_bytes*align_out), e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE1, input_row_bytes),
    e(_TARGET_CNA, rk.REG_CNA_WEIGHT_SIZE2, 0x1010000|align_out),
    e(_TARGET_CNA, rk.REG_CNA_CBUF_CON0, ((12-data_banks)<<4)|data_banks),
    e(_TARGET_CNA, rk.REG_CNA_CBUF_CON1, align_in//32), e(_TARGET_CNA, rk.REG_CNA_CVT_CON0, 0xb),
    *(e(_TARGET_CNA, reg, 0x10000) for reg in (rk.REG_CNA_CVT_CON1, rk.REG_CNA_CVT_CON2, rk.REG_CNA_CVT_CON3, rk.REG_CNA_CVT_CON4)),
    e(_TARGET_CNA, rk.REG_CNA_FEATURE_DATA_ADDR, 0), e(_TARGET_CNA, rk.REG_CNA_DMA_CON0, 0xf000f),
    e(_TARGET_CNA, rk.REG_CNA_DMA_CON1, line_stride), e(_TARGET_CNA, rk.REG_CNA_DMA_CON2, 0),
    e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE0, (1<<16)|m), e(_TARGET_CNA, rk.REG_CNA_FC_DATA_SIZE1, align_in),
    e(_TARGET_CNA, rk.REG_CNA_DCOMP_ADDR0, 0), e(_TARGET_CORE, rk.REG_CORE_MISC_CFG, 0x201),
    e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_0, (m-1)<<16), e(_TARGET_CORE, rk.REG_CORE_DATAOUT_SIZE_1, align_out-1),
    e(_TARGET_CORE, rk.REG_CORE_RESERVED_3030, 0), e(_TARGET_DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x1e4),
    e(_TARGET_DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002), e(_TARGET_DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
    e(_TARGET_DPU, rk.REG_DPU_DST_SURF_STRIDE, 0x10), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, m-1), e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, (notch<<16)|notch),
    e(_TARGET_DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, ((align_out-1)<<16)|(align_out-1)), e(_TARGET_DPU, rk.REG_DPU_BS_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_BS_OW_CFG, 0x126), e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_0, align_out-1),
    e(_TARGET_DPU, rk.REG_DPU_WDMA_SIZE_1, (m-1)<<16), e(_TARGET_DPU, rk.REG_DPU_BN_CFG, 0x53),
    e(_TARGET_DPU, rk.REG_DPU_EW_CFG, 0x383), e(_TARGET_DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x10001),
    e(_TARGET_DPU, rk.REG_DPU_SURFACE_ADD, 0x40), e(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0xd))
  relocs = tuple(RKReloc(0, word, ref.buffer.kind, ref.buffer.index, ref.buffer.addend+ref.layout.base_offset)
                 for word,ref in ((18,plan.lhs), (24,plan.rhs), (31,plan.out)))
  return RKImage(target, (RKStage(RKEngine.CMAC, commands, relocs, RK_STAGE_RESET),), constants=plan.constants)

def emit_program(plan:RKProgram, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Compose arbitrary typed engine steps into one ordered sequential image."""
  images:list[RKImage] = []
  for step in plan.steps:
    if isinstance(step, RKDPUProgram): images.append(emit_dpu(step, target))
    elif isinstance(step, RKContract): images.append(emit_contract(step, target))
    elif isinstance(step, RKReduce): images.append(emit_reduce(step, target))
    else: raise TypeError(f"unsupported Rockchip program step {type(step).__name__}")
  stages:list[RKStage] = []
  constants, constant_offsets = bytearray(), {}
  for image in images:
    if image.constants not in constant_offsets:
      constant_offsets[image.constants] = len(constants)
      constants.extend(image.constants)
    constant_base = constant_offsets[image.constants]
    for stage in image.stages:
      relocs = tuple(RKReloc(len(stages), reloc.word, reloc.kind,
        reloc.index+(constant_base if reloc.kind is RKBufferKind.CONSTANT else 0), reloc.addend, reloc.shift, reloc.mask, reloc.field_shift)
        for reloc in stage.relocs)
      stages.append(RKStage(stage.engine, stage.commands, relocs, stage.flags))
  return RKImage(target, tuple(stages), plan.scratch, bytes(constants))

def emit_reduce(plan:RKReduce, target:RKTarget=RKTarget.RK3588) -> RKImage:
  """Emit the proven direct PPU global-MAX program for a dense FP16 HWC8 surface."""
  if target is not RKTarget.RK3588 or plan.op is not Ops.MAX: raise ValueError("unsupported Rockchip PPU reduction")
  height, width, channels = plan.src.layout.logical_shape
  if channels != 8 or not 2 <= height <= 16 or not 2 <= width <= 16: raise ValueError("PPU global MAX requires 2..16 x 2..16 x 8")
  h, w, c = height-1, width-1, channels-1
  regs = (
    (_TARGET_PPU, rk.REG_PPU_S_POINTER, 0xe), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_S_POINTER, 0xe),
    (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_WIDTH, w), (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_HEIGHT, h),
    (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_IN_CHANNEL, c), (_TARGET_PPU, rk.REG_PPU_DATA_CUBE_OUT_CHANNEL, c),
    (_TARGET_PPU, rk.REG_PPU_OPERATION_MODE_CFG, 0x11),
    (_TARGET_PPU, rk.REG_PPU_POOLING_KERNEL_CFG, (h<<20)|(w<<16)|(h<<8)|w),
    (_TARGET_PPU, rk.REG_PPU_DST_SURF_STRIDE, 1), (_TARGET_PPU, rk.REG_PPU_DATA_FORMAT, 0x10002),
    (_TARGET_PPU, rk.REG_PPU_MISC_CTRL, 3), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_CUBE_IN_WIDTH, w),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_CUBE_IN_HEIGHT, h), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_CUBE_IN_CHANNEL, c),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_SRC_LINE_STRIDE, plan.src.layout.strides_bytes[0]),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_SRC_SURF_STRIDE, height*plan.src.layout.strides_bytes[0]),
    (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_DATA_FORMAT, 2), (_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_OPERATION_ENABLE, 1))
  commands = [_command(*x) for x in regs]
  dst_word = len(commands)
  commands.append(_command(_TARGET_PPU, rk.REG_PPU_DST_BASE_ADDR, 0))
  src_word = len(commands)
  commands.append(_command(_TARGET_PPU_RDMA, rk.REG_PPU_RDMA_SRC_BASE_ADDR, 0))
  commands.append(_command(_TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x60))
  relocs = (RKReloc(0, dst_word, plan.out.buffer.kind, plan.out.buffer.index, plan.out.buffer.addend+plan.out.layout.base_offset),
            RKReloc(0, src_word, plan.src.buffer.kind, plan.src.buffer.index, plan.src.buffer.addend+plan.src.layout.base_offset))
  return RKImage(target, (RKStage(RKEngine.PPU, tuple(commands), relocs, RK_STAGE_RESET),))
