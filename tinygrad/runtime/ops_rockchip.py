# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Any, TYPE_CHECKING
import pickle, base64, itertools, time, struct, sys, functools, ctypes, mmap, os, math, numpy as np
from dataclasses import replace
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate, storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar
from tinygrad.helpers import DEBUG, all_same, getenv, flatten, get_single_element
from tinygrad.device import Compiled, Compiler, Allocator
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp, AxisType, bitcast
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_python import storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar, _load, load, _store, generic_wmma_helper
from tinygrad.runtime.support.hcq import HCQBuffer
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.uop.ops import PatternMatcher, UPat
from tinygrad.helpers import mv_address, getenv

def _load(m, i, dtype: DType):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return from_storage_scalar(m[i], dtype)

def load(inp, j, dtype: DType):
  if len(inp) >= 3: return [_load(m, x+j if x is not None else None, dtype) if gate else default for (m,x),default,gate in zip(*inp[:3])]
  return [_load(m, x+j if x is not None else None, dtype) for m,x in inp[0]]

def _store(m, i, v, dtype: DType):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = to_storage_scalar(v, dtype)

# here are the models for the WMMA instruction on the different hardware
def generic_wmma_helper(inp, warp_size, WARP_THREADS, K, NUM_A, NUM_B, NUM_C, a_elem, b_elem, c_map):
  for cc, tinp, num in zip(("A", "B", "C"), inp, (NUM_A, NUM_B, NUM_C)):
    assert len(tinp) == num, f"{cc} must have {num} elements per thread, it has {len(tinp)}"
    assert len(flatten(tinp)) == num * warp_size, f"WMMA must have {num * warp_size} total elements for {cc} in WMMA"
  assert warp_size > 0 and warp_size % WARP_THREADS == 0, f"must have multiples of {WARP_THREADS} warp threads"
  out = [inp[2][elem_idx][:] for elem_idx in range(NUM_C)]
  for goff in range(0, warp_size, WARP_THREADS):
    for lane_id in range(WARP_THREADS):
      for elem_idx in range(NUM_C): # calculate new muls and add to acc
        (c_i, c_j) = c_map(lane_id, elem_idx)
        out[elem_idx][goff+lane_id] += sum(a_elem(inp[0], _k, c_j, goff) * b_elem(inp[1], c_i, _k, goff) for _k in range(K))
  return out

class RockchipProgram:
  def __init__(self, dev:'RockchipDevice', name:str, lib:bytes,
               runtimevars:dict[str, int]|None=None, **kwargs):
    prg = pickle.loads(lib)
    self.cna_meta = None
    self.cna_kind = None
    self.fallback_uops: list[UOp]|None = None
    if isinstance(prg, tuple) and len(prg) == 3 and prg[0] in ("rkcna_v1", "rkcna_gemv_v1"):
      self.cna_kind = prg[0]
      self.cna_meta, self.fallback_uops = prg[1], prg[2]
      self.uops = self.fallback_uops
    else:
      self.uops: list[UOp] = prg
    self.uop_to_index: dict[UOp, int] = {u:i for i,u in enumerate(self.uops)}
    self.loop_ends: dict[UOp, int] = {u.src[1]:i for i, u in enumerate(self.uops) if u.op == Ops.END}
    self.runtimevars = runtimevars or {}
    self.device = dev
    self.q = []
    self.hardware_ops = {Ops.SHL:0, Ops.TRUNC:0, Ops.CUSTOM:0, Ops.MUL:0, Ops.NEG:0, Ops.MAX:0, Ops.EXP2:0, Ops.CMPLT:0, Ops.CMPEQ:0, Ops.ADD:2, Ops.FDIV:3, Ops.SUB:4}
    self.cmd_buf_size = 16384
    self.exp2_inv_scale = 1.0
    self.lut_size = 513
  def check_lut_enable(self, op, arg):
    return op in (Ops.EXP2, Ops.TRUNC) or (op is Ops.CUSTOM and arg == "silu")
  def reg(self, val, shift, mask):
    return ((val) << shift) & mask
  def emit_raw(self, target, reg, value):
    # Pack the values into a 64-bit integer as per hardware spec
    target = target + 0x1
    packed_value = ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF)
    self.q.append(packed_value)
  def fill_lut(self, lut):
    for table_id, base in ((0, 0), (1, self.lut_size)):
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG,
          self.reg(1, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK) |
          self.reg(table_id, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK) |
          self.reg(0, rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__MASK))
      for i in range(self.lut_size):
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
          self.reg(lut[base + i], rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))

  def boilerplate(self, op, size, arg):
    if self.lut_enable:
      lut = [0] * self.lut_size * 2
      index_shift = 5
      index_scale = 0.0
      if op is Ops.EXP2:
        x_min, x_max = -2.0, 2.0
        step = (x_max - x_min) / (len(lut) - 1)
        index_scale = (1 << index_shift) / step
        
        max_val = max(math.exp2(x_min), math.exp2(x_max))
        self.inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
        for i in range(len(lut)):
          x = x_min + i * step
          y = math.exp2(x) * self.inv_scale
          q = int(math.floor((y + 1.0) * 2**14 + 0.5))
          lut[i] = np.clip(q, 0, 32767)
      elif op is Ops.CUSTOM and arg == "silu":
        x_min, x_max = 0, 5.8
        step = (x_max - x_min) / (self.lut_size - 1)
        index_scale = (1 << index_shift) / step

        max_val = max(x_min / (1.0 + math.exp(-x_min)), x_max / (1.0 + math.exp(-x_max)))
        self.inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
        for i in range(self.lut_size * 2):
          x = (i - self.lut_size + (i < self.lut_size)) * step
          y = x / (1.0 + math.exp(-x)) * self.inv_scale 
          q = int(math.floor(y * (2**15 - 1) + 0.5)) if y >= 0.0 else int(math.ceil(y * (2**15 - 1) - 0.5))
          lut[i] = np.clip(q, -32768, 32767)
      elif op is Ops.TRUNC:
        max_val = 1 << 14
        for table_id in range(2):
          base = table_id * self.lut_size
          for i in range(self.lut_size):
            lut[base + i] = 0 if (i % 2 == 0) else max_val
      bn_mul_operand = int(np.float16(index_scale).view(np.int16)) if index_scale!=0 else 0x3C00

      self.fill_lut(lut)
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_CFG,
          self.reg(1, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK) |
          self.reg(1, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK) |
          self.reg(2, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__MASK))
      index_select = 14 if op is Ops.TRUNC else 5
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_INFO,
          self.reg(index_select, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK) |
          self.reg(index_select, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK))
      if op is Ops.TRUNC:
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
            self.reg(0x00000000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_END,
            self.reg(0x44000000, rk.DPU_LUT_LE_END_LUT_LE_END__SHIFT, rk.DPU_LUT_LE_END_LUT_LE_END__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_START,
            self.reg(0x44000000, rk.DPU_LUT_LO_START_LUT_LO_START__SHIFT, rk.DPU_LUT_LO_START_LUT_LO_START__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
            self.reg(0x44800000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
      else:
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
            self.reg(0xffffc000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
            self.reg(0x00004000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE,
          self.reg(23107, rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__SHIFT,
                  rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT,
          self.reg(22, rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__SHIFT,
                  rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
        self.reg(2, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
        self.reg(bn_mul_operand, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
      
    elif op is Ops.CUSTOM and arg == "cmplt_diff2bool":
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
      # DPU_BS perform ALU first then MUL
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
        self.reg(0x33800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        self.reg(0x4000, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
        self.reg(4, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT, rk.DPU_BN_CFG_BN_RELUX_EN__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
        self.reg(0x7C00, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE,
        self.reg(0x3F800000, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__SHIFT, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__MASK))
    elif op is Ops.CUSTOM and arg == "cmpeq_diff_zero_to_nan_to_32800":
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(2, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        self.reg(0x7C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
        self.reg(1, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
    elif op is Ops.CUSTOM and arg == "cmpeq_32800_to_bool":
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
        self.reg(0, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
        self.reg(0x47001F00, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        self.reg(0x3C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
      # REG_DPU_OUT_CVT_SHIFT need manual reset to 0
      self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
        self.reg(0, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
    elif op is Ops.SHL:
      self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
        self.reg(-2, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))

    burst_len = 15
    output_mode  = 2
    flying_mode = 1
    channel = 7
    dataout_height = 0
    dataout_width = math.ceil(size / ((dataout_height+1) * (channel+1))) - 1

    precision_float16 = 2

    ew_cvt_type = 0
    ew_data_mode = 1
    ew_data_size = 2
    ew_relu_bypass = arg != "relu"
    ew_alu_algo = self.hardware_ops.get(op, 0)
    ew_op_src = 1
    erdma_data_size_16bit=2
    if self.lut_enable: 
      ew_data_mode = 0; ew_data_size = 0; ew_op_src = 0; 
    
    self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
        self.reg(burst_len, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(output_mode, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
        self.reg(flying_mode, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
        self.reg(precision_float16, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
        self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
        self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
        self.reg(dataout_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(ew_cvt_type, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
        self.reg(ew_data_mode, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(ew_data_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(ew_alu_algo, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(op == Ops.MUL, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
        self.reg(ew_relu_bypass, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(op in [Ops.MUL, Ops.FDIV] or self.lut_enable, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        self.reg(self.lut_enable == False, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(ew_op_src, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
        self.reg(self.lut_enable == True, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
        self.reg(arg in ["cmplt_diff2bool", "cmpeq_diff_zero_to_nan_to_32800", "cmpeq_32800_to_bool"], rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK) 
      )
    # 0 or 1 both passed test_div, do not emit OUT_CVT_SCALE for other ops
    self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      self.reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK)) if op == Ops.FDIV else None

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
        self.reg(dataout_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
        self.reg(dataout_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
        self.reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
        self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
        self.reg(erdma_data_size_16bit, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))

  def submit(self, uop):
    # TODO fix special if, maybe MUL output defaulted as fp32 amd need FP16TOFP32
    if uop != Ops.FDIV: 
      # EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(2) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(1) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1));
      self.q.append(0x2001000178495044), 
    self.q.append(0x0081000000180008), # EMIT(REG_PC_OPERATION_ENABLE, PC_OPERATION_ENABLE_RESERVED_0(12))
    tasks = ctypes.cast(self.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
    assert len(self.q) <= self.cmd_buf_size
    regcmd = ctypes.cast(self.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * self.cmd_buf_size)).contents
    for i in range(len(self.q)):
      regcmd[i] = self.q[i]

    tasks[0].flags  = 0
    tasks[0].op_idx = 4
    tasks[0].enable_mask = 0x18
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = len(self.q)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = self.cmd_buf.meta.dma_addr

    # TODO: update parameter name as driver updated
    print("NPU Submit")
    submit_res = rk.struct_rknpu_submit(
            flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
            timeout=6000,
            task_start=0,
            task_number=1,
            task_counter=0,
            priority=0,
            task_obj_addr=self.task_buf.meta.obj_addr,   # Placeholder, would be actual address in real code
            regcfg_obj_addr=0,
            task_base_addr=0,
            user_data=0,
            core_mask=1,
            fence_fd=-1,
            subcore_task=(rk.struct_rknpu_subcore_task * 5)(
                rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
                rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
                rk.struct_rknpu_subcore_task(task_start=2, task_number=0),
            )
    )
    res = rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl,__payload=submit_res)
    # os.system("cd ~/npu/ops_reg/ && python dump.py 5")
    # print(res)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    if self.cna_meta is not None:
      return self._run_cna_group(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait, **kw)
    self.device.reset_npu()
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    void_ops = {Ops.END, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.STORE}
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      values: dict[UOp, Any] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      exec_masks = [[True] * warp_size]
      i = 0
      if DEBUG >= 7:
        for idx, u in enumerate(self.uops):
          print(idx, u.op, u.dtype, u.arg, [v.op for v in u.src])
      while i < len(self.uops):
        u = self.uops[i]
        src_values = [values[v] for v in u.src if v.op not in void_ops]
        src_dtypes = [v.dtype for v in u.src if v.op not in void_ops]
        if getenv("TRACE"): print(i, u.op, u.dtype, u.arg, src_values, src_dtypes)
        if u.op is Ops.END:
          i = self.uop_to_index[u.src[1]]
          continue
        if u.op is Ops.IF:
          exec_masks.append([x and y for x,y in zip(exec_masks[-1], src_values[0])])
          i += 1
          continue
        if u.op is Ops.ENDIF:
          exec_masks.pop()
          i += 1
          continue
        if u.op in (Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.GROUP):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert u.dtype is not None, f"{u.op} is missing a dtype"
        if u.op is Ops.STORE:
          assert len(src_values) == 2, f"STORE must be lowered to 2 srcs, got {len(src_values)}"
          store_gate = exec_masks[-1]
          for j,val in enumerate(src_values[1] if u.max_numel() > 1 else [src_values[1]]):
            for (m,o),v,g in zip(src_values[0], val, store_gate):
              if g: _store(m, o+j, v, src_dtypes[1].scalar())
          i += 1
          continue
        if u.op is Ops.AFTER: values[u] = src_values[0]
        elif u.op in {Ops.PARAM, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          storage_fmt = storage_fmt_for_dtype(u.dtype.base.scalar())
          if storage_fmt is None: raise RuntimeError(f"dtype={u.dtype} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if u.op is Ops.DEFINE_REG:
            # REGs are per thread
            values[u] = [memoryview(bytearray(u.max_numel()*u.dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(u.max_numel()*u.dtype.itemsize)) if u.op is not Ops.PARAM else pbufs.pop(0)
            values[u] = [buf.cast(storage_fmt)] * warp_size
        elif u.op is Ops.DEFINE_VAR:
          values[u] = [pvals.pop(0)] * warp_size
        elif u.op is Ops.SPECIAL:
          if u.arg[0] == 'g': values[u] = [idxs[2-int(u.arg[-1])]] * warp_size
          elif u.arg[0] == 'l': values[u] = [x[2-int(u.arg[-1])] for x in warp]
        elif u.op is Ops.CONST: values[u] = [u.arg] * warp_size
        elif u.op is Ops.INDEX:
          ret:list = []
          if isinstance(src_dtypes[0], ImageDType):
            assert len(src_values) == 3, "image index must be 3 srcs"
            for m,oy,ox in zip(*src_values):
              if ox < 0 or ox >= src_dtypes[0].shape[1] or oy < 0 or oy >= src_dtypes[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*src_dtypes[0].shape[1]*4))
          else:
            assert len(src_values) == 2, "non-image index must be 2 srcs"
            for m,o in zip(*src_values): ret.append((m,o))
          values[u] = ret
        elif u.op is Ops.CAST and isinstance(u.dtype, PtrDType):
          values[u] = src_values[0]
        elif u.op is Ops.RANGE:
          if u not in values: values[u] = [0] * warp_size
          else:
            for j in range(len(values[u])):
              values[u][j] += 1
          if values[u][0] == src_values[0][0]:
            del values[u]
            i = self.loop_ends[u] + 1
            continue
        elif u.op is Ops.STACK: values[u] = src_values
        elif u.op is Ops.BITCAST: values[u] = [bitcast(x, src_dtypes[0], u.dtype) for x in src_values[0]]
        elif u.op is Ops.CAST:
          values[u] = [truncate.get(u.dtype, lambda dt: dt)(u.dtype.const(x)) for x in src_values[0]]
        elif u.op is Ops.LOAD:
          if (load_sz := u.max_numel()) > 1:
            # buf and gate are not vecs
            values[u] = [load([src_values[k] if k in [0,2] else src_values[k][j] \
                               for k in range(len(src_values))], j, u.dtype.scalar()) for j in range(load_sz)]
          else:
            values[u] = load(src_values, 0, u.dtype)
        elif u.op is Ops.GEP: values[u] = src_values[0][get_single_element(u.arg)]
        elif u.op is Ops.WMMA:
          first_src_dtype = u.src[0].dtype
          assert isinstance(first_src_dtype, DType) # mypy
          dims, dtype_in, device, threads = u.arg[1], first_src_dtype.scalar(), u.arg[4], u.arg[5]
          wmma_helper = functools.partial(generic_wmma_helper, src_values, warp_size)
          # TODO: refactor these to a shared TensorCoreLayout
          if device == "METAL":
            # A (2 elements on 32 threads): row major
            def a_b_elem(x, i, j, goff): return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            # (i, j), C, D (2 elements on 32 threads): row major same as A/B
            def c_map(lane, elem): return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            values[u] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif device == "AMD" and threads == 64:
            def a_elem(x, k, row, goff): return x[k%(dims[2]//4)][goff + (k//(dims[2]//4))*16 + row]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, (lane//16)*4 + elem)
            values[u] = wmma_helper(64, dims[2], len(src_values[0]), len(src_values[1]), len(src_values[2]), a_elem, b_elem, c_map)
          elif device == "AMD" and len(src_values[0]) == 8: # RDNA4
            def a_elem(x, k, row, goff): return x[k - [0, 4, 4, 8][k//4]][goff + row + [0, 16, 0, 16][k//4]]
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)
            def c_map(lane, elem): return (lane%16, (lane//16)*8 + elem)
            values[u] = wmma_helper(32, 16, 8, 8, 8, a_elem, b_elem, c_map)
          elif device == "AMD":
            # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
            def a_elem(x, k, row, goff):
              assert x[k][goff+row] == x[k][goff+row+16], "warp elements not duplicated properly across lanes"
              return x[k][goff+row]
            # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
            def b_elem(x, col, k, goff): return a_elem(x, k, col, goff)  # pylint: disable=arguments-out-of-order
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            values[u] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif device == "CUDA":
            # (col, row) given (lane, elem) for C & D (4 elements on 32 threads); shared by all tc shapes with M=16 N=8
            def c_map(lane, elem): return (elem%2 + (lane%4)*2, lane//4 + (elem//2)*8)

            if dims == (8,16,16):
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2 + (k//8)*4][goff + (k//2)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2 + (k//8)*2][goff + (k//2)%4 + col*4]
              values[u] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,32):
              def a_elem(x, k, row, goff): return x[k%4 + (row//8)*4 + (k//16)*8][goff + (k//4)%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%4 + (k//16)*4][goff + (k//4)%4  + col*4]
              values[u] = wmma_helper(32, 32, 16, 8, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.half:
              def a_elem(x, k, row, goff): return x[k%2 + (row//8)*2][goff + k//2 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k%2][goff + k//2 + col*4]
              values[u] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            elif dims == (8,16,8) and dtype_in == dtypes.float:
              def a_elem(x, k, row, goff): return x[(k//4)*2 + row//8][goff + k%4 + (row%8)*4]
              def b_elem(x, col, k, goff): return x[k//4][goff + k%4 + col*4]
              values[u] = wmma_helper(32, 8, 4, 2, 4, a_elem, b_elem, c_map)

            else: raise NotImplementedError(f"unimplemented tensor core {u.arg}")
          else: raise NotImplementedError(f"unimplemented tensor core {u.arg}")
        elif u.op is Ops.CUSTOM or u.op in GroupOp.ALU:
          assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {u.op}"
          assert all_same([u.dtype] + src_dtypes) or u.op in {*GroupOp.Comparison, Ops.WHERE}, f"dtype mismatch on {u.op}"
          eff_op = u.op
          if u.op is Ops.CMPLT or (u.op in self.hardware_ops and u.dtype.scalar() in [dtypes.float16]):
            self.device.reset_npu()
            self.q = []
            self.lut_enable = self.check_lut_enable(eff_op, u.arg)
            if len(src_values)==1:
              if u.op is Ops.NEG:
                src_values.append([-1]*len(src_values[0]))
                eff_op = Ops.MUL
              else:
                src_values.append(src_values[0])
                eff_op = u.op
            self.boilerplate(op=eff_op, size=len(src_values[0]), arg=u.arg)

            src = memoryview(bytearray(np.asarray(src_values[0], dtype=np.float16).tobytes()))
            src2 = memoryview(bytearray(np.asarray(src_values[1], dtype=np.float16).tobytes()))
            self.task_buf = self.device._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task_buf")
            self.cmd_buf = self.device._gpu_alloc(self.cmd_buf_size, 0, name="cmd_buf")
            self.input_buf = self.device._gpu_alloc(src.nbytes, 0, name="input")
            self.weight_buf = self.device._gpu_alloc(src2.nbytes, 0, name="weight")
            self.output_buf = self.device._gpu_alloc(src.nbytes, 0, name="output")
            try:
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              self.device._gpu_sync(self.input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
              self.device._gpu_sync(self.weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)

              self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
                  self.reg(self.output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT,
                            rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
              self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
                self.reg(self.input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT,
                          rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
              self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
                self.reg(self.weight_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT,
                          rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))

              self.submit(eff_op)
              self.device._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)

              dst = memoryview(bytearray(self.output_buf.size))
              ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
              if getenv("TRACE"):  print(dst.tobytes().hex())
              # fp16 2B, self.output_buf.size//2
              result = struct.unpack(f'<{self.output_buf.size//2}e', dst.tobytes())
              if self.lut_enable:
                raw = np.rint(np.array(result, dtype=np.float32))
                # q14 decode
                if eff_op is Ops.EXP2:
                  result = ((raw.astype(np.uint16) / 2**14) - 1) / self.inv_scale
                elif u.arg == "silu":
                  result = raw.astype(np.int16) / (2**15 - 1) / self.inv_scale
              values[u] = list(result)
              if getenv("TRACE"):  print('src', src_values[0])
              if getenv("TRACE"):  print('src2', src_values[1])
              if getenv("TRACE"):  print('result', values[u])
              try:
                if getenv("TRACE"): print('expected', [exec_alu(eff_op, u.dtype, p) for p in zip(*src_values)])
              except: pass
            finally:
              self.device._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf, self.weight_buf, self.output_buf])
          else:
            allow_fallback = u.op in (Ops.XOR, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR)
            if allow_fallback:
              print('ALLOWED FALLBACK TO CPU', u.op, u.dtype)
              values[u] = [exec_alu(u.op, u.dtype, p) for p in zip(*src_values)]
            else:
              print('<!> EXIT OPERATION NOT SUPPORTED', u.op, u.dtype, src_values)
              if getenv("TRACE"): print('expected', [exec_alu(u.op, u.dtype, p) for p in zip(*src_values)])
        assert u in values, (u.op, u.dtype, u.src, u.arg)
        i += 1
    return time.perf_counter() - st

  def _run_cna_group(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    if len(bufs) != 3: raise NotImplementedError(f"rkcna_v1 expected 3 buffers, got {len(bufs)}")
    st = time.perf_counter()
    out_buf, a_buf, b_buf = bufs
    solved = None
    if self.cna_kind == "rkcna_gemv_v1" and self.cna_meta is not None:
      m, n, k = (int(self.cna_meta[x]) for x in ("m", "n", "k"))
      batch = int(self.cna_meta.get("batch", 1))
      src_itemsize = int(self.cna_meta.get("src_itemsize", 2))
      out_itemsize = int(self.cna_meta.get("out_itemsize", 2))
      a_len = m*k*src_itemsize if batch == 1 else m*k*src_itemsize
      b_len = batch*k*n*src_itemsize
      out_len = batch*m*n*out_itemsize
      if len(a_buf) != a_len or len(b_buf) != b_len or len(out_buf) != out_len:
        raise NotImplementedError(f"rkcna_gemv_v1 buffer sizes do not match meta {self.cna_meta}: {[len(x) for x in bufs]}")
      solved = (m, n, k, src_itemsize, out_itemsize, batch)
    else:
      for out_itemsize in (4, 2):
        if len(out_buf) % out_itemsize != 0: continue
        out_elems = len(out_buf) // out_itemsize
        for src_itemsize in (2, 4):
          if len(a_buf) % src_itemsize != 0 or len(b_buf) % src_itemsize != 0: continue
          a_elems, b_elems = len(a_buf) // src_itemsize, len(b_buf) // src_itemsize
          if a_elems * b_elems % out_elems != 0: continue
          k = math.isqrt(a_elems * b_elems // out_elems)
          if k > 0 and k*k*out_elems == a_elems*b_elems and a_elems % k == 0 and b_elems % k == 0:
            m, n = a_elems // k, b_elems // k
            if m*n == out_elems: solved = (m, n, k, src_itemsize, out_itemsize, 1); break
        if solved is not None: break
    if solved is None: raise NotImplementedError(f"rkcna_v1 cannot infer GEMM shape from sizes {[len(x) for x in bufs]}")
    m, n, k, src_itemsize, out_itemsize, batch = solved
    out_fp16 = out_itemsize == 2
    a_src = np.frombuffer(a_buf, dtype=np.float32 if src_itemsize == 4 else np.float16).astype(np.float16).reshape(m, k)
    b_src = np.frombuffer(b_buf, dtype=np.float32 if src_itemsize == 4 else np.float16).astype(np.float16).reshape(batch, k, n)
    align_in, align_out, _ = _rk_gemm_layout(m, n, k)
    input_packed = np.zeros((m, align_in), dtype=np.float16)
    input_packed[:, :k] = a_src
    weight_packed = np.zeros((batch, align_out * align_in), dtype=np.float16)
    for bi in range(batch):
      weight = np.zeros((align_out, align_in), dtype=np.float16)
      weight[:n, :k] = b_src[bi].T
      weight_packed[bi] = weight.reshape(align_out // 16, 16, align_in // 32, 32).transpose(0, 2, 1, 3).ravel()
    weight_packed_flat = weight_packed.reshape(-1)
    row_stride_bytes = align_out * 4
    out_nbytes_one = max(256, m * row_stride_bytes)
    out_nbytes = batch * out_nbytes_one
    task_buf = self.device._gpu_alloc(64*1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="rkcna_task")
    cmd_buf = self.device._gpu_alloc(512*1024, 0, name="rkcna_cmd")
    input_buf = self.device._gpu_alloc(input_packed.nbytes, 0, name="rkcna_input")
    weight_buf = self.device._gpu_alloc(weight_packed_flat.nbytes, 0, name="rkcna_weight")
    output_buf = self.device._gpu_alloc(out_nbytes, 0, name="rkcna_output")
    try:
      ctypes.memmove(input_buf.va_addr, mv_address(memoryview(input_packed.reshape(-1))), input_packed.nbytes)
      ctypes.memmove(weight_buf.va_addr, mv_address(memoryview(weight_packed_flat)), weight_packed_flat.nbytes)
      self.device._gpu_sync(input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.device._gpu_sync(weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      regcmd = ctypes.cast(cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * (cmd_buf.size // 8))).contents
      task_regs = [_rk_make_gemm_regs(m, n, k, input_buf.meta.dma_addr,
        weight_buf.meta.dma_addr + bi * weight_packed.strides[0], output_buf.meta.dma_addr + bi * out_nbytes_one, out_fp16=out_fp16) for bi in range(batch)]
      offsets, offset = [], 0
      for regs in task_regs:
        offsets.append(offset)
        offset += _rk_align_up(len(regs) + RK_PC_CHAIN_TAIL_QWORDS, 2)
      if offset > cmd_buf.size // 8: raise RuntimeError("rkcna command buffer too small")
      for bi, regs in enumerate(task_regs):
        base = offsets[bi]
        for i,qword in enumerate(regs): regcmd[base+i] = qword
        if bi + 1 < batch:
          next_addr = cmd_buf.meta.dma_addr + offsets[bi+1] * ctypes.sizeof(ctypes.c_uint64)
          tail = [_rk_E(0x0101, rk.REG_PC_BASE_ADDRESS, next_addr & 0xFFFFFFF0), _rk_E(0x0101, rk.REG_PC_REGISTER_AMOUNTS, _rk_ceil_div(len(task_regs[bi+1]), 2) + 1), _rk_E(0x0041, 0, 0), _rk_E(0x0081, rk.REG_PC_OPERATION_ENABLE, (6 << 1) | 1)]
        else:
          tail = [_rk_E(0x0001, 0, 0), _rk_E(0x0101, rk.REG_PC_REGISTER_AMOUNTS, 0), _rk_E(0x0041, 0, 0), _rk_E(0x0081, rk.REG_PC_OPERATION_ENABLE, (6 << 1) | 1)]
        for i,qword in enumerate(tail): regcmd[base+len(regs)+i] = qword
      tasks = ctypes.cast(task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
      if batch > len(tasks): raise RuntimeError("rkcna task buffer too small")
      for bi, regs in enumerate(task_regs):
        tasks[bi].flags = 0; tasks[bi].op_idx = 0; tasks[bi].enable_mask = 0xd; tasks[bi].int_mask = 0x300; tasks[bi].int_clear = 0x1ffff; tasks[bi].int_status = 0
        tasks[bi].regcfg_amount = len(regs); tasks[bi].regcfg_offset = 0; tasks[bi].regcmd_addr = cmd_buf.meta.dma_addr + offsets[bi] * ctypes.sizeof(ctypes.c_uint64)
      self.device.reset_npu()
      submit_res = rk.struct_rknpu_submit(flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG, timeout=6000,
        task_start=0, task_number=batch, task_counter=0, priority=0, task_obj_addr=task_buf.meta.obj_addr, regcfg_obj_addr=0,
        task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
        subcore_task=(rk.struct_rknpu_subcore_task * 5)(rk.struct_rknpu_subcore_task(task_start=0, task_number=batch), rk.struct_rknpu_subcore_task(task_start=batch, task_number=0), rk.struct_rknpu_subcore_task(task_start=batch, task_number=0)))
      rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl, __payload=submit_res)
      self.device._gpu_sync(output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      raw = memoryview(bytearray(output_buf.size))
      ctypes.memmove(mv_address(raw), output_buf.va_addr, output_buf.size)
      result_batches = []
      for bi in range(batch):
        chunk = raw[bi*out_nbytes_one:(bi+1)*out_nbytes_one]
        if out_fp16:
          surface = np.frombuffer(chunk, dtype=np.float16, count=len(chunk)//2)
          result_batches.append(surface[_rk_gemm_output_indices(m, n, align_out, True)].copy().reshape(-1))
        else:
          surface = np.frombuffer(chunk, dtype=np.float32, count=len(chunk)//4)
          result_batches.append(surface[(np.arange(m) * align_out)[:, None] + np.arange(n)].copy().reshape(-1))
      result = np.concatenate(result_batches)
      out_buf.cast('B')[:] = memoryview(result).cast('B')
      if DEBUG >= 3:
        print(f"{'RKCNA_GEMV_RUN' if self.cna_kind == 'rkcna_gemv_v1' else 'RKCNA_RUN'} m={m} n={n} k={k} batch={batch} out={'fp16' if out_fp16 else 'fp32'}")
      return time.perf_counter() - st
    finally:
      self.device._gpu_free_multiple([task_buf, cmd_buf, input_buf, weight_buf, output_buf])
  
def _rk_shift_to_muldiv(x, op):
  vmax = max(abs(x.src[0].vmin), abs(x.src[0].vmax))
  smin, smax = x.src[1].vmin, x.src[1].vmax
  if not all(math.isfinite(v) for v in (vmax, smin, smax)): return None
  if smin < 0: return None
  smax_i = int(smax)
  if smax_i != smax or smax_i > 15: return None
  if vmax > 65504: return None
  if x.op is Ops.SHL and vmax * (1 << smax_i) > 65504: return None
  pow2 = x.src[1].cast(dtypes.float16).exp2()
  return x.src[0].cast(dtypes.float16).alu(op, pow2).cast(x.dtype)

RK_FP16_BYTES, RK_FP32_BYTES, RK_CBUF_ENTRY_BYTES, RK_CBUF_ENTRIES_PER_BANK = 2, 4, 128, 256
RK_CBUF_BANKS, RK_MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP = 12, 32, 13
RK_CBUF_BANK_SIZE = RK_CBUF_ENTRIES_PER_BANK * RK_CBUF_ENTRY_BYTES
RK_MIN_WIDE_FEATURE_GRAINS, RK_PC_CHAIN_TAIL_QWORDS = 80, 4
RK_GEMM_INPUT_BANKS, RK_GEMM_MAX_ALIGN_IN = RK_CBUF_BANKS - 2, RK_CBUF_BANKS * RK_MIN_CHANNEL_TILE

def _rk_ceil_div(x, y): return (x + y - 1) // y
def _rk_align_up(x, align): return _rk_ceil_div(x, align) * align
def _rk_E(target, reg_addr, value): return (target << 48) | ((value & 0xFFFFFFFF) << 16) | reg_addr

def _rk_gemm_layout(m, n, k):
  aligned_k = max(RK_MIN_CHANNEL_TILE, _rk_align_up(k, RK_MIN_CHANNEL_TILE))
  align_out = max(RK_MIN_CHANNEL_TILE, _rk_align_up(n, RK_MIN_CHANNEL_TILE))
  align_in = max(aligned_k, align_out)
  eff_k = align_in if align_in != aligned_k else k
  return align_in, align_out, eff_k

def _rk_gemm_output_indices(m, n, align_out, out_fp16):
  row_stride = align_out * 2 if out_fp16 else align_out
  row_start = np.arange(m, dtype=np.int64) * row_stride
  col_idx = (np.arange(n, dtype=np.int64) // 16) * 32 + (np.arange(n, dtype=np.int64) % 16) if out_fp16 else np.arange(n, dtype=np.int64)
  return row_start[:, None] + col_idx[None, :]

def _rk_make_gemm_regs(m, n, k, in_dma, wt_dma, out_dma, out_fp16=False):
  cna, core, dpu = rk.CNA + 1, rk.CORE + 1, rk.DPU + 1
  align_in, align_out, eff_k = _rk_gemm_layout(m, n, k)
  input_row_bytes = align_in * RK_FP16_BYTES
  out_precision, size_e = (2, 1) if out_fp16 else (5, 3)
  even_rows_per_two_banks = (_rk_ceil_div(2 * RK_CBUF_BANK_SIZE, input_row_bytes) + 1) & ~1
  feature_grains = max(RK_MIN_WIDE_FEATURE_GRAINS, even_rows_per_two_banks)
  data_banks = int(np.clip(_rk_ceil_div(m * input_row_bytes, RK_CBUF_BANK_SIZE), 1, RK_CBUF_BANKS-1))
  line_stride = 4 * min(_rk_ceil_div(eff_k, RK_MIN_CHANNEL_TILE), RK_LINE_STRIDE_GROUP_CAP)
  notch_val = 8 * min(align_out // RK_MIN_CHANNEL_TILE, RK_LINE_STRIDE_GROUP_CAP) - 1
  return [
    _rk_E(dpu, rk.REG_DPU_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
    _rk_E(cna, rk.REG_CNA_CONV_CON1, (2 << 4) | (2 << 7) | (1 << 29)),
    _rk_E(cna, rk.REG_CNA_CONV_CON2, feature_grains << 4),
    _rk_E(cna, rk.REG_CNA_CONV_CON3, (1 << 3) | 1),
    _rk_E(cna, rk.REG_CNA_DATA_SIZE0, (1 << 16) | m),
    _rk_E(cna, rk.REG_CNA_DATA_SIZE1, ((align_in - 1) << 16) | align_in),
    _rk_E(cna, rk.REG_CNA_DATA_SIZE2, 1), _rk_E(cna, rk.REG_CNA_DATA_SIZE3, m),
    _rk_E(cna, rk.REG_CNA_WEIGHT_SIZE0, input_row_bytes * align_out), _rk_E(cna, rk.REG_CNA_WEIGHT_SIZE1, input_row_bytes),
    _rk_E(cna, rk.REG_CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | align_out),
    _rk_E(cna, rk.REG_CNA_CBUF_CON0, ((RK_CBUF_BANKS - data_banks) << 4) | data_banks), _rk_E(cna, rk.REG_CNA_CBUF_CON1, _rk_ceil_div(align_in, RK_MIN_CHANNEL_TILE)),
    _rk_E(cna, rk.REG_CNA_CVT_CON0, (1 << 3) | (1 << 1) | 1), _rk_E(cna, rk.REG_CNA_CVT_CON1, 1 << 16),
    _rk_E(cna, rk.REG_CNA_CVT_CON2, 1 << 16), _rk_E(cna, rk.REG_CNA_CVT_CON3, 1 << 16), _rk_E(cna, rk.REG_CNA_CVT_CON4, 1 << 16),
    _rk_E(cna, rk.REG_CNA_FEATURE_DATA_ADDR, in_dma), _rk_E(cna, rk.REG_CNA_DMA_CON0, (15 << 16) | 15), _rk_E(cna, rk.REG_CNA_DMA_CON1, line_stride), _rk_E(cna, rk.REG_CNA_DMA_CON2, 0),
    _rk_E(cna, rk.REG_CNA_FC_DATA_SIZE0, (1 << 16) | m), _rk_E(cna, rk.REG_CNA_FC_DATA_SIZE1, align_in), _rk_E(cna, rk.REG_CNA_DCOMP_ADDR0, wt_dma),
    _rk_E(core, rk.REG_CORE_MISC_CFG, (2 << 8) | 1), _rk_E(core, rk.REG_CORE_DATAOUT_SIZE_0, ((m - 1) << 16) | 0), _rk_E(core, rk.REG_CORE_DATAOUT_SIZE_1, align_out - 1), _rk_E(core, 0x3030, 0),
    _rk_E(dpu, rk.REG_DPU_FEATURE_MODE_CFG, (15 << 5) | (2 << 1)), _rk_E(dpu, rk.REG_DPU_DATA_FORMAT, (out_precision << 29) | (2 << 26) | 2), _rk_E(dpu, rk.REG_DPU_DST_BASE_ADDR, out_dma),
    _rk_E(dpu, rk.REG_DPU_DST_SURF_STRIDE, 1 << 4), _rk_E(dpu, rk.REG_DPU_DATA_CUBE_WIDTH, 0), _rk_E(dpu, rk.REG_DPU_DATA_CUBE_HEIGHT, m - 1),
    _rk_E(dpu, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, (notch_val << 16) | notch_val), _rk_E(dpu, rk.REG_DPU_DATA_CUBE_CHANNEL, ((align_out - 1) << 16) | (align_out - 1)),
    _rk_E(dpu, rk.REG_DPU_BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1), _rk_E(dpu, rk.REG_DPU_BS_OW_CFG, (size_e << 8) | (size_e << 5) | (size_e << 2) | (1 << 1)),
    _rk_E(dpu, rk.REG_DPU_WDMA_SIZE_0, align_out - 1), _rk_E(dpu, rk.REG_DPU_WDMA_SIZE_1, ((m - 1) << 16) | 0),
    _rk_E(dpu, rk.REG_DPU_BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1), _rk_E(dpu, rk.REG_DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1),
    _rk_E(dpu, rk.REG_DPU_OUT_CVT_SCALE, ((1 << 16) | 1) if out_fp16 else 0), _rk_E(dpu, rk.REG_DPU_SURFACE_ADD, (1 * 4) << 4),
  ]

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  has_threads = False
  code_for_op = {k:v for k,v in python_alu.items() if k not in [Ops.MULACC, Ops.RECIPROCAL, Ops.CMPNE]} | {Ops.FDIV: 0}
  # hacks, turned unsupported dtype to half and lut function to Ops.CUSTOM
  compiler = RockchipCompiler()

  def __init__(self, target):
    super().__init__(target)
    self.tensor_cores = tc.rockchip_cmac
  def _rk_trunc_fix(x):
    if x.tag == "rk_trunc": return None
    xh = x.src[0].cast(dtypes.half)
    zero = UOp.const(dtypes.half, 0)
    neg = xh.alu(Ops.CMPLT, zero)
    shifted = xh.alu(Ops.SUB, UOp.const(dtypes.half, 0.49951171875))
    absx = UOp(Ops.WHERE, dtypes.half, src=(shifted.alu(Ops.CMPLT, zero), shifted.alu(Ops.NEG), shifted))
    mag = absx.alu(Ops.TRUNC).rtag("rk_trunc")
    signed = UOp(Ops.WHERE, dtypes.half, src=(neg, mag.alu(Ops.NEG).alu(Ops.ADD, UOp.const(dtypes.half, 1)), mag))
    return signed.cast(x.dtype)
  pre_matcher = PatternMatcher([
    (UPat.const(dtypes.floats, 0).alu(Ops.CMPLT, UPat.var("x", dtypes.floats)).where(UPat.var("x", dtypes.floats), UPat.const(dtypes.floats, 0)),
     lambda x: UOp(Ops.CUSTOM, dtypes.half, src=(x.cast(dtypes.half),), arg="relu")),
  ])
  extra_matcher = PatternMatcher([
    (UPat(Ops.MUL, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.MUL, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.ADD, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.ADD, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.MAX, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.MAX, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.SHL, dtypes.uint, name="x"),
     lambda x: _rk_shift_to_muldiv(x, Ops.MUL)),
    (UPat(Ops.SHR, dtypes.uint, name="x"),
      lambda x: _rk_shift_to_muldiv(x, Ops.FDIV)),
    (UPat(Ops.ADD, dtypes.float, name="x"),
      lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half))),
    (UPat(Ops.MUL, dtypes.float, name="x"),
      lambda x: x.src[0].cast(dtypes.half).alu(Ops.MUL, x.src[1].cast(dtypes.half))),
    (UPat(Ops.MAX, dtypes.float, name="x"),
      lambda x: x.src[0].cast(dtypes.half).alu(Ops.MAX, x.src[1].cast(dtypes.half))),
    (UPat(Ops.NEG, dtypes.float, name="x"),
      lambda x: x.src[0].cast(dtypes.half).alu(Ops.NEG)),
    (UPat(Ops.EXP2, dtypes.float, name="x"),
      lambda x: x.src[0].cast(dtypes.half).alu(Ops.EXP2)),
    (UPat(Ops.TRUNC, dtypes.floats, name="x"),
      _rk_trunc_fix),
    # (UPat.var("x", dtypes.floats).alu(Ops.FDIV,
    #   UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
    #  lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
    # (UPat.var("x", dtypes.floats) * UPat.const(dtypes.floats, 1).alu(Ops.FDIV,
    #   UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
    #  lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
    (UPat(Ops.CMPLT, name="x"),
     lambda x: UOp(Ops.CUSTOM, dtypes.float16, src=(x.src[1].cast(dtypes.float16).alu(Ops.SUB, x.src[0].cast(dtypes.float16)),),
                   arg="cmplt_diff2bool").cast(dtypes.bool)),
    (UPat(Ops.CMPEQ, name="x"),
     lambda x: UOp(Ops.CUSTOM, dtypes.float16, arg="cmpeq_32800_to_bool", src=(
       UOp(Ops.CUSTOM, dtypes.float16, arg="cmpeq_diff_zero_to_nan_to_32800", src=(
         x.src[1].cast(dtypes.float16).alu(Ops.SUB, x.src[0].cast(dtypes.float16)),),
       ),
     )).cast(dtypes.bool)),
    # CMPNE(x) = 1 - CMPEQ(x)
    (UPat(Ops.CMPNE, name="x"),
      lambda x: UOp.const(dtypes.float16, 1).alu(
        Ops.SUB,
        x.src[0].cast(dtypes.float16).alu(Ops.CMPEQ, x.src[1].cast(dtypes.float16)).cast(dtypes.float16)
      ).cast(dtypes.bool)),
    # ax + b(1-x) 
    (UPat(Ops.WHERE, name="w", src=(UPat.var("c", dtypes.bool), UPat.var("a", dtypes.floats), UPat.var("b", dtypes.floats))),
     lambda w,c,a,b: a.cast(dtypes.float16).alu(Ops.MUL, c.cast(dtypes.float16)).alu(Ops.ADD,
       b.cast(dtypes.float16).alu(Ops.MUL, UOp.const(dtypes.float16, 1).alu(Ops.SUB, c.cast(dtypes.float16)))).cast(w.dtype)),
    (UPat(Ops.WHERE, name="w", src=(UPat.var("c", dtypes.bool), UPat.var("a", dtypes.ints), UPat.var("b", dtypes.ints))),
     lambda w,c,a,b: a.cast(dtypes.float16).alu(Ops.MUL, c.cast(dtypes.float16)).alu(Ops.ADD,
       b.cast(dtypes.float16).alu(Ops.MUL, UOp.const(dtypes.float16, 1).alu(Ops.SUB, c.cast(dtypes.float16)))).cast(w.dtype)),
  ])
  def _coalesce_wmma_to_cna(self, uops:list[UOp]):
    wmmas = [u for u in uops if u.op is Ops.WMMA and isinstance(u.arg, tuple) and len(u.arg) >= 6 and u.arg[4] == "ROCKCHIP"]
    if not wmmas: return None
    first = wmmas[0]
    dims, dtype_in, dtype_out = first.arg[1], first.arg[2], first.arg[3]
    if dims != (16, 1, 32):
      if DEBUG >= 3: print("RKCNA_FALLBACK:unsupported_cmac_dims")
      return None
    if not all(w.arg[1] == dims and w.arg[2] == dtype_in and w.arg[3] == dtype_out and w.arg[4] == "ROCKCHIP" for w in wmmas):
      if DEBUG >= 3: print("RKCNA_FALLBACK:mixed_wmma_atoms")
      return None
    meta = {
      "version": 1, "m": 1, "n": dims[0], "k": dims[2], "batch": 1,
      "a_dtype": dtype_in, "b_dtype": dtype_in, "c_dtype": dtype_out,
      "acc_dtype": dtype_out, "out_dtype": dtype_out, "post_op": None,
      "atoms": len(wmmas), "dims": dims,
    }
    if DEBUG >= 3:
      print(f"RKCNA_MATCH m={meta['m']} n={meta['n']} k={meta['k']} batch={meta['batch']} atoms={meta['atoms']} k_slices=1 c_slices=1 slots=(-1,-1,-1)")
    return meta

  @staticmethod
  def _root_params(u:UOp) -> set[UOp]:
    if u.op is Ops.PARAM: return {u}
    ret:set[UOp] = set()
    for s in u.src: ret.update(RockchipRenderer._root_params(s))
    return ret

  def _match_gemv_to_cna(self, uops:list[UOp]):
    if any(u.op is Ops.WMMA for u in uops): return None
    if any(u.op in {Ops.CDIV, Ops.CMOD, Ops.FLOORDIV, Ops.FLOORMOD} for u in uops): return None
    params = [u for u in uops if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType)]
    if len(params) != 3: return None
    out, a, b = params
    if a.dtype.base.scalar() is not dtypes.half or b.dtype.base.scalar() is not dtypes.half: return None
    if out.dtype.base.scalar() not in (dtypes.half, dtypes.float): return None
    out_stores = [u for u in uops if u.op is Ops.STORE and out in self._root_params(u.src[0])]
    if len(out_stores) != 1: return None
    loads = [u for u in uops if u.op is Ops.LOAD]
    load_roots = [self._root_params(u) for u in loads]
    if any(out in roots for roots in load_roots): return None
    if {a, b} - set().union(*load_roots) if load_roots else {a, b}: return None
    if not any(u.op is Ops.MUL and a in self._root_params(u.src[0]) | self._root_params(u.src[1]) and
               b in self._root_params(u.src[0]) | self._root_params(u.src[1]) for u in uops): return None
    if not any(u.op is Ops.ADD for u in uops): return None
    out_elems, a_elems, b_elems = out.dtype.size, a.dtype.size, b.dtype.size
    if min(out_elems, a_elems, b_elems) <= 0 or a_elems * b_elems % out_elems != 0: return None
    k = math.isqrt(a_elems * b_elems // out_elems)
    if k <= 0 or k*k*out_elems != a_elems*b_elems or a_elems % k != 0 or b_elems % k != 0: return None
    m, n = a_elems // k, b_elems // k
    if m*n != out_elems or (m != 1 and n != 1): return None
    out_ranges = [u for u in uops if u.op is Ops.RANGE and isinstance(u.arg, tuple) and len(u.arg) > 1 and u.arg[1] is not AxisType.REDUCE]
    if len(out_ranges) > 1: return None
    batch = 1
    sink = next((u for u in uops if u.op is Ops.SINK), None)
    local_opts = [x for x in (getattr(sink.arg, "applied_opts", ()) if sink is not None else ()) if getattr(getattr(x, "op", None), "name", None) == "LOCAL"]
    if n != 1 and len(local_opts) > 1:
      candidates = sorted([int(x.arg) for x in local_opts if isinstance(x.arg, int) and x.arg > 1], reverse=True)
      batch = next((x for x in candidates if n % x == 0), 1)
      if batch == 1: return None
      n //= batch
    meta = {
      "version": 1, "kind": "gemv", "m": m, "n": n, "k": k, "batch": batch,
      "a_dtype": a.dtype.base.scalar(), "b_dtype": b.dtype.base.scalar(), "out_dtype": out.dtype.base.scalar(),
      "src_itemsize": a.dtype.base.scalar().itemsize, "out_itemsize": out.dtype.base.scalar().itemsize,
    }
    if DEBUG >= 3: print(f"RKCNA_GEMV_MATCH m={m} n={n} k={k} batch={batch}")
    return meta

  def render(self, uops:list[UOp]) -> str:
    if (meta := self._coalesce_wmma_to_cna(uops)) is not None:
      return base64.b64encode(pickle.dumps(("rkcna_v1", meta, uops))).decode()
    if (meta := self._match_gemv_to_cna(uops)) is not None:
      return base64.b64encode(pickle.dumps(("rkcna_gemv_v1", meta, uops))).decode()
    return base64.b64encode(pickle.dumps(uops)).decode()
  
class RockchipAllocator(Allocator['RockchipDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer], functools.partial(RockchipProgram, self))
  def create_flink_name(self, handle: int, name:str, virt_address:int|None=None, obj_addr:int|None=None, dma_address:int|None=None) -> int:
    flink_req = rk.struct_drm_gem_flink(handle=handle, name=0)
    result = rk.DRM_IOCTL_GEM_FLINK(self.fd_ctl, __payload=flink_req)
    # print(f"SUCCESS: Created flink name {flink_req.name} for handle {handle} {name} {hex(dma_address)}")
    return flink_req.name
  def _gpu_alloc(self, size:int, flags, name:str) -> HCQBuffer:
    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mem_map.offset)
    mem_create.flink_name = self.create_flink_name(mem_create.handle, name, virt_address=va_addr, obj_addr=mem_create.obj_addr, dma_address=mem_create.dma_addr)

    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)
  def _gpu_sync(self, buf:HCQBuffer, flags:int) -> None:
    if not getenv("ROCKCHIP_MEM_SYNC", 0): return
    rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.fd_ctl, __payload=rk.struct_rknpu_mem_sync(
      flags=flags, reserved=0, obj_addr=buf.meta.obj_addr, offset=0, size=buf.size))
  def _gpu_free(self, buf:HCQBuffer) -> None:
    FileIOInterface.munmap(buf.va_addr, buf.size)
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, __payload=rk.struct_rknpu_mem_destroy(
      handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr))
  def _gpu_free_multiple(self, buf_list) -> None:
    for buf in buf_list: self._gpu_free(buf)
  def reset_npu(self):
    rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, __payload=rk.struct_rknpu_action(flags=rk.RKNPU_ACT_RESET, value=0))
