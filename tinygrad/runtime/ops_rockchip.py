from typing import Any
import base64, time, struct, functools, ctypes, mmap, os, numpy as np
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, mv_address, to_mv
from tinygrad.device import Compiled, Compiler, Allocator, BufferSpec
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import python_alu, Ops, UOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_cpu import HCQBuffer
from tinygrad.runtime.support.hcq import FileIOInterface, HCQAllocatorBase
from tinygrad.runtime.support.rockchip import (
  REGCMD_RESERVED, RK_TEMPLATE_MAGIC, build_conv1x1_template,
  build_elementwise_template, build_lut, build_wmma_template, conv_params, decode_template, encode_template, lut_enabled, pack_conv_input,
  pack_conv_weights, apply_patches, conv1x1_meta, elementwise_meta, parse_fused_matmul_name, submit_template,
  unpack_conv_output, validate_template, wmma_params,
)
from tinygrad.runtime.autogen import rockchip as rk

def _rk_env(name:str, default:int) -> int:
  try: return int(os.getenv(name, str(default)))
  except ValueError: return default

class RockchipProgram:
  def __init__(self, dev:'RockchipDevice', name:str, lib:bytes, **kwargs):
    self.template = decode_template(lib) if lib.startswith(RK_TEMPLATE_MAGIC) else None
    if self.template is not None: validate_template(self.template, getattr(dev, "target", "rk3588-rknpu2"))
    if self.template is None: raise RuntimeError("unsupported Rockchip program: missing RKTemplatePackage magic")
    self.ew_meta = None
    self.conv_meta = None
    self.ew_arg = None
    if self.template is not None and self.template.family == "elementwise":
      slots = {p.role:p.arg_index for p in self.template.patches}
      self.ew_meta = (self.template.op, self.template.size, slots["output"], slots["input"], slots["weight"])
      self.ew_arg = self.template.meta.get("arg") if self.template.meta is not None else None
    elif self.template is not None and self.template.family == "conv1x1":
      slots = {p.role:p.arg_index for p in self.template.patches}
      assert self.template.meta is not None
      self.conv_meta = (slots["output"], slots["input"], slots["weight"], self.template.meta["in_channels"],
                        self.template.meta["out_channels"], self.template.meta["spatial"])
    self.ew_template = self.template if self.template.family == "elementwise" else None
    self.conv_template = self.template if self.template.family == "conv1x1" else None
    self.device = dev
    self.q = []
    self.hardware_ops = {
      Ops.WMMA:0, Ops.TRUNC:0, Ops.CUSTOM:0, Ops.MUL:0, Ops.NEG:0, Ops.MAX:0,
      Ops.EXP2:0, Ops.CMPLT:0, Ops.CMPEQ:0, Ops.ADD:2, Ops.FDIV:3, Ops.SUB:4,
    }
    self.cmd_buf_size = 16384
    self.exp2_inv_scale = 1.0
    self.lut_size = 513
    self.fused_matmul_meta = parse_fused_matmul_name(name)
    self.fused_matmul_hits = 0
    self.fused_matmul_fallbacks = 0

  def _dtype_from_code(self, code:int):
    if code == 0: return np.float16
    if code == 1: return np.float32
    raise RuntimeError(f"dtype_code_{code}")

  def _run_wmma_matmul(self, a_matrix:np.ndarray, b_matrix:np.ndarray) -> np.ndarray:
    m, k = a_matrix.shape
    if b_matrix.shape[0] != k: raise RuntimeError("k_mismatch")
    n = int(b_matrix.shape[1])
    wmma_meta = wmma_params(int(m), int(n), int(k))
    in_pack = np.zeros(wmma_meta["align_in"] * wmma_meta["m"], dtype=np.float16)
    wt_pack = np.zeros(wmma_meta["align_out"] * wmma_meta["align_in"], dtype=np.float16)
    if (wmma_meta["m"], wmma_meta["n"], wmma_meta["k"]) == (64, 64, 64):
      for mm in range(1, 65):
        for kk in range(1, 65):
          plane = (kk - 1) // 8
          offset = (kk - 1) % 8
          in_pack[plane * 64 * 8 + (mm - 1) * 8 + offset] = a_matrix[mm - 1, kk - 1]
      for nn in range(1, 65):
        for kk in range(1, 65):
          kpg, cpg = (nn - 1) // 16, (kk - 1) // 32
          wt_idx = ((cpg * 32) * 16) + (kpg * 16 * wmma_meta["align_in"]) + ((kk - 1) % 32) + (((nn - 1) % 16) * 32)
          wt_pack[wt_idx] = b_matrix[kk - 1, nn - 1]
    else:
      in_pack = in_pack.reshape(wmma_meta["m"], wmma_meta["align_in"])
      wt_pack = wt_pack.reshape(wmma_meta["align_out"], wmma_meta["align_in"])
      in_pack[:, :wmma_meta["k"]] = a_matrix
      wt_pack[:wmma_meta["n"], :wmma_meta["k"]] = b_matrix.T
      in_pack = in_pack.reshape(-1)
      wt_pack = wt_pack.reshape(-1)
    src = memoryview(bytearray(in_pack.tobytes()))
    src2 = memoryview(bytearray(wt_pack.tobytes()))
    self.task_buf = self.device._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task_buf")
    self.cmd_buf = self.device._gpu_alloc(self.cmd_buf_size, 0, name="cmd_buf")
    self.input_buf = self.device._gpu_alloc(src.nbytes, 0, name="input")
    self.weight_buf = self.device._gpu_alloc(src2.nbytes, 0, name="weight")
    out_stride = wmma_meta["align_out"] * dtypes.float32.itemsize
    out_nbytes = max(0x100, (wmma_meta["m"]-1)*out_stride + wmma_meta["n"]*dtypes.float32.itemsize)
    self.output_buf = self.device._gpu_alloc(out_nbytes, 0, name="output")
    try:
      ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
      ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
      self.device._gpu_sync(self.input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.device._gpu_sync(self.weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      template = build_wmma_template(wmma_meta)
      self.q = list(template.regcmd)
      addrs = {"output":self.output_buf.meta.dma_addr, "input":self.input_buf.meta.dma_addr, "weight":self.weight_buf.meta.dma_addr}
      apply_patches(self.q, template.patches, addrs)
      submit_template(self.device.fd_ctl, template, self.q, self.task_buf, self.cmd_buf, self.cmd_buf_size)
      self.device._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      dst = memoryview(bytearray(self.output_buf.size))
      ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
      raw = np.frombuffer(dst.tobytes(), dtype=np.float32)
      out = np.empty((m, n), dtype=np.float32)
      if (wmma_meta["m"], wmma_meta["n"], wmma_meta["k"]) in {(64, 64, 64), (256, 256, 256)}:
        c2 = 4
        for col in range(n):
          plane, offset = col // c2, col % c2
          plane_base = plane * m * c2
          for row in range(m): out[row, col] = raw[plane_base + row * c2 + offset]
      else:
        stride = wmma_meta["align_out"]
        for row in range(m): out[row, :] = raw[row * stride:row * stride + n]
      return out
    finally:
      self.device._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf, self.weight_buf, self.output_buf])

  def _run_fused_matmul(self, bufs:tuple[Any, ...]) -> None:
    if self.fused_matmul_meta is None: raise RuntimeError("missing_meta")
    m = self.fused_matmul_meta["m"]
    n = self.fused_matmul_meta["n"]
    k = self.fused_matmul_meta["k"]
    batch = self.fused_matmul_meta["batch"]
    a_slot, b_slot, c_slot = self.fused_matmul_meta["a_slot"], self.fused_matmul_meta["b_slot"], self.fused_matmul_meta["c_slot"]
    if max(a_slot, b_slot, c_slot) >= len(bufs): raise RuntimeError("slot_oob")
    a_arr = np.frombuffer(bufs[a_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["a_dt"]))
    b_arr = np.frombuffer(bufs[b_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["b_dt"]))
    c_arr = np.frombuffer(bufs[c_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["c_dt"]))
    if self.fused_matmul_meta["ta"] == 0 and (self.fused_matmul_meta["a_ms"], self.fused_matmul_meta["a_ks"]) != (k, 1):
      raise RuntimeError("lhs_layout")
    if self.fused_matmul_meta["ta"] == 1 and (self.fused_matmul_meta["a_ms"], self.fused_matmul_meta["a_ks"]) != (1, m):
      raise RuntimeError("lhs_layout")
    if self.fused_matmul_meta["tb"] == 0 and (self.fused_matmul_meta["b_ks"], self.fused_matmul_meta["b_ns"]) != (n, 1):
      raise RuntimeError("rhs_layout")
    if self.fused_matmul_meta["tb"] == 1 and (self.fused_matmul_meta["b_ks"], self.fused_matmul_meta["b_ns"]) != (1, k):
      raise RuntimeError("rhs_layout")
    if (self.fused_matmul_meta["c_ms"], self.fused_matmul_meta["c_ns"]) != (n, 1): raise RuntimeError("out_layout")
    if batch > 1 and self.fused_matmul_meta["a_bs"] != m*k: raise RuntimeError("lhs_batch_stride")
    if batch > 1 and self.fused_matmul_meta["b_bs"] != k*n: raise RuntimeError("rhs_batch_stride")
    if batch > 1 and self.fused_matmul_meta["c_bs"] != m*n: raise RuntimeError("out_batch_stride")
    if len(a_arr) < (batch-1)*self.fused_matmul_meta["a_bs"] + m*k: raise RuntimeError("lhs_buffer_too_small")
    if len(b_arr) < (batch-1)*self.fused_matmul_meta["b_bs"] + k*n: raise RuntimeError("rhs_buffer_too_small")
    if len(c_arr) < (batch-1)*self.fused_matmul_meta["c_bs"] + m*n: raise RuntimeError("out_buffer_too_small")

    for bidx in range(batch):
      a_base = bidx * self.fused_matmul_meta["a_bs"]
      b_base = bidx * self.fused_matmul_meta["b_bs"]
      c_base = bidx * self.fused_matmul_meta["c_bs"]
      a_block = a_arr[a_base:a_base + m*k]
      b_block = b_arr[b_base:b_base + k*n]
      if len(a_block) != m*k or len(b_block) != k*n: raise RuntimeError("batch_slice_oob")
      if self.fused_matmul_meta["ta"] == 0:
        a_matrix = a_block.reshape(m, k).astype(np.float16, copy=False)
      else:
        a_matrix = a_block.reshape(k, m).T.astype(np.float16, copy=False)
      if self.fused_matmul_meta["tb"] == 0:
        b_matrix = b_block.reshape(k, n).astype(np.float16, copy=False)
      else:
        b_matrix = b_block.reshape(n, k).T.astype(np.float16, copy=False)
      out_matrix = self._run_wmma_matmul(a_matrix, b_matrix)
      # Validate one dot-product entry to avoid silently returning bad fused outputs.
      ref00 = float(np.dot(a_matrix[0, :].astype(np.float32), b_matrix[:, 0].astype(np.float32)))
      got00 = float(out_matrix[0, 0])
      tol = max(1e-2, abs(ref00) * 5e-3)
      if not np.isfinite(got00) or abs(got00 - ref00) > tol: raise RuntimeError("npu_verify_mismatch")
      c_block = c_arr[c_base:c_base + m*n]
      if len(c_block) != m*n: raise RuntimeError("out_batch_slice_oob")
      if self.fused_matmul_meta["c_dt"] == 0: c_block[:] = out_matrix.astype(np.float16).reshape(-1)
      else: c_block[:] = out_matrix.reshape(-1)

  def _run_elementwise(self, op, size:int, out_slot:int, lhs_slot:int, rhs_slot:int, bufs:tuple[Any, ...], arg=None) -> None:
    src, src2 = memoryview(bufs[lhs_slot])[:size*2], memoryview(bufs[rhs_slot])[:size*2]
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
      self.q, self.lut_enable = list(self.ew_template.regcmd), False
      addrs = {"output":self.output_buf.meta.dma_addr, "input":self.input_buf.meta.dma_addr, "weight":self.weight_buf.meta.dma_addr}
      apply_patches(self.q, self.ew_template.patches, addrs)
      submit_template(self.device.fd_ctl, self.ew_template, self.q, self.task_buf, self.cmd_buf, self.cmd_buf_size)
      self.device._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      if op is Ops.EXP2 or (op is Ops.CUSTOM and arg == "silu"):
        raw = np.rint(np.array(struct.unpack(f'<{src.nbytes//2}e', ctypes.string_at(self.output_buf.va_addr, src.nbytes)), dtype=np.float32))
        _, _, inv_scale = build_lut(op, arg, self.lut_size)
        out = ((raw.astype(np.uint16) / 2**14) - 1) / inv_scale if op is Ops.EXP2 else raw.astype(np.int16) / (2**15 - 1) / inv_scale
        ctypes.memmove(mv_address(memoryview(bufs[out_slot])[:src.nbytes]), mv_address(memoryview(out.astype(np.float16))), src.nbytes)
      else:
        ctypes.memmove(mv_address(memoryview(bufs[out_slot])[:src.nbytes]), self.output_buf.va_addr, src.nbytes)
    finally:
      self.device._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf, self.weight_buf, self.output_buf])

  def _run_conv1x1(self, out_slot:int, in_slot:int, weight_slot:int, in_channels:int, out_channels:int, spatial:int, bufs:tuple[Any, ...]) -> None:
    if in_channels == 1:
      in_full = np.zeros(3 * spatial, dtype=np.float16)
      wt_full = np.zeros(out_channels * 3, dtype=np.float16)
      in_full[:spatial] = np.frombuffer(bufs[in_slot], dtype=np.float16, count=spatial)
      wt_full.reshape(out_channels, 3)[:, 0] = np.frombuffer(bufs[weight_slot], dtype=np.float16, count=out_channels)
      return self._run_conv1x1(0, 1, 2, 3, out_channels, spatial, (bufs[out_slot], in_full, wt_full))
    p = conv_params(in_channels, out_channels, spatial)
    input_packed = pack_conv_input(memoryview(bufs[in_slot]), p)
    weight_packed = pack_conv_weights(memoryview(bufs[weight_slot]), p)
    packed_input_size = ((in_channels + int(p["align_c"]) - 1) // int(p["align_c"])) * int(p["width_stride"]) * int(p["align_c"]) * 2
    packed_weight_size = weight_packed.nbytes
    packed_output_size = ((out_channels + int(p["align_out_c"]) - 1) // int(p["align_out_c"]))
    packed_output_size *= int(p["out_width_stride"]) * int(p["align_out_c"]) * 2
    self.task_buf = self.device._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task_buf")
    self.cmd_buf = self.device._gpu_alloc(self.cmd_buf_size, 0, name="cmd_buf")
    self.input_buf = self.device._gpu_alloc(packed_input_size, 0, name="input")
    self.weight_buf = self.device._gpu_alloc(REGCMD_RESERVED + packed_weight_size, 0, name="weight")
    self.output_buf = self.device._gpu_alloc(packed_output_size, 0, name="output")
    try:
      ctypes.memset(self.input_buf.va_addr, 0, packed_input_size)
      ctypes.memmove(self.input_buf.va_addr, mv_address(memoryview(input_packed)), input_packed.nbytes)
      ctypes.memmove(self.weight_buf.va_addr + REGCMD_RESERVED, mv_address(memoryview(weight_packed)), packed_weight_size)
      self.device._gpu_sync(self.input_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.device._gpu_sync(self.weight_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.device._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      self.q = list(self.conv_template.regcmd)
      addrs = {"output":self.output_buf.meta.dma_addr, "input":self.input_buf.meta.dma_addr, "weight":self.weight_buf.meta.dma_addr}
      apply_patches(self.q, self.conv_template.patches, addrs)
      submit_template(self.device.fd_ctl, self.conv_template, self.q, self.task_buf, self.cmd_buf, self.cmd_buf_size)
      self.device._gpu_sync(self.output_buf, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
      dst = memoryview(bytearray(self.output_buf.size))
      ctypes.memmove(mv_address(dst), self.output_buf.va_addr, self.output_buf.size)
      out = unpack_conv_output(dst, p)
      ctypes.memmove(mv_address(memoryview(bufs[out_slot])[:out.nbytes]), mv_address(memoryview(out)), out.nbytes)
    finally:
      self.device._gpu_free_multiple([self.task_buf, self.cmd_buf, self.input_buf, self.weight_buf, self.output_buf])

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    self.device.reset_npu()
    st = time.perf_counter()
    if self.ew_meta is not None:
      op, size, out_slot, lhs_slot, rhs_slot = self.ew_meta
      self._run_elementwise(op, size, out_slot, lhs_slot, rhs_slot, bufs, self.ew_arg)
      return time.perf_counter() - st
    if self.fused_matmul_meta is not None:
      try:
        self._run_fused_matmul(bufs)
        self.fused_matmul_hits += 1
        return time.perf_counter() - st
      except Exception as e:
        reason = str(e)
        self.fused_matmul_fallbacks += 1
    if self.conv_meta is not None:
      out_slot, in_slot, weight_slot, in_channels, out_channels, spatial = self.conv_meta
      self._run_conv1x1(out_slot, in_slot, weight_slot, in_channels, out_channels, spatial, bufs)
      return time.perf_counter() - st
    raise RuntimeError(f"unsupported Rockchip template family {self.template.family if self.template is not None else None}")

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  has_threads = False
  tensor_cores = tc.rockchip
  hardware_ops = {Ops.MUL, Ops.MAX, Ops.ADD, Ops.SUB}
  code_for_op = {k:v for k,v in python_alu.items() if k not in [Ops.MULACC, Ops.RECIPROCAL, Ops.CMPNE]} | {Ops.FDIV: 0}
  # hacks, turned unsupported dtype to half and lut function to Ops.CUSTOM
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
    (UPat(Ops.WMMA, dtype=dtypes.float.vec(4), name="x"),
     lambda x: UOp(Ops.WMMA, dtypes.half.vec(4), x.src,
                   (x.arg[0], x.arg[1], x.arg[2], dtypes.half, *x.arg[4:])).cast(dtypes.float.vec(4))),
    (UPat(Ops.MUL, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.MUL, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.ADD, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.ADD, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.MAX, dtypes.int, name="x"),
     lambda x: x.src[0].cast(dtypes.float16).alu(Ops.MAX, x.src[1].cast(dtypes.float16)).cast(dtypes.int)),
    (UPat(Ops.ADD, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.ADD, x.src[1].cast(dtypes.half))),
    (UPat(Ops.MAX, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.MAX, x.src[1].cast(dtypes.half))),
    (UPat(Ops.NEG, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.NEG)),
    (UPat(Ops.EXP2, dtypes.float, name="x"),
     lambda x: x.src[0].cast(dtypes.half).alu(Ops.EXP2)),
    (UPat(Ops.TRUNC, dtypes.floats, name="x"),
     _rk_trunc_fix),
    (UPat.var("x", dtypes.floats).alu(Ops.FDIV,
      UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
     lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
    (UPat.var("x", dtypes.floats) * UPat.const(dtypes.floats, 1).alu(Ops.FDIV,
      UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
     lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
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
  def render(self, uops:list[UOp]) -> str:
    if (ew_meta:=elementwise_meta(uops, self.hardware_ops)) is not None:
      op, size, out_slot, lhs_slot, rhs_slot, arg = ew_meta
      return base64.b64encode(encode_template(build_elementwise_template(op, size, out_slot, lhs_slot, rhs_slot, arg))).decode()
    if (conv_meta:=conv1x1_meta(uops)) is not None:
      out_slot, in_slot, weight_slot, in_channels, out_channels, spatial = conv_meta
      return base64.b64encode(encode_template(build_conv1x1_template(
        conv_params(in_channels, out_channels, spatial), out_slot, in_slot, weight_slot))).decode()
    raise RuntimeError("unsupported Rockchip program: no RKTemplatePackage match")

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

RockchipRenderer.compiler = RockchipCompiler()

class RockchipRegisterAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev._gpu_alloc(size, 0)
  def _do_copy(self, src_addr, dest_addr, src_size):
    ctypes.memmove(dest_addr, src_addr, src_size)

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self._do_copy(mv_address(src), dest.va_addr, src.nbytes)

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self._do_copy(src.va_addr, mv_address(dest), src.size)

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    return to_mv(ctypes.cast(int, src.va_addr), src.size)

class RockchipAllocator(Allocator['RockchipDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.target = os.getenv("ROCKCHIP_TARGET", "rk3588-rknpu2")
    self.drm_path = os.getenv("ROCKCHIP_DRM", "/dev/dri/card1")
    self.fd_ctl = FileIOInterface(self.drm_path, os.O_RDWR)
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
    mem_create.flink_name = self.create_flink_name(
      mem_create.handle, name, virt_address=va_addr, obj_addr=mem_create.obj_addr, dma_address=mem_create.dma_addr)

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
