from typing import Any
import base64, time, struct, functools, ctypes, mmap, os, itertools, math, numpy as np
from pathlib import Path
from tinygrad.dtype import dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, get_single_element, mv_address, to_mv
from tinygrad.device import Compiled, Compiler, Allocator, BufferSpec
from tinygrad.codegen.opt import tc
from tinygrad.uop.ops import exec_alu, python_alu, Ops, UOp, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad.runtime.ops_cpu import HCQBuffer
from tinygrad.runtime.ops_python import storage_fmt_for_dtype, to_storage_scalar, from_storage_scalar, load, _store
from tinygrad.runtime.support.hcq import FileIOInterface, HCQAllocatorBase
from tinygrad.runtime.support.rockchip import (
  REGCMD_RESERVED, RK_TEMPLATE_MAGIC, RKTemplatePackage, build_conv1x1_template,
  build_elementwise_template, build_fused_matmul_template, build_lut, build_wmma_template, conv_params, decode_template, encode_template,
  lut_enabled, pack_conv_input, pack_conv_weights, apply_patches, conv1x1_meta, elementwise_meta, fused_matmul_meta, submit_template,
  pool2d_meta, unpack_conv_output, validate_template, wmma_params,
)
from tinygrad.runtime.autogen import rockchip as rk

def _rk_env(name:str, default:int) -> int:
  try: return int(os.getenv(name, str(default)))
  except ValueError: return default

def _safe_div(x, y):
  try: return x / y
  except ZeroDivisionError: return math.nan if x == 0 else math.copysign(math.inf, x)

def _discover_rockchip_drm(sysfs:str="/sys/class/drm", devfs:str="/dev/dri") -> str:
  if override:=os.getenv("ROCKCHIP_DRM"): return override
  probed = []
  for card in sorted(Path(sysfs).glob("card[0-9]*")):
    if not card.name[4:].isdigit(): continue
    driver = Path(os.path.realpath(card / "device" / "driver")).name
    path = str(Path(devfs) / card.name)
    probed.append(f"{path}:{driver or 'unknown'}")
    if driver.lower() == "rknpu": return path
  raise RuntimeError("failed to find Rockchip RKNPU DRM node; probed " + ", ".join(probed or [str(Path(sysfs) / "card*")]))

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
    elif self.template is not None and self.template.family == "fused_matmul":
      assert self.template.meta is not None
      self.fused_matmul_meta = self.template.meta
    elif self.template is not None and self.template.family == "fill_zero":
      assert self.template.meta is not None
      self.fill_zero_meta = self.template.meta
    elif self.template is not None and self.template.family == "fill_const":
      assert self.template.meta is not None
      self.fill_const_meta = self.template.meta
    elif self.template is not None and self.template.family == "fill_range":
      assert self.template.meta is not None
      self.fill_range_meta = self.template.meta
    elif self.template is not None and self.template.family == "pool2d":
      assert self.template.meta is not None
      self.pool_meta = self.template.meta
    elif self.template is not None and self.template.family == "uops":
      assert self.template.meta is not None
      self.uops = self.template.meta["uops"]
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
    if not hasattr(self, "fused_matmul_meta"): self.fused_matmul_meta = None
    if not hasattr(self, "fill_zero_meta"): self.fill_zero_meta = None
    if not hasattr(self, "fill_const_meta"): self.fill_const_meta = None
    if not hasattr(self, "fill_range_meta"): self.fill_range_meta = None
    if not hasattr(self, "pool_meta"): self.pool_meta = None
    if not hasattr(self, "uops"): self.uops = None
    self.fused_matmul_hits = 0
    self.fused_matmul_fallbacks = 0

  def _dtype_from_code(self, code:int):
    if code == 0: return np.float16
    if code == 1: return np.float32
    if code == 2: return np.int32
    raise RuntimeError(f"dtype_code_{code}")

  def _run_wmma_matmul(self, a_matrix:np.ndarray, b_matrix:np.ndarray) -> np.ndarray:
    m, k = a_matrix.shape
    if b_matrix.shape[0] != k: raise RuntimeError("k_mismatch")
    n = int(b_matrix.shape[1])
    wmma_meta = wmma_params(int(m), int(n), int(k))
    in_pack = np.zeros(wmma_meta["align_in"] * wmma_meta["m"], dtype=np.float16)
    wt_pack = np.zeros(wmma_meta["align_out"] * wmma_meta["align_in"], dtype=np.float16)
    if m == n == k and k % 64 == 0 and 64 <= k <= 256:
      in_pack[:] = a_matrix.reshape(m, -1, 8).transpose(1, 0, 2).ravel()
      wt = np.zeros((wmma_meta["align_out"], wmma_meta["align_in"]), dtype=np.float16)
      wt[:n, :k] = b_matrix.T[:n, :k]
      wt_pack[:] = wt.reshape(wmma_meta["align_out"]//16, 16, wmma_meta["align_in"]//32, 32).transpose(0, 2, 1, 3).ravel()
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
    if batch > 1 and self.fused_matmul_meta["a_bs"] not in (0, m*k): raise RuntimeError("lhs_batch_stride")
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

  def _run_fused_matmul_cpu(self, bufs:tuple[Any, ...]) -> None:
    m, n, k, batch = (self.fused_matmul_meta[x] for x in ("m", "n", "k", "batch"))
    a_slot, b_slot, c_slot = self.fused_matmul_meta["a_slot"], self.fused_matmul_meta["b_slot"], self.fused_matmul_meta["c_slot"]
    a_arr = np.frombuffer(bufs[a_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["a_dt"]))
    b_arr = np.frombuffer(bufs[b_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["b_dt"]))
    c_arr = np.frombuffer(bufs[c_slot], dtype=self._dtype_from_code(self.fused_matmul_meta["c_dt"]))
    for bidx in range(batch):
      a_base, b_base, c_base = (bidx * self.fused_matmul_meta[x] for x in ("a_bs", "b_bs", "c_bs"))
      a_block, b_block = a_arr[a_base:a_base + m*k], b_arr[b_base:b_base + k*n]
      a_matrix = (a_block.reshape(m, k) if self.fused_matmul_meta["ta"] == 0 else a_block.reshape(k, m).T).astype(np.float16)
      b_matrix = (b_block.reshape(k, n) if self.fused_matmul_meta["tb"] == 0 else b_block.reshape(n, k).T).astype(np.float16)
      out = a_matrix.astype(np.float32) @ b_matrix.astype(np.float32)
      c_block = c_arr[c_base:c_base + m*n]
      if self.fused_matmul_meta["c_dt"] == 0: c_block[:] = out.astype(np.float16).reshape(-1)
      else: c_block[:] = out.reshape(-1)

  def _run_fill_zero(self, bufs:tuple[Any, ...]) -> None:
    out_slot = self.fill_zero_meta["out_slot"]
    size = self.fill_zero_meta["size"]
    memoryview(bufs[out_slot]).cast("B")[:size * 2] = b"\x00" * (size * 2)

  def _run_fill_const(self, bufs:tuple[Any, ...]) -> None:
    out_slot = self.fill_const_meta["out_slot"]
    size = self.fill_const_meta["size"]
    dtype = self._dtype_from_code(self.fill_const_meta["dtype_code"])
    out = np.full(size, self.fill_const_meta["value"], dtype=dtype)
    memoryview(bufs[out_slot]).cast("B")[:out.nbytes] = memoryview(out).cast("B")

  def _run_fill_range(self, bufs:tuple[Any, ...]) -> None:
    out_slot = self.fill_range_meta["out_slot"]
    size = self.fill_range_meta["size"]
    dtype = self._dtype_from_code(self.fill_range_meta["dtype_code"])
    out = np.arange(size, dtype=dtype)
    memoryview(bufs[out_slot]).cast("B")[:out.nbytes] = memoryview(out).cast("B")

  def _run_pool2d(self, bufs:tuple[Any, ...]) -> None:
    op = self.pool_meta["op"]
    out_slot, in_slot = self.pool_meta["out_slot"], self.pool_meta["in_slot"]
    in_h, in_w, channels = (int(self.pool_meta[k]) for k in ("in_h", "in_w", "channels"))
    x = np.frombuffer(bufs[in_slot], dtype=np.float16, count=in_h * in_w * channels).reshape(in_h, in_w, channels)
    if op.startswith("global"):
      if op == "globalmax": out = np.max(x, axis=(0, 1), keepdims=True).astype(np.float16)
      elif op == "globalmin": out = np.min(x, axis=(0, 1), keepdims=True).astype(np.float16)
      else: out = np.mean(x.astype(np.float32), axis=(0, 1), keepdims=True).astype(np.float16)
    else:
      out_h, out_w = in_h - 1, in_w - 1
      out = np.empty((out_h, out_w, channels), dtype=np.float16)
      for y in range(out_h):
        for x0 in range(out_w):
          window = x[y:y + 2, x0:x0 + 2]
          if op == "max": out[y, x0] = np.max(window, axis=(0, 1))
          elif op == "min": out[y, x0] = np.min(window, axis=(0, 1))
          else: out[y, x0] = np.mean(window.astype(np.float32), axis=(0, 1)).astype(np.float16)
    memoryview(bufs[out_slot]).cast("B")[:out.nbytes] = memoryview(out).cast("B")

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
    if spatial > 256:
      out = np.zeros((out_channels, spatial), dtype=np.float32)
      for start in range(0, spatial, 256):
        end = min(start + 256, spatial)
        tmp = memoryview(bytearray(out_channels * (end - start) * dtypes.float16.itemsize))
        inp = memoryview(bufs[in_slot]).cast("B")[start * in_channels * dtypes.float16.itemsize:end * in_channels * dtypes.float16.itemsize]
        self._run_conv1x1(0, 1, 2, in_channels, out_channels, end - start, (tmp, inp, bufs[weight_slot]))
        out[:, start:end] = np.frombuffer(tmp, dtype=np.float16, count=out_channels * (end - start)).reshape(out_channels, end - start)
      outh = out.astype(np.float16)
      memoryview(bufs[out_slot]).cast("B")[:outh.nbytes] = memoryview(outh).cast("B")
      return
    if in_channels > 4:
      inp = np.frombuffer(bufs[in_slot], dtype=np.float16, count=in_channels*spatial).reshape(in_channels, spatial)
      wt = np.frombuffer(bufs[weight_slot], dtype=np.float16, count=out_channels*in_channels).reshape(out_channels, in_channels)
      out = np.zeros((out_channels, spatial), dtype=np.float32)
      for start in range(0, in_channels, 4):
        end = min(start + 4, in_channels)
        tmp = memoryview(bytearray(out_channels * spatial * dtypes.float16.itemsize))
        self._run_conv1x1(0, 1, 2, end-start, out_channels, spatial, (tmp, inp[start:end].reshape(-1), wt[:, start:end].reshape(-1)))
        out += np.frombuffer(tmp, dtype=np.float16, count=out_channels*spatial).reshape(out_channels, spatial).astype(np.float32)
      outh = out.astype(np.float16)
      memoryview(bufs[out_slot]).cast("B")[:outh.nbytes] = memoryview(outh).cast("B")
      return
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

  def _run_uops(self, bufs:tuple[Any, ...], global_size:tuple[int, int, int], local_size:tuple[int, int, int], vals:tuple[int, ...]) -> None:
    assert self.uops is not None
    base_warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    void_ops = {Ops.END, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP, Ops.STORE}
    loop_ends: dict[int, int] = {srcs[1]:i for i, (uop, _, srcs, _) in enumerate(self.uops) if uop == Ops.END}
    has_control_flow = any(op in (Ops.RANGE, Ops.IF, Ops.ENDIF) for op,_,_,_ in self.uops)
    vectorize_global = not has_control_flow and all(x == 1 for x in local_size) and math.prod(global_size) > 1
    if vectorize_global:
      global_iter, chunk_size = itertools.product(*[range(x) for x in global_size[::-1]]), 16384
      global_iters = ((tuple(0 for _ in global_size), chunk) for chunk in iter(lambda: list(itertools.islice(global_iter, chunk_size)), []))
    else:
      global_iters = ((idxs, base_warp) for idxs in itertools.product(*[range(x) for x in global_size[::-1]]))

    for idxs,warp in global_iters:
      warp_size = len(warp)
      values: dict[int, Any] = {}
      pbufs, pvals = list(bufs), list(vals)
      i = 0
      while i < len(self.uops):
        uop, dtype, srcs, arg = self.uops[i]
        src_values = [values[v] for v in srcs if self.uops[v][0] not in void_ops]
        src_dtypes = [self.uops[v][1] for v in srcs if self.uops[v][0] not in void_ops]
        if uop is Ops.END:
          i = srcs[1]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.GROUP):
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        if uop is Ops.STORE:
          for j,val in enumerate(src_values[1] if src_dtypes[1].count > 1 else [src_values[1]]):
            for (m,o,g),v in zip(src_values[0], val):
              if g and o is not None and 0 <= o+j < len(m): _store(m, o+j, v, src_dtypes[1].scalar())
          i += 1
          continue
        if uop is Ops.AFTER: values[i] = src_values[0]
        elif uop in {Ops.PARAM, Ops.BUFFER, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          assert isinstance(dtype, PtrDType)
          storage_fmt = storage_fmt_for_dtype(dtype.base.scalar())
          if storage_fmt is None: raise RuntimeError(f"{dtype=} is not supported")
          if uop is Ops.DEFINE_REG: values[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          elif uop is Ops.PARAM: values[i] = [pbufs.pop(0).cast(storage_fmt)] * warp_size
          else: values[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt)] * warp_size
        elif uop is Ops.DEFINE_VAR: values[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0] == 'g': values[i] = [x[2-int(arg[-1])] for x in warp] if vectorize_global else [idxs[2-int(arg[-1])]] * warp_size
          elif arg[0] == 'l': values[i] = [0] * warp_size if vectorize_global else [x[2-int(arg[-1])] for x in warp]
        elif uop is Ops.CONST: values[i] = [arg] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(src_dtypes[0], ImageDType):
            for m,ox,oy in zip(src_values[0], src_values[1][0], src_values[1][1]):
              ret.append((m, None) if ox < 0 or ox >= src_dtypes[0].shape[1] or oy < 0 or oy >= src_dtypes[0].shape[0] else (m, ox*4 + oy*src_dtypes[0].shape[1]*4))
          else:
            for m,o in zip(src_values[0], src_values[1]): ret.append((m,o))
          values[i] = [(m,o,g) for (m,o),g in zip(ret, src_values[2] if len(src_values) == 3 else [True]*len(ret))]
        elif uop is Ops.CAST and isinstance(dtype, PtrDType): values[i] = src_values[0]
        elif uop is Ops.RANGE:
          if i not in values: values[i] = [0] * warp_size
          else:
            for j in range(len(values[i])): values[i][j] += 1
          if values[i][0] == src_values[0][0]:
            del values[i]
            i = loop_ends[i] + 1
            continue
        elif uop is Ops.STACK or ((vectorize_op:=getattr(Ops, "VECTORIZE", None)) is not None and uop is vectorize_op): values[i] = src_values
        elif uop is Ops.BITCAST:
          packed = struct.pack(str(warp_size) + storage_fmt_for_dtype(src_dtypes[0].scalar()), *[to_storage_scalar(x, src_dtypes[0].scalar()) for x in src_values[0]])
          values[i] = [from_storage_scalar(x, dtype.scalar()) for x in struct.unpack(str(warp_size) + storage_fmt_for_dtype(dtype.scalar()), packed)]
        elif uop is Ops.CAST:
          values[i] = src_values[0] if dtype.scalar() is dtypes.half and src_dtypes[0].scalar() in dtypes.ints else \
            [truncate.get(dtype, lambda dt: dt)(dtype.const(x)) for x in src_values[0]]
        elif uop is Ops.LOAD:
          values[i] = [load([src_values[k][j] if k != 0 and src_dtypes[k].count > 1 else src_values[k] for k in range(len(src_values))], j, dtype.scalar())
                       for j in range(dtype.count)] if dtype.count > 1 else load(src_values, 0, dtype)
        elif uop is Ops.GEP:
          v = src_values[0][get_single_element(arg)]
          values[i] = v if isinstance(v, (list, tuple)) else [v]
        elif uop is Ops.CUSTOM:
          if arg == "relu": values[i] = [dtype.const(max(0, x)) for x in src_values[0]]
          elif arg == "silu": values[i] = [dtype.const(float(x) / (1 + math.exp(-float(x)))) for x in src_values[0]]
          elif arg == "cmplt_diff2bool": values[i] = [dtype.const(1.0 if x > 0 else 0.0) for x in src_values[0]]
          elif arg == "cmpeq_diff_zero_to_nan_to_32800": values[i] = [dtype.const(32800.0 if x == 0 else 0.0) for x in src_values[0]]
          elif arg == "cmpeq_32800_to_bool":
            target = float(np.float16(32800.0))
            values[i] = [dtype.const(1.0 if float(x) == target else 0.0) for x in src_values[0]]
          else: raise RuntimeError(f"unsupported Rockchip custom uop {arg}")
        elif uop is Ops.FDIV:
          values[i] = [dtype.const(_safe_div(float(a), float(b))) for a, b in zip(*src_values)]
        elif uop is Ops.WMMA or uop in GroupOp.ALU:
          if uop is not Ops.WMMA: assert all_same([len(x) for x in src_values]), f"{[len(x) for x in src_values]} doesn't match on {uop}"
          def _intlike(x): return isinstance(x, (int, np.integer, bool)) or (isinstance(x, float) and x.is_integer())
          if dtype.scalar() is dtypes.half and all(all(_intlike(x) for x in p) for p in zip(*src_values)):
            values[i] = [python_alu[uop](*p) for p in zip(*src_values)]
          else:
            values[i] = [exec_alu(uop, dtype, p) for p in zip(*src_values)]
        assert i in values, (uop, dtype, srcs, arg)
        i += 1

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
        self._run_fused_matmul_cpu(bufs)
        self.fused_matmul_fallbacks += 1
        if getenv("DEBUG", 0) >= 2: print(f"Rockchip fused_matmul fallback: {reason}")
        return time.perf_counter() - st
    if self.fill_zero_meta is not None:
      self._run_fill_zero(bufs)
      return time.perf_counter() - st
    if self.fill_const_meta is not None:
      self._run_fill_const(bufs)
      return time.perf_counter() - st
    if self.fill_range_meta is not None:
      self._run_fill_range(bufs)
      return time.perf_counter() - st
    if self.pool_meta is not None:
      self._run_pool2d(bufs)
      return time.perf_counter() - st
    if self.conv_meta is not None:
      out_slot, in_slot, weight_slot, in_channels, out_channels, spatial = self.conv_meta
      self._run_conv1x1(out_slot, in_slot, weight_slot, in_channels, out_channels, spatial, bufs)
      return time.perf_counter() - st
    if self.uops is not None:
      self._run_uops(bufs, global_size, local_size, vals)
      return time.perf_counter() - st
    raise RuntimeError(f"unsupported Rockchip template family {self.template.family if self.template is not None else None}")

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  has_threads = False
  tensor_cores = tc.rockchip
  hardware_ops = {Ops.MUL, Ops.MAX, Ops.ADD, Ops.SUB}
  code_for_op = {k:v for k,v in python_alu.items() if k not in [Ops.MULACC, Ops.CMPNE]} | {Ops.FDIV: 0}
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
    (UPat(Ops.TRUNC, dtypes.floats, name="x"),
     _rk_trunc_fix),
    (UPat.var("x", dtypes.floats).alu(Ops.FDIV,
      UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
     lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
    (UPat.var("x", dtypes.floats) * UPat.const(dtypes.floats, 1).alu(Ops.FDIV,
      UPat.const(dtypes.floats, 1) + (UPat.var("x", dtypes.floats) * UPat.cvar("c", dtypes.floats, vec=False)).exp2()),
     lambda x, c: UOp(Ops.CUSTOM, x.dtype, src=(x,), arg="silu")),
  ])
  def render(self, uops:list[UOp]) -> str:
    if (zero_store:=self._zero_store_meta(uops)) is not None:
      return base64.b64encode(encode_template(RKTemplatePackage(
        1, "rk3588-rknpu2", "fill_zero", (), meta=zero_store,
      ))).decode()
    if (const_store:=self._const_store_meta(uops)) is not None:
      return base64.b64encode(encode_template(RKTemplatePackage(
        1, "rk3588-rknpu2", "fill_const", (), meta=const_store,
      ))).decode()
    if (range_store:=self._range_store_meta(uops)) is not None:
      return base64.b64encode(encode_template(RKTemplatePackage(
        1, "rk3588-rknpu2", "fill_range", (), meta=range_store,
      ))).decode()
    if (pool_meta:=pool2d_meta(uops)) is not None:
      return base64.b64encode(encode_template(RKTemplatePackage(
        1, "rk3588-rknpu2", "pool2d", (), meta=pool_meta,
      ))).decode()
    if (mm_meta:=fused_matmul_meta(uops)) is not None:
      return base64.b64encode(encode_template(build_fused_matmul_template(mm_meta))).decode()
    if (ew_meta:=elementwise_meta(uops, self.hardware_ops)) is not None:
      op, size, out_slot, lhs_slot, rhs_slot, arg = ew_meta
      return base64.b64encode(encode_template(build_elementwise_template(op, size, out_slot, lhs_slot, rhs_slot, arg))).decode()
    if (conv_meta:=conv1x1_meta(uops)) is not None:
      out_slot, in_slot, weight_slot, in_channels, out_channels, spatial = conv_meta
      return base64.b64encode(encode_template(build_conv1x1_template(
        conv_params(in_channels, out_channels, spatial), out_slot, in_slot, weight_slot))).decode()
    uop_to_idx = {u:i for i,u in enumerate(uops)}
    packed_uops = tuple((u.op, u.dtype, [uop_to_idx[s] for s in u.src], u.arg) for u in uops)
    return base64.b64encode(encode_template(RKTemplatePackage(1, "rk3588-rknpu2", "uops", (), meta={"uops":packed_uops}))).decode()

  def _zero_store_meta(self, uops:list[UOp]) -> dict[str, int]|None:
    if (meta:=self._const_store_meta(uops)) is None: return None
    return meta if float(meta["value"]) == 0.0 else None

  def _const_store_meta(self, uops:list[UOp]) -> dict[str, int]|None:
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores or any(len(x.src) != 2 for x in stores): return None
    fill_values, bufs = [], []
    for store in stores:
      idx, value = store.src
      if value.op is Ops.CONST:
        fill_values.append(float(getattr(value.arg, "val", value.arg)))
      elif value.op is Ops.STACK and all(v.op is Ops.CONST for v in value.src):
        vals = [float(getattr(v.arg, "val", v.arg)) for v in value.src]
        if any(v != vals[0] and not (math.isnan(v) and math.isnan(vals[0])) for v in vals): return None
        fill_values.append(vals[0])
      else:
        return None
      if idx.op is Ops.CAST and len(idx.src) == 1: idx = idx.src[0]
      if idx.op is not Ops.INDEX or len(idx.src) != 2: return None
      bufs.append(idx.src[0])
    if any(v != fill_values[0] and not (math.isnan(v) and math.isnan(fill_values[0])) for v in fill_values): return None
    if len(set(bufs)) != 1: return None
    buf = bufs[0]
    if buf.op is not Ops.PARAM or not hasattr(buf.dtype, "size"): return None
    base = getattr(getattr(buf.dtype, "base", None), "scalar", lambda: None)()
    dtype_code = 0 if base is dtypes.half else 1 if base is dtypes.float else 2 if base is dtypes.int else None
    if dtype_code is None: return None
    return {"out_slot": int(buf.arg), "size": int(buf.dtype.size), "dtype_code": dtype_code, "value": fill_values[0]}

  def _range_store_meta(self, uops:list[UOp]) -> dict[str, int]|None:
    stores = [u for u in uops if u.op is Ops.STORE]
    if not stores or any(len(x.src) != 2 for x in stores): return None
    bufs = []
    for store in stores:
      idx, value = store.src
      if idx.op is Ops.CAST and len(idx.src) == 1: idx = idx.src[0]
      if idx.op is not Ops.INDEX or len(idx.src) != 2 or value is not idx.src[1]: return None
      bufs.append(idx.src[0])
    if len(set(bufs)) != 1: return None
    buf = bufs[0]
    if buf.op is not Ops.PARAM or not hasattr(buf.dtype, "size"): return None
    base = getattr(getattr(buf.dtype, "base", None), "scalar", lambda: None)()
    dtype_code = 0 if base is dtypes.half else 1 if base is dtypes.float else 2 if base is dtypes.int else None
    if dtype_code is None: return None
    return {"out_slot": int(buf.arg), "size": int(buf.dtype.size), "dtype_code": dtype_code}

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
    self.drm_path = _discover_rockchip_drm()
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
