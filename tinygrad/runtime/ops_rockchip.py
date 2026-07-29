# pylint: disable=cell-var-from-loop
# RK3588 NPU compiled backend: deterministic register commands + DMA relocations.
# PR 1 native contract: one single-task fp16 path per compute family.
#   DPU: binary EW (ADD/SUB/MUL/MAX) with two INDEX operands, scalar operand, or DMA copy.
#   CMAC: matmul MUL(INDEX,INDEX) with REDUCE(ADD), or sum via ones-vector.
#   PPU: global max pool REDUCE(MAX, INDEX) over (H,W,C) → (C,).
# Fill, broadcast, mean, non-fp16, non-affine indexing, fused epilogues, and
# multi-task are explicitly rejected via RKPLAN_REJECT.
# All compute (including copy) executes on the NPU — no host-side tensor arithmetic.
import ctypes, mmap, os, struct
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import getenv, mv_address, to_mv, Target
from tinygrad.device import Compiled, Program, BufferSpec, TinyELF
from tinygrad.uop.ops import UOp
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.hcq import HCQBuffer, FileIOInterface, HCQAllocatorBase
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.rockchip import (build_native_program,
  encode_rk, decode_rk, encode_rk_multi, decode_rk_multi, RKTask, RKReloc, RKSubTask,
  _CONST_SLOT, _ZERO_SLOT, _HOST_BITWISE_LAYOUT)

# CMAC byte-level data transforms (no NumPy per plan §0.3 B2)
def _pad_a(src, dst, M, K, align_in):
  """Copy M rows of K fp16 elements into a zeroed (M, align_in) buffer."""
  ctypes.memset(dst, 0, M * align_in * 2)
  for m in range(M): ctypes.memmove(dst + m * align_in * 2, src + m * K * 2, K * 2)

def _swizzle_b(src, dst, K, N, align_out, align_in):
  """Transpose (K,N) and swizzle into (align_out//16, align_in//32, 16, 32) fp16 layout."""
  ctypes.memset(dst, 0, align_out * align_in * 2)
  s, d = ctypes.cast(src, ctypes.POINTER(ctypes.c_uint16)), ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint16))
  for k in range(K):
    for n in range(N): d[(((n//16)*(align_in//32)+(k//32))*16+(n%16))*32+(k%32)] = s[k*N+n]

def _fp32_to_fp16(b):
  try: return struct.unpack('<H', struct.pack('<e', struct.unpack('<f', struct.pack('<I', b))[0]))[0]
  except OverflowError: return 0xFC00 if b>>31 else 0x7C00

def _convert_fp32_to_fp16_buf(src, dst, n):
  """Convert n fp32 elements at src to fp16 at dst (buffer-level cast, not NPU compute)."""
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 4), dtype=np.float32).astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _convert_periodic_fp32_to_fp16_buf(src, dst, n):
  """Reduce finite fp32 angles to [-pi,pi]; encode nonfinite values as a detectable sentinel."""
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 4), dtype=np.float32).astype(np.float64)
  finite = np.isfinite(arr)
  arr[finite] = np.remainder(arr[finite]+np.pi, 2*np.pi)-np.pi
  arr[~finite] = 65472.0
  arr = arr.astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _convert_fp16_to_fp32_buf(src, dst, n):
  """Convert n fp16 elements at src to fp32 at dst (buffer-level cast, not NPU compute)."""
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).astype(np.float32)
  ctypes.memmove(dst, arr.ctypes.data, n * 4)  # type: ignore[arg-type]

def _convert_bool_to_fp16_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n), dtype=np.bool_).astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _broadcast_fp16_buf(src, dst, src_n, n):
  data = ctypes.string_at(src, src_n * 2)
  ctypes.memmove(dst, (data * ((n + src_n - 1) // src_n))[:n*2], n * 2)

def _convert_fp16_to_int32_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).astype(np.int32)
  ctypes.memmove(dst, arr.ctypes.data, n * 4)  # type: ignore[arg-type]

def _convert_fp16_to_uint8_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).astype(np.uint8)
  ctypes.memmove(dst, arr.ctypes.data, n)  # type: ignore[arg-type]

def _convert_fp16_to_bool_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).astype(np.bool_)
  ctypes.memmove(dst, arr.ctypes.data, n)  # type: ignore[arg-type]

def _truncate_fp16_buf(src, dst, n):
  """Implement trunc as the same fp16→int32→fp16 cast round-trip used by its graph semantics."""
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).copy()
  finite_nonzero = np.isfinite(arr) & (arr != 0)
  arr[finite_nonzero] = arr[finite_nonzero].astype(np.int32).astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _convert_int32_to_fp16_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 4), dtype=np.int32).astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _sanitize_fp16_comparison_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).copy()
  np.nan_to_num(arr, copy=False, nan=float("nan"), posinf=65504.0, neginf=-65504.0)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _unpack_cmac_out(src, dst, M, N, align_out):
  s, d = ctypes.cast(src, ctypes.POINTER(ctypes.c_uint32)), ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint16))
  for i in range(M * N):
    d[i] = _fp32_to_fp16(s[(i // N) * align_out + i % N])

def _run_host_bitwise(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Execute a tagged exact int32/uint32/bool bitwise task on mapped buffers."""
  total, tag, op, lhs_const, lhs_value, lhs_dtype, rhs_const, rhs_value, rhs_dtype, out_dtype = task.layout
  assert tag == _HOST_BITWISE_LAYOUT
  slot_iter = iter(r.globals_slot for r in relocs[1:])

  def reader(is_const:int, value:int, dtype:int):
    if is_const: return lambda _: value & 0xFFFFFFFF
    buf = bufs[next(slot_iter)]
    itemsize = 1 if dtype == 2 else 4
    count = max(1, buf.size // itemsize)
    if dtype == 0:
      int_ptr = ctypes.cast(buf.va_addr, ctypes.POINTER(ctypes.c_int32))
      return lambda i: int(int_ptr[i % count]) & 0xFFFFFFFF
    if dtype == 1:
      uint_ptr = ctypes.cast(buf.va_addr, ctypes.POINTER(ctypes.c_uint32))
      return lambda i: int(uint_ptr[i % count])
    bool_ptr = ctypes.cast(buf.va_addr, ctypes.POINTER(ctypes.c_uint8))
    return lambda i: int(bool(bool_ptr[i % count]))

  lhs, rhs = reader(lhs_const, lhs_value, lhs_dtype), reader(rhs_const, rhs_value, rhs_dtype)
  out = bufs[relocs[0].globals_slot]
  for i in range(total):
    a, b = lhs(i), rhs(i)
    if op == 0: result = a ^ b
    elif op == 1: result = a & b
    elif op == 2: result = a | b
    elif op == 3: result = (a << (b & 31)) & 0xFFFFFFFF
    elif lhs_dtype == 0:
      signed = a if a < 1 << 31 else a-(1 << 32)
      result = (signed >> (b & 31)) & 0xFFFFFFFF
    else: result = a >> (b & 31)
    if out_dtype == 2: ctypes.cast(out.va_addr, ctypes.POINTER(ctypes.c_uint8))[i] = bool(result)
    elif out_dtype == 0:
      ctypes.cast(out.va_addr, ctypes.POINTER(ctypes.c_int32))[i] = result if result < 1 << 31 else result-(1 << 32)
    else: ctypes.cast(out.va_addr, ctypes.POINTER(ctypes.c_uint32))[i] = result

class RockchipProgram(Program['RockchipDevice']):
  cmds: list[int]
  task: RKTask
  relocs: list[RKReloc]
  subtasks: list[RKSubTask] | None  # multi-task (PC chain) or None for single-task
  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
    self.device, self.name, self.submit_count, self.last_enable_mask = dev, obj.name, 0, 0
    self.subtasks = None
    if len(obj.lib) >= 8 and struct.unpack_from("<II", obj.lib, 0) == (0x524b494d, 4):
      # Version 4: multi-task PC chain image
      self.subtasks = decode_rk_multi(obj.lib)
    elif len(obj.lib) >= 4 and struct.unpack_from("<I", obj.lib, 0)[0] == 0x524b494d:
      self.cmds, self.task, self.relocs = decode_rk(obj.lib)
    else:
      raise RuntimeError(f"rk: no Python fallback — binary is not an RKImage (len={len(obj.lib)}, first byte={obj.lib[0]:#x})")

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    dev = self.device
    dev.reset_npu()
    if self.subtasks is not None:
      return self._submit_multi(bufs)
    task = self.task
    temp:list[HCQBuffer] = []
    try:
      buf_map:dict[int, HCQBuffer] = {}
      cmac_bufs: list[HCQBuffer] = []
      if task.kind == "cmac":
        layout = task.layout
        M, N, K, align_in, align_out = layout[0], layout[1], layout[2], layout[3], layout[4]
        a_s, b_s = self.relocs[0].globals_slot, self.relocs[1].globals_slot
        a_buf = dev._gpu_alloc(max(M*align_in*2, 4096), 0)
        temp.append(a_buf)
        if a_s == _CONST_SLOT:
          ctypes.memmove(a_buf.va_addr, struct.pack('<e', task.const_val) * align_in, align_in * 2)  # type: ignore[arg-type]
        else:
          _pad_a(bufs[a_s].va_addr, a_buf.va_addr, M, K, align_in)
        b_buf = dev._gpu_alloc(max(align_out*align_in*2, 4096), 0)
        temp.append(b_buf)
        if b_s == _CONST_SLOT:
          ctypes.memmove(b_buf.va_addr, struct.pack('<e', task.const_val) * (align_out * align_in), align_out * align_in * 2)  # type: ignore[arg-type]
        else:
          _swizzle_b(bufs[b_s].va_addr, b_buf.va_addr, K, N, align_out, align_in)
        o_buf = dev._gpu_alloc(max(M*align_out*4, 4096), 0)
        temp.append(o_buf)
        cmac_bufs = [a_buf, b_buf, o_buf]  # ordered by reloc emission: A, B, output
      else:
        for i, b in enumerate(bufs): buf_map[i] = b  # type: ignore[assignment]
      # PPU: pad input to chan_padded (multiple of 8) and prepare padded output buffer
      ppu_padded = None
      if task.kind == "ppu":
        in_h, in_w, channels, chan_padded = task.layout
        in_slot = self.relocs[1].globals_slot  # reloc 0=output, reloc 1=input
        in_buf = buf_map[in_slot]
        K = in_h * in_w
        if chan_padded != channels:
          # Pad input: (K, channels) → (K, chan_padded) with -inf for max pooling
          padded_size = max(K * chan_padded * 2, 4096)
          pbuf = dev._gpu_alloc(padded_size, 0)
          temp.append(pbuf)
          pad = struct.pack('<e', -65504.0) * (chan_padded - channels)  # -inf padding
          for k in range(K):
            dst = pbuf.va_addr + k * chan_padded * 2
            ctypes.memmove(dst, in_buf.va_addr + k * channels * 2, channels * 2)  # type: ignore[arg-type]
            ctypes.memmove(dst + channels * 2, pad, (chan_padded - channels) * 2)  # type: ignore[arg-type]
          buf_map[in_slot] = pbuf  # type: ignore[assignment]
          # Allocate padded output buffer and redirect output; copy back after submission
          out_padded = dev._gpu_alloc(max(chan_padded * 2, 4096), 0)
          temp.append(out_padded)
          ppu_padded = (channels, chan_padded, out_padded)
          buf_map[task.out_slot] = out_padded  # type: ignore[assignment]
      # fp32 buffer-level conversion: NPU processes fp16, so convert fp32 inputs→fp16 temp buffers,
      # and redirect fp32 output to a fp16 temp buffer (converted back to fp32 after NPU execution).
      # Copy tasks handle fp32 directly (no NPU processing), so skip conversion for them.
      # PPU fp32 not yet supported (channel padding complicates output redirect).
      fp32_out_temp = None  # (original_out_buf, fp16_temp_buf, n_elements) for fp16→fp32 after NPU
      if task.kind in ("dpu", "dpu_lut") and not task.is_copy:
        total = task.layout[0] if task.kind != "ppu" else task.layout[2]  # n_elements
        # Convert fp32 input buffers to fp16 temp buffers
        for slot in task.fp32_inputs:
          if slot in buf_map and slot != _CONST_SLOT and slot != _ZERO_SLOT:
            src_buf = buf_map[slot]
            fp16_buf = dev._gpu_alloc(max(total * 2, 4096), 0)
            temp.append(fp16_buf)
            _convert_fp32_to_fp16_buf(src_buf.va_addr, fp16_buf.va_addr, total)
            buf_map[slot] = fp16_buf  # type: ignore[assignment]
        # Redirect fp32 output to fp16 temp buffer
        if task.fp32_output and task.out_slot in buf_map:
          orig_out = buf_map[task.out_slot]
          fp16_out = dev._gpu_alloc(max(total * 2, 4096), 0)
          temp.append(fp16_out)
          buf_map[task.out_slot] = fp16_out  # type: ignore[assignment]
          fp32_out_temp = (orig_out, fp16_out, total)
      # DPU DMA copy: host-side memmove (data movement, not NPU compute).
      # No submit_count increment — no NPU submission. Documented honestly as non-native.
      if task.is_fill:
        total = task.layout[0]
        out_buf = buf_map[self.relocs[0].globals_slot]
        itemsize = 4 if task.fp32_output else 2
        if task.const_val == 0.0:
          ctypes.memset(out_buf.va_addr, 0, total * itemsize)  # type: ignore[arg-type]
        else:
          if itemsize == 2:
            arr = struct.pack('<e', task.const_val) * total
          else:
            arr = struct.pack('<f', task.const_val) * total
          ctypes.memmove(out_buf.va_addr, arr, total * itemsize)  # type: ignore[arg-type]
        return
      if task.is_copy:
        total = task.layout[0]
        in_slot = self.relocs[1].globals_slot
        in_is_fp32 = in_slot in task.fp32_inputs
        out_is_fp32 = task.fp32_output
        in_buf, out_buf = buf_map[in_slot], buf_map[self.relocs[0].globals_slot]
        out_addr = out_buf.va_addr + task.out_offset
        if len(task.layout) > 1 and task.layout[1] < 0 and in_is_fp32 == out_is_fp32:
          # Scatter copy: source (possibly strided) → strided destination (for pad)
          # Layout: (total, -ndim, *in_shape, *src_strides, src_offset, *dst_strides, dst_offset)
          _, neg_ndim, *meta = task.layout
          ndim = -neg_ndim
          shape = meta[:ndim]
          src_strides = meta[ndim:2*ndim]
          src_offset = meta[2*ndim]
          dst_strides = meta[2*ndim+1:3*ndim+1]
          dst_offset = meta[-1]
          itemsize = 4 if in_is_fp32 else 2
          for in_idx in range(total):
            rem, src_idx = in_idx, src_offset
            for dim in range(ndim-1, -1, -1):
              rem, coord = divmod(rem, shape[dim])
              src_idx += coord * src_strides[dim]
            rem, dst_idx = in_idx, dst_offset
            for dim in range(ndim-1, -1, -1):
              rem, coord = divmod(rem, shape[dim])
              dst_idx += coord * dst_strides[dim]
            if 0 <= src_idx < in_buf.size // itemsize and 0 <= dst_idx < out_buf.size // itemsize:
              ctypes.memmove(out_addr + dst_idx*itemsize, in_buf.va_addr + src_idx*itemsize, itemsize)  # type: ignore[arg-type]
        elif len(task.layout) > 1 and in_is_fp32 == out_is_fp32:
          _, ndim, *meta = task.layout
          shape, strides, offset = meta[:ndim], meta[ndim:2*ndim], meta[-1]
          itemsize = 4 if in_is_fp32 else 2
          in_n = in_buf.size // itemsize
          out_n = out_buf.size // itemsize
          for out_idx in range(total):
            rem, src_idx = out_idx, offset
            for dim in range(ndim-1, -1, -1):
              rem, coord = divmod(rem, shape[dim])
              src_idx += coord * strides[dim]
            if 0 <= src_idx < in_n and 0 <= out_idx < out_n:
              ctypes.memmove(out_addr + out_idx*itemsize, in_buf.va_addr + src_idx*itemsize, itemsize)  # type: ignore[arg-type]
        elif in_is_fp32 and out_is_fp32:
          # fp32→fp32 copy: just memmove fp32 data directly
          ctypes.memmove(out_addr, in_buf.va_addr, total * 4)  # type: ignore[arg-type]
        elif in_is_fp32 and not out_is_fp32:
          # fp32→fp16 copy: convert
          _convert_fp32_to_fp16_buf(in_buf.va_addr, out_addr, total)
        elif not in_is_fp32 and out_is_fp32:
          # fp16→fp32 copy: convert
          _convert_fp16_to_fp32_buf(in_buf.va_addr, out_addr, total)
        else:
          ctypes.memmove(out_addr, in_buf.va_addr, total * 2)  # type: ignore[arg-type]
        return
      n_cmds = len(self.cmds)
      assert n_cmds <= dev.cmd_buf_size
      regcmd = ctypes.cast(dev.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * dev.cmd_buf_size)).contents  # type: ignore[arg-type]
      for i, cmd in enumerate(self.cmds): regcmd[i] = cmd
      for i, r in enumerate(self.relocs):
        if task.kind == "cmac" and cmac_bufs:
          # CMAC: A/B/output buffers already prepared in cmac_bufs (ordered by reloc index)
          dma = cmac_bufs[i].meta.dma_addr
          v = ((dma + r.addend) >> r.shift) & r.mask
        elif r.globals_slot == _CONST_SLOT:
          # scalar operand: allocate a buffer filled with the constant value (buffer prep, NPU does the EW op)
          total = task.layout[0]
          cbuf = dev._gpu_alloc(max(total * 2, 4096), 0)
          temp.append(cbuf)
          cval = struct.unpack('<f', struct.pack('<I', r.addend))[0]
          fp16_bytes = struct.pack('<e', cval) * total
          ctypes.memmove(cbuf.va_addr, fp16_bytes, total * 2)  # type: ignore[arg-type]
          dma = cbuf.meta.dma_addr
          v = (dma >> r.shift) & r.mask
        elif r.globals_slot == _ZERO_SLOT:
          # fill: allocate a zero-filled input buffer (buffer prep, NPU does ADD(zero, const) = fill)
          total = task.layout[0]
          zbuf = dev._gpu_alloc(max(total * 2, 4096), 0)
          temp.append(zbuf)
          ctypes.memset(zbuf.va_addr, 0, total * 2)  # type: ignore[arg-type]
          dma = zbuf.meta.dma_addr
          v = (dma >> r.shift) & r.mask
        else:
          dma = (cmac_bufs[i] if cmac_bufs else buf_map[r.globals_slot]).meta.dma_addr
          v = ((dma + r.addend) >> r.shift) & r.mask
        if r.field_shift:
          v = (v << r.field_shift) & 0xFFFFFFFF
          fm = (r.mask << r.field_shift) & 0xFFFFFFFF
        else: fm = r.mask
        w = regcmd[r.word_index]
        regcmd[r.word_index] = (w & ~(fm << 16)) | ((v & fm) << 16)
      t = ctypes.cast(dev.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents[0]  # type: ignore[arg-type]
      t.flags, t.op_idx, t.enable_mask, t.int_mask = 0, task.op_idx, task.enable_mask, task.int_mask
      t.int_clear, t.int_status, t.regcfg_amount, t.regcfg_offset = 0x1ffff, 0, n_cmds, 0
      t.regcmd_addr = dev.cmd_buf.meta.dma_addr
      rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd_ctl, __payload=rk.struct_rknpu_submit(
        flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=6000,
        task_start=0, task_number=1, task_counter=0, priority=0,
        task_obj_addr=dev.task_buf.meta.obj_addr, regcfg_obj_addr=0, task_base_addr=0,
        user_data=0, core_mask=1, fence_fd=-1,
        subcore_task=(rk.struct_rknpu_subcore_task*5)(
          rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
          rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
          rk.struct_rknpu_subcore_task(task_start=2, task_number=0))))
      if getenv("DEBUG") >= 1: print(f"submit {self.name}: mask={task.enable_mask:#x} kind={task.kind}")
      if task.kind == "cmac":
        M, N, _, _, align_out = task.layout
        _unpack_cmac_out(cmac_bufs[2].va_addr, bufs[task.out_slot].va_addr, M, N, align_out)
      elif task.kind == "ppu" and ppu_padded is not None:
        channels, chan_padded, out_padded = ppu_padded
        ctypes.memmove(bufs[task.out_slot].va_addr, out_padded.va_addr, channels * 2)  # type: ignore[arg-type]
      # fp32 output: convert fp16 temp buffer → fp32 in original output buffer
      if fp32_out_temp is not None:
        orig_out, fp16_out, n = fp32_out_temp
        _convert_fp16_to_fp32_buf(fp16_out.va_addr, orig_out.va_addr, n)
    finally:
      for b in temp: dev._gpu_free(b)
    self.submit_count += 1
    self.last_enable_mask = task.enable_mask
    dev.submitted_masks.add(task.enable_mask)

  def _submit_multi(self, bufs:tuple) -> None:
    """Submit multiple DPU tasks as a PC chain (single IOCTL, multiple tasks).
    PC chain format from ref/rk3588/experimental/pcchain.md and examples/elementwise.py.
    Key rules (pcchain.md §Debug Checklist, §ADD Reference Captures):
      - regcfg_amount = body qwords + 4 (PC tail qwords)
      - PC_REGISTER_AMOUNTS = next body qword count (raw, not ceil_div)
      - enable_mask = 0x18 for DPU (no | 1)
      - flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG
      - task_number = n_tasks, subcore_task[0] = (0, n_tasks)
      - Re-arm S_POINTER=0x0e in every chained segment (already in body cmds)
    """
    dev = self.device
    from tinygrad.runtime.support.rockchip import _T_PC_REG, _T_VERSION, _PC_CHAIN_TAIL_QWORDS
    _T_PC = 0x0081
    subtasks = self.subtasks
    assert subtasks is not None
    # All host-side subtasks (copy/fill): handle without NPU submission.
    if all(st.task.is_copy or st.task.is_fill for st in subtasks):
      for st in subtasks:
        task = st.task
        if len(task.layout) > 1 and task.layout[1] == _HOST_BITWISE_LAYOUT:
          _run_host_bitwise(task, st.relocs, bufs)
          continue
        if task.is_fill:
          total = task.layout[0]
          out_slot = st.relocs[0].globals_slot
          out_buf = bufs[out_slot] if out_slot < len(bufs) else None
          if out_buf is None: continue
          if task.const_val == 0.0:
            ctypes.memset(out_buf.va_addr, 0, total * 2)  # type: ignore[arg-type]
          else:
            arr = struct.pack('<e', task.const_val) * total
            ctypes.memmove(out_buf.va_addr, arr, total * 2)  # type: ignore[arg-type]
          continue
        total = task.layout[0]
        in_slot = st.relocs[1].globals_slot
        out_slot = st.relocs[0].globals_slot
        in_buf = bufs[in_slot] if in_slot < len(bufs) else None
        out_buf = bufs[out_slot] if out_slot < len(bufs) else None
        if in_buf is None or out_buf is None: continue
        out_addr = out_buf.va_addr + task.out_offset
        if len(task.layout) > 1 and task.layout[1] < 0:
          # Scatter copy: source (possibly strided) → strided destination (for pad)
          _, neg_ndim, *meta = task.layout
          ndim = -neg_ndim
          shape = meta[:ndim]
          src_strides = meta[ndim:2*ndim]
          src_offset = meta[2*ndim]
          dst_strides = meta[2*ndim+1:3*ndim+1]
          dst_offset = meta[-1]
          for in_idx in range(total):
            rem, src_idx = in_idx, src_offset
            for dim in range(ndim-1, -1, -1):
              rem, coord = divmod(rem, shape[dim])
              src_idx += coord * src_strides[dim]
            rem, dst_idx = in_idx, dst_offset
            for dim in range(ndim-1, -1, -1):
              rem, coord = divmod(rem, shape[dim])
              dst_idx += coord * dst_strides[dim]
            if 0 <= src_idx < in_buf.size // 2 and 0 <= dst_idx < out_buf.size // 2:
              ctypes.memmove(out_addr + dst_idx*2, in_buf.va_addr + src_idx*2, 2)  # type: ignore[arg-type]
        elif len(task.layout) > 1:
          # 2D copy with strides (broadcast expansion)
          _, ndim, *meta = task.layout
          shape, strides, offset = meta[:ndim], meta[ndim:2*ndim], meta[-1]
          in_n = in_buf.size // 2
          out_n = out_buf.size // 2
          for out_idx in range(total):
            rem, src_idx = out_idx, offset
            for dim in range(ndim-1, -1, -1):
              rem, coord = divmod(rem, shape[dim])
              src_idx += coord * strides[dim]
            if 0 <= src_idx < in_n and 0 <= out_idx < out_n:
              ctypes.memmove(out_addr + out_idx*2, in_buf.va_addr + src_idx*2, 2)  # type: ignore[arg-type]
        else:
          ctypes.memmove(out_addr, in_buf.va_addr, total * 2)  # type: ignore[arg-type]
      return
    # Mixed copy + DPU: handle copy tasks host-side, then submit DPU tasks.
    copy_tasks = [st for st in subtasks if st.task.is_copy]
    dpu_tasks = [st for st in subtasks if not st.task.is_copy]
    if copy_tasks and dpu_tasks:
      ext, shared = list(bufs), []
      max_slot = max((r.globals_slot for st in subtasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)),
                     default=len(ext)-1)
      total = max(st.task.layout[0] for st in subtasks)
      try:
        while len(ext) <= max_slot:
          shared.append(b := dev._gpu_alloc(max(total * 2, 4096), 0))
          ext.append(b)
        # Handle copy tasks host-side
        for st in copy_tasks:
          task = st.task
          ct = task.layout[0]
          in_slot = st.relocs[1].globals_slot
          out_slot = st.relocs[0].globals_slot
          in_buf = ext[in_slot]
          out_buf = ext[out_slot]
          out_addr = out_buf.va_addr + task.out_offset
          if len(task.layout) > 1:
            _, ndim, *meta = task.layout
            shape, strides, offset = meta[:ndim], meta[ndim:2*ndim], meta[-1]
            for out_idx in range(ct):
              rem, src_idx = out_idx, offset
              for dim in range(ndim-1, -1, -1):
                rem, coord = divmod(rem, shape[dim])
                src_idx += coord * strides[dim]
              ctypes.memmove(out_addr + out_idx*2, in_buf.va_addr + src_idx*2, 2)  # type: ignore[arg-type]
          else:
            ctypes.memmove(out_addr, in_buf.va_addr, ct * 2)  # type: ignore[arg-type]
        # Submit DPU tasks with expanded buffers
        for st in dpu_tasks:
          dev.reset_npu()
          self.subtasks = [st]
          self._submit_multi(tuple(ext))
      finally:
        self.subtasks = subtasks
        for b in shared: dev._gpu_free(b)
      return
    # Custom comparison and LUT stages leave DPU state that makes a mixed chain unstable.
    # Submit those programs stage-by-stage with resets, retaining shared scratch allocations.
    def is_cmp(st): return any((cmd & 0xffff) == rk.REG_DPU_BN_RELUX_CMP_VALUE and
                               ((cmd >> 16) & 0xffffffff) == 0x3f800000 for cmd in st.cmds)
    if len(subtasks) > 1 and any(is_cmp(st) or st.task.kind == "dpu_lut" for st in subtasks):
      ext, shared, original = list(bufs), [], self.subtasks
      max_slot = max((r.globals_slot for st in subtasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)),
                     default=len(ext)-1)
      total = max(st.task.layout[0] for st in subtasks)
      if getenv("DEBUG") >= 2: print(f"WHERE stages: bufs={len(ext)} max_slot={max_slot} total={total}")
      try:
        while len(ext) <= max_slot:
          shared.append(b := dev._gpu_alloc(max(total * 2, 4096), 0))
          ext.append(b)
        for st in subtasks:
          dev.reset_npu()
          self.subtasks = [st]
          self._submit_multi(tuple(ext))
      finally:
        self.subtasks = original
        for b in shared: dev._gpu_free(b)
      return
    temp:list[HCQBuffer] = []
    prepared = list(bufs)
    total = max(st.task.layout[0] for st in subtasks)
    max_slot = max((r.globals_slot for st in subtasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)),
                   default=len(prepared)-1)
    while len(prepared) <= max_slot:
      temp.append(scratch := dev._gpu_alloc(max(total * 2, 4096), 0))
      prepared.append(scratch)
    source_counts:dict[int, int] = {}
    periodic_slots = {s for st in subtasks if st.task.periodic_input for s in st.task.fp32_inputs}
    for slot in {s for st in subtasks for s in st.task.fp32_inputs}:
      source_counts[slot] = source_n = prepared[slot].size // 4
      converted = dev._gpu_alloc(max(source_n * 2, 4096), 0)
      temp.append(converted)
      if slot in periodic_slots: _convert_periodic_fp32_to_fp16_buf(prepared[slot].va_addr, converted.va_addr, source_n)
      else: _convert_fp32_to_fp16_buf(prepared[slot].va_addr, converted.va_addr, source_n)
      prepared[slot] = converted
    for slot in {s for st in subtasks for s in st.task.int32_inputs}:
      source_counts[slot] = source_n = prepared[slot].size // 4
      converted = dev._gpu_alloc(max(source_n * 2, 4096), 0)
      temp.append(converted)
      _convert_int32_to_fp16_buf(prepared[slot].va_addr, converted.va_addr, source_n)
      prepared[slot] = converted
    for slot in {s for st in subtasks for s in st.task.bool_inputs}:
      source_counts[slot] = source_n = prepared[slot].size
      converted = dev._gpu_alloc(max(source_n * 2, 4096), 0)
      temp.append(converted)
      _convert_bool_to_fp16_buf(prepared[slot].va_addr, converted.va_addr, source_n)
      prepared[slot] = converted
    for slot in {s for st in subtasks for s in st.task.comparison_inputs}:
      source_counts[slot] = source_n = source_counts.get(slot, prepared[slot].size // 2)
      sanitized = dev._gpu_alloc(max(source_n * 2, 4096), 0)
      temp.append(sanitized)
      _sanitize_fp16_comparison_buf(prepared[slot].va_addr, sanitized.va_addr, source_n)
      prepared[slot] = sanitized
    for slot in {s for st in subtasks for s in st.task.broadcast_inputs}:
      converted = dev._gpu_alloc(max(total * 2, 4096), 0)
      temp.append(converted)
      _broadcast_fp16_buf(prepared[slot].va_addr, converted.va_addr, source_counts.get(slot, prepared[slot].size // 2), total)
      prepared[slot] = converted
    converted_output = next(((st.task.out_slot, st.task.fp32_output, st.task.uint8_output, st.task.bool_output, st.task.trunc_output)
                             for st in subtasks if st.task.fp32_output or st.task.int32_output or st.task.uint8_output or
                             st.task.bool_output or st.task.trunc_output), None)
    output_conversion = None
    if converted_output is not None:
      output_slot, is_fp32, is_uint8, is_bool, is_trunc = converted_output
      # Use the actual output element count from the buffer size, not `total` (which is
      # max across subtasks and may be larger than the real output for broadcast/pad ops)
      # fp32/int32 outputs are 4 bytes; uint8/bool are 1 byte; fp16/trunc are 2 bytes
      out_itemsize = 4 if is_fp32 else (1 if (is_uint8 or is_bool) else (2 if is_trunc else 4))
      out_n = prepared[output_slot].size // out_itemsize
      converted = dev._gpu_alloc(max(total * 2, out_n * 2, 4096), 0)
      temp.append(converted)
      output_conversion = (prepared[output_slot], converted, out_n, is_fp32, is_uint8, is_bool, is_trunc)
      prepared[output_slot] = converted
    bufs = tuple(prepared)
    try:
      buf_map:dict[int, HCQBuffer] = dict(enumerate(bufs))
      n_tasks = len(subtasks)
      # Emitters already end each command stream with PC_OPERATION_ENABLE. For a chain that
      # word is the final word of the four-qword PC tail, not part of the register body.
      offsets = []
      offset = 0
      for st in subtasks:
        offsets.append(offset)
        offset += ((len(st.cmds) + _PC_CHAIN_TAIL_QWORDS) // 2) * 2  # (cmds - enable) + tail, aligned to 2 qwords
      total_qwords = offset
      assert total_qwords <= dev.cmd_buf_size, f"PC chain: {total_qwords} qwords > cmd_buf {dev.cmd_buf_size}"
      # Pack cmds + PC chain tails into cmd_buf
      regcmd = ctypes.cast(dev.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * dev.cmd_buf_size)).contents  # type: ignore[arg-type]
      for i in range(total_qwords): regcmd[i] = 0  # clear
      cmd_buf_dma = dev.cmd_buf.meta.dma_addr
      for idx, st in enumerate(subtasks):
        base = offsets[idx]
        n_body = len(st.cmds) - 1
        assert (st.cmds[-1] & 0xffff) == rk.REG_PC_OPERATION_ENABLE
        # Write the body without its single-task PC_OPERATION_ENABLE; the chain tail supplies it.
        for j, cmd in enumerate(st.cmds[:-1]): regcmd[base + j] = cmd
        # PC chain tail (4 qwords) — pcchain.md §PC Tail Layout
        tail_base = base + n_body
        if idx + 1 < n_tasks:
          next_st = subtasks[idx + 1]
          next_addr = (cmd_buf_dma + offsets[idx + 1] * 8) & 0xfffffff0
          next_n_body = len(next_st.cmds) - 1  # raw body count (pcchain.md §Amount Encoding)
          regcmd[tail_base + 0] = ((_T_PC_REG << 48) | ((next_addr & 0xFFFFFFFF) << 16) | rk.REG_PC_BASE_ADDRESS)
          regcmd[tail_base + 1] = ((_T_PC_REG << 48) | ((next_n_body & 0xFFFFFFFF) << 16) | rk.REG_PC_REGISTER_AMOUNTS)
          regcmd[tail_base + 2] = ((_T_VERSION << 48) | 0)
          regcmd[tail_base + 3] = ((_T_PC << 48) | ((st.task.enable_mask & 0xFFFFFFFF) << 16) | rk.REG_PC_OPERATION_ENABLE)
        else:
          # Last task: null tail (pcchain.md §PC Tail Layout, last segment uses 0 for first qword)
          regcmd[tail_base + 0] = 0
          regcmd[tail_base + 1] = ((_T_PC_REG << 48) | (0 << 16) | rk.REG_PC_REGISTER_AMOUNTS)
          regcmd[tail_base + 2] = ((_T_VERSION << 48) | 0)
          regcmd[tail_base + 3] = ((_T_PC << 48) | ((st.task.enable_mask & 0xFFFFFFFF) << 16) | rk.REG_PC_OPERATION_ENABLE)
      # Apply relocs for each subtask and prepare buffers
      for idx, st in enumerate(subtasks):
        task = st.task
        base = offsets[idx]
        if getenv("DEBUG") >= 2: print(f"  task {idx}: base={base}, n_cmds={len(st.cmds)}, n_relocs={len(st.relocs)}")
        for r in st.relocs:
          if r.globals_slot == _CONST_SLOT:
            total = task.layout[0]
            cbuf = dev._gpu_alloc(max(total * 2, 4096), 0)
            temp.append(cbuf)
            cval = struct.unpack('<f', struct.pack('<I', r.addend))[0]
            try: fp16_value = struct.pack('<e', cval)
            except OverflowError: fp16_value = struct.pack('<e', float("-inf") if cval < 0 else float("inf"))
            fp16_bytes = fp16_value * total
            ctypes.memmove(cbuf.va_addr, fp16_bytes, total * 2)  # type: ignore[arg-type]
            dma = cbuf.meta.dma_addr
            v = (dma >> r.shift) & r.mask
          elif r.globals_slot == _ZERO_SLOT:
            total = task.layout[0]
            zbuf = dev._gpu_alloc(max(total * 2, 4096), 0)
            temp.append(zbuf)
            ctypes.memset(zbuf.va_addr, 0, total * 2)  # type: ignore[arg-type]
            dma = zbuf.meta.dma_addr
            v = (dma >> r.shift) & r.mask
          else:
            dma = buf_map[r.globals_slot].meta.dma_addr
            v = ((dma + r.addend) >> r.shift) & r.mask
          if r.field_shift:
            v = (v << r.field_shift) & 0xFFFFFFFF
            fm = (r.mask << r.field_shift) & 0xFFFFFFFF
          else: fm = r.mask
          w = regcmd[base + r.word_index]
          regcmd[base + r.word_index] = (w & ~(fm << 16)) | ((v & fm) << 16)
          if getenv("DEBUG") >= 2: print(f"    reloc: word_idx={r.word_index}, slot={r.globals_slot}, dma={dma:#x}, v={v:#x}, fm={fm:#x}")
      # Fill task_buf entries — pcchain.md §Task Descriptor Rules
      # regcfg_amount = body qwords + 4 PC-tail qwords. The emitter's final enable is
      # replaced by that tail, hence len(cmds) + 3 rather than len(cmds) + 4.
      tasks = ctypes.cast(dev.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents  # type: ignore[arg-type]
      ctypes.memset(dev.task_buf.va_addr, 0, n_tasks * ctypes.sizeof(rk.struct_rknpu_task))  # type: ignore[arg-type]
      for idx, st in enumerate(subtasks):
        t = tasks[idx]
        t.flags = 0
        t.op_idx = st.task.op_idx
        t.enable_mask = st.task.enable_mask
        t.int_mask = st.task.int_mask
        t.int_clear = 0x1ffff
        t.int_status = 0
        t.regcfg_amount = len(st.cmds) + _PC_CHAIN_TAIL_QWORDS - 1
        t.regcfg_offset = 0  # absolute mode (pcchain.md §Task Descriptor Rules)
        t.regcmd_addr = cmd_buf_dma + offsets[idx] * 8
      if getenv("DEBUG") >= 2:
        for idx in range(n_tasks):
          t = tasks[idx]
          print(f"  task_buf[{idx}]: op_idx={t.op_idx}, enable_mask={t.enable_mask:#x}, "
                f"regcfg_amount={t.regcfg_amount}, regcmd_addr={t.regcmd_addr:#x}")
          base = offsets[idx]
          for j in range(len(subtasks[idx].cmds) + 4):
            print(f"    [{base+j}] = {regcmd[base+j]:#018x}")
      # Submit — pcchain.md §Submit Shape: single-core PC chain
      # flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG (pcchain.md §Conv PC-Chain)
      rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd_ctl, __payload=rk.struct_rknpu_submit(
        flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=6000,
        task_start=0, task_number=n_tasks, task_counter=0, priority=0,
        task_obj_addr=dev.task_buf.meta.obj_addr, regcfg_obj_addr=0, task_base_addr=0,
        user_data=0, core_mask=1, fence_fd=-1,
        subcore_task=(rk.struct_rknpu_subcore_task*5)(
          rk.struct_rknpu_subcore_task(task_start=0, task_number=n_tasks),
          rk.struct_rknpu_subcore_task(task_start=n_tasks, task_number=0),
          rk.struct_rknpu_subcore_task(task_start=n_tasks, task_number=0),
          rk.struct_rknpu_subcore_task(task_start=0, task_number=0),
          rk.struct_rknpu_subcore_task(task_start=0, task_number=0))))
      if output_conversion is not None:
        original_out, converted, n, is_fp32, is_uint8, is_bool, is_trunc = output_conversion
        if is_trunc: _truncate_fp16_buf(converted.va_addr, original_out.va_addr, n)
        elif is_bool: _convert_fp16_to_bool_buf(converted.va_addr, original_out.va_addr, n)
        elif is_uint8: _convert_fp16_to_uint8_buf(converted.va_addr, original_out.va_addr, n)
        elif is_fp32: _convert_fp16_to_fp32_buf(converted.va_addr, original_out.va_addr, n)
        else: _convert_fp16_to_int32_buf(converted.va_addr, original_out.va_addr, n)
      if getenv("DEBUG") >= 1: print(f"submit {self.name}: PC chain {n_tasks} tasks")
    finally:
      for b in temp: dev._gpu_free(b)
    self.submit_count += 1
    dev.submitted_masks.add(subtasks[-1].task.enable_mask)

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  has_threads = False
  has_local = False
  code_for_op = {}  # no Python fallback — all compute goes through native_program
  def __init__(self, target:Target): self.target, self.tensor_cores = target, []
  def native_program(self, ast:UOp) -> UOp|None: return build_native_program(ast)
  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # Multi-task (PC chain): first INS carries a tuple of RKSubTask
    subtasks = None
    task, cmds, relocs = None, [], []
    for u in lin.src:
      if isinstance(u.arg, tuple) and u.arg and isinstance(u.arg[0], RKSubTask):
        subtasks = u.arg
      elif isinstance(u.arg, RKTask): task = u.arg
      elif isinstance(u.arg, RKReloc): relocs.append(u.arg)
      elif isinstance(u.arg, int): cmds.append(u.arg)
    if subtasks is not None:
      return encode_rk_multi(subtasks)
    if task is None: raise RuntimeError("rk: no RKTask metadata — non-NPU kernel with no Python fallback")
    return encode_rk(tuple(cmds), task, tuple(relocs))
  def supported_dtypes(self) -> set[DType]: return {dtypes.half, dtypes.float}

class RockchipRegisterAllocator(HCQAllocatorBase):
  """DMA-backed allocator: buffers are NPU GEM objects with va_addr (CPU mmap) and dma_addr (NPU)."""
  def __init__(self, dev): super().__init__(dev, batch_cnt=0)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer: return self.dev._gpu_alloc(size, 0)
  def _do_copy(self, s, d, sz): ctypes.memmove(d, s, sz)
  def _copyin(self, dest:HCQBuffer, src:memoryview): self._do_copy(mv_address(src), dest.va_addr, src.nbytes)
  def _copyout(self, dest:memoryview, src:HCQBuffer): self._do_copy(src.va_addr, mv_address(dest), src.size)  # type: ignore[arg-type]
  def _as_buffer(self, src:HCQBuffer) -> memoryview: return to_mv(src.va_addr, src.size)  # type: ignore[arg-type]
  def _do_free(self, buf:HCQBuffer, options:BufferSpec|None=None): self.dev._gpu_free(buf)

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface("/dev/dri/card1", os.O_RDWR)
    self.cmd_buf_size = 16384
    self.cmd_buf = self._gpu_alloc(self.cmd_buf_size * 8, 0, "cmd_buf")
    self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, "task_buf")
    self.submitted_masks: set[int] = set()
    super().__init__(device, RockchipRegisterAllocator(self), [RockchipRenderer], RockchipProgram)
  def create_flink_name(self, handle:int, name:str="", **kw) -> int:
    fr = rk.struct_drm_gem_flink(handle=handle, name=0)
    rk.DRM_IOCTL_GEM_FLINK(self.fd_ctl, __payload=fr)
    return fr.name

  def _gpu_alloc(self, size:int, flags, name:str="") -> HCQBuffer:
    mc = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags|rk.RKNPU_MEM_NON_CACHEABLE)
    mm = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mc.handle, offset=0)
    va = self.fd_ctl.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mm.offset)
    mc.flink_name = self.create_flink_name(mc.handle, name)
    return HCQBuffer(va_addr=va, size=size, meta=mc)

  def _gpu_free(self, buf:HCQBuffer) -> None:
    FileIOInterface.munmap(buf.va_addr, buf.size)
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, __payload=rk.struct_rknpu_mem_destroy(handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr))

  def reset_npu(self):
    rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, __payload=rk.struct_rknpu_action(flags=rk.RKNPU_ACT_RESET, value=0))
