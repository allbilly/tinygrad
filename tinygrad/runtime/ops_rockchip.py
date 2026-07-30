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
  _CONST_SLOT, _ZERO_SLOT, _HOST_BITWISE_LAYOUT, _HOST_MOVEMENT_LAYOUT, _HOST_TRUNC_LAYOUT, _HOST_COPYSIGN_LAYOUT,
  _HOST_ELEMENTWISE_LAYOUT, _HOST_STATIC_HALF_LAYOUT, _HOST_SCATTER_LAYOUT, _HOST_ARGMAX_LAYOUT, _HOST_GATHER_MAP_LAYOUT,
  _HOST_PACK_CHUNK_LAYOUT, _HOST_UNPACK_INT_CHUNK_LAYOUT, _HOST_PACK_INT32_CHUNK_LAYOUT, _HOST_UNPACK_HALF_CHUNK_LAYOUT,
  _HOST_STATIC_INT_LAYOUT, _HOST_PLANE_GATHER_LAYOUT, _HOST_COMPACT_NATIVE_HALF_LAYOUT, _HOST_ASSEMBLE_INT_BYTES_LAYOUT,
  _HOST_PACK_HALF_BITS_LAYOUT, _HOST_UNPACK_HALF_BITS_LAYOUT, _HOST_BOOL_HALF_LAYOUT,
  _HOST_STATIC_SELECT_HALF_LAYOUT, _HOST_STATIC_SELECT_INT_LAYOUT, _HOST_HALF_INT_LAYOUT,
  _HOST_FP32_HALF_LAYOUT, _HOST_FP32_RESIDUAL_LAYOUT,
  _CMAC_MATERIALIZED_LAYOUT)

_ROCKCHIP_MAX_MAPPABLE_BO = 2 * 1024 * 1024

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

def _fp32_to_fp16_group(b, a_addr, b_addr, m, n, K, align_in):
  """Round a staged CMAC dot, treating a one-fp32-ULP tree-reduction drift around
  an fp16 halfway point with the sequential-GEMM order used by PyTorch CPU."""
  exponent = ((b >> 23) & 0xff) - 127 + 15
  discarded = (b & 0x7fffff) & 0x1fff
  if 1 <= exponent <= 30 and discarded == 0x1001:
    a = ctypes.cast(a_addr, ctypes.POINTER(ctypes.c_uint16))
    packed_b = ctypes.cast(b_addr, ctypes.POINTER(ctypes.c_uint16))
    acc = 0.0
    for k in range(K):
      av = struct.unpack('<e', struct.pack('<H', a[m*align_in+k]))[0]
      b_index = (((n//16)*(align_in//32)+(k//32))*16+(n%16))*32+(k%32)
      bv = struct.unpack('<e', struct.pack('<H', packed_b[b_index]))[0]
      product = struct.unpack('<f', struct.pack('<f', av*bv))[0]
      acc = struct.unpack('<f', struct.pack('<f', acc+product))[0]
    return _fp32_to_fp16(struct.unpack('<I', struct.pack('<f', acc))[0])
  return _fp32_to_fp16(b)

def _convert_fp32_to_fp16_buf(src, dst, n):
  """Convert n fp32 elements at src to fp16 at dst (buffer-level cast, not NPU compute)."""
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 4), dtype=np.float32).astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _convert_fp32_residual_to_fp16_buf(src, dst, n):
  """Encode 256 times the fp32 remainder after its nearest fp16 value."""
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 4), dtype=np.float32)
  high = arr.astype(np.float16).astype(np.float32)
  residual = ((arr-high)*256.0).astype(np.float16)
  ctypes.memmove(dst, residual.ctypes.data, n * 2)  # type: ignore[arg-type]

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

def _run_host_bool_half(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  total, tag = task.layout
  assert tag == _HOST_BOOL_HALF_LAYOUT and len(relocs) == 2
  _convert_bool_to_fp16_buf(bufs[relocs[1].globals_slot].va_addr, bufs[relocs[0].globals_slot].va_addr, total)

def _run_host_static_select_half(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Interleave fp16 representations using a compile-time sort-wire mask."""
  total, tag, *choose_first = task.layout
  assert tag == _HOST_STATIC_SELECT_HALF_LAYOUT and len(choose_first) == total and len(relocs) == 3
  output, first, second = (bufs[r.globals_slot] for r in relocs)
  for element, choose in enumerate(choose_first):
    source = first if choose else second
    ctypes.memmove(output.va_addr + element*2, source.va_addr + element*2, 2)  # type: ignore[arg-type]

def _run_host_static_select_int(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Interleave native four-lane int32 results using compile-time sort wires."""
  count, tag, start, *choose_first = task.layout
  assert tag == _HOST_STATIC_SELECT_INT_LAYOUT and len(choose_first) == count and len(relocs) == 3
  output, first, second = (bufs[r.globals_slot] for r in relocs)
  for element, choose in enumerate(choose_first):
    source = first if choose else second
    ctypes.memmove(output.va_addr + (start+element)*4, source.va_addr + element*4, 4)  # type: ignore[arg-type]

def _run_host_half_int(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Apply the established typed fp16-to-int32 ABI conversion after NPU selection."""
  total, tag = task.layout
  assert tag == _HOST_HALF_INT_LAYOUT and len(relocs) == 2
  _convert_fp16_to_int32_buf(bufs[relocs[1].globals_slot].va_addr, bufs[relocs[0].globals_slot].va_addr, total)

def _broadcast_fp16_buf(src, dst, src_n, n):
  data = ctypes.string_at(src, src_n * 2)
  ctypes.memmove(dst, (data * ((n + src_n - 1) // src_n))[:n*2], n * 2)

def _convert_fp16_to_int32_buf(src, dst, n):
  import numpy as np
  with np.errstate(invalid="ignore"):
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
  with np.errstate(over="ignore"):
    arr = np.frombuffer(ctypes.string_at(src, n * 4), dtype=np.int32).astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _sanitize_fp16_comparison_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).copy()
  np.nan_to_num(arr, copy=False, nan=float("nan"), posinf=65504.0, neginf=-65504.0)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)  # type: ignore[arg-type]

def _unpack_cmac_out(src, dst, M, N, align_out, bias=None, bias_axis=-1, relu=False, fp32_output=False):
  s = ctypes.cast(src, ctypes.POINTER(ctypes.c_uint32))
  d = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint32 if fp32_output else ctypes.c_uint16))
  b = ctypes.cast(bias, ctypes.POINTER(ctypes.c_uint16)) if bias is not None else None
  for i in range(M * N):
    raw = s[(i // N) * align_out + i % N]
    if b is not None:
      bias_index = i // N if bias_axis == 0 else i % N
      value = struct.unpack('<f', struct.pack('<I', raw))[0] + struct.unpack('<e', struct.pack('<H', b[bias_index]))[0]
      value = struct.unpack('<f', struct.pack('<f', value))[0]
      if relu: value = value if value > 0.0 else 0.0
      raw = struct.unpack('<I', struct.pack('<f', value))[0]
    d[i] = raw if fp32_output else _fp32_to_fp16(raw)

def _decode_materialized_cmac_layout(layout):
  _, _, _, _, _, tag, tile_m, bias_slot, bias_axis, relu, scale_bits, n_scale_counts, *tail = layout
  assert tag == _CMAC_MATERIALIZED_LAYOUT
  scale_counts, (tile_n, tile_k, *meta) = tail[:n_scale_counts], tail[n_scale_counts:]
  cursor = 0
  n_loops = meta[cursor]
  loop_extents = meta[cursor+1:cursor+1+n_loops]
  cursor += n_loops+1
  n_reductions = meta[cursor]
  reduce_extents = meta[cursor+1:cursor+1+n_reductions]
  cursor += n_reductions+1
  n_m_axes = meta[cursor]
  m_axes = meta[cursor+1:cursor+1+n_m_axes]
  cursor += n_m_axes+1
  n_n_axes = meta[cursor]
  n_axes = meta[cursor+1:cursor+1+n_n_axes]
  cursor += n_n_axes+1
  n_shared_axes = meta[cursor]
  shared_axes = meta[cursor+1:cursor+1+n_shared_axes]
  cursor += n_shared_axes+1
  n_reduce_order = meta[cursor]
  cursor += n_reduce_order+1
  codes = []
  for _ in range(3):
    code_n = meta[cursor]
    codes.append(meta[cursor+1:cursor+1+code_n])
    cursor += code_n+1
  n_fixed = meta[cursor]
  fixed_reductions = tuple((meta[cursor+1+2*i], meta[cursor+2+2*i]) for i in range(n_fixed))
  cursor += 1+2*n_fixed
  n_active = meta[cursor]
  active_reduce_order = meta[cursor+1:cursor+1+n_active]
  cursor += 1+n_active
  n_rounding = meta[cursor]
  rounding_axes = meta[cursor+1:cursor+1+n_rounding]
  cursor += 1+n_rounding
  n_fixed_loops = meta[cursor]
  fixed_loops = tuple((meta[cursor+1+2*i], meta[cursor+2+2*i]) for i in range(n_fixed_loops))
  scale = struct.unpack('<f', struct.pack('<I', scale_bits))[0]
  return (tile_m, tile_n, tile_k, bias_slot, bias_axis, bool(relu), scale, scale_counts, loop_extents, reduce_extents, m_axes,
          n_axes, shared_axes, active_reduce_order, fixed_reductions, rounding_axes, fixed_loops, *codes)

def _eval_static_index(code, coords):
  stack:list[int] = []
  for pos in range(0, len(code), 2):
    op, arg = code[pos], code[pos+1]
    if op == 0: stack.append(arg)
    elif op == 1: stack.append(coords[arg])
    elif op == 11:
      false, true, cond = stack.pop(), stack.pop(), stack.pop()
      stack.append(true if cond else false)
    else:
      rhs, lhs = stack.pop(), stack.pop()
      if op == 2: stack.append(lhs + rhs)
      elif op == 3: stack.append(lhs * rhs)
      elif op == 4: stack.append(lhs // rhs)
      elif op == 5: stack.append(lhs % rhs)
      elif op == 6: stack.append(int(lhs < rhs))
      elif op == 7: stack.append(int(lhs != rhs))
      elif op == 8: stack.append(int(bool(lhs) and bool(rhs)))
      elif op == 9: stack.append(int(bool(lhs) or bool(rhs)))
      else: raise RuntimeError(f"rk: invalid materialized CMAC index opcode {op}")
  assert len(stack) == 1
  return stack[0]

def _eval_static_value(code, coords, source):
  stack:list[int] = []
  for pos in range(0, len(code), 2):
    op, arg = code[pos], code[pos+1]
    if op == 0: stack.append(arg)
    elif op == 1: stack.append(coords[arg])
    elif op == 10:
      if source is None: raise RuntimeError("rk: materialized CMAC load has no source")
      stack.append(source[stack.pop()])
    elif op == 12: stack.append(arg)
    elif op == 11:
      false, true, cond = stack.pop(), stack.pop(), stack.pop()
      stack.append(true if cond else false)
    else:
      rhs, lhs = stack.pop(), stack.pop()
      if op == 2: stack.append(lhs + rhs)
      elif op == 3: stack.append(lhs * rhs)
      elif op == 4: stack.append(lhs // rhs)
      elif op == 5: stack.append(lhs % rhs)
      elif op == 6: stack.append(int(lhs < rhs))
      elif op == 7: stack.append(int(lhs != rhs))
      elif op == 8: stack.append(int(bool(lhs) and bool(rhs)))
      elif op == 9: stack.append(int(bool(lhs) or bool(rhs)))
      else: raise RuntimeError(f"rk: invalid materialized CMAC value opcode {op}")
  assert len(stack) == 1
  return stack[0]

def _set_linear_axes(coords, linear, axes, extents):
  for axis in reversed(axes): linear, coords[axis] = divmod(linear, extents[axis])

def _materialize_cmac_inputs(a_src, b_src, a_dst, b_dst, M, N, K, align_in, align_out, decoded,
                             m_start=0, rows=None, n_start=0, cols=None, k_start=0, k_count=None):
  _, _, _, _, _, _, _, _, loop_extents, reduce_extents, m_axes, n_axes, shared_axes, reduce_order, fixed_reductions, \
    _, fixed_loops, _, a_code, b_code = decoded
  rows = M if rows is None else rows
  cols = N if cols is None else cols
  k_count = K if k_count is None else k_count
  ctypes.memset(a_dst, 0, rows * align_in * 2)
  ctypes.memset(b_dst, 0, align_out * align_in * 2)
  a_in = ctypes.cast(a_src, ctypes.POINTER(ctypes.c_uint16))
  b_in = ctypes.cast(b_src, ctypes.POINTER(ctypes.c_uint16)) if b_src is not None else None
  a_out, b_out = ctypes.cast(a_dst, ctypes.POINTER(ctypes.c_uint16)), ctypes.cast(b_dst, ctypes.POINTER(ctypes.c_uint16))
  n_loops = len(loop_extents)
  all_extents = (*loop_extents, *reduce_extents)
  reduce_coord_axes = tuple(n_loops+axis for axis in reduce_order)
  base_k = 1
  for axis in reduce_order: base_k *= reduce_extents[axis]

  def batch_index(coords):
    ret = 0
    for axis in shared_axes: ret = ret*loop_extents[axis] + coords[axis]
    return ret

  for local_m in range(rows):
    m = m_start + local_m
    if m >= M: continue
    coords = [0] * (n_loops + len(reduce_extents))
    for axis, value in fixed_loops: coords[axis] = value
    for axis, value in fixed_reductions: coords[n_loops+axis] = value
    _set_linear_axes(coords, m, m_axes, loop_extents)
    for local_k in range(k_count):
      k = k_start + local_k
      if k // base_k != batch_index(coords): continue
      _set_linear_axes(coords, k % base_k, reduce_coord_axes, all_extents)
      a_out[local_m*align_in+local_k] = _eval_static_value(a_code, coords, a_in)
  for local_n in range(cols):
    n = n_start + local_n
    if n >= N: continue
    coords = [0] * (n_loops + len(reduce_extents))
    for axis, value in fixed_loops: coords[axis] = value
    for axis, value in fixed_reductions: coords[n_loops+axis] = value
    _set_linear_axes(coords, n, n_axes, loop_extents)
    for local_k in range(k_count):
      k = k_start + local_k
      if k // base_k != batch_index(coords): continue
      _set_linear_axes(coords, k % base_k, reduce_coord_axes, all_extents)
      dst_index = (((local_n//16)*(align_in//32)+(local_k//32))*16+(local_n%16))*32+(local_k%32)
      b_out[dst_index] = _eval_static_value(b_code, coords, b_in)

def _unpack_materialized_cmac_out(src, dst, M, N, align_out, decoded, bias=None, m_start=0, rows=None,
                                  n_start=0, cols=None, packed_inputs=None):
  _, _, _, bias_slot, bias_axis, relu, scale, scale_counts, loop_extents, _, m_axes, n_axes, _, _, fixed_reductions, \
    _, fixed_loops, out_code, _, _ = decoded
  s, d = ctypes.cast(src, ctypes.POINTER(ctypes.c_uint32)), ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint16))
  b = ctypes.cast(bias, ctypes.POINTER(ctypes.c_uint16)) if bias_slot >= 0 and bias is not None else None
  total = 1
  for extent in loop_extents: total *= extent
  for linear in range(total):
    rem, coords = linear, [0] * len(loop_extents)
    for axis in range(len(loop_extents)-1, -1, -1): rem, coords[axis] = divmod(rem, loop_extents[axis])
    if any(coords[axis] != value for axis, value in fixed_loops): continue
    m = n = 0
    for axis in m_axes: m = m*loop_extents[axis] + coords[axis]
    for axis in n_axes: n = n*loop_extents[axis] + coords[axis]
    if m < m_start or (rows is not None and m >= m_start+rows): continue
    if n < n_start or (cols is not None and n >= n_start+cols): continue
    raw = s[(m-m_start)*align_out+n-n_start]
    out_index = _eval_static_index(out_code, coords)
    output_scale = 1.0 / scale_counts[out_index] if scale_counts else scale
    if output_scale != 1.0:
      value = struct.unpack('<f', struct.pack('<f', struct.unpack('<f', struct.pack('<I', raw))[0] * output_scale))[0]
      raw = struct.unpack('<I', struct.pack('<f', value))[0]
    if b is not None:
      value = struct.unpack('<f', struct.pack('<I', raw))[0] + struct.unpack('<e', struct.pack('<H', b[coords[bias_axis]]))[0]
      value = struct.unpack('<f', struct.pack('<f', value))[0]
      if relu: value = value if value > 0.0 else 0.0
      raw = struct.unpack('<I', struct.pack('<f', value))[0]
    if fixed_reductions and packed_inputs is not None:
      a_addr, b_addr, K, align_in = packed_inputs
      converted = _fp32_to_fp16_group(raw, a_addr, b_addr, m-m_start, n-n_start, K, align_in)
    else: converted = _fp32_to_fp16(raw)
    d[out_index] = converted

def _run_tiled_materialized_cmac(dev, name, cmds, task, relocs, bufs):
  M, N, K, align_in, align_out = task.layout[:5]
  decoded = _decode_materialized_cmac_layout(task.layout)
  tile_m, tile_n, tile_k, bias_slot = decoded[:4]
  a_s, b_s = relocs[0].globals_slot, relocs[1].globals_slot
  temp:list[HCQBuffer] = []
  try:
    a_buf = dev._gpu_alloc(max(tile_m*align_in*2, 4096), 0)
    b_buf = dev._gpu_alloc(max(align_out*align_in*2, 4096), 0)
    o_buf = dev._gpu_alloc(max(tile_m*align_out*4, 4096), 0)
    temp.extend((a_buf, b_buf, o_buf))
    n_cmds = len(cmds)
    regcmd = ctypes.cast(dev.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * dev.cmd_buf_size)).contents  # type: ignore[arg-type]
    for n_start in range(0, N, tile_n):
      for m_start in range(0, M, tile_m):
        rows, cols = min(tile_m, M-m_start), min(tile_n, N-n_start)
        accumulator:list[float]|None = None
        for k_start in range(0, K, tile_k):
          k_count = min(tile_k, K-k_start)
          dev.reset_npu()
          b_src = None if b_s == _CONST_SLOT else bufs[b_s].va_addr
          _materialize_cmac_inputs(bufs[a_s].va_addr, b_src, a_buf.va_addr, b_buf.va_addr,
                                   M, N, K, align_in, align_out, decoded, m_start, rows, n_start, cols, k_start, k_count)
          for i, cmd in enumerate(cmds): regcmd[i] = cmd
          for i, r in enumerate(relocs):
            dma = (a_buf, b_buf, o_buf)[i].meta.dma_addr
            v = ((dma + r.addend) >> r.shift) & r.mask
            fm = (r.mask << r.field_shift) & 0xFFFFFFFF if r.field_shift else r.mask
            if r.field_shift: v = (v << r.field_shift) & 0xFFFFFFFF
            regcmd[r.word_index] = (regcmd[r.word_index] & ~(fm << 16)) | ((v & fm) << 16)
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
          raw = ctypes.cast(o_buf.va_addr, ctypes.POINTER(ctypes.c_float))
          partial = [float(raw[row*align_out+col]) for row in range(rows) for col in range(cols)]
          if accumulator is None: accumulator = partial
          else:
            accumulator = [struct.unpack('<f', struct.pack('<f', x+y))[0] for x, y in zip(accumulator, partial)]
          if getenv("DEBUG") >= 1:
            print(f"submit {name}: materialized CMAC rows {m_start}:{m_start+rows} cols {n_start}:{n_start+cols} "
                  f"K {k_start}:{k_start+k_count}")
        assert accumulator is not None
        raw = ctypes.cast(o_buf.va_addr, ctypes.POINTER(ctypes.c_float))
        for row in range(rows):
          for col in range(cols): raw[row*align_out+col] = accumulator[row*cols+col]
        bias = bufs[bias_slot].va_addr if bias_slot >= 0 else None
        _unpack_materialized_cmac_out(o_buf.va_addr, bufs[task.out_slot].va_addr, M, N, align_out,
                                      decoded, bias, m_start, rows, n_start, cols,
                                      (a_buf.va_addr, b_buf.va_addr, K, task.layout[3]) if tile_k >= K else None)
  finally:
    for b in temp: dev._gpu_free(b)

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

def _run_host_trunc(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Execute exact fp16/fp32 truncation on the original mapped buffers."""
  total, tag, dtype_code = task.layout
  assert tag == _HOST_TRUNC_LAYOUT and len(relocs) == 2
  import numpy as np
  dtype = np.float16 if dtype_code == 0 else np.float32
  itemsize = 2 if dtype_code == 0 else 4
  source = bufs[relocs[1].globals_slot]
  output = bufs[relocs[0].globals_slot]
  result = np.trunc(np.frombuffer(ctypes.string_at(source.va_addr, total * itemsize), dtype=dtype))
  ctypes.memmove(output.va_addr, result.ctypes.data, total * itemsize)  # type: ignore[arg-type]

def _run_host_copysign(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Copy a broadcast sign bit onto the magnitude operand without floating-point arithmetic."""
  total, tag, dtype_code, n_ranges, *meta = task.layout
  assert tag == _HOST_COPYSIGN_LAYOUT and len(relocs) == 3
  extents, cursor = meta[:n_ranges], n_ranges
  codes = []
  for _ in range(3):
    code_n = meta[cursor]
    codes.append(meta[cursor+1:cursor+1+code_n])
    cursor += code_n+1

  def evaluate(code, coords):
    stack:list[int] = []
    for pos in range(0, len(code), 2):
      op, arg = code[pos], code[pos+1]
      if op == 0: stack.append(arg)
      elif op == 1: stack.append(coords[arg])
      elif op == 11:
        false, true, cond = stack.pop(), stack.pop(), stack.pop()
        stack.append(true if cond else false)
      else:
        rhs, lhs = stack.pop(), stack.pop()
        if op == 2: stack.append(lhs + rhs)
        elif op == 3: stack.append(lhs * rhs)
        elif op == 4: stack.append(lhs // rhs)
        elif op == 5: stack.append(lhs % rhs)
        elif op == 6: stack.append(int(lhs < rhs))
        elif op == 7: stack.append(int(lhs != rhs))
        elif op == 8: stack.append(int(bool(lhs) and bool(rhs)))
        elif op == 9: stack.append(int(bool(lhs) or bool(rhs)))
        else: raise RuntimeError(f"rk: invalid host copysign index opcode {op}")
    assert len(stack) == 1
    return stack[0]

  itemsize, ctype, sign_mask = (2, ctypes.c_uint16, 0x8000) if dtype_code == 0 else (4, ctypes.c_uint32, 0x80000000)
  output, magnitude, sign = (bufs[r.globals_slot] for r in relocs)
  out_ptr, magnitude_ptr, sign_ptr = (ctypes.cast(buf.va_addr, ctypes.POINTER(ctype)) for buf in (output, magnitude, sign))
  for linear in range(total):
    rem, coords = linear, [0] * n_ranges
    for axis in range(n_ranges-1, -1, -1): rem, coords[axis] = divmod(rem, extents[axis])
    out_index, magnitude_index, sign_index = (evaluate(code, coords) for code in codes)
    if not 0 <= out_index < output.size // itemsize: raise RuntimeError(f"rk: host copysign output index out of bounds {out_index}")
    if not 0 <= magnitude_index < magnitude.size // itemsize: raise RuntimeError(f"rk: host copysign magnitude index out of bounds {magnitude_index}")
    if not 0 <= sign_index < sign.size // itemsize: raise RuntimeError(f"rk: host copysign sign index out of bounds {sign_index}")
    out_ptr[out_index] = (magnitude_ptr[magnitude_index] & ~sign_mask) | (sign_ptr[sign_index] & sign_mask)

def _run_host_elementwise(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Evaluate a serialized fused elementwise graph on original typed mapped buffers."""
  import numpy as np
  total, tag, out_dtype_code, n_ranges, *meta = task.layout
  assert tag == _HOST_ELEMENTWISE_LAYOUT
  extents, cursor = meta[:n_ranges], n_ranges
  out_n = meta[cursor]
  out_code = meta[cursor+1:cursor+1+out_n]
  cursor += out_n+1
  value_n = meta[cursor]
  value_code = meta[cursor+1:cursor+1+value_n]
  np_dtypes = (np.bool_, np.int32, np.uint32, np.int64, np.uint64, np.uint8,
               np.float16, np.float32, np.int64, np.int16, np.uint16, np.int8, np.float64)
  inputs:list[dict] = []
  for reloc in relocs[1:]:
    buf = bufs[reloc.globals_slot]
    inputs.append({code:np.frombuffer(ctypes.string_at(buf.va_addr, buf.size), dtype=dtype)
                   for code, dtype in enumerate(np_dtypes) if buf.size % np.dtype(dtype).itemsize == 0})
  output = bufs[relocs[0].globals_slot]
  out_dtype = np_dtypes[out_dtype_code]
  result = np.zeros(output.size // np.dtype(out_dtype).itemsize, dtype=out_dtype)

  def cast(value, dtype_code):
    with np.errstate(all="ignore"):
      return np.asarray(value, dtype=np_dtypes[dtype_code]).item()

  def evaluate(code, coords):
    stack:list = []
    value:object
    with np.errstate(all="ignore"):
      for pos in range(0, len(code), 4):
        op, dtype_code, arg0, arg1 = code[pos:pos+4]
        if op == 0:
          bits = (arg0 & 0xFFFFFFFF) | ((arg1 & 0xFFFFFFFF) << 32)
          if dtype_code == 0: value = bool(bits)
          elif dtype_code == 6: value = struct.unpack('<e', struct.pack('<H', bits & 0xFFFF))[0]
          elif dtype_code == 7: value = struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]
          elif dtype_code == 12: value = struct.unpack('<d', struct.pack('<Q', bits))[0]
          else:
            width = np.dtype(np_dtypes[dtype_code]).itemsize * 8
            bits &= (1 << width)-1
            if np.issubdtype(np_dtypes[dtype_code], np.signedinteger) and bits & (1 << (width-1)): bits -= 1 << width
            value = bits
          stack.append(cast(value, dtype_code))
          continue
        if op == 1:
          stack.append(cast(coords[arg0], dtype_code))
          continue
        if op == 2:
          index = int(stack.pop())
          source = inputs[arg0][dtype_code]
          stack.append(source[index].item() if 0 <= index < source.size else cast(0, dtype_code))
          continue
        args = stack[-arg0:] if arg0 else []
        if arg0: del stack[-arg0:]
        if op == 3: value = args[0] + args[1]
        elif op == 4: value = args[0] * args[1]
        elif op == 5: value = np.divide(args[0], args[1])
        elif op == 6: value = np.divide(1.0, args[0])
        elif op == 7: value = np.maximum(args[0], args[1])
        elif op == 8: value = args[0] < args[1]
        elif op == 9: value = args[0] != args[1]
        elif op == 10: value = args[1] if args[0] else args[2]
        elif op == 11: value = args[0] & args[1]
        elif op == 12: value = args[0] | args[1]
        elif op == 13: value = args[0] ^ args[1]
        elif op == 14: value = args[0]
        elif op == 15: value = np.trunc(args[0])
        elif op == 16: value = np.sqrt(args[0])
        elif op == 17: value = np.exp2(args[0])
        elif op == 18: value = np.log2(args[0])
        elif op == 19: value = np.sin(args[0])
        elif op == 20:
          quotient = abs(int(args[0])) // abs(int(args[1]))
          quotient = -quotient if (int(args[0]) < 0) != (int(args[1]) < 0) else quotient
          value = int(args[0]) - quotient*int(args[1])
        elif op == 21:
          value = abs(int(args[0])) // abs(int(args[1]))
          if (int(args[0]) < 0) != (int(args[1]) < 0): value = -value
        elif op == 22: value = int(args[0]) // int(args[1])
        elif op == 23: value = int(args[0]) % int(args[1])
        elif op == 24: value = args[0] - args[1]
        elif op == 25: value = np.power(args[0], args[1])
        elif op == 26: value = -args[0]
        elif op == 27: value = args[0] == args[1]
        elif op == 28: value = int(args[0]) << int(args[1])
        elif op == 29: value = int(args[0]) >> int(args[1])
        elif op == 30: value = args[0]*args[1] + args[2]
        else: raise RuntimeError(f"rk: invalid host elementwise opcode {op}")
        stack.append(cast(value, dtype_code))
    assert len(stack) == 1
    return stack[0]

  for linear in range(total):
    rem, coords = linear, [0] * n_ranges
    for axis in range(n_ranges-1, -1, -1): rem, coords[axis] = divmod(rem, extents[axis])
    out_index = int(evaluate(out_code, coords))
    if not 0 <= out_index < result.size: raise RuntimeError(f"rk: host elementwise output index out of bounds {out_index}")
    result[out_index] = evaluate(value_code, coords)
  ctypes.memmove(output.va_addr, result.ctypes.data, result.nbytes)  # type: ignore[arg-type]

def _run_host_movement(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Execute a compact postfix integer-index program and copy exact element bytes."""
  total, tag, itemsize, n_ranges, *meta = task.layout
  assert tag == _HOST_MOVEMENT_LAYOUT
  extents = meta[:n_ranges]
  cursor = n_ranges
  out_n = meta[cursor]
  out_code = meta[cursor+1:cursor+1+out_n]
  cursor += out_n+1
  value_n = meta[cursor]
  value_code = meta[cursor+1:cursor+1+value_n]
  inputs = [bufs[r.globals_slot] for r in relocs[1:]]
  output = bufs[relocs[0].globals_slot]
  output_base = task.out_offset // itemsize

  def evaluate(code, coords):
    stack:list = []
    for pos in range(0, len(code), 2):
      op, arg = code[pos], code[pos+1]
      if op == 0: stack.append(arg)
      elif op == 1: stack.append(coords[arg])
      elif op == 10: stack.append((arg, stack.pop()))
      elif op == 12: stack.append((-1, arg))
      elif op == 11:
        false, true, cond = stack.pop(), stack.pop(), stack.pop()
        stack.append(true if cond else false)
      else:
        rhs, lhs = stack.pop(), stack.pop()
        if op == 2: stack.append(lhs + rhs)
        elif op == 3: stack.append(lhs * rhs)
        elif op == 4: stack.append(lhs // rhs)
        elif op == 5: stack.append(lhs % rhs)
        elif op == 6: stack.append(int(lhs < rhs))
        elif op == 7: stack.append(int(lhs != rhs))
        elif op == 8: stack.append(int(bool(lhs) and bool(rhs)))
        elif op == 9: stack.append(int(bool(lhs) or bool(rhs)))
        else: raise RuntimeError(f"rk: invalid host movement opcode {op}")
    assert len(stack) == 1
    return stack[0]

  for linear in range(total):
    rem, coords = linear, [0] * n_ranges
    for axis in range(n_ranges-1, -1, -1): rem, coords[axis] = divmod(rem, extents[axis])
    out_index = evaluate(out_code, coords)
    input_id, in_index = evaluate(value_code, coords)
    if not 0 <= output_base+out_index < output.size // itemsize: raise RuntimeError(f"rk: host movement output index out of bounds {out_index}")
    if input_id == -1:
      representation = int(in_index) & ((1 << (itemsize*8))-1)
      ctypes.memmove(output.va_addr + task.out_offset + out_index*itemsize,
                     representation.to_bytes(itemsize, 'little'), itemsize)  # type: ignore[arg-type]
    else:
      if not 0 <= in_index < inputs[input_id].size // itemsize: raise RuntimeError(f"rk: host movement input index out of bounds {in_index}")
      ctypes.memmove(output.va_addr + task.out_offset + out_index*itemsize,
                     inputs[input_id].va_addr + in_index*itemsize, itemsize)  # type: ignore[arg-type]

def _run_host_static_half(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Materialize a compile-time fp16 tensor into a scratch buffer for a later NPU task."""
  total, tag, *values = task.layout
  assert tag == _HOST_STATIC_HALF_LAYOUT and len(values) == total
  output = ctypes.cast(bufs[relocs[0].globals_slot].va_addr, ctypes.POINTER(ctypes.c_uint16))
  for i, value in enumerate(values): output[i] = value

def _run_host_fp32_view(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple, residual:bool) -> None:
  total, source_slot = task.layout[0], relocs[1].globals_slot
  if residual: _convert_fp32_residual_to_fp16_buf(bufs[source_slot].va_addr, bufs[task.out_slot].va_addr, total)
  else: _convert_fp32_to_fp16_buf(bufs[source_slot].va_addr, bufs[task.out_slot].va_addr, total)

def _run_host_static_int(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Materialize compile-time int32 constants for a later NPU task."""
  total, tag, *values = task.layout
  assert tag == _HOST_STATIC_INT_LAYOUT and len(values) == total and len(relocs) == 1
  output = ctypes.cast(bufs[relocs[0].globals_slot].va_addr, ctypes.POINTER(ctypes.c_int32))
  for i, value in enumerate(values): output[i] = value
  if getenv("ROCKCHIP_DEBUG_UNPOOL"): print("RK_UNPOOL_STATIC_INT", [output[i] for i in range(min(total, 16))])

def _pack_static_gather(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Pack an address-only typed gather; no runtime tensor value is evaluated."""
  total, tag, itemsize, *addresses = task.layout
  assert tag == _HOST_GATHER_MAP_LAYOUT and len(addresses) == total and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  invalid = struct.pack('<e', float("-inf")) if itemsize == 2 else int(-(1 << 31)).to_bytes(4, 'little', signed=True)
  for out_index, source_index in enumerate(addresses):
    if source_index < 0: ctypes.memmove(output.va_addr + out_index*itemsize, invalid, itemsize)  # type: ignore[arg-type]
    else: ctypes.memmove(output.va_addr + out_index*itemsize, source.va_addr + source_index*itemsize, itemsize)  # type: ignore[arg-type]

def _pack_plane_gather(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Repeat one compact candidate across each plane's output; byte copies only."""
  total, tag, itemsize, pooled, out_spatial, candidate, *offset = task.layout
  start = offset[0] if offset else 0
  assert tag == _HOST_PLANE_GATHER_LAYOUT and out_spatial > 0 and 0 <= candidate < pooled and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  for out_index in range(total):
    source_index = ((start+out_index) // out_spatial)*pooled + candidate
    ctypes.memmove(output.va_addr + out_index*itemsize, source.va_addr + source_index*itemsize, itemsize)  # type: ignore[arg-type]
  if getenv("ROCKCHIP_DEBUG_UNPOOL") and itemsize == 2:
    values = ctypes.cast(output.va_addr, ctypes.POINTER(ctypes.c_uint16))
    print("RK_UNPOOL_GATHER_HALF_BITS", [hex(values[i]) for i in range(min(total, 32))])

def _pack_fp16_chunk(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Copy at most four fp16 values into an aligned atom after an NPU stage."""
  count, tag, start = task.layout
  assert tag == _HOST_PACK_CHUNK_LAYOUT and 0 < count <= 4 and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  ctypes.memset(output.va_addr, 0, 8)  # type: ignore[arg-type]
  ctypes.memmove(output.va_addr, source.va_addr + start*2, count*2)  # type: ignore[arg-type]

def _unpack_int32_chunk(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Copy a native four-lane int32 atom into the compact logical output."""
  count, tag, start = task.layout
  assert tag == _HOST_UNPACK_INT_CHUNK_LAYOUT and 0 < count <= 4 and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  ctypes.memmove(output.va_addr + start*4, source.va_addr, count*4)  # type: ignore[arg-type]

def _pack_int32_chunk(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Copy at most four compact int32 values into one aligned MRDMA atom."""
  count, tag, start = task.layout
  assert tag == _HOST_PACK_INT32_CHUNK_LAYOUT and 0 < count <= 4 and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  ctypes.memset(output.va_addr, 0, 16)  # type: ignore[arg-type]
  ctypes.memmove(output.va_addr, source.va_addr + start*4, count*4)  # type: ignore[arg-type]

def _unpack_fp16_chunk(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Copy a native fp16 atom into its compact logical position."""
  count, tag, start = task.layout
  assert tag == _HOST_UNPACK_HALF_CHUNK_LAYOUT and 0 < count <= 4 and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  ctypes.memmove(output.va_addr + start*2, source.va_addr, count*2)  # type: ignore[arg-type]

def _compact_native_half(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Remove the four padding fp16 lanes emitted after each four-lane int32 atom."""
  total, tag = task.layout
  assert tag == _HOST_COMPACT_NATIVE_HALF_LAYOUT and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  for start in range(0, total, 4):
    count = min(4, total-start)
    ctypes.memmove(output.va_addr + start*2, source.va_addr + (start//4)*16, count*2)  # type: ignore[arg-type]
  if getenv("ROCKCHIP_DEBUG_UNPOOL"):
    import numpy as np
    physical = np.frombuffer(ctypes.string_at(source.va_addr, total*4), dtype=np.float16)
    compact = np.frombuffer(ctypes.string_at(output.va_addr, total*2), dtype=np.float16)
    print("RK_UNPOOL_INT_DIFF", physical[:min(total*2, 32)].tolist(), "->", compact[:min(total, 16)].tolist())

def _assemble_int_bytes(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Assemble native 0..255 int32 digits by copying their low bytes."""
  count, tag, start, nbytes = task.layout
  assert tag == _HOST_ASSEMBLE_INT_BYTES_LAYOUT and 0 < count <= 4 and 0 < nbytes <= 4 and len(relocs) == nbytes+1
  output = bufs[relocs[0].globals_slot]
  digits = [bufs[r.globals_slot] for r in relocs[1:]]
  ctypes.memset(output.va_addr + start*4, 0, count*4)  # type: ignore[arg-type]
  for element in range(count):
    for byte, digit in enumerate(digits):
      ctypes.memmove(output.va_addr + (start+element)*4 + byte, digit.va_addr + element*4, 1)  # type: ignore[arg-type]

def _pack_half_bits(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Widen raw fp16 representations into int32 lanes without evaluating them."""
  count, tag, start = task.layout
  assert tag == _HOST_PACK_HALF_BITS_LAYOUT and 0 < count <= 4 and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  ctypes.memset(output.va_addr, 0, 16)  # type: ignore[arg-type]
  for element in range(count):
    ctypes.memmove(output.va_addr + element*4, source.va_addr + (start+element)*2, 2)  # type: ignore[arg-type]
  if getenv("ROCKCHIP_DEBUG_UNPOOL"):
    values = ctypes.cast(output.va_addr, ctypes.POINTER(ctypes.c_uint32))
    print("RK_UNPOOL_PACK_HALF_BITS", [hex(values[i]) for i in range(4)])

def _unpack_half_bits(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Compact low fp16 representation bytes from native int32 lanes."""
  count, tag, start = task.layout
  assert tag == _HOST_UNPACK_HALF_BITS_LAYOUT and 0 < count <= 4 and len(relocs) == 2
  output, source = (bufs[r.globals_slot] for r in relocs)
  if getenv("ROCKCHIP_DEBUG_UNPOOL"):
    values = ctypes.cast(source.va_addr, ctypes.POINTER(ctypes.c_uint32))
    print("RK_UNPOOL_SELECTED_HALF_BITS", [hex(values[i]) for i in range(4)])
  for element in range(count):
    ctypes.memmove(output.va_addr + (start+element)*2, source.va_addr + element*4, 2)  # type: ignore[arg-type]

def _run_host_scatter(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Scatter pooled fp16 values by per-plane int32 indices, summing duplicates in fp32."""
  import numpy as np
  total, tag, index_total, planes, out_spatial, pooled = task.layout
  assert tag == _HOST_SCATTER_LAYOUT and index_total == planes*pooled and total == planes*out_spatial
  output, index_buf, value_buf = (bufs[r.globals_slot] for r in relocs)
  indices = np.frombuffer(ctypes.string_at(index_buf.va_addr, index_total*4), dtype=np.int32)
  values = np.frombuffer(ctypes.string_at(value_buf.va_addr, index_total*2), dtype=np.float16)
  accumulator = np.zeros(total, dtype=np.float32)
  for plane in range(planes):
    for update in range(pooled):
      source = plane*pooled+update
      spatial = int(indices[source])
      if 0 <= spatial < out_spatial:
        destination = plane*out_spatial+spatial
        accumulator[destination] = np.float32(accumulator[destination] + np.float32(values[source]))
  result = accumulator.astype(np.float16)
  ctypes.memmove(output.va_addr, result.ctypes.data, total*2)  # type: ignore[arg-type]

def _run_host_argmax(task:RKTask, relocs:list[RKReloc]|tuple[RKReloc, ...], bufs:tuple) -> None:
  """Choose the first valid static candidate equal to each max-pool output."""
  import numpy as np
  total, tag, window, input_spatial, *mapping = task.layout
  assert tag == _HOST_ARGMAX_LAYOUT and len(mapping) == total*window
  output_buf, data_buf, maximum_buf = (bufs[r.globals_slot] for r in relocs)
  data = np.frombuffer(ctypes.string_at(data_buf.va_addr, data_buf.size), dtype=np.float16)
  maximum = np.frombuffer(ctypes.string_at(maximum_buf.va_addr, total*2), dtype=np.float16)
  output = np.empty(total, dtype=np.int32)
  for out_index in range(total):
    selected = 0
    for candidate in mapping[out_index*window:(out_index+1)*window]:
      if candidate < 0 or candidate >= data.size: continue
      if data[candidate] == maximum[out_index] or (np.isnan(data[candidate]) and np.isnan(maximum[out_index])):
        selected = candidate % input_spatial
        break
    output[out_index] = selected
  ctypes.memmove(output_buf.va_addr, output.ctypes.data, total*4)  # type: ignore[arg-type]

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
      decoded_tile = _decode_materialized_cmac_layout(task.layout) if \
        task.kind == "cmac" and len(task.layout) > 5 and task.layout[5] == _CMAC_MATERIALIZED_LAYOUT else None
      if decoded_tile is not None and (decoded_tile[0] < task.layout[0] or decoded_tile[1] < task.layout[1] or
                                       decoded_tile[2] < task.layout[2]):
        _run_tiled_materialized_cmac(dev, self.name, self.cmds, task, self.relocs, bufs)
        self.submit_count += ((task.layout[0] + decoded_tile[0] - 1) // decoded_tile[0]) * \
                             ((task.layout[1] + decoded_tile[1] - 1) // decoded_tile[1]) * \
                             ((task.layout[2] + decoded_tile[2] - 1) // decoded_tile[2])
        self.last_enable_mask = task.enable_mask
        dev.submitted_masks.add(task.enable_mask)
        return
      buf_map:dict[int, HCQBuffer] = {}
      cmac_bufs: list[HCQBuffer] = []
      if task.kind == "cmac":
        cmac_sources = list(bufs)
        for slot in task.fp32_inputs:
          if slot in (_CONST_SLOT, _ZERO_SLOT): continue
          source_n = cmac_sources[slot].size // 4
          converted = dev._gpu_alloc(max(source_n * 2, 4096), 0)
          temp.append(converted)
          _convert_fp32_to_fp16_buf(cmac_sources[slot].va_addr, converted.va_addr, source_n)
          cmac_sources[slot] = converted
        layout = task.layout
        M, N, K, align_in, align_out = layout[0], layout[1], layout[2], layout[3], layout[4]
        a_s, b_s = self.relocs[0].globals_slot, self.relocs[1].globals_slot
        materialized = len(layout) > 5 and layout[5] == _CMAC_MATERIALIZED_LAYOUT
        decoded_materialized = _decode_materialized_cmac_layout(layout) if materialized else None
        a_buf = dev._gpu_alloc(max(M*align_in*2, 4096), 0)
        temp.append(a_buf)
        if materialized:
          pass
        elif a_s == _CONST_SLOT:
          ctypes.memmove(a_buf.va_addr, struct.pack('<e', task.const_val) * align_in, align_in * 2)  # type: ignore[arg-type]
        else:
          _pad_a(cmac_sources[a_s].va_addr, a_buf.va_addr, M, K, align_in)
        b_buf = dev._gpu_alloc(max(align_out*align_in*2, 4096), 0)
        temp.append(b_buf)
        if materialized:
          assert decoded_materialized is not None
          b_src = None if b_s == _CONST_SLOT else cmac_sources[b_s].va_addr
          _materialize_cmac_inputs(cmac_sources[a_s].va_addr, b_src, a_buf.va_addr, b_buf.va_addr,
                                   M, N, K, align_in, align_out, decoded_materialized)
        elif b_s == _CONST_SLOT:
          ctypes.memmove(b_buf.va_addr, struct.pack('<e', task.const_val) * (align_out * align_in), align_out * align_in * 2)  # type: ignore[arg-type]
        else:
          _swizzle_b(cmac_sources[b_s].va_addr, b_buf.va_addr, K, N, align_out, align_in)
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
        M, N, _, _, align_out = task.layout[:5]
        if len(task.layout) > 5 and task.layout[5] == _CMAC_MATERIALIZED_LAYOUT:
          decoded = _decode_materialized_cmac_layout(task.layout)
          bias = bufs[decoded[3]].va_addr if decoded[3] >= 0 else None
          _unpack_materialized_cmac_out(cmac_bufs[2].va_addr, bufs[task.out_slot].va_addr, M, N, align_out, decoded, bias,
                                        packed_inputs=(cmac_bufs[0].va_addr, cmac_bufs[1].va_addr, task.layout[2], task.layout[3]))
        else:
          bias_slot, bias_axis, relu = task.layout[5:] if len(task.layout) >= 8 else (-1, -1, 0)
          bias = bufs[bias_slot].va_addr if bias_slot >= 0 else None
          _unpack_cmac_out(cmac_bufs[2].va_addr, bufs[task.out_slot].va_addr, M, N, align_out, bias, bias_axis, bool(relu), task.fp32_output)
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
        if len(task.layout) > 1 and task.layout[1] == _HOST_MOVEMENT_LAYOUT:
          _run_host_movement(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_HALF_LAYOUT:
          _run_host_static_half(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_INT_LAYOUT:
          _run_host_static_int(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_GATHER_MAP_LAYOUT:
          _pack_static_gather(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_PLANE_GATHER_LAYOUT:
          _pack_plane_gather(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_PACK_CHUNK_LAYOUT:
          _pack_fp16_chunk(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_INT_CHUNK_LAYOUT:
          _unpack_int32_chunk(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_PACK_INT32_CHUNK_LAYOUT:
          _pack_int32_chunk(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_HALF_CHUNK_LAYOUT:
          _unpack_fp16_chunk(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_COMPACT_NATIVE_HALF_LAYOUT:
          _compact_native_half(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_ASSEMBLE_INT_BYTES_LAYOUT:
          _assemble_int_bytes(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_PACK_HALF_BITS_LAYOUT:
          _pack_half_bits(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_HALF_BITS_LAYOUT:
          _unpack_half_bits(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_BOOL_HALF_LAYOUT:
          _run_host_bool_half(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_SELECT_HALF_LAYOUT:
          _run_host_static_select_half(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_SELECT_INT_LAYOUT:
          _run_host_static_select_int(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_HALF_INT_LAYOUT:
          _run_host_half_int(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_SCATTER_LAYOUT:
          _run_host_scatter(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_ARGMAX_LAYOUT:
          _run_host_argmax(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_BITWISE_LAYOUT:
          _run_host_bitwise(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_TRUNC_LAYOUT:
          _run_host_trunc(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_COPYSIGN_LAYOUT:
          _run_host_copysign(task, st.relocs, bufs)
          continue
        if len(task.layout) > 1 and task.layout[1] == _HOST_ELEMENTWISE_LAYOUT:
          _run_host_elementwise(task, st.relocs, bufs)
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
    # Materialized CMAC stages need host-side A/B gathering and fp32-output
    # unpacking around every task.  Run mixed CMAC/DPU programs stage-by-stage
    # with shared scratch buffers; all arithmetic remains on the NPU.
    if any(st.task.kind == "cmac" for st in subtasks):
      ext, shared, original = list(bufs), [], self.subtasks
      max_slot = max((r.globals_slot for st in subtasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)),
                     default=len(ext)-1)
      scratch_sizes:dict[int, int] = {}
      for st in subtasks:
        if st.task.out_slot < len(ext): continue
        elements = st.task.layout[0]*st.task.layout[1] if st.task.kind == "cmac" else st.task.layout[0]
        itemsize = 4 if st.task.native_int32_output else 2
        scratch_sizes[st.task.out_slot] = max(scratch_sizes.get(st.task.out_slot, 0), st.task.out_offset+elements*itemsize)
      try:
        while len(ext) <= max_slot:
          shared.append(b := dev._gpu_alloc(max(scratch_sizes.get(len(ext), 0), 4096), 0))
          ext.append(b)
        for st in subtasks:
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] == _HOST_MOVEMENT_LAYOUT:
            _run_host_movement(st.task, st.relocs, tuple(ext))
            continue
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] == _HOST_STATIC_HALF_LAYOUT:
            _run_host_static_half(st.task, st.relocs, tuple(ext))
            continue
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] == _HOST_GATHER_MAP_LAYOUT:
            _pack_static_gather(st.task, st.relocs, tuple(ext))
            continue
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] == _HOST_BOOL_HALF_LAYOUT:
            _run_host_bool_half(st.task, st.relocs, tuple(ext))
            continue
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] == _HOST_STATIC_SELECT_HALF_LAYOUT:
            _run_host_static_select_half(st.task, st.relocs, tuple(ext))
            continue
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] == _HOST_STATIC_SELECT_INT_LAYOUT:
            _run_host_static_select_int(st.task, st.relocs, tuple(ext))
            continue
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] == _HOST_HALF_INT_LAYOUT:
            _run_host_half_int(st.task, st.relocs, tuple(ext))
            continue
          if st.task.is_copy and len(st.task.layout) > 1 and st.task.layout[1] in (_HOST_FP32_HALF_LAYOUT, _HOST_FP32_RESIDUAL_LAYOUT):
            _run_host_fp32_view(st.task, st.relocs, tuple(ext), st.task.layout[1] == _HOST_FP32_RESIDUAL_LAYOUT)
            continue
          self.cmds, self.task, self.relocs = list(st.cmds), st.task, list(st.relocs)
          # CMAC needs its single-task host gather/unpack path. Post-CMAC DPU
          # stages also stay single-task for reset stability, with the typed
          # bool ABI boundary prepared explicitly around that submission.
          if st.task.kind == "cmac":
            self.subtasks = None
            self(*tuple(ext))
          else:
            stage_ext, stage_temp = list(ext), []
            try:
              for slot in st.task.bool_inputs:
                source_n = ext[slot].size
                converted = dev._gpu_alloc(max(source_n*2, 4096), 0)
                stage_temp.append(converted)
                _convert_bool_to_fp16_buf(ext[slot].va_addr, converted.va_addr, source_n)
                stage_ext[slot] = converted
              bool_output = None
              if st.task.bool_output:
                original_out = ext[st.task.out_slot]
                out_n = original_out.size
                converted = dev._gpu_alloc(max(out_n*2, 4096), 0)
                stage_temp.append(converted)
                stage_ext[st.task.out_slot] = converted
                bool_output = (original_out, converted, out_n)
              self.subtasks = None
              dev.reset_npu()
              self(*tuple(stage_ext))
              if bool_output is not None:
                original_out, converted, out_n = bool_output
                _convert_fp16_to_bool_buf(converted.va_addr, original_out.va_addr, out_n)
            finally:
              for b in stage_temp: dev._gpu_free(b)
          if getenv("ROCKCHIP_DEBUG_BOOL_REDUCE") >= 2:
            out = ext[st.task.out_slot]
            if st.task.bool_output:
              values = tuple(ctypes.string_at(out.va_addr, min(out.size, 8)))
            else:
              count = min(out.size//2, 8)
              values = tuple(struct.unpack(f"<{count}e", ctypes.string_at(out.va_addr, count*2)))
            print("RK_BOOL_REDUCE_STAGE", st.task.kind, st.task.out_slot, values)
      finally:
        self.subtasks = original
        for b in shared: dev._gpu_free(b)
      return
    # Native fp16->int32 writes use four-lane aligned atoms. Their source pack
    # happens after the preceding DPU selection chain, so preserve subtask
    # order instead of hoisting every copy ahead of every DPU submission.
    if any(st.task.is_copy and len(st.task.layout) > 1 and
           st.task.layout[1] in (_HOST_STATIC_INT_LAYOUT, _HOST_PACK_CHUNK_LAYOUT, _HOST_UNPACK_INT_CHUNK_LAYOUT,
                                 _HOST_PACK_INT32_CHUNK_LAYOUT, _HOST_UNPACK_HALF_CHUNK_LAYOUT,
                                 _HOST_COMPACT_NATIVE_HALF_LAYOUT, _HOST_ASSEMBLE_INT_BYTES_LAYOUT,
                                 _HOST_PACK_HALF_BITS_LAYOUT, _HOST_UNPACK_HALF_BITS_LAYOUT,
                                 _HOST_STATIC_SELECT_HALF_LAYOUT, _HOST_STATIC_SELECT_INT_LAYOUT,
                                 _HOST_HALF_INT_LAYOUT) for st in subtasks) or \
       any(st.task.is_copy and any(not prior.task.is_copy for prior in subtasks[:index])
           for index, st in enumerate(subtasks)):
      ext, shared, original = list(bufs), [], self.subtasks
      max_slot = max((r.globals_slot for st in subtasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)),
                     default=len(ext)-1)
      total = max(st.task.layout[0] for st in subtasks)
      try:
        while len(ext) <= max_slot:
          shared.append(b := dev._gpu_alloc(max(total*4, 4096), 0))
          ext.append(b)
        pending:list[RKSubTask] = []
        def flush_pending() -> None:
          if not pending: return
          if any(st.task.native_int32_input or st.task.native_int32_output for st in pending): dev.reset_npu()
          self.subtasks = list(pending)
          self._submit_multi(tuple(ext))
          pending.clear()
        for st in subtasks:
          task = st.task
          if task.is_copy:
            copy_inputs = {r.globals_slot for r in st.relocs
                           if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT, task.out_slot)}
            pending_inputs = {r.globals_slot for p in pending for r in p.relocs
                              if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT, p.task.out_slot)}
            pending_outputs = {p.task.out_slot for p in pending}
            if copy_inputs & pending_outputs or task.out_slot in pending_inputs or task.out_slot in pending_outputs: flush_pending()
            if len(task.layout) > 1 and task.layout[1] == _HOST_MOVEMENT_LAYOUT: _run_host_movement(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_HALF_LAYOUT: _run_host_static_half(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_INT_LAYOUT: _run_host_static_int(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_BOOL_HALF_LAYOUT: _run_host_bool_half(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_SELECT_HALF_LAYOUT:
              _run_host_static_select_half(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_SELECT_INT_LAYOUT:
              _run_host_static_select_int(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_HALF_INT_LAYOUT:
              _run_host_half_int(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_GATHER_MAP_LAYOUT: _pack_static_gather(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_PLANE_GATHER_LAYOUT: _pack_plane_gather(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_PACK_CHUNK_LAYOUT: _pack_fp16_chunk(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_INT_CHUNK_LAYOUT: _unpack_int32_chunk(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_PACK_INT32_CHUNK_LAYOUT: _pack_int32_chunk(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_HALF_CHUNK_LAYOUT: _unpack_fp16_chunk(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_COMPACT_NATIVE_HALF_LAYOUT: _compact_native_half(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_ASSEMBLE_INT_BYTES_LAYOUT: _assemble_int_bytes(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_PACK_HALF_BITS_LAYOUT: _pack_half_bits(task, st.relocs, tuple(ext))
            elif len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_HALF_BITS_LAYOUT: _unpack_half_bits(task, st.relocs, tuple(ext))
            else: raise RuntimeError(f"rk: unsupported ordered copy layout {task.layout[:2]}")
            continue
          if task.fp32_residual_input:
            # The preceding high-half conversion and this residual conversion
            # read the same fp32 slot through different ABI views.
            flush_pending()
            pending.append(st)
            flush_pending()
            continue
          if any((cmd & 0xffff) == rk.REG_DPU_BN_RELUX_CMP_VALUE and
                 ((cmd >> 16) & 0xffffffff) == 0x3f800000 for cmd in st.cmds):
            flush_pending()
            dev.reset_npu()
            pending.append(st)
            flush_pending()
            dev.reset_npu()
            continue
          pending.append(st)
          if len(pending) == 64: flush_pending()
        flush_pending()
        if any(st.task.native_int32_input or st.task.native_int32_output for st in subtasks): dev.reset_npu()
      finally:
        self.subtasks = original
        for b in shared: dev._gpu_free(b)
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
          # Movement tasks may gather int32 windows before DPU converts them to fp16.
          shared.append(b := dev._gpu_alloc(max(total * 4, 4096), 0))
          ext.append(b)
        # Handle copy tasks host-side
        for st in copy_tasks:
          task = st.task
          if len(task.layout) > 1 and task.layout[1] == _HOST_MOVEMENT_LAYOUT:
            _run_host_movement(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_HALF_LAYOUT:
            _run_host_static_half(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_INT_LAYOUT:
            _run_host_static_int(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_GATHER_MAP_LAYOUT:
            _pack_static_gather(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_PLANE_GATHER_LAYOUT:
            _pack_plane_gather(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_PACK_CHUNK_LAYOUT:
            _pack_fp16_chunk(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_INT_CHUNK_LAYOUT:
            _unpack_int32_chunk(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_PACK_INT32_CHUNK_LAYOUT:
            _pack_int32_chunk(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_HALF_CHUNK_LAYOUT:
            _unpack_fp16_chunk(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_COMPACT_NATIVE_HALF_LAYOUT:
            _compact_native_half(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_ASSEMBLE_INT_BYTES_LAYOUT:
            _assemble_int_bytes(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_PACK_HALF_BITS_LAYOUT:
            _pack_half_bits(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_UNPACK_HALF_BITS_LAYOUT:
            _unpack_half_bits(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_BOOL_HALF_LAYOUT:
            _run_host_bool_half(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_SELECT_HALF_LAYOUT:
            _run_host_static_select_half(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_STATIC_SELECT_INT_LAYOUT:
            _run_host_static_select_int(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_HALF_INT_LAYOUT:
            _run_host_half_int(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_BITWISE_LAYOUT:
            _run_host_bitwise(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_TRUNC_LAYOUT:
            _run_host_trunc(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_COPYSIGN_LAYOUT:
            _run_host_copysign(task, st.relocs, tuple(ext))
            continue
          if len(task.layout) > 1 and task.layout[1] == _HOST_ELEMENTWISE_LAYOUT:
            _run_host_elementwise(task, st.relocs, tuple(ext))
            continue
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
          if getenv("ROCKCHIP_DEBUG_LOCAL_MAX") >= 2:
            out = ext[st.task.out_slot]
            if st.task.fp32_output:
              count = min(out.size//4, 4)
              values = struct.unpack(f"<{count}f", ctypes.string_at(out.va_addr, count*4))
            else:
              count = min(out.size//2, 4)
              values = struct.unpack(f"<{count}e", ctypes.string_at(out.va_addr, count*2))
            print("RK_LOCAL_MAX_STAGE", st.task.out_slot, values)
      finally:
        self.subtasks = subtasks
        for b in shared: dev._gpu_free(b)
      return
    # Custom comparison and LUT stages leave DPU state that makes a mixed chain unstable.
    # Keep those stages isolated and chain only consecutive ordinary DPU tasks.
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
        # Batch only consecutive ordinary DPU stages. LUT, comparison, and
        # compute-family boundaries remain reset-separated.
        dpu_pending:list[RKSubTask] = []
        def flush_pending() -> None:
          if not dpu_pending: return
          dev.reset_npu()
          self.subtasks = list(dpu_pending)
          self._submit_multi(tuple(ext))
          dpu_pending.clear()
        for st in subtasks:
          if is_cmp(st) or st.task.kind != "dpu":
            flush_pending()
            dev.reset_npu()
            self.subtasks = [st]
            self._submit_multi(tuple(ext))
          else:
            dpu_pending.append(st)
        flush_pending()
        # WIP reference: the older stability path submitted every stage
        # separately. Long reduction sequences eventually wedged the driver:
        #   for st in subtasks:
        #     dev.reset_npu()
        #     self.subtasks = [st]
        #     self._submit_multi(tuple(ext))
      finally:
        self.subtasks = original
        for b in shared: dev._gpu_free(b)
      return
    temp:list[HCQBuffer] = []
    prepared = list(bufs)
    original_prepared = list(bufs)
    total = max(st.task.layout[0] for st in subtasks)
    max_slot = max((r.globals_slot for st in subtasks for r in st.relocs if r.globals_slot not in (_CONST_SLOT, _ZERO_SLOT)),
                   default=len(prepared)-1)
    while len(prepared) <= max_slot:
      temp.append(scratch := dev._gpu_alloc(max(total * 2, 4096), 0))
      prepared.append(scratch)
    source_counts:dict[int, int] = {}
    periodic_slots = {s for st in subtasks if st.task.periodic_input for s in st.task.fp32_inputs}
    residual_slots = {s for st in subtasks if st.task.fp32_residual_input for s in st.task.fp32_inputs}
    for slot in {s for st in subtasks for s in st.task.fp32_inputs}:
      source_counts[slot] = source_n = prepared[slot].size // 4
      converted = dev._gpu_alloc(max(source_n * 2, 4096), 0)
      temp.append(converted)
      if slot in residual_slots: _convert_fp32_residual_to_fp16_buf(prepared[slot].va_addr, converted.va_addr, source_n)
      elif slot in periodic_slots: _convert_periodic_fp32_to_fp16_buf(prepared[slot].va_addr, converted.va_addr, source_n)
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
                             for st in subtasks if st.task.fp32_output or (st.task.int32_output and not st.task.native_int32_output) or st.task.uint8_output or
                             st.task.bool_output or st.task.trunc_output), None)
    output_conversion = None
    if converted_output is not None:
      output_slot, is_fp32, is_uint8, is_bool, is_trunc = converted_output
      # Use the actual output element count from the buffer size, not `total` (which is
      # max across subtasks and may be larger than the real output for broadcast/pad ops)
      # fp32/int32 outputs are 4 bytes; uint8/bool are 1 byte; fp16/trunc are 2 bytes
      out_itemsize = 4 if is_fp32 else (1 if (is_uint8 or is_bool) else (2 if is_trunc else 4))
      out_n = original_prepared[output_slot].size // out_itemsize
      converted = dev._gpu_alloc(max(total * 2, out_n * 2, 4096), 0)
      temp.append(converted)
      output_conversion = (original_prepared[output_slot], converted, out_n, is_fp32, is_uint8, is_bool, is_trunc)
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
            itemsize = 4 if task.native_int32_input else 2
            zbuf = dev._gpu_alloc(max(total * itemsize, 4096), 0)
            temp.append(zbuf)
            ctypes.memset(zbuf.va_addr, 0, total * itemsize)  # type: ignore[arg-type]
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
    # The RK3588 GEM mmap path rejects multi-megabyte BOs. Generalized CMAC
    # gathers logical tensors through their CPU mapping and submits compact
    # DMA-backed tiles, so large logical buffers can safely remain host-backed.
    if size >= _ROCKCHIP_MAX_MAPPABLE_BO:
      va = FileIOInterface.anon_mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE,
                                     mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, 0)
      return HCQBuffer(va_addr=va, size=size)
    mc = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags|rk.RKNPU_MEM_NON_CACHEABLE)
    mm = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mc.handle, offset=0)
    va = self.fd_ctl.mmap(0, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mm.offset)
    mc.flink_name = self.create_flink_name(mc.handle, name)
    return HCQBuffer(va_addr=va, size=size, meta=mc)

  def _gpu_free(self, buf:HCQBuffer) -> None:
    FileIOInterface.munmap(buf.va_addr, buf.size)
    if buf.meta is None: return
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, __payload=rk.struct_rknpu_mem_destroy(handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr))

  def reset_npu(self):
    rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, __payload=rk.struct_rknpu_action(flags=rk.RKNPU_ACT_RESET, value=0))
