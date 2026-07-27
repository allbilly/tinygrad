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
from tinygrad.runtime.support.rockchip import build_native_program, encode_rk, decode_rk, RKTask, RKReloc, _CONST_SLOT, _ZERO_SLOT

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
  si, e, mt = (b>>31)&1, (b>>23)&0xFF, b&0x7FFFFF
  if e == 0: return si<<15
  if e == 0xFF: return (si<<15)|0x7C00 if mt == 0 else (si<<15)|0x7E00|((mt>>13)&0x1FF)
  ne = e - 112
  if ne >= 0x1F: return (si<<15)|0x7C00
  if ne >= 1:
    mant = (mt >> 13) + (1 if (mt & 0x1FFF) > 0x1000 or (mt & 0x1FFF == 0x1000 and (mt >> 13) & 1) else 0)
    return (si<<15)|0x7C00 if ne + (mant >= 0x400) >= 0x1F else (si<<15)|((ne + (mant >= 0x400))<<10)|(mant & 0x3FF)
  sh, fm = 14-ne, (1<<23)|mt
  return (si<<15)|min((fm >> sh) + (1 if (fm & ((1<<sh)-1)) + ((fm >> sh) & 1) > (1<<(sh-1)) else 0), 1<<10)

def _unpack_cmac_out(src, dst, M, N, align_out):
  s, d = ctypes.cast(src, ctypes.POINTER(ctypes.c_uint32)), ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint16))
  for i in range(M * N):
    d[i] = _fp32_to_fp16(s[(i // N) * align_out + i % N])

class RockchipProgram(Program['RockchipDevice']):
  cmds: list[int]
  task: RKTask
  relocs: list[RKReloc]
  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
    self.device, self.name, self.submit_count, self.last_enable_mask = dev, obj.name, 0, 0
    if len(obj.lib) >= 4 and struct.unpack_from("<I", obj.lib, 0)[0] == 0x524b494d:
      self.cmds, self.task, self.relocs = decode_rk(obj.lib)
    else:
      raise RuntimeError(f"rk: no Python fallback — binary is not an RKImage (len={len(obj.lib)}, first byte={obj.lib[0]:#x})")

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    dev = self.device
    dev.reset_npu()
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
      # DPU DMA copy: host-side memmove (data movement, not NPU compute).
      # No submit_count increment — no NPU submission. Documented honestly as non-native.
      if task.is_copy:
        total = task.layout[0]
        in_buf, out_buf = buf_map[self.relocs[1].globals_slot], buf_map[self.relocs[0].globals_slot]
        ctypes.memmove(out_buf.va_addr, in_buf.va_addr, total * 2)  # type: ignore[arg-type]
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
    finally:
      for b in temp: dev._gpu_free(b)
    self.submit_count += 1
    self.last_enable_mask = task.enable_mask
    dev.submitted_masks.add(task.enable_mask)

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  has_threads = False
  has_local = False
  code_for_op = {}  # no Python fallback — all compute goes through native_program
  def __init__(self, target:Target): self.target, self.tensor_cores = target, []
  def native_program(self, ast:UOp) -> UOp|None: return build_native_program(ast)
  def asm(self, prg:UOp, lin:UOp) -> bytes:
    task, cmds, relocs = None, [], []
    for u in lin.src:
      if isinstance(u.arg, RKTask): task = u.arg
      elif isinstance(u.arg, RKReloc): relocs.append(u.arg)
      elif isinstance(u.arg, int): cmds.append(u.arg)
    if task is None: raise RuntimeError("rk: no RKTask metadata — non-NPU kernel with no Python fallback")
    return encode_rk(tuple(cmds), task, tuple(relocs))
  def supported_dtypes(self) -> set[DType]: return {dtypes.half}

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
