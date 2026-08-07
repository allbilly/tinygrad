from __future__ import annotations
import ctypes, mmap, os, struct, time
import numpy as np
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, suppress_finalizing, to_mv
from tinygrad.renderer.rockchip import (RKBufferKind, RKEWOut, RockchipRenderer, decode_image, patch_image,
  emit_ew_stage, RKArg, RKImage, _EW_CHUNK, _EW_SLOT_BYTES, _MAX_EW_ELEMS_FP16)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer

# PC-chain up to `_EW_CHAIN` tasks/ioctl. No soft-reset on hot path.
# Regcmd/task scratch floors match allbilly elementwise (64KiB / 16KiB), not 4096.
_EW_CHAIN = 64
_PC_TAIL = 4
_CMD_BUF_MIN = 65536
_TASK_BUF_MIN = 16384

def _pc(t, r, v=0): return (t << 48) | ((v & 0xffffffff) << 16) | r
def _align_up(x:int, a:int) -> int: return (x + a - 1) & ~(a - 1)

def _pack_half_slots(src:np.ndarray, dst_mv:memoryview, count:int) -> None:
  """Contiguous half → 64B slots (≤8 halfs at start of each slot)."""
  dst = np.frombuffer(dst_mv, dtype=np.uint16)
  nslots = (count + _EW_CHUNK - 1) // _EW_CHUNK
  dst[:nslots * (_EW_SLOT_BYTES // 2)] = 0
  src_u16 = src.view(np.uint16) if src.dtype == np.float16 else src
  for slot, start in enumerate(range(0, count, _EW_CHUNK)):
    n = min(_EW_CHUNK, count - start)
    base = slot * (_EW_SLOT_BYTES // 2)
    dst[base:base + n] = src_u16[start:start + n]

def _fp32_slots_to_half(scratch_mv:memoryview, count:int) -> None:
  """In-slot f32→half (first `n` floats of each 64B slot → first `n` halfs)."""
  for slot, start in enumerate(range(0, count, _EW_CHUNK)):
    n = min(_EW_CHUNK, count - start)
    off = slot * _EW_SLOT_BYTES
    f32 = np.frombuffer(scratch_mv, dtype=np.float32, count=_EW_SLOT_BYTES // 4, offset=off).copy()
    h = f32[:n].astype(np.float16)
    slot_u16 = np.frombuffer(scratch_mv, dtype=np.uint16, count=_EW_SLOT_BYTES // 2, offset=off)
    slot_u16[:] = 0
    slot_u16[:n] = h.view(np.uint16)

def _unpack_half_slots(scratch_mv:memoryview, count:int) -> bytes:
  out = np.empty(count, dtype=np.float16)
  src = np.frombuffer(scratch_mv, dtype=np.uint16)
  for slot, start in enumerate(range(0, count, _EW_CHUNK)):
    n = min(_EW_CHUNK, count - start)
    base = slot * (_EW_SLOT_BYTES // 2)
    out[start:start + n] = src[base:base + n].view(np.float16)
  return out.tobytes()

def _arg_as_half_u16(buf:HCQBuffer, nelem:int, itemsize:int=2) -> np.ndarray:
  """Read ARG as half bit-patterns; float ARG (itemsize=4) is cast on the host."""
  mv = to_mv(int(buf.va_addr), buf.size)
  if itemsize == 4:
    return np.frombuffer(mv, dtype=np.float32, count=nelem).astype(np.float16).view(np.uint16).copy()
  return np.frombuffer(mv, dtype=np.uint16, count=nelem)

class RockchipAllocator(LRUAllocator['RockchipDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer: return self.dev._gpu_alloc(size)
  def _copyin(self, dest:HCQBuffer, src:memoryview): ctypes.memmove(int(dest.va_addr), from_mv(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:HCQBuffer): ctypes.memmove(from_mv(dest), int(src.va_addr), dest.nbytes)
  def _as_buffer(self, src:HCQBuffer): return to_mv(int(src.va_addr), src.size)
  def _offset(self, buf:HCQBuffer, size:int, offset:int): return buf.offset(offset, size)
  def _free(self, buf:HCQBuffer, options:BufferSpec): self.dev._gpu_free(buf)

class RockchipProgram(Program['RockchipDevice']):
  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
    self.dev, self.name, self.image = dev, obj.name, decode_image(obj.lib)
    self.scratch = tuple(dev._gpu_alloc(x.size) for x in self.image.scratch)
    self.submit_count = 0
    self._cmd_buf:HCQBuffer|None = None
    self._cmd_cap = 0
    self._task_buf:HCQBuffer|None = None
  @suppress_finalizing
  def __del__(self):
    for buf in getattr(self, "scratch", ()): self.dev._gpu_free(buf)
    if getattr(self, "_cmd_buf", None) is not None: self.dev._gpu_free(self._cmd_buf)
    if getattr(self, "_task_buf", None) is not None: self.dev._gpu_free(self._task_buf)
  def _dma(self, buf:HCQBuffer) -> int: return int(buf.meta.dma_addr)+int(buf.va_addr)-int(buf.base.va_addr)
  def _submit_pcchain(self, bodies:list[tuple[int, ...]]) -> None:
    """Submit EW tasks as one PC chain (no soft-reset)."""
    n = len(bodies)
    if not (1 <= n <= _EW_CHAIN): raise ValueError(f"EW PC-chain length {n} out of range")
    offsets:list[int] = []
    off = 0
    for b in bodies:
      offsets.append(off)
      off += _align_up(len(b) + _PC_TAIL, 2)
    need = off * 8
    if self._cmd_buf is None or self._cmd_cap < need:
      if self._cmd_buf is not None: self.dev._gpu_free(self._cmd_buf)
      self._cmd_buf = self.dev._gpu_alloc(max(need, _CMD_BUF_MIN))
      self._cmd_cap = self._cmd_buf.size
    tneed = n * ctypes.sizeof(rk.struct_rknpu_task)
    if self._task_buf is None or self._task_buf.size < tneed:
      if self._task_buf is not None: self.dev._gpu_free(self._task_buf)
      self._task_buf = self.dev._gpu_alloc(max(tneed, _TASK_BUF_MIN), rk.RKNPU_MEM_KERNEL_MAPPING)
    cmd, task = self._cmd_buf, self._task_buf
    ctypes.memset(int(cmd.va_addr), 0, need)
    base_dma = self._dma(cmd)
    for i, body in enumerate(bodies):
      base = offsets[i]
      ctypes.memmove(int(cmd.va_addr) + base * 8, (ctypes.c_uint64 * len(body))(*body), len(body) * 8)
      if i + 1 < n:
        next_addr = (base_dma + offsets[i + 1] * 8) & 0xFFFFFFF0
        next_amt = len(bodies[i + 1])
        tail = (_pc(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS, next_addr),
                _pc(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, next_amt),
                _pc(rk.TARGET_VERSION, 0, 0),
                _pc(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
      else:
        tail = (_pc(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS, 0),
                _pc(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, 0),
                _pc(rk.TARGET_VERSION, 0, 0),
                _pc(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18))
      ctypes.memmove(int(cmd.va_addr) + (base + len(body)) * 8,
                     (ctypes.c_uint64 * _PC_TAIL)(*tail), _PC_TAIL * 8)
      desc = rk.struct_rknpu_task(0, 4, 0x18, 0x300, 0x1ffff, 0, len(body) + _PC_TAIL, 0,
                                  base_dma + base * 8)
      ctypes.memmove(int(task.va_addr) + i * ctypes.sizeof(desc), ctypes.addressof(desc), ctypes.sizeof(desc))
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
      flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=6000,
      task_start=0, task_number=n, task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr,
      regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
      subcore_task=(rk.struct_rknpu_subcore_task*5)(
        rk.struct_rknpu_subcore_task(0, n), rk.struct_rknpu_subcore_task(n, 0), rk.struct_rknpu_subcore_task(n, 0)))
    self.submit_count += 1
    self.dev.submit_count += 1
  def _run_ew_op_fp32(self, op, address) -> None:
    """OUT=5: ≤8-elem mtx512 chunks; host f32→half after each op."""
    bodies:list[tuple[int, ...]] = []
    def flush() -> None:
      if not bodies: return
      self._submit_pcchain(bodies)
      bodies.clear()
    for slot, start in enumerate(range(0, op.count, _EW_CHUNK)):
      n = min(_EW_CHUNK, op.count - start)
      off = slot * _EW_SLOT_BYTES
      stage = emit_ew_stage(0,
        RKArg(op.dst.kind, op.dst.index, op.dst.addend + off),
        RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend + off),
        RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend + off), n, op.ew_cfg, RKEWOut.FP32)
      bodies.append(patch_image(RKImage(self.image.target, (stage,)), address)[0])
      if len(bodies) >= _EW_CHAIN: flush()
    flush()
    if op.cvt_scratch is not None:
      mv = to_mv(int(self.scratch[op.cvt_scratch].va_addr), self.scratch[op.cvt_scratch].size)
      _fp32_slots_to_half(mv, op.count)
  def _run_ew_ops_fp16(self, address) -> None:
    """OUT=2: contiguous half; chain all ops in one/few ioctls (no host cvt)."""
    bodies:list[tuple[int, ...]] = []
    def flush() -> None:
      if not bodies: return
      self._submit_pcchain(bodies)
      bodies.clear()
    for op in self.image.ew_ops:
      for start in range(0, op.count, _MAX_EW_ELEMS_FP16):
        n = min(_MAX_EW_ELEMS_FP16, op.count - start)
        off = start * 2  # half bytes
        stage = emit_ew_stage(0,
          RKArg(op.dst.kind, op.dst.index, op.dst.addend + off),
          RKArg(op.lhs.kind, op.lhs.index, op.lhs.addend + off),
          RKArg(op.rhs.kind, op.rhs.index, op.rhs.addend + off), n, op.ew_cfg, RKEWOut.FP16)
        bodies.append(patch_image(RKImage(self.image.target, (stage,)), address)[0])
        if len(bodies) >= _EW_CHAIN: flush()
    flush()
  def __call__(self, *bufs:HCQBuffer, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    del global_size, local_size, vals, kwargs
    fp16 = self.image.out_precision is RKEWOut.FP16
    if self.image.constants and self.scratch:
      for i in range(len(self.image.constants) // 2):
        if i >= len(self.scratch): break
        val = struct.unpack_from("<e", self.image.constants, i * 2)[0]
        ctypes.memset(int(self.scratch[i].va_addr), 0, self.scratch[i].size)
        hbits = struct.pack("<e", val)
        if fp16:
          # contiguous half fill for the largest EW count using this scratch
          n = max((op.count for op in self.image.ew_ops), default=self.scratch[i].size // 2)
          n = min(n, self.scratch[i].size // 2)
          for lane in range(n):
            ctypes.memmove(int(self.scratch[i].va_addr) + lane * 2, hbits, 2)
        else:
          nslots = self.scratch[i].size // _EW_SLOT_BYTES
          for s in range(nslots):
            for lane in range(_EW_CHUNK):
              ctypes.memmove(int(self.scratch[i].va_addr) + s * _EW_SLOT_BYTES + lane * 2, hbits, 2)
    for g in self.image.gathers:
      src_u16 = _arg_as_half_u16(bufs[g.src_index], max(g.offsets) + 1 if g.offsets else 0, g.itemsize)
      dst_mv = to_mv(int(self.scratch[g.dst_scratch].va_addr), self.scratch[g.dst_scratch].size)
      ctypes.memset(int(self.scratch[g.dst_scratch].va_addr), 0, self.scratch[g.dst_scratch].size)
      dst = np.frombuffer(dst_mv, dtype=np.uint16)
      if fp16:
        for i, off in enumerate(g.offsets): dst[i] = src_u16[off]
      else:
        for i, off in enumerate(g.offsets):
          slot, lane = divmod(i, _EW_CHUNK)
          dst[slot * (_EW_SLOT_BYTES // 2) + lane] = src_u16[off]
    for p in self.image.packs:
      src = _arg_as_half_u16(bufs[p.src_index], p.count, p.itemsize).view(np.float16)
      dst_mv = to_mv(int(self.scratch[p.dst_scratch].va_addr), self.scratch[p.dst_scratch].size)
      ctypes.memset(int(self.scratch[p.dst_scratch].va_addr), 0, self.scratch[p.dst_scratch].size)
      if fp16:
        ctypes.memmove(int(self.scratch[p.dst_scratch].va_addr), src[:p.count].tobytes(), p.count * 2)
      else:
        _pack_half_slots(src[:p.count], dst_mv, p.count)
    def address(kind:RKBufferKind, index:int) -> int:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return self._dma(bufs[index])
      if kind is RKBufferKind.SCRATCH:
        if index >= len(self.scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
        return self._dma(self.scratch[index])
      raise RuntimeError(f"unsupported RKBufferKind {kind}")
    start = time.perf_counter()
    if fp16: self._run_ew_ops_fp16(address)
    else:
      for op in self.image.ew_ops: self._run_ew_op_fp32(op, address)
    if (fill:=self.image.fill) is not None:
      v = struct.unpack_from("<e", self.image.constants, 0)[0] if self.image.constants else 0.0
      out_bits = np.full(fill.count, v, dtype=np.float16).tobytes()
      if fill.dst.kind is RKBufferKind.ARG:
        ctypes.memmove(int(bufs[fill.dst.index].va_addr), out_bits, len(out_bits))
      else:
        ctypes.memmove(int(self.scratch[fill.dst.index].va_addr), out_bits, len(out_bits))
    if (hout:=self.image.half_out) is not None:
      if fp16:
        out_bits = bytes(to_mv(int(self.scratch[hout.src_scratch].va_addr), hout.count * 2))
      else:
        mv = to_mv(int(self.scratch[hout.src_scratch].va_addr), self.scratch[hout.src_scratch].size)
        out_bits = _unpack_half_slots(mv, hout.count)
      if hout.dst.kind is RKBufferKind.ARG:
        ctypes.memmove(int(bufs[hout.dst.index].va_addr), out_bits, len(out_bits))
      else:
        ctypes.memmove(int(self.scratch[hout.dst.index].va_addr), out_bits, len(out_bits))
    return time.perf_counter()-start if wait else None

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(os.getenv("ROCKCHIP_DRM", "/dev/dri/card1"), os.O_RDWR)
    self.submit_count = 0
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer], RockchipProgram)
  def _gpu_alloc(self, size:int, flags:int=0) -> HCQBuffer:
    alloc = max(4096, (size+4095)&-4096)
    meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=alloc, flags=flags|rk.RKNPU_MEM_NON_CACHEABLE)
    mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=meta.handle, reserved=0, offset=0)
    return HCQBuffer(self.fd_ctl.mmap(0, alloc, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset), size, meta=meta)
  def _gpu_free(self, buf:HCQBuffer):
    FileIOInterface.munmap(int(buf.base.va_addr), max(4096, (buf.base.size+4095)&-4096))
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr)
  def reset_npu(self):
    try: rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=13, value=0)  # RKNPU_ACT_CLR_TOTAL_RW_AMOUNT
    except OSError: pass
    try: rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=rk.RKNPU_ACT_RESET, value=0)
    except OSError: pass
  def synchronize(self): pass
