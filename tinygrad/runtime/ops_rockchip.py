from __future__ import annotations
import ctypes, mmap, os, struct, time
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, suppress_finalizing, to_mv
from tinygrad.renderer.rockchip import RKBufferKind, RK_STAGE_RESET, RockchipRenderer, decode_image, patch_image
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer

# PC-chain layout from allbilly/rk3588 experimental/pcchain.md + rockchip-2607 DPU fix:
# body without OPERATION_ENABLE + 4-qword PC tail; REGISTER_AMOUNTS = next body qword count.
# N=1 uses a null tail (same format). Single submit with task_number=N.
def _pc(t, r, v=0): return (t << 48) | ((v & 0xffffffff) << 16) | r

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
    self.submit_count = 0  # DRM_IOCTL_RKNPU_SUBMIT calls for this program
  @suppress_finalizing
  def __del__(self):
    for buf in getattr(self, "scratch", ()): self.dev._gpu_free(buf)
  def _dma(self, buf:HCQBuffer) -> int: return int(buf.meta.dma_addr)+int(buf.va_addr)-int(buf.base.va_addr)
  def _submit(self, streams:tuple[tuple[int, ...], ...]) -> None:
    """Submit N DPU stages as one PC-chained job (add3: producer→scratch→consumer in 1 ioctl).

    Emitters end each stream with PC_OPERATION_ENABLE; that word is replaced by the 4-qword PC
    tail (BASE_ADDRESS / REGISTER_AMOUNTS / VERSION / ENABLE). Last segment uses a null tail.
    Descriptor regcfg_amount = body qwords + 4. Segment stride is aligned to 2 qwords.
    """
    bodies = [list(s[:-1]) for s in streams]  # drop single-task ENABLE; chain tail re-supplies it
    offs, o = [], 0
    for b in bodies: offs.append(o); o += (len(b) + 5) // 2 * 2  # body + 4-qword tail, align 2
    cmd = self.dev._gpu_alloc(o * 8)
    task = self.dev._gpu_alloc(len(bodies) * ctypes.sizeof(rk.struct_rknpu_task), rk.RKNPU_MEM_KERNEL_MAPPING)
    try:
      ctypes.memset(int(cmd.va_addr), 0, o * 8)
      reg, dma, n = (ctypes.c_uint64 * o).from_address(int(cmd.va_addr)), self._dma(cmd), len(bodies)
      for i, b in enumerate(bodies):
        base = offs[i]
        for j, w in enumerate(b): reg[base + j] = w
        # PC tail: next segment DMA + next body length (0 / null for last)
        nxt, amt = ((dma + offs[i + 1] * 8) & 0xfffffff0, len(bodies[i + 1])) if i + 1 < n else (0, 0)
        reg[base + len(b) + 0] = _pc(rk.TARGET_PC_REG, rk.REG_PC_BASE_ADDRESS, nxt) if nxt else 0
        reg[base + len(b) + 1] = _pc(rk.TARGET_PC_REG, rk.REG_PC_REGISTER_AMOUNTS, amt)
        reg[base + len(b) + 2] = _pc(rk.TARGET_VERSION, 0)
        reg[base + len(b) + 3] = _pc(rk.TARGET_PC, rk.REG_PC_OPERATION_ENABLE, 0x18)
      # absolute-mode descriptors: regcfg_amount includes the PC tail
      descs = (rk.struct_rknpu_task * n)(*[rk.struct_rknpu_task(0, 4, 0x18, 0x300, 0x1ffff, 0, len(b) + 4, 0, dma + offs[i] * 8)
                                           for i, b in enumerate(bodies)])
      ctypes.memmove(int(task.va_addr), ctypes.addressof(descs), ctypes.sizeof(descs))
      # flags = PC | BLOCK | PINGPONG; single-core chain on subcore 0
      rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
        flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=6000, task_start=0, task_number=n,
        task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr, regcfg_obj_addr=0, task_base_addr=0, user_data=0,
        core_mask=1, fence_fd=-1, subcore_task=(rk.struct_rknpu_subcore_task*5)(
          rk.struct_rknpu_subcore_task(0, n), rk.struct_rknpu_subcore_task(n, 0), rk.struct_rknpu_subcore_task(n, 0)))
      self.submit_count += 1
      self.dev.submit_count += 1
    finally: self.dev._gpu_free(cmd); self.dev._gpu_free(task)
  def __call__(self, *bufs:HCQBuffer, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    del global_size, local_size, vals, kwargs
    # Scalar EW: image.constants holds packed fp16s; splat each into scratch[i].
    if self.image.constants and self.scratch:
      for i in range(len(self.image.constants) // 2):
        if i >= len(self.scratch): break
        val = struct.unpack_from("<e", self.image.constants, i * 2)[0]
        n = self.scratch[i].size // 2
        ctypes.memmove(int(self.scratch[i].va_addr), struct.pack("<e", val) * n, n * 2)
    # Gather: scratch[dst][i] = arg[src][offsets[i]] before EW.
    for g in self.image.gathers:
      src = to_mv(int(bufs[g.src_index].va_addr), bufs[g.src_index].size).cast('H')
      dst = to_mv(int(self.scratch[g.dst_scratch].va_addr), self.scratch[g.dst_scratch].size).cast('H')
      for i, off in enumerate(g.offsets): dst[i] = src[off]
    def address(kind:RKBufferKind, index:int) -> int:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return self._dma(bufs[index])
      if kind is RKBufferKind.SCRATCH:
        if index >= len(self.scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
        return self._dma(self.scratch[index])
      raise RuntimeError(f"unsupported RKBufferKind {kind}")
    start = time.perf_counter()
    if any(s.flags & RK_STAGE_RESET for s in self.image.stages): self.dev.reset_npu()
    self._submit(patch_image(self.image, address))
    return time.perf_counter()-start if wait else None

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(os.getenv("ROCKCHIP_DRM", "/dev/dri/card1"), os.O_RDWR)
    self.submit_count = 0  # total DRM_IOCTL_RKNPU_SUBMIT ioctls on this device
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer], RockchipProgram)
  def _gpu_alloc(self, size:int, flags:int=0) -> HCQBuffer:
    alloc = max(4096, (size+4095)&-4096)
    meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=alloc, flags=flags|rk.RKNPU_MEM_NON_CACHEABLE)
    mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=meta.handle, reserved=0, offset=0)
    return HCQBuffer(self.fd_ctl.mmap(0, alloc, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset), size, meta=meta)
  def _gpu_free(self, buf:HCQBuffer):
    FileIOInterface.munmap(int(buf.base.va_addr), max(4096, (buf.base.size+4095)&-4096))
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr)
  def reset_npu(self): rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=rk.RKNPU_ACT_RESET, value=0)
  def synchronize(self): pass
