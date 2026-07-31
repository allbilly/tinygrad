from __future__ import annotations
import ctypes, mmap, os, time
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, suppress_finalizing, to_mv
from tinygrad.renderer.rockchip import RKBufferKind, RKEngine, RK_STAGE_RESET, RockchipRenderer, decode_image, patch_image
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer

_TASK = {RKEngine.DPU:(4, 0x18, 0x300), RKEngine.CMAC:(0, 0x0d, 0x300),
         RKEngine.CONV:(0, 0x0d, 0x300), RKEngine.PPU:(1, 0x60, 0xc00)}

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
    self.constants = dev._gpu_alloc(len(self.image.constants)) if self.image.constants else None
    if self.constants is not None: ctypes.memmove(int(self.constants.va_addr), self.image.constants, len(self.image.constants))

  @suppress_finalizing
  def __del__(self):
    for buf in self.scratch: self.dev._gpu_free(buf)
    if self.constants is not None: self.dev._gpu_free(self.constants)

  def _dma(self, buf:HCQBuffer) -> int:
    if buf.meta is None: raise RuntimeError("Rockchip program requires an NPU DMA buffer")
    return int(buf.meta.dma_addr) + int(buf.va_addr) - int(buf.base.va_addr)

  def __call__(self, *bufs:HCQBuffer, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    del global_size, local_size, vals, kwargs
    def address(kind:RKBufferKind, index:int) -> int:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return self._dma(bufs[index])
      if kind is RKBufferKind.SCRATCH:
        if index >= len(self.scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
        return self._dma(self.scratch[index])
      if self.constants is None or index >= len(self.image.constants): raise RuntimeError(f"RKImage constant offset {index} is invalid")
      return self._dma(self.constants) + index

    start = time.perf_counter()
    for stage, commands in zip(self.image.stages, patch_image(self.image, address)):
      if stage.flags & RK_STAGE_RESET: self.dev.reset_npu()
      cmd = self.dev._gpu_alloc(len(commands)*8)
      task = self.dev._gpu_alloc(ctypes.sizeof(rk.struct_rknpu_task), rk.RKNPU_MEM_KERNEL_MAPPING)
      try:
        ctypes.memmove(int(cmd.va_addr), (ctypes.c_uint64*len(commands))(*commands), len(commands)*8)
        op_idx, enable_mask, int_mask = _TASK[stage.engine]
        descriptor = rk.struct_rknpu_task(flags=0, op_idx=op_idx, enable_mask=enable_mask, int_mask=int_mask, int_clear=0x1ffff,
          int_status=0, regcfg_amount=len(commands), regcfg_offset=0, regcmd_addr=self._dma(cmd))
        ctypes.memmove(int(task.va_addr), ctypes.addressof(descriptor), ctypes.sizeof(descriptor))
        rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
          flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=6000, task_start=0, task_number=1,
          task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr, regcfg_obj_addr=0, task_base_addr=0, user_data=0,
          core_mask=1, fence_fd=-1, subcore_task=(rk.struct_rknpu_subcore_task*5)(rk.struct_rknpu_subcore_task(0,1)))
      finally:
        self.dev._gpu_free(cmd)
        self.dev._gpu_free(task)
    return time.perf_counter()-start if wait else None

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface("/dev/dri/card1", os.O_RDWR)
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer], RockchipProgram)

  def _gpu_alloc(self, size:int, flags:int=0) -> HCQBuffer:
    alloc_size = max(4096, (size+4095)&-4096)
    meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=alloc_size, flags=flags|rk.RKNPU_MEM_NON_CACHEABLE)
    mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=meta.handle, reserved=0, offset=0)
    va = self.fd_ctl.mmap(0, alloc_size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset)
    return HCQBuffer(va, size, meta=meta)

  def _gpu_free(self, buf:HCQBuffer):
    FileIOInterface.munmap(int(buf.base.va_addr), max(4096, (buf.base.size+4095)&-4096))
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr)

  def reset_npu(self): rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=rk.RKNPU_ACT_RESET, value=0)
  def synchronize(self): pass
