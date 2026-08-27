from __future__ import annotations
import collections, ctypes, itertools, mmap, os, time, typing, weakref as wr
import numpy as np
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, suppress_finalizing, to_mv
from tinygrad.renderer.rockchip import (RKBufferKind, RKEWMode, RockchipRenderer, RockchipBoolRenderer, decode_image, patch_stage,
  emit_ew_stage, emit_cmac_stage, RKArg, RKGather, RKEWOp, RKCMAC, _MAX_EW_ELEMS_FP16)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer

_PC_TAIL, _CMD_BUF_MIN, _TASK_BUF_MIN = 4, 65536, 16384
_CMD_PREFETCH_GUARD = mmap.PAGESIZE
_SUBMIT_TIMEOUT_MS = max(1, int(os.getenv("ROCKCHIP_SUBMIT_TIMEOUT_MS", "6000")))
_SUBMIT_RETRIES = max(0, int(os.getenv("ROCKCHIP_SUBMIT_RETRIES", "4")))
_MAX_EW_GROUP_OPS = 48
_TASK_DESC_BYTES = ctypes.sizeof(rk.struct_rknpu_task)

def _pc(target:int, reg:int, value:int=0) -> int: return (target << 48) | ((value & 0xffffffff) << 16) | reg
def _align_up(value:int, alignment:int) -> int: return (value + alignment - 1) & ~(alignment - 1)
class RockchipAllocator(LRUAllocator['RockchipDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer: return self.dev._gpu_alloc(size)
  def _copyin(self, dest:HCQBuffer, src:memoryview):
    ctypes.memmove(int(dest.va_addr), from_mv(src), src.nbytes)
    self.dev._sync_buffer(dest, rk.RKNPU_MEM_SYNC_TO_DEVICE)
  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev._sync_buffer(src, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    ctypes.memmove(from_mv(dest), int(src.va_addr), dest.nbytes)
  def _as_buffer(self, src:HCQBuffer):
    self.dev._sync_buffer(src, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    return to_mv(int(src.va_addr), src.size)
  def _offset(self, buf:HCQBuffer, size:int, offset:int): return buf.offset(offset, size)
  def _free(self, buf:HCQBuffer, options:BufferSpec): self.dev._gpu_free(buf)

class RockchipProgram(Program['RockchipDevice']):
  def __init__(self, dev:'RockchipDevice', obj:TinyELF):
    self.dev, self.name, self.image = dev, obj.name, decode_image(obj.lib)
    self._scratch_offsets:list[int] = []
    self._scratch_size = 0
    for spec in self.image.scratch:
      self._scratch_size = _align_up(self._scratch_size, spec.alignment)
      self._scratch_offsets.append(self._scratch_size)
      self._scratch_size += spec.size
    self._buffers:dict[str,HCQBuffer] = {}
    self.scratch:tuple[HCQBuffer, ...] = ()
    self.submit_count = 0
    self._pcchain_bodies:tuple[tuple[int, ...], ...]|None = None
    self._scratch_ew_bodies:dict[tuple[RKEWOp, ...], tuple[tuple[int, ...], ...]] = {}
    dev._touch_program(self)
    self._ensure_scratch()

  def _ensure_scratch(self) -> None:
    if self.scratch or not self._scratch_size: return
    arena=self._ensure_buffer("scratch",self._scratch_size,self._scratch_size)
    self.scratch = tuple(arena.offset(offset, spec.size)
      for offset,spec in zip(self._scratch_offsets, self.image.scratch))

  def _release_resources(self) -> None:
    self.scratch,self._pcchain_bodies = (),None
    for buf in (buffers:=getattr(self,"_buffers",{})).values(): self.dev._gpu_free(buf)
    buffers.clear()
    getattr(self,"_scratch_ew_bodies",{}).clear()

  @suppress_finalizing
  def __del__(self):
    self._release_resources()
    self.dev._forget_program(self)

  def _dma(self, buf:HCQBuffer) -> int: return int(buf.meta.dma_addr)+int(buf.va_addr)-int(buf.base.va_addr)

  def _ensure_buffer(self, attr:str, size:int, minimum:int, flags:int=0) -> HCQBuffer:
    if (buf:=self._buffers.get(attr)) is None or buf.size < size:
      new = self.dev._gpu_alloc(max(size, minimum), flags)
      self._buffers[attr] = new
      if buf is not None: self.dev._gpu_free(buf)
      return new
    return buf

  def _submit(self, cmd:HCQBuffer, task:HCQBuffer, n:int, standalone:bool=False, retry:bool=True) -> None:
    subcores = ((0, n),) if standalone else ((0, n), (n, 0), (n, 0))
    retries = _SUBMIT_RETRIES if retry else 0
    for attempt in range(retries+1):
      try:
        for buffer in (cmd, task): self.dev._sync_buffer(buffer, rk.RKNPU_MEM_SYNC_TO_DEVICE)
        rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
          flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=_SUBMIT_TIMEOUT_MS,
          task_start=0, task_number=n, task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr,
          regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
          subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*x) for x in subcores)))
        break
      except TimeoutError:
        self.dev.timeout_retries += 1
        if attempt == retries: raise
        self.dev.reset_npu()
    self.submit_count += 1
    self.dev.submit_count += 1
    self.dev.task_count += n

  # Submit contiguous FP16 EW tasks as one blocking PC chain, or one stateful DPU/CMAC body with its direct PC tail.
  def _submit_bodies(self, bodies:typing.Iterable[tuple[int, ...]], standalone:bool=False, cmac:bool=False) -> None:
    """Materialize one physical command/task batch while retaining each submission ABI."""
    bodies=tuple(bodies); sizes=tuple(map(len,bodies)); n=len(bodies)  # noqa: E702
    if not sizes or not all(0<s<1<<16 for s in sizes) or standalone and n!=1 or cmac and not standalone: raise ValueError("invalid NPU command body")  # noqa: E501
    if not standalone and self._pcchain_bodies==bodies and all(name in self._buffers for name in ("cmd","task")): self._submit(self._buffers["cmd"],self._buffers["task"],n); return  # noqa: E501,E702
    tail_size=_PC_TAIL if cmac or not standalone else 1; offsets=(0,*itertools.accumulate(_align_up(size+tail_size,2) for size in sizes))  # noqa: E501,E702
    prefix="standalone_" if standalone else ""; cmd_size=offsets[-1]*8+_CMD_PREFETCH_GUARD  # noqa: E702
    cmd=self._ensure_buffer(f"{prefix}cmd",cmd_size,_CMD_BUF_MIN); task=self._ensure_buffer(f"{prefix}task",n*_TASK_DESC_BYTES,_TASK_BUF_MIN,rk.RKNPU_MEM_KERNEL_MAPPING)  # noqa: E501,E702
    ctypes.memset(int(cmd.va_addr),0,cmd_size); base_dma=self._dma(cmd)  # noqa: E702
    for i,(body,size) in enumerate(zip(bodies,sizes)):
      base=offsets[i]; ctypes.memmove(int(cmd.va_addr)+base*8,(ctypes.c_uint64*size)(*body),size*8)  # noqa: E702
      # REGISTER_AMOUNTS=0 terminates a chain. Keep its speculative base-address fetch inside the mapped
      # zero-filled guard page: RK3588 can otherwise race completion with an IOMMU read from address zero.
      next_addr=(base_dma+(offsets[i+1] if i+1<n else offsets[-1])*8)&0xfffffff0; next_amount=sizes[i+1] if i+1<n else 0  # noqa: E702
      tail=(_pc(0x0001,0),_pc(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS),_pc(rk.TARGET_VERSION,0),_pc(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)) if cmac else (_pc(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0x18),) if standalone else (_pc(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,next_addr),_pc(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,next_amount),_pc(rk.TARGET_VERSION,0),_pc(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0x18))  # noqa: E501
      ctypes.memmove(int(cmd.va_addr)+(base+size)*8,(ctypes.c_uint64*len(tail))(*tail),len(tail)*8)
      desc=rk.struct_rknpu_task(0,0 if cmac else 4,0xd if cmac else 0x18,0x300,0x1ffff,0,size if cmac else size+len(tail),0,base_dma+base*8)  # noqa: E501
      ctypes.memmove(int(task.va_addr)+i*_TASK_DESC_BYTES,ctypes.addressof(desc),_TASK_DESC_BYTES)
    if not standalone: self._pcchain_bodies=bodies
    if standalone: self.dev.reset_npu()
    try: self._submit(cmd,task,n,standalone=standalone,**({"retry":False} if cmac else {}))
    finally:
      if standalone: self.dev.reset_npu()

  def _run_ew_ops(self, address, ops:tuple[RKEWOp, ...]) -> None:
    if not ops: return
    M=RKEWMode
    if all(op.mode==M.INT16 and not op.submit_barrier and
           all(arg.kind is RKBufferKind.SCRATCH for arg in (op.dst,op.lhs,op.rhs)) for op in ops):
      if (cached:=self._scratch_ew_bodies.get(ops)) is None: self._scratch_ew_bodies[ops]=cached=tuple(stage for op in ops for stage in self._tile(op,_MAX_EW_ELEMS_FP16,address))  # noqa: E501
      self._submit_bodies(cached)
      return
    def run_group(group:tuple[RKEWOp, ...]) -> None:
      chain:list[tuple[int, ...]] = []
      body_precision = 0
      def flush_chain(reset:bool=False) -> None:
        nonlocal body_precision
        if chain:
          self._submit_bodies(chain)
          if reset or body_precision >= 64: self.dev.reset_npu()
          chain.clear()
        body_precision = 0
      for op in group:
        mode=op.mode
        if mode==M.COMPARE:
          if body_precision: flush_chain(True)
          flush_chain()
          for body in self._tile(op,_MAX_EW_ELEMS_FP16,address):
            self._submit_bodies((body,),True)
          continue
        precision=128 if mode==M.HALF_TO_FLOAT else 64 if mode in (M.HALF_TO_INT32,M.INT32_TO_HALF) else \
          16 if mode==M.INT16 else 32 if mode==M.INT32 else 0
        if chain and precision!=body_precision and not (mode==M.INT16_TO_INT32 and body_precision in (0,16)):
          flush_chain(not precision and mode!=M.INT16_TO_INT32 and bool(body_precision))
        if mode in (M.HALF_TO_FLOAT,M.HALF_TO_INT32,M.INT32_TO_HALF): chain.append(patch_stage(emit_ew_stage(op),address))
        else:
          limit,itemsize=(_MAX_EW_ELEMS_FP16//2,4) if mode==M.INT32 else (8,1) if mode==M.INT16_TO_INT32 else (_MAX_EW_ELEMS_FP16,2)  # noqa: E501
          flags=dict(dst_step=4,src_step=2) if mode==M.INT16_TO_INT32 else {}
          chain.extend(self._tile(op,limit,address,itemsize,**flags))
        body_precision=precision
        if precision==128 and len(chain)>=16: flush_chain()
      flush_chain()
    cuts=(0,*(i for i,op in enumerate(ops) if i and op.submit_barrier),len(ops))
    for begin,end in zip(cuts,cuts[1:]):
      group=ops[begin:end]
      split=group[0].mode in (M.STATEFUL,M.HALF_TO_INT16,M.FLOAT_TO_HALF) and len(group)>_MAX_EW_GROUP_OPS and max(op.count for op in group)<=_MAX_EW_ELEMS_FP16 and not any(op.mode in (M.INT32,M.INT16_TO_INT32,M.HALF_TO_INT32,M.INT32_TO_HALF,M.HALF_TO_FLOAT) for op in group)  # noqa: E501
      tiled=group[0].mode in (M.STATEFUL,M.FLOAT_TO_HALF) and group[0].count>_MAX_EW_ELEMS_FP16 and all(op.count==group[0].count and op.mode in (M.HALF,M.STATEFUL,M.FLOAT_TO_HALF) for op in group)  # noqa: E501
      if split:
        for start in range(0,len(group),_MAX_EW_GROUP_OPS):
          op=group[start]
          run_group((op._replace(submit_barrier=False,mode=M.STATEFUL if op.mode==M.HALF else op.mode),*group[start+1:start+_MAX_EW_GROUP_OPS]))  # noqa: E501
      elif tiled:
        tiles=(self._tile(op,_MAX_EW_ELEMS_FP16,address,mode=M.STATEFUL if i==0 and op.mode==M.HALF else op.mode)
               for i,op in enumerate(group))
        for bodies in zip(*tiles): self._submit_bodies(bodies)
      else: run_group(group)

  def _tile(self, op:RKEWOp, limit:int, address, itemsize:int=2, dst_step:int=1, src_step:int=1, **flags):
    for start in range(0, op.count, limit):
      args=tuple(arg._replace(addend=arg.addend+start*itemsize*(dst_step if i==0 else src_step)) for i,arg in enumerate((op.dst,op.lhs,op.rhs)))  # noqa: E501
      yield patch_stage(emit_ew_stage(op._replace(dst=args[0],lhs=args[1],rhs=args[2],count=min(limit,op.count-start),**flags)),address)

  def __call__(self, *bufs:HCQBuffer, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    del global_size, local_size, vals, kwargs
    self.dev._touch_program(self)
    self._ensure_scratch()
    def buffer(kind:RKBufferKind, index:int) -> HCQBuffer:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return bufs[index]
      if index >= len(self.scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
      return self.scratch[index]
    self.dev._sync_buffers(bufs, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    linear:dict[int, np.ndarray] = {}
    def view(arg:RKArg,dtype,itemsize:int) -> np.ndarray:
      if arg.addend%itemsize: raise RuntimeError("unaligned RKGather")
      raw=buffer(arg.kind,arg.index)
      return np.frombuffer(to_mv(int(raw.va_addr),raw.size),dtype=dtype)[arg.addend//itemsize:]
    def apply_gathers(gathers:tuple[RKGather, ...]) -> None:
      for gather in gathers:
        lane_dtype = {1:np.uint8, 2:np.uint16, 4:np.uint32}[gather.itemsize]
        dst,lanes=view(gather.dst,lane_dtype,gather.itemsize),linear.setdefault(gather.count,np.arange(gather.count,dtype=np.intp))
        if gather.index is not None:
          assert gather.src is not None
          src=view(gather.src,lane_dtype,gather.itemsize)
          idx=view(gather.index,{2:np.int16,4:np.int32}[gather.index_itemsize],gather.index_itemsize)[:gather.count].astype(np.intp)
          if len(idx)!=gather.count or len(src)<(gather.count if gather.scatter else gather.src_count) or len(dst)<(gather.dst_count if gather.scatter else gather.count): raise RuntimeError("runtime RKGather exceeds buffer")  # noqa: E501
          valid=(idx>=0)&(idx<(gather.dst_count if gather.scatter else gather.index_limit or gather.src_count))
          if gather.scatter:
            dst[idx[valid]]=src[lanes[valid]]
            continue
          index=gather.base+lanes*gather.lane_stride+idx*gather.index_scale
          valid&=(index>=0)&(index<gather.src_count)
          dst_index=lanes
          dst[dst_index]=gather.fill_bits
        else:
          dest=buffer(gather.dst.kind,gather.dst.index)
          if gather.dst.kind is RKBufferKind.SCRATCH and not gather.partial and not gather.dst.addend and not gather.dst_addend: ctypes.memset(int(dest.va_addr),0,dest.size)  # noqa: E501
          dst_index=gather.dst_addend+lanes*gather.dst_stride
          if gather.values:
            dst[dst_index]=gather.values[0] if len(gather.values)==1 else gather.values
            continue
          assert gather.src is not None
          src=view(gather.src,lane_dtype,gather.itemsize)
          index=np.asarray(gather.offsets,dtype=np.intp) if gather.offsets else np.full(gather.count,gather.base,dtype=np.intp)
          valid=index>=0 if gather.offsets else np.ones(gather.count,dtype=np.bool_)
          if gather.offsets and not gather.partial: dst[dst_index]=gather.fill_bits
          for divisor,limit,stride in gather.axes: index+=(lanes//divisor%limit)*stride
        dst[dst_index[valid]]=src[index[valid]]
    cursor=next((i for i,op in enumerate(self.image.program) if not isinstance(op,RKGather) or op.scatter),len(self.image.program))
    apply_gathers(self.image.program[:cursor])  # type: ignore[arg-type]
    self.dev._sync_buffers((*bufs,*((arena,) if (arena:=self._buffers.get("scratch")) is not None else ())),rk.RKNPU_MEM_SYNC_TO_DEVICE)
    def address(kind:RKBufferKind,index:int) -> int: return self._dma(buffer(kind,index))
    start = time.perf_counter()
    ew_ops=tuple(op for op in self.image.program if isinstance(op,RKEWOp))
    native_int16=any(op.mode in (RKEWMode.INT16,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT16) for op in ew_ops)
    if ew_ops and (self.dev._native_int16 and not native_int16 or any(op.mode==RKEWMode.HALF_TO_FLOAT for op in ew_ops)): self.dev.reset_npu()  # noqa: E501
    for _,items in itertools.groupby(self.image.program[cursor:],type):
      group=tuple(items); current=group[0]  # noqa: E702
      if isinstance(current,RKCMAC): self._submit_bodies((patch_stage(emit_cmac_stage(current),address),),True,True)
      elif isinstance(current,RKEWOp): self._run_ew_ops(address,group)  # type: ignore[arg-type]
      else:
        touched={(arg.kind,arg.index) for gather in group for arg in (gather.src,gather.index,gather.dst) if arg is not None}  # type: ignore[union-attr]  # noqa: E501
        self.dev._sync_buffers(tuple(buffer(kind,index) for kind,index in touched),rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        apply_gathers(group)  # type: ignore[arg-type]
        self.dev._sync_buffers(tuple(buffer(g.dst.kind,g.dst.index) for g in group),rk.RKNPU_MEM_SYNC_TO_DEVICE)  # type: ignore[union-attr]  # noqa: E501
    if ew_ops: self.dev._native_int16 = native_int16
    return time.perf_counter()-start if wait else None

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(os.getenv("ROCKCHIP_DRM", "/dev/dri/card1"), os.O_RDWR)
    self.submit_count = self.task_count = self.timeout_retries = 0
    self._native_int16 = False
    self.reset_npu()
    self._program_resource_limit = max(1, int(os.getenv("ROCKCHIP_PROGRAM_CACHE", "32")))
    self._program_resources:collections.OrderedDict[int, wr.ReferenceType[RockchipProgram]] = collections.OrderedDict()
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer, RockchipBoolRenderer], RockchipProgram)
  def _touch_program(self, program:RockchipProgram) -> None:
    self._program_resources.pop(id(program), None)
    self._program_resources[id(program)] = wr.ref(program)
    while len(self._program_resources) > self._program_resource_limit:
      _, reference = self._program_resources.popitem(last=False)
      if (old:=reference()) is not None: old._release_resources()
  def _forget_program(self, program:RockchipProgram) -> None: self._program_resources.pop(id(program), None)
  def _gpu_alloc(self, size:int, flags:int=0) -> HCQBuffer:
    alloc = max(4096, (size+4095)&-4096)
    try: meta = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl,size=alloc,flags=flags|rk.RKNPU_MEM_NON_CONTIGUOUS|rk.RKNPU_MEM_CACHEABLE|rk.RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT)  # noqa: E501
    except OSError as exc: raise MemoryError(f"RKNPU GEM allocation failed for {alloc} bytes") from exc
    try:
      mapping = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=meta.handle, reserved=0, offset=0)
      mapped = self.fd_ctl.mmap(0, alloc, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, mapping.offset)
    except Exception as exc:
      try: rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=meta.handle, reserved=0, obj_addr=meta.obj_addr)
      except (OSError, RuntimeError): pass
      raise MemoryError(f"RKNPU GEM mapping failed for {alloc} bytes") from exc
    return HCQBuffer(mapped, size, meta=meta)
  def _sync_buffer(self, buf:HCQBuffer, flags:int):
    rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.fd_ctl, flags=flags, reserved=0, obj_addr=buf.meta.obj_addr, offset=0, size=buf.meta.size)
  def _sync_buffers(self, bufs:tuple[HCQBuffer, ...], flags:int):
    unique:dict[int,HCQBuffer] = {}
    for buf in bufs: unique.setdefault(buf.meta.obj_addr,buf)
    for buf in unique.values(): self._sync_buffer(buf,flags)
  def _gpu_free(self, buf:HCQBuffer):
    FileIOInterface.munmap(int(buf.base.va_addr), max(4096, (buf.base.size+4095)&-4096))
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr)
  def reset_npu(self):
    rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=rk.RKNPU_ACT_RESET, value=0)
    self._native_int16 = False
  def synchronize(self): pass
