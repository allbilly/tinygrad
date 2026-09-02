from __future__ import annotations
import ctypes, itertools, mmap, os, threading, time, typing
import numpy as np
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, round_up, to_mv
from tinygrad.renderer.rockchip import (RKBufferKind, RKEWMode, RockchipRenderer, RockchipBoolRenderer, decode_image,
  emit_ew_stage, emit_cmac_stage, RKArg, RKGather, RKEWOp, RKCMAC, _MAX_EW_ELEMS_FP16)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer

_PC_TAIL, _CMD_BUF_MIN, _TASK_BUF_MIN, _MAX_PC_TASKS = 4, 65536, 16384, 0xfff
_CMD_PREFETCH_GUARD = mmap.PAGESIZE
_SUBMIT_TIMEOUT_MS = max(1, int(os.getenv("ROCKCHIP_SUBMIT_TIMEOUT_MS", "6000")))
_MAX_EW_GROUP_OPS = 48
_EW_MODE_INFO=((128,0,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1),(16,_MAX_EW_ELEMS_FP16,2,1,1),(32,_MAX_EW_ELEMS_FP16//2,4,1,1),(0,8,1,4,2),(64,0,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1),(64,0,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1))  # noqa: E501
_TASK_DESC_BYTES = ctypes.sizeof(rk.struct_rknpu_task)

def _pc(target:int, reg:int, value:int=0) -> int: return (target << 48) | ((value & 0xffffffff) << 16) | reg
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
    self._scratch_offsets=(0,*itertools.accumulate(round_up(size,4096) for size in self.image.scratch)); self._ew_modes={op.mode for op in self.image.program if isinstance(op,RKEWOp)}  # noqa: E501,E702

  def _dma(self, buf:HCQBuffer) -> int: return int(buf.meta.dma_addr)+int(buf.va_addr)-int(buf.base.va_addr)

  def _submit(self, cmd:HCQBuffer, task:HCQBuffer, n:int, standalone:bool=False) -> None:
    if not 0<n<=_MAX_PC_TASKS: raise ValueError("invalid NPU task count")
    subcores = ((0, n),) if standalone else ((0, n), (n, 0), (n, 0))
    self.dev._check_healthy()
    try:
      for buffer in (cmd, task): self.dev._sync_buffer(buffer, rk.RKNPU_MEM_SYNC_TO_DEVICE)
      rk.DRM_IOCTL_RKNPU_SUBMIT(self.dev.fd_ctl,
        flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG, timeout=_SUBMIT_TIMEOUT_MS,
        task_start=0, task_number=n, task_counter=0, priority=0, task_obj_addr=task.meta.obj_addr,
        regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=1, fence_fd=-1,
        subcore_task=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*x) for x in subcores)))
    except TimeoutError as exc:
      self.dev.timeout_retries,self.dev._poisoned=self.dev.timeout_retries+1,True
      raise RuntimeError("RKNPU submit timed out; platform NPU reset or power cycle required") from exc
    self.dev.submit_count += 1; self.dev.task_count += n  # noqa: E702

  # Submit contiguous FP16 EW tasks as one blocking PC chain, or one stateful DPU/CMAC body with its direct PC tail.
  def _submit_bodies(self, bodies:typing.Iterable[tuple[int, ...]], standalone:bool=False, cmac:bool=False) -> None:
    """Materialize one physical command/task batch while retaining each submission ABI."""
    bodies=tuple(bodies); sizes=tuple(map(len,bodies)); n=len(bodies)  # noqa: E702
    if not sizes or not all(0<s<1<<16 for s in sizes) or standalone and n!=1 or cmac and not standalone: raise ValueError("invalid NPU command body")  # noqa: E501
    tail_size=_PC_TAIL if cmac or not standalone else 1; offsets=(0,*itertools.accumulate(round_up(size+tail_size,2) for size in sizes)); cmd_size=offsets[-1]*8+_CMD_PREFETCH_GUARD  # noqa: E501,E702
    cmd,task=self.dev._replace_submit_buffers(cmd_size,n*_TASK_DESC_BYTES)
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
    if standalone: self.dev.reset_npu()
    try: self._submit(cmd,task,n,standalone=standalone)
    finally:
      if standalone and not self.dev._poisoned: self.dev.reset_npu()

  def _run_ew_ops(self, address, ops:tuple[RKEWOp, ...], rearm:bool=False) -> None:
    if not ops: return
    M=RKEWMode; program_modes=getattr(self,"_ew_modes",None) or {op.mode for op in ops}  # noqa: E702
    def run_group(group:tuple[RKEWOp, ...], pc_limit:int) -> None:
      chain:list[tuple[int, ...]]=[]; precision=0  # noqa: E702
      def flush_chain(reset:bool=False) -> None:
        if not chain: return
        for chunk in itertools.batched(chain,pc_limit): self._submit_bodies(chunk)
        if reset or precision>=64 and not rearm: self.dev.reset_npu()
        chain.clear()
      for op in group:
        mode=op.mode
        if mode==M.COMPARE:
          flush_chain(bool(precision))
          for body in self._tile(op,_MAX_EW_ELEMS_FP16,address): self._submit_bodies((body,),True)
          continue
        next_precision,limit,itemsize,dst_step,src_step=_EW_MODE_INFO[mode]
        if chain and next_precision!=precision and not (mode==M.INT16_TO_INT32 and precision in (0,16)):
          flush_chain(not next_precision and mode!=M.INT16_TO_INT32 and bool(precision) and precision!=16)
        chain.extend((emit_ew_stage(op,address),) if not limit else self._tile(op,limit,address,itemsize,dst_step,src_step,mode=M.BOUNDED if not chain and mode in (M.HALF,M.BOUNDED) else M.HALF if mode==M.BOUNDED else mode))  # noqa: E501
        precision=next_precision
        if precision==128 and len(chain)>=16: flush_chain()
      flush_chain()
    cuts=tuple(sorted({0,len(ops),*(j for i,o in enumerate(ops) if o.submit_barrier for j in ((i,) if o.mode!=M.BOUNDED else (i,i+1)) if j)}))
    for begin,end in zip(cuts,cuts[1:]):
      section=ops[begin:end]; limit=_MAX_EW_GROUP_OPS if M.HALF in program_modes and M.INT16 in program_modes or section[0].mode in (M.BOUNDED,M.HALF_TO_INT16,M.FLOAT_TO_HALF) and len(section)>_MAX_EW_GROUP_OPS and max(op.count for op in section)<=_MAX_EW_ELEMS_FP16 and not any(op.mode in (M.INT32,M.INT16_TO_INT32,M.HALF_TO_INT32,M.INT32_TO_HALF,M.HALF_TO_FLOAT) for op in section) else _MAX_PC_TASKS  # noqa: E501,E702
      for group in map(tuple,itertools.batched(section,limit)):
        tiled=len(group)>1 and group[0].mode in (M.HALF,M.BOUNDED,M.FLOAT_TO_HALF) and group[0].count>_MAX_EW_ELEMS_FP16 and all(op.count==group[0].count and op.mode in (M.HALF,M.BOUNDED,M.FLOAT_TO_HALF) for op in group)  # noqa: E501
        if tiled:
          tiles=(self._tile(op,_MAX_EW_ELEMS_FP16,address,mode=M.BOUNDED if i==0 and op.mode in (M.HALF,M.BOUNDED) else M.HALF if op.mode==M.BOUNDED else op.mode) for i,op in enumerate(group))  # noqa: E501
          for bodies in zip(*tiles): self._submit_bodies(bodies)
        else: run_group(tuple(op._replace(submit_barrier=False) for op in group),limit)

  def _tile(self, op:RKEWOp, limit:int, address, itemsize:int=2, dst_step:int=1, src_step:int=1, **flags):
    for start in range(0, op.count, limit):
      args=tuple(arg._replace(addend=arg.addend+start*itemsize*(dst_step if i==0 else src_step)) for i,arg in enumerate((op.dst,op.lhs,op.rhs)))  # noqa: E501
      yield emit_ew_stage(op._replace(dst=args[0],lhs=args[1],rhs=args[2],count=min(limit,op.count-start),**flags),address)

  def __call__(self, *bufs:HCQBuffer, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    del global_size, local_size, vals, kwargs
    with self.dev._lock: return self._run(bufs,wait)

  def _run(self, bufs:tuple[HCQBuffer, ...], wait:bool):
    arena=self.dev._ensure_buffer("scratch",self._scratch_offsets[-1],self._scratch_offsets[-1]) if self._scratch_offsets[-1] else None
    scratch=tuple(arena.offset(offset,size) for offset,size in zip(self._scratch_offsets,self.image.scratch)) if arena is not None else ()
    def buffer(kind:RKBufferKind, index:int) -> HCQBuffer:
      if kind is RKBufferKind.ARG:
        if index >= len(bufs): raise RuntimeError(f"RKImage argument slot {index} is not bound")
        return bufs[index]
      if index >= len(scratch): raise RuntimeError(f"RKImage scratch slot {index} is not declared")
      return scratch[index]
    self.dev._sync_buffers(bufs, rk.RKNPU_MEM_SYNC_FROM_DEVICE)
    linear:dict[int, np.ndarray] = {}
    def view(arg:RKArg,dtype,itemsize:int) -> np.ndarray:
      if arg.addend%itemsize: raise RuntimeError("unaligned RKGather")
      raw=buffer(arg.kind,arg.index)
      return np.frombuffer(to_mv(int(raw.va_addr),raw.size),dtype=dtype)[arg.addend//itemsize:]
    def apply_gathers(gathers:tuple[RKGather, ...]) -> None:
      for gather in gathers:
        lane_dtype={1:np.uint8,2:np.uint16,4:np.uint32}[gather.itemsize]
        dst,lanes=view(gather.dst,lane_dtype,gather.itemsize),linear.setdefault(gather.count,np.arange(gather.count,dtype=np.intp))
        if gather.index is None and gather.dst.kind is RKBufferKind.SCRATCH and not gather.partial and not gather.dst.addend and not gather.dst_addend: ctypes.memset(int((dest:=buffer(gather.dst.kind,gather.dst.index)).va_addr),0,dest.size)  # noqa: E501
        dst_index=gather.dst_addend+lanes*gather.dst_stride
        if gather.values:
          dst[dst_index]=gather.values[0] if len(gather.values)==1 else gather.values
          continue
        assert gather.src is not None
        src=view(gather.src,lane_dtype,gather.itemsize)
        if gather.index is not None:
          source_index=view(gather.index,{2:np.int16,4:np.int32}[gather.index_itemsize],gather.index_itemsize)[:gather.count].astype(np.intp)
          if len(source_index)!=gather.count or len(dst)<gather.count: raise RuntimeError("runtime RKGather exceeds buffer")
          source_index,dst_index=source_index,lanes
        else:
          source_index=np.asarray(gather.offsets,dtype=np.intp) if gather.offsets else np.full(gather.count,gather.base,dtype=np.intp)
          for divisor,limit,stride in gather.axes: source_index+=(lanes//divisor%limit)*stride
        valid=(source_index>=0)&(source_index<len(src))&(dst_index>=0)&(dst_index<len(dst))
        if not gather.partial and (gather.offsets or gather.index is not None and gather.dst.kind is RKBufferKind.SCRATCH): dst[dst_index]=gather.fill_bits  # noqa: E501
        dst[dst_index[valid]]=src[source_index[valid]]
    cursor=next((i for i,op in enumerate(self.image.program) if not isinstance(op,RKGather)),len(self.image.program))
    apply_gathers(self.image.program[:cursor])  # type: ignore[arg-type]
    self.dev._sync_buffers((*bufs,*((arena,) if arena is not None else ())),rk.RKNPU_MEM_SYNC_TO_DEVICE)
    def address(arg:RKArg) -> int: return self._dma(buffer(arg.kind,arg.index))+arg.addend
    start = time.perf_counter()
    ew_ops=tuple(op for op in self.image.program if isinstance(op,RKEWOp))
    native_int16=any(op.mode in (RKEWMode.INT16,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT16) for op in ew_ops)
    if ew_ops and (self.dev._native_int16 and not native_int16 or any(op.mode==RKEWMode.HALF_TO_FLOAT for op in ew_ops)): self.dev.reset_npu()  # noqa: E501
    for index,group in enumerate(groups:=tuple(tuple(items) for _,items in itertools.groupby(self.image.program[cursor:],type))):
      rearm=index+2<len(groups) and all(isinstance(op,RKEWOp) and op.mode==RKEWMode.INT32_TO_HALF for op in group) and isinstance(groups[index+1][0],RKGather) and (isinstance(next_op:=groups[index+2][0],RKEWOp) and next_op.mode==RKEWMode.BOUNDED or all(isinstance(op,RKEWOp) and op.mode==RKEWMode.INT32_TO_HALF for op in groups[index+2]))  # noqa: E501
      if isinstance(current:=group[0],RKCMAC): self._submit_bodies((emit_cmac_stage(current,address),),True,True)
      elif isinstance(current,RKEWOp): self._run_ew_ops(address,group,rearm)  # type: ignore[arg-type]
      else:
        touched={(arg.kind,arg.index) for gather in group for arg in (gather.src,gather.index,gather.dst) if arg is not None}  # type: ignore[union-attr]  # noqa: E501
        self.dev._sync_buffers(tuple(buffer(kind,index) for kind,index in touched),rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        apply_gathers(group)  # type: ignore[arg-type]
        self.dev._sync_buffers(tuple(buffer(g.dst.kind,g.dst.index) for g in group),rk.RKNPU_MEM_SYNC_TO_DEVICE)  # type: ignore[union-attr]  # noqa: E501
    if ew_ops: self.dev._native_int16 = ew_ops[-1].mode in (RKEWMode.INT16,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT16)
    return time.perf_counter()-start if wait else None

class RockchipDevice(Compiled):
  def __init__(self, device:str):
    self.fd_ctl = FileIOInterface(os.getenv("ROCKCHIP_DRM", "/dev/dri/card1"), os.O_RDWR)
    self.submit_count,self.task_count,self.timeout_retries,self._poisoned=0,0,0,False
    self._lock, self._buffers = threading.Lock(), dict[str,HCQBuffer]()
    self.reset_npu()
    super().__init__(device, RockchipAllocator(self), [RockchipRenderer, RockchipBoolRenderer], RockchipProgram)
  def _check_healthy(self):
    if self._poisoned: raise RuntimeError("RKNPU is unavailable after a submit timeout; platform NPU reset or power cycle required")
  def _ensure_buffer(self, attr:str, size:int, minimum:int, flags:int=0) -> HCQBuffer:
    if (buf:=self._buffers.get(attr)) is None or buf.size < size:
      new = self._gpu_alloc(max(size, minimum), flags)
      self._buffers[attr] = new
      if buf is not None: self._gpu_free(buf)
      return new
    return buf
  def _replace_submit_buffers(self, cmd_size:int, task_size:int) -> tuple[HCQBuffer,HCQBuffer]:
    old=tuple(self._buffers.get(name) for name in ("cmd","task")); fresh=(self._gpu_alloc(max(cmd_size,_CMD_BUF_MIN)),self._gpu_alloc(max(task_size,_TASK_BUF_MIN),rk.RKNPU_MEM_KERNEL_MAPPING))  # noqa: E501,E702
    self._buffers.update(zip(("cmd","task"),fresh))
    for buf in (x for x in old if x is not None): self._gpu_free(buf)
    return fresh
  def _gpu_alloc(self, size:int, flags:int=0) -> HCQBuffer:
    alloc = max(self._check_healthy() or 4096, (size+4095)&-4096)
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
    self._check_healthy() or rk.DRM_IOCTL_RKNPU_MEM_SYNC(self.fd_ctl, flags=flags, obj_addr=buf.meta.obj_addr, offset=0, size=buf.meta.size)
  def _sync_buffers(self, bufs:tuple[HCQBuffer, ...], flags:int):
    unique:dict[int,HCQBuffer] = {}
    for buf in bufs: unique.setdefault(buf.meta.obj_addr,buf)
    for buf in unique.values(): self._sync_buffer(buf,flags)
  def _gpu_free(self, buf:HCQBuffer):
    FileIOInterface.munmap(int(buf.base.va_addr), max(4096, (buf.base.size+4095)&-4096))
    if not self._poisoned: rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=buf.meta.handle, reserved=0, obj_addr=buf.meta.obj_addr)
  def reset_npu(self):
    self._check_healthy() or rk.DRM_IOCTL_RKNPU_ACTION(self.fd_ctl, flags=rk.RKNPU_ACT_RESET, value=0)
    self._native_int16 = False
  def finalize(self):
    for buf in self._buffers.values(): self._gpu_free(buf)
    self._buffers.clear()
