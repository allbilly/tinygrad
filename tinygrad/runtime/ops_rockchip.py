from __future__ import annotations
import array, ctypes, functools, itertools, mmap, operator, os, threading, time, typing
from tinygrad.device import BufferSpec, Compiled, LRUAllocator, Program, TinyELF
from tinygrad.helpers import from_mv, round_up, to_mv
from tinygrad.renderer.rockchip import (RKBufferKind, RKEWMode, RockchipRenderer, RockchipBoolRenderer, decode_image,
  emit_ew_stage, emit_cmac_stage, RKArg, RKGather, RKEWOp, RKCMAC, _MAX_EW_ELEMS_FP16, _cmd as _pc)
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer, MMIOInterface

_PC_TAIL, _CMD_BUF_MIN, _TASK_BUF_MIN, _MAX_PC_TASKS = 4, 65536, 16384, 0xfff
_CMD_PREFETCH_GUARD = mmap.PAGESIZE
_SUBMIT_TIMEOUT_MS = max(1, int(os.getenv("ROCKCHIP_SUBMIT_TIMEOUT_MS", "6000")))
_MAX_EW_GROUP_OPS = 48
_EW_MODE_INFO=((128,0,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1),(16,_MAX_EW_ELEMS_FP16,2,1,1),(32,_MAX_EW_ELEMS_FP16//2,4,1,1),(0,8,1,4,2),(64,0,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1),(64,0,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1),(0,_MAX_EW_ELEMS_FP16,2,1,1))  # noqa: E501
_TASK_DESC_BYTES = ctypes.sizeof(rk.struct_rknpu_task)

_RAW_FORMATS, _INDEX_FORMATS = {1:"B",2:"H",4:"I"}, {2:"h",4:"i"}

def _rk_buffer_view(raw:HCQBuffer, arg:RKArg, fmt:str, itemsize:int) -> MMIOInterface:
  """Return the same typed suffix that NumPy frombuffer(...)[addend//itemsize:] exposed."""
  if arg.addend%itemsize: raise RuntimeError("unaligned RKGather")
  if raw.size%itemsize: raise RuntimeError("mis-sized RKGather view")
  start,count,_=slice(arg.addend//itemsize,None).indices(raw.size//itemsize)
  return raw.cpu_view().view(offset=start*itemsize,size=(count-start)*itemsize,fmt=fmt)

def _regular_gather_payload(gather:RKGather, src:MMIOInterface) -> array.array|memoryview|None:
  """Copy regular affine blocks, retaining a vectorized view for a single contiguous or strided leaf."""
  axes,code=tuple(sorted(gather.axes)),_RAW_FORMATS[gather.itemsize]
  if not axes or gather.base<0 or any(divisor<=0 or limit<=0 or stride<=0 for divisor,limit,stride in axes): return None
  periods=tuple(divisor*limit for divisor,limit,_ in axes)
  if gather.count%periods[-1] or any(divisor%period for period,(divisor,_,_) in zip(periods,axes[1:])) or \
     gather.base+sum((limit-1)*stride for _,limit,stride in axes)>=len(src): return None
  if len(axes)==1 and axes[0][:2]==(1,gather.count): return src.mv[gather.base:gather.base+gather.count*axes[0][2]:axes[0][2]]
  def block(index:int, base:int) -> array.array:
    divisor,limit,stride=axes[index]
    if index==0 and divisor==1: return array.array(code,src.mv[base:base+limit*stride:stride])
    chunks=(array.array(code,[value]) for value in src.mv[base:base+limit*stride:stride]) if index==0 else (block(index-1,base+i*stride) for i in range(limit))  # noqa: E501
    return functools.reduce(operator.iadd,(chunk*(divisor//(periods[index-1] if index else 1)) for chunk in chunks),array.array(code))
  return block(len(axes)-1,gather.base)*(gather.count//periods[-1])

def _apply_gathers(gathers:tuple[RKGather, ...], buffer:typing.Callable[[RKBufferKind,int],HCQBuffer]) -> None:
  """Apply host-addressed raw-lane movement; all numeric tensor operations remain on the NPU."""
  for gather in gathers:
    raw_dst,code=buffer(gather.dst.kind,gather.dst.index),_RAW_FORMATS[gather.itemsize]
    dst=_rk_buffer_view(raw_dst,gather.dst,code,gather.itemsize)
    dst_limit=len(dst)
    begin,step=(0,1) if gather.index is not None else (gather.dst_addend,gather.dst_stride)
    span,bounded=slice(begin,begin+gather.count*step,step),begin>=0 and begin+max(0,gather.count-1)*step<dst_limit
    src=_rk_buffer_view(buffer(gather.src.kind,gather.src.index),gather.src,code,gather.itemsize) if gather.src is not None else dst
    src_limit,offsets,overlap=len(src),gather.offsets,src.addr < dst.addr+dst.nbytes and dst.addr < src.addr+src.nbytes
    # Preserve every input before writes, including whole-scratch padding and index-buffer aliases.
    if gather.src is not None and overlap: src.mv=memoryview(bytes(src.mv)).cast(src.fmt)
    if gather.index is not None:
      indices=_rk_buffer_view(buffer(gather.index.kind,gather.index.index),gather.index,_INDEX_FORMATS[gather.index_itemsize],gather.index_itemsize)
      if len(indices)<gather.count or dst_limit<gather.count: raise RuntimeError("runtime RKGather exceeds buffer")
      offsets=tuple(indices.mv[:gather.count])
    if gather.index is None and gather.dst.kind is RKBufferKind.SCRATCH and not gather.partial and not gather.dst.addend and not gather.dst_addend:
      ctypes.memset(int(raw_dst.va_addr),0,raw_dst.size)
    if gather.values:
      if not bounded: raise IndexError("RKGather destination exceeds buffer")
      values=array.array(code,gather.values)
      dst.mv[span]=values*gather.count if len(values)==1 else values
      continue
    assert gather.src is not None
    fill=not gather.partial and bool(gather.offsets or gather.index is not None and gather.dst.kind is RKBufferKind.SCRATCH)
    if fill and not bounded and gather.count: raise IndexError("RKGather destination exceeds buffer")
    if gather.index is None and not offsets and bounded:
      if (payload:=_regular_gather_payload(gather,src)) is not None:
        dst.mv[span]=payload
        continue
    # Build one raw payload before assignment; inactive partial lanes retain their existing destination bits.
    if gather.index is None and offsets and bounded:
      picked=operator.itemgetter(*offsets)(src.mv) if gather.count>1 and min(offsets)>=0 and max(offsets)<src_limit else (
        src.mv[index] if 0<=index<src_limit else dst.mv[begin+lane*step] if gather.partial else gather.fill_bits for lane,index in enumerate(offsets))  # noqa: E501
      dst.mv[span]=array.array(code,picked)
      continue
    source_indices=offsets or (gather.base+sum((lane//divisor%limit)*stride for divisor,limit,stride in gather.axes) for lane in range(gather.count))
    writes=((begin+lane*step,src.mv[index] if 0<=index<src_limit else gather.fill_bits) for lane,index in enumerate(source_indices) if 0<=begin+lane*step<dst_limit and (fill or 0<=index<src_limit))  # noqa: E501
    for lane,value in writes: dst.mv[lane]=value

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
    self.dev._npu_clean=False; self.dev.submit_count += 1; self.dev.task_count += n  # noqa: E702

  # Submit contiguous FP16 EW tasks as one blocking PC chain, or one stateful DPU/CMAC body with its direct PC tail.
  def _submit_bodies(self, bodies:typing.Iterable[tuple[int, ...]], standalone:bool=False, cmac:bool=False) -> None:
    """Materialize one physical command/task batch while retaining each submission ABI."""
    bodies=tuple(bodies); sizes=tuple(map(len,bodies)); n=len(bodies)  # noqa: E702
    if not sizes or not all(0<s<1<<16 for s in sizes) or standalone and n!=1 or cmac and not standalone: raise ValueError("invalid NPU command body")  # noqa: E501
    tail_size=_PC_TAIL if cmac or not standalone else 1; offsets=(0,*itertools.accumulate(round_up(size+tail_size,2) for size in sizes)); cmd_size=offsets[-1]*8+_CMD_PREFETCH_GUARD  # noqa: E501,E702
    cmd,task=self.dev._replace_submit_buffers(cmd_size,n*_TASK_DESC_BYTES)
    ctypes.memset(int(cmd.va_addr),0,cmd_size); base_dma=self._dma(cmd); words=array.array("Q"); descs=[]  # noqa: E702
    for i,(body,size) in enumerate(zip(bodies,sizes)):
      base=offsets[i]; words.extend(itertools.repeat(0,base-len(words))); words.extend(body)  # noqa: E702
      # REGISTER_AMOUNTS=0 terminates a chain. Keep its speculative base-address fetch inside the mapped
      # zero-filled guard page: RK3588 can otherwise race completion with an IOMMU read from address zero.
      next_addr=(base_dma+(offsets[i+1] if i+1<n else offsets[-1])*8)&0xfffffff0; next_amount=sizes[i+1] if i+1<n else 0  # noqa: E702
      tail=(_pc(0x0001,0),_pc(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS),_pc(rk.TARGET_VERSION,0),_pc(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd)) if cmac else (_pc(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0x18),) if standalone else (_pc(rk.TARGET_PC_REG,rk.REG_PC_BASE_ADDRESS,next_addr),_pc(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS,next_amount),_pc(rk.TARGET_VERSION,0),_pc(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0x18))  # noqa: E501
      words.extend(tail); descs.append(rk.struct_rknpu_task(0,0 if cmac else 4,0xd if cmac else 0x18,0x300,0x1ffff,0,size if cmac else size+len(tail),0,base_dma+base*8))  # noqa: E501,E702
    ctypes.memmove(int(cmd.va_addr),words.buffer_info()[0],len(words)*8); packed_tasks=(rk.struct_rknpu_task*n)(*descs); ctypes.memmove(int(task.va_addr),ctypes.addressof(packed_tasks),n*_TASK_DESC_BYTES)  # noqa: E501,E702
    if standalone and not getattr(self.dev,"_npu_clean",False): self.dev.reset_npu()
    try: self._submit(cmd,task,n,standalone=standalone)
    finally:
      if standalone and not self.dev._poisoned: self.dev.reset_npu()

  def _run_ew_ops(self, address, ops:tuple[RKEWOp, ...], rearm:bool=False) -> None:
    """Partition barriers into precision runs, retaining each exact submission and reset boundary."""
    if not ops: return
    M=RKEWMode; program_modes=getattr(self,"_ew_modes",None) or {op.mode for op in ops}  # noqa: E702
    cuts=tuple(sorted({0,len(ops),*(j for i,o in enumerate(ops) if o.submit_barrier for j in ((i,) if o.mode!=M.BOUNDED else (i,i+1)) if j)}))
    for begin,end in zip(cuts,cuts[1:]):
      section=ops[begin:end]; limit=_MAX_EW_GROUP_OPS if M.HALF in program_modes and M.INT16 in program_modes or section[0].mode in (M.BOUNDED,M.HALF_TO_INT16,M.FLOAT_TO_HALF) and len(section)>_MAX_EW_GROUP_OPS and max(op.count for op in section)<=_MAX_EW_ELEMS_FP16 and not any(op.mode in (M.INT32,M.INT16_TO_INT32,M.HALF_TO_INT32,M.INT32_TO_HALF,M.HALF_TO_FLOAT) for op in section) else _MAX_PC_TASKS  # noqa: E501,E702
      for group in map(tuple,itertools.batched(section,limit)):
        tiled=len(group)>1 and group[0].mode in (M.HALF,M.BOUNDED,M.FLOAT_TO_HALF) and group[0].count>_MAX_EW_ELEMS_FP16 and all(op.count==group[0].count and op.mode in (M.HALF,M.BOUNDED,M.FLOAT_TO_HALF) for op in group)  # noqa: E501
        # COMPARE is always standalone. INT16_TO_INT32 may finish an INT16 run without a reset.
        precisions=tuple(_EW_MODE_INFO[op.mode][0] if op.mode!=M.COMPARE else -1 for op in group)
        splits=(0,*(i for i,(left,right) in enumerate(zip(precisions,precisions[1:]),1) if left<0 or right<0 or left!=right and not (group[i].mode==M.INT16_TO_INT32 and left in (0,16))),len(group))  # noqa: E501
        for first,last in zip(splits,splits[1:]):
          run=group[first:last]; precision=precisions[last-1]  # noqa: E702
          if run[0].mode==M.COMPARE:
            for body in self._tile(run[0],_MAX_EW_ELEMS_FP16,address): self._submit_bodies((body,),True)
            continue
          tiles=((emit_ew_stage(op,address),) if not info[1] else self._tile(op,info[1],address,*info[2:],mode=M.BOUNDED if i==0 and op.mode in (M.HALF,M.BOUNDED) else M.HALF if op.mode==M.BOUNDED else op.mode) for i,op in enumerate(run) for info in (_EW_MODE_INFO[op.mode],))  # noqa: E501
          bodies=tuple(itertools.chain.from_iterable(zip(*tiles) if tiled else tiles)); capacity=16 if precision==128 else len(bodies)  # noqa: E702
          reset=last<len(group) and ((following:=group[last].mode)==M.COMPARE and bool(precision) or following!=M.COMPARE and not _EW_MODE_INFO[following][0] and following!=M.INT16_TO_INT32 and precision not in (0,16))  # noqa: E501
          for start in range(0,len(bodies),capacity):
            for chunk in itertools.batched(bodies[start:start+capacity],_MAX_PC_TASKS if tiled else limit): self._submit_bodies(chunk)
            # A full 16-body FLOAT flush has already emptied the chain; rearm suppresses its normal reset.
            # Only a remaining partial FLOAT chain can require the following mode's transition reset.
            if precision>=64 and not rearm or reset and start+capacity>=len(bodies) and (precision!=128 or len(bodies)%16): self.dev.reset_npu()

  def _tile(self, op:RKEWOp, limit:int, address, itemsize:int=2, dst_step:int=1, src_step:int=1, **flags):
    for start in range(0, max(1,op.count), limit):
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
    cursor=next((i for i,op in enumerate(self.image.program) if not isinstance(op,RKGather)),len(self.image.program))
    _apply_gathers(self.image.program[:cursor],buffer)  # type: ignore[arg-type]
    self.dev._sync_buffers((*bufs,*((arena,) if arena is not None else ())),rk.RKNPU_MEM_SYNC_TO_DEVICE); addresses:dict[RKArg,int]={}  # noqa: E702
    def address(arg:RKArg) -> int: return addresses[arg] if arg in addresses else addresses.setdefault(arg,self._dma(buffer(arg.kind,arg.index))+arg.addend)  # noqa: E501
    start = time.perf_counter()
    ew_ops=tuple(op for op in self.image.program if isinstance(op,RKEWOp))
    native_int16=any(op.mode in (RKEWMode.INT16,RKEWMode.INT16_TO_INT32,RKEWMode.HALF_TO_INT16) for op in ew_ops)
    if ew_ops and self.dev._native_int16 and not native_int16: self.dev.reset_npu()
    for index,group in enumerate(groups:=tuple(tuple(items) for _,items in itertools.groupby(self.image.program[cursor:],type))):
      rearm=index+2<len(groups) and all(isinstance(op,RKEWOp) and op.mode==RKEWMode.INT32_TO_HALF for op in group) and isinstance(groups[index+1][0],RKGather) and (isinstance(next_op:=groups[index+2][0],RKEWOp) and next_op.mode==RKEWMode.BOUNDED or all(isinstance(op,RKEWOp) and op.mode==RKEWMode.INT32_TO_HALF for op in groups[index+2]))  # noqa: E501
      if isinstance(current:=group[0],RKCMAC): self._submit_bodies((emit_cmac_stage(current,address),),True,True)
      elif isinstance(current,RKEWOp): self._run_ew_ops(address,group,rearm)  # type: ignore[arg-type]
      else:
        touched={(arg.kind,arg.index) for gather in group for arg in (gather.src,gather.index,gather.dst) if arg is not None}  # type: ignore[union-attr]  # noqa: E501
        self.dev._sync_buffers(tuple(buffer(kind,index) for kind,index in touched),rk.RKNPU_MEM_SYNC_FROM_DEVICE)
        _apply_gathers(group,buffer)  # type: ignore[arg-type]
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
    return HCQBuffer(mapped,size,meta=meta,view=MMIOInterface(mapped,size))
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
    self._native_int16,self._npu_clean=False,True
  def finalize(self):
    for buf in self._buffers.values(): self._gpu_free(buf)
    self._buffers.clear()
