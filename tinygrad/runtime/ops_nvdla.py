# pylint: disable=cell-var-from-loop
# NVDLA compiled backend: direct MMIO register programming via /dev/mem.
# No KMD needed — registers are written directly to NVDLA's MMIO space.
# Port of the rockchip backend, replacing RKNPU ioctl submit with direct MMIO writes.
# Current VP-tested scope: native SDP elementwise ADD/MUL.
# fp16 only. No LUT, no CMAC, no PDP in this first pass.
#
# Memory model: buffers are allocated from a DRAM region mmap'd via /dev/mem.
# The NVDLA hardware accesses DRAM directly via physical addresses.
# On the VP simulator: MMIO at 0x10200000, DRAM at 0xC0000000 (from gen_cfg2c.py).
import ctypes, mmap, os, struct
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import mv_address, to_mv, Target
from tinygrad.device import Compiled, Program, BufferSpec, TinyELF
from tinygrad.uop.ops import UOp
from tinygrad.renderer import Renderer
from tinygrad.runtime.support.hcq import HCQBuffer, HCQAllocatorBase
from tinygrad.runtime.autogen import nvdla as nv
from tinygrad.runtime.support.nvdla import (build_native_program, NVDLAImage, NVDLATask,
  NVDLAReloc, _CONST_SLOT, _ZERO_SLOT, encode_nvdla, decode_nvdla)

# ---- buffer-level dtype conversion (same as rockchip, no NPU compute) ----
def _convert_fp32_to_fp16_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 4), dtype=np.float32).astype(np.float16)
  ctypes.memmove(dst, arr.ctypes.data, n * 2)

def _convert_fp16_to_fp32_buf(src, dst, n):
  import numpy as np
  arr = np.frombuffer(ctypes.string_at(src, n * 2), dtype=np.float16).astype(np.float32)
  ctypes.memmove(dst, arr.ctypes.data, n * 4)

class NVDLAProgram(Program['NVDLADevice']):
  image: NVDLAImage|None
  task: NVDLATask
  relocs: list[NVDLAReloc]
  def __init__(self, dev:'NVDLADevice', obj:TinyELF):
    self.device, self.name, self.submit_count = dev, obj.name, 0
    self.image = None
    self.task = NVDLATask("sdp", (1,), 0, ())
    self.relocs = []
    # Decode image from ELF lib if it's an encoded NVDLAImage
    if len(obj.lib) >= 8:
      magic = struct.unpack_from("<I", obj.lib, 0)[0]
      if magic == 0x4E56444C:  # "NVDA"
        self.image = decode_nvdla(obj.lib)
        self.task = self.image.task
        self.relocs = list(self.image.relocs)

  def _set_image(self, img: NVDLAImage):
    """Called by the renderer to set the image directly (bypasses ELF for PR1)."""
    self.image = img
    self.task = img.task
    self.relocs = list(img.relocs)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    dev = self.device
    img = self.image
    assert img is not None, "nvdla: no image set"
    task = img.task
    temp: list[HCQBuffer] = []
    try:
      buf_map: dict[int, HCQBuffer] = dict(enumerate(bufs))  # type: ignore[assignment]
      total = task.layout[0]
      sdp_group = dev.sdp_group
      # fp32 buffer-level conversion: NPU processes fp16 internally
      if task.kind == "sdp" and not task.is_copy:
        for slot in task.fp32_inputs:
          if slot in buf_map and slot != _CONST_SLOT and slot != _ZERO_SLOT:
            src_buf = buf_map[slot]
            fp16_buf = dev._alloc_dram(max(total * 2, 4096))
            temp.append(fp16_buf)
            _convert_fp32_to_fp16_buf(src_buf.va_addr, fp16_buf.va_addr, total)
            buf_map[slot] = fp16_buf  # type: ignore[assignment]
        if task.fp32_output:
          fp16_out = dev._alloc_dram(max(total * 2, 4096))
          temp.append(fp16_out)
          buf_map[task.out_slot] = fp16_out  # type: ignore[assignment]
      # BDMA copy: host-side memmove (data movement, not NPU compute)
      if task.is_copy:
        in_slot = task.in_slots[0]
        in_buf, out_buf = buf_map[in_slot], buf_map[task.out_slot]
        in_is_fp32 = in_slot in task.fp32_inputs
        out_is_fp32 = task.fp32_output
        if in_is_fp32 and not out_is_fp32:
          _convert_fp32_to_fp16_buf(in_buf.va_addr, out_buf.va_addr, total)
        elif not in_is_fp32 and out_is_fp32:
          _convert_fp16_to_fp32_buf(in_buf.va_addr, out_buf.va_addr, total)
        elif in_is_fp32 and out_is_fp32:
          ctypes.memmove(out_buf.va_addr, in_buf.va_addr, total * 4)  # type: ignore[arg-type]
        else:
          ctypes.memmove(out_buf.va_addr, in_buf.va_addr, total * 2)  # type: ignore[arg-type]
        return
      # Prepare constant/zero buffers for SDP
      for rel in self.relocs:
        if rel.globals_slot == _CONST_SLOT:
          cbuf = dev._alloc_dram(max(total * 2, 4096))
          temp.append(cbuf)
          cval = struct.unpack('<f', struct.pack('<I', rel.addend))[0]
          try: fp16_bytes = struct.pack('<e', cval) * total
          except OverflowError: fp16_bytes = struct.pack('<e', float("-inf") if cval < 0 else float("inf")) * total
          ctypes.memmove(cbuf.va_addr, fp16_bytes, total * 2)  # type: ignore[arg-type]
          buf_map[_CONST_SLOT] = cbuf  # type: ignore[assignment]
        elif rel.globals_slot == _ZERO_SLOT:
          zbuf = dev._alloc_dram(max(total * 2, 4096))
          temp.append(zbuf)
          ctypes.memset(zbuf.va_addr, 0, total * 2)  # type: ignore[arg-type]
          buf_map[_ZERO_SLOT] = zbuf  # type: ignore[assignment]
      # Write register commands to MMIO, applying relocs with physical addresses
      mmio = dev.mmio_base  # mmap'd NVDLA MMIO region
      for i, cmd in enumerate(img.cmds):
        offset = cmd & 0xFFFFFFFF
        value = (cmd >> 32) & 0xFFFFFFFF
        if task.kind == "sdp":
          if offset in (nv.NVDLA_SDP_S_POINTER_0, nv.NVDLA_SDP_RDMA_S_POINTER_0): value = sdp_group
          elif offset == nv.NVDLA_GLB_S_INTR_STATUS_0: value = 1 << sdp_group
        # Apply relocs that reference this command
        for rel in self.relocs:
          if rel.cmd_index == i:
            buf = buf_map[rel.globals_slot]
            phys = buf.meta + rel.addend  # phys_addr stored in meta
            v = phys & rel.mask  # NVDLA uses 32-bit LOW addr; HIGH is 0 for VP DRAM
            if rel.field_shift: v = (v << rel.field_shift) & 0xFFFFFFFF
            value = v
        # Write to MMIO: *(uint32_t*)(mmio + offset) = value
        mmio.seek(offset)
        mmio.write(struct.pack('<I', value))
      # Poll for completion
      if task.kind == "sdp":
        dev._poll_sdp_done(sdp_group)
        dev.sdp_group ^= 1
      elif task.kind == "bdma":
        dev._poll_bdma_done()
      # fp32 output: convert fp16 → fp32
      if task.fp32_output:
        out_buf = buf_map[task.out_slot]
        _convert_fp16_to_fp32_buf(out_buf.va_addr, out_buf.va_addr, total)
    finally:
      for b in temp: dev._free_dram(b)
    self.submit_count += 1

class NVDLARenderer(Renderer):
  device = "NVDLA"
  has_threads = False
  has_local = False
  code_for_op = {}
  def __init__(self, target:Target): self.target, self.tensor_cores = target, []
  def native_program(self, ast:UOp) -> UOp|None: return build_native_program(ast)
  def asm(self, prg:UOp, lin:UOp) -> bytes:
    # The INS args carry the NVDLAImage + packed cmds + relocs
    img = None
    for u in lin.src:
      if isinstance(u.arg, NVDLAImage): img = u.arg
    if img is None: raise RuntimeError("nvdla: no NVDLAImage metadata")
    return encode_nvdla(img)
  def supported_dtypes(self) -> set[DType]: return {dtypes.half, dtypes.float}

class NVDLAAllocator(HCQAllocatorBase):
  """DRAM-backed allocator: buffers are mmap'd from /dev/mem at the NVDLA DRAM range."""
  def __init__(self, dev): super().__init__(dev, batch_cnt=0)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer: return self.dev._alloc_dram(size)
  def _do_copy(self, s, d, sz): ctypes.memmove(d, s, sz)
  def _copyin(self, dest:HCQBuffer, src:memoryview): self._do_copy(mv_address(src), dest.va_addr, src.nbytes)
  def _copyout(self, dest:memoryview, src:HCQBuffer): self._do_copy(src.va_addr, mv_address(dest), src.size)  # type: ignore[arg-type]
  def _as_buffer(self, src:HCQBuffer) -> memoryview: return to_mv(src.va_addr, src.size)  # type: ignore[arg-type]
  def _do_free(self, buf:HCQBuffer, options:BufferSpec|None=None): self.dev._free_dram(buf)

class NVDLADevice(Compiled):
  def __init__(self, device:str):
    # Open /dev/mem for MMIO and DRAM access
    self.fd_mem = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    # mmap NVDLA MMIO register space
    self.mmio_base = mmap.mmap(self.fd_mem, nv.NVDLA_MMIO_SIZE, mmap.MAP_SHARED,
                               mmap.PROT_READ | mmap.PROT_WRITE, offset=nv.NVDLA_MMIO_BASE)
    # DRAM allocator state: bump allocator within the DRAM region
    self.dram_base = nv.NVDLA_DRAM_BASE
    self.dram_size = nv.NVDLA_DRAM_SIZE
    self.dram_offset = 0  # current allocation offset within DRAM region
    self.sdp_group = 0
    self.dram_mmap = mmap.mmap(self.fd_mem, self.dram_size, mmap.MAP_SHARED,
                               mmap.PROT_READ | mmap.PROT_WRITE, offset=self.dram_base)
    self.dram_allocs: list[tuple[int, int]] = []  # (offset, size) for free tracking
    super().__init__(device, NVDLAAllocator(self), [NVDLARenderer], NVDLAProgram)

  def _alloc_dram(self, size:int) -> HCQBuffer:
    """Allocate a buffer from the DRAM region. Returns HCQBuffer with va_addr and phys_addr."""
    size = (size + 4095) & ~4095  # page-align
    # Simple bump allocator (no free for PR1 — buffers are small and short-lived)
    if self.dram_offset + size > self.dram_size:
      # Reset bump allocator if we've used up the region
      self.dram_offset = 0
    phys_addr = self.dram_base + self.dram_offset
    va_addr = self.dram_offset  # offset into dram_mmap
    buf = HCQBuffer(va_addr=ctypes.addressof(ctypes.c_char.from_buffer(self.dram_mmap, va_addr)),
                    size=size, meta=phys_addr)
    self.dram_offset += size
    return buf

  def _free_dram(self, buf:HCQBuffer) -> None:
    """No-op for PR1 (bump allocator doesn't free)."""
    pass

  def _poll_sdp_done(self, group:int) -> None:
    """Poll and acknowledge the selected SDP producer group's GLB completion interrupt."""
    status_off = nv.NVDLA_GLB_S_INTR_STATUS_0
    done_mask = 1 << group
    for _ in range(100000):
      self.mmio_base.seek(status_off)
      status = struct.unpack('<I', self.mmio_base.read(4))[0]
      if status & done_mask:
        self.mmio_base.seek(status_off)
        self.mmio_base.write(struct.pack('<I', done_mask))
        return
    raise RuntimeError("nvdla: SDP poll timeout")

  def _poll_bdma_done(self) -> None:
    """Poll BDMA status register for completion."""
    status_off = nv.NVDLA_BDMA_STATUS_0
    for _ in range(10000000):
      self.mmio_base.seek(status_off)
      status = struct.unpack('<I', self.mmio_base.read(4))[0]
      if status & 0x1: return
    raise RuntimeError("nvdla: BDMA poll timeout")

  def finalize(self):
    self.mmio_base.close()
    self.dram_mmap.close()
    os.close(self.fd_mem)
