from __future__ import annotations
import math, struct, time
from tinygrad.helpers import DEBUG
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.system import PCIDevice, System

# Polaris10 / GFX8 register indices. These are dword offsets in BAR5.
mmMC_VM_FB_LOCATION = 0x0809
mmMC_VM_MX_L1_TLB_CNTL = 0x0819
mmVM_L2_CNTL = 0x0500
mmVM_L2_CNTL2 = 0x0501
mmVM_L2_CNTL3 = 0x0502
mmVM_CONTEXT0_CNTL = 0x0504
mmVM_CONTEXT0_CNTL2 = 0x050c
mmVM_INVALIDATE_REQUEST = 0x051e
mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR = 0x0546
mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR = 0x054f
mmVM_CONTEXT0_PAGE_TABLE_START_ADDR = 0x0557
mmVM_CONTEXT0_PAGE_TABLE_END_ADDR = 0x055f
mmVM_L2_CNTL4 = 0x0578
mmHDP_MEM_COHERENCY_FLUSH_CNTL = 0x1520
mmBIF_DOORBELL_APER_EN = 0x1501
mmSRBM_GFX_CNTL = 0x0391
mmCP_MEC_CNTL = 0x208d
mmCP_PQ_STATUS = 0x2147
mmCP_MEC_DOORBELL_RANGE_LOWER = 0x2149
mmCP_MEC_DOORBELL_RANGE_UPPER = 0x214a
mmRLC_CNTL = 0xec00
mmRLC_SAFE_MODE = 0xec05
mmRLC_CP_SCHEDULERS = 0xecaa
mmSH_MEM_BASES = 0x230a
mmSH_MEM_APE1_BASE = 0x230b
mmSH_MEM_APE1_LIMIT = 0x230c
mmSH_MEM_CONFIG = 0x230d
mmCP_PQ_WPTR_POLL_CNTL = 0x3083
mmCP_MQD_BASE_ADDR = 0x3245
mmCP_HQD_ACTIVE = 0x3247
mmCP_HQD_VMID = 0x3248
mmCP_HQD_PERSISTENT_STATE = 0x3249
mmCP_HQD_QUANTUM = 0x324c
mmCP_HQD_PQ_BASE_LO = 0x324d
mmCP_HQD_PQ_BASE_HI = 0x324e
mmCP_HQD_PQ_RPTR = 0x324f
mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO = 0x3250
mmCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x3251
mmCP_HQD_PQ_WPTR_POLL_ADDR_LO = 0x3252
mmCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x3253
mmCP_HQD_PQ_DOORBELL_CONTROL = 0x3254
mmCP_HQD_PQ_WPTR = 0x3255
mmCP_HQD_PQ_CONTROL = 0x3256
mmCP_HQD_IB_CONTROL = 0x325a
mmCP_HQD_IQ_TIMER = 0x325b
mmCP_HQD_DEQUEUE_REQUEST = 0x325d
mmCP_MQD_CONTROL = 0x3267
mmCP_HQD_EOP_BASE_ADDR_LO = 0x326a
mmCP_HQD_EOP_BASE_ADDR_HI = 0x326b
mmCP_HQD_EOP_CONTROL = 0x326c
mmCP_HQD_EOP_RPTR = 0x326d
mmCP_HQD_EOP_WPTR = 0x326e
mmCP_HQD_EOP_EVENTS = 0x326f
mmCP_HQD_CTX_SAVE_CONTROL = 0x3272
mmCP_HQD_ERROR = 0x3278
mmCP_HQD_EOP_WPTR_MEM = 0x3279
mmCP_HQD_EOP_DONES = 0x327a

DOORBELL_MEC_RING0, DOORBELL_MEC_RING7 = 0x10, 0x17
VI_MQD_ALLOC_DWORDS, MQD_HQD_WORD = 261, 128

def _field(val:int, mask:int, shift:int, field:int) -> int:
  return (val & ~mask) | ((field << shift) & mask)

class ViMqd:
  """Polaris vi_mqd_allocation, matching gfx_v8_0_mqd_init."""
  def __init__(self):
    self.words = [0] * VI_MQD_ALLOC_DWORDS
    self.words[259] = self.words[260] = 0xffffffff

  def hqd(self, reg:int) -> int: return self.words[MQD_HQD_WORD + reg - mmCP_MQD_BASE_ADDR]
  def set_hqd(self, reg:int, val:int): self.words[MQD_HQD_WORD + reg - mmCP_MQD_BASE_ADDR] = val & 0xffffffff
  def to_bytes(self) -> bytes: return struct.pack(f"<{VI_MQD_ALLOC_DWORDS}I", *self.words)

class PolarisGMC:
  def __init__(self, adev:'PolarisAMDev'): self.adev = adev
  def flush_hdp(self):
    self.adev.wreg(mmHDP_MEM_COHERENCY_FLUSH_CNTL, 1)
    self.adev.rreg(mmHDP_MEM_COHERENCY_FLUSH_CNTL)

class PolarisGFX:
  xccs = 1
  def __init__(self, adev:'PolarisAMDev'):
    self.adev = adev
    self.queue_args:tuple|None = None

  def setup_ring(self, ring_addr:int, ring_size:int, rptr_addr:int, wptr_addr:int, eop_addr:int, eop_size:int,
                 _is_aql:bool=False, _is_aql2:bool=False, mqd_addr:int=0, mqd_view=None) -> int:
    if _is_aql or _is_aql2: raise RuntimeError("Polaris supports the native compute queue only")
    a = self.adev
    self.queue_args = (ring_addr, ring_size, rptr_addr, wptr_addr, eop_addr, eop_size, False, False, mqd_addr, mqd_view)
    a.deactivate_hqd()
    a.enable_compute()
    a.srbm_select(1, 0, 0, 0)

    # Populate the complete VI MQD before committing it. MEC can DMA-read this
    # structure during activation; pointing it at a zero page leaves RPTR stuck.
    mqd = ViMqd()
    mqd.words[0], mqd.words[11], mqd.words[32] = 0xC0310800, 1, 3
    for i in (23, 24, 26, 27): mqd.words[i] = 0xffffffff
    cu_addr = mqd_addr + 259 * 4
    mqd.words[126], mqd.words[127] = cu_addr & 0xffffffff, (cu_addr >> 32) & 0xffffffff

    mqd.set_hqd(mmCP_HQD_VMID, 0)
    persistent = _field(a.rreg(mmCP_HQD_PERSISTENT_STATE), 0x3ff00, 8, 0x53) & ~1
    mqd.set_hqd(mmCP_HQD_PERSISTENT_STATE, persistent)
    quantum = _field(_field(_field(a.rreg(mmCP_HQD_QUANTUM), 1, 0, 1), 0x6, 1, 1), 0xfffffff0, 4, 10)
    mqd.set_hqd(mmCP_HQD_QUANTUM, quantum)

    pq_base = ring_addr >> 8
    mqd.set_hqd(mmCP_HQD_PQ_BASE_LO, pq_base & 0xffffffff)
    mqd.set_hqd(mmCP_HQD_PQ_BASE_HI, (pq_base >> 32) & 0xffffffff)
    mqd.set_hqd(mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO, rptr_addr & 0xfffffffc)
    mqd.set_hqd(mmCP_HQD_PQ_RPTR_REPORT_ADDR_HI, (rptr_addr >> 32) & 0xffff)
    mqd.set_hqd(mmCP_HQD_PQ_WPTR_POLL_ADDR_LO, wptr_addr & 0xfffffffc)
    mqd.set_hqd(mmCP_HQD_PQ_WPTR_POLL_ADDR_HI, (wptr_addr >> 32) & 0xffff)
    mqd.set_hqd(mmCP_HQD_PQ_RPTR, 0)
    mqd.set_hqd(mmCP_HQD_PQ_WPTR, 0)
    mqd.set_hqd(mmCP_HQD_PQ_DOORBELL_CONTROL, (DOORBELL_MEC_RING0 << 2) | (1 << 30))
    qsize = int(math.log2(ring_size // 4)) - 1
    rptr_block_size = int(math.log2(4096 // 4)) - 1
    mqd.set_hqd(mmCP_HQD_PQ_CONTROL, (qsize & 0x3f) | ((rptr_block_size << 8) & 0x3f00) | (1 << 30) | (1 << 31))

    eop_base = eop_addr >> 8
    mqd.set_hqd(mmCP_HQD_EOP_BASE_ADDR_LO, eop_base & 0xffffffff)
    mqd.set_hqd(mmCP_HQD_EOP_BASE_ADDR_HI, (eop_base >> 32) & 0xffffffff)
    mqd.set_hqd(mmCP_HQD_EOP_CONTROL, _field(a.rreg(mmCP_HQD_EOP_CONTROL), 0xf000, 12, int(math.log2(eop_size // 4)) - 1))
    for reg in (mmCP_HQD_EOP_RPTR, mmCP_HQD_EOP_WPTR, mmCP_HQD_EOP_WPTR_MEM,
                mmCP_HQD_EOP_DONES, mmCP_HQD_EOP_EVENTS, mmCP_HQD_ERROR): mqd.set_hqd(reg, a.rreg(reg))

    mqd.set_hqd(mmCP_MQD_BASE_ADDR, mqd_addr & 0xfffffffc)
    mqd.set_hqd(mmCP_MQD_BASE_ADDR + 1, (mqd_addr >> 32) & 0xffffffff)
    mqd.set_hqd(mmCP_MQD_CONTROL, _field(a.rreg(mmCP_MQD_CONTROL), 0xf, 0, 0))
    mqd.set_hqd(mmCP_HQD_IB_CONTROL, _field(_field(a.rreg(mmCP_HQD_IB_CONTROL), 0x300000, 20, 3), 0xc0000, 18, 3))
    mqd.set_hqd(mmCP_HQD_IQ_TIMER, _field(a.rreg(mmCP_HQD_IQ_TIMER), 0x3000000, 24, 3))
    mqd.set_hqd(mmCP_HQD_CTX_SAVE_CONTROL, _field(a.rreg(mmCP_HQD_CTX_SAVE_CONTROL), 0x3000000, 24, 3))
    mqd.set_hqd(mmCP_HQD_ACTIVE, 1)
    a.upload_mqd(mqd_addr, mqd.to_bytes(), mqd_view)

    # gfx_v8_0_mqd_commit register order. ACTIVE is in the final range and the
    # in-memory MQD has already been made visible above.
    a.wreg(mmCP_PQ_WPTR_POLL_CNTL, a.rreg(mmCP_PQ_WPTR_POLL_CNTL) & ~0x80000000)
    for reg in range(mmCP_HQD_VMID, mmCP_HQD_EOP_CONTROL + 1): a.wreg(reg, mqd.hqd(reg))
    for reg in (mmCP_HQD_EOP_RPTR, mmCP_HQD_EOP_WPTR, mmCP_HQD_EOP_WPTR_MEM): a.wreg(reg, mqd.hqd(reg))
    for reg in range(mmCP_HQD_EOP_EVENTS, mmCP_HQD_ERROR + 1): a.wreg(reg, mqd.hqd(reg))
    for reg in range(mmCP_MQD_BASE_ADDR, mmCP_HQD_ACTIVE + 1): a.wreg(reg, mqd.hqd(reg))
    a.srbm_select()
    a.gmc.flush_hdp()
    if not a.hqd_active(): raise RuntimeError("Polaris compute HQD did not activate")
    if DEBUG >= 2: print(f"am {a.devfmt}: GFX8 HQD active ring={ring_addr:#x} doorbell={DOORBELL_MEC_RING0:#x}")
    return DOORBELL_MEC_RING0

  def reset_mec(self):
    if self.queue_args is not None: self.setup_ring(*self.queue_args)

class PolarisAMDev:
  """Experimental direct GFX8 device for a Linux-initialized Polaris10.

  Linux owns firmware/SMU initialization. After amdgpu is unbound, AM owns a
  native compute HQD and the CPU-visible 256 MiB VRAM aperture directly. This
  path is gated by AMD_POLARIS_EXPERIMENTAL until ring fetch is proven safe.
  """
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt = pci_dev, pci_dev.pcibus
    self.vram, self.doorbell32, self.mmio = pci_dev.map_bar(0), pci_dev.map_bar(2, fmt='I'), pci_dev.map_bar(5, fmt='I')
    self.ip_ver = {am.GC_HWIP:(8, 0, 3), am.SDMA0_HWIP:(3, 0, 0), am.NBIF_HWIP:(5, 0, 0)}
    fb = self.rreg(mmMC_VM_FB_LOCATION)
    fb_base, fb_end = (fb & 0xffff) << 24, (((fb >> 16) & 0xffff) << 24) | 0xffffff
    if fb in (0, 0xffffffff) or fb_end < fb_base: raise RuntimeError(f"invalid Polaris FB aperture {fb:#x}")
    self.vram_size, self.visible_vram_size = fb_end - fb_base + 1, self.vram.nbytes
    # BAR0 starts at VRAM offset zero even when it exposes only the first 256 MiB.
    self.visible_vram_base = fb_base
    self.gart_start = self.rreg(mmVM_CONTEXT0_PAGE_TABLE_START_ADDR) << 12
    self.gart_end = (self.rreg(mmVM_CONTEXT0_PAGE_TABLE_END_ADDR) << 12) | 0xfff
    self.gart_base = self.rreg(mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR) << 12
    self.gart_size, self.gart_table_off = self.gart_end - self.gart_start + 1, self.gart_base - self.visible_vram_base
    gart_table_size = self.gart_size // 0x1000 * 8
    if self.gart_size <= 0 or self.gart_table_off < 0 or self.gart_table_off + gart_table_size > self.visible_vram_size:
      raise RuntimeError(f"Polaris GART table is not CPU-visible: {self.gart_base:#x}")
    if not (self.rreg(mmVM_CONTEXT0_CNTL) & 1) and DEBUG >= 1:
      print(f"am {self.devfmt}: enabling preserved GART context (cntl={self.rreg(mmVM_CONTEXT0_CNTL):#x})")
    # Linux may leave VMID0 disabled when no userspace queue exists. The page
    # table itself and its aperture survive unbind; restore gfx_v8_0_gart_enable.
    tlb = self.rreg(mmMC_VM_MX_L1_TLB_CNTL)
    tlb = ((tlb | 0x43) & ~0x38) | (3 << 3)
    self.wreg(mmMC_VM_MX_L1_TLB_CNTL, tlb)
    self.wreg(mmVM_L2_CNTL, 0x30103)
    self.wreg(mmVM_L2_CNTL2, 0x30003)
    self.wreg(mmVM_L2_CNTL3, 0x24100003)
    self.wreg(mmVM_L2_CNTL4, 0)
    self.wreg(mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR, 0)
    self.wreg(mmVM_CONTEXT0_CNTL2, 0)
    self.wreg(mmVM_CONTEXT0_CNTL, 0x11)
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.gmc, self.gfx, self.sdma = PolarisGMC(self), PolarisGFX(self), None
    if DEBUG >= 1:
      print(f"am {self.devfmt}: Polaris hot takeover GART {self.gart_start:#x}+{self.gart_size:#x} table={self.gart_base:#x}")

  def rreg(self, reg:int) -> int: return self.mmio[reg]
  def wreg(self, reg:int, val:int): self.mmio[reg] = val & 0xffffffff
  def upload_mqd(self, addr:int, data:bytes, view=None):
    if view is not None:
      view.view(fmt='B')[:len(data)] = data
      System.memory_barrier()
    else:
      off = addr - self.visible_vram_base
      if off < 0 or off + len(data) > self.visible_vram_size: raise RuntimeError(f"MQD address is not writable: {addr:#x}")
      self.vram[off:off+len(data)] = data
      self.gmc.flush_hdp()

  def map_gart(self, off:int, paddrs:list[int]):
    pte_off = self.gart_table_off + (off // 0x1000) * 8
    for i, paddr in enumerate(paddrs):
      self.vram[pte_off+i*8:pte_off+(i+1)*8] = struct.pack("<Q", (paddr & 0x0000fffffffff000) | 0x77)
    self.gmc.flush_hdp()
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
    self.rreg(mmVM_INVALIDATE_REQUEST)

  def unmap_gart(self, off:int, size:int):
    pte_off, npages = self.gart_table_off + (off // 0x1000) * 8, (size + 0xfff) // 0x1000
    self.vram[pte_off:pte_off+npages*8] = bytes(npages * 8)
    self.gmc.flush_hdp()
    self.wreg(mmVM_INVALIDATE_REQUEST, 1)
  def srbm_select(self, me:int=0, pipe:int=0, queue:int=0, vmid:int=0):
    self.wreg(mmSRBM_GFX_CNTL, ((pipe & 3) << 0) | ((me & 3) << 2) | ((vmid & 0xf) << 4) | ((queue & 7) << 8))

  def hqd_active(self) -> bool:
    self.srbm_select(1, 0, 0, 0)
    active = bool(self.rreg(mmCP_HQD_ACTIVE) & 1)
    self.srbm_select()
    return active

  def deactivate_hqd(self, timeout_s:float=1.0):
    self.srbm_select(1, 0, 0, 0)
    if self.rreg(mmCP_HQD_ACTIVE) & 1:
      self.wreg(mmCP_HQD_DEQUEUE_REQUEST, 1)
      deadline = time.monotonic() + timeout_s
      while time.monotonic() < deadline and self.rreg(mmCP_HQD_ACTIVE) & 1: time.sleep(0.001)
    self.wreg(mmCP_HQD_DEQUEUE_REQUEST, 0)
    self.wreg(mmCP_HQD_PQ_RPTR, 0)
    self.wreg(mmCP_HQD_PQ_WPTR, 0)
    self.srbm_select()

  def enable_compute(self):
    # VMID0 flat addressing, UC memory type, and the native MEC doorbell range.
    self.srbm_select()
    self.wreg(mmSH_MEM_CONFIG, (3 << 3) | (3 << 5) | (3 << 8))
    self.wreg(mmSH_MEM_BASES, 0)
    self.wreg(mmSH_MEM_APE1_BASE, 1)
    self.wreg(mmSH_MEM_APE1_LIMIT, 0)
    data = self.rreg(mmRLC_CNTL)
    self.wreg(mmRLC_SAFE_MODE, (data | 1) & ~0x1e)
    deadline = time.monotonic() + 0.05
    while time.monotonic() < deadline and self.rreg(mmRLC_SAFE_MODE) & 1: time.sleep(0.001)
    self.wreg(mmRLC_CP_SCHEDULERS, (self.rreg(mmRLC_CP_SCHEDULERS) & 0xffffff00) | (1 << 5) | 0x80)
    self.wreg(mmCP_MEC_CNTL, 1 << 28)  # ME1 live, ME2 halted
    self.wreg(mmBIF_DOORBELL_APER_EN, self.rreg(mmBIF_DOORBELL_APER_EN) | 1)
    self.wreg(mmCP_MEC_DOORBELL_RANGE_LOWER, 0)
    self.wreg(mmCP_MEC_DOORBELL_RANGE_UPPER, DOORBELL_MEC_RING7 << 2)
    self.wreg(mmCP_PQ_STATUS, self.rreg(mmCP_PQ_STATUS) | 2)

  def recover(self, force:bool=False) -> bool:
    if not force: return False
    self.gfx.reset_mec()
    return True

  def fini(self): self.deactivate_hqd()
