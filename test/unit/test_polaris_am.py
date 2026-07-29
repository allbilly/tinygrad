import unittest
from tinygrad.runtime.support.am.polaris import (
  PolarisAMDev, ViMqd, DOORBELL_MEC_RING0, VI_MQD_ALLOC_DWORDS, MQD_HQD_WORD, mmCP_HQD_ACTIVE, mmCP_HQD_PQ_BASE_LO, mmCP_HQD_PQ_BASE_HI,
  mmCP_HQD_PQ_DOORBELL_CONTROL, mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO, mmCP_HQD_PQ_WPTR_POLL_ADDR_LO,
  mmCP_MEC_CNTL, mmMC_VM_FB_LOCATION, mmSH_MEM_CONFIG, mmVM_CONTEXT0_CNTL, mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR,
  mmVM_CONTEXT0_PAGE_TABLE_START_ADDR, mmVM_CONTEXT0_PAGE_TABLE_END_ADDR,
)

class FakeMMIO:
  def __init__(self, nbytes:int, vals=None, offset:int=0):
    self.nbytes, self.vals, self.offset = nbytes, vals if vals is not None else {}, offset
  def __getitem__(self, idx): return self.vals.get(self.offset + idx, 0)
  def __setitem__(self, idx, val):
    if isinstance(idx, slice):
      start, stop, step = idx.indices(self.nbytes)
      if step != 1: raise ValueError("slice step")
      for i, x in enumerate(val[:stop-start]): self.vals[self.offset + start + i] = x
    else: self.vals[self.offset + idx] = val
  def view(self, offset=0, size=None, fmt=None): return FakeMMIO(size or self.nbytes-offset, self.vals, self.offset+offset)

class FakePCI:
  pcibus = "0000:09:00.0"
  def __init__(self):
    self.bars = {0:FakeMMIO(256 << 20), 2:FakeMMIO(2 << 20), 5:FakeMMIO(256 << 10)}
    self.bars[5][mmMC_VM_FB_LOCATION] = 0xf4fff400
    self.bars[5][mmCP_MEC_CNTL] = 0x50000000
    self.bars[5][mmVM_CONTEXT0_CNTL] = 0x11
    self.bars[5][mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR] = 0xf400200
    self.bars[5][mmVM_CONTEXT0_PAGE_TABLE_START_ADDR] = 0xff00000
    self.bars[5][mmVM_CONTEXT0_PAGE_TABLE_END_ADDR] = 0xff0ffff
  def map_bar(self, bar, fmt='B'): return self.bars[bar]

class TestPolarisAM(unittest.TestCase):
  def test_hot_takeover_and_hqd(self):
    dev = PolarisAMDev(FakePCI())  # type: ignore[arg-type]
    self.assertEqual(dev.visible_vram_base, 0xf400000000)
    self.assertEqual(dev.vram_size, 4 << 30)
    self.assertEqual(dev.visible_vram_size, 256 << 20)
    self.assertEqual(dev.gart_start, 0xff00000000)
    self.assertEqual(dev.gart_size, 256 << 20)

    base = dev.visible_vram_base
    doorbell = dev.gfx.setup_ring(base+0x100000, 0x10000, base+0x200000, base+0x200040,
                                  base+0x201000, 0x1000, mqd_addr=base+0x202000)
    self.assertEqual(doorbell, DOORBELL_MEC_RING0)
    self.assertTrue(dev.hqd_active())
    self.assertEqual(dev.rreg(mmCP_HQD_ACTIVE), 1)
    self.assertEqual(dev.rreg(mmCP_HQD_PQ_BASE_LO), ((base+0x100000) >> 8) & 0xffffffff)
    self.assertEqual(dev.rreg(mmCP_HQD_PQ_BASE_HI), (base+0x100000) >> 40)
    self.assertEqual(dev.rreg(mmCP_HQD_PQ_RPTR_REPORT_ADDR_LO), (base+0x200000) & 0xfffffffc)
    self.assertEqual(dev.rreg(mmCP_HQD_PQ_WPTR_POLL_ADDR_LO), (base+0x200040) & 0xfffffffc)
    self.assertEqual(dev.rreg(mmCP_HQD_PQ_DOORBELL_CONTROL), (DOORBELL_MEC_RING0 << 2) | (1 << 30))
    self.assertEqual(dev.rreg(mmCP_MEC_CNTL), 1 << 28)
    self.assertEqual(dev.rreg(mmSH_MEM_CONFIG), (3 << 3) | (3 << 5) | (3 << 8))
    mqd_bytes = bytes(dev.vram.vals.get(0x202000 + i, 0) for i in range(VI_MQD_ALLOC_DWORDS * 4))
    mqd = memoryview(mqd_bytes).cast('I')
    self.assertEqual(mqd[0], 0xC0310800)
    self.assertEqual(mqd[11], 1)
    self.assertEqual(mqd[259], 0xffffffff)
    self.assertEqual(mqd[MQD_HQD_WORD + mmCP_HQD_ACTIVE - 0x3245], 1)
    self.assertEqual(len(ViMqd().to_bytes()), VI_MQD_ALLOC_DWORDS * 4)

if __name__ == "__main__":
  unittest.main()
