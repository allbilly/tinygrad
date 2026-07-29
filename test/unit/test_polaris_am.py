import unittest
from types import SimpleNamespace
from tinygrad.runtime.ops_amd import AMDComputeQueue, _kfd_doorbell_params
from tinygrad.runtime.autogen.am import pm4_soc15
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
  def test_vi_ring_padding_is_direct_only(self):
    class FakePM4:
      PACKET3_NOP = 0x10
      @staticmethod
      def PACKET3(cmd, count): return 0xffff1000 if (cmd, count) == (0x10, 0x3fff) else 0

    for direct, expected_wptr in ((False, 3), (True, 256)):
      desc = SimpleNamespace(ring=[0] * 512, put_value=0)
      desc.signal_doorbell = lambda dev: None
      dev = SimpleNamespace(target=(8, 0, 3), xccs=1, compute_queue=desc, is_am=lambda: direct)
      queue = AMDComputeQueue.__new__(AMDComputeQueue)
      queue.dev, queue.pm4, queue._q, queue.binded_device = dev, FakePM4, [1, 2, 3], None
      queue._submit(dev)
      self.assertEqual(desc.put_value, expected_wptr)
      self.assertEqual(desc.ring[:3], [1, 2, 3])
      if direct: self.assertTrue(all(x == 0xffff1000 for x in desc.ring[3:256]))

  def test_gfx8_kfd_doorbell_uses_queue_id_slot(self):
    queue = SimpleNamespace(doorbell_offset=0x12345000, queue_id=7)
    self.assertEqual(_kfd_doorbell_params(queue, 80003), (0x12345000, 0x1000, 28))
    self.assertEqual(_kfd_doorbell_params(SimpleNamespace(doorbell_offset=0x12345678), 110000),
                     (0x12344000, 0x2000, 0x1678))

  def test_gfx8_release_mem_uses_ci_packet_layout(self):
    queue = AMDComputeQueue.__new__(AMDComputeQueue)
    queue.dev, queue.pm4, queue._q, queue.binded_device = SimpleNamespace(target=(8, 0, 3)), pm4_soc15, [], None
    queue.release_mem(address=0x12345000, value=7, data_sel=pm4_soc15.data_sel__mec_release_mem__send_32_bit_low,
                      int_sel=pm4_soc15.int_sel__mec_release_mem__none, cache_flush=True)
    self.assertEqual(len(queue._q), 7)  # CI/VI has no trailing ctxid dword.
    self.assertEqual(queue._q[0], pm4_soc15.PACKET3(pm4_soc15.PACKET3_RELEASE_MEM, 5) | 2)
    self.assertTrue(queue._q[1] & pm4_soc15.EOP_TC_ACTION_EN)
    self.assertFalse(queue._q[1] & pm4_soc15.EOP_TC_NC_ACTION_EN)
    self.assertEqual((queue._q[1] >> 25) & 3, 2)
    self.assertEqual(queue._q[2], pm4_soc15.DATA_SEL(1) | pm4_soc15.INT_SEL(3))

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

    dev.quiesce(timeout_s=0)
    self.assertEqual(dev.rreg(mmCP_MEC_CNTL), (1 << 30) | (1 << 28))
    self.assertEqual(dev.rreg(mmVM_CONTEXT0_CNTL), 0)

if __name__ == "__main__":
  unittest.main()
