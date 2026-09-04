import unittest
from types import SimpleNamespace
from tinygrad.codegen.opt import tc
from tinygrad.runtime.ops_amd import AMDComputeQueue, AMDQueueDesc, GFX8GC, _gfx8_props, _kfd_doorbell_params
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
  def test_gfx803_has_no_tensor_cores(self):
    self.assertEqual(tc.get_amd("gfx803"), [])

  def test_gfx8_topology_matches_kernel(self):
    props = _gfx8_props(cu_count=32, se_count=4, sh_per_se=1, cu_per_sh=9)
    self.assertEqual((props['simd_count'], props['simd_per_cu']), (128, 4))
    self.assertEqual((props['array_count'], props['simd_arrays_per_engine']), (4, 1))
    self.assertEqual(props['cu_per_simd_array'], 9)
    self.assertEqual(GFX8GC().regCOMPUTE_RESOURCE_LIMITS.fields['waves_per_sh'], (0, 9))

  def test_gfx8_dispatch_does_not_disable_shader_engine_zero(self):
    for direct in (False, True):
      with self.subTest(direct=direct):
        dev = SimpleNamespace(target=(8, 0, 3), is_am=lambda: direct, sqtt_enabled=False, xccs=1, tmpring_size=0x800400)
        queue = AMDComputeQueue(SimpleNamespace(soc=SimpleNamespace(CS_PARTIAL_FLUSH=0), pm4=pm4_soc15, gc=GFX8GC(), nbio=None))
        queue.dev = dev
        writes = []
        queue.wreg = lambda reg, *args, **kwargs: writes.append(reg.name)  # type: ignore[method-assign]
        queue.q = queue.pkt3 = lambda *args: None  # type: ignore[method-assign]
        prg = SimpleNamespace(dev=dev, enable_private_segment_sgpr=False, enable_dispatch_ptr=False, prog_addr=0x100000,
                              rsrc1=0, rsrc2=0, rsrc3=0, wave32=False)
        queue.exec(prg, SimpleNamespace(bind_data=[], buf=SimpleNamespace(va_addr=0x200000)), (1, 1, 1), (1, 1, 1))
        self.assertNotIn("regCOMPUTE_STATIC_THREAD_MGMT_SE0", writes)
        self.assertNotIn("regCOMPUTE_PGM_RSRC3", writes)
        self.assertEqual(writes.count("regCOMPUTE_START_X"), 1)
        self.assertLess(writes.index("regCOMPUTE_START_X"), writes.index("regCOMPUTE_PGM_LO"))

  def test_gfx8_dispatch_programs_private_scratch(self):
    scratch = SimpleNamespace(va_addr=0x123456789000, size=8 << 20)
    dev = SimpleNamespace(target=(8, 0, 3), is_am=lambda: False, sqtt_enabled=False, xccs=1,
                          tmpring_size=0x800400, scratch=scratch)
    queue = AMDComputeQueue(SimpleNamespace(soc=SimpleNamespace(CS_PARTIAL_FLUSH=0), pm4=pm4_soc15, gc=GFX8GC(), nbio=None))
    queue.dev = dev
    writes = {}
    queue.wreg = lambda reg, *args, **kwargs: writes.setdefault(reg.name, (args, kwargs))  # type: ignore[method-assign]
    queue.q = queue.pkt3 = lambda *args: None  # type: ignore[method-assign]
    prg = SimpleNamespace(dev=dev, enable_private_segment_sgpr=True, private_segment_size=120, enable_dispatch_ptr=False,
                          prog_addr=0x100000, rsrc1=0, rsrc2=1, rsrc3=0, wave32=False)
    queue.exec(prg, SimpleNamespace(bind_data=[], buf=SimpleNamespace(va_addr=0x200000)), (1, 1, 1), (1, 1, 1))
    self.assertEqual(writes['regCOMPUTE_TMPRING_SIZE'][0], (0x800400,))
    scratch_desc = writes['regCOMPUTE_USER_DATA_0'][0][:4]
    self.assertEqual(scratch_desc[:3], (0x56789000, 0x80001234, 8 << 20))
    self.assertTrue(scratch_desc[3] & (1 << 23))  # ADD_TID_ENABLE

  def test_drm_submit_callback_reuses_ring(self):
    submitted = []
    ring_buf = SimpleNamespace()
    desc = AMDQueueDesc(ring=FakeMMIO(0x1000), read_ptr=FakeMMIO(8), write_ptr=FakeMMIO(8), doorbell=FakeMMIO(8),
                        put_value=7, submit_ib=lambda ib, size: submitted.append((ib, size)), ring_buf=ring_buf)  # type: ignore[arg-type]
    dev = SimpleNamespace(error_state=None)
    desc.signal_doorbell(dev)
    self.assertEqual(submitted, [(ring_buf, 28)])
    self.assertEqual(desc.put_value, 0)
    self.assertEqual(desc.read_ptr[0], 0)
    self.assertEqual(desc.write_ptr[0], 0)

  def test_non_kfd_signal_does_not_emit_event(self):
    dev = SimpleNamespace(target=(8, 0, 3), xccs=1, iface=SimpleNamespace(has_queue_events=False), is_am=lambda: False)
    queue = AMDComputeQueue.__new__(AMDComputeQueue)
    queue.dev, queue.pm4, queue._q, queue.binded_device = dev, pm4_soc15, [], None
    queue.signal(SimpleNamespace(value_addr=0x12345000, owner=dev, is_timeline=True), 7)  # type: ignore[arg-type]
    self.assertEqual(len(queue._q), 7)

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
    queue.dev = SimpleNamespace(target=(8, 0, 3), is_am=lambda: False)
    queue.pm4, queue._q, queue.binded_device = pm4_soc15, [], None
    queue.release_mem(address=0x12345000, value=7, data_sel=pm4_soc15.data_sel__mec_release_mem__send_32_bit_low,
                      int_sel=pm4_soc15.int_sel__mec_release_mem__none, cache_flush=True)
    self.assertEqual(len(queue._q), 7)  # CI/VI has no trailing ctxid dword.
    self.assertEqual(queue._q[0], pm4_soc15.PACKET3(pm4_soc15.PACKET3_RELEASE_MEM, 5) | 2)
    self.assertTrue(queue._q[1] & pm4_soc15.EOP_TC_ACTION_EN)
    self.assertFalse(queue._q[1] & pm4_soc15.EOP_TC_NC_ACTION_EN)
    self.assertEqual((queue._q[1] >> 25) & 3, 2)
    self.assertEqual(queue._q[2], pm4_soc15.DATA_SEL(1) | pm4_soc15.INT_SEL(3))

    queue.dev, queue._q = SimpleNamespace(target=(8, 0, 3), is_am=lambda: True), []
    queue.release_mem(address=0x12345000, value=7, data_sel=pm4_soc15.data_sel__mec_release_mem__send_32_bit_low,
                      int_sel=pm4_soc15.int_sel__mec_release_mem__none, cache_flush=True)
    self.assertEqual(len(queue._q), 7)
    self.assertEqual(queue._q[0], pm4_soc15.PACKET3(pm4_soc15.PACKET3_RELEASE_MEM, 5))
    self.assertTrue(queue._q[1] & pm4_soc15.EOP_TCL1_ACTION_EN)
    self.assertTrue(queue._q[1] & pm4_soc15.EOP_TC_ACTION_EN)
    self.assertEqual(queue._q[2], pm4_soc15.DATA_SEL(1))

  def test_gfx8_direct_barrier_matches_linux_compute_sync(self):
    queue = AMDComputeQueue.__new__(AMDComputeQueue)
    queue.dev = SimpleNamespace(target=(8, 0, 3), is_am=lambda: True)
    queue.pm4, queue._q, queue.binded_device = pm4_soc15, [], None
    queue.memory_barrier()
    self.assertEqual(len(queue._q), 7)
    self.assertEqual(queue._q[2:4], [0xffffffff, 0xff])
    self.assertEqual(queue._q[4:6], [0, 0])

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
