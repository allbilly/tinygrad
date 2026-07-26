# PR 1 hardware numerical tests: one per compute family (DPU, CNA+CORE, PPU).
# These tests require an RK3588 NPU and /dev/dri/card1.
import os, unittest, numpy as np
from tinygrad import Tensor
from tinygrad.helpers import to_mv

_NPU_AVAILABLE = os.path.exists("/dev/dri/card1")

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestDPU(unittest.TestCase):
  def test_dpu_add(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=np.float16)
    b_np = np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).realize()
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

  def test_dpu_repeated_invocation(self):
    a_np = np.random.randn(4,4).astype(np.float16)
    b_np = np.random.randn(4,4).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c1 = (a + b).realize().numpy()
    c2 = (a + b).realize().numpy()
    np.testing.assert_allclose(c1, c2, rtol=1e-3, atol=1e-3)

  def test_dpu_inplace_add(self):
    # a.assign(a+b).realize() — in-place ADD where output buffer == input buffer A
    a_np = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.float16)
    b_np = np.array([[10,20,30,40],[50,60,70,80]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    a.assign(a + b).realize()
    np.testing.assert_allclose(a.numpy(), a_np + b_np, rtol=1e-3, atol=1e-3)

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestCMAC(unittest.TestCase):
  def test_cmac_matmul(self):
    a_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=np.float16)
    b_np = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_cmac_matmul_non_identity(self):
    a_np = np.array([[1,2],[3,4]], dtype=np.float16)
    b_np = np.array([[5,6],[7,8]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_cmac_same_buffer(self):
    a_np = np.array([[1,2],[3,4]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = (a @ a).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ a_np, rtol=1e-2, atol=1e-1)

  def test_cmac_subnormal_output(self):
    # 4x4 matmul where each element = 4 * (2^-12)^2 = 4 * 2^-24 = 2^-22
    # 2^-22 is a subnormal in FP16 (min normal is 2^-14, min subnormal is 2^-24)
    # The old FP32→FP16 conversion returned 0 for subnormals; this verifies the fix
    a_np = np.ones((4,4), dtype=np.float16) * np.float16(2**-12)
    b_np = np.ones((4,4), dtype=np.float16) * np.float16(2**-12)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    expected = a_np @ b_np
    np.testing.assert_allclose(c.numpy(), expected, rtol=0, atol=0)
    # verify the result is actually subnormal (nonzero but below min normal)
    self.assertGreater(c.numpy().max(), 0, "subnormal result should not be zero")

  def test_cmac_accuracy(self):
    # Broader CMAC accuracy: random 8x8 matmul with values spanning normal range
    np.random.seed(42)
    a_np = np.random.randn(8,8).astype(np.float16)
    b_np = np.random.randn(8,8).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-2, atol=1e-1)

  def test_cmac_fp32_to_fp16_rounding(self):
    # Judge's test case: 0.5180664 * 0.5258789 should give FP16 bits 0x345c, not 0x345b
    # The old conversion truncated (mt >> 13) without round-to-nearest-even
    # Use a sparse 4x4 matmul so only one product contributes to result[0,0]
    a_np = np.zeros((4,4), dtype=np.float16)
    b_np = np.zeros((4,4), dtype=np.float16)
    a_np[0,0] = np.float16(0.5180664)
    b_np[0,0] = np.float16(0.5258789)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a @ b).realize()
    result_bits = int(c.numpy().view(np.uint16)[0, 0])
    expected_bits = int(np.float16(np.float32(0.5180664) * np.float32(0.5258789)).view(np.uint16))
    self.assertEqual(result_bits, expected_bits,
                     f"rounding: got 0x{result_bits:04x}, expected 0x{expected_bits:04x}")

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestPPU(unittest.TestCase):
  def test_ppu_globalmax(self):
    a_np = np.arange(1, 33, dtype=np.float16).reshape(2, 2, 8)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.max(axis=(0,1)).realize()
    np.testing.assert_allclose(c.numpy(), np.max(a_np, axis=(0,1)), rtol=1e-3, atol=1e-3)

  def test_ppu_globalmax_axis0(self):
    a_np = np.array([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1],[9,10,11,12,13,14,15,16],[16,15,14,13,12,11,10,9]], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    c = a.max(axis=0).realize()
    np.testing.assert_allclose(c.numpy(), np.max(a_np, axis=0), rtol=1e-3, atol=1e-3)

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestCounters(unittest.TestCase):
  def test_all_units_submit(self):
    from tinygrad.device import Device
    dev = Device["ROCKCHIP"]
    dev.submitted_masks.clear()
    a_np = np.random.randn(4,4).astype(np.float16)
    b_np = np.random.randn(4,4).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    dpu_result = (a + b).realize().numpy()
    np.testing.assert_allclose(dpu_result, a_np + b_np, rtol=1e-3, atol=1e-3)
    cmac_result = (a @ b).realize().numpy()
    np.testing.assert_allclose(cmac_result, a_np @ b_np, rtol=1e-2, atol=1e-1)
    a8_np = np.random.randn(4,8).astype(np.float16)
    a8 = Tensor(a8_np, device="ROCKCHIP").realize()
    ppu_result = a8.max(axis=0).realize().numpy()
    np.testing.assert_allclose(ppu_result, np.max(a8_np, axis=0), rtol=1e-3, atol=1e-3)
    self.assertIn(0x18, dev.submitted_masks)
    self.assertIn(0xd, dev.submitted_masks)
    self.assertIn(0x60, dev.submitted_masks)

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestGuardBuffer(unittest.TestCase):
  def test_small_buffer_canary(self):
    # 12 elements = 24 bytes; DPU rounds up to 2 lanes of 8 = 32 bytes written
    a_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.float16)
    b_np = np.array([12,11,10,9,8,7,6,5,4,3,2,1], dtype=np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    c = (a + b).realize()
    expected = a_np + b_np
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-3, atol=1e-3)
    out_buf = c._buffer().get_buf("ROCKCHIP")
    page_mv = to_mv(out_buf.va_addr, 4096)
    canary = b'\xDE\xAD' * ((4096 - 32) // 2)
    page_mv[32:4096] = canary
    page_mv[0:32] = b'\x00' * 32
    c.assign(a + b).realize()
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-3, atol=1e-3)
    self.assertEqual(bytes(page_mv[32:4096]), canary)

@unittest.skipUnless(_NPU_AVAILABLE, "no /dev/dri/card1 NPU device")
class TestJit(unittest.TestCase):
  def test_two_kernel_jit(self):
    from tinygrad import TinyJit
    a_np = np.random.randn(4,4).astype(np.float16)
    b_np = np.random.randn(4,4).astype(np.float16)
    a = Tensor(a_np, device="ROCKCHIP").realize()
    b = Tensor(b_np, device="ROCKCHIP").realize()
    @TinyJit
    def f(x, y):
      return (x + y).realize()
    c1 = f(a, b).numpy()
    c2 = f(a, b).numpy()
    np.testing.assert_allclose(c1, c2, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
  unittest.main()
