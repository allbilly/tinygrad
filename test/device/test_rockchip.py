import math, os, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.runtime.support.rockchip_telemetry import clear, drain

@unittest.skipUnless(os.path.exists("/dev/dri/card1"), "no RK3588 NPU")
class TestRockchip(unittest.TestCase):
  def test_fused_fp32_intermediate_lerp_native_dpu(self):
    rng = np.random.default_rng(12)
    x,y,z = (rng.uniform(-1,1,33).astype(np.float16) for _ in range(3))
    tx,ty,tz = (Tensor(value,device="ROCKCHIP",dtype=dtypes.half) for value in (x,y,z))
    actual = tx.lerp(ty,tz).realize().numpy()
    expected = (x.astype(np.float32)+(y.astype(np.float32)-x.astype(np.float32))*z.astype(np.float32)).astype(np.float16)
    np.testing.assert_equal(actual,expected)

  def test_zero_base_power_masks_exp2_zero_before_evaluation(self):
    exponent = np.array([-2,-1,0,1,2,3], dtype=np.float16)
    actual = (0**Tensor(exponent, device="ROCKCHIP")).realize().numpy()
    np.testing.assert_equal(actual, np.array([np.inf,np.inf,1,0,0,0], dtype=np.float16))

  def test_wide_fp16_fill_native_dpu_tiles(self):
    actual = Tensor.ones(65536,dtype=dtypes.half,device="ROCKCHIP").realize().numpy()
    np.testing.assert_equal(actual, np.ones(65536,dtype=np.float16))

  def test_dense_fp16_row_sum_native_cmac(self):
    data = np.linspace(-2, 2, 8*32, dtype=np.float16).reshape(8,32)
    actual = Tensor(data, device="ROCKCHIP").realize().sum(axis=1).realize().numpy()
    np.testing.assert_equal(actual, data.astype(np.float32).sum(axis=1).astype(np.float16))

  def test_small_dynamic_fp16_gemm_native_pack_compute_unpack(self):
    rng = np.random.default_rng(19)
    for size in (4,8,9):
      with self.subTest(size=size):
        x, y = (rng.uniform(-.25, .25, (size,size)).astype(np.float16) for _ in range(2))
        actual = (Tensor(x, device="ROCKCHIP").realize()@Tensor(y, device="ROCKCHIP").realize()).realize().numpy()
        expected = (x.astype(np.float32)@y.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_fused_cmac_bias_relu_avoids_intermediate_rounding(self):
    rng = np.random.default_rng(0)
    x = rng.uniform(-2,2,(1,8,5,5)).astype(np.float16)
    weight = rng.uniform(-2,2,(8,8,1,1)).astype(np.float16)
    bias = rng.uniform(-2,2,(8,)).astype(np.float16)
    tx, tw, tb = (Tensor(value,device="ROCKCHIP",dtype=dtypes.half) for value in (x,weight,bias))
    actual = tx.conv2d(tw,tb).relu().conv2d(tw,tb).realize().numpy()
    first = np.maximum(np.einsum("nchw,oc->nohw", x.astype(np.float32), weight[:,:,0,0].astype(np.float32))+
                       bias.astype(np.float32)[None,:,None,None], 0).astype(np.float16)
    expected = (np.einsum("nchw,oc->nohw", first.astype(np.float32), weight[:,:,0,0].astype(np.float32))+
                bias.astype(np.float32)[None,:,None,None]).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_batched_direct_spatial_conv_uses_cna(self):
    rng = np.random.default_rng(17)
    x = rng.uniform(-2,2,(2,4,9,9)).astype(np.float16)
    weight = rng.uniform(-2,2,(4,4,3,3)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try: actual = Tensor(x,device="ROCKCHIP").conv2d(Tensor(weight,device="ROCKCHIP")).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.zeros((2,4,7,7),dtype=np.float32)
    for ky in range(3):
      for kx in range(3):
        expected += np.einsum("nihw,oi->nohw",x[:,:,ky:ky+7,kx:kx+7].astype(np.float32),weight[:,:,ky,kx].astype(np.float32))
    np.testing.assert_allclose(actual,expected.astype(np.float16),rtol=1e-3,atol=1e-6)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("CONV") == 2 for event in kernels))

  def test_depthwise_channel_planes_use_cna(self):
    rng = np.random.default_rng(29)
    x = rng.uniform(-1,1,(2,3,11,28)).astype(np.float16)
    weight = rng.uniform(-1,1,(3,1,3,3)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try: actual = Tensor(x,device="ROCKCHIP").conv2d(Tensor(weight,device="ROCKCHIP"),groups=3).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.zeros((2,3,9,26),dtype=np.float32)
    for ky in range(3):
      for kx in range(3): expected += x[:,:,ky:ky+9,kx:kx+26].astype(np.float32)*weight[:,0,ky,kx][None,:,None,None]
    np.testing.assert_allclose(actual,expected.astype(np.float16),rtol=1e-3,atol=1e-6)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("CONV") == 6 for event in kernels))

  def test_grouped_channel_tiles_use_cna(self):
    rng = np.random.default_rng(30)
    x = rng.uniform(-1,1,(2,4,5,5)).astype(np.float16)
    weight = rng.uniform(-1,1,(6,2,3,3)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try: actual = Tensor(x,device="ROCKCHIP").conv2d(Tensor(weight,device="ROCKCHIP"),groups=2).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.zeros((2,6,3,3),dtype=np.float32)
    for group in range(2):
      for ky in range(3):
        for kx in range(3):
          expected[:,group*3:(group+1)*3] += np.einsum("nihw,oi->nohw",
            x[:,group*2:(group+1)*2,ky:ky+3,kx:kx+3].astype(np.float32),
            weight[group*3:(group+1)*3,:,ky,kx].astype(np.float32))
    np.testing.assert_allclose(actual,expected.astype(np.float16),rtol=1e-3,atol=1e-6)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("CONV") == 4 for event in kernels))

  def test_valid_sliding_max_uses_ppu(self):
    rng = np.random.default_rng(31)
    x = rng.uniform(-8,8,(2,3,9,9)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try: actual = Tensor(x,device="ROCKCHIP").max_pool2d((5,5),stride=1).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.empty((2,3,5,5),dtype=np.float16)
    for y in range(5):
      for x0 in range(5): expected[:,:,y,x0] = np.max(x[:,:,y:y+5,x0:x0+5],axis=(2,3))
    np.testing.assert_equal(actual,expected)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("PPU") == 1 for event in kernels))

  def test_channel16_direct_spatial_conv_uses_cna(self):
    rng = np.random.default_rng(18)
    x = rng.uniform(-1,1,(1,16,9,9)).astype(np.float16)
    weight = rng.uniform(-1,1,(16,16,3,3)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try: actual = Tensor(x,device="ROCKCHIP").conv2d(Tensor(weight,device="ROCKCHIP")).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.zeros((1,16,7,7),dtype=np.float32)
    for ky in range(3):
      for kx in range(3):
        expected += np.einsum("nihw,oi->nohw",x[:,:,ky:ky+7,kx:kx+7].astype(np.float32),weight[:,:,ky,kx].astype(np.float32))
    np.testing.assert_allclose(actual,expected.astype(np.float16),rtol=1e-3,atol=1e-6)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("CONV") == 1 for event in kernels))

  def test_nhwc_channel10_convolution_uses_cbuf_split_cna(self):
    rng = np.random.default_rng(32)
    x = rng.uniform(-.5,.5,(2,9,9,10)).astype(np.float16)
    weight = rng.uniform(-.5,.5,(3,3,10,20)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try:
      actual = Tensor(x,device="ROCKCHIP").permute(0,3,1,2).conv2d(
        Tensor(weight,device="ROCKCHIP").permute(3,2,0,1)).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.zeros((2,20,7,7),dtype=np.float32)
    for ky in range(3):
      for kx in range(3):
        expected += np.einsum("byxc,co->boyx",x[:,ky:ky+7,kx:kx+7].astype(np.float32),weight[ky,kx].astype(np.float32))
    np.testing.assert_allclose(actual,expected.astype(np.float16),rtol=1e-3,atol=1e-5)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("CONV") == 4 for event in kernels))

  def test_k65_contractions_use_k96_cmac_tile(self):
    rng = np.random.default_rng(33)
    for lhs_shape,rhs_shape in (((65,),(65,45)),((45,65),(65,))):
      with self.subTest(lhs=lhs_shape,rhs=rhs_shape):
        lhs = rng.uniform(-.5,.5,lhs_shape).astype(np.float16)
        rhs = rng.uniform(-.5,.5,rhs_shape).astype(np.float16)
        actual = (Tensor(lhs,device="ROCKCHIP").realize()@Tensor(rhs,device="ROCKCHIP").realize()).realize().numpy()
        expected = (lhs.astype(np.float32)@rhs.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-6)

  def test_rgb_strided_spatial_conv_uses_cna(self):
    rng = np.random.default_rng(21)
    x = rng.uniform(-1,1,(2,3,7,10)).astype(np.float16)
    weight = rng.uniform(-1,1,(4,3,3,3)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try: actual = Tensor(x,device="ROCKCHIP").conv2d(Tensor(weight,device="ROCKCHIP"),stride=(2,1)).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.zeros((2,4,3,8),dtype=np.float32)
    for ky in range(3):
      for kx in range(3):
        expected += np.einsum("nihw,oi->nohw",x[:,:,ky:ky+6:2,kx:kx+8].astype(np.float32),weight[:,:,ky,kx].astype(np.float32))
    np.testing.assert_allclose(actual,expected.astype(np.float16),rtol=1e-3,atol=1e-6)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("CONV") == 2 for event in kernels))

  def test_pointwise_spatial_conv_uses_cna(self):
    rng = np.random.default_rng(23)
    x = rng.uniform(-1,1,(1,16,8,8)).astype(np.float16)
    weight = rng.uniform(-1,1,(16,16,1,1)).astype(np.float16)
    old_telemetry = os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_TELEMETRY"] = "memory"
    clear()
    try: actual = Tensor(x,device="ROCKCHIP").conv2d(Tensor(weight,device="ROCKCHIP")).realize().numpy()
    finally:
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY",None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry
    expected = np.einsum("nchw,oc->nohw",x.astype(np.float32),weight[:,:,0,0].astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-6)
    kernels = [event for event in drain() if event["kind"] == "kernel"]
    self.assertTrue(any(event.get("engine_counts",{}).get("CONV") == 1 for event in kernels))

  def test_contiguous_fp16_sum_native_dpu_cmac(self):
    rng = np.random.default_rng(4)
    for count in (2, 16, 60, 135, 720, 16384):
      with self.subTest(count=count):
        data = rng.uniform(-0.5, 0.5, count).astype(np.float16)
        actual = Tensor(data, device="ROCKCHIP").realize().sum().realize().item()
        expected = data.astype(np.float32).sum().astype(np.float16).item()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)
    np.random.seed(0)
    official = np.random.uniform(-2, 2, (45,3)).astype(np.float16)
    actual = Tensor(official, device="ROCKCHIP").realize().sum().realize().item()
    expected = official.astype(np.float32).sum().astype(np.float16).item()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_contiguous_fp16_sum_commits_fp32_accumulator(self):
    data = np.linspace(-2,2,135,dtype=np.float16)
    actual = Tensor(data,device="ROCKCHIP").realize().sum(dtype=dtypes.float).realize().numpy()
    np.testing.assert_equal(actual,data.astype(np.float32).sum())

  def test_nested_dense_fp16_sum_native_dpu_cmac(self):
    data = np.linspace(-1,1,64,dtype=np.float16).reshape(4,4,4)
    tensor = Tensor(data,device="ROCKCHIP").realize()
    for axes in ((0,1), (0,2), (1,2)):
      with self.subTest(axes=axes):
        actual = tensor.sum(axes).sum().realize().item()
        expected = data.astype(np.float32).sum(axis=axes).astype(np.float16).astype(np.float32).sum().astype(np.float16).item()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_short_scalar_fp16_product_native_cmac_dpu(self):
    for data in (np.array([1,2,3],dtype=np.float16), np.full(9,2,dtype=np.float16),
                 np.linspace(.75,1.25,9,dtype=np.float16)):
      with self.subTest(count=data.size):
        actual = Tensor(data,device="ROCKCHIP").realize().prod().realize().item()
        expected = np.multiply.accumulate(data,dtype=np.float16)[-1].item()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_affine_fp16_product_native_cmac_dpu(self):
    data = np.linspace(.75,1.25,3*4*5*6,dtype=np.float16).reshape(3,4,5,6)
    tensor = Tensor(data,device="ROCKCHIP").realize()
    for axis in (1,3):
      with self.subTest(axis=axis):
        actual = tensor.prod(axis=axis).realize().numpy()
        expected = np.multiply.accumulate(data,axis=axis,dtype=np.float16).take(-1,axis=axis)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_masked_affine_fp16_cumprod_native_cmac_dpu(self):
    data = np.linspace(.8,1.2,10,dtype=np.float16)
    actual = Tensor(data,device="ROCKCHIP").realize().cumprod(0).realize().numpy()
    np.testing.assert_allclose(actual, np.cumprod(data,dtype=np.float16), rtol=1e-3, atol=1e-6)

  def test_contiguous_fp16_mean_native_cmac(self):
    np.random.seed(0)
    data = np.random.uniform(-2, 2, (3,4,5,6)).astype(np.float16)
    actual = Tensor(data, device="ROCKCHIP").realize().mean().realize().item()
    expected = (data.astype(np.float32).sum()*np.float32(1/data.size)).astype(np.float16).item()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_contiguous_fp16_relu_sum_native_dpu_cmac(self):
    np.random.seed(0)
    data = np.random.uniform(-2, 2, (3,4,5)).astype(np.float16)
    actual = Tensor(data, device="ROCKCHIP").realize().relu().sum().relu().realize().item()
    expected = np.maximum(data.astype(np.float32), 0).sum().astype(np.float16).item()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_affine_fp16_reductions_native_cmac(self):
    np.random.seed(0)
    data = np.random.uniform(-2, 2, (3,4,5,6)).astype(np.float16)
    tensor = Tensor(data, device="ROCKCHIP").realize()
    for axes in (3, (1,3), (0,2), (1,2), 1):
      with self.subTest(axes=axes):
        actual = tensor.sum(axis=axes).realize().numpy()
        expected = data.astype(np.float32).sum(axis=axes).astype(np.float16)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

    actual = tensor.mean(axis=(1,2)).realize().numpy()
    expected = (data.astype(np.float32).sum(axis=(1,2))*np.float32(1/20)).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)
    tiny = np.random.uniform(-2, 2, (4,2,2)).astype(np.float16)
    actual = Tensor(tiny, device="ROCKCHIP").realize().sum(axis=(0,2)).realize().numpy()
    np.testing.assert_allclose(actual, tiny.astype(np.float32).sum(axis=(0,2)).astype(np.float16), rtol=1e-3, atol=1e-6)

  def test_wide_dense_row_sum_native_two_level_cmac(self):
    data = np.linspace(-.25,.25,256*256,dtype=np.float16).reshape(256,256)
    tensor = Tensor(data,device="ROCKCHIP").realize()
    for scale in (1.0, 0.25):
      with self.subTest(scale=scale):
        actual = (tensor.sum(axis=1)*scale).realize().numpy()
        expected = (data.astype(np.float32).sum(axis=1)*scale).astype(np.float16)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_multi_source_affine_sum_native_cmac_dpu(self):
    lhs = np.linspace(-.125,.125,256*256,dtype=np.float16).reshape(256,256)
    rhs = np.linspace(.0625,-.0625,256*64,dtype=np.float16).reshape(256,64)
    actual = Tensor.cat(Tensor(lhs,device="ROCKCHIP").realize(), Tensor(rhs,device="ROCKCHIP").realize(), dim=1).sum(axis=1).realize().numpy()
    expected = np.concatenate((lhs,rhs),axis=1).astype(np.float32).sum(axis=1).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_masked_affine_prefix_sum_native_cmac(self):
    data = np.array([1,-2,3,-4,5,-6,7,-8,9,-10], dtype=np.float16)
    actual = Tensor(data, device="ROCKCHIP").realize().cumsum(0).realize().numpy()
    np.testing.assert_equal(actual, np.cumsum(data, dtype=np.float16))

  def test_windowed_affine_average_native_cmac(self):
    data = np.linspace(-1,1,2*2*11*28,dtype=np.float16).reshape(2,2,11,28)
    actual = Tensor(data,device="ROCKCHIP").realize().avg_pool2d(2).realize().numpy()
    expected = data[:,:,:10,:].astype(np.float32).reshape(2,2,5,2,14,2).mean(axis=(3,5)).astype(np.float16)
    np.testing.assert_equal(actual, expected)

  def test_pointwise_affine_variance_reduction_stays_native(self):
    # A recycled 30-lane upload previously exposed stale NaNs from the last physical atom to selector CMAC padding.
    self.assertTrue(np.isnan((Tensor.zeros(30,device="ROCKCHIP")/0).realize().numpy()).all())
    rng = np.random.default_rng(35)
    data = rng.uniform(-2,2,30).astype(np.float16).reshape(1,2,3,1,5)
    actual = Tensor(data,device="ROCKCHIP").realize().var(axis=(0,4),correction=5).realize().numpy()
    self.assertTrue(np.isposinf(actual).all())

  def test_pointwise_fp16_expression_before_affine_reduction(self):
    rng = np.random.default_rng(17)
    x = rng.uniform(-1, 1, (3,4,5)).astype(np.float16)
    y = rng.uniform(-1, 1, (3,4,5)).astype(np.float16)
    tx, ty = Tensor(x, device="ROCKCHIP").realize(), Tensor(y, device="ROCKCHIP").realize()
    actual = ((tx+ty)*tx).sum(axis=1).realize().numpy()
    expected = ((x+y)*x).astype(np.float32).sum(axis=1).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_global_max_hwc8_native_ppu(self):
    for height,width in ((2,2), (4,4), (16,16)):
      with self.subTest(shape=(height,width,8)):
        data = np.linspace(-8, 8, height*width*8, dtype=np.float16).reshape(height,width,8)
        actual = Tensor(data, device="ROCKCHIP").realize().max(axis=(0,1)).realize().numpy()
        np.testing.assert_equal(actual, data.max(axis=(0,1)))

  def test_scalar_multiaxis_max_native_cmac_ppu(self):
    data = np.array([[[[-2,3,1], [4,-1,2]]]],dtype=np.float16)
    actual = Tensor(data,device="ROCKCHIP").realize().max_pool2d((2,2)).realize().numpy()
    np.testing.assert_equal(actual, np.array([[[[4]]]],dtype=np.float16))

  def test_padded_ceil_max_pool_native_cmac_ppu(self):
    data = np.array([[[[-8,-7,-6,-5,-4,-3], [-2,-1,0,1,2,3], [4,5,6,7,8,9],
                       [10,11,12,13,14,15], [16,17,18,19,20,21], [22,23,24,25,26,27]]]], dtype=np.float16)
    actual = Tensor(data,device="ROCKCHIP").realize().max_pool2d((3,3),stride=3,padding=1,ceil_mode=True).realize().numpy()
    padded = np.pad(data, ((0,0),(0,0),(1,2),(1,2)), constant_values=-np.inf)
    expected = np.empty((1,1,3,3), dtype=np.float16)
    for y in range(3):
      for x in range(3): expected[:,:,y,x] = padded[:,:,y*3:y*3+3,x*3:x*3+3].max(axis=(2,3))
    np.testing.assert_equal(actual, expected)

  def test_windowed_affine_max_pool_native_cmac_ppu(self):
    data = np.linspace(-4,4,2*11*28,dtype=np.float16).reshape(2,1,11,28)
    actual = Tensor(data,device="ROCKCHIP").realize().max_pool2d((2,2),stride=2).realize().numpy()
    expected = data[:,:,:10,:].reshape(2,1,5,2,14,2).max(axis=(3,5))
    np.testing.assert_equal(actual, expected)
    actual = Tensor(data,device="ROCKCHIP").realize().max_pool2d((2,2),padding=1).realize().numpy()
    padded = np.pad(data, ((0,0),(0,0),(1,1),(1,1)), constant_values=-np.inf)
    expected = padded[:,:,:12,:].reshape(2,1,6,2,15,2).max(axis=(3,5))
    np.testing.assert_equal(actual, expected)

  def test_wide_atom_dilated_max_pool_native_cmac_ppu(self):
    data = np.linspace(-4,4,3*2*17*14,dtype=np.float16).reshape(3,2,17,14)
    actual = Tensor(data,device="ROCKCHIP").realize().max_pool2d((5,5),dilation=(2,3)).realize().numpy()
    expected = np.empty((3,2,2,1),dtype=np.float16)
    for y in range(2):
      expected[:,:,y,0] = np.stack([data[:,:,y*5+ky*2,kx*3] for ky in range(5) for kx in range(5)]).max(axis=0)
    np.testing.assert_equal(actual, expected)

  def test_dense_fp16_global_extrema_native_dpu(self):
    for count in (2, 8, 9, 135):
      with self.subTest(count=count):
        data = np.linspace(-8, 8, count, dtype=np.float16)
        tensor = Tensor(data, device="ROCKCHIP").realize()
        np.testing.assert_equal(tensor.max().realize().item(), data.max().item())
    data = np.linspace(-8, 8, 135, dtype=np.float16)
    tensor = Tensor(data, device="ROCKCHIP").realize()
    np.testing.assert_equal((tensor.max()*0.5).realize().item(), (data.max()*np.float16(0.5)).item())
    np.testing.assert_equal(tensor.min().realize().item(), data.min().item())

  def test_affine_fp16_max_native_ppu_batches(self):
    data = np.linspace(-8, 8, 3*4*5*6, dtype=np.float16).reshape(3,4,5,6)
    actual = Tensor(data, device="ROCKCHIP").realize().max(axis=1).realize().numpy()
    np.testing.assert_equal(actual, data.max(axis=1))

  def test_static_conditional_fp16_reformat_native_cmac(self):
    data = np.arange(20, dtype=np.float16).reshape(4,5)
    tensor = Tensor(data, device="ROCKCHIP").realize()
    for diagonal in (-2,0,2):
      with self.subTest(diagonal=diagonal):
        np.testing.assert_equal(tensor.tril(diagonal).realize().numpy(), np.tril(data, diagonal))

  def test_affine_hwc8_movements_native_dpu(self):
    data = np.arange(2*3*8, dtype=np.float16).reshape(2,3,8)
    tensor = Tensor(data, device="ROCKCHIP").realize()
    for actual,expected in ((tensor.permute(1,0,2).contiguous(), data.transpose(1,0,2)),
                            (tensor.flip(0).contiguous(), data[::-1])):
      np.testing.assert_equal(actual.realize().numpy(), expected)
    expanded = Tensor(data[:,:1], device="ROCKCHIP").realize().expand(2,3,8).contiguous().realize().numpy()
    np.testing.assert_equal(expanded, np.broadcast_to(data[:,:1], (2,3,8)))

  def test_affine_fp16_broadcast_alu_native_cmac_dpu(self):
    lhs = np.linspace(1, 3, 27, dtype=np.float16).reshape(3,9)
    rhs = np.array([[1], [2], [4]], dtype=np.float16)
    x, y = Tensor(lhs, device="ROCKCHIP").realize(), Tensor(rhs, device="ROCKCHIP").realize()
    np.testing.assert_equal((x+y).realize().numpy(), lhs+rhs)
    np.testing.assert_allclose((x/y).realize().numpy(), lhs/rhs, rtol=1e-3, atol=1e-6)

  def test_zero_masked_affine_surface_native_cmac_dpu(self):
    lhs, rhs = np.arange(64,dtype=np.float16).reshape(8,8), np.arange(36,dtype=np.float16).reshape(6,6)
    actual = (Tensor(lhs,device="ROCKCHIP").realize()+Tensor(rhs,device="ROCKCHIP").realize().pad((1,1,1,1))).realize().numpy()
    np.testing.assert_equal(actual, lhs+np.pad(rhs, ((1,1),(1,1))))

  def test_multiaxis_tall_affine_contraction_native_cmac(self):
    data = np.linspace(-1,1,50,dtype=np.float16).reshape(1,2,5,5)
    weight = np.linspace(-.5,.5,36,dtype=np.float16).reshape(2,2,3,3)
    expected = Tensor(data).conv2d(Tensor(weight)).numpy()
    actual = Tensor(data,device="ROCKCHIP").realize().conv2d(Tensor(weight,device="ROCKCHIP").realize()).realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_sparse_pair_affine_contraction_native_cmac(self):
    data = np.linspace(-1,1,3*5*7,dtype=np.float16).reshape(1,3,5,7)
    for weight,groups in ((np.linspace(-.5,.5,6*3*3,dtype=np.float16).reshape(6,1,3,3),3),
                          (np.linspace(-.5,.5,6*3*3*5,dtype=np.float16).reshape(6,3,3,5),1)):
      with self.subTest(groups=groups):
        expected = Tensor(data,device="CPU").conv2d(Tensor(weight,device="CPU"),groups=groups).numpy()
        actual = Tensor(data,device="ROCKCHIP").realize().conv2d(
          Tensor(weight,device="ROCKCHIP").realize(),groups=groups).realize().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_zero_aware_windowed_contraction_native_cmac(self):
    data = np.linspace(-1,1,6*2*11,dtype=np.float16).reshape(6,2,11)
    weight = np.linspace(-.5,.5,6*2*5,dtype=np.float16).reshape(6,2,5)
    expected = Tensor(data,device="CPU").conv2d(Tensor(weight,device="CPU"),padding=(1,1)).numpy()
    actual = Tensor(data,device="ROCKCHIP").realize().conv2d(
      Tensor(weight,device="ROCKCHIP").realize(),padding=(1,1)).realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_tall_k4_contraction_native_cmac(self):
    data = np.linspace(-1,1,4*9*9,dtype=np.float16).reshape(1,4,9,9)
    weight = np.linspace(-.5,.5,4*4,dtype=np.float16).reshape(4,4,1,1)
    expected = Tensor(data,device="CPU").conv2d(Tensor(weight,device="CPU")).numpy()
    actual = Tensor(data,device="ROCKCHIP").realize().conv2d(Tensor(weight,device="ROCKCHIP").realize()).realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_second_cmac_output_atom_native_contraction(self):
    lhs = np.linspace(-1,1,4*9,dtype=np.float16).reshape(4,9)
    rhs = np.linspace(-.5,.5,9*24,dtype=np.float16).reshape(9,24)
    expected = Tensor(lhs,device="CPU").matmul(Tensor(rhs,device="CPU")).numpy()
    actual = Tensor(lhs,device="ROCKCHIP").realize().matmul(Tensor(rhs,device="ROCKCHIP").realize()).realize().numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-6)

  def test_wide_cmac_output_groups_native_contraction(self):
    for m,k,n in ((4,9,40),(1,64,40)):
      with self.subTest(m=m,k=k,n=n):
        lhs = np.linspace(-1,1,m*k,dtype=np.float16).reshape(m,k)
        rhs = np.linspace(-.5,.5,k*n,dtype=np.float16).reshape(k,n)
        expected = Tensor(lhs,device="CPU").matmul(Tensor(rhs,device="CPU")).numpy()
        actual = Tensor(lhs,device="ROCKCHIP").realize().matmul(Tensor(rhs,device="ROCKCHIP").realize()).realize().numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-6)

  def test_direct_aligned_contraction_windows_native_cmac(self):
    for input_shape,weight_shape in (((1,4,9,9),(4,4,3,3)), ((8,1,11),(6,1,2)), ((8,3,11),(6,3,2))):
      with self.subTest(input_shape=input_shape):
        data = np.linspace(-1,1,np.prod(input_shape),dtype=np.float16).reshape(input_shape)
        weight = np.linspace(-.5,.5,np.prod(weight_shape),dtype=np.float16).reshape(weight_shape)
        expected = Tensor(data,device="CPU").conv2d(Tensor(weight,device="CPU")).numpy()
        actual = Tensor(data,device="ROCKCHIP").realize().conv2d(Tensor(weight,device="ROCKCHIP").realize()).realize().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_multi_broadcast_and_tiled_m_contractions_native(self):
    for input_shape,weight_shape,groups in (((1,1,11),(6,1,1),1), ((8,3,11),(6,1,5),3)):
      with self.subTest(input_shape=input_shape):
        data = np.linspace(-1,1,np.prod(input_shape),dtype=np.float16).reshape(input_shape)
        weight = np.linspace(-.5,.5,np.prod(weight_shape),dtype=np.float16).reshape(weight_shape)
        expected = Tensor(data,device="CPU").conv2d(Tensor(weight,device="CPU"),groups=groups).numpy()
        actual = Tensor(data,device="ROCKCHIP").realize().conv2d(
          Tensor(weight,device="ROCKCHIP").realize(),groups=groups).realize().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_zero_masked_affine_contraction_native_cmac(self):
    data, weight = np.arange(3,dtype=np.float16).reshape(1,1,3), np.array([[[1,-1]]],dtype=np.float16)
    expected = Tensor(data).conv2d(Tensor(weight),padding=(0,1)).numpy()
    actual = Tensor(data,device="ROCKCHIP").realize().conv2d(
      Tensor(weight,device="ROCKCHIP").realize(),padding=(0,1)).realize().numpy()
    np.testing.assert_equal(actual, expected)

  def test_sparse_affine_movements_native_cmac(self):
    cases = (
      (np.arange(9, dtype=np.float16).reshape(3,3), lambda x:x.T, lambda x:x.T),
      (np.arange(27, dtype=np.float16).reshape(3,3,3), lambda x:x.transpose(0,2), lambda x:x.transpose(2,1,0)),
      (np.arange(360, dtype=np.float16).reshape(3,4,5,6), lambda x:x.permute(3,2,1,0), lambda x:x.transpose(3,2,1,0)),
      (np.arange(432, dtype=np.float16).reshape(4,3,6,6), lambda x:x.flip((0,1,3)), lambda x:x[::-1,::-1,:,::-1]),
      (np.arange(72, dtype=np.float16).reshape(4,3,1,6), lambda x:x.expand(6,1,4,3,2,6),
       lambda x:np.broadcast_to(x, (6,1,4,3,2,6))))
    for data,tensor_op,numpy_op in cases:
      with self.subTest(shape=data.shape):
        actual = tensor_op(Tensor(data, device="ROCKCHIP").realize()).contiguous().realize().numpy()
        np.testing.assert_equal(actual, numpy_op(data))
    scalar = np.array([[[[0.1953125]]]], dtype=np.float16)
    for _ in range(8):
      actual = Tensor(scalar, device="ROCKCHIP").realize().expand(4,3,2,6).contiguous().realize().numpy()
      np.testing.assert_equal(actual, np.broadcast_to(scalar, (4,3,2,6)))

    square = np.arange(64,dtype=np.float16).reshape(8,8)
    actual = Tensor(square,device="ROCKCHIP").realize().T.contiguous().realize().numpy()
    np.testing.assert_equal(actual,square.T)

  def test_static_padding_constants_native_reformat(self):
    data = np.linspace(-2,2,3*3*3*3,dtype=np.float16).reshape(3,3,3,3)
    for fill in (5.0,3.456,np.inf,-np.inf):
      with self.subTest(fill=fill):
        actual = Tensor(data,device="ROCKCHIP").realize().pad((1,2,3,4),value=float(fill)).contiguous().realize().numpy()
        expected = np.pad(data,((0,0),(0,0),(3,4),(1,2)),constant_values=fill)
        np.testing.assert_equal(actual,expected)

  def test_static_split_and_modulo_movements_native_cmac(self):
    data = np.arange(9,dtype=np.float16).reshape(3,3)
    tensor = Tensor(data,device="ROCKCHIP").realize()
    np.testing.assert_equal(tensor.repeat(3,3,4).contiguous().realize().numpy(), np.tile(data,(3,3,4)))
    rolled = np.arange(32,dtype=np.float16).reshape(4,8)
    np.testing.assert_equal(Tensor(rolled,device="ROCKCHIP").realize().roll(1).contiguous().realize().numpy(), np.roll(rolled,1))
    large = np.arange(72,dtype=np.float16).reshape(4,6,3)
    repeated = Tensor(large,device="ROCKCHIP").realize().repeat(2,4,3,3,2,2).contiguous().realize().numpy()
    np.testing.assert_equal(repeated, np.tile(large,(2,4,3,3,2,2)))

  def test_static_cast_nearest_reformat_native_cmac(self):
    data = np.arange(2*3*13,dtype=np.float16).reshape(2,3,13)
    tensor = Tensor(data,device="ROCKCHIP").realize()
    for mode,indexes in (("nearest", (0,1,2,4,5,7,8,10,11)), ("nearest-exact", (0,2,3,5,6,7,9,10,12))):
      with self.subTest(mode=mode):
        np.testing.assert_equal(tensor.interpolate((9,),mode=mode).realize().numpy(), data[...,indexes])

  def test_large_constant_run_broadcast_native_cmac(self):
    lhs = np.arange(32*32*32,dtype=np.float16).reshape(1,32,32,32)/np.float16(1024)
    rhs = np.arange(32,dtype=np.float16).reshape(1,32,1,1)/np.float16(32)
    actual = (Tensor(lhs,device="ROCKCHIP").realize()+Tensor(rhs,device="ROCKCHIP").realize()).realize().numpy()
    np.testing.assert_equal(actual,lhs+rhs)

  def test_input_dma_atom_padding_survives_lut_then_cmac(self):
    zero, negative = Tensor([0], dtype=dtypes.half, device="ROCKCHIP"), Tensor([-.7], dtype=dtypes.half, device="ROCKCHIP")
    np.testing.assert_equal((zero.log2()*negative).realize().numpy(), np.array([np.inf], dtype=np.float16))
    scalar = np.array([[[[0.1953125]]]], dtype=np.float16)
    actual = Tensor(scalar, device="ROCKCHIP").realize().expand(4,3,2,6).contiguous().realize().numpy()
    np.testing.assert_equal(actual, np.broadcast_to(scalar, (4,3,2,6)))

  def test_unaligned_contiguous_slice_native_dpu(self):
    data = np.arange(24, dtype=np.float16)
    tensor = Tensor(data, device="ROCKCHIP").realize()
    for offset,length in (*((offset,16) for offset in range(1,8)), (3,3)):
      with self.subTest(offset=offset, length=length):
        actual = tensor[offset:offset+length].contiguous().realize().numpy()
        np.testing.assert_equal(actual, data[offset:offset+length])

  def test_python_fallback_mapped_buffer_coherence(self):
    old_fallback, old_telemetry = os.environ.get("ROCKCHIP_FALLBACK"), os.environ.get("ROCKCHIP_TELEMETRY")
    os.environ["ROCKCHIP_FALLBACK"], os.environ["ROCKCHIP_TELEMETRY"] = "PYTHON", "memory"
    try:
      clear()
      data = np.linspace(-1, 1, 17, dtype=np.float16)
      x = Tensor(data, device="ROCKCHIP").realize()
      native_before = (x+0.25).realize()
      fallback = native_before.tan().realize()
      actual = (fallback*2).realize().numpy()
      np.testing.assert_allclose(actual, (np.tan(data+0.25)*2).astype(np.float16), rtol=2e-3, atol=2e-3)
      lanes = [event["lane"] for event in drain() if event["kind"] == "kernel"]
      self.assertEqual(lanes[-3:], ["RK_DPU", "PYTHON", "RK_DPU"])
    finally:
      if old_fallback is None: os.environ.pop("ROCKCHIP_FALLBACK", None)
      else: os.environ["ROCKCHIP_FALLBACK"] = old_fallback
      if old_telemetry is None: os.environ.pop("ROCKCHIP_TELEMETRY", None)
      else: os.environ["ROCKCHIP_TELEMETRY"] = old_telemetry

  def test_dpu_binary_and_multistage(self):
    rng = np.random.default_rng(1)
    values = [rng.uniform(-2, 2, 16).astype(np.float16) for _ in range(4)]
    a, b, c, d = (Tensor(x, device="ROCKCHIP").realize() for x in values)
    for out, expected in ((a+b, values[0]+values[1]), (a*b, values[0]*values[1]),
                          (a.maximum(b), np.maximum(values[0], values[1])), (a/b, values[0]/values[1]),
                          (((a+b)*c)+d, ((values[0]+values[1])*values[2])+values[3])):
      np.testing.assert_allclose(out.realize().numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_dpu_division_infinite_numerator_sign(self):
    data = np.array([-2, -1, 1, 2, np.nan], dtype=np.float16)
    x = Tensor(data, device="ROCKCHIP").realize()
    for numerator in (np.inf, -np.inf):
      np.testing.assert_equal((numerator/x).realize().numpy(), numerator/data)

  def test_dpu_scalar_and_fill(self):
    data = np.linspace(-2, 2, 16, dtype=np.float16)
    x = Tensor(data, device="ROCKCHIP").realize()
    np.testing.assert_allclose((x*2).realize().numpy(), data*2, rtol=1e-3, atol=1e-3)
    np.testing.assert_equal(Tensor.full((16,), 3.5, dtype=dtypes.half, device="ROCKCHIP").realize().numpy(), np.full(16, 3.5, np.float16))
    np.testing.assert_equal(Tensor.ones((), dtype=dtypes.half, device="ROCKCHIP").realize().numpy(), np.ones((), np.float16))
    np.testing.assert_equal(Tensor.full((2925,), 4, dtype=dtypes.int, device="ROCKCHIP").realize().numpy(), np.full(2925, 4, np.int32))
    np.testing.assert_equal(Tensor.full((6,), 4, dtype=dtypes.float, device="ROCKCHIP").realize().numpy(), np.full(6, 4, np.float32))
    np.testing.assert_equal(Tensor.full((257,), 4, dtype=dtypes.float, device="ROCKCHIP").realize().numpy(), np.full(257, 4, np.float32))

  def test_where_uses_native_mask(self):
    a = np.linspace(-2, 2, 16, dtype=np.float16)
    b = np.linspace(3, 6, 16, dtype=np.float16)
    ta, tb = Tensor(a, device="ROCKCHIP").realize(), Tensor(b, device="ROCKCHIP").realize()
    np.testing.assert_equal((ta<0).where(ta, tb).realize().numpy(), np.where(a<0, a, b))
    special = np.array([-np.inf, -2, 0, 2, np.inf], dtype=np.float16)
    ts = Tensor(special, device="ROCKCHIP").realize()
    np.testing.assert_equal((ts<0).where(ts, 1).realize().numpy(), np.where(special<0, special, 1))

  def test_infinite_threshold_masked_fill_native_dpu(self):
    data = np.array([-np.inf,-2,0,.2,np.inf,np.nan], dtype=np.float16)
    for greater in (True,False):
      tensor = Tensor(data, device="ROCKCHIP").realize()
      condition = tensor>0.1 if greater else tensor<0.1
      actual = tensor.masked_fill(condition.detach(), -float("inf")).realize().numpy()
      expected = np.where(data>0.1 if greater else data<0.1, -np.inf, data)
      np.testing.assert_equal(actual, expected)

  def test_fp16_abs_specials_and_finite_extrema(self):
    data = np.array([-2, -0., 0., 2., np.inf, -np.inf, np.nan, -np.nan], dtype=np.float16)
    x = Tensor(data, device="ROCKCHIP").realize()
    np.testing.assert_equal(x.abs().realize().numpy(), np.abs(data))
    finite = data[:4]
    np.testing.assert_equal(Tensor(finite, device="ROCKCHIP").maximum(0).realize().numpy(), np.maximum(finite, np.float16(0)))
    np.testing.assert_equal(Tensor(finite, device="ROCKCHIP").sign().realize().numpy(), np.sign(finite))

  def test_stable_hardsigmoid_saturation(self):
    data = np.concatenate((np.linspace(-400,-300,1001), np.linspace(300,400,1001))).astype(np.float16)
    expected = np.concatenate((np.zeros(1001,np.float16), np.ones(1001,np.float16)))
    np.testing.assert_equal(Tensor(data, device="ROCKCHIP").hardsigmoid().realize().numpy(), expected)

  def test_generated_exp2_lut(self):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    data = encodings.view(np.float16)
    data = data[np.isfinite(data) & (data >= -2) & (data <= 2)]
    actual = Tensor(data, device="ROCKCHIP").realize().exp2().realize().numpy()
    reference = np.exp2(data.astype(np.float32))
    absolute = np.abs(actual.astype(np.float32)-reference)
    relative = absolute/reference
    ulp = np.abs(actual.view(np.uint16).astype(np.int32)-reference.astype(np.float16).view(np.uint16).astype(np.int32))
    order = np.argsort(data.astype(np.float32), kind="stable")
    self.assertEqual(data.size, 32770)
    self.assertLessEqual(float(absolute.max()), 0.0011)
    self.assertLessEqual(float(relative.max()), 0.0009)
    self.assertLessEqual(int(ulp.max()), 1)
    self.assertTrue(np.all(np.diff(actual[order].astype(np.float32)) >= 0))
    special = np.array([np.inf, -np.inf, np.nan], dtype=np.float16)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").exp2().realize().numpy(), np.array([np.inf, 0, np.nan], dtype=np.float16))

  def test_generated_two_level_exp_lut(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    actual = Tensor(data, device="ROCKCHIP").exp().realize().numpy()
    reference = np.exp(data.astype(np.float32))
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-6)
    special = np.array([np.inf, -np.inf, np.nan], dtype=np.float16)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").exp().realize().numpy(), np.array([np.inf, 0, np.nan], dtype=np.float16))

  def test_generated_two_level_expm1_lut(self):
    data = np.linspace(-2, 0, 4097, dtype=np.float16)
    actual = (Tensor(data, device="ROCKCHIP").exp()-1).realize().numpy()
    np.testing.assert_allclose(actual, np.expm1(data.astype(np.float32)).astype(np.float16), rtol=1.2e-3, atol=1e-6)

  def test_generated_tanh_luts_and_local_polynomial(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").tanh().realize().numpy(),
                               np.tanh(data.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    extreme = np.array([-400, -300, 300, 400], dtype=np.float16)
    np.testing.assert_equal(Tensor(extreme, device="ROCKCHIP").tanh().realize().numpy(), np.tanh(extreme))

  def test_generated_sin_luts_and_split_range_reduction(self):
    data = np.array([-10000, -1000, -100, -10, -2, -.125, -.04, -0., 0., .04, .125, 2, 10, 100, 1000, 10000], dtype=np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").sin().realize().numpy(),
                               np.sin(data.astype(np.float32)).astype(np.float16), rtol=3e-3, atol=3e-3)
    special = np.array([np.nan, np.inf, -np.inf], dtype=np.float16)
    self.assertTrue(np.isnan(Tensor(special, device="ROCKCHIP").sin().realize().numpy()).all())

  def test_generated_inverse_trig_assets(self):
    unit = np.linspace(-1, 1, 4097, dtype=np.float16)
    np.testing.assert_allclose(Tensor(unit, device="ROCKCHIP").asin().realize().numpy(),
                               np.arcsin(unit.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(Tensor(unit, device="ROCKCHIP").acos().realize().numpy(),
                               np.arccos(unit.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    broad = np.linspace(-8, 8, 4097, dtype=np.float16)
    np.testing.assert_allclose(Tensor(broad, device="ROCKCHIP").atan().realize().numpy(),
                               np.arctan(broad.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    invalid = np.array([-300, 300], dtype=np.float16)
    self.assertTrue(np.isnan(Tensor(invalid, device="ROCKCHIP").asin().realize().numpy()).all())

  def test_generated_atanh_assets(self):
    data = np.linspace(-.9995, .9995, 4097, dtype=np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").atanh().realize().numpy(),
                               np.arctanh(data.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    special = np.array([-2, -1, 1, 2, np.nan], dtype=np.float16)
    with np.errstate(divide="ignore", invalid="ignore"): expected = np.arctanh(special)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").atanh().realize().numpy(), expected)

  def test_generated_inverse_hyperbolic_assets(self):
    asinh_data = np.concatenate((np.linspace(-8, 8, 4097), [-300, 300])).astype(np.float16)
    np.testing.assert_allclose(Tensor(asinh_data, device="ROCKCHIP").asinh().realize().numpy(),
                               np.arcsinh(asinh_data.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    acosh_data = np.concatenate((np.linspace(1, 9, 4097), [300])).astype(np.float16)
    np.testing.assert_allclose(Tensor(acosh_data, device="ROCKCHIP").acosh().realize().numpy(),
                               np.arccosh(acosh_data.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    invalid = np.array([-300, -1, 0, .5], dtype=np.float16)
    self.assertTrue(np.isnan(Tensor(invalid, device="ROCKCHIP").acosh().realize().numpy()).all())

  def test_generated_hyperbolic_assets(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    for function, reference in ((lambda x:x.sinh(), np.sinh), (lambda x:x.cosh(), np.cosh)):
      np.testing.assert_allclose(function(Tensor(data, device="ROCKCHIP")).realize().numpy(),
                                 reference(data.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    extreme = np.array([-300, 300], dtype=np.float16)
    with np.errstate(over="ignore"): sinh_expected, cosh_expected = np.sinh(extreme), np.cosh(extreme)
    np.testing.assert_equal(Tensor(extreme, device="ROCKCHIP").sinh().realize().numpy(), sinh_expected)
    np.testing.assert_equal(Tensor(extreme, device="ROCKCHIP").cosh().realize().numpy(), cosh_expected)

  def test_generated_erf_asset(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").erf().realize().numpy(),
                               np.vectorize(math.erf)(data.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    extreme = np.array([-300, 300], dtype=np.float16)
    np.testing.assert_equal(Tensor(extreme, device="ROCKCHIP").erf().realize().numpy(), np.array([-1, 1], dtype=np.float16))

  def test_generated_softplus_assets_and_logsigmoid(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").softplus().realize().numpy(),
                               np.logaddexp(data.astype(np.float32), 0).astype(np.float16), rtol=1e-3, atol=1e-6)
    local = np.linspace(-.5, .5, 2049, dtype=np.float16)
    for beta in (3, 1/3):
      expected = (np.logaddexp(beta*local.astype(np.float32), 0)/beta).astype(np.float16)
      np.testing.assert_allclose(Tensor(local, device="ROCKCHIP").softplus(beta=beta).realize().numpy(), expected, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").logsigmoid().realize().numpy(),
                               -np.logaddexp(-data.astype(np.float32), 0).astype(np.float16), rtol=1e-3, atol=1e-6)
    extreme = np.array([-300, 300], dtype=np.float16)
    np.testing.assert_equal(Tensor(extreme, device="ROCKCHIP").softplus().realize().numpy(), np.array([0, 300], dtype=np.float16))

  def test_generated_mish_assets(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    expected = (data.astype(np.float32)*np.tanh(np.logaddexp(data.astype(np.float32), 0))).astype(np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").mish().realize().numpy(), expected, rtol=1e-3, atol=1e-6)

  def test_generated_hardswish_assets(self):
    data = np.linspace(-4, 4, 8193, dtype=np.float16)
    expected = (data.astype(np.float32)*np.minimum(6, np.maximum(0, data.astype(np.float32)+3))/6).astype(np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").hardswish().realize().numpy(), expected, rtol=1e-3, atol=1e-6)

  def test_generated_quick_gelu_assets(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    expected = (data.astype(np.float32)/(1+np.exp(-1.702*data.astype(np.float32)))).astype(np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").quick_gelu().realize().numpy(), expected, rtol=1e-3, atol=1e-6)

  def test_generated_gelu_assets(self):
    data = np.linspace(-4, 4, 8193, dtype=np.float16)
    x = data.astype(np.float32)
    for approximate in ("tanh", "none"):
      expected = (.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3))) if approximate == "tanh" else
                  .5*x*(1+np.vectorize(math.erf)(x/np.sqrt(2)))).astype(np.float16)
      actual = Tensor(data, device="ROCKCHIP").gelu(approximate=approximate).realize().numpy()
      np.testing.assert_allclose(actual, expected, rtol=1.4e-3, atol=1.3e-4)

  def test_generated_elu_family_assets(self):
    data = np.linspace(-10, 10, 8193, dtype=np.float16)
    x = data.astype(np.float32)
    variants = ((lambda value:value.elu(), np.where(x > 0, x, np.expm1(x))),
                (lambda value:value.elu(.1), np.where(x > 0, x, .1*np.expm1(x))),
                (lambda value:value.selu(), 1.0507*np.where(x > 0, x, 1.67326*np.expm1(x))))
    for function, expected in variants:
      np.testing.assert_allclose(function(Tensor(data, device="ROCKCHIP")).realize().numpy(), expected.astype(np.float16),
                                 rtol=1e-3, atol=1e-6)

  def test_generated_celu_assets(self):
    data = np.linspace(-4, 4, 8193, dtype=np.float16)
    x = data.astype(np.float32)
    for alpha in range(1,5):
      expected = np.where(x > 0, x, alpha*np.expm1(x/alpha)).astype(np.float16)
      np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").celu(alpha).realize().numpy(), expected, rtol=1e-3, atol=1e-6)

  def test_generated_two_level_sigmoid_lut(self):
    data = np.linspace(-2, 2, 4097, dtype=np.float16)
    expected = (1/(1+np.exp(-data.astype(np.float32)))).astype(np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").sigmoid().realize().numpy(), expected, rtol=1e-3, atol=1e-6)
    special = np.array([np.inf, -np.inf, np.nan], dtype=np.float16)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").sigmoid().realize().numpy(), np.array([1, 0, np.nan], dtype=np.float16))

  def test_scaled_sigmoid_composition(self):
    data = np.concatenate((np.linspace(-2,2,4097), np.linspace(-400,-300,1001), np.linspace(300,400,1001))).astype(np.float16)
    with np.errstate(over="ignore"): expected = (data.astype(np.float32)/(1+np.exp(-1.702*data.astype(np.float32)))).astype(np.float16)
    actual = Tensor(data, device="ROCKCHIP").quick_gelu().realize().numpy()
    error = np.abs(actual.astype(np.float32)-expected.astype(np.float32))
    self.assertLessEqual(float(error.max()), 1e-3)
    self.assertLessEqual(float((error[np.abs(expected)>.05]/np.abs(expected[np.abs(expected)>.05])).max()), 3.1e-3)

  def test_generated_refined_sqrt_lut(self):
    data = np.linspace(0, 16, 2049, dtype=np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").sqrt().realize().numpy(),
                               np.sqrt(data.astype(np.float32)).astype(np.float16), rtol=1e-3, atol=1e-6)
    special = np.array([-1, -0., 0., np.inf, np.nan], dtype=np.float16)
    with np.errstate(invalid="ignore"): expected = np.sqrt(special)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").sqrt().realize().numpy(), expected)

  def test_generated_refined_rsqrt_lut(self):
    data = np.geomspace(2**-8, 4, 2049).astype(np.float16)
    expected = (1/np.sqrt(data.astype(np.float32))).astype(np.float16)
    np.testing.assert_allclose(Tensor(data, device="ROCKCHIP").rsqrt().realize().numpy(), expected, rtol=1e-3, atol=1e-6)
    special = np.array([-1, 0., np.inf, np.nan], dtype=np.float16)
    with np.errstate(divide="ignore", invalid="ignore"): expected_special = 1/np.sqrt(special)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").rsqrt().realize().numpy(), expected_special)

  def test_generated_logarithm_luts(self):
    bits = np.arange(1 << 16, dtype=np.uint16)
    data = bits.view(np.float16)
    data = data[np.isfinite(data) & (data >= 2**-8) & (data <= 4)]
    for function, reference in ((lambda x:x.log2(), np.log2), (lambda x:x.log10(), np.log10)):
      actual = function(Tensor(data, device="ROCKCHIP")).realize().numpy()
      expected = reference(data.astype(np.float32)).astype(np.float16)
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)
    special = np.array([-1., -0., 0., 1., np.inf, np.nan], dtype=np.float16)
    with np.errstate(divide="ignore", invalid="ignore"): expected_special = np.log2(special)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").log2().realize().numpy(), expected_special)
    zero, negative = Tensor([0.], device="ROCKCHIP"), Tensor([-.7], device="ROCKCHIP")
    np.testing.assert_equal((zero.log2()*negative).exp2().realize().numpy(), np.array([np.inf], dtype=np.float16))

  def test_generated_roundoff_lut(self):
    data = np.linspace(-16, 16, 4097, dtype=np.float16)
    np.testing.assert_equal(Tensor(data, device="ROCKCHIP").round().realize().numpy(), np.round(data))
    special = np.array([-np.inf,-2.5,-1.5,-.5,-0.,0.,.5,1.5,2.5,np.inf,np.nan], dtype=np.float16)
    np.testing.assert_equal(Tensor(special, device="ROCKCHIP").round().realize().numpy(), np.round(special))
    for function, reference in ((lambda x:x.trunc(), np.trunc), (lambda x:x.floor(), np.floor), (lambda x:x.ceil(), np.ceil)):
      np.testing.assert_equal(function(Tensor(data, device="ROCKCHIP")).realize().numpy(), reference(data))

  def test_generated_pow8_two_level_lut(self):
    data = np.concatenate((np.linspace(-4.1, 4.1, 513, dtype=np.float32).astype(np.float16),
                           np.array([np.inf, -np.inf, np.nan], dtype=np.float16)))
    actual = (Tensor(data, device="ROCKCHIP").realize()**8).realize().numpy()
    with np.errstate(all="ignore"): expected = np.power(data.astype(np.float32), 8).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_generated_positive_pow55_multirange_lut(self):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    all_half = encodings.view(np.float16)
    data = all_half[np.isfinite(all_half) & (all_half >= 0) & (all_half <= 4)]
    special = np.array([-2, -1, np.inf, np.nan, -0.0], dtype=np.float16)
    for values in (data, special):
      actual = (Tensor(values, device="ROCKCHIP").realize()**5.5).realize().numpy()
      with np.errstate(all="ignore"): expected = np.power(values.astype(np.float32), np.float32(5.5)).astype(np.float16)
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_generated_negative_pow55_shifted_multirange_luts(self):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    all_half = encodings.view(np.float16)
    data = all_half[np.isfinite(all_half) & (all_half >= 0) & (all_half <= 8)]
    special = np.array([-2, -1, -np.inf, np.inf, np.nan, -0.0, .133056640625, .1331787109375], dtype=np.float16)
    for values in (data, special):
      actual = (Tensor(values, device="ROCKCHIP").realize()**-5.5).realize().numpy()
      with np.errstate(all="ignore"): expected = np.power(values.astype(np.float32), np.float32(-5.5)).astype(np.float16)
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_generated_constant_base_pow55_split_luts(self):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    all_half = encodings.view(np.float16)
    data = all_half[np.isfinite(all_half) & (all_half >= -2) & (all_half <= 2)]
    special = np.array([-np.inf, np.inf, np.nan], dtype=np.float16)
    for values in (data, special):
      actual = (5.5**Tensor(values, device="ROCKCHIP").realize()).realize().numpy()
      with np.errstate(all="ignore"): expected = np.power(np.float32(5.5), values.astype(np.float32)).astype(np.float16)
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_negative_base_pow55_native_parity(self):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    all_half = encodings.view(np.float16)
    data = all_half[np.isfinite(all_half) & (all_half >= -2) & (all_half <= 2)]
    actual = ((-5.5)**Tensor(data, device="ROCKCHIP").realize()).realize().numpy()
    with np.errstate(all="ignore"): expected = np.power(np.float32(-5.5), data.astype(np.float32)).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_constant_base_pow8_four_generated_bands(self):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    all_half = encodings.view(np.float16)
    data = all_half[np.isfinite(all_half) & (all_half >= -2) & (all_half <= 2)]
    special = np.array([-np.inf, np.inf, np.nan], dtype=np.float16)
    for values in (data, special):
      actual = (8.0**Tensor(values, device="ROCKCHIP").realize()).realize().numpy()
      with np.errstate(all="ignore"): expected = np.power(np.float32(8.0), values.astype(np.float32)).astype(np.float16)
      np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

  def test_linear_sigmoid_workload(self):
    rng = np.random.default_rng(2)
    a_np = rng.uniform(-0.25, 0.25, (1,32)).astype(np.float16)
    w_np = rng.uniform(-0.25, 0.25, (8,32)).astype(np.float16)
    a, w = Tensor(a_np, device="ROCKCHIP").realize(), Tensor(w_np, device="ROCKCHIP").realize()
    actual = (a@w.T).realize().sigmoid().realize().numpy()
    logits = a_np@w_np.T
    np.testing.assert_allclose(actual, 1/(1+np.exp(-logits)), rtol=6e-3, atol=6e-3)

if __name__ == "__main__": unittest.main()
