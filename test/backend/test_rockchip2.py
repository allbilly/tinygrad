"""Rockchip-only hardware regressions beyond the unchanged upstream test_ops census.

Run: FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP python -m pytest test/backend/test_rockchip2.py -q -n0
"""
from __future__ import annotations
import math, unittest
import numpy as np
import torch
from tinygrad import Tensor, Device, dtypes
from test.backend.test_ops import slow_test
from test.backend import test_ops as _test_ops
from test.backend import test_rockchip as _base
from test.backend.test_rockchip import (_FP16, _TEST_OPS_HELPER, _ew_submits, _fp16_fp32_golden_test_op, _fp16_test_op)

def _only_local_tests(cls):
  """Keep inherited fixtures/helpers without recollecting the upstream tests from the base class."""
  for name in dir(cls.__bases__[0]):
    if name.startswith("test_") and name not in cls.__dict__: setattr(cls, name, None)
  return cls

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIndexedLossCandidates(_test_ops.TestOps):
  """Upstream indexed-loss candidates; passing methods move unchanged to test_rockchip.py."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_cross_entropy_reductions = _test_ops.TestOps.test_cross_entropy_reductions
  test_cross_entropy_smoothing = _test_ops.TestOps.test_cross_entropy_smoothing
  test_nll_loss_weight = _test_ops.TestOps.test_nll_loss_weight
  test_nll_loss_3d_weight = _test_ops.TestOps.test_nll_loss_3d_weight

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipTranscendentalCandidates(_test_ops.TestOps):
  """Remaining no-LUT transcendental candidates; passing methods move unchanged to test_rockchip.py."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_acosh = _test_ops.TestOps.test_acosh
  test_asinh = _test_ops.TestOps.test_asinh
  test_atan = _test_ops.TestOps.test_atan
  test_cos = _test_ops.TestOps.test_cos
  test_sigmoid_alt_extreme = _test_ops.TestOps.test_sigmoid_alt_extreme
  test_tan = _test_ops.TestOps.test_tan

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCompositeCandidates(_test_ops.TestOps):
  """Remaining cumulative, normalization, and attention candidates."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_logcumsumexp_numerical = _test_ops.TestOps.test_logcumsumexp_numerical
  test_scaled_dot_product_attention = _test_ops.TestOps.test_scaled_dot_product_attention
  test_scaled_dot_product_attention_causal = _test_ops.TestOps.test_scaled_dot_product_attention_causal
  test_scaled_dot_product_attention_gqa = _test_ops.TestOps.test_scaled_dot_product_attention_gqa
  test_scaled_dot_product_attention_mismatch_ls = _test_ops.TestOps.test_scaled_dot_product_attention_mismatch_ls
  test_softmax_argmax = _test_ops.TestOps.test_softmax_argmax

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipEdgeCandidates(_test_ops.TestOps):
  """Remaining forward power, masked-selection, and scatter edge candidates."""

  @classmethod
  def setUpClass(cls): _test_ops.helper_test_op = _fp16_test_op

  @classmethod
  def tearDownClass(cls): _test_ops.helper_test_op = _TEST_OPS_HELPER

  test_cast_relu = _test_ops.TestOpsUint8.test_cast_relu
  test_masked_select = _test_ops.TestOps.test_masked_select
  test_pow_const_direct = _test_ops.TestOps.test_pow_const_direct
  test_pow_int = _test_ops.TestOps.test_pow_int
  test_scatter_reduce_prod_zeros = _test_ops.TestOps.test_scatter_reduce_prod_zeros

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchip(_base.TestRockchip):
  def test_add_scalar_constfold(self):
    # Tensor(1)+0.5 folds on device=None — no NPU submit
    self._check(0, Tensor(1) + 0.5, np.array(1.5, dtype=np.float16))
  def test_add_empty(self):
    # rank-0 buffers materialize without NPU submit
    a, b = self._half((), 5), self._half((), 6)
    self._check(0, a + b, (a.numpy().astype(np.float32) + b.numpy()).astype(np.float16))
  def test_scalar_mul_empty(self):
    # rank-0 scalar mul — no NPU submit
    a = self._half((), 15)
    self._check(0, a * 2, (a.numpy().astype(np.float32) * 2).astype(np.float16))
    self._check(0, 2 * a, (a.numpy().astype(np.float32) * 2).astype(np.float16))
  def test_tiny_div(self):
    lhs = np.array([-2.0, 3.0, 4.0], dtype=np.float16)
    rhs = np.array([0.5, -2.0, 8.0], dtype=np.float16)
    self._check(1, Tensor(lhs) / Tensor(rhs), (lhs.astype(np.float32) / rhs).astype(np.float16))
  def test_infinite_division_sign(self):
    values = np.array([-3.0, -0.5, 0.5, 2.0], dtype=np.float16)
    self._check(1, math.inf / Tensor(values), (np.float16(np.inf) / values).astype(np.float16))
    self._check(1, -math.inf / Tensor(values), (np.float16(-np.inf) / values).astype(np.float16))
  def test_maximum_fp16(self):
    a, b = self._half((45, 65), 17), self._half((45, 65), 18)
    self._check(_ew_submits(45*65), a.maximum(b), np.maximum(a.numpy(), b.numpy()))
  def test_max_pool2d_simple_submit(self):
    xn = np.array([[[[-1, 2, 0], [3, -4, 1]]]], dtype=np.float16)
    ref = torch.nn.functional.max_pool2d(torch.from_numpy(xn), kernel_size=(2, 2)).numpy()
    self._check(1, Tensor(xn).max_pool2d(kernel_size=(2, 2)), ref)
  def test_avg_pool2d_valid_count_submit(self):
    xn = np.arange(16, dtype=np.float16).reshape(1, 1, 4, 4)
    args = dict(kernel_size=(3, 3), padding=1, count_include_pad=False)
    ref = torch.nn.functional.avg_pool2d(torch.from_numpy(xn), **args).numpy()
    self._check(1, Tensor(xn).avg_pool2d(**args), ref)
  def test_interpolate_nearest_submit(self):
    x = self._half((2, 3, 13), 502)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(9,), mode="nearest").numpy()
    self._check(1, x.interpolate((9,), mode="nearest"), ref, **_FP16)
  def test_interpolate_linear_submit(self):
    x = self._half((2, 3, 52), 503)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(29,), mode="linear").numpy()
    self._check(1, x.interpolate((29,), mode="linear"), ref, **_FP16)
  def test_interpolate_bilinear_submit(self):
    x = self._half((2, 3, 12, 20), 504)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(9, 31), mode="bilinear").numpy()
    self._check(2, x.interpolate((9, 31), mode="linear"), ref, **_FP16)
  def test_interpolate_trilinear_submit(self):
    x = self._half((1, 1, 3, 2, 4), 505)
    ref = torch.nn.functional.interpolate(torch.from_numpy(x.numpy()), size=(2, 4, 3), mode="trilinear").numpy()
    self._check(3, x.interpolate((2, 4, 3), mode="linear"), ref, **_FP16)
  def test_padded_conv2d(self):
    self._check_conv2d((1,4,9,9), (4,4,3,3), 103, padding=1)
  def test_simple_conv2d_reduce63(self):
    self._check_conv2d((1,7,9,9), (4,7,3,3), 207)
  def test_simple_conv2d_reduce72(self):
    self._check_conv2d((1,8,9,9), (4,8,3,3), 208)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaxPoolOps(_base.TestRockchipMaxPoolOps):
  def test_max_pool2d_return_indices_int16(self):
    data = np.array([[[[-32768, 7, 7, -9], [3, 7, -4, -4], [5, 5, 2, 1], [5, -8, 2, 2]]]], dtype=np.int16)
    self._check_int16(data, kernel_size=(3,3), padding=1, return_indices=True, expected_submits=2)
    global_data = np.random.default_rng(2609).integers(-32768, 32768, (1,1,12,13), dtype=np.int16)
    self._check_int16(global_data, kernel_size=(12,13), return_indices=True, expected_submits=2)
  def test_max_pool2d_return_indices_wide(self):
    data = np.zeros((1,1,50,50), dtype=np.float16)
    data[0,0,46,49] = 1
    expected = torch.nn.functional.max_pool2d(torch.from_numpy(data), kernel_size=(5,5), stride=(6,5), return_indices=True)[1].int().numpy()
    got = Tensor(data, device="ROCKCHIP").max_pool2d((5,5), stride=(6,5), return_indices=True)[1].realize().numpy()
    self.assertGreater(int(expected.max()), 2048)
    np.testing.assert_array_equal(got, expected)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaxUnpoolOps(_base.TestRockchipMaxUnpoolOps):
  def test_max_unpool2d_int16(self):
    values = np.array([[[[-32768, 32767], [7, -9]]]], dtype=np.int16)
    indices = np.array([[[[0, 3], [12, 15]]]], dtype=np.int32)
    expected = torch.nn.functional.max_unpool2d(torch.from_numpy(values).float(), torch.from_numpy(indices).long(), 2).int().numpy()
    before = Device["ROCKCHIP"].submit_count, Device["ROCKCHIP"].task_count
    got = Tensor(values, device="ROCKCHIP").max_unpool2d(Tensor(indices, device="ROCKCHIP"), 2).realize().numpy()
    np.testing.assert_array_equal(got, expected)
    self.assertEqual((Device["ROCKCHIP"].submit_count-before[0], Device["ROCKCHIP"].task_count-before[1]), (2,40))

    duplicate_values = np.array([[[[30000, 30000]]]], dtype=np.int16)
    duplicate_indices = np.zeros((1,1,1,2), dtype=np.int32)
    expected = np.zeros((1,1,1,4), dtype=np.int32)
    expected[0,0,0,0] = 60000
    got = Tensor(duplicate_values, device="ROCKCHIP").max_unpool2d(
      Tensor(duplicate_indices, device="ROCKCHIP"), (1,2)).realize().numpy()
    np.testing.assert_array_equal(got, expected)

    data = np.array([[[[-32768, 7, 7, -9], [3, 7, -4, -4], [5, 5, 2, 1], [5, -8, 2, 2]]]], dtype=np.int16)
    pooled, pooled_indices = torch.nn.functional.max_pool2d(torch.from_numpy(data), (2,2), return_indices=True)
    expected = torch.nn.functional.max_unpool2d(pooled.float(), pooled_indices, (2,2)).int().numpy()
    got = Tensor.max_unpool2d(
      *Tensor.max_pool2d(Tensor(data, device="ROCKCHIP"), (2,2), return_indices=True), (2,2)).realize().numpy()
    np.testing.assert_array_equal(got, expected)
  def test_max_unpool2d_nonfinite_bits(self):
    values = np.array([math.inf, -math.inf, math.nan, 3.5], dtype=np.float16).reshape(4,1,1,1)
    indices = np.arange(4, dtype=np.int32).reshape(4,1,1,1)
    expected = np.zeros((4,1,2,2), dtype=np.float16)
    expected.reshape(4,4)[np.arange(4),np.arange(4)] = values.reshape(4)
    got = Tensor(values, device="ROCKCHIP").max_unpool2d(Tensor(indices, device="ROCKCHIP"), 2).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
  def test_max_unpool2d_wide_indices(self):
    values = np.arange(1, 15, dtype=np.float16).reshape(7,1,1,2)
    indices = np.array([[2049,2499]]*7, dtype=np.int32).reshape(7,1,1,2)
    expected = np.zeros((7,1,50,50), dtype=np.float16)
    expected.reshape(7,2500)[:,2049] = values.reshape(7,2)[:,0]
    expected.reshape(7,2500)[:,2499] = values.reshape(7,2)[:,1]
    got = Tensor(values, device="ROCKCHIP").max_unpool2d(Tensor(indices, device="ROCKCHIP"), (1,2), output_size=(50,50)).realize().numpy()
    np.testing.assert_array_equal(got, expected)
  @slow_test
  def test_max_unpool2d_wide(self):
    args = {"kernel_size":(5,5), "stride":(6,5)}
    _fp16_test_op([(8,3,50,50)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args), **args),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args), **args), forward_only=True)
  def test_max_unpool2d_bounded(self):
    _fp16_test_op([(1,3,7,6)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                                 kernel_size=(2,2), output_size=(99,99,7,6)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                    kernel_size=(2,2), output_size=(99,99,7,6)), forward_only=True)
  def test_max_unpool2d_padded(self):
    args = {"kernel_size":(3,3), "stride":(6,7), "padding":1}
    _fp16_test_op([(8,3,30,30)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args),
                                                 **args, output_size=(30,30)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args),
                                    **args, output_size=(30,30)), forward_only=True)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipAdaptivePoolOps(_base.TestRockchipAdaptivePoolOps):
  def test_adaptive_avg_pool2d_equivalent(self):
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (2,2)),
                  lambda x: x.avg_pool2d(kernel_size=2, stride=2))
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (1,1)),
                  lambda x: x.avg_pool2d(kernel_size=4, stride=4))
  def test_adaptive_max_pool2d_equivalent(self):
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_max_pool2d(x, (2,2)),
                  lambda x: x.max_pool2d(kernel_size=2, stride=2))
    _fp16_test_op([(1,3,4,4)], lambda x: torch.nn.functional.adaptive_max_pool2d(x, (1,1)),
                  lambda x: x.max_pool2d(kernel_size=4, stride=4))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipPolynomialLossOps(_base.TestRockchipPolynomialLossOps):
  def test_square_and_cubic(self):
    _fp16_test_op([(4,5)], lambda x: x.square(), lambda x: x.square())
    _fp16_test_op([(4,5)], lambda x: x*x*x, lambda x: x*x*x)
  def test_horner_polynomial(self):
    _fp16_test_op([(4,5)], lambda x: (x*0.5-1.25)*x+0.75, lambda x: (x*0.5-1.25)*x+0.75)
  def test_mse_loss(self):
    _fp16_fp32_golden_test_op([(3,4), (3,4)], lambda x,y: (x-y).square().mean(), lambda x,y: (x-y).square().mean())
  def test_l1_loss(self):
    _fp16_fp32_golden_test_op([(3,4), (3,4)], lambda x,y: (x-y).abs().mean(), lambda x,y: (x-y).abs().mean())

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipOneHotOps(_base.TestRockchipOneHotOps):
  def test_one_hot_full_int32_bytes(self):
    values = np.array([0, 5, 256, 65536, 1 << 24, -1], dtype=np.int32)
    expected = np.zeros((len(values), 6), dtype=np.int32)
    expected[0,0] = expected[1,5] = 1
    got = Tensor(values, device="ROCKCHIP").one_hot(6).realize().numpy()
    np.testing.assert_array_equal(got, expected)
  def test_one_hot_beyond_fp16_integer_range(self):
    expected = np.zeros((1, 2050), dtype=np.int32)
    expected[0,2049] = 1
    got = Tensor(np.array([2049], dtype=np.int32), device="ROCKCHIP").one_hot(2050).realize().numpy()
    np.testing.assert_array_equal(got, expected)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipGatherOps(_base.TestRockchipGatherOps):
  def test_gather_int16_full_index_bytes(self):
    source = np.array([-32768, -1, 0, 1, 7, 8, 9, 32767], dtype=np.int16)
    indices = np.array([7, 0, 3, 99, -1], dtype=np.int32)
    expected = np.array([32767, -32768, 1, 0, 0], dtype=np.int16)
    before = Device["ROCKCHIP"].submit_count
    got = Tensor(source, device="ROCKCHIP").gather(0, Tensor(indices, device="ROCKCHIP")).realize().numpy()
    np.testing.assert_array_equal(got, expected)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 1)
  def test_gather_nonfinite_full_index_bytes(self):
    source = np.array([math.inf, -math.inf, math.nan], dtype=np.float16)
    indices = np.array([0, 1, 2, 256, 65536, 1 << 24, -1], dtype=np.int32)
    expected = np.zeros(len(indices), dtype=np.float16)
    expected[:3] = source
    got = Tensor(source, device="ROCKCHIP").gather(0, Tensor(indices, device="ROCKCHIP")).realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMaskedSelectOps(_base.TestRockchipMaskedSelectOps):
  def test_dynamic_threshold(self):
    _fp16_test_op([(32,10)], lambda x:x.masked_select(x>0.5), lambda x:x.masked_select(x>0.5), forward_only=True)
  def test_dynamic_scalar_true_small(self):
    _fp16_test_op([(32,)], lambda x:x.masked_select(torch.tensor(True)),
                  lambda x:x.masked_select(Tensor(True)), forward_only=True)
  def test_fixed_masked_select_pad_and_truncate(self):
    values = np.array([-3., 2., -1., 4., 5., -2.], dtype=np.float16)
    source = Tensor(values, device="ROCKCHIP")
    padded = source.masked_select(source > 0, size=8, fill_value=-7).numpy()
    np.testing.assert_array_equal(padded, np.array([2., 4., 5., -7., -7., -7., -7., -7.], dtype=np.float16))
    source = Tensor(values, device="ROCKCHIP")
    np.testing.assert_array_equal(source.masked_select(source > 0, size=2).numpy(), np.array([2., 4.], dtype=np.float16))
  def test_fixed_masked_select_exact_bits(self):
    values = np.array([0x7e01, 0xfc00, 0x8000, 0x0000, 0x0001, 0x3c00, 0x7c00], dtype=np.uint16).view(np.float16)
    expected = np.array([0x0001, 0x3c00, 0x7c00, 0x8000, 0x8000], dtype=np.uint16)
    source = Tensor(values, device="ROCKCHIP")
    got = source.masked_select(source > 0, size=5, fill_value=-0.0).numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected)
  def test_fixed_masked_select_int16(self):
    values = np.array([-32768, -30000, -1, 0, 1, 30000, 32767], dtype=np.int16)
    mask = np.array([True, False, True, False, False, True, True])
    source, predicate = Tensor(values, device="ROCKCHIP"), Tensor(mask, device="ROCKCHIP")
    got = source.masked_select(predicate, size=6, fill_value=-12345).numpy()
    np.testing.assert_array_equal(got, np.array([-32768, -1, 30000, 32767, -12345, -12345], dtype=np.int16))
    source, predicate = Tensor(values, device="ROCKCHIP"), Tensor(mask, device="ROCKCHIP")
    np.testing.assert_array_equal(source.masked_select(predicate, size=2).numpy(), np.array([-32768, -1], dtype=np.int16))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipNonzeroOps(_base.TestRockchipNonzeroOps):
  def test_dynamic_nonzero_rank_two(self):
    _fp16_test_op([(32,10)], lambda x:(x>0.5).nonzero().int(), lambda x:(x>0.5).nonzero(), forward_only=True)
  def test_dynamic_nonzero_rank_one(self):
    _fp16_test_op([(20,)], lambda x:(x>0.5).nonzero().int(), lambda x:(x>0.5).nonzero(), forward_only=True)
  def test_dynamic_nonzero_rank_three(self):
    _fp16_test_op([(10,5,3)], lambda x:(x>0.5).nonzero().int(), lambda x:(x>0.5).nonzero(), forward_only=True)
  def test_dynamic_nonzero_scalars(self):
    for value in (0, 1, 0.0, 2.5, True, False):
      _fp16_test_op(None, lambda x:x.nonzero().int(), lambda x:x.nonzero(), vals=[value], forward_only=True)
  def test_fixed_nonzero_pad_and_truncate(self):
    values = np.array([1., 0., 2., 0., 3.], dtype=np.float16)
    source = Tensor(values, device="ROCKCHIP")
    np.testing.assert_array_equal(source.nonzero(size=3).numpy(), np.array([[0], [2], [4]], dtype=np.int32))
    source = Tensor(values, device="ROCKCHIP")
    np.testing.assert_array_equal(source.nonzero(size=5, fill_value=-1).numpy(),
                                  np.array([[0], [2], [4], [-1], [-1]], dtype=np.int32))
  def test_fixed_nonzero_rank_two(self):
    source = Tensor(np.array([[1., 0.], [0., 2.]], dtype=np.float16), device="ROCKCHIP")
    np.testing.assert_array_equal(source.nonzero(size=2).numpy(), np.array([[0, 0], [1, 1]], dtype=np.int32))
  def test_fixed_nonzero_fp16_specials(self):
    values = np.array([0x0000, 0x8000, 0xbc00, 0x7e01, 0x7c00], dtype=np.uint16).view(np.float16)
    source = Tensor(values, device="ROCKCHIP")
    np.testing.assert_array_equal(source.nonzero(size=5, fill_value=-1).numpy(),
                                  np.array([[2], [3], [4], [-1], [-1]], dtype=np.int32))
  def test_fixed_nonzero_empty_scalar_and_dtype(self):
    empty = Tensor(np.empty((0,), dtype=np.float16), device="ROCKCHIP")
    np.testing.assert_array_equal(empty.nonzero(size=3, fill_value=-1).numpy(), np.full((3, 1), -1, dtype=np.int32))
    self.assertEqual(Tensor(5, dtype=dtypes.half, device="ROCKCHIP").nonzero(size=4).shape, (4, 0))
    self.assertEqual(Tensor(np.array([1., 0.], dtype=np.float16), device="ROCKCHIP").nonzero(size=3, fill_value=-1.5).dtype,
                     dtypes.int)
  def test_int32_nonzero_exact_bytes(self):
    values = np.array([0, -1, 256, -65536, 0, 2147483647, -2147483648], dtype=np.int32)
    got = Tensor(values, device="ROCKCHIP").nonzero(size=6, fill_value=-1).numpy()
    np.testing.assert_array_equal(got, np.array([[1], [2], [3], [5], [6], [-1]], dtype=np.int32))
  def test_int16_nonzero_exact_bytes(self):
    values = np.array([0, -32768, 256, -256, 0, 32767, -1], dtype=np.int16)
    got = Tensor(values, device="ROCKCHIP").nonzero(size=6, fill_value=-1).numpy()
    np.testing.assert_array_equal(got, np.array([[1], [2], [3], [5], [6], [-1]], dtype=np.int32))
  def test_int16_nonzero_rank_two(self):
    values = np.array([[0, -32768], [256, 0]], dtype=np.int16)
    got = Tensor(values, device="ROCKCHIP").nonzero(size=3, fill_value=-1).numpy()
    np.testing.assert_array_equal(got, np.array([[0, 1], [1, 0], [-1, -1]], dtype=np.int32))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipFancyIndexOps(_base.TestRockchipFancyIndexOps):
  def test_fancy_indexing_int16(self):
    source = np.array([[-32768,-1,0,1], [7,8,9,32767]], dtype=np.int16)
    rows = np.array([1,0,-1], dtype=np.int32)
    got = Tensor(source, device="ROCKCHIP")[Tensor(rows, device="ROCKCHIP")].realize().numpy()
    np.testing.assert_array_equal(got, source[rows])

    cols = np.array([3,0,1], dtype=np.int32)
    got = Tensor(source, device="ROCKCHIP")[Tensor(rows, device="ROCKCHIP"), Tensor(cols, device="ROCKCHIP")].realize().numpy()
    np.testing.assert_array_equal(got, source[rows,cols])
  def test_slice_fancy_indexing_dim_inject_and_collapse_leading_none(self):
    _,b,_,d,_,_,j,_,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[None,b,2,d,None], lambda x:x[None,j,2,o,None])
  def test_slice_fancy_indexing_dim_inject_and_collapse_ellipsis(self):
    _,_,_,d,_,_,_,_,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[...,1,d,None], lambda x:x[...,1,o,None])
  def test_slice_fancy_indexing_dim_collapse_int_middle(self):
    a,b,_,d,e,i,j,_,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,3,d,e], lambda x:x[i,j,3,o,p])
  def test_slice_fancy_indexing_dim_collapse_int_scalars(self):
    _,b,_,d,_,_,j,_,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[1,b,2,d,2], lambda x:x[1,j,2,o,2])
  def test_slice_fancy_indexing_dim_collapse_int_sparse(self):
    a,_,_,_,e,i,_,_,_,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,2,2,2,e], lambda x:x[i,2,2,2,p])
  def test_slice_fancy_indexing_dim_collapse_int_sliced(self):
    _,_,_,d,_,_,_,_,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[1,:,3:11:2,d,0:2], lambda x:x[1,:,3:11:2,o,0:2])
  def test_slice_fancy_indexing_no_dim_collapse_outer(self):
    _,b,c,d,_,_,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[:,b,c,d,:], lambda x:x[:,j,k,o,:])
  def test_slice_fancy_indexing_no_dim_collapse_trailing(self):
    a,b,_,_,_,i,j,_,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,...], lambda x:x[i,j,...])
  def test_slice_fancy_indexing_no_dim_collapse_spanning(self):
    a,_,_,_,e,i,_,_,_,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,...,e], lambda x:x[i,...,p])
  def test_slice_fancy_indexing_no_dim_collapse_middle(self):
    _,_,c,_,e,_,_,k,_,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[...,c,:,e], lambda x:x[...,k,:,p])
  def test_slice_fancy_indexing_dim_inject_none_leading(self):
    _,b,c,d,e,_,j,k,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[None,b,c,d,e], lambda x:x[None,j,k,o,p])
  def test_slice_fancy_indexing_dim_inject_none_trailing(self):
    a,b,c,d,_,i,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,c,d,None], lambda x:x[i,j,k,o,None])
  def test_slice_fancy_indexing_dim_inject_none_middle(self):
    a,b,_,d,e,i,j,_,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,None,d,e], lambda x:x[i,j,None,o,p])
  def test_slice_fancy_indexing_dim_inject_none_ends(self):
    _,b,c,d,_,_,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[None,b,c,d,None], lambda x:x[None,j,k,o,None])
  def test_slice_fancy_indexing_dim_inject_none_static_axis(self):
    a,_,_,d,e,i,_,_,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,:,None,d,e], lambda x:x[i,:,None,o,p])
  def test_slice_fancy_indexing_dim_inject_none_only(self):
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[None,None,None,None,None], lambda x:x[None,None,None,None,None])
  def test_slice_fancy_indexing_dim_inject_none_double_leading(self):
    _,b,c,d,e,_,j,k,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[None,None,b,c,d,e], lambda x:x[None,None,j,k,o,p])
  def test_slice_fancy_indexing_dim_inject_none_pairs(self):
    _,b,c,_,_,_,j,k,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[None,None,b,c,None,None], lambda x:x[None,None,j,k,None,None])
  def test_slice_fancy_indexing_dim_inject_none_internal_pair(self):
    a,_,c,d,e,i,_,k,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,None,None,c,d,e], lambda x:x[i,None,None,k,o,p])
  def test_slice_fancy_indexing_dim_inject_none_internal_and_trailing(self):
    a,_,c,_,_,i,_,k,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,None,None,c,None,None], lambda x:x[i,None,None,k,None,None])
  def test_slice_fancy_indexing_dim_inject_none_sparse(self):
    _,b,_,d,e,_,j,_,o,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[None,None,b,None,d,e], lambda x:x[None,None,j,None,o,p])
  def test_slice_fancy_indexing_list_with_tensor(self):
    a,_,_,_,_,i,_,_,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(a,)], lambda x:x[(i,)])
  def test_slice_fancy_indexing_list_with_tensor_and_scalar(self):
    a,_,_,_,_,i,_,_,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(a,1)], lambda x:x[(i,1)])
  def test_slice_fancy_indexing_list_with_tensor_and_tuple(self):
    a,_,_,_,_,i,_,_,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(a,(1,1))], lambda x:x[(i,(1,1))])
  def test_slice_fancy_indexing_list_indices_static(self):
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[((0,),)], lambda x:x[((0,),)])
  def test_slice_fancy_indexing_list_indices_leading(self):
    _,b,c,d,_,_,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(0,),b,c,d,:], lambda x:x[(0,),j,k,o,:])
  def test_slice_fancy_indexing_list_indices_broadcast(self):
    _,b,c,d,_,_,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[[[[0]]],b,c,d,[[1]]], lambda x:x[[[[0]]],j,k,o,[[1]]])
  def test_slice_fancy_indexing_list_indices_negative(self):
    _,b,c,d,_,_,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(1,0,-1),b,c,d,:], lambda x:x[(1,0,-1),j,k,o,:])
  def test_slice_fancy_indexing_list_indices_trailing(self):
    a,b,c,_,_,i,j,k,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,c,(1,2,3),...], lambda x:x[i,j,k,(1,2,3),...])
  def test_slice_fancy_indexing_list_indices_column(self):
    a,b,c,_,_,i,j,k,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,c,[[1],[2],[3]],...], lambda x:x[i,j,k,[[1],[2],[3]],...])
  def test_slice_fancy_indexing_list_indices_mixed(self):
    a,_,c,_,e,i,_,k,_,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,(2,1,0),c,(-2,1,0),e], lambda x:x[i,(2,1,0),k,(-2,1,0),p])
  def test_slice_fancy_indexing_tuple_indices_static(self):
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(((0,),),)], lambda x:x[(((0,),),)])
  def test_slice_fancy_indexing_tuple_indices_leading(self):
    _,b,c,d,_,_,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(0,),b,c,d,:], lambda x:x[(0,),j,k,o,:])
  def test_slice_fancy_indexing_tuple_indices_negative(self):
    _,b,c,d,_,_,j,k,o,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[(1,0),b,c,d,:], lambda x:x[(1,0),j,k,o,:])
  def test_slice_fancy_indexing_tuple_indices_trailing(self):
    a,b,c,_,_,i,j,k,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,b,c,(1,2,3),...], lambda x:x[i,j,k,(1,2,3),...])
  def test_slice_fancy_indexing_tuple_indices_column(self):
    a,_,c,_,_,i,_,k,_,_ = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[a,((2,),(1,),(0,)),c,(2,1,0)],
                  lambda x:x[i,((2,),(1,),(0,)),k,(2,1,0)])
  def test_slice_fancy_indexing_tuple_indices_none(self):
    _,_,c,_,e,_,_,k,_,p = self._get_index_randoms()
    _fp16_test_op([(2,5,6,5,3,4)], lambda x:x[1,(2,1,0),None,c,(2,1,0),e],
                  lambda x:x[1,(2,1,0),None,k,(2,1,0),p])
  def test_fancy_indexing_negative_nonfinite_bits(self):
    source = np.array([math.inf, -math.inf, math.nan], dtype=np.float16)
    indices = np.array([-1, -2, -3], dtype=np.int32)
    got = Tensor(source, device="ROCKCHIP")[Tensor(indices, device="ROCKCHIP")].realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), source[::-1].view(np.uint16))
  def test_multi_fancy_indexing_negative_nonfinite_bits(self):
    source = np.array([[math.inf, -math.inf, math.nan], [1.0, -0.0, 0.0]], dtype=np.float16)
    rows = Tensor(np.array([[0, 0, 0], [-1, -1, -1]], dtype=np.int32), device="ROCKCHIP")
    columns = Tensor(np.array([-3, -2, -1], dtype=np.int32), device="ROCKCHIP")
    got = Tensor(source, device="ROCKCHIP")[rows, columns].realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint16), source.view(np.uint16))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipScatterOps(_base.TestRockchipScatterOps):
  def test_scatter_upstream_numeric_cases(self):
    index = torch.randint(3, size=[3,4,5], dtype=torch.int64)
    dynamic_index = Tensor(index.numpy().astype(np.int32))
    for dim in (0,1,2,-1,-2,-3):
      with self.subTest(dim=dim):
        _fp16_test_op([(4,5,6), (4,5,6)], lambda x,src,dim=dim: x.scatter(dim, index, src),
                      lambda x,src,dim=dim: x.scatter(dim, dynamic_index, src), forward_only=True)
    _fp16_test_op([(3,4,5), (3,4,5)], lambda x,src: x.scatter(1, index, src),
                  lambda x,src: x.scatter(1, dynamic_index, src), forward_only=True)
    _fp16_test_op([(10,3,10), (10,10,10)], lambda x,src: x.scatter(1, index, src),
                  lambda x,src: x.scatter(1, dynamic_index, src), forward_only=True)

  def test_scatter_static_tensor_source(self):
    index = torch.arange(4, dtype=torch.int64).reshape(2,2)
    _fp16_test_op([(2,4), (2,2)], lambda x,src: x.scatter(1, index, src),
                  lambda x,src: x.scatter(1, Tensor.arange(4, dtype=dtypes.int32).reshape(2,2), src), forward_only=True)
  def test_scatter_static_dim0(self):
    index = torch.arange(4, dtype=torch.int64).reshape(2,2)
    _fp16_test_op([(4,2), (2,2)], lambda x,src: x.scatter(0, index, src),
                  lambda x,src: x.scatter(0, Tensor.arange(4, dtype=dtypes.int32).reshape(2,2), src), forward_only=True)
  def test_scatter_static_scalar_source(self):
    index = torch.arange(4, dtype=torch.int64).reshape(2,2)
    _fp16_test_op([(2,4)], lambda x: x.scatter(1, index, value=0.5),
                  lambda x: x.scatter(1, Tensor.arange(4, dtype=dtypes.int32).reshape(2,2), src=0.5), forward_only=True)
  def test_scatter_dynamic_scalar_source_nonfinite(self):
    source = np.array([[1., -2., 3.], [-4., 5., -6.]], dtype=np.float16)
    indices = np.array([[2, 0], [1, 1]], dtype=np.int32)
    expected = source.copy()
    expected[0, [2, 0]] = math.inf
    expected[1, 1] = math.inf
    got = Tensor(source, device="ROCKCHIP").scatter(1, Tensor(indices, device="ROCKCHIP"), src=math.inf).numpy()
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
  def test_scatter_reduce_static_sum(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="sum"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="sum"), forward_only=True)
  def test_scatter_reduce_static_max(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="amax"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="amax"), forward_only=True)
  def test_scatter_reduce_static_product(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="prod"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="prod"), forward_only=True)
  def test_scatter_reduce_static_min(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="amin"),
                  lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="amin"), forward_only=True)
  def test_scatter_reduce_static_mean(self):
    index = torch.zeros((2,4), dtype=torch.int64)
    _fp16_fp32_golden_test_op([(1,4), (2,4)], lambda x,src: x.scatter_reduce(0, index, src, reduce="mean"),
      lambda x,src: x.scatter_reduce(0, Tensor.zeros(2,4, dtype=dtypes.int32, buffer=False), src, reduce="mean"), forward_only=True)
  def test_scatter_dynamic_integer_index(self):
    base = np.array([0x7c00, 0xfc00, 0x7e01, 0x8000], dtype=np.uint16).view(np.float16)
    for index,source in (([1,3], [0x0000,0xfe01]), ([0,0], [0x7c00,0xfe01])):
      with self.subTest(index=index):
        indices = np.array(index, dtype=np.int32)
        values = np.array(source, dtype=np.uint16).view(np.float16)
        expected = base.copy()
        for lane,dst in enumerate(indices): expected[dst] = values[lane]
        got = Tensor(base).scatter(0, Tensor(indices), Tensor(values)).realize().numpy()
        np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
  def test_scatter_dynamic_multidim_tensor_source(self):
    shape = (2,3,4)
    bits = np.resize(np.array([0x3c00,0x7c00,0x8000,0x7e01,0xfc00,0x0000,0xfe01,0x4000], dtype=np.uint16), np.prod(shape))
    values = np.resize(np.array([0xbc00,0x7d01,0x3555,0x8000,0x7c00,0xfe01], dtype=np.uint16), np.prod(shape)).view(np.float16).reshape(shape)
    coordinates = np.indices(shape)
    for dim in range(len(shape)):
      with self.subTest(dim=dim):
        base = bits.view(np.float16).reshape(shape)
        axis = dim % len(shape)
        indices = (coordinates[(axis+1)%len(shape)] % shape[axis]).astype(np.int32)
        expected = base.copy()
        for position in np.ndindex(indices.shape):
          target = list(position)
          target[axis] = int(indices[position])
          expected[tuple(target)] = values[position]
        got = Tensor(base).scatter(dim, Tensor(indices), Tensor(values)).realize().numpy()
        np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
  def test_scatter_dynamic_int16(self):
    base = np.array([-32768, -30000, -1, 0, 1, 30000, 32767, 123], dtype=np.int16)
    indices = np.array([2, 2, 7, 0, 7, 4], dtype=np.int32)
    values = np.array([-32768, 32767, -30000, 30000, 1234, -1234], dtype=np.int16)
    expected = base.copy()
    for lane,dst in enumerate(indices): expected[dst] = values[lane]
    got = Tensor(base).scatter(0, Tensor(indices), Tensor(values)).realize().numpy()
    self.assertEqual(got.dtype, np.int16)
    np.testing.assert_array_equal(got, expected)
  def test_scatter_dynamic_multidim_int16(self):
    base = np.array([[-32768, -1, 32767], [30000, 0, -30000]], dtype=np.int16)
    indices = np.array([[2, 0, 2], [1, 1, 0]], dtype=np.int32)
    values = np.array([[1234, -1234, 30000], [32767, -32768, 1]], dtype=np.int16)
    expected = base.copy()
    for position in np.ndindex(indices.shape):
      target = list(position)
      target[1] = int(indices[position])
      expected[tuple(target)] = values[position]
    got = Tensor(base).scatter(1, Tensor(indices), Tensor(values)).realize().numpy()
    self.assertEqual(got.dtype, np.int16)
    np.testing.assert_array_equal(got, expected)
  def test_scatter_dynamic_scalar_reductions(self):
    base = np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float16)
    indices = Tensor(np.array([0, 0, 2, 1], dtype=np.int32)).realize()
    for mode,value,expected in (("add", 2.0, [5.0, 0.0, 5.0, 4.0]),
                                ("multiply", -1.0, [1.0, 2.0, -3.0, 4.0])):
      with self.subTest(mode=mode):
        got = Tensor(base).scatter(0, indices, value, reduce=mode).realize().numpy()
        np.testing.assert_array_equal(got, np.array(expected, dtype=np.float16))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBroadcastOps(_base.TestRockchipBroadcastOps):
  def test_broadcast_full_arithmetic(self):
    for torch_op, tinygrad_op in ((torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul), (torch.div, Tensor.div)):
      for shapes in (((5,3,14,16), (5,1,14,1)), ((1,3,1,7,1), (2,1,5,1,8))):
        with self.subTest(op=torch_op.__name__, shapes=shapes): _fp16_test_op(shapes, torch_op, tinygrad_op)
  def test_broadcast_partial_arithmetic(self):
    shapes = (((1,32,32,32), (1,32,1,1)), ((5,13,24,16,2), (1,13,24,1,1)), ((4,1), (4,5)), ((1,4), (5,4)))
    for torch_op, tinygrad_op in ((torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul), (torch.div, Tensor.div)):
      for pair in shapes:
        with self.subTest(op=torch_op.__name__, shapes=pair): _fp16_test_op(pair, torch_op, tinygrad_op)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipBitwiseOps(_base.TestRockchipBitwiseOps):
  def test_bitwise_not_exact_bytes(self):
    values = np.array([dtypes.int32.min, -16777217, -65536, -256, -1, 0,
                       1, 255, 256, 65535, 16777216, dtypes.int32.max], dtype=np.int32).reshape(3,4)
    booleans = np.array([[False, True, False], [True, True, False]], dtype=np.bool_)
    before = Device["ROCKCHIP"].submit_count, Device["ROCKCHIP"].task_count
    np.testing.assert_array_equal((~Tensor(values).permute(1,0)).realize().numpy(), np.bitwise_not(values.T))
    np.testing.assert_array_equal((~Tensor(booleans).flip(1)).realize().numpy(), np.logical_not(booleans[:,::-1]))
    self.assertEqual((Device["ROCKCHIP"].submit_count-before[0], Device["ROCKCHIP"].task_count-before[1]), (2,2))

  def test_bitwise_binary_exact_bytes(self):
    lhs = np.array([0x00000000, 0xffffffff, 0x80000000, 0x7fffffff, 0x00ff00ff, 0xff00ff00,
                    0x01020304, 0x10204080, 0xaaaaaaaa, 0x55555555, 0xdeadbeef, 0x12345678], dtype=np.uint32).view(np.int32).reshape(3,4)
    rhs = np.array([0xffffffff, 0x00000000, 0x7fffffff, 0x80000000, 0xff00ff00, 0x00ff00ff,
                    0x80706050, 0x01020408, 0x55555555, 0xaaaaaaaa, 0x0f0f0f0f, 0x87654321], dtype=np.uint32).view(np.int32).reshape(3,4)
    before = Device["ROCKCHIP"].submit_count, Device["ROCKCHIP"].task_count
    for tinygrad_op,numpy_op in ((Tensor.bitwise_and, np.bitwise_and), (Tensor.bitwise_or, np.bitwise_or),
                                 (Tensor.bitwise_xor, np.bitwise_xor)):
      np.testing.assert_array_equal(tinygrad_op(Tensor(lhs).permute(1,0), Tensor(rhs).permute(1,0)).realize().numpy(),
                                    numpy_op(lhs.T, rhs.T))
    self.assertEqual((Device["ROCKCHIP"].submit_count-before[0], Device["ROCKCHIP"].task_count-before[1]), (3,284))

  def test_shift_all_counts(self):
    values = np.asarray([0, 1, -1, np.iinfo(np.int32).min, np.iinfo(np.int32).max, 0x12345678]*6, dtype=np.int32)[:32]
    shifts = np.arange(32, dtype=np.int32)
    signed, unsigned = Tensor(values), Tensor(values.view(np.uint32))
    np.testing.assert_array_equal((signed << 2).numpy(), np.left_shift(values, 2))
    np.testing.assert_array_equal((signed << Tensor(shifts)).numpy(), np.left_shift(values, shifts))
    np.testing.assert_array_equal((signed >> Tensor(shifts)).numpy(), np.right_shift(values, shifts))
    unsigned_shifts = shifts.view(np.uint32)
    np.testing.assert_array_equal(((unsigned >> Tensor(unsigned_shifts)).cast(dtypes.int32)).numpy(),
                                  np.right_shift(values.view(np.uint32), unsigned_shifts).view(np.int32))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIntegerDivisionOps(unittest.TestCase):
  def test_signed_limits_and_rounding(self):
    lhs = np.array([dtypes.int32.min, dtypes.int32.max, -7, 7, -1, 1, 0], dtype=np.int32)
    rhs = np.array([-1, 1, 3, -3, 2, -2, 5], dtype=np.int32)
    before = Device["ROCKCHIP"].submit_count, Device["ROCKCHIP"].task_count
    trunc = Tensor(lhs).div(Tensor(rhs), rounding_mode="trunc").realize().numpy()
    floor = (Tensor(lhs)//Tensor(rhs)).realize().numpy()
    modulo = (Tensor(lhs)%Tensor(rhs)).realize().numpy()
    fmod = Tensor(lhs).fmod(Tensor(rhs)).realize().numpy()
    np.testing.assert_array_equal(trunc, np.array([dtypes.int32.min, dtypes.int32.max, -2, -2, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(floor, np.array([dtypes.int32.min, dtypes.int32.max, -3, -3, -1, -1, 0], dtype=np.int32))
    np.testing.assert_array_equal(modulo, np.array([0, 0, 2, -2, 1, -1, 0], dtype=np.int32))
    np.testing.assert_array_equal(fmod, np.array([0, 0, -1, 1, -1, 1, 0], dtype=np.int32))
    self.assertEqual((Device["ROCKCHIP"].submit_count-before[0], Device["ROCKCHIP"].task_count-before[1]), (4, 13980))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIntegerProductOps(unittest.TestCase):
  def test_signed_limits(self):
    lhs = np.array([dtypes.int32.min, dtypes.int32.max, -65537, -46341, -256, -1, 0, 1, 255, 46341, 65537], dtype=np.int32)
    rhs = np.array([-1, 2, 65537, 46341, -257, dtypes.int32.min, 9, -7, 257, 46341, 65537], dtype=np.int32)
    before = Device["ROCKCHIP"].submit_count, Device["ROCKCHIP"].task_count
    got = (Tensor(lhs)*Tensor(rhs)).realize().numpy()
    np.testing.assert_array_equal(got, (lhs.astype(np.int64)*rhs.astype(np.int64)).astype(np.int32))
    self.assertEqual((Device["ROCKCHIP"].submit_count-before[0], Device["ROCKCHIP"].task_count-before[1]), (1, 1468))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipLogicalPredicateOps(_base.TestRockchipLogicalPredicateOps):
  def test_isclose_exact_fp16(self):
    lhs = np.array([0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e01], dtype=np.uint16).view(np.float16)
    rhs = np.array([0x8000, 0x0000, 0x0000, 0x03ff, 0x3c01, 0x7bff, 0x7c00, 0xfc00, 0x7e01], dtype=np.uint16).view(np.float16)
    expected = np.array([True, True, False, True, False, True, True, True, False])
    before = Device["ROCKCHIP"].submit_count
    np.testing.assert_array_equal(Tensor(lhs).isclose(Tensor(rhs)).realize().numpy(), expected)
    np.testing.assert_array_equal(Tensor(lhs).isclose(Tensor(rhs), equal_nan=True).realize().numpy(), expected | np.isnan(lhs) & np.isnan(rhs))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 2)

  def test_logical_and(self):
    _fp16_test_op(None, lambda x:(1 < x) & (x < 2), forward_only=True, vals=[[1.2, 1.2, 1.2, 3.2]])

  def test_logical_or(self):
    _fp16_test_op(None, lambda x:(x < -1) | (x > 1), forward_only=True,
                  vals=[[-math.inf, -2., -1., -0., 0., 1., 2., math.inf, math.nan]])

  def test_logical_xor(self):
    _fp16_test_op(None, lambda x:(x < 0) ^ (x > 1), forward_only=True,
                  vals=[[-math.inf, -2., -1., -0., 0., 1., 2., math.inf, math.nan]])

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipInt16EWOps(_base.TestRockchipInt16EWOps):
  def test_add_sub(self):
    a = [-30000,-1200,-7,-1,0,1,1200,30000]
    b = [1000,-30,2,-1,1,2,-30,1000]
    self._check([-29000,-1230,-5,-2,1,3,1170,31000], lambda x,y:x+y, a, b)
    self._check([-31000,-1170,-9,0,-1,-1,1230,29000], lambda x,y:x-y, a, b)
  def test_mul_max(self):
    a, b = [-100,-24,-7,-1,0,1,24,100], [20,-6,2,-1,1,2,-3,5]
    self._check([-2000,144,-14,1,0,2,-72,500], lambda x,y:x*y, a, b)
    self._check([20,-6,2,-1,1,2,24,100], lambda x,y:x.maximum(y), a, b)
  def test_min_abs_neg(self):
    a = [-30000,-1200,-7,-1,0,1,1200,30000]
    b = [1000,-30,2,-1,1,2,-30,1000]
    self._check([-30000,-1200,-7,-1,0,1,-30,1000], lambda x,y:x.minimum(y), a, b)
    self._check([30000,1200,7,1,0,1,1200,30000], lambda x:x.abs(), a)
    self._check([30000,1200,7,1,0,-1,-1200,-30000], lambda x:-x, a)
  def test_saturating_limit(self):
    self._check([32767,-32768], lambda x,y:x+y, [32000,-32000], [1000,-1000])
    self._check([32767,-32768], lambda x,y:x-y, [32000,-32000], [-1000,1000])
    self._check([32767,-32768], lambda x,y:x*y, [300,-300], [300,300])
    self._check([32767], lambda x:x.abs(), [-32768])
    self._check([32767], lambda x:-x, [-32768])
  def test_broadcast_chain(self):
    a = [[-8,-4,0,4], [8,12,16,20]]
    self._check([[-9,-9,5,17], [17,17,37,49]], lambda x,y:(x+y).maximum(-3)*2-3, a, [2,-2,4,6])
  def test_fused_concat_neg(self):
    a = np.asarray([[-32768,-7,0], [1,1200,32767]], dtype=np.int16)
    b = np.asarray([[3,-4,5], [6,-7,8]], dtype=np.int16)
    expected = np.clip(-np.concatenate((a, b), axis=1).astype(np.int32), -32768, 32767).astype(np.int16)
    self._check(expected, lambda x,y:Tensor.cat(x, y, dim=1).neg(), a, b, expected_tasks=1)
  def test_fused_stack_abs(self):
    a = np.asarray([[-32768,-7,0], [1,1200,32767]], dtype=np.int16)
    b = np.asarray([[3,-4,5], [6,-7,8]], dtype=np.int16)
    expected = np.clip(np.abs(np.stack((a, b)).astype(np.int32)), 0, 32767).astype(np.int16)
    self._check(expected, lambda x,y:Tensor.stack(x, y).abs(), a, b, expected_tasks=3)
  def test_fused_permute_add(self):
    a = np.asarray([[-32768,-7,0], [1,1200,32760]], dtype=np.int16)
    b = np.arange(6, dtype=np.int16).reshape(3,2)
    self._check(a.T+b, lambda x,y:x.permute(1,0)+y, a, b, expected_tasks=1)
  def test_fused_pad_clip(self):
    a = np.asarray([[-32768,-7,0], [1,1200,32767]], dtype=np.int16)
    expected = np.clip(np.pad(a, ((1,1),(2,1)), constant_values=-7), -5, 6)
    self._check(expected, lambda x:x.pad(((1,1),(2,1)), value=-7).clip(-5, 6), a, expected_tasks=2)
  def test_fused_repeat_add(self):
    a = np.asarray([[-30000,-7,0], [1,1200,30000]], dtype=np.int16)
    b = np.ones((4,9), dtype=np.int16)
    self._check(np.tile(a, (2,3))+b, lambda x,y:x.repeat(2,3)+y, a, b, expected_tasks=1)
  def test_int32_writeback(self):
    a = [-30000,-1200,-7,-1,0,1,1200,30000]
    b = [1000,-30,2,-1,1,2,-30,1000]
    self._check([-29000,-1230,-5,-2,1,3,1170,31000], lambda x,y:(x+y).cast(dtypes.int32), a, b,
                output_dtype=np.int32)
    self._check(a, lambda x:x.cast(dtypes.int32), a, output_dtype=np.int32)
  def test_sum_unrolled(self):
    values = np.asarray([[-32768, 32767, 1], [30000, 2000, -1234]], dtype=np.int16)
    self._check(values.sum(1, dtype=np.int32), lambda x:x.sum(1), values, output_dtype=np.int32,
                expected_tasks=6, expected_submits=2)
    self._check(values.sum(0, dtype=np.int32), lambda x:x.sum(0), values, output_dtype=np.int32,
                expected_tasks=4, expected_submits=2)
    self._check(values.sum(dtype=np.int32), lambda x:x.sum(), values, output_dtype=np.int32,
                expected_tasks=12, expected_submits=2)
  def test_sum_loop(self):
    values = (np.arange(40*257, dtype=np.uint32)*7919).astype(np.uint16).view(np.int16).reshape(40, 257)
    self._check(values.sum(1, dtype=np.int32), lambda x:x.sum(1), values, output_dtype=np.int32,
                expected_tasks=1542, expected_submits=2)
  def test_cumsum_unrolled(self): self._check_cumsum((17,), 0, 68)
  def test_cumsum_1d(self): self._check_cumsum((257,), 0, 8738)
  def test_cumsum_last_axis(self): self._check_cumsum((2,257), 1, 16962)
  def test_cumsum_first_axis(self): self._check_cumsum((257,2), 0, 16962)
  def test_product_loop(self):
    values = np.ones((40,257), dtype=np.int16)
    values[:, :3] = (300,300,-1)
    values[::2, 3] = 0
    self._check(self._saturating_cumprod(values, 1)[:, -1], lambda x:x.prod(1), values, expected_tasks=257)
  def test_cumprod_1d(self):
    values = np.ones(257, dtype=np.int16)
    values[:3] = (300,300,-1)
    self._check(self._saturating_cumprod(values, 0), lambda x:x.cumprod(0), values, expected_tasks=257)
  def test_cumprod_last_axis(self):
    values = np.ones((2,257), dtype=np.int16)
    values[:, :3] = ((300,300,-1), (-300,-300,-1))
    self._check(self._saturating_cumprod(values, 1), lambda x:x.cumprod(1), values, expected_tasks=257)
  def test_cumprod_first_axis(self):
    values = np.ones((257,2), dtype=np.int16)
    values[:3, :] = ((300,-300), (300,-300), (-1,-1))
    self._check(self._saturating_cumprod(values, 0), lambda x:x.cumprod(0), values, expected_tasks=258, expected_submits=2)
  def test_any_reduction_loop(self):
    values = np.zeros(257, dtype=np.int16)
    values[128] = -32768
    self._check_bool(lambda x:x.any(), values, expected_tasks=259)
    matrix = np.zeros((2,257), dtype=np.int16)
    matrix[0,128] = -32768
    self._check_bool(lambda x:x.any(1), matrix, expected_tasks=259)
  def test_all_reduction_loop(self):
    values = np.ones(257, dtype=np.int16)
    values[128] = 0
    self._check_bool(lambda x:x.all(), values, expected_tasks=259)
    matrix = np.ones((257,2), dtype=np.int16)
    matrix[128,0] = 0
    self._check_bool(lambda x:x.all(0), matrix, expected_tasks=259)
  def test_nonzero_count_unrolled(self):
    values = np.asarray([0,-32768,0,1,-1,0,32767], dtype=np.int16)
    self._check(np.count_nonzero(values), lambda x:(x != 0).sum(), values, output_dtype=np.int32, expected_tasks=10)
  def test_nonzero_count_loop(self):
    values = np.zeros(257, dtype=np.int16)
    values[[0,128,256]] = (-1,-32768,32767)
    self._check(np.count_nonzero(values), lambda x:(x != 0).sum(), values, output_dtype=np.int32, expected_tasks=260)
  def test_nonzero_count_axes(self):
    values = np.zeros((2,257), dtype=np.int16)
    values[0,[0,128,256]], values[1,::3] = (-1,-32768,32767), 1
    self._check(np.count_nonzero(values, axis=1), lambda x:(x != 0).sum(1), values,
                output_dtype=np.int32, expected_tasks=260)
    values = values.T.copy()
    self._check(np.count_nonzero(values, axis=0), lambda x:(x != 0).sum(0), values,
                output_dtype=np.int32, expected_tasks=260)
  def test_compare_ordering(self):
    a = [-32768,-32768,-30000,-1,0,1,30000,32767]
    b = [32767,-32768,30000,0,0,-1,-30000,-32768]
    for op in (lambda x,y:x<y, lambda x,y:x>y, lambda x,y:x<=y, lambda x,y:x>=y): self._check_bool(op, a, b)
  def test_compare_equality(self):
    a = [-32768,-32768,-30000,-1,0,1,30000,32767]
    b = [32767,-32768,30000,0,0,-1,-30000,-32768]
    for op in (lambda x,y:x==y, lambda x,y:x!=y): self._check_bool(op, a, b)
  def test_compare_logical(self):
    a = np.asarray([-32768,-32768,-30000,-1,0,1,30000,32767], dtype=np.int16)
    b = np.asarray([32767,-32768,30000,0,0,-1,-30000,-32768], dtype=np.int16)
    for op in (lambda x,y:(x<y)&(x!=0), lambda x,y:(x<y)|(x!=0), lambda x,y:(x<y)^(x!=0)): self._check_bool(op, a, b)
    self._check_bool(lambda x,y:(x<y).logical_not(), a, b, expected=np.logical_not(a<b))
  def test_where_logical_broadcast(self):
    a = np.asarray([[-8,-4,0,4], [8,12,16,20]], dtype=np.int16)
    b = np.asarray([2,-2,4,6], dtype=np.int16)
    condition = (a < b) & (a != 0)
    self._check(np.where(condition, a+b, a-b), lambda x,y:((x<y)&(x!=0)).where(x+y, x-y), a, b)
  def test_hard_activation(self):
    values = np.asarray([-32768,-7,-6,-1,0,1,6,7,32767], dtype=np.int16)
    self._check(np.clip(values, 0, 6), lambda x:x.relu6(), values, expected_tasks=2)
    self._check(np.clip(values, -3, 4), lambda x:x.hardtanh(-3, 4), values, expected_tasks=2)
  def test_leaky_relu_integral(self):
    values = np.asarray([-32768,-20000,-1200,-1,0,1,1200,20000,32767], dtype=np.int16)
    for slope, tasks in ((2, 2), (3, 3)):
      wide = values.astype(np.int32)
      expected = np.where(values < 0, np.clip(wide*slope, -32768, 32767), wide).astype(np.int16)
      self._check(expected, lambda x,slope=slope:x.leaky_relu(slope), values, expected_tasks=tasks)
  def test_compare_broadcast_scalar(self):
    a = [[-32768,-1,0,32767], [32767,2,-3,-32768]]
    self._check_bool(lambda x,y:x>=y, a, [-32768,0,0,32767])
    self._check_bool(lambda x,y:x<y, a, 1)
    self._check_bool(lambda x,y:x<y, 1, a)
  def test_compare_large(self):
    a = np.arange(131072, dtype=np.uint16).view(np.int16)
    self._check_bool(lambda x,y:x<y, a, np.roll(a, 1))
  def test_int32_byte_add_wrap(self):
    a = np.array([0x7fffffff, -1, -0x80000000, 0x12345678], dtype=np.int32)
    b = np.array([1, 1, -1, 0x6dcba988], dtype=np.int32)
    c = np.array([0, -1, 1, 0], dtype=np.int32)
    with np.errstate(over="ignore"): expected = np.add(np.add(a, b, dtype=np.int32), c, dtype=np.int32)
    before = Device["ROCKCHIP"].submit_count
    got = (Tensor(a, device="ROCKCHIP") + Tensor(b, device="ROCKCHIP") + Tensor(c, device="ROCKCHIP")).numpy()
    np.testing.assert_array_equal(got, expected)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 1)
  def test_reduce_extrema_loop(self):
    values = np.arange(40*257, dtype=np.uint16).view(np.int16).reshape(40, 257)
    self._check(values.max(1), lambda x:x.max(1), values, expected_tasks=256)
    self._check(values.min(1), lambda x:x.min(1), values, expected_tasks=256)
  def test_cumulative_extrema(self):
    for count in (17, 257):
      values = (np.arange(count, dtype=np.uint32)*7919).astype(np.uint16).view(np.int16)
      self._check(np.maximum.accumulate(values), lambda x:x.cummax(0)[0], values, expected_tasks=count-1)
      self._check(np.minimum.accumulate(values), lambda x:x.cummin(0)[0], values, expected_tasks=count-1)
  def test_arg_extrema(self):
    values = np.asarray([32767, -32768, 7, 7, -1, 32767, 0, -32768], dtype=np.int16)
    self._check_index(np.argmax(values), lambda x:x.argmax(), values, 1, 22)
    self._check_index(np.argmin(values), lambda x:x.argmin(), values, 1, 31)
    matrix = values[:4].reshape(1, 4).repeat(3, 0)
    self._check_index(np.argmax(matrix, axis=1), lambda x:x.argmax(1), matrix, 2, 14)
    self._check_index(np.argmin(matrix, axis=1), lambda x:x.argmin(1), matrix, 2, 16)
    self._check_index(np.argmax(matrix, axis=0), lambda x:x.argmax(0), matrix, 2, 12)
    self._check_index(np.argmin(matrix, axis=0), lambda x:x.argmin(0), matrix, 2, 14)
    wide = (np.arange(257, dtype=np.uint32)*7919).astype(np.uint16).view(np.int16)
    self._check_index(np.argmax(wide), lambda x:x.argmax(), wide, 2, 520)
    self._check_index(np.argmin(wide), lambda x:x.argmin(), wide, 2, 522)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipArgExtremaOps(_base.TestRockchipArgExtremaOps):
  def test_argmax_first_tie(self): self._test("max", ((2.0, 2.0), (1.0, 2.0, 2.0)))
  def test_argmin_first_tie(self): self._test("min", ((2.0, 2.0), (3.0, 2.0, 2.0)))
  def test_argmax_axes(self): self._test_axes("max")
  def test_argmin_axes(self): self._test_axes("min")
  def test_argmax_global(self): self._test_global("max")
  def test_argmin_global(self): self._test_global("min")

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSortValueOps(_base.TestRockchipSortValueOps):
  def test_sort_values_trivial(self):
    for shape in ((0,), (0,5), (1,), (1,5)):
      with self.subTest(shape=shape):
        _fp16_test_op([shape], lambda x: x.sort(0).values, lambda x: x.sort(0)[0], forward_only=True)
  def test_sort_values_axes(self):
    for axis in (-1, 0, 1):
      for descending in (True, False):
        with self.subTest(axis=axis, descending=descending):
          _fp16_test_op([(8,8,6)], lambda x, axis=axis, descending=descending: x.sort(axis, descending).values,
                        lambda x, axis=axis, descending=descending: x.sort(axis, descending)[0], forward_only=True)
  def test_sort_values_repeated(self):
    values = np.array([0, 1] * 9, dtype=np.float16)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _fp16_test_op(None, lambda x, descending=descending: x.sort(stable=True, descending=descending).values,
                      lambda x, descending=descending: x.sort(descending=descending)[0], vals=[values], forward_only=True)
  def test_sort_values_infinity(self):
    values = np.array([-np.inf, 2.0], dtype=np.float16)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _fp16_test_op(None, lambda x, descending=descending: x.sort(descending=descending).values,
                      lambda x, descending=descending: x.sort(descending=descending)[0], vals=[values], forward_only=True)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSortIndexOps(_base.TestRockchipSortIndexOps):
  def test_sort_indices_trivial(self):
    for shape in ((0,), (0,5), (1,), (1,5)):
      with self.subTest(shape=shape):
        _TEST_OPS_HELPER([shape], lambda x: x.sort(0).indices.int(), lambda x: x.sort(0)[1], forward_only=True)
  def test_sort_indices_last_descending(self): self._axis(-1, True)
  def test_sort_indices_last_ascending(self): self._axis(-1, False)
  def test_sort_indices_axis0_descending(self): self._axis(0, True)
  def test_sort_indices_axis0_ascending(self): self._axis(0, False)
  def test_sort_indices_axis1_ascending(self): self._axis(1, False)
  def test_sort_indices_repeated(self):
    values = np.array([0, 1] * 9, dtype=np.float16)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _TEST_OPS_HELPER(None, lambda x, descending=descending: x.sort(stable=True, descending=descending).indices.int(),
                         lambda x, descending=descending: x.sort(descending=descending)[1], vals=[values], forward_only=True)
  def test_sort_indices_signed_zero_and_infinity(self):
    values = np.array([0.0, -0.0, np.inf, np.inf, -np.inf, -np.inf], dtype=np.float16)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _TEST_OPS_HELPER(None, lambda x, descending=descending: x.sort(stable=True, descending=descending).indices.int(),
                         lambda x, descending=descending: x.sort(descending=descending)[1], vals=[values], forward_only=True)
  def test_sort_int32_limits(self):
    values = np.array([0, -(1 << 31), (1 << 31)-1, -1, 0, -(1 << 31)], dtype=np.int32)
    for descending in (False, True):
      with self.subTest(descending=descending):
        _TEST_OPS_HELPER(None, lambda x, descending=descending: x.sort(stable=True, descending=descending).indices.int(),
                         lambda x, descending=descending: x.sort(descending=descending)[1], vals=[values], forward_only=True)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipTopKOps(_base.TestRockchipTopKOps):
  def test_topk_1d(self): self._test((8,), 3, -1, True)
  def test_topk_axis0_largest(self): self._test((5,5,4), 4, 0, True)
  def test_topk_axis1_smallest(self): self._test((5,5,4), 4, 1, False)
  def test_topk_repeated(self):
    values = np.array([1,1,0,1,0,1,0,0,1,0,0,0,1,0], dtype=np.float16)
    for largest,expected in ((True, [0,1,3]), (False, [2,4,6])):
      with self.subTest(largest=largest):
        result_values, result_indices = Tensor(values).topk(3, largest=largest)
        np.testing.assert_array_equal(result_values.numpy(), values[expected])
        np.testing.assert_array_equal(result_indices.numpy(), expected)
    with self.assertRaises((RuntimeError, ValueError)): Tensor(np.zeros(4, dtype=np.float16)).topk(5)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSignOps(_base.TestRockchipSignOps):
  def test_sign_nonfinite(self):
    values = np.array([-math.inf, -1., -0., 0., 1., math.inf, math.nan], dtype=np.float16)
    before = Device["ROCKCHIP"].submit_count
    got = Tensor(values).sign().realize().numpy()
    expected = np.array([-1., -1., 0., 0., 1., 1., 0.], dtype=np.float16)
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 4)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipCastOps(_base.TestRockchipCastOps):
  def test_cast_float(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(3,3)], lambda x:x.float(), forward_only=True)
    values = np.array([0x8000,0x0000,0xbc00,0x3c00,0x0001,0x8001,0x7c00,0x7e01,0x7bff], dtype=np.uint16).view(np.float16)
    got = Tensor(values).float().realize().numpy()
    np.testing.assert_array_equal(got.view(np.uint32), values.astype(np.float32).view(np.uint32))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 2)
  def test_cast_bool(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(3,3)], lambda x:x.bool(), forward_only=True)
    _fp16_test_op(None, lambda x:x.bool(), vals=[[-2., -0., 0., 1., math.inf, -math.inf, math.nan]], forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 6)
  def test_cast_int(self):
    before = Device["ROCKCHIP"].submit_count
    _fp16_test_op([(3,3)], lambda x:x.int(), forward_only=True)
    _fp16_test_op(None, lambda x:x.int(), vals=[[-2.9, -2.5, -1.9, -0.9, -0., 0., 0.9, 1.9, 2.5, 2.9]], forward_only=True)
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 4)
  def test_cast_integer_and_bool_to_float(self):
    integers = np.array([-2048, -257, -1, 0, 1, 257, 2048], dtype=np.int32)
    booleans = np.array([False, True, True, False], dtype=np.bool_)
    np.testing.assert_array_equal(Tensor(integers).float().numpy(), integers.astype(np.float32))
    np.testing.assert_array_equal(Tensor(booleans).float().numpy(), booleans.astype(np.float32))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipComparisonOps(_base.TestRockchipComparisonOps):
  def test_cmp_ne(self): self._test_cmp(lambda x,y:x != y, reverse=False)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipIntegralRoundingOps(_base.TestRockchipIntegralRoundingOps):
  def test_all_fp16_encodings(self):
    values = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    before = Device["ROCKCHIP"].submit_count
    with np.errstate(invalid="ignore"):
      for name in ("floor", "ceil", "trunc"):
        np.testing.assert_equal(getattr(Tensor(values), name)().numpy(), getattr(np, name)(values))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 3)
  def test_round_all_fp16_encodings(self):
    values = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    before = Device["ROCKCHIP"].submit_count
    with np.errstate(invalid="ignore"):
      np.testing.assert_equal(Tensor(values).round().numpy(), np.round(values))
    self.assertEqual(Device["ROCKCHIP"].submit_count-before, 10)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipWhereOps(_base.TestRockchipWhereOps):
  def test_where_tensor(self):
    _fp16_test_op([(100,), (100,), (100,)], lambda x,a,b: torch.where(x > .1, a, b), lambda x,a,b: (x > .1).where(a, b),
                  forward_only=True)
  def test_where_scalar(self):
    _fp16_test_op([(2,3,4,5)], lambda x: torch.where(x < -.2, 3.0, -2.0), lambda x: (x < -.2).where(3.0, -2.0),
                  forward_only=True)
  def test_where_broadcast(self):
    _fp16_test_op([(2,3,4,5), (5,), (1,3,1,1)], lambda x,a,b: torch.where(x > 0, a, b),
                  lambda x,a,b: (x > 0).where(a, b), forward_only=True)
  def test_where_cmpne_exact(self):
    values = [[-2., -1., 0., 1., 2.], [-2., 0., 0., 0., 2.], [4., 3., 2., 1., 0.], [-4., -3., -2., -1., 0.]]
    _fp16_test_op(None, lambda x,y,a,b: torch.where(x != y, a, b), lambda x,y,a,b: (x != y).where(a, b), vals=values,
                  forward_only=True)
  def test_where_boolean_composition(self):
    _fp16_test_op([(100,)], lambda x: torch.where((x > -.5) & (x < .5), x*2, x-1),
                  lambda x: ((x > -.5) & (x < .5)).where(x*2, x-1), forward_only=True)
    _fp16_test_op([(100,)], lambda x: torch.where((x < -.5) | (x > .5), x+2, x-2),
                  lambda x: ((x < -.5) | (x > .5)).where(x+2, x-2), forward_only=True)
  def test_where_nested(self):
    _fp16_test_op([(100,), (100,), (100,)], lambda x,a,b: torch.where(x < -.5, a, torch.where(x > .5, b, x)),
                  lambda x,a,b: (x < -.5).where(a, (x > .5).where(b, x)), forward_only=True)
  def test_masked_fill_finite(self):
    _fp16_test_op([(32,10)], lambda x: x.masked_fill(x > .1, -3.25), lambda x: x.masked_fill(x > .1, -3.25), forward_only=True)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipMovementOps(_base.TestRockchipMovementOps):
  def test_int16_raw_movement(self):
    values = np.array([[-32768, -256, -1, 0], [1, 256, 32767, 123]], dtype=np.int16)
    got = Tensor(values, device="ROCKCHIP").permute(1, 0).flip(0)[1:4].numpy()
    np.testing.assert_array_equal(got, np.flip(values.T, axis=0)[1:4])

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipConcatOps(_base.TestRockchipConcatOps):
  def test_int16_concat_stack_repeat(self):
    a = np.array([[-32768, -1, 0], [1, 256, 32767]], dtype=np.int16)
    b = np.array([[-30000, 30000], [-256, 123]], dtype=np.int16)
    got = Tensor.cat(Tensor(a, device="ROCKCHIP"), Tensor(b, device="ROCKCHIP"), dim=1).numpy()
    np.testing.assert_array_equal(got, np.concatenate((a, b), axis=1))
    got = Tensor.stack(Tensor(a, device="ROCKCHIP"), Tensor(-a, device="ROCKCHIP"), dim=1).numpy()
    np.testing.assert_array_equal(got, np.stack((a, -a), axis=1))
    np.testing.assert_array_equal(Tensor(b, device="ROCKCHIP").repeat(2, 1).numpy(), np.tile(b, (2, 1)))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipPaddingOps(_base.TestRockchipPaddingOps):
  def test_int16_constant_padding(self):
    values = np.array([[-32768, -1, 0], [1, 256, 32767]], dtype=np.int16)
    got = Tensor(values, device="ROCKCHIP").pad(((1, 2), (2, 1)), value=-12345).numpy()
    np.testing.assert_array_equal(got, np.pad(values, ((1, 2), (2, 1)), constant_values=-12345))

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipSqrtOps(_base.TestRockchipSqrtOps):
  def test_sqrt_nonfinite(self):
    values = np.array([-4.0, 0.0, 2**-14, 0.25, 2.0, 65504.0, math.inf, math.nan], dtype=np.float16)
    with np.errstate(invalid="ignore", divide="ignore"):
      np.testing.assert_allclose(Tensor(values).sqrt().numpy(), np.sqrt(values), equal_nan=True, **_FP16)
      np.testing.assert_allclose(Tensor(values).rsqrt().numpy(), 1/np.sqrt(values), equal_nan=True, **_FP16)

@_only_local_tests
@unittest.skipUnless(Device.DEFAULT == "ROCKCHIP", "ROCKCHIP device only")
class TestRockchipReductionOps(_base.TestRockchipReductionOps):
  def test_std_mean_fp16_input_only(self):
    values = np.arange(12, dtype=np.float16).reshape(3, 4)
    got = Tensor.stack(*Tensor(values).std_mean()).numpy()
    expected = np.array([values.astype(np.float32).std(ddof=1), values.astype(np.float32).mean()], dtype=np.float16)
    np.testing.assert_allclose(got, expected, **_FP16)

  def test_non_fp16_reductions(self):
    self.skipTest("Rockchip DPU accepts FP16 tensors only; FP32 dtype, boolean, and integer reductions are excluded")
