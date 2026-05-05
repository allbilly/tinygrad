import os, sys, time, math, unittest, functools, platform, warnings, subprocess, pickle, ctypes, tempfile
from pathlib import Path
import numpy as np
import torch

if os.getenv("ROCKCHIP") == "1" and "DEV" not in os.environ: os.environ["DEV"] = "ROCKCHIP"

from tinygrad.helpers import getenv, CI, DEBUG, DEV, IMAGE, Context
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.tensor import _to_np_dtype
from tinygrad.uop.ops import Ops
from tinygrad.runtime.support.rockchip import (
  RKPatch, RKTaskTemplate, RKTemplatePackage, build_conv1x1_template, build_elementwise_template, build_lut, build_wmma_template, conv_params,
  decode_template, encode_template, apply_patches, lut_enabled, pack_conv_input, pack_conv_weights, patch_regcmd, rkcmd, unpack_conv_output,
  validate_template, submit_plan, submit_template, wmma_params,
)
from tinygrad.runtime.autogen import rockchip as rk

if getenv("TINY_BACKEND"):
  import tinygrad.nn.torch # noqa: F401 # pylint: disable=unused-import
  torch.set_default_device("tiny")

warnings.filterwarnings("ignore", message="Non-empty compiler output encountered")

FORWARD_ONLY = getenv("FORWARD_ONLY", 0)
PRINT_TENSORS = getenv("PRINT_TENSORS", 0)
COMPILE_ONLY = Device.DEFAULT == "NULL"
RK3588_REF = Path.home() / "rk3588"

def slow_test(test_func):
  return unittest.skipIf(getenv("SKIP_SLOW_TEST"), "Skipping slow test")(test_func)

def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-2, high=2):
  if tinygrad_fxn is None: tinygrad_fxn = torch_fxn
  ts, tst = prepare_test_op(low, high, shps, vals, forward_only)

  st = time.monotonic()
  out = torch_fxn(*ts)
  torch_fp = time.monotonic() - st

  # move inputs to a different device, test the device of intermediate tensors are correct
  if mt:=getenv("MOVE_TENSOR", ""):
    for t in tst: t.to_(mt)

  st = time.monotonic()
  ret = tinygrad_fxn(*tst).realize()
  tinygrad_fp = time.monotonic() - st

  def compare(s, tinygrad_output, torch_output, atol, rtol):
    if COMPILE_ONLY: return
    if PRINT_TENSORS: print(s, tinygrad_output, torch_output)
    try:
      assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={tinygrad_output.shape} | torch={torch_output.shape}"
      assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={tinygrad_output.dtype} | torch={torch_output.dtype}"
      if np.issubdtype(tinygrad_output.dtype, np.floating):
        np.testing.assert_allclose(tinygrad_output, torch_output, atol=atol, rtol=rtol)
      else:
        np.testing.assert_equal(tinygrad_output, torch_output)
    except Exception as e:
      raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")

  if DEBUG >= 6:
    np.set_printoptions(linewidth=200, suppress=True)
    print(ret.numpy())
    print(out.detach().cpu().numpy())
  compare("forward pass", ret.numpy(), out.detach().cpu().numpy(), atol=atol, rtol=rtol)

  torch_fbp, tinygrad_fbp = np.nan, np.nan
  if not forward_only and not FORWARD_ONLY and ts and tst:
    st = time.monotonic()
    torch_grads = torch.autograd.grad(torch_fxn(*ts).sum(), ts)
    torch_fbp = time.monotonic() - st

    st = time.monotonic()
    # NOTE: we now have to recompute the forward pass since we realized it
    tiny_grads = tinygrad_fxn(*tst).sum().gradient(*tst)
    Tensor.realize(*tiny_grads)
    tinygrad_fbp = time.monotonic() - st

    for i, (t, torch_grad) in enumerate(zip(tiny_grads, torch_grads)):
      compare(f"backward pass tensor {i}", t.numpy(), torch_grad.detach().cpu().numpy(), atol=grad_atol, rtol=grad_rtol)

  if not CI:
    print("\ntesting %40r   torch/tinygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % \
          (shps, torch_fp*1000, tinygrad_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(low, high, shps, vals, forward_only=False):
  if shps is None:
    ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
  else:
    np.random.seed(0)
    np_data = [np.random.uniform(low=low, high=high, size=size).astype(_to_np_dtype(dtypes.default_float)) for size in shps]
    ts = [torch.tensor(data, requires_grad=(not forward_only)) for data in np_data]
  for i in range(len(ts)):
    # NOTE: torch default int64 for python ints input
    if ts[i].dtype == torch.int64: ts[i] = ts[i].type(torch.int32)
  tst = [Tensor(x.detach().cpu().numpy(), requires_grad=(not forward_only and not FORWARD_ONLY)) for x in ts]
  return ts, tst

class TestRockchipSupport(unittest.TestCase):
  def test_rkcmd_pack(self):
    self.assertEqual(rkcmd(0x12, 0x3456, 0x89abcdef), ((0x13 & 0xffff) << 48) | ((0x89abcdef & 0xffffffff) << 16) | 0x3456)

  def test_conv_packers(self):
    p = conv_params(3, 4, 5)
    src = memoryview(np.arange(15, dtype=np.float16).tobytes())
    packed_in = pack_conv_input(src, p)
    self.assertEqual(packed_in.dtype, np.float16)
    self.assertEqual(packed_in.size, p["width_stride"] * (p["in_channels"] if p["use_nhwc"] else p["align_c"]))
    packed_wt = pack_conv_weights(memoryview(np.arange(12, dtype=np.float16).tobytes()), p)
    self.assertEqual(packed_wt.dtype, np.float16)
    self.assertEqual(packed_wt.size, p["out_channels"] * p["align_c"])
    out_p = {**p, "out_channels":4}
    packed_out = np.arange(p["out_width_stride"] * 8, dtype=np.float16)
    unpacked = unpack_conv_output(memoryview(packed_out.tobytes()), out_p)
    self.assertEqual(unpacked.dtype, np.float16)
    self.assertEqual(unpacked.size, 4 * 5)

  def test_wmma_params(self):
    p = wmma_params(64, 64, 64)
    self.assertEqual(p["align_in"], 64)
    self.assertEqual(p["align_out"], 64)
    self.assertEqual(p["notch_val"], 0)

  def test_wmma_template_patches(self):
    pkg = build_wmma_template(wmma_params(64, 64, 64), 6, 7, 8)
    self.assertEqual(pkg.family, "wmma")
    self.assertEqual(pkg.op, Ops.WMMA)
    self.assertEqual([p.role for p in pkg.patches], ["input", "weight", "output"])
    self.assertEqual([p.arg_index for p in pkg.patches], [7, 8, 6])
    regcmd = list(pkg.regcmd)
    for patch, value in zip(pkg.patches, [0x1000, 0x2000, 0x3000]):
      patch_regcmd(regcmd, patch, value)
    self.assertIn(0x1000, [(cmd >> 16) & 0xffffffff for cmd in regcmd])
    self.assertIn(0x2000, [(cmd >> 16) & 0xffffffff for cmd in regcmd])
    self.assertIn(0x3000, [(cmd >> 16) & 0xffffffff for cmd in regcmd])

  def test_lut_builder(self):
    lut, index_scale, inv_scale = build_lut(Ops.EXP2, None, 513)
    self.assertEqual(len(lut), 1026)
    self.assertGreater(index_scale, 0)
    self.assertGreater(inv_scale, 0)
    trunc_lut, trunc_scale, trunc_inv_scale = build_lut(Ops.TRUNC, None, 513)
    self.assertEqual(trunc_lut[:4], [0, 1 << 14, 0, 1 << 14])
    self.assertEqual(trunc_scale, 0)
    self.assertIsNone(trunc_inv_scale)
    self.assertTrue(lut_enabled(Ops.EXP2, None))
    self.assertTrue(lut_enabled(Ops.TRUNC, None))
    self.assertTrue(lut_enabled(Ops.CUSTOM, "silu"))
    self.assertFalse(lut_enabled(Ops.ADD, None))

  def test_template_roundtrip(self):
    pkg = RKTemplatePackage(
      version=1, target="rk3588-rknpu2", family="elementwise", regcmd=(rkcmd(1, 2, 0),),
      patches=(RKPatch("regcmd", 0, "dma32", 1, "input"),),
      tasks=(RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=0, regcfg_amount=1),),
    )
    self.assertEqual(decode_template(encode_template(pkg)), pkg)
    with self.assertRaisesRegex(RuntimeError, "unsupported Rockchip template package"):
      decode_template(b"not-a-template")

  def test_template_patch_dma32(self):
    regcmd = [rkcmd(1, 2, 0)]
    patch_regcmd(regcmd, RKPatch("regcmd", 0, "dma32", 0, "input"), 0x12345678)
    self.assertEqual(regcmd[0], rkcmd(1, 2, 0x12345678))

  def test_template_patch_regfield_overflow(self):
    regcmd = [rkcmd(1, 2, 0)]
    patch_regcmd(regcmd, RKPatch("regcmd", 0, "regfield", None, "scalar", shift=4, mask=0xf0), 0xf)
    self.assertEqual(regcmd[0], rkcmd(1, 2, 0xf0))
    with self.assertRaisesRegex(RuntimeError, "overflow"):
      patch_regcmd(regcmd, RKPatch("regcmd", 0, "regfield", None, "scalar", shift=4, mask=0xf0), 0x10)

  def test_template_apply_patches_requires_roles(self):
    regcmd = [rkcmd(1, 2, 0), rkcmd(1, 3, 0)]
    patches = (RKPatch("regcmd", 0, "dma32", None, "input"), RKPatch("regcmd", 1, "dma32", None, "output"))
    apply_patches(regcmd, patches, {"input":0x1000, "output":0x2000})
    self.assertEqual((regcmd[0] >> 16) & 0xffffffff, 0x1000)
    self.assertEqual((regcmd[1] >> 16) & 0xffffffff, 0x2000)
    with self.assertRaisesRegex(RuntimeError, "missing Rockchip patch role output"):
      apply_patches(regcmd, patches, {"input":0x1000})

  def test_template_validate_task_bounds(self):
    pkg = RKTemplatePackage(
      version=1, target="rk3588-rknpu2", family="elementwise", regcmd=(rkcmd(1, 2, 0),),
      tasks=(RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=1, regcfg_amount=1),),
    )
    with self.assertRaisesRegex(RuntimeError, "out of bounds"):
      validate_template(pkg)
    validate_template(RKTemplatePackage(
      version=1, target="rk3588-rknpu2", family="pcchain", regcmd=tuple(range(64)),
      tasks=(RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=32 * 8, regcfg_amount=32),),
    ))

  def test_elementwise_template_patches(self):
    pkg = build_elementwise_template(Ops.ADD, 64, 3, 4, 5)
    self.assertEqual(pkg.family, "elementwise")
    self.assertEqual(pkg.op, Ops.ADD)
    self.assertEqual(pkg.size, 64)
    self.assertEqual([p.role for p in pkg.patches], ["output", "input", "weight"])
    self.assertEqual([p.arg_index for p in pkg.patches], [3, 4, 5])
    regcmd = list(pkg.regcmd)
    for patch, value in zip(pkg.patches, [0x1000, 0x2000, 0x3000]):
      patch_regcmd(regcmd, patch, value)
    self.assertEqual((regcmd[pkg.patches[0].offset] >> 16) & 0xffffffff, 0x1000)
    self.assertEqual((regcmd[pkg.patches[1].offset] >> 16) & 0xffffffff, 0x2000)
    self.assertEqual((regcmd[pkg.patches[2].offset] >> 16) & 0xffffffff, 0x3000)

  def test_elementwise_compile_shape(self):
    from tinygrad.codegen import to_program
    from tinygrad.device import Target
    from tinygrad.runtime.ops_rockchip import RockchipRenderer
    ast = (Tensor.empty(64, dtype=dtypes.half) + Tensor.empty(64, dtype=dtypes.half)).schedule_linear().src[0].src[0]
    pkg = decode_template(to_program(ast, RockchipRenderer(Target("ROCKCHIP"))).src[-1].arg)
    self.assertEqual(pkg.family, "elementwise")
    self.assertEqual(pkg.op, Ops.ADD)
    self.assertEqual(pkg.size, 64)
    self.assertEqual([p.role for p in pkg.patches], ["output", "input", "weight"])

  def test_lut_elementwise_compile_shape(self):
    from tinygrad.codegen import to_program
    from tinygrad.device import Target
    from tinygrad.runtime.ops_rockchip import RockchipRenderer
    ast = Tensor.empty(64, dtype=dtypes.half).exp2().schedule_linear().src[0].src[0]
    pkg = decode_template(to_program(ast, RockchipRenderer(Target("ROCKCHIP"))).src[-1].arg)
    self.assertEqual(pkg.family, "elementwise")
    self.assertEqual(pkg.op, Ops.EXP2)
    self.assertEqual(pkg.size, 64)
    self.assertEqual([p.arg_index for p in pkg.patches], [0, 1, 1])

  def test_fused_matmul_compile_metadata_package(self):
    from tinygrad.codegen import to_program
    from tinygrad.device import Target
    from tinygrad.runtime.ops_rockchip import RockchipRenderer
    ast = Tensor.empty(4, 4).half().matmul(Tensor.empty(4, 4).half()).schedule_linear().src[0].src[0]
    prg = to_program(ast, RockchipRenderer(Target("ROCKCHIP")))
    pkg = decode_template(prg.src[-1].arg)
    self.assertFalse(prg.arg.name.startswith("rkmm_v1_"))
    self.assertEqual(pkg.family, "fused_matmul")
    self.assertEqual((pkg.meta["m"], pkg.meta["n"], pkg.meta["k"]), (4, 4, 4))
    self.assertEqual((pkg.meta["a_slot"], pkg.meta["b_slot"], pkg.meta["c_slot"]), (1, 2, 0))

  def test_unsupported_compile_requires_template(self):
    from tinygrad.codegen import to_program
    from tinygrad.device import Target
    from tinygrad.runtime.ops_rockchip import RockchipRenderer
    ast = Tensor.empty(64, dtype=dtypes.half).reciprocal().schedule_linear().src[0].src[0]
    with self.assertRaisesRegex(RuntimeError, "no RKTemplatePackage match"):
      to_program(ast, RockchipRenderer(Target("ROCKCHIP")))

  def test_runtime_boundary_smoke(self):
    runtime = Path(__file__).parents[1] / "tinygrad" / "runtime" / "ops_rockchip.py"
    old_runtime = runtime.with_name("ops_rockchip_old.py")
    src = runtime.read_text()
    if old_runtime.exists(): self.assertLess(len(src.splitlines()), len(old_runtime.read_text().splitlines()))
    for symbol in ("def _conv_params", "def _pack_conv_input", "def _pack_conv_weights", "def _unpack_conv_output",
                   "def _wmma_params", "def _parse_fused_matmul_name", "def boilerplate"):
      self.assertNotIn(symbol, src)

  def test_drm_discovery_prefers_rknpu(self):
    from tinygrad.runtime.ops_rockchip import _discover_rockchip_drm
    old = os.environ.get("ROCKCHIP_DRM")
    try:
      os.environ.pop("ROCKCHIP_DRM", None)
      with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sysfs, devfs = root / "sys", root / "dev"
        for card, driver in (("card0", "rockchip-drm"), ("card1", "RKNPU")):
          (sysfs / card / "device").mkdir(parents=True)
          (sysfs / card / "device" / "driver").symlink_to(root / driver)
        self.assertEqual(_discover_rockchip_drm(str(sysfs), str(devfs)), str(devfs / "card1"))
    finally:
      if old is None: os.environ.pop("ROCKCHIP_DRM", None)
      else: os.environ["ROCKCHIP_DRM"] = old

  def test_emit_runtime_boilerplate_lut_path(self):
    from tinygrad.runtime.support.rockchip import emit_runtime_boilerplate
    mock = type("MockRockchipProgram", (), {})()
    mock.lut_enable = True
    mock.lut_size = 513
    mock.inv_scale = 1.0
    mock.q = []
    mock.hardware_ops = {}
    mock.reg = rk_field = lambda v, s, m: (v << s) & m
    mock.emit_raw = lambda target, reg, value: mock.q.append(rkcmd(target, reg, value))
    def fill_lut(lut):
      mock.q.append(len(lut))
    mock.fill_lut = fill_lut
    emit_runtime_boilerplate(mock, Ops.TRUNC, 64, None)
    self.assertGreater(len(mock.q), 0)
    self.assertIn(1026, mock.q)

  def test_conv_template_patches(self):
    pkg = build_conv1x1_template(conv_params(3, 4, 5), 6, 7, 8)
    self.assertEqual(pkg.family, "conv1x1")
    self.assertEqual(pkg.meta["in_channels"], 3)
    self.assertEqual([p.role for p in pkg.patches], ["input", "weight", "output"])
    self.assertEqual([p.arg_index for p in pkg.patches], [7, 8, 6])
    self.assertEqual(pkg.patches[1].addend, 16384)
    regcmd = list(pkg.regcmd)
    for patch, value in zip(pkg.patches, [0x1000, 0x2000, 0x3000]):
      patch_regcmd(regcmd, patch, value)
    self.assertEqual((regcmd[18] >> 16) & 0xffffffff, 0x1000)
    self.assertEqual((regcmd[24] >> 16) & 0xffffffff, 0x2000 + 16384)
    self.assertEqual((regcmd[32] >> 16) & 0xffffffff, 0x3000)

  def test_pcchain_multitask_submit_plan(self):
    tasks = tuple(RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff,
                                 regcfg_offset=i * 64, regcfg_amount=16) for i in range(3))
    pkg = RKTemplatePackage(version=1, target="rk3588-rknpu2", family="pcchain", regcmd=tuple(range(48)), tasks=tasks)
    plan = submit_plan(pkg, flags=0x7)
    self.assertEqual(plan.task_number, 3)
    self.assertEqual(plan.core_mask, 1)
    self.assertEqual(plan.subcore_task, ((0, 3), (0, 0), (0, 0), (0, 0), (0, 0)))
    official = submit_plan(pkg, flags=0x5, official=True)
    self.assertEqual(official.task_number, 9)
    self.assertEqual(official.core_mask, 0)
    self.assertEqual(official.subcore_task[:3], ((0, 3), (0, 3), (0, 3)))

  def test_reject_old_pickle(self):
    from tinygrad.runtime.ops_rockchip import RockchipProgram
    with self.assertRaisesRegex(RuntimeError, "missing RKTemplatePackage magic"):
      RockchipProgram(object(), "old_pickle", pickle.dumps([]))

  def test_program_rejects_wrong_template_target(self):
    from tinygrad.runtime.ops_rockchip import RockchipProgram
    pkg = RKTemplatePackage(
      version=1, target="rk9999-rknpu", family="noop", regcmd=(rkcmd(1, 2, 0),),
      tasks=(RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=0, regcfg_amount=1),),
    )
    dev = type("FakeRockchipDevice", (), {"target":"rk3588-rknpu2"})()
    with self.assertRaisesRegex(RuntimeError, "compiled for rk9999-rknpu"):
      RockchipProgram(dev, "wrong_target", encode_template(pkg))

  def test_conv1x1_spatial_tiling(self):
    from tinygrad.runtime.ops_rockchip import RockchipProgram
    prog = object.__new__(RockchipProgram)
    calls = []
    def stub(out_slot, in_slot, weight_slot, in_channels, out_channels, spatial, bufs):
      calls.append(spatial)
      out = np.full((out_channels, spatial), len(calls), dtype=np.float16)
      memoryview(bufs[out_slot]).cast("B")[:out.nbytes] = memoryview(out).cast("B")
    prog._run_conv1x1 = stub
    out = bytearray(3 * 513 * dtypes.float16.itemsize)
    inp = bytearray(2 * 513 * dtypes.float16.itemsize)
    wt = bytearray(3 * 2 * dtypes.float16.itemsize)
    RockchipProgram._run_conv1x1(prog, 0, 1, 2, 2, 3, 513, (out, inp, wt))
    self.assertEqual(calls, [256, 256, 1])
    got = np.frombuffer(out, dtype=np.float16).reshape(3, 513)
    self.assertEqual(float(got[0, 0]), 1.0)
    self.assertEqual(float(got[0, 256]), 2.0)
    self.assertEqual(float(got[0, 512]), 3.0)

  def test_pool2d_reference_runtime(self):
    from tinygrad.runtime.ops_rockchip import RockchipProgram
    from tinygrad.runtime.support.rockchip import RKTemplatePackage, encode_template
    dev = type("FakeRockchipDevice", (), {"target":"rk3588-rknpu2"})()
    pkg = RKTemplatePackage(
      version=1, target="rk3588-rknpu2", family="pool2d", regcmd=(),
      meta={"op":"max", "out_slot":0, "in_slot":1, "in_h":4, "in_w":4, "channels":2},
    )
    prog = RockchipProgram(dev, "pool2d", encode_template(pkg))
    out = bytearray(3 * 3 * 2 * dtypes.float16.itemsize)
    inp = np.arange(4 * 4 * 2, dtype=np.float16)
    prog._run_pool2d((out, inp.tobytes()))
    got = np.frombuffer(out, dtype=np.float16).reshape(3, 3, 2)
    ref = np.max(inp.reshape(4, 4, 2)[:2, :2], axis=(0, 1))
    self.assertTrue(np.allclose(got[0, 0], ref))

class TestRockchipHardware(unittest.TestCase):
  @staticmethod
  def _pcchain_add_segment(input_dma, weight_dma, output_dma, next_dma, elements):
    width = (elements + 7) // 8 - 1
    body = [
      rkcmd(rk.DPU, rk.REG_DPU_S_POINTER, 0x0000000E),
      rkcmd(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG, 0x000001E5),
      rkcmd(rk.DPU, rk.REG_DPU_DATA_FORMAT, 0x48000002),
      rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x00070007),
      rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH, width),
      rkcmd(rk.DPU, rk.REG_DPU_EW_CFG, 0x108202C0),
      rkcmd(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x00010001),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER, 0x0000000E),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, width),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 7),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000008),
      rkcmd(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, output_dma),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, input_dma),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, weight_dma),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x00017849),
    ]
    return body + [
      rkcmd(0x100, rk.REG_PC_BASE_ADDRESS, next_dma & 0xFFFFFFF0) if next_dma else 0,
      rkcmd(0x100, rk.REG_PC_REGISTER_AMOUNTS, len(body)) if next_dma else 0,
      rkcmd(0x40, 0, 0),
      rkcmd(0x80, rk.REG_PC_OPERATION_ENABLE, 0x18),
    ]

  @staticmethod
  def _run_ref_script(args, timeout=30):
    if not (RK3588_REF / args[0]).exists(): raise unittest.SkipTest(f"missing {RK3588_REF / args[0]}")
    env = {**os.environ, "PYTHONPATH":str(RK3588_REF)}
    ret = subprocess.run([sys.executable, *args], cwd=RK3588_REF, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    if ret.returncode != 0: raise AssertionError(f"{' '.join(args)} failed\nstdout:\n{ret.stdout}\nstderr:\n{ret.stderr}")
    return ret.stdout

  @unittest.skipUnless(getenv("ROCKCHIP_HW_TESTS"), "set ROCKCHIP_HW_TESTS=1 to run Rockchip NPU hardware tests")
  def test_pcchain_add_hardware(self):
    out = self._run_ref_script(["experimental/add_pcchain.py", "--segment-elements", "4096"])
    self.assertIn("PASS", out)

  @unittest.skipUnless(getenv("ROCKCHIP_HW_TESTS"), "set ROCKCHIP_HW_TESTS=1 to run Rockchip NPU hardware tests")
  def test_pcchain_submit_template_hardware(self):
    tasks_n, elements, block_qwords = 3, 4096, 32
    dev = Device["ROCKCHIP"]
    task_buf = cmd_buf = input_buf = weight_buf = output_buf = None
    try:
      task_buf = dev._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, "pcchain_tasks")
      cmd_buf = dev._gpu_alloc(tasks_n * block_qwords * ctypes.sizeof(ctypes.c_uint64), 0, "pcchain_cmd")
      input_buf = dev._gpu_alloc(tasks_n * elements * ctypes.sizeof(ctypes.c_uint16), 0, "pcchain_input")
      weight_buf = dev._gpu_alloc(tasks_n * elements * ctypes.sizeof(ctypes.c_uint16), 0, "pcchain_weight")
      output_buf = dev._gpu_alloc(tasks_n * elements * ctypes.sizeof(ctypes.c_uint16), 0, "pcchain_output")
      inp = np.repeat(np.array([3, 7, 13], dtype=np.uint16), elements)
      weight = np.repeat(np.array([5, 11, 17], dtype=np.uint16), elements)
      ctypes.memmove(input_buf.va_addr, inp.ctypes.data, inp.nbytes)
      ctypes.memmove(weight_buf.va_addr, weight.ctypes.data, weight.nbytes)
      ctypes.memset(output_buf.va_addr, 0, output_buf.size)
      q = []
      for task_idx in range(tasks_n):
        next_dma = cmd_buf.meta.dma_addr + (task_idx + 1) * block_qwords * 8 if task_idx + 1 < tasks_n else 0
        seg = self._pcchain_add_segment(input_buf.meta.dma_addr + task_idx * elements * 2, weight_buf.meta.dma_addr + task_idx * elements * 2,
                                        output_buf.meta.dma_addr + task_idx * elements * 2, next_dma, elements)
        q += seg + [0] * (block_qwords - len(seg))
      tasks = tuple(RKTaskTemplate(4, 0x18, 0x300, 0x1ffff, i * block_qwords * 8, 20) for i in range(tasks_n))
      dev.reset_npu()
      submit_template(dev.fd_ctl, RKTemplatePackage(1, "rk3588-rknpu2", "pcchain", tuple(q), tasks=tasks), q, task_buf, cmd_buf, len(q))
      got = np.frombuffer(ctypes.string_at(output_buf.va_addr, output_buf.size), dtype=np.uint16).reshape(tasks_n, elements)
      np.testing.assert_equal(got, np.array([[8], [18], [30]], dtype=np.uint16).repeat(elements, axis=1))
    finally:
      dev._gpu_free_multiple([b for b in [task_buf, cmd_buf, input_buf, weight_buf, output_buf] if b is not None])

  @unittest.skipUnless(getenv("ROCKCHIP_HW_MULTICORE"), "set ROCKCHIP_HW_MULTICORE=1 to run risky Rockchip multicore hardware tests")
  def test_multicore_split3_hardware(self):
    out = self._run_ref_script([
      "experimental/multicore_elementwise.py", "--tile-flat", "--ops", "ADD", "--n", "12288", "--tiles", "3",
      "--execution", "unsafe-split3", "--allow-unsafe-submit", "--timeout", "10000",
    ], timeout=40)
    self.assertIn("PASS", out)

class TestOps(unittest.TestCase):
  @staticmethod
  def _matmul_data(ash, bsh):
    np.random.seed(0)
    return np.random.uniform(-2, 2, size=ash).astype(np.float32), np.random.uniform(-2, 2, size=bsh).astype(np.float32)

  def _matmul_runner(self, out:Tensor):
    from tinygrad.codegen import to_program
    from tinygrad.engine.realize import get_runtime
    ast = out.schedule_linear().src[0].src[0]
    return get_runtime(Device.DEFAULT, to_program(ast, Device[Device.DEFAULT].renderer))

  def _run_fused_case(self, ash, bsh):
    a_np, b_np = self._matmul_data(ash, bsh)
    expected = torch.tensor(a_np).half().matmul(torch.tensor(b_np).half()).cpu().numpy()
    probe = Tensor(a_np).half().matmul(Tensor(b_np).half())
    prg = self._matmul_runner(probe)
    self.assertIsNotNone(getattr(prg, "fused_matmul_meta", None))
    hits_before = prg.fused_matmul_hits
    fallbacks_before = prg.fused_matmul_fallbacks
    out = Tensor(a_np).half().matmul(Tensor(b_np).half())
    got = out.realize().numpy()
    np.testing.assert_allclose(got, expected, atol=5e-3, rtol=5e-3)
    used = prg.fused_matmul_hits == hits_before + 1
    fallback = prg.fused_matmul_fallbacks > fallbacks_before
    self.assertTrue(used or fallback)

  def helper_test_exception(self, shps, torch_fxn, tinygrad_fxn=None, expected=None, forward_only=False, exact=False, vals=None, low=-1.5, high=1.5):
    if DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV": self.skipTest('helper_test_exception fails in CI CUDA')
    ts, tst = prepare_test_op(low, high, shps, vals, forward_only)
    if tinygrad_fxn is None:
      tinygrad_fxn = torch_fxn
    with self.assertRaises(expected) as torch_cm:
      torch_fxn(*ts)
    with self.assertRaises(expected) as tinygrad_cm:
      tinygrad_fxn(*tst)
    if exact: self.assertEqual(str(torch_cm.exception), str(tinygrad_cm.exception))
    if not CI: print("\ntesting %40r   torch/tinygrad exception: %s / %s" % (shps, torch_cm.exception, tinygrad_cm.exception), end="")

  def test_tiny_add(self):
    helper_test_op([(3), (3)], lambda x,y: x+y, Tensor.add, forward_only=True)
  def test_tiny_mul(self):
    helper_test_op([(64), (64)], lambda x,y: x*y, Tensor.mul, forward_only=True)

  def test_add(self):
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y, Tensor.add)
    helper_test_op([(45,68), (45,68)], lambda x,y: x+y)
    helper_test_op([(), ()], lambda x,y: x+y)
  def test_add3(self):
    helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: x+y+z)

  def test_broadcasted_add(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x+y)
    helper_test_op([(45,65), ()], lambda x,y: x+y)

  def test_broadcasted_add_2(self):
    helper_test_op([(45,65), (65,)], lambda x,y: x+y)

  def test_broadcasted_elementwise(self):
    fxns = [(lambda x,y: x+y, lambda x,y: x+y), (lambda x,y: x-y, lambda x,y: x-y), (lambda x,y: x*y, lambda x,y: x*y),
            (lambda x,y: x/y, lambda x,y: x/y), (torch.maximum, lambda x,y: x.maximum(y)), (torch.minimum, lambda x,y: x.minimum(y))]
    for torch_fxn, tinygrad_fxn in fxns:
      for shps in [[(45,65), (45,1)], [(45,65), ()], [(45,65), (65,)]]:
        with self.subTest(torch_fxn=torch_fxn, shps=shps):
          helper_test_op(shps, torch_fxn, tinygrad_fxn, forward_only=True)

  def test_sub(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y, Tensor.sub)
    helper_test_op([(45,65), (45,65)], lambda x,y: x-y)
    helper_test_op([(), ()], lambda x,y: x-y)
  def test_scalar_sub(self):
    helper_test_op([(45,65)], lambda x: x-2)
    helper_test_op([()], lambda x: x-2)
  def test_scalar_rsub(self):
    helper_test_op([(45,65)], lambda x: 2-x)
    helper_test_op([()], lambda x: 2-x)

  def test_mul(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y, Tensor.mul)
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y, Tensor.mul)
    helper_test_op([(64,64), (64,64)], lambda x,y: x*y)
    helper_test_op([(), ()], lambda x,y: x*y)
  
  def test_scalar_mul(self):
    helper_test_op([(45,65)], lambda x: x*2)
    helper_test_op([(45,65)], lambda x: x*-1)
    helper_test_op([(45,65)], lambda x: 255*x)
    helper_test_op([(45,65)], lambda x: 2*x)
    helper_test_op([()], lambda x: x*2)
    helper_test_op([()], lambda x: 2*x)

  def test_div(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y, Tensor.div)
    helper_test_op([(45,65), (45,65)], lambda x,y: x/y)
    helper_test_op([(), ()], lambda x,y: x/y)

  def test_scalar_div(self):
    helper_test_op([(45,65)], lambda x: x/255)
    helper_test_op([(45,65)], lambda x: x/1)
    helper_test_op([(45,65)], lambda x: 1/x)
    helper_test_op([(45,65)], lambda x: x/2)
    helper_test_op([(45,65)], lambda x: 2/x)
    helper_test_op([()], lambda x: x/2)
    helper_test_op([()], lambda x: 2/x)

  def test_div_rounding_mode(self):
    for denominator in [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]:
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])

    for denominator in [-10.0, -5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0, 10.0]:
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])
      helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[5.0, 6, 7, 0, -5, -6, -7], [denominator]])

    for numerator in [110, 111, -110, -111]:
      for denominator in [55, -55]:
        helper_test_op(None, lambda x,y: x.div(y, rounding_mode=None), forward_only=True, vals=[[numerator], [denominator]])
        helper_test_op(None, lambda x,y: x.div(y, rounding_mode="trunc"), forward_only=True, vals=[[numerator], [denominator]])
        helper_test_op(None, lambda x,y: x.div(y, rounding_mode="floor"), forward_only=True, vals=[[numerator], [denominator]])

    self.helper_test_exception(None, lambda x,y: x.div(y, rounding_mode="typo"), forward_only=True, vals=[[5], [0]], expected=RuntimeError)

  def test_div_int(self):
    helper_test_op(None, lambda x,y: x/y, Tensor.div, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x,y: x//y, forward_only=True, vals=[[5, 6, 7],[1, 2, 3]])
    helper_test_op(None, lambda x: x/2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, lambda x: x//2, forward_only=True, vals=[[3, 4, 5]])
    helper_test_op(None, functools.partial(torch.div, rounding_mode="trunc"), Tensor.idiv, forward_only=True,
                   vals=[[-4, 7, 5, 4, -7, 8], [2, -3, 8, -2, 3, 5]])
    if not COMPILE_ONLY:
      x = Tensor(2**64 - 1, dtype=dtypes.uint64).idiv(1)
      np.testing.assert_equal(x.numpy(), 2**64 - 1)

  def test_mod(self):
    a = [-4, 7, 5, 4, -7, 8, -9]
    b = [2, -3, 8, -2, 3, 5, -5]
    for float_a in [True, False]:
      for float_b in [True, False]:
        va = [float(ai) for ai in a] if float_a else a
        vb = [float(bi) for bi in b] if float_b else b
        helper_test_op(None, lambda x,y: x%y, Tensor.mod, forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x,y: x%y, forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x: x%2, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x%3, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x%3.5, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100%x, forward_only=True, vals=[va])
        helper_test_op(None, lambda x: 100.5%x, forward_only=True, vals=[va])

  def test_fmod(self):
    a = [-4, 7, 5, 4, -7, 8, -9]
    b = [2, -3, 8, -2, 3, 5, -5]
    for float_a in [True, False]:
      for float_b in [True, False]:
        va = [float(ai) for ai in a] if float_a else a
        vb = [float(bi) for bi in b] if float_b else b
        helper_test_op(None, lambda x,y: x.fmod(y), forward_only=True, vals=[va, vb])
        helper_test_op(None, lambda x: x.fmod(2), forward_only=True, vals=[va])
        helper_test_op(None, lambda x: x.fmod(3.5), forward_only=True, vals=[va])

  def test_mul_naninf(self):
    helper_test_op([(45,65)], lambda x: x*math.inf)
    helper_test_op([(45,65)], lambda x: x*-math.inf)
    helper_test_op([(45,65)], lambda x: x*math.nan)

  def test_div_naninf(self):
    helper_test_op([(45,65)], lambda x: x/math.inf)
    helper_test_op([(45,65)], lambda x: x/-math.inf)
    helper_test_op([(45,65)], lambda x: x/math.nan)
    helper_test_op([(45,65)], lambda x: math.inf/x)
    helper_test_op([(45,65)], lambda x: (-math.inf)/x)
    helper_test_op([(45,65)], lambda x: math.nan/x)

  def test_pow_full(self):
    helper_test_op([(45,65), (45,65)], lambda x,y: x**y)
    helper_test_op([(45,65), (45,65)], lambda x,y: x.pow(y))

  def test_pow(self):
    helper_test_op([(45,65)], lambda x: x**0)
    helper_test_op([(45,65)], lambda x: x**1)
    helper_test_op([(45,65)], lambda x: x**2)
    helper_test_op([(45,65)], lambda x: x**3)
    helper_test_op([(45,65)], lambda x: x**-2)
    helper_test_op([()], lambda x: x**2)
    helper_test_op([()], lambda x: x**-2)
    helper_test_op([(45,65)], lambda x: x**3, low=-30, high=-27)
    helper_test_op([()], lambda x: x**3, low=-30, high=-27)
    helper_test_op([(45,65)], lambda x: x**0.2, low=-30, high=-27)
    helper_test_op([(45,65)], lambda x: x**1.2, low=-30, high=-27)
    helper_test_op([()], lambda x: x**0.2, low=-30, high=-27)
    helper_test_op([()], lambda x: x**1.2, low=-30, high=-27)
    a, b = Tensor([0.0], requires_grad=True), torch.tensor([0.0], requires_grad=True)
    helper_test_op([], lambda: b**1.1, lambda: a**1.1)

  def test_pow_const(self):
    helper_test_op([(45,65)], lambda x: x**0.0)
    helper_test_op([(45,65)], lambda x: x**1.0)
    helper_test_op([(45,65)], lambda x: x**-1.0)
    helper_test_op([(45,65)], lambda x: x**8.0)
    helper_test_op([(45,65)], lambda x: x**5.5)
    helper_test_op([(45,65)], lambda x: x**-5.5)
    helper_test_op([(45,65)], lambda x: 1.0**x)
    helper_test_op([(45,65)], lambda x: 5.5**x)
    helper_test_op([(45,65)], lambda x: (-5.5)**x)
    helper_test_op([(45,65)], lambda x: 8.0**x)
    helper_test_op([(45,65)], lambda x: x**2.0)
    helper_test_op([(45,65)], lambda x: 2.0**x)
    helper_test_op([()], lambda x: x**2.0)
    helper_test_op([()], lambda x: 2.0**x)
    helper_test_op(None, lambda x: 0**x, vals=[[-2.,-1,0,1,2,3]])
    helper_test_op(None, lambda x: 0.7**x, vals=[[-2.,-1,0,1,2,3]])
    helper_test_op(None, lambda x: (-2)**x, vals=[[-2.,-1,0,1,2,3]])
    helper_test_op(None, lambda x: 0.7**x, vals=[[-2,-1,0,1,2,3]], forward_only=True)

  def test_pow_const_direct(self):
    def get_tiny_gradient(x, c):
      t = Tensor([x], dtype=dtypes.float)
      return (t ** c)[0].gradient(t)[0].item()
    def get_torch_gradient(x, c):
      t = torch.tensor([x], dtype=torch.float, requires_grad=True)
      return torch.autograd.grad(t ** c, t)[0].item()
    for x in [-math.inf, 0, 1, math.inf]:
      for c in [-1, 0, 0.3, 1, 2]:
        tiny_out = get_tiny_gradient(x, c)
        torch_out = get_torch_gradient(x, c)
        if math.isnan(tiny_out):
          if Device.DEFAULT == "WEBGPU": continue
          assert math.isnan(torch_out)
        else:
          self.assertAlmostEqual(tiny_out, torch_out, msg=f"{x}, {c}")

  def test_pow_zero_tensor(self):
    helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.0]])
    if Device.DEFAULT != "WEBGPU":
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [0.3]])
      helper_test_op(None, lambda x,y: x**y, vals=[[0.0], [-0.3]])

  def test_exp2_log2_zero_times_negative(self):
    helper_test_op(None, lambda x,y: (x.log2()*y).exp2(), lambda x,y: (x.log2()*y).exp2(), vals=[[0.0], [-0.7]], forward_only=True)

  def test_pow_zero_const(self):
    helper_test_op(None, lambda x: x**0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**0.0, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-0.3, vals=[[0.0]])
    helper_test_op(None, lambda x: x**-1.0, vals=[[-1.0, 0.0, 1.0]])

  def test_int_pow_const_int(self):
    helper_test_op(None, lambda x: x**0, vals=[[-2,0,2]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**1, vals=[[-2,0,2]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**2, vals=[[-2,0,2]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**7, vals=[[11,12,13]], forward_only=True, atol=0)
    helper_test_op(None, lambda x: x**29, vals=[[-2,0,2]], forward_only=True, atol=0)
    self.helper_test_exception(None, lambda x: x**-2, vals=[[-2,0,2]], forward_only=True, expected=RuntimeError)

  def test_pow_int(self):
    def _test(base, exponent): helper_test_op(None, lambda x,y: x**y, vals=[base, exponent], forward_only=True)
    for base in ([1, 2, 3], [-1, -2, -3]):
      for exponent in ([2, 3, 4], [-2, -3, -4]):
        _test(base, exponent)
    _test([0, 0, 0], [0, 1, 2])

  def test_xor(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor^tor, lambda: ten^ten, forward_only=True)
    helper_test_op([], lambda: tor^0x1337, lambda: ten^0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337^tor, lambda: 0x1337^ten, forward_only=True)
    self.helper_test_exception([(4), (4)], lambda x,y: x.bitwise_xor(y), expected=RuntimeError)

  def test_and(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor&tor, lambda: ten&ten, forward_only=True)
    helper_test_op([], lambda: tor&0x1337, lambda: ten&0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337&tor, lambda: 0x1337&ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0&tor1, lambda: ten0&ten1, forward_only=True)
    helper_test_op(None, lambda x: (1 < x) & (x < 2), forward_only=True, vals=[[1.2, 1.2, 1.2, 3.2]])
    self.helper_test_exception([(4), (4)], lambda x,y: x.bitwise_and(y), expected=RuntimeError)

  def test_or(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor|tor, lambda: ten|ten, forward_only=True)
    helper_test_op([], lambda: tor|0x1337, lambda: ten|0x1337, forward_only=True)
    helper_test_op([], lambda: 0x1337|tor, lambda: 0x1337|ten, forward_only=True)

    data = [[True, True, False, False], [True, False, True, False]]
    tor0, tor1 = torch.tensor(data[0], dtype=torch.bool),  torch.tensor(data[1], dtype=torch.bool)
    ten0, ten1 = Tensor(data[0], dtype=dtypes.bool), Tensor(data[1], dtype=dtypes.bool)
    helper_test_op([], lambda: tor0|tor1, lambda: ten0|ten1, forward_only=True)
    self.helper_test_exception([(4), (4)], lambda x,y: x.bitwise_or(y), expected=RuntimeError)

  def test_bitwise_not(self):
    data = [[1,-8,1],[32,1,6]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor.bitwise_not(), lambda: ten.bitwise_not(), forward_only=True)
    helper_test_op([], lambda: ~tor, lambda: ~ten, forward_only=True)

    data = [[True, False]]
    tor = torch.tensor(data, dtype=torch.bool)
    ten = Tensor(data, dtype=dtypes.bool)
    helper_test_op([], lambda: tor.bitwise_not(), lambda: ten.bitwise_not(), forward_only=True)
    helper_test_op([], lambda: ~tor, lambda: ~ten, forward_only=True)

    self.helper_test_exception([(4)], lambda x: x.bitwise_not(), expected=RuntimeError)

  def test_lshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    helper_test_op([], lambda: tor << 0, lambda: (ten << 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 2, lambda: (ten << 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor << 31, lambda: (ten << 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.__lshift__(2), lambda: ten.__lshift__(2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_left_shift(2), lambda: ten.lshift(2).cast(dtypes.int32), forward_only=True)

  def test_rshift(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.uint32)
    helper_test_op([], lambda: tor >> 0, lambda: (ten >> 0).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 2, lambda: (ten >> 2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor >> 31, lambda: (ten >> 31).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.__rshift__(2), lambda: ten.__rshift__(2).cast(dtypes.int32), forward_only=True)
    helper_test_op([], lambda: tor.bitwise_right_shift(2), lambda: ten.rshift(2).cast(dtypes.int32), forward_only=True)

  def test_lshift_signed(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor << 0, lambda: ten << 0, forward_only=True)
    helper_test_op([], lambda: tor << 2, lambda: ten << 2, forward_only=True)
    helper_test_op([], lambda: tor << 31, lambda: ten << 31, forward_only=True)

  def test_rshift_signed(self):
    data = [[0,1,2],[1<<8,1<<16,1<<31-1]]
    tor = torch.tensor(data, dtype=torch.int)
    ten = Tensor(data, dtype=dtypes.int32)
    helper_test_op([], lambda: tor >> 0, lambda: ten >> 0, forward_only=True)
    helper_test_op([], lambda: tor >> 2, lambda: ten >> 2, forward_only=True)
    helper_test_op([], lambda: tor >> 31, lambda: ten >> 31, forward_only=True)

  def test_idiv_shift_rewrite_negative(self):
    helper_test_op(None, lambda x: x//2, lambda x: x//2, vals=[[-1, -2, -3, -4, -5]], forward_only=True)

  def test_neg(self):
    helper_test_op([(45,65)], lambda x: -x)
    helper_test_op([(45,65)], lambda x: x.neg())
    helper_test_op([()], lambda x: x.neg())

  def test_maximum(self):
    helper_test_op([(45,65), (45,65)], torch.maximum, Tensor.maximum)
    helper_test_op([(), ()], torch.maximum, Tensor.maximum)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., -4.], 3.])
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    helper_test_op(None, torch.maximum, Tensor.maximum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.max], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.min], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.maximum, Tensor.maximum, vals=[[True, False, False], 3], forward_only=True)

  def test_minimum(self):
    helper_test_op([(45,65), (45,65)], torch.minimum, Tensor.minimum)
    helper_test_op([(), ()], torch.minimum, Tensor.minimum)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], 3.])
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1., 0., 3., -4.], [-1., -2., 3., 0.]])
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.max], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum,
                   vals=[[-1234, 0, 1234, dtypes.int.max, dtypes.int.min], dtypes.int.min], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], True], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], [True, True, False]], forward_only=True)

    # test applying to different dtype
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[1, 2, 3], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 1.2], forward_only=True)
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], 3], forward_only=True)

  # failed Not equal to tolerance rtol=1e-3, atol=1e-06
  def test_exp2(self):
    helper_test_op([(45,65)], torch.exp2, Tensor.exp2, rtol=1e-3, atol=5e-3)
    # helper_test_op(None, torch.exp2, Tensor.exp2, vals=[[math.inf, -math.inf, math.nan]])
    helper_test_op([()], torch.exp2, Tensor.exp2, atol=8e-3, rtol=1e-2)
  # TODO test silu(-x)=silu(x)-x
  def test_silu(self):
    helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.silu, rtol=1e-3, atol=5e-3)
    helper_test_op([()], torch.nn.functional.silu, Tensor.silu)

  def test_relu(self):
    helper_test_op([(64,64)], lambda x: x.relu())
    helper_test_op([()], lambda x: x.relu())
  def test_relu_exact(self):
    helper_test_op(None, lambda x: x.relu(), vals=[[-1.,0,1]])
  def test_relu_maximum_exact(self):
    helper_test_op(None, lambda x: torch.maximum(x, torch.zeros_like(x, requires_grad=False)), lambda x: Tensor.maximum(x, 0), vals=[[-1.,0,1]])
  def test_relu6(self):
    helper_test_op([(45,65)], torch.nn.functional.relu6, Tensor.relu6, high=100)
    helper_test_op([()], torch.nn.functional.relu6, Tensor.relu6)

  def test_arange(self):
    helper_test_op([], lambda: torch.arange(10, dtype=torch.int32), lambda: Tensor.arange(10), forward_only=True)
    helper_test_op([], lambda: torch.arange(36, dtype=torch.int32), lambda: Tensor.arange(36), forward_only=True)
    helper_test_op([], lambda: torch.arange(5, 10, 3, dtype=torch.int32), lambda: Tensor.arange(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.arange(10, 5, -3, dtype=torch.int32), lambda: Tensor.arange(10, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(11, 5, -3, dtype=torch.int32), lambda: Tensor.arange(11, 5, -3), forward_only=True)
    helper_test_op([], lambda: torch.arange(1, 78, 2, dtype=torch.int32), lambda: Tensor.arange(1, 78, 2), forward_only=True)
    helper_test_op([], lambda: torch.arange(5.5, 175.5, 2.5), lambda: Tensor.arange(5.5, 175.5, 2.5), forward_only=True)
    helper_test_op([], lambda: torch.arange(-30.2, -0.3, 0.75), lambda: Tensor.arange(-30.2, -0.3, 0.75), forward_only=True)
    helper_test_op([], lambda: torch.arange(-50.3, -380.2, -2.25), lambda: Tensor.arange(-50.3, -380.2, -2.25), forward_only=True)
    helper_test_op([], lambda: torch.arange(128, dtype=torch.int8), lambda: Tensor.arange(128, dtype=dtypes.int8), forward_only=True)
    helper_test_op([], lambda: torch.arange(-128, 128, dtype=torch.int8),
                   lambda: Tensor.arange(-128, 128, dtype=dtypes.int8), forward_only=True)
    helper_test_op([], lambda: torch.arange(127, -129, -1, dtype=torch.int8),
                   lambda: Tensor.arange(127, -129, -1, dtype=dtypes.int8), forward_only=True)

  def test_arange_big(self):
    helper_test_op([], lambda: torch.arange(256, dtype=torch.int32), lambda: Tensor.arange(256), forward_only=True)

  def test_arange_4096(self):
    helper_test_op([], lambda: torch.arange(4096, dtype=torch.int32), lambda: Tensor.arange(4096), forward_only=True)

  def test_linspace(self):
    helper_test_op([], lambda: torch.linspace(5, 10, 3), lambda: Tensor.linspace(5, 10, 3), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 1), lambda: Tensor.linspace(5, 10, 1), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 0), lambda: Tensor.linspace(5, 10, 0), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 30), lambda: Tensor.linspace(5, 10, 30), forward_only=True)
    helper_test_op([], lambda: torch.linspace(-5.5, 5.5, 10), lambda: Tensor.linspace(-5.5, 5.5, 10), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5.5, -5.5, 10), lambda: Tensor.linspace(5.5, -5.5, 10), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 3, dtype=torch.int32),
                   lambda: Tensor.linspace(5, 10, 3, dtype="int32"), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, 10, 20, dtype=torch.int32),
                   lambda: Tensor.linspace(5, 10, 20, dtype="int32"), forward_only=True)
    helper_test_op([], lambda: torch.linspace(5, -5, 20, dtype=torch.int32),
                   lambda: Tensor.linspace(5, -5, 20, dtype="int32"), forward_only=True)
    self.helper_test_exception([], lambda: torch.linspace(5, 10, 3, dtype=torch.bool),
                               lambda: Tensor.linspace(5, 10, 3, dtype="bool"), expected=(RuntimeError, ValueError))
    self.helper_test_exception([], lambda: torch.linspace(1, 2, -1), lambda: Tensor.linspace(1, 2, -1), expected=(RuntimeError, ValueError))

  def test_logical_not(self):
    helper_test_op(None, torch.logical_not, Tensor.logical_not, vals=[[True, False, True]], forward_only=True)
    helper_test_op(None, torch.logical_not, Tensor.logical_not, vals=[[1., 2., 0., 0.5]], forward_only=True)

  def test_isinf(self):
    val = [float('-inf'), 0., float('inf'), float('nan'), 1.1]
    helper_test_op(None, torch.isinf, Tensor.isinf, vals=[val], forward_only=True)
    if not COMPILE_ONLY:
      np.testing.assert_equal(Tensor(val).isinf(detect_positive=True, detect_negative=False).numpy(), [False, False, True, False, False])
      np.testing.assert_equal(Tensor(val).isinf(detect_positive=False, detect_negative=True).numpy(), [True, False, False, False, False])

  def test_isnan(self):
    helper_test_op(None, torch.isnan, Tensor.isnan, vals=[[float('-inf'), 0., float('inf'), float('nan'), 1.1]], forward_only=True)

  def test_isfinite(self):
    helper_test_op(None, torch.isfinite, Tensor.isfinite, vals=[[float('-inf'), 0., float('inf'), float('nan'), 1.1]], forward_only=True)

  def test_lerp(self):
    helper_test_op([(45,35), (45,35), (45,35)], lambda x,y,z: x.lerp(y,z))
    helper_test_op(None, lambda x,y,z: x.lerp(y,z), vals=[[1.,2.,3.], [4.,5.,6.], 0.5])

  def test_tril(self):
    helper_test_op([(3,3)], lambda x: x.tril())
    helper_test_op([(3,3)], lambda x: x.tril(1))
    helper_test_op([(3,3)], lambda x: x.tril(2))
    helper_test_op([(3,3)], lambda x: x.tril(-1))
    helper_test_op([(3,3)], lambda x: x.tril(-2))
    helper_test_op([(4,5)], lambda x: x.tril(4))
    helper_test_op([(4,5)], lambda x: x.tril(5))
    helper_test_op([(4,5)], lambda x: x.tril(6))
    helper_test_op([(4,5)], lambda x: x.tril(-4))
    helper_test_op([(4,5)], lambda x: x.tril(-5))
    helper_test_op([(4,5)], lambda x: x.tril(-6))
    helper_test_op([(5,3,3)], lambda x: x.tril())
    helper_test_op([(5,0,3)], lambda x: x.tril())
    helper_test_op([(5,3,3)], lambda x: x.tril(1))
    helper_test_op(None, lambda x: x.tril(), vals=[[[True] * 3] * 3], forward_only=True)

  def test_triu(self):
    helper_test_op([(3,3)], lambda x: x.triu())
    helper_test_op([(3,3)], lambda x: x.triu(1))
    helper_test_op([(3,3)], lambda x: x.triu(2))
    helper_test_op([(3,3)], lambda x: x.triu(-1))
    helper_test_op([(3,3)], lambda x: x.triu(-2))
    helper_test_op([(4,5)], lambda x: x.triu(4))
    helper_test_op([(4,5)], lambda x: x.triu(5))
    helper_test_op([(4,5)], lambda x: x.triu(6))
    helper_test_op([(4,5)], lambda x: x.triu(-4))
    helper_test_op([(4,5)], lambda x: x.triu(-5))
    helper_test_op([(4,5)], lambda x: x.triu(-6))
    helper_test_op([(5,3,3)], lambda x: x.triu())
    helper_test_op([(5,0,3)], lambda x: x.triu())
    helper_test_op([(5,3,3)], lambda x: x.triu(1))
    helper_test_op(None, lambda x: x.triu(), vals=[[[True] * 3] * 3], forward_only=True)

  def test_sum_fake(self):
    helper_test_op([(256, 1)], lambda x: x.sum(axis=1))

  def test_sum_collapse(self):
    helper_test_op([], lambda: torch.ones(256,256).sum(axis=1), lambda: Tensor.ones(256,256).sum(axis=1), forward_only=True)

  def test_sum_collapse_neg(self):
    helper_test_op([], lambda: (-torch.ones(3,3)).sum(axis=1), lambda: (-Tensor.ones(3,3)).sum(axis=1), forward_only=True)

  def test_sum_pad_collapse(self):
    helper_test_op([], lambda: torch.nn.functional.pad(torch.ones(256,256), pad=(0,64,0,0)).sum(axis=1),
                       lambda: Tensor.ones(256,256).pad(((0,0), (0,64))).sum(axis=1), forward_only=True)

  def test_sum_twice(self):
    helper_test_op([(4, 4, 4)], lambda x: x.sum((0, 1)).sum())

  def test_sum_cat_collapse(self):
    helper_test_op([], lambda: torch.cat((torch.ones(256,256), torch.zeros(256,64)), dim=1).sum(axis=1),
                       lambda: Tensor.cat(Tensor.ones(256,256), Tensor.zeros(256,64), dim=1).sum(axis=1), forward_only=True)

  def test_max_dont_collapse(self):
    helper_test_op([], lambda: torch.ones(256,256).max(1)[0], lambda: Tensor.ones(256,256).max(1), forward_only=True)

  def test_topo_sort(self):
    helper_test_op([(45,65)], lambda x: (x+x)*x, grad_atol=1e-6)
    helper_test_op([()], lambda x: (x+x)*x, grad_atol=1e-6)

  def test_flip_eye_crash(self):
    helper_test_op([], lambda: (torch.eye(10)@torch.eye(10).flip(0)),
                       lambda: (Tensor.eye(10)@Tensor.eye(10).flip(0)), forward_only=True)

  def test_broadcast_full(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((5,3,14,16), (5,1,14,1)), ((1,3,1,7,1), (2,1,5,1,8))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          if tinygrad_op != Tensor.pow:
            helper_test_op(shapes, torch_op, tinygrad_op)
          else:
            helper_test_op(shapes, torch_op, tinygrad_op, low=0, high=3)

  def test_broadcast_simple(self):
    helper_test_op([(45,65), (45,1)], lambda x,y: x/y)
    helper_test_op([(45,65), ()], lambda x,y: x/y)

  @slow_test
  def test_broadcast_partial(self):
    for torch_op, tinygrad_op in [(torch.add, Tensor.add), (torch.sub, Tensor.sub), (torch.mul, Tensor.mul),
                                  (torch.div, Tensor.div), (torch.pow, Tensor.pow)]:
      for shapes in [((1,32,32,32), (1,32,1,1)), ((5,13,24,16,2), (1,13,24,1,1)),
                     ((4,1), (4,5)), ((1,4), (5,4))]:
        with self.subTest(op=torch_op.__name__, shapes=shapes):
          if tinygrad_op != Tensor.pow:
            helper_test_op(shapes, torch_op, tinygrad_op)
          else:
            helper_test_op(shapes, torch_op, tinygrad_op, low=0, high=3)

  # failed special case math.inf
  def _test_cmp(self, fxn, reverse=True):
    # test different dtypes
    helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    # helper_test_op(None, fxn, fxn, forward_only=True, vals=[[0,1,2], [2,1,0]])
    # helper_test_op(None, fxn, fxn, forward_only=True, vals=[[True, True, False], [False,True,False]])
    
    ### RK3588 failed
    # test broadcasting
    # for shps in [[(3, 4, 5), (3, 4, 5)], [(3, 4, 5), (5,)], [(5,), (3, 4, 5)]]:
      # helper_test_op(shps, fxn, fxn, forward_only=True)
    
    # test cmp with const
    # helper_test_op(None, lambda x,y: fxn(x,2), lambda x,y: fxn(x,2), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    # if reverse: helper_test_op(None, lambda x,y: fxn(2,y), lambda x,y: fxn(2,y), forward_only=True, vals=[[0.,1,2], [2.,1,0]])
    
    ### RK3588 failed
    # test special floats  # TODO: fix nan
    # specials = [0.0, 1.0, -1.0, math.inf, -math.inf]#, math.nan]
    # for s0 in specials:
      # for s1 in specials:
        # helper_test_op(None, fxn, fxn, forward_only=True, vals=[[s0], [s1]])
  def test_cmp_lt(self): self._test_cmp(lambda x,y: x<y)
  def test_cmp_gt(self): self._test_cmp(lambda x,y: x>y)
  def test_cmp_eq(self): self._test_cmp(lambda x,y: x==y, reverse=False)
  def test_cmp_ge(self): self._test_cmp(lambda x,y: x>=y)
  def test_cmp_le(self): self._test_cmp(lambda x,y: x<=y)
  def test_cmp_ne(self):
    helper_test_op(None, torch.minimum, Tensor.minimum, vals=[[True, False, False], [True, True, False]], forward_only=True)

  def test_cmp_ne_backwards(self):
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt != 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t != 0)).sum().backward()
    np.testing.assert_allclose(t.grad.cpu().numpy(), tt.grad.numpy(), rtol=1e-5)

  def test_cmp_lt_backwards(self):
    tt = Tensor.randn(4, requires_grad=True)
    (tt*(tt < 0)).sum().backward()
    t = torch.tensor(tt.numpy(), requires_grad=True)
    (t*(t < 0)).sum().backward()
    np.testing.assert_allclose(t.grad.cpu().numpy(), tt.grad.numpy(), rtol=1e-5)

  def test_where(self):
    helper_test_op(
      [(100,)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32),
      lambda x: (x > 0.5).where(4, 2), forward_only=True)

    for shps in [[(8,),(1,),(1,)], [(10,10),(10,),(10,)], [(100,)]*3, [(10,10)]*3]:
      helper_test_op(
        shps,
        lambda x, a, b: torch.where(x > 0.5, a, b),
        lambda x, a, b: (x > 0.5).where(a, b), forward_only=True)

  def test_where_permute(self):
    helper_test_op(
      [(5, 5)],
      lambda x: torch.where(x > 0.5, 4, 2).type(torch.int32).permute((1, 0)),
      lambda x: (x > 0.5).where(4, 2).permute((1, 0)), forward_only=True)
  def test_trunc(self):
    # helper_test_op([()], lambda x: x.trunc(), forward_only=True)
    # helper_test_op([(45,35)], lambda x: x.trunc(), forward_only=True)
    helper_test_op(None, lambda x: x.half().trunc(), lambda x: x.cast(dtypes.float16).trunc(),
                   vals=[[1.499, 1.5, 1.501, 1.0, 2.1, 0.0, -5.0, -2.499, -2.5, -2.501]],
                   forward_only=True)

  def test_pad(self):
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)), lambda x: x.pad((1,2,3,4)))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4)), lambda x: x.pad(((3,4),(1,2))))
    helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4)), lambda x: x.pad(((-3,4),(-1,2))))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (1,2,3,4), value=5),
                   lambda x: x.pad((1,2,3,4), value=5))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,-3,4), value=5),
                   lambda x: x.pad((-1,2,-3,4), value=5))

  def test_pad_replicate_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="replicate"), lambda x: x.pad((0,2,3,2), mode="replicate"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="replicate"), lambda x: x.pad((0,2), mode="replicate"))
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,4,1,2),mode="replicate"),lambda x:x.pad((1,2,3,4,1,2),mode="replicate"))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,2,-1), mode="replicate"), lambda x: x.pad((-1,2,2,-1), mode="replicate"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-3,0,-3), mode="replicate"), lambda x: x.pad((3,-3,0,-3), mode="replicate"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-5,1,-5), mode="replicate"), lambda x: x.pad((3,-5,1,-5), mode="replicate"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,0,0,-5), mode="replicate"), lambda x: x.pad((0,0,0,-5), mode="replicate"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,11,0,30), mode="replicate"), lambda x: x.pad((3,11,0,30), mode="replicate"))

  def test_pad_reflect_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="reflect"), lambda x: x.pad((0,2,3,2), mode="reflect"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="reflect"), lambda x: x.pad((0,2), mode="reflect"))
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,4,1,2), mode="reflect"),
                                  lambda x: x.pad((1,2,3,4,1,2), mode="reflect"))
    helper_test_op([(3,3,3,3)], lambda x: torch.nn.functional.pad(x, (-1,2,2,-1), mode="reflect"), lambda x: x.pad((-1,2,2,-1), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-3,0,-3), mode="reflect"), lambda x: x.pad((3,-3,0,-3), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (3,-5,1,-5), mode="reflect"), lambda x: x.pad((3,-5,1,-5), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,0,0,-5), mode="reflect"), lambda x: x.pad((0,0,0,-5), mode="reflect"))
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (4,4,0,4), mode="reflect"), lambda x:x.pad((4,4,0,4),mode="reflect"))
    self.helper_test_exception([(1,1,5,5)],
                                lambda x: torch.nn.functional.pad(x, (3,5,0,0),mode="reflect"), lambda x: x.pad((3,5,0,0),mode="reflect"),
                                expected=(RuntimeError, ValueError))

  def test_pad_circular_mode(self):
    helper_test_op([(1,1,5,5)], lambda x: torch.nn.functional.pad(x, (0,2,3,2), mode="circular"), lambda x: x.pad((0,2,3,2), mode="circular"))
    helper_test_op([(5,5,5)], lambda x: torch.nn.functional.pad(x, (0,2), mode="circular"), lambda x: x.pad((0,2), mode="circular"))
    helper_test_op([(1,1,5,5,5)], lambda x: torch.nn.functional.pad(x, (1,2,3,5,1,2),mode="circular"),lambda x:x.pad((1,2,3,5,1,2),mode="circular"))
    self.helper_test_exception([(1,1,5,5)],
                                lambda x: torch.nn.functional.pad(x, (3,6,0,0), mode="circular"), lambda x: x.pad((3,6,0,0), mode="circular"),
                                expected=(RuntimeError, ValueError))
    with self.assertRaises(NotImplementedError):
      Tensor.randn(1,1,5,5).pad((3,-5,1,-5), mode="circular")

  def test_pad_reshape(self):
    helper_test_op([(1, 2)],
                   lambda x: torch.nn.functional.pad(x, (0, 1, 1, 0)).reshape((3, 2)),
                   lambda x: x.pad((0, 1, 1, 0)).reshape((3, 2)))
    helper_test_op([(1, 2)],
                   lambda x: torch.nn.functional.pad(x, (0, 2, 1, 1)).reshape((4, 3)),
                   lambda x: x.pad((0, 2, 1, 1)).reshape((4, 3)))
    helper_test_op([(1, 1, 1, 2)],
                   lambda x: torch.nn.functional.pad(x, (0, 4, 2, 2, 1, 2, 0, 2)).reshape((4, 3, 6, 5)),
                   lambda x: x.pad(((0, 2), (1, 2), (2, 2), (0, 4))).reshape((4, 3, 6, 5)))

  def test_pad_slice(self):
    for value in 0., 3.456:
      helper_test_op([(1)], lambda x: torch.nn.functional.pad(x,(1,0), value=value)[0], lambda x: x.pad(((1,0),), value=value)[0])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(1,0), value=value)[0], lambda x: x.pad(((1,0),), value=value)[0])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(3,0), value=value)[0:1], lambda x: x.pad(((3,0),), value=value)[0:1])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(0,3), value=value)[6], lambda x: x.pad(((0,3),), value=value)[6])
      helper_test_op([(4)], lambda x: torch.nn.functional.pad(x,(0,3), value=value)[4:6], lambda x: x.pad(((0,3),), value=value)[4:6])
      helper_test_op([(5,5)], lambda x: torch.nn.functional.pad(x,(0,0,1,0), value=value)[0], lambda x: x.pad(((1,0),(0,0)), value=value)[0])
      helper_test_op([(2,2)], lambda x: torch.nn.functional.pad(x,(0,1,0,0), value=value)[0,2], lambda x: x.pad(((0,0),(0,1)), value=value)[0,2])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(0,0,1,0), value=value)[0,2], lambda x: x.pad(((1,0),(0,0)), value=value)[0,2])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(0,0,0,2), value=value)[5], lambda x: x.pad(((0,2),(0,0)), value=value)[5])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(0,0,0,2), value=value)[3:5], lambda x: x.pad(((0,2),(0,0)), value=value)[3:5])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,0,0,0), value=value)[1,0], lambda x: x.pad(((0,0),(3,0)), value=value)[1,0])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,0,0,0), value=value)[1,0:4], lambda x: x.pad(((0,0),(3,0)), value=value)[1,0:4])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,4,1,2), value=value)[0], lambda x: x.pad(((1,2),(3,4)), value=value)[0])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,4,1,2), value=value)[:,1], lambda x: x.pad(((1,2),(3,4)), value=value)[:,1])
      helper_test_op([(4,4)], lambda x: torch.nn.functional.pad(x,(3,4,1,2), value=value)[:,4], lambda x: x.pad(((1,2),(3,4)), value=value)[:,4])
      helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x,(0,3,0,0), value=value)[:,4:6], lambda x: x.pad(((0,0),(0,3)), value=value)[:,4:6])
      helper_test_op([(3,3)], lambda x: torch.nn.functional.pad(x,(0,1,3,2), value=value)[0:2,:], lambda x: x.pad(((3,2),(0,1)), value=value)[0:2,:])
      helper_test_op([(3,3,3)], lambda x: torch.nn.functional.pad(x,(1,1,0,1,3,2), value=value)[0:2,:,:],
                                lambda x: x.pad(((3,2),(0,1),(1,1)), value=value)[0:2,:,:])
      helper_test_op([(3,3,3)], lambda x: torch.nn.functional.pad(x,(1,1,0,1,3,2), value=value)[2:4,:,:],
                                lambda x: x.pad(((3,2),(0,1),(1,1)), value=value)[2:4,:,:])

  def test_padding_add(self):
    helper_test_op([(64,64), (60,60)],
      lambda x,w: x+torch.nn.functional.pad(w, (2,2,2,2)),
      lambda x,w: x+w.pad((2,2,2,2)))

  def test_permute(self):
    helper_test_op([(1,2,3,4)], lambda x: x.permute((3,0,2,1)))
    helper_test_op([(3,4,5,6)], lambda x: x.permute((3,2,1,0)))
    helper_test_op([(3,4,5,6)], lambda x: x.permute((-2,-1,1,0)))
    helper_test_op([()], lambda x: x.permute(()))
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,2)), expected=RuntimeError)
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,1,2,3,3,3)), expected=RuntimeError)
    self.helper_test_exception([(3,4,5,6)], lambda x: x.permute((0,0,1,2,3)), expected=RuntimeError)

  def test_reshape(self):
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((12,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((-1,3,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((-1,1,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.reshape((4,3,6,6)), lambda x: x.reshape((None,None,6,6)))
    helper_test_op([()], lambda x: x.reshape(()))
    helper_test_op([(1,)], lambda x: x.reshape(()))
    helper_test_op([()], lambda x: x.reshape((1,)))
    helper_test_op([()], lambda x: x.reshape((1,1,1)))
    self.helper_test_exception([(3,4)], lambda x: x.reshape((-1,-1,2)), expected=RuntimeError)
    self.helper_test_exception([(3,4)], lambda x: x.reshape((-1,-1,-1,2)), expected=RuntimeError)

  def test_flip(self):
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,1)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,1,3)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((3,)))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((0,1,3)).flip(0))
    helper_test_op([(4,3,6,6)], lambda x: x.flip((-1,)))
    helper_test_op([()], lambda x: x.flip(()))
    helper_test_op([(1,)], lambda x: x.flip(()))
    helper_test_op([(4,3,6,6)], lambda x: x.flip(()))

  def test_slice_in_bounds_1dim(self):
    helper_test_op([(3)], lambda x: x[1:3])
    helper_test_op([(3)], lambda x: x[0:2])
    helper_test_op([(3)], lambda x: x[-2:2])

  def test_slice_on_0dim_tensor(self):
    helper_test_op([()], lambda x: x[None])
    with self.assertRaises(IndexError):
      a = Tensor(3.14)
      a[0]

  def test_slice_int_indexing(self):
    helper_test_op([(3)], lambda x: x[0])
    helper_test_op([(3)], lambda x: x[2])
    helper_test_op([(3)], lambda x: x[-1])
    helper_test_op([(3)], lambda x: x[-3])
    helper_test_op([(10,10)], lambda x: x[1])
    helper_test_op([(3,3,3)], lambda x: x[1,1,1])

  def test_slice_in_bounds_multidim(self):
    helper_test_op([(3,3,3)], lambda x: x[1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2, 0:-1])

  def test_slice_with_none(self):
    helper_test_op([(3,3,3)], lambda x: x[None])
    helper_test_op([(3,3,3)], lambda x: x[1:2, None])
    helper_test_op([(3,3,3)], lambda x: x[1:2, None, 1:2])
    helper_test_op([(3,3,3)], lambda x: x[1:2, 1:2, None, -1])
    helper_test_op([(3,3,3)], lambda x: x[None, None, 1, None, 2, 0:2])

  def test_slice_one_endpoint_out_of_bounds(self):
    helper_test_op([(3,3,3)], lambda x: x[0:4])
    helper_test_op([(3,3,3)], lambda x: x[-6:4])
    helper_test_op([(3,3,3)], lambda x: x[1:50])
    helper_test_op([(3,3,3)], lambda x: x[1:50, 1:2, -1])

  def test_slice_stride_gt_one(self):
    helper_test_op([(7,5,10)], lambda x: x[::2, ::3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, ::3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, 3, ::4])
    helper_test_op([(7,5,10)], lambda x: x[1:5:2, None, None, 3, None, ::4])

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  def test_slice_negative_strides(self):
    a = np.random.randn(10, 10, 10).astype(np.float32)
    t = Tensor(a)
    np.testing.assert_allclose(a[::-1], t[::-1].numpy())
    np.testing.assert_allclose(a[::-2], t[::-2].numpy())
    np.testing.assert_allclose(a[:, 2:0:-1], t[:, 2:0:-1].numpy())
    np.testing.assert_allclose(a[:, 2:0:-1, 3:1:-2], t[:, 2:0:-1, 3:1:-2].numpy())
    np.testing.assert_allclose(a[4:0:-3, 2:0:-1, -1:-5:-2], t[4:0:-3, 2:0:-1, -1:-5:-2].numpy())
    np.testing.assert_allclose(a[2:5:-1, :, :], t[2:5:-1, :, :].numpy())
    np.testing.assert_allclose(a[:, 2:5:-1, :], t[:, 2:5:-1, :].numpy())
    np.testing.assert_allclose(a[:, :, 2:5:-1], t[:, :, 2:5:-1].numpy())

  def test_slice_both_endpoints_out_of_bounds(self):
    helper_test_op([(3,3,3)], lambda x: x[5:10])
    helper_test_op([(3,3,3)], lambda x: x[-15:-7])

  def test_slice_start_gt_end(self):
    helper_test_op([(3,3,3)], lambda x: x[-2:2])
    helper_test_op([(3,3,3)], lambda x: x[-2:-5])

  def test_slice_errors(self):
    a = Tensor.ones(4, 3)
    b = Tensor(2)
    with self.assertRaisesRegex(IndexError, "too many"): a[1, 77, 77, 77]
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, 3]
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, -4]
    with self.assertRaisesRegex(IndexError, "single ellipsis"): a[..., ...]
    with self.assertRaises(ValueError): a[::0, 1]
    with self.assertRaises(TypeError): a[:Tensor([3]), 1]
    with self.assertRaises(IndexError): b[:]

  def test_slice_ellipsis(self):
    helper_test_op([(3,3,3,3)], lambda x: x[..., 0])
    helper_test_op([(3,3,3,3)], lambda x: x[0, ...])
    helper_test_op([(3,3,3,3)], lambda x: x[0, ..., 0])
    helper_test_op([(3,3,3,3)], lambda x: x[0:3, ..., 2:3])
    helper_test_op([(3,3,3,3)], lambda x: x[None, 0:3, ..., 0, None])

  def test_double_slice(self):
    helper_test_op([(4,4)], lambda x: x[:, 1:2][1:2])
    helper_test_op([(4,4)], lambda x: x[1:3][1:2])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][0:1])
    helper_test_op([(4,4)], lambda x: x[:, 1:2][:, 0:1])

  def test_slice_with_const_tensor(self):
    t = Tensor.zeros(1, dtype=dtypes.int)
    helper_test_op([(3,3,3)], lambda x: x[:, [0], :], lambda x: x[:, t, :])
    helper_test_op([(3,3,3)], lambda x: x[:, [0], :], lambda x: x[:, t.contiguous(), :])

  def test_slice_empty(self):
    helper_test_op([(10,10)], lambda x: x[1:1])

  def test_slice_zero_in_shape(self):
    helper_test_op([(10,10)], lambda x: x[1:1])
    helper_test_op([(3,3,3)], lambda x: x[-2:-5])

  def test_transpose(self):
    helper_test_op([(3,3)], lambda x: x.T)
    helper_test_op([(3,3,3)], lambda x: x.transpose(1,2))
    helper_test_op([(3,3,3)], lambda x: x.transpose(0,2))

  def test_view(self):
    helper_test_op([(4,3,6,6)], lambda x: x.view((12,6,6)))
    helper_test_op([(4,3,6,6)], lambda x: x.view((-1,3,6,6)))
    helper_test_op([(6,)], lambda x: x.view(2, 3))
    helper_test_op([(6,1)], lambda x: x.view([2, 3]))
    helper_test_op([(1,6)], lambda x: x.view((3, 2)))
    helper_test_op([(3,2)], lambda x: x.view((2, 3)))
    helper_test_op([(3,2)], lambda x: x.view(6))

  def test_squeeze(self):
    helper_test_op([(1,3,6,6)], lambda x: x.squeeze(0))
    helper_test_op([(4,3,1,6)], lambda x: x.squeeze(1))
    helper_test_op([(4,3,6,6)], lambda x: x.squeeze(3))
    self.helper_test_exception([(4,3,6,6)], lambda x: x.squeeze(50), expected=IndexError)
    self.helper_test_exception([(4,3,6,6)], lambda x: x.squeeze(50), expected=IndexError)
    helper_test_op([(4,3,6,1)], lambda x: x.squeeze(-1))
    helper_test_op([(4,3,6,6)], lambda x: x.squeeze())
    helper_test_op([(1,3,6,6)], lambda x: x.squeeze())
    helper_test_op([(2,3,1)], lambda x: x.squeeze())
    helper_test_op([()], lambda x: x.squeeze(-1))
    helper_test_op([()], lambda x: x.squeeze(0))
    helper_test_op([()], lambda x: x.squeeze())
    self.helper_test_exception([()], lambda x: x.squeeze(10), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.squeeze(1), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.squeeze(-2), expected=IndexError)

  def test_unsqueeze(self):
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(0))
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(4))
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(-1))
    helper_test_op([(4,3,6,6)], lambda x: x.unsqueeze(-3))
    helper_test_op([()], lambda x: x.unsqueeze(0))

  def test_flatten(self):
    for axis in range(3):
      helper_test_op([(4,3,6,6)], lambda x: x.flatten(start_dim=axis))
    for axis in range(3):
      helper_test_op([(4,3,6,6)], lambda x: x.flatten(end_dim=axis))
    helper_test_op([(4,3,6,6)], lambda x: x.flatten(start_dim=1, end_dim=3))
    helper_test_op([()], lambda x: x.flatten())
    helper_test_op([(1,)], lambda x: x.flatten())

  def test_unflatten(self):
    helper_test_op([(4,3,6,6)], lambda x: x.unflatten(0, (2, 2)))
    helper_test_op([(4,3,6,6)], lambda x: x.unflatten(3, (3, 2)))
    helper_test_op([(4,3,6,6)], lambda x: x.unflatten(-1, (3, 2, 1)))

  def test_expand(self):
    helper_test_op([(4,3,1,6)], lambda x: x.expand((4,3,2,6)))
    helper_test_op([(1,1,1,1)], lambda x: x.expand((4,3,2,6)))
    helper_test_op([(4,3,1,6)], lambda x: x.expand((6,1,4,3,2,6)))
    helper_test_op([(4,3,1,6)], lambda x: x.expand((0,1,4,3,2,6)))
    helper_test_op([(4,3,1,6)], lambda x: x.expand((4,3,0,6)))
    helper_test_op([()], lambda x: x.expand((4,3,2,6)))
    helper_test_op([()], lambda x: x.expand([]))

  def test_diag(self):
    helper_test_op([(5,)], lambda x: x.diag())

  def test_diagonal(self):
    helper_test_op([(5,5)], lambda x: x.diagonal())
    helper_test_op([(3,4)], lambda x: x.diagonal())
    helper_test_op([(4,3)], lambda x: x.diagonal())
    helper_test_op([(3,3,3)], lambda x: x.diagonal(dim1=-2, dim2=-1))
    helper_test_op([(4,5,6)], lambda x: x.diagonal(dim1=-2, dim2=-1))
    helper_test_op([(2,3,4,5)], lambda x: x.diagonal(dim1=-2, dim2=-1))
    helper_test_op([(5,5)], lambda x: x.diagonal(offset=1))
    helper_test_op([(5,5)], lambda x: x.diagonal(offset=-1))
    helper_test_op([(3,5)], lambda x: x.diagonal(offset=2))
    self.helper_test_exception([(3,3)], lambda x: x.diagonal(dim1=0, dim2=0), expected=RuntimeError)

  def test_roll(self):
    helper_test_op([(2, 4)], lambda x: x.roll(1))
    helper_test_op([(2, 4)], lambda x: x.roll((1,)))
    self.helper_test_exception([(2, 4)], lambda x: x.roll((1, 2)), expected=RuntimeError)
    helper_test_op([(2, 4)], lambda x: x.roll(1, 0))
    helper_test_op([(2, 4)], lambda x: x.roll(-1, 0))
    helper_test_op([(2, 4)], lambda x: x.roll(shifts=(2, 1), dims=(0, 1)))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(1, 0))
    helper_test_op([(2, 4)], lambda x: x.roll(1, -1))
    helper_test_op([(2, 4)], lambda x: x.roll(-1, -1))
    helper_test_op([(2, 4)], lambda x: x.roll(5, 0))
    helper_test_op([(2, 4)], lambda x: x.roll(-5, 0))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(2, -3), dims=(0, 2)))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(1, 2, -1), dims=(0, 1, 2)))
    helper_test_op([(2, 4)], lambda x: x.roll(0, 0))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(0, 0), dims=(0, 1)))
    helper_test_op([(2, 4, 6)], lambda x: x.roll(shifts=(0, 2), dims=(0, 1)))
    self.helper_test_exception([(3, 3)], lambda x: x.roll(shifts=1, dims=(0, 1)), expected=RuntimeError)
    self.helper_test_exception([(10,)], lambda x: x.roll(shifts=(1, 2), dims=0), expected=RuntimeError)

  def test_detach(self):
    helper_test_op([(4,3,6,6)], lambda x: x.detach(), forward_only=True)
    helper_test_op([()], lambda x: x.detach(), forward_only=True)

  @slow_test
  def test_cat(self):
    for dim in range(-2, 3):
      helper_test_op([(45,65,9), (45,65,9), (45,65,9)], lambda x,y,z: torch.cat((x,y,z), dim),
                     lambda x,y,z: x.cat(y, z, dim=dim))
    helper_test_op([(45,0,9), (45,0,9), (45,0,9)], lambda x,y,z: torch.cat((x,y,z), 0),
                   lambda x,y,z: x.cat(y, z, dim=0))
    helper_test_op([(45,0,9), (45,1,9), (45,2,9)], lambda x,y,z: torch.cat((x,y,z), 1),
                   lambda x,y,z: x.cat(y, z, dim=1))
    helper_test_op([(45,0,9), (45,0,9), (45,0,9)], lambda x,y,z: torch.cat((x,y,z), 1),
                   lambda x,y,z: x.cat(y, z, dim=1))

  def test_stack(self):
    for dim in range(-1, 3):
      helper_test_op([(5,6,3), (5,6,3), (5,6,3)], lambda x,y,z: torch.stack((x, y, z), dim),
                     lambda x,y,z: Tensor.stack(x, y, z, dim=dim))
      helper_test_op([(5,6,3), (5,6,3), (5,6,3)], lambda x,y,z: torch.stack((x, y, z), dim),
                     lambda x,y,z: Tensor.stack((x, y, z), dim=dim))

  def test_stack_slice(self):
    helper_test_op([(4)], lambda x: torch.stack([x for i in range(3)])[0,:], lambda x: Tensor.stack(*[x for i in range(3)])[0,:])
    helper_test_op([(5)], lambda x: torch.stack([x for i in range(3)])[0,0], lambda x: Tensor.stack(*[x for i in range(3)])[0,0])
    helper_test_op([(4,4)], lambda x: torch.stack([x for i in range(4)])[3], lambda x: Tensor.stack(*[x for i in range(4)])[3])

  def test_sum(self):
    helper_test_op([(45,3)], lambda x: x.sum())
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=3))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,3)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(0,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,2)))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1))
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1, keepdim=True))
    helper_test_op([()], lambda x: x.sum())
    helper_test_op([()], lambda x: x.sum(0))
    helper_test_op([()], lambda x: x.sum(-1))
    helper_test_op([()], lambda x: x.sum(()))
    self.helper_test_exception([(3,4,5,6)], lambda x: x.sum(5), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum(1), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum((1,)), expected=IndexError)

  def test_max(self):
    helper_test_op([(45,3)], lambda x: x.max())
    helper_test_op([(45,3)], lambda x: x.max().mul(0.5))
    helper_test_op([(3,4,5,6)], lambda x: x.max(axis=1)[0], lambda x: x.max(axis=1))
    helper_test_op([()], lambda x: x.max())
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[-2**31, 0]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.max(), forward_only=True, vals=[[True, False]])

  def test_mean(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean())
    helper_test_op([()], lambda x: x.mean())
    helper_test_op([(3,4,5,6)], lambda x: x.mean(axis=(1,2)))
    helper_test_op([(1,0,3,0,5)], lambda x: x.mean(axis=(1,3)))

  def test_mean_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean(axis=(1,2)))

  def test_var(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var())
    helper_test_op([(15, 25, 35)], lambda x: x.var(correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.var(correction=5))

  @slow_test
  def test_var_axis(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var(0))
    helper_test_op([(15, 25, 35)], lambda x: x.var(2))
    helper_test_op([(15, 25, 35)], lambda x: x.var([1, 2]))
    helper_test_op([(15, 25, 35)], lambda x: x.var(0, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.var(2, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.var([1, 2], correction=0))

  def test_var_zero_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="var\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3)))
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3), correction=0))
      helper_test_op([(1,0,3,0,5)], lambda x: x.var(axis=(1,3), correction=5))

  def test_var_one_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="var\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3)))
      helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3), correction=5), forward_only=True)
      helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4), correction=5), forward_only=True)
    helper_test_op([(1,)], lambda x: x.var(axis=(0,), correction=0))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,3), correction=0))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4)))
    helper_test_op([(1,2,3,1,5)], lambda x: x.var(axis=(0,4), correction=0))

  def test_var_keepdim(self):
    helper_test_op([(15, 25, 35)], lambda x: x.var(keepdim=True))
    helper_test_op([(15, 25, 35)], lambda x: x.var(0, keepdim=True, correction=0))

  @slow_test
  def test_std(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std())
    helper_test_op([(15, 25, 35)], lambda x: x.std(correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.std(correction=5))

  @slow_test
  def test_std_axis(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std(0))
    helper_test_op([(15, 25, 35)], lambda x: x.std(2))
    helper_test_op([(15, 25, 35)], lambda x: x.std([1, 2]))
    helper_test_op([(15, 25, 35)], lambda x: x.std(0, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.std(2, correction=0))
    helper_test_op([(15, 25, 35)], lambda x: x.std([1, 2], correction=0))

  def test_std_zero_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3)))
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3), correction=0))
      helper_test_op([(1,0,3,0,5)], lambda x: x.std(axis=(1,3), correction=5))

  def test_std_one_in_axis(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3)))
      helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3), correction=5))
      helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4), correction=5))
    helper_test_op([(1,)], lambda x: x.std(axis=(0,), correction=0), forward_only=True)
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,3), correction=0), forward_only=True)
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4)))
    helper_test_op([(1,2,3,1,5)], lambda x: x.std(axis=(0,4), correction=0))

  def test_std_keepdim(self):
    helper_test_op([(15, 25, 35)], lambda x: x.std(keepdim=True))
    helper_test_op([(15, 25, 35)], lambda x: x.std(0, keepdim=True, correction=0))

  @slow_test
  def test_std_mean(self):
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x)),
                                 lambda x: Tensor.stack(*x.std_mean()))
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x, correction=5)),
                                 lambda x: Tensor.stack(*x.std_mean(correction=5)))
    helper_test_op([(15,25,35)], lambda x: torch.stack(torch.std_mean(x, keepdim=True, correction=0)),
                                 lambda x: Tensor.stack(*x.std_mean(keepdim=True, correction=0)))
    helper_test_op([(3,4,5,6)], lambda x: torch.stack(torch.std_mean(x, axis=(1,2))),
                                lambda x: Tensor.stack(*x.std_mean(axis=(1,2))))

  def test_std_mean_loaded_nan(self):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", message="std_mean\\(\\): degrees of freedom is <= 0")
      helper_test_op([(1,0,3,0,5)], lambda x: torch.stack(torch.std_mean(x, axis=(1,3))),
                                    lambda x: Tensor.stack(*x.std_mean(axis=(1,3))))

  def test_argmax(self):
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[2, 2]])
    helper_test_op(None, lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True, vals=[[1, 2, 2]])
    helper_test_op([(10,20)], lambda x: x.argmax().type(torch.int32), lambda x: x.argmax(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(0, False).type(torch.int32), lambda x: x.argmax(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, False).type(torch.int32), lambda x: x.argmax(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmax(1, True).type(torch.int32), lambda x: x.argmax(1, True), forward_only=True)
    helper_test_op(None, lambda x: (~x).argmax().type(torch.int32), lambda x: (~x).argmax(), forward_only=True, vals=[[2, 2]])

  def test_argmin(self):
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[2, 2]])
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[3, 2, 2]])
    if not COMPILE_ONLY:
      np.testing.assert_equal(Tensor([2,2]).argmin().numpy(), 0)
      np.testing.assert_equal(Tensor([3,2,2]).argmin().numpy(), 1)
    helper_test_op([(10,20)], lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(0, False).type(torch.int32), lambda x: x.argmin(0, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, False).type(torch.int32), lambda x: x.argmin(1, False), forward_only=True)
    helper_test_op([(10,20)], lambda x: x.argmin(1, True).type(torch.int32), lambda x: x.argmin(1, True), forward_only=True)
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[-2**31, 0]])
    helper_test_op(None, lambda x: x.type(torch.int32).argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.type(torch.int32).argmin().type(torch.int32), lambda x: x.argmin(), forward_only=True, vals=[[True, False]])

  def test_sort(self):
    for shape in [(0,), (0,5), (1,), (1,5)]:
      helper_test_op([shape], lambda x: x.sort(0).values, lambda x: x.sort(0)[0], forward_only=True)
      helper_test_op([shape], lambda x: x.sort(0).indices.type(torch.int32), lambda x: x.sort(0)[1], forward_only=True)
    for dim in [-1, 0, 1]:
      for descending in [True, False]:
        helper_test_op([(8,8,6)], lambda x: x.sort(dim, descending).values, lambda x: x.sort(dim, descending)[0], forward_only=True)
        helper_test_op([(8,8,6)], lambda x: x.sort(dim, descending).indices.type(torch.int32), lambda x: x.sort(dim, descending)[1],
                       forward_only=True)
    helper_test_op(None, lambda x: x.sort(stable=True).values, lambda x: x.sort()[0], forward_only=True, vals=[[0, 1] * 9])
    helper_test_op(None, lambda x: x.sort(stable=True).indices.type(torch.int32), lambda x: x.sort()[1], forward_only=True, vals=[[0, 1] * 9])
    helper_test_op(None, lambda x: x.sort(stable=True, descending=True).values,
                   lambda x: x.sort(descending=True)[0], forward_only=True, vals=[[0, 1] * 9])
    helper_test_op(None, lambda x: x.sort(stable=True, descending=True).indices.type(torch.int32),
                   lambda x: x.sort(descending=True)[1], forward_only=True, vals=[[0, 1] * 9])

  def test_argsort(self):
    for dim in [-1, 0, 1]:
      for descending in [True, False]:
        helper_test_op([(8,8,6)], lambda x: torch.argsort(x, dim=dim, descending=descending, stable=True).type(torch.int32),
                                  lambda x: x.argsort(dim, descending), forward_only=True)

  def test_topk(self):
    helper_test_op([(8)], lambda x: x.topk(3).values, lambda x: x.topk(3)[0], forward_only=True)
    helper_test_op([(8)], lambda x: x.topk(3).indices.type(torch.int32), lambda x: x.topk(3)[1], forward_only=True)
    for dim in [0, 1, -1]:
      for largest in [True, False]:
        for sorted_ in [True]:
          helper_test_op([(5,5,4)],
                          lambda x: x.topk(4, dim, largest, sorted_).values,
                          lambda x: x.topk(4, dim, largest, sorted_)[0], forward_only=True)
          helper_test_op([(5,5,4)],
                          lambda x: x.topk(4, dim, largest, sorted_).indices.type(torch.int32),
                          lambda x: x.topk(4, dim, largest, sorted_)[1], forward_only=True)
    if not COMPILE_ONLY:
      value, indices = Tensor([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]).topk(3)
      np.testing.assert_equal(value.numpy(), [1, 1, 1])
      np.testing.assert_equal(indices.numpy(), [0, 1, 3])
      value, indices = Tensor([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]).topk(3, largest=False)
      np.testing.assert_equal(value.numpy(), [0, 0, 0])
      np.testing.assert_equal(indices.numpy(), [2, 4, 6])
    self.helper_test_exception([(4)], lambda x: x.topk(5), expected=(RuntimeError, ValueError))

  @slow_test
  def test_einsum(self):
    helper_test_op([()], lambda a: torch.einsum('->', a), lambda a: Tensor.einsum('->', a))
    helper_test_op([(10,10)], lambda a: torch.einsum('ij->ji', a), lambda a: Tensor.einsum('ij->ji', a))
    helper_test_op([(10,10)], lambda a: torch.einsum('ij -> ji', a), lambda a: Tensor.einsum('ij -> ji', a))
    helper_test_op([(10,10)], lambda a: torch.einsum('ji', a), lambda a: Tensor.einsum('ji', a))
    helper_test_op([(4,6,8)], lambda a: torch.einsum('jki', a), lambda a: Tensor.einsum('jki', a))
    helper_test_op([(4,6,8)], lambda a: torch.einsum('dog', a), lambda a: Tensor.einsum('dog', a))
    helper_test_op([(4,6),(6,8)], lambda a, b: torch.einsum('ij,jk', a, b), lambda a, b: Tensor.einsum('ij,jk', a, b))
    helper_test_op([(4,6,8)], lambda a: torch.einsum('ijk->', a), lambda a: Tensor.einsum('ijk->', a))
    helper_test_op([(5,5)], lambda a: torch.einsum('ij->j', a), lambda a: Tensor.einsum('ij->j', a))
    helper_test_op([(5,5)], lambda a: torch.einsum('ij->i', a), lambda a: Tensor.einsum('ij->i', a))
    helper_test_op([(3,4), (4,)], lambda a,b: torch.einsum('ik,k->i', a,b), lambda a,b: Tensor.einsum('ik,k->i', a, b))
    helper_test_op([(3,4), (4,5)], lambda a,b: torch.einsum('ik,kj->ij', a,b), lambda a,b: Tensor.einsum('ik,kj->ij', a, b))
    helper_test_op([(3,4), (4,5)], lambda a,b: torch.einsum('jk,ki->ji', a,b), lambda a,b: Tensor.einsum('jk,ki->ji', a, b))
    helper_test_op([(5),(5)], lambda a,b: torch.einsum('i,i->i', [a,b]), lambda a,b: Tensor.einsum('i,i->i', [a,b]))
    helper_test_op([(5,6),(5,6)], lambda a,b: torch.einsum('ij,ij->ij', a,b), lambda a,b: Tensor.einsum('ij,ij->ij', a,b))
    helper_test_op([(5,), (5,)], lambda a,b: torch.einsum('i,j->ij', a,b), lambda a,b: Tensor.einsum('i,j->ij',a,b))
    helper_test_op([(2,4,6),(2,6,8)], lambda a,b: torch.einsum('ijk,ikl->ijl', [a, b]), lambda a,b: Tensor.einsum('ijk,ikl->ijl', [a, b]))
    helper_test_op([(2,4,5),(2,5,7)], lambda a,b: torch.einsum('ijk,ikl->jil', [a, b]), lambda a,b: Tensor.einsum('ijk,ikl->jil', [a, b]))
    helper_test_op([(4,2,5),(2,5,7)], lambda a,b: torch.einsum('jik,ikl->jil', [a, b]), lambda a,b: Tensor.einsum('jik,ikl->jil', [a, b]))
    helper_test_op([(2,4,6),(2,6,8)], lambda a,b: torch.einsum('ijk,ika->ija', [a, b]), lambda a,b: Tensor.einsum('ijk,ika->ija', [a, b]))
    helper_test_op([(3,5,8,10),(11,7,5,13,8)], lambda a,b: torch.einsum('pqrs,tuqvr->pstuv', a,b),
                                                lambda a,b: Tensor.einsum('pqrs,tuqvr->pstuv', a,b), atol=1e-5)
    helper_test_op([(3,8,10,5),(11,5,7,13,8)], lambda a,b: torch.einsum('prsq,tquvr->pstuv', a,b),
                                                lambda a,b: Tensor.einsum('prsq,tquvr->pstuv', a,b), atol=1e-5)
    helper_test_op([(3,5,8,10),(11,7,5,13,8)], lambda a,b: torch.einsum('zqrs,tuqvr->zstuv', a,b),
                                                lambda a,b: Tensor.einsum('zqrs,tuqvr->zstuv', a,b), atol=1e-5)
    helper_test_op([(2,3),(5,3,7),(2,7)], lambda a,b,c: torch.einsum('ik,jkl,il->ij', [a,b,c]), lambda a,b,c: Tensor.einsum('ik,jkl,il->ij', [a,b,c]))

  @slow_test
  def test_einsum_ellipsis(self):
    helper_test_op([(3, 8, 9), (3, 8, 9)], lambda a, b: torch.einsum('...id, ...jd -> ...ij', [a, b]),
               lambda a, b: Tensor.einsum('...id, ...jd -> ...ij', [a, b]))
    helper_test_op([(3, 8, 9), (3, 8, 9)], lambda a, b: torch.einsum('...id, ...jd', [a, b]),
               lambda a, b: Tensor.einsum('...id, ...jd', [a, b]))
    helper_test_op([(2, 3, 4, 5), (5, 2, 4)], lambda a, b: torch.einsum('i...j,ji...->...', [a, b]),
               lambda a, b: Tensor.einsum('i...j,ji...->...', [a, b]))
    helper_test_op([(32, 7, 24, 24, 24), (32, 7, 24, 24, 24)], lambda a, b: torch.einsum('ij...,ij...->ij', [a, b]),
               lambda a, b: Tensor.einsum('ij...,ij...->ij', [a, b]))
    self.helper_test_exception([(2, 3, 4), (2, 3, 4)], lambda a, b: torch.einsum('...ik..., ...jk ->', [a, b]),
                lambda a, b: Tensor.einsum('...ik..., ...jk ->', [a, b]), expected=(RuntimeError, IndexError))
    self.helper_test_exception([(2, 3, 4), (2, 3, 4)], lambda a, b: torch.einsum('i...j,ji...->...', [a, b]),
                lambda a, b: Tensor.einsum('i...j,ji...->...', [a, b]), expected=RuntimeError)

  def test_einsum_trace(self):
    helper_test_op([(5,), (5,)], lambda a, b: torch.einsum('i,i', a, b), lambda a, b: Tensor.einsum('i,i', a, b))
    helper_test_op([(4, 4)], lambda a: torch.einsum('ii->i', a), lambda a: Tensor.einsum('ii->i', a))
    helper_test_op([(4, 4)], lambda a: torch.einsum('ii->', a), lambda a: Tensor.einsum('ii->', a))
    helper_test_op([(3, 5, 5)], lambda a: torch.einsum('...ii->...i', a), lambda a: Tensor.einsum('...ii->...i', a))
    helper_test_op([(3, 5, 5)], lambda a: torch.einsum('...ii->...', a), lambda a: Tensor.einsum('...ii->...', a))

  def test_einsum_shape_check(self):
    self.helper_test_exception([(3,8,10,5), (11,5,13,16,8)], lambda a, b: torch.einsum('pqrs,tuqvr->pstuv', [a, b]),
                lambda a, b: Tensor.einsum('pqrs,tuqvr->pstuv', [a, b]), expected=RuntimeError)

  def test_einsum_arity_check1(self):
    self.helper_test_exception([(10,15), (15,20), (20,10)], lambda a, b, c: torch.einsum('ij,jk->ij', [a, b, c]),
                lambda a, b, c: Tensor.einsum('ij,jk->ij', [a, b, c]), expected=(ValueError, RuntimeError))

  def test_einsum_arity_check2(self):
    self.helper_test_exception([(10,10)], lambda a: torch.einsum('ij,jk->ij', a),
                lambda a: Tensor.einsum('ij,jk->ij', a), expected=(ValueError, RuntimeError))

  def test_any(self):
    helper_test_op([(3,4,5,6)], lambda x: x.any(), forward_only=True)
    helper_test_op(None, lambda x: x.any(), vals=[[True, True]], forward_only=True)
    helper_test_op(None, lambda x: x.any(), vals=[[True, False]], forward_only=True)
    helper_test_op(None, lambda x: x.any(), vals=[[False, False]], forward_only=True)
    helper_test_op([()], lambda x: x.any(), forward_only=True)

  def test_any_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.any(axis=(1,2)), forward_only=True)

  def test_any_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.any(axis=(1,3)), forward_only=True)

  def test_all(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(), forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, True]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[True, False]], forward_only=True)
    helper_test_op(None, lambda x: x.all(), vals=[[False, False]], forward_only=True)
    helper_test_op([()], lambda x: x.all(), forward_only=True)

  def test_all_axis(self):
    helper_test_op([(3,4,5,6)], lambda x: x.all(axis=(1,2)), forward_only=True)

  def test_all_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.all(axis=(1,3)), forward_only=True)

  def test_all_large(self):
    for exp in [15, 16, 20]:
      helper_test_op(None, lambda: torch.ones(2**exp).bool().all(), lambda: Tensor.ones(2**exp).bool().all(), vals=[], forward_only=True)

  def test_isclose(self):
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x), forward_only=True)
    helper_test_op([(3, 4, 5, 6), (3, 4, 5, 6)], lambda x,y: x.isclose(y), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x, equal_nan=True), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-6), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-9), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-6, atol=0.0), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-9, atol=0.0), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-6, rtol=0.01), forward_only=True)
    helper_test_op([(3, 4, 5, 6)], lambda x: x.isclose(x + 1e-9, rtol=0.01), forward_only=True)
    helper_test_op(None, lambda x,y: x.isclose(y), vals=[[1e-7, 1e-8, 1e-9], [0.0, 0.0, 0.0]], forward_only=True)

  def test_isclose_edge_cases(self):
    for a in [math.inf, -math.inf, math.nan, 0.0]:
      for b in [math.inf, -math.inf, math.nan, 0.0]:
        helper_test_op(None, lambda x,y: x.isclose(y), vals=[[a], [b]], forward_only=True)
        helper_test_op(None, lambda x,y: x.isclose(y, equal_nan=True), vals=[[a], [b]], forward_only=True)

  def test_cumsum(self):
    helper_test_op([(10)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(20,)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=1), lambda x: Tensor.cumsum(x, axis=1))
    helper_test_op([(20,30,40)], lambda x: torch.cumsum(x, dim=2), lambda x: Tensor.cumsum(x, axis=2))
    helper_test_op([(20,30,40)], lambda x: torch.cumsum(x, dim=-1), lambda x: Tensor.cumsum(x, axis=-1))

  def test_cumprod(self):
    helper_test_op([(10)], lambda x: torch.cumprod(x, dim=0), lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(20,)], lambda x: torch.cumprod(x, dim=0), lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(20,30)], lambda x: torch.cumprod(x, dim=0), lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(20,30)], lambda x: torch.cumprod(x, dim=1), lambda x: Tensor.cumprod(x, axis=1))
    helper_test_op([(20,30,40)], lambda x: torch.cumprod(x, dim=2), lambda x: Tensor.cumprod(x, axis=2))
    helper_test_op([(20,30,40)], lambda x: torch.cumprod(x, dim=-1), lambda x: Tensor.cumprod(x, axis=-1))

  def test_cummax(self):
    helper_test_op([(10)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(10)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)
    helper_test_op([(5,6)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(5,6)], lambda x: torch.cummax(x, dim=1).values, lambda x: Tensor.cummax(x, axis=1)[0])
    helper_test_op([(5,6,7)], lambda x: torch.cummax(x, dim=2).values, lambda x: Tensor.cummax(x, axis=2)[0])
    helper_test_op([(5,6,7)], lambda x: torch.cummax(x, dim=-1).values, lambda x: Tensor.cummax(x, axis=-1)[0])

  def test_cummin(self):
    helper_test_op([(10)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(10)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)
    helper_test_op([(5,6)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(5,6)], lambda x: torch.cummin(x, dim=1).values, lambda x: Tensor.cummin(x, axis=1)[0])
    helper_test_op([(5,6,7)], lambda x: torch.cummin(x, dim=2).values, lambda x: Tensor.cummin(x, axis=2)[0])
    helper_test_op([(5,6,7)], lambda x: torch.cummin(x, dim=-1).values, lambda x: Tensor.cummin(x, axis=-1)[0])

  def test_small_cumsum(self):
    helper_test_op([(10)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))

  @slow_test
  def test_simple_cumsum(self):
    helper_test_op([(512)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(1022)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))

  def test_cumsum_zero_axis(self):
    helper_test_op([(2,0,4)], lambda x: torch.cumsum(x, dim=1), lambda x: Tensor.cumsum(x, axis=1))
    helper_test_op([(0,3)], lambda x: torch.cumsum(x, dim=0), lambda x: Tensor.cumsum(x, axis=0))
    helper_test_op([(2,3,0)], lambda x: torch.cumsum(x, dim=2), lambda x: Tensor.cumsum(x, axis=2))

  def test_small_cumprod(self):
    helper_test_op([(10)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))

  @slow_test
  def test_simple_cumprod(self):
    helper_test_op([(512)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(1022)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))

  def test_cumprod_zero_axis(self):
    helper_test_op([(2, 0, 4)],lambda x: torch.cumprod(x, dim=1),lambda x: Tensor.cumprod(x, axis=1))
    helper_test_op([(0, 3)],lambda x: torch.cumprod(x, dim=0),lambda x: Tensor.cumprod(x, axis=0))
    helper_test_op([(2, 3, 0)],lambda x: torch.cumprod(x, dim=2),lambda x: Tensor.cumprod(x, axis=2))

  def test_small_cummax(self):
    helper_test_op([(10)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(10)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)

  @slow_test
  def test_simple_cummax(self):
    helper_test_op([(512)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(512)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)
    helper_test_op([(1022)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(1022)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)

  def test_cummax_zero_axis(self):
    helper_test_op([(2,0,4)], lambda x: torch.cummax(x, dim=1).values, lambda x: Tensor.cummax(x, axis=1)[0])
    helper_test_op([(2,0,4)], lambda x: torch.cummax(x, dim=1).indices.int(), lambda x: Tensor.cummax(x, axis=1)[1], forward_only=True)
    helper_test_op([(0,3)], lambda x: torch.cummax(x, dim=0).values, lambda x: Tensor.cummax(x, axis=0)[0])
    helper_test_op([(0,3)], lambda x: torch.cummax(x, dim=0).indices.int(), lambda x: Tensor.cummax(x, axis=0)[1], forward_only=True)
    helper_test_op([(2,3,0)], lambda x: torch.cummax(x, dim=2).values, lambda x: Tensor.cummax(x, axis=2)[0])
    helper_test_op([(2,3,0)], lambda x: torch.cummax(x, dim=2).indices.int(), lambda x: Tensor.cummax(x, axis=2)[1], forward_only=True)

  def test_small_cummin(self):
    helper_test_op([(10)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(10)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)

  @slow_test
  def test_simple_cummin(self):
    helper_test_op([(512)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(512)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)
    helper_test_op([(1022)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(1022)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)

  def test_cummin_zero_axis(self):
    helper_test_op([(2,0,4)], lambda x: torch.cummin(x, dim=1).values, lambda x: Tensor.cummin(x, axis=1)[0])
    helper_test_op([(2,0,4)], lambda x: torch.cummin(x, dim=1).indices.int(), lambda x: Tensor.cummin(x, axis=1)[1], forward_only=True)
    helper_test_op([(0,3)], lambda x: torch.cummin(x, dim=0).values, lambda x: Tensor.cummin(x, axis=0)[0])
    helper_test_op([(0,3)], lambda x: torch.cummin(x, dim=0).indices.int(), lambda x: Tensor.cummin(x, axis=0)[1], forward_only=True)
    helper_test_op([(2,3,0)], lambda x: torch.cummin(x, dim=2).values, lambda x: Tensor.cummin(x, axis=2)[0])
    helper_test_op([(2,3,0)], lambda x: torch.cummin(x, dim=2).indices.int(), lambda x: Tensor.cummin(x, axis=2)[1], forward_only=True)

  def test_prod(self):
    helper_test_op(None, lambda x: x.prod(), vals=[[1.0, 2.0, 3.0]])
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=3), lambda x: x.prod(axis=3))
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1), lambda x: x.prod(axis=1))
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1, keepdim=True), lambda x: x.prod(axis=1, keepdim=True))
    helper_test_op([()], lambda x: x.prod())
    helper_test_op([()], lambda x: x.prod(0))
    helper_test_op([()], lambda x: x.prod(-1))

  def test_min(self):
    helper_test_op([(3,3)], lambda x: x.min())
    helper_test_op([(45,3)], lambda x: x.min())
    helper_test_op([(45,3)], lambda x: x.min().mul(0.5))
    helper_test_op([()], lambda x: x.min())
    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[0, -2**31]])
    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[-2**31, 0]])
    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[False, True]])
    helper_test_op(None, lambda x: x.min(), forward_only=True, vals=[[True, False]])

  def test_floor(self):
    helper_test_op([(45,65)], lambda x: x.floor(), forward_only=True)
    helper_test_op([()], lambda x: x.floor(), forward_only=True)

  def test_ceil(self):
    helper_test_op([(45,65)], lambda x: x.ceil(), forward_only=True)
    helper_test_op([()], lambda x: x.ceil(), forward_only=True)

  def test_round(self):
    helper_test_op([(45,65)], lambda x: x.round(), forward_only=True)
    helper_test_op([()], lambda x: x.round(), forward_only=True)

  def test_abs(self):
    helper_test_op([(45,65)], lambda x: x.abs())
    helper_test_op([()], lambda x: x.abs())

  def test_sqrt(self):
    helper_test_op([(45,65)], lambda x: x.sqrt(), low=0.1, high=4.0)
    helper_test_op([()], lambda x: x.sqrt(), vals=[4.0], forward_only=True)

  def test_rsqrt(self):
    helper_test_op([(45,65)], lambda x: x.rsqrt(), low=0.1, high=4.0)
    helper_test_op([()], lambda x: x.rsqrt(), vals=[4.0], forward_only=True)

  def test_sign(self):
    helper_test_op([(45,65)], lambda x: x.sign())
    helper_test_op([()], lambda x: x.sign())

  def test_sin(self):
    helper_test_op([(45,65)], lambda x: x.sin())
    helper_test_op([()], lambda x: x.sin())

  def test_cos(self):
    helper_test_op([(45,65)], lambda x: x.cos())
    helper_test_op([()], lambda x: x.cos())

  def test_tan(self):
    helper_test_op([(45,65)], lambda x: x.tan(), low=-1.5, high=1.5)
    helper_test_op([()], lambda x: x.tan())

  def test_asin(self):
    helper_test_op([(45,65)], lambda x: x.asin(), low=-1, high=1)

  def test_acos(self):
    helper_test_op([(45,65)], lambda x: x.acos(), low=-1, high=1)

  def test_atan(self):
    helper_test_op([(45,65)], lambda x: x.atan())

  def test_log(self):
    helper_test_op([(45,65)], torch.log, Tensor.log, low=0.1, high=4.0)
    helper_test_op([()], torch.log, Tensor.log, vals=[4.0], forward_only=True)

  def test_log10(self):
    helper_test_op([(45,65)], torch.log10, Tensor.log10, low=0.1, high=4.0)
    helper_test_op([()], torch.log10, Tensor.log10, vals=[4.0], forward_only=True)

  def test_log2(self):
    helper_test_op([(45,65)], torch.log2, Tensor.log2, low=0.1, high=4.0)
    helper_test_op([()], torch.log2, Tensor.log2, vals=[4.0], forward_only=True)

  def test_exp(self):
    helper_test_op([(45,65)], torch.exp, Tensor.exp)
    helper_test_op([()], torch.exp, Tensor.exp)

  def test_sigmoid(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid)
    helper_test_op([()], torch.sigmoid, Tensor.sigmoid)

  def test_logsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.logsigmoid, Tensor.logsigmoid)
    helper_test_op([()], torch.nn.functional.logsigmoid, Tensor.logsigmoid)

  def test_softplus(self):
    helper_test_op([(45,65)], torch.nn.functional.softplus, Tensor.softplus, grad_atol=1e-6)
    helper_test_op([()], torch.nn.functional.softplus, Tensor.softplus)

  def test_leaky_relu(self):
    helper_test_op([(45,65)], torch.nn.functional.leaky_relu, Tensor.leaky_relu)
    helper_test_op([()], torch.nn.functional.leaky_relu, Tensor.leaky_relu)

  def test_celu(self):
    for val in range(1, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.celu(x, val), lambda x: x.celu(val))
      helper_test_op([()], lambda x: torch.nn.functional.celu(x, val), lambda x: x.celu(val))

  def test_selu(self):
    helper_test_op([(45,65)], torch.nn.functional.selu, Tensor.selu)
    helper_test_op([()], torch.nn.functional.selu, Tensor.selu)

  def test_softsign(self):
    helper_test_op([(45,65)], torch.nn.functional.softsign, Tensor.softsign)
    helper_test_op([()], torch.nn.functional.softsign, Tensor.softsign)

  def test_swish(self):
    helper_test_op([(45,65)], torch.nn.functional.silu, Tensor.swish)
    helper_test_op([()], torch.nn.functional.silu, Tensor.swish)

  def test_abs_exact(self):
    for v in [-1., -0., 0., 1., math.inf, -math.inf, math.nan, -math.nan]:
      helper_test_op(None, torch.abs, Tensor.abs, vals=[[v]], forward_only=math.isnan(v))

  def test_sign_exact(self):
    helper_test_op(None, torch.sign, Tensor.sign, vals=[[-1.,0,1]])

  def test_softsign_exact(self):
    helper_test_op(None, torch.nn.functional.softsign, Tensor.softsign, vals=[[-1.,0,1]])

  def test_sigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)
    x = Tensor([300.0])
    self.assertAlmostEqual(x.sigmoid()[0].gradient(x)[0].item(), 0.0)
    x = Tensor([-300.0])
    self.assertAlmostEqual(x.sigmoid()[0].gradient(x)[0].item(), 0.0)

  def test_sigmoid_alt_extreme(self):
    def sigmoid(x:Tensor): return x.exp() / (1 + x.exp())
    x = Tensor([300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)
    x = Tensor([-300.0])
    self.assertAlmostEqual(sigmoid(x)[0].gradient(x)[0].item(), 0.0)

  def test_hardsigmoid(self):
    helper_test_op([(45,65)], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)
    helper_test_op([()], torch.nn.functional.hardsigmoid, Tensor.hardsigmoid)

  def test_hardsigmoid_extreme(self):
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=300, high=400)
    helper_test_op([(45,65)], torch.sigmoid, Tensor.sigmoid, low=-400, high=-300)

  def test_copysign(self):
    helper_test_op([(45,65), (45,65)], torch.copysign, Tensor.copysign)
    helper_test_op([(45,65), (45,1)], torch.copysign, Tensor.copysign)
    helper_test_op([(45,1), (1,65)], torch.copysign, Tensor.copysign)
    helper_test_op([(), ()], torch.copysign, Tensor.copysign)

  def test_copysign_exact(self):
    v = [-1., -0., 0., 1., math.inf, -math.inf, math.nan]
    for i in v:
      for j in v:
        helper_test_op(None, torch.copysign, Tensor.copysign, vals=[[i], [j]], forward_only=math.isinf(i) or math.isnan(i))

  def test_logaddexp(self):
    helper_test_op([(45,65), (45,65)], torch.logaddexp, Tensor.logaddexp)
    helper_test_op(None, torch.logaddexp, Tensor.logaddexp, vals=[[-1.], [-1.0, 2, 3]])
    helper_test_op(None, torch.logaddexp, Tensor.logaddexp, vals=[[-100.0, -200, -300], [-1.0, 2, 3]])
    helper_test_op(None, torch.logaddexp, Tensor.logaddexp, vals=[[1.0, 2000, 30000], [-1.0, 2, 3]])

  def test_erf(self):
    helper_test_op([(45,65)], torch.erf, Tensor.erf)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=300, high=400)
    helper_test_op([(45,65)], torch.erf, Tensor.erf, low=-400, high=-300)
    helper_test_op([()], torch.erf, Tensor.erf)

  def test_gelu(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu)

  def test_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, low=300, high=400)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.gelu(x, approximate="tanh"), Tensor.gelu, low=-400, high=-300)

  def test_quick_gelu(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)
    helper_test_op([()], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu)

  def test_quick_gelu_extreme(self):
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=300, high=400)
    helper_test_op([(45,65)], lambda x: x * torch.sigmoid(1.702 * x), Tensor.quick_gelu, low=-400, high=-300)

  def test_elu(self):
    helper_test_op([(45,65)], torch.nn.functional.elu, Tensor.elu)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.elu(x, alpha=0.1), lambda x: Tensor.elu(x, alpha=0.1))
    helper_test_op([()], torch.nn.functional.elu, Tensor.elu)

  def test_hardswish(self):
    helper_test_op([(45,65)], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)
    helper_test_op([()], torch.nn.functional.hardswish, Tensor.hardswish, grad_atol=1e-6)

  def test_mish(self):
    helper_test_op([(45,65)], torch.nn.functional.mish, Tensor.mish)
    helper_test_op([()], torch.nn.functional.mish, Tensor.mish)

  def test_tanh(self):
    helper_test_op([(45,65)], lambda x: x.tanh())
    helper_test_op([()], lambda x: x.tanh())

  def test_tanh_extreme(self):
    helper_test_op([(45,65)], lambda x: x.tanh(), grad_atol=1e-6, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.tanh(), grad_atol=1e-6, low=300, high=303)

  def test_hardtanh(self):
    for val in range(10, 30, 5):
      helper_test_op([(45,65)], lambda x: torch.nn.functional.hardtanh(x, -val, val), lambda x: x.hardtanh(-val, val), grad_atol=1e-6)
      helper_test_op([()], lambda x: torch.nn.functional.hardtanh(x, -val, val), lambda x: x.hardtanh(-val, val), grad_atol=1e-6)

  def test_sinh(self):
    helper_test_op([(45,65)], lambda x: x.sinh())
    helper_test_op([(45,65)], lambda x: x.sinh(), low=-300, high=-297, forward_only=True)
    helper_test_op([(45,65)], lambda x: x.sinh(), low=300, high=303, forward_only=True)

  def test_cosh(self):
    helper_test_op([(45,65)], lambda x: x.cosh())
    helper_test_op([(45,65)], lambda x: x.cosh(), low=-300, high=-297, forward_only=True)
    helper_test_op([(45,65)], lambda x: x.cosh(), low=300, high=303, forward_only=True)

  def test_asinh(self):
    helper_test_op([(45,65)], lambda x: x.asinh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.asinh(), atol=1e-2, rtol=2e-2, grad_rtol=2e-2, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.asinh(), grad_atol=1e-6, low=300, high=303)

  def test_acosh(self):
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-3, grad_rtol=1e-2, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.acosh(), grad_atol=1e-6, low=300, high=303)

  def test_atanh(self):
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6)
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6, low=-300, high=-297)
    helper_test_op([(45,65)], lambda x: x.atanh(), grad_atol=1e-6, low=300, high=303)

  # # slow test
  # def test_max_pool2d(self):
  #   for ksz in [(3,3)]:
  #     with self.subTest(kernel_size=ksz):
  #       helper_test_op([(1,1,10,10)],
  #         lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
  #         lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_mulacc_with_zero_strides(self):
    helper_test_op(
      [],
      lambda: torch.tensor(1.0).reshape((1,1,1)).expand(2,4,3).mul(torch.tensor(1.0).reshape((1,1,1)).expand(2,4,3)).sum(-1),
      lambda: Tensor(1.0).reshape((1,1,1)).expand(2,4,3).mul(Tensor(1.0).reshape((1,1,1)).expand(2,4,3)).sum(-1),
      forward_only=True
    )
    a = [[1.,1.,1.,1.], [1.,1.,1.,1.]]
    b = [1.,1.,1.,1.]
    helper_test_op(
      [],
      lambda: torch.tensor(a).reshape((2,4,1)).expand(2,4,3).mul(torch.tensor(b).reshape((1,4,1)).expand(2,4,3)).sum([0,2]),
      lambda: Tensor(a).reshape((2,4,1)).expand(2,4,3).mul(Tensor(b).reshape((1,4,1)).expand(2,4,3)).sum([0,2]),
      forward_only=True
    )
    helper_test_op(
      [],
      lambda: torch.ones((1,2)).matmul(torch.ones((2,3))), lambda: Tensor.ones((1,2)).dot(Tensor.ones((2,3))),
      forward_only=True
    )

  @unittest.skipIf(IMAGE>0, "no 1d dot for images")
  def test_dot_1d(self):
    helper_test_op([(65), (65)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    helper_test_op([(65), (65,45)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    helper_test_op([(45,65), (65)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    helper_test_op([(8,45,65), (65)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    self.helper_test_exception([(4), (1,2)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                               expected=RuntimeError, forward_only=True)
    self.helper_test_exception([(2,1), (4)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                               expected=RuntimeError, forward_only=True)
    self.helper_test_exception([(1), (4)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                               expected=RuntimeError, forward_only=True)

  @slow_test
  def test_dot(self):
    helper_test_op([(45,65), (65,100)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    self.helper_test_exception([(2,4), (1,3)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                               expected=RuntimeError, forward_only=True)
    self.helper_test_exception([(2,1), (4,3)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                               expected=RuntimeError, forward_only=True)
    with self.assertRaises(RuntimeError):
      a = Tensor(3.14).half()
      a.matmul(a)

  def test_matmul_simple(self):
    helper_test_op([(4), (4,4)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_matmul(self):
    helper_test_op([(64), (64,99)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  @unittest.skipIf(IMAGE>0, "no batched matmul on images")
  def test_matmul_batched(self):
    helper_test_op([(3), (1,3,3,5)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  @unittest.skipIf(IMAGE>0, "no batched matmul on images")
  def test_matmul_batched_vector(self):
    helper_test_op([(4,3), (1,3,3,5)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_gemm_with_zeros_shape(self):
    helper_test_op([(8,8), (8,0)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=1e-7, rtol=1e-7, grad_atol=1e-7, grad_rtol=1e-7, forward_only=True)
    helper_test_op([(0,8), (8,8)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=1e-7, rtol=1e-7, grad_atol=1e-7, grad_rtol=1e-7, forward_only=True)
    helper_test_op([(0,8), (8,0)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=1e-7, rtol=1e-7, grad_atol=1e-7, grad_rtol=1e-7, forward_only=True)
    helper_test_op([(8,0), (0,8)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=1e-7, rtol=1e-7, grad_atol=1e-7, grad_rtol=1e-7, forward_only=True)
    helper_test_op([(0,0), (0,0)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=1e-7, rtol=1e-7, grad_atol=1e-7, grad_rtol=1e-7, forward_only=True)
    helper_test_op([(0), (0,8)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=1e-7, rtol=1e-7, grad_atol=1e-7, grad_rtol=1e-7, forward_only=True)
    helper_test_op([(0), (0)], lambda x,y: x.half().matmul(y.half()), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=1e-7, rtol=1e-7, grad_atol=1e-7, grad_rtol=1e-7, forward_only=True)

  def test_gemm_fp16(self):
    i = 64
    helper_test_op([(i,i), (i,i)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    # helper_test_op([(2,2), (2,2)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3)
    # shapes = [((4,4), (4,4)), ((33,33), (33,33)), ((34,34), (34,34)), ((65,33), (33,65)), ((394,394), (394,394)),
    #           ((1,8192), (8192,8192)), ((1,8192), (8192,8193)), ((1,8193), (8193,8192)), ((1,8193), (8193,8193)),
    #           ((1,768), (768,2048)), ((1,2048), (2048,2048)), ((1,4096), (4096,4096))]

  def test_gemm_fp16_shapes(self):
    for m,n,k in [(4,4,4), (8,8,8), (9,9,9), (32,32,32), (64,64,64), (128,128,128), (192,192,192), (256,256,256),
                  (1,1,65), (1,45,65), (45,1,65), (45,100,65), (64,99,64), (1,4,4), (1,99,64),
                  (4,8,16), (16,4,8), (8,32,4), (12,34,56), (50,10,20), (32,32,1)]:
      with self.subTest(m=m, n=n, k=k):
        helper_test_op([(m,k), (k,n)], lambda x,y: x.half().matmul(y.half()),
                       atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_gemm_2x2x1_manual(self):
    helper_test_op(None, lambda x,y: x.half().matmul(y.half()), lambda x,y: x.half()@y.half(),
                   vals=[np.array([[1], [3]], dtype=np.float32), np.array([[5, 6]], dtype=np.float32)],
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_small_gemm(self):
    helper_test_op([(8,8), (8,8)], lambda x,y: x.half().matmul(y.half()), lambda x,y: x.half()@y.half(),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_9_gemm(self):
    helper_test_op([(9,9), (9,9)], lambda x,y: x.half().matmul(y.half()), lambda x,y: x.half()@y.half(),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_small_gemm_padded(self):
    np.random.seed(0)
    vals = [np.pad(np.random.uniform(-2, 2, size=(9,9)).astype(np.float32), ((0,7),(0,7))),
            np.pad(np.random.uniform(-2, 2, size=(9,9)).astype(np.float32), ((0,7),(0,7)))]
    helper_test_op(None, lambda x,y: x.half().matmul(y.half()), lambda x,y: x.half()@y.half(), vals=vals,
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_small_gemm_range(self):
    helper_test_op(None, lambda x,y: x.half().matmul(y.half()), lambda x,y: x.half()@y.half(),
                   vals=[np.arange(0,64,dtype=np.float32).reshape(8,8), np.arange(64,128,dtype=np.float32).reshape(8,8)],
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_small_gemm_eye(self):
    helper_test_op(None, lambda x,y: x.half().matmul(y.half()), lambda x,y: x.half()@y.half(),
                   vals=[np.eye(8).astype(np.float32), np.eye(8).astype(np.float32)],
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)




  def test_gemm(self):
    helper_test_op([(64,64), (64,64)], lambda x,y: x.half().matmul(y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  @slow_test
  def test_big_gemm(self):
    helper_test_op([(256,256), (256,256)], lambda x,y: x.half().matmul(y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  @slow_test
  def test_broadcastdot(self):
    helper_test_op([(10,45,65), (65,45)], lambda x,y: x.half() @ y.half(), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    with self.assertRaises(RuntimeError):
      a = Tensor(3.14).half()
      b = Tensor.ones(3,3).half()
      a @ b

  @slow_test
  def test_multidot(self):
    helper_test_op([(10,45,65), (10,65,45)], lambda x,y: x.half() @ y.half(), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)
    helper_test_op([(3,3,45,65), (3,3,65,45)], lambda x,y: x.half() @ y.half(), lambda x,y: Tensor.dot(x.half(), y.half()),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_sum_simple(self):
    helper_test_op(None, lambda x: x.sum(), vals=[[1., 1.]], forward_only=True)

  def test_sum_full(self):
    helper_test_op([(16384)], lambda x: x.sum(), forward_only=True, atol=5e-2, rtol=1e-2)

  def test_sum_relu(self):
    helper_test_op([(3,4,5)], lambda x: x.relu().sum().relu(), forward_only=True, atol=5e-2, rtol=1e-2)

  def test_sum_tiny(self):
    helper_test_op([(4,2,2)], lambda x: x.sum(axis=(0,2)), forward_only=True, atol=5e-2, rtol=1e-2)

  def test_sum(self):
    helper_test_op([(45,3)], lambda x: x.sum(), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=3), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,3)), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(0,2)), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=(1,2)), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([(3,4,5,6)], lambda x: x.sum(axis=1, keepdim=True), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([()], lambda x: x.sum(), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([()], lambda x: x.sum(0), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([()], lambda x: x.sum(-1), forward_only=True, atol=5e-2, rtol=1e-2)
    helper_test_op([()], lambda x: x.sum(()), forward_only=True, atol=5e-2, rtol=1e-2)
    self.helper_test_exception([(3,4,5,6)], lambda x: x.sum(5), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum(1), expected=IndexError)
    self.helper_test_exception([()], lambda x: x.sum((1,)), expected=IndexError)

  def test_sum_dtype_arg(self):
    helper_test_op([(45,3)], lambda x: x.sum(), lambda x: x.sum(dtype=dtypes.float32), forward_only=True, atol=5e-2, rtol=1e-2)
    if is_dtype_supported(dtypes.float64):
      helper_test_op([(45,3)], lambda x: x.sum(dtype=torch.float64), lambda x: x.sum(dtype=dtypes.float64), forward_only=True, atol=5e-2, rtol=1e-2)
    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).sum(dtype="")

  def test_sum_with_zeros_shape(self):
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(0,)), forward_only=True)
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(1,)), forward_only=True)
    helper_test_op([(4, 0)], lambda x: x.sum(axis=(0,1)), forward_only=True)

  def test_prod(self):
    helper_test_op(None, lambda x: x.prod(), vals=[[1.0, 2.0, 3.0]], forward_only=True)
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=3), lambda x: x.prod(axis=3), forward_only=True)
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1), lambda x: x.prod(axis=1), forward_only=True)
    helper_test_op([(3,4,5,6)], lambda x: x.prod(dim=1, keepdim=True), lambda x: x.prod(axis=1, keepdim=True), forward_only=True)
    helper_test_op([()], lambda x: x.prod(), forward_only=True)
    helper_test_op([()], lambda x: x.prod(0), forward_only=True)
    helper_test_op([()], lambda x: x.prod(-1), forward_only=True)

  def test_prod_dtype_arg(self):
    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).prod(dtype="")

  def test_const_reduce(self):
    helper_test_op([(3,3)], lambda x: torch.full_like(x, 2).sum(), lambda x: (x.full_like(2)).sum(), forward_only=True)
    helper_test_op([(3,3)], lambda x: torch.full_like(x, 2).prod(), lambda x: (x.full_like(2)).prod(), forward_only=True)
    helper_test_op([(3,3)], lambda x: torch.full_like(x, 2).max(), lambda x: (x.full_like(2)).max(), forward_only=True)

  def test_mean(self):
    helper_test_op([(3,4,5,6)], lambda x: x.mean(), forward_only=True, atol=1e-2, rtol=1e-2)
    helper_test_op([()], lambda x: x.mean(), forward_only=True, atol=1e-2, rtol=1e-2)

  def test_mean_zero_axis(self):
    helper_test_op([(1,0,3,0,5)], lambda x: x.mean(axis=(1,3)), forward_only=True, atol=1e-2, rtol=1e-2)

  def test_softmax(self):
    helper_test_op([(45,65)], torch.nn.Softmax(dim=1), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], torch.nn.Softmax(dim=0), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.Softmax(dim=0), Tensor.softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.Softmax(dim=-1), Tensor.softmax, atol=1e-7, grad_atol=1e-7)

  def test_softmax_other_axis(self):
    helper_test_op([(10,10,10)], lambda x: x.softmax(0), atol=1e-7, grad_atol=2e-7)
    helper_test_op([(10,10,10)], lambda x: x.softmax(1), atol=1e-7, grad_atol=2e-7)
    helper_test_op([(10,10,10)], lambda x: x.softmax(2), atol=1e-7, grad_atol=2e-7)

  def test_softmax_argmax(self):
    helper_test_op([(45,65)], lambda x: x.softmax(0).argmax().type(torch.int32),
                              lambda x: x.softmax(0).argmax(), forward_only=True, atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: x.softmax(1).argmax().type(torch.int32),
                              lambda x: x.softmax(1).argmax(), forward_only=True, atol=1e-7, grad_atol=1e-7)

  def test_log_softmax(self):
    helper_test_op([(45,65)], torch.nn.LogSoftmax(dim=1), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], torch.nn.LogSoftmax(dim=0), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.LogSoftmax(dim=0), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], torch.nn.LogSoftmax(dim=-1), Tensor.log_softmax, atol=1e-7, grad_atol=1e-7)

  def test_log_softmax_other_axis(self):
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: x.log_softmax(2), atol=1e-7, grad_atol=1e-7)

  def test_normalize(self):
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x), lambda x: x.normalize(), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, dim=0), lambda x: x.normalize(dim=0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(10,10,10)], lambda x: torch.nn.functional.normalize(x, dim=2), lambda x: x.normalize(dim=2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=1), lambda x: x.normalize(p=1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=3, dim=0), lambda x: x.normalize(p=3, dim=0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=0), lambda x: x.normalize(p=0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.nn.functional.normalize(x, p=-1), lambda x: x.normalize(p=-1), atol=1e-7, grad_atol=1e-7)

  def test_logsumexp(self):
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=0, keepdim=True), lambda x: x.logsumexp(0, True), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=1), lambda x: x.logsumexp(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logsumexp(x, dim=1, keepdim=True), lambda x: x.logsumexp(1, True), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6)], lambda x: torch.logsumexp(x, dim=2), lambda x: x.logsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logsumexp(x, dim=2), lambda x: x.logsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logsumexp(x, dim=3), lambda x: x.logsumexp(3), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logsumexp(x, dim=0), lambda x: x.logsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logsumexp(x, dim=-1), lambda x: x.logsumexp(-1), atol=1e-7, grad_atol=1e-7)

  @slow_test
  def test_logcumsumexp(self):
    helper_test_op([(45,65)], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45,65)], lambda x: torch.logcumsumexp(x, dim=1), lambda x: x.logcumsumexp(1), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6)], lambda x: torch.logcumsumexp(x, dim=2), lambda x: x.logcumsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logcumsumexp(x, dim=2), lambda x: x.logcumsumexp(2), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(6,6,6,6)], lambda x: torch.logcumsumexp(x, dim=3), lambda x: x.logcumsumexp(3), atol=1e-7, grad_atol=1e-7)
    helper_test_op([(45)], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(0), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(), atol=1e-7, grad_atol=1e-7)
    helper_test_op([()], lambda x: torch.logcumsumexp(x, dim=-1), lambda x: x.logcumsumexp(-1), atol=1e-7, grad_atol=1e-7)

  def test_logcumsumexp_numerical(self):
    helper_test_op(None, lambda x: torch.logcumsumexp(x, dim=0), lambda x: x.logcumsumexp(), atol=1e-7, grad_atol=1e-7, vals=[[0.0, 100.0]])

  def test_full_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.full_like(b, 4), lambda: Tensor.full_like(a, 4), forward_only=True)

  def test_full(self):
    helper_test_op([], lambda: torch.full((45,65), 4, dtype=torch.int32), lambda: Tensor.full((45,65), 4), forward_only=True)

  def test_negative_dims(self):
    creation_methods = [
      Tensor.empty, Tensor.rand, Tensor.zeros, Tensor.ones, Tensor.randn, Tensor.randint,
      Tensor.normal, Tensor.uniform, Tensor.scaled_uniform, Tensor.glorot_uniform]
    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 2)
      with self.assertRaises(ValueError): method((2, -3))
      with self.assertRaises(ValueError): method((2, -3, 0))

  def test_negative_dims_full(self):
    with self.assertRaises(ValueError): Tensor.full((-3,), 2)
    with self.assertRaises(ValueError): Tensor.full((2, -3), 4)
    with self.assertRaises(ValueError): Tensor.full((2, -3, 0), 4)

  def test_negative_dims_eye(self):
    with self.assertRaises(ValueError): Tensor.eye(-3, 3)
    with self.assertRaises(ValueError): Tensor.eye(3, -3)
    with self.assertRaises(ValueError): Tensor.eye(-3, -3)

  def test_negative_dims_kaiming(self):
    for method in [Tensor.kaiming_uniform, Tensor.kaiming_normal]:
      with self.assertRaises(ValueError): method(-3, 3)
      with self.assertRaises(ValueError): method((-3, 3), 3)
      with self.assertRaises(ValueError): method((-3, -3), 3)

  def test_zeros(self):
    helper_test_op([], lambda: torch.zeros(45,65), lambda: Tensor.zeros(45,65), forward_only=True)
    helper_test_op([], lambda: torch.zeros([45,65]), lambda: Tensor.zeros([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.zeros([]), lambda: Tensor.zeros([]), forward_only=True)

  def test_zeros_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.zeros_like(b), lambda: Tensor.zeros_like(a), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.zeros_like(b), lambda: Tensor.zeros_like(a), forward_only=True)

  def test_empty_0(self):
    helper_test_op([], lambda: torch.empty(45,65)*0/0, lambda: Tensor.empty(45,65)*0/0, forward_only=True)

  def test_ones(self):
    helper_test_op([], lambda: torch.ones(45,65), lambda: Tensor.ones(45,65), forward_only=True)
    helper_test_op([], lambda: torch.ones([45,65]), lambda: Tensor.ones([45,65]), forward_only=True)
    helper_test_op([], lambda: torch.ones([]), lambda: Tensor.ones([]), forward_only=True)

  def test_ones_like(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.float32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
    helper_test_op([], lambda: torch.ones_like(b), lambda: Tensor.ones_like(a), forward_only=True)

    a = Tensor([[1,2,3],[4,5,6]], dtype=dtypes.int32)
    b = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int32)
    helper_test_op([], lambda: torch.ones_like(b), lambda: Tensor.ones_like(a), forward_only=True)

  def test_eye(self):
    helper_test_op([], lambda: torch.eye(10), lambda: Tensor.eye(10), forward_only=True)
    helper_test_op([], lambda: torch.eye(3, 5), lambda: Tensor.eye(3, 5), forward_only=True)
    helper_test_op([], lambda: torch.eye(5, 3), lambda: Tensor.eye(5, 3), forward_only=True)
    helper_test_op([], lambda: torch.eye(1), lambda: Tensor.eye(1), forward_only=True)
    helper_test_op([], lambda: torch.eye(0), lambda: Tensor.eye(0), forward_only=True)

  def test_split(self):
    def tensor(s): return torch.arange(math.prod(s), dtype=torch.int32).reshape(s), Tensor.arange(math.prod(s)).reshape(s)
    test_cases = [
      (tensor((10,)),       5, {}),
      (tensor((10,)), [1,4,5], {}),
      (tensor((10,)),       3, {}),
      (tensor((3,4,)),      1, {}),
      (tensor((3,4,)),      1, {'dim':1}),
      (tensor((4,4,)),  [2,2], {}),
      (tensor((4,4,)),  [2,2], {'dim':1}),
      (tensor((10000,)), 2500, {}),
    ]
    for (tor, ten), sizes, args in test_cases:
      tor_splits, ten_splits = tor.split(sizes, **args), ten.split(sizes, **args)
      assert len(tor_splits) == len(ten_splits)
      for tor_chunk, ten_chunk in zip(tor_splits, ten_splits):
        helper_test_op([], lambda: tor_chunk, lambda: ten_chunk, forward_only=True)

  def test_chunk(self):
    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(6, 1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(6, 0)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(6, 0)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 1).chunk(3, -1)
    ten = Tensor.arange(13).repeat((8, 1)).chunk(3, -1)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

    tor = torch.arange(13, dtype=torch.int32).repeat(8, 3, 3).chunk(3, -2)
    ten = Tensor.arange(13).repeat((8, 3, 3)).chunk(3, -2)
    assert len(tor) == len(ten)
    for i in range(len(tor)):
      helper_test_op([], lambda: tor[i], lambda: ten[i], forward_only=True)

  def test_unfold(self):
    helper_test_op([(8,)], lambda x: x.unfold(0, 2, 1))
    helper_test_op([(8,)], lambda x: x.unfold(0, 2, 2))
    helper_test_op([(8,)], lambda x: x.unfold(0, 7, 3))
    helper_test_op([(3,3,3)], lambda x: x.unfold(2, 2, 8))
    helper_test_op([(3,3,3)], lambda x: x.unfold(1, 0, 8))
    helper_test_op([(3,3,3,3,3)], lambda x: x.unfold(-1, 2, 2))

    self.helper_test_exception([(8,)], lambda x: x.unfold(0, 9, 3), expected=RuntimeError)
    self.helper_test_exception([(8,)], lambda x: x.unfold(1, 8, 3), expected=IndexError)
    self.helper_test_exception([(8,)], lambda x: x.unfold(0, 9, 3), expected=RuntimeError)
    self.helper_test_exception([(8,)], lambda x: x.unfold(0, 1, -1), expected=RuntimeError)

  def test_meshgrid(self):
    x, xt = torch.tensor([0.,1.,2.], requires_grad=True), Tensor([0.,1.,2.], requires_grad=True)
    y, yt = torch.tensor([3.,4.,5.,6.], requires_grad=True), Tensor([3.,4.,5.,6.], requires_grad=True)
    z, zt = torch.tensor([7.,8.,9.], requires_grad=True), Tensor([7.,8.,9.], requires_grad=True)
    for indexing in ("ij", "xy"):
      tor = torch.meshgrid(x, indexing=indexing)
      ten = xt.meshgrid(indexing=indexing)
      self.assertEqual(len(tor), len(ten))
      for tor_i, ten_i in zip(tor, ten):
        helper_test_op([], lambda: tor_i, lambda: ten_i)

  def test_sd_big_conv(self):
    helper_test_op([(1,256,64,64), (512,256,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-3)

  def test_large_bs_conv(self):
    helper_test_op([(4096,3,3,3), (1,3,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-3)

  def test_large_ic_conv(self):
    helper_test_op([(1,2048,3,3), (1,2048,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w))

  def test_biased_conv2d(self):
    C = 8
    helper_test_op([(1,C,5,5), (C,C,1,1), (C,)],
      lambda x,w,b: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w,b).relu(),w,b),
      lambda x,w,b: Tensor.conv2d(x,w,b).relu().conv2d(w,b))

  def test_simple_conv2d(self):
    helper_test_op([(1,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  def test_simple_conv2d_bias(self):
    helper_test_op([(1,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv2d(x,w,b),
      lambda x,w,b: Tensor.conv2d(x,w,b), grad_rtol=1e-5)

  @slow_test
  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv3d(self):
    helper_test_op([(1,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  @slow_test
  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_padded_conv3d(self):
    helper_test_op([(1,4,5,5,5), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv3d(x,w,padding=1),
      lambda x,w: Tensor.conv2d(x,w,padding=[1,1,1,1,1,1]), grad_rtol=1e-5)

  def test_simple_conv2d_m4(self):
    helper_test_op([(1,16,9,9), (16,16,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), atol=1e-05, grad_rtol=1e-5)

  def test_simple_conv2d_1x1(self):
    np.random.seed(0)
    x = np.random.uniform(-2, 2, size=(1,4,9,9)).astype(np.float16)
    w = np.random.uniform(-2, 2, size=(4,4,1,1)).astype(np.float16)
    expected = torch.nn.functional.conv2d(torch.tensor(x.astype(np.float32)), torch.tensor(w.astype(np.float32))).numpy()
    np.testing.assert_allclose(Tensor(x).conv2d(Tensor(w)).realize().numpy(), expected, atol=5e-2, rtol=5e-2)

  def test_simple_conv2d_1x1_m4(self):
    np.random.seed(0)
    x = np.random.uniform(-2, 2, size=(1,16,32,32)).astype(np.float16)
    w = np.random.uniform(-2, 2, size=(16,16,1,1)).astype(np.float16)
    expected = torch.nn.functional.conv2d(torch.tensor(x.astype(np.float32)), torch.tensor(w.astype(np.float32))).numpy()
    with Context(TC=0):
      np.testing.assert_allclose(Tensor(x).conv2d(Tensor(w)).realize().numpy(), expected, atol=5e-2, rtol=5e-2)

  @slow_test
  def test_nested_conv2d(self):
    helper_test_op([(1,32,9,9), (32,32,3,3), (32,32,3,3)],
      lambda x,w1,w2: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,w1).relu(), w2),
      lambda x,w1,w2: x.conv2d(w1).relu().conv2d(w2))

  def test_simple_conv2d_nhwc(self):
    helper_test_op([(2,9,9,10), (3,3,10,20)],
      lambda x,w: torch.nn.functional.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)),
      lambda x,w: Tensor.conv2d(x.permute(0,3,1,2),w.permute(3,2,0,1)), atol=1e-5, grad_rtol=1e-5)

  def test_simple_conv2d_batched(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  def test_simple_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w),
      lambda x,w: Tensor.conv_transpose2d(x,w), grad_rtol=1e-5)

  def test_bias_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3), (4,)],
      lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b),
      lambda x,w,b: Tensor.conv_transpose2d(x,w,b), grad_rtol=1e-5)

  def test_grouped_conv_transpose2d(self):
    helper_test_op([(2,4,9,9), (4,4,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose2d(x,w,groups=2),
      lambda x,w: Tensor.conv_transpose2d(x,w,groups=2), grad_rtol=1e-5)

  @slow_test
  def test_padded_conv_transpose2d(self):
    for padding in [(1,2), (2,1), 2, 1, 0]:
      helper_test_op([(2,4,9,9), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w,padding=padding),
        lambda x,w: Tensor.conv_transpose2d(x,w,padding=padding), grad_rtol=1e-5)
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w: torch.nn.functional.conv_transpose2d(x,w,padding=(1,1,1)),
                   lambda x,w: Tensor.conv_transpose2d(x,w,padding=(1,1,1)), expected=(RuntimeError, ValueError))

  @slow_test
  def test_dilated_conv_transpose2d(self):
    for dilation in [(1,2), (2,1), 2, 1]:
      helper_test_op([(2,4,9,9), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w,dilation=dilation),
        lambda x,w: Tensor.conv_transpose2d(x,w,dilation=dilation), grad_rtol=1e-5)

  def test_strided_conv_transpose2d(self):
    for stride in [(2,1), (1,2), 1]:
      helper_test_op([(2,4,4,5), (4,4,3,3)],
        lambda x,w: torch.nn.functional.conv_transpose2d(x,w, stride=stride),
        lambda x,w: Tensor.conv_transpose2d(x,w,stride=stride), atol=1e-5, grad_rtol=1e-5)

  @slow_test
  def test_output_padded_conv_transpose2d(self):
    for output_padding, stride in [((1,1), (2,3)), ((2,1), (3,2))]:
      helper_test_op([(2,4,6,5), (4,4,3,3),(4,)],
        lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride),
        lambda x,w,b: Tensor.conv_transpose2d(x,w,b,output_padding=output_padding,stride=stride), grad_rtol=1e-5)

  @slow_test
  @unittest.skipIf(IMAGE>0, "no conv3d on images")
  def test_simple_conv_transpose3d(self):
    helper_test_op([(2,4,9,9,9), (4,4,3,3,3)],
      lambda x,w: torch.nn.functional.conv_transpose3d(x,w),
      lambda x,w: Tensor.conv_transpose2d(x,w), grad_rtol=1e-5)

  @unittest.skipIf((IMAGE>0), "no conv1d on images")
  def test_conv1d(self):
    for bs in [1,8]:
      for cin in [1,3]:
        for H in [1,2,5]:
          for groups in [1,3] if cin == 3 and H == 5 else [1]:
            with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H):
              helper_test_op([(bs,cin,11), (6,cin//groups,H)],
                lambda x,w: torch.nn.functional.conv1d(x,w,groups=groups),
                lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_simple_padding_conv1d(self):
    bs = 6
    cin = 2
    groups = 1
    H = 5
    p = (1,1)
    helper_test_op([(bs,cin,11), (6,cin//groups,H)],
      lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w),
      lambda x,w: Tensor.conv2d(x,w,padding=p))

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_strided_conv1d_simple(self):
    bs, H = 2, 3
    helper_test_op([(bs,1,5), (1,1,H)],
      lambda x,w: torch.nn.functional.conv1d(x,w,stride=2),
      lambda x,w: Tensor.conv2d(x,w,stride=2))

  @unittest.skipIf(IMAGE>0, "no conv1d on images")
  def test_asymmetric_padding_conv1d(self):
    for p in [(0,1), (2,1), (2,0)]:
      with self.subTest(p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n), (1,1,k)],
              lambda x,w: torch.nn.functional.conv1d(torch.nn.functional.pad(x, p),w),
              lambda x,w: Tensor.conv2d(x,w,padding=p))

  def _test_conv2d(self, bs=1, cin=1, cout=6):
    for H in [2,3]:
      for W in [1,3,5]:
        for groups in [1,3] if cin == 3 and cout == 6 and H == 3 and W == 3 else [1]:
          with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H, width=W):
            helper_test_op([(bs,cin,5,7), (cout,cin//groups,H,W)],
              lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
              lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)
  def test_conv2d(self): self._test_conv2d(bs=1, cin=3)
  @slow_test
  def test_conv2d_bs_4_cin_3(self): self._test_conv2d(bs=4, cin=3, cout=2)
  def test_conv2d_bs_1_cin_1(self): self._test_conv2d(bs=1, cin=1)
  @slow_test
  def test_conv2d_bs_4_cin_1(self): self._test_conv2d(bs=4, cin=1)

  def test_conv2d_errors(self):
    self.helper_test_exception([(1,1,6,7), (6,1,3,3)],
                               lambda x,w:torch.nn.functional.conv2d(x,w,dilation=3),
                               lambda x,w: Tensor.conv2d(x,w,dilation=3), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w:torch.nn.functional.conv2d(x,w), lambda x,w: Tensor.conv2d(x,w),
                               expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,16,2,2), (32,16,3,3)], lambda x,w:torch.nn.functional.conv2d(x,w,padding=(1,1,1)),
                               lambda x,w: Tensor.conv2d(x,w,padding=(1,1,1)), expected=(RuntimeError, ValueError))

  @slow_test
  def test_large_input_conv2d(self):
    bs = 4
    cin = 16
    groups = 1
    H = 5
    W = 2
    helper_test_op([(bs,cin,64,64), (6,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), atol=1e-4, grad_atol=3e-4, grad_rtol=1e-4)

  def test_simple_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 1
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  def test_medium_grouped_conv2d(self):
    bs = 1
    groups = 2
    rcout = 2
    cin = 2
    helper_test_op([(bs,groups*cin,1,1), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  def test_depthwise_conv2d(self):
    bs = 1
    groups = 32
    rcout = 1
    cin = 1
    helper_test_op([(bs,groups*cin,32,32), (groups*rcout,cin,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  def test_grouped_conv2d(self):
    bs = 4
    groups = 5
    rcout = 7
    cin = 3
    helper_test_op([(bs,groups*cin,5,5), (groups*rcout,cin,3,3)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  def test_fancy_conv2d(self):
    bs = 2
    cin = 3
    cout = 1
    groups = 3
    H,W = 3,3
    helper_test_op([(bs,cin,11,28), (groups*cout,cin//groups,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,groups=groups),
      lambda x,w: Tensor.conv2d(x,w,groups=groups), grad_rtol=1e-5)

  @slow_test
  def test_strided_conv2d_simple(self):
    bs,H,W = 2,3,1
    helper_test_op([(bs,1,5,1), (1,1,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,stride=2),
      lambda x,w: Tensor.conv2d(x,w,stride=2))

  @unittest.skipUnless(Device.DEFAULT == "CPU" and DEV.renderer == "LLVM", "DEVECTORIZE=0 only for LLVM")
  def test_strided_conv2d_simple_vec(self):
    with Context(DEVECTORIZE=0): self.test_strided_conv2d_simple()

  @slow_test
  def test_strided_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    with self.subTest(stride := 2):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=2),
        lambda x,w: Tensor.conv2d(x,w,stride=stride))
    with self.subTest(stride := (2,1)):
      helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
        lambda x,w: torch.nn.functional.conv2d(x,w,stride=stride),
        lambda x,w: Tensor.conv2d(x,w,stride=(2,1)))

  def test_negative_padding_conv2d(self):
    n,k = 10, 3
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:-1, 1:-1],w),
      lambda x,w: Tensor.conv2d(x,w,padding=-1))
    helper_test_op([(1,1,n,n), (1,1,k,k)],
      lambda x,w: torch.nn.functional.conv2d(x[:, :, 1:, 1:],w),
      lambda x,w: Tensor.conv2d(x,w,padding=(-1,0,-1,0)))

  def test_simple_padding_conv2d(self):
    p = (1,1,1,1)
    helper_test_op(None,
      lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w),
      lambda x,w: Tensor.conv2d(x,w,padding=p), vals=[[[[[2.,3.]]]], [[[[1.]]]]])

  def test_asymmetric_padding_conv2d(self):
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(p):
        for n in [3,4]:
          for k in [2]:
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w),
              lambda x,w: Tensor.conv2d(x,w,padding=p))
            helper_test_op([(1,1,n,n), (1,1,k,k)],
              lambda x,w: torch.nn.functional.conv2d(torch.nn.functional.pad(x, p),w),
              lambda x,w: Tensor.conv2d(x,w,padding=p))

  def test_padded_conv2d_p21(self):
    bs,cin,H,W,padding = 4, 3, 3, 3, (2,1)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding),
      lambda x,w: Tensor.conv2d(x,w,padding=padding))

  def test_padded_conv2d_p22(self):
    bs,cin,H,W,padding = 4, 3, 3, 3, (2,2)
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding),
      lambda x,w: Tensor.conv2d(x,w,padding=padding))

  def test_padded_conv2d_1x1(self):
    bs,cin,H,W,padding = 4, 3, 1, 1, 2
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding),
      lambda x,w: Tensor.conv2d(x,w,padding=padding))

  def test_padded_conv2d_bs1(self):
    bs,cin,H,W,padding = 1, 3, 3, 3, 1
    helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
      lambda x,w: torch.nn.functional.conv2d(x,w,padding=padding),
      lambda x,w: Tensor.conv2d(x,w,padding=padding))

  def test_dilated_conv2d(self):
    bs = 4
    cin = 3
    H,W = 3,3
    for d in [2, (2,1)]:
      with self.subTest(dilation := d):
        helper_test_op([(bs,cin,11,28), (4,cin,H,W)],
          lambda x,w: torch.nn.functional.conv2d(x,w,dilation=dilation),
          lambda x,w: Tensor.conv2d(x,w,dilation=dilation))

  def test_max_pool2d_simple(self):
    ksz = (2,2)
    helper_test_op([(1,1,2,3)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
      lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  @slow_test
  def test_max_pool2d(self):
    for ksz in [(2,2), (3,3), 2, 3, (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([(32,2,11,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  @slow_test
  def test_max_pool2d_padding(self):
    for ksz in [(2,2), (3,3), 2, 3, (3,2)]:
      for p in [1, (1,0), (0,1)]:
        with self.subTest(kernel_size=ksz, padding=p):
          helper_test_op([(4,2,11,28)],
            lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz, padding=p),
            lambda x: Tensor.max_pool2d(x, kernel_size=ksz, padding=p))
    self.helper_test_exception([(4,2,110,28)], lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), padding=(1,1,1)),
                   lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), padding=(1,1,1)), expected=(RuntimeError, ValueError))

  @slow_test
  def test_max_pool2d_asymmetric_padding(self):
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding=p):
        helper_test_op([(4,2,111,28)],
          lambda x: torch.nn.functional.max_pool2d(torch.nn.functional.pad(x, p, value=float("-inf")), kernel_size=(5,5)),
          lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), padding=p))

  @slow_test
  def test_max_pool2d_padding_int(self):
    ksz = (2,2)
    helper_test_op([(4,2,11,28)],
      lambda x: torch.nn.functional.max_pool2d(x.int(), kernel_size=ksz, padding=1),
      lambda x: Tensor.max_pool2d(x.int(), kernel_size=ksz, padding=1), forward_only=True)

  @slow_test
  def test_max_pool2d_bigger_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(4,2,11,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=stride))

  @slow_test
  def test_max_pool2d_bigger_stride_dilation(self):
    for stride, dilation in zip([(2,3), (3,2), 2, 3, 4], [(3,2), (2,3), 2, 3, 6]):
      with self.subTest(stride=stride):
        helper_test_op([(4,2,11,28)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=stride, dilation=dilation),
          lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=stride, dilation=dilation))

  def test_max_pool2d_unit_stride(self):
    helper_test_op([(3, 2, 17, 14)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=1),
      lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=1))

  @slow_test
  def test_max_pool2d_smaller_stride(self):
    for stride in [(2,3), (3,2), 2, 3]:
      with self.subTest(stride=stride):
        helper_test_op([(3, 2, 17, 14)],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), stride=stride),
          lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), stride=stride))

  @slow_test
  def test_max_pool2d_dilation(self):
    for dilation in [(2, 3), (3, 2), 2, 3]:
      helper_test_op([(3, 2, 17, 14)],
        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(5,5), dilation=dilation),
        lambda x: Tensor.max_pool2d(x, kernel_size=(5,5), dilation=dilation))

  def test_max_pool2d_ceil_mode(self):
    shape = (1,1,6,6)
    for ksz in [(3,3), 3, (3,2), 4]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True),
          lambda x: Tensor.max_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True))

  def test_max_pool2d_ceil_mode_output_size_reduce_by_one(self):
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True))

  def test_max_pool2d_return_indices(self):
    helper_test_op([(2,3,6,6)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), return_indices=True)[1], forward_only=True)
    helper_test_op([(1,1,10,10)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,2), dilation=(2,3), return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,2), dilation=(2,3), return_indices=True)[1], forward_only=True)
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,3), padding=1, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), padding=1, return_indices=True)[1], forward_only=True)
    helper_test_op([(1, 1, 7, 7)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), ceil_mode=True, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), ceil_mode=True, return_indices=True)[1],
      forward_only=True)
    helper_test_op([(1,1,12,13)],
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(12, 13), return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(12, 13), return_indices=True)[1],
      forward_only=True)
    helper_test_op(None,
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(3,3), stride=1, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=1, return_indices=True)[1],
      vals=[[[[[1]*6]*6]]], forward_only=True)
    helper_test_op(None,
      lambda x: torch.nn.functional.max_pool2d(x, kernel_size=(2,2), stride=1, return_indices=True)[1].type(torch.int32),
      lambda x: Tensor.max_pool2d(x, kernel_size=(2,2), stride=1, return_indices=True)[1],
      vals=[[[[[1,2]*3]*6]]], forward_only=True)

  @slow_test
  def test_max_unpool2d(self):
    args = {"kernel_size":(5,5), "stride":(6,5)}
    helper_test_op([(8,3,50,50)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args), **args),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args), **args), forward_only=True)
    args = {"kernel_size":(3,3), "stride":(6,7), "padding":1}
    helper_test_op([(8,3,30,30)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, return_indices=True, **args), **args, output_size=(30,30)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, return_indices=True, **args), **args, output_size=(30,30)), forward_only=True)
    helper_test_op([(1,3,7,6)],
      lambda x: torch.nn.functional.max_unpool2d(*torch.nn.functional.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                                 kernel_size=(2,2), output_size=(99,99,7,6)),
      lambda x: Tensor.max_unpool2d(*Tensor.max_pool2d(x, kernel_size=(2,2), return_indices=True),
                                    kernel_size=(2,2), output_size=(99,99,7,6)), forward_only=True)

  def test_max_unpool2d_inf(self):
    data = [[[[math.inf, -math.inf, math.nan], [1.0, 2.0, 3.0]]]]
    ksz = (2,2)
    helper_test_op((),
      lambda: torch.nn.functional.max_unpool2d(
        *torch.nn.functional.max_pool2d(torch.tensor(data), kernel_size=ksz, return_indices=True),
        kernel_size=ksz
      ),
      lambda: Tensor.max_unpool2d(
        *Tensor.max_pool2d(Tensor(data), kernel_size=ksz, return_indices=True),
        kernel_size=ksz
      ),
      forward_only=True)

  @slow_test
  def test_avg_pool2d(self):
    shape = (32,2,11,28)
    for ksz in [(2,2), (3,3), (3,2), (5,5), (5,1)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz), rtol=1e-5)
    helper_test_op([(1,1,8,8)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(1,2), padding=(0,1), stride=(5,1)), rtol=1e-5)

  @slow_test
  def test_avg_pool2d_padding(self):
    shape = (32,2,11,28)
    for ksz in [(2,2), (3,3), 2, 3, (3,2)]:
      for p in [1, (1,0), (0,1)]:
        with self.subTest(kernel_size=ksz, padding=p):
          helper_test_op([shape],
            lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=p),
            lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=p), rtol=1e-5)
    with self.assertRaises(ValueError):
      Tensor.avg_pool2d(Tensor.randn((32,2,11,28)), kernel_size=(2,2), padding=(1,1,1))

  @slow_test
  def test_avg_pool2d_padding_not_counted(self):
    shape = (32,2,11,28)
    for ksz in [(2,2), (3,3), 2, 3, (3,2)]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, count_include_pad=False),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, count_include_pad=False), rtol=1e-5)

  def test_avg_pool2d_ceil_mode(self):
    shape = (1,1,6,6)
    for ksz in [(3,3), 3, (3,2), 4]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True), rtol=1e-5)

  def test_avg_pool2d_ceil_mode_padding_not_counted(self):
    shape = (1,1,6,6)
    for ksz in [(3,3), 3, (3,2), 4]:
      with self.subTest(kernel_size=ksz):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True, count_include_pad=False),
          lambda x: Tensor.avg_pool2d(x, kernel_size=ksz, padding=1, stride=3, ceil_mode=True, count_include_pad=False), rtol=1e-5)

  def test_avg_pool2d_ceil_mode_output_size_reduce_by_one(self):
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True))

  def test_avg_pool2d_ceil_mode_include_pad_output_size_reduce_by_one(self):
    helper_test_op([(1,1,5,5)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True, count_include_pad=True),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(3,3), stride=3, padding=1, ceil_mode=True, count_include_pad=True))

  def test_avg_pool2d_asymmetric_padding(self):
    shape = (32,2,11,28)
    for p in [(0,1,0,1), (2,1,2,1), (2,0,2,1)]:
      with self.subTest(padding=p):
        helper_test_op([shape],
          lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(5,5), padding=1),
          lambda x: Tensor.avg_pool2d(x, kernel_size=(5,5), padding=1), rtol=1e-5)
    self.helper_test_exception([shape], lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(2,2), padding=(1,1,1)),
                               lambda x: Tensor.avg_pool2d(x, kernel_size=(2,2), padding=(1,1,1)), expected=(RuntimeError, ValueError))

  def test_global_avg_pool2d(self):
    helper_test_op([(32,2,11,28)],
      lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=(11,28)),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(11,28)), rtol=1e-5)

  def test_avg_pool3d(self):
    helper_test_op([(1,1,16,16,16)],
      lambda x: torch.nn.functional.avg_pool3d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
      lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8,8), stride=5, padding=1, count_include_pad=False),
      atol=1e-6, rtol=1e-5, forward_only=True)

  @unittest.skip("needs Rockchip interpolate template support")
  def test_interpolate_linear(self):
    for in_sz, out_sz in [((52,),(29,)), ((29,),(52,))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="linear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"))

  @unittest.skip("needs Rockchip interpolate template support")
  def test_interpolate_linear_corners_aligned(self):
    for in_sz, out_sz in [((52,),(29,)), ((29,),(52,))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="linear", align_corners=True),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear", align_corners=True))

  @unittest.skip("needs Rockchip interpolate template support")
  def test_interpolate_nearest(self, mode="nearest"):
    for in_sz, out_sz in [((13,),(9,)), ((9,),(13,))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode=mode),
        lambda x: Tensor.interpolate(x, size=out_sz, mode=mode))
    for in_sz, out_sz in [((13,10),(9,11)), ((13,9),(11,10)), ((9,11),(10,13))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode=mode),
        lambda x: Tensor.interpolate(x, size=out_sz, mode=mode))
    for in_sz, out_sz in [((5,2,8),(3,6,4))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode=mode),
        lambda x: Tensor.interpolate(x, size=out_sz, mode=mode))

  @unittest.skip("needs Rockchip interpolate template support")
  def test_interpolate_nearest_exact(self): self.test_interpolate_nearest("nearest-exact")

  @unittest.skip("needs Rockchip interpolate template support")
  @slow_test
  def test_interpolate_bilinear(self):
    for in_sz, out_sz in [((12,20),(9,31)), ((12,9),(31,20)), ((9,31),(20,12))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="bilinear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"), atol=1e-4)

  @unittest.skip("needs Rockchip interpolate template support")
  @slow_test
  def test_interpolate_bilinear_corners_aligned(self):
    for in_sz, out_sz in [((12,20),(9,31)), ((12,9),(31,20)), ((9,31),(20,12))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="bilinear", align_corners=True),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear", align_corners=True), atol=1e-4)

  @unittest.skip("needs Rockchip interpolate template support")
  def test_interpolate_trilinear(self):
    for in_sz, out_sz in [((5,2,8),(3,6,4))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="trilinear"),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear"), atol=1e-4)

  @unittest.skip("needs Rockchip interpolate template support")
  def test_interpolate_trilinear_corners_aligned(self):
    for in_sz, out_sz in [((5,2,8),(3,6,4))]:
      helper_test_op([(2,3)+in_sz],
        lambda x: torch.nn.functional.interpolate(x, size=out_sz, mode="trilinear", align_corners=True),
        lambda x: Tensor.interpolate(x, size=out_sz, mode="linear", align_corners=True), atol=1e-4)

  @unittest.skip("needs Rockchip strided copy/movement template support; closest dump is ~/npu/ops_rknn/rknn copy.gdb")
  def test_multicat(self):
    for dim in range(-1, 2):
      helper_test_op([(45,65), (45,65), (45,65)], lambda x,y,z: torch.cat((x,y,z), dim), lambda x,y,z: x.cat(y, z, dim=dim))

  @unittest.skip("needs Rockchip strided copy/movement template support; closest dump is ~/npu/ops_rknn/rknn copy.gdb")
  def test_stack_max(self):
    helper_test_op(None, lambda x,y: torch.stack((x, y)).max(axis=0)[0], lambda x,y: Tensor.stack(x, y).max(axis=0), vals=[[1.], [2.]])

  @unittest.skip("needs Rockchip strided copy/movement template support; closest dump is ~/npu/ops_rknn/rknn copy.gdb")
  def test_repeat(self):
    x = Tensor.randn(4, 6, 3)
    base_repeats = [2, 4, 3]

    for reps in [[], [4], [2, 1], [3, 2, 2]]:
      repeats = base_repeats + reps
      helper_test_op([(4, 6, 3)], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))
      helper_test_op([()], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))

    with self.assertRaises(ValueError):
      x.repeat((2, 4))

    np.testing.assert_allclose(x.repeat((2, 0, 4)).numpy(), Tensor.zeros(8, 0, 12).numpy())

  @unittest.skip("needs Rockchip strided copy/movement template support; closest dump is ~/npu/ops_rknn/rknn copy.gdb")
  def test_repeat_interleave(self):
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(6))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, 1))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, 0))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, -1))
    helper_test_op([(3, 3)], lambda x: x.repeat_interleave(2, -2))

  @unittest.skip("needs Rockchip strided copy/movement template support; closest dump is ~/npu/ops_rknn/rknn copy.gdb")
  def test_simple_repeat(self):
    repeats = [3, 3, 4]
    helper_test_op([(3, 3)], lambda x: x.repeat(*repeats), lambda x: x.repeat(repeats))

  @unittest.skip("needs Rockchip clip lowering; refs: ~/rk3588/experimental/min_add_common.py and ~/npu/ops_rknn/{min,max}.cpp")
  def test_clip(self):
    helper_test_op([(45,65)], lambda x: x.clip(-2.3, 1.2))
    helper_test_op(None, lambda x: x.clip(-2.5, 1.5), vals=[[-3.0, -2.5, 0, 1.5, 2]])
    helper_test_op([(45,65)], lambda x: x.clip(0, 0))
    helper_test_op([(45,65)], lambda x: x.clip(10, 100))
    helper_test_op([(45,65)], lambda x: x.clip(0, 0.1))
    helper_test_op([(45,65)], lambda x: x.clip(-0.3, -0.2))
    helper_test_op([(45,65)], lambda x: x.clip(3, 0))
    helper_test_op([(45,65)], lambda x: x.clip(None, 0))
    helper_test_op([(45,65)], lambda x: x.clip(0, None))
    self.helper_test_exception([(45,65)], lambda x: x.clip(None, None), expected=RuntimeError)

  def test_matvecmat(self):
    helper_test_op([(1,128), (128,128), (128,128)], lambda x,y,z: (x.half()@y.half()).relu()@z.half(),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  def test_matvec(self):
    helper_test_op([(1,128), (128,128)], lambda x,y: (x.half()@y.half()).relu(),
                   atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3, forward_only=True)

  @unittest.skip("this test is broken #862")
  def test_max_nan(self):
    n = Tensor([1, float("nan")]).max().numpy()
    assert math.isnan(n.item()), f"{n.item()} is not nan"

  @unittest.skipIf(COMPILE_ONLY, "test requires runtime")
  def test_inf_where(self):
    x = Tensor.full((3, 3), float("inf"))
    n = (x < 0).where(x, 1).numpy()
    assert np.all(n == 1.)

  def _get_index_randoms(self):
    a = torch.randint(low=-1, high=1, size=(2,1,1,1,1,1), dtype=torch.int64, requires_grad=False)
    b = torch.randint(high=1, size=(1,3,1,1,1,1), dtype=torch.int64, requires_grad=False)
    c = torch.randint(low=-5, high=5, size=(1,1,4,1,1,1), dtype=torch.int64, requires_grad=False)
    d = torch.randint(high=4, size=(2,1,1,5,1,1), dtype=torch.int64, requires_grad=False)
    e = torch.randint(high=1, size=(1,1,1,1,6,1), dtype=torch.int64, requires_grad=False)
    i, j, k, o, p = [Tensor(tor.detach().cpu().numpy().astype(np.int32), requires_grad=False) for tor in [a,b,c,d,e]]
    return a,b,c,d,e,i,j,k,o,p

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_fancy_indexing_inf(self):
    data = [math.inf, -math.inf, math.nan]
    helper_test_op((), lambda: torch.tensor(data)[torch.tensor([0, 1, 2])], lambda: Tensor(data)[Tensor([0, 1, 2])])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  @slow_test
  def test_slice_fancy_indexing_no_dim_collapse(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,d,e], lambda x: x[i,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[:,b,c,d,:], lambda x: x[:,j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,...], lambda x: x[i,j,...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,...,e], lambda x: x[i,...,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[...,c,:,e], lambda x: x[...,k,:,p])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_slice_fancy_indexing_dim_collapse_int(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,c,d,e], lambda x: x[1,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,3,d,e], lambda x: x[i,j,3,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,2,d,2], lambda x: x[1,j,2,o,2])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,2,2,2,e], lambda x: x[i,2,2,2,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,:,3:11:2,d,0:2], lambda x: x[1,:,3:11:2,o,0:2])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  @slow_test
  def test_slice_fancy_indexing_dim_inject_none(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,c,d,e], lambda x: x[None,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,d,None], lambda x: x[i,j,k,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,None,d,e], lambda x: x[i,j,None,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,c,d,None], lambda x: x[None,j,k,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,:,None,d,e], lambda x: x[i,:,None,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,None,None,None], lambda x: x[None,None,None,None,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,b,c,d,e], lambda x: x[None,None,j,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,b,c,None,None], lambda x: x[None,None,j,k,None,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,None,None,c,d,e], lambda x: x[i,None,None,k,o,p])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,None,None,c,None,None], lambda x: x[i,None,None,k,None,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,None,b,None,d,e], lambda x: x[None,None,j,None,o,p])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_slice_fancy_indexing_dim_inject_and_collapse(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()  # noqa
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,b,None,d,1], lambda x: x[1,j,None,o,1])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[None,b,2,d,None], lambda x: x[None,j,2,o,None])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[...,1,d,None], lambda x: x[...,1,o,None])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_slice_fancy_indexing_with_tensors(self):
    helper_test_op([(2,3)], lambda x: x[torch.tensor([[0,0,0],[0,0,0]]), torch.tensor(1)],
                            lambda x: x[Tensor([[0,0,0],[0,0,0]]), Tensor(1)])
    helper_test_op([(2,3)], lambda x: x[torch.tensor([1]), torch.tensor([[0,0,0],[0,0,0]])],
                            lambda x: x[Tensor([1]), Tensor([[0,0,0],[0,0,0]])])
    helper_test_op([(2,3)], lambda x: x[torch.tensor([[0,0,0],[0,0,0]]), torch.tensor([2,1,1])],
                            lambda x: x[Tensor([[0,0,0],[0,0,0]]), Tensor([2,1,1])])
    helper_test_op([(2,3)], lambda x: x[torch.tensor([[0,1,-1],[-1,-2,0]]), torch.tensor([2,1,-1])],
                            lambda x: x[Tensor([[0,1,-1],[-1,-2,0]]), Tensor([2,1,-1])])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  @slow_test
  def test_slice_fancy_indexing_list_indices(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[((0,),)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(0,),b,c,d,:], lambda x: x[(0,),j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[[[[0]]],b,c,d,[[1]]], lambda x: x[[[[0]]],j,k,o,[[1]]])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(1,0,-1),b,c,d,:], lambda x: x[(1,0,-1),j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,(1,2,3),...], lambda x: x[i,j,k,(1,2,3),...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,[[1],[2],[3]],...], lambda x: x[i,j,k,[[1],[2],[3]],...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,(2,1,0),c,(-2,1,0),e], lambda x: x[i,(2,1,0),k,(-2,1,0),p])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  @slow_test
  def test_slice_fancy_indexing_tuple_indices(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(((0,),),)], lambda x: x[(((0,),),)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(0,),b,c,d,:], lambda x: x[(0,),j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(1,0),b,c,d,:], lambda x: x[(1,0),j,k,o,:])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,b,c,(1,2,3),...], lambda x: x[i,j,k,(1,2,3),...])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[a,((2,),(1,),(0,)),c,(2,1,0)], lambda x: x[i,((2,),(1,),(0,)),k,(2,1,0)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[1,(2,1,0),None,c,(2,1,0),e], lambda x: x[1,(2,1,0),None,k,(2,1,0),p])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  @slow_test
  def test_slice_fancy_indexing_list_with_tensors(self):
    a,b,c,d,e,i,j,k,o,p = self._get_index_randoms()
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(a,)], lambda x: x[(i,)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(a,1)], lambda x: x[(i,1)])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(a,(1,1))], lambda x: x[(i,(1,1))])
    helper_test_op([(2,5,6,5,3,4)], lambda x: x[(a,b,c,d,e)], lambda x: x[(i,j,k,o,p)])

  @unittest.skip("needs Rockchip gather/fancy-index lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_slice_fancy_indexing_errors(self):
    a = Tensor.ones(10,11,12)
    with self.assertRaises(IndexError): a[Tensor(1.1)]
    with self.assertRaises(IndexError): a[[1.1]]
    with self.assertRaises(IndexError): a[Tensor([True, False])]
    with self.assertRaises(IndexError): a[[True, False]]
    with self.assertRaises((IndexError, ValueError)): a[Tensor.randint(3,1,1,1), Tensor.randint(1,4,1,1), Tensor.randint(2,4,4,1)]
    with self.assertRaises((IndexError, ValueError)): a[Tensor.randint(3,1,1,1), Tensor.randint(1,4,1,1,1)]
    helper_test_op([(5, 6)], lambda x: x[[True, False, 2]])

  @unittest.skip("needs Rockchip gather/scatter lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_gather(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=0, index=b), lambda x: x.gather(dim=0, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=1, index=b), lambda x: x.gather(dim=1, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=2, index=b), lambda x: x.gather(dim=2, index=a))
    helper_test_op([(3,4,5)], lambda x: x.gather(dim=0, index=b), lambda x: x.gather(dim=0, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=-1, index=b), lambda x: x.gather(dim=-1, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=-2, index=b), lambda x: x.gather(dim=-2, index=a))
    helper_test_op([(4,5,6)], lambda x: x.gather(dim=-3, index=b), lambda x: x.gather(dim=-3, index=a))
    self.helper_test_exception([(4,5,6)], lambda x: x.gather(dim=0, index=torch.tensor([1], dtype=torch.int64)),
                                          lambda x: x.gather(dim=0, index=Tensor([1], dtype=dtypes.int32)), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(2,1,1)], lambda x: x.gather(dim=0, index=b),
                                          lambda x: x.gather(dim=0, index=a), expected=(RuntimeError, AssertionError))
    helper_test_op(None, lambda x: x.gather(dim=0, index=torch.tensor([2, 1, 0, 1, 2], requires_grad=False)),
                         lambda x: x.gather(dim=0, index=Tensor([2, 1, 0, 1, 2])), vals=[[1., 2., 3.]])
    helper_test_op(None, lambda x: x.gather(dim=0, index=torch.tensor([2, 1, 0, 1, 2], requires_grad=False)),
                         lambda x: x.gather(dim=0, index=Tensor([2, 1, 0, 1, 2])), vals=[[-float("inf"), 2., 3.]])

  @unittest.skip("needs Rockchip gather/scatter lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_scatter(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    for dim in (0,1,2,-1,-2,-3):
      helper_test_op([(4,5,6), (4,5,6)], lambda x,src: x.scatter(dim=dim, index=b, src=src),
                                         lambda x,src: x.scatter(dim=dim, index=a, src=src), forward_only=True)
    helper_test_op([(3,4,5), (3,4,5)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                       lambda x,src: x.scatter(dim=1, index=a, src=src), forward_only=True)
    helper_test_op([(10,3,10), (10,10,10)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                            lambda x,src: x.scatter(dim=1, index=a, src=src), forward_only=True)
    self.helper_test_exception([(2,3,10), (10,10,10)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                                       lambda x,src: x.scatter(dim=1, index=a, src=src), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(10,3,10), (10,3,10)], lambda x,src: x.scatter(dim=1, index=b, src=src),
                                                       lambda x,src: x.scatter(dim=1, index=a, src=src), expected=(RuntimeError, AssertionError))
    self.helper_test_exception([(3,4,5), (3,4,5)], lambda x,src: x.scatter(dim=1, index=b, src=src, mode="typo"),
                                                   lambda x,src: x.scatter(dim=1, index=a, src=src, mode="typo"), expected=TypeError)
    self.helper_test_exception([(3,4,5), (3,4,5)], lambda x,src: x.half().scatter(dim=1, index=b, src=src),
                                                   lambda x,src: x.half().scatter(dim=1, index=a, src=src), expected=RuntimeError)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=3), lambda x: x.scatter(dim=1, index=a, src=3), forward_only=True)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf")),
      lambda x: x.scatter(dim=1, index=a, src=float("inf")), forward_only=True)
    b = torch.tensor([0,0], requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op(None, lambda x,src: x.scatter(0, b, src), lambda x,src: x.scatter(0, a, src), forward_only=True,
                   vals=[[1.,2.,3.,4.], [1.,0.]])

  @unittest.skip("needs Rockchip gather/scatter lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_scatter_add(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf"), reduce="add"),
      lambda x: x.scatter(dim=1, index=a, src=float("inf"), reduce="add"), forward_only=True)
    if Device.DEFAULT != "WEBGPU":
      helper_test_op([(4,5,6)],
        lambda x: x.scatter(1, b, float("nan"), reduce="add"),
        lambda x: x.scatter(1, a, float("nan"), reduce="add"), forward_only=True)

  @unittest.skip("needs Rockchip gather/scatter lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_scatter_mul(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    helper_test_op([(4,5,6)], lambda x: x.scatter(dim=1, index=b, value=float("inf"), reduce="multiply"),
      lambda x: x.scatter(dim=1, index=a, src=float("inf"), reduce="multiply"), forward_only=True)
    if Device.DEFAULT != "WEBGPU":
      helper_test_op([(4,5,6)],
        lambda x: x.scatter(1, b, float("nan"), reduce="multiply"),
        lambda x: x.scatter(1, a, float("nan"), reduce="multiply"), forward_only=True)

  def test_scatter_no_reduce_tensor_src(self):
    with self.assertRaises(TypeError):
      Tensor.ones(4).scatter(dim=1, index=Tensor([0]), src=Tensor.ones(4), reduce="add")

  @unittest.skip("needs Rockchip gather/scatter lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  @slow_test
  def test_scatter_reduce(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), requires_grad=False)
    for reduce in ("sum", "prod", "mean", "amin", "amax"):
      for dim in (-1,1,-3):
        helper_test_op([(3,4,5), (3,4,5)],
          lambda x,src: x.scatter_reduce(dim=dim, index=b, src=src, reduce=reduce),
          lambda x,src: x.scatter_reduce(dim=dim, index=a, src=src, reduce=reduce), forward_only=True)
        helper_test_op([(3,4,5), (3,4,5)],
          lambda x,src: x.scatter_reduce(dim=dim, index=b, src=src, reduce=reduce, include_self=False),
          lambda x,src: x.scatter_reduce(dim=dim, index=a, src=src, reduce=reduce, include_self=False), forward_only=True)

  @unittest.skip("needs Rockchip gather/scatter lowering; nearest refs: ~/rk3588/examples/where.py and ~/npu/ops_rknn/rknn copy.gdb")
  def test_scatter_reduce_prod_zeros(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    x = Tensor.zeros([4,5,6]).float()
    y = torch.zeros([4,5,6]).float()
    helper_test_op([(4,5,6)],
      lambda src: y.scatter_reduce(dim=1, index=b, src=src, reduce="prod"),
      lambda src: x.scatter_reduce(dim=1, index=a, src=src, reduce="prod"), forward_only=True)

  def test_scatter_reduce_errors(self):
    b = torch.randint(3, size=[3,4,5], dtype=torch.int64, requires_grad=False)
    a = Tensor(b.detach().cpu().numpy().astype(np.int32), dtype=dtypes.int32, requires_grad=False)
    self.helper_test_exception([(4,5,6), (4,5,6)],
      lambda x,src: x.scatter_reduce(dim=0, index=b, src=src, reduce="INVALID"),
      lambda x,src: x.scatter_reduce(dim=0, index=a, src=src, reduce="INVALID"),
      RuntimeError)
    self.helper_test_exception([(4,5,6), (4,5,6)],
      lambda x,src: x.half().scatter_reduce(dim=0, index=b, src=src, reduce="sum"),
      lambda x,src: x.half().scatter_reduce(dim=0, index=a, src=src, reduce="sum"),
      RuntimeError)

  @unittest.skip("needs Rockchip attention/softmax chain support; refs: ~/rk3588/examples/gemm.py and ~/npu/ops_rknn/act/log_softmax.py")
  def test_scaled_dot_product_attention(self):
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64)], torch.nn.functional.scaled_dot_product_attention,
                   Tensor.scaled_dot_product_attention)
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64), (32,8,16,16)],
                   lambda x,y,z,m: torch.nn.functional.scaled_dot_product_attention(x,y,z,attn_mask=m),
                   lambda x,y,z,m: Tensor.scaled_dot_product_attention(x,y,z,attn_mask=m))

  @unittest.skip("needs Rockchip attention/softmax chain support; refs: ~/rk3588/examples/gemm.py and ~/npu/ops_rknn/act/log_softmax.py")
  @slow_test
  def test_scaled_dot_product_attention_mismatch_ls(self):
    helper_test_op([(32,8,4,64), (32,8,16,64), (32,8,16,64)], torch.nn.functional.scaled_dot_product_attention,
                   Tensor.scaled_dot_product_attention)

  @unittest.skip("needs Rockchip attention/softmax chain support; refs: ~/rk3588/examples/gemm.py and ~/npu/ops_rknn/act/log_softmax.py")
  @slow_test
  def test_scaled_dot_product_attention_causal(self):
    helper_test_op([(32,8,16,64), (32,8,16,64), (32,8,16,64)],
                   lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z,is_causal=True),
                   lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,is_causal=True))
    self.helper_test_exception([(32,8,16,64), (32,8,16,64), (32,8,16,64), (32,8,16,16)],
      lambda x,y,z,m: torch.nn.functional.scaled_dot_product_attention(x,y,z,is_causal=True,attn_mask=m),
      lambda x,y,z,m: Tensor.scaled_dot_product_attention(x,y,z,is_causal=True,attn_mask=m),
      expected=RuntimeError)

  @unittest.skip("needs Rockchip attention/softmax chain support; refs: ~/rk3588/examples/gemm.py and ~/npu/ops_rknn/act/log_softmax.py")
  def test_scaled_dot_product_attention_gqa(self):
    helper_test_op([(32,32,16,64), (32,8,16,64), (32,8,16,64)],
                   lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z,enable_gqa=True),
                   lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,enable_gqa=True))

  @unittest.skip("needs Rockchip attention/softmax chain support; refs: ~/rk3588/examples/gemm.py and ~/npu/ops_rknn/act/log_softmax.py")
  def test_scaled_dot_product_attention_gqa_errors(self):
    self.helper_test_exception([(32,31,16,64), (32,8,16,64), (32,8,16,64)],
      lambda x,y,z: torch.nn.functional.scaled_dot_product_attention(x,y,z),
      lambda x,y,z: Tensor.scaled_dot_product_attention(x,y,z,enable_gqa=True),
      expected=(AssertionError, RuntimeError, ValueError))

  @unittest.skip("needs Rockchip BCE/loss-chain support; nearest refs: ~/npu/ops_rknn/act/sigmoid.py and ~/npu/ops_rknn/act/logsigmoid.py")
  def test_binary_crossentropy(self):
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),y.clip(0,1)),
                                       lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,y.clip(0,1)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,y.clip(0,1)),
                                       lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1)))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(),y.clip(0,1)),
                                       lambda x,y: x.binary_crossentropy_logits(y.clip(0,1)))

  @unittest.skip("needs Rockchip BCE/loss-chain support; nearest refs: ~/npu/ops_rknn/act/sigmoid.py and ~/npu/ops_rknn/act/logsigmoid.py")
  def test_binary_crossentropy_reductions(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy(x.sigmoid(), y.clip(0,1), reduction=r),
                                         lambda x,y: x.sigmoid().binary_crossentropy(y.clip(0,1), reduction=r))
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x, y.clip(0,1), reduction=r),
                                         lambda x,y: x.binary_crossentropy_logits(y.clip(0,1), reduction=r))

  @unittest.skip("needs Rockchip BCE/loss-chain support; nearest refs: ~/npu/ops_rknn/act/sigmoid.py and ~/npu/ops_rknn/act/logsigmoid.py")
  def test_binary_crossentropy_logits_pos_weights(self):
    pos_weight = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.binary_cross_entropy_with_logits(x,y.clip(0,1),
      pos_weight=torch.tensor(pos_weight)),
      lambda x,y: x.binary_crossentropy_logits(y.clip(0,1), pos_weight=Tensor(pos_weight)))

  @unittest.skip("needs Rockchip cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_cross_entropy_class_probabilities(self):
    helper_test_op([(32,), (32,)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))
    helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))
    helper_test_op([(32,4,4,4), (32,4,4,4)], lambda x,y: torch.nn.functional.cross_entropy(x, y), lambda x,y: x.cross_entropy(y))

  @unittest.skip("needs Rockchip cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_cross_entropy_class_indices(self):
    classes = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
    helper_test_op([(32,10)], lambda x: torch.nn.functional.cross_entropy(x, torch.tensor(classes)),
                              lambda x: x.cross_entropy(Tensor(classes)))
    self.helper_test_exception([(32,10), (32,1)], lambda x,y: torch.nn.functional.cross_entropy(x, y),
                                                  lambda x,y: x.cross_entropy(y), expected=(AssertionError, RuntimeError))

  @unittest.skip("needs Rockchip cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_cross_entropy_reductions(self):
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, reduction=r),
                                         lambda x,y: x.cross_entropy(y, reduction=r))
    self.helper_test_exception([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, reduction="typo"),
                                                   lambda x,y: x.cross_entropy(y, reduction="typo"), expected=ValueError)

  @unittest.skip("needs Rockchip cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_cross_entropy_smoothing(self):
    for ls in (0., 0.3, 0.7, 1.):
      helper_test_op([(32,10), (32,10)], lambda x,y: torch.nn.functional.cross_entropy(x, y, label_smoothing=ls),
                                         lambda x,y: x.cross_entropy(y, label_smoothing=ls))
      classes = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
      helper_test_op([(32,10)], lambda x: torch.nn.functional.cross_entropy(x, torch.tensor(classes), label_smoothing=ls),
                                lambda x: x.cross_entropy(Tensor(classes), label_smoothing=ls))

  @unittest.skip("needs Rockchip sparse cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_sparse_categorical_crossentropy(self):
    classes = np.random.randint(0, 10, (12,), dtype=np.int32).tolist()
    helper_test_op([(12,10)], lambda x: torch.nn.CrossEntropyLoss()(x, torch.tensor(classes)),
                              lambda x: x.sparse_categorical_crossentropy(Tensor(classes)))
    helper_test_op([(12,10)],
                   lambda x: torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=classes[0], label_smoothing=0.3)(x, torch.tensor(classes)),
                   lambda x: x.sparse_categorical_crossentropy(Tensor(classes), reduction="mean", ignore_index=classes[0], label_smoothing=0.3))
    classes = np.random.randint(0, 10, (3,12), dtype=np.int32).tolist()
    helper_test_op([(3,12,10)], lambda x: torch.nn.CrossEntropyLoss()(x.permute(0,2,1), torch.tensor(classes)),
                                lambda x: x.sparse_categorical_crossentropy(Tensor(classes)))

  @unittest.skip("needs Rockchip sparse cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_sparse_categorical_crossentropy_reductions(self):
    for r in ("mean", "sum", "none"):
      classes = np.random.randint(0, 10, (12,), dtype=np.int32).tolist()
      helper_test_op([(12,10)], lambda x: torch.nn.CrossEntropyLoss(reduction=r)(x, torch.tensor(classes)),
                                lambda x: x.sparse_categorical_crossentropy(Tensor(classes), reduction=r))

  @unittest.skip("needs Rockchip sparse cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_sparse_categorical_crossentropy_ignore_index(self):
    classes = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    for i in (-1, 0, 3):
      helper_test_op([(12,10)], lambda x: torch.nn.CrossEntropyLoss(ignore_index=i)(x, torch.tensor(classes)),
                                lambda x: x.sparse_categorical_crossentropy(Tensor(classes), ignore_index=i))

  @unittest.skip("needs Rockchip sparse cross-entropy/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_sparse_categorical_crossentropy_label_smoothing(self):
    for s in (0.3, 0.9):
      classes = np.random.randint(0, 10, (12,), dtype=np.int32).tolist()
      helper_test_op([(12,10)], lambda x: torch.nn.CrossEntropyLoss(label_smoothing=s)(x, torch.tensor(classes)),
                                lambda x: x.sparse_categorical_crossentropy(Tensor(classes), label_smoothing=s))

  @unittest.skip("needs Rockchip NLL/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_nll_loss(self):
    target = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
    helper_test_op([(32,10)],
                   lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target)),
                   lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target)))

  @unittest.skip("needs Rockchip NLL/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_nll_loss_3d(self):
    target = np.random.randint(0, 10, (32,3,3,3), dtype=np.int32).tolist()
    helper_test_op([(32,10,3,3,3)],
                   lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target)),
                   lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target)))

  @unittest.skip("needs Rockchip NLL/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_nll_loss_reductions(self):
    target = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10)],
        lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target), reduction=r),
      lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target), reduction=r))
    self.helper_test_exception([(32,10)],
      lambda x: torch.nn.functional.nll_loss(x, torch.tensor(target), reduction="typo"),
      lambda x: x.nll_loss(Tensor(target), reduction="typo"), expected=ValueError)

  @unittest.skip("needs Rockchip NLL/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_nll_loss_weight(self):
    target = np.random.randint(0, 10, (32,), dtype=np.int32).tolist()
    weight = np.random.normal(0, 1, (10,)).astype(np.float32).tolist()
    for r in ("mean", "sum", "none"):
      helper_test_op([(32,10)],
        lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target), torch.tensor(weight), reduction=r),
        lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target), Tensor(weight), reduction=r))

  @unittest.skip("needs Rockchip NLL/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_nll_loss_3d_weight(self):
    target = np.random.randint(0, 10, (16,3,3,3), dtype=np.int32).tolist()
    weight = np.random.normal(0, 1, (10,)).astype(np.float32).tolist()
    for r in ("mean", "sum", "none"):
      helper_test_op([(16,10,3,3,3)],
          lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target), torch.tensor(weight), reduction=r),
          lambda x: x.log_softmax(axis=1).nll_loss(Tensor(target), Tensor(weight), reduction=r))

  @unittest.skip("needs Rockchip NLL/log_softmax chain support; nearest refs: ~/npu/ops_rknn/act/log_softmax.py")
  def test_nll_loss_ignore_index(self):
    logits = [[2.0, 0.5, -1.0],
              [1.5, 2.5, -0.5],
              [0.0, -2.0, 1.0]]
    target = [0, 1, 2]
    helper_test_op(None, lambda x: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x, dim=1), torch.tensor(target), ignore_index=1),
                         lambda x: x.log_softmax().nll_loss(Tensor(target), ignore_index=1),
                         vals=[logits])

  @unittest.skip("needs Rockchip one_hot/masked-select lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_one_hot(self):
    data = [1, 2, 4]
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data), 6).type(torch.int32),
                       lambda: Tensor(data).one_hot(6), forward_only=True)
    with self.assertRaises(ValueError): Tensor(data).one_hot(-1)
    data = [[[1, 2, 3], [0, 3, 5]], [[1, 2, 3], [0, 3, 5]]]
    helper_test_op([], lambda: torch.nn.functional.one_hot(torch.tensor(data), 8).type(torch.int32),
                       lambda: Tensor(data).one_hot(8), forward_only=True)

  @unittest.skip("needs Rockchip one_hot/masked-select lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_masked_fill(self):
    helper_test_op([(32,10)], lambda x: x.masked_fill((x>0.1).detach(), -math.inf))
    helper_test_op([(32,10)], lambda x: x.masked_fill((x<0.1).detach(), -math.inf))

  @unittest.skip("needs Rockchip mask-select lowering; nearest refs: ~/npu/ops_rknn/cast.cpp")
  def test_masked_select(self):
    helper_test_op([(32, 10)], lambda x: x.masked_select(x>0.5), lambda x: x.masked_select(x>0.5), forward_only=True)
    helper_test_op([(32, 10)], lambda x: x.masked_select(torch.tensor(True)), lambda x: x.masked_select(Tensor(True)), forward_only=True)

  @unittest.skip("needs Rockchip one_hot/masked-select lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_nonzero(self):
    helper_test_op([(32, 10)], lambda x: (x>0.5).nonzero().int(), lambda x: (x>0.5).nonzero(), forward_only=True)
    helper_test_op([(20,)], lambda x: (x>0.5).nonzero().int(), lambda x: (x>0.5).nonzero(), forward_only=True)
    helper_test_op([(10, 5, 3)], lambda x: (x>0.5).nonzero().int(), lambda x: (x>0.5).nonzero(), forward_only=True)

  @unittest.skip("needs Rockchip cast/bitcast lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_cast(self):
    helper_test_op([(3, 3)], lambda x: x.float())
    helper_test_op(None, lambda x: x.float(), vals=[[0, 1, 2, 3]], forward_only=True)
    helper_test_op(None, lambda x: x.float(), vals=[[True, False]], forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.int(), forward_only=True)
    helper_test_op([(3, 3)], lambda x: x.bool(), forward_only=True)

  @unittest.skip("needs Rockchip cast/bitcast lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_bitcast(self):
    helper_test_op([(3, 3)], lambda x: x.view(torch.int32), lambda x: x.bitcast(dtypes.int32), forward_only=True)

  @unittest.skip("needs Rockchip int-or lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_int_or(self):
    t = (Tensor([0], dtype='int') | 0xFFFFFFFF).item()
    if not COMPILE_ONLY: assert t == -1

@unittest.skipUnless(is_dtype_supported(dtypes.uchar), f"no uint8 on {Device.DEFAULT}")
class TestOpsUint8(unittest.TestCase):
  @unittest.skip("needs Rockchip cast/bitcast lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_cast(self):
    helper_test_op([(2,3,64,64)], lambda x: x.type(torch.uint8), lambda x: x.cast('uint8'), forward_only=True, low=0, high=255)

  @unittest.skip("needs Rockchip cast/bitcast lowering; nearest ref: ~/npu/ops_rknn/cast.cpp")
  def test_cast_relu(self):
    helper_test_op([(2,3,64,64)], lambda x: x.relu().type(torch.uint8), lambda x: x.relu().cast('uint8'), forward_only=True)

  def test_interpolate_bilinear(self):
    out_sz = (10, 10)
    helper_test_op([(2,3,64,64)],
      lambda x: torch.nn.functional.interpolate((10*x).relu().type(torch.uint8), size=out_sz, mode="bilinear"),
      lambda x: Tensor.interpolate((10*x).relu().cast('uint8'), size=out_sz, mode="linear"), forward_only=True)

  def test_interpolate_nearest(self):
    out_sz = (10, 10)
    helper_test_op([(2,3,64,64)],
      lambda x: torch.nn.functional.interpolate((10*x).relu().type(torch.uint8), size=out_sz, mode="nearest"),
      lambda x: Tensor.interpolate((10*x).relu().cast('uint8'), size=out_sz, mode="nearest"), forward_only=True)

  def test_interpolate_nearest_exact(self):
    out_sz = (10, 10)
    helper_test_op([(2,3,64,64)],
      lambda x: torch.nn.functional.interpolate((10*x).relu().type(torch.uint8), size=out_sz, mode="nearest-exact"),
      lambda x: Tensor.interpolate((10*x).relu().cast('uint8'), size=out_sz, mode="nearest-exact"), forward_only=True)

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
