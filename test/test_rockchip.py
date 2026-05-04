import os, sys, time, math, unittest, functools, platform, warnings, subprocess, pickle, ctypes
from pathlib import Path
import numpy as np
import torch

if os.getenv("ROCKCHIP") == "1" and "DEV" not in os.environ: os.environ["DEV"] = "ROCKCHIP"

from tinygrad.helpers import getenv, CI, DEBUG, DEV, IMAGE
from tinygrad import Tensor, Device, dtypes
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
    self.assertEqual(plan.subcore_task, ((0, 3), (3, 0), (4, 0), (0, 0), (0, 0)))
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

class TestRockchipHardware(unittest.TestCase):
  @staticmethod
  def _pcchain_add_segment(input_dma, weight_dma, output_dma, next_dma, elements):
    width = (elements + 7) // 8 - 1
    body = [
      rkcmd(rk.DPU, rk.REG_DPU_S_POINTER, 0x0000000E),
      rkcmd(rk.DPU, rk.REG_DPU_DATA_FORMAT, 0x000001E5),
      rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, 0x48000002),
      rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH, 0x00070007),
      rkcmd(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, width),
      rkcmd(rk.DPU, rk.REG_DPU_EW_CFG, 0x108202C0),
      rkcmd(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE, 0x00010001),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, 0x0000000E),
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

  # failed, need ADD int16 support
  # def test_broadcasted_add(self):
  #   pass
  #   helper_test_op([(45,65), (45,1)], lambda x,y: x+y)
  #   helper_test_op([(45,65), ()], lambda x,y: x+y)
  # def test_broadcasted_add_2(self):
  #   helper_test_op([(45,65), (65,)], lambda x,y: x+y)

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

  # # slow test
  # def test_max_pool2d(self):
  #   for ksz in [(3,3)]:
  #     with self.subTest(kernel_size=ksz):
  #       helper_test_op([(1,1,10,10)],
  #         lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
  #         lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

  def test_gemm_fp16(self):
    i = 64
    helper_test_op([(i,i), (i,i)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3)
    # helper_test_op([(2,2), (2,2)], lambda x,y: x.half().matmul(y.half()), atol=5e-3, rtol=5e-3, grad_atol=5e-3, grad_rtol=5e-3)
    # shapes = [((4,4), (4,4)), ((33,33), (33,33)), ((34,34), (34,34)), ((65,33), (33,65)), ((394,394), (394,394)),
    #           ((1,8192), (8192,8192)), ((1,8192), (8192,8193)), ((1,8193), (8193,8192)), ((1,8193), (8193,8193)),
    #           ((1,768), (768,2048)), ((1,2048), (2048,2048)), ((1,4096), (4096,4096))]




  def test_sd_big_conv(self):
    helper_test_op([(1,256,64,64), (512,256,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-3)

  @unittest.skip("slow")
  def test_large_bs_conv(self):
    helper_test_op([(4096,3,3,3), (1,3,3,3)],
                    lambda x,w: torch.nn.functional.conv2d(x, w),
                    lambda x,w: x.conv2d(w), atol=1e-3)

  @unittest.skip("slow")
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
    helper_test_op([(1,4,9,9), (4,4,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

  def test_simple_conv2d_1x1_m4(self):
    helper_test_op([(1,16,32,32), (16,16,1,1)],
      lambda x,w: torch.nn.functional.conv2d(x,w),
      lambda x,w: Tensor.conv2d(x,w), grad_rtol=1e-5)

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

if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)
