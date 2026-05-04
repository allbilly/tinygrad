from dataclasses import dataclass
import ctypes, math, os, pickle, numpy as np
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.uop.ops import Ops, UOp
from tinygrad.runtime.autogen import rockchip as rk

RK_TEMPLATE_MAGIC = b"RKTP"
RK_TEMPLATE_VERSION = 1
REGCMD_RESERVED = 16384
FP16_ATOM_ELEMENTS = 16
NPU_CBUF_BANK_SIZE = 32768
NPU_CBUF_BANKS = 12
RK_DPU_OUTPUT_GROUP = 8
EW_ALU_OPS = {Ops.MUL:0, Ops.MAX:0, Ops.ADD:2, Ops.FDIV:3, Ops.SUB:4}
CORE_RESERVED_ZERO_ADDR = 0x3030
DPU_RESERVED_ZERO_ADDR = 0x40c4

@dataclass(frozen=True)
class RKPatch:
  section: str
  offset: int
  kind: str
  arg_index: int|None
  role: str
  shift: int = 0
  mask: int = 0xffffffff
  addend: int = 0

@dataclass(frozen=True)
class RKTaskTemplate:
  op_idx: int
  enable_mask: int
  int_mask: int
  int_clear: int
  regcfg_offset: int
  regcfg_amount: int
  core_mask: int = 1
  flags: int = 0

@dataclass(frozen=True)
class RKTemplatePackage:
  version: int
  target: str
  family: str
  regcmd: tuple[int, ...]
  patches: tuple[RKPatch, ...] = ()
  tasks: tuple[RKTaskTemplate, ...] = ()
  op: Ops|None = None
  size: int = 0
  meta: dict[str, int|bool]|None = None

@dataclass(frozen=True)
class RKSubmitPlan:
  flags: int
  task_number: int
  core_mask: int
  subcore_task: tuple[tuple[int, int], ...]

def rk_field(val:int, shift:int, mask:int) -> int: return (val << shift) & mask

def rkcmd(target:int, reg:int, value:int) -> int: return (((target + 1) & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def encode_template(pkg:RKTemplatePackage) -> bytes:
  validate_template(pkg)
  return RK_TEMPLATE_MAGIC + bytes([RK_TEMPLATE_VERSION]) + pickle.dumps(pkg)

def decode_template(lib:bytes) -> RKTemplatePackage:
  if not lib.startswith(RK_TEMPLATE_MAGIC) or len(lib) <= len(RK_TEMPLATE_MAGIC): raise RuntimeError("unsupported Rockchip template package")
  version = lib[len(RK_TEMPLATE_MAGIC)]
  if version != RK_TEMPLATE_VERSION: raise RuntimeError(f"unsupported Rockchip template version {version}")
  pkg = pickle.loads(lib[len(RK_TEMPLATE_MAGIC)+1:])
  if not isinstance(pkg, RKTemplatePackage): raise RuntimeError("invalid Rockchip template payload")
  validate_template(pkg)
  return pkg

def validate_template(pkg:RKTemplatePackage, target:str|None=None) -> None:
  if pkg.version != RK_TEMPLATE_VERSION: raise RuntimeError(f"unsupported Rockchip template version {pkg.version}")
  if target is not None and pkg.target != target: raise RuntimeError(f"compiled for {pkg.target}, running on {target}")
  if not pkg.family: raise RuntimeError("missing Rockchip template family")
  for task in pkg.tasks:
    if task.regcfg_offset < 0 or task.regcfg_amount < 0 or task.regcfg_offset % 8 or task.regcfg_offset // 8 + task.regcfg_amount > len(pkg.regcmd):
      raise RuntimeError("Rockchip template task register range out of bounds")
  for patch in pkg.patches:
    if patch.section != "regcmd": raise RuntimeError(f"unsupported Rockchip patch section {patch.section}")
    if patch.offset < 0 or patch.offset >= len(pkg.regcmd): raise RuntimeError("Rockchip patch offset out of bounds")
    if patch.kind not in {"dma32", "dma32_add", "u32", "regfield"}: raise RuntimeError(f"unsupported Rockchip patch kind {patch.kind}")

def patch_regcmd(regcmd:list[int], patch:RKPatch, value:int) -> None:
  if patch.section != "regcmd": raise RuntimeError(f"unsupported Rockchip patch section {patch.section}")
  if patch.offset < 0 or patch.offset >= len(regcmd): raise RuntimeError("Rockchip patch offset out of bounds")
  raw = value + patch.addend
  if raw < 0: raise RuntimeError("Rockchip patch value underflow")
  cmd, old_value = regcmd[patch.offset], (regcmd[patch.offset] >> 16) & 0xffffffff
  if patch.kind in {"dma32", "dma32_add", "u32"}:
    shifted = raw << patch.shift
    if shifted & ~patch.mask: raise RuntimeError("Rockchip patch value overflow")
    new_value = (old_value & ~patch.mask) | (shifted & patch.mask)
  elif patch.kind == "regfield":
    shifted = raw << patch.shift
    if shifted & ~patch.mask: raise RuntimeError("Rockchip patch value overflow")
    new_value = (old_value & ~patch.mask) | (shifted & patch.mask)
  else:
    raise RuntimeError(f"unsupported Rockchip patch kind {patch.kind}")
  regcmd[patch.offset] = (cmd & 0xffff000000000000) | ((new_value & 0xffffffff) << 16) | (cmd & 0xffff)

def apply_patches(regcmd:list[int], patches:tuple[RKPatch, ...], values:dict[str, int]) -> None:
  for patch in patches:
    if patch.role not in values: raise RuntimeError(f"missing Rockchip patch role {patch.role}")
    patch_regcmd(regcmd, patch, values[patch.role])

def submit_plan(template:RKTemplatePackage, flags:int, official:bool=False) -> RKSubmitPlan:
  task_count = len(template.tasks)
  if task_count <= 0: raise RuntimeError("Rockchip template has no tasks")
  if official:
    return RKSubmitPlan(flags, task_count * 3, 0, ((0, task_count), (0, task_count), (0, task_count), (0, 0), (0, 0)))
  if template.family == "pcchain": return RKSubmitPlan(flags, task_count, template.tasks[0].core_mask, ((0, task_count), (0, 0), (0, 0), (0, 0), (0, 0)))
  return RKSubmitPlan(flags, task_count, template.tasks[0].core_mask, ((0, task_count), (task_count, 0), (task_count + 1, 0), (0, 0), (0, 0)))

def submit_template(fd_ctl, template:RKTemplatePackage, q:list[int], task_buf, cmd_buf, cmd_buf_size:int, timeout:int=6000) -> None:
  tasks = ctypes.cast(task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
  assert len(q) <= cmd_buf_size
  regcmd = ctypes.cast(cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * cmd_buf_size)).contents
  for i in range(len(q)): regcmd[i] = q[i]

  for i, task in enumerate(template.tasks):
    tasks[i].flags = task.flags
    tasks[i].op_idx = task.op_idx
    tasks[i].enable_mask = task.enable_mask
    tasks[i].int_mask = task.int_mask
    tasks[i].int_clear = task.int_clear
    tasks[i].int_status = 0
    tasks[i].regcfg_amount = task.regcfg_amount
    tasks[i].regcfg_offset = 0 if template.family == "pcchain" else task.regcfg_offset
    tasks[i].regcmd_addr = cmd_buf.meta.dma_addr + task.regcfg_offset if template.family == "pcchain" else cmd_buf.meta.dma_addr

  plan = submit_plan(template, rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG)
  submit_res = rk.struct_rknpu_submit(
    flags=plan.flags, timeout=timeout, task_start=0, task_number=plan.task_number, task_counter=0, priority=0,
    task_obj_addr=task_buf.meta.obj_addr, regcfg_obj_addr=0, task_base_addr=0, user_data=0, core_mask=plan.core_mask, fence_fd=-1,
    subcore_task=(rk.struct_rknpu_subcore_task * 5)(
      *(rk.struct_rknpu_subcore_task(task_start=s, task_number=n) for s,n in plan.subcore_task),
    )
  )
  rk.DRM_IOCTL_RKNPU_SUBMIT(fd_ctl, __payload=submit_res)

def align_up(val:int, align:int) -> int: return val if align <= 0 else ((val + align - 1) // align) * align

def conv_params(in_channels:int, out_channels:int, spatial:int) -> dict[str, int|bool]:
  align_c = max(8, min(1 << (max(1, in_channels) - 1).bit_length(), 16))
  align_out_c = max(FP16_ATOM_ELEMENTS, align_up(out_channels, FP16_ATOM_ELEMENTS))
  width_stride = align_up(spatial, max(1, (16 + align_c - 1) // align_c))
  row_bytes = width_stride * align_c * dtypes.float16.itemsize
  feature_grains = min(2, max(2, (2 * NPU_CBUF_BANK_SIZE + row_bytes - 1) // row_bytes))
  data_bytes = width_stride * feature_grains * align_c * dtypes.float16.itemsize
  data_bank = max(1, min(NPU_CBUF_BANKS - 1, (data_bytes + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE))
  out_width_stride = spatial if spatial < 4 else align_up(spatial, 4)
  return {
    "in_channels":in_channels, "out_channels":out_channels, "spatial":spatial, "align_c":align_c, "align_out_c":align_out_c,
    "width_stride":width_stride, "out_width_stride":out_width_stride, "data_bank":data_bank, "feature_grains":feature_grains,
    "surface_add":out_width_stride * (align_out_c // RK_DPU_OUTPUT_GROUP),
    "cbuf_entries":max(1, (width_stride * align_c + 31) // 32) * (4 if align_c < 16 else 1),
    "use_nhwc":align_c // in_channels == 2,
  }

def pack_conv_input(src:memoryview, p:dict[str, int|bool]) -> np.ndarray:
  in_channels, spatial, align_c, width_stride = (int(p[x]) for x in ("in_channels", "spatial", "align_c", "width_stride"))
  nchw = np.frombuffer(src, dtype=np.float16, count=in_channels*spatial).reshape(1, in_channels, 1, spatial)
  if p["use_nhwc"]:
    packed = np.zeros((1, 1, width_stride, in_channels), dtype=np.float16)
    packed[:, :, :spatial, :] = nchw.transpose(0, 2, 3, 1)
    return packed.reshape(-1)
  padded = np.zeros((1, align_c, 1, width_stride), dtype=np.float16)
  padded[:, :in_channels, :, :spatial] = nchw
  return padded.reshape(1, 1, align_c, 1, width_stride).transpose(0, 1, 3, 4, 2).reshape(-1)

def pack_conv_weights(src:memoryview, p:dict[str, int|bool]) -> np.ndarray:
  out_channels, in_channels, align_c = (int(p[x]) for x in ("out_channels", "in_channels", "align_c"))
  weights = np.frombuffer(src, dtype=np.float16, count=out_channels*in_channels).reshape(out_channels, in_channels, 1, 1)
  packed = np.zeros((out_channels, align_c, 1, 1), dtype=np.float16)
  packed[:, :in_channels] = weights
  return packed.transpose(0, 2, 3, 1).reshape(-1)

def unpack_conv_output(src:memoryview, p:dict[str, int|bool]) -> np.ndarray:
  out_channels, spatial, align_out_c, out_width_stride = (int(p[x]) for x in ("out_channels", "spatial", "align_out_c", "out_width_stride"))
  c2 = 8 if align_out_c >= 8 else align_out_c
  c1 = (out_channels + c2 - 1) // c2
  packed = np.frombuffer(src, dtype=np.float16, count=c1*out_width_stride*c2).reshape(1, c1, 1, out_width_stride, c2)
  return packed.transpose(0, 1, 4, 2, 3).reshape(1, c1*c2, 1, out_width_stride)[:, :out_channels, :, :spatial].reshape(-1)

def wmma_params(m:int, n:int, k:int) -> dict[str, int]:
  m, n, k = max(1, m), max(1, n), max(1, k)
  align_in = max(32, align_up(k, 32))
  align_out = max(32, align_up(n, 32))
  data_in_width, data_in_height = 1, m
  dataout_width, dataout_height = 1, m
  out_width_stride = 1
  is_kn_64, is_kn_256, is_kn_512, is_kn_lg_512 = k == 64 and n == 64, k == 256 and n == 256, k == 512 and n == 512, k > 512 and n > 512
  is_matmul_64, is_matmul_256 = m == 64 and k == 64 and n == 64, m == 256 and k == 256 and n == 256
  feature_grains = data_in_height + 1
  if k > 7872:
    feature_grains = 2
  elif 128 < k <= 192:
    feature_grains = data_in_height
  elif k > 192 and k != 256:
    denom = align_in * dtypes.float16.itemsize
    grains = (2 * 32768 + denom - 1) // denom
    grains = (grains + 1) & ~1
    feature_grains = max(80, grains)
  weight_bytes_per_kernel = align_in * dtypes.float16.itemsize
  fd_bytes = data_in_width * data_in_height * align_in * dtypes.float16.itemsize
  data_bank = max(1, min(11, (fd_bytes + 32768 - 1) // 32768))
  line_stride = data_in_width * 4
  if 32 < k < 512 and k not in (64, 256): line_stride = min(13, (k + 31) // 32) * 4
  surf_groups = data_in_height // 4
  surf_stride = (line_stride * (surf_groups - 1) + int(surf_groups == 0)) * int(align_in >= 64)
  if (32 < k < 64) or (64 < k <= 128) or (128 < k < 256) or (256 < k < 512): surf_stride = 0
  dst_surf_stride = 64 if is_matmul_64 else (256 if is_matmul_256 else out_width_stride)
  notch_blocks = min(13, align_out // 32)
  notch_val = 8 * notch_blocks - 1
  if is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or k > 7872: notch_val = 0
  return {
    "m":m, "n":n, "k":k, "align_in":align_in, "align_out":align_out,
    "data_in_width":data_in_width, "data_in_height":data_in_height,
    "dataout_width":dataout_width, "dataout_height":dataout_height,
    "feature_grains":feature_grains, "weight_bytes_per_kernel":weight_bytes_per_kernel,
    "data_bank":data_bank, "line_stride":line_stride, "surf_stride":surf_stride,
    "dst_surf_stride":dst_surf_stride, "notch_val":notch_val,
  }

def build_lut(op:Ops, arg, lut_size:int) -> tuple[list[int], float, float|None]:
  lut = [0] * lut_size * 2
  index_shift, index_scale, inv_scale = 5, 0.0, None
  if op is Ops.EXP2:
    x_min, x_max = -2.0, 2.0
    step = (x_max - x_min) / (len(lut) - 1)
    index_scale = (1 << index_shift) / step
    max_val = max(math.exp2(x_min), math.exp2(x_max))
    inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
    for i in range(len(lut)):
      x = x_min + i * step
      y = math.exp2(x) * inv_scale
      q = int(math.floor((y + 1.0) * 2**14 + 0.5))
      lut[i] = int(np.clip(q, 0, 32767))
  elif op is Ops.CUSTOM and arg == "silu":
    x_min, x_max = 0, 5.8
    step = (x_max - x_min) / (lut_size - 1)
    index_scale = (1 << index_shift) / step
    max_val = max(x_min / (1.0 + math.exp(-x_min)), x_max / (1.0 + math.exp(-x_max)))
    inv_scale = 1.0 / max_val if max_val > 1.0 else 1.0
    for i in range(lut_size * 2):
      x = (i - lut_size + (i < lut_size)) * step
      y = x / (1.0 + math.exp(-x)) * inv_scale
      q = int(math.floor(y * (2**15 - 1) + 0.5)) if y >= 0.0 else int(math.ceil(y * (2**15 - 1) - 0.5))
      lut[i] = int(np.clip(q, -32768, 32767))
  elif op is Ops.TRUNC:
    max_val = 1 << 14
    for table_id in range(2):
      base = table_id * lut_size
      for i in range(lut_size): lut[base + i] = 0 if (i % 2 == 0) else max_val
  return lut, index_scale, inv_scale

def lut_enabled(op:Ops, arg) -> bool:
  return op in (Ops.EXP2, Ops.TRUNC) or (op is Ops.CUSTOM and arg == "silu")

def emit_runtime_boilerplate(prg, op, size, arg, feature_addr=0, weight_addr=0, dst_addr=0, wmma_meta:dict[str, int]|None=None):
  if prg.lut_enable:
    lut, index_scale, inv_scale = build_lut(op, arg, prg.lut_size)
    if inv_scale is not None: prg.inv_scale = inv_scale
    bn_mul_operand = int(np.float16(index_scale).view(np.int16)) if index_scale!=0 else 0x3C00

    prg.fill_lut(lut)
    prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_CFG,
        prg.reg(1, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK) |
        prg.reg(1, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK) |
        prg.reg(2, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__MASK))
    index_select = 14 if op is Ops.TRUNC else 5
    prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_INFO,
        prg.reg(index_select, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK) |
        prg.reg(index_select, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK))
    if op is Ops.TRUNC:
      prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
          prg.reg(0x00000000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
      prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_END,
          prg.reg(0x44000000, rk.DPU_LUT_LE_END_LUT_LE_END__SHIFT, rk.DPU_LUT_LE_END_LUT_LE_END__MASK))
      prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_START,
          prg.reg(0x44000000, rk.DPU_LUT_LO_START_LUT_LO_START__SHIFT, rk.DPU_LUT_LO_START_LUT_LO_START__MASK))
      prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
          prg.reg(0x44800000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
    else:
      prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START,
          prg.reg(0xffffc000, rk.DPU_LUT_LE_START_LUT_LE_START__SHIFT, rk.DPU_LUT_LE_START_LUT_LE_START__MASK))
      prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END,
          prg.reg(0x00004000, rk.DPU_LUT_LO_END_LUT_LO_END__SHIFT, rk.DPU_LUT_LO_END_LUT_LO_END__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE,
        prg.reg(23107, rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__SHIFT,
                rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT,
        prg.reg(22, rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__SHIFT,
                rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
      prg.reg(2, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
      prg.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
      prg.reg(bn_mul_operand, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))

  elif op is Ops.CUSTOM and arg == "cmplt_diff2bool":
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
      prg.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      prg.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
      prg.reg(0x33800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
      prg.reg(0x4000, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
      prg.reg(4, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
      prg.reg(1, rk.DPU_BN_CFG_BN_RELUX_EN__SHIFT, rk.DPU_BN_CFG_BN_RELUX_EN__MASK) |
      prg.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
      prg.reg(0x7C00, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE,
      prg.reg(0x3F800000, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__SHIFT, rk.DPU_BN_RELUX_CMP_VALUE_BN_RELUX_CMP_DAT__MASK))
  elif op is Ops.CUSTOM and arg == "cmpeq_diff_zero_to_nan_to_32800":
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
      prg.reg(2, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      prg.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
      prg.reg(0x7C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
      prg.reg(1, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
  elif op is Ops.CUSTOM and arg == "cmpeq_32800_to_bool":
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
      prg.reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      prg.reg(0, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
      prg.reg(0x47001F00, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
      prg.reg(0x3C00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
      prg.reg(0, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))

  if op is Ops.WMMA:
    p = wmma_meta if wmma_meta is not None else wmma_params(2, 2, 2)
    prg.emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
      prg.reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      prg.reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      prg.reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))

    is_kn_64 = p["k"] == 64 and p["n"] == 64
    is_kn_256 = p["k"] == 256 and p["n"] == 256
    is_kn_512 = p["k"] == 512 and p["n"] == 512
    is_kn_lg_512 = p["k"] > 512 and p["n"] > 512
    is_m_1_kn_768 = p["m"] == 1 and p["k"] == 768 and p["n"] == 768
    is_m_1_k768_n2048 = p["m"] == 1 and p["k"] == 768 and p["n"] == 2048
    is_m_1_kn_2048 = p["m"] == 1 and p["k"] == 2048 and p["n"] == 2048
    conv_con1 = prg.reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) | \
                prg.reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK)
    if not (is_kn_64 or is_kn_256 or is_kn_512 or is_kn_lg_512 or is_m_1_kn_768 or is_m_1_k768_n2048 or is_m_1_kn_2048):
      conv_con1 |= prg.reg(1, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK)
    prg.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1)
    prg.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON2,
      prg.reg(p["feature_grains"], rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CONV_CON3,
      prg.reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
      prg.reg(1, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE0,
      prg.reg(p["data_in_width"], rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
      prg.reg(p["data_in_height"], rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE1,
      prg.reg(p["align_in"]-1, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK) |
      prg.reg(p["align_in"], rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE2,
      prg.reg(p["dataout_width"], rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_DATA_SIZE3,
      prg.reg(p["dataout_width"]*p["dataout_height"], rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0,
      prg.reg(p["weight_bytes_per_kernel"]*p["align_out"], rk.CNA_WEIGHT_SIZE0_WEIGHT_BYTES__SHIFT, rk.CNA_WEIGHT_SIZE0_WEIGHT_BYTES__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1,
      prg.reg(p["weight_bytes_per_kernel"], rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__SHIFT,
               rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
      prg.reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK) |
      prg.reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK) |
      prg.reg(p["align_out"], rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CBUF_CON0,
      prg.reg(12-p["data_bank"], rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      prg.reg(p["data_bank"], rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CBUF_CON1,
      prg.reg((p["data_in_width"]*p["align_in"]+31)//32, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT,
               rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON0,
      prg.reg(1, rk.CNA_CVT_CON0_DATA_SIGN__SHIFT, rk.CNA_CVT_CON0_DATA_SIGN__MASK) |
      prg.reg(1, rk.CNA_CVT_CON0_CVT_TYPE__SHIFT, rk.CNA_CVT_CON0_CVT_TYPE__MASK) |
      prg.reg(1, rk.CNA_CVT_CON0_CVT_BYPASS__SHIFT, rk.CNA_CVT_CON0_CVT_BYPASS__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON1,
      prg.reg(1, rk.CNA_CVT_CON1_CVT_SCALE0__SHIFT, rk.CNA_CVT_CON1_CVT_SCALE0__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON2,
      prg.reg(1, rk.CNA_CVT_CON2_CVT_SCALE1__SHIFT, rk.CNA_CVT_CON2_CVT_SCALE1__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON3,
      prg.reg(1, rk.CNA_CVT_CON3_CVT_SCALE2__SHIFT, rk.CNA_CVT_CON3_CVT_SCALE2__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_CVT_CON4,
      prg.reg(1, rk.CNA_CVT_CON4_CVT_SCALE3__SHIFT, rk.CNA_CVT_CON4_CVT_SCALE3__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR,
      prg.reg(feature_addr, rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__SHIFT,
                rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__MASK))

    prg.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON0,
      prg.reg(15, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__MASK) |
      prg.reg(15, rk.CNA_DMA_CON0_DATA_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_DATA_BURST_LEN__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON1,
      prg.reg(p["line_stride"], rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_DMA_CON2,
      prg.reg(p["surf_stride"], rk.CNA_DMA_CON2_SURF_STRIDE__SHIFT, rk.CNA_DMA_CON2_SURF_STRIDE__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0,
      prg.reg(p["data_in_width"], rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK) |
      prg.reg(p["data_in_height"], rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1,
      prg.reg(p["align_in"], rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__SHIFT, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK))
    prg.emit_raw(rk.CNA, rk.REG_CNA_DCOMP_ADDR0,
      prg.reg(weight_addr, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__SHIFT,
                rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK))

    prg.emit_raw(rk.CORE, rk.REG_CORE_MISC_CFG,
      prg.reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK) |
      prg.reg(1, rk.CORE_MISC_CFG_QD_EN__SHIFT, rk.CORE_MISC_CFG_QD_EN__MASK))
    prg.emit_raw(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0,
      prg.reg(p["dataout_height"]-1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
      prg.reg(p["dataout_width"]-1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
    prg.emit_raw(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1,
      prg.reg(p["align_out"]-1, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK))

    prg.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      prg.reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      prg.reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      prg.reg(5, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      prg.reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      prg.reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      prg.reg(dst_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT,
                rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      prg.reg(p["dst_surf_stride"], rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      prg.reg(p["dataout_width"]-1, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      prg.reg(p["dataout_height"]-1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,
      prg.reg(p["notch_val"], rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__MASK) |
      prg.reg(p["notch_val"], rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      prg.reg(p["align_out"]-1, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      prg.reg(p["align_out"]-1, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
      prg.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      prg.reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      prg.reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      prg.reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      prg.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
      prg.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
      prg.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
      prg.reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      prg.reg(p["align_out"]-1, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      prg.reg(p["dataout_height"]-1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      prg.reg(p["dataout_width"]-1, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
      prg.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      prg.reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      prg.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      prg.reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
      prg.reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      prg.reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      prg.reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      prg.reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      prg.reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    prg.emit_raw(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      prg.reg(p["dst_surf_stride"]*4, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    return

  burst_len = 15
  output_mode  = 2
  flying_mode = 1
  channel = 7
  dataout_height = 0
  dataout_width = math.ceil(size / ((dataout_height+1) * (channel+1))) - 1

  precision_float16 = 2

  ew_cvt_type = 0
  ew_data_mode = 1
  ew_data_size = 2
  ew_relu_bypass = arg != "relu"
  ew_alu_algo = prg.hardware_ops.get(op, 0)
  ew_op_src = 1
  erdma_data_size_16bit=2
  if prg.lut_enable:
    ew_data_mode = 0; ew_data_size = 0; ew_op_src = 0

  prg.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      prg.reg(burst_len, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      prg.reg(output_mode, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      prg.reg(flying_mode, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
  prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      prg.reg(precision_float16, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      prg.reg(precision_float16, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      prg.reg(precision_float16, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
  prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      prg.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      prg.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
  prg.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      prg.reg(dataout_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
  prg.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
      prg.reg(ew_cvt_type, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
      prg.reg(ew_data_mode, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
      prg.reg(ew_data_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
      prg.reg(ew_alu_algo, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
      prg.reg(op == Ops.MUL, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
      prg.reg(ew_relu_bypass, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      prg.reg(op in [Ops.MUL, Ops.FDIV] or prg.lut_enable, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      prg.reg(prg.lut_enable == False, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      prg.reg(ew_op_src, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
      prg.reg(prg.lut_enable == True, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      prg.reg(arg in ["cmplt_diff2bool", "cmpeq_diff_zero_to_nan_to_32800", "cmpeq_32800_to_bool"],
              rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
  prg.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
    prg.reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK)) if op == Ops.FDIV else None
  prg.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      prg.reg(dataout_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
  prg.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      prg.reg(dataout_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
  prg.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      prg.reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
  prg.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      prg.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      prg.reg(erdma_data_size_16bit, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))

def build_elementwise_template(op:Ops, size:int, out_arg:int=0, input_arg:int=1, weight_arg:int=2, arg=None,
                               target:str="rk3588-rknpu2") -> RKTemplatePackage:
  if op not in EW_ALU_OPS:
    if not (lut_enabled(op, arg) or op is Ops.CUSTOM): raise RuntimeError(f"unsupported Rockchip elementwise op {op}")
    mock = type("RockchipTemplateEmitter", (), {})()
    mock.q, mock.lut_enable, mock.lut_size, mock.inv_scale, mock.hardware_ops = [], lut_enabled(op, arg), 513, 1.0, EW_ALU_OPS
    mock.reg = rk_field
    mock.emit_raw = lambda target, reg, value: mock.q.append(rkcmd(target, reg, value))
    def fill_lut(lut):
      for table_id, base in ((0, 0), (1, mock.lut_size)):
        mock.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG,
          rk_field(1, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK) |
          rk_field(table_id, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK) |
          rk_field(0, rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ADDR__MASK))
        for i in range(mock.lut_size):
          mock.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
            rk_field(lut[base + i], rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))
    mock.fill_lut = fill_lut
    emit_runtime_boilerplate(mock, op, size, arg)
    regcmd = mock.q
    output_patch, input_patch, weight_patch = len(regcmd), len(regcmd)+1, len(regcmd)+2
    regcmd += [
      rkcmd(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0),
      rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0),
      0x2001000178495044,
      0x0081000000180008,
    ]
    patches = (
      RKPatch("regcmd", output_patch, "dma32", out_arg, "output", mask=rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK),
      RKPatch("regcmd", input_patch, "dma32", input_arg, "input", mask=rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK),
      RKPatch("regcmd", weight_patch, "dma32", weight_arg, "weight", mask=rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK),
    )
    tasks = (RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=0, regcfg_amount=len(regcmd)),)
    return RKTemplatePackage(RK_TEMPLATE_VERSION, target, "elementwise", tuple(regcmd), patches, tasks, op=op, size=size, meta={"arg":arg})
  channel, dataout_height = 7, 0
  dataout_width = math.ceil(size / ((dataout_height + 1) * (channel + 1))) - 1
  regcmd = [
    rkcmd(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      rk_field(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      rk_field(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      rk_field(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK)),
    rkcmd(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      rk_field(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      rk_field(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      rk_field(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK)),
    rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      rk_field(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      rk_field(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK)),
    rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      rk_field(dataout_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK)),
    rkcmd(rk.DPU, rk.REG_DPU_EW_CFG,
      rk_field(0, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
      rk_field(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
      rk_field(2, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
      rk_field(EW_ALU_OPS[op], rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
      rk_field(op == Ops.MUL, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
      rk_field(arg != "relu", rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      rk_field(op in [Ops.MUL, Ops.FDIV], rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      rk_field(True, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      rk_field(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
      rk_field(False, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      rk_field(False, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK)),
    rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      rk_field(dataout_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK)),
    rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      rk_field(dataout_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK)),
    rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      rk_field(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK)),
    rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      rk_field(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      rk_field(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK)),
  ]
  output_patch, input_patch, weight_patch = len(regcmd), len(regcmd)+1, len(regcmd)+2
  regcmd += [
    rkcmd(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
    rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0),
    rkcmd(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0),
  ]
  if op is not Ops.FDIV: regcmd.append(0x2001000178495044)
  regcmd.append(0x0081000000180008)
  patches = (
    RKPatch("regcmd", output_patch, "dma32", out_arg, "output", mask=rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK),
    RKPatch("regcmd", input_patch, "dma32", input_arg, "input", mask=rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK),
    RKPatch("regcmd", weight_patch, "dma32", weight_arg, "weight", mask=rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK),
  )
  tasks = (RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=0, regcfg_amount=len(regcmd)),)
  return RKTemplatePackage(RK_TEMPLATE_VERSION, target, "elementwise", tuple(regcmd), patches, tasks, op=op, size=size)

def _regcmd_index(regcmd:list[int], target:int, reg:int) -> int:
  for i, cmd in enumerate(regcmd):
    if ((cmd >> 48) & 0xffff) == ((target + 1) & 0xffff) and (cmd & 0xffff) == reg: return i
  raise RuntimeError(f"missing Rockchip register command target={target:#x} reg={reg:#x}")

def build_wmma_template(p:dict[str, int], out_arg:int=0, input_arg:int=1, weight_arg:int=2,
                        target:str="rk3588-rknpu2") -> RKTemplatePackage:
  mock = type("RockchipTemplateEmitter", (), {})()
  mock.q, mock.lut_enable, mock.lut_size, mock.inv_scale, mock.hardware_ops = [], False, 513, 1.0, EW_ALU_OPS
  mock.reg = rk_field
  mock.emit_raw = lambda target, reg, value: mock.q.append(rkcmd(target, reg, value))
  mock.fill_lut = lambda lut: None
  emit_runtime_boilerplate(mock, Ops.WMMA, int(p["m"]) * int(p["k"]), None, 0, 0, 0, p)
  regcmd = mock.q
  input_patch = _regcmd_index(regcmd, rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR)
  weight_patch = _regcmd_index(regcmd, rk.CNA, rk.REG_CNA_DCOMP_ADDR0)
  output_patch = _regcmd_index(regcmd, rk.DPU, rk.REG_DPU_DST_BASE_ADDR)
  regcmd.append(0x00810000000d0008)
  patches = (
    RKPatch("regcmd", input_patch, "dma32", input_arg, "input", mask=rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__MASK),
    RKPatch("regcmd", weight_patch, "dma32", weight_arg, "weight", mask=rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK),
    RKPatch("regcmd", output_patch, "dma32", out_arg, "output", mask=rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK),
  )
  tasks = (RKTaskTemplate(op_idx=4, enable_mask=0x18, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=0, regcfg_amount=len(regcmd)),)
  return RKTemplatePackage(RK_TEMPLATE_VERSION, target, "wmma", tuple(regcmd), patches, tasks, op=Ops.WMMA, size=int(p["m"]) * int(p["k"]), meta=dict(p))

def build_fused_matmul_template(meta:dict[str, int], target:str="rk3588-rknpu2") -> RKTemplatePackage:
  return RKTemplatePackage(RK_TEMPLATE_VERSION, target, "fused_matmul", (), meta=meta)

def build_conv1x1_template(p:dict[str, int|bool], out_arg:int=0, input_arg:int=1, weight_arg:int=2,
                           target:str="rk3588-rknpu2") -> RKTemplatePackage:
  out_channels, in_channels, spatial = (int(p[x]) for x in ("out_channels", "in_channels", "spatial"))
  align_c, align_out_c = int(p["align_c"]), int(p["align_out_c"])
  regcmd = [
    rkcmd(rk.DPU, rk.REG_DPU_S_POINTER, (1 << 3) | (1 << 2) | (1 << 1)),
    rkcmd(rk.CNA, rk.REG_CNA_CONV_CON1, (2 << 7) | (2 << 4) |
      ((1 << 30) | (1 << 29) | ((7 + in_channels) << 12) if in_channels in (1, 3, 4) else 0)),
    rkcmd(rk.CNA, rk.REG_CNA_CONV_CON2, int(p["feature_grains"]) << 4),
    rkcmd(rk.CNA, rk.REG_CNA_CONV_CON3, (1 << 3) | 1),
    rkcmd(rk.CNA, rk.REG_CNA_DATA_SIZE0, (int(p["width_stride"]) << 16) | 1),
    rkcmd(rk.CNA, rk.REG_CNA_DATA_SIZE1, ((in_channels - 1) << 16) | align_c),
    rkcmd(rk.CNA, rk.REG_CNA_DATA_SIZE2, spatial),
    rkcmd(rk.CNA, rk.REG_CNA_DATA_SIZE3, spatial),
    rkcmd(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, out_channels * align_c * dtypes.float16.itemsize),
    rkcmd(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1, align_c * dtypes.float16.itemsize),
    rkcmd(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2, (1 << 24) | (1 << 16) | out_channels),
    rkcmd(rk.CNA, rk.REG_CNA_CBUF_CON0, ((NPU_CBUF_BANKS - int(p["data_bank"])) << 4) | int(p["data_bank"])),
    rkcmd(rk.CNA, rk.REG_CNA_CBUF_CON1, int(p["cbuf_entries"])),
    rkcmd(rk.CNA, rk.REG_CNA_CVT_CON0, 0x1 if p["use_nhwc"] else 0xB),
    rkcmd(rk.CNA, rk.REG_CNA_CVT_CON1, 1 << 16),
    rkcmd(rk.CNA, rk.REG_CNA_CVT_CON2, 1 << 16),
    rkcmd(rk.CNA, rk.REG_CNA_CVT_CON3, 1 << 16),
    rkcmd(rk.CNA, rk.REG_CNA_CVT_CON4, 1 << 16),
    rkcmd(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR, 0),
    rkcmd(rk.CNA, rk.REG_CNA_DMA_CON0, (15 << 16) | 15),
    rkcmd(rk.CNA, rk.REG_CNA_DMA_CON1, int(p["width_stride"]) if in_channels in (1, 3, 4) else int(p["width_stride"]) * 4),
    rkcmd(rk.CNA, rk.REG_CNA_DMA_CON2, 0),
    rkcmd(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0, (spatial << 16) | 1),
    rkcmd(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1, align_c),
    rkcmd(rk.CNA, rk.REG_CNA_DCOMP_ADDR0, 0),
    rkcmd(rk.CNA, rk.REG_CNA_CVT_CON5, (1 << max(1, min(in_channels if p["use_nhwc"] else align_c, 8))) - 1),
    rkcmd(rk.CORE, rk.REG_CORE_MISC_CFG, 2 << 8),
    rkcmd(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0, spatial - 1),
    rkcmd(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1, align_out_c - 1),
    ((rk.CORE | 0x1) << 48) | CORE_RESERVED_ZERO_ADDR,
    rkcmd(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG, (15 << 5) | (2 << 1)),
    rkcmd(rk.DPU, rk.REG_DPU_DATA_FORMAT, (2 << 29) | (2 << 26) | 2),
    rkcmd(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, 0),
    rkcmd(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, int(p["out_width_stride"]) << 4),
    rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH, spatial - 1),
    rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, 0),
    rkcmd(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL, ((out_channels - 1) << 16) | (align_out_c - 1)),
    rkcmd(rk.DPU, rk.REG_DPU_BS_CFG, 0x53),
    rkcmd(rk.DPU, rk.REG_DPU_BS_OW_CFG, (1 << 8) | (1 << 5) | (1 << 2) | (1 << 1)),
    rkcmd(rk.DPU, rk.REG_DPU_WDMA_SIZE_0, align_out_c - 1),
    rkcmd(rk.DPU, rk.REG_DPU_WDMA_SIZE_1, spatial - 1),
    rkcmd(rk.DPU, rk.REG_DPU_BN_CFG, 0x53),
    rkcmd(rk.DPU, rk.REG_DPU_EW_CFG, 0x383),
    rkcmd(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE, 1),
    rkcmd(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE, (1 << 16) | 1),
    rkcmd(rk.DPU, rk.REG_DPU_SURFACE_ADD, int(p["surface_add"]) << 4),
    (0x1 << 48) | DPU_RESERVED_ZERO_ADDR,
    (0x81 << 48) | (0xD << 16) | rk.REG_PC_OPERATION_ENABLE,
  ]
  patches = (
    RKPatch("regcmd", 18, "dma32", input_arg, "input", mask=rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__MASK),
    RKPatch("regcmd", 24, "dma32_add", weight_arg, "weight", mask=rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK, addend=REGCMD_RESERVED),
    RKPatch("regcmd", 32, "dma32", out_arg, "output", mask=rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK),
  )
  tasks = (RKTaskTemplate(op_idx=1, enable_mask=0xd, int_mask=0x300, int_clear=0x1ffff, regcfg_offset=0, regcfg_amount=len(regcmd)),)
  return RKTemplatePackage(RK_TEMPLATE_VERSION, target, "conv1x1", tuple(regcmd), patches, tasks, meta=dict(p))

def elementwise_meta(uops:list[UOp], hardware_ops:set[Ops]) -> tuple[Ops, int, int, int, int, object]|None:
  params = [(i,u) for i,u in enumerate(uops) if u.op is Ops.PARAM]
  if len(params) not in (2, 3) or any(not isinstance(u.dtype, PtrDType) or u.dtype.base.scalar() is not dtypes.half for _,u in params): return None
  size = params[0][1].dtype.size
  if any(u.dtype.size != size for _,u in params): return None

  def load_info(u:UOp):
    gep = None
    if u.op is Ops.GEP: gep, u = u.arg, u.src[0]
    if u.op is not Ops.LOAD: return None
    idx = u.src[0]
    if idx.op is Ops.CAST: idx = idx.src[0]
    if idx.op is not Ops.INDEX: return None
    return (uops.index(idx.src[0]), idx.src[1], gep)

  flat_op, flat_arg = None, None
  for u in uops:
    if u.op is not Ops.STORE: continue
    dst = u.src[0]
    if dst.op is Ops.CAST: dst = dst.src[0]
    if dst.op is not Ops.INDEX or uops.index(dst.src[0]) != params[0][0]: return None
    vals = u.src[1].src if u.src[1].op is Ops.STACK else (u.src[1],)
    for j,v in enumerate(vals):
      if v.dtype.scalar() is not dtypes.half: return None
      if len(params) == 2:
        if not (lut_enabled(v.op, v.arg) or v.op is Ops.CUSTOM): return None
        if flat_op is None: flat_op, flat_arg = v.op, v.arg
        if flat_op is not v.op or flat_arg != v.arg: return None
        src = load_info(v.src[0]) if len(v.src) == 1 else None
        if src is None or src[0] != params[1][0] or src[1] is not dst.src[1]: return None
        if src[2] is not None and src[2] != (j,): return None
      else:
        if v.op not in hardware_ops or len(v.src) != 2: return None
        if flat_op is None: flat_op = v.op
        if flat_op is not v.op: return None
        lhs, rhs = load_info(v.src[0]), load_info(v.src[1])
        if lhs is None or rhs is None or lhs[0] != params[1][0] or rhs[0] != params[2][0]: return None
        if lhs[1] is not dst.src[1] or rhs[1] is not dst.src[1] or lhs[2] != rhs[2]: return None
        if lhs[2] is not None and lhs[2] != (j,): return None
  return (flat_op, size, 0, 1, 1 if len(params) == 2 else 2, flat_arg) if flat_op is not None else None

def conv1x1_meta(uops:list[UOp]) -> tuple[int, int, int, int, int, int]|None:
  if os.getenv("ROCKCHIP_NATIVE_CONV", "1") == "0": return None
  params = [(i,u) for i,u in enumerate(uops) if u.op is Ops.PARAM]
  if len(params) != 3 or any(not isinstance(u.dtype, PtrDType) or u.dtype.base.scalar() is not dtypes.half for _,u in params): return None
  out_size, in_size, weight_size = (u.dtype.size for _,u in params)
  candidates = []
  for spatial in range(1, min(out_size, in_size) + 1):
    if out_size % spatial == 0 and in_size % spatial == 0 and (out_size // spatial) * (in_size // spatial) == weight_size:
      candidates.append(spatial)
  if not candidates: return None
  spatial = max(candidates)
  out_channels, in_channels = out_size // spatial, in_size // spatial
  if out_channels <= 0 or in_channels <= 0 or spatial <= 0 or in_channels > 4: return None
  if not any(u.op is Ops.WMMA for u in uops): return None
  return (params[0][1].arg, params[1][1].arg, params[2][1].arg, in_channels, out_channels, spatial)

FUSED_MATMUL_KEYS = ("m", "n", "k", "batch", "a_bs", "b_bs", "c_bs", "a_ms", "a_ks", "b_ks", "b_ns", "c_ms", "c_ns",
                     "a_slot", "b_slot", "c_slot", "ta", "tb", "a_dt", "b_dt", "c_dt")

def fused_matmul_meta(uops:list[UOp]) -> dict[str, int]|None:
  for u in uops:
    if u.op is not Ops.SINK or not hasattr(u.arg, "applied_opts"): continue
    for opt in u.arg.applied_opts:
      if isinstance(opt, tuple) and len(opt) == 2 and opt[0] == "ROCKCHIP_FUSED_MATMUL":
        vals = opt[1]
        if isinstance(vals, tuple) and len(vals) == len(FUSED_MATMUL_KEYS) and all(isinstance(x, int) and x >= 0 for x in vals):
          return dict(zip(FUSED_MATMUL_KEYS, vals))
  return None
