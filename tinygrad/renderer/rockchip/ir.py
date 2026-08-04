from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, IntEnum

from tinygrad.dtype import dtypes, DType
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId
from tinygrad.uop.ops import Ops

class RKTarget(IntEnum): RK3588 = 1
class RKEngine(IntEnum):
  DPU = 1
  CMAC = 2
  PPU = 3
  CONV = 4
class RKBufferKind(IntEnum):
  ARG = 0
  SCRATCH = 1
  CONSTANT = 2
class RKLayoutKind(Enum):
  LINEAR = "linear"
  DPU_FEATURE = "dpu_feature"
  CMAC_ACTIVATION = "cmac_activation"
  CMAC_WEIGHT = "cmac_weight"
  PPU_HWC8 = "ppu_hwc8"
  CNA_ACTIVATION = "cna_activation"
  CNA_WEIGHT = "cna_weight"
  CONV_OUTPUT = "conv_output"
class RKReformatKind(Enum):
  COALESCED_DPU = "coalesced_dpu"
  SELECTOR_CMAC = "selector_cmac"
class RKConvSplit(Enum):
  NONE = "none"
  BY_Y = "by_y"
  BY_K = "by_k"
  BY_YK = "by_yk"

@dataclass(frozen=True)
class RKArg:
  kind: RKBufferKind
  index: int
  addend: int = 0

RK_ALU_OPS = frozenset((Ops.ADD, Ops.MUL, Ops.MAX, Ops.FDIV, Ops.SUB))

@dataclass(frozen=True)
class RKALUStage:
  op: Ops
  dst: RKArg
  lhs: RKArg|float
  rhs: RKArg|float
  count: int
  out_dtype: DType = dtypes.half
  def __post_init__(self):
    if self.op not in RK_ALU_OPS: raise ValueError(f"unsupported RK DPU ALU operation {self.op}")

@dataclass(frozen=True)
class RKFusedALUStage:
  """One ordered BS/BN/EW arithmetic pipeline over flying MRDMA data."""
  dst: RKArg
  main: RKArg
  bs_op: Ops
  bs: RKArg
  bn_op: Ops
  bn: RKArg|float
  ew_op: Ops
  ew: RKArg
  count: int
  def __post_init__(self):
    if (self.bs_op,self.bn_op,self.ew_op) != (Ops.SUB,Ops.MUL,Ops.ADD):
      raise ValueError("unsupported RK fused DPU arithmetic pipeline")
    if not 0 < self.count <= 8: raise ValueError("RK fused DPU arithmetic needs one eight-channel atom")

@dataclass(frozen=True)
class RKMaskStage:
  dst: RKArg
  src: RKArg
  count: int

@dataclass(frozen=True)
class RKLUTStage:
  lut: RKLUTId
  dst: RKArg
  src: RKArg
  count: int

RKDPUStage = RKALUStage|RKFusedALUStage|RKMaskStage|RKLUTStage

@dataclass(frozen=True)
class RKScratch:
  size: int
  alignment: int = 4096

@dataclass(frozen=True)
class RKDPUProgram:
  stages: tuple[RKDPUStage, ...]
  scratch: tuple[RKScratch, ...] = ()

@dataclass(frozen=True)
class RKLayout:
  logical_shape: tuple[int, ...]
  physical_shape: tuple[int, ...]
  strides_bytes: tuple[int, ...]
  dtype: DType
  base_offset: int = 0
  row_alignment: int = 16
  channel_alignment: int = 8
  padding: tuple[tuple[int, int], ...] = ()
  kind: RKLayoutKind = RKLayoutKind.LINEAR
  padding_value: float|int|None = None
  def __post_init__(self):
    rank = len(self.logical_shape)
    if len(self.physical_shape) != rank or len(self.strides_bytes) != rank or self.padding and len(self.padding) != rank:
      raise ValueError("RKLayout rank mismatch")
    if any(logical < 0 or physical < logical for logical,physical in zip(self.logical_shape, self.physical_shape)):
      raise ValueError("RKLayout physical shape does not contain its logical shape")
    if self.base_offset < 0 or self.row_alignment <= 0 or self.channel_alignment <= 0 or any(stride <= 0 for stride in self.strides_bytes):
      raise ValueError("invalid RKLayout alignment or stride")
  def byte_size(self) -> int:
    return 0 if any(extent == 0 for extent in self.physical_shape) else \
      self.base_offset+sum((extent-1)*stride for extent,stride in zip(self.physical_shape,self.strides_bytes))+self.dtype.itemsize
  def is_dense(self) -> bool:
    stride = self.dtype.itemsize
    for extent,actual in zip(reversed(self.physical_shape),reversed(self.strides_bytes)):
      if actual != stride: return False
      stride *= extent
    return True
  def can_view_as(self, other:RKLayout) -> bool:
    return self.dtype is other.dtype and self.base_offset == other.base_offset and self.is_dense() and other.is_dense() and \
      math.prod(self.logical_shape) == math.prod(other.logical_shape) and math.prod(self.physical_shape) == math.prod(other.physical_shape)
  def padding_is_initialized(self) -> bool:
    has_padding = any(logical != physical for logical,physical in zip(self.logical_shape,self.physical_shape)) or \
      any(before or after for before,after in self.padding)
    return not has_padding or self.padding_value is not None
  def is_legal_for(self, engine:RKEngine) -> bool:
    if engine is RKEngine.DPU:
      return self.kind in (RKLayoutKind.LINEAR,RKLayoutKind.DPU_FEATURE,RKLayoutKind.CONV_OUTPUT) and \
        self.dtype in (dtypes.half,dtypes.float,dtypes.int) and self.strides_bytes[-1] == self.dtype.itemsize
    if engine is RKEngine.CMAC:
      return self.kind in (RKLayoutKind.LINEAR,RKLayoutKind.CMAC_ACTIVATION,RKLayoutKind.CMAC_WEIGHT) and \
        self.dtype in (dtypes.half,dtypes.float) and self.strides_bytes[-1] == self.dtype.itemsize
    if engine is RKEngine.PPU:
      return self.kind is RKLayoutKind.PPU_HWC8 and self.dtype is dtypes.half and len(self.physical_shape) == 3 and \
        self.physical_shape[-1] == 8 and self.strides_bytes[-1] == 2
    assert engine is RKEngine.CONV
    return self.kind in (RKLayoutKind.CNA_ACTIVATION,RKLayoutKind.CNA_WEIGHT,RKLayoutKind.CONV_OUTPUT) and \
      self.dtype is dtypes.half and self.strides_bytes[-1] == 2
  def requires_reformat_for(self, engine:RKEngine) -> bool: return not self.is_legal_for(engine)
  def validate_for(self, engine:RKEngine) -> None:
    if not self.is_legal_for(engine): raise ValueError(f"{self.kind.value} layout is not legal for {engine.name}")

@dataclass(frozen=True)
class RKTensorRef:
  buffer: RKArg
  layout: RKLayout

@dataclass(frozen=True)
class RKEpilogue:
  bias: RKTensorRef|None = None
  relu: bool = False

@dataclass(frozen=True)
class RKContract:
  out: RKTensorRef
  lhs: RKTensorRef
  rhs: RKTensorRef
  reduce_axis: int
  constants: bytes = b""
  epilogue: RKEpilogue|None = None
  compact_output: bool = False

@dataclass(frozen=True)
class RKSpatialConv:
  """One proven FP16 spatial convolution over packed RK3588 surfaces."""
  out: RKTensorRef
  src: RKTensorRef
  weight: RKTensorRef
  in_channels: int
  out_channels: int
  input_height: int
  input_width: int
  kernel_height: int
  kernel_width: int
  output_height: int
  output_width: int
  stride_y: int
  stride_x: int
  input_width_stride: int
  output_width_stride: int

@dataclass(frozen=True)
class RKConvTile:
  y_start: int
  input_y_start: int
  input_height: int
  output_height: int
  k_start: int
  out_channels: int
  data_banks: int
  weight_banks: int
  def __post_init__(self):
    if min(self.y_start,self.input_y_start,self.k_start) < 0 or \
       min(self.input_height,self.output_height,self.out_channels,self.data_banks,self.weight_banks) <= 0 or \
       self.data_banks+self.weight_banks > 12: raise ValueError("invalid RK3588 CBUF convolution tile")

@dataclass(frozen=True)
class RKConvTiling:
  split: RKConvSplit
  y_step: int
  k_step: int
  tiles: tuple[RKConvTile, ...]
  def __post_init__(self):
    if not self.tiles or min(self.y_step,self.k_step) <= 0: raise ValueError("empty RK3588 convolution tiling")

@dataclass(frozen=True)
class RKReduce:
  out: RKTensorRef
  src: RKTensorRef
  op: Ops
  reduce_axis: int

@dataclass(frozen=True)
class RKPool(RKReduce):
  """One proven FP16 sliding-pooling task over dense HWC8 surfaces."""
  out: RKTensorRef
  src: RKTensorRef
  op: Ops
  kernel_height: int
  kernel_width: int
  stride_y: int
  stride_x: int

@dataclass(frozen=True)
class RKReformatPlan:
  """One logical static physical-layout transform, before selecting engine tasks."""
  out: RKTensorRef
  src: RKTensorRef
  mapping: tuple[int, ...]
  fill: float = 0.0
  def __post_init__(self):
    out_count, src_count = math.prod(self.out.layout.logical_shape), math.prod(self.src.layout.logical_shape)
    if len(self.mapping) != out_count or any(index < -1 or index >= src_count for index in self.mapping):
      raise ValueError("RKReformatPlan mapping is outside its logical surfaces")

RKProgramStep = RKDPUProgram|RKContract|RKSpatialConv|RKReduce

@dataclass(frozen=True)
class RKProgram:
  steps: tuple[RKProgramStep, ...]
  scratch: tuple[RKScratch, ...] = ()
  def __post_init__(self):
    if not self.steps: raise ValueError("Rockchip program has no steps")
    if any(isinstance(step, RKDPUProgram) and step.scratch and step.scratch != self.scratch for step in self.steps):
      raise ValueError("Rockchip step scratch does not match program resources")

@dataclass(frozen=True)
class RKLegalizedReformat:
  """A semantic reformat paired with one selected, UOp-free physical task schedule."""
  plan: RKReformatPlan
  kind: RKReformatKind
  program: RKProgram

@dataclass(frozen=True)
class RKPlanCost:
  task_count: int
  command_words: int
  reset_count: int
  constant_bytes: int
  scratch_bytes: int
  estimated_read_bytes: int
  estimated_write_bytes: int
  estimated_macs: int
  @property
  def stage_count(self) -> int: return self.task_count

class RKRejectKind(Enum):
  UNSUPPORTED_INPUT_DTYPE = "unsupported_input_dtype"
  UNSUPPORTED_OUTPUT_DTYPE = "unsupported_output_dtype"
  UNSUPPORTED_ALU = "unsupported_alu"
  UNSUPPORTED_LAYOUT = "unsupported_layout"
  UNALIGNED_ROW = "unaligned_row"
  REQUIRES_REFORMAT = "requires_reformat"
  UNSUPPORTED_BROADCAST = "unsupported_broadcast"
  UNSUPPORTED_REDUCTION = "unsupported_reduction"
  UNSUPPORTED_CONTRACTION = "unsupported_contraction"
  UNSUPPORTED_DYNAMIC_PACK = "unsupported_dynamic_pack"
  PLAN_STAGE_LIMIT = "plan_stage_limit"
  LUT_DOMAIN_UNPROVEN = "lut_domain_unproven"
  NUMERICAL_CONTRACT = "numerical_contract"

@dataclass(frozen=True)
class RKReject:
  kind: RKRejectKind
  detail: str
  node_op: Ops|None = None
  fingerprint: tuple = ()

class RKLowerKind(Enum):
  NATIVE = "native"
  NOT_APPLICABLE = "not_applicable"
  UNSUPPORTED = "unsupported"

@dataclass(frozen=True)
class RKLowerResult:
  kind: RKLowerKind
  plan: RKDPUProgram|RKContract|RKSpatialConv|RKReduce|RKLegalizedReformat|RKProgram|None = None
  reject: RKReject|None = None
  def __post_init__(self):
    valid = {RKLowerKind.NATIVE:self.plan is not None and self.reject is None,
             RKLowerKind.NOT_APPLICABLE:self.plan is None and self.reject is None,
             RKLowerKind.UNSUPPORTED:self.plan is None and self.reject is not None}
    if not valid[self.kind]: raise ValueError(f"invalid {self.kind.value} Rockchip lowering result")
