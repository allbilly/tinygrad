from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, IntEnum

from tinygrad.dtype import dtypes, DType
from tinygrad.runtime.autogen.rockchip_lut import RKLUTId
from tinygrad.uop.ops import Ops
from tinygrad.renderer.rockchip.access import RKAccessMap, RKMultiSourceMap

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
  PPU_HWC = "ppu_hwc"
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
  lhs: RKArg|float|bytes
  rhs: RKArg|float|bytes
  count: int
  out_dtype: DType = dtypes.half
  out_cvt_offset: int = 0
  def __post_init__(self):
    if self.op not in RK_ALU_OPS: raise ValueError(f"unsupported RK DPU ALU operation {self.op}")
    if not 0 <= self.out_cvt_offset <= 0xffffffff: raise ValueError("RK DPU output conversion offset does not fit 32 bits")
    if self.out_cvt_offset and self.out_dtype is not dtypes.int: raise ValueError("RK DPU output conversion offset requires int32 output")
    if any(isinstance(value,bytes) and (self.out_dtype is not dtypes.bool or len(value) != self.count) for value in (self.lhs,self.rhs)):
      raise ValueError("RK DPU byte operand requires one bool value per output")

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
class RKFusedMulStage:
  """Multiply four compact FP16 channel vectors through BS, BN, and EW; WDMA writes whole eight-channel atoms."""
  dst: RKArg
  main: RKArg
  bs: RKArg
  bn: RKArg
  ew: RKArg
  count: int
  def __post_init__(self):
    if not 0 < self.count <= 256: raise ValueError("RK fused DPU multiply requires 1..256 channels")
    if self.count > 8 and self.count%8: raise ValueError("RK fused DPU multiply requires complete atoms after the first eight channels")

@dataclass(frozen=True)
class RKStridedAtomGatherStage:
  """Gather one aligned eight-lane FP16 atom from each strided source row."""
  dst: RKArg
  src: RKArg
  rows: int
  src_row_stride: int
  def __post_init__(self):
    if not 1 <= self.rows <= 128: raise ValueError("RK strided atom gather supports 1..128 rows")
    if not 8 <= self.src_row_stride <= 128 or self.src_row_stride % 8:
      raise ValueError("RK strided atom gather row stride must be 8..128 aligned FP16 values")
    if self.dst.addend % 16 or self.src.addend % 16:
      raise ValueError("RK strided atom gather surfaces must be 16-byte aligned")

@dataclass(frozen=True)
class RKCopyStage:
  dst: RKArg
  src: RKArg|bool
  count: int
  dtype: DType
  def __post_init__(self):
    if self.dtype not in (dtypes.bool,dtypes.int,dtypes.float): raise ValueError("RK native copy only supports bool, int32, or FP32")
    if not 0 < self.count <= 16384: raise ValueError("RK native copy extent exceeds DPU width")

@dataclass(frozen=True)
class RKCastStage:
  dst: RKArg
  src: RKArg
  count: int
  src_dtype: DType
  dst_dtype: DType
  def __post_init__(self):
    if (self.src_dtype,self.dst_dtype) != (dtypes.bool,dtypes.half) or not 0 < self.count <= 8:
      raise ValueError("RK native cast supports one bool-to-FP16 atom")

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

RKDPUStage = RKALUStage|RKFusedALUStage|RKFusedMulStage|RKStridedAtomGatherStage|RKCopyStage|RKCastStage|RKMaskStage|RKLUTStage

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
      return self.kind is RKLayoutKind.PPU_HWC and self.dtype is dtypes.half and len(self.physical_shape) == 3 and \
        2 <= self.physical_shape[-1] <= 8 and self.strides_bytes[-1] == 2
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
class RKContractionPlan:
  """Logical dense contraction before CMAC tile and physical-layout legalization."""
  out: RKTensorRef
  lhs: RKTensorRef
  rhs: RKTensorRef
  logical_m: int
  logical_n: int
  logical_k: int
  reduction_axes: tuple[int, ...]
  constants: bytes = b""
  epilogue: RKEpilogue|None = None
  def __post_init__(self):
    if min(self.logical_m,self.logical_n,self.logical_k) <= 0 or not self.reduction_axes:
      raise ValueError("invalid logical RK contraction geometry")
    if math.prod(self.out.layout.logical_shape) != self.logical_m*self.logical_n or \
       math.prod(self.lhs.layout.logical_shape) != self.logical_m*self.logical_k or \
       math.prod(self.rhs.layout.logical_shape) != self.logical_n*self.logical_k:
      raise ValueError("RK contraction surfaces do not match logical M/N/K")

@dataclass(frozen=True)
class RKCMACTask:
  """One fully legalized physical CMAC invocation."""
  out: RKTensorRef
  lhs: RKTensorRef
  rhs: RKTensorRef
  reduce_axis: int
  constants: bytes = b""
  epilogue: RKEpilogue|None = None
  compact_output: bool = False
  @property
  def physical_m(self) -> int: return self.lhs.layout.physical_shape[0]
  @property
  def physical_n(self) -> int: return self.rhs.layout.physical_shape[0]
  @property
  def physical_k(self) -> int: return self.lhs.layout.physical_shape[-1]

@dataclass(frozen=True)
class RKConvTask:
  """One fully legalized FP16 CONV engine task over packed RK3588 surfaces."""
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
  data_banks: int|None = None
  weight_banks: int|None = None
  pad_top: int = 0
  pad_bottom: int = 0
  pad_left: int = 0
  pad_right: int = 0
  dilation_y: int = 1
  dilation_x: int = 1

@dataclass(frozen=True)
class RKDeconvTask(RKConvTask):
  """One fully legalized FP16 CNA deconvolution task over packed RK3588 surfaces."""
  transpose_stride_y: int = 1
  transpose_stride_x: int = 1
  output_padding_y: int = 0
  output_padding_x: int = 0
  hardware_pad_top: int|None = None
  hardware_pad_left: int|None = None

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
class RKConvPlan:
  """Logical dense convolution over already-legal physical surfaces, before task tiling."""
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
  tiling: RKConvTiling
  pad_top: int = 0
  pad_bottom: int = 0
  pad_left: int = 0
  pad_right: int = 0
  dilation_y: int = 1
  dilation_x: int = 1

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
  pad_top: int = 0
  pad_bottom: int = 0
  pad_left: int = 0
  pad_right: int = 0

@dataclass(frozen=True)
class RKReformatPlan:
  """One logical static physical-layout transform, before selecting engine tasks."""
  out: RKTensorRef
  src: RKTensorRef
  access: RKAccessMap
  fill: float = 0.0
  def __post_init__(self):
    out_count, src_count = math.prod(self.out.layout.logical_shape), math.prod(self.src.layout.logical_shape)
    mapping = self.access.expand()
    if len(mapping) != out_count or any(index < -1 or index >= src_count for index in mapping):
      raise ValueError("RKReformatPlan mapping is outside its logical surfaces")
  @property
  def mapping(self) -> tuple[int, ...]: return self.access.expand()

@dataclass(frozen=True)
class RKMultiSourceReformatPlan:
  """One logical static transform selecting every output from one of several surfaces."""
  out: RKTensorRef
  sources: tuple[RKTensorRef, ...]
  access: RKMultiSourceMap
  def __post_init__(self):
    if not self.sources or self.access.count != math.prod(self.out.layout.logical_shape):
      raise ValueError("RK multi-source reformat has an invalid output map")
    if any(source < 0 or source >= len(self.sources) or index < 0 or
           index >= math.prod(self.sources[source].layout.logical_shape) for source,index in self.access.values()):
      raise ValueError("RK multi-source reformat mapping is outside its logical surfaces")
  @property
  def mapping(self) -> tuple[tuple[int,int], ...]: return self.access.expand()

RKProgramStep = RKDPUProgram|RKCMACTask|RKConvTask|RKReduce

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
  plan: RKReformatPlan|RKMultiSourceReformatPlan
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
  plan: RKDPUProgram|RKCMACTask|RKConvTask|RKReduce|RKLegalizedReformat|RKProgram|None = None
  reject: RKReject|None = None
  def __post_init__(self):
    valid = {RKLowerKind.NATIVE:self.plan is not None and self.reject is None,
             RKLowerKind.NOT_APPLICABLE:self.plan is None and self.reject is None,
             RKLowerKind.UNSUPPORTED:self.plan is None and self.reject is not None}
    if not valid[self.kind]: raise ValueError(f"invalid {self.kind.value} Rockchip lowering result")
