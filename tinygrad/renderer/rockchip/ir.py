from __future__ import annotations
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
class RKBufferKind(IntEnum):
  ARG = 0
  SCRATCH = 1
  CONSTANT = 2
class RKLayoutKind(Enum):
  LINEAR = "linear"
  CMAC_WEIGHT = "cmac_weight"

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

RKDPUStage = RKALUStage|RKMaskStage|RKLUTStage

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
  def __post_init__(self):
    rank = len(self.logical_shape)
    if len(self.physical_shape) != rank or len(self.strides_bytes) != rank or self.padding and len(self.padding) != rank:
      raise ValueError("RKLayout rank mismatch")
    if any(logical < 0 or physical < logical for logical,physical in zip(self.logical_shape, self.physical_shape)):
      raise ValueError("RKLayout physical shape does not contain its logical shape")
    if self.base_offset < 0 or self.row_alignment <= 0 or self.channel_alignment <= 0: raise ValueError("invalid RKLayout alignment")

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

@dataclass(frozen=True)
class RKReduce:
  out: RKTensorRef
  src: RKTensorRef
  op: Ops
  reduce_axis: int

RKProgramStep = RKDPUProgram|RKContract|RKReduce

@dataclass(frozen=True)
class RKProgram:
  steps: tuple[RKProgramStep, ...]
  scratch: tuple[RKScratch, ...] = ()
  def __post_init__(self):
    if not self.steps: raise ValueError("Rockchip program has no steps")
    if any(isinstance(step, RKDPUProgram) and step.scratch and step.scratch != self.scratch for step in self.steps):
      raise ValueError("Rockchip step scratch does not match program resources")

@dataclass(frozen=True)
class RKPlanCost:
  stage_count: int
  constant_bytes: int
  scratch_bytes: int

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
  plan: RKDPUProgram|RKContract|RKReduce|RKProgram|None = None
  reject: RKReject|None = None
  def __post_init__(self):
    valid = {RKLowerKind.NATIVE:self.plan is not None and self.reject is None,
             RKLowerKind.NOT_APPLICABLE:self.plan is None and self.reject is None,
             RKLowerKind.UNSUPPORTED:self.plan is None and self.reject is not None}
    if not valid[self.kind]: raise ValueError(f"invalid {self.kind.value} Rockchip lowering result")
