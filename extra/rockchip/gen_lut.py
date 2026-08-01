#!/usr/bin/env python3
import hashlib, math, pathlib, struct
from dataclasses import dataclass
from typing import Callable

SIZE = 513
def half(value:float) -> float: return struct.unpack("<e", struct.pack("<e", value))[0]
def hardswish(x:float) -> float: return x * min(6.0, max(0.0, x+3.0)) / 6.0
def quick_gelu(x:float) -> float: return x/(1+math.exp(-1.702*x))
def quick_gelu_local(z:float) -> float:
  x, xh = -1.5+z/4, half(-1.5+z/4)
  staged = half(xh*half(1/(1+math.exp(-half(xh*1.702)))))
  return 0.5*quick_gelu(x)+0.5*staged
def gelu(x:float, approximate_tanh:bool) -> float:
  return 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3))) if approximate_tanh else 0.5*x*(1+math.erf(x/math.sqrt(2)))
def mish(x:float) -> float: return x*math.tanh(math.log1p(math.exp(x)))

@dataclass(frozen=True)
class LUT:
  name: str
  function: Callable[[float], float]
  domain: float
  index_scale: float
  output_scale: float
  minus_exp: int
  replace_zero: bool = False
  corrections: tuple[tuple[int, int, int], ...] = ()
  post_scale: float = 1.0

LUTS = (
  LUT("EXP2", math.exp2, 2.0, 8192.0, 8192.0, 13),
  LUT("HARDSWISH", hardswish, 2.0, 8192.0, 16384.0, 14, True),
  LUT("HARDSWISH_LOCAL", lambda z: hardswish(half(z/16))*16, 2.0, 8192.0, 32768.0, 15, True),
  LUT("TANH", math.tanh, 4.0, 4096.0, 32768.0, 15),
  LUT("TANH_LOCAL", lambda z: 4*math.tanh(z/16), 4.0, 4096.0, 32768.0, 15),
  LUT("SIGMOID", lambda x: 1/(1+math.exp(-x)), 8.0, 2048.0, 32768.0, 15),
  LUT("SIGMOID_LOCAL", lambda x: 1/(1+math.exp(-x)), 2.0, 8192.0, 32768.0, 15),
  LUT("QUICK_GELU", quick_gelu, 2.0, 8192.0, 16384.0, 14, True,
      ((0, 276, 4), (0, 375, 1), (0, 408, 1), (0, 427, 1), (1, 49, 1))),
  LUT("QUICK_GELU_LOCAL", quick_gelu_local, 2.0, 8192.0, 32768.0, 15, True),
  LUT("GELU_TANH", lambda x: gelu(x, True)/(4 if x >= 0 else 1), 4.0, 4096.0, 32768.0, 15, True),
  LUT("GELU_TANH_LOCAL", lambda z: 2*gelu(z/8, True), 4.0, 4096.0, 32768.0, 15, True),
  LUT("GELU_EXACT", lambda x: gelu(x, False)/(4 if x >= 0 else 1), 4.0, 4096.0, 32768.0, 15, True),
  LUT("GELU_EXACT_LOCAL", lambda z: 2*gelu(z/8, False), 4.0, 4096.0, 32768.0, 15, True),
  LUT("ERF", math.erf, 4.0, 4096.0, 32768.0, 15, True),
  LUT("ERF_LOCAL", lambda z: 3*math.erf(z/16), 4.0, 4096.0, 32768.0, 15, True),
  LUT("ELU1", lambda x: math.expm1(x) if x < 0 else 0, 8.0, 2048.0, 32768.0, 15, True),
  LUT("ELU1_LOCAL", lambda z: 2*math.expm1(z/4) if z < 0 else 0, 2.0, 8192.0, 32768.0, 15, True),
  LUT("ELU01", lambda x: .8*math.expm1(x) if x < 0 else 0, 8.0, 2048.0, 32768.0, 15, True),
  LUT("ELU01_LOCAL", lambda z: 1.6*math.expm1(z/4) if z < 0 else 0, 2.0, 8192.0, 32768.0, 15, True),
  LUT("SELU", lambda x: .5*1.0507*1.67326*math.expm1(x) if x < 0 else 0, 8.0, 2048.0, 32768.0, 15, True),
  LUT("SELU_LOCAL", lambda z: 1.0507*1.67326*math.expm1(z/4) if z < 0 else 0, 2.0, 8192.0, 32768.0, 15, True),
  LUT("MISH", lambda x: mish(x)/(8 if x >= 0 else 1), 8.0, 2048.0, 32768.0, 15, True),
  LUT("MISH_LOCAL", lambda z: mish(z/2), 2.0, 8192.0, 32768.0, 15, True),
  LUT("LOGSIGMOID", lambda x: -math.log1p(math.exp(-abs(x))), 8.0, 2048.0, 32768.0, 15, True),
  LUT("LOGSIGMOID_TAIL", lambda x: -32*math.log1p(math.exp(-abs(x))), 16.0, 1024.0, 32768.0, 15, True),
  LUT("SOFTPLUS1", lambda x: -math.log1p(math.exp(-abs(x))), 8.0, 2048.0, 32768.0, 15, True),
  LUT("SOFTPLUS1_TAIL", lambda x: -21*math.log1p(math.exp(-abs(x))), 16.0, 1024.0, 32768.0, 15, True),
  LUT("SOFTPLUS3", lambda x: -math.log1p(math.exp(-abs(3*x))), 8/3, 6144.0, 32768.0, 15, True, ((0,344,1),(0,345,1)), 1/3),
  LUT("SOFTPLUS3_TAIL", lambda x: -21*math.log1p(math.exp(-abs(3*x))), 16/3, 3072.0, 32768.0, 15, True, (), 1/3),
  LUT("SOFTPLUS13", lambda x: -3*math.log1p(math.exp(-abs(x/3))), 8.0, 2048.0, 8192.0, 13, True),
  LUT("SINH", math.sinh, 2.0, 8192.0, 8192.0, 13, True),
  LUT("SINH_LOCAL", lambda x: 4*math.sinh(x), .25, 65504.0, 32768.0, 15, True),
  LUT("COSH", math.cosh, 2.0, 8192.0, 8192.0, 13),
  LUT("SQRT", lambda x: math.sqrt(max(0, x)), 4.0, 4090.0, 16384.0, 14, True),
  LUT("RSQRT", lambda x: max(.5, min(4.0, 1/math.sqrt(x) if x > 0 else 4.0)), 4.0, 4090.0, 8192.0, 13),
  LUT("EXP", lambda x: math.exp(x)/(8 if x >= 0 else 1), 2.0, 8192.0, 32768.0, 15),
  LUT("EXP_LOCAL", math.exp, .25, 65504.0, 16384.0, 14),
  LUT("CELU2", lambda x: 2*math.expm1(x/2) if x < 0 else 0, 4.0, 4096.0, 16384.0, 14, True),
  LUT("CELU2_LOCAL", lambda x: 2*math.expm1(x/2) if x < 0 else 0, .5, 32768.0, 32768.0, 15, True),
  LUT("CELU3", lambda x: 3*math.expm1(x/3) if x < 0 else 0, 4.0, 4096.0, 8192.0, 13, True),
  LUT("CELU3_LOCAL", lambda x: 3*math.expm1(x/3) if x < 0 else 0, .5, 32768.0, 32768.0, 15, True),
  LUT("CELU4", lambda x: 4*math.expm1(x/4) if x < 0 else 0, 4.0, 4096.0, 8192.0, 13, True),
  LUT("CELU4_LOCAL", lambda x: 4*math.expm1(x/4) if x < 0 else 0, .5, 32768.0, 32768.0, 15, True),
  LUT("LOG2", lambda x: max(-2.0, min(2.0, math.log2(x) if x > 0 else -2.0)), 4.0, 4096.0, 8192.0, 13, True),
  LUT("LOG2_LOCAL", lambda z: 4*math.log2(1+z/12.5), 2.0, 8192.0, 32768.0, 15, True),
  # Rejected broad LOG +16 corrections at LO 295/296/311/312: fixed normalized low inputs but regressed direct positive inputs.
  LUT("LOG", lambda x: max(-math.log(4), min(math.log(4), math.log(x) if x > 0 else -math.log(4))),
      4.0, 4096.0, 16384.0, 14, True),
  LUT("LOG_LOCAL", lambda z: 4*math.log(1+z/12.5), 2.0, 8192.0, 32768.0, 15, True, ((0,77,8),(0,78,8))),
  LUT("LOG10", lambda x: max(-math.log10(4), min(math.log10(4), math.log10(x) if x > 0 else -math.log10(4))),
      4.0, 4096.0, 32768.0, 15, True),
  LUT("LOG10_LOCAL", lambda z: 4*math.log10(1+z/12.5), 2.0, 8192.0, 32768.0, 15, True),
  LUT("ROUNDOFF", lambda x: x, 1.0, 1.0, 1.0, 0),
  LUT("ASIN", lambda x: .5*math.asin(min(1.0, max(0.0, x))), 1.0, 16384.0, 32768.0, 15, True),
  LUT("ASIN_DETAIL", lambda x: 4*math.asin(min(1.0, abs(x))) if x < 0 else .5*math.asin(max(-1.0, 1-x)),
      .25, 65504.0, 32768.0, 15, True),
  LUT("ACOS", lambda x: .25*math.acos(max(-1.0, x)) if x < 0 else .5*math.acos(min(1.0, x)),
      1.0, 16384.0, 32768.0, 15, True, ((0,512,-12868),)),
  LUT("ACOS_ENDPOINT", lambda x: math.acos(max(-1.0, 1-x)) if x >= 0 else 0.0,
      .25, 65504.0, 32768.0, 15, True),
  LUT("ACOS_FINE_ENDPOINT", lambda x: 8*math.acos(max(-1.0, 1-x/64)) if x >= 0 else 0.0,
      .25, 65504.0, 32768.0, 15, True),
  LUT("ATAN", lambda x: math.atan(max(0.0, x)), 1.0, 16384.0, 32768.0, 15, True),
  LUT("ATAN_DETAIL", lambda x: 4*math.atan(abs(x)/4) if x < 0 else .5*(math.pi/2 if x == 0 else math.atan(1/x)),
      1.0, 16384.0, 32768.0, 15, True, ((0,512,-25735),)),
  LUT("SIN", math.sin, math.pi, 16384/math.pi, 32768.0, 15, True),
  LUT("SIN_LOCAL", lambda z: 8*math.sin(z/16), 2.0, 8192.0, 32768.0, 15, True),
  LUT("COS", math.cos, math.pi, 16384/math.pi, 32768.0, 15, True),
  LUT("COS_LOCAL", lambda x: 2*math.cos(x), 2.0, 8192.0, 32768.0, 15, True),
  LUT("ATANH", lambda x: .25*math.atanh(min(.99951171875, max(0.0, x))), 1.0, 16384.0, 32768.0, 15, True),
  LUT("ATANH_DETAIL", lambda x: 4*math.atanh(min(.99951171875, abs(x))) if x < 0 else
      .125*math.atanh(1-max(.00048828125, x)), .25, 65504.0, 32768.0, 15, True,
      ((0,512,1-round(.125*math.atanh(1-.00048828125)*32768)),)),
  LUT("ASINH_CORE", lambda z: 4*math.asinh(abs(z)/8) if z < 0 else .5*math.asinh(z),
      2.0, 8192.0, 32768.0, 15, True),
  LUT("ASINH_RANGE", lambda z: .25*math.asinh(2+abs(z)) if z < 0 else .125*math.asinh(19*z),
      16.0, 1024.0, 32768.0, 15, True, ((0,512,round(.25*math.asinh(2)*32768)-1),)),
)

def compile_lut(spec:LUT) -> tuple[list[int], str]:
  if spec.name == "ROUNDOFF":
    values = [0 if i % 2 == 0 else 1 << 14 for i in range(SIZE)] * 2
    return values, hashlib.sha256(struct.pack(f"<{len(values)}h", *values)).hexdigest()
  step = 32.0/spec.index_scale
  values = []
  for table in range(2):
    for i in range(SIZE):
      x = (-(512-i) if table == 0 else i)*step
      raw = max(-32768, min(32767, round(spec.function(x)*spec.output_scale)))
      values.append(1 if spec.replace_zero and raw == 0 else raw)
  for table, index, correction in spec.corrections: values[table*SIZE+index] += correction
  payload = struct.pack(f"<{len(values)}h", *values)
  return values, hashlib.sha256(payload).hexdigest()

parts = ["# generated by extra/rockchip/gen_lut.py; do not edit", "from enum import IntEnum",
         "class RKLUT(IntEnum):\n  EXP2 = 1\n  HARDSWISH = 2\n  HARDSWISH_LOCAL = 3\n  TANH = 4\n  TANH_LOCAL = 5\n"
         "  SIGMOID = 6\n  SIGMOID_LOCAL = 7\n  QUICK_GELU = 8\n  QUICK_GELU_LOCAL = 9\n"
         "  GELU_TANH = 10\n  GELU_TANH_LOCAL = 11\n  GELU_EXACT = 12\n  GELU_EXACT_LOCAL = 13\n"
         "  ERF = 14\n  ERF_LOCAL = 15\n  ELU1 = 16\n  ELU1_LOCAL = 17\n  ELU01 = 18\n  ELU01_LOCAL = 19\n"
         "  SELU = 20\n  SELU_LOCAL = 21\n  MISH = 22\n  MISH_LOCAL = 23\n"
         "  LOGSIGMOID = 24\n  LOGSIGMOID_TAIL = 25\n  SOFTPLUS1 = 26\n  SOFTPLUS1_TAIL = 27\n"
         "  SOFTPLUS3 = 28\n  SOFTPLUS3_TAIL = 29\n  SOFTPLUS13 = 30\n  SINH = 31\n  SINH_LOCAL = 32\n  COSH = 33\n"
         "  SQRT = 34\n  RSQRT = 35\n  EXP = 36\n  EXP_LOCAL = 37\n"
         "  CELU2 = 38\n  CELU2_LOCAL = 39\n  CELU3 = 40\n  CELU3_LOCAL = 41\n  CELU4 = 42\n  CELU4_LOCAL = 43\n"
         "  LOG2 = 44\n  LOG2_LOCAL = 45\n  LOG = 46\n  LOG_LOCAL = 47\n  LOG10 = 48\n  LOG10_LOCAL = 49\n"
         "  ROUNDOFF = 50\n  ASIN = 51\n  ASIN_DETAIL = 52\n  ACOS = 53\n  ACOS_ENDPOINT = 54\n  ACOS_FINE_ENDPOINT = 55\n"
         "  ATAN = 56\n  ATAN_DETAIL = 57\n  SIN = 58\n  SIN_LOCAL = 59\n  COS = 60\n  COS_LOCAL = 61\n"
         "  ATANH = 62\n  ATANH_DETAIL = 63\n  ASINH_CORE = 64\n  ASINH_RANGE = 65", "RK_LUT_SCHEMA = 25"]
for spec in LUTS:
  values, sha = compile_lut(spec)
  prefix = f"RK_LUT_{spec.name}"
  rows = "\n".join("  " + ", ".join(map(str, values[i:i+16])) + "," for i in range(0, len(values), 16))
  parts += [f'{prefix}_SHA256 = "{sha}"', f"{prefix}_DOMAIN = (-{spec.domain!r}, {spec.domain!r})",
            f"{prefix}_ENTRIES = {SIZE}", f"{prefix}_BN_MUL = {struct.unpack('<H', struct.pack('<e', spec.index_scale))[0]}",
            f"{prefix}_MINUS_EXP = {spec.minus_exp}", f"{prefix}_POST_SCALE = {spec.post_scale!r}", f"{prefix} = (\n{rows}\n)"]

# Preserve the exhaustive EXP2 simulator contract used by the compiler tests.
exp_values, _ = compile_lut(LUTS[0])
errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 2: continue
  position, base = ((x+2)*256, 0) if x < 0 else (x*256, SIZE)
  index = min(511, max(0, math.floor(position)))
  got = half(((1-(position-index))*exp_values[base+index] + (position-index)*exp_values[base+index+1]) / 8192)
  errors.append((abs(got-math.exp2(x)), abs(got-math.exp2(x))/math.exp2(x)))
parts[8:8] = [f"RK_LUT_EXP2_VERIFIED_INPUTS = {len(errors)}", f"RK_LUT_EXP2_SIM_MAX_ABS_ERROR = {max(x[0] for x in errors)!r}",
              f"RK_LUT_EXP2_SIM_MAX_REL_ERROR = {max(x[1] for x in errors)!r}"]
pathlib.Path(__file__).parents[2].joinpath("tinygrad/runtime/autogen/rockchip_lut.py").write_text("\n".join(parts)+"\n")
