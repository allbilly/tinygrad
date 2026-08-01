#!/usr/bin/env python3
import hashlib, math, pathlib, struct

SIZE, INDEX_SCALE, OUTPUT_SCALE, MINUS_EXP = 513, 8192.0, 8192.0, 13
STEP = 32.0 / INDEX_SCALE
exp2 = [max(-32768, min(32767, round(math.exp2((-(512-i)*STEP) if table == 0 else i*STEP) * OUTPUT_SCALE)))
        for table in range(2) for i in range(SIZE)]
def exp_value(table:int, index:int) -> int:
  x = (-(512-index)*STEP) if table == 0 else index*STEP
  return max(-32768, min(32767, round(math.exp(x) / (8 if x >= 0 else 1) * 32768)))
exp = [exp_value(table, i) for table in range(2) for i in range(SIZE)]
EXP_LOCAL_SCALE, EXP_LOCAL_STEP = 65504.0, 32.0/65504.0
exp_local = [max(-32768, min(32767, round(math.exp((-(512-i)*EXP_LOCAL_STEP) if table == 0 else i*EXP_LOCAL_STEP) * 16384)))
             for table in range(2) for i in range(SIZE)]
def expm1_value(table:int, index:int) -> int:
  x = (-(512-index)*STEP) if table == 0 else index*STEP
  return max(-32768, min(32767, round(math.expm1(x) / (8 if x >= 0 else 1) * 32768)))
expm1 = [expm1_value(table, i) for table in range(2) for i in range(SIZE)]
# WIP rejected on RK3588: a Q16 table of expm1(x)+0.5 restores the offset with enough staged error to regress ELU/SELU.
EXPM1_LOCAL_SCALE, EXPM1_LOCAL_STEP = 65504.0, 32.0/65504.0
expm1_local = [max(-32768, min(32767, round(math.expm1((-(512-i)*EXPM1_LOCAL_STEP) if table == 0 else i*EXPM1_LOCAL_STEP) /
  (2 if table else 1) * 131072))) for table in range(2) for i in range(SIZE)]
TANH_SCALE, TANH_STEP = 2048.0, 32.0/2048.0
tanh = [max(-32768, min(32767, round(math.tanh((-(512-i)*TANH_STEP) if table == 0 else i*TANH_STEP) * 32768)))
        for table in range(2) for i in range(SIZE)]
TANH_MID_SCALE, TANH_MID_STEP = 32768.0, 32.0/32768.0
tanh_mid = [max(-32768, min(32767, round(math.tanh((-(512-i)*TANH_MID_STEP) if table == 0 else i*TANH_MID_STEP) * 65536)))
            for table in range(2) for i in range(SIZE)]
TANH_LOCAL_SCALE, TANH_LOCAL_STEP = 65504.0, 32.0/65504.0
tanh_local = [max(-32768, min(32767, round(math.tanh((-(512-i)*TANH_LOCAL_STEP) if table == 0 else i*TANH_LOCAL_STEP) * 262144)))
              for table in range(2) for i in range(SIZE)]
ASIN_SCALE, ASIN_STEP = 16384.0, 32.0/16384.0
asin = [max(-32768, min(32767, round(math.asin((-(512-i)*ASIN_STEP) if table == 0 else i*ASIN_STEP) * 16384)))
        for table in range(2) for i in range(SIZE)]
ASIN_LOCAL_SCALE, ASIN_LOCAL_STEP = 65504.0, 32.0/65504.0
asin_local = [max(-32768, min(32767, round(math.asin((-(512-i)*ASIN_LOCAL_STEP) if table == 0 else i*ASIN_LOCAL_STEP) * 262144)))
              for table in range(2) for i in range(SIZE)]
asin_edge = []
for table in range(2):
  for i in range(SIZE):
    raw = max(-32768, min(32767, round(math.acos(1-abs((-(512-i)*ASIN_LOCAL_STEP) if table == 0 else i*ASIN_LOCAL_STEP)) * 32768)))
    asin_edge.append(1 if raw == 0 else raw)
acos = [max(-32768, min(32767, round(math.acos((-(512-i)*ASIN_STEP) if table == 0 else i*ASIN_STEP) * 8192)))
        for table in range(2) for i in range(SIZE)]
ATAN_SCALE, ATAN_STEP = 2048.0, 32.0/2048.0
atan = [max(-32768, min(32767, round(math.atan((-(512-i)*ATAN_STEP) if table == 0 else i*ATAN_STEP) * 16384)))
        for table in range(2) for i in range(SIZE)]
atanh = [max(-32768, min(32767, round(math.atanh(max(-.875, min(.875,
  (-(512-i)*ASIN_STEP) if table == 0 else i*ASIN_STEP))) * 8192))) for table in range(2) for i in range(SIZE)]
atanh_edge = []
for table in range(2):
  for i in range(SIZE):
    distance = abs((-(512-i)*ASIN_LOCAL_STEP) if table == 0 else i*ASIN_LOCAL_STEP)
    atanh_edge.append(max(-32768, min(32767, round((8 if distance == 0 else math.atanh(1-distance))*4096))))
HYPER_BROAD_SCALE, HYPER_BROAD_STEP = 32.0, 1.0
HYPER_MID_SCALE, HYPER_MID_STEP = 2048.0, 32.0/2048.0
ACOSH_EDGE_SCALE, ACOSH_EDGE_STEP = 65504.0, 32.0/65504.0
def signed_sample(table:int, index:int, step:float) -> float: return (-(512-index)*step) if table == 0 else index*step
asinh_broad = [max(-32768, min(32767, round(math.asinh(signed_sample(table, i, HYPER_BROAD_STEP))*4096))) or 1
                 for table in range(2) for i in range(SIZE)]
asinh_mid = [max(-32768, min(32767, round(math.asinh(signed_sample(table, i, HYPER_MID_STEP))*8192))) or 1
             for table in range(2) for i in range(SIZE)]
asinh_near = [max(-32768, min(32767, round(math.asinh(signed_sample(table, i, STEP))*16384))) or 1
              for table in range(2) for i in range(SIZE)]
acosh_broad = [max(-32768, min(32767, round(math.acosh(max(1, signed_sample(table, i, HYPER_BROAD_STEP)))*4096))) or 1
               for table in range(2) for i in range(SIZE)]
acosh_mid = [max(-32768, min(32767, round(math.acosh(1+max(0, signed_sample(table, i, HYPER_MID_STEP)))*8192))) or 1
             for table in range(2) for i in range(SIZE)]
acosh_edge = [max(-32768, min(32767, round(math.acosh(1+max(0, signed_sample(table, i, ACOSH_EDGE_STEP)-ACOSH_EDGE_STEP))*32768))) or 1
              for table in range(2) for i in range(SIZE)]
sinh_lut = [max(-32768, min(32767, round(math.sinh(signed_sample(table, i, STEP))*8192))) or 1
            for table in range(2) for i in range(SIZE)]
cosh_lut = [max(-32768, min(32767, round(math.cosh(signed_sample(table, i, STEP))*8192)))
            for table in range(2) for i in range(SIZE)]
erf_lut = [max(-32768, min(32767, round(math.erf(signed_sample(table, i, STEP))*32768))) or 1
           for table in range(2) for i in range(SIZE)]
erf_local = [max(-32768, min(32767, round(math.erf(signed_sample(table, i, ASIN_LOCAL_STEP))*65536))) or 1
             for table in range(2) for i in range(SIZE)]
def sigmoid_value(x:float) -> float: return 1/(1+math.exp(-x))
SIGMOID_SCALE, SIGMOID_STEP = 2048.0, 32.0/2048.0
sigmoid = [max(-32768, min(32767, round(sigmoid_value((-(512-i)*SIGMOID_STEP) if table == 0 else i*SIGMOID_STEP) * 32768)))
           for table in range(2) for i in range(SIZE)]
SIGMOID_LOCAL_SCALE, SIGMOID_LOCAL_STEP = 8192.0, 32.0/8192.0
sigmoid_local = [max(-32768, min(32767, round(sigmoid_value((-(512-i)*SIGMOID_LOCAL_STEP) if table == 0 else i*SIGMOID_LOCAL_STEP) * 32768)))
                 for table in range(2) for i in range(SIZE)]
SQRT_SCALE, SQRT_STEP = 4090.0, 32.0/4090.0
sqrt_lut = []
for table in range(2):
  for i in range(SIZE):
    x = (-(512-i)*SQRT_STEP) if table == 0 else i*SQRT_STEP
    raw = max(-32768, min(32767, round(math.sqrt(max(0, x))*16384)))
    sqrt_lut.append(1 if raw == 0 else raw)
rsqrt_lut = []
for table in range(2):
  for i in range(SIZE):
    x = (-(512-i)*SQRT_STEP) if table == 0 else i*SQRT_STEP
    value = max(.5, min(4.0, 1/math.sqrt(x) if x > 0 else 4.0))
    rsqrt_lut.append(max(-32768, min(32767, round(value*8192))))
LOG_SCALE, LOG_STEP = 4096.0, 32.0/4096.0
log2_lut, log2_local, log10_lut, log10_local = [], [], [], []
for table in range(2):
  for i in range(SIZE):
    x = (-(512-i)*LOG_STEP) if table == 0 else i*LOG_STEP
    broad = max(-2.0, min(2.0, math.log2(x) if x > 0 else -2.0))
    local_x = (-(512-i)*STEP) if table == 0 else i*STEP
    local = 4*math.log2(1+local_x/12.5)
    for output, value, scale in ((log2_lut, broad, 8192), (log2_local, local, 32768)):
      raw = max(-32768, min(32767, round(value*scale)))
      output.append(1 if raw == 0 else raw)
    broad10 = max(-math.log10(4), min(math.log10(4), math.log10(x) if x > 0 else -math.log10(4)))
    local10 = 4*math.log10(1+local_x/12.5)
    for output, value in ((log10_lut, broad10), (log10_local, local10)):
      raw = max(-32768, min(32767, round(value*32768)))
      output.append(1 if raw == 0 else raw)
roundoff = [0 if i % 2 == 0 else 1 << 14 for i in range(SIZE)] * 2
def digest(values:list[int]) -> str: return hashlib.sha256(struct.pack(f"<{len(values)}h", *values)).hexdigest()
def half(value:float) -> float: return struct.unpack("<e", struct.pack("<e", value))[0]
errors = []
exp_errors = []
expm1_errors = []
tanh_errors = []
asin_errors = []
acos_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 2: continue
  position, base = ((x+2)*256, 0) if x < 0 else (x*256, SIZE)
  index = min(511, max(0, math.floor(position)))
  got = half(((1-(position-index))*exp2[base+index] + (position-index)*exp2[base+index+1]) / OUTPUT_SCALE)
  reference = math.exp2(x)
  errors.append((abs(got-reference), abs(got-reference)/reference))
  exp_position, exp_base = ((x/EXP_LOCAL_STEP+512, 0) if x < 0 else (x/EXP_LOCAL_STEP, SIZE)) if abs(x) < .25 else (position, base)
  exp_index = min(511, max(0, math.floor(exp_position)))
  exp_table, exp_scale = (exp_local, 16384) if abs(x) < .25 else (exp, 32768)
  exp_got = ((1-(exp_position-exp_index))*exp_table[exp_base+exp_index] +
             (exp_position-exp_index)*exp_table[exp_base+exp_index+1]) / exp_scale
  if abs(x) >= .25 and x >= 0: exp_got *= 8
  exp_got, exp_reference = half(exp_got), math.exp(x)
  exp_errors.append((abs(exp_got-exp_reference), abs(exp_got-exp_reference)/exp_reference))
  expm1_position, expm1_base = ((x/EXPM1_LOCAL_STEP+512, 0) if x < 0 else (x/EXPM1_LOCAL_STEP, SIZE)) if abs(x) < .25 else (position, base)
  expm1_index = min(511, max(0, math.floor(expm1_position)))
  expm1_table, expm1_scale, restore = (expm1_local, 131072, 2 if x >= 0 else 1) if abs(x) < .25 else \
    (expm1, 32768, 8 if x >= 0 else 1)
  expm1_got = half(half(((1-(expm1_position-expm1_index))*expm1_table[expm1_base+expm1_index] +
    (expm1_position-expm1_index)*expm1_table[expm1_base+expm1_index+1]) / expm1_scale)*restore)
  expm1_reference = math.expm1(x)
  expm1_errors.append((abs(expm1_got-expm1_reference), abs(expm1_got-expm1_reference)/max(abs(expm1_reference), 2**-24)))
  tanh_step, tanh_table, tanh_scale = (TANH_LOCAL_STEP, tanh_local, 262144) if abs(x) < .125 else \
    (TANH_MID_STEP, tanh_mid, 65536) if abs(x) < .5 else (TANH_STEP, tanh, 32768)
  tanh_position, tanh_base = (x/tanh_step+512, 0) if x < 0 else (x/tanh_step, SIZE)
  tanh_index = min(511, max(0, math.floor(tanh_position)))
  tanh_got = half(((1-(tanh_position-tanh_index))*tanh_table[tanh_base+tanh_index] +
    (tanh_position-tanh_index)*tanh_table[tanh_base+tanh_index+1]) / tanh_scale)
  tanh_reference = math.tanh(x)
  tanh_errors.append((abs(tanh_got-tanh_reference), abs(tanh_got-tanh_reference)/max(abs(tanh_reference), 2**-24)))
  if -1 <= x <= 1:
    if abs(x) > .875:
      asin_position, asin_base, asin_table, asin_scale = (1-abs(x))/ASIN_LOCAL_STEP, SIZE, asin_edge, 32768
      asin_index = min(511, max(0, math.floor(asin_position)))
      edge_got = half(((1-(asin_position-asin_index))*asin_table[asin_base+asin_index] +
        (asin_position-asin_index)*asin_table[asin_base+asin_index+1]) / asin_scale)
      if abs(x) == 1: edge_got = 0.0
      asin_got = half(math.copysign(half(half(math.pi/2)-edge_got), x))
    else:
      asin_step, asin_table, asin_scale = (ASIN_LOCAL_STEP, asin_local, 262144) if abs(x) < .125 else (ASIN_STEP, asin, 16384)
      asin_position, asin_base = (x/asin_step+512, 0) if x < 0 else (x/asin_step, SIZE)
      asin_index = min(511, max(0, math.floor(asin_position)))
      asin_got = half(((1-(asin_position-asin_index))*asin_table[asin_base+asin_index] +
        (asin_position-asin_index)*asin_table[asin_base+asin_index+1]) / asin_scale)
    asin_reference = math.asin(x)
    asin_errors.append((abs(asin_got-asin_reference), abs(asin_got-asin_reference)/max(abs(asin_reference), 2**-24)))
    if abs(x) > .875:
      acos_got = edge_got if x >= 0 else half(half(math.pi)-edge_got)
    else:
      acos_position, acos_base = (x/ASIN_STEP+512, 0) if x < 0 else (x/ASIN_STEP, SIZE)
      acos_index = min(511, max(0, math.floor(acos_position)))
      acos_got = half(((1-(acos_position-acos_index))*acos[acos_base+acos_index] +
        (acos_position-acos_index)*acos[acos_base+acos_index+1]) / 8192)
    acos_reference = math.acos(x)
    acos_errors.append((abs(acos_got-acos_reference), abs(acos_got-acos_reference)/max(abs(acos_reference), 2**-24)))
sigmoid_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -8 <= x <= 8: continue
  local = abs(x) < 2
  step, table = (SIGMOID_LOCAL_STEP, sigmoid_local) if local else (SIGMOID_STEP, sigmoid)
  position, base = ((x/step+512, 0) if x < 0 else (x/step, SIZE))
  index = min(511, max(0, math.floor(position)))
  got = half(((1-(position-index))*table[base+index] + (position-index)*table[base+index+1]) / 32768)
  reference = sigmoid_value(x)
  sigmoid_errors.append((abs(got-reference), abs(got-reference)/reference))
atan_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -8 <= x <= 8: continue
  if abs(x) < .3:
    x2, x3 = half(x*x), half(x*half(x*x))
    got = half(half(x-half(x3/3))+half(half(x3*x2)/5))
  else:
    position, base = (x/ATAN_STEP+512, 0) if x < 0 else (x/ATAN_STEP, SIZE)
    index = min(511, max(0, math.floor(position)))
    got = half(((1-(position-index))*atan[base+index] + (position-index)*atan[base+index+1]) / 16384)
  reference = math.atan(x)
  atan_errors.append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24)))
atanh_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -1 < x < 1: continue
  if abs(x) < .3:
    x2, x3 = half(x*x), half(x*half(x*x))
    got = half(half(x+half(x3/3))+half(half(x3*x2)/5))
  elif abs(x) > .875:
    distance = half(1-abs(x))
    position, base = distance/ASIN_LOCAL_STEP, SIZE
    index = min(511, max(0, math.floor(position)))
    got = math.copysign(half(((1-(position-index))*atanh_edge[base+index] +
      (position-index)*atanh_edge[base+index+1]) / 4096), x)
  else:
    position, base = (x/ASIN_STEP+512, 0) if x < 0 else (x/ASIN_STEP, SIZE)
    index = min(511, max(0, math.floor(position)))
    got = half(((1-(position-index))*atanh[base+index] + (position-index)*atanh[base+index+1]) / 8192)
  reference = math.atanh(x)
  atanh_errors.append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24)))
def interpolate(table:list[int], step:float, scale:float, x:float) -> float:
  position, base = (x/step+512, 0) if x < 0 else (x/step, SIZE)
  index = min(511, max(0, math.floor(position)))
  return half(((1-(position-index))*table[base+index] + (position-index)*table[base+index+1]) / scale)
asinh_errors, acosh_errors = [], []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x): continue
  if -512 <= x <= 512:
    if abs(x) < .3:
      x2, x3 = half(x*x), half(x*half(x*x))
      asinh_got = half(half(x-half(x3/6))+half(half(x3*x2)*(3/40)))
    elif abs(x) < 2: asinh_got = interpolate(asinh_near, STEP, 16384, x)
    elif abs(x) < 8: asinh_got = interpolate(asinh_mid, HYPER_MID_STEP, 8192, x)
    else: asinh_got = interpolate(asinh_broad, HYPER_BROAD_STEP, 4096, x)
    reference = math.asinh(x)
    asinh_errors.append((abs(asinh_got-reference), abs(asinh_got-reference)/max(abs(reference), 2**-24)))
  if 1 <= x <= 512:
    distance = x-1
    if distance < .125: acosh_got = 0.0 if distance == 0 else interpolate(acosh_edge, ACOSH_EDGE_STEP, 32768, distance+ACOSH_EDGE_STEP)
    elif distance < 8: acosh_got = interpolate(acosh_mid, HYPER_MID_STEP, 8192, distance)
    else: acosh_got = interpolate(acosh_broad, HYPER_BROAD_STEP, 4096, x)
    reference = math.acosh(x)
    acosh_errors.append((abs(acosh_got-reference), abs(acosh_got-reference)/max(abs(reference), 2**-24)))
sinh_errors, cosh_errors = [], []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 2: continue
  if abs(x) < .3:
    x2, x3 = half(x*x), half(x*half(x*x))
    sinh_got = half(half(x+half(x3/6))+half(half(x3*x2)/120))
  else: sinh_got = interpolate(sinh_lut, STEP, 8192, x)
  cosh_got = interpolate(cosh_lut, STEP, 8192, x)
  sinh_reference, cosh_reference = math.sinh(x), math.cosh(x)
  sinh_errors.append((abs(sinh_got-sinh_reference), abs(sinh_got-sinh_reference)/max(abs(sinh_reference), 2**-24)))
  cosh_errors.append((abs(cosh_got-cosh_reference), abs(cosh_got-cosh_reference)/cosh_reference))
erf_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 2: continue
  if abs(x) < .05:
    x2, x3 = half(x*x), half(x*half(x*x))
    got = half(half(half(x-half(x3/3))+half(half(x3*x2)/10))*(2/math.sqrt(math.pi)))
  elif abs(x) < .25: got = interpolate(erf_local, ASIN_LOCAL_STEP, 65536, x)
  else: got = interpolate(erf_lut, STEP, 32768, x)
  reference = math.erf(x)
  erf_errors.append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24)))
sqrt_errors = []
rsqrt_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not 2**-8 <= x <= 4: continue
  position, base = x/SQRT_STEP, SIZE
  index = min(511, max(0, math.floor(position)))
  got = half(((1-(position-index))*sqrt_lut[base+index] + (position-index)*sqrt_lut[base+index+1]) / 16384)
  for _ in range(3): got = half(half(got+half(x/got))*.5)
  reference = math.sqrt(x)
  sqrt_errors.append((abs(got-reference), abs(got-reference)/reference))
  low_1, low_2 = float(x < .0625), float(x < .00390625)
  factor_1, factor_2 = half(1+half(low_1*15)), half(1+half(low_2*15))
  scaled = half(half(x*factor_1)*factor_2)
  position = scaled/SQRT_STEP
  index = min(511, max(0, math.floor(position)))
  seed = half(((1-(position-index))*rsqrt_lut[SIZE+index] + (position-index)*rsqrt_lut[SIZE+index+1]) / 8192)
  correction = half(1.5-half(half(min(scaled, 4.0)*half(seed*seed))*.5))
  refined = half(seed*correction)
  out_factor_1, out_factor_2 = half(1+half(low_1*3)), half(1+half(low_2*3))
  got = half(half(refined*out_factor_1)*out_factor_2)
  reference = 1/math.sqrt(x)
  rsqrt_errors.append((abs(got-reference), abs(got-reference)/reference))
def rows(values:list[int]) -> str: return "\n".join("  " + ", ".join(map(str, values[i:i+16])) + "," for i in range(0, len(values), 16))
output = f'''# generated by extra/rockchip/gen_lut.py; do not edit
from enum import IntEnum
class RKLUTId(IntEnum):
  EXP2 = 1
  ROUNDOFF = 2
  EXP = 3
  EXP_LOCAL = 4
  SIGMOID = 5
  SIGMOID_LOCAL = 6
  SQRT = 7
  RSQRT = 8
  LOG2 = 9
  LOG2_LOCAL = 10
  LOG10 = 11
  LOG10_LOCAL = 12
  EXPM1 = 13
  EXPM1_LOCAL = 14
  TANH = 15
  TANH_LOCAL = 16
  TANH_MID = 17
  ASIN = 18
  ASIN_LOCAL = 19
  ASIN_EDGE = 20
  ACOS = 21
  ATAN = 22
  ATANH = 23
  ATANH_EDGE = 24
  ASINH = 25
  ASINH_MID = 26
  ACOSH = 27
  ACOSH_MID = 28
  ACOSH_EDGE = 29
  ASINH_NEAR = 30
  SINH = 31
  COSH = 32
  ERF = 33
  ERF_LOCAL = 34
RK_LUT_SCHEMA = 29
RK_LUT_EXP2_SHA256 = "{digest(exp2)}"
RK_LUT_EXP2_DOMAIN = (-2.0, 2.0)
RK_LUT_EXP2_ENTRIES = {SIZE}
RK_LUT_EXP2_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_EXP2_MINUS_EXP = {MINUS_EXP}
RK_LUT_EXP2_VERIFIED_INPUTS = {len(errors)}
RK_LUT_EXP2_SIM_MAX_ABS_ERROR = {max(x[0] for x in errors)!r}
RK_LUT_EXP2_SIM_MAX_REL_ERROR = {max(x[1] for x in errors)!r}
RK_LUT_EXP2 = (\n{rows(exp2)}\n)
RK_LUT_ROUNDOFF_SHA256 = "{digest(roundoff)}"
RK_LUT_ROUNDOFF_DOMAIN = (-65504.0, 65504.0)
RK_LUT_ROUNDOFF_ENTRIES = {SIZE}
RK_LUT_ROUNDOFF = (\n{rows(roundoff)}\n)
RK_LUT_EXP_SHA256 = "{digest(exp)}"
RK_LUT_EXP_DOMAIN = (-2.0, 2.0)
RK_LUT_EXP_ENTRIES = {SIZE}
RK_LUT_EXP_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_EXP_MINUS_EXP = 15
RK_LUT_EXP_VERIFIED_INPUTS = {len(exp_errors)}
RK_LUT_EXP_SIM_MAX_ABS_ERROR = {max(x[0] for x in exp_errors)!r}
RK_LUT_EXP_SIM_MAX_REL_ERROR = {max(x[1] for x in exp_errors)!r}
RK_LUT_EXP = (\n{rows(exp)}\n)
RK_LUT_EXP_LOCAL_SHA256 = "{digest(exp_local)}"
RK_LUT_EXP_LOCAL_DOMAIN = (-0.25, 0.25)
RK_LUT_EXP_LOCAL_ENTRIES = {SIZE}
RK_LUT_EXP_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', EXP_LOCAL_SCALE))[0]}
RK_LUT_EXP_LOCAL_MINUS_EXP = 14
RK_LUT_EXP_LOCAL = (\n{rows(exp_local)}\n)
RK_LUT_EXPM1_SHA256 = "{digest(expm1)}"
RK_LUT_EXPM1_DOMAIN = (-2.0, 2.0)
RK_LUT_EXPM1_ENTRIES = {SIZE}
RK_LUT_EXPM1_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_EXPM1_MINUS_EXP = 15
RK_LUT_EXPM1_VERIFIED_INPUTS = {len(expm1_errors)}
RK_LUT_EXPM1_SIM_MAX_ABS_ERROR = {max(x[0] for x in expm1_errors)!r}
RK_LUT_EXPM1_SIM_MAX_REL_ERROR = {max(x[1] for x in expm1_errors)!r}
RK_LUT_EXPM1_LOCAL_SHA256 = "{digest(expm1_local)}"
RK_LUT_EXPM1_LOCAL_DOMAIN = (-0.25, 0.25)
RK_LUT_EXPM1_LOCAL_ENTRIES = {SIZE}
RK_LUT_EXPM1_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', EXPM1_LOCAL_SCALE))[0]}
RK_LUT_EXPM1_LOCAL_MINUS_EXP = 17
RK_LUT_EXPM1 = (\n{rows(expm1)}\n)
RK_LUT_EXPM1_LOCAL = (\n{rows(expm1_local)}\n)
RK_LUT_TANH_SHA256 = "{digest(tanh)}"
RK_LUT_TANH_DOMAIN = (-2.0, 2.0)
RK_LUT_TANH_ENTRIES = {SIZE}
RK_LUT_TANH_BN_MUL = {struct.unpack('<H', struct.pack('<e', TANH_SCALE))[0]}
RK_LUT_TANH_MINUS_EXP = 15
RK_LUT_TANH_VERIFIED_INPUTS = {len(tanh_errors)}
RK_LUT_TANH_SIM_MAX_ABS_ERROR = {max(x[0] for x in tanh_errors)!r}
RK_LUT_TANH_SIM_MAX_REL_ERROR = {max(x[1] for x in tanh_errors)!r}
RK_LUT_TANH_LOCAL_SHA256 = "{digest(tanh_local)}"
RK_LUT_TANH_LOCAL_DOMAIN = (-0.125, 0.125)
RK_LUT_TANH_LOCAL_ENTRIES = {SIZE}
RK_LUT_TANH_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', TANH_LOCAL_SCALE))[0]}
RK_LUT_TANH_LOCAL_MINUS_EXP = 18
RK_LUT_TANH_MID_SHA256 = "{digest(tanh_mid)}"
RK_LUT_TANH_MID_DOMAIN = (-0.5, 0.5)
RK_LUT_TANH_MID_ENTRIES = {SIZE}
RK_LUT_TANH_MID_BN_MUL = {struct.unpack('<H', struct.pack('<e', TANH_MID_SCALE))[0]}
RK_LUT_TANH_MID_MINUS_EXP = 16
RK_LUT_TANH = (\n{rows(tanh)}\n)
RK_LUT_TANH_LOCAL = (\n{rows(tanh_local)}\n)
RK_LUT_TANH_MID = (\n{rows(tanh_mid)}\n)
RK_LUT_ASIN_SHA256 = "{digest(asin)}"
RK_LUT_ASIN_DOMAIN = (-1.0, 1.0)
RK_LUT_ASIN_ENTRIES = {SIZE}
RK_LUT_ASIN_BN_MUL = {struct.unpack('<H', struct.pack('<e', ASIN_SCALE))[0]}
RK_LUT_ASIN_MINUS_EXP = 14
RK_LUT_ASIN_VERIFIED_INPUTS = {len(asin_errors)}
RK_LUT_ASIN_SIM_MAX_ABS_ERROR = {max(x[0] for x in asin_errors)!r}
RK_LUT_ASIN_SIM_MAX_REL_ERROR = {max(x[1] for x in asin_errors)!r}
RK_LUT_ASIN = (\n{rows(asin)}\n)
RK_LUT_ASIN_LOCAL_SHA256 = "{digest(asin_local)}"
RK_LUT_ASIN_LOCAL_DOMAIN = (-0.125, 0.125)
RK_LUT_ASIN_LOCAL_ENTRIES = {SIZE}
RK_LUT_ASIN_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', ASIN_LOCAL_SCALE))[0]}
RK_LUT_ASIN_LOCAL_MINUS_EXP = 18
RK_LUT_ASIN_LOCAL = (\n{rows(asin_local)}\n)
RK_LUT_ASIN_EDGE_SHA256 = "{digest(asin_edge)}"
RK_LUT_ASIN_EDGE_DOMAIN = (-0.125, 0.125)
RK_LUT_ASIN_EDGE_ENTRIES = {SIZE}
RK_LUT_ASIN_EDGE_BN_MUL = {struct.unpack('<H', struct.pack('<e', ASIN_LOCAL_SCALE))[0]}
RK_LUT_ASIN_EDGE_MINUS_EXP = 15
RK_LUT_ASIN_EDGE = (\n{rows(asin_edge)}\n)
RK_LUT_ACOS_SHA256 = "{digest(acos)}"
RK_LUT_ACOS_DOMAIN = (-1.0, 1.0)
RK_LUT_ACOS_ENTRIES = {SIZE}
RK_LUT_ACOS_BN_MUL = {struct.unpack('<H', struct.pack('<e', ASIN_SCALE))[0]}
RK_LUT_ACOS_MINUS_EXP = 13
RK_LUT_ACOS_VERIFIED_INPUTS = {len(acos_errors)}
RK_LUT_ACOS_SIM_MAX_ABS_ERROR = {max(x[0] for x in acos_errors)!r}
RK_LUT_ACOS_SIM_MAX_REL_ERROR = {max(x[1] for x in acos_errors)!r}
RK_LUT_ACOS = (\n{rows(acos)}\n)
RK_LUT_ATAN_SHA256 = "{digest(atan)}"
RK_LUT_ATAN_DOMAIN = (-8.0, 8.0)
RK_LUT_ATAN_ENTRIES = {SIZE}
RK_LUT_ATAN_BN_MUL = {struct.unpack('<H', struct.pack('<e', ATAN_SCALE))[0]}
RK_LUT_ATAN_MINUS_EXP = 14
RK_LUT_ATAN_VERIFIED_INPUTS = {len(atan_errors)}
RK_LUT_ATAN_SIM_MAX_ABS_ERROR = {max(x[0] for x in atan_errors)!r}
RK_LUT_ATAN_SIM_MAX_REL_ERROR = {max(x[1] for x in atan_errors)!r}
RK_LUT_ATAN = (\n{rows(atan)}\n)
RK_LUT_ATANH_SHA256 = "{digest(atanh)}"
RK_LUT_ATANH_DOMAIN = (-0.875, 0.875)
RK_LUT_ATANH_ENTRIES = {SIZE}
RK_LUT_ATANH_BN_MUL = {struct.unpack('<H', struct.pack('<e', ASIN_SCALE))[0]}
RK_LUT_ATANH_MINUS_EXP = 13
RK_LUT_ATANH_VERIFIED_INPUTS = {len(atanh_errors)}
RK_LUT_ATANH_SIM_MAX_ABS_ERROR = {max(x[0] for x in atanh_errors)!r}
RK_LUT_ATANH_SIM_MAX_REL_ERROR = {max(x[1] for x in atanh_errors)!r}
RK_LUT_ATANH = (\n{rows(atanh)}\n)
RK_LUT_ATANH_EDGE_SHA256 = "{digest(atanh_edge)}"
RK_LUT_ATANH_EDGE_DOMAIN = (0.0, 0.125)
RK_LUT_ATANH_EDGE_ENTRIES = {SIZE}
RK_LUT_ATANH_EDGE_BN_MUL = {struct.unpack('<H', struct.pack('<e', ASIN_LOCAL_SCALE))[0]}
RK_LUT_ATANH_EDGE_MINUS_EXP = 12
RK_LUT_ATANH_EDGE = (\n{rows(atanh_edge)}\n)
RK_LUT_ASINH_SHA256 = "{digest(asinh_broad)}"
RK_LUT_ASINH_DOMAIN = (-512.0, 512.0)
RK_LUT_ASINH_ENTRIES = {SIZE}
RK_LUT_ASINH_BN_MUL = {struct.unpack('<H', struct.pack('<e', HYPER_BROAD_SCALE))[0]}
RK_LUT_ASINH_MINUS_EXP = 12
RK_LUT_ASINH_VERIFIED_INPUTS = {len(asinh_errors)}
RK_LUT_ASINH_SIM_MAX_ABS_ERROR = {max(x[0] for x in asinh_errors)!r}
RK_LUT_ASINH_SIM_MAX_REL_ERROR = {max(x[1] for x in asinh_errors)!r}
RK_LUT_ASINH = (\n{rows(asinh_broad)}\n)
RK_LUT_ASINH_MID_SHA256 = "{digest(asinh_mid)}"
RK_LUT_ASINH_MID_DOMAIN = (-8.0, 8.0)
RK_LUT_ASINH_MID_ENTRIES = {SIZE}
RK_LUT_ASINH_MID_BN_MUL = {struct.unpack('<H', struct.pack('<e', HYPER_MID_SCALE))[0]}
RK_LUT_ASINH_MID_MINUS_EXP = 13
RK_LUT_ASINH_MID = (\n{rows(asinh_mid)}\n)
RK_LUT_ASINH_NEAR_SHA256 = "{digest(asinh_near)}"
RK_LUT_ASINH_NEAR_DOMAIN = (-2.0, 2.0)
RK_LUT_ASINH_NEAR_ENTRIES = {SIZE}
RK_LUT_ASINH_NEAR_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_ASINH_NEAR_MINUS_EXP = 14
RK_LUT_ASINH_NEAR = (\n{rows(asinh_near)}\n)
RK_LUT_ACOSH_SHA256 = "{digest(acosh_broad)}"
RK_LUT_ACOSH_DOMAIN = (1.0, 512.0)
RK_LUT_ACOSH_ENTRIES = {SIZE}
RK_LUT_ACOSH_BN_MUL = {struct.unpack('<H', struct.pack('<e', HYPER_BROAD_SCALE))[0]}
RK_LUT_ACOSH_MINUS_EXP = 12
RK_LUT_ACOSH_VERIFIED_INPUTS = {len(acosh_errors)}
RK_LUT_ACOSH_SIM_MAX_ABS_ERROR = {max(x[0] for x in acosh_errors)!r}
RK_LUT_ACOSH_SIM_MAX_REL_ERROR = {max(x[1] for x in acosh_errors)!r}
RK_LUT_ACOSH = (\n{rows(acosh_broad)}\n)
RK_LUT_ACOSH_MID_SHA256 = "{digest(acosh_mid)}"
RK_LUT_ACOSH_MID_DOMAIN = (1.0, 9.0)
RK_LUT_ACOSH_MID_ENTRIES = {SIZE}
RK_LUT_ACOSH_MID_BN_MUL = {struct.unpack('<H', struct.pack('<e', HYPER_MID_SCALE))[0]}
RK_LUT_ACOSH_MID_MINUS_EXP = 13
RK_LUT_ACOSH_MID = (\n{rows(acosh_mid)}\n)
RK_LUT_ACOSH_EDGE_SHA256 = "{digest(acosh_edge)}"
RK_LUT_ACOSH_EDGE_DOMAIN = (1.0, 1.125)
RK_LUT_ACOSH_EDGE_ENTRIES = {SIZE}
RK_LUT_ACOSH_EDGE_BN_MUL = {struct.unpack('<H', struct.pack('<e', ACOSH_EDGE_SCALE))[0]}
RK_LUT_ACOSH_EDGE_MINUS_EXP = 15
RK_LUT_ACOSH_EDGE = (\n{rows(acosh_edge)}\n)
RK_LUT_SINH_SHA256 = "{digest(sinh_lut)}"
RK_LUT_SINH_DOMAIN = (-2.0, 2.0)
RK_LUT_SINH_ENTRIES = {SIZE}
RK_LUT_SINH_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_SINH_MINUS_EXP = 13
RK_LUT_SINH_VERIFIED_INPUTS = {len(sinh_errors)}
RK_LUT_SINH_SIM_MAX_ABS_ERROR = {max(x[0] for x in sinh_errors)!r}
RK_LUT_SINH_SIM_MAX_REL_ERROR = {max(x[1] for x in sinh_errors)!r}
RK_LUT_SINH = (\n{rows(sinh_lut)}\n)
RK_LUT_COSH_SHA256 = "{digest(cosh_lut)}"
RK_LUT_COSH_DOMAIN = (-2.0, 2.0)
RK_LUT_COSH_ENTRIES = {SIZE}
RK_LUT_COSH_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_COSH_MINUS_EXP = 13
RK_LUT_COSH_VERIFIED_INPUTS = {len(cosh_errors)}
RK_LUT_COSH_SIM_MAX_ABS_ERROR = {max(x[0] for x in cosh_errors)!r}
RK_LUT_COSH_SIM_MAX_REL_ERROR = {max(x[1] for x in cosh_errors)!r}
RK_LUT_COSH = (\n{rows(cosh_lut)}\n)
RK_LUT_ERF_SHA256 = "{digest(erf_lut)}"
RK_LUT_ERF_DOMAIN = (-2.0, 2.0)
RK_LUT_ERF_ENTRIES = {SIZE}
RK_LUT_ERF_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_ERF_MINUS_EXP = 15
RK_LUT_ERF_VERIFIED_INPUTS = {len(erf_errors)}
RK_LUT_ERF_SIM_MAX_ABS_ERROR = {max(x[0] for x in erf_errors)!r}
RK_LUT_ERF_SIM_MAX_REL_ERROR = {max(x[1] for x in erf_errors)!r}
RK_LUT_ERF = (\n{rows(erf_lut)}\n)
RK_LUT_ERF_LOCAL_SHA256 = "{digest(erf_local)}"
RK_LUT_ERF_LOCAL_DOMAIN = (-0.25, 0.25)
RK_LUT_ERF_LOCAL_ENTRIES = {SIZE}
RK_LUT_ERF_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', ASIN_LOCAL_SCALE))[0]}
RK_LUT_ERF_LOCAL_MINUS_EXP = 16
RK_LUT_ERF_LOCAL = (\n{rows(erf_local)}\n)
RK_LUT_SIGMOID_SHA256 = "{digest(sigmoid)}"
RK_LUT_SIGMOID_DOMAIN = (-8.0, 8.0)
RK_LUT_SIGMOID_ENTRIES = {SIZE}
RK_LUT_SIGMOID_BN_MUL = {struct.unpack('<H', struct.pack('<e', SIGMOID_SCALE))[0]}
RK_LUT_SIGMOID_MINUS_EXP = 15
RK_LUT_SIGMOID_VERIFIED_INPUTS = {len(sigmoid_errors)}
RK_LUT_SIGMOID_SIM_MAX_ABS_ERROR = {max(x[0] for x in sigmoid_errors)!r}
RK_LUT_SIGMOID_SIM_MAX_REL_ERROR = {max(x[1] for x in sigmoid_errors)!r}
RK_LUT_SIGMOID = (\n{rows(sigmoid)}\n)
RK_LUT_SIGMOID_LOCAL_SHA256 = "{digest(sigmoid_local)}"
RK_LUT_SIGMOID_LOCAL_DOMAIN = (-2.0, 2.0)
RK_LUT_SIGMOID_LOCAL_ENTRIES = {SIZE}
RK_LUT_SIGMOID_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', SIGMOID_LOCAL_SCALE))[0]}
RK_LUT_SIGMOID_LOCAL_MINUS_EXP = 15
RK_LUT_SIGMOID_LOCAL = (\n{rows(sigmoid_local)}\n)
RK_LUT_SQRT_SHA256 = "{digest(sqrt_lut)}"
RK_LUT_SQRT_DOMAIN = (-4.0, 4.0)
RK_LUT_SQRT_ENTRIES = {SIZE}
RK_LUT_SQRT_BN_MUL = {struct.unpack('<H', struct.pack('<e', SQRT_SCALE))[0]}
RK_LUT_SQRT_MINUS_EXP = 14
RK_LUT_SQRT_VERIFIED_INPUTS = {len(sqrt_errors)}
RK_LUT_SQRT_SIM_MAX_ABS_ERROR = {max(x[0] for x in sqrt_errors)!r}
RK_LUT_SQRT_SIM_MAX_REL_ERROR = {max(x[1] for x in sqrt_errors)!r}
RK_LUT_SQRT = (\n{rows(sqrt_lut)}\n)
RK_LUT_RSQRT_SHA256 = "{digest(rsqrt_lut)}"
RK_LUT_RSQRT_DOMAIN = (-4.0, 4.0)
RK_LUT_RSQRT_ENTRIES = {SIZE}
RK_LUT_RSQRT_BN_MUL = {struct.unpack('<H', struct.pack('<e', SQRT_SCALE))[0]}
RK_LUT_RSQRT_MINUS_EXP = 13
RK_LUT_RSQRT_VERIFIED_INPUTS = {len(rsqrt_errors)}
RK_LUT_RSQRT_SIM_MAX_ABS_ERROR = {max(x[0] for x in rsqrt_errors)!r}
RK_LUT_RSQRT_SIM_MAX_REL_ERROR = {max(x[1] for x in rsqrt_errors)!r}
RK_LUT_RSQRT = (\n{rows(rsqrt_lut)}\n)
RK_LUT_LOG2_SHA256 = "{digest(log2_lut)}"
RK_LUT_LOG2_DOMAIN = (-4.0, 4.0)
RK_LUT_LOG2_ENTRIES = {SIZE}
RK_LUT_LOG2_BN_MUL = {struct.unpack('<H', struct.pack('<e', LOG_SCALE))[0]}
RK_LUT_LOG2_MINUS_EXP = 13
RK_LUT_LOG2 = (\n{rows(log2_lut)}\n)
RK_LUT_LOG2_LOCAL_SHA256 = "{digest(log2_local)}"
RK_LUT_LOG2_LOCAL_DOMAIN = (-2.0, 2.0)
RK_LUT_LOG2_LOCAL_ENTRIES = {SIZE}
RK_LUT_LOG2_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_LOG2_LOCAL_MINUS_EXP = 15
RK_LUT_LOG2_LOCAL = (\n{rows(log2_local)}\n)
RK_LUT_LOG10_SHA256 = "{digest(log10_lut)}"
RK_LUT_LOG10_DOMAIN = (-4.0, 4.0)
RK_LUT_LOG10_ENTRIES = {SIZE}
RK_LUT_LOG10_BN_MUL = {struct.unpack('<H', struct.pack('<e', LOG_SCALE))[0]}
RK_LUT_LOG10_MINUS_EXP = 15
RK_LUT_LOG10 = (\n{rows(log10_lut)}\n)
RK_LUT_LOG10_LOCAL_SHA256 = "{digest(log10_local)}"
RK_LUT_LOG10_LOCAL_DOMAIN = (-2.0, 2.0)
RK_LUT_LOG10_LOCAL_ENTRIES = {SIZE}
RK_LUT_LOG10_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_LOG10_LOCAL_MINUS_EXP = 15
RK_LUT_LOG10_LOCAL = (\n{rows(log10_local)}\n)\n'''
pathlib.Path(__file__).parents[2].joinpath("tinygrad/runtime/autogen/rockchip_lut.py").write_text(output)
