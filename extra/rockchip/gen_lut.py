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
SOFTPLUS_SCALE, SOFTPLUS_STEP = 4096.0, 32.0/4096.0
softplus_neg = [max(-32768, min(32767, round((math.log1p(math.exp(signed_sample(table, i, SOFTPLUS_STEP)))-.5)*65536))) or 1
                for table in range(2) for i in range(SIZE)]
SOFTPLUS_DIV3_NEAR_SCALE, SOFTPLUS_DIV3_NEAR_STEP = 16384.0, 32.0/16384.0
softplus_div3_near = [max(-32768, min(32767, round(math.log1p(math.exp(3*signed_sample(table, i, SOFTPLUS_DIV3_NEAR_STEP)))/3*131072))) or 1
                      for table in range(2) for i in range(SIZE)]
SOFTPLUS_DIV3_FAR_SCALE, SOFTPLUS_DIV3_FAR_STEP = 8192.0, 32.0/8192.0
softplus_div3_far = [max(-32768, min(32767, round(math.log1p(math.exp(3*signed_sample(table, i, SOFTPLUS_DIV3_FAR_STEP)))/3*1048576))) or 1
                     for table in range(2) for i in range(SIZE)]
MISH_SCALE, MISH_STEP = 4096.0, 32.0/4096.0
def mish_value(x:float) -> float: return x*math.tanh(math.log1p(math.exp(x)))
mish_lut = [max(-32768, min(32767, round(mish_value(signed_sample(table, i, MISH_STEP))*16384))) or 1
            for table in range(2) for i in range(SIZE)]
MISH_MID_SCALE, MISH_MID_STEP = 8192.0, 32.0/8192.0
mish_mid = [max(-32768, min(32767, round(mish_value(signed_sample(table, i, MISH_MID_STEP))*65536))) or 1
            for table in range(2) for i in range(SIZE)]
# WIP reference: Q18/Q19 immediate-local MISH tables were less accurate on hardware than the FP16 Horner series.
def hardswish_value(x:float) -> float: return x*min(6.0, max(0.0, x+3.0))/6.0
HARDSWISH_SCALE, HARDSWISH_STEP = 8192.0, 32.0/8192.0
hardswish = [max(-32768, min(32767, round(hardswish_value(signed_sample(table, i, HARDSWISH_STEP))*16384))) or 1
             for table in range(2) for i in range(SIZE)]
# WIP reference: a second table consuming z=16*x and returning 16*hardswish(float16(z/16)) regressed FP16 subnormals at the zero entry.
# hardswish_local = [max(-32768, min(32767, round(hardswish_value(struct.unpack("<e", struct.pack("<e",
#   signed_sample(table, i, STEP)/16))[0])*16*32768))) or 1 for table in range(2) for i in range(SIZE)]
def quick_gelu_value(x:float) -> float: return x/(1+math.exp(-1.702*x))
QUICK_GELU_SCALE, QUICK_GELU_STEP = 8192.0, 32.0/8192.0
quick_gelu = [max(-32768, min(32767, round(quick_gelu_value(signed_sample(table, i, QUICK_GELU_STEP))*16384))) or 1
              for table in range(2) for i in range(SIZE)]
# WIP reference: the 2607 emitter also added `(0,276,4)`, but that moves exact x=-0.921875 one FP16 ULP away under this emitter.
# Two- and four-count trials at index 277 likewise moved its exact knot; one count covers both exact and staged PyTorch samples.
for table, index, correction in ((0,277,1), (0,375,1), (0,408,1), (0,427,1), (1,49,1)):
  quick_gelu[table*SIZE+index] += correction
quick_gelu_local = []
for table in range(2):
  for i in range(SIZE):
    z = signed_sample(table, i, STEP)
    x = -1.5+z/4
    ideal = quick_gelu_value(x)
    xh = struct.unpack("<e", struct.pack("<e", x))[0]
    scaled = struct.unpack("<e", struct.pack("<e", xh*1.702))[0]
    sigmoid_staged = struct.unpack("<e", struct.pack("<e", 1/(1+math.exp(-scaled))))[0]
    staged = struct.unpack("<e", struct.pack("<e", xh*sigmoid_staged))[0]
    quick_gelu_local.append(max(-32768, min(32767, round((.5*ideal+.5*staged)*32768))) or 1)
def gelu_value(x:float, approximate_tanh:bool) -> float:
  return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3))) if approximate_tanh else .5*x*(1+math.erf(x/math.sqrt(2)))
GELU_SCALE, GELU_STEP = 4096.0, 32.0/4096.0
def gelu_table(approximate_tanh:bool) -> list[int]:
  return [max(-32768, min(32767, round(gelu_value(x:=signed_sample(table, i, GELU_STEP), approximate_tanh)/
    (4 if x >= 0 else 1)*32768))) or 1 for table in range(2) for i in range(SIZE)]
def gelu_local_table(approximate_tanh:bool) -> list[int]:
  return [max(-32768, min(32767, round(2*gelu_value(signed_sample(table, i, GELU_STEP)/8, approximate_tanh)*32768))) or 1
          for table in range(2) for i in range(SIZE)]
gelu_tanh, gelu_tanh_local = gelu_table(True), gelu_local_table(True)
gelu_exact, gelu_exact_local = gelu_table(False), gelu_local_table(False)
ELU_BROAD_SCALE, ELU_BROAD_STEP = 2048.0, 32.0/2048.0
ELU_LOCAL_SCALE, ELU_LOCAL_STEP = 8192.0, 32.0/8192.0
def elu_table(negative_scale:float, gain:float, local:bool) -> list[int]:
  step = ELU_LOCAL_STEP if local else ELU_BROAD_STEP
  return [max(-32768, min(32767, round(negative_scale*gain*math.expm1(
    signed_sample(table, i, step)/(4 if local else 1))*32768))) or 1 for table in range(2) for i in range(SIZE)]
ELU_VARIANTS = {"ELU1":(1.0,1.0,2.0), "ELU01":(.1,8.0,16.0), "SELU":(1.0507*1.67326,.5,1.0)}
elu_tables = {name:(elu_table(scale, broad_gain, False), elu_table(scale, local_gain, True))
              for name,(scale,broad_gain,local_gain) in ELU_VARIANTS.items()}
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
softplus_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 0: continue
  got, reference = interpolate(softplus_neg, SOFTPLUS_STEP, 65536, x)+.5, math.log1p(math.exp(x))
  softplus_errors.append((abs(got-reference), abs(got-reference)/reference))
softplus_div3_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 0: continue
  got = interpolate(softplus_div3_near, SOFTPLUS_DIV3_NEAR_STEP, 131072, x) if x >= -2.5/3 else \
        interpolate(softplus_div3_far, SOFTPLUS_DIV3_FAR_STEP, 1048576, x)
  reference = math.log1p(math.exp(3*x))/3
  softplus_div3_errors.append((abs(got-reference), abs(got-reference)/reference))
mish_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 2: continue
  if abs(x) < .125:
    polynomial = half(-.016+half(x*(-86/1875)))
    polynomial = half(.32+half(x*polynomial))
    polynomial = half(.6+half(x*polynomial))
    got = half(x*polynomial)
  else: got = interpolate(mish_mid, MISH_MID_STEP, 65536, x) if abs(x) < .5 else interpolate(mish_lut, MISH_STEP, 16384, x)
  reference = mish_value(x)
  mish_errors.append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24)))
hardswish_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 2: continue
  got = half(half(half(x*x)*(1/6))+half(x*.5)) if -.125 < x < 15/128 else interpolate(hardswish, HARDSWISH_STEP, 16384, x)
  reference = hardswish_value(x)
  hardswish_errors.append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24), abs(reference)))
quick_gelu_errors = []
for bits in range(1 << 16):
  x = struct.unpack("<e", struct.pack("<H", bits))[0]
  if not math.isfinite(x) or not -2 <= x <= 2: continue
  if -.16 < x < .16: got = half(half(x*.5)+half(half(x*x)*.4253))
  elif -2 < x < -1:
    local_input = half(half(x+1.5)*4)
    got = interpolate(quick_gelu_local, STEP, 32768, local_input)
  else: got = interpolate(quick_gelu, QUICK_GELU_STEP, 16384, x)
  reference = quick_gelu_value(x)
  quick_gelu_errors.append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24), abs(reference)))
gelu_errors = {True:[], False:[]}
for approximate_tanh in (True, False):
  broad, local = (gelu_tanh, gelu_tanh_local) if approximate_tanh else (gelu_exact, gelu_exact_local)
  for bits in range(1 << 16):
    x = struct.unpack("<e", struct.pack("<H", bits))[0]
    if not math.isfinite(x) or not -4 <= x <= 4: continue
    if abs(x) < .04: got = half(half(x*.5)+half(half(x*x)*(1/math.sqrt(2*math.pi))))
    elif abs(x) < .5: got = half(interpolate(local, GELU_STEP, 32768, half(x*8))*.5)
    else:
      got = interpolate(broad, GELU_STEP, 32768, x)
      if x >= 0: got = half(got*4)
    reference = gelu_value(x, approximate_tanh)
    gelu_errors[approximate_tanh].append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24), abs(reference)))
elu_errors = {name:[] for name in ELU_VARIANTS}
for name,(negative_scale,broad_gain,local_gain) in ELU_VARIANTS.items():
  broad, local = elu_tables[name]
  for bits in range(1 << 16):
    x = struct.unpack("<e", struct.pack("<H", bits))[0]
    if not math.isfinite(x) or not -8 <= x <= 0: continue
    if x < -.5: got = half(interpolate(broad, ELU_BROAD_STEP, 32768, x)/broad_gain)
    elif x < -.03: got = half(interpolate(local, ELU_LOCAL_STEP, 32768, half(x*4))/local_gain)
    else: got = half(half(x*negative_scale)+half(half(x*x)*(negative_scale/2)))
    reference = negative_scale*math.expm1(x)
    elu_errors[name].append((abs(got-reference), abs(got-reference)/max(abs(reference), 2**-24), abs(reference)))
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
  SOFTPLUS_NEG = 35
  SOFTPLUS_DIV3_NEAR = 36
  SOFTPLUS_DIV3_FAR = 37
  MISH = 38
  MISH_MID = 39
  HARDSWISH = 40
  QUICK_GELU = 41
  QUICK_GELU_LOCAL = 42
  GELU_TANH = 43
  GELU_TANH_LOCAL = 44
  GELU_EXACT = 45
  GELU_EXACT_LOCAL = 46
  ELU1 = 47
  ELU1_LOCAL = 48
  ELU01 = 49
  ELU01_LOCAL = 50
  SELU = 51
  SELU_LOCAL = 52
RK_LUT_SCHEMA = 43
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
RK_LUT_SOFTPLUS_NEG_SHA256 = "{digest(softplus_neg)}"
RK_LUT_SOFTPLUS_NEG_DOMAIN = (-2.0, 0.0)
RK_LUT_SOFTPLUS_NEG_ENTRIES = {SIZE}
RK_LUT_SOFTPLUS_NEG_BN_MUL = {struct.unpack('<H', struct.pack('<e', SOFTPLUS_SCALE))[0]}
RK_LUT_SOFTPLUS_NEG_MINUS_EXP = 16
RK_LUT_SOFTPLUS_NEG_VERIFIED_INPUTS = {len(softplus_errors)}
RK_LUT_SOFTPLUS_NEG_SIM_MAX_ABS_ERROR = {max(x[0] for x in softplus_errors)!r}
RK_LUT_SOFTPLUS_NEG_SIM_MAX_REL_ERROR = {max(x[1] for x in softplus_errors)!r}
RK_LUT_SOFTPLUS_NEG = (\n{rows(softplus_neg)}\n)
RK_LUT_SOFTPLUS_DIV3_NEAR_SHA256 = "{digest(softplus_div3_near)}"
RK_LUT_SOFTPLUS_DIV3_NEAR_DOMAIN = (-0.8333333333333334, 0.0)
RK_LUT_SOFTPLUS_DIV3_NEAR_ENTRIES = {SIZE}
RK_LUT_SOFTPLUS_DIV3_NEAR_BN_MUL = {struct.unpack('<H', struct.pack('<e', SOFTPLUS_DIV3_NEAR_SCALE))[0]}
RK_LUT_SOFTPLUS_DIV3_NEAR_MINUS_EXP = 17
RK_LUT_SOFTPLUS_DIV3_NEAR_VERIFIED_INPUTS = {len(softplus_div3_errors)}
RK_LUT_SOFTPLUS_DIV3_NEAR_SIM_MAX_ABS_ERROR = {max(x[0] for x in softplus_div3_errors)!r}
RK_LUT_SOFTPLUS_DIV3_NEAR_SIM_MAX_REL_ERROR = {max(x[1] for x in softplus_div3_errors)!r}
RK_LUT_SOFTPLUS_DIV3_NEAR = (\n{rows(softplus_div3_near)}\n)
RK_LUT_SOFTPLUS_DIV3_FAR_SHA256 = "{digest(softplus_div3_far)}"
RK_LUT_SOFTPLUS_DIV3_FAR_DOMAIN = (-2.0, -0.8333333333333334)
RK_LUT_SOFTPLUS_DIV3_FAR_ENTRIES = {SIZE}
RK_LUT_SOFTPLUS_DIV3_FAR_BN_MUL = {struct.unpack('<H', struct.pack('<e', SOFTPLUS_DIV3_FAR_SCALE))[0]}
RK_LUT_SOFTPLUS_DIV3_FAR_MINUS_EXP = 20
RK_LUT_SOFTPLUS_DIV3_FAR = (\n{rows(softplus_div3_far)}\n)
RK_LUT_MISH_SHA256 = "{digest(mish_lut)}"
RK_LUT_MISH_DOMAIN = (-2.0, 2.0)
RK_LUT_MISH_ENTRIES = {SIZE}
RK_LUT_MISH_BN_MUL = {struct.unpack('<H', struct.pack('<e', MISH_SCALE))[0]}
RK_LUT_MISH_MINUS_EXP = 14
RK_LUT_MISH_VERIFIED_INPUTS = {len(mish_errors)}
RK_LUT_MISH_SIM_MAX_ABS_ERROR = {max(x[0] for x in mish_errors)!r}
RK_LUT_MISH_SIM_MAX_REL_ERROR = {max(x[1] for x in mish_errors)!r}
RK_LUT_MISH = (\n{rows(mish_lut)}\n)
RK_LUT_MISH_MID_SHA256 = "{digest(mish_mid)}"
RK_LUT_MISH_MID_DOMAIN = (-0.5, 0.5)
RK_LUT_MISH_MID_ENTRIES = {SIZE}
RK_LUT_MISH_MID_BN_MUL = {struct.unpack('<H', struct.pack('<e', MISH_MID_SCALE))[0]}
RK_LUT_MISH_MID_MINUS_EXP = 16
RK_LUT_MISH_MID = (\n{rows(mish_mid)}\n)
RK_LUT_HARDSWISH_SHA256 = "{digest(hardswish)}"
RK_LUT_HARDSWISH_DOMAIN = (-2.0, 2.0)
RK_LUT_HARDSWISH_ENTRIES = {SIZE}
RK_LUT_HARDSWISH_BN_MUL = {struct.unpack('<H', struct.pack('<e', HARDSWISH_SCALE))[0]}
RK_LUT_HARDSWISH_MINUS_EXP = 14
RK_LUT_HARDSWISH_VERIFIED_INPUTS = {len(hardswish_errors)}
RK_LUT_HARDSWISH_SIM_MAX_ABS_ERROR = {max(x[0] for x in hardswish_errors)!r}
RK_LUT_HARDSWISH_SIM_MAX_REL_ERROR = {max(x[1] for x in hardswish_errors if x[2] > .01)!r}
RK_LUT_HARDSWISH = (\n{rows(hardswish)}\n)
RK_LUT_QUICK_GELU_SHA256 = "{digest(quick_gelu)}"
RK_LUT_QUICK_GELU_DOMAIN = (-2.0, 2.0)
RK_LUT_QUICK_GELU_ENTRIES = {SIZE}
RK_LUT_QUICK_GELU_BN_MUL = {struct.unpack('<H', struct.pack('<e', QUICK_GELU_SCALE))[0]}
RK_LUT_QUICK_GELU_MINUS_EXP = 14
RK_LUT_QUICK_GELU_VERIFIED_INPUTS = {len(quick_gelu_errors)}
RK_LUT_QUICK_GELU_SIM_MAX_ABS_ERROR = {max(x[0] for x in quick_gelu_errors)!r}
RK_LUT_QUICK_GELU_SIM_MAX_REL_ERROR = {max(x[1] for x in quick_gelu_errors if x[2] > .01)!r}
RK_LUT_QUICK_GELU = (\n{rows(quick_gelu)}\n)
RK_LUT_QUICK_GELU_LOCAL_SHA256 = "{digest(quick_gelu_local)}"
RK_LUT_QUICK_GELU_LOCAL_DOMAIN = (-2.0, -1.0)
RK_LUT_QUICK_GELU_LOCAL_ENTRIES = {SIZE}
RK_LUT_QUICK_GELU_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', INDEX_SCALE))[0]}
RK_LUT_QUICK_GELU_LOCAL_MINUS_EXP = 15
RK_LUT_QUICK_GELU_LOCAL = (\n{rows(quick_gelu_local)}\n)
RK_LUT_GELU_TANH_SHA256 = "{digest(gelu_tanh)}"
RK_LUT_GELU_TANH_DOMAIN = (-4.0, 4.0)
RK_LUT_GELU_TANH_ENTRIES = {SIZE}
RK_LUT_GELU_TANH_BN_MUL = {struct.unpack('<H', struct.pack('<e', GELU_SCALE))[0]}
RK_LUT_GELU_TANH_MINUS_EXP = 15
RK_LUT_GELU_TANH_VERIFIED_INPUTS = {len(gelu_errors[True])}
RK_LUT_GELU_TANH_SIM_MAX_ABS_ERROR = {max(x[0] for x in gelu_errors[True])!r}
RK_LUT_GELU_TANH_SIM_MAX_REL_ERROR = {max(x[1] for x in gelu_errors[True] if x[2] > .01)!r}
RK_LUT_GELU_TANH = (\n{rows(gelu_tanh)}\n)
RK_LUT_GELU_TANH_LOCAL_SHA256 = "{digest(gelu_tanh_local)}"
RK_LUT_GELU_TANH_LOCAL_DOMAIN = (-0.5, 0.5)
RK_LUT_GELU_TANH_LOCAL_ENTRIES = {SIZE}
RK_LUT_GELU_TANH_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', GELU_SCALE))[0]}
RK_LUT_GELU_TANH_LOCAL_MINUS_EXP = 15
RK_LUT_GELU_TANH_LOCAL = (\n{rows(gelu_tanh_local)}\n)
RK_LUT_GELU_EXACT_SHA256 = "{digest(gelu_exact)}"
RK_LUT_GELU_EXACT_DOMAIN = (-4.0, 4.0)
RK_LUT_GELU_EXACT_ENTRIES = {SIZE}
RK_LUT_GELU_EXACT_BN_MUL = {struct.unpack('<H', struct.pack('<e', GELU_SCALE))[0]}
RK_LUT_GELU_EXACT_MINUS_EXP = 15
RK_LUT_GELU_EXACT_VERIFIED_INPUTS = {len(gelu_errors[False])}
RK_LUT_GELU_EXACT_SIM_MAX_ABS_ERROR = {max(x[0] for x in gelu_errors[False])!r}
RK_LUT_GELU_EXACT_SIM_MAX_REL_ERROR = {max(x[1] for x in gelu_errors[False] if x[2] > .01)!r}
RK_LUT_GELU_EXACT = (\n{rows(gelu_exact)}\n)
RK_LUT_GELU_EXACT_LOCAL_SHA256 = "{digest(gelu_exact_local)}"
RK_LUT_GELU_EXACT_LOCAL_DOMAIN = (-0.5, 0.5)
RK_LUT_GELU_EXACT_LOCAL_ENTRIES = {SIZE}
RK_LUT_GELU_EXACT_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', GELU_SCALE))[0]}
RK_LUT_GELU_EXACT_LOCAL_MINUS_EXP = 15
RK_LUT_GELU_EXACT_LOCAL = (\n{rows(gelu_exact_local)}\n)
{''.join(f'''RK_LUT_{name}_SHA256 = "{digest(elu_tables[name][0])}"
RK_LUT_{name}_DOMAIN = (-8.0, 0.0)
RK_LUT_{name}_ENTRIES = {SIZE}
RK_LUT_{name}_BN_MUL = {struct.unpack('<H', struct.pack('<e', ELU_BROAD_SCALE))[0]}
RK_LUT_{name}_MINUS_EXP = 15
RK_LUT_{name}_VERIFIED_INPUTS = {len(elu_errors[name])}
RK_LUT_{name}_SIM_MAX_ABS_ERROR = {max(x[0] for x in elu_errors[name])!r}
RK_LUT_{name}_SIM_MAX_REL_ERROR = {max(x[1] for x in elu_errors[name] if x[2] > .01)!r}
RK_LUT_{name} = (\n{rows(elu_tables[name][0])}\n)
RK_LUT_{name}_LOCAL_SHA256 = "{digest(elu_tables[name][1])}"
RK_LUT_{name}_LOCAL_DOMAIN = (-0.5, 0.0)
RK_LUT_{name}_LOCAL_ENTRIES = {SIZE}
RK_LUT_{name}_LOCAL_BN_MUL = {struct.unpack('<H', struct.pack('<e', ELU_LOCAL_SCALE))[0]}
RK_LUT_{name}_LOCAL_MINUS_EXP = 15
RK_LUT_{name}_LOCAL = (\n{rows(elu_tables[name][1])}\n)
''' for name in ELU_VARIANTS)}
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
