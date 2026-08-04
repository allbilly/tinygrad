from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RKIdentityMap:
  count: int
  def __post_init__(self):
    if self.count < 0: raise ValueError("negative RK identity extent")
  def expand(self) -> tuple[int, ...]: return tuple(range(self.count))

@dataclass(frozen=True)
class RKAffineMap:
  base: int
  stride: int
  count: int
  def __post_init__(self):
    if self.count < 0 or self.base < -1 or self.base == -1 and self.stride:
      raise ValueError("invalid RK affine access map")
  def expand(self) -> tuple[int, ...]: return tuple(self.base+self.stride*index for index in range(self.count))

@dataclass(frozen=True)
class RKPadMap:
  prefix: int
  source_start: int
  source_count: int
  suffix: int
  def __post_init__(self):
    if min(self.prefix,self.source_start,self.source_count,self.suffix) < 0:
      raise ValueError("invalid RK padding access map")
  def expand(self) -> tuple[int, ...]:
    return (-1,)*self.prefix+tuple(range(self.source_start,self.source_start+self.source_count))+(-1,)*self.suffix

@dataclass(frozen=True)
class RKPeriodicMap:
  pattern: tuple[int, ...]
  repeats: int
  def __post_init__(self):
    if not self.pattern or self.repeats < 2: raise ValueError("invalid RK periodic access map")
  def expand(self) -> tuple[int, ...]: return self.pattern*self.repeats

@dataclass(frozen=True)
class RKAffineSegment:
  base: int
  stride: int
  count: int
  def __post_init__(self):
    if self.count <= 0 or self.base < -1 or self.base == -1 and self.stride:
      raise ValueError("invalid RK piecewise-affine segment")
  def expand(self) -> tuple[int, ...]: return tuple(self.base+self.stride*index for index in range(self.count))

@dataclass(frozen=True)
class RKPiecewiseAffineMap:
  segments: tuple[RKAffineSegment, ...]
  def __post_init__(self):
    if not self.segments: raise ValueError("empty RK piecewise-affine access map")
  def expand(self) -> tuple[int, ...]: return tuple(value for segment in self.segments for value in segment.expand())

@dataclass(frozen=True)
class RKStaticSelectorMap:
  indexes: tuple[int, ...]
  def expand(self) -> tuple[int, ...]: return self.indexes

RKAccessMap = RKIdentityMap|RKAffineMap|RKPadMap|RKPeriodicMap|RKPiecewiseAffineMap|RKStaticSelectorMap

def _period(values:tuple[int, ...]) -> int:
  """Return the minimal exact period using the KMP prefix table."""
  prefix = [0]*len(values)
  for index in range(1,len(values)):
    matched = prefix[index-1]
    while matched and values[index] != values[matched]: matched = prefix[matched-1]
    if values[index] == values[matched]: matched += 1
    prefix[index] = matched
  period = len(values)-prefix[-1]
  return period if period < len(values) and len(values)%period == 0 else len(values)

def compact_access_map(indexes:tuple[int, ...]) -> RKAccessMap:
  """Retain a static output-to-input map in its smallest common structural form."""
  count = len(indexes)
  if indexes == tuple(range(count)): return RKIdentityMap(count)
  if count <= 1: return RKAffineMap(indexes[0] if indexes else 0,0,count)
  stride = indexes[1]-indexes[0]
  if (indexes[0] >= 0 or indexes[0] == -1 and stride == 0) and all(value == indexes[0]+stride*index for index,value in enumerate(indexes)):
    return RKAffineMap(indexes[0],stride,count)
  prefix = next((index for index,value in enumerate(indexes) if value != -1),count)
  suffix = next((index for index,value in enumerate(reversed(indexes)) if value != -1),count-prefix)
  middle = indexes[prefix:count-suffix]
  if middle and middle[0] >= 0 and middle == tuple(range(middle[0],middle[0]+len(middle))):
    return RKPadMap(prefix,middle[0],len(middle),suffix)
  period = _period(indexes)
  if period < count: return RKPeriodicMap(indexes[:period],count//period)
  segments:list[RKAffineSegment] = []
  start = 0
  while start < count:
    segment_stride = indexes[start+1]-indexes[start] if start+1 < count and indexes[start] >= 0 and indexes[start+1] >= 0 else 0
    end = start+1
    while end < count and indexes[end] == indexes[start]+segment_stride*(end-start) and \
          (indexes[start] >= 0 or indexes[end] == -1): end += 1
    segments.append(RKAffineSegment(indexes[start],segment_stride,end-start))
    start = end
  return RKPiecewiseAffineMap(tuple(segments)) if 3*len(segments) < count else RKStaticSelectorMap(indexes)
