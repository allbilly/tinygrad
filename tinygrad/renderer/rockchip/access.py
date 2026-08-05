from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Iterator
import math

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

@dataclass(frozen=True)
class RKMultiSourceAffineSegment:
  source: int
  base: int
  stride: int
  count: int
  def __post_init__(self):
    if self.source < 0 or self.base < 0 or self.count <= 0: raise ValueError("invalid RK multi-source affine segment")
  def values(self) -> Iterator[tuple[int,int]]:
    return ((self.source,self.base+self.stride*index) for index in range(self.count))

@dataclass(frozen=True)
class RKMultiSourceAccessMap:
  segments: tuple[RKMultiSourceAffineSegment, ...]
  def __post_init__(self):
    if not self.segments: raise ValueError("empty RK multi-source access map")
  @property
  def count(self) -> int: return sum(segment.count for segment in self.segments)
  def values(self) -> Iterator[tuple[int,int]]:
    return (value for segment in self.segments for value in segment.values())
  def expand(self) -> tuple[tuple[int,int], ...]: return tuple(self.values())

@dataclass(frozen=True)
class RKMultiSourceAffineGridMap:
  """A dense output grid selecting one source by one axis while preserving an affine source layout."""
  extents: tuple[int, ...]
  output_strides: tuple[int, ...]
  source_strides: tuple[int, ...]
  selector_axis: int
  selector_sources: tuple[int, ...]
  selector_bases: tuple[int, ...]
  def __post_init__(self):
    if not self.extents or len(self.extents) != len(self.output_strides) or len(self.extents) != len(self.source_strides) or \
       not 0 <= self.selector_axis < len(self.extents) or self.extents[self.selector_axis] != len(self.selector_sources) or \
       len(self.selector_sources) != len(self.selector_bases) or min(*self.extents,*self.output_strides,*self.source_strides,
       *self.selector_sources,*self.selector_bases) < 0:
      raise ValueError("invalid RK multi-source affine grid")
    stride = 1
    for axis in sorted(range(len(self.extents)),key=self.output_strides.__getitem__):
      if self.output_strides[axis] != stride: raise ValueError("RK multi-source output grid is not dense")
      stride *= self.extents[axis]
  @property
  def count(self) -> int: return math.prod(self.extents)
  def values(self) -> Iterator[tuple[int,int]]:
    for linear in range(self.count):
      coordinates = tuple(linear//stride%extent for extent,stride in zip(self.extents,self.output_strides))
      selector = coordinates[self.selector_axis]
      source_index = self.selector_bases[selector]+sum(coordinate*stride for coordinate,stride in zip(coordinates,self.source_strides))
      yield self.selector_sources[selector],source_index
  def expand(self) -> tuple[tuple[int,int], ...]: return tuple(self.values())

RKMultiSourceMap = RKMultiSourceAccessMap|RKMultiSourceAffineGridMap

def compact_multi_source_map(indexes:tuple[tuple[int,int], ...]) -> RKMultiSourceAccessMap:
  """Compress consecutive selections from one source into affine segments."""
  if not indexes: raise ValueError("empty RK multi-source mapping")
  segments:list[RKMultiSourceAffineSegment] = []
  start = 0
  while start < len(indexes):
    source,base = indexes[start]
    stride = indexes[start+1][1]-base if start+1 < len(indexes) and indexes[start+1][0] == source else 0
    end = start+1
    while end < len(indexes) and indexes[end] == (source,base+stride*(end-start)): end += 1
    segments.append(RKMultiSourceAffineSegment(source,base,stride,end-start))
    start = end
  return RKMultiSourceAccessMap(tuple(segments))

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
