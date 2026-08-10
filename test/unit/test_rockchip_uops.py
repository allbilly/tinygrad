from tinygrad.dtype import dtypes
from tinygrad.renderer.rockchip import RKArg, RKBufferKind, RKLayout, RKValue, _lower_uop_program
from tinygrad.uop.ops import UOp


def _program(dtype, value, count:int=4):
  out, axis = UOp.param(0, dtype, (count,)), UOp.range(count, 0)
  return list(out.index(axis).store(value(axis)).end(axis).sink().toposort())


def test_rkvalue_is_the_typed_physical_abi():
  value = RKValue(RKArg(RKBufferKind.ARG, 0), dtypes.half, 1, RKLayout.FP16)
  assert value.dtype is dtypes.half and value.count == 1 and value.layout is RKLayout.FP16


def test_generic_fp16_uops_lower_in_dependency_order():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:lhs.index(i).load() + rhs.index(i).load() * 2.0))
  assert image is not None
  assert len(image.ew_ops) == 2
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG and image.ew_ops[-1].dst.index == 0


def test_generic_where_owns_ternary_arity():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  def select(i):
    left, right = lhs.index(i).load(), rhs.index(i).load()
    return (left < right).where(left, right)
  image = _lower_uop_program(_program(dtypes.half, select))
  assert image is not None
  assert any(op.compare for op in image.ew_ops)
  assert image.ew_ops[-1].dst.kind is RKBufferKind.ARG and image.ew_ops[-1].dst.index == 0


def test_generic_static_index_becomes_gather():
  source = UOp.param(1, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.half, lambda i:source.index(3-i).load()))
  assert image is not None and len(image.gathers) == 1
  assert image.gathers[0].base == 3 and image.gathers[0].axes == ((1, 4, -1),)


def test_generic_bool_store_has_explicit_boundary_conversion():
  lhs, rhs = UOp.param(1, dtypes.half, (4,)), UOp.param(2, dtypes.half, (4,))
  image = _lower_uop_program(_program(dtypes.bool, lambda i:lhs.index(i).load() < rhs.index(i).load()))
  assert image is not None
  assert image.ew_ops[-1].int32_output and image.ew_ops[-1].bool_output


def test_generic_int16_uses_canonical_native_layout():
  source = UOp.param(1, dtypes.int16, (4,))
  image = _lower_uop_program(_program(dtypes.int16, lambda i:source.index(i).load() + UOp.const(3, dtypes.int16)))
  assert image is not None and len(image.ew_ops) == 1
  assert image.ew_ops[0].int16_input and image.ew_ops[0].int16_output
