from tinygrad.renderer.rockchip.ir import RKContractionPlan, RKCMACTask, RKEngine, RKLayoutKind

def legalize_contraction_plan(plan:RKContractionPlan) -> tuple[RKCMACTask, ...]:
  """Legalize the currently proven direct dense contraction into one physical CMAC task."""
  plan.lhs.layout.validate_for(RKEngine.CMAC)
  plan.rhs.layout.validate_for(RKEngine.CMAC)
  plan.out.layout.validate_for(RKEngine.CMAC)
  if plan.rhs.layout.kind is not RKLayoutKind.CMAC_WEIGHT:
    raise ValueError("RK contraction RHS is not in CMAC weight layout")
  if plan.logical_m != 1 or plan.logical_k != 32 or not 4 <= plan.logical_n <= 16:
    raise ValueError("RK direct contraction is outside the proven M=1, K=32, 4<=N<=16 contract")
  if plan.lhs.layout.logical_shape != (plan.logical_m,plan.logical_k) or \
     plan.rhs.layout.logical_shape != (plan.logical_n,plan.logical_k) or \
     plan.out.layout.logical_shape != (plan.logical_m,plan.logical_n):
    raise ValueError("RK direct contraction layouts do not match logical geometry")
  return (RKCMACTask(plan.out,plan.lhs,plan.rhs,plan.reduction_axes[0],plan.constants,plan.epilogue),)
