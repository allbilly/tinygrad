RK_MAX_CONSTANT_BYTES = 2*1024*1024
RK_MAX_AFFINE_VISITS = 65536
# Static identity masks may describe more logical coordinates than physical source terms. Keep their compiler-work fence separate.
RK_MAX_STATIC_MASK_VISITS = 2*RK_MAX_AFFINE_VISITS
RK_MAX_PROGRAM_STAGES = 400
RK_MAX_AFFINE_WINDOW = 192
RK_MAX_CMAC_SELECTOR_WINDOW = 1504
# A dense 64x64 transpose pack proves a 2,048-lane CMAC source window; keep ordinary affine selectors on their narrower contract.
RK_MAX_TILED_CMAC_SELECTOR_WINDOW = 2048
RK_MAX_TILED_CONTRACT_VISITS = 4*RK_MAX_AFFINE_VISITS
# Compact prefix CMAC is exact through a 1,024-lane padded source window. Bound both the hardware surface and compiler work.
RK_MAX_PREFIX_WINDOW = 1024
RK_MAX_PREFIX_VISITS = RK_MAX_PREFIX_WINDOW**2
