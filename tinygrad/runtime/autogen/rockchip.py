# generated from RK3588 rknpu_ioctl.h and rknpu_register.h; minimal backend subset
import ctypes, functools

def _ioctl(nr, payload_type, fd, payload=None, **kwargs):
  made = payload or payload_type(**kwargs)
  ret = fd.ioctl((3 << 30) | (ctypes.sizeof(made) << 16) | (ord('d') << 8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

class struct_rknpu_mem_create(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('handle', ctypes.c_uint32), ('flags', ctypes.c_uint32), ('size', ctypes.c_uint64), ('obj_addr', ctypes.c_uint64),
              ('dma_addr', ctypes.c_uint64), ('sram_size', ctypes.c_uint64)]
class struct_rknpu_mem_map(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('handle', ctypes.c_uint32), ('reserved', ctypes.c_uint32), ('offset', ctypes.c_uint64)]
class struct_rknpu_mem_destroy(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('handle', ctypes.c_uint32), ('reserved', ctypes.c_uint32), ('obj_addr', ctypes.c_uint64)]
class struct_rknpu_task(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('flags', ctypes.c_uint32), ('op_idx', ctypes.c_uint32), ('enable_mask', ctypes.c_uint32), ('int_mask', ctypes.c_uint32),
              ('int_clear', ctypes.c_uint32), ('int_status', ctypes.c_uint32), ('regcfg_amount', ctypes.c_uint32),
              ('regcfg_offset', ctypes.c_uint32), ('regcmd_addr', ctypes.c_uint64)]
class struct_rknpu_subcore_task(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('task_start', ctypes.c_uint32), ('task_number', ctypes.c_uint32)]
class struct_rknpu_submit(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('flags', ctypes.c_uint32), ('timeout', ctypes.c_uint32), ('task_start', ctypes.c_uint32), ('task_number', ctypes.c_uint32),
              ('task_counter', ctypes.c_uint32), ('priority', ctypes.c_int32), ('task_obj_addr', ctypes.c_uint64),
              ('regcfg_obj_addr', ctypes.c_uint64), ('task_base_addr', ctypes.c_uint64), ('user_data', ctypes.c_uint64),
              ('core_mask', ctypes.c_uint32), ('fence_fd', ctypes.c_int32), ('subcore_task', struct_rknpu_subcore_task * 5)]
class struct_rknpu_action(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('flags', ctypes.c_uint32), ('value', ctypes.c_uint32)]

DRM_IOCTL_RKNPU_ACTION = functools.partial(_ioctl, 0x40, struct_rknpu_action)
DRM_IOCTL_RKNPU_SUBMIT = functools.partial(_ioctl, 0x41, struct_rknpu_submit)
DRM_IOCTL_RKNPU_MEM_CREATE = functools.partial(_ioctl, 0x42, struct_rknpu_mem_create)
DRM_IOCTL_RKNPU_MEM_MAP = functools.partial(_ioctl, 0x43, struct_rknpu_mem_map)
DRM_IOCTL_RKNPU_MEM_DESTROY = functools.partial(_ioctl, 0x44, struct_rknpu_mem_destroy)
RKNPU_MEM_NON_CACHEABLE, RKNPU_MEM_KERNEL_MAPPING = 0, 8
RKNPU_JOB_PC, RKNPU_JOB_BLOCK, RKNPU_JOB_PINGPONG, RKNPU_ACT_RESET = 1, 0, 4, 6

REG_PC_OPERATION_ENABLE = 0x00000008
REG_DPU_S_POINTER = 0x00004004
REG_DPU_FEATURE_MODE_CFG = 0x0000400c
REG_DPU_DATA_FORMAT = 0x00004010
REG_DPU_DST_BASE_ADDR = 0x00004020
REG_DPU_DST_SURF_STRIDE = 0x00004024
REG_DPU_DATA_CUBE_WIDTH = 0x00004030
REG_DPU_DATA_CUBE_HEIGHT = 0x00004034
REG_DPU_DATA_CUBE_NOTCH_ADDR = 0x00004038
REG_DPU_DATA_CUBE_CHANNEL = 0x0000403c
REG_DPU_BS_CFG = 0x00004040
REG_DPU_BS_ALU_CFG = 0x00004044
REG_DPU_BS_MUL_CFG = 0x00004048
REG_DPU_BS_OW_CFG = 0x00004050
REG_DPU_WDMA_SIZE_0 = 0x00004058
REG_DPU_WDMA_SIZE_1 = 0x0000405c
REG_DPU_BN_CFG = 0x00004060
REG_DPU_BN_ALU_CFG = 0x00004064
REG_DPU_BN_MUL_CFG = 0x00004068
REG_DPU_BN_RELUX_CMP_VALUE = 0x0000406c
REG_DPU_EW_CFG = 0x00004070
REG_DPU_EW_CVT_SCALE_VALUE = 0x00004078
REG_DPU_OUT_CVT_OFFSET = 0x00004080
REG_DPU_OUT_CVT_SCALE = 0x00004084
REG_DPU_OUT_CVT_SHIFT = 0x00004088
REG_DPU_SURFACE_ADD = 0x000040c0
REG_DPU_LUT_ACCESS_CFG, REG_DPU_LUT_ACCESS_DATA = 0x00004100, 0x00004104
REG_DPU_LUT_CFG, REG_DPU_LUT_INFO = 0x00004108, 0x0000410c
REG_DPU_LUT_LE_START, REG_DPU_LUT_LE_END = 0x00004110, 0x00004114
REG_DPU_LUT_LO_START, REG_DPU_LUT_LO_END = 0x00004118, 0x0000411c
REG_DPU_LUT_LO_SLOPE_SCALE, REG_DPU_LUT_LO_SLOPE_SHIFT = 0x00004128, 0x0000412c
REG_DPU_RDMA_RDMA_S_POINTER = 0x00005004
REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH = 0x0000500c
REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT = 0x00005010
REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL = 0x00005014
REG_DPU_RDMA_RDMA_SRC_BASE_ADDR = 0x00005018
REG_DPU_RDMA_RDMA_ERDMA_CFG = 0x00005034
REG_DPU_RDMA_RDMA_EW_BASE_ADDR = 0x00005038
REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG = 0x00005044
REG_DPU_RDMA_RDMA_WEIGHT = 0x00005068
