import fcntl, os, tempfile, unittest
from unittest.mock import patch

from tinygrad.runtime.ops_rockchip import _RKNPU_LOCKS, _acquire_rknpu_lock

class TestRockchipRuntime(unittest.TestCase):
  def test_process_lock_serializes_rknpu_access(self):
    with tempfile.TemporaryDirectory() as lock_dir, patch.dict(os.environ, {"ROCKCHIP_LOCK_DIR":lock_dir}):
      lock_path = os.path.join(lock_dir,"tinygrad-rknpu.lock")
      lock_fd = _acquire_rknpu_lock()
      self.assertEqual(_acquire_rknpu_lock(),lock_fd)
      competing_fd = os.open(lock_path,os.O_RDWR)
      try:
        with self.assertRaises(BlockingIOError): fcntl.flock(competing_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.close(lock_fd)
        _RKNPU_LOCKS.pop(lock_path)
        fcntl.flock(competing_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
      finally:
        os.close(competing_fd)

if __name__ == "__main__": unittest.main()
