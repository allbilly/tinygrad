import fcntl, os, pytest, signal, threading
from pathlib import Path

def _needs_rockchip_hardware(items) -> bool:
  if os.getenv("DEV", "").split(":", 1)[0].upper() == "ROCKCHIP": return True
  return any(item.path.as_posix().endswith("test/device/test_rockchip.py") for item in items)

@pytest.fixture(scope="session", autouse=True)
def rockchip_hardware_test_lock(request):
  """Keep complete RK3588 pytest sessions from corrupting each other's device state."""
  if not _needs_rockchip_hardware(request.session.items):
    yield
    return
  lock_path = Path(os.getenv("ROCKCHIP_TEST_LOCK", "~/.cache/tinygrad/rk3588-test.lock")).expanduser()
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  with lock_path.open("a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    yield

@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
  t = threading.Timer(int(os.getenv("TEST_TIMEOUT", 300)), os.kill, args=(os.getpid(), signal.SIGABRT))
  t.start()
  try: yield
  finally:
    t.cancel()
    t.join()
