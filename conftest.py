import datetime, fcntl, json, os, platform, pytest, signal, subprocess, threading
from collections import Counter
from pathlib import Path

_rockchip_methods:dict[str, dict] = {}

def _telemetry_path() -> Path|None:
  return Path(value).expanduser() if (value:=os.getenv("ROCKCHIP_TELEMETRY")) else None

def _drain_rockchip_events() -> tuple[list[dict], list[dict]]:
  if _telemetry_path() is None: return [], []
  from tinygrad.runtime.support.rockchip_telemetry import drain
  events = drain()
  return [x for x in events if x["kind"] == "kernel"], [x for x in events if x["kind"] == "reject"]

def _coverage_outcome(failed:bool, skipped:bool, kernels:list[dict]) -> str:
  if failed: return "FAIL"
  if skipped: return "SKIP_UPSTREAM"
  if not kernels: return "PASS_FRONTEND"
  lanes = {x["lane"] for x in kernels}
  native, fallback = any(x.startswith("RK_") for x in lanes), "PYTHON" in lanes
  if native and not fallback: return "PASS_NATIVE"
  if fallback and not native: return "PASS_FALLBACK"
  return "PASS_MIXED"

def _first_reject(rejects:list[dict]) -> dict|None:
  return min(rejects, key=lambda x: x["sequence"]) if rejects else None

def _failure_kind(failed:bool, rejects:list[dict], kernels:list[dict]) -> str|None:
  if not failed: return None
  if rejects: return "NATIVE_REJECT"
  if any(x.get("outcome") == "FAIL" for x in kernels): return "DEVICE_FAILURE"
  if kernels: return "POST_EXECUTION_FAILURE"
  return "NON_DEVICE_FAILURE"

def _failure_info(report) -> dict|None:
  if not report.failed: return None
  crash = getattr(report.longrepr, "reprcrash", None)
  message = str(getattr(crash, "message", None) or report.longrepr)
  return {"class": message.partition(":")[0], "message": message}

def _failure_events(method:dict, phase_failed:bool) -> tuple[list[dict], list[dict]]:
  failed_subcases = [x for x in method["subcases"] if x["raw_outcome"] == "FAIL"]
  kernels = (method["kernels"] if phase_failed else []) + [k for subcase in failed_subcases for k in subcase["kernels"]]
  rejects = (method["rejects"] if phase_failed else []) + [r for subcase in failed_subcases for r in subcase["rejects"]]
  return kernels, rejects

def _read_identity(path:str) -> str|None:
  try: return Path(path).read_bytes().replace(b"\0", b",").decode(errors="replace").strip("\n,") or None
  except OSError: return None

def _write_rockchip_coverage(exitstatus:int) -> None:
  if (path:=_telemetry_path()) is None: return
  methods = list(_rockchip_methods.values())
  for method in methods:
    failed_subcases = [x for x in method["subcases"] if x["raw_outcome"] == "FAIL"]
    sub_failed = bool(failed_subcases)
    phase_failed = any(x == "FAIL" for x in method["phase_outcomes"].values())
    phase_skipped = any(x == "SKIP" for x in method["phase_outcomes"].values())
    kernels = method["kernels"] + [k for subcase in method["subcases"] for k in subcase["kernels"]]
    failure_kernels, failure_rejects = _failure_events(method, phase_failed)
    method["outcome"] = _coverage_outcome(sub_failed or phase_failed, phase_skipped, kernels)
    method["first_reject"] = _first_reject(failure_rejects)
    method["failure_kind"] = _failure_kind(sub_failed or phase_failed, failure_rejects, failure_kernels)
  try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
  except (OSError, subprocess.CalledProcessError): commit = None
  report = {"schema_version": 2, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "commit": commit,
    "exit_status": exitstatus, "environment": {"DEV": os.getenv("DEV"), "DEFAULT_FLOAT": os.getenv("DEFAULT_FLOAT"),
      "FORWARD_ONLY": os.getenv("FORWARD_ONLY"), "ROCKCHIP_FALLBACK": os.getenv("ROCKCHIP_FALLBACK", "0")},
    "hardware": {"hostname": platform.node(), "machine": platform.machine(), "kernel": platform.release(),
      "device_tree": _read_identity("/proc/device-tree/compatible"), "rknpu_version": _read_identity("/sys/module/rknpu/version"),
      "rknpu_srcversion": _read_identity("/sys/module/rknpu/srcversion")},
    "summary": dict(sorted(Counter(x["outcome"] for x in methods).items())), "methods": methods}
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_name(path.name+".partial")
  partial.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
  os.replace(partial, path)

def pytest_configure(config):
  if _telemetry_path() is not None:
    from tinygrad.runtime.support.rockchip_telemetry import clear
    clear()
    _rockchip_methods.clear()

def _new_method(item) -> dict:
  return {"nodeid": item.nodeid, "class": item.cls.__name__ if item.cls else None, "test": item.name,
    "outcome": None, "first_reject": None, "failure_kind": None, "phase_outcomes": {}, "phase_failures": {},
    "kernels": [], "rejects": [], "subcases": []}

def pytest_collection_modifyitems(items):
  if _telemetry_path() is not None:
    for item in items: _rockchip_methods[item.nodeid] = _new_method(item)

def pytest_runtest_setup(item):
  if _telemetry_path() is None: return
  _rockchip_methods.setdefault(item.nodeid, _new_method(item))

def pytest_runtest_logreport(report):
  if _telemetry_path() is None or report.nodeid not in _rockchip_methods: return
  method = _rockchip_methods[report.nodeid]
  kernels, rejects = _drain_rockchip_events()
  if hasattr(report, "context"):
    raw_outcome = "FAIL" if report.failed else "SKIP" if report.skipped else "PASS"
    subcase = {"index": len(method["subcases"]), "message": report.context.msg, "params": dict(report.context.kwargs),
      "raw_outcome": raw_outcome, "outcome": _coverage_outcome(report.failed, report.skipped, kernels),
      "first_reject": _first_reject(rejects), "failure_kind": _failure_kind(report.failed, rejects, kernels),
      "failure": _failure_info(report), "kernels": kernels, "rejects": rejects}
    method["subcases"].append(subcase)
  else:
    method["phase_outcomes"][report.when] = "FAIL" if report.failed else "SKIP" if report.skipped else "PASS"
    if (failure:=_failure_info(report)) is not None: method["phase_failures"][report.when] = failure
    method["kernels"].extend(kernels)
    method["rejects"].extend(rejects)

def pytest_sessionfinish(session, exitstatus):
  _write_rockchip_coverage(exitstatus)

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
