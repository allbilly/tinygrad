from __future__ import annotations
import itertools, os, threading
from typing import Any

_events:list[dict[str, Any]] = []
_sequence = itertools.count()
_lock = threading.Lock()

def enabled() -> bool: return bool(os.getenv("ROCKCHIP_TELEMETRY"))

def record(kind:str, **fields:Any) -> None:
  if not enabled(): return
  with _lock: _events.append({"sequence": next(_sequence), "kind": kind, **fields})

def drain() -> list[dict[str, Any]]:
  with _lock:
    events, _events[:] = list(_events), []
  return events

def clear() -> None:
  with _lock: _events.clear()
