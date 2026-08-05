#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from typing import Any

BEGIN, END = "<!-- BEGIN GENERATED COVERAGE -->", "<!-- END GENERATED COVERAGE -->"
METHOD_OUTCOMES = ("PASS_NATIVE", "PASS_MIXED", "PASS_FALLBACK", "PASS_FRONTEND", "SKIP_UPSTREAM", "FAIL")

def _table(headers:tuple[str, ...], rows:list[tuple[object, ...]]) -> list[str]:
  return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
          *("| " + " | ".join(map(str,row)) + " |" for row in rows)]

def _first_reject(method:dict[str, Any]) -> dict[str, Any]:
  rejects = [*method.get("rejects",()), *(reject for subcase in method.get("subcases",()) for reject in subcase.get("rejects",()))]
  return min(rejects,key=lambda reject:reject.get("sequence",float("inf")),default={})

def render_coverage(report:dict[str, Any], source_name:str) -> str:
  if report.get("schema_version") != 2: raise ValueError(f"unsupported telemetry schema {report.get('schema_version')!r}")
  methods = report.get("methods")
  if not isinstance(methods,list): raise ValueError("telemetry methods must be a list")
  method_counts = Counter(method.get("outcome","UNKNOWN") for method in methods)
  if unknown := set(method_counts)-set(METHOD_OUTCOMES): raise ValueError(f"unknown method outcomes {sorted(unknown)}")
  subcases = [subcase for method in methods for subcase in method.get("subcases",())]
  subcase_counts = Counter(subcase.get("outcome","UNKNOWN") for subcase in subcases)
  kernels = [kernel for method in methods for kernel in method.get("kernels",())]
  lane_counts = Counter(kernel.get("lane","UNKNOWN") for kernel in kernels)
  native = [kernel for kernel in kernels if str(kernel.get("lane","")).startswith("RK_")]
  quality_counts = Counter(kernel.get("native_quality","UNKNOWN") for kernel in native)
  fallback_methods = [method for method in methods if method.get("outcome") in ("PASS_MIXED","PASS_FALLBACK")]
  reject_counts = Counter(_first_reject(method).get("reject_kind","host_without_native_reject") for method in fallback_methods)
  failures = [method.get("test",method.get("nodeid","UNKNOWN")) for method in methods if method.get("outcome") == "FAIL"]
  environment, hardware = report.get("environment",{}), report.get("hardware",{})
  lines = ["## Generated current census", "",
    f"Generated from `{source_name}` at `{report.get('generated_at','unknown')}` for commit `{report.get('commit','unknown')}`.", "",
    *_table(("Method outcome","Count"), [(outcome,method_counts[outcome]) for outcome in METHOD_OUTCOMES]),
    "", f"Total methods: **{len(methods)}**. Subcases: **{len(subcases)}** " +
    "(" + ", ".join(f"{key}={value}" for key,value in sorted(subcase_counts.items())) + ").", "",
    *_table(("Kernel lane","Count"), [(lane,lane_counts[lane]) for lane in sorted(lane_counts)]), "",
    *_table(("Native quality","Count"), [(quality,quality_counts[quality]) for quality in sorted(quality_counts)]), "",
    "First recorded native rejection or routing classification among fallback-using methods:", "",
    *_table(("Reject kind","Methods"), sorted(reject_counts.items(),key=lambda item:(-item[1],item[0]))), "",
    f"Environment: `DEV={environment.get('DEV','unknown')}` `FORWARD_ONLY={environment.get('FORWARD_ONLY','unknown')}` " +
    f"`DEFAULT_FLOAT={environment.get('DEFAULT_FLOAT','unknown')}` `ROCKCHIP_FALLBACK={environment.get('ROCKCHIP_FALLBACK','unknown')}`.",
    f"Hardware: `{hardware.get('device_tree','unknown')}`, kernel `{hardware.get('kernel','unknown')}`, " +
    f"RKNPU `{hardware.get('rknpu_version','unknown')}`.", ""]
  lines += (["Failures:", "", *(f"- `{failure}`" for failure in failures), ""] if failures else ["Failures: **0**.", ""])
  return "\n".join(lines).rstrip()+"\n"

def update_marked_file(path:Path, generated:str, check:bool=False) -> None:
  current = path.read_text()
  if current.count(BEGIN) != 1 or current.count(END) != 1 or current.index(BEGIN) > current.index(END):
    raise ValueError(f"{path} must contain one ordered generated-coverage marker pair")
  wanted = current[:current.index(BEGIN)+len(BEGIN)] + "\n" + generated + current[current.index(END):]
  if check:
    if current != wanted: raise SystemExit(f"{path} is stale; regenerate it from the current telemetry JSON")
  else: path.write_text(wanted)

def main() -> None:
  parser = argparse.ArgumentParser(description="Generate the current Rockchip coverage summary from telemetry JSON")
  parser.add_argument("telemetry",type=Path)
  parser.add_argument("--output",type=Path,default=Path(__file__).with_name("coverage.md"))
  parser.add_argument("--expect-methods",type=int,default=425)
  parser.add_argument("--check",action="store_true")
  args = parser.parse_args()
  report = json.loads(args.telemetry.read_text())
  if len(report.get("methods",())) != args.expect_methods:
    raise SystemExit(f"expected {args.expect_methods} methods, found {len(report.get('methods',()))}")
  update_marked_file(args.output,render_coverage(report,args.telemetry.name),args.check)

if __name__ == "__main__": main()
