import tempfile, unittest
from pathlib import Path

from extra.rockchip.gen_coverage import BEGIN, END, render_coverage, update_marked_file

class TestRockchipCoverage(unittest.TestCase):
  def test_generated_summary_tracks_methods_subcases_kernels_and_rejects(self):
    report = {"schema_version":2,"commit":"abc123","generated_at":"now",
      "environment":{"DEV":"ROCKCHIP","FORWARD_ONLY":"1","DEFAULT_FLOAT":"HALF","ROCKCHIP_FALLBACK":"CLANG"},
      "hardware":{"device_tree":"rk3588","kernel":"test","rknpu_version":"0.9.8"},
      "methods":[
        {"test":"native","outcome":"PASS_NATIVE","subcases":[{"outcome":"PASS_NATIVE"}],"rejects":[],
         "kernels":[{"lane":"RK_DPU","native_quality":"EFFICIENT"}]},
        {"test":"mixed","outcome":"PASS_MIXED","subcases":[],"rejects":[{"reject_kind":"unsupported_layout"}],
         "kernels":[{"lane":"RK_DPU","native_quality":"CORRECTNESS_FALLBACK"},{"lane":"HOST"}]},
        {"test":"subcase_fallback","outcome":"PASS_FALLBACK","rejects":[],"kernels":[{"lane":"HOST"}],
         "subcases":[{"outcome":"PASS_FALLBACK","rejects":[{"reject_kind":"numerical_contract","sequence":3}]}]},
        {"test":"pre_renderer","outcome":"PASS_FALLBACK","rejects":[],"subcases":[],"kernels":[{"lane":"HOST"}]},
        {"test":"failed","outcome":"FAIL","subcases":[],"rejects":[],"kernels":[]}]}
    summary = render_coverage(report,"result.json")
    for expected in ("PASS_NATIVE | 1", "PASS_MIXED | 1", "PASS_FALLBACK | 2", "FAIL | 1", "RK_DPU | 2", "HOST | 3",
                     "unsupported_layout | 1", "numerical_contract | 1", "host_without_native_reject | 1", "- `failed`"):
      self.assertIn(expected,summary)

  def test_marked_file_update_and_check(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/"coverage.md"
      path.write_text(f"before\n{BEGIN}\nold\n{END}\nafter\n")
      update_marked_file(path,"new\n")
      self.assertEqual(path.read_text(),f"before\n{BEGIN}\nnew\n{END}\nafter\n")
      update_marked_file(path,"new\n",check=True)
      with self.assertRaises(SystemExit): update_marked_file(path,"different\n",check=True)

if __name__ == "__main__": unittest.main()
