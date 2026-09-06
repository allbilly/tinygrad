# Notes

- Always speak English only
- comments and docstrings are not counted in sz.py, no need delete them for lines cleanup

- Run tests with -n0` to prevent race condition
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Run Rockchip hardware census tests with `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF DEV=ROCKCHIP`
- Use `.venv/bin/python ~/rk3588/examples/simple_add.py` as the authoritative Rockchip NPU health gate; do not run `elementwise.py` as a health check

# Captain/review rule

- For line-cleanup or backend feature work, before implementation define the exact `sz.py` addition/deletion budget and name the replacement functions.
- Reject a generic framework or IR for a single operation. Any feature adding more than 250 executable lines, or remaining net-positive after cleanup, requires explicit user approval.
- Do not merge work that makes zero old code obsolete. Keep experiments in private `/tmp` or a persistent isolated worktree; promote only proven results to the shared tree, with small milestone commits when commits are authorized.
- Shared acceptance requires the actual production `to_program` path (never a hand-built matcher/render shortcut), no CPU/GPU numeric fallback, an exact full 445-case census, and pre/post `.venv/bin/python ~/rk3588/examples/simple_add.py` health checks.
- Comments and docstrings are excluded from `sz.py` and should be retained.
