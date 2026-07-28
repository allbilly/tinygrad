# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README` for profiling
- NEVER EVER run git stash, git checkout 
- NEVER change / revert staged files
- backup code to /tmp with timestamp before any modification or just do not remove rockchip related WIP code or old code, just comment them for ref, wont count in sz.py anyway

# NPU
do not remove rockchip related WIP code or old code, just comment them for ref, wont count in sz.py anyway
do not remove rockchip related WIP code or old code, just comment them for ref, wont count in sz.py anyway
do not remove rockchip related WIP code or old code, just comment them for ref, wont count in sz.py anyway

use FORWARD_ONLY=1 and DEFAULT_FLOAT=HALF

u can clone any needed reference repo into ref/ with git clone --depth=1
- allbilly/npu
- allbilly/rk3588
- other branches in this repo
- nvdla/sw
- nvdla/hw
- https://gitlab.freedesktop.org/mesa/mesa
- https://github.com/mtx512/rk3588-npu
