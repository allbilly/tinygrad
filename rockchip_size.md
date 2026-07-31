# Rockchip backend size audit

This audit uses `. .venv/bin/activate && python sz.py`. `sz.py` counts source
lines containing Python tokens, excludes docstrings, and excludes
`tinygrad/runtime/autogen`. The numbers therefore differ slightly from
`wc -l` and should be treated as implementation-size signals, not physical
file sizes.

## Baseline and current size

| Scope | Before | Current | Change |
|---|---:|---:|---:|
| `runtime/support/rockchip.py` | 13,422 | 13,031 | -391 |
| `runtime/ops_rockchip.py` | 2,276 | 2,087 | -189 |
| Both Rockchip files | 15,698 | 15,118 | **-580** |
| Entire `tinygrad/runtime` | 22,955 | 22,375 | -580 |
| Entire repository | 40,007 | 39,427 | -580 |

The two Rockchip files are still 67.6% of counted runtime code and 38.3% of
the repository. Their physical source size moved from 18,059 to 17,479 lines,
also a reduction of 580.

Three semantics-preserving dispatch rewrites produced the current reduction:

- `fc69aeff3`: replace the repeated ordered classifier chain with ordered
  classifier tables, preserving all 133 active classifiers and their order;
- `927996958`: replace repeated typed-host tag chains with one runner table
  and path-specific allowlists, preserving which layouts each path accepts.
- `6cb4310f6`: table-drive all 74 ordinary LUT marker variants, preserving
  every tuned builder and the roundoff-specific configuration.

PR1 remains 173/173 and the static baselines are unchanged. The definitive
post-refactor inventory passes 405 methods, 13 expected skips, and 126
subtests in 2373.12 seconds with zero failures.

## What occupies the lines

The baseline token-line split for `support/rockchip.py` was:

| Category | Counted lines | Share |
|---|---:|---:|
| Typed-host and general graph lowering | 5,417 | 40.4% |
| Native and LUT task lowering | 4,985 | 37.1% |
| LUT builders | 742 | 5.5% |
| Register emitters | 678 | 5.1% |
| LUT recognition and planner | 564 | 4.2% |
| General graph recognizers | 493 | 3.7% |
| Native-program dispatch | 241 | 1.8% |
| Formats, constants, and data classes | 176 | 1.3% |
| Codec | 126 | 0.9% |

The important result is that lowering accounts for 10,402 lines, or 77.5%
of the support file. LUT builders are visible but are not the dominant source
of size.

The baseline split for `ops_rockchip.py` was:

| Category | Counted lines | Share |
|---|---:|---:|
| Program submission and runtime control | 1,003 | 44.1% |
| Typed-host runtime implementations | 807 | 35.5% |
| CMAC and conversion helpers | 394 | 17.3% |
| Renderer and device classes | 72 | 3.2% |

The largest current top-level units reinforce the same conclusion:

- `RockchipProgram`: 769 counted lines;
- `_emit_dpu_lut`: 103 after dispatch compaction (388 before);
- `_try_tensor_pow_subtasks`: 271;
- `_try_arg_extrema_subtasks`: 246;
- `_try_pool_index_subtasks`: 226;
- `_try_argsort_selected_subtasks`: 214;
- `_run_host_elementwise`: 215.

Most large blocks are distinct lowering algorithms and hardware state
handling, not whitespace or trivial helper duplication. They need focused
algorithm-level refactors and full hardware regression, not mechanical line
joining.

## LUT compiler assessment

There are 63 `_build_*_lut` functions: 42 static builders occupying 483
counted lines and 21 parameterized builders occupying 255. The parameterized
set includes runtime values such as CELU/ELU alpha, Softplus and LogSigmoid
beta, LOG function scale, generic EXP2 input scale, boolean approximation
modes, and finite power-table regions.

Moving the current builders into another ordinary `tinygrad` module would not
change `sz.py`. Moving handwritten implementation into `runtime/autogen`
would reduce the displayed number only because that directory is excluded;
that would hide complexity rather than remove it.

A real external compiler should instead use this layout:

```text
extra/rockchip/compile_luts.py          generator and tuned formulas
tinygrad/runtime/autogen/rockchip_luts.py  committed generated artifact
tinygrad/runtime/support/rockchip.py    compact lookup + dynamic builders
```

The generated artifact should contain a format version, generator revision,
SHA-256 digest, table name, two signed-int16 513-entry halves, quantized BN
multiplier, output scale, index scale, minus exponent, and any measured knot
corrections. A regeneration check must compare the committed artifact byte for
byte. The runtime should lazily decode/cache tables so normal import time does
not pay for every LUT.

Recommended split:

1. Precompile the 42 static builders and finite boolean/region variants.
2. Keep compact runtime builders for truly variable alpha, beta, and scale.
3. Keep graph recognition, two-level LUT selection, special-value restoration,
   task ordering, and `_emit_dpu_lut` in the backend; these are runtime policy,
   not table generation.
4. Commit the generated Python artifact because current package data includes
   Python modules but not arbitrary binary assets.
5. Add software checks for table length/range/metadata and hardware checks for
   every public LUT family before deleting any old builder implementation.

This can realistically remove roughly 400-500 counted runtime builder lines.
It cannot remove the remaining 103-line emitter or thousands of lines of
two-level task composition. The repeated builder-selection portion of
`_emit_dpu_lut` has already been compacted without changing a table value.

## Refactoring order

1. Compact exact dispatch duplication and prove ordered equivalence.
2. Introduce the versioned external generator for static tables.
3. Refactor repeated task-composition primitives only after capturing exact
   stage lists and buffer-slot dependencies in tests.
4. Leave register sequences and NPU reset/barrier paths until last; they are
   small relative to lowering and have the highest state-pollution risk.

Every milestone must retain `FORWARD_ONLY=1 DEFAULT_FLOAT=HALF`, pass PR1,
preserve mypy/Ruff baselines, pass `git diff --check`, and finish with the
complete forward-only Rockchip TestOps inventory.
