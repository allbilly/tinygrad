# CMAC matrixizer design from the 515cb6b2 base

This is a host-only design and admission artifact.  `tinygrad/renderer/rockchip.py`
and `tinygrad/runtime/ops_rockchip.py` do not import it, and it never opens a
device, reads a tensor buffer, decodes FP16 data, or performs tensor arithmetic.
The only bytes it emits are the frozen donor command words used for a static
serialization oracle.

## Static donor evidence

The proposal is derived from source history, not a new hardware claim.

| donor | source fact used |
| --- | --- |
| `fc12eac61` | frozen 46-word FP16 CMAC body; relocations at command words 18, 24, and 31; SHA-256 `e1a4fb0194156e87680375eab9594f22f9ae545b4be50776819fb8e83c5e4af1` |
| `95f508096` | direct FP16 contraction shape M=1, K=32, logical N=4..16 |
| `9a6325415` | affine static index admission and explicit rejection of non-affine layouts |
| `da09c1fd9` | static affine reductions split into CMAC output tiles |
| `52f34b131` | sparse selector payload and packing through a shared pipeline |
| `f09be028b` | independent M and N tiling metadata |
| `eda240f95` | K-window tiling metadata and accumulation sequencing |
| `94d160c81` | FP32 accumulator/output-boundary representation is a separate contract |

These commits establish a reusable shape/packing vocabulary.  They do not
prove that the current EW-only image codec or current runtime can execute CMAC;
that is deliberately left as a later device milestone.

## Prototype interface

The new host-only module is
[`tinygrad/renderer/rockchip_cmac.py`](/home/orangepi/tinygrad/tinygrad/renderer/rockchip_cmac.py).
Its public interface is:

```python
plan_cmac(*, family: CMACFamily, axes: CMACAxes, output_map: StaticIndex,
          lhs_map: StaticIndex, lhs: RKArg, out: RKArg, lhs_count: int,
          rhs_map: StaticIndex|None = None, rhs: RKArg|None = None,
          rhs_count: int = 0, bias: RKArg|None = None,
          input_dtype: str = "fp16", accumulator_dtype: str = "fp32",
          output_dtype: str = "fp16", operation: str = "add",
          dynamic: bool = False, local: bool = False,
          scale: float = 1.0) -> CMACPlan|CMACFallback

validate_cmac_plan(plan: CMACPlan, lhs_count: int, rhs_count: int) -> None
emit_cmac_stage(tile: CMACTile, lhs: RKArg|None = None,
                rhs: RKArg|None = None, out: RKArg|None = None) -> RKStage
serialize_cmac_stage(stage: RKStage) -> bytes
```

`CMACAxes` partitions arbitrary static axes into `m`, `n`, `k`, and optional
`batch` groups.  `StaticIndex` is either an affine expression or a complete
static table.  The planner exhaustively enumerates every coordinate, emits
source gathers and physical packing positions, and validates dense output
coverage before returning a plan.

The six normalized non-scalar families are `ADD`, `SCALED_ADD`, `MADD`,
`SCALED_MADD`, `BIAS_MADD`, and `AFFINE_MADD`.  `LOCAL_ADD` is a separate
static-local admission and must be requested with `local=True`.  Every family
uses the same M/K/N matrixization, so later production lowering need not keep
one recognizer per graph spelling.

The donor limits are intentionally conservative: M tiles are one row, N tiles
are at most 16 logical outputs, and K tiles are at most 32 terms.  A larger K
creates ordered tiles with one reusable scratch accumulator and a barrier at
each subsequent K window.  The planner records `CMACTile.lhs/rhs` gathers,
static pack-reuse IDs, `scratch_slots`, and barrier indices; it does not fill a
buffer with tensor values.  FP16 input plus FP32 accumulation is mandatory;
FP16 output records `final_round=True`, while FP32 output keeps the accumulator
boundary.  `MAX/MIN/ARG*`, integer, boolean, dynamic, bad-axis, bad-bound, and
unsupported-precision requests return an explicit `CMACFallback` reason.

## Production interface and line budget

The eventual production route should preserve the current `RKStage` command and
relocation representation and add only a CMAC section to `RKImage`:

```python
@dataclass(frozen=True)
class RKImage:
  ...
  cmac_stages: tuple[RKStage, ...] = ()

def encode_image(image: RKImage) -> bytes       # append CMAC count/rows
def decode_image(blob: bytes) -> RKImage       # validate CMAC rows/relocs
def _run_cmac_stages(self, stages: tuple[RKStage, ...], address, buffer,
                     scratch: RKArg|None, final_round: bool) -> None
```

No shared tinygrad core change is needed.  The proposed executable-line cap is
186 additions, with each row an allocation ceiling rather than an optimistic
average:

| surface | functions/fields | cap |
| --- | --- | ---: |
| renderer | CMAC admission wrapper, matrix-to-`RKStage` conversion, route/fallback telemetry | 86 |
| image codec | `RKImage.cmac_stages`, header count, stage/reloc encode/decode and validation | 34 |
| runtime | CMAC task descriptor, DMA relocation, K-tile scratch/barrier sequencing, final FP16 round boundary | 66 |
| total |  | **186** |

The 173 executable lines in the host-only oracle are an artifact of this
milestone, not production device code.  If the route is later integrated, its
static checks are folded into the renderer cap; they must not be copied into a
second runtime interpreter.

## Deletion unlocked after a real route census

At exact base `515cb6b2c`, the six candidate route bodies account for these
executable lines (measured with the repository token-line rule):

| current route | lines |
| --- | ---: |
| `_lower_dot_loop_reduction` | 25 |
| `_lower_scalar_loop_reduction` | 40 |
| `_lower_mapped_add_loop_reduction` | 18 |
| `_lower_vectorized_unrolled_add_reduction` | 88 |
| `_lower_vectorized_mul_add_reduction` | 42 |
| `_lower_multi_scalar_local_reductions` | 23 |
| **route bodies** | **236** |

Once exhaustive route telemetry proves every admitted shape uses the matrixizer
and every rejected shape reaches the existing generic EW oracle, the route
matcher/dispatch-only helpers can be removed as well.  The available shared
helper/matcher budget is approximately 74 executable lines, making the
realistic conditional deletion **310 lines**, not 58–59.  This is conditional:
the helpers must not be deleted from the current tree until the full 445-case
hardware census and an independent image differential show that all old route
outputs are preserved.  This artifact itself makes no such hardware claim.

