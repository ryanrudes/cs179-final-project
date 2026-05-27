# Directional reach envelope

The directional reach envelope estimates how far a robot’s tool frame can reach from its base as a function of direction. Retargeting uses it to shrink Cartesian targets that would lie outside the UR3e workspace before inverse kinematics runs.

Runnable API walkthroughs: [examples/reach_envelope_python.py](examples/reach_envelope_python.py) and [examples/reach_envelope_cpp.cpp](examples/reach_envelope_cpp.cpp) (see [examples/README.md](examples/README.md)).

## What it represents

For each direction **d** (unit vector from the robot base), the envelope stores the maximum observed tool distance **r** along that direction:

```
limit(d) = max { ‖p‖ : p = FK(q), q sampled uniformly in joint limits }
```

Directions are parameterized in spherical coordinates relative to the base frame:

| Coordinate | Range | Meaning |
|------------|-------|---------|
| **θ** (theta) | `[0, π]` | Polar angle from +Z (`θ = arccos(z)`) |
| **φ** (phi) | `[-π, π]` | Azimuth in the XY plane (`φ = atan2(y, x)`) |

The discrete approximation is a 2D grid `bin_radii[θᵢ, φⱼ]` of size `n_theta × n_phi` (defaults: **64 × 128**). Each cell holds the maximum radius seen among all Monte Carlo samples that fell into that bin.

This is **not** a tight analytic workspace boundary. It is a statistical upper envelope: more FK samples improve coverage of the bin grid, but the grid resolution is fixed by `n_theta` and `n_phi`.

## Role in the pipeline

```
Joint-limited random q  →  FK(tool frame)  →  scatter-max into (θ, φ) bins
                                                      ↓
                                            DirectionalReachEnvelope
                                                      ↓
Demo Cartesian targets  →  scale_positions (radial clamp)  →  retarget IK
```

During retarget (`src/retarget/run.py`), a cached envelope for the target robot’s `tool0` frame scales each demo’s Cartesian positions via `scale_cartesian_to_robot`. A safety margin (`REACH_SAFETY_MARGIN = 0.9`) keeps targets slightly inside the estimated boundary.

Retarget loads or builds that cache via `from_robot_cached` using CLI flags (defaults match `REACH_SAMPLE_COUNT_RETARGET` and `REACH_BINS_*`):

```bash
uv run cs179 retarget \
  --robot ur3e_description \
  --reach-n-samples 10000000 \
  --reach-n-theta 64 \
  --reach-n-phi 128 \
  --reach-safety 0.9 \
  --reach-force-rebuild   # optional
```

Use the same `--reach-n-*` values as `reach-envelope build` if you pre-build the NPZ; otherwise retarget builds the cache on first run.

Pass **`--no-native`** on `retarget`, `reach-envelope build`, or `reach-envelope visualize` to force the Python Monte Carlo builder and Python `reach_limits` / `scale_positions` even when `cs179._native` is installed. In scripts: `from reachability import set_use_native_envelope; set_use_native_envelope(False)`.

Visualization (`uv run cs179 reach-envelope visualize`) draws the same envelope as a mesh in Viser; that path is separate from retarget and does not start the Viser server from the retarget command. Build-only: `uv run cs179 reach-envelope build` (writes `data/reach_envelopes/*.npz`, no Viser). `cs179 reachability` remains as a deprecated alias for `visualize`.

## Building the envelope

### Monte Carlo loop

1. Sample a random configuration `q` within `model.lowerPositionLimit` / `upperPositionLimit`.
2. Run `forwardKinematics` and `updateFramePlacements`.
3. Read the tool frame translation `p` in the base frame.
4. Compute `(θ, φ)` from `p`, map to bin indices, and update  
   `bin_radii[ti, pj] = max(bin_radii[ti, pj], ‖p‖)`.

Work proceeds in **batches** (default **50 000** samples per batch) so progress can be reported and memory stays bounded. Retarget builds use **10 000 000** samples by default (`REACH_SAMPLE_COUNT_RETARGET`).

### Empty bins

Random sampling may leave some `(θ, φ)` cells empty. Post-processing fills them:

1. **Neighborhood spread** — For each empty cell, set its value to the max of its 3×3 neighborhood (nearest-edge behavior at θ boundaries; φ uses clamped indices, not wrap, matching SciPy `maximum_filter(..., mode="nearest")`).
2. **Global fallback** — Any cell still empty receives `max(bin_radii)`.

### Disk cache

`DirectionalReachEnvelope.from_robot_cached` stores envelopes under `data/reach_envelopes/` as compressed NPZ files:

`{robot_key}_{frame_name}_{n_theta}x{n_phi}_{n_samples}.npz`

(e.g. `ur3_official_description_tool0_256x512_50000000.npz`). Rebuild with `--force-rebuild` on the CLI or `force_rebuild=True` in Python when robot, frame, bins, or sample count change. Loading a cache whose shape does not match requested `n_theta`/`n_phi` raises an error.

## Querying the envelope

After construction, the grid is treated as a function on `(θ, φ)`:

- **`reach_limits(directions)`** — Unit directions `(N, 3)` → reach `(N,)`. Directions are converted to `(θ, φ)` and **bilinearly interpolated** on cell-centered grid points  
  `θᵢ = (i + ½) π / n_theta`,  
  `φⱼ = (j + ½) 2π / n_phi − π`.  
  Out-of-range queries use the global max radius as fill value (same as SciPy `RegularGridInterpolator` with `bounds_error=False`).

- **`scale_positions(positions, pivot, safety)`** — For each point, compute offset from `pivot` (default origin), look up the directional limit, and scale the offset radially so `‖offset‖ ≤ safety × limit` (scale capped at 1).

- **`boundary_mesh`** — Python-only helper that tessellates the envelope surface for Viser (not implemented in C++).

## Implementation layout

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **C++ core** | `cpp/include/cs179/reach_envelope.hpp`, `cpp/src/reach_envelope.cpp` | Build, fill-empty, interpolate, scale |
| **pybind11** | `cpp/bindings_reachability.cpp` | Expose `cs179._native.DirectionalReachEnvelope` |
| **Python API** | `src/reachability/envelope.py` | Public class, cache I/O, Viser mesh, fallback when `_native` is absent |
| **CLI** | `src/cli_reach_envelope.py`, `src/app/reach_envelope.py` | `cs179 reach-envelope build` / `visualize` |

When `_native` is built with Pinocchio (`./scripts/build_loader.sh`) and `--no-native` was not passed, Python delegates the hot path automatically:

- `_build_bin_radii_batched` / `from_robot` → C++ `build_from_robot`
- `reach_limits` / `scale_positions` / `max_radius` → C++ when a `_native` wrapper is attached (including after `load` from NPZ)

Cache, `save`/`load`, and `boundary_mesh` remain in Python. C++ has **no NPZ loader**; use Python cache + `load`, pass `bin_radii` into `DirectionalReachEnvelope(...)`, or call `build_from_robot` in C++.

### C++ batch scaling

After you have a `cs179::DirectionalReachEnvelope`, clamp positions in the robot base frame (row-major `N×3`, optional pivot, `safety` e.g. `0.9`):

```cpp
auto [scaled, scales] = envelope.scale_positions(positions, n_points, pivot, safety);
// scaled[i*3 + {0,1,2}], scales[i] in (0, 1]
```

## Parallelization (C++ build)

The expensive step is repeated FK over millions of samples. The C++ builder parallelizes **within each batch**; batches themselves run sequentially (for progress reporting and simple merging).

```
For each batch of n_batch samples:
┌─────────────────────────────────────────────────────────────┐
│  OpenMP parallel region (num_threads = hardware / OMP_MAX)  │
│                                                             │
│  Per thread tid:                                            │
│    • own std::mt19937 RNG (seeded from batch offset + tid)  │
│    • own pinocchio::Data (FK is not thread-safe on one Data)│
│    • own thread_bins[tid] copy of the full bin grid         │
│                                                             │
│  #pragma omp for schedule(static) over samples in batch     │
│    sample q → FK → accumulate_position → thread_bins[tid]   │
└─────────────────────────────────────────────────────────────┘
         ↓
  Merge: bin_radii[i] = max(bin_radii[i], thread_bins[tid][i]) for all tid
         then zero thread_bins[tid] for the next batch
```

Design choices:

- **Thread-local bin buffers** avoid atomics on every sample (scatter-max into a shared grid would contend heavily).
- **One `pinocchio::Data` per thread** — Pinocchio mutates `Data` during FK; sharing it across threads would race.
- **`schedule(static)`** — Even sample chunks across threads when FK cost is roughly uniform.
- **Merge after each batch** — O(n_theta × n_phi × n_threads), negligible compared to FK.

If OpenMP is not available at configure time, the same loop runs **single-threaded** (still in C++ with batched structure).

The legacy **Python fallback** (`_NATIVE_ENVELOPE` missing) is single-threaded: a Python `for` loop over samples per batch, then vectorized `np.maximum.at` for bin updates.

## Python vs C++ sampling note

The C++ builder draws each `q[i]` uniformly in `[lowerPositionLimit[i], upperPositionLimit[i]]` per configuration coordinate. The Python fallback uses `pin.randomConfiguration(model)`, which respects joint types (e.g. SO(3) joints) via Pinocchio’s joint-aware sampler. For standard UR3e chains the results are similar; they are not bit-identical for the same sample count.

## Constants (defaults)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `REACH_BINS_THETA` | 64 | Polar bins |
| `REACH_BINS_PHI` | 128 | Azimuth bins |
| `REACH_BUILD_BATCH_SIZE` | 50 000 | Samples per build batch |
| `REACH_SAMPLE_COUNT_RETARGET` | 10 000 000 | FK samples when retarget builds its cached UR3e envelope |
| `REACH_SAMPLE_COUNT_VIZ` | 10 000 000 | Default `--n-samples` for `reach-envelope build` / `visualize` |
| `REACH_SAFETY_MARGIN` | 0.9 | Default safety; override with `--reach-safety` on `retarget` / `reach-envelope visualize` |
| `REACH_CACHE_DIR` | `data/reach_envelopes` | On-disk cache root |

## Building and testing

```bash
./scripts/build_loader.sh          # _native with RLDS loader + reach envelope
uv run cs179 reach-envelope build --robot ur3_official_description --n-theta 256 --n-phi 512 --n-samples 50000000
uv run cs179 reach-envelope visualize --robot ur3_official_description
uv run pytest tests/test_reach_envelope.py
```

**`reach-envelope`** — `build` (cache only) and `visualize` (Viser). Options: `--robot`, `--n-theta`, `--n-phi`, `--n-samples`, `--force-rebuild`, `--compare-robot`, `--compare-samples`. Visualize-only: `--mesh-theta` / `--mesh-phi` (default **same as** bin counts), `--fk-samples`, `--no-robot`, `--no-block`.

**`retarget`** — uses the same grid via `--reach-n-theta`, `--reach-n-phi`, `--reach-n-samples`, `--reach-force-rebuild` (prefixed to avoid clashing with dataset options).

`cs179 reachability` is a deprecated alias for `reach-envelope visualize`.

CMake option `CS179_BUILD_REACHABILITY` (default ON) links Pinocchio from the project venv’s `cmeel.prefix`. Requires the same Python interpreter for dev headers (`Python3_EXECUTABLE` from `uv`).

Tests check native interpolation against SciPy on a shared `bin_radii` grid and smoke-test a native build on UR3e.
