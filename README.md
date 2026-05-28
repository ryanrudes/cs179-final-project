# CS179 — DROID retargeting

Download robot proprioception from RLDS (DROID), build a directional reach envelope, and retarget demonstrations onto a **UR3e** (default). DROID was recorded on Franka Panda arms; Panda is used only for elbow-side hints during retargeting.

Python **3.12+**, dependencies and CLI via [uv](https://docs.astral.sh/uv/).

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- CMake **3.18+** and a **C++17** compiler
- Optional: CUDA `nvcc` (only for the sample `vector_add` kernel, not required for download/retarget)

## Install

```bash
git clone <repo-url>
cd cs179_final_project
uv sync --dev
```

Build the native extension (recommended — faster reach envelope and retarget):

```bash
./scripts/build_loader.sh
```

This produces `src/cs179/_native*.so` (RLDS loader, reach envelope, NLopt retarget). No GPU required.

If you change C++ or bindings, rebuild with the same command.

## Quick start

**1. Download** DROID proprio to disk (default: 100-demo subset):

```bash
uv run cs179 download --dataset-url DROID_100
```

Cache layout: `data/droid_100/` with sharded `.npy` observations and `metadata/metadata.json`.

**2. Retarget** cached demos onto UR3e:

```bash
uv run cs179 retarget --dataset-url DROID_100
```

Headless batch GPU (no Meshcat during run; save joints for replay over SSH):

```bash
uv run cs179 retarget --dataset-url DROID_100 \
  --gpu --disable-visualization --no-plots --save-joints
```

Replay later (forward port 7000, open http://127.0.0.1:7000/static/ on your laptop):

```bash
ssh -L 7000:localhost:7000 -L 6000:localhost:6000 user@host
uv run cs179 retarget replay --demo 0
```

First run builds or loads a reach envelope under `data/reach_envelopes/`. Defaults: **64×128** bins, **10M** FK samples. Use smaller values for a quick try:

```bash
uv run cs179 retarget --dataset-url DROID_100 \
  --reach-n-theta 64 --reach-n-phi 128 --reach-n-samples 1000000
```

**3. Config** — weights, IK, and solver settings in [`config/default.yaml`](config/default.yaml). Override with `--config path/to.yaml`.

## CLI overview

```bash
uv run cs179 --help
```

| Command | Purpose |
|--------|---------|
| `cs179 download` | Stream RLDS episodes to local `.npy` shards |
| `cs179 retarget` | Map cached Cartesian demos onto `--robot` (default `ur3e_description`) |
| `cs179 reach-envelope build` | Precompute reach envelope NPZ only |
| `cs179 reach-envelope visualize` | Viser viewer for envelope + optional robot mesh |

Common retarget flags:

| Flag | Default | Notes |
|------|---------|--------|
| `--robot` | `ur3e_description` | Target robot (reach + IK) |
| `--panda` | `panda_description` | Source arm for elbow hints |
| `--data-dir` | `data` | Base dir; dataset name appended |
| `--start-demo` / `--end-demo` | `0` / all | Demo index range |
| `--reach-safety` | `0.9` | Scale Cartesian targets into workspace |
| `--no-native` | off | Force Python reach + SciPy retarget |
| `--reach-force-rebuild` | off | Ignore cached envelope NPZ |

## Reach envelope only

```bash
uv run cs179 reach-envelope build --robot ur3e_description
uv run cs179 reach-envelope visualize --robot ur3e_description
```

Caches: `data/reach_envelopes/{robot}_{frame}_{n_theta}x{n_phi}_{n_samples}.npz`.

## Tests

```bash
uv run pytest
```

Loader tests expect `data/droid_100` from `cs179 download`. Native retarget tests need `./scripts/build_loader.sh`.

## Project layout

```text
src/cli.py              # Typer entry point (cs179)
src/rlds/               # RLDS download + cache loader
src/retarget/           # Retargeting + stats/plots
src/reachability/       # Directional reach envelope
cpp/                    # Native reach + retarget (Pinocchio, NLopt)
config/default.yaml     # Retarget cost weights and solver
data/                   # Downloaded datasets and envelope caches (gitignored)
```

## Notes

- **Intended pipeline:** DROID (Panda) demos → **UR3e** via `--robot ur3e_description`. Retargeting onto Panda (`--robot panda_description`) is supported but not the main use case.
- After editing `src/cli.py` or `src/cli_reach_envelope.py`, reinstall if the CLI looks stale: `uv pip install -e . --force-reinstall --no-deps`.
- Examples: [`examples/README.md`](examples/README.md).
