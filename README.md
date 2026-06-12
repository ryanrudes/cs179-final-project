# CS179 — DROID retargeting

Download robot proprioception from RLDS (DROID), build a directional reach envelope, and retarget demonstrations onto a **UR3e** (default). DROID was recorded on Franka Panda arms; Panda is used only for elbow-side hints during retargeting.

Python **3.12+**, dependencies and CLI via [uv](https://docs.astral.sh/uv/).

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- CMake **3.18+** and a **C++17** compiler
- Optional: CUDA `nvcc` (required for GPU-accelerated motion retargeting)

## Install

```bash
git clone <repo-url>
cd cs179_final_project
uv sync --dev
```

Build the native extension. If you have CUDA available, build with GPU acceleration:

```bash
./scripts/build_native.sh
```

Otherwise, build the CPU/loader-only extension (no CUDA/GPU required):

```bash
./scripts/build_loader.sh
```

This produces `src/cs179/_native*.so`. If you change C++ files, CUDA kernels, or Python bindings, rebuild using the appropriate script.

## Quick start

**1. Download** DROID proprio to disk (default: 100-demo subset):

```bash
uv run cs179 download --dataset-url DROID_100
```

Cache layout: `data/droid_100/` with sharded `.npy` observations and `metadata/metadata.json`.

**Alternative:** You can skip `cs179 download` and use the pre-cached DROID subset from [Kaggle — droid-motion-data](https://www.kaggle.com/datasets/ryanrudes/droid-motion-data). Extract it under `data/droid/` so the layout matches above.

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

## Performance Benchmarks

Motion retargeting can be run on CPU or GPU. Below is an apples-to-apples throughput comparison of retargeting performance, comparing the highly optimized C++ native NLopt solver (running on all available CPU cores) to the batched CUDA implementation on an Intel Xeon E5-2640 v3 (16 threads) and an NVIDIA RTX A5000:

| Backend | Hardware Utilized | Solver Throughput (frames/sec) | Speedup |
|---------|-------------------|--------------------------------|---------|
| C++ CPU (NLopt) | 16 Cores | ~1,042 | 1x |
| GPU CUDA (Pure) | NVIDIA RTX A5000 | ~929,216 | **891x** |

When streaming the entire dataset (10,000 demos, ~2.8 million frames) through the full pipeline—which includes data loading, Cartesian scaling, and GPU optimization—the system achieves **~492,000 pipeline frames/sec** (approx. 1,734 demos/sec).

The gap between pure solver throughput (~929k frames/sec) and full-pipeline throughput (~492k frames/sec) reflects the surrounding work—loading cached `.npy` shards, Cartesian scaling, host/device staging, and Python orchestration—which becomes the dominant cost once the solver itself is this fast.

> **Note:** The GPU implementation avoids iterative Python/C++ overhead by running Jacobi projected gradient descent completely inside shared memory. To maintain maximum throughput during dataset generation, run headless with `--disable-visualization --no-plots`, which skips Meshcat playback, matplotlib plots, and the CPU forward-kinematic evaluations they require.

### Hardware-level GPU Profiling (Nsight Compute)

To profile the low-level execution characteristics of the CUDA retargeting kernel, we run a hardware-level profiling script using NVIDIA Nsight Compute (`ncu`). This profiling isolates the `retarget_trajectory_kernel` and measures hardware metrics like achieved occupancy, SM throughput, and shared memory utilization.

You can run this profiling script using:
```bash
uv run scripts/benchmark_nsight.py             # single 10-demo batch
uv run scripts/benchmark_nsight.py --demos 100 # custom batch size
uv run scripts/benchmark_nsight.py --sweep     # 10/100/1000-demo scaling table
```

Hardware metrics captured on an NVIDIA RTX A5000 across batch sizes (`--sweep`):

| Metric | 10 demos | 100 demos | 1000 demos |
|--------|---------:|----------:|-----------:|
| Total Kernel Duration | 12.34 ms | 16.27 ms | 98.41 ms |
| Compute (SM) Throughput | 2.92% | 26.37% | 44.30% |
| Achieved Occupancy | 16.67% | 16.67% | 16.66% |
| Shared Mem Throughput | 0.11% | 0.97% | 1.64% |
| Registers Per Thread | 116 | 116 | 116 |
| Dynamic Shared Memory / Block | 42.25 KiB | 58.25 KiB | 96.25 KiB |

*Note: The dynamic shared memory allocation keeps the entire solver state (joint trajectories and kinematics cache) on-chip within L1/Shared Memory, minimizing latency during iterative projected gradient descent steps. It grows with batch size because all demos in a launch are padded to the longest trajectory in the batch, and larger batches include longer demos.*

**Why the hardware utilization numbers are low (and expected).** These metrics describe how much of the GPU's theoretical capacity the kernel uses, not how fast the pipeline runs—the two are decoupled here by design:

- **Small batches underfeed the GPU.** The kernel launches one block per demo, so a 10-demo profile occupies only ~10 of the RTX A5000's 64 SMs and most of the GPU sits idle. The sweep confirms this directly: SM throughput climbs from 2.92% at 10 demos to 44.30% at 1000 demos with the same kernel, while a 100× larger batch takes only ~8× longer to solve. The low utilization at small batches is underfeeding, not kernel inefficiency.
- **Shared memory limits residency.** With 42–96 KiB of dynamic shared memory per block, only a few blocks fit per SM, capping achieved occupancy at 16.67% regardless of batch size. This is a deliberate latency-vs-occupancy trade: keeping all solver state on-chip avoids global memory traffic and synchronization inside the iterative solve loop.
- **High register pressure.** The generated FK/Jacobian device code holds many intermediates in registers (116/thread), further reducing resident warps. The alternative—spilling to local memory—would be slower.
- **Workload shape.** Each block runs a small, structured optimizer (fixed-size vector math, limit projection, block-level sync), not a dense arithmetic workload, so low shared-memory and SM throughput percentages are normal even for a well-tuned kernel.

The meaningful end-to-end metric is the throughput table above: at ~492k pipeline frames/sec, the full ~2.8M-frame dataset retargets in under 6 seconds.

### Custom CUDA Kernels and Kinematics Codegen

The performance of the GPU backend relies on two key architectural components:
1. **Custom GPU Solver Kernel (`retarget_trajectory_kernel`)**: Implemented in [cuda/src/retarget_gpu.cu](cuda/src/retarget_gpu.cu). This kernel performs batched, warp-level projected gradient descent completely within GPU shared memory. By storing the active trajectory, joint limits, optimization weights, and kinematics cache on-chip, it completely avoids global memory latency and host-to-device synchronization overhead during iterative projection/solve loops.
2. **Forward Kinematics & Analytical Jacobian Codegen**: Evaluating forward kinematics (FK) and analytical local spatial Jacobians in parallel at 15 Hz is computationally intensive. To optimize this, highly optimized, **register-allocated, operation-minimized C++ kinematics kernels** are precomputed symbolically using a **separate custom generation codebase**. The resulting C++ kernels (committed under the [kernels/](kernels/) directory, e.g., for the UR3e tool0 frame) are translated into float32 CUDA device inline functions via our generator script [scripts/generate_device_fk.py](scripts/generate_device_fk.py). These generated device headers ([cuda/include/generated/ur3e_tool0_fk_device.cuh](cuda/include/generated/ur3e_tool0_fk_device.cuh)) are compile-time inlined within the CUDA kernel, allowing ultra-fast, instruction-minimized joint-to-Cartesian evaluations per warp thread.


## Tests

```bash
uv run pytest
```

Loader tests expect `data/droid_100` from `cs179 download` or the [Kaggle cache](https://www.kaggle.com/datasets/ryanrudes/droid-motion-data). Native retarget tests need `./scripts/build_loader.sh`.

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
