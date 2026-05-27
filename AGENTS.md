## Learned User Preferences

- Only create git commits when the user explicitly asks.
- Prefer minimal, focused changes; match existing layout and naming when extending the project.
- Name classes for their real scope (e.g. generic RLDS loaders, not `Droid*` once multiple datasets are supported); prefer shorter names when an enum already conveys robotics context.
- With `--no-adapter`, infer observation field shapes from the first step block seen during download; do not hardcode joint/gripper/cartesian dimensions in the downloader. With adapters, shapes come from adapter `Field` specs (DROID-style layout).
- Store downloader metadata as JSON (`metadata/metadata.json`), not pickle or `dtype=object` `.npy` files; do not add legacy metadata, pickle loaders, or on-disk cache fallbacks—assume data was written by the current downloader.
- Keep `trim_partial_shards=True` as the default; use `--no-trim-partial-shards` only when faster finalize or padded final shards are preferred.
- When materializing full RLDS episodes, concatenate all step batches from `iter_episode_step_batches` (see `stack_episode`); never use only the first `batch(10_000)` (avoids silent truncation on long episodes).
- Default `data_dir` to `data` and append the dataset name parsed from the RLDS URL (e.g. `data/droid_100`).
- Store observation shards as `{observation_key}/{shard_id:05d}.npy` under the dataset directory.
- When refactoring or splitting modules, preserve behavior exactly—no implicit changes to retargeting, solver, or initialization order.
- Prefer conventional Python module splits (shared helpers, explicit imports) over duplicating or reordering logic across files.
- When testing native RLDS/loader code, prefer loader-only builds (`./scripts/build_loader.sh` or `build_native.sh --no-cuda`) so `nvcc` is not required.

## Learned Workspace Facts

### Project layout and tooling

- **Examples:** `examples/` — runnable API walkthroughs (`examples/README.md`, `reach_envelope_python.py`, `reach_envelope_cpp.cpp`); add new examples there when requested. Python: `uv run python examples/reach_envelope_python.py [--quick] [--no-native]`. C++: `cmake -DCS179_BUILD_EXAMPLES=ON`, target `reach_envelope_cpp` (uses `humanoidRandom`, no urdfdom).

- **Python ≥3.12**, managed with **uv** (`uv sync --dev`, `uv run …`). **Hatchling** wheel includes `cs179`, `rlds`, `app`, `retarget`, `reachability`, and top-level `cli` (`src/cli.py` via `force-include`).
- **Two Python surfaces:** (1) installable CUDA package `cs179` (`vector_add` via `_native`); (2) application modules `cli`, `app`, `rlds`, `retarget`, `reachability`. Runtime deps in `pyproject.toml`: **typer**, **tensorflow**, **tensorflow-datasets**, **absl-py**, **rich**, **scipy**, **pin**, **robot-descriptions**, **yourdfpy**, **viser**, **matplotlib**, **meshcat**, **numpy**, **pyyaml**.
- **CLI:** `uv run cs179 <subcommand>` (entry point `cli:main`). Subcommands: `download`, `retarget`, `reach-envelope` (`build` | `visualize`), `reachability` (deprecated alias for `reach-envelope visualize`). Shared flags in `src/cli_native.py`: `--no-native` (`NoNativeOption`, forces Python reach envelope and retargeting via `set_use_native_envelope(False)` / `set_use_native_retarget(False)` on retarget, reach-envelope build/visualize, and deprecated reachability); `--reach-safety` (`ReachSafetyOption`, default `REACH_SAFETY_MARGIN=0.9`, must be in (0,1]; retarget scaling and visualize boundary mesh only—not on build). **`cs179 retarget --config`** selects retarget YAML (default `config/default.yaml`). After editing **`src/cli.py`** or **`src/cli_reach_envelope.py`** (Hatchling `force-include`), run `uv pip install -e . --force-reinstall --no-deps` if `uv sync` leaves stale `site-packages` copies.
- **Native build:** Root `CMakeLists.txt` is CXX-only when `CS179_BUILD_CUDA=OFF`. Loader-only: `./scripts/build_loader.sh` (also `cs179_loader_test`) or `./scripts/build_native.sh --no-cuda`; `build_native.sh` auto-falls back if `nvcc` is missing. Full CUDA: `./scripts/build_native.sh` with `nvcc`. `_native` uses `bindings_rlds.cpp` always, `bindings_reachability.cpp` + `bindings_retarget.cpp` when Pinocchio/NLopt available, `bindings_cuda.cpp` only with CUDA; `RldsObservationLoader` always, `vector_add` only with CUDA. Rebuild `_native` after any C++/binding change.

### RLDS download and cache

- **Classes:** `RldsObservationDownloader`, `RldsObservationLoader` in `src/rlds/`.
- **Cache root:** `resolve_data_dir("data", url)` → e.g. `data/droid_100/`. **Metadata dir:** `data/{dataset}/metadata/` with `metadata.json` plus index arrays: `demo_lengths.npy`, `demo_offsets.npy`, `shard_lengths.npy`, `shard_offsets.npy`, `total_steps.npy`.
- **Shards:** `data/{dataset}/{observation_key}/{shard_id:05d}.npy`.
- **Default proprio keys** (naive / DROID-style): `joint_position`, `gripper_position`, `cartesian_position`. **Dataset adapters** in `src/rlds/adapters/` normalize to those keys by default; `--no-adapter` keeps native RLDS keys.
- **TFDS decode:** On nested `steps` features, `tfds.decode.PartialDecoding` does not work; use per-field `tfds.decode.SkipDecoding` (skipping images is the largest decode win). Downloader streams episodes via `iter_episode_step_batches` (batch size 10_000).
- **`control_hz`:** documented per dataset in `src/rlds/timing.py` (DROID 15 Hz, KUKA 3 Hz); written to `metadata.json` on download and used by `Retargeter` for velocity/acceleration costs.

### Retarget and reachability

- **`uv run cs179 retarget`** retargets cached proprio demos onto UR3e; requires prior **`uv run cs179 download`** (or existing cache under `data/{dataset}/`). Uses `RldsObservationLoader` / `iter_retarget_demos`, not live TFDS. Target frame: `tool0`. Reach cache: `--reach-n-theta`, `--reach-n-phi`, `--reach-n-samples`, `--reach-force-rebuild` (defaults 64×128, 10M samples); `--reach-safety` clamps Cartesian targets via `scale_cartesian_to_robot`.
- **Retarget config YAML:** `config/default.yaml` (Hatchling force-include); sections `runtime`, `cost` (`units`, `weights`, `elbow_margin`), `ik_seed`, `optimizer`, `frames`—avoid `_unit`/`_weight` suffixes when nested under `cost.units`/`cost.weights`. `src/retarget/config.py` (`RetargetConfig` nested dataclasses); CLI `--config path/to.yaml` (default `config/default.yaml`). `RetargetConfig.to_native_dict()` → C++ `RetargetParams`.
- **Retarget optimizer:** `src/retarget/core.py` `Retargeter` — native **NLopt** `LD_LBFGS` with joint-limit bounds when `cs179._native` built; Python fallback **SciPy** `L-BFGS-B`. IK seeding, joint velocity/acceleration penalties, reach envelope scaling. `--no-native` disables native reach envelope and retarget (SciPy path); it does not isolate reach vs retarget—use `set_use_native_envelope` / `set_use_native_retarget` in code to compare one backend at a time.
- **`src/reachability/`:** envelope (`envelope.py`), Viser (`viz.py`); backends in `src/app/reach_envelope.py`. `uv run cs179 reach-envelope build` (writes `data/reach_envelopes/{robot}_{frame}_{n_theta}x{n_phi}_{n_samples}.npz`, no Viser) or `visualize` (not started from retarget). Viser defaults: **directional envelope only** (other layers off); no FK cloud unless `--fk-samples`; `--compare-robot` optional (default `None`). `--compare-samples` overrides compare FK count; otherwise compare uses the same `--n-samples` and bin grid as the primary robot. Visualize `--mesh-theta`/`--mesh-phi` default to `--n-theta`/`--n-phi`; use `--force-rebuild` after changing bins or sample count. See **C++ directional reach envelope** and `REACH_ENVELOPE.md`.
- **Retarget viz:** Meshcat + matplotlib on by default; `--disable-visualization` turns off Meshcat only. Headless batch: `--disable-visualization --no-plots`.

### C++ directional reach envelope

- **Core:** `cpp/include/cs179/reach_envelope.hpp`, `cpp/src/reach_envelope.cpp` — `cs179::DirectionalReachEnvelope` (Monte Carlo FK, empty-bin fill, bilinear `reach_limits`, `scale_positions`).
- **Bindings:** `cpp/bindings_reachability.cpp` → `cs179._native.DirectionalReachEnvelope`; `cpp/bindings.cpp` registers `bind_reachability` only when `CS179_HAVE_REACHABILITY`. Module load requires `import pinocchio` (Pinocchio `Model&` pybind bridge).
- **CMake:** `CS179_BUILD_REACHABILITY` (default ON); `CS179_BUILD_EXAMPLES` (OFF) builds `reach_envelope_cpp` from `examples/`. Pinocchio from venv `cmeel.prefix` via `pinocchioTargets.cmake`, link `pinocchio::pinocchio_default` only (avoids urdfdom_headers find_package failure). Dev headers from `Python3_EXECUTABLE`’s `sys.base_prefix` (not system Python); links `Python3::Module`, numpy, `boost_python312`, `eigenpy`. OpenMP when found.
- **OpenMP:** per-batch parallel loop — per-thread `pinocchio::Data`, RNG, full bin grid; static schedule; merge max after batch (no atomics on shared grid).
- **Python split:** `envelope.py` delegates `_build_bin_radii_batched`, `reach_limits`, `scale_positions`, `max_radius` to native when `_native` is built and not disabled; `--no-native` / `set_use_native_envelope(False)` forces Python. Native bindings require **C-contiguous** `(N,3)` row-major arrays; `envelope.py` calls `np.ascontiguousarray` before native `reach_limits`/`scale_positions` (e.g. `out[:, :3]` from `(N,6)` poses is non-contiguous). Cache/save/load/boundary_mesh stay Python. Defaults: 64×128 bins, 50k batch, 10M retarget samples, `REACH_SAFETY_MARGIN=0.9`, cache `data/reach_envelopes`.
- **Sampling:** C++ draws each `q[i]` uniform in joint limits; Python fallback uses `pin.randomConfiguration` — not bit-identical for the same `n_samples`.
- **Tests:** `tests/test_reach_envelope.py` — interpolation parity, non-contiguous xyz slice vs native (`test_native_scale_positions_non_contiguous_xyz_slice`), build smoke (skipped without native).

### C++ native retarget

- **Core:** `cpp/include/cs179/retarget.hpp`, `cpp/src/retarget.cpp`, `retarget_params.hpp` — `cs179::Retargeter` (Pinocchio FK + **NLopt** `LD_LBFGS`, central-diff gradients; not a custom solver).
- **Bindings:** `cpp/bindings_retarget.cpp` → `cs179._native.Retargeter`; built with reachability (`CS179_HAVE_REACHABILITY`). NLopt **v2.9.1** via CMake FetchContent when Pinocchio is available.
- **Python split:** `core.py` delegates to native when built and not disabled; warm-started trajectories usually match closely—joint paths can diverge while native improves pose error vs SciPy local minima.
- **seed_ik:** C++ damped least-squares on `JJt` must use Eigen `householderQr().solve(nu)` (matches NumPy `linalg.solve`); `ldlt()` and column-pivoting QR diverge on ill-conditioned targets (e.g. fine reach bins 1024×1024). Public `cs179::seed_ik` / `cs179._native.seed_ik` (pybind `forcecast` for float32 targets).
- **Quality tests:** `tests/test_retarget_native_quality.py` (slow, needs `data/droid_100`) and `scripts/compare_retarget_native.py` — assert **task-space pose** error parity, not joint equality; fails only when native position error exceeds Python by configured margins. Quick smoke: `tests/test_retarget_native.py` (`test_native_seed_ik_matches_python` guards C++ vs Python `seed_ik` on hard scaled targets).
- **Rebuild:** `./scripts/build_loader.sh` or `build_native.sh` after C++/binding changes (`_native` gitignored under `src/cs179/`).

### CUDA Python API

- `from cs179 import vector_add` — contiguous 1-D `float32` numpy arrays via `_native.vector_add(a, b, out)`.
- Example kernel: `cs179::vector_add` in `cuda/src/kernels.cu` (`cuda/include/kernels.cuh`).
- `_native*.so` under `src/cs179/` is gitignored; rebuild after CUDA or binding changes.

### C++ RLDS cache loader

- **`cs179::RldsObservationLoader`** in `cpp/include/cs179/rlds_loader.hpp`; static lib **`cs179_core`** (mmap, minimal NPY, **nlohmann/json** metadata via FetchContent).
- Mirrors Python `RldsObservationLoader`: `get_demo`, `get_step_range`, `get_demo_views` (zero-copy mmap when a demo fits one shard).
- Pybind11: `from cs179._native import RldsObservationLoader` (same method names as Python).
- Rebuild/test loader: `./scripts/build_loader.sh` → `_native` in `src/cs179/` plus `build/cs179_loader_test` (also builds reach envelope and native retarget when Pinocchio is available). `uv run pytest tests/test_rlds_loader.py` (needs `data/droid_100`); `tests/test_reach_envelope.py` when native envelope is built; `tests/test_retarget_native.py` / slow `tests/test_retarget_native_quality.py` when native retarget is built; `tests/test_vector_add.py` skips when `_native` lacks `vector_add` (full CUDA build required to run).
