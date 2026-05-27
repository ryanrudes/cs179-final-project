# Examples

Runnable snippets for project APIs. Add new examples here when documenting a feature.

| Example | Language | Run |
|---------|----------|-----|
| [reach_envelope_python.py](reach_envelope_python.py) | Python | `uv run python examples/reach_envelope_python.py` |
| [reach_envelope_cpp.cpp](reach_envelope_cpp.cpp) | C++ | See below |

## Reach envelope

Theory, CLI flags, and cache layout: [REACH_ENVELOPE.md](../REACH_ENVELOPE.md).

### Python

Uses `reachability` (pure Python or `cs179._native` when built). Pass `--quick` for a fast smoke run.

```bash
uv run python examples/reach_envelope_python.py
uv run python examples/reach_envelope_python.py --quick --no-native
```

### C++

Links against `cs179_core` (`cs179::DirectionalReachEnvelope`). Uses Pinocchio’s `humanoidRandom` model so the example builds without urdfdom. For real robots, use the Python example or load a `pinocchio::Model` from URDF in your own target.

```bash
./scripts/build_loader.sh
cmake -B build -DCS179_BUILD_EXAMPLES=ON -DCS179_BUILD_PYTHON=ON
cmake --build build --target reach_envelope_cpp
./build/reach_envelope_cpp
```
