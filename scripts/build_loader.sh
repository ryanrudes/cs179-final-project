#!/usr/bin/env bash
# Build RLDS loader + Python bindings without CUDA (no nvcc required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
OUTPUT_DIR="${ROOT}/src/cs179"

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

PYTHON_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"

cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DPython3_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCS179_BUILD_CUDA=OFF \
    -DCS179_BUILD_LOADER_TEST=ON

cmake --build "${BUILD_DIR}" --config Release

find "${BUILD_DIR}" -maxdepth 3 -name '_native*.so' -exec cp {} "${OUTPUT_DIR}/" \;

echo "Built _native (RLDS loader + reach envelope + retarget when Pinocchio is available) into ${OUTPUT_DIR}"
echo "C++ smoke test: ${BUILD_DIR}/cs179_loader_test [data/droid_100]"
