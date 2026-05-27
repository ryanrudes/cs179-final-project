#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
OUTPUT_DIR="${ROOT}/src/cs179"

mkdir -p "${BUILD_DIR}" "${OUTPUT_DIR}"

PYTHON_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"

CUDA_FLAG=ON
if [[ "${1:-}" == "--no-cuda" ]]; then
    CUDA_FLAG=OFF
elif ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found; building loader-only _native (pass --cuda to require CUDA)." >&2
    CUDA_FLAG=OFF
fi

CMAKE_ARGS=(
    -S "${ROOT}"
    -B "${BUILD_DIR}"
    -DPython3_EXECUTABLE="${PYTHON_EXECUTABLE}"
    -DCMAKE_BUILD_TYPE=Release
    -DCS179_BUILD_CUDA="${CUDA_FLAG}"
)

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --config Release

find "${BUILD_DIR}" -maxdepth 3 -name '_native*.so' -exec cp {} "${OUTPUT_DIR}/" \;

if [[ "${CUDA_FLAG}" == ON ]]; then
    echo "Built _native (RLDS loader + CUDA) into ${OUTPUT_DIR}"
else
    echo "Built _native (RLDS loader only) into ${OUTPUT_DIR}"
fi
