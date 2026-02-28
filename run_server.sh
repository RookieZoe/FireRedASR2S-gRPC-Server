#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory (handles symlinks and relative paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set cuBLAS library path from pip's nvidia-cublas package
# This fixes the CUBLAS_STATUS_INVALID_VALUE error by ensuring pip's cuBLAS 12.8
# takes precedence over system's cuBLAS 12.9
CUBLAS_LIB="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="$CUBLAS_LIB:${LD_LIBRARY_PATH:-}"

# Add vendor directory to PYTHONPATH for fireredasr2s imports
export PYTHONPATH="$SCRIPT_DIR/vendor/FireRedASR2S:${PYTHONPATH:-}"

# Execute the gRPC server with all arguments passed through
exec "$SCRIPT_DIR/.venv/bin/python" -m asr2s_grpc.serve "$@"
