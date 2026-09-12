#!/usr/bin/env bash
set -euo pipefail

MTP_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VIRTUAL_ENV="$MTP_REPO_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="${FREETOKEN_SOURCE_ROOT:-$MTP_REPO_ROOT}/python"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XDG_CACHE_HOME="$MTP_REPO_ROOT/.cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TVM_FFI_CACHE_DIR="$XDG_CACHE_HOME/tvm-ffi"
export FLASHINFER_WORKSPACE_BASE="$MTP_REPO_ROOT"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TMPDIR="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR"
exec "$@"
