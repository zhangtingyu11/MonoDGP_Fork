#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
PYTHON="$ROOT/.venv-cu129/bin/python"
export CUDA_HOME=/usr/local/cuda-12.9
export TORCH_CUDA_ARCH_LIST=8.9

cd "$(dirname "$0")"
"$PYTHON" setup.py build_ext --inplace
