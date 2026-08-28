#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
PYTHON="$ROOT/.venv-cu129/bin/python"
CONFIG="$ROOT/configs/monodgp_exp46.yaml"
OUTPUT="$ROOT/outputs/V2-0046_实验46_确定性优化新基线"
COMMAND="$PYTHON tools/train_val.py --config $CONFIG"

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_HOME=/usr/local/cuda-12.9
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export ALL_PROXY=socks5h://127.0.0.1:7897
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export all_proxy="$ALL_PROXY"

cd "$ROOT" || exit 90
"$PYTHON" tools/write_run_manifest.py \
    --config "$CONFIG" --command "$COMMAND" || exit $?
"$PYTHON" tools/train_val.py --config "$CONFIG" \
    2>&1 | tee "$OUTPUT/train_console.log"
status=${PIPESTATUS[0]}
printf 'train_exit\t%d\n' "$status" > "$OUTPUT/status.tsv"
exit "$status"
