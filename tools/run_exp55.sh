#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
PYTHON="$ROOT/.venv-cu129/bin/python"
CONFIG=${MONODGP_FORMAL_CONFIG:-"$ROOT/configs/monodgp_exp55.yaml"}
OUTPUT=${MONODGP_FORMAL_OUTPUT:-"$ROOT/outputs/V2-0055_实验55_Exp47_MixUp概率0.3"}
COMMAND="$PYTHON tools/train_val.py --config $CONFIG"

if [[ -z "${TMUX:-}" ]]; then
    printf 'Formal experiments must run inside a persistent tmux session.\n' >&2
    exit 88
fi

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
if ! mkdir "$OUTPUT"; then
    printf 'Refusing to overwrite existing Exp55 output: %s\n' "$OUTPUT" >&2
    exit 91
fi

publish_status() {
    local runner_status=$1
    local manifest_status=$2
    local train_status=$3
    local tee_status=$4
    local status_tmp="$OUTPUT/status.tsv.tmp.$$"
    if ! printf \
            'runner_exit\t%d\nmanifest_exit\t%d\ntrain_exit\t%d\ntee_exit\t%d\n' \
            "$runner_status" "$manifest_status" \
            "$train_status" "$tee_status" > "$status_tmp"; then
        printf 'Failed to write Exp55 status receipt.\n' >&2
        return 96
    fi
    if ! mv "$status_tmp" "$OUTPUT/status.tsv"; then
        printf 'Failed to publish Exp55 status receipt.\n' >&2
        return 96
    fi
}

"$PYTHON" tools/write_run_manifest.py \
    --config "$CONFIG" --command "$COMMAND"
manifest_status=$?
if [[ "$manifest_status" != 0 ]]; then
    publish_status "$manifest_status" "$manifest_status" -1 -1 \
        || exit $?
    exit "$manifest_status"
fi

"$PYTHON" tools/train_val.py --config "$CONFIG" \
    2>&1 | tee "$OUTPUT/train_console.log"
pipeline_status=("${PIPESTATUS[@]}")
train_status=${pipeline_status[0]}
tee_status=${pipeline_status[1]}
runner_status=$train_status
if [[ "$runner_status" == 0 && "$tee_status" != 0 ]]; then
    runner_status=95
fi

publish_status "$runner_status" "$manifest_status" \
    "$train_status" "$tee_status" || exit $?
exit "$runner_status"
