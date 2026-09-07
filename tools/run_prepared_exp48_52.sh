#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
PYTHON="$ROOT/.venv-cu129/bin/python"
EXP_NUMBER="${1:-}"

if [[ -z "${TMUX:-}" ]]; then
    printf 'Formal experiments must run inside a persistent tmux session.\n' >&2
    exit 88
fi

case "$EXP_NUMBER" in
    48) MODEL_NAME='V2-0048_实验48_全Query三维IoU分类乘深度' ;;
    49) MODEL_NAME='V2-0049_实验49_全样本确定性虚拟焦距' ;;
    50) MODEL_NAME='V2-0050_实验50_MixUp成功样本虚拟焦距' ;;
    51) MODEL_NAME='V2-0051_实验51_第121轮启用质量头训练' ;;
    52) MODEL_NAME='V2-0052_实验52_Cosine学习率' ;;
    *)
        printf 'Usage: %s {48|49|50|51|52}\n' "$0" >&2
        exit 89
        ;;
esac

CONFIG="$ROOT/configs/monodgp_exp${EXP_NUMBER}.yaml"
OUTPUT="$ROOT/outputs/$MODEL_NAME"
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
if ! mkdir "$OUTPUT"; then
    printf 'Refusing to overwrite existing Exp%s output: %s\n' \
        "$EXP_NUMBER" "$OUTPUT" >&2
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
        printf 'Failed to write Exp%s status receipt.\n' \
            "$EXP_NUMBER" >&2
        return 96
    fi
    if ! mv "$status_tmp" "$OUTPUT/status.tsv"; then
        printf 'Failed to publish Exp%s status receipt.\n' \
            "$EXP_NUMBER" >&2
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
