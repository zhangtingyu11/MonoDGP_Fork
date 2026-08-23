#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
approved_python="$repo_dir/.venv-cu129/bin/python"
exp35_wrapper_pid="1058441"
exp35_wrapper_start_ticks="230585832"
exp35_output_dir="$repo_dir/outputs/V2-0035_实验33完全关闭MixUp"
exp35_status_file="$exp35_output_dir/experiment_exit_status.txt"
exp36_output_dir="$repo_dir/outputs/V2-0036_实验33目标保护跨焦距MixUp"
orchestration_dir="$repo_dir/outputs/_orchestration"
orchestration_log="$orchestration_dir/exp36_after_exp35_20260821.log"
orchestration_status="$orchestration_dir/exp36_after_exp35_20260821.status"

declare -A expected_hashes=(
    ["configs/monodgp.yaml"]="7876048f164939345116f218e49da1ddf7c41183a31dfbd67abe444ea0f454ea"
    ["configs/monodgp_exp29.yaml"]="6eaf26c54c722c37a980ef057a88e14d075e9a7b9d2c486683326983aa031a3b"
    ["configs/monodgp_exp30.yaml"]="8b9d9fa38f34be9d37b672359b67c633489ce065ef119d9d0ce6558a9ab11bba"
    ["configs/monodgp_exp33.yaml"]="70201e0b62ffac323a7f9176fb9d14636090c25133ed0f24ba7a1a0b4c1e5cb1"
    ["configs/monodgp_exp36.yaml"]="4202459349eaab244ab0469d1e8cc1917af1e781e0ed6b1a32eee4ad05e9ea5f"
    ["lib/datasets/kitti/kitti_dataset.py"]="c639b2121901a76e036eadfe76bb32de6e5599bd3cf48b9c98428fb497b4fa0b"
    ["lib/datasets/kitti/mixup_geometry.py"]="629d240c9db8fe1eced9ee1434cc9748b8ac255ea1577612e345cc6b1ca6a850"
    ["lib/helpers/config_helper.py"]="c300e9512b7880b60276e38150ebd771b2ef55dc505c3c11bd340bfc28aa2d27"
    ["lib/helpers/decode_helper.py"]="7fe7382b93835a82dcba11d6385dedc171e6a0e8d4ee1bbe3a02d6c87423de51"
    ["lib/helpers/swanlab_helper.py"]="bf83effaab9ac8fdf35a994cfc6023ef06ea7dd10a278c51ca21f705ef3288c7"
    ["lib/helpers/tester_helper.py"]="8afbb0c3efaf0b1b3159c67a5ab6537e955980d94bf6b5b8d3388f7bcfabe08e"
    ["lib/helpers/trainer_helper.py"]="96d5d90c9915fff64dfc6036c7cb78407f79698dd0a945b0c0c9214cd8065077"
    ["lib/models/monodgp/matcher.py"]="26eba7a2dd15f39d228ce98ae79751dc9ce7eeab828bf41cf441852d3783e5ca"
    ["lib/models/monodgp/monodgp.py"]="42f3f616e61ac26f71acbdd76881a719be1f0292eae56335d0f6e10a7f43796e"
    ["tools/smoke_exp36_b1.py"]="e57508217333a9a896b5bf52000c8c46d8a62bb2ebd9a63c5fd17e9a054117ea"
    ["tools/train_val.py"]="ba274e75e8e124dd42c8a67f62fb5bc7cb3cadb5410022d40d309f92bffce9ea"
    ["tools/write_run_manifest.py"]="860ce6261105821a5233cd14a2712c52da9ab99fd791e6db244ea9571fa4d303"
    ["run_experiment_36.sh"]="e6264d9ce9ae13477a64525c8c7e87f7c35b08a865bdae0691bc5abec42932f7"
)

mkdir -p "$orchestration_dir"
if [[ -e "$orchestration_log" || -e "$orchestration_status" ]]; then
    echo "Refusing to overwrite Experiment 36 orchestration receipts" >&2
    exit 10
fi
exec >>"$orchestration_log" 2>&1

started_at="$(date --iso-8601=seconds)"
finished=0

write_failure_status() {
    local exit_code="$1"
    if [[ "$finished" -eq 0 ]]; then
        printf 'status=failed\nexit_code=%s\nstarted_at=%s\nfinished_at=%s\n' \
            "$exit_code" "$started_at" "$(date --iso-8601=seconds)" \
            > "$orchestration_status"
    fi
}
trap 'write_failure_status "$?"' EXIT

process_is_original_exp35_wrapper() {
    [[ -r "/proc/$exp35_wrapper_pid/stat" ]] || return 1
    local current_start_ticks
    current_start_ticks="$(awk '{print $22}' "/proc/$exp35_wrapper_pid/stat")"
    [[ "$current_start_ticks" == "$exp35_wrapper_start_ticks" ]]
}

verify_exp36_source_snapshot() {
    local relative_path absolute_path actual_hash
    for relative_path in "${!expected_hashes[@]}"; do
        absolute_path="$repo_dir/$relative_path"
        if [[ ! -f "$absolute_path" ]]; then
            printf 'Experiment 36 source file is missing: %s\n' \
                "$absolute_path" >&2
            return 1
        fi
        actual_hash="$(sha256sum "$absolute_path")"
        actual_hash="${actual_hash%% *}"
        if [[ "$actual_hash" != "${expected_hashes[$relative_path]}" ]]; then
            printf 'Experiment 36 source drift: %s expected=%s actual=%s\n' \
                "$relative_path" "${expected_hashes[$relative_path]}" \
                "$actual_hash" >&2
            return 1
        fi
    done
}

printf '[%s] Waiting for Experiment 35 wrapper pid=%s start_ticks=%s\n' \
    "$(date --iso-8601=seconds)" "$exp35_wrapper_pid" \
    "$exp35_wrapper_start_ticks"
while process_is_original_exp35_wrapper; do
    sleep 30
done

printf '[%s] Experiment 35 wrapper exited; checking completion receipt\n' \
    "$(date --iso-8601=seconds)"
for _ in {1..12}; do
    [[ -f "$exp35_status_file" ]] && break
    sleep 5
done
if [[ ! -f "$exp35_status_file" ]]; then
    printf 'Experiment 35 status receipt is missing: %s\n' \
        "$exp35_status_file" >&2
    exit 20
fi

exp35_status=""
exp35_exit_code=""
while IFS='=' read -r key value; do
    case "$key" in
        status) exp35_status="$value" ;;
        exit_code) exp35_exit_code="$value" ;;
    esac
done < "$exp35_status_file"
if [[ "$exp35_status" != "completed" || "$exp35_exit_code" != "0" ]]; then
    printf 'Experiment 35 did not complete successfully: status=%s exit_code=%s\n' \
        "$exp35_status" "$exp35_exit_code" >&2
    exit 21
fi
if [[ ! -f "$exp35_output_dir/checkpoint.pth" \
        || ! -f "$exp35_output_dir/checkpoint_best.pth" ]]; then
    echo 'Experiment 35 completed but expected checkpoints are missing' >&2
    exit 22
fi
if ! verify_exp36_source_snapshot; then
    exit 23
fi
if [[ -e "$exp36_output_dir" ]]; then
    printf 'Refusing to overwrite Experiment 36 output: %s\n' \
        "$exp36_output_dir" >&2
    exit 24
fi

printf '[%s] Running Experiment 36 GPU B1 forward/loss/backward smoke\n' \
    "$(date --iso-8601=seconds)"
cd "$repo_dir"
CUDA_VISIBLE_DEVICES=0 "$approved_python" tools/smoke_exp36_b1.py \
    --config configs/monodgp_exp36.yaml

printf '[%s] Preflight passed; starting Experiment 36\n' \
    "$(date --iso-8601=seconds)"
MONODGP_CUDA_VISIBLE_DEVICES=0 ./run_experiment_36.sh

finished=1
printf 'status=completed\nexit_code=0\nstarted_at=%s\nfinished_at=%s\n' \
    "$started_at" "$(date --iso-8601=seconds)" \
    > "$orchestration_status"
trap - EXIT
printf '[%s] Experiment 36 completed successfully\n' \
    "$(date --iso-8601=seconds)"
