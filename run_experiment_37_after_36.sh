#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
approved_python="$repo_dir/.venv-cu129/bin/python"
exp36_wrapper_pid="3206996"
exp36_wrapper_start_ticks="235253057"
exp36_output_dir="$repo_dir/outputs/V2-0036_实验33目标保护跨焦距MixUp"
exp36_status_file="$exp36_output_dir/experiment_exit_status.txt"
exp37_output_dir="$repo_dir/outputs/V2-0037_实验36_MixUp绑定确定性虚拟焦距"
orchestration_dir="$repo_dir/outputs/_orchestration"
orchestration_log="$orchestration_dir/exp37_after_exp36_20260822_v6.log"
orchestration_status="$orchestration_dir/exp37_after_exp36_20260822_v6.status"

declare -A expected_hashes=(
    ["configs/monodgp.yaml"]="5aee78fb38020a5353fa0b0f25d054d3149f48c8132e7b8b3d45a9e85c4accab"
    ["configs/monodgp_exp29.yaml"]="6eaf26c54c722c37a980ef057a88e14d075e9a7b9d2c486683326983aa031a3b"
    ["configs/monodgp_exp30.yaml"]="8b9d9fa38f34be9d37b672359b67c633489ce065ef119d9d0ce6558a9ab11bba"
    ["configs/monodgp_exp33.yaml"]="70201e0b62ffac323a7f9176fb9d14636090c25133ed0f24ba7a1a0b4c1e5cb1"
    ["configs/monodgp_exp36.yaml"]="4202459349eaab244ab0469d1e8cc1917af1e781e0ed6b1a32eee4ad05e9ea5f"
    ["configs/monodgp_exp37.yaml"]="b8568043803e26c2081a77bdde648199afe6b11b3403f3db15d9d533b44126c1"
    ["lib/datasets/kitti/kitti_dataset.py"]="854a91054a4cd282d9978556f1212342242ee963bb8a9ea83d2454b29eda2eb2"
    ["lib/datasets/kitti/mixup_geometry.py"]="85a48ddfd0a99a10c8c99c4bb9ef42bbbddf9c56c833c6d58e2c8d70c7983f85"
    ["lib/helpers/config_helper.py"]="c300e9512b7880b60276e38150ebd771b2ef55dc505c3c11bd340bfc28aa2d27"
    ["lib/helpers/decode_helper.py"]="7fe7382b93835a82dcba11d6385dedc171e6a0e8d4ee1bbe3a02d6c87423de51"
    ["lib/helpers/swanlab_helper.py"]="bf83effaab9ac8fdf35a994cfc6023ef06ea7dd10a278c51ca21f705ef3288c7"
    ["lib/helpers/tester_helper.py"]="dcb95dbb9b64103b82c4b092744c2db40211a47fa6e92620657ccc301ab5d37c"
    ["lib/helpers/trainer_helper.py"]="c23a043c43e300d625a96c9f2176d462beee174aa067f44c4352bf322b505e46"
    ["lib/losses/focal_loss.py"]="25c31eeb649647ce4f9d60ab61be5160d35b8860d5018103ad05f652fb3f75e1"
    ["lib/losses/asymmetric_interval_depth_loss.py"]="1583f3d8b9a2f099a1e87fe1d01e28f4e1203cccaf087d92431b87c5855fd848"
    ["lib/models/monodgp/iou3d_match_cost.py"]="91aaa430a14fd0ca343ae61f532a383dee5358659c2af27f9c271f892c533d52"
    ["lib/models/monodgp/matcher.py"]="26eba7a2dd15f39d228ce98ae79751dc9ce7eeab828bf41cf441852d3783e5ca"
    ["lib/models/monodgp/monodgp.py"]="22959003ee1bfa1dd7eee53e95b166e7e4f1f02f3a65d0491a0211b8473fd4f3"
    ["tools/smoke_exp37_b16.py"]="651d65c6196355d75845471cf1654b523fe3d84a2cd9d5b31a075c06ebed2d7c"
    ["tools/train_val.py"]="ba274e75e8e124dd42c8a67f62fb5bc7cb3cadb5410022d40d309f92bffce9ea"
    ["tools/write_run_manifest.py"]="c7f6791816b0432be2d7242b5f0414393703964dec3b7329a0421e10e5ea6be0"
    ["run_experiment_37.sh"]="902a732f6611b2099a8ef8a799deba215c96d736ed9760d986cced85cc712420"
)

mkdir -p "$orchestration_dir"
if [[ -e "$orchestration_log" || -e "$orchestration_status" ]]; then
    echo "Refusing to overwrite Experiment 37 orchestration receipts" >&2
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

process_is_original_exp36_wrapper() {
    [[ -r "/proc/$exp36_wrapper_pid/stat" ]] || return 1
    local current_start_ticks
    current_start_ticks="$(awk '{print $22}' "/proc/$exp36_wrapper_pid/stat")"
    [[ "$current_start_ticks" == "$exp36_wrapper_start_ticks" ]]
}

verify_exp37_source_snapshot() {
    local relative_path absolute_path actual_hash
    for relative_path in "${!expected_hashes[@]}"; do
        absolute_path="$repo_dir/$relative_path"
        [[ -f "$absolute_path" ]] || {
            printf 'Experiment 37 source missing: %s\n' "$absolute_path" >&2
            return 1
        }
        actual_hash="$(sha256sum "$absolute_path")"
        actual_hash="${actual_hash%% *}"
        if [[ "$actual_hash" != "${expected_hashes[$relative_path]}" ]]; then
            printf 'Experiment 37 source drift: %s expected=%s actual=%s\n' \
                "$relative_path" "${expected_hashes[$relative_path]}" \
                "$actual_hash" >&2
            return 1
        fi
    done
}

printf '[%s] Waiting for Experiment 36 wrapper pid=%s start_ticks=%s\n' \
    "$(date --iso-8601=seconds)" "$exp36_wrapper_pid" \
    "$exp36_wrapper_start_ticks"
while process_is_original_exp36_wrapper; do
    sleep 30
done

printf '[%s] Experiment 36 wrapper exited; checking completion receipt\n' \
    "$(date --iso-8601=seconds)"
for _ in {1..12}; do
    [[ -f "$exp36_status_file" ]] && break
    sleep 5
done
if [[ ! -f "$exp36_status_file" ]]; then
    echo "Experiment 36 status receipt is missing" >&2
    exit 20
fi
exp36_status=""
exp36_exit_code=""
while IFS='=' read -r key value; do
    case "$key" in
        status) exp36_status="$value" ;;
        exit_code) exp36_exit_code="$value" ;;
    esac
done < "$exp36_status_file"
if [[ "$exp36_status" != "completed" || "$exp36_exit_code" != "0" ]]; then
    printf 'Experiment 36 did not complete: status=%s exit_code=%s\n' \
        "$exp36_status" "$exp36_exit_code" >&2
    exit 21
fi
if [[ ! -f "$exp36_output_dir/checkpoint.pth" \
        || ! -f "$exp36_output_dir/checkpoint_best.pth" ]]; then
    echo "Experiment 36 completed but checkpoints are missing" >&2
    exit 22
fi
verify_exp37_source_snapshot || exit 23
if [[ -e "$exp37_output_dir" ]]; then
    echo "Refusing to overwrite Experiment 37 output: $exp37_output_dir" >&2
    exit 24
fi

printf '[%s] Running Experiment 37 real KITTI B16 correctness/timing gate\n' \
    "$(date --iso-8601=seconds)"
cd "$repo_dir"
CUDA_VISIBLE_DEVICES=0 "$approved_python" tools/smoke_exp37_b16.py --epoch 7

printf '[%s] Preflight passed; starting Experiment 37\n' \
    "$(date --iso-8601=seconds)"
MONODGP_CUDA_VISIBLE_DEVICES=0 ./run_experiment_37.sh

finished=1
printf 'status=completed\nexit_code=0\nstarted_at=%s\nfinished_at=%s\n' \
    "$started_at" "$(date --iso-8601=seconds)" > "$orchestration_status"
trap - EXIT
printf '[%s] Experiment 37 completed successfully\n' \
    "$(date --iso-8601=seconds)"
