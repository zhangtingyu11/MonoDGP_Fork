#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
approved_python="$repo_dir/.venv-cu129/bin/python"
exp37_wrapper_pid="1264253"
exp37_wrapper_start_ticks="240037504"
exp37_output_dir="$repo_dir/outputs/V2-0037_实验36_MixUp绑定确定性虚拟焦距"
exp37_status_file="$exp37_output_dir/experiment_exit_status.txt"
exp38_output_dir="$repo_dir/outputs/V2-0038_实验37全Query三维IoU分类"
orchestration_dir="$repo_dir/outputs/_orchestration"
orchestration_log="$orchestration_dir/exp38_after_exp37_20260822.log"
orchestration_status="$orchestration_dir/exp38_after_exp37_20260822.status"

declare -A expected_hashes=(
    ["configs/monodgp.yaml"]="5aee78fb38020a5353fa0b0f25d054d3149f48c8132e7b8b3d45a9e85c4accab"
    ["configs/monodgp_exp29.yaml"]="6eaf26c54c722c37a980ef057a88e14d075e9a7b9d2c486683326983aa031a3b"
    ["configs/monodgp_exp30.yaml"]="8b9d9fa38f34be9d37b672359b67c633489ce065ef119d9d0ce6558a9ab11bba"
    ["configs/monodgp_exp33.yaml"]="70201e0b62ffac323a7f9176fb9d14636090c25133ed0f24ba7a1a0b4c1e5cb1"
    ["configs/monodgp_exp36.yaml"]="4202459349eaab244ab0469d1e8cc1917af1e781e0ed6b1a32eee4ad05e9ea5f"
    ["configs/monodgp_exp37.yaml"]="b8568043803e26c2081a77bdde648199afe6b11b3403f3db15d9d533b44126c1"
    ["configs/monodgp_exp38.yaml"]="c00d8955843655133d8a6a6c4413d8b82f5f055ed32b659c486bb35ec99706e4"
    ["lib/datasets/kitti/kitti_dataset.py"]="854a91054a4cd282d9978556f1212342242ee963bb8a9ea83d2454b29eda2eb2"
    ["lib/datasets/kitti/mixup_geometry.py"]="85a48ddfd0a99a10c8c99c4bb9ef42bbbddf9c56c833c6d58e2c8d70c7983f85"
    ["lib/helpers/config_helper.py"]="c300e9512b7880b60276e38150ebd771b2ef55dc505c3c11bd340bfc28aa2d27"
    ["lib/helpers/decode_helper.py"]="7fe7382b93835a82dcba11d6385dedc171e6a0e8d4ee1bbe3a02d6c87423de51"
    ["lib/helpers/swanlab_helper.py"]="bf83effaab9ac8fdf35a994cfc6023ef06ea7dd10a278c51ca21f705ef3288c7"
    ["lib/helpers/tester_helper.py"]="01e8ba8db79cc396318535c2b6fa6a9c4d6a19c6021b84b67457f3b5ef65c702"
    ["lib/helpers/trainer_helper.py"]="c23a043c43e300d625a96c9f2176d462beee174aa067f44c4352bf322b505e46"
    ["lib/helpers/quality_ranking_monitor.py"]="3eeee37c9838cc21c705acc7b267f785caf0a412cf1061f6b3a9b3b34e9079c6"
    ["lib/losses/focal_loss.py"]="282b5a0efdbc29ef9e917cf71b52fc13b165981313c269daf0865ff12f1282bc"
    ["lib/losses/asymmetric_interval_depth_loss.py"]="1583f3d8b9a2f099a1e87fe1d01e28f4e1203cccaf087d92431b87c5855fd848"
    ["lib/losses/query_quality_ranking_loss.py"]="e3ed53d00daed77502680d627f0cf8487f9d456659465a258f26c35ab44ff327"
    ["lib/models/monodgp/iou3d_match_cost.py"]="91aaa430a14fd0ca343ae61f532a383dee5358659c2af27f9c271f892c533d52"
    ["lib/models/monodgp/matcher.py"]="26eba7a2dd15f39d228ce98ae79751dc9ce7eeab828bf41cf441852d3783e5ca"
    ["lib/models/monodgp/monodgp.py"]="c186c5afa863797cc4c1f812d247f53256b63c88fd8e97f3ec430e5d46b556b0"
    ["tools/train_val.py"]="ba274e75e8e124dd42c8a67f62fb5bc7cb3cadb5410022d40d309f92bffce9ea"
    ["tools/write_run_manifest.py"]="db1204f4c517999bb5f79fc8d29f7440d78480cee1f736c5380e309c4c9d429d"
    ["run_experiment_38.sh"]="92f183fcfe6a727b57cb30f9fe356e88c75f12bc32c8a14ae606666e0c5019dc"
)

mkdir -p "$orchestration_dir"
if [[ -e "$orchestration_log" || -e "$orchestration_status" ]]; then
    echo "Refusing to overwrite Experiment 38 orchestration receipts" >&2
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

process_is_original_exp37_wrapper() {
    [[ -r "/proc/$exp37_wrapper_pid/stat" ]] || return 1
    local current_start_ticks
    current_start_ticks="$(awk '{print $22}' "/proc/$exp37_wrapper_pid/stat")"
    [[ "$current_start_ticks" == "$exp37_wrapper_start_ticks" ]]
}

verify_exp38_source_snapshot() {
    local relative_path absolute_path actual_hash
    for relative_path in "${!expected_hashes[@]}"; do
        absolute_path="$repo_dir/$relative_path"
        [[ -f "$absolute_path" ]] || {
            printf 'Experiment 38 source missing: %s\n' "$absolute_path" >&2
            return 1
        }
        actual_hash="$(sha256sum "$absolute_path")"
        actual_hash="${actual_hash%% *}"
        if [[ "$actual_hash" != "${expected_hashes[$relative_path]}" ]]; then
            printf 'Experiment 38 source drift: %s expected=%s actual=%s\n' \
                "$relative_path" "${expected_hashes[$relative_path]}" \
                "$actual_hash" >&2
            return 1
        fi
    done
}

printf '[%s] Waiting for Experiment 37 wrapper pid=%s start_ticks=%s\n' \
    "$(date --iso-8601=seconds)" "$exp37_wrapper_pid" \
    "$exp37_wrapper_start_ticks"
while process_is_original_exp37_wrapper; do
    sleep 30
done

printf '[%s] Experiment 37 wrapper exited; checking completion receipt\n' \
    "$(date --iso-8601=seconds)"
for _ in {1..12}; do
    [[ -f "$exp37_status_file" ]] && break
    sleep 5
done
if [[ ! -f "$exp37_status_file" ]]; then
    echo "Experiment 37 status receipt is missing" >&2
    exit 20
fi
exp37_status=""
exp37_exit_code=""
while IFS='=' read -r key value; do
    case "$key" in
        status) exp37_status="$value" ;;
        exit_code) exp37_exit_code="$value" ;;
    esac
done < "$exp37_status_file"
if [[ "$exp37_status" != "completed" || "$exp37_exit_code" != "0" ]]; then
    printf 'Experiment 37 did not complete: status=%s exit_code=%s\n' \
        "$exp37_status" "$exp37_exit_code" >&2
    exit 21
fi
if [[ ! -f "$exp37_output_dir/checkpoint.pth" \
        || ! -f "$exp37_output_dir/checkpoint_best.pth" ]]; then
    echo "Experiment 37 completed but checkpoints are missing" >&2
    exit 22
fi
exp37_epoch="$($approved_python -c \
    "import torch; print(torch.load('$exp37_output_dir/checkpoint.pth', map_location='cpu', weights_only=False).get('epoch', -1))")"
if [[ "$exp37_epoch" != "250" ]]; then
    printf 'Experiment 37 completion checkpoint is not epoch 250: %s\n' \
        "$exp37_epoch" >&2
    exit 23
fi
verify_exp38_source_snapshot || exit 24
if [[ -e "$exp38_output_dir" ]]; then
    echo "Refusing to overwrite Experiment 38 output: $exp38_output_dir" >&2
    exit 25
fi

printf '[%s] Experiment 38 gates passed; starting formal run\n' \
    "$(date --iso-8601=seconds)"
cd "$repo_dir"
MONODGP_CUDA_VISIBLE_DEVICES=0 ./run_experiment_38.sh

finished=1
printf 'status=completed\nexit_code=0\nstarted_at=%s\nfinished_at=%s\n' \
    "$started_at" "$(date --iso-8601=seconds)" \
    > "$orchestration_status"
trap - EXIT
printf '[%s] Experiment 38 completed successfully\n' \
    "$(date --iso-8601=seconds)"
