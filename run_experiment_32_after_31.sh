#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
approved_python="$repo_dir/.venv-cu129/bin/python"
exp31_wrapper_pid="3676520"
exp31_wrapper_start_ticks="209306868"
exp31_status_file="$repo_dir/outputs/V2-0031_相机归一化跨焦距MixUp/experiment_exit_status.txt"
exp31_latest_checkpoint="$repo_dir/outputs/V2-0031_相机归一化跨焦距MixUp/checkpoint.pth"
exp31_best_checkpoint="$repo_dir/outputs/V2-0031_相机归一化跨焦距MixUp/checkpoint_best.pth"
exp32_output_dir="$repo_dir/outputs/V2-0032_全query同GT三维IoU质量排序"
orchestration_dir="$repo_dir/outputs/_orchestration"
orchestration_log="$orchestration_dir/exp32_after_exp31_20260818.log"
orchestration_status="$orchestration_dir/exp32_after_exp31_20260818.status"

declare -A expected_hashes=(
    ["configs/monodgp.yaml"]="7876048f164939345116f218e49da1ddf7c41183a31dfbd67abe444ea0f454ea"
    ["configs/monodgp_exp29.yaml"]="6eaf26c54c722c37a980ef057a88e14d075e9a7b9d2c486683326983aa031a3b"
    ["configs/monodgp_exp30.yaml"]="8b9d9fa38f34be9d37b672359b67c633489ce065ef119d9d0ce6558a9ab11bba"
    ["configs/monodgp_exp31.yaml"]="847fbb3ca1ffd5879b0753852dc4dbacfb06b4e469ae9f7d0049135db801aaba"
    ["configs/monodgp_exp32.yaml"]="bbb744c55aa80a4e26b4cdb2d61aad469d8ff9c98dbd1e9a2133d7406468fbce"
    ["lib/datasets/kitti/kitti_dataset.py"]="9985a694273dc4081d02f728064f29e2c7b1ee30fec890f8d10c9261b629fbf7"
    ["lib/datasets/kitti/kitti_utils.py"]="54216ccf4586c56f47b84971d13f6c26b81c90dc977e222a71cd4aa9fa6511e6"
    ["lib/datasets/kitti/mixup_geometry.py"]="29bc77c52eb4eaa9be87c38e13d1df2ddb16dee8a5ab2300d3f8d7cd611485cb"
    ["lib/helpers/config_helper.py"]="c300e9512b7880b60276e38150ebd771b2ef55dc505c3c11bd340bfc28aa2d27"
    ["lib/helpers/decode_helper.py"]="7fe7382b93835a82dcba11d6385dedc171e6a0e8d4ee1bbe3a02d6c87423de51"
    ["lib/helpers/tester_helper.py"]="8afbb0c3efaf0b1b3159c67a5ab6537e955980d94bf6b5b8d3388f7bcfabe08e"
    ["lib/helpers/trainer_helper.py"]="ad214eb430795c4bef37332c2b5c363cc465a8aa217c687759ef7290629cec1d"
    ["lib/helpers/quality_ranking_monitor.py"]="575cef7fad0b160953c33a884cfdaea76c886becce784794c5e79143d93b66b2"
    ["lib/helpers/swanlab_helper.py"]="e82f787a926eb56457d79f6cf6a34cf7d5ad4fcf3cc57c1fbe4c1546b5ed3edc"
    ["lib/helpers/gradient_monitor.py"]="fe0ef7df00280d9c9fd5ff2f495a99a046b7c7242df2eb159c81d00353fb3cb1"
    ["lib/losses/asymmetric_interval_depth_loss.py"]="1583f3d8b9a2f099a1e87fe1d01e28f4e1203cccaf087d92431b87c5855fd848"
    ["lib/losses/query_quality_ranking_loss.py"]="e3ed53d00daed77502680d627f0cf8487f9d456659465a258f26c35ab44ff327"
    ["lib/models/monodgp/monodgp.py"]="d718621897191ce6575fdc09d759038d4c6a82d46476a2f3651ff9478d66f595"
    ["lib/models/monodgp/matcher.py"]="26eba7a2dd15f39d228ce98ae79751dc9ce7eeab828bf41cf441852d3783e5ca"
    ["lib/models/monodgp/iou3d_match_cost.py"]="91aaa430a14fd0ca343ae61f532a383dee5358659c2af27f9c271f892c533d52"
    ["tools/write_run_manifest.py"]="860ce6261105821a5233cd14a2712c52da9ab99fd791e6db244ea9571fa4d303"
    ["run_experiment_32.sh"]="e1ef92d77e526e752305a152f8c1092ae9ea3923a6d4fff4c6b496a0c9ec029e"
)

mkdir -p "$orchestration_dir"
if [[ -e "$orchestration_log" || -e "$orchestration_status" ]]; then
    echo "Refusing to overwrite existing Experiment 32 orchestration receipts" >&2
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

process_is_original_exp31_wrapper() {
    [[ -r "/proc/$exp31_wrapper_pid/stat" ]] || return 1
    current_start_ticks="$(awk '{print $22}' "/proc/$exp31_wrapper_pid/stat")"
    [[ "$current_start_ticks" == "$exp31_wrapper_start_ticks" ]]
}

verify_exp32_source_snapshot() {
    local relative_path absolute_path actual_hash
    for relative_path in "${!expected_hashes[@]}"; do
        absolute_path="$repo_dir/$relative_path"
        if [[ ! -f "$absolute_path" ]]; then
            printf 'Experiment 32 source file is missing: %s\n' \
                "$absolute_path" >&2
            return 1
        fi
        actual_hash="$(sha256sum "$absolute_path")"
        actual_hash="${actual_hash%% *}"
        if [[ "$actual_hash" != "${expected_hashes[$relative_path]}" ]]; then
            printf 'Experiment 32 source drift: %s expected=%s actual=%s\n' \
                "$relative_path" "${expected_hashes[$relative_path]}" \
                "$actual_hash" >&2
            return 1
        fi
    done
}

printf '[%s] Waiting for Experiment 31 wrapper pid=%s start_ticks=%s\n' \
    "$(date --iso-8601=seconds)" "$exp31_wrapper_pid" \
    "$exp31_wrapper_start_ticks"

while process_is_original_exp31_wrapper; do
    sleep 30
done

printf '[%s] Experiment 31 wrapper exited; checking completion receipt\n' \
    "$(date --iso-8601=seconds)"

for _ in {1..12}; do
    [[ -f "$exp31_status_file" ]] && break
    sleep 5
done

if [[ ! -f "$exp31_status_file" ]]; then
    printf 'Experiment 31 status receipt is missing: %s\n' \
        "$exp31_status_file" >&2
    exit 20
fi

exp31_status=""
exp31_exit_code=""
while IFS='=' read -r key value; do
    case "$key" in
        status) exp31_status="$value" ;;
        exit_code) exp31_exit_code="$value" ;;
    esac
done < "$exp31_status_file"

if [[ "$exp31_status" != "completed" || "$exp31_exit_code" != "0" ]]; then
    printf 'Experiment 31 did not complete successfully: status=%s exit_code=%s\n' \
        "$exp31_status" "$exp31_exit_code" >&2
    exit 21
fi

if [[ ! -f "$exp31_latest_checkpoint" || ! -f "$exp31_best_checkpoint" ]]; then
    printf 'Experiment 31 completion receipt exists but checkpoints are missing\n' >&2
    exit 22
fi

if ! verify_exp32_source_snapshot; then
    exit 23
fi

if [[ -e "$exp32_output_dir" ]]; then
    printf 'Refusing to overwrite Experiment 32 output: %s\n' \
        "$exp32_output_dir" >&2
    exit 24
fi

printf '[%s] Experiment 31 completed successfully; starting Experiment 32\n' \
    "$(date --iso-8601=seconds)"

cd "$repo_dir"
MONODGP_CUDA_VISIBLE_DEVICES=0 ./run_experiment_32.sh

finished=1
printf 'status=completed\nexit_code=0\nstarted_at=%s\nfinished_at=%s\n' \
    "$started_at" "$(date --iso-8601=seconds)" \
    > "$orchestration_status"
trap - EXIT
printf '[%s] Experiment 32 completed successfully\n' \
    "$(date --iso-8601=seconds)"
