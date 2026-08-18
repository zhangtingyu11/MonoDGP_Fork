#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
exp30_wrapper_pid="3461642"
exp30_wrapper_start_ticks="200403255"
exp30_status_file="$repo_dir/outputs/V2-0030_实验29完整P2几何修正/experiment_exit_status.txt"
exp31_output_dir="$repo_dir/outputs/V2-0031_相机归一化跨焦距MixUp"
orchestration_dir="$repo_dir/outputs/_orchestration"
orchestration_log="$orchestration_dir/exp31_after_exp30_20260817.log"
orchestration_status="$orchestration_dir/exp31_after_exp30_20260817.status"

declare -A expected_hashes=(
    ["configs/monodgp.yaml"]="7876048f164939345116f218e49da1ddf7c41183a31dfbd67abe444ea0f454ea"
    ["configs/monodgp_exp29.yaml"]="6eaf26c54c722c37a980ef057a88e14d075e9a7b9d2c486683326983aa031a3b"
    ["configs/monodgp_exp30.yaml"]="8b9d9fa38f34be9d37b672359b67c633489ce065ef119d9d0ce6558a9ab11bba"
    ["configs/monodgp_exp31.yaml"]="847fbb3ca1ffd5879b0753852dc4dbacfb06b4e469ae9f7d0049135db801aaba"
    ["lib/datasets/kitti/kitti_dataset.py"]="9985a694273dc4081d02f728064f29e2c7b1ee30fec890f8d10c9261b629fbf7"
    ["lib/datasets/kitti/kitti_utils.py"]="54216ccf4586c56f47b84971d13f6c26b81c90dc977e222a71cd4aa9fa6511e6"
    ["lib/datasets/kitti/mixup_geometry.py"]="29bc77c52eb4eaa9be87c38e13d1df2ddb16dee8a5ab2300d3f8d7cd611485cb"
    ["lib/helpers/trainer_helper.py"]="f5a854a70bf315f8118d2e357199d4c40ae1508ac9d9634b2e342831b6923fd4"
    ["tools/write_run_manifest.py"]="66819ae3ed4c21fd8693a229425cebbc744c960324f5f5e9d6bcc4bbec114232"
    ["run_experiment_31.sh"]="efb367a77231cb95ab87b6beca4a93fc01592fb7260cba6b170a0fbcfe561fc7"
)

mkdir -p "$orchestration_dir"
exec >>"$orchestration_log" 2>&1

started_at="$(date --iso-8601=seconds)"
finished=0

write_failure_status() {
    exit_code=$?
    if [[ "$finished" -eq 0 ]]; then
        printf 'status=failed\nexit_code=%s\nstarted_at=%s\nfinished_at=%s\n' \
            "$exit_code" "$started_at" "$(date --iso-8601=seconds)" \
            > "$orchestration_status"
    fi
}
trap write_failure_status EXIT

process_is_original_exp30_wrapper() {
    [[ -r "/proc/$exp30_wrapper_pid/stat" ]] || return 1
    current_start_ticks="$(awk '{print $22}' "/proc/$exp30_wrapper_pid/stat")"
    [[ "$current_start_ticks" == "$exp30_wrapper_start_ticks" ]]
}

verify_exp31_source_snapshot() {
    for relative_path in "${!expected_hashes[@]}"; do
        absolute_path="$repo_dir/$relative_path"
        if [[ ! -f "$absolute_path" ]]; then
            printf 'Experiment 31 source file is missing: %s\n' \
                "$absolute_path" >&2
            return 1
        fi
        actual_hash="$(sha256sum "$absolute_path")"
        actual_hash="${actual_hash%% *}"
        if [[ "$actual_hash" != "${expected_hashes[$relative_path]}" ]]; then
            printf 'Experiment 31 source drift: %s expected=%s actual=%s\n' \
                "$relative_path" "${expected_hashes[$relative_path]}" \
                "$actual_hash" >&2
            return 1
        fi
    done
}

printf '[%s] Waiting for Experiment 30 wrapper pid=%s start_ticks=%s\n' \
    "$(date --iso-8601=seconds)" "$exp30_wrapper_pid" \
    "$exp30_wrapper_start_ticks"

while process_is_original_exp30_wrapper; do
    sleep 30
done

printf '[%s] Experiment 30 wrapper exited; checking receipt\n' \
    "$(date --iso-8601=seconds)"

for _ in {1..12}; do
    [[ -f "$exp30_status_file" ]] && break
    sleep 5
done

if [[ ! -f "$exp30_status_file" ]]; then
    printf 'Experiment 30 status receipt is missing: %s\n' \
        "$exp30_status_file" >&2
    exit 20
fi

exp30_status=""
exp30_exit_code=""
while IFS='=' read -r key value; do
    case "$key" in
        status) exp30_status="$value" ;;
        exit_code) exp30_exit_code="$value" ;;
    esac
done < "$exp30_status_file"

if [[ "$exp30_status" != "completed" || "$exp30_exit_code" != "0" ]]; then
    printf 'Experiment 30 did not complete successfully: status=%s exit_code=%s\n' \
        "$exp30_status" "$exp30_exit_code" >&2
    exit 21
fi

if ! verify_exp31_source_snapshot; then
    exit 23
fi

if [[ -e "$exp31_output_dir" ]]; then
    printf 'Refusing to overwrite Experiment 31 output: %s\n' \
        "$exp31_output_dir" >&2
    exit 22
fi

printf '[%s] Experiment 30 completed successfully; starting Experiment 31\n' \
    "$(date --iso-8601=seconds)"

cd "$repo_dir"
./run_experiment_31.sh

finished=1
printf 'status=completed\nexit_code=0\nstarted_at=%s\nfinished_at=%s\n' \
    "$started_at" "$(date --iso-8601=seconds)" \
    > "$orchestration_status"
trap - EXIT
printf '[%s] Experiment 31 completed successfully\n' \
    "$(date --iso-8601=seconds)"
