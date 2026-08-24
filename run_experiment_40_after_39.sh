#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
approved_python="$repo_dir/.venv-cu129/bin/python"
exp39_wrapper_pid="2145993"
exp39_wrapper_start_ticks="5983324"
exp39_output_dir="$repo_dir/outputs/V2-0039_实验38_触发NMS候选对IoU排序"
exp39_status_file="$exp39_output_dir/experiment_exit_status.txt"
exp40_output_dir="$repo_dir/outputs/V2-0040_实验40_NMS排序Loss权重1"
orchestration_dir="$repo_dir/outputs/_orchestration"
orchestration_log="$orchestration_dir/exp40_after_exp39_20260823.log"
orchestration_status="$orchestration_dir/exp40_after_exp39_20260823.status"

declare -A expected_hashes=(
    ["configs/monodgp.yaml"]="7405b075bb789549025ed0144d415e66f98b418d493dfde6b90f54ba5267b0e1"
    ["configs/monodgp_exp29.yaml"]="6eaf26c54c722c37a980ef057a88e14d075e9a7b9d2c486683326983aa031a3b"
    ["configs/monodgp_exp30.yaml"]="8b9d9fa38f34be9d37b672359b67c633489ce065ef119d9d0ce6558a9ab11bba"
    ["configs/monodgp_exp33.yaml"]="70201e0b62ffac323a7f9176fb9d14636090c25133ed0f24ba7a1a0b4c1e5cb1"
    ["configs/monodgp_exp36.yaml"]="4202459349eaab244ab0469d1e8cc1917af1e781e0ed6b1a32eee4ad05e9ea5f"
    ["configs/monodgp_exp37.yaml"]="b8568043803e26c2081a77bdde648199afe6b11b3403f3db15d9d533b44126c1"
    ["configs/monodgp_exp38.yaml"]="0adccd6163d123aa67eefb879200ad61893a3751406f0650d26292b3eb967200"
    ["configs/monodgp_exp39.yaml"]="bae9d0e42156882740599448338545af080fb3d69daa2d90c3dd41fec9b6347d"
    ["configs/monodgp_exp40.yaml"]="c047d4e8c2d95d048db70bb904657ddd4bb36b9d35683158e4a1d83d6c72f51e"
    ["lib/datasets/kitti/kitti_dataset.py"]="854a91054a4cd282d9978556f1212342242ee963bb8a9ea83d2454b29eda2eb2"
    ["lib/datasets/kitti/mixup_geometry.py"]="85a48ddfd0a99a10c8c99c4bb9ef42bbbddf9c56c833c6d58e2c8d70c7983f85"
    ["lib/helpers/config_helper.py"]="c300e9512b7880b60276e38150ebd771b2ef55dc505c3c11bd340bfc28aa2d27"
    ["lib/helpers/decode_helper.py"]="7fe7382b93835a82dcba11d6385dedc171e6a0e8d4ee1bbe3a02d6c87423de51"
    ["lib/helpers/quality_ranking_monitor.py"]="3eeee37c9838cc21c705acc7b267f785caf0a412cf1061f6b3a9b3b34e9079c6"
    ["lib/helpers/swanlab_helper.py"]="713924c28c23027b34d6f2bbf42e19e29f8e6678f31fed65d5f6b5f018e3571d"
    ["lib/helpers/tester_helper.py"]="003cc819d1b670deb00e2fc3966da6ddc0c40891cee21b68fc594d4bc6fc703f"
    ["lib/helpers/trainer_helper.py"]="9b0aa1d6a2fb14683dd35a665acc0cdbb18051265c46bfd9c48fa82e069774b8"
    ["lib/losses/focal_loss.py"]="282b5a0efdbc29ef9e917cf71b52fc13b165981313c269daf0865ff12f1282bc"
    ["lib/losses/nms_aware_iou_ranking_loss.py"]="6ae32605b234bedb789329191bd19b5dd46969aee3db67781acf969468098603"
    ["lib/models/monodgp/iou3d_match_cost.py"]="91aaa430a14fd0ca343ae61f532a383dee5358659c2af27f9c271f892c533d52"
    ["lib/models/monodgp/matcher.py"]="26eba7a2dd15f39d228ce98ae79751dc9ce7eeab828bf41cf441852d3783e5ca"
    ["lib/models/monodgp/monodgp.py"]="61d5e20ce170d9ae2769065745fed0ec5f8c82cb27aacb97597f25de68db6c55"
    ["tools/train_val.py"]="ba274e75e8e124dd42c8a67f62fb5bc7cb3cadb5410022d40d309f92bffce9ea"
    ["tools/write_run_manifest.py"]="75c3a8d336fad01b4a1b7951984d1bd24b8b4d7ae67179bd90136b6da798730b"
    ["run_experiment_40.sh"]="f093bd18b46b5ea9a240ef392238eb3ae249761ca16707ed027b6fc5302937e3"
)

mkdir -p "$orchestration_dir"
if [[ -e "$orchestration_log" || -e "$orchestration_status" ]]; then
    echo "Refusing to overwrite Experiment 40 orchestration receipts" >&2
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

process_is_original_exp39_wrapper() {
    [[ -r "/proc/$exp39_wrapper_pid/stat" ]] || return 1
    local current_start_ticks
    current_start_ticks="$(awk '{print $22}' "/proc/$exp39_wrapper_pid/stat")"
    [[ "$current_start_ticks" == "$exp39_wrapper_start_ticks" ]]
}

verify_exp40_source_snapshot() {
    local relative_path absolute_path actual_hash
    for relative_path in "${!expected_hashes[@]}"; do
        absolute_path="$repo_dir/$relative_path"
        [[ -f "$absolute_path" ]] || {
            printf 'Experiment 40 source missing: %s\n' "$absolute_path" >&2
            return 1
        }
        actual_hash="$(sha256sum "$absolute_path")"
        actual_hash="${actual_hash%% *}"
        if [[ "$actual_hash" != "${expected_hashes[$relative_path]}" ]]; then
            printf 'Experiment 40 source drift: %s expected=%s actual=%s\n' \
                "$relative_path" "${expected_hashes[$relative_path]}" \
                "$actual_hash" >&2
            return 1
        fi
    done
}

verify_single_variable_delta() {
    cd "$repo_dir"
    "$approved_python" -c "from copy import deepcopy
from lib.helpers.config_helper import load_config
a=deepcopy(load_config('configs/monodgp_exp39.yaml'))
b=deepcopy(load_config('configs/monodgp_exp40.yaml'))
assert a['model']['iou_classification']['nms_ranking']['loss_coef'] == 0.1
assert b['model']['iou_classification']['nms_ranking']['loss_coef'] == 1.0
a['model']['iou_classification']['nms_ranking']['loss_coef'] = 1.0
a['model_name'] = b['model_name']
a['trainer']['swanlab'] = b['trainer']['swanlab']
assert a == b, 'Experiment 40 has an unexpected effective-config delta'"
}

printf '[%s] Waiting for Experiment 39 wrapper pid=%s start_ticks=%s\n' \
    "$(date --iso-8601=seconds)" "$exp39_wrapper_pid" \
    "$exp39_wrapper_start_ticks"
while process_is_original_exp39_wrapper; do
    sleep 30
done

printf '[%s] Experiment 39 wrapper exited; checking completion receipt\n' \
    "$(date --iso-8601=seconds)"
for _ in {1..12}; do
    [[ -f "$exp39_status_file" ]] && break
    sleep 5
done
if [[ ! -f "$exp39_status_file" ]]; then
    echo "Experiment 39 status receipt is missing" >&2
    exit 20
fi
exp39_status=""
exp39_exit_code=""
while IFS='=' read -r key value; do
    case "$key" in
        status) exp39_status="$value" ;;
        exit_code) exp39_exit_code="$value" ;;
    esac
done < "$exp39_status_file"
if [[ "$exp39_status" != "completed" || "$exp39_exit_code" != "0" ]]; then
    printf 'Experiment 39 did not complete: status=%s exit_code=%s\n' \
        "$exp39_status" "$exp39_exit_code" >&2
    exit 21
fi
if [[ ! -f "$exp39_output_dir/checkpoint.pth" \
        || ! -f "$exp39_output_dir/checkpoint_best.pth" \
        || ! -f "$exp39_output_dir/checkpoint_best_bev_nms_0_80.pth" ]]; then
    echo "Experiment 39 completed but expected checkpoints are missing" >&2
    exit 22
fi
exp39_epoch="$($approved_python -c \
    "import torch; print(torch.load('$exp39_output_dir/checkpoint.pth', map_location='cpu', weights_only=False).get('epoch', -1))")"
if [[ "$exp39_epoch" != "250" ]]; then
    printf 'Experiment 39 completion checkpoint is not epoch 250: %s\n' \
        "$exp39_epoch" >&2
    exit 23
fi
verify_exp40_source_snapshot || exit 24
verify_single_variable_delta || exit 25
if [[ -e "$exp40_output_dir" ]]; then
    echo "Refusing to overwrite Experiment 40 output: $exp40_output_dir" >&2
    exit 26
fi

printf '[%s] Experiment 40 gates passed; starting formal run\n' \
    "$(date --iso-8601=seconds)"
cd "$repo_dir"
MONODGP_CUDA_VISIBLE_DEVICES=0 ./run_experiment_40.sh

finished=1
printf 'status=completed\nexit_code=0\nstarted_at=%s\nfinished_at=%s\n' \
    "$started_at" "$(date --iso-8601=seconds)" \
    > "$orchestration_status"
trap - EXIT
printf '[%s] Experiment 40 completed successfully\n' \
    "$(date --iso-8601=seconds)"
