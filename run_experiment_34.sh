#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
approved_python="$repo_dir/.venv-cu129/bin/python"
config_path="configs/monodgp_exp34.yaml"
cuda_devices="${MONODGP_CUDA_VISIBLE_DEVICES:-0}"

cd "$repo_dir"

model_name="$($approved_python -c \
    "from lib.helpers.config_helper import load_config; print(load_config('$config_path')['model_name'])")"
output_dir="$repo_dir/outputs/$model_name"
if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite an existing experiment: $output_dir" >&2
    exit 1
fi

command_text="CUDA_VISIBLE_DEVICES=$cuda_devices PYTHONUNBUFFERED=1 $approved_python tools/train_val.py --config $config_path"
CUDA_VISIBLE_DEVICES="$cuda_devices" "$approved_python" \
    tools/write_run_manifest.py \
    --config "$config_path" \
    --command "$command_text"

status_file="$output_dir/experiment_exit_status.txt"
started_at="$(date --iso-8601=seconds)"
if CUDA_VISIBLE_DEVICES="$cuda_devices" PYTHONUNBUFFERED=1 \
    "$approved_python" tools/train_val.py --config "$config_path"; then
    printf 'status=completed\nexit_code=0\nstarted_at=%s\nfinished_at=%s\n' \
        "$started_at" "$(date --iso-8601=seconds)" > "$status_file"
else
    exit_code=$?
    printf 'status=failed\nexit_code=%s\nstarted_at=%s\nfinished_at=%s\n' \
        "$exit_code" "$started_at" "$(date --iso-8601=seconds)" > "$status_file"
    exit "$exit_code"
fi
