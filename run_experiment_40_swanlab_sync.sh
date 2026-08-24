#!/usr/bin/env bash
set -u

repo_dir="/home/zhangtingyu/Project/Mono3D/MonoDGP"
output_dir="$repo_dir/outputs/V2-0040_实验40_NMS排序Loss权重1"
swanlab_bin="$repo_dir/.venv-cu129/bin/swanlab"
owner_session="monodgp_exp40_after_exp39_20260823"
sync_log="$output_dir/swanlab_cloud_sync.log"
cloud_run_id="e40r1nms"

while tmux has-session -t "$owner_session" 2>/dev/null; do
    run_dir=""
    if [[ -d "$output_dir/swanlog" ]]; then
        run_dir="$(find "$output_dir/swanlog" -mindepth 1 -maxdepth 1 \
            -type d -name 'run-*' -print -quit)"
    fi
    if [[ -n "$run_dir" ]]; then
        "$swanlab_bin" sync -i "$cloud_run_id" "$run_dir" \
            >> "$sync_log" 2>&1 || true
    fi
    sleep 60
done
