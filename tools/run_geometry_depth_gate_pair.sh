#!/usr/bin/env bash
# User-approved sequence (2026-09-11): 64 then 65, 250 epochs each.
set -uo pipefail
ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
if [[ -z "${TMUX:-}" ]]; then
    printf 'This sequence requires a named persistent tmux session.\n' >&2
    exit 88
fi
cd "$ROOT" || exit 90
for number in 64 65; do
    if [[ "$number" == 64 ]]; then
        output="$ROOT/outputs/V2-0064_实验64_几何合格度深度减力"
    else
        output="$ROOT/outputs/V2-0065_实验65_深度梯度统一均值减力对照"
    fi
    MONODGP_FORMAL_CONFIG="$ROOT/configs/monodgp_exp${number}.yaml" \
    MONODGP_FORMAL_OUTPUT="$output" bash "$ROOT/tools/run_exp55.sh"
    result=$?
    if [[ "$result" != 0 ]]; then
        printf 'Experiment %s exited %s; sequence stops.\n' "$number" "$result" >&2
        exit "$result"
    fi
done
