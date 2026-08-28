#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
EXP46_SESSION=monodgp_exp46_baseline_rerun1_20260827
EXP46_STATUS="$ROOT/outputs/V2-0046_实验46_确定性优化新基线/status.tsv"
EXP47_OUTPUT="$ROOT/outputs/V2-0047_实验47_去除三维IoU质量头"
RELAY_DIR="$ROOT/outputs/exp46_to_exp47_relay"
RELAY_STATUS="$RELAY_DIR/status.tsv"

mkdir -p "$RELAY_DIR"
: > "$RELAY_STATUS"
printf 'waiting_for_exp46\t%s\n' "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS"

while [[ ! -f "$EXP46_STATUS" ]]; do
    if ! tmux has-session -t "$EXP46_SESSION" 2>/dev/null; then
        printf 'blocked_exp46_session_missing_without_status\t%s\n' \
            "$(date --iso-8601=seconds)" >> "$RELAY_STATUS"
        exit 92
    fi
    sleep 30
done

IFS=$'\t' read -r exp46_label exp46_code < "$EXP46_STATUS"
printf 'exp46_receipt\t%s\t%s\t%s\n' \
    "$exp46_label" "$exp46_code" "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS"
if [[ "$exp46_label" != train_exit || "$exp46_code" != 0 ]]; then
    printf 'blocked_exp46_not_successful\t%s\n' \
        "$(date --iso-8601=seconds)" >> "$RELAY_STATUS"
    exit 93
fi
if [[ -e "$EXP47_OUTPUT" ]]; then
    printf 'blocked_exp47_output_exists\t%s\n' \
        "$(date --iso-8601=seconds)" >> "$RELAY_STATUS"
    exit 94
fi

printf 'exp47_starting\t%s\n' "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS"
bash "$ROOT/tools/run_exp47.sh"
exp47_code=$?
printf 'exp47_exit\t%d\t%s\n' \
    "$exp47_code" "$(date --iso-8601=seconds)" >> "$RELAY_STATUS"
exit "$exp47_code"
