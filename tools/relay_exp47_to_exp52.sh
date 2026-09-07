#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
EXP47_SESSION=monodgp_exp46_to_exp47_relay_20260827
EXP47_STATUS="$ROOT/outputs/V2-0047_实验47_去除三维IoU质量头/status.tsv"
RELAY_DIR="$ROOT/outputs/exp47_to_exp52_relay"
RELAY_STATUS="$RELAY_DIR/status.tsv"

if [[ -z "${TMUX:-}" ]]; then
    printf 'The Exp47-to-Exp52 relay must run inside tmux.\n' >&2
    exit 88
fi

if ! mkdir "$RELAY_DIR"; then
    printf 'Refusing duplicate relay; state directory already exists: %s\n' \
        "$RELAY_DIR" >&2
    exit 94
fi
printf 'waiting_for_exp47\t%s\n' "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS" || exit 96

while [[ ! -f "$EXP47_STATUS" ]]; do
    if ! tmux has-session -t "$EXP47_SESSION" 2>/dev/null; then
        printf 'blocked_exp47_session_missing_without_status\t%s\n' \
            "$(date --iso-8601=seconds)" >> "$RELAY_STATUS" || exit 96
        exit 92
    fi
    sleep 30
done

IFS=$'\t' read -r exp47_label exp47_code < "$EXP47_STATUS"
printf 'exp47_receipt\t%s\t%s\t%s\n' \
    "$exp47_label" "$exp47_code" "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS" || exit 96
if [[ "$exp47_label" != train_exit || "$exp47_code" != 0 ]]; then
    printf 'blocked_exp47_not_successful\t%s\n' \
        "$(date --iso-8601=seconds)" >> "$RELAY_STATUS" || exit 96
    exit 93
fi

for exp_number in 48 49 50 51 52; do
    printf 'exp%s_starting\t%s\n' \
        "$exp_number" "$(date --iso-8601=seconds)" \
        >> "$RELAY_STATUS" || exit 96
    bash "$ROOT/tools/run_exp${exp_number}.sh"
    exp_code=$?
    printf 'exp%s_exit\t%d\t%s\n' \
        "$exp_number" "$exp_code" "$(date --iso-8601=seconds)" \
        >> "$RELAY_STATUS" || exit 96
    if [[ "$exp_code" != 0 ]]; then
        printf 'blocked_after_exp%s_failure\t%s\n' \
            "$exp_number" "$(date --iso-8601=seconds)" \
            >> "$RELAY_STATUS" || exit 96
        exit "$exp_code"
    fi
done

printf 'relay_complete\t%s\n' "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS" || exit 96
