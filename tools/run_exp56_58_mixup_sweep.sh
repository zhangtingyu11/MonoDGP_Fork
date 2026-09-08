#!/usr/bin/env bash
set -u
set -o pipefail

ROOT=/home/zhangtingyu/Project/Mono3D/MonoDGP
RELAY_DIR="$ROOT/outputs/exp56_58_mixup_probability_sweep"
RELAY_STATUS="$RELAY_DIR/status.tsv"

if [[ -z "${TMUX:-}" ]]; then
    printf 'The Exp56-58 sweep must run inside tmux.\n' >&2
    exit 88
fi

if ! mkdir "$RELAY_DIR"; then
    printf 'Refusing duplicate sweep; state directory already exists: %s\n' \
        "$RELAY_DIR" >&2
    exit 94
fi

printf 'sweep_start\t%s\n' "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS" || exit 96

for exp_number in 56 57 58; do
    config="$ROOT/configs/monodgp_exp${exp_number}.yaml"
    case "$exp_number" in
        56) output="$ROOT/outputs/V2-0056_实验56_Exp47_MixUp概率0.2" ;;
        57) output="$ROOT/outputs/V2-0057_实验57_Exp47_MixUp概率0.1" ;;
        58) output="$ROOT/outputs/V2-0058_实验58_Exp47_关闭MixUp" ;;
    esac
    printf 'exp%s_starting\t%s\n' \
        "$exp_number" "$(date --iso-8601=seconds)" \
        >> "$RELAY_STATUS" || exit 96
    MONODGP_FORMAL_CONFIG="$config" \
    MONODGP_FORMAL_OUTPUT="$output" \
        bash "$ROOT/tools/run_exp55.sh"
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

printf 'sweep_complete\t%s\n' "$(date --iso-8601=seconds)" \
    >> "$RELAY_STATUS" || exit 96
