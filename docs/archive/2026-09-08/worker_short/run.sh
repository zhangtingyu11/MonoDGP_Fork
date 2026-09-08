#!/usr/bin/env bash
set -uo pipefail
cd /home/zhangtingyu/Project/Mono3D/MonoDGP || exit 90
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for trial in A B compare; do
  /home/zhangtingyu/Project/Mono3D/MonoDGP/.venv-cu129/bin/python /tmp/monodgp-worker-repro-OII7ks/check.py "$trial" > "/tmp/monodgp-worker-repro-OII7ks/$trial.log" 2>&1
  code=$?
  printf '%s\n' "$code" > "/tmp/monodgp-worker-repro-OII7ks/$trial.exitcode"
  if [[ "$code" != 0 ]]; then exit "$code"; fi
done
