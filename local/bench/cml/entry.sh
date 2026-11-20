#!/usr/bin/env bash
# CloudML custom_train entry for the Automodel#4122 DDP vs FSDP2 grid.
# Env: RUN_NAME (results go to $AM_WS/bench/runs/$RUN_NAME), NPROC, STEPS, WARMUP.
set -euo pipefail

AM_WS=/planning-users/yangyeyuan/automodel-4122
RUN_NAME=${RUN_NAME:-local-$(date +%m%d-%H%M%S)}
export RESULTS=$AM_WS/bench/runs/$RUN_NAME
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/entry.log") 2>&1

# shellcheck disable=SC1091
source "$AM_WS/env.sh"
{
    hostname
    nvidia-smi
    nvidia-smi topo -m
    lscpu | head -20
    free -g
} > "$RESULTS/sysinfo.txt" 2>&1 || true
env | grep -viE 'key|secret|token|passw|cred|^AK=' | sort > "$RESULTS/env.txt"
md5sum "$AM_WS"/bench/ddp_vs_fsdp2.py "$AM_WS"/bench/grid.sh > "$RESULTS/code.md5"

echo "[entry] host=$(hostname) run=$RUN_NAME gpus=$(nvidia-smi -L | wc -l) start=$(date -Is)"
bash "$AM_WS/bench/grid.sh"
echo "[entry] done $(date -Is)"
