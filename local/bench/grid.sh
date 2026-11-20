#!/bin/bash
# DDP vs FSDP2 grid for Automodel#4122 on NPROC GPUs of one node.
# Env: NPROC (default: all visible GPUs), RESULTS (output dir), STEPS, WARMUP.
# Needs the automodel-4122 env (source $AM_WS/env.sh).
set -u
export HF_HUB_OFFLINE=1
BENCH=$AM_WS/bench
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
RESULTS=${RESULTS:-$BENCH/results_${NPROC}gpu}
OUT=$RESULTS/grid.jsonl
FAIL=$RESULTS/grid.failures
STEPS=${STEPS:-30}
WARMUP=${WARMUP:-10}
mkdir -p "$RESULTS/logs" "$RESULTS/profiles"

run() {  # run <tag> <args...>; skips tags that already have a result in $OUT
    local tag=$1; shift
    if [[ -f $OUT ]] && grep -q "\"tag\": \"$tag\"" "$OUT"; then return; fi
    if ! uv run --project "$AM_ROOT" --locked --no-sync torchrun --standalone --nproc-per-node "$NPROC" \
        "$BENCH/ddp_vs_fsdp2.py" --warmup "$WARMUP" --steps "$STEPS" --tag "$tag" --out "$OUT" "$@" \
        > "$RESULTS/logs/$tag.log" 2>&1; then
        echo "$tag $*" >> "$FAIL"
        grep -m1 -E "OutOfMemoryError|Error:" "$RESULTS/logs/$tag.log" >> "$FAIL"
    fi
    grep -E '^\{' "$RESULTS/logs/$tag.log" | tail -1 | cut -c1-200
}

# "none" has no gradient sync across ranks: it is the compute-only lower bound, not a valid trainer.
VARIANTS=(
    "none|--mode none"
    "ddp|--mode ddp"
    "ddp_bv|--mode ddp --bucket-view 1"
    "fsdp2|--mode fsdp2"
    "fsdp2_raf|--mode fsdp2 --reshard-after-forward 1"
    "replicate|--mode replicate"
    "fsdp2_shard|--mode fsdp2_shard"
)
SHAPES=("512 1 1" "512 4 1" "2048 1 1" "4096 1 1" "512 1 4")

for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
    for prec in bf16 fp32_master; do
        for shape in "${SHAPES[@]}"; do
            read -r seq mbs ga <<< "$shape"
            if [[ $ga != 1 && $model != Qwen3-0.6B ]]; then continue; fi
            for v in "${VARIANTS[@]}"; do
                name=${v%%|*}; vargs=${v#*|}
                # shellcheck disable=SC2086
                run "${model}_${prec}_s${seq}_b${mbs}_ga${ga}_${name}" --model "$AM_WS/models/$model" \
                    --precision "$prec" --seq-len "$seq" --mbs "$mbs" --ga "$ga" $vargs
            done
        done
    done
done

# Repeat the row that was noisy on 1 GPU.
for v in "${VARIANTS[@]}"; do
    name=${v%%|*}; vargs=${v#*|}
    # shellcheck disable=SC2086
    run "rep2_Qwen3-1.7B_bf16_s4096_b1_ga1_${name}" --model "$AM_WS/models/Qwen3-1.7B" --precision bf16 \
        --seq-len 4096 --mbs 1 $vargs
done

# Profiles (rank 0) for the most CPU-bound shape and one compute-bound shape.
for shape in "Qwen3-0.6B 512 1" "Qwen3-4B 2048 1"; do
    read -r model seq mbs <<< "$shape"
    for v in "${VARIANTS[@]}"; do
        name=${v%%|*}; vargs=${v#*|}
        # shellcheck disable=SC2086
        run "prof_${model}_bf16_s${seq}_b${mbs}_${name}" --model "$AM_WS/models/$model" --precision bf16 \
            --seq-len "$seq" --mbs "$mbs" --steps 10 --profile-dir "$RESULTS/profiles" $vargs
    done
done
echo GRID_DONE
