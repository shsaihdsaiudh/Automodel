#!/bin/bash
# Single-GPU grid for Automodel#4122. Run from the dev machine with the automodel-4122 env loaded.
set -u
export HF_HUB_OFFLINE=1
cd "$AM_WS/bench"
OUT=${OUT:-grid_1gpu.jsonl}
FAIL=${FAIL:-grid_1gpu.failures}
STEPS=${STEPS:-30}
WARMUP=${WARMUP:-10}

run() {  # run <tag> <args...>; skips tags that already have a result in $OUT
    local tag=$1; shift
    if [[ -f $OUT ]] && grep -q "\"tag\": \"$tag\"" "$OUT"; then return; fi
    # --standalone picks a free rendezvous port; a fixed 29500 collided between back-to-back runs.
    if ! uv run --project "$AM_ROOT" --locked --no-sync torchrun --standalone --nproc-per-node 1 ddp_vs_fsdp2.py \
        --warmup "$WARMUP" --steps "$STEPS" --tag "$tag" --out "$OUT" "$@" > "logs/$tag.log" 2>&1; then
        echo "$tag $*" >> "$FAIL"
        tail -3 "logs/$tag.log" >> "$FAIL"
    fi
    grep -E '^\{' "logs/$tag.log" | tail -1
}

VARIANTS=(
    "none|--mode none"
    "ddp|--mode ddp"
    "ddp_bv|--mode ddp --bucket-view 1"
    "fsdp2|--mode fsdp2"
    "fsdp2_raf|--mode fsdp2 --reshard-after-forward 1"
    "replicate|--mode replicate"
)
SHAPES=("512 1 1" "512 4 1" "2048 1 1" "4096 1 1" "512 1 4")

mkdir -p logs
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
    for prec in bf16 fp32_master; do
        for shape in "${SHAPES[@]}"; do
            read -r seq mbs ga <<< "$shape"
            if [[ $ga != 1 && $model != Qwen3-0.6B ]]; then continue; fi
            for v in "${VARIANTS[@]}"; do
                name=${v%%|*}; vargs=${v#*|}
                tag="${model}_${prec}_s${seq}_b${mbs}_ga${ga}_${name}"
                # shellcheck disable=SC2086
                run "$tag" --model "$AM_WS/models/$model" --precision "$prec" \
                    --seq-len "$seq" --mbs "$mbs" --ga "$ga" $vargs
            done
        done
    done
done

# Profiles for the most CPU-bound shape.
for v in "${VARIANTS[@]}"; do
    name=${v%%|*}; vargs=${v#*|}
    # shellcheck disable=SC2086
    run "prof_Qwen3-0.6B_bf16_s512_b1_${name}" --model "$AM_WS/models/Qwen3-0.6B" --precision bf16 \
        --seq-len 512 --mbs 1 --steps 10 --profile-dir profiles $vargs
done
echo GRID_DONE
