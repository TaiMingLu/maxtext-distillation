#!/bin/bash
# Exp 5: Mechanism study — collect logits using test_orbax_eval.py (proven working code)
# Runs PPL evaluation on each model, which produces per-token log-likelihoods.
# Run on v6e-8.

cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
export PYTHONPATH="$(pwd):$(pwd)/lm-evaluation-harness:${PYTHONPATH:-}"

HF_TOKENIZER="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B"
EVAL_DIR="/home/terry/gcs-bucket/rebuttal/exp5_mechanism"
mkdir -p "$EVAL_DIR"

# Install lm-evaluation-harness once upfront
cd ~/maxtext/lm-evaluation-harness
pip install -e . -q 2>&1 | tail -3 || echo "WARNING: pip install had issues, continuing..."
cd ~/maxtext

echo "========================================"
echo "Exp 5: Mechanism Study"
echo "========================================"

# Use test_orbax_eval.py exactly as the existing eval pipeline does.
# This is the same code that works for all other evaluations.
run_eval() {
    local CKPT="$1"
    local MODEL="$2"
    local LABEL="$3"
    local SAVE_DIR="$EVAL_DIR/$LABEL"

    # Check if ALL 13 datasets are in the results file (not just file exists)
    if [ -f "$SAVE_DIR/exp5_${LABEL}.json" ]; then
        local n_datasets=$(python3.10 -c "import json; d=json.load(open('$SAVE_DIR/exp5_${LABEL}.json')); print(len(d.get('ppl',{})))" 2>/dev/null || echo 0)
        if [ "$n_datasets" -ge 13 ]; then
            echo "[$LABEL] All 13 datasets done, skipping."
            return 0
        fi
        echo "[$LABEL] Only $n_datasets/13 datasets, re-running..."
        rm -f "$SAVE_DIR/exp5_${LABEL}.json"
    fi

    echo ""
    echo "--- [$LABEL] model=$MODEL ---"

    # Force-release TPU devices held by stale JAX processes
    pkill -9 -f "python3.10" 2>/dev/null || true
    sleep 3
    # Release /dev/vfio device handles
    for vfio in /dev/vfio/*; do
        fuser -k "$vfio" 2>/dev/null || true
    done
    sleep 10

    cd ~/maxtext/lm-evaluation-harness

    python3.10 -u scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
        load_parameters_path="$CKPT" \
        run_name="exp5_${LABEL}" \
        model_name="$MODEL" \
        max_target_length=4096 \
        dtype=bfloat16 \
        scan_layers=true \
        attention=dot_product \
        --hf_model_path="$HF_TOKENIZER" \
        --eval_mode=ppl \
        --eval_save_dir="$SAVE_DIR" \
        --ppl_batch_size=8 \
        --ppl_seq_length=4096 || {
        echo "WARNING: [$LABEL] failed, continuing..."
        cd ~/maxtext
        return 0
    }
    cd ~/maxtext
}

# Baseline
run_eval \
    "/home/terry/gcs-bucket/rebuttal/baselines/llama3.1-1b-s43/24999/items" \
    "llama3.1-1b" \
    "baseline_1b"

# Teachers
run_eval \
    "/home/terry/gcs-bucket/rebuttal/teachers/llama05b-50B-s42/checkpoint_24999/0/items" \
    "llama3.1-05b" \
    "teacher_05b"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/teachers/llama1b-50B-s42/checkpoint_24999/0/items" \
    "llama3.1-1b" \
    "teacher_1b"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/teachers/llama3b-50B-s42/checkpoint_24999/0/items" \
    "llama3.1-3b" \
    "teacher_3b"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/teachers/llama8b-50B-s42/checkpoint_24999/0/items" \
    "llama3.1-8b" \
    "teacher_8b"

# Distilled students — best alpha
run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A05B-a02/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A05B_a02"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A1B-a04/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A1B_a04"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A3B-a06/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A3B_a06"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A8B-a06/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A8B_a06"

# Distilled students — pure KD (alpha=1.0)
run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A05B-a1/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A05B_a1"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A1B-a1/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A1B_a1"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A3B-a1/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A3B_a1"

run_eval \
    "/home/terry/gcs-bucket/rebuttal/distilled/A8B-a1/24999/items/0/items" \
    "llama3.1-1b" \
    "distilled_A8B_a1"

echo "========================================"
echo "Exp 5 complete. Results in $EVAL_DIR"
echo "========================================"
