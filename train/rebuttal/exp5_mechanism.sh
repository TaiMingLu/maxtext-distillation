#!/bin/bash
# Exp 5: Mechanism study — comprehensive logit analysis
# Runs logit_analysis.py once per model to avoid JAX re-init issues.
# Each run saves to disk; subsequent runs skip already-computed models.
# Run on v6e-8.

set -eo pipefail
cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
export PYTHONPATH="$(pwd):$(pwd)/lm-evaluation-harness:${PYTHONPATH:-}"

OUTPUT_DIR="/home/terry/gcs-bucket/rebuttal/exp5_mechanism"
TOKENIZER="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B"

echo "========================================"
echo "Exp 5: Mechanism Study — Logit Analysis"
echo "========================================"

# Run analysis for each model one at a time
# The script skips models that already have saved .npz files
for MODEL in \
    baseline_1b \
    teacher_05b teacher_1b teacher_3b teacher_8b \
    distilled_A05B_a02 distilled_A1B_a04 distilled_A3B_a06 distilled_A8B_a06 \
    distilled_A05B_a1 distilled_A1B_a1 distilled_A3B_a1 distilled_A8B_a1; do

    echo ""
    echo "--- Processing: $MODEL ---"

    python3.10 -u rebuttal/logit_analysis.py \
        --output_dir "$OUTPUT_DIR" \
        --data_path unused \
        --tokenizer_path "$TOKENIZER" \
        --num_sequences 200 \
        --seq_length 2048 \
        --batch_size 4 \
        --top_k 100 \
        --models "$MODEL" \
        --skip_pairwise || {
        echo "WARNING: $MODEL failed, continuing..."
        continue
    }
done

# Final run: compute pairwise stats and summary from saved data
echo ""
echo "--- Computing pairwise stats and summary ---"
python3.10 -u rebuttal/logit_analysis.py \
    --output_dir "$OUTPUT_DIR" \
    --data_path unused \
    --tokenizer_path "$TOKENIZER" \
    --num_sequences 200 \
    --seq_length 2048 \
    --batch_size 4 \
    --top_k 100 \
    --models all

echo "========================================"
echo "Exp 5 complete."
echo "========================================"
