#!/bin/bash
# Exp 5: Mechanism study — comprehensive logit analysis
# Runs forward passes on all models and computes per-token logit statistics.
# Run on v6e-8.

set -euo pipefail
cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"

export PYTHONPATH="$(pwd):$(pwd)/lm-evaluation-harness:${PYTHONPATH:-}"

echo "========================================"
echo "Exp 5: Mechanism Study — Logit Analysis"
echo "========================================"

python3.10 -u rebuttal/logit_analysis.py \
    --output_dir /home/terry/gcs-bucket/rebuttal/exp5_mechanism \
    --data_path /home/terry/gcs-bucket/rebuttal/data/fineweb-edu \
    --tokenizer_path /home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B \
    --num_sequences 200 \
    --seq_length 2048 \
    --batch_size 4 \
    --top_k 100

echo "========================================"
echo "Exp 5 complete."
echo "========================================"
