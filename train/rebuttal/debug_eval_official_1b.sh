#!/bin/bash
cd ~/maxtext
source ~/maxtext_env/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/lm-evaluation-harness:${PYTHONPATH:-}"

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"

cd ~/maxtext/lm-evaluation-harness
pip install -e . -q 2>&1 | tail -1 || true

echo "=== Evaluating official Llama 3.2 1B checkpoint ==="
python3.10 -u scripts/test_orbax_eval.py ../MaxText/configs/base.yml \
    load_parameters_path="/home/terry/gcs-bucket/rebuttal/converted/llama3.2-1b-official/0/items" \
    run_name="debug_official_1b" \
    model_name="llama3.2-1b-official" \
    max_target_length=4096 \
    dtype=bfloat16 \
    scan_layers=true \
    attention=dot_product \
    --hf_model_path="/home/terry/gcs-bucket/rebuttal/hf_models/Llama-3.1-8B" \
    --eval_mode=ppl \
    --eval_save_dir="/home/terry/gcs-bucket/rebuttal/debug_official_1b" \
    --ppl_batch_size=4 \
    --ppl_seq_length=4096

echo "=== Done ==="
