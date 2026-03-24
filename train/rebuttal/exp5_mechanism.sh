#!/bin/bash
# Exp 5: Mechanism study — logit analysis on existing checkpoints
# Usage: bash train/rebuttal/exp5_mechanism.sh
#
# Runs inference on all models (baselines, teachers, distilled students)
# and computes logit statistics for the mechanism analysis.
# Run on a single TPU (v6e-8 or larger).

set -euo pipefail
cd ~/maxtext
source ~/maxtext_env/bin/activate

BUCKET_NAME="${BUCKET_NAME:?BUCKET_NAME not set}"
BUCKET="gs://$BUCKET_NAME"
OUTPUT_DIR="$BUCKET/rebuttal/exp5_mechanism"

# Eval data paths
FINEWEB_TEST="/home/terry/gcs-bucket/rebuttal/data/fineweb-edu"

echo "========================================"
echo "Exp 5: Mechanism Study — Logit Analysis"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# --- Helper function to run inference and collect logits ---
run_logit_analysis() {
  local CKPT_PATH="$1"
  local MODEL_NAME="$2"
  local RUN_LABEL="$3"

  echo ""
  echo "--- Analyzing: $RUN_LABEL (model=$MODEL_NAME) ---"

  python3.10 -u -m MaxText.decode MaxText/configs/base.yml \
    run_name="exp5_${RUN_LABEL}" \
    base_output_directory="$OUTPUT_DIR" \
    load_parameters_path="$CKPT_PATH" \
    model_name="$MODEL_NAME" \
    per_device_batch_size=4 \
    max_target_length=8192 \
    scan_layers=false \
    attention=dot_product
}

# --- Baselines ---
echo ""
echo "========== BASELINES =========="

run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/baselines/llama3.1-1b-s43/24999/items" \
  "llama3.1-1b" \
  "baseline_1b"

# --- Teachers (50B tokens) ---
echo ""
echo "========== TEACHERS =========="

run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/teachers/llama05b-50B-s42/checkpoint_24999/0/items" \
  "llama3.1-05b" \
  "teacher_05b"

run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/teachers/llama1b-50B-s42/checkpoint_24999/0/items" \
  "llama3.1-1b" \
  "teacher_1b"

run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/teachers/llama3b-50B-s42/checkpoint_24999/0/items" \
  "llama3.1-3b" \
  "teacher_3b"

run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/teachers/llama8b-50B-s42/checkpoint_24999/0/items" \
  "llama3.1-8b" \
  "teacher_8b"

# --- Distilled 1.7B students (best alpha per teacher) ---
echo ""
echo "========== DISTILLED STUDENTS =========="

# Weak teacher (0.7B), alpha=0.2
run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/distilled/A05B-a02/24999/0/items" \
  "llama3.1-1b" \
  "distilled_A05B_a02"

# Same teacher (1.7B), alpha=0.4
run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/distilled/A1B-a04/24999/0/items" \
  "llama3.1-1b" \
  "distilled_A1B_a04"

# Strong teacher (3.8B), alpha=0.6
run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/distilled/A3B-a06/24999/0/items" \
  "llama3.1-1b" \
  "distilled_A3B_a06"

# Strongest teacher (8.0B), alpha=0.6
run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/distilled/A8B-a06/24999/0/items" \
  "llama3.1-1b" \
  "distilled_A8B_a06"

# Also pure KD variants (alpha=1.0) to compare
run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/distilled/A05B-a1/24999/0/items" \
  "llama3.1-1b" \
  "distilled_A05B_a1"

run_logit_analysis \
  "/home/terry/gcs-bucket/rebuttal/distilled/A8B-a1/24999/0/items" \
  "llama3.1-1b" \
  "distilled_A8B_a1"

echo ""
echo "========================================"
echo "Exp 5 inference complete."
echo "TODO: Run logit_analysis.py to compute metrics from saved logits."
echo "========================================"
