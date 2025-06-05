#!/bin/bash
# Refined leadership embedding analysis workflow script
# with bge-m3 model and TSDAE pre-training

set -e  # Exit on error

echo "====================================================================="
echo "        REFINED LEADERSHIP EMBEDDING ANALYSIS WORKFLOW"
echo "====================================================================="

# Create directories if they don't exist
mkdir -p data/processed
mkdir -p models
mkdir -p data/visualizations/model_comparison
mkdir -p data/visualizations/model_evaluations

# Step 1: Generate improved pairs dataset with randomized positions
echo -e "\n[1/6] Generating comprehensive pairs dataset with randomized positions..."
python scripts/build_ipip_pairs_improved.py

# Step 2: Train with bge-m3 model, TSDAE pre-training, and GIST loss
echo -e "\n[2/6] Training with bge-m3 model, TSDAE pre-training, and GIST loss..."
python scripts/train_ipip_mnrl.py \
  --base_model "BAAI/bge-m3" \
  --loss_fn gist \
  --guide_model "all-MiniLM-L6-v2" \
  --pooling_mode cls \
  --batch_size 96 \
  --epochs 10 \
  --num_phases 5 \
  --use_fp16 \
  --tsdae_pretrain \
  --tsdae_epochs 1 \
  --tsdae_batch_size 16

# Step 3: Evaluate models with validation metrics
echo -e "\n[3/6] Evaluating model with comprehensive validation metrics..."
# Find the latest model
LATEST_MODEL=$(ls -td models/ipip_gist_* | head -1)
python scripts/evaluate_model_with_validation.py --model "$LATEST_MODEL" --dataset ipip

# Step 4: Apply model to leadership constructs
echo -e "\n[4/6] Applying model to leadership constructs..."
python scripts/evaluate_mnrl_on_leadership.py --model_path "$LATEST_MODEL"

# Step 5: Analyze embedding similarities
echo -e "\n[5/6] Analyzing embedding similarities within and across constructs..."
python scripts/evaluate_model_with_validation.py --model "$LATEST_MODEL" --dataset leadership

# Step 6: Compare embedding similarities
echo -e "\n[6/6] Comparing same-construct vs different-construct similarities..."
if [ -f "scripts/evaluate_model_with_validation.py" ]; then
  python scripts/evaluate_model_with_validation.py --model "$LATEST_MODEL" --dataset ipip
  python scripts/evaluate_model_with_validation.py --model "$LATEST_MODEL" --dataset leadership
fi

echo -e "\n====================================================================="
echo "                      WORKFLOW COMPLETE!"
echo "====================================================================="
echo "Results and visualizations are available in:"
echo "  - data/visualizations/model_comparison/"
echo "  - data/visualizations/model_evaluations/"
echo "  - data/processed/ipip_construct_statistics.csv"
echo "====================================================================="