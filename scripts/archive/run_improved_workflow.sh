#!/bin/bash
# Comprehensive workflow script for improved leadership embedding analysis

set -e  # Exit on error

echo "====================================================================="
echo "         IMPROVED LEADERSHIP EMBEDDING ANALYSIS WORKFLOW"
echo "====================================================================="

# Create directories if they don't exist
mkdir -p data/processed
mkdir -p models
mkdir -p data/visualizations/model_comparison

# Step 1: Generate improved comprehensive anchor-positive pairs
echo -e "\n[1/5] Generating comprehensive pairs dataset..."
python scripts/build_ipip_pairs_improved.py

# Step 2: Train model with MultipleNegativesRankingLoss
echo -e "\n[2/5] Training model with MultipleNegativesRankingLoss..."
python scripts/train_ipip_mnrl.py

# Step 3: Train model with original GISTEmbedLoss for comparison
# This is optional - uncomment if you want to retrain with original method
echo -e "\n[3/5] Training model with GISTEmbedLoss (optional comparison)..."
# python scripts/train_gist_ipip.py

# Step 4: Compare model performance
echo -e "\n[4/5] Comparing model performance on IPIP and leadership data..."
python scripts/compare_model_performance.py

# Step 5: Apply best model to leadership data
echo -e "\n[5/5] Applying model to leadership data..."
# Automatically selects the best model based on comparison results
python scripts/apply_best_model_to_leadership.py

echo -e "\n====================================================================="
echo "                      WORKFLOW COMPLETE!"
echo "====================================================================="
echo "Results and visualizations are available in:"
echo "  - data/visualizations/model_comparison/"
echo "  - data/visualizations/leadership_clusters/"
echo "=====================================================================" 