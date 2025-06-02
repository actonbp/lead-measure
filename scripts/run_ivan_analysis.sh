#!/bin/bash
# Ivan's enhanced embedding analysis workflow
# Implements randomized pairs, TSDAE pre-training, and enhanced GIST training

set -e  # Exit on error

echo "====================================================================="
echo "         IVAN'S ENHANCED EMBEDDING ANALYSIS WORKFLOW"
echo "====================================================================="
echo "This workflow implements Ivan Hernandez's improvements (January 2025):"
echo "- Randomized pair generation to avoid ordering bias"
echo "- TSDAE pre-training for domain adaptation"
echo "- BGE-M3 model optimized for clustering"
echo "- Larger batch sizes (96) for better GIST performance"
echo "- t-SNE visualization and statistical similarity analysis"
echo "====================================================================="

# Create directories if they don't exist
mkdir -p data/processed
mkdir -p models
mkdir -p data/visualizations/ivan_analysis

# Step 1: Generate randomized pairs
echo -e "\n[1/4] Generating randomized pairs dataset..."
python scripts/ivan_analysis/build_pairs_randomized.py

# Step 2: Train with TSDAE pre-training and GIST loss
echo -e "\n[2/4] Training with TSDAE pre-training and GIST loss..."
python scripts/ivan_analysis/train_with_tsdae.py

# Step 3: Visualize and analyze IPIP results
echo -e "\n[3/4] Analyzing IPIP embeddings..."
python scripts/ivan_analysis/visualize_and_analyze.py \
    --model models/ivan_tsdae_gist_final \
    --dataset ipip

# Step 4: Compare with baseline model
echo -e "\n[4/4] Comparing with baseline model..."
python scripts/ivan_analysis/visualize_and_analyze.py \
    --model models/ivan_tsdae_gist_final \
    --compare-baseline

echo -e "\n====================================================================="
echo "                      WORKFLOW COMPLETE!"
echo "====================================================================="
echo "Results and visualizations are available in:"
echo "  - data/visualizations/ivan_analysis/"
echo "  - models/ivan_tsdae_gist_final/ (trained model)"
echo "  - models/tsdae_pretrained/ (TSDAE pre-trained model)"
echo ""
echo "Key metrics are saved in:"
echo "  - data/visualizations/ivan_analysis/similarity_analysis.csv"
echo "  - data/visualizations/ivan_analysis/model_comparison.csv"
echo "====================================================================="