#!/bin/bash
# Run model training and evaluation with different loss functions
# This script tests different loss functions to find the best one for IPIP construct classification

set -e  # Exit immediately if any command fails

echo "===== Starting Loss Function Comparison Experiment ====="
echo "This script will train multiple models with different loss functions and compare their performance."

# Create experiment output directory
EXPERIMENT_DIR="experiment_results/loss_function_comparison_$(date +%Y%m%d_%H%M)"
mkdir -p "$EXPERIMENT_DIR"
echo "Results will be saved to: $EXPERIMENT_DIR"

# Save experiment metadata
cat > "$EXPERIMENT_DIR/experiment_info.txt" << EOF
Loss Function Comparison Experiment
==================================
Date: $(date)
Description: Training and evaluating models with different loss functions on IPIP data

Loss functions tested:
- Multiple Negatives Ranking Loss (MNRL)
- Triplet Loss
- Contrastive Loss
- Cosine Similarity Loss

Each model is trained with the same hyperparameters except for the loss function:
- Base model: all-mpnet-base-v2
- Batch size: 32
- Learning rate: 2e-5
- Epochs: 10
EOF

# Function to train and evaluate a model with a specific loss function
train_and_evaluate() {
    loss_fn=$1
    epochs=$2
    
    echo
    echo "===== Training model with loss function: $loss_fn, epochs: $epochs ====="
    
    # Train the model
    TRAIN_OUTPUT="$EXPERIMENT_DIR/${loss_fn}_${epochs}_train_output.txt"
    echo "Training with $loss_fn loss function for $epochs epochs..."
    python scripts/train_ipip_mnrl.py --loss_fn "$loss_fn" --epochs "$epochs" > "$TRAIN_OUTPUT" 2>&1
    
    # Extract model path from training output
    MODEL_PATH=$(grep "Model saved to" "$TRAIN_OUTPUT" | tail -1 | sed 's/.*Model saved to \(.*\)/\1/')
    if [ -z "$MODEL_PATH" ]; then
        # Try alternative format
        MODEL_PATH=$(grep "Training complete. Model saved to" "$TRAIN_OUTPUT" | tail -1 | sed 's/.*Training complete. Model saved to \(.*\)/\1/')
    fi
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Could not determine model path from training output. Check $TRAIN_OUTPUT"
        return 1
    fi
    
    echo "Model trained and saved to: $MODEL_PATH"
    
    # Evaluate the model
    EVAL_OUTPUT="$EXPERIMENT_DIR/${loss_fn}_${epochs}_eval_output.txt"
    echo "Evaluating model..."
    python scripts/evaluate_trained_ipip_model.py --model_path "$MODEL_PATH" > "$EVAL_OUTPUT" 2>&1
    
    # Extract evaluation directory from output
    EVAL_DIR=$(grep "Results saved to" "$EVAL_OUTPUT" | tail -1 | sed 's/.*Results saved to \(.*\)/\1/')
    if [ -z "$EVAL_DIR" ]; then
        echo "Could not determine evaluation directory from output. Check $EVAL_OUTPUT"
        return 1
    fi
    
    echo "Evaluation complete. Results saved to: $EVAL_DIR"
    
    # Copy key evaluation files to experiment directory
    cp "$EVAL_DIR/evaluation_metrics.txt" "$EXPERIMENT_DIR/${loss_fn}_${epochs}_metrics.txt"
    cp "$EVAL_DIR/confusion_matrix.png" "$EXPERIMENT_DIR/${loss_fn}_${epochs}_confusion_matrix.png"
    cp "$EVAL_DIR/tsne_combined.png" "$EXPERIMENT_DIR/${loss_fn}_${epochs}_tsne_combined.png"
    
    # Extract key metrics for summary
    ARI=$(grep "Adjusted Rand Index" "$EVAL_DIR/evaluation_metrics.txt" | sed 's/.*: \(.*\)/\1/')
    NMI=$(grep "Normalized Mutual Information" "$EVAL_DIR/evaluation_metrics.txt" | sed 's/.*: \(.*\)/\1/')
    PURITY=$(grep "Cluster Purity" "$EVAL_DIR/evaluation_metrics.txt" | sed 's/.*: \(.*\)/\1/')
    
    echo "Metrics: ARI=$ARI, NMI=$NMI, Purity=$PURITY"
    
    # Add to summary file
    echo "$loss_fn,$epochs,$ARI,$NMI,$PURITY,$MODEL_PATH,$EVAL_DIR" >> "$EXPERIMENT_DIR/metrics_summary.csv"
}

# Initialize summary file
echo "loss_function,epochs,ari,nmi,purity,model_path,eval_dir" > "$EXPERIMENT_DIR/metrics_summary.csv"

# Run experiments with different loss functions
train_and_evaluate "mnrl" 10
train_and_evaluate "triplet" 10
train_and_evaluate "contrastive" 10
train_and_evaluate "cosine" 10

# Try MNRL with more epochs to see if it improves
train_and_evaluate "mnrl" 15

# Generate summary report
echo
echo "===== Experiment Summary ====="
echo "Generating summary report..."

# Create a formatted summary report
cat > "$EXPERIMENT_DIR/experiment_summary.md" << EOF
# Loss Function Comparison Experiment

**Date:** $(date)

This experiment compared different loss functions for training embedding models on IPIP data.

## Results Summary

| Loss Function | Epochs | ARI | NMI | Purity |
|---------------|--------|-----|-----|--------|
EOF

# Add data rows from CSV, skipping header
tail -n +2 "$EXPERIMENT_DIR/metrics_summary.csv" | while IFS=, read -r loss epochs ari nmi purity model eval; do
    echo "| $loss | $epochs | $ari | $nmi | $purity |" >> "$EXPERIMENT_DIR/experiment_summary.md"
done

# Add plots section
cat >> "$EXPERIMENT_DIR/experiment_summary.md" << EOF

## Best Performing Model

$(cat "$EXPERIMENT_DIR/metrics_summary.csv" | sort -t, -k3 -nr | head -2 | while IFS=, read -r loss epochs ari nmi purity model eval; do
    if [ -n "$loss" ]; then
        echo "**Loss Function:** $loss"
        echo "**Epochs:** $epochs"
        echo "**ARI:** $ari"
        echo "**NMI:** $nmi"
        echo "**Purity:** $purity"
        echo "**Model Path:** $model"
    fi
done)

## Visualizations

The visualizations for each model can be found in the experiment directory:
\`$EXPERIMENT_DIR\`

EOF

echo "Summary report generated: $EXPERIMENT_DIR/experiment_summary.md"
echo
echo "===== Experiment Complete ====="
echo "All results saved to: $EXPERIMENT_DIR"