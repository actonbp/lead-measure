#!/bin/bash
# Script to run the complete holdout validation pipeline
# This addresses the training bias issue in the original analysis

echo "🚀 Starting Holdout Validation Pipeline"
echo "======================================="

# Activate virtual environment if it exists
if [ -d "leadmeasure_env" ]; then
    echo "Activating virtual environment..."
    source leadmeasure_env/bin/activate
fi

# Step 1: Create holdout splits (if not already done)
if [ ! -f "data/processed/ipip_train_pairs_holdout.jsonl" ]; then
    echo ""
    echo "Step 1: Creating holdout validation splits..."
    echo "---------------------------------------------"
    python3 scripts/ivan_analysis/create_holdout_splits.py
    if [ $? -ne 0 ]; then
        echo "❌ Error creating holdout splits"
        exit 1
    fi
else
    echo ""
    echo "✅ Holdout splits already exist, skipping step 1"
fi

# Step 2: Train model on holdout data
echo ""
echo "Step 2: Training model on holdout data"
echo "--------------------------------------"
echo "Choose training configuration:"
echo "1) Standard parameters (slower, less memory)"
echo "2) Mac Studio optimized (faster, requires 64GB memory)"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "2" ]; then
    echo "Using Mac Studio optimized parameters..."
    python3 scripts/ivan_analysis/train_with_holdout.py --optimized
else
    echo "Using standard parameters..."
    python3 scripts/ivan_analysis/train_with_holdout.py
fi

if [ $? -ne 0 ]; then
    echo "❌ Error during training"
    exit 1
fi

# Step 3: Validate results
echo ""
echo "Step 3: Validating holdout results"
echo "----------------------------------"
python3 scripts/ivan_analysis/validate_holdout_results.py

if [ $? -ne 0 ]; then
    echo "❌ Error during validation"
    exit 1
fi

echo ""
echo "✅ Holdout validation pipeline complete!"
echo ""
echo "Results saved to:"
echo "- Visualizations: data/visualizations/holdout_validation/"
echo "- Summary: data/visualizations/holdout_validation/holdout_validation_summary.txt"
echo ""
echo "Key files created:"
echo "- ipip_train_pairs_holdout.jsonl - Training pairs (80% of data)"
echo "- ipip_holdout_items.csv - Holdout items for validation (20% of data)"
echo "- models/ivan_holdout_gist_final/ - Model trained on training split only"
echo ""