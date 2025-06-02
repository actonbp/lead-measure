# Leadership Embedding Analysis Scripts (2025)

This directory contains the current scripts for the leadership embedding analysis project. For older versions, see the [archive](./archive) folder.

## Primary Workflow

The recommended workflow is to run the refined automated script:

```bash
chmod +x run_refined_workflow.sh
./run_refined_workflow.sh
```

This will execute all steps in sequence:
1. Generate comprehensive balanced pairs with randomized positions
2. Train with BGE-M3 model, TSDAE pre-training, and GIST loss
3. Evaluate with comprehensive validation metrics
4. Apply model to leadership constructs with detailed analysis

## Individual Scripts

If you need to run steps individually:

### 1. Data Preparation
- **build_ipip_pairs_improved.py**: Generates comprehensive balanced anchor-positive pairs from IPIP items

### 2. Training
- **train_ipip_mnrl.py**: Trains a model using MultipleNegativesRankingLoss on the comprehensive pairs

### 3. Evaluation & Application
- **compare_model_performance.py**: Evaluates different models on IPIP and leadership data
- **apply_best_model_to_leadership.py**: Applies best model to leadership data
- **evaluate_trained_ipip_model.py**: Detailed evaluation of a specific model

### 4. Utility Scripts
- **analyze_construct_counts.py**: Analyzes the distribution of constructs in datasets
- **cleanup_models.py**: Removes old model folders keeping only the most recent ones
- **cleanup_data.py**: Cleans up temporary and redundant data files while preserving essential ones
- **organize_data.sh**: Helper script for data organization

## Documentation

For detailed information about the improved workflow, see [docs/improved_workflow.md](../docs/improved_workflow.md). 