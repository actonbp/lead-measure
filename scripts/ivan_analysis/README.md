# Ivan's Enhanced Analysis Scripts

**📖 For the latest unified pipeline, see [README_CONSOLIDATED.md](README_CONSOLIDATED.md)**

This directory contains the implementation of Ivan Hernandez's enhanced embedding analysis approach (June 2025).

## ✅ TRAINING COMPLETE (June 3, 2025)

**Status**: Holdout validation training finished at 3:18 PM
**Model**: `models/gist_holdout_unified_final`

## 🚨 CRITICAL NEXT STEP

```bash
# MUST BE RUN FROM MAIN PROJECT DIRECTORY!
cd /Users/acton/Documents/GitHub/lead-measure
python3 scripts/ivan_analysis/validate_holdout_results.py
```

This will run the final unbiased comparison between holdout IPIP and leadership constructs.

## Quick Start (Background Training)

```bash
# Start training in background with platform auto-detection
python3 scripts/ivan_analysis/run_training_background.py start --mode holdout --high-memory

# Monitor progress
python3 scripts/ivan_analysis/run_training_background.py monitor
```

## Overview

These scripts implement several key improvements over our previous methods:
- Triple randomization in pair generation
- TSDAE pre-training for domain adaptation  
- BGE-M3 model optimized for clustering
- Enhanced training parameters (batch size 96, FP16)
- Statistical validation with t-tests and Cohen's d

## Scripts

### 1. `build_pairs_randomized.py`
Generates training pairs with triple randomization to eliminate ordering bias:
- Randomizes within pairs
- Randomizes anchor/positive positions
- Additional pointwise randomization
- Removes any NaN values

**Usage:**
```bash
python scripts/ivan_analysis/build_pairs_randomized.py
```

**Output:** `data/processed/ipip_pairs_randomized.jsonl`

### 2. `train_with_tsdae.py`
Trains embeddings with TSDAE pre-training and GIST loss:
- TSDAE pre-training on all IPIP texts
- BGE-M3 base model with CLS pooling
- Batch size 96 with FP16 training
- 5 phases of 10 epochs each
- Saves checkpoints after each phase

**Usage:**
```bash
python scripts/ivan_analysis/train_with_tsdae.py
```

**Outputs:**
- `models/tsdae_pretrained/` - TSDAE pre-trained model
- `models/ivan_tsdae_gist_phase{1-5}/` - Phase checkpoints
- `models/ivan_tsdae_gist_final/` - Final model

### 3. `visualize_and_analyze.py`
Creates visualizations and performs statistical analysis:
- t-SNE visualizations with median centroids
- Similarity analysis (same vs different constructs)
- Statistical tests (paired t-test, Cohen's d)
- Model comparison with baseline

**Usage:**
```bash
# Analyze a specific model
python scripts/ivan_analysis/visualize_and_analyze.py --model models/ivan_tsdae_gist_final --dataset ipip

# Compare with baseline
python scripts/ivan_analysis/visualize_and_analyze.py --model models/ivan_tsdae_gist_final --compare-baseline
```

**Outputs:**
- `data/visualizations/ivan_analysis/tsne_*.png` - t-SNE visualizations
- `data/visualizations/ivan_analysis/similarity_analysis.csv` - Statistical results
- `data/visualizations/ivan_analysis/model_comparison.csv` - Baseline comparison

## Running the Analysis

### Recommended: Step-by-Step with Validation

Use the interactive runner for controlled execution:
```bash
# Check environment first
python scripts/ivan_analysis/run_analysis_steps.py --check

# Run step by step
python scripts/ivan_analysis/run_analysis_steps.py --step 1  # Generate pairs
python scripts/ivan_analysis/run_analysis_steps.py --step 2  # Train model
python scripts/ivan_analysis/run_analysis_steps.py --step 3  # Analyze IPIP
python scripts/ivan_analysis/run_analysis_steps.py --step 4  # Compare baseline

# Or use Make commands
make ivan-step1
make ivan-step2
make ivan-step3
make ivan-step4
```

### Alternative: Run Complete Workflow

For experienced users:
```bash
# Using Make
make ivan-all

# Or using shell script
./scripts/run_ivan_analysis.sh
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

## Key Results

Ivan's approach achieves:
- **99.43%** probability of correctly ranking same-construct similarity (vs 81.66% baseline)
- **Cohen's d = 2.487** (massive effect size)
- Clear visual separation of constructs in t-SNE space

## Integration with Main Pipeline

These scripts are designed to work alongside our existing pipeline. The trained models can be used with our standard evaluation scripts:

```bash
# Use Ivan's model with our standard evaluation
python scripts/evaluate_model_with_validation.py --model models/ivan_tsdae_gist_final --dataset ipip

# Apply to leadership data
python scripts/evaluate_mnrl_on_leadership.py --model_path models/ivan_tsdae_gist_final
```