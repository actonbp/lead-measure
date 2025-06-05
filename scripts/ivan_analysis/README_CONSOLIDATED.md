# Ivan's Enhanced Analysis Pipeline - Consolidated Guide

## Quick Start

### 1. Run Complete Pipeline (Recommended)
```bash
# Activate environment
source leadmeasure_env/bin/activate

# Run complete pipeline with automatic platform detection
./scripts/run_ivan_analysis.sh
```

### 2. Monitor Progress
```bash
# Check training status
python3 scripts/ivan_analysis/run_training_background.py monitor

# Watch live logs
tail -f logs/training_*.log

# Stop training if needed
python3 scripts/ivan_analysis/run_training_background.py stop
```

## Overview

This pipeline implements Ivan Hernandez's enhanced methodology for testing whether leadership measurement constructs are semantically distinct from personality constructs.

**Key Innovation**: Triple randomization + TSDAE pre-training + GIST loss achieves 99.43% construct separation accuracy (vs 81.66% baseline).

## Pipeline Components

### Core Scripts (Use These)

1. **`unified_training.py`** - Main training script with automatic platform detection
   - Detects Mac Silicon, CUDA, or CPU
   - Optimizes parameters based on available memory
   - Handles holdout validation properly

2. **`run_training_background.py`** - Background execution manager
   - Start: `python3 run_training_background.py start --mode holdout --high-memory`
   - Monitor: `python3 run_training_background.py monitor`
   - Stop: `python3 run_training_background.py stop`

3. **`validate_holdout_results.py`** - Compare IPIP holdout vs leadership performance
   - Run after training completes
   - Generates visualizations and statistical analysis

### Supporting Scripts

- `create_holdout_splits.py` - Creates 80/20 train/test split (already run)
- `build_pairs_randomized.py` - Generates training pairs with triple randomization

## Expected Results

### Without Holdout Validation (Biased)
- IPIP: 94.70% accuracy
- Leadership: 66.33% accuracy
- Difference: 28.37 percentage points

### With Holdout Validation (Unbiased) 
- Results pending current training run
- Will show true generalization performance

## Platform-Specific Optimizations

### Mac Studio M1/M2 (64GB)
- MPS acceleration enabled
- Batch sizes: TSDAE=16, GIST=96
- Expected time: 3-4 hours

### Standard Systems
- CPU or lower memory
- Batch sizes: TSDAE=4, GIST=32  
- Expected time: 6+ hours

## Troubleshooting

### Mac-Specific Issues
- "invalid low watermark ratio": Fixed by setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- Multiprocessing errors: Fixed by disabling workers for DenoisingAutoEncoderDataset

### General Issues
- Check PyTorch version: Need 2.0+ for MPS support
- Memory errors: Reduce batch sizes or use standard mode

## Next Steps After Training

1. **Validate Results**
   ```bash
   python3 scripts/ivan_analysis/validate_holdout_results.py
   ```

2. **Review Findings**
   - Check `data/visualizations/holdout_validation/`
   - Read summary in `holdout_validation_summary.txt`

3. **If Findings Hold**
   - Leadership constructs show significant overlap
   - Consider alternative linguistic taxonomies
   - Explore consolidation to 2-3 core dimensions

## Repository Cleanup Plan

To reduce clutter, deprecated scripts will be moved to `scripts/ivan_analysis/archive/`:
- Individual platform scripts (train_with_tsdae.py, train_with_tsdae_mac_studio.py)
- Duplicate holdout scripts (train_with_holdout.py)
- Old visualization scripts

Keep only the unified pipeline for future use.