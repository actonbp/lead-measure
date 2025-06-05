# Leadership Embedding Analysis Scripts (Updated June 4, 2025)

This directory contains scripts for the leadership measurement analysis project. The main pipeline is in `ivan_analysis/` which implements the validated methodology.

## 🚀 Primary Pipeline: Ivan's Enhanced Analysis

The validated approach achieving 87.4% IPIP vs 62.9% leadership accuracy:

### Quick Start (Recommended)
```bash
# Run complete pipeline with platform auto-detection
./ivan_analysis/run_complete_pipeline.sh
```

### Key Scripts in `ivan_analysis/`

1. **Data Preparation**
   - `build_pairs_randomized.py` - Creates 41,723 randomized training pairs
   - `create_holdout_splits.py` - Implements construct-level holdout validation

2. **Training Pipeline**
   - `unified_training.py` - Main training script with TSDAE + GIST
   - `train_with_tsdae_mac_studio.py` - Mac Silicon optimized version
   - `run_training_background.py` - Background execution wrapper

3. **Analysis & Validation**
   - `validate_holdout_results.py` - Computes final accuracy metrics
   - `visualize_and_analyze.py` - Generates t-SNE visualizations
   - `run_analysis_steps.py` - Orchestrates analysis pipeline

### Background Training (For Long Tasks)
```bash
# Start training in background
python3 ivan_analysis/run_training_background.py start --mode holdout --high-memory

# Monitor progress
python3 ivan_analysis/run_training_background.py monitor

# Run validation when complete
python3 ivan_analysis/validate_holdout_results.py
```

## 📁 Legacy Scripts (Historical Reference)

### Archived Approaches
Located in `archive/` - earlier implementations before Ivan's methodology:
- Various MNRL implementations
- GIST loss experiments  
- Different model architectures

### Utility Scripts (Still Active)
- `analyze_construct_counts.py` - Dataset statistics
- `cleanup_models.py` - Model directory management
- `cleanup_data.py` - Data file cleanup
- `create_comprehensive_report.py` - Report generation

## 📊 Current Results

Using Ivan's enhanced methodology:
- **IPIP Accuracy**: 87.4% (Cohen's d = 1.116)
- **Leadership Accuracy**: 62.9% (Cohen's d = 0.368)
- **Statistical Significance**: p < 2.22e-16
- **Training Time**: ~3-4 hours on Mac Studio, 4-6 hours on CPU

## 🔧 Technical Details

### Model Configuration
- **Base Model**: BAAI/bge-m3 (optimized for clustering)
- **Pre-training**: TSDAE for domain adaptation
- **Fine-tuning**: GIST loss with large batches (96)
- **Validation**: Construct-level holdout (no data leakage)

### Platform Optimizations
- **Mac Silicon**: MPS acceleration, batch size 96
- **Standard CPU**: Batch size 32-64 depending on memory
- **Auto-detection**: Scripts detect platform and optimize

## 📝 Adding New Scripts

When creating new analysis scripts:
1. **Check existing functionality first** - avoid duplication
2. **Use ivan_analysis/ patterns** - maintain consistency
3. **Include docstrings** - explain purpose and usage
4. **Add to appropriate directory** - main vs ivan_analysis
5. **Update this README** - document new additions

## 🎯 Key Outputs

Scripts generate outputs in:
- `models/gist_holdout_unified_final/` - Trained model
- `data/processed/` - Training pairs and embeddings
- `data/visualizations/` - Analysis plots and figures
- `logs/` - Training progress and debugging

## 🚨 Common Issues

1. **Memory errors**: Reduce batch size in training scripts
2. **Missing model**: Check `models/` directory exists
3. **Import errors**: Ensure virtual environment is activated
4. **Slow training**: Use Mac Studio or reduce epochs

## 📚 Documentation

- Full methodology: `scripts/ivan_analysis/README_CONSOLIDATED.md`
- Parameter details: `scripts/ivan_analysis/SETUP.md`
- Main project docs: `docs/improved_workflow.md` 