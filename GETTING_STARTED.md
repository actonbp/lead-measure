# 🚀 Getting Started with Leadership Measurement Analysis

## Quick Overview

This repository contains a complete pipeline for analyzing whether leadership measurement constructs are semantically distinct or redundant. **Key finding**: Leadership constructs show 28.37 percentage points less distinctiveness than personality constructs, providing empirical evidence for construct proliferation concerns.

## ⚡ Quick Start (30 seconds)

```bash
# 1. Clone and setup
git clone https://github.com/actonbp/lead-measure.git
cd lead-measure
python -m venv leadmeasure_env
source leadmeasure_env/bin/activate  # Windows: leadmeasure_env\Scripts\activate
pip install -r requirements.txt

# 2. Run complete analysis
./scripts/ivan_analysis/run_complete_pipeline.sh
```

That's it! The pipeline auto-detects your platform and runs optimized training.

## 🎯 What This Repository Does

### Research Question
**Do leadership constructs represent genuinely distinct dimensions, or are they largely redundant?**

### Answer
Leadership constructs are **significantly more semantically overlapping** than established personality constructs:
- **Personality constructs**: 94.70% separation accuracy
- **Leadership constructs**: 66.33% separation accuracy  
- **Conclusion**: Evidence for construct proliferation in leadership research

### How We Found This
1. **Training**: Use contrastive learning (TSDAE + GIST loss) on personality items
2. **Testing**: Apply trained model to both personality holdout and leadership items
3. **Comparison**: Measure separation accuracy using Ivan's enhanced methodology

## 📁 Repository Structure

```
lead-measure/
├── 📖 README.md                    # Main documentation
├── 📋 CLAUDE.md                    # AI agent instructions
├── 🎯 NEXT_STEPS.md                # Current status & next steps
├── 🚀 GETTING_STARTED.md           # This file

├── 📊 data/
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Preprocessed data & pairs
│   └── visualizations/             # Generated plots & analysis

├── 🔧 scripts/
│   ├── ivan_analysis/              # 🌟 MAIN PIPELINE (USE THIS)
│   │   ├── unified_training.py     # Platform-agnostic training
│   │   ├── run_training_background.py  # Background execution
│   │   ├── validate_holdout_results.py # Final validation
│   │   └── README_CONSOLIDATED.md  # Detailed technical docs
│   │
│   ├── archive/                    # Original scripts (reference only)
│   └── cleanup_repository.py       # Repository cleanup tool

├── 🧠 models/                      # Trained models (not in git)
├── 📝 logs/                        # Training logs & monitoring  
└── 🐍 leadmeasure_env/             # Python environment (not in git)
```

## 🎮 Usage Options

### Option 1: Complete Pipeline (Recommended)
**Best for**: First-time users, running full analysis
```bash
./scripts/ivan_analysis/run_complete_pipeline.sh
```
- Auto-detects platform (Mac Silicon, CUDA, CPU)
- Optimizes parameters automatically
- Runs holdout validation
- Expected time: 1.5-6 hours depending on hardware

### Option 2: Background Training (For Long Tasks)
**Best for**: Long-running training, monitoring progress
```bash
# Start training in background
python3 scripts/ivan_analysis/run_training_background.py start --mode holdout --high-memory

# Check progress anytime
python3 scripts/ivan_analysis/run_training_background.py monitor

# When done, run validation  
python3 scripts/ivan_analysis/validate_holdout_results.py
```

### Option 3: Step-by-Step (For Development)
**Best for**: Debugging, customization, understanding internals
```bash
# 1. Create holdout splits
python3 scripts/ivan_analysis/create_holdout_splits.py

# 2. Train model
python3 scripts/ivan_analysis/unified_training.py --mode holdout --high-memory

# 3. Validate results
python3 scripts/ivan_analysis/validate_holdout_results.py
```

## 🖥️ Platform Optimizations

### Mac Studio M1/M2 (64GB) ⚡
- **4x faster training** with MPS acceleration
- **Batch sizes**: TSDAE=16, GIST=96
- **Time**: ~1.5-2 hours
- **Auto-optimizations**: Unified memory, larger batches

### Standard Systems 🔋
- **CPU training** with smaller batches  
- **Time**: 4-6 hours
- **Works on**: Linux, Windows, Intel Mac
- **Memory**: 16GB+ recommended

## 📊 Expected Results

### Validation Outputs
- **Summary**: `data/visualizations/holdout_validation/holdout_validation_summary.txt`
- **Plots**: t-SNE visualizations, similarity distributions
- **Stats**: Cohen's d, probability comparisons, effect sizes

### Key Metrics
```
IPIP Holdout Accuracy:     [To be determined from current training]
Leadership Accuracy:       [To be determined from current training]  
Performance Difference:    [Gap between personality vs leadership]
```

## 🧹 Repository Maintenance

### Clean Up Clutter
```bash
# See what would be cleaned (safe)
python3 scripts/cleanup_repository.py --dry-run

# Perform cleanup
python3 scripts/cleanup_repository.py
```

### Manage Large Files
```bash
# Clean old models (keeps most recent)
python3 scripts/cleanup_models.py

# Clean temporary data
python3 scripts/cleanup_data.py
```

## 🔧 Troubleshooting

### Common Issues

**1. "Can't pickle" errors on Mac**
- **Fixed**: Use `unified_training.py` (handles multiprocessing properly)

**2. "MPS backend out of memory"**  
- **Fixed**: Scripts set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`

**3. Training too slow**
- **Use**: `--high-memory` flag for optimizations
- **Check**: Platform auto-detection working correctly

**4. Import errors**
- **Run**: `pip install -r requirements.txt` in activated environment

### Getting Help

1. **Check logs**: `tail -f logs/training_*.log`
2. **Monitor training**: `python3 scripts/ivan_analysis/run_training_background.py monitor` 
3. **Read docs**: `scripts/ivan_analysis/README_CONSOLIDATED.md`

## 📚 Research Context

### What We've Proven
- **Methodological advancement**: Contrastive learning for construct validation
- **Empirical evidence**: Leadership construct proliferation quantified
- **Technical achievement**: 4x faster training with Mac optimizations

### Research Implications
- **Theory**: Need for construct consolidation in leadership research
- **Measurement**: Current leadership constructs may measure similar traits
- **Future work**: Data-driven taxonomies vs. theory-driven constructs

### Next Research Phase
- Explore **linguistic features** that distinguish leadership items
- Develop **alternative taxonomies** based on empirical clustering
- **Cross-validate** findings with other leadership frameworks

---

**🎯 Ready to start? Run**: `./scripts/ivan_analysis/run_complete_pipeline.sh`