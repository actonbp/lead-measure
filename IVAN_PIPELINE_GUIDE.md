# Ivan's Enhanced Analysis Pipeline - Step-by-Step Guide

This guide provides the exact steps to run Ivan Hernandez's enhanced methodology for comparing leadership vs personality construct separation using holdout validation.

## Prerequisites

1. **Virtual Environment**: Ensure `leadmeasure_env` is created and has all dependencies
2. **Data Files**: IPIP and leadership data should be in `data/processed/`
3. **Hardware**: Works on Mac Silicon, NVIDIA GPU, or CPU (auto-detects)

## Quick Start Options

### Option A: Background Execution (Recommended - Takes 3-12 hours)
```bash
cd /Users/acton/Documents/GitHub/lead-measure
source leadmeasure_env/bin/activate
python3 scripts/ivan_analysis/run_training_background.py start --mode holdout --high-memory
# Check progress later with:
python3 scripts/ivan_analysis/run_training_background.py monitor
```

### Option B: Foreground Execution (Must Keep Terminal Open)
```bash
cd /Users/acton/Documents/GitHub/lead-measure
./scripts/ivan_analysis/run_complete_pipeline.sh
```

## One-Command Pipeline (Recommended)

```bash
cd /Users/acton/Documents/GitHub/lead-measure
./scripts/ivan_analysis/run_complete_pipeline.sh
```

This single command handles everything:
- Platform detection (Mac Silicon/NVIDIA/CPU)
- Memory optimization based on hardware
- Creates holdout splits (80% train, 20% test)
- Trains model with TSDAE + GIST loss
- Validates IPIP holdout vs leadership constructs
- Generates key visualizations

**Note**: This runs in the foreground and can take 3-12 hours. For background execution with monitoring, see the section below.

## Manual Step-by-Step Process

If you prefer to run steps individually:

### Step 1: Environment Setup (< 1 minute)
```bash
cd /Users/acton/Documents/GitHub/lead-measure
source leadmeasure_env/bin/activate
```

### Step 2: Create Holdout Splits (< 1 minute) 
```bash
python3 scripts/ivan_analysis/create_holdout_splits.py --method stratified --holdout-ratio 0.2
```

**Output**: 
- `data/processed/ipip_train_pairs_holdout.jsonl` (training data)
- `data/processed/ipip_holdout_items.csv` (test items)

### Step 3: Train Model with Holdout Validation ⏱️ **THIS IS THE LONG STEP (3-12 hours)**
```bash
python3 scripts/ivan_analysis/unified_training.py --mode holdout --high-memory --skip-tsdae
```

**⚠️ TIME REQUIREMENTS**:
- **Mac Silicon (M1/M2)**: 3-4 hours
- **NVIDIA GPU**: 2-6 hours (depending on GPU model)
- **CPU**: 8-12 hours

**💡 TIP**: Use background execution (see above) so you don't need to keep terminal open!

**Platform-specific optimizations**:
- **Mac Silicon (M1/M2)**: MPS acceleration, batch sizes optimized for 32-64GB memory
- **NVIDIA GPU**: CUDA acceleration with appropriate batch sizes
- **CPU**: Conservative settings for broader compatibility

### Step 4: Validate Results (< 5 minutes)
```bash
python3 scripts/ivan_analysis/validate_holdout_results.py
```

## Background Execution (Recommended for Long Training)

Since training can take 3-12 hours depending on hardware, background execution allows you to:
- Start training and close your terminal
- Check progress periodically
- Continue other work while training runs

### Starting Training in Background

```bash
cd /Users/acton/Documents/GitHub/lead-measure
source leadmeasure_env/bin/activate
python3 scripts/ivan_analysis/run_training_background.py start --mode holdout --high-memory
```

This will:
- Create a background process that continues even if you close terminal
- Save process ID to `training_pid.txt`
- Write all output to `logs/training_background.log`

### Monitoring Progress

Check training progress anytime:

```bash
python3 scripts/ivan_analysis/run_training_background.py monitor
```

This shows:
- Current training status (running/stopped)
- Training progress (epochs, loss values)
- Estimated time remaining
- Last 20 lines of training output

### Checking if Training is Still Running

```bash
python3 scripts/ivan_analysis/run_training_background.py status
```

Returns:
- Process ID if running
- "No training process found" if completed or stopped

### Viewing Full Logs

```bash
# View complete training log
tail -f logs/training_background.log

# View last 100 lines
tail -100 logs/training_background.log
```

### Stopping Training

If you need to stop training:

```bash
python3 scripts/ivan_analysis/run_training_background.py stop
```

### After Training Completes

Once training finishes (check with `status` command), run validation:

```bash
python3 scripts/ivan_analysis/validate_holdout_results.py
```

## Expected Results

**IPIP Holdout Constructs** (never seen during training):
- **~87% accuracy** in construct separation
- Cohen's d ≈ 1.1 (large effect)

**Leadership Constructs**:
- **~63% accuracy** in construct separation
- Cohen's d ≈ 0.37 (small effect)

**Key Finding**: **~24 percentage point difference** providing empirical evidence for leadership construct redundancy.

## Output Files

### Models
- `models/gist_holdout_unified_final/` - Trained model

### Key Visualizations
- `data/visualizations/holdout_validation/holdout_validation_summary.txt` - Statistical summary
- `data/visualizations/construct_holdout_validation/top5_coherent_constructs_tsne.png` - **Main figure**
- `data/visualizations/construct_holdout_validation/clean_performance_summary.png` - Performance comparison

### Detailed Results
- `data/visualizations/construct_holdout_validation/holdout_validation_results.json` - Exact metrics
- `data/visualizations/construct_holdout_validation/holdout_constructs_summary.md` - Analysis summary

## Platform Detection Details

The pipeline automatically detects:

1. **Mac Silicon (M1/M2)**: 
   - Uses MPS acceleration
   - Optimizes batch sizes for unified memory
   - High-memory mode for 32GB+ systems

2. **NVIDIA GPU**:
   - Uses CUDA acceleration
   - Detects GPU memory automatically
   - Enables high-memory optimizations

3. **CPU Only**:
   - Conservative batch sizes
   - Longer training time
   - Still produces valid results

## Time Estimates

### Total Pipeline Time
- **Mac Studio M1 (64GB)**: 3-4 hours
- **NVIDIA GPU**: 2-6 hours (depending on GPU)
- **CPU only**: 8-12 hours

### Time Breakdown by Step
1. **Environment Setup**: < 1 minute
2. **Create Holdout Splits**: < 1 minute
3. **Model Training**: ⏱️ **3-12 hours** (THIS IS THE LONG STEP)
4. **Validation Analysis**: < 5 minutes

**Note**: Over 95% of the time is spent in Step 3 (Model Training). All other steps are quick.

## Troubleshooting

### Common Issues

1. **Virtual environment not found**:
   ```bash
   python3 -m venv leadmeasure_env
   source leadmeasure_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Missing data files**:
   - Ensure IPIP.csv and leadership data are in `data/` directory
   - Run preprocessing scripts if needed

3. **Memory issues**:
   - Remove `--high-memory` flag for systems with <16GB RAM
   - Reduce batch sizes manually in the script

4. **Permission denied on shell script**:
   ```bash
   chmod +x scripts/ivan_analysis/run_complete_pipeline.sh
   ```

## Next Steps After Pipeline Completion

1. **Review Results**: Check validation summary and key visualizations
2. **Analyze Leadership Clusters**: Examine what makes up leadership construct groupings
3. **Prepare for Part 2**: Use results to understand leadership construct composition
4. **Documentation**: Update analysis with new findings

## Contact/Support

If issues arise, check:
- `logs/` directory for detailed error messages
- GPU memory usage with `nvidia-smi` (if applicable)
- Available disk space (models can be large)