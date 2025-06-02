# Setup Guide for Ivan's Enhanced Analysis

This guide provides step-by-step instructions for running Ivan's enhanced embedding analysis in an organized, reproducible manner.

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- ~10GB disk space for models
- GPU optional but recommended for faster training

## Initial Setup

### 1. Clone/Navigate to Repository
```bash
cd /Users/bryanacton/Documents/GitHub/lead-measure
```

### 2. Set Up Virtual Environment

**Option A: Using Make (Recommended)**
```bash
make ivan-setup
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python3 -m venv leadmeasure_env

# Activate it
source leadmeasure_env/bin/activate  # On macOS/Linux
# or
leadmeasure_env\Scripts\activate     # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r scripts/ivan_analysis/requirements.txt
```

### 3. Verify Setup
```bash
# With Make
make ivan-check

# Or directly
python scripts/ivan_analysis/run_analysis_steps.py --check
```

## Running the Analysis

### Recommended: Step-by-Step Execution

This approach lets you verify each step before proceeding:

#### Step 1: Generate Randomized Pairs
```bash
make ivan-step1
# or
python scripts/ivan_analysis/run_analysis_steps.py --step 1
```
- Creates: `data/processed/ipip_pairs_randomized.jsonl`
- Time: ~1 minute

#### Step 2: Train Model
```bash
make ivan-step2
# or
python scripts/ivan_analysis/run_analysis_steps.py --step 2
```
- Creates: `models/tsdae_pretrained/` and `models/ivan_tsdae_gist_final/`
- Time: 30-60 minutes (GPU) or 2-4 hours (CPU)
- Note: Shows real-time training progress

#### Step 3: Analyze IPIP
```bash
make ivan-step3
# or
python scripts/ivan_analysis/run_analysis_steps.py --step 3
```
- Creates: t-SNE visualizations and similarity analysis
- Time: ~5 minutes

#### Step 4: Compare with Baseline
```bash
make ivan-step4
# or
python scripts/ivan_analysis/run_analysis_steps.py --step 4
```
- Creates: Model comparison report
- Time: ~5 minutes (downloads BGE-M3 on first run)

### Alternative: Run All Steps
```bash
make ivan-all
# or
python scripts/ivan_analysis/run_analysis_steps.py --all
```

### Interactive Mode
```bash
python scripts/ivan_analysis/run_analysis_steps.py
```
This provides a menu-driven interface for running steps.

## Checking Progress

```bash
python scripts/ivan_analysis/run_analysis_steps.py --status
```

Shows which steps have been completed and when.

## Starting Fresh

If you need to re-run from scratch:
```bash
make ivan-clean
# Then start from step 1
```

## Expected Outputs

After completing all steps:

```
data/
├── processed/
│   └── ipip_pairs_randomized.jsonl      # ~500K randomized pairs
└── visualizations/
    └── ivan_analysis/
        ├── tsne_perplexity15.png         # Fine-detail t-SNE
        ├── tsne_perplexity30.png         # Global t-SNE
        ├── similarity_analysis.csv        # Statistical results
        └── model_comparison.csv           # Baseline comparison

models/
├── tsdae_pretrained/                      # TSDAE model
└── ivan_tsdae_gist_final/                # Final trained model
```

## Key Metrics to Verify

After completion, check these metrics:

1. **Similarity Analysis** (`similarity_analysis.csv`):
   - Probability(same > diff): Should be ~99%
   - Cohen's d: Should be > 2.0

2. **Model Comparison** (`model_comparison.csv`):
   - Baseline: ~81% probability
   - Trained: ~99% probability

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in `train_with_tsdae.py`:
  - TSDAE_BATCH_SIZE = 4 (instead of 8)
  - GIST_BATCH_SIZE = 48 (instead of 96)

### CUDA/GPU Issues
- Set CPU-only mode:
  ```bash
  export CUDA_VISIBLE_DEVICES=""
  ```

### Package Conflicts
- Use exact versions from `requirements.txt`
- Consider using conda instead of pip for better dependency resolution

### Training Takes Too Long
- Use GPU if available
- Reduce number of training phases (NUM_TRAINING_PHASES = 3)
- Skip TSDAE pre-training (comment out in script)

## Next Steps

Once the analysis is complete:

1. **Apply to Leadership Data**:
   ```bash
   # Modify visualize_and_analyze.py to support leadership dataset
   # Then run:
   python scripts/ivan_analysis/visualize_and_analyze.py \
       --model models/ivan_tsdae_gist_final \
       --dataset leadership
   ```

2. **Integration with Main Pipeline**:
   ```bash
   # Use Ivan's model with existing evaluation scripts
   python scripts/evaluate_model_with_validation.py \
       --model models/ivan_tsdae_gist_final \
       --dataset ipip
   ```

3. **Generate Comprehensive Report**:
   ```bash
   python scripts/create_comprehensive_report.py \
       --include-ivan-analysis
   ```

## For AI Agents

To run this analysis programmatically:

```python
import subprocess
import os

# Set up environment
os.chdir('/Users/bryanacton/Documents/GitHub/lead-measure')

# Check environment
result = subprocess.run(['make', 'ivan-check'], capture_output=True)
if result.returncode != 0:
    subprocess.run(['make', 'ivan-setup'])

# Run analysis
subprocess.run(['make', 'ivan-all'])

# Verify results
import pandas as pd
results = pd.read_csv('data/visualizations/ivan_analysis/similarity_analysis.csv')
print(f"Success rate: {results['probability_same_higher'].iloc[0]:.2%}")
```