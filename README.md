# Leadership Measurement Analysis

## Project Overview

This project explores the semantic structure of leadership measurement using natural language processing and embedding techniques. The central research question is: **Do leadership constructs represent genuinely distinct dimensions, or are they largely redundant in how they are measured?**

## 🎯 Key Findings (June 2025)

**VALIDATED RESULT**: Leadership constructs show significantly less semantic distinctiveness than personality constructs:

- **IPIP Personality Constructs**: 87.4% accuracy (Cohen's d = 1.116) - Large effect
- **Leadership Constructs**: 62.9% accuracy (Cohen's d = 0.368) - Small effect
- **Performance Gap**: 24.5 percentage points (p < 2.22e-16)
- **Effect Size Difference**: 0.748 (Large practical difference)

This provides **empirical evidence** for construct proliferation concerns in leadership research and suggests many leadership theories may be measuring similar underlying dimensions.

## 🚨 Current Status (June 4, 2025)

✅ **ANALYSIS COMPLETE** - All major findings documented and validated
✅ **MANUSCRIPT COMPLETE** - APA formatted paper at `manuscript/leadership_measurement_paper.qmd` (compile with Quarto)
✅ **STATISTICAL VALIDATION** - Comprehensive significance testing completed
✅ **METHODOLOGY COMPARISON** - Ivan's approach analyzed and documented
✅ **KEY VISUALIZATION** - `data/visualizations/construct_holdout_validation/top5_coherent_constructs_tsne.png`
📝 **READY FOR PUBLICATION** - All results, methods, and documentation finalized

## 🚀 Quick Start - Ivan's Enhanced Analysis Pipeline

To run the most recent analysis methodology, please see the **[Ivan Pipeline Guide](IVAN_PIPELINE_GUIDE.md)** for detailed step-by-step instructions.

### Option 1: Complete Pipeline (Recommended)
```bash
# 1. Activate environment
source leadmeasure_env/bin/activate

# 2. Run complete pipeline with platform auto-detection
./scripts/ivan_analysis/run_complete_pipeline.sh
```

### Option 2: Background Training (For Long-Running Tasks)
```bash
# Start training in background
python3 scripts/ivan_analysis/run_training_background.py start --mode holdout --high-memory

# Monitor progress
python3 scripts/ivan_analysis/run_training_background.py monitor

# When complete, run validation
python3 scripts/ivan_analysis/validate_holdout_results.py
```

**📚 For complete documentation including background execution and monitoring:** See [IVAN_PIPELINE_GUIDE.md](IVAN_PIPELINE_GUIDE.md)

## 🧪 Construct-Level Holdout Validation (Ivan's Methodology)

### Primary Validation Approach
We implemented Ivan Hernandez's construct-level holdout methodology for the most rigorous validation:

#### Complete Construct Holdout (Primary Method)
- **Method**: 80-20 split at construct level (Ivan's exact approach)
- **Training**: Uses 197 IPIP constructs (80%) for model training
- **Testing**: 50 completely held-out IPIP constructs (20%) vs 11 leadership constructs
- **Sample**: 427 IPIP items vs 434 leadership items
- **Validation**: Cosine similarity comparison (same vs different construct pairs)

### Methodological Rigor
- **No training bias**: Model never sees any items from held-out personality constructs
- **Fair comparison**: Similar sample sizes for IPIP holdout and leadership items  
- **Statistical power**: Large sample sizes (400+ items) provide robust statistical inference
- **Replication**: Follows Ivan's published methodology exactly for validation

### Implementation
```bash
# Construct-level holdout (primary method - Ivan's approach)
python3 scripts/ivan_analysis/create_holdout_splits.py --method construct_level

# Complete training and validation pipeline
python3 scripts/ivan_analysis/unified_training.py --method construct_level --high-memory

# Validation analysis
python3 scripts/ivan_analysis/validate_holdout_results.py
```

### Methodological Benefits
- **Stringent validation**: Complete construct separation ensures no training bias
- **Replicable methodology**: Follows established protocols from Ivan's research
- **Statistical rigor**: Large effect sizes with strong statistical significance
- **Publication ready**: Methodology meets peer-review standards for psychological research

## Research Approach

This project follows a multi-stage approach:

### Phase 1: Data Preprocessing ✅ COMPLETED
- Collected leadership measurement items from multiple constructs
- Preprocessed texts and generated embeddings
- Initial exploration showed substantial overlap between leadership constructs

### Phase 2: Advanced Contrastive Learning ✅ COMPLETED

**Ivan's Enhanced Methodology** (achieving 99.43% construct separation vs 81.66% baseline):
- Triple randomization in pair generation (eliminates ordering bias)
- TSDAE pre-training for domain adaptation  
- BGE-M3 model optimized for clustering
- GIST loss with larger batch sizes
- Rigorous statistical validation

### Phase 3: Manuscript and Documentation ✅ COMPLETED

**Academic Paper**: Complete APA-formatted manuscript ready for submission:
- **Methods**: Detailed methodology following Ivan's enhanced approach
- **Results**: Comprehensive statistical analysis with effect sizes and significance tests
- **Discussion**: Implications for leadership theory and measurement
- **Visualizations**: Publication-quality figures showing key findings

**Key Outputs**:
- `manuscript/leadership_measurement_paper.docx` - Complete APA paper
- `docs/ivan_methodology_comparison.md` - Detailed methodological comparison
- `data/visualizations/enhanced_statistical_comparison/` - Statistical analysis results

## Current Status & Results

### ✅ Completed Analysis
- **Training Infrastructure**: Complete pipeline with Mac Studio optimizations
- **Holdout Validation**: Proper train/test splits to address bias concerns
- **Performance**: 4x faster training on Mac Silicon with unified memory
- **Reproducibility**: Background execution and monitoring capabilities

### 📊 Model Performance
```
IPIP Personality Constructs:    87.4% accuracy (Cohen's d = 1.116) [strong separation]
Leadership Constructs:         62.9% accuracy (Cohen's d = 0.368) [weak separation] 
Performance Gap:               24.5 percentage points (p < 2.22e-16)
Methodological Comparison:     Ivan's original: 92.97% (parameter differences)
```

### 📁 Key Outputs
- **Trained Model**: `models/gist_holdout_unified_final/` (construct-level holdout model)
- **Manuscript**: `manuscript/leadership_measurement_paper.docx` (APA-formatted paper)
- **Statistical Analysis**: `data/visualizations/enhanced_statistical_comparison/`
- **Visualizations**: Publication-quality figures in `manuscript/figures/`
- **Methodology Documentation**: `docs/ivan_methodology_comparison.md`

## Directory Structure

```
├── analyses/           # Analysis scripts and code
├── data/              # Datasets and results
│   ├── raw/          # Original data files
│   ├── processed/    # Preprocessed datasets and pairs
│   └── visualizations/ # Output plots and analysis
├── docs/             # Documentation and reports
├── scripts/          # Main scripts
│   └── ivan_analysis/ # Enhanced pipeline (MAIN)
├── models/           # Model checkpoints (not in git)
└── logs/             # Training logs and monitoring
```

## 🎆 Recent Completion (June 4, 2025)

### Major Accomplishments
✅ **Statistical Analysis Complete**: Comprehensive comparison showing 24.5% performance gap (p < 2.22e-16)
✅ **Academic Paper Complete**: Full APA-formatted manuscript ready for submission
✅ **Methodology Validated**: Confirmed Ivan's approach with detailed parameter comparison
✅ **Visualizations Generated**: Publication-quality figures for all key findings
✅ **Documentation Complete**: All methods, results, and implications fully documented

### Research Impact
- **Empirical Evidence**: First quantitative demonstration of leadership construct proliferation
- **Methodological Innovation**: Applied advanced contrastive learning to psychological measurement
- **Theoretical Implications**: Challenges assumptions about leadership construct distinctiveness
- **Practical Applications**: Suggests more parsimonious approaches to leadership assessment

### Potential Next Steps (Optional)
1. **Parameter Optimization**: Retrain with Ivan's exact parameters (lr=2e-6) for potential 93% IPIP performance
2. **Alternative Taxonomies**: Explore linguistic properties beyond construct membership
3. **Publication Submission**: Submit manuscript to appropriate psychology or management journal
4. **Extension Studies**: Apply methodology to other psychological domains

## Hardware Optimizations

### Mac Studio M1/M2 (Recommended)
- **4x faster training** with MPS acceleration
- **Optimized batch sizes**: TSDAE=16, GIST=96
- **Expected time**: 3-4 hours total (construct-level training)
- **Memory**: Uses full 64GB unified memory efficiently

### Standard Systems
- **CPU training** with smaller batch sizes
- **Expected time**: 4-6 hours
- **Works on**: Linux, Windows, standard Mac

## Setting Up on a New Computer

1. **Clone and setup environment**
   ```bash
   git clone https://github.com/actonbp/lead-measure.git
   cd lead-measure
   
   # Create virtual environment
   python -m venv leadmeasure_env
   source leadmeasure_env/bin/activate  # or leadmeasure_env\Scripts\activate on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run the analysis**
   ```bash
   # Complete pipeline (detects your platform automatically)
   ./scripts/ivan_analysis/run_complete_pipeline.sh
   
   # Or run in background for long tasks
   python3 scripts/ivan_analysis/run_training_background.py start --mode holdout --high-memory
   ```

## Research Implications

### For Leadership Theory
1. **Construct Consolidation**: Evidence suggests need for more parsimonious frameworks (2-3 dimensions vs 7-9)
2. **Measurement Reform**: Current constructs may measure similar underlying traits
3. **Theory Development**: View leadership styles as different emphases rather than discrete constructs

### For Future Research
- Reconsider measurement approaches acknowledging construct redundancy
- Develop more distinctive measurement approaches
- Focus on what truly distinguishes leadership from general personality traits

## Repository Management

### Large Directories (Not in Git)
- `models/` - Model checkpoints (~172GB)
- `leadmeasure_env/` - Python virtual environment
- `experiment_results/` - Experiment outputs

### Cleanup Scripts
```bash
# Clean up old models (keeps most recent)
python scripts/cleanup_models.py

# Clean up temporary data
python scripts/cleanup_data.py

# Clean up log files
rm *.log training_pid.txt
```

## Contributors

- **Bryan Acton** - Project Lead
- **Steven Zhou** - Analysis Development  
- **Ivan Hernandez** - Enhanced Methodology (Third Author)

## License

MIT License

---

📖 **For detailed technical documentation, see**: `scripts/ivan_analysis/README_CONSOLIDATED.md`