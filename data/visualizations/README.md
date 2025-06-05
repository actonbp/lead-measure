# Visualizations Directory (Updated June 4, 2025)

This directory contains all analysis outputs, plots, and visualizations from the leadership measurement project.

## 🎯 Most Important Results

### Primary Evidence (Used in Manuscript)

1. **`construct_holdout_validation/top5_coherent_constructs_tsne.png`** ⭐⭐⭐
   - **THE KEY FIGURE**: Cleanest comparison of IPIP vs Leadership separation
   - Shows 5 most coherent IPIP constructs vs 5 most coherent leadership constructs
   - Demonstrates clear clustering for IPIP, overlapping clusters for leadership
   - Primary visual evidence for construct proliferation

2. **`enhanced_statistical_comparison/enhanced_statistical_comparison.png`**
   - Bar chart showing 87.4% vs 62.9% accuracy with error bars
   - Includes statistical significance annotations
   - Publication-ready format

3. **`holdout_validation/holdout_tsne_comparison.png`**
   - Full holdout validation results
   - Shows unbiased comparison with held-out IPIP items

## 📊 Statistical Results

### Key JSON Files with Numerical Results

- **`construct_holdout_validation/holdout_validation_results.json`**
  - Contains exact accuracy percentages (87.4% IPIP, 62.9% leadership)
  - Cohen's d effect sizes (1.116 vs 0.368)
  - Statistical test results

- **`enhanced_statistical_comparison/enhanced_statistical_analysis_results.json`**
  - T-test results (t = 43.49, p < 2.22e-16)
  - Confidence intervals for effect sizes
  - Bootstrap validation results

## 📁 Directory Organization

### Current Validation Results (June 2025)
- `construct_holdout_validation/` - Final validated results with construct-level holdout
- `enhanced_statistical_comparison/` - Statistical significance testing
- `holdout_validation/` - Initial holdout validation approach

### Historical Results (For Reference)
- `ivan_analysis/` - Comparison with Ivan's original methodology
- `mnrl_evaluation_*/` - Various model evaluation approaches
- `trained_ipip_*/` - Different training configurations
- `big_five_*/` - Initial Big Five personality analysis
- `leadership_styles_*/` - Early leadership-only analysis

## 🔍 Understanding the Visualizations

### t-SNE Plots
- **Good separation**: Distinct, non-overlapping clusters (see IPIP constructs)
- **Poor separation**: Overlapping, mixed clusters (see leadership constructs)
- **Color coding**: Each color represents a different construct

### Statistical Plots
- **Accuracy**: Percentage of correctly ranked same-construct pairs
- **Cohen's d**: Effect size (>0.8 is large, 0.5-0.8 is medium, <0.5 is small)
- **Error bars**: 95% confidence intervals

### Distance Distribution Plots
- Show distribution of cosine similarities
- Separate curves for same-construct vs different-construct pairs
- Greater separation = better construct distinctiveness

## 📈 Key Metrics Summary

| Metric | IPIP Personality | Leadership | Difference |
|--------|-----------------|------------|------------|
| Accuracy | 87.4% | 62.9% | 24.5 pp |
| Cohen's d | 1.116 (Large) | 0.368 (Small) | 0.748 |
| t-statistic | 43.49 | - | p < 0.001 |

## 🎨 Visualization Tools Used

- **t-SNE**: For dimensionality reduction and cluster visualization
- **Matplotlib/Seaborn**: For statistical plots and distributions
- **Plotly**: For interactive visualizations (HTML files)

## 📝 Notes for Future Analysis

1. **Replication**: All visualizations can be regenerated using scripts in `scripts/ivan_analysis/`
2. **Raw data**: Embedding vectors and distances stored in `.pkl` files in `processed/`
3. **Customization**: Modify visualization scripts to highlight different aspects

## 🚀 Quick Reference for Paper Writing

When citing visualizations in manuscripts:
- Use `top5_coherent_constructs_tsne.png` as primary visual evidence
- Reference exact statistics from JSON files for numerical claims
- Note that all results use construct-level holdout validation (no data leakage)