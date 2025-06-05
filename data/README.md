# Data Directory (Updated June 4, 2025)

This directory contains all datasets, processed outputs, and visualizations for the leadership measurement analysis project.

## Directory Structure

### Core Data Directories

- **[`raw/`](raw/)**: Original, unmodified data sources
  - `IPIP.csv` - International Personality Item Pool (3,320 items, 246 constructs)
  - `Measures_text_long.csv` - Leadership measurement items (434 items, 11 constructs)
  - Original survey instruments and documentation

- **[`processed/`](processed/)**: Cleaned datasets and model outputs
  - **Key Pairs Files:**
    - `ipip_pairs_randomized.jsonl` - 41,723 randomized training pairs
    - `ipip_holdout_items.csv` - Holdout validation items
    - `ipip_train_pairs_holdout.jsonl` - Training pairs excluding holdout items
  - **Leadership Data:**
    - `leadership_focused_clean.csv` - Cleaned leadership items
    - `leadership_ipip_mapping.csv` - Mapping of leadership to IPIP constructs
  - **Analysis Results:**
    - `ipip_construct_statistics.csv` - Construct-level statistics
    - Various `.pkl` files with embedding results

- **[`visualizations/`](#visualizations)**: Analysis outputs and figures
  - **Key Validated Results:**
    - `construct_holdout_validation/` - Final validation results
    - `enhanced_statistical_comparison/` - Statistical significance tests
    - `ivan_analysis/` - Methodology comparison results

- **[`metadata/`](metadata/)**: Dataset documentation
  - `leadership_measures_data_description.md` - Detailed data documentation

## Key Visualizations Directory

The `visualizations/` folder contains critical analysis outputs:

### Most Important Results (June 2025)

1. **`construct_holdout_validation/top5_coherent_constructs_tsne.png`** ⭐
   - Cleanest visualization showing IPIP vs Leadership construct separation
   - Used in manuscript as primary evidence

2. **`enhanced_statistical_comparison/`**
   - Contains statistical validation (t-tests, effect sizes)
   - `enhanced_statistical_analysis_results.json` - Full statistical results

3. **`holdout_validation/`**
   - Unbiased holdout validation results
   - Shows 87.4% IPIP vs 62.9% leadership accuracy

### Historical Results
- Various model evaluations (`mnrl_evaluation_*`, `ipip_evaluation/`)
- Different training approaches (`trained_ipip_*`)
- Baseline comparisons (`big_five_*`, `leadership_styles_*`)

## Data Statistics

### IPIP Dataset
- **Items**: 3,320 personality items
- **Constructs**: 246 distinct personality constructs
- **Average items per construct**: 13.5
- **Format**: CSV with ProcessedText, StandardConstruct columns

### Leadership Dataset  
- **Items**: 434 leadership measurement items
- **Constructs**: 11 leadership styles/theories
- **Average items per construct**: 39.5
- **Format**: CSV with ProcessedText, StandardConstruct columns

## Key Findings in Data

The processed data and visualizations demonstrate:
- **87.4% accuracy** for IPIP personality construct separation (Cohen's d = 1.116)
- **62.9% accuracy** for leadership construct separation (Cohen's d = 0.368)
- **24.5 percentage point gap** indicating leadership construct overlap

## Data Processing Pipeline

1. **Raw Data** → Cleaning scripts remove stems and standardize format
2. **Cleaned Data** → Pair generation creates training samples
3. **Training Pairs** → Model training with GIST loss
4. **Trained Models** → Generate embeddings for analysis
5. **Embeddings** → Statistical analysis and visualization

## Adding New Data

When adding new leadership measures:
1. Place original files in `raw/` with clear naming
2. Document source in `metadata/` 
3. Run preprocessing to create cleaned version in `processed/`
4. Update construct counts in this README
5. Regenerate pairs if adding to training data 