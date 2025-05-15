# Improved Leadership Embedding Analysis Workflow (May 2025)

*Last Updated: May 15, 2025*

This document outlines the improved methodology for analyzing leadership measurement constructs using natural language processing and embedding techniques. These improvements address the performance bottlenecks identified in our initial approach.

## Key Improvements

Our analysis identified that the main bottlenecks in model performance were related to:

1. **Data Quality & Quantity**: Limited number of pairs for training
2. **Training Signal**: Imbalanced representation of constructs
3. **Loss Function**: GISTEmbedLoss may not be optimal for our dataset size

The improved workflow addresses these issues with:

1. **Comprehensive Pair Generation**: Generate all possible within-construct pairs rather than just one per item
2. **Balanced Training Data**: Rebalancing strategies to prevent over/under-representation of constructs
3. **Alternative Loss Functions**: Using MultipleNegativesRankingLoss which works better with limited data

## Workflow Overview

The complete workflow is automated in `scripts/run_improved_workflow.sh` and consists of these steps:

1. **Generate comprehensive anchor-positive pairs**
   - Script: `build_ipip_pairs_improved.py`
   - Generates all possible within-construct pairs from IPIP items
   - Implements rebalancing to ensure fair representation across constructs
   - Output: `data/processed/ipip_pairs_comprehensive.jsonl`

2. **Train model with MultipleNegativesRankingLoss**
   - Script: `train_ipip_mnrl.py`
   - Uses the comprehensive pairs dataset
   - Applies MultipleNegativesRankingLoss for more effective training with limited data
   - Output: `models/ipip_mnrl/`

3. **Compare model performance** 
   - Script: `compare_model_performance.py`
   - Evaluates and compares different models on both IPIP and leadership data
   - Calculates clustering metrics (ARI, NMI, purity)
   - Outputs: Comparison metrics and visualizations

4. **Apply best model to leadership data**
   - Script: `apply_best_model_to_leadership.py`
   - Automatically selects the best performing model based on IPIP ARI score
   - Applies that model to leadership items
   - Generates visualizations and analysis of leadership construct clustering

## Running the Improved Workflow

Simply run:

```bash
chmod +x scripts/run_improved_workflow.sh
./scripts/run_improved_workflow.sh
```

## Key Scripts

### 1. Comprehensive Pair Generation (`build_ipip_pairs_improved.py`)

This script addresses data quality by:
- Generating all possible within-construct pairs instead of just one per item
- Implementing rebalancing strategies to prevent construct over/under-representation
- Supporting different rebalancing methods:
  - `sampling`: Targets a balanced number of pairs per construct
  - `cap`: Simple cap on pairs per construct
  - `both`: Combined approach

### 2. Alternative Loss Function Training (`train_ipip_mnrl.py`)

This script improves training by:
- Using `MultipleNegativesRankingLoss` instead of `GISTEmbedLoss`
- Each positive pair in a batch implicitly uses all other sentences as negatives
- Enables more efficient use of batch data for stronger training signal
- Works better with smaller datasets by maximizing the contrastive signal

### 3. Model Comparison (`compare_model_performance.py`)

This script evaluates different approaches by:
- Comparing multiple models on both IPIP and leadership data
- Calculating clustering metrics (ARI, NMI, purity) for each approach
- Generating visualizations to illustrate performance differences
- Providing quantitative evidence for which approach works best

### 4. Best Model Application (`apply_best_model_to_leadership.py`)

This script tests our research hypothesis by:
- Automatically selecting the best-performing model on IPIP data
- Applying it to leadership items
- Calculating metrics that indicate whether leadership constructs form distinct clusters
- Generating visualizations and detailed analysis

## Evaluation Metrics

To evaluate how well our models separate constructs, we use:

- **Adjusted Rand Index (ARI)**: Measures how well the clustering matches the true labels
  - Perfect match = 1.0
  - Random match = 0.0
  - Negative values indicate worse than random

- **Normalized Mutual Information (NMI)**: Measures the mutual information between clusters and true labels
  - Higher values (towards 1.0) indicate better alignment
  
- **Cluster Purity**: For each cluster, what fraction belongs to the most common construct
  - Higher values indicate more homogeneous clusters

## Expected Results

If our hypothesis is correct:
- IPIP personality constructs should form relatively distinct clusters (higher metrics)
- Leadership constructs will show substantial overlap (lower metrics)

This approach provides a more robust test of construct distinctiveness than our previous method. 