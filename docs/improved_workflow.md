# Improved Leadership Embedding Analysis Workflow (May 2025)

*Last Updated: May 20, 2025*

This document outlines the improved methodology for analyzing leadership measurement constructs using natural language processing and embedding techniques. These improvements address the performance bottlenecks identified in our initial approach.

## Key Improvements

Our analysis identified that the main bottlenecks in model performance were related to:

1. **Data Quality & Quantity**: Limited number of pairs for training
2. **Training Signal**: Imbalanced representation of constructs
3. **Model Selection**: Previous models were not optimized for clustering tasks
4. **Pre-training Approach**: No domain adaptation for specialized text

The improved workflow addresses these issues with:

1. **Comprehensive Pair Generation**: Generate all possible within-construct pairs with randomized positions
2. **Balanced Training Data**: Rebalancing strategies to prevent over/under-representation of constructs
3. **Optimized Model Selection**: Using `BAAI/bge-m3`, a model specifically optimized for clustering
4. **TSDAE Pre-training**: Implementing denoising autoencoder pre-training for domain adaptation
5. **Improved Training Efficiency**: 
   - Larger batch sizes for GISTEmbedLoss
   - FP16 mixed precision training
   - Multi-phase incremental training

## Workflow Overview

The complete workflow is automated in `scripts/run_refined_workflow.sh` and consists of these steps:

1. **Generate comprehensive anchor-positive pairs with randomized positions**
   - Script: `build_ipip_pairs_improved.py`
   - Generates all possible within-construct pairs from IPIP items
   - Randomizes the position of anchor and positive items
   - Implements rebalancing to ensure fair representation across constructs
   - Output: `data/processed/ipip_pairs_comprehensive.jsonl`

2. **Train model with BGE-M3, TSDAE and GISTEmbedLoss**
   - Script: `train_ipip_mnrl.py`
   - Performs TSDAE pre-training for domain adaptation
   - Uses the `BAAI/bge-m3` model optimized for clustering
   - Applies GISTEmbedLoss with large batch sizes for more effective training
   - Implements multi-phase training with fp16 precision
   - Output: `models/ipip_gist_*/`

3. **Evaluate model with comprehensive metrics** 
   - Script: `evaluate_model_with_validation.py`
   - Performs k-fold cross-validation for robust metrics
   - Calculates clustering metrics (ARI, NMI, purity)
   - Generates t-SNE visualizations and confusion matrices
   - Outputs: Comprehensive evaluation metrics and visualizations

4. **Apply model to leadership data**
   - Script: `evaluate_mnrl_on_leadership.py`
   - Applies the trained model to leadership items
   - Generates visualizations and analysis of leadership construct clustering
   - Outputs: Leadership evaluation metrics and visualizations

5. **Compare embedding similarities**
   - Scripts: `evaluate_model_with_validation.py`
   - Analyzes the similarity of embeddings within vs. across constructs
   - Performs statistical tests (paired t-test, Cohen's d)
   - Calculates the probability of same-construct items having higher similarity
   - Outputs: Similarity analysis reports and visualizations

## Running the Refined Workflow

To execute the complete refined workflow:

```bash
chmod +x scripts/run_refined_workflow.sh
./scripts/run_refined_workflow.sh
```

## Key Scripts

### 1. Comprehensive Pair Generation with Randomization (`build_ipip_pairs_improved.py`)

This script addresses data quality by:
- Generating all possible within-construct pairs instead of just one per item
- Randomizing the position of anchor and positive items for better training
- Implementing rebalancing strategies to prevent construct over/under-representation
- Supporting different rebalancing methods:
  - `sampling`: Targets a balanced number of pairs per construct
  - `cap`: Simple cap on pairs per construct
  - `both`: Combined approach

### 2. Advanced Training with TSDAE Pre-training (`train_ipip_mnrl.py`)

This script implements our new training approach:
- Uses `BAAI/bge-m3` model specifically designed for clustering tasks
- Performs TSDAE pre-training for domain adaptation before fine-tuning
- Supports `GISTEmbedLoss` with configurable guide models
- Implements multi-phase training with incremental epochs for better optimization
- Larger batch sizes (96 vs. 32 previously) for more effective training
- FP16 mixed precision training for memory efficiency and faster training

### 3. Comprehensive Evaluation (`evaluate_model_with_validation.py`)

This script provides rigorous model evaluation:
- Performs k-fold cross-validation for more reliable metrics
- Tests multiple clustering algorithms for robustness
- Generates t-SNE visualizations for better interpretability
- Creates detailed confusion matrices for cluster analysis
- Calculates metrics like ARI, NMI, and purity
- Analyzes embedding similarity within and across constructs
- Performs statistical significance testing for similarity differences

### 4. Leadership Construct Analysis (`evaluate_mnrl_on_leadership.py`)

This script tests our research hypothesis by:
- Applying the trained model to leadership items
- Clustering leadership items and evaluating semantic distinctiveness
- Generating visualizations showing relationships between leadership constructs
- Calculating similarity metrics between leadership constructs
- Creating detailed reports on construct overlap and cluster composition

## Evaluation Metrics

We use multiple metrics to evaluate model performance:

### Clustering Metrics

- **Adjusted Rand Index (ARI)**: Measures how well the clustering matches the true labels
  - Perfect match = 1.0
  - Random match = 0.0
  - Negative values indicate worse than random

- **Normalized Mutual Information (NMI)**: Measures the mutual information between clusters and true labels
  - Higher values (towards 1.0) indicate better alignment
  
- **Cluster Purity**: For each cluster, what fraction belongs to the most common construct
  - Higher values indicate more homogeneous clusters

### Similarity Analysis Metrics

- **Same vs. Different Construct Similarity**: Compare cosine similarity within and across constructs
  - Higher within-construct similarity indicates better embedding quality

- **Paired t-test**: Tests if the difference between same-construct and different-construct similarities is statistically significant
  - Lower p-values indicate more confident results

- **Cohen's d**: Effect size measurement for the similarity difference
  - d > 0.8: Large effect
  - d > 0.5: Medium effect
  - d > 0.2: Small effect

- **Probability(same > diff)**: The probability that a same-construct pair has higher similarity than a different-construct pair
  - Values closer to 100% indicate better construct separation

## Expected Results

If our hypothesis is correct:
- IPIP personality constructs should form relatively distinct clusters (higher metrics)
- IPIP constructs should show statistically significant higher within-construct similarity
- Leadership constructs will show substantial overlap (lower clustering metrics)
- Leadership constructs will show less differentiation in similarity analysis

The refined approach with TSDAE pre-training, the BGE-M3 model, and improved training procedures should provide a more robust and powerful test of construct distinctiveness than our previous methods. 