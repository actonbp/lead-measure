# Ivan's Enhanced Embedding Analysis Approach (June 2025)

*Last Updated: June 2025*

This document describes the enhanced methodology developed by Ivan Hernandez for analyzing leadership measurement constructs using improved embedding techniques. This approach achieves significantly better construct separation compared to previous methods.

## Overview

Ivan's approach introduces several key improvements that dramatically increase the model's ability to distinguish between constructs:

- **Baseline Model Performance**: 81.66% probability of correctly ranking same-construct similarity
- **Ivan's Enhanced Model**: 99.43% probability of correctly ranking same-construct similarity

## Key Innovations

### 1. Randomized Pair Generation

The original approach had an ordering bias where items appearing earlier in the construct list would more frequently appear as anchors. Ivan's solution implements **triple randomization**:

```python
# First randomization: shuffle within each pair
pairs[i] = tuple(random.sample(pairs[i], len(pairs[i])))

# Second randomization: randomly swap anchor/positive
(a, b) if random.random() < 0.5 else (b, a)

# Third randomization: another pointwise swap
if random.random() < 0.5:
    anchors_positives_switched.append((positive, anchor))
```

This ensures balanced representation and removes ordering artifacts from the training data.

### 2. TSDAE Pre-training

**TSDAE (Transformer-based Sequential Denoising Auto-Encoder)** pre-training helps the model adapt to domain-specific vocabulary before contrastive learning:

- Performs unsupervised denoising on all IPIP texts
- Helps model understand survey item language patterns
- Only requires 1 epoch of pre-training
- Uses tied encoder-decoder weights for efficiency

### 3. Optimized Model Selection

Instead of general-purpose models, Ivan uses **BAAI/bge-m3**, which ranks as the best clustering model under 1B parameters on the MTEB leaderboard:

- Specifically optimized for clustering tasks
- Better suited for our construct separation objective
- CLS pooling instead of mean pooling

### 4. Enhanced Training Parameters

Several training improvements maximize GIST loss effectiveness:

- **Larger batch size**: 96 (vs. 32 previously) - critical for GIST loss
- **FP16 training**: Enables larger batches with memory efficiency
- **Incremental training**: 5 phases of 10 epochs each
- **NO_DUPLICATES batch sampling**: Ensures diversity within batches

### 5. Statistical Validation

Ivan's approach includes rigorous statistical analysis:

```
Paired t-test: t=87.103, p=0.000e+00
Cohen's d (paired) = 2.487
Probability(same > diff) = 99.43%
```

This represents a massive effect size (Cohen's d > 2.4) compared to baseline (d = 0.868).

### 6. Enhanced Visualization

Uses t-SNE with median centroids to show construct relationships:
- Perplexity tuning (15 and 30) for different granularities
- Median centroids show construct centers
- Clear visual separation of construct clusters

## Implementation in Our Framework

We've integrated Ivan's approach into our existing pipeline:

### Directory Structure
```
scripts/
├── ivan_analysis/
│   ├── build_pairs_randomized.py      # Randomized pair generation
│   ├── train_with_tsdae.py           # TSDAE + GIST training
│   └── visualize_and_analyze.py      # t-SNE + similarity analysis
└── run_ivan_analysis.sh              # Complete workflow
```

### Running the Analysis

```bash
chmod +x scripts/run_ivan_analysis.sh
./scripts/run_ivan_analysis.sh
```

This executes the complete pipeline:
1. Generates randomized pairs
2. Performs TSDAE pre-training
3. Trains with GIST loss (5 phases)
4. Creates visualizations and statistical analysis

### Output Files

```
data/
├── processed/
│   └── ipip_pairs_randomized.jsonl   # Randomized training pairs
└── visualizations/
    └── ivan_analysis/
        ├── tsne_perplexity15.png     # t-SNE visualization (fine detail)
        ├── tsne_perplexity30.png     # t-SNE visualization (global structure)
        ├── similarity_analysis.csv    # Statistical metrics
        └── model_comparison.csv       # Baseline vs trained comparison

models/
├── tsdae_pretrained/                  # TSDAE pre-trained model
└── ivan_tsdae_gist_final/            # Final trained model
```

## Applying to Leadership Data

The next step is to apply this enhanced model to leadership constructs:

```bash
python scripts/ivan_analysis/visualize_and_analyze.py \
    --model models/ivan_tsdae_gist_final \
    --dataset leadership
```

This will reveal whether leadership constructs show similar semantic overlap patterns as found in our previous analyses, but with much higher confidence due to the improved model performance.

## Key Takeaways

1. **Randomization matters**: Ordering biases in training data can significantly impact results
2. **Domain adaptation helps**: TSDAE pre-training improves domain-specific understanding
3. **Model selection is critical**: Using clustering-optimized models yields better results
4. **Batch size affects GIST loss**: Larger batches provide more negative examples
5. **Statistical validation**: Proper metrics confirm the dramatic improvement

## Future Enhancements

- Apply to leadership data with same rigor
- Explore different guide models for GIST loss
- Test on external validation sets
- Compare with other state-of-the-art clustering models