# Project Status Update: IPIP Construct Embedding Model

## 1. Overall Goal

The primary objective of this project is to develop a sentence embedding model that can effectively understand the semantic relationships between IPIP (International Personality Item Pool) items and their corresponding personality constructs. The ultimate aim is to have a model that can accurately group or cluster IPIP items based on their underlying constructs, which will later be tested on leadership items.

## 2. What We've Tried So Far

We have iteratively experimented with different model architectures, training configurations, and hyperparameters.

### a. Initial Model: `sentence-transformers/all-mpnet-base-v2`
*   **Initial Training**: Started with 5 epochs and a batch size of 32.
    *   *Metrics (5 epochs)*: ARI: 0.0609, NMI: 0.7658, Purity: 0.3932
*   **Increased Epochs**:
    *   *15 epochs*: ARI: 0.0687, NMI: 0.7849, Purity: 0.4068
    *   *50 epochs*: ARI: 0.0768, NMI: 0.7998, Purity: 0.4166
    *   *200 epochs*: The model was trained, but evaluation metrics started showing diminishing returns and potential signs of overfitting or hitting the model's capacity with this dataset and loss function.
*   **Observation**: The `all-mpnet-base-v2` model showed initial learning, but performance gains plateaued, indicating a need for a more powerful base model or different training strategies.

### b. Upgraded Model: `Snowflake/snowflake-arctic-embed-m-v1.5`
*   **Training Configuration**: Introduced a cosine learning rate scheduler with warmup.
    *   *25 epochs (Batch Size 32)*: ARI: 0.0644, NMI: 0.7879, Purity: 0.4016
    *   *100 epochs (Batch Size 32)*: ARI: 0.0661, NMI: 0.7992, Purity: 0.4005
*   **Observation**: The Snowflake model, while powerful, did not immediately outperform the later stages of the `all-mpnet-base-v2` model on our specific task with these configurations. The metrics seemed to plateau again, suggesting that simply more epochs with this setup might not be the most effective path.

### c. Current Advanced Model Attempt: `Salesforce/SFR-Embedding-Mistral`
This is one of the current state-of-the-art (SOTA) embedding models.
*   **Initial Setup**:
    *   Student Model: `Salesforce/SFR-Embedding-Mistral`
    *   Guide Model: `sentence-transformers/all-MiniLM-L6-v2` (for GISTEmbedLoss)
    *   Epochs: Target 60
    *   Learning Rate: 1e-5
    *   Learning Rate Scheduler: `cosine_with_restarts`
    *   Initial Batch Size: 32, Gradient Accumulation: 2 (Effective: 64)
*   **Troubleshooting & Adjustments**:
    *   Encountered several `TypeError` issues with `GISTEmbedLoss` and `SentenceTransformerTrainingArguments` parameters, which were resolved by correcting parameter names (e.g., `model` instead of `student_model`, `guide` instead of `guide_model`, `lr_scheduler_type`).
    *   Installed `tensorboard` dependency.
    *   Faced MPS (Apple Silicon GPU) out-of-memory (OOM) errors.
        *   Attempt 1: Enabled `gradient_checkpointing=True`. Still OOM.
        *   Attempt 2 (Current): Reduced `BATCH_SIZE` from 32 to 16 (effective batch size now 32), while keeping `gradient_checkpointing=True`.

## 3. Summary of Key Results (Best for each model family)

| Model Base                             | Epochs | ARI    | NMI    | Purity | Notes                                      |
|----------------------------------------|--------|--------|--------|--------|--------------------------------------------|
| `all-mpnet-base-v2`                    | 50     | 0.0768 | 0.7998 | 0.4166 | Showed some learning, then plateaued.      |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | 100    | 0.0661 | 0.7992 | 0.4005 | Did not surpass `mpnet` significantly.   |
| `Salesforce/SFR-Embedding-Mistral`     | N/A    | -      | -      | -      | *Currently training*                       |

**General Observation on Metrics**: While NMI scores have been relatively high (suggesting good information overlap between predicted and true clusters), ARI and Purity scores have remained somewhat modest. This indicates that while the model captures some construct-related information, the resulting clusters are not yet highly pure or perfectly aligned with the ground truth constructs.

## 4. What We're Trying Now (Current Run)

*   **Model**: `Salesforce/SFR-Embedding-Mistral`
*   **Epochs**: 60
*   **Batch Size**: 16 (Effective: 32 due to gradient accumulation of 2)
*   **Learning Rate Scheduler**: `cosine_with_restarts`
*   **Warmup Steps**: 10% of total steps
*   **Key Optimizations**: `gradient_checkpointing=True`
*   **Status**: Currently running. This is expected to be a lengthy training process (potentially several hours).

## 5. Improved Approach (2025 Update)

After identifying performance bottlenecks in our previous approaches, we've developed an improved methodology that addresses the core limitations.

### 5.1 Root Causes of Performance Plateau

Our analysis identified that the main bottlenecks in model performance were related to:

1. **Limited Training Data**: Only ~3,800 pairs were used for training, with just one positive example per anchor.
2. **Imbalanced Representation**: Constructs with more items were overrepresented in training.
3. **Loss Function Limitations**: GISTEmbedLoss might not be optimal for the size and structure of our dataset.

### 5.2 New Comprehensive Approach

We've implemented a series of improvements:

1. **Comprehensive Pair Generation**:
   * New script: `build_ipip_pairs_improved.py`
   * Generates *all possible* within-construct pairs, not just one per item
   * Increases training signal dramatically while keeping the same underlying data

2. **Balanced Training Data**:
   * Implemented rebalancing strategies to prevent over/under-representation
   * Ensures each construct has appropriate representation in training
   * Options for sampling or capping pairs per construct

3. **Alternative Loss Function**:
   * New script: `train_ipip_mnrl.py` 
   * Uses `MultipleNegativesRankingLoss` instead of `GISTEmbedLoss`
   * Makes more efficient use of batch data by treating all non-matching examples as negatives
   * Better suited to smaller datasets like ours

4. **Automated Comparison and Evaluation**:
   * Added `compare_model_performance.py` to objectively evaluate different approaches
   * Automatically applies best model to leadership data with `apply_best_model_to_leadership.py`

### 5.3 Preliminary Results and Next Steps

The improved workflow has been implemented and is currently being tested. We expect:

1. **Higher Metrics**: Significant improvements in ARI, NMI, and Purity scores for IPIP constructs
2. **More Rigorous Test**: A more definitive answer about whether leadership constructs are truly distinct
3. **Automated Comparison**: Clear evidence about which approach works best

Full results will be added after the complete workflow is executed. For more details, see [docs/improved_workflow.md](docs/improved_workflow.md).

## 6. Potential Next Steps (Post-Improved Approach)

1. **If Improved Approach Shows Significant Gains**:
   * Apply to leadership data and analyze results
   * Explore potential construct-specific patterns in embeddings
   * Consider visualization improvements for clearer presentation

2. **If Improved Approach Still Shows Limitations**:
   * Explore more sophisticated data augmentation techniques
   * Consider task-specific pretraining strategies
   * Investigate potential fundamental limitations in the construct definitions themselves

This iterative process of training, evaluating, and adjusting is key to finding the optimal approach for this task. 