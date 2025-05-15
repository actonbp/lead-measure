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

## 5. Potential Next Steps (Post-Current Run)

1.  **Evaluate the Current `Mistral` Model**:
    *   Once the 60-epoch training completes, we will run `scripts/evaluate_trained_ipip_model.py` to get ARI, NMI, and Purity scores.
    *   We will also examine the t-SNE plots to visually assess cluster separation.

2.  **If `Mistral` Shows Significant Improvement & Continued Learning Trend**:
    *   Consider training for more epochs (e.g., 100-150) if the loss is still decreasing and metrics are improving.

3.  **If `Mistral` Plateaus or Shows Modest Improvement**:
    *   **Try Another SOTA Model**: Consult the MTEB leaderboard again for other high-performing models that are compatible with `sentence-transformers` and `GISTEmbedLoss`.
    *   **Re-evaluate Data**:
        *   Examine the `data/IPIP.csv` and `data/processed/ipip_pairs.jsonl`.
        *   Are there ambiguities in the items or constructs?
        *   Could data augmentation (e.g., paraphrasing items) be beneficial?
        *   Is the "anchor-positive" pair generation strategy optimal?
    *   **Explore Different Loss Functions**: While GISTEmbedLoss is powerful for distillation, other contrastive or triplet losses available in `sentence-transformers` might yield better separation for this specific task if GIST is not proving effective enough.
    *   **Hyperparameter Tuning**: Beyond epochs and learning rate, explore other hyperparameters like weight decay, different optimizer parameters, or more complex scheduler configurations if a promising model is identified but needs further refinement.

This iterative process of training, evaluating, and adjusting is key to finding the optimal approach for this task. 