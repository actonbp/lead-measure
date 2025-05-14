"""Train a SentenceTransformer on IPIP anchor-positive pairs using GISTEmbedLoss.

Usage (from repo root):
    python scripts/train_gist_ipip.py

Outputs
    • models/gist_ipip/  – final model + checkpoints

This script trains a sentence embedding model on IPIP personality items using the GIST loss approach.
The trained model can then be used to evaluate how well it clusters personality constructs, and
potentially be applied to leadership items to assess construct distinctiveness.

Hardware Requirements:
- At least 16GB RAM recommended
- GPU acceleration recommended but not required
- Disk space for model checkpoints (~1-2GB)
"""
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
# Fix import path for training arguments
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ---------------------------------------------------------------------------
# Configuration – edit here if you want to change encoders or hyper-params
# ---------------------------------------------------------------------------
DATA_FILE = "data/processed/ipip_pairs.jsonl"  # anchor-positive JSONL
STUDENT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # student model to train
GUIDE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # guide model (frozen)

OUTPUT_DIR = "models/gist_ipip"  # where to save checkpoints and final model
BATCH_SIZE = 16  # reduced from 32 to save memory
EPOCHS = 5
LEARNING_RATE = 1e-5

# ---------------------------------------------------------------------------
# Main script – setup transformer models and train
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure our output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading data from {DATA_FILE}")
    dataset = load_dataset("json", data_files=DATA_FILE)["train"]
    print(f"Dataset has {len(dataset)} examples")

    # Initialize student (to train) and guide (frozen reference) models
    print(f"Initializing models:\n - Student: {STUDENT_MODEL_NAME}\n - Guide: {GUIDE_MODEL_NAME}")
    student = SentenceTransformer(STUDENT_MODEL_NAME)
    guide = SentenceTransformer(GUIDE_MODEL_NAME)

    # Use GISTEmbedLoss for training
    loss = losses.GISTEmbedLoss(
        model=student,
        guide=guide,
        temperature=0.01,
        margin_strategy="absolute",
        margin=0.0,
    )

    # Configure training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch
        learning_rate=LEARNING_RATE,
        save_steps=0.5,  # checkpoint after each 50% of training (reduced from 20%)
        save_total_limit=1,  # keep only 1 checkpoint to save disk space
        logging_steps=0.1,  # log each 10% of training
        seed=42,  # reproducibility
    )

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=student,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )

    # Run training
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Training complete. Final model saved to {OUTPUT_DIR}")
    
    # Print a message about next steps
    print("\nNext Steps:")
    print("1. Evaluate how well this model clusters personality constructs")
    print("2. Apply this model to leadership items to test construct distinctiveness") 