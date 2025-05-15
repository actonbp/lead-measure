"""Train a SentenceTransformer on IPIP anchor-positive pairs using GISTEmbedLoss.

This script:
1. Loads a 'student' SentenceTransformer model to be fine-tuned.
2. Loads a 'guide' SentenceTransformer model (which is frozen).
3. Loads IPIP anchor-positive pairs.
4. Trains the student model using GISTEmbedLoss, which encourages the student
   to match the guide's embeddings for the *same* text, while also minimizing
   the distance between embeddings of anchor-positive pairs.

Key features of GISTEmbedLoss:
- Leverages a pre-trained, high-quality 'guide' model without needing its architecture
  to match the 'student'.
- Efficiently distills knowledge from the guide by focusing on matching embedding
  outputs rather than internal model states.

Usage:
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
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import GISTEmbedLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset
import torch # ensure torch is imported to check for MPS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration – edit here if you want to change encoders or hyper-params
# ---------------------------------------------------------------------------
DATA_FILE = "data/processed/ipip_pairs.jsonl"  # anchor-positive JSONL
STUDENT_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"  # student model to train
GUIDE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # guide model (frozen)

OUTPUT_DIR = "models/gist_ipip_mistral_cosine_60_epochs"  # where to save checkpoints and final model
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 2 # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# ---------------------------------------------------------------------------
# Main script – setup transformer models and train
# ---------------------------------------------------------------------------
def main():
    # Ensure our output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading data from {DATA_FILE}")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    logger.info(f"Dataset has {len(dataset)} examples")

    # Initialize student (to train) and guide (frozen reference) models
    logger.info(f"Initializing models:\n - Student: {STUDENT_MODEL_NAME}\n - Guide: {GUIDE_MODEL_NAME}")
    student_model = SentenceTransformer(STUDENT_MODEL_NAME)
    guide_model = SentenceTransformer(GUIDE_MODEL_NAME)

    # Use GISTEmbedLoss for training
    loss = GISTEmbedLoss(model=student_model, guide=guide_model)

    # MPS check for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("MPS device is available. Using MPS.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"MPS not available. Using {device}.")
    student_model.to(device)
    guide_model.to(device)

    # Calculate total steps for warmup calculation if using a scheduler that needs it
    # Total steps = (num_examples / (batch_size * grad_accum)) * num_epochs
    total_steps = (len(dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS
    warmup_steps = int(total_steps * 0.1) # 10% of total steps for warmup

    logger.info(f"Training parameters: Batch Size: {BATCH_SIZE}, Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    logger.info(f"Scheduler: cosine_with_restarts, Warmup Steps: {warmup_steps}, Total Steps: {total_steps}")

    # Configure training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # accumulate gradients for 2 steps
        learning_rate=LEARNING_RATE,
        save_strategy="epoch", # Save a checkpoint at the end of each epoch
        # save_steps=0.5,  # Save every 0.5 epochs (e.g., mid-epoch and end-epoch)
        save_total_limit=2, # Only keep the last 2 checkpoints + the final model
        logging_steps=0.1, # Log every 10% of an epoch
        dataloader_num_workers=0, # Set to 0 or more if you have issues with multi-processing
        dataloader_pin_memory=True, # Might improve performance if your GPU supports it
        # LR Scheduler specific arguments
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=warmup_steps,
        metric_for_best_model="loss", # Optional: if you want to track the best model based on loss
        greater_is_better=False,     # Optional: for loss, lower is better
        report_to=["tensorboard"], # Log to tensorboard
        gradient_checkpointing=True, # Enable gradient checkpointing
    )

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset, # type: ignore
        loss=loss,
    )

    # Run training
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Training complete. Final model saved to {OUTPUT_DIR}")
    
    # Print a message about next steps
    logger.info("\nNext Steps:")
    logger.info("1. Evaluate how well this model clusters personality constructs")
    logger.info("2. Apply this model to leadership items to test construct distinctiveness")

if __name__ == "__main__":
    main() 