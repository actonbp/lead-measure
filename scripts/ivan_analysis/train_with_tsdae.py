#!/usr/bin/env python3
"""
Train IPIP embeddings using Ivan's enhanced approach with TSDAE pre-training.

Key improvements:
1. TSDAE pre-training for domain adaptation
2. BGE-M3 model optimized for clustering
3. Larger batch size (96) for better GIST loss performance
4. FP16 training for memory efficiency
5. Incremental training with multiple phases

Based on Ivan Hernandez's methodology (January 2025)
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import sys
import nltk
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    models,
    InputExample,
    SentenceTransformerTrainer,
    losses,
    util,
    datasets,
    evaluation
)
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_with_tsdae.log')
    ]
)
logger = logging.getLogger(__name__)

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

# Configuration
PAIRS_FILE = "data/processed/ipip_pairs_randomized.jsonl"
IPIP_CSV = "data/IPIP.csv"
OUTPUT_DIR = "models/ivan_tsdae_gist"
TSDAE_MODEL_DIR = "models/tsdae_pretrained"

# Training parameters (Ivan's settings - optimized for Mac Studio Metal)
TSDAE_EPOCHS = 3       # Increased for better domain adaptation
TSDAE_BATCH_SIZE = 32  # 8x increase leveraging unified memory
GIST_BATCH_SIZE = 128  # 4x increase for Metal parallel processing
GIST_EPOCHS_PER_PHASE = 10
NUM_TRAINING_PHASES = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
USE_FP16 = False  # MPS doesn't support FP16

def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        logger.info("NLTK punkt_tab not available, trying punkt")
        nltk.download('punkt', quiet=True)

def load_ipip_texts(csv_path: str) -> List[str]:
    """Load all unique IPIP texts for TSDAE pre-training."""
    logger.info(f"Loading IPIP texts from {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin-1").dropna()
    all_texts = df['text'].unique().tolist()
    logger.info(f"Loaded {len(all_texts)} unique texts")
    return all_texts

def load_pairs(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """Load anchor-positive pairs from JSONL file."""
    logger.info(f"Loading pairs from {jsonl_path}")
    anchors = []
    positives = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            anchors.append(data['anchor'])
            positives.append(data['positive'])
    
    logger.info(f"Loaded {len(anchors)} pairs")
    return anchors, positives

def perform_tsdae_pretraining(
    model_name: str = "BAAI/bge-m3",
    texts: List[str] = None,
    output_dir: str = TSDAE_MODEL_DIR
) -> SentenceTransformer:
    """
    Perform TSDAE pre-training on domain-specific texts.
    
    TSDAE (Transformer-based Sequential Denoising Auto-Encoder) helps the model
    adapt to the specific domain vocabulary and structure.
    """
    logger.info("Starting TSDAE pre-training...")
    
    # Build model components
    word_embedding = models.Transformer(model_name)
    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(), 
        pooling_mode="cls"
    )
    model = SentenceTransformer(modules=[word_embedding, pooling])
    
    # Prepare dataset for denoising
    train_dataset = datasets.DenoisingAutoEncoderDataset(texts)
    dataloader = DataLoader(train_dataset, batch_size=TSDAE_BATCH_SIZE, shuffle=True)
    
    # TSDAE loss with tied encoder-decoder weights
    loss = losses.DenoisingAutoEncoderLoss(
        model=model,
        tie_encoder_decoder=True,
    )
    
    # Train
    logger.info(f"Training TSDAE for {TSDAE_EPOCHS} epochs...")
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=TSDAE_EPOCHS,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
    )
    
    # Save the pre-trained model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    logger.info(f"TSDAE pre-training complete. Model saved to {output_dir}")
    
    return model

def train_with_gist_loss(
    model: SentenceTransformer,
    anchors: List[str],
    positives: List[str],
    guide_model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = OUTPUT_DIR,
    num_phases: int = NUM_TRAINING_PHASES
):
    """
    Train the model using GIST loss with Ivan's parameters.
    
    GIST (Guided In-batch Sampling Training) uses a guide model to help
    identify hard negatives within each batch.
    """
    logger.info("Starting GIST loss training...")
    
    # Load guide model
    guide_model = SentenceTransformer(guide_model_name)
    
    # Create dataset
    train_dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
    })
    
    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=GIST_EPOCHS_PER_PHASE,
        per_device_train_batch_size=GIST_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=USE_FP16,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=50,
        seed=42,
    )
    
    # GIST loss
    loss = losses.GISTEmbedLoss(model, guide_model)
    
    # Train in phases
    for phase in range(num_phases):
        logger.info(f"\nTraining phase {phase + 1}/{num_phases}")
        
        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=loss,
            args=args
        )
        
        trainer.train()
        
        # Save checkpoint after each phase
        phase_output_dir = f"{output_dir}_phase{phase + 1}"
        model.save(phase_output_dir)
        logger.info(f"Saved model checkpoint to {phase_output_dir}")
    
    # Save final model
    final_output_dir = f"{output_dir}_final"
    model.save(final_output_dir)
    logger.info(f"Training complete. Final model saved to {final_output_dir}")
    
    return model

def main():
    """Main execution function."""
    # Download NLTK data
    download_nltk_data()
    
    # Create output directories
    Path("models").mkdir(exist_ok=True)
    
    # Load data
    all_texts = load_ipip_texts(IPIP_CSV)
    anchors, positives = load_pairs(PAIRS_FILE)
    
    # Step 1: TSDAE pre-training
    if Path(TSDAE_MODEL_DIR).exists():
        logger.info(f"Loading existing TSDAE model from {TSDAE_MODEL_DIR}")
        model = SentenceTransformer(TSDAE_MODEL_DIR)
    else:
        model = perform_tsdae_pretraining(
            model_name="BAAI/bge-m3",
            texts=all_texts,
            output_dir=TSDAE_MODEL_DIR
        )
    
    # Step 2: GIST loss training
    model = train_with_gist_loss(
        model=model,
        anchors=anchors,
        positives=positives,
        guide_model_name="all-MiniLM-L6-v2",
        output_dir=OUTPUT_DIR,
        num_phases=NUM_TRAINING_PHASES
    )
    
    logger.info("\nTraining pipeline complete!")
    logger.info(f"Final model saved to: {OUTPUT_DIR}_final")
    logger.info(f"Intermediate checkpoints saved with suffix _phase1 through _phase{NUM_TRAINING_PHASES}")

if __name__ == "__main__":
    main()