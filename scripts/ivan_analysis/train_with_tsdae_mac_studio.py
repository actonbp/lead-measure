I think, back to your choice, I think you run the script that's like Mac optimized or whatever, and you go step by step. So it's not necessarily the full pipeline, although we'll add to the documentation later that that pipeline is available. And maybe we'll have an argument to say like what kind of computer do you have, Mac or PC or whatever, and that can be something they put in. But like, if I want to give this to a colleague, they should be able to just run the entire pipeline and kind of have directions to do stuff. But like anyways, you know, I think for now, we just want to run the Mac optimized for me, Mac studio optimized. We'll just say Mac optimized.#!/usr/bin/env python3
"""
Enhanced training script optimized for Mac Studio M1/M2 with 64GB unified memory.
Takes advantage of large memory capacity for bigger batch sizes and faster training.
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
        logging.FileHandler('train_with_tsdae_mac_studio.log')
    ]
)
logger = logging.getLogger(__name__)

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

# Configuration
PAIRS_FILE = "data/processed/ipip_pairs_randomized.jsonl"
IPIP_CSV = "data/IPIP.csv"
OUTPUT_DIR = "models/ivan_mac_studio_gist"
TSDAE_MODEL_DIR = "models/tsdae_mac_studio_pretrained"

# Training parameters optimized for Mac Studio M1/M2 64GB
TSDAE_EPOCHS = 1
TSDAE_BATCH_SIZE = 16  # Increased from 4 for 64GB memory
GIST_BATCH_SIZE = 128  # Increased from 32 for 64GB memory  
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

def optimize_for_mac_studio():
    """Optimize PyTorch settings for Mac Studio M1/M2."""
    # Enable MPS if available
    if torch.backends.mps.is_available():
        logger.info("✓ MPS (Metal Performance Shaders) available - optimized for Mac Studio")
        # Set memory management for large unified memory
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'  # Use up to 70% of memory (conservative)
    else:
        logger.warning("MPS not available, falling back to CPU")
    
    # Optimize for large memory systems
    torch.set_num_threads(8)  # Utilize M1/M2 performance cores
    
    logger.info(f"Available memory: ~64GB unified memory")
    logger.info(f"Batch sizes optimized for Mac Studio: TSDAE={TSDAE_BATCH_SIZE}, GIST={GIST_BATCH_SIZE}")

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
    Perform TSDAE pre-training optimized for Mac Studio.
    """
    logger.info("Starting TSDAE pre-training (Mac Studio optimized)...")
    
    # Build model components
    word_embedding = models.Transformer(model_name)
    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(), 
        pooling_mode="cls"
    )
    model = SentenceTransformer(modules=[word_embedding, pooling])
    
    # Prepare dataset for denoising - use larger chunks with 64GB memory
    train_dataset = datasets.DenoisingAutoEncoderDataset(texts)
    dataloader = DataLoader(
        train_dataset, 
        batch_size=TSDAE_BATCH_SIZE, 
        shuffle=True,
        num_workers=4,  # Utilize multiple cores
        pin_memory=True  # Optimize for unified memory
    )
    
    # TSDAE loss with tied encoder-decoder weights
    loss = losses.DenoisingAutoEncoderLoss(
        model=model,
        tie_encoder_decoder=True,
    )
    
    # Train with optimized settings
    logger.info(f"Training TSDAE for {TSDAE_EPOCHS} epochs with batch size {TSDAE_BATCH_SIZE}...")
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=TSDAE_EPOCHS,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        use_amp=USE_FP16,  # Mixed precision for memory efficiency
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
    Train with GIST loss optimized for Mac Studio 64GB memory.
    """
    logger.info("Starting GIST loss training (Mac Studio optimized)...")
    
    # Load guide model
    guide_model = SentenceTransformer(guide_model_name)
    
    # Create dataset
    train_dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
    })
    
    # Training arguments optimized for Mac Studio
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=GIST_EPOCHS_PER_PHASE,
        per_device_train_batch_size=GIST_BATCH_SIZE,  # Much larger with 64GB
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=USE_FP16,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=25,  # More frequent logging with larger batches
        seed=42,
        dataloader_num_workers=4,  # Utilize M1/M2 cores
        dataloader_pin_memory=True,  # Optimize for unified memory
        save_total_limit=2,  # Keep fewer checkpoints to save space
    )
    
    # GIST loss
    loss = losses.GISTEmbedLoss(model, guide_model)
    
    # Train in phases with enhanced monitoring
    for phase in range(num_phases):
        logger.info(f"\nTraining phase {phase + 1}/{num_phases}")
        logger.info(f"Using batch size: {GIST_BATCH_SIZE} (optimized for 64GB unified memory)")
        
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
        
        # Log memory usage (helpful for monitoring)
        if torch.backends.mps.is_available():
            logger.info(f"Phase {phase + 1} complete - MPS optimized training")
    
    # Save final model
    final_output_dir = f"{output_dir}_final"
    model.save(final_output_dir)
    logger.info(f"Training complete. Final model saved to {final_output_dir}")
    
    return model

def main():
    """Main execution function optimized for Mac Studio."""
    logger.info("🚀 Starting Ivan's Enhanced Analysis on Mac Studio M1/M2 (64GB)")
    
    # Optimize for Mac Studio
    optimize_for_mac_studio()
    
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
    
    logger.info("\n🎉 Mac Studio optimized training pipeline complete!")
    logger.info(f"Final model saved to: {OUTPUT_DIR}_final")
    logger.info(f"Intermediate checkpoints: {OUTPUT_DIR}_phase1 through {OUTPUT_DIR}_phase{NUM_TRAINING_PHASES}")
    logger.info("\nNext steps:")
    logger.info("1. Run analysis: python scripts/ivan_analysis/run_analysis_steps.py --step 3")
    logger.info("2. Compare baseline: python scripts/ivan_analysis/run_analysis_steps.py --step 4")

if __name__ == "__main__":
    main()