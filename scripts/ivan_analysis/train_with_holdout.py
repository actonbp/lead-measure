#!/usr/bin/env python3
"""
Enhanced training script with holdout validation - optimized for Mac Studio M1/M2.
Trains only on the training split to enable fair comparison with leadership data.

Key differences from original:
1. Uses ipip_train_pairs_holdout.jsonl (80% of data) instead of all pairs
2. Loads only training items for TSDAE pre-training
3. Includes enhanced optimizations: larger batch sizes, better guide model
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
import argparse

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
        logging.FileHandler('train_with_holdout.log')
    ]
)
logger = logging.getLogger(__name__)

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

# Configuration
TRAIN_PAIRS_FILE = "data/processed/ipip_train_pairs_holdout.jsonl"
HOLDOUT_ITEMS_FILE = "data/processed/ipip_holdout_items.csv"
HOLDOUT_INFO_FILE = "data/processed/ipip_holdout_info.json"
IPIP_CSV = "data/IPIP.csv"
OUTPUT_DIR = "models/ivan_holdout_gist"
TSDAE_MODEL_DIR = "models/tsdae_holdout_pretrained"

# Default training parameters
DEFAULT_TSDAE_EPOCHS = 1
DEFAULT_TSDAE_BATCH_SIZE = 4
DEFAULT_GIST_BATCH_SIZE = 32
DEFAULT_GIST_EPOCHS_PER_PHASE = 10
DEFAULT_NUM_TRAINING_PHASES = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_RATIO = 0.1

# Optimized parameters for Mac Studio (--optimized flag)
OPTIMIZED_TSDAE_EPOCHS = 3  # Better domain adaptation
OPTIMIZED_TSDAE_BATCH_SIZE = 16
OPTIMIZED_GIST_BATCH_SIZE = 96  # Leverage 64GB memory
OPTIMIZED_GUIDE_MODEL = "BAAI/bge-m3"  # Better guide model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train with holdout validation')
    parser.add_argument('--optimized', action='store_true', 
                       help='Use optimized parameters for Mac Studio 64GB')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if models exist')
    return parser.parse_args()

def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        logger.info("NLTK punkt_tab not available, trying punkt")
        nltk.download('punkt', quiet=True)

def optimize_for_mac_studio():
    """Optimize PyTorch settings for Mac Studio M1/M2."""
    # Enable MPS fallback for operations not yet implemented in MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Disable memory watermark to avoid ratio errors (0.0 = no limit)
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    if torch.backends.mps.is_available():
        logger.info("✓ MPS (Metal Performance Shaders) available - optimized for Mac Studio")
        logger.info("  - MPS fallback enabled for unsupported operations")
        logger.info("  - Memory watermark disabled (using full 64GB unified memory)")
    else:
        logger.warning("MPS not available, falling back to CPU")
    
    torch.set_num_threads(8)
    logger.info("Mac Studio optimizations applied")

def load_training_texts() -> List[str]:
    """Load only training texts for TSDAE pre-training (excluding holdout)."""
    logger.info("Loading training texts (excluding holdout)...")
    
    # Load full IPIP data
    df_full = pd.read_csv(IPIP_CSV, encoding="latin-1").dropna()
    
    # Load holdout items to exclude
    df_holdout = pd.read_csv(HOLDOUT_ITEMS_FILE)
    holdout_texts = set(df_holdout['text'].tolist())
    
    # Get only training texts
    train_texts = [text for text in df_full['text'].unique() 
                   if text not in holdout_texts]
    
    logger.info(f"Loaded {len(train_texts)} training texts (excluded {len(holdout_texts)} holdout texts)")
    return train_texts

def load_train_pairs(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """Load anchor-positive pairs from training JSONL file."""
    logger.info(f"Loading training pairs from {jsonl_path}")
    anchors = []
    positives = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            anchors.append(data['anchor'])
            positives.append(data['positive'])
    
    logger.info(f"Loaded {len(anchors)} training pairs")
    return anchors, positives

def load_holdout_info():
    """Load and display holdout split information."""
    with open(HOLDOUT_INFO_FILE, 'r') as f:
        info = json.load(f)
    
    logger.info("\n=== Holdout Split Information ===")
    logger.info(f"Train items: {info['n_train_items']} ({info['train_size']*100:.0f}%)")
    logger.info(f"Holdout items: {info['n_holdout_items']} ({(1-info['train_size'])*100:.0f}%)")
    logger.info(f"Training pairs: {info['n_train_pairs']}")
    logger.info(f"Constructs: {info['n_constructs']}")
    
    return info

def perform_tsdae_pretraining(
    model_name: str,
    texts: List[str],
    output_dir: str,
    epochs: int,
    batch_size: int,
    use_fp16: bool = False
) -> SentenceTransformer:
    """Perform TSDAE pre-training on training data only."""
    logger.info(f"Starting TSDAE pre-training with {epochs} epochs, batch size {batch_size}...")
    
    # Determine device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Build model components
    word_embedding = models.Transformer(model_name)
    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(), 
        pooling_mode="cls"
    )
    model = SentenceTransformer(modules=[word_embedding, pooling], device=device)
    
    # Prepare dataset for denoising
    train_dataset = datasets.DenoisingAutoEncoderDataset(texts)
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # TSDAE loss
    loss = losses.DenoisingAutoEncoderLoss(
        model=model,
        tie_encoder_decoder=True,
    )
    
    # Train
    logger.info(f"Training TSDAE for {epochs} epochs...")
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        use_amp=use_fp16,
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
    guide_model_name: str,
    output_dir: str,
    batch_size: int,
    epochs_per_phase: int,
    num_phases: int,
    learning_rate: float,
    warmup_ratio: float,
    use_fp16: bool = False
):
    """Train with GIST loss on training data only."""
    logger.info(f"Starting GIST loss training with batch size {batch_size}...")
    
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
        num_train_epochs=epochs_per_phase,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=use_fp16,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=max(1, 100 // batch_size),  # Adjust logging frequency
        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_total_limit=2,
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
    args = parse_args()
    
    logger.info("🚀 Starting Holdout Validation Training")
    
    if args.optimized:
        logger.info("Using OPTIMIZED parameters for Mac Studio 64GB")
        optimize_for_mac_studio()
        tsdae_epochs = OPTIMIZED_TSDAE_EPOCHS
        tsdae_batch_size = OPTIMIZED_TSDAE_BATCH_SIZE
        gist_batch_size = OPTIMIZED_GIST_BATCH_SIZE
        guide_model = OPTIMIZED_GUIDE_MODEL
        model_name = "BAAI/bge-m3"
    else:
        logger.info("Using default parameters")
        tsdae_epochs = DEFAULT_TSDAE_EPOCHS
        tsdae_batch_size = DEFAULT_TSDAE_BATCH_SIZE
        gist_batch_size = DEFAULT_GIST_BATCH_SIZE
        guide_model = "all-MiniLM-L6-v2"
        model_name = "BAAI/bge-m3"
    
    # Download NLTK data
    download_nltk_data()
    
    # Create output directories
    Path("models").mkdir(exist_ok=True)
    
    # Load holdout information
    holdout_info = load_holdout_info()
    
    # Load training data only
    train_texts = load_training_texts()
    anchors, positives = load_train_pairs(TRAIN_PAIRS_FILE)
    
    # Step 1: TSDAE pre-training
    if Path(TSDAE_MODEL_DIR).exists() and not args.force_retrain:
        logger.info(f"Loading existing TSDAE model from {TSDAE_MODEL_DIR}")
        model = SentenceTransformer(TSDAE_MODEL_DIR)
    else:
        model = perform_tsdae_pretraining(
            model_name=model_name,
            texts=train_texts,
            output_dir=TSDAE_MODEL_DIR,
            epochs=tsdae_epochs,
            batch_size=tsdae_batch_size,
            use_fp16=False  # MPS doesn't support FP16
        )
    
    # Step 2: GIST loss training
    model = train_with_gist_loss(
        model=model,
        anchors=anchors,
        positives=positives,
        guide_model_name=guide_model,
        output_dir=OUTPUT_DIR,
        batch_size=gist_batch_size,
        epochs_per_phase=DEFAULT_GIST_EPOCHS_PER_PHASE,
        num_phases=DEFAULT_NUM_TRAINING_PHASES,
        learning_rate=DEFAULT_LEARNING_RATE,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        use_fp16=False
    )
    
    logger.info("\n🎉 Holdout validation training complete!")
    logger.info(f"Final model saved to: {OUTPUT_DIR}_final")
    logger.info("\nThis model was trained on:")
    logger.info(f"- {holdout_info['n_train_items']} training items ({holdout_info['train_size']*100:.0f}%)")
    logger.info(f"- {holdout_info['n_train_pairs']} training pairs")
    logger.info(f"- Excluded {holdout_info['n_holdout_items']} holdout items for fair validation")
    logger.info("\nNext step: Run validate_holdout_results.py to compare performance")

if __name__ == "__main__":
    main()