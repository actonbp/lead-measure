"""Train a SentenceTransformer on IPIP anchor-positive pairs using configurable loss functions.

This script:
1. Optionally pre-trains a base model using TSDAE (Transformer-based Denoising Auto-Encoder).
2. Loads IPIP anchor-positive pairs from the comprehensive/balanced dataset.
3. Trains the model using a specified loss function (default: GISTEmbedLoss).

Supported loss functions:
- GISTEmbedLoss (default): Uses a guide model to provide training signal
- MultipleNegativesRankingLoss: Treats other batch items as negatives, efficient for small datasets
- ContrastiveLoss: Traditional contrastive learning approach with positive/negative pairs
- TripletLoss: Learns from triplets of anchor, positive, and negative examples
- CosineSimilarityLoss: Optimizes for cosine similarity between positive pairs

Usage:
    python scripts/train_ipip_mnrl.py [--loss_fn gist] [--base_model BAAI/bge-m3] [--epochs 10]

For help on all options:
    python scripts/train_ipip_mnrl.py --help

Outputs
    • models/ipip_[loss_type]_[timestamp]/  – final model + checkpoints
    • models/tsdae_[timestamp]/  – TSDAE pre-trained model (if enabled)

Hardware Requirements:
- At least 16GB RAM recommended
- GPU acceleration recommended but not required
- Disk space for model checkpoints (~1-2GB)
"""
import logging
import json
import os
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, 
    models, 
    losses, 
    InputExample,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    datasets
)
from sentence_transformers.training_args import BatchSamplers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_ipip_pairs(file_path):
    """Load IPIP anchor-positive pairs from JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            examples.append(InputExample(
                texts=[data['anchor'], data['positive']], 
                label=1.0
            ))
    return examples

def get_loss_function(loss_name, model, **kwargs):
    """Get the appropriate loss function based on name."""
    loss_name = loss_name.lower()
    
    if loss_name == 'gist' or loss_name == 'gistembedloss':
        guide_model = kwargs.get('guide_model')
        if guide_model is None:
            guide_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Using default guide model: all-MiniLM-L6-v2")
        return losses.GISTEmbedLoss(model, guide_model)
    
    elif loss_name == 'mnrl' or loss_name == 'multiplenegativesrankingloss':
        return losses.MultipleNegativesRankingLoss(model=model)
    
    elif loss_name == 'contrastive' or loss_name == 'contrastiveloss':
        return losses.ContrastiveLoss(model=model)
    
    elif loss_name == 'triplet' or loss_name == 'tripletloss':
        margin = kwargs.get('triplet_margin', 0.5)
        logger.info(f"Using TripletLoss with margin={margin}")
        return losses.TripletLoss(model=model, triplet_margin=margin)
    
    elif loss_name == 'cosine' or loss_name == 'cosinesimilarityloss':
        return losses.CosineSimilarityLoss(model=model)
    
    else:
        logger.warning(f"Unknown loss function '{loss_name}'. Using GISTEmbedLoss as default.")
        guide_model = SentenceTransformer("all-MiniLM-L6-v2")
        return losses.GISTEmbedLoss(model, guide_model)

def load_all_ipip_items(file_path="data/IPIP.csv"):
    """Load all IPIP items for TSDAE pretraining."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")
    
    if 'text' not in df.columns:
        logger.error(f"'text' column not found in {file_path}")
        return []
    
    # Filter out invalid items
    valid_items = [text for text in df['text'].dropna().tolist() 
                  if isinstance(text, str) and len(text.strip()) > 3]
    
    logger.info(f"Loaded {len(valid_items)} valid items for pretraining")
    return valid_items

def tsdae_pretrain(base_model, all_items, tsdae_epochs=1, batch_size=16, output_path=None):
    """Perform TSDAE pretraining on the model."""
    if not all_items:
        logger.warning("No items provided for TSDAE pretraining. Skipping.")
        return SentenceTransformer(base_model)
    
    logger.info(f"Starting TSDAE pretraining with {len(all_items)} items for {tsdae_epochs} epochs")
    
    # Initialize model
    word_emb = models.Transformer(base_model)
    pooling = models.Pooling(word_emb.get_word_embedding_dimension(), "cls")
    tsdae_model = SentenceTransformer(modules=[word_emb, pooling])
    
    # Prepare dataset
    train_dataset = datasets.DenoisingAutoEncoderDataset(all_items)
    tsdae_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup loss
    tsdae_loss = losses.DenoisingAutoEncoderLoss(
        model=tsdae_model,
        tie_encoder_decoder=True,
    )
    
    # Train model
    tsdae_model.fit(
        train_objectives=[(tsdae_dataloader, tsdae_loss)],
        epochs=tsdae_epochs,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        output_path=output_path
    )
    
    logger.info("TSDAE pretraining completed")
    return tsdae_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SentenceTransformer on IPIP with configurable parameters")
    
    # Model parameters
    parser.add_argument('--base_model', type=str, default="BAAI/bge-m3",
                        help="Base model to use for fine-tuning")
    parser.add_argument('--pooling_mode', type=str, default="cls", choices=["mean", "max", "cls"],
                        help="Pooling strategy for embeddings")
    parser.add_argument('--guide_model', type=str, default="all-MiniLM-L6-v2",
                        help="Guide model to use with GISTEmbedLoss")
    
    # Loss function parameters
    parser.add_argument('--loss_fn', type=str, default="gist", 
                        choices=["gist", "mnrl", "contrastive", "triplet", "cosine"],
                        help="Loss function to use for training")
    parser.add_argument('--triplet_margin', type=float, default=0.5,
                        help="Margin for triplet loss (only used if loss_fn=triplet)")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=96,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs per phase (total epochs = epochs * num_phases)")
    parser.add_argument('--num_phases', type=int, default=5,
                        help="Number of training phases, each with 'epochs' epochs")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help="Ratio of warmup steps to total steps")
    parser.add_argument('--checkpoint_steps', type=int, default=None,
                        help="Save checkpoint every N steps (default: once per epoch)")
    parser.add_argument('--use_fp16', action='store_true',
                        help="Use mixed precision training (fp16)")
    
    # TSDAE pretraining parameters
    parser.add_argument('--tsdae_pretrain', action='store_true',
                        help="Perform TSDAE pretraining before fine-tuning")
    parser.add_argument('--tsdae_epochs', type=int, default=1,
                        help="Number of epochs for TSDAE pretraining")
    parser.add_argument('--tsdae_batch_size', type=int, default=16,
                        help="Batch size for TSDAE pretraining")
    
    # Data parameters
    parser.add_argument('--train_file', type=str, default="data/processed/ipip_pairs_comprehensive.jsonl",
                        help="Path to training data file")
    parser.add_argument('--ipip_file', type=str, default="data/IPIP.csv",
                        help="Path to IPIP items file (for TSDAE pretraining)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up output path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"models/ipip_{args.loss_fn}_{timestamp}"
    
    # Create output directories if they don't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Save training configuration
    config = vars(args)
    config['timestamp'] = timestamp
    
    with open(f"{output_path}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Check for GPU
    if torch.cuda.is_available():
        logger.info("CUDA device is available. Using GPU.")
        device = "cuda"
    elif torch.backends.mps.is_available():
        logger.info("MPS device is available. Using MPS.")
        device = "mps"
    else:
        logger.info("No GPU found. Using CPU.")
        device = "cpu"
    
    # Step 1: TSDAE Pretraining (if enabled)
    if args.tsdae_pretrain:
        logger.info("Starting TSDAE pretraining phase...")
        tsdae_output_path = f"models/tsdae_{timestamp}"
        os.makedirs(tsdae_output_path, exist_ok=True)
        
        # Load all items for TSDAE
        all_items = load_all_ipip_items(args.ipip_file)
        
        # Perform TSDAE pretraining
        model = tsdae_pretrain(
            args.base_model, 
            all_items, 
            tsdae_epochs=args.tsdae_epochs, 
            batch_size=args.tsdae_batch_size,
            output_path=tsdae_output_path
        )
    else:
        # Create standard SentenceTransformer model without pretraining
        logger.info(f"Initializing model: {args.base_model}")
        word_embedding_model = models.Transformer(args.base_model)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                      pooling_mode=args.pooling_mode)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    model.to(device)
    
    # Step 2: Prepare for fine-tuning
    logger.info(f"Loading data from {args.train_file}")
    train_examples = load_ipip_pairs(args.train_file)
    logger.info(f"Dataset has {len(train_examples)} examples")
    config['num_examples'] = len(train_examples)
    
    # Update config with number of examples
    with open(f"{output_path}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Step 3: Multi-phase fine-tuning with SentenceTransformerTrainer
    logger.info(f"Starting {args.num_phases} training phases, each with {args.epochs} epochs")
    
    # Initialize guide model for GIST loss if needed
    guide_model = None
    if args.loss_fn == 'gist':
        guide_model = SentenceTransformer(args.guide_model)
        logger.info(f"Initialized guide model: {args.guide_model}")
    
    # Create a dataset for SentenceTransformerTrainer
    from datasets import Dataset
    
    # Extract the text pairs from train_examples
    anchor_texts = []
    positive_texts = []
    for example in train_examples:
        anchor_texts.append(example.texts[0])
        positive_texts.append(example.texts[1])
    
    train_dataset = Dataset.from_dict({
        "anchor": anchor_texts,
        "positive": positive_texts,
    })
    
    for phase in range(args.num_phases):
        phase_output_path = f"{output_path}/phase_{phase+1}"
        os.makedirs(phase_output_path, exist_ok=True)
        
        logger.info(f"Starting training phase {phase+1}/{args.num_phases}...")
        
        # Configure training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=phase_output_path,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            fp16=args.use_fp16,
            bf16=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="epoch" if args.checkpoint_steps is None else "steps",
            save_steps=args.checkpoint_steps if args.checkpoint_steps else 500,
        )
        
        # Get the appropriate loss function
        if args.loss_fn == 'gist':
            loss = losses.GISTEmbedLoss(model, guide_model)
        else:
            loss = get_loss_function(args.loss_fn, model, triplet_margin=args.triplet_margin)
        
        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=loss,
            args=training_args
        )
        
        logger.info(f"Phase {phase+1} training parameters: Loss: {args.loss_fn}, "
                   f"Batch Size: {args.batch_size}, Epochs: {args.epochs}")
        
        # Train for this phase
        trainer.train()
        
        # Save the model after each phase
        model.save(f"{phase_output_path}/model")
        
        # Visualize results after each phase (except the last one)
        if phase < args.num_phases - 1:
            logger.info(f"Phase {phase+1} complete. Model saved to {phase_output_path}")
            logger.info(f"Starting phase {phase+2}...")
        
    # Save the final model to the main output directory
    model.save(output_path)
    logger.info(f"All training phases complete. Final model saved to {output_path}")

if __name__ == "__main__":
    main() 