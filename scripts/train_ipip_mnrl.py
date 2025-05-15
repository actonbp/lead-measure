"""Train a SentenceTransformer on IPIP anchor-positive pairs using configurable loss functions.

This script:
1. Loads a SentenceTransformer model to be fine-tuned.
2. Loads IPIP anchor-positive pairs from the comprehensive/balanced dataset.
3. Trains the model using a specified loss function (default: MultipleNegativesRankingLoss).

Supported loss functions:
- MultipleNegativesRankingLoss (default): Treats other batch items as negatives, efficient for small datasets
- ContrastiveLoss: Traditional contrastive learning approach with positive/negative pairs
- TripletLoss: Learns from triplets of anchor, positive, and negative examples
- CosineSimilarityLoss: Optimizes for cosine similarity between positive pairs

Usage:
    python scripts/train_ipip_mnrl.py [--loss_fn mnrl] [--base_model all-mpnet-base-v2] [--epochs 10]

For help on all options:
    python scripts/train_ipip_mnrl.py --help

Outputs
    • models/ipip_[loss_type]_[timestamp]/  – final model + checkpoints

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
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
from pathlib import Path

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
    
    if loss_name == 'mnrl' or loss_name == 'multiplenegativesrankingloss':
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
        logger.warning(f"Unknown loss function '{loss_name}'. Using MultipleNegativesRankingLoss as default.")
        return losses.MultipleNegativesRankingLoss(model=model)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SentenceTransformer on IPIP with configurable parameters")
    
    # Model parameters
    parser.add_argument('--base_model', type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Base model to use for fine-tuning")
    parser.add_argument('--pooling_mode', type=str, default="mean", choices=["mean", "max", "cls"],
                        help="Pooling strategy for embeddings")
    
    # Loss function parameters
    parser.add_argument('--loss_fn', type=str, default="mnrl", 
                        choices=["mnrl", "contrastive", "triplet", "cosine"],
                        help="Loss function to use for training")
    parser.add_argument('--triplet_margin', type=float, default=0.5,
                        help="Margin for triplet loss (only used if loss_fn=triplet)")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help="Ratio of warmup steps to total steps")
    parser.add_argument('--checkpoint_steps', type=int, default=None,
                        help="Save checkpoint every N steps (default: once per epoch)")
    
    # Data parameters
    parser.add_argument('--train_file', type=str, default="data/processed/ipip_pairs_comprehensive.jsonl",
                        help="Path to training data file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up output path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"models/ipip_{args.loss_fn}_{timestamp}"
    
    # Create output dir if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save training configuration
    config = vars(args)
    config['timestamp'] = timestamp
    
    with open(f"{output_path}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    logger.info(f"Loading data from {args.train_file}")
    train_examples = load_ipip_pairs(args.train_file)
    logger.info(f"Dataset has {len(train_examples)} examples")
    config['num_examples'] = len(train_examples)
    
    # Update config with number of examples
    with open(f"{output_path}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare model
    logger.info(f"Initializing model: {args.base_model}")
    
    # Create SentenceTransformer model
    word_embedding_model = models.Transformer(args.base_model)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                  pooling_mode=args.pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
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
    
    model.to(device)
    
    # Prepare dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Get the specified loss function
    train_loss = get_loss_function(args.loss_fn, model, triplet_margin=args.triplet_margin)
    
    # Calculate total steps and warmup steps
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    checkpoint_steps = args.checkpoint_steps or len(train_dataloader)
    
    logger.info(f"Training parameters: Loss: {args.loss_fn}, Batch Size: {args.batch_size}, "
               f"Epochs: {args.epochs}, LR: {args.learning_rate}")
    logger.info(f"Warmup Steps: {warmup_steps}, Total Steps: {len(train_dataloader) * args.epochs}")
    
    # Train the model
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.learning_rate},
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_path=output_path,
        checkpoint_save_steps=checkpoint_steps,
        optimizer_class=torch.optim.AdamW
    )
    
    logger.info(f"Training complete. Model saved to {output_path}")

if __name__ == "__main__":
    main() 