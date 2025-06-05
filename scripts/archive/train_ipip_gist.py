"""Train a SentenceTransformer on IPIP anchor-positive pairs using GIST loss.

This script:
1. Loads a student SentenceTransformer model to be fine-tuned
2. Loads a guide SentenceTransformer model (smaller, faster model)
3. Loads IPIP anchor-positive pairs from the comprehensive/balanced dataset
4. Trains the student model using GISTEmbedLoss with guidance from the guide model

The GIST (Guided In-batch Similarity Training) loss approach:
- Uses a smaller, faster model as a guide model
- Helps the student model learn by providing similarity guidance
- Provides efficient training with limited data

Usage:
    python scripts/train_ipip_gist.py [--student_model all-mpnet-base-v2] [--guide_model all-MiniLM-L6-v2] [--epochs 10]

For help on all options:
    python scripts/train_ipip_gist.py --help

Outputs:
    • models/ipip_gist_[timestamp]/  – final model + checkpoints

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
    anchors = []
    positives = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Validate both anchor and positive
                if not data['anchor'] or not data['positive']:
                    logger.warning(f"Skipping pair with empty text: {data}")
                    continue
                    
                # Append to lists
                anchors.append(data['anchor'])
                positives.append(data['positive'])
                
                # Create sentence-transformers example
                examples.append(InputExample(
                    texts=[data['anchor'], data['positive']], 
                    label=1.0
                ))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
            except KeyError:
                logger.warning(f"Skipping line with missing keys: {line[:50]}...")
    
    logger.info(f"Loaded {len(examples)} valid pairs")
    return examples, anchors, positives

class GISTEmbedLoss(losses.CosineSimilarityLoss):
    """
    Implementation of GIST Embedding Loss.
    
    Uses a guide model to provide similarity guidance to a student model.
    """
    def __init__(self, model, guide_model, loss_margin=0.3):
        super(GISTEmbedLoss, self).__init__(model)
        self.guide_model = guide_model
        self.loss_margin = loss_margin
        
    def __call__(self, batch_features):
        embeddings = batch_features['sentence_embedding']  # From student model
        labels = batch_features['label']
        
        # Get embeddings for first and second sentences
        embeddings_a = embeddings[0::2]  # Even indices (0, 2, 4...)
        embeddings_b = embeddings[1::2]  # Odd indices (1, 3, 5...)
        
        # Compute cosine similarity between pairs
        cos_sim = self.similarity_fct(embeddings_a, embeddings_b)
        
        # If we have labels, compute guided loss
        if labels is not None:
            with torch.no_grad():
                batch_texts = batch_features.get('texts', None)
                if batch_texts is None:
                    raise ValueError("GISTEmbedLoss requires 'texts' in batch_features")
                
                # Get text pairs
                texts_a = [text for i, text in enumerate(batch_texts) if i % 2 == 0]
                texts_b = [text for i, text in enumerate(batch_texts) if i % 2 == 1]
                
                # Get guide model embeddings
                guide_embeddings_a = self.guide_model.encode(texts_a, convert_to_tensor=True)
                guide_embeddings_b = self.guide_model.encode(texts_b, convert_to_tensor=True)
                
                # Compute guide model similarities
                guide_cos_sim = self.similarity_fct(guide_embeddings_a, guide_embeddings_b)
                
            # Compute loss: try to match or exceed guide model similarity
            similarity_targets = torch.clamp(guide_cos_sim + self.loss_margin, min=0.0, max=1.0)
            losses = torch.nn.functional.mse_loss(cos_sim, similarity_targets, reduction='none')
            return losses.mean()
        
        return cos_sim

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SentenceTransformer on IPIP using GIST loss")
    
    # Model parameters
    parser.add_argument('--student_model', type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Student model to be fine-tuned")
    parser.add_argument('--guide_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Guide model for GIST loss")
    parser.add_argument('--pooling_mode', type=str, default="mean", choices=["mean", "max", "cls"],
                        help="Pooling strategy for embeddings")
    parser.add_argument('--loss_margin', type=float, default=0.3,
                        help="Margin for GIST loss (student should exceed guide by this amount)")
    
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
    output_path = f"models/ipip_gist_{timestamp}"
    
    # Create output dir if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save training configuration
    config = vars(args)
    config['timestamp'] = timestamp
    
    with open(f"{output_path}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    logger.info(f"Loading data from {args.train_file}")
    train_examples, anchors, positives = load_ipip_pairs(args.train_file)
    logger.info(f"Dataset has {len(train_examples)} examples")
    config['num_examples'] = len(train_examples)
    
    # Update config with number of examples
    with open(f"{output_path}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare student model
    logger.info(f"Initializing student model: {args.student_model}")
    word_embedding_model = models.Transformer(args.student_model)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                  pooling_mode=args.pooling_mode)
    student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Prepare guide model
    logger.info(f"Initializing guide model: {args.guide_model}")
    guide_model = SentenceTransformer(args.guide_model)
    
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
    
    student_model.to(device)
    guide_model.to(device)
    
    # Create custom GIST loss
    train_loss = GISTEmbedLoss(
        model=student_model,
        guide_model=guide_model,
        loss_margin=args.loss_margin
    )
    
    # Prepare dataloader with text pairs
    train_data = []
    for i in range(len(anchors)):
        train_data.append({
            'texts': [anchors[i], positives[i]],
            'label': 1.0
        })
    
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    
    # Calculate total steps and warmup steps
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    checkpoint_steps = args.checkpoint_steps or len(train_dataloader)
    
    logger.info(f"Training parameters: Student: {args.student_model}, Guide: {args.guide_model}, "
               f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.learning_rate}")
    logger.info(f"Warmup Steps: {warmup_steps}, Total Steps: {len(train_dataloader) * args.epochs}")
    
    # Train the model
    logger.info("Starting training...")
    
    # The fit method expects a list of train objectives
    train_objectives = [(train_dataloader, train_loss)]
    
    student_model.fit(
        train_objectives=train_objectives,
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