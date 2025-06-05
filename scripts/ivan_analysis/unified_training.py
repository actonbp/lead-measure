#!/usr/bin/env python3
"""
Unified training script for Ivan's enhanced analysis approach.
Automatically detects platform and optimizes accordingly.

This replaces the multiple separate scripts with a single, configurable script
that handles:
- Platform detection (Mac Silicon, Windows, Linux)
- Holdout validation
- Optimized parameters based on hardware
- Proper multiprocessing handling for each platform
"""

import os
import json
import pandas as pd
import numpy as np
import platform
import argparse
from datetime import datetime
from pathlib import Path
import logging
import sys
import nltk
from typing import List, Tuple, Optional, Dict
import torch
import multiprocessing as mp

# Import sentence transformers components
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    models,
    SentenceTransformerTrainer,
    losses,
    datasets
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
        logging.FileHandler('unified_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Disable wandb by default
os.environ['WANDB_DISABLED'] = 'true'


class PlatformConfig:
    """Platform-specific configuration."""
    
    @staticmethod
    def detect_platform() -> Dict:
        """Detect platform and hardware capabilities."""
        system = platform.system()
        machine = platform.machine()
        
        config = {
            'system': system,
            'machine': machine,
            'is_mac_silicon': False,
            'has_mps': False,
            'has_cuda': False,
            'device': 'cpu',
            'num_workers': 4,
            'multiprocessing_context': None
        }
        
        # Mac Silicon detection
        if system == 'Darwin' and machine in ['arm64', 'aarch64']:
            config['is_mac_silicon'] = True
            config['has_mps'] = torch.backends.mps.is_available()
            if config['has_mps']:
                config['device'] = 'mps'
            # Mac-specific multiprocessing settings
            config['num_workers'] = 0  # Avoid pickle issues with TSDAE
            config['multiprocessing_context'] = 'fork'
            
        # CUDA detection for other platforms
        elif torch.cuda.is_available():
            config['has_cuda'] = True
            config['device'] = 'cuda'
            
        # Windows-specific settings
        if system == 'Windows':
            config['num_workers'] = 0  # Windows has issues with multiprocessing
            
        return config
    
    @staticmethod
    def get_optimized_params(config: Dict, use_high_memory: bool = False) -> Dict:
        """Get optimized parameters based on platform."""
        params = {
            'tsdae_epochs': 1,
            'tsdae_batch_size': 4,
            'gist_batch_size': 32,
            'gist_epochs_per_phase': 10,
            'num_training_phases': 5,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'guide_model': 'all-MiniLM-L6-v2',
            'base_model': 'BAAI/bge-m3'
        }
        
        # Mac Silicon with high memory (64GB+)
        if config['is_mac_silicon'] and use_high_memory:
            params.update({
                'tsdae_epochs': 3,
                'tsdae_batch_size': 16,
                'gist_batch_size': 96,
                'guide_model': 'BAAI/bge-m3'  # Better guide model
            })
            
        # CUDA with high memory
        elif config['has_cuda'] and use_high_memory:
            params.update({
                'tsdae_epochs': 3,
                'tsdae_batch_size': 32,
                'gist_batch_size': 128
            })
            
        return params


def setup_environment(config: Dict):
    """Set up environment based on platform."""
    if config['is_mac_silicon']:
        # Mac Silicon optimizations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        # Fix for multiprocessing issues
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        torch.set_num_threads(8)  # Use performance cores
        
        logger.info("Mac Silicon optimizations applied:")
        logger.info("  - MPS fallback enabled")
        logger.info("  - Memory watermark disabled")
        logger.info("  - Fork safety disabled for multiprocessing")
        
    elif config['has_cuda']:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        

def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        logger.info("NLTK punkt_tab not available, trying punkt")
        nltk.download('punkt', quiet=True)


def load_holdout_data(data_type: str = 'train', method: str = 'item_level') -> Tuple[List[str], List[str], Dict]:
    """Load holdout training or validation data.
    
    Args:
        data_type: 'train' or 'validation'
        method: 'item_level' or 'construct_level' (Ivan's method)
    """
    if data_type == 'train':
        if method == 'construct_level':
            # Load construct-level training pairs (Ivan's method)
            pairs_file = "data/processed/ipip_construct_train_pairs.jsonl"
            logger.info(f"Loading construct-level training pairs from {pairs_file}")
            
            anchors = []
            positives = []
            with open(pairs_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    anchors.append(data['anchor'])
                    positives.append(data['positive'])
            
            # Load training texts for TSDAE (only from training constructs)
            df_full = pd.read_csv("data/IPIP.csv", encoding="latin-1").dropna()
            df_holdout = pd.read_csv("data/processed/ipip_construct_holdout_items.csv")
            holdout_texts = set(df_holdout['text'].tolist())
            train_texts = [text for text in df_full['text'].unique() if text not in holdout_texts]
            
            # Load info
            with open("data/processed/ipip_construct_holdout_info.json", 'r') as f:
                info = json.load(f)
                
            logger.info(f"✅ CONSTRUCT-LEVEL: {len(anchors)} pairs, {len(train_texts)} texts, {info['split_stats']['train_constructs']} constructs")
            return anchors, positives, {'texts': train_texts, 'info': info}
            
        else:  # item_level (original method)
            # Load training pairs
            pairs_file = "data/processed/ipip_train_pairs_holdout.jsonl"
            logger.info(f"Loading item-level training pairs from {pairs_file}")
            
            anchors = []
            positives = []
            with open(pairs_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    anchors.append(data['anchor'])
                    positives.append(data['positive'])
                    
            # Load training texts for TSDAE
            df_full = pd.read_csv("data/IPIP.csv", encoding="latin-1").dropna()
            df_holdout = pd.read_csv("data/processed/ipip_holdout_items.csv")
            holdout_texts = set(df_holdout['text'].tolist())
            train_texts = [text for text in df_full['text'].unique() if text not in holdout_texts]
            
            # Load info
            with open("data/processed/ipip_holdout_info.json", 'r') as f:
                info = json.load(f)
                
            logger.info(f"📊 ITEM-LEVEL: {len(anchors)} pairs, {len(train_texts)} texts")
            return anchors, positives, {'texts': train_texts, 'info': info}
        
    else:  # validation
        # Load all pairs for validation
        pairs_file = "data/processed/ipip_pairs_randomized.jsonl"
        logger.info(f"Loading all pairs from {pairs_file}")
        
        anchors = []
        positives = []
        with open(pairs_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                anchors.append(data['anchor'])
                positives.append(data['positive'])
                
        # Load all texts
        df = pd.read_csv("data/IPIP.csv", encoding="latin-1").dropna()
        all_texts = df['text'].unique().tolist()
        
        return anchors, positives, {'texts': all_texts}


def perform_tsdae_pretraining(
    model_name: str,
    texts: List[str],
    output_dir: str,
    config: Dict,
    params: Dict
) -> SentenceTransformer:
    """Perform TSDAE pre-training with platform-specific optimizations."""
    logger.info(f"Starting TSDAE pre-training on {config['device']}...")
    logger.info(f"Epochs: {params['tsdae_epochs']}, Batch size: {params['tsdae_batch_size']}")
    
    # Build model
    word_embedding = models.Transformer(model_name)
    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(), 
        pooling_mode="cls"
    )
    model = SentenceTransformer(modules=[word_embedding, pooling], device=config['device'])
    
    # Prepare dataset
    train_dataset = datasets.DenoisingAutoEncoderDataset(texts)
    
    # Configure DataLoader with platform-specific settings
    dataloader_kwargs = {
        'batch_size': params['tsdae_batch_size'],
        'shuffle': True,
        'num_workers': config['num_workers'],
        'pin_memory': (config['device'] == 'cuda'),  # Only for CUDA
    }
    
    # Only add multiprocessing_context if we have workers
    if config['num_workers'] > 0 and config['multiprocessing_context']:
        dataloader_kwargs['multiprocessing_context'] = config['multiprocessing_context']
        
    dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    
    # TSDAE loss
    loss = losses.DenoisingAutoEncoderLoss(
        model=model,
        tie_encoder_decoder=True,
    )
    
    # Train
    logger.info("Training TSDAE...")
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=params['tsdae_epochs'],
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        use_amp=(config['device'] == 'cuda'),  # AMP only for CUDA
    )
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    logger.info(f"TSDAE pre-training complete. Model saved to {output_dir}")
    
    return model


def train_with_gist_loss(
    model: SentenceTransformer,
    anchors: List[str],
    positives: List[str],
    config: Dict,
    params: Dict,
    output_dir: str
):
    """Train with GIST loss using platform-specific optimizations."""
    logger.info(f"Starting GIST loss training on {config['device']}...")
    logger.info(f"Batch size: {params['gist_batch_size']}")
    
    # Load guide model
    guide_model = SentenceTransformer(params['guide_model'])
    
    # Create dataset
    train_dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
    })
    
    # Training arguments with platform-specific settings
    training_args_kwargs = {
        'output_dir': output_dir,
        'num_train_epochs': params['gist_epochs_per_phase'],
        'per_device_train_batch_size': params['gist_batch_size'],
        'learning_rate': params['learning_rate'],
        'warmup_ratio': params['warmup_ratio'],
        'fp16': (config['device'] == 'cuda'),  # FP16 only for CUDA
        'bf16': False,
        'batch_sampler': BatchSamplers.NO_DUPLICATES,
        'save_strategy': "no",
        'logging_steps': max(1, 100 // params['gist_batch_size']),
        'seed': 42,
        'dataloader_num_workers': config['num_workers'],
        'dataloader_pin_memory': (config['device'] == 'cuda'),
        'save_total_limit': 2,
    }
    
    args = SentenceTransformerTrainingArguments(**training_args_kwargs)
    
    # GIST loss
    loss = losses.GISTEmbedLoss(model, guide_model)
    
    # Train in phases
    for phase in range(params['num_training_phases']):
        logger.info(f"\nTraining phase {phase + 1}/{params['num_training_phases']}")
        
        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=loss,
            args=args
        )
        
        trainer.train()
        
        # Save checkpoint
        phase_output_dir = f"{output_dir}_phase{phase + 1}"
        model.save(phase_output_dir)
        logger.info(f"Saved checkpoint to {phase_output_dir}")
    
    # Save final model
    final_output_dir = f"{output_dir}_final"
    model.save(final_output_dir)
    logger.info(f"Training complete. Final model saved to {final_output_dir}")
    
    return model


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Unified training script for Ivan\'s analysis')
    parser.add_argument('--mode', choices=['full', 'holdout'], default='holdout',
                       help='Training mode: full (all data) or holdout (80/20 split)')
    parser.add_argument('--method', choices=['item_level', 'construct_level'], default='item_level',
                       help='Holdout method: item_level (original) or construct_level (Ivan\'s method)')
    parser.add_argument('--high-memory', action='store_true',
                       help='Use high memory optimizations (for 32GB+ systems)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if models exist')
    parser.add_argument('--skip-tsdae', action='store_true',
                       help='Skip TSDAE pre-training if model exists')
    args = parser.parse_args()
    
    logger.info("🚀 Starting Unified Training Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Method: {args.method}")
    
    # Detect platform and configure
    config = PlatformConfig.detect_platform()
    params = PlatformConfig.get_optimized_params(config, args.high_memory)
    
    logger.info(f"\n📱 Platform Configuration:")
    logger.info(f"  System: {config['system']} ({config['machine']})")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"  Mac Silicon: {config['is_mac_silicon']}")
    logger.info(f"  High Memory Mode: {args.high_memory}")
    
    # Setup environment
    setup_environment(config)
    download_nltk_data()
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    
    # Define paths based on mode and method
    if args.mode == 'holdout':
        if args.method == 'construct_level':
            tsdae_dir = "models/tsdae_construct_holdout_unified"
            gist_dir = "models/gist_construct_holdout_unified"
            logger.info(f"\n🎯 IVAN'S CONSTRUCT-LEVEL HOLDOUT METHOD")
        else:
            tsdae_dir = "models/tsdae_holdout_unified"
            gist_dir = "models/gist_holdout_unified"
            logger.info(f"\n📊 ITEM-LEVEL HOLDOUT METHOD")
            
        anchors, positives, data = load_holdout_data('train', args.method)
        texts = data['texts']
        logger.info(f"\n📊 Holdout Training Configuration:")
        if 'n_train_items' in data['info']:
            logger.info(f"  Training items: {data['info']['n_train_items']}")
        logger.info(f"  Training pairs: {len(anchors)}")
        logger.info(f"  Training texts: {len(texts)}")
    else:
        tsdae_dir = "models/tsdae_full_unified"
        gist_dir = "models/gist_full_unified"
        anchors, positives, data = load_holdout_data('validation')
        texts = data['texts']
        logger.info(f"\n📊 Full Training Configuration:")
        logger.info(f"  Total texts: {len(texts)}")
        logger.info(f"  Total pairs: {len(anchors)}")
    
    # Step 1: TSDAE pre-training
    if Path(tsdae_dir).exists() and not args.force_retrain and args.skip_tsdae:
        logger.info(f"\n✅ Loading existing TSDAE model from {tsdae_dir}")
        model = SentenceTransformer(tsdae_dir)
    else:
        logger.info(f"\n🔄 Starting TSDAE pre-training...")
        model = perform_tsdae_pretraining(
            model_name=params['base_model'],
            texts=texts,
            output_dir=tsdae_dir,
            config=config,
            params=params
        )
    
    # Step 2: GIST loss training
    logger.info(f"\n🔄 Starting GIST loss training...")
    model = train_with_gist_loss(
        model=model,
        anchors=anchors,
        positives=positives,
        config=config,
        params=params,
        output_dir=gist_dir
    )
    
    logger.info("\n✅ Training pipeline complete!")
    logger.info(f"Final model saved to: {gist_dir}_final")
    
    if args.mode == 'holdout':
        logger.info("\n📋 Next steps:")
        logger.info("1. Run validation: python scripts/ivan_analysis/validate_holdout_results.py")
        logger.info("2. Compare with leadership data")


if __name__ == "__main__":
    # Handle multiprocessing based on platform
    # Don't set start method if we're not using multiprocessing
    if platform.system() != 'Darwin':  # Not Mac
        mp.set_start_method('spawn', force=True)
    main()