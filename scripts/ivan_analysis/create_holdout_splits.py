#!/usr/bin/env python3
"""
Create holdout validation splits for IPIP data with stratified sampling by construct.

This script:
1. Loads the original IPIP data
2. Creates 80/20 train/holdout splits stratified by construct
3. Generates training pairs from train set only
4. Saves holdout items for later validation

This addresses the training bias issue where the model was trained and tested
on the same IPIP items.
"""

import pandas as pd
import json
import random
from pathlib import Path
import logging
import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('create_holdout_splits.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_CSV = "data/IPIP.csv"
OUTPUT_TRAIN_PAIRS = "data/processed/ipip_train_pairs_holdout.jsonl"
OUTPUT_HOLDOUT_ITEMS = "data/processed/ipip_holdout_items.csv"
OUTPUT_HOLDOUT_INFO = "data/processed/ipip_holdout_info.json"

# NEW: Ivan's construct-level holdout outputs
OUTPUT_CONSTRUCT_TRAIN_PAIRS = "data/processed/ipip_construct_train_pairs.jsonl"
OUTPUT_CONSTRUCT_HOLDOUT_ITEMS = "data/processed/ipip_construct_holdout_items.csv"
OUTPUT_CONSTRUCT_HOLDOUT_INFO = "data/processed/ipip_construct_holdout_info.json"

TEXT_COL = "text"
LABEL_COL = "label"
TRAIN_SIZE = 0.8
RANDOM_STATE = 42

# Ensure reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def load_and_preprocess_data(input_path):
    """Load IPIP data and handle any preprocessing."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding="latin-1").dropna()
    logger.info(f"Loaded {len(df)} items with {df[LABEL_COL].nunique()} unique constructs")
    return df

def create_stratified_splits(df, text_col, label_col, train_size=0.8):
    """Create stratified train/holdout splits ensuring each construct is represented in both sets."""
    
    # Group items by construct
    construct_to_items = defaultdict(list)
    for idx, row in df.iterrows():
        construct_to_items[row[label_col]].append({
            'text': row[text_col],
            'label': row[label_col],
            'index': idx
        })
    
    train_items = []
    holdout_items = []
    split_stats = {}
    
    # Split each construct separately to ensure stratification
    for construct, items in construct_to_items.items():
        if len(items) < 2:
            logger.warning(f"Construct '{construct}' has only {len(items)} items. Skipping...")
            continue
        
        # Shuffle items within construct
        random.shuffle(items)
        
        # Calculate split point
        n_train = max(1, int(len(items) * train_size))
        
        # Ensure at least 1 item in holdout if possible
        if len(items) > 1 and n_train == len(items):
            n_train = len(items) - 1
        
        train_items.extend(items[:n_train])
        holdout_items.extend(items[n_train:])
        
        split_stats[construct] = {
            'total': len(items),
            'train': n_train,
            'holdout': len(items) - n_train
        }
    
    logger.info(f"Created splits: {len(train_items)} train items, {len(holdout_items)} holdout items")
    
    return train_items, holdout_items, split_stats

def generate_randomized_pairs(items):
    """Generate all possible pairs with Ivan's randomization approach from training items only."""
    
    # Group items by construct
    construct_to_texts = defaultdict(list)
    for item in items:
        construct_to_texts[item['label']].append(item['text'])
    
    # Generate all possible pairs for each construct
    all_pairs = []
    pair_stats = {}
    
    for construct, texts in construct_to_texts.items():
        if len(texts) >= 2:
            # Generate all possible combinations of 2 items within this construct
            pairs = list(combinations(texts, 2))
            
            # Apply triple randomization as in Ivan's approach
            # First randomization: shuffle within each pair
            for i in range(len(pairs)):
                pairs[i] = tuple(random.sample(pairs[i], len(pairs[i])))
            
            # Second randomization: flatten and randomize order
            shuffled = [
                (a, b) if random.random() < 0.5 else (b, a)
                for a, b in pairs
            ]
            
            # Third randomization: switch orders randomly pointwise
            final_pairs = []
            for anchor, positive in shuffled:
                if random.random() < 0.5:
                    final_pairs.append({'anchor': positive, 'positive': anchor})
                else:
                    final_pairs.append({'anchor': anchor, 'positive': positive})
            
            all_pairs.extend(final_pairs)
            pair_stats[construct] = len(final_pairs)
    
    logger.info(f"Generated {len(all_pairs)} total training pairs")
    
    return all_pairs, pair_stats

def save_results(train_items, holdout_items, train_pairs, split_stats, pair_stats):
    """Save all results to appropriate files."""
    
    # Save training pairs in JSONL format
    logger.info(f"Saving {len(train_pairs)} training pairs to {OUTPUT_TRAIN_PAIRS}")
    with open(OUTPUT_TRAIN_PAIRS, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')
    
    # Save holdout items as CSV
    logger.info(f"Saving {len(holdout_items)} holdout items to {OUTPUT_HOLDOUT_ITEMS}")
    holdout_df = pd.DataFrame(holdout_items)
    holdout_df.to_csv(OUTPUT_HOLDOUT_ITEMS, index=False)
    
    # Save split information
    info = {
        'train_size': TRAIN_SIZE,
        'random_state': RANDOM_STATE,
        'n_train_items': len(train_items),
        'n_holdout_items': len(holdout_items),
        'n_train_pairs': len(train_pairs),
        'n_constructs': len(split_stats),
        'split_stats': split_stats,
        'pair_stats': pair_stats
    }
    
    logger.info(f"Saving split information to {OUTPUT_HOLDOUT_INFO}")
    with open(OUTPUT_HOLDOUT_INFO, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Print summary
    logger.info("\n=== Split Summary ===")
    logger.info(f"Total constructs: {len(split_stats)}")
    logger.info(f"Train items: {len(train_items)} ({len(train_items)/(len(train_items)+len(holdout_items))*100:.1f}%)")
    logger.info(f"Holdout items: {len(holdout_items)} ({len(holdout_items)/(len(train_items)+len(holdout_items))*100:.1f}%)")
    logger.info(f"Training pairs generated: {len(train_pairs)}")
    
    # Check construct representation
    train_constructs = set(item['label'] for item in train_items)
    holdout_constructs = set(item['label'] for item in holdout_items)
    logger.info(f"Constructs in train set: {len(train_constructs)}")
    logger.info(f"Constructs in holdout set: {len(holdout_constructs)}")
    logger.info(f"Constructs in both: {len(train_constructs & holdout_constructs)}")

def create_construct_level_splits(df):
    """
    Create construct-level holdout splits following Ivan's methodology.
    
    This splits constructs (not items) into 80/20 train/test.
    Training uses ALL items from 80% of constructs.
    Testing uses ALL items from 20% of constructs.
    """
    logger.info("Creating construct-level holdout splits (Ivan's method)")
    
    # Group items by construct (following Ivan's code exactly)
    construct_to_items = defaultdict(list)
    for idx, row in df.iterrows():
        construct_to_items[row[LABEL_COL]].append(row[TEXT_COL])
    
    # Split constructs 80/20 (following Ivan's code exactly)
    construct_list = list(construct_to_items.keys())
    threshold = int(len(construct_list) * 0.8)
    
    construct_to_items_train = construct_list[:threshold]  # 80% of constructs
    construct_to_items_test = construct_list[threshold:]   # 20% of constructs
    
    logger.info(f"Total constructs: {len(construct_list)}")
    logger.info(f"Training constructs: {len(construct_to_items_train)} ({len(construct_to_items_train)/len(construct_list)*100:.1f}%)")
    logger.info(f"Test constructs: {len(construct_to_items_test)} ({len(construct_to_items_test)/len(construct_list)*100:.1f}%)")
    
    # Generate training pairs from training constructs only (following Ivan's code)
    construct_pairs_train = {}
    total_possible_pairs = 0
    
    for construct in construct_to_items_train:
        items = construct_to_items[construct]
        if len(items) >= 2:  # Need at least 2 items to form a pair
            # Generate all possible combinations of 2 items within this construct
            pairs = list(combinations(items, 2))
            # Randomize within pairs (following Ivan's code)
            for i in range(len(pairs)):
                pairs[i] = tuple(random.sample(pairs[i], len(pairs[i])))
            construct_pairs_train[construct] = pairs
            total_possible_pairs += len(pairs)
    
    # Flatten and randomize training pairs (following Ivan's code)
    shuffled_train = [
        (a, b) if random.random() < 0.5 else (b, a)
        for pairs in construct_pairs_train.values()
        for a, b in pairs
    ]
    
    # Remove any NaN values (following Ivan's code)
    anchors_train, positives_train = map(list, zip(*shuffled_train))
    
    indices_to_remove_train = [
        i for i, (anchor, positive) in enumerate(zip(anchors_train, positives_train))
        if isinstance(anchor, float) or isinstance(positive, float)
    ]
    
    cleaned_anchors_train = [anchors_train[i] for i in range(len(anchors_train)) if i not in indices_to_remove_train]
    cleaned_positives_train = [positives_train[i] for i in range(len(positives_train)) if i not in indices_to_remove_train]
    
    # Create training pairs in JSONL format
    train_pairs = [
        {"anchor": anchor, "positive": positive}
        for anchor, positive in zip(cleaned_anchors_train, cleaned_positives_train)
    ]
    
    # Create holdout items (from test constructs only)
    holdout_items = []
    train_items = []
    
    for idx, row in df.iterrows():
        item_dict = {"text": row[TEXT_COL], "label": row[LABEL_COL]}
        if row[LABEL_COL] in construct_to_items_test:
            holdout_items.append(item_dict)
        elif row[LABEL_COL] in construct_to_items_train:
            train_items.append(item_dict)
    
    # Create split statistics
    split_stats = {
        'total_constructs': len(construct_list),
        'train_constructs': len(construct_to_items_train),
        'test_constructs': len(construct_to_items_test),
        'train_construct_names': construct_to_items_train,
        'test_construct_names': construct_to_items_test,
        'train_items': len(train_items),
        'test_items': len(holdout_items),
        'train_pairs': len(train_pairs)
    }
    
    logger.info(f"Generated {len(train_pairs)} training pairs from {len(construct_to_items_train)} constructs")
    logger.info(f"Holdout contains {len(holdout_items)} items from {len(construct_to_items_test)} constructs")
    
    return train_items, holdout_items, train_pairs, split_stats

def save_construct_results(train_items, holdout_items, train_pairs, split_stats):
    """Save construct-level holdout results."""
    
    # Save training pairs in JSONL format
    logger.info(f"Saving {len(train_pairs)} construct-level training pairs to {OUTPUT_CONSTRUCT_TRAIN_PAIRS}")
    with open(OUTPUT_CONSTRUCT_TRAIN_PAIRS, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')
    
    # Save holdout items as CSV
    logger.info(f"Saving {len(holdout_items)} construct-level holdout items to {OUTPUT_CONSTRUCT_HOLDOUT_ITEMS}")
    holdout_df = pd.DataFrame(holdout_items)
    holdout_df.to_csv(OUTPUT_CONSTRUCT_HOLDOUT_ITEMS, index=False)
    
    # Save split information
    info = {
        'method': 'construct_level_holdout',
        'description': 'Ivan methodology: 80% constructs for training, 20% constructs for testing',
        'train_size': TRAIN_SIZE,
        'random_state': RANDOM_STATE,
        'split_stats': split_stats
    }
    
    logger.info(f"Saving construct-level split information to {OUTPUT_CONSTRUCT_HOLDOUT_INFO}")
    with open(OUTPUT_CONSTRUCT_HOLDOUT_INFO, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Print summary
    logger.info("\n=== Construct-Level Split Summary ===")
    logger.info(f"Total constructs: {split_stats['total_constructs']}")
    logger.info(f"Train constructs: {split_stats['train_constructs']} ({split_stats['train_constructs']/split_stats['total_constructs']*100:.1f}%)")
    logger.info(f"Test constructs: {split_stats['test_constructs']} ({split_stats['test_constructs']/split_stats['total_constructs']*100:.1f}%)")
    logger.info(f"Train items: {split_stats['train_items']}")
    logger.info(f"Test items: {split_stats['test_items']}")
    logger.info(f"Training pairs generated: {split_stats['train_pairs']}")
    logger.info("✅ NO CONSTRUCT OVERLAP between train and test!")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create holdout splits for IPIP data')
    parser.add_argument('--method', choices=['item_level', 'construct_level', 'both'], 
                       default='both', help='Holdout method to use')
    args = parser.parse_args()
    
    logger.info("Starting holdout split creation for IPIP data")
    
    # Load data
    df = load_and_preprocess_data(INPUT_CSV)
    
    if args.method in ['item_level', 'both']:
        logger.info("\n" + "="*50)
        logger.info("CREATING ITEM-LEVEL HOLDOUT (Original Method)")
        logger.info("="*50)
        
        # Create stratified splits (original method)
        train_items, holdout_items, split_stats = create_stratified_splits(
            df, TEXT_COL, LABEL_COL, TRAIN_SIZE
        )
        
        # Generate training pairs (only from training items)
        train_pairs, pair_stats = generate_randomized_pairs(train_items)
        
        # Save results
        save_results(train_items, holdout_items, train_pairs, split_stats, pair_stats)
        
        logger.info("Item-level holdout split creation completed!")
    
    if args.method in ['construct_level', 'both']:
        logger.info("\n" + "="*50)
        logger.info("CREATING CONSTRUCT-LEVEL HOLDOUT (Ivan's Method)")
        logger.info("="*50)
        
        # Create construct-level splits (Ivan's method)
        train_items, holdout_items, train_pairs, split_stats = create_construct_level_splits(df)
        
        # Save results
        save_construct_results(train_items, holdout_items, train_pairs, split_stats)
        
        logger.info("Construct-level holdout split creation completed!")
    
    logger.info("\n✅ All requested holdout split creation completed successfully!")

if __name__ == "__main__":
    main()