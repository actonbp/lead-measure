#!/usr/bin/env python3
"""
Build comprehensive IPIP pairs with Ivan's randomization improvements.

This script generates all possible within-construct pairs from IPIP items
with the following enhancements:
1. Randomizes which item is anchor vs positive to avoid ordering bias
2. Double randomization for better distribution
3. Removes any NaN or float values from pairs

Based on Ivan Hernandez's improvements (January 2025)
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('build_pairs_randomized.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_CSV = "data/IPIP.csv"
OUTPUT_JSONL = "data/processed/ipip_pairs_randomized.jsonl"
TEXT_COL = "text"
LABEL_COL = "label"

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

def load_and_preprocess_data(input_path):
    """Load IPIP data and handle any preprocessing."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, encoding="latin-1").dropna()
    logger.info(f"Loaded {len(df)} items with {df[LABEL_COL].nunique()} unique constructs")
    return df

def generate_randomized_pairs(df, text_col, label_col):
    """Generate all possible pairs with Ivan's randomization approach."""
    # Group items by construct
    construct_to_items = defaultdict(list)
    for idx, row in df.iterrows():
        construct_to_items[row[label_col]].append(row[text_col])
    
    # Generate all possible pairs for each construct
    construct_pairs = {}
    total_possible_pairs = 0
    
    for construct, items in construct_to_items.items():
        if len(items) >= 2:  # Need at least 2 items to form a pair
            # Generate all possible combinations of 2 items within this construct
            pairs = list(combinations(items, 2))
            
            # First randomization: shuffle within each pair
            for i in range(len(pairs)):
                pairs[i] = tuple(random.sample(pairs[i], len(pairs[i])))
            
            construct_pairs[construct] = pairs
            total_possible_pairs += len(pairs)
    
    logger.info(f"Generated {total_possible_pairs} total pairs across {len(construct_pairs)} constructs")
    
    # Flatten and apply second randomization
    shuffled = [
        (a, b) if random.random() < 0.5 else (b, a)
        for pairs in construct_pairs.values()
        for a, b in pairs
    ]
    
    anchors, positives = map(list, zip(*shuffled))
    
    # Third randomization: switch orders randomly pointwise
    anchors_positives_switched = []
    for anchor, positive in zip(anchors, positives):
        if random.random() < 0.5:
            anchors_positives_switched.append((positive, anchor))
        else:
            anchors_positives_switched.append((anchor, positive))
    
    anchors, positives = map(list, zip(*anchors_positives_switched))
    
    # Remove any NaN or float values
    indices_to_remove = [
        i for i, (anchor, positive) in enumerate(zip(anchors, positives))
        if isinstance(anchor, float) or isinstance(positive, float)
    ]
    
    cleaned_anchors = [anchors[i] for i in range(len(anchors)) if i not in indices_to_remove]
    cleaned_positives = [positives[i] for i in range(len(positives)) if i not in indices_to_remove]
    
    if indices_to_remove:
        logger.info(f"Removed {len(indices_to_remove)} pairs containing NaN or float values")
    
    logger.info(f"Final dataset contains {len(cleaned_anchors)} pairs")
    
    return cleaned_anchors, cleaned_positives, construct_pairs

def save_pairs_to_jsonl(anchors, positives, output_path):
    """Save the pairs to JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for anchor, positive in zip(anchors, positives):
            record = {
                "anchor": anchor,
                "positive": positive
            }
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved {len(anchors)} pairs to {output_path}")

def main():
    """Main execution function."""
    # Load data
    df = load_and_preprocess_data(INPUT_CSV)
    
    # Generate randomized pairs
    anchors, positives, construct_pairs = generate_randomized_pairs(df, TEXT_COL, LABEL_COL)
    
    # Save to JSONL
    save_pairs_to_jsonl(anchors, positives, OUTPUT_JSONL)
    
    # Print statistics
    logger.info("\nPair Generation Statistics:")
    for construct, pairs in sorted(construct_pairs.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        logger.info(f"  {construct}: {len(pairs)} pairs")
    
    logger.info("\nProcess completed successfully!")

if __name__ == "__main__":
    main()