"""Generate comprehensive anchor-positive text pairs from IPIP personality items.

This script reads the IPIP personality items CSV file and generates all possible
anchor-positive pairs within each construct for training. It also implements
rebalancing strategies to prevent over/under-representation of constructs.

Usage:
    python scripts/build_ipip_pairs_improved.py

Outputs:
    data/processed/ipip_pairs_comprehensive.jsonl - A JSONL file containing balanced anchor-positive pairs
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_CSV = "data/IPIP.csv"
OUTPUT_JSONL = "data/processed/ipip_pairs_comprehensive.jsonl"
TEXT_COL = "text"
LABEL_COL = "label"

# Rebalancing configuration
MAX_PAIRS_PER_CONSTRUCT = 500  # Cap on pairs per construct to prevent dominance
MIN_PAIRS_PER_CONSTRUCT = 20   # Minimum pairs to ensure for each construct
REBALANCE_METHOD = "sampling"  # Options: "sampling", "cap", "both"
                               # "sampling" = subsample from larger constructs
                               # "cap" = hard cap at MAX_PAIRS
                               # "both" = apply both strategies

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

def main():
    # Ensure output directory exists
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Reading {INPUT_CSV} ...")
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed. Retrying with latin-1 …")
        df = pd.read_csv(INPUT_CSV, encoding="latin-1")

    # Drop rows with missing values in key columns
    initial_len = len(df)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    logger.info(f"Loaded {len(df)} rows (dropped {initial_len - len(df)} with NA).")
    
    # Group items by construct
    construct_to_items = defaultdict(list)
    for idx, row in df.iterrows():
        construct_to_items[row[LABEL_COL]].append((idx, row[TEXT_COL]))
    
    logger.info(f"Found {len(construct_to_items)} unique constructs.")
    
    # Generate all possible pairs for each construct
    construct_pairs = {}
    total_possible_pairs = 0
    
    for construct, items in construct_to_items.items():
        if len(items) >= 2:  # Need at least 2 items to form a pair
            # Generate all possible combinations of 2 items within this construct
            pairs = list(combinations(items, 2))
            construct_pairs[construct] = pairs
            total_possible_pairs += len(pairs)
            logger.info(f"Construct '{construct}': {len(items)} items → {len(pairs)} possible pairs")
    
    logger.info(f"Generated {total_possible_pairs} total possible pairs across {len(construct_pairs)} constructs.")
    
    # Rebalance pairs to prevent over/under-representation
    rebalanced_pairs = []
    
    if REBALANCE_METHOD == "sampling" or REBALANCE_METHOD == "both":
        # Calculate target number of pairs per construct for balanced representation
        target_pairs = max(MIN_PAIRS_PER_CONSTRUCT, 
                          min(MAX_PAIRS_PER_CONSTRUCT, 
                             total_possible_pairs // len(construct_pairs)))
        
        logger.info(f"Rebalancing using {REBALANCE_METHOD} method. Target pairs per construct: {target_pairs}")
        
        # Sample pairs from each construct
        for construct, pairs in construct_pairs.items():
            if len(pairs) <= target_pairs:
                # Keep all pairs for under-represented constructs
                selected_pairs = pairs
            else:
                # Randomly sample for over-represented constructs
                selected_pairs = random.sample(pairs, target_pairs)
            
            rebalanced_pairs.extend([(p[0][0], p[0][1], p[1][0], p[1][1], construct) 
                                    for p in selected_pairs])
    
    elif REBALANCE_METHOD == "cap":
        # Simply cap each construct at MAX_PAIRS
        for construct, pairs in construct_pairs.items():
            if len(pairs) > MAX_PAIRS_PER_CONSTRUCT:
                selected_pairs = random.sample(pairs, MAX_PAIRS_PER_CONSTRUCT)
            else:
                selected_pairs = pairs
            
            rebalanced_pairs.extend([(p[0][0], p[0][1], p[1][0], p[1][1], construct) 
                                    for p in selected_pairs])
    
    # Shuffle the pairs for good measure
    random.shuffle(rebalanced_pairs)
    
    # Write the pairs to output file
    pairs_written = 0
    constructs_in_output = set()
    
    logger.info(f"Writing {len(rebalanced_pairs)} rebalanced pairs to {OUTPUT_JSONL}...")
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for anchor_idx, anchor_text, positive_idx, positive_text, construct in rebalanced_pairs:
            pair = {
                "anchor": anchor_text,
                "positive": positive_text,
                "construct": construct  # Include construct for reference
            }
            f_out.write(json.dumps(pair) + "\n")
            pairs_written += 1
            constructs_in_output.add(construct)
    
    # Generate statistics for the output file
    logger.info(f"Successfully wrote {pairs_written} balanced pairs across {len(constructs_in_output)} constructs.")
    
    # Count pairs per construct in output
    construct_counts = defaultdict(int)
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f_in:
        for line in f_in:
            pair = json.loads(line)
            construct_counts[pair["construct"]] += 1
    
    # Show distribution
    logger.info("Distribution of pairs per construct in output:")
    for construct, count in sorted(construct_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {construct}: {count} pairs")
    
    # Print next steps
    logger.info("\nNext steps:")
    logger.info("1. Update training script to use the new comprehensive pairs file")
    logger.info("2. Run training with the improved dataset")

if __name__ == "__main__":
    main() 