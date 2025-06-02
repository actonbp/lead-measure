"""Generate comprehensive anchor-positive text pairs from IPIP personality items.

This script reads the IPIP personality items CSV file and generates all possible
anchor-positive pairs within each construct for training. It also implements
rebalancing strategies to prevent over/under-representation of constructs.

The script includes validation checks to ensure:
1. No empty or blank text items are included in pairs
2. All generated pairs have valid text content
3. The number of pairs is consistent with what's expected mathematically

Usage:
    python scripts/build_ipip_pairs_improved.py

Outputs:
    data/processed/ipip_pairs_comprehensive.jsonl - A JSONL file containing balanced anchor-positive pairs
    data/processed/ipip_construct_statistics.csv - Statistics about constructs and their items
"""
import pandas as pd
import json
import random
from pathlib import Path
import logging
import sys
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import math
import csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("build_ipip_pairs_improved.log")]
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_CSV = "data/IPIP.csv"
OUTPUT_JSONL = "data/processed/ipip_pairs_comprehensive.jsonl"
STATS_CSV = "data/processed/ipip_construct_statistics.csv"
TEXT_COL = "text"
LABEL_COL = "label"

# Rebalancing configuration
MAX_PAIRS_PER_CONSTRUCT = 500  # Cap on pairs per construct to prevent dominance
MIN_PAIRS_PER_CONSTRUCT = 20   # Minimum pairs to ensure for each construct
REBALANCE_METHOD = "sampling"  # Options: "sampling", "cap", "both"
                               # "sampling" = subsample from larger constructs
                               # "cap" = hard cap at MAX_PAIRS
                               # "both" = apply both strategies

# Text validation settings
MIN_TEXT_LENGTH = 3           # Minimum characters for valid text
REQUIRE_ALPHABETIC = True     # Require at least one alphabetic character

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

def validate_text(text):
    """Validate that text is non-empty and meets minimum requirements."""
    if not isinstance(text, str):
        return False
    
    # Check for minimum length after stripping whitespace
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False
    
    # Check for at least one alphabetic character if required
    if REQUIRE_ALPHABETIC and not any(c.isalpha() for c in text):
        return False
        
    return True

def expected_pairs(n):
    """Calculate the expected number of pairs for n items."""
    if n < 2:
        return 0
    return math.comb(n, 2)  # n choose 2 = n! / (2! * (n-2)!)

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

    # Initial dataset statistics
    initial_len = len(df)
    logger.info(f"Initial dataset contains {initial_len} rows and {len(df.columns)} columns")
    logger.info(f"Column names: {', '.join(df.columns.tolist())}")
    
    # Check for missing values in important columns
    missing_text = df[TEXT_COL].isna().sum()
    missing_label = df[LABEL_COL].isna().sum()
    logger.info(f"Missing values: {TEXT_COL}: {missing_text}, {LABEL_COL}: {missing_label}")
    
    # Data cleaning: Drop rows with missing values in key columns
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    logger.info(f"After dropping NAs: {len(df)} rows (removed {initial_len - len(df)} rows)")
    
    # Further text validation
    # Filter out rows with invalid text
    valid_text_mask = df[TEXT_COL].apply(validate_text)
    invalid_count = (~valid_text_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} rows with invalid text (too short or non-alphabetic)")
        df = df[valid_text_mask]
        logger.info(f"After text validation: {len(df)} rows")
    
    # Group items by construct
    construct_to_items = defaultdict(list)
    for idx, row in df.iterrows():
        # Additional validation at the row level
        if validate_text(row[TEXT_COL]):
            construct_to_items[row[LABEL_COL]].append((idx, row[TEXT_COL]))
    
    # Generate construct statistics
    construct_stats = []
    logger.info(f"Found {len(construct_to_items)} unique constructs with valid items.")
    for construct, items in construct_to_items.items():
        expected_pair_count = expected_pairs(len(items))
        construct_stats.append({
            'construct': construct,
            'item_count': len(items),
            'expected_pairs': expected_pair_count
        })
    
    # Save construct statistics
    stats_df = pd.DataFrame(construct_stats)
    stats_df = stats_df.sort_values(by='item_count', ascending=False)
    stats_df.to_csv(STATS_CSV, index=False)
    
    # Display construct statistics summary
    total_items = stats_df['item_count'].sum()
    total_expected_pairs = stats_df['expected_pairs'].sum()
    logger.info(f"Total items across all constructs: {total_items}")
    logger.info(f"Theoretical maximum possible pairs: {total_expected_pairs}")
    
    # Log construct counts by size
    size_bins = [0, 1, 2, 5, 10, 20, 50, 100, float('inf')]
    size_labels = ['0', '1', '2-4', '5-9', '10-19', '20-49', '50-99', '100+']
    construct_size_counts = []
    
    for i in range(len(size_bins)-1):
        count = ((stats_df['item_count'] > size_bins[i]) & 
                 (stats_df['item_count'] <= size_bins[i+1])).sum()
        construct_size_counts.append(count)
        
    logger.info("Construct count by number of items:")
    for label, count in zip(size_labels, construct_size_counts):
        logger.info(f"  {label} items: {count} constructs")
    
    # Log the top 10 largest constructs
    logger.info("Top 10 largest constructs:")
    for _, row in stats_df.head(10).iterrows():
        logger.info(f"  {row['construct']}: {row['item_count']} items ({row['expected_pairs']} possible pairs)")
    
    # Generate all possible pairs for each construct
    construct_pairs = {}
    total_possible_pairs = 0
    
    for construct, items in construct_to_items.items():
        if len(items) >= 2:  # Need at least 2 items to form a pair
            # Generate all possible combinations of 2 items within this construct
            pairs = list(combinations(items, 2))
            
            # Validate the generated pairs and randomize position of items in each pair
            valid_pairs = []
            for p in pairs:
                if (validate_text(p[0][1]) and validate_text(p[1][1]) and 
                    p[0][1].strip() != p[1][1].strip()):  # Ensure texts aren't identical
                    # Randomly swap the order of items in the pair (50% chance)
                    if random.random() < 0.5:
                        valid_pairs.append((p[1], p[0]))  # Swap positions
                    else:
                        valid_pairs.append(p)  # Keep original order
            
            if len(valid_pairs) > 0:
                construct_pairs[construct] = valid_pairs
                total_possible_pairs += len(valid_pairs)
                
                # Check if the pair count matches the expected count
                expected = expected_pairs(len(items))
                if len(valid_pairs) != expected:
                    logger.warning(f"Construct '{construct}': Expected {expected} pairs but generated {len(valid_pairs)}")
                else:
                    logger.info(f"Construct '{construct}': {len(items)} items → {len(valid_pairs)} pairs (correct)")
            else:
                logger.warning(f"Construct '{construct}': No valid pairs generated from {len(items)} items")
    
    logger.info(f"Generated {total_possible_pairs} total valid pairs across {len(construct_pairs)} constructs.")
    
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
    
    # Final validation before writing
    valid_rebalanced_pairs = []
    for pair in rebalanced_pairs:
        anchor_idx, anchor_text, positive_idx, positive_text, construct = pair
        
        # Final text validation
        if (validate_text(anchor_text) and validate_text(positive_text) and 
            anchor_text.strip() != positive_text.strip()):
            valid_rebalanced_pairs.append(pair)
    
    if len(valid_rebalanced_pairs) != len(rebalanced_pairs):
        logger.warning(f"Removed {len(rebalanced_pairs) - len(valid_rebalanced_pairs)} invalid pairs in final validation")
        rebalanced_pairs = valid_rebalanced_pairs
    
    # Write the pairs to output file
    pairs_written = 0
    constructs_in_output = set()
    
    logger.info(f"Writing {len(rebalanced_pairs)} validated pairs to {OUTPUT_JSONL}...")
    
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
    
    # Validate output file
    logger.info("Validating output file...")
    valid_lines = 0
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f_in:
        for line_num, line in enumerate(f_in, 1):
            try:
                pair = json.loads(line)
                if (validate_text(pair["anchor"]) and 
                    validate_text(pair["positive"]) and 
                    pair["anchor"] != pair["positive"]):
                    valid_lines += 1
                else:
                    logger.error(f"Line {line_num}: Invalid content in pair")
            except json.JSONDecodeError:
                logger.error(f"Line {line_num}: Invalid JSON")
    
    logger.info(f"Output validation: {valid_lines}/{pairs_written} lines are valid")
    
    # Print next steps
    logger.info("\nNext steps:")
    logger.info("1. Check the statistics in data/processed/ipip_construct_statistics.csv")
    logger.info("2. Run training with the validated dataset using: python scripts/train_ipip_mnrl.py --base_model all-MiniLM-L6-v2")

if __name__ == "__main__":
    main()