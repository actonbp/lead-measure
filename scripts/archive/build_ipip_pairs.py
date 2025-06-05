"""Generate anchor-positive text pairs from IPIP personality items.

This script reads the IPIP personality items CSV file and generates anchor-positive
pairs for training with the GIST loss function. Pairs are constructed such that
both items belong to the same personality construct.

Usage:
    python scripts/build_ipip_pairs.py

Outputs:
    data/processed/ipip_pairs.jsonl - A JSONL file containing anchor-positive pairs
"""
import pandas as pd
import json
import random
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_CSV = "data/IPIP.csv"
OUTPUT_JSONL = "data/processed/ipip_pairs.jsonl"
TEXT_COL = "text"
LABEL_COL = "label"

# Ensure reproducibility
random.seed(42)

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

    # Group indices by label for fast sampling
    label_to_indices = {}
    for idx, label in enumerate(df[LABEL_COL]):
        label_to_indices.setdefault(label, []).append(idx)

    pairs_created = 0
    constructs_used = set()

    logger.info(f"Generating anchor-positive pairs...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for idx, row in df.iterrows():
            label = row[LABEL_COL]
            constructs_used.add(label)
            
            # Find other items with same label (excluding self)
            candidates = [i for i in label_to_indices[label] if i != idx]
            
            if candidates:
                # Select a random positive example
                positive_idx = random.choice(candidates)
                positive_text = df.iloc[positive_idx][TEXT_COL]
                
                # Write the pair to output file
                pair = {
                    "anchor": row[TEXT_COL],
                    "positive": positive_text
                }
                f_out.write(json.dumps(pair) + "\n")
                pairs_created += 1
    
    logger.info(f"Created {pairs_created} anchor-positive pairs across {len(constructs_used)} constructs.")
    logger.info(f"Output saved to {OUTPUT_JSONL}")
    
    # Print next steps
    logger.info("\nNext steps:")
    logger.info("1. Run evaluation: python scripts/evaluate_ipip_model.py")
    logger.info("2. Train full model: python scripts/train_gist_ipip.py")

if __name__ == "__main__":
    main() 