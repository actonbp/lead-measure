#!/usr/bin/env python3
"""
Generate embeddings for leadership measure items using various models.

This script loads processed leadership measurement items and generates
embeddings using multiple models (API-based and local).
"""

import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse
from pathlib import Path

# For API-based embeddings
import openai
import anthropic

# For local model embeddings
from sentence_transformers import SentenceTransformer

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "processed" / "embeddings"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_leadership_items(filename):
    """Load leadership items from a processed CSV file."""
    filepath = PROCESSED_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find processed data file: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} leadership items from {filename}")
    return df

def generate_openai_embeddings(texts, model="text-embedding-ada-002"):
    """Generate embeddings using OpenAI's API."""
    embeddings = []
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating OpenAI embeddings"):
        batch = texts[i:i+batch_size]
        response = openai.Embedding.create(input=batch, model=model)
        batch_embeddings = [item["embedding"] for item in response["data"]]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def generate_transformer_embeddings(texts, model_name="all-mpnet-base-v2"):
    """Generate embeddings using a local transformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def save_embeddings(embeddings, metadata, model_name, output_file):
    """Save embeddings and metadata to disk."""
    output_path = OUTPUT_DIR / output_file
    
    # Create a dictionary with embeddings and metadata
    output_data = {
        "model": model_name,
        "dimensions": embeddings.shape[1],
        "count": embeddings.shape[0],
        "created": pd.Timestamp.now().isoformat(),
        "metadata": metadata,
        "embeddings": embeddings.tolist()
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Saved {embeddings.shape[0]} embeddings to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for leadership items")
    parser.add_argument("--input", required=True, help="Input CSV file in processed directory")
    parser.add_argument("--model", default="all-mpnet-base-v2", 
                        choices=["all-mpnet-base-v2", "openai", "anthropic"],
                        help="Embedding model to use")
    parser.add_argument("--output", help="Output filename (defaults to input filename with model name)")
    
    args = parser.parse_args()
    
    # Load leadership items
    df = load_leadership_items(args.input)
    
    # Prepare texts for embedding
    texts = df["item_text"].tolist()
    
    # Generate embeddings based on model choice
    if args.model == "openai":
        embeddings = generate_openai_embeddings(texts)
        model_name = "text-embedding-ada-002"
    elif args.model == "anthropic":
        # Placeholder for Anthropic embeddings
        raise NotImplementedError("Anthropic embeddings not yet implemented")
    else:
        # Use local transformer model
        embeddings = generate_transformer_embeddings(texts, args.model)
        model_name = args.model
    
    # Prepare metadata (subset of DataFrame excluding the actual item text)
    metadata = df.drop(columns=["item_text"]).to_dict(orient="records")
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        input_stem = Path(args.input).stem
        model_suffix = args.model.replace("-", "_")
        output_file = f"{input_stem}_{model_suffix}_embeddings.json"
    
    # Save embeddings
    save_embeddings(embeddings, metadata, model_name, output_file)

if __name__ == "__main__":
    main() 