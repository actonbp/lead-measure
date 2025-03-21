#!/usr/bin/env python3
"""
Analyze similarities between leadership constructs based on embeddings.

This script loads pre-computed embeddings and analyzes semantic similarities
between leadership items and constructs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "processed" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "analyses" / "python" / "outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_embeddings(filename):
    """Load embeddings from a JSON file."""
    filepath = EMBEDDINGS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find embeddings file: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    embeddings = np.array(data["embeddings"])
    metadata = pd.DataFrame(data["metadata"])
    model_info = {k: v for k, v in data.items() if k not in ["embeddings", "metadata"]}
    
    print(f"Loaded {len(embeddings)} embeddings from {filename}")
    return embeddings, metadata, model_info

def compute_item_similarities(embeddings):
    """Compute pairwise cosine similarities between all items."""
    similarities = cosine_similarity(embeddings)
    return similarities

def compute_construct_similarities(embeddings, metadata, construct_col="construct"):
    """Compute average similarities between constructs."""
    constructs = metadata[construct_col].unique()
    n_constructs = len(constructs)
    
    # Create a mapping of constructs to their indices
    construct_to_idx = {construct: i for i, construct in enumerate(constructs)}
    
    # Initialize similarity matrix
    construct_similarities = np.zeros((n_constructs, n_constructs))
    construct_counts = np.zeros((n_constructs, n_constructs))
    
    # Compute item-level similarities
    item_similarities = compute_item_similarities(embeddings)
    
    # Aggregate to construct level
    for i in range(len(metadata)):
        for j in range(len(metadata)):
            if i != j:  # Skip self-comparisons
                construct_i = metadata[construct_col].iloc[i]
                construct_j = metadata[construct_col].iloc[j]
                
                idx_i = construct_to_idx[construct_i]
                idx_j = construct_to_idx[construct_j]
                
                construct_similarities[idx_i, idx_j] += item_similarities[i, j]
                construct_counts[idx_i, idx_j] += 1
    
    # Average the similarities
    # Replace zeros with ones in counts to avoid division by zero
    construct_counts = np.where(construct_counts == 0, 1, construct_counts)
    avg_construct_similarities = construct_similarities / construct_counts
    
    return avg_construct_similarities, constructs

def visualize_construct_similarities(similarities, constructs, output_file):
    """Create a heatmap of construct similarities."""
    plt.figure(figsize=(12, 10))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(similarities, dtype=bool))
    
    # Customize the heatmap
    sns.heatmap(
        similarities,
        mask=mask,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        xticklabels=constructs,
        yticklabels=constructs,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title("Semantic Similarity Between Leadership Constructs", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    output_path = OUTPUT_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")
    
    plt.close()

def visualize_embedding_space(embeddings, metadata, construct_col="construct", output_file=None, method="tsne"):
    """Visualize the embedding space using dimensionality reduction."""
    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = "t-SNE"
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = "UMAP"
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'tsne' or 'umap'.")
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "construct": metadata[construct_col]
    })
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot points colored by construct
    constructs = plot_df["construct"].unique()
    palette = sns.color_palette("husl", len(constructs))
    
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="construct",
        palette=palette,
        alpha=0.7,
        s=100
    )
    
    plt.title(f"Leadership Constructs in Embedding Space ({method_name})", fontsize=16)
    plt.xlabel(f"{method_name} Dimension 1", fontsize=12)
    plt.ylabel(f"{method_name} Dimension 2", fontsize=12)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    
    # Save the figure if output_file is provided
    if output_file:
        output_path = OUTPUT_DIR / output_file
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved embedding visualization to {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze similarities between leadership constructs")
    parser.add_argument("--input", required=True, help="Input embeddings file in embeddings directory")
    parser.add_argument("--construct", default="construct", help="Column name for construct in metadata")
    parser.add_argument("--output-prefix", default="leadership", help="Prefix for output files")
    parser.add_argument("--reduction", default="tsne", choices=["tsne", "umap"], help="Dimensionality reduction method")
    
    args = parser.parse_args()
    
    # Load embeddings
    embeddings, metadata, model_info = load_embeddings(args.input)
    
    # Compute construct similarities
    construct_similarities, constructs = compute_construct_similarities(
        embeddings, metadata, construct_col=args.construct)
    
    # Visualize construct similarities
    heatmap_file = f"{args.output_prefix}_construct_similarities.png"
    visualize_construct_similarities(construct_similarities, constructs, heatmap_file)
    
    # Visualize embedding space
    embedding_file = f"{args.output_prefix}_embedding_space_{args.reduction}.png"
    visualize_embedding_space(
        embeddings, metadata, construct_col=args.construct, 
        output_file=embedding_file, method=args.reduction)
    
    # Save construct similarity matrix
    similarity_df = pd.DataFrame(construct_similarities, index=constructs, columns=constructs)
    similarity_csv = OUTPUT_DIR / f"{args.output_prefix}_construct_similarities.csv"
    similarity_df.to_csv(similarity_csv)
    print(f"Saved similarity matrix to {similarity_csv}")

if __name__ == "__main__":
    main() 