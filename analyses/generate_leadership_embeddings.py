#!/usr/bin/env python3
"""
Generate Leadership Embeddings and Visualizations

This script generates embeddings for leadership measurement items using sentence-transformers,
and creates visualizations to explore semantic relationships between leadership constructs.

Usage:
    python generate_leadership_embeddings.py --dataset focused_clean --model all-mpnet-base-v2

Output:
    - Saved embeddings in numpy format
    - UMAP visualizations of the embedding space
    - Similarity matrices between leadership constructs
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import umap
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"

# Ensure directories exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Define available models and datasets
AVAILABLE_MODELS = {
    "all-mpnet-base-v2": "Most accurate model, but slower",
    "all-MiniLM-L6-v2": "Fast and efficient model, good balance of speed and accuracy",
    "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual model, handles multiple languages",
}

AVAILABLE_DATASETS = {
    "focused_clean": "Focused dataset with stems removed and gender-neutral language",
    "focused": "Focused dataset with only Fischer & Sitkin constructs",
    "original_clean": "Complete dataset with stems removed and gender-neutral language",
}

def load_dataset(dataset_name):
    """Load the specified processed dataset."""
    file_path = PROCESSED_DIR / f"leadership_{dataset_name}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} items from {file_path}")
    return df

def generate_embeddings(df, model_name, text_column, use_cache=True):
    """Generate embeddings using the specified model."""
    # Set up embedding cache path
    cache_file = EMBEDDINGS_DIR / f"{model_name}_{text_column}.pkl"
    
    # Return cached embeddings if they exist and use_cache is True
    if cache_file.exists() and use_cache:
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Load the model
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Extract text for embedding
    if text_column in df.columns:
        texts = df[text_column].tolist()
    else:
        raise ValueError(f"Column '{text_column}' not found in dataset")
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create embedding results
    embedding_results = {
        'model': model_name,
        'text_column': text_column,
        'embeddings': embeddings,
        'ids': df.index.tolist(),
        'texts': texts,
        'metadata': {col: df[col].tolist() for col in df.columns if col != text_column},
    }
    
    # Cache embeddings
    print(f"Saving embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embedding_results, f)
    
    return embedding_results

def visualize_embeddings_umap(embedding_results, color_by='StandardConstruct', n_neighbors=15, min_dist=0.1):
    """Create UMAP visualization of embeddings."""
    print("Generating UMAP visualization...")
    
    # Extract embeddings and metadata
    embeddings = embedding_results['embeddings']
    model_name = embedding_results['model']
    text_column = embedding_results['text_column']
    
    # Perform dimensionality reduction with UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'text': embedding_results['texts'],
    })
    
    # Add metadata for coloring
    if color_by in embedding_results['metadata']:
        plot_df[color_by] = embedding_results['metadata'][color_by]
    else:
        print(f"Warning: Column '{color_by}' not found in metadata. Available columns: {list(embedding_results['metadata'].keys())}")
        color_by = list(embedding_results['metadata'].keys())[0]
        plot_df[color_by] = embedding_results['metadata'][color_by]
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=plot_df, x='x', y='y', hue=color_by, palette='tab20', alpha=0.7)
    plt.title(f"UMAP Visualization of Leadership Constructs\nModel: {model_name}, Text: {text_column}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save visualization
    output_path = VISUALIZATIONS_DIR / f"umap_{model_name}_{text_column}_{color_by}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved UMAP visualization to {output_path}")
    plt.close()

def calculate_construct_similarities(embedding_results):
    """Calculate and visualize the similarities between leadership constructs."""
    print("Calculating construct similarities...")
    
    # Extract embeddings and metadata
    embeddings = embedding_results['embeddings']
    model_name = embedding_results['model']
    text_column = embedding_results['text_column']
    
    # Extract construct information
    if 'StandardConstruct' in embedding_results['metadata']:
        constructs = embedding_results['metadata']['StandardConstruct']
    else:
        constructs = embedding_results['metadata']['Behavior']
    
    # Get unique constructs
    unique_constructs = sorted(set(constructs))
    n_constructs = len(unique_constructs)
    
    # Create construct-level embeddings (mean of item embeddings)
    construct_embeddings = {}
    for construct in unique_constructs:
        indices = [i for i, c in enumerate(constructs) if c == construct]
        construct_embeddings[construct] = np.mean(embeddings[indices], axis=0)
    
    # Calculate cosine similarity between constructs
    construct_vectors = np.array([construct_embeddings[c] for c in unique_constructs])
    similarity_matrix = cosine_similarity(construct_vectors)
    
    # Create similarity dataframe
    similarity_df = pd.DataFrame(similarity_matrix, 
                                index=unique_constructs, 
                                columns=unique_constructs)
    
    # Visualize similarity matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_df, annot=True, cmap='viridis', vmin=0, vmax=1)
    plt.title(f"Semantic Similarity Between Leadership Constructs\nModel: {model_name}, Text: {text_column}")
    
    # Save visualization
    output_path = VISUALIZATIONS_DIR / f"similarity_matrix_{model_name}_{text_column}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved similarity matrix to {output_path}")
    plt.close()
    
    # Save similarity data
    similarity_df.to_csv(VISUALIZATIONS_DIR / f"similarity_matrix_{model_name}_{text_column}.csv")
    
    return similarity_df

def main():
    parser = argparse.ArgumentParser(description="Generate leadership embeddings and visualizations")
    parser.add_argument('--dataset', choices=AVAILABLE_DATASETS.keys(), default='focused_clean',
                        help='Dataset to use for embeddings')
    parser.add_argument('--model', choices=AVAILABLE_MODELS.keys(), default='all-mpnet-base-v2',
                        help='Sentence transformer model to use')
    parser.add_argument('--text-column', default='ProcessedText',
                        help='Column containing text to embed (ProcessedText or Text)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force regeneration of embeddings even if cached version exists')
    
    args = parser.parse_args()
    
    # Print info about selected options
    print(f"Selected dataset: {args.dataset} - {AVAILABLE_DATASETS[args.dataset]}")
    print(f"Selected model: {args.model} - {AVAILABLE_MODELS[args.model]}")
    print(f"Using text from column: {args.text_column}")
    
    # Load dataset
    df = load_dataset(args.dataset)
    
    # Generate embeddings
    embedding_results = generate_embeddings(df, args.model, args.text_column, not args.no_cache)
    
    # Visualize embeddings
    visualize_embeddings_umap(embedding_results)
    
    # Calculate construct similarities
    similarity_df = calculate_construct_similarities(embedding_results)
    
    print("\nEmbedding generation and visualization completed successfully.")

if __name__ == "__main__":
    main() 