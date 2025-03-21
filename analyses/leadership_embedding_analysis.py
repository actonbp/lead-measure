#!/usr/bin/env python3
"""
Leadership Construct Similarity Analysis using Embeddings

This script analyzes the semantic similarity between leadership constructs
using embedding models to test whether different leadership styles represent
distinct constructs or have substantial overlap.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import umap
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_leadership_data(file_path=None):
    """Load the leadership measures dataset."""
    if file_path is None:
        file_path = RAW_DATA_DIR / "Measures_text_long.csv"
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {file_path}")
    
    # Load data
    print(f"Loading leadership measures from {file_path}")
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    df = df.dropna(subset=['Text'])  # Remove items with no text
    
    print(f"Loaded {len(df)} leadership items across {df['Behavior'].nunique()} constructs")
    return df

def generate_embeddings(texts, model_name="all-mpnet-base-v2"):
    """Generate embeddings for the given texts using a sentence transformer model."""
    print(f"Generating embeddings using {model_name} model...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def compute_construct_similarities(df, embeddings):
    """Compute similarities between leadership constructs based on item embeddings."""
    print("Computing construct-level similarities...")
    
    # Group by constructs (Behavior column)
    constructs = df['Behavior'].unique()
    construct_indices = {construct: df.index[df['Behavior'] == construct].tolist() 
                         for construct in constructs}
    
    # Calculate mean embeddings for each construct
    construct_embeddings = {}
    for construct, indices in construct_indices.items():
        if indices:  # Check if there are any items for this construct
            construct_embeddings[construct] = np.mean(embeddings[indices], axis=0)
    
    # Compute pairwise similarities between construct embeddings
    construct_list = list(construct_embeddings.keys())
    similarity_matrix = np.zeros((len(construct_list), len(construct_list)))
    
    for i, c1 in enumerate(construct_list):
        for j, c2 in enumerate(construct_list):
            e1 = construct_embeddings[c1].reshape(1, -1)
            e2 = construct_embeddings[c2].reshape(1, -1)
            similarity_matrix[i, j] = cosine_similarity(e1, e2)[0, 0]
    
    return similarity_matrix, construct_list

def analyze_construct_distances(df, embeddings):
    """Analyze intra-construct and inter-construct distances."""
    print("Analyzing construct distances...")
    
    # Calculate all pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    # Convert to distances (1 - similarity)
    distances = 1 - similarities
    
    # Calculate intra-construct and inter-construct distances
    intra_distances = []
    inter_distances = []
    
    constructs = df['Behavior'].unique()
    
    for i in range(len(df)):
        for j in range(i+1, len(df)):  # Only upper triangle to avoid duplicates
            if df.iloc[i]['Behavior'] == df.iloc[j]['Behavior']:
                intra_distances.append(distances[i, j])
            else:
                inter_distances.append(distances[i, j])
    
    # Convert to numpy arrays
    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)
    
    # Calculate statistics
    intra_mean = np.mean(intra_distances)
    intra_std = np.std(intra_distances)
    inter_mean = np.mean(inter_distances)
    inter_std = np.std(inter_distances)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
    effect_size = (inter_mean - intra_mean) / pooled_std
    
    print(f"\nDistance Analysis Results:")
    print(f"Intra-construct distances: mean = {intra_mean:.4f}, std = {intra_std:.4f}")
    print(f"Inter-construct distances: mean = {inter_mean:.4f}, std = {inter_std:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    
    results = {
        'intra_mean': intra_mean,
        'intra_std': intra_std,
        'inter_mean': inter_mean,
        'inter_std': inter_std,
        'effect_size': effect_size,
        'intra_distances': intra_distances,
        'inter_distances': inter_distances
    }
    
    return results

def visualize_construct_similarity(similarity_matrix, construct_list, output_file=None):
    """Visualize the similarity matrix between leadership constructs."""
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        mask=mask,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        xticklabels=construct_list,
        yticklabels=construct_list,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title("Semantic Similarity Between Leadership Constructs", fontsize=16)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {output_file}")
    else:
        plt.show()
    
    plt.close()

def visualize_embeddings_2d(df, embeddings, output_file=None):
    """Create a 2D visualization of the leadership items using UMAP."""
    print("Creating 2D visualization of embeddings...")
    
    # Reduce dimensionality to 2D using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'construct': df['Behavior'],
        'item': df['Text'].str[:30] + '...'  # First 30 chars of item text
    })
    
    # Get top N most frequent constructs for clearer visualization
    top_n = 10
    top_constructs = df['Behavior'].value_counts().nlargest(top_n).index.tolist()
    plot_df['plot_construct'] = plot_df['construct'].apply(
        lambda x: x if x in top_constructs else 'Other')
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Use a good colormap with distinct colors
    palette = sns.color_palette("husl", n_colors=len(plot_df['plot_construct'].unique()))
    
    # Plot the points
    scatter = sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='plot_construct',
        palette=palette,
        alpha=0.7,
        s=50
    )
    
    plt.title("Leadership Items in Semantic Space (2D UMAP Projection)", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Leadership Construct")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved UMAP visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    return plot_df, embeddings_2d

def visualize_distance_distributions(distance_results, output_file=None):
    """Visualize the distribution of intra-construct and inter-construct distances."""
    plt.figure(figsize=(10, 6))
    
    sns.histplot(
        distance_results['intra_distances'], 
        alpha=0.5, 
        label=f"Intra-construct (mean={distance_results['intra_mean']:.4f})",
        kde=True
    )
    
    sns.histplot(
        distance_results['inter_distances'], 
        alpha=0.5, 
        label=f"Inter-construct (mean={distance_results['inter_mean']:.4f})",
        kde=True
    )
    
    plt.axvline(distance_results['intra_mean'], color='blue', linestyle='--')
    plt.axvline(distance_results['inter_mean'], color='orange', linestyle='--')
    
    plt.title(f"Distribution of Semantic Distances Between Leadership Items\nEffect Size (d) = {distance_results['effect_size']:.4f}", 
              fontsize=14)
    plt.xlabel("Semantic Distance (1 - Cosine Similarity)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved distance distribution plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def save_processed_data(df, embeddings, embedding_model, output_file=None):
    """Save processed data and embeddings for future use."""
    if output_file is None:
        output_file = PROCESSED_DIR / "leadership_embeddings.pkl"
    
    # Create a dictionary with all relevant data
    data = {
        'df': df,
        'embeddings': embeddings,
        'embedding_model': embedding_model,
        'processed_date': pd.Timestamp.now().isoformat()
    }
    
    # Save to pickle file
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved processed data and embeddings to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze leadership constructs using embeddings")
    parser.add_argument("--data-file", type=str, help="Path to leadership measures data file")
    parser.add_argument("--model", default="all-mpnet-base-v2", 
                      help="Embedding model to use (default: all-mpnet-base-v2)")
    parser.add_argument("--skip-embeddings", action="store_true", 
                      help="Skip embedding generation (use pre-computed embeddings)")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save outputs (default: PROJECT_ROOT/outputs)")
    
    args = parser.parse_args()
    
    # Set output directory
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    data_file = Path(args.data_file) if args.data_file else None
    leadership_df = load_leadership_data(data_file)
    
    # Generate or load embeddings
    if args.skip_embeddings:
        # Load pre-computed embeddings (placeholder - would need implementation)
        raise NotImplementedError("Loading pre-computed embeddings not yet implemented")
    else:
        # Generate embeddings
        embeddings = generate_embeddings(leadership_df['Text'].tolist(), model_name=args.model)
    
    # Analyze distances between constructs
    distance_results = analyze_construct_distances(leadership_df, embeddings)
    
    # Compute construct-level similarities
    similarity_matrix, construct_list = compute_construct_similarities(leadership_df, embeddings)
    
    # Save processed data
    save_processed_data(leadership_df, embeddings, args.model)
    
    # Visualizations
    visualize_construct_similarity(
        similarity_matrix, 
        construct_list,
        output_file=OUTPUT_DIR / "construct_similarity_heatmap.png"
    )
    
    visualize_distance_distributions(
        distance_results,
        output_file=OUTPUT_DIR / "distance_distributions.png"
    )
    
    visualize_embeddings_2d(
        leadership_df, 
        embeddings,
        output_file=OUTPUT_DIR / "leadership_embedding_2d.png"
    )
    
    print("\nAnalysis complete. Key findings:")
    if distance_results['effect_size'] < 0.2:
        print("- Small effect size suggests substantial overlap between leadership constructs")
    elif distance_results['effect_size'] < 0.5:
        print("- Medium effect size suggests moderate distinction between leadership constructs")
    else:
        print("- Large effect size suggests leadership constructs are relatively distinct")
    
    print(f"- On average, items within the same construct are {distance_results['intra_mean']:.4f} units apart")
    print(f"- On average, items from different constructs are {distance_results['inter_mean']:.4f} units apart")
    
    print("\nNext steps could include:")
    print("1. Compare these results with a similar analysis on personality trait measures")
    print("2. Implement triplet training to potentially improve embedding quality")
    print("3. Perform cluster analysis to identify natural groupings in the semantic space")
    print("4. Analyze specific construct pairs that show particularly high or low similarity")

if __name__ == "__main__":
    main() 