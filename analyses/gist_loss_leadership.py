#!/usr/bin/env python3
"""
GIST Loss Implementation for Leadership Style Analysis

This script implements the GIST loss approach to analyze positive leadership styles
from the processed datasets, visualizing the clusters with UMAP.

Usage:
    python gist_loss_leadership.py

Output:
    - Visualizations of embeddings in the visualizations directory
    - Trained model saved to models directory
    - Analysis reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Import SentenceTransformers
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Positive Leadership styles to include
POSITIVE_LEADERSHIP_STYLES = [
    'Authentic', 
    'Charismatic', 
    'Consideration', 
    'Initiating Structure',
    'Empowering', 
    'Ethical', 
    'Instrumental', 
    'Servant', 
    'Transformational'
]

def load_leadership_data(dataset_name="focused_clean"):
    """Load the specified processed leadership dataset."""
    file_path = PROCESSED_DIR / f"leadership_{dataset_name}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Leadership dataset not found at {file_path}")
    
    print(f"Loading leadership dataset from {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} leadership items")
    return df

def filter_positive_leadership_styles(df, construct_column="StandardConstruct"):
    """Filter for only positive leadership styles."""
    # Filter to only positive styles
    positive_df = df[df[construct_column].isin(POSITIVE_LEADERSHIP_STYLES)].copy()
    
    print(f"Filtered to {len(positive_df)} items from positive leadership styles")
    print(positive_df[construct_column].value_counts())
    
    return positive_df

def prepare_gist_training_data(df, text_column="ProcessedText", label_column="StandardConstruct"):
    """
    Prepare training data for GIST loss fine-tuning.
    
    GIST loss works with positive pairs (items from the same category).
    """
    # Split data into train/test
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[label_column], random_state=42
    )
    
    # Group items by leadership style
    style_to_items = {}
    for style in train_df[label_column].unique():
        style_items = train_df[train_df[label_column] == style][text_column].tolist()
        style_to_items[style] = style_items
    
    # Create training examples - pairs from the same leadership style
    train_examples = []
    
    for style, items in style_to_items.items():
        # Ensure we have at least 2 items per style
        if len(items) < 2:
            continue
            
        # Create positive pairs
        for i, anchor in enumerate(items):
            # Sample a few positive items for each anchor
            pos_indices = np.random.choice(
                [j for j in range(len(items)) if j != i],
                min(5, len(items) - 1),
                replace=False
            )
            
            for pos_idx in pos_indices:
                train_examples.append(InputExample(texts=[anchor, items[pos_idx]]))
    
    print(f"Created {len(train_examples)} training pairs")
    
    return train_examples, train_df, test_df

def create_evaluator(df, style_column="StandardConstruct", text_column="ProcessedText"):
    """Create evaluator for measuring model performance during training."""
    # Create arrays for sentences1, sentences2, and scores
    sentences1 = []
    sentences2 = []
    scores = []
    
    styles = df[style_column].unique()
    
    # Generate positive pairs (same style)
    for style in styles:
        style_items = df[df[style_column] == style][text_column].tolist()
        if len(style_items) >= 2:
            for i in range(min(len(style_items), 20)):
                for j in range(i+1, min(len(style_items), 20)):
                    sentences1.append(style_items[i])
                    sentences2.append(style_items[j])
                    scores.append(1.0)  # 1.0 = similar
    
    # Generate negative pairs (different styles)
    for i in range(min(len(sentences1), 200)):
        # Randomly select two different styles
        style1, style2 = np.random.choice(styles, 2, replace=False)
        
        # Get items from each style
        item1 = np.random.choice(df[df[style_column] == style1][text_column].tolist())
        item2 = np.random.choice(df[df[style_column] == style2][text_column].tolist())
        
        sentences1.append(item1)
        sentences2.append(item2)
        scores.append(0.0)  # 0.0 = dissimilar
    
    return evaluation.EmbeddingSimilarityEvaluator(sentences1=sentences1, sentences2=sentences2, scores=scores)

def train_gist_model(train_examples, evaluator, model_name="all-mpnet-base-v2", epochs=2):
    """Train a SentenceTransformer model using GIST loss."""
    print(f"Training model with GIST loss using {model_name} as base model")
    
    # Initialize model
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Initialize MultipleNegativesRankingLoss (serves as GIST loss implementation)
    # This loss function works with positive pairs and uses in-batch negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Set up warmup steps
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(MODELS_DIR / f"leadership_gist_{model_name}")
    )
    
    return model

def generate_raw_embeddings(df, text_column="ProcessedText", model_name="all-mpnet-base-v2"):
    """Generate raw embeddings without GIST loss for comparison."""
    print(f"Generating raw embeddings using {model_name}")
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings, model

def generate_gist_embeddings(model, df, text_column="ProcessedText"):
    """Generate embeddings using the GIST-trained model."""
    print("Generating embeddings with GIST-trained model")
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def analyze_distances(embeddings, df, style_column="StandardConstruct"):
    """Analyze intra-style and inter-style distances."""
    print("Analyzing semantic distances between leadership styles")
    
    # Calculate all pairwise distances
    similarities = cosine_similarity(embeddings)
    distances = 1 - similarities
    
    # Get style labels
    styles = df[style_column].values
    
    # Initialize containers for distances
    intra_style_distances = []
    inter_style_distances = []
    
    # Collect distances
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):  # Upper triangle only
            if styles[i] == styles[j]:
                intra_style_distances.append(distances[i, j])
            else:
                inter_style_distances.append(distances[i, j])
    
    # Convert to numpy arrays
    intra_distances = np.array(intra_style_distances)
    inter_distances = np.array(inter_style_distances)
    
    # Calculate statistics
    intra_mean = np.mean(intra_distances)
    intra_std = np.std(intra_distances)
    inter_mean = np.mean(inter_distances)
    inter_std = np.std(inter_distances)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
    effect_size = (inter_mean - intra_mean) / pooled_std
    
    print(f"Intra-style distances: mean = {intra_mean:.4f}, std = {intra_std:.4f}")
    print(f"Inter-style distances: mean = {inter_mean:.4f}, std = {inter_std:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot histograms of distances
    sns.histplot(intra_distances, alpha=0.5, label=f"Within style: μ={intra_mean:.4f}", kde=True)
    sns.histplot(inter_distances, alpha=0.5, label=f"Between styles: μ={inter_mean:.4f}", kde=True)
    
    plt.axvline(intra_mean, color='blue', linestyle='--')
    plt.axvline(inter_mean, color='orange', linestyle='--')
    
    plt.title(f"Leadership Style Semantic Distance Distribution (Effect Size d={effect_size:.4f})", fontsize=14)
    plt.xlabel("Semantic Distance (1 - Cosine Similarity)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / "leadership_distance_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved distance distribution to {output_file}")
    
    return {
        "intra_mean": intra_mean,
        "intra_std": intra_std,
        "inter_mean": inter_mean,
        "inter_std": inter_std,
        "effect_size": effect_size
    }

def visualize_embeddings_umap(raw_embeddings, gist_embeddings, df, style_column="StandardConstruct"):
    """Create comparative UMAP visualization of raw vs GIST embeddings."""
    print("Creating comparative UMAP visualization")
    
    # Apply UMAP to both embeddings
    reducer_raw = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    
    reducer_gist = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    
    raw_2d = reducer_raw.fit_transform(raw_embeddings)
    gist_2d = reducer_gist.fit_transform(gist_embeddings)
    
    # Create DataFrames for plotting
    raw_df = pd.DataFrame({
        "x": raw_2d[:, 0],
        "y": raw_2d[:, 1],
        "style": df[style_column],
        "text": df["ProcessedText"]
    })
    
    gist_df = pd.DataFrame({
        "x": gist_2d[:, 0],
        "y": gist_2d[:, 1],
        "style": df[style_column],
        "text": df["ProcessedText"]
    })
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Get unique styles and set color palette
    styles = df[style_column].unique()
    palette = sns.color_palette("husl", len(styles))
    
    # Plot raw embeddings
    for i, style in enumerate(styles):
        subset = raw_df[raw_df["style"] == style]
        axes[0].scatter(
            subset["x"], 
            subset["y"], 
            c=[palette[i]] * len(subset),
            label=style,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    axes[0].set_title("Raw Embeddings - UMAP Projection", fontsize=16)
    axes[0].set_xlabel("UMAP Dimension 1", fontsize=12)
    axes[0].set_ylabel("UMAP Dimension 2", fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Plot GIST embeddings
    for i, style in enumerate(styles):
        subset = gist_df[gist_df["style"] == style]
        axes[1].scatter(
            subset["x"], 
            subset["y"], 
            c=[palette[i]] * len(subset),
            label=style,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    axes[1].set_title("GIST-trained Embeddings - UMAP Projection", fontsize=16)
    axes[1].set_xlabel("UMAP Dimension 1", fontsize=12)
    axes[1].set_ylabel("UMAP Dimension 2", fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / "leadership_umap_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP comparison to {output_file}")
    
    return {
        "raw_df": raw_df,
        "gist_df": gist_df,
        "raw_2d": raw_2d,
        "gist_2d": gist_2d
    }

def perform_cluster_analysis(embeddings, df, embedding_2d, style_column="StandardConstruct", prefix="gist"):
    """Perform HDBSCAN clustering and compare to actual style labels."""
    print(f"Performing HDBSCAN clustering on {prefix} embeddings")
    
    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    
    # Cluster using the 2D UMAP embeddings for better results
    cluster_labels = clusterer.fit_predict(embedding_2d)
    
    # Create result DataFrame
    result_df = df.copy()
    result_df["cluster"] = cluster_labels
    result_df["cluster_label"] = result_df["cluster"].apply(
        lambda x: f"Cluster {x}" if x >= 0 else "Noise"
    )
    
    # Add UMAP coordinates
    result_df["umap_x"] = embedding_2d[:, 0]
    result_df["umap_y"] = embedding_2d[:, 1]
    
    # Calculate cluster metrics
    true_labels = pd.Categorical(df[style_column]).codes
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    ami_score = adjusted_mutual_info_score(true_labels, cluster_labels)
    
    print(f"Number of clusters found: {len(set(cluster_labels) - {-1})}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Adjusted Mutual Information: {ami_score:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Clusters
    cluster_palette = sns.color_palette("husl", len(result_df["cluster_label"].unique()))
    
    for i, cluster in enumerate(sorted(result_df["cluster_label"].unique())):
        mask = result_df["cluster_label"] == cluster
        axes[0].scatter(
            result_df.loc[mask, "umap_x"],
            result_df.loc[mask, "umap_y"],
            label=cluster,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    axes[0].set_title(f"HDBSCAN Clusters ({prefix})", fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    
    # Plot 2: True styles
    style_palette = sns.color_palette("deep", len(result_df[style_column].unique()))
    
    for i, style in enumerate(sorted(result_df[style_column].unique())):
        mask = result_df[style_column] == style
        axes[1].scatter(
            result_df.loc[mask, "umap_x"],
            result_df.loc[mask, "umap_y"],
            label=style,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    axes[1].set_title("True Leadership Styles", fontsize=14)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    
    # Add metrics to figure
    plt.figtext(
        0.5, 0.01,
        f"Clustering Metrics - ARI: {ari_score:.3f}, AMI: {ami_score:.3f}\n"
        f"Number of clusters found: {len(set(cluster_labels) - {-1})}, "
        f"Noise points: {(cluster_labels == -1).sum()}",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / f"leadership_{prefix}_cluster_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved cluster analysis to {output_file}")
    
    # Create heatmap showing style-cluster relationship
    plt.figure(figsize=(12, 8))
    
    # Create crosstab of style vs cluster
    style_cluster_crosstab = pd.crosstab(
        result_df[style_column],
        result_df["cluster_label"],
        normalize="index"  # Normalize by row (style)
    )
    
    # Plot heatmap
    sns.heatmap(
        style_cluster_crosstab,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    
    plt.title(f"Relationship Between Leadership Styles and Clusters ({prefix})", fontsize=14)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file = VISUALIZATIONS_DIR / f"leadership_{prefix}_style_cluster_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    print(f"Saved style-cluster heatmap to {heatmap_file}")
    
    return {
        "ari_score": ari_score,
        "ami_score": ami_score,
        "num_clusters": len(set(cluster_labels) - {-1}),
        "noise_points": (cluster_labels == -1).sum()
    }

def save_results(raw_embeddings, gist_embeddings, df, metrics, model_name):
    """Save embeddings, data, and metrics for future use."""
    results = {
        "raw_embeddings": raw_embeddings,
        "gist_embeddings": gist_embeddings,
        "data": df,
        "metrics": metrics,
        "model_name": model_name
    }
    
    output_file = PROCESSED_DIR / "leadership_gist_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Saved results to {output_file}")
    
    # Also save a simple text report
    report = f"""
    Leadership Style Analysis with GIST Loss
    =======================================
    
    Model: {model_name}
    Dataset: leadership_focused_clean.csv
    Styles analyzed: {', '.join(POSITIVE_LEADERSHIP_STYLES)}
    
    Raw Embeddings:
    - Effect size: {metrics['raw']['distance']['effect_size']:.4f}
    - Intra-style distance: {metrics['raw']['distance']['intra_mean']:.4f}
    - Inter-style distance: {metrics['raw']['distance']['inter_mean']:.4f}
    - Clusters found: {metrics['raw']['clustering']['num_clusters']}
    - ARI score: {metrics['raw']['clustering']['ari_score']:.4f}
    - AMI score: {metrics['raw']['clustering']['ami_score']:.4f}
    
    GIST Embeddings:
    - Effect size: {metrics['gist']['distance']['effect_size']:.4f}
    - Intra-style distance: {metrics['gist']['distance']['intra_mean']:.4f}
    - Inter-style distance: {metrics['gist']['distance']['inter_mean']:.4f}
    - Clusters found: {metrics['gist']['clustering']['num_clusters']}
    - ARI score: {metrics['gist']['clustering']['ari_score']:.4f}
    - AMI score: {metrics['gist']['clustering']['ami_score']:.4f}
    
    Interpretation:
    - Effect size values below 0.2 indicate high overlap between leadership constructs
    - Higher ARI/AMI scores suggest better alignment between clusters and true constructs
    
    Questions for experts:
    1. Are the small effect sizes expected, suggesting leadership constructs are indeed overlapping?
    2. Is GIST loss an appropriate method for this type of construct validation?
    3. Given the results, should we consider a more fundamental restructuring of leadership constructs?
    """
    
    report_file = VISUALIZATIONS_DIR / "leadership_analysis_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"Saved text report to {report_file}")

def main():
    # 1. Load and preprocess data
    leadership_df = load_leadership_data(dataset_name="focused_clean")
    positive_df = filter_positive_leadership_styles(leadership_df)
    
    # 2. Generate raw embeddings for comparison
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Fast model for quick results
    raw_embeddings, base_model = generate_raw_embeddings(positive_df, model_name=model_name)
    
    # 3. Prepare training data for GIST model
    train_examples, train_df, test_df = prepare_gist_training_data(positive_df)
    evaluator = create_evaluator(test_df)
    
    # 4. Train model with GIST loss
    gist_model = train_gist_model(train_examples, evaluator, model_name)
    
    # 5. Generate GIST embeddings
    gist_embeddings = generate_gist_embeddings(gist_model, positive_df)
    
    # 6. Compare raw vs GIST embeddings
    metrics = {"raw": {}, "gist": {}}
    
    # 6.1 Analyze raw embeddings
    raw_distance_metrics = analyze_distances(raw_embeddings, positive_df)
    metrics["raw"]["distance"] = raw_distance_metrics
    
    # 6.2 Analyze GIST embeddings
    gist_distance_metrics = analyze_distances(gist_embeddings, positive_df)
    metrics["gist"]["distance"] = gist_distance_metrics
    
    # 7. Visualize with UMAP
    umap_results = visualize_embeddings_umap(raw_embeddings, gist_embeddings, positive_df)
    
    # 8. Perform cluster analysis
    raw_cluster_metrics = perform_cluster_analysis(
        raw_embeddings, positive_df, umap_results["raw_2d"], prefix="raw"
    )
    metrics["raw"]["clustering"] = raw_cluster_metrics
    
    gist_cluster_metrics = perform_cluster_analysis(
        gist_embeddings, positive_df, umap_results["gist_2d"], prefix="gist"
    )
    metrics["gist"]["clustering"] = gist_cluster_metrics
    
    # 9. Save results
    save_results(raw_embeddings, gist_embeddings, positive_df, metrics, model_name)
    
    print("\nAnalysis complete! Results saved to the visualizations directory.")
    print("Key findings:")
    print(f"Raw embeddings effect size: {raw_distance_metrics['effect_size']:.4f}")
    print(f"GIST embeddings effect size: {gist_distance_metrics['effect_size']:.4f}")
    
    if raw_distance_metrics['effect_size'] < 0.2 and gist_distance_metrics['effect_size'] < 0.2:
        print("\nBoth methods show small effect sizes, suggesting substantial overlap between leadership constructs.")
    elif gist_distance_metrics['effect_size'] > raw_distance_metrics['effect_size']:
        print("\nGIST loss improved separation between leadership constructs, but overlap still exists.")
    else:
        print("\nRaw embeddings show better separation between constructs than GIST loss approach.")

if __name__ == "__main__":
    main()