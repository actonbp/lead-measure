#!/usr/bin/env python3
"""
Simplified Embedding Analysis for Leadership and Personality Traits

This script analyzes both Big Five personality traits and leadership styles
using sentence embeddings and UMAP visualization without fine-tuning.

Usage:
    python simplified_embedding_analysis.py

Output:
    - Visualizations of embeddings in the visualizations directory
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
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Import SentenceTransformers
from sentence_transformers import SentenceTransformer

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Big Five trait mapping - categories that map to the Big Five
BIG_FIVE_MAPPING = {
    'Openness': [
        'Complexity', 'Imagination', 'Creativity/Originality', 'Intellect', 
        'Intellectual Openness', 'Understanding', 'Depth', 'Culture', 'Ingenuity'
    ],
    'Conscientiousness': [
        'Orderliness', 'Dutifulness', 'Achievement-striving', 'Competence', 
        'Organization', 'Efficiency', 'Industriousness/Perseverance/Persistence',
        'Purposefulness', 'Deliberateness', 'Methodicalness'
    ],
    'Extraversion': [
        'Gregariousness', 'Assertiveness', 'Warmth', 'Talkativeness', 
        'Sociability', 'Vitality/Enthusiasm/Zest', 'Exhibitionism',
        'Leadership', 'Friendliness', 'Positive Expressivity'
    ],
    'Agreeableness': [
        'Compassion', 'Cooperation', 'Sympathy', 'Empathy', 'Nurturance',
        'Pleasantness', 'Tenderness', 'Morality', 'Docility'
    ],
    'Neuroticism': [
        'Anxiety', 'Emotionality', 'Anger', 'Distrust', 'Negative-Valence',
        'Stability', 'Emotional Stability'
    ]
}

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

def load_ipip_data():
    """Load and preprocess the IPIP personality items."""
    ipip_path = DATA_DIR / "ipip.csv"
    
    if not ipip_path.exists():
        raise FileNotFoundError(f"IPIP dataset not found at {ipip_path}")
    
    print(f"Loading IPIP dataset from {ipip_path}")
    
    # Load the dataset, handle encoding issues
    try:
        ipip_df = pd.read_csv(ipip_path, encoding='utf-8')
    except UnicodeDecodeError:
        ipip_df = pd.read_csv(ipip_path, encoding='latin1')
    
    print(f"Loaded {len(ipip_df)} personality items")
    return ipip_df

def filter_big_five_traits(ipip_df):
    """Filter items to only include Big Five personality traits."""
    # Map each item to its Big Five category if applicable
    def map_to_big_five(label):
        for big_five, labels in BIG_FIVE_MAPPING.items():
            if label in labels:
                return big_five
        return None
    
    # Apply the mapping
    ipip_df['big_five'] = ipip_df['label'].apply(map_to_big_five)
    
    # Filter to only include items mapped to Big Five
    big_five_df = ipip_df.dropna(subset=['big_five'])
    
    print(f"Filtered to {len(big_five_df)} items mapped to Big Five traits")
    print(big_five_df['big_five'].value_counts())
    
    return big_five_df

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

def generate_embeddings(df, text_column, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings for the given texts."""
    print(f"Generating embeddings using {model_name} model...")
    
    model = SentenceTransformer(model_name)
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def analyze_distances(embeddings, df, label_column):
    """Analyze intra-category and inter-category distances."""
    print(f"Analyzing semantic distances between {label_column} categories")
    
    # Calculate all pairwise distances
    similarities = cosine_similarity(embeddings)
    distances = 1 - similarities
    
    # Get category labels
    categories = df[label_column].values
    
    # Initialize containers for distances
    intra_distances = []
    inter_distances = []
    
    # Collect distances
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):  # Upper triangle only
            if categories[i] == categories[j]:
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
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
    effect_size = (inter_mean - intra_mean) / pooled_std
    
    print(f"Intra-category distances: mean = {intra_mean:.4f}, std = {intra_std:.4f}")
    print(f"Inter-category distances: mean = {inter_mean:.4f}, std = {inter_std:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot histograms of distances
    sns.histplot(intra_distances, alpha=0.5, label=f"Within category: μ={intra_mean:.4f}", kde=True)
    sns.histplot(inter_distances, alpha=0.5, label=f"Between categories: μ={inter_mean:.4f}", kde=True)
    
    plt.axvline(intra_mean, color='blue', linestyle='--')
    plt.axvline(inter_mean, color='orange', linestyle='--')
    
    plt.title(f"{label_column} Semantic Distance Distribution (Effect Size d={effect_size:.4f})", fontsize=14)
    plt.xlabel("Semantic Distance (1 - Cosine Similarity)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / f"{label_column.lower()}_distance_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved distance distribution to {output_file}")
    
    return {
        "intra_mean": intra_mean,
        "intra_std": intra_std,
        "inter_mean": inter_mean,
        "inter_std": inter_std,
        "effect_size": effect_size
    }

def visualize_embeddings_umap(embeddings, df, label_column, name):
    """Create UMAP visualization of embeddings colored by category."""
    print(f"Creating UMAP visualization for {name}")
    
    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        "x": embedding_2d[:, 0],
        "y": embedding_2d[:, 1],
        "category": df[label_column],
        "text": df.iloc[:, 3]  # Using the 4th column as text (varies by dataset)
    })
    
    # Plot with distinct colors
    plt.figure(figsize=(14, 10))
    
    # Get unique categories and set color palette
    categories = plot_df["category"].unique()
    palette = sns.color_palette("husl", len(categories))
    
    # Create scatter plot for each category
    for i, category in enumerate(categories):
        subset = plot_df[plot_df["category"] == category]
        plt.scatter(
            subset["x"], 
            subset["y"], 
            c=[palette[i]] * len(subset),
            label=category,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    plt.title(f"{name} UMAP Projection", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / f"{name.lower().replace(' ', '_')}_umap_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP visualization to {output_file}")
    
    return plot_df, embedding_2d

def perform_cluster_analysis(embeddings, df, embedding_2d, label_column, name):
    """Perform HDBSCAN clustering and compare to actual category labels."""
    print(f"Performing HDBSCAN clustering for {name}")
    
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
    true_labels = pd.Categorical(df[label_column]).codes
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
    
    axes[0].set_title(f"HDBSCAN Clusters ({name})", fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    
    # Plot 2: True categories
    category_palette = sns.color_palette("deep", len(result_df[label_column].unique()))
    
    for i, category in enumerate(sorted(result_df[label_column].unique())):
        mask = result_df[label_column] == category
        axes[1].scatter(
            result_df.loc[mask, "umap_x"],
            result_df.loc[mask, "umap_y"],
            label=category,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    axes[1].set_title(f"True {name} Categories", fontsize=14)
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
    output_file = VISUALIZATIONS_DIR / f"{name.lower().replace(' ', '_')}_cluster_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved cluster analysis to {output_file}")
    
    # Create heatmap showing category-cluster relationship
    plt.figure(figsize=(12, 8))
    
    # Create crosstab of category vs cluster
    category_cluster_crosstab = pd.crosstab(
        result_df[label_column],
        result_df["cluster_label"],
        normalize="index"  # Normalize by row (category)
    )
    
    # Plot heatmap
    sns.heatmap(
        category_cluster_crosstab,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    
    plt.title(f"Relationship Between {name} Categories and Clusters", fontsize=14)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file = VISUALIZATIONS_DIR / f"{name.lower().replace(' ', '_')}_category_cluster_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    print(f"Saved category-cluster heatmap to {heatmap_file}")
    
    return {
        "ari_score": ari_score,
        "ami_score": ami_score,
        "num_clusters": len(set(cluster_labels) - {-1}),
        "noise_points": (cluster_labels == -1).sum()
    }

def save_results(data):
    """Save results as a pickle file."""
    output_file = PROCESSED_DIR / "embedding_analysis_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved results to {output_file}")

def create_comparison_report(big_five_metrics, leadership_metrics):
    """Create a text report comparing Big Five and leadership results."""
    report = f"""
    Embedding Analysis Comparison: Big Five vs. Leadership Styles
    ===========================================================
    
    Effect Size Comparison:
    ----------------------
    Big Five Effect Size: {big_five_metrics['distance']['effect_size']:.4f}
    Leadership Effect Size: {leadership_metrics['distance']['effect_size']:.4f}
    
    Intra-Category Distances:
    -----------------------
    Big Five: {big_five_metrics['distance']['intra_mean']:.4f} ± {big_five_metrics['distance']['intra_std']:.4f}
    Leadership: {leadership_metrics['distance']['intra_mean']:.4f} ± {leadership_metrics['distance']['intra_std']:.4f}
    
    Inter-Category Distances:
    -----------------------
    Big Five: {big_five_metrics['distance']['inter_mean']:.4f} ± {big_five_metrics['distance']['inter_std']:.4f}
    Leadership: {leadership_metrics['distance']['inter_mean']:.4f} ± {leadership_metrics['distance']['inter_std']:.4f}
    
    Clustering Metrics:
    -----------------
    Big Five - Adjusted Rand Index: {big_five_metrics['clustering']['ari_score']:.4f}
    Leadership - Adjusted Rand Index: {leadership_metrics['clustering']['ari_score']:.4f}
    
    Big Five - Adjusted Mutual Information: {big_five_metrics['clustering']['ami_score']:.4f}
    Leadership - Adjusted Mutual Information: {leadership_metrics['clustering']['ami_score']:.4f}
    
    Clusters Found:
    -------------
    Big Five: {big_five_metrics['clustering']['num_clusters']} clusters (vs. 5 true categories)
    Leadership: {leadership_metrics['clustering']['num_clusters']} clusters (vs. {len(POSITIVE_LEADERSHIP_STYLES)} true categories)
    
    Interpretation:
    -------------
    {
    "Big Five traits show stronger separation" if big_five_metrics['distance']['effect_size'] > leadership_metrics['distance']['effect_size'] 
    else "Leadership styles show stronger separation"
    } with an effect size difference of {abs(big_five_metrics['distance']['effect_size'] - leadership_metrics['distance']['effect_size']):.4f}.
    
    The clustering metrics also show that {
    "Big Five categories are more easily distinguished" if big_five_metrics['clustering']['ari_score'] > leadership_metrics['clustering']['ari_score']
    else "Leadership categories are more easily distinguished"
    } by unsupervised clustering.
    
    Questions for further analysis:
    1. Why are leadership constructs more similar to each other than personality traits?
    2. Which specific leadership constructs show the greatest overlap?
    3. Would a different embedding approach (e.g., fine-tuning) improve the separation?
    """
    
    report_file = VISUALIZATIONS_DIR / "big_five_leadership_comparison_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"Saved comparison report to {report_file}")
    return report

def main():
    # Set up model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Part 1: Analyze Big Five personality traits
    print("\n==== Analyzing Big Five Personality Traits ====\n")
    # 1. Load and preprocess data
    ipip_df = load_ipip_data()
    big_five_df = filter_big_five_traits(ipip_df)
    
    # 2. Generate embeddings
    big_five_embeddings = generate_embeddings(big_five_df, text_column="text", model_name=model_name)
    
    # 3. Analyze distances
    big_five_distance_metrics = analyze_distances(big_five_embeddings, big_five_df, label_column="big_five")
    
    # 4. Visualize with UMAP
    big_five_umap_df, big_five_umap_coords = visualize_embeddings_umap(
        big_five_embeddings, big_five_df, label_column="big_five", name="Big Five Personality Traits"
    )
    
    # 5. Perform cluster analysis
    big_five_cluster_metrics = perform_cluster_analysis(
        big_five_embeddings, big_five_df, big_five_umap_coords, 
        label_column="big_five", name="Big Five Personality Traits"
    )
    
    # Part 2: Analyze leadership styles
    print("\n==== Analyzing Leadership Styles ====\n")
    # 1. Load and preprocess data
    leadership_df = load_leadership_data(dataset_name="focused_clean")
    positive_leadership_df = filter_positive_leadership_styles(leadership_df)
    
    # 2. Generate embeddings
    leadership_embeddings = generate_embeddings(
        positive_leadership_df, text_column="ProcessedText", model_name=model_name
    )
    
    # 3. Analyze distances
    leadership_distance_metrics = analyze_distances(
        leadership_embeddings, positive_leadership_df, label_column="StandardConstruct"
    )
    
    # 4. Visualize with UMAP
    leadership_umap_df, leadership_umap_coords = visualize_embeddings_umap(
        leadership_embeddings, positive_leadership_df, 
        label_column="StandardConstruct", name="Leadership Styles"
    )
    
    # 5. Perform cluster analysis
    leadership_cluster_metrics = perform_cluster_analysis(
        leadership_embeddings, positive_leadership_df, leadership_umap_coords,
        label_column="StandardConstruct", name="Leadership Styles"
    )
    
    # Aggregate results
    results = {
        "big_five": {
            "data": big_five_df,
            "embeddings": big_five_embeddings,
            "distance": big_five_distance_metrics,
            "clustering": big_five_cluster_metrics
        },
        "leadership": {
            "data": positive_leadership_df,
            "embeddings": leadership_embeddings,
            "distance": leadership_distance_metrics,
            "clustering": leadership_cluster_metrics
        },
        "model_name": model_name
    }
    
    # Save results
    save_results(results)
    
    # Generate comparison report
    comparison_report = create_comparison_report(
        results["big_five"], results["leadership"]
    )
    
    print("\nAnalysis complete! Results saved to the visualizations directory.")
    print(f"Big Five effect size: {big_five_distance_metrics['effect_size']:.4f}")
    print(f"Leadership effect size: {leadership_distance_metrics['effect_size']:.4f}")
    
    print("\nKey comparison findings:")
    print(f"- {'Big Five traits' if big_five_distance_metrics['effect_size'] > leadership_distance_metrics['effect_size'] else 'Leadership styles'} show stronger categorical separation")
    print(f"- {'Big Five categories' if big_five_cluster_metrics['ari_score'] > leadership_cluster_metrics['ari_score'] else 'Leadership categories'} are more easily distinguished by clustering")

if __name__ == "__main__":
    main()