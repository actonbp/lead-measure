"""Automatically select and apply the best performing model to leadership data.

This script:
1. Reads the comparison metrics to determine the best model based on IPIP ARI score
2. Applies that model to leadership data
3. Generates visualizations and analysis of leadership construct clustering

Usage:
    python scripts/apply_best_model_to_leadership.py

Outputs:
    data/visualizations/leadership_clusters/ - Visualizations of leadership item clustering
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
COMPARISON_METRICS = "data/visualizations/model_comparison/comparison_metrics.csv"
LEADERSHIP_CSV = "data/processed/leadership_focused_clean.csv"
OUTPUT_DIR = Path("data/visualizations/leadership_clusters")

def load_leadership_data():
    """Load and prepare leadership data for evaluation."""
    logger.info(f"Loading leadership data from {LEADERSHIP_CSV}...")
    df = pd.read_csv(LEADERSHIP_CSV)
    
    # Clean data
    df = df.dropna(subset=["Text", "StandardConstruct"])
    logger.info(f"Loaded {len(df)} leadership items across {df['StandardConstruct'].nunique()} constructs")
    
    # Return dataframe
    return df

def find_best_model():
    """Find the best performing model based on IPIP ARI score."""
    try:
        metrics_df = pd.read_csv(COMPARISON_METRICS)
        
        # Filter for IPIP dataset results
        ipip_results = metrics_df[metrics_df["dataset"] == "IPIP"]
        
        # Get model with highest ARI
        best_idx = ipip_results["ari"].idxmax()
        best_row = ipip_results.loc[best_idx]
        
        best_model_name = best_row["model"]
        best_model_path = None
        
        # Find the path for this model name in the comparison file
        for model_info in metrics_df[["model", "model_type"]].drop_duplicates().to_dict("records"):
            if model_info["model"] == best_model_name:
                # Look up the path from our list of models
                if "MNRL" in best_model_name:
                    best_model_path = "models/ipip_mnrl"
                elif "GIST" in best_model_name:
                    best_model_path = "models/gist_ipip"
                else:
                    best_model_path = "sentence-transformers/all-mpnet-base-v2"
        
        logger.info(f"Best model: {best_model_name} (ARI: {best_row['ari']:.4f})")
        logger.info(f"Model path: {best_model_path}")
        
        return best_model_name, best_model_path, best_row["ari"]
    
    except Exception as e:
        logger.error(f"Error finding best model: {str(e)}. Using default model instead.")
        return "Default model", "sentence-transformers/all-mpnet-base-v2", 0.0

def cluster_leadership_items(model, leadership_df):
    """Cluster leadership items using the provided model."""
    texts = leadership_df["Text"].tolist()
    constructs = leadership_df["StandardConstruct"].tolist()
    
    # Get unique constructs for labeling
    unique_constructs = sorted(set(constructs))
    n_clusters = len(unique_constructs)
    construct_to_id = {construct: i for i, construct in enumerate(unique_constructs)}
    construct_ids = [construct_to_id[c] for c in constructs]
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} leadership items...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Perform k-means clustering
    logger.info(f"Clustering embeddings into {n_clusters} clusters (one per construct)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Calculate metrics
    ari = adjusted_rand_score(construct_ids, predicted_clusters)
    nmi = normalized_mutual_info_score(construct_ids, predicted_clusters)
    
    # Calculate purity score
    cluster_to_constructs = defaultdict(list)
    for i, cluster_id in enumerate(predicted_clusters):
        cluster_to_constructs[cluster_id].append(construct_ids[i])
    
    purity = 0
    for cluster_id, labels in cluster_to_constructs.items():
        most_common = Counter(labels).most_common(1)[0][0]
        correct_assignments = sum(1 for label in labels if label == most_common)
        purity += correct_assignments
    
    purity /= len(construct_ids)
    
    logger.info("Leadership clustering metrics:")
    logger.info(f"- Adjusted Rand Index: {ari:.4f} (higher is better, max=1)")
    logger.info(f"- Normalized Mutual Information: {nmi:.4f} (higher is better, max=1)")
    logger.info(f"- Cluster Purity: {purity:.4f} (higher is better, max=1)")
    
    # Generate t-SNE visualization
    logger.info("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(embeddings)
    
    # Create plots
    create_visualizations(embedded, constructs, predicted_clusters, unique_constructs)
    
    # Save results
    save_results(leadership_df, predicted_clusters, constructs, ari, nmi, purity)
    
    return {
        "ari": ari,
        "nmi": nmi,
        "purity": purity,
        "embeddings": embeddings,
        "tsne": embedded,
        "predicted_clusters": predicted_clusters
    }

def create_visualizations(embedded, true_constructs, predicted_clusters, unique_constructs):
    """Create visualizations of the clustering."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot with true construct labels
    plt.figure(figsize=(14, 12))
    plt.title("Leadership Items by Construct (True Labels)", fontsize=16)
    
    # Use a subset of constructs for visualization clarity if there are many
    if len(unique_constructs) > 10:
        # Get the top N constructs by frequency
        top_constructs = [c for c, _ in Counter(true_constructs).most_common(10)]
        for construct in top_constructs:
            idx = [i for i, c in enumerate(true_constructs) if c == construct]
            plt.scatter(
                embedded[idx, 0], embedded[idx, 1], 
                label=construct, alpha=0.7, s=30
            )
        # Plot "Other" for remaining constructs
        other_idx = [i for i, c in enumerate(true_constructs) if c not in top_constructs]
        if other_idx:
            plt.scatter(
                embedded[other_idx, 0], embedded[other_idx, 1],
                label="Other Constructs", alpha=0.3, s=15, color="gray"
            )
    else:
        # Plot all constructs if there aren't too many
        for construct in unique_constructs:
            idx = [i for i, c in enumerate(true_constructs) if c == construct]
            plt.scatter(
                embedded[idx, 0], embedded[idx, 1],
                label=construct, alpha=0.7, s=30
            )
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "leadership_tsne_true_constructs.png", dpi=300)
    
    # Plot with predicted clusters
    plt.figure(figsize=(14, 12))
    plt.title("Leadership Items by Predicted Cluster", fontsize=16)
    
    n_clusters = len(set(predicted_clusters))
    for cluster_id in range(n_clusters):
        idx = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        plt.scatter(
            embedded[idx, 0], embedded[idx, 1],
            label=f"Cluster {cluster_id}", alpha=0.7, s=30
        )
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "leadership_tsne_predicted_clusters.png", dpi=300)
    logger.info(f"Visualizations saved to {OUTPUT_DIR}")

def save_results(df, predicted_clusters, constructs, ari, nmi, purity):
    """Save clustering results and metrics."""
    # Save comparison of clusters vs true constructs
    comparison_df = pd.DataFrame({
        "Text": df["Text"],
        "TrueConstruct": constructs,
        "PredictedCluster": predicted_clusters
    })
    comparison_df.to_csv(OUTPUT_DIR / "leadership_clustering_comparison.csv", index=False)
    
    # Analyze cluster composition
    cluster_analysis = []
    for cluster_id in sorted(set(predicted_clusters)):
        # Get items in this cluster
        cluster_items = comparison_df[comparison_df["PredictedCluster"] == cluster_id]
        
        # Count constructs in this cluster
        construct_counts = cluster_items["TrueConstruct"].value_counts()
        top_constructs = construct_counts.nlargest(3)
        
        # Calculate percentages
        total = len(cluster_items)
        percentages = top_constructs / total * 100
        
        cluster_analysis.append({
            "ClusterID": cluster_id,
            "Size": total,
            "MostCommonConstruct": top_constructs.index[0] if len(top_constructs) > 0 else "None",
            "MostCommonCount": top_constructs.iloc[0] if len(top_constructs) > 0 else 0,
            "MostCommonPct": percentages.iloc[0] if len(percentages) > 0 else 0,
            "SecondCommonConstruct": top_constructs.index[1] if len(top_constructs) > 1 else "None",
            "SecondCommonCount": top_constructs.iloc[1] if len(top_constructs) > 1 else 0,
            "SecondCommonPct": percentages.iloc[1] if len(percentages) > 1 else 0,
            "ThirdCommonConstruct": top_constructs.index[2] if len(top_constructs) > 2 else "None",
            "ThirdCommonCount": top_constructs.iloc[2] if len(top_constructs) > 2 else 0,
            "ThirdCommonPct": percentages.iloc[2] if len(percentages) > 2 else 0,
        })
    
    # Save cluster analysis
    pd.DataFrame(cluster_analysis).to_csv(OUTPUT_DIR / "cluster_composition_analysis.csv", index=False)
    
    # Save metrics to file
    metrics_path = OUTPUT_DIR / "leadership_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Leadership Construct Clustering Metrics\n")
        f.write("=====================================\n\n")
        f.write(f"Dataset: {LEADERSHIP_CSV} with {len(df)} items\n")
        f.write("\nClustering quality metrics:\n")
        f.write(f"- Adjusted Rand Index: {ari:.4f}\n")
        f.write(f"- Normalized Mutual Information: {nmi:.4f}\n")
        f.write(f"- Cluster Purity: {purity:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("- Higher values indicate better correspondence between clusters and constructs (max=1.0)\n")
        f.write("- Low values suggest leadership constructs may not be semantically distinct\n")
        f.write(f"\nGenerated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    logger.info(f"Results saved to {OUTPUT_DIR}")

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find best model from comparison metrics
    best_model_name, best_model_path, best_ari = find_best_model()
    
    # Load model
    logger.info(f"Loading best model from {best_model_path}...")
    model = SentenceTransformer(best_model_path)
    
    # Load leadership data
    leadership_df = load_leadership_data()
    
    # Cluster leadership items
    logger.info("Clustering leadership items...")
    results = cluster_leadership_items(model, leadership_df)
    
    # Report results
    logger.info("\n=== Summary ===")
    logger.info(f"Model: {best_model_name}")
    logger.info(f"IPIP ARI: {best_ari:.4f}")
    logger.info(f"Leadership ARI: {results['ari']:.4f}")
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    main() 