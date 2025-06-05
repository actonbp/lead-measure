"""Apply trained GIST model to leadership items.

This script takes a pretrained GIST model (trained on IPIP personality items) and
applies it to leadership items to evaluate whether the model can effectively
distinguish between leadership constructs.

Usage:
    python scripts/apply_gist_to_leadership.py

Outputs:
    - data/visualizations/leadership_clusters/ - Visualizations of leadership item clusters
    - data/visualizations/leadership_clusters/leadership_metrics.txt - Quantitative evaluation metrics
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
LEADERSHIP_CSV = "data/processed/leadership_focused_clean.csv"  # 434 items
GIST_MODEL_DIR = "models/gist_ipip"  # The GIST model trained on IPIP

# Output directories
OUTPUT_DIR = Path("data/visualizations/leadership_clusters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_leadership_data():
    """Load the leadership items dataset."""
    logger.info(f"Loading leadership data from {LEADERSHIP_CSV}...")
    df = pd.read_csv(LEADERSHIP_CSV)
    
    # Clean the dataframe
    df = df.dropna(subset=["Text", "StandardConstruct"])
    
    logger.info(f"Loaded {len(df)} leadership items across {df['StandardConstruct'].nunique()} constructs")
    
    # Show construct counts
    construct_counts = df["StandardConstruct"].value_counts()
    logger.info("Construct counts:")
    for construct, count in construct_counts.items():
        logger.info(f"- {construct}: {count} items")
    
    return df

def load_model():
    """Load the trained GIST model."""
    logger.info(f"Loading pretrained GIST model from {GIST_MODEL_DIR}...")
    model = SentenceTransformer(GIST_MODEL_DIR)
    return model

def evaluate_clustering(model, leadership_df):
    """Evaluate how well the model clusters leadership items by construct."""
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
    
    # Calculate clustering metrics
    ari = adjusted_rand_score(construct_ids, predicted_clusters)
    nmi = normalized_mutual_info_score(construct_ids, predicted_clusters)
    
    logger.info("Clustering metrics:")
    logger.info(f"- Adjusted Rand Index: {ari:.4f} (higher is better, max=1)")
    logger.info(f"- Normalized Mutual Information: {nmi:.4f} (higher is better, max=1)")
    
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
    logger.info(f"- Cluster Purity: {purity:.4f} (higher is better, max=1)")
    
    # Generate t-SNE visualization
    try:
        logger.info("Generating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(embeddings)
        
        # Plot with true construct labels
        plt.figure(figsize=(14, 12))
        plt.title("Leadership Items by Construct (True Labels)", fontsize=16)
        
        # Use a subset of constructs for visualization clarity if there are many
        if len(unique_constructs) > 10:
            # Get the top N constructs by frequency
            top_constructs = [c for c, _ in Counter(constructs).most_common(10)]
            for construct in top_constructs:
                idx = [i for i, c in enumerate(constructs) if c == construct]
                plt.scatter(
                    embedded[idx, 0], embedded[idx, 1], 
                    label=construct, alpha=0.7, s=30
                )
            # Plot "Other" for remaining constructs
            other_idx = [i for i, c in enumerate(constructs) if c not in top_constructs]
            if other_idx:
                plt.scatter(
                    embedded[other_idx, 0], embedded[other_idx, 1],
                    label="Other Constructs", alpha=0.3, s=15, color="gray"
                )
        else:
            # Plot all constructs if there aren't too many
            for construct in unique_constructs:
                idx = [i for i, c in enumerate(constructs) if c == construct]
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
        
        for cluster_id in range(n_clusters):
            idx = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
            plt.scatter(
                embedded[idx, 0], embedded[idx, 1],
                label=f"Cluster {cluster_id}", alpha=0.7, s=30
            )
        
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "leadership_tsne_predicted_clusters.png", dpi=300)
        
    except Exception as e:
        logger.warning(f"Error generating visualizations: {str(e)}")
    
    # Save comparison of clusters vs true constructs
    comparison_df = pd.DataFrame({
        "Text": texts,
        "TrueConstruct": constructs,
        "PredictedCluster": predicted_clusters
    })
    comparison_df.to_csv(OUTPUT_DIR / "leadership_clustering_comparison.csv", index=False)
    
    # Save metrics to file
    metrics_path = OUTPUT_DIR / "leadership_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Leadership Construct Clustering Metrics\n")
        f.write("=====================================\n\n")
        f.write(f"Dataset: {LEADERSHIP_CSV} with {len(texts)} items across {n_clusters} constructs\n")
        f.write(f"Model used: Pretrained GIST model from {GIST_MODEL_DIR}\n\n")
        f.write("Clustering quality metrics:\n")
        f.write(f"- Adjusted Rand Index: {ari:.4f}\n")
        f.write(f"- Normalized Mutual Information: {nmi:.4f}\n")
        f.write(f"- Cluster Purity: {purity:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("- Higher values indicate better correspondence between clusters and constructs (max=1.0)\n")
        f.write("- Low values suggest leadership constructs may not be semantically distinct\n")
        f.write(f"\nGenerated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    logger.info(f"Results saved to {OUTPUT_DIR}")
    return {
        "ari": ari,
        "nmi": nmi,
        "purity": purity
    }

def main():
    # Load leadership data
    leadership_df = load_leadership_data()
    
    # Load pretrained model
    model = load_model()
    
    # Evaluate clustering
    metrics = evaluate_clustering(model, leadership_df)
    
    # Show conclusion
    logger.info("\nAnalysis complete!")
    if max(metrics.values()) < 0.3:
        logger.info("The low clustering metrics suggest that leadership constructs are NOT semantically distinct.")
        logger.info("This supports the hypothesis that leadership constructs have substantial overlap.")
    else:
        logger.info("The clustering metrics suggest that leadership constructs DO have some semantic distinction.")
    
    logger.info("\nNext steps:")
    logger.info("1. Compare these results with the personality construct clustering")
    logger.info("2. Analyze the semantic overlap between specific leadership constructs")
    logger.info("3. Consider alternative clustering approaches based on linguistic properties")

if __name__ == "__main__":
    main() 