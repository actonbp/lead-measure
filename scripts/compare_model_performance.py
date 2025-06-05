"""Compare performance of different embedding models on IPIP and leadership data.

This script evaluates and compares multiple trained models on both IPIP and leadership
datasets, computing clustering metrics for each approach.

Usage:
    python scripts/compare_model_performance.py

Outputs:
    data/visualizations/model_comparison/
      - comparison_metrics.csv - CSV file with all metrics
      - ipip_comparison_plot.png - Visualizations of IPIP metric comparisons
      - leadership_comparison_plot.png - Visualizations of leadership metric comparisons
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
import argparse
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
IPIP_CSV = "data/IPIP.csv"
LEADERSHIP_CSV = "data/processed/leadership_focused_clean.csv"
OUTPUT_DIR = Path("data/visualizations/model_comparison")

# Models to compare - adjust this list to include the models you want to evaluate
MODELS = [
    # Baseline models
    {"name": "Base mpnet", "path": "sentence-transformers/all-mpnet-base-v2", "type": "baseline"},
    
    # GIST-trained models
    {"name": "GIST (original pairs)", "path": "models/gist_ipip", "type": "gist"},
    
    # MNRL-trained models
    {"name": "MNRL (comprehensive pairs)", "path": "models/ipip_mnrl", "type": "mnrl"},
    
    # Add other models as needed
]

def load_ipip_data():
    """Load and prepare IPIP data for evaluation."""
    logger.info(f"Loading IPIP data from {IPIP_CSV}...")
    try:
        df = pd.read_csv(IPIP_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed. Retrying with latin-1...")
        df = pd.read_csv(IPIP_CSV, encoding="latin-1")
    
    # Clean data
    df = df.dropna(subset=["text", "label"])
    logger.info(f"Loaded {len(df)} valid IPIP items across {df['label'].nunique()} constructs")
    
    # Randomly select a subset for evaluation (to save time if needed)
    # df = df.sample(n=min(len(df), 1000), random_state=42)
    
    # Return texts and labels
    return df["text"].tolist(), df["label"].tolist()

def load_leadership_data():
    """Load and prepare leadership data for evaluation."""
    logger.info(f"Loading leadership data from {LEADERSHIP_CSV}...")
    df = pd.read_csv(LEADERSHIP_CSV)
    
    # Clean data
    df = df.dropna(subset=["Text", "StandardConstruct"])
    logger.info(f"Loaded {len(df)} leadership items across {df['StandardConstruct'].nunique()} constructs")
    
    # Return texts and labels
    return df["Text"].tolist(), df["StandardConstruct"].tolist()

def evaluate_clustering(model, texts, true_labels, dataset_name):
    """Evaluate clustering performance of a model on texts with known labels."""
    logger.info(f"Evaluating {model.name} on {dataset_name} dataset...")
    
    # Get unique labels and create mapping
    unique_labels = sorted(set(true_labels))
    n_clusters = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    true_label_ids = [label_to_id[label] for label in true_labels]
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Perform k-means clustering
    logger.info(f"Clustering embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_label_ids, predicted_clusters)
    nmi = normalized_mutual_info_score(true_label_ids, predicted_clusters)
    
    # Calculate purity
    cluster_to_true_labels = defaultdict(list)
    for i, cluster_id in enumerate(predicted_clusters):
        cluster_to_true_labels[cluster_id].append(true_label_ids[i])
    
    purity = 0
    for cluster_id, labels in cluster_to_true_labels.items():
        most_common = Counter(labels).most_common(1)[0][0]
        correct_assignments = sum(1 for label in labels if label == most_common)
        purity += correct_assignments
    
    purity /= len(true_label_ids)
    
    logger.info(f"Metrics for {model.name} on {dataset_name}:")
    logger.info(f"- ARI: {ari:.4f}, NMI: {nmi:.4f}, Purity: {purity:.4f}")
    
    return {
        "model": model.name,
        "model_type": model.model_type,
        "dataset": dataset_name,
        "ari": ari,
        "nmi": nmi,
        "purity": purity
    }

def plot_comparison(results_df, dataset, output_path):
    """Create bar plots comparing metrics across models."""
    # Filter for specific dataset
    df = results_df[results_df["dataset"] == dataset].copy()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define the metrics to plot
    metrics = ["ari", "nmi", "purity"]
    
    # Define positions for the bars
    x = np.arange(len(df))
    width = 0.25
    
    # Create bars for each metric
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[metric], width, label=metric.upper())
    
    # Add labels and title
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Clustering Performance on {dataset} Dataset', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=45, ha='right')
    ax.legend()
    
    # Color bars by model_type
    # Colors for different model types
    color_dict = {
        "baseline": "lightgray",
        "gist": "lightblue",
        "mnrl": "lightgreen",
    }
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plot saved to {output_path}")

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ipip_texts, ipip_labels = load_ipip_data()
    leadership_texts, leadership_labels = load_leadership_data()
    
    results = []
    
    # Evaluate each model
    for model_info in MODELS:
        # Skip if model path doesn't exist
        if not os.path.exists(model_info["path"]) and not model_info["path"].startswith("sentence-transformers"):
            logger.warning(f"Model path {model_info['path']} does not exist. Skipping.")
            continue
        
        # Load the model
        try:
            logger.info(f"Loading model from {model_info['path']}...")
            model = SentenceTransformer(model_info["path"])
            model.name = model_info["name"]  # Add name attribute
            model.model_type = model_info["type"]  # Add type attribute
            
            # Evaluate on IPIP
            ipip_results = evaluate_clustering(model, ipip_texts, ipip_labels, "IPIP")
            results.append(ipip_results)
            
            # Evaluate on leadership
            leadership_results = evaluate_clustering(model, leadership_texts, leadership_labels, "Leadership")
            results.append(leadership_results)
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_info['name']}: {str(e)}")
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    csv_path = OUTPUT_DIR / "comparison_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Create plots
    plot_comparison(results_df, "IPIP", OUTPUT_DIR / "ipip_comparison_plot.png")
    plot_comparison(results_df, "Leadership", OUTPUT_DIR / "leadership_comparison_plot.png")
    
    # Print summary
    logger.info("\nSummary of results:")
    for dataset in ["IPIP", "Leadership"]:
        df_dataset = results_df[results_df["dataset"] == dataset]
        best_ari_idx = df_dataset["ari"].idxmax()
        best_model = df_dataset.loc[best_ari_idx, "model"]
        best_ari = df_dataset.loc[best_ari_idx, "ari"]
        
        logger.info(f"Best model for {dataset} dataset: {best_model} (ARI: {best_ari:.4f})")

if __name__ == "__main__":
    main() 