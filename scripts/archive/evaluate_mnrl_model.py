"""Evaluate the trained IPIP MNRL model on test data.

This script:
1. Loads the IPIP dataset and prepares the test set (20% of data)
2. Loads the MNRL model that was trained with comprehensive pairs
3. Evaluates how well the model clusters test set items by personality construct
4. Generates visualizations and metrics for reporting

Usage:
    python scripts/evaluate_mnrl_model.py

Output:
    data/visualizations/mnrl_evaluation/
    - evaluation_metrics.txt - Text summary of key metrics
    - confusion_matrix.png - Confusion matrix showing clustering performance
    - tsne_true_labels.png - t-SNE visualization with true labels
    - tsne_predicted_clusters.png - t-SNE visualization with predicted clusters
    - tsne_combined.png - Combined visualization showing both true and predicted
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
import random
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
import argparse
import datetime

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate trained MNRL model on IPIP test data")
parser.add_argument("--model_path", type=str, default="models/ipip_mnrl_20250515_1328", 
                    help="Path to the trained model directory")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for results (default: automatically generated)")
args = parser.parse_args()

# Configuration
IPIP_CSV = "data/IPIP.csv"
TRAINED_MODEL_PATH = args.model_path
MODEL_NAME = Path(TRAINED_MODEL_PATH).name

# Define output directory with timestamp if not provided
if args.output_dir is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    RESULTS_DIR = Path(f"data/visualizations/mnrl_evaluation_{timestamp}")
else:
    RESULTS_DIR = Path(args.output_dir)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_ipip_data():
    """Load IPIP data, handling potential encoding issues."""
    print(f"Loading IPIP data from {IPIP_CSV}...")
    try:
        df = pd.read_csv(IPIP_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️  UTF-8 decoding failed. Retrying with latin-1 …")
        df = pd.read_csv(IPIP_CSV, encoding="latin-1")
    
    df = df.dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} valid IPIP items across {df['label'].nunique()} constructs")
    return df

def get_test_dataset(df, test_size=0.2):
    """Shuffle data and return the test set portion as a Hugging Face Dataset."""
    df_shuffled = df.sample(frac=1.0, random_state=42)  # Ensure consistent shuffle
    split_idx = int(len(df_shuffled) * (1 - test_size))
    test_df = df_shuffled.iloc[split_idx:]
    print(f"Created test set with {len(test_df)} items")
    return Dataset.from_pandas(test_df)

def calculate_cluster_distances(embeddings, predicted_clusters, unique_clusters):
    """Calculate mean distances between cluster centroids."""
    # Calculate cluster centroids
    centroids = {}
    for cluster_id in unique_clusters:
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        if cluster_indices:  # Check if the cluster has members
            centroids[cluster_id] = np.mean(embeddings[cluster_indices], axis=0)
    
    # Calculate distances between centroids
    distances = {}
    for c1 in unique_clusters:
        for c2 in unique_clusters:
            if c1 < c2 and c1 in centroids and c2 in centroids:  # Avoid duplicates
                distances[(c1, c2)] = np.linalg.norm(centroids[c1] - centroids[c2])
    
    return distances

def plot_confusion_matrix(true_label_ids, predicted_clusters, label_names, n_clusters):
    """Generate and save a confusion matrix visualization."""
    cm = confusion_matrix(true_label_ids, predicted_clusters)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create confusion matrix plot
    plt.figure(figsize=(14, 12))
    
    # If there are too many clusters, show the most populated ones
    if n_clusters > 20:
        # Find most populated true clusters
        label_counts = Counter(true_label_ids)
        top_labels = [label for label, _ in label_counts.most_common(20)]
        
        # Get indices for top labels
        top_indices = [idx for idx, label in enumerate(range(n_clusters)) if label in top_labels]
        
        # Subset confusion matrix
        cm_subset = cm_normalized[top_indices, :][:, top_indices]
        subset_label_names = [label_names[i] for i in top_indices]
        
        sns.heatmap(cm_subset, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=subset_label_names, yticklabels=subset_label_names)
        plt.title(f"Confusion Matrix (Top 20 Constructs) - {MODEL_NAME}")
    else:
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f"Confusion Matrix - {MODEL_NAME}")
    
    plt.ylabel('True Construct')
    plt.xlabel('Predicted Cluster')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

def evaluate_clustering(model, test_dataset):
    """Evaluate how well the model clusters items by their construct labels."""
    texts = test_dataset["text"]
    true_labels = test_dataset["label"]
    unique_labels = sorted(list(set(true_labels)))
    n_clusters = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    true_label_ids = [label_to_id[label] for label in true_labels]
    
    print(f"Generating embeddings for {len(texts)} test items using model from {TRAINED_MODEL_PATH}...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Clustering embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Calculate metrics
    ari = adjusted_rand_score(true_label_ids, predicted_clusters)
    nmi = normalized_mutual_info_score(true_label_ids, predicted_clusters)
    
    print(f"Clustering metrics:")
    print(f"- Adjusted Rand Index: {ari:.4f}")
    print(f"- Normalized Mutual Information: {nmi:.4f}")
    
    # Calculate purity
    cluster_to_true_labels = defaultdict(list)
    for i, cluster_id in enumerate(predicted_clusters):
        cluster_to_true_labels[cluster_id].append(true_label_ids[i])
    
    purity = 0
    for cluster_id, labels_in_cluster in cluster_to_true_labels.items():
        if not labels_in_cluster: continue
        most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
        correct_assignments = sum(1 for label in labels_in_cluster if label == most_common_label)
        purity += correct_assignments
    
    purity /= len(true_label_ids)
    print(f"- Cluster Purity: {purity:.4f}")
    
    # Calculate inter-cluster distances
    unique_clusters = sorted(list(set(predicted_clusters)))
    cluster_distances = calculate_cluster_distances(embeddings, predicted_clusters, unique_clusters)
    
    # Create confusion matrix
    plot_confusion_matrix(true_label_ids, predicted_clusters, unique_labels, n_clusters)
    
    # Create t-SNE visualization
    try:
        print("Generating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embedded = tsne.fit_transform(embeddings)
        
        # Plot with true labels
        plt.figure(figsize=(14, 12))
        # If too many unique labels, plot top 20 by frequency
        if len(unique_labels) > 20:
            label_counts = Counter(true_labels)
            top_labels = [label for label, _ in label_counts.most_common(20)]
            
            for label in top_labels:
                idx = [i for i, l in enumerate(true_labels) if l == label]
                plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.7, s=30)
                
            # Plot "other" category for remaining labels
            other_idx = [i for i, l in enumerate(true_labels) if l not in top_labels]
            if other_idx:
                plt.scatter(embedded[other_idx, 0], embedded[other_idx, 1], 
                            label="Other Constructs", alpha=0.3, s=15, color="gray")
        else:
            for label in unique_labels:
                idx = [i for i, l in enumerate(true_labels) if l == label]
                plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.7, s=30)
        
        plt.title(f"t-SNE of IPIP Test Items (True Labels) - {MODEL_NAME}")
        if len(unique_labels) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tsne_true_labels.png", dpi=300)
        plt.close()
        
        # Plot with predicted clusters
        plt.figure(figsize=(14, 12))
        for cluster_id in range(min(20, n_clusters)):
            idx = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], 
                        label=f"Cluster {cluster_id}", alpha=0.7, s=30)
        
        plt.title(f"t-SNE of IPIP Test Items (Predicted Clusters) - {MODEL_NAME}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tsne_predicted_clusters.png", dpi=300)
        plt.close()
        
        # Combined visualization with both true and predicted
        plt.figure(figsize=(18, 16))
        
        # Create a mapping from true label to a color
        unique_true_labels = sorted(list(set(true_labels)))
        cmap = plt.cm.get_cmap('tab20', len(unique_true_labels))
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_true_labels)}
        
        # Plot points with color for true label and marker style for predicted cluster
        for i in range(len(embedded)):
            true_label = true_labels[i]
            color = label_to_color[true_label]
            cluster_id = predicted_clusters[i]
            marker = 'o'  # Default marker
            
            plt.scatter(embedded[i, 0], embedded[i, 1], 
                      color=color, marker=marker, alpha=0.7, s=50,
                      edgecolors='black' if predicted_clusters[i] == true_label_ids[i] else 'red')
        
        plt.title(f"t-SNE of IPIP Test Items (Combined Visualization) - {MODEL_NAME}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tsne_combined.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    # Return metrics and results
    return {
        "ari": ari, 
        "nmi": nmi, 
        "purity": purity,
        "cluster_distances": cluster_distances,
        "model_name": MODEL_NAME,
        "test_size": len(texts),
        "num_constructs": n_clusters
    }

def main():
    """Main function to load trained model and run evaluation."""
    # Load data and prepare test set
    ipip_df = load_ipip_data()
    test_dataset = get_test_dataset(ipip_df)
    
    # Load the fine-tuned model
    print(f"Loading fine-tuned model from {TRAINED_MODEL_PATH}...")
    model = SentenceTransformer(TRAINED_MODEL_PATH)
    
    # Evaluate clustering performance
    metrics = evaluate_clustering(model, test_dataset)
    
    # Calculate avg/min/max cluster distances
    if metrics["cluster_distances"]:
        distances = list(metrics["cluster_distances"].values())
        metrics["avg_cluster_distance"] = np.mean(distances)
        metrics["min_cluster_distance"] = np.min(distances)
        metrics["max_cluster_distance"] = np.max(distances)
    else:
        metrics["avg_cluster_distance"] = 0
        metrics["min_cluster_distance"] = 0
        metrics["max_cluster_distance"] = 0
    
    # Save metrics to file
    metrics_file = RESULTS_DIR / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"IPIP MNRL Model Evaluation Metrics\n")
        f.write("===============================\n\n")
        f.write(f"Model: {metrics['model_name']}\n")
        f.write(f"Test Set Size: {metrics['test_size']} items\n")
        f.write(f"Number of Constructs: {metrics['num_constructs']}\n\n")
        f.write("Clustering Quality Metrics:\n")
        f.write(f"- Adjusted Rand Index: {metrics['ari']:.4f}\n")
        f.write(f"- Normalized Mutual Information: {metrics['nmi']:.4f}\n")
        f.write(f"- Cluster Purity: {metrics['purity']:.4f}\n\n")
        f.write("Cluster Distance Metrics:\n")
        f.write(f"- Average Distance Between Clusters: {metrics['avg_cluster_distance']:.4f}\n")
        f.write(f"- Minimum Distance Between Clusters: {metrics['min_cluster_distance']:.4f}\n")
        f.write(f"- Maximum Distance Between Clusters: {metrics['max_cluster_distance']:.4f}\n\n")
        f.write(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    print(f"Evaluation complete! Results saved to {RESULTS_DIR}")
    print(f"Metrics saved to {metrics_file}")
    
    # Return results path for use in reporting
    return RESULTS_DIR

if __name__ == "__main__":
    main()