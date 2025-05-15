"""Evaluate a pre-trained model on IPIP personality items.

This script:
1. Loads the IPIP dataset and prepares the test set.
2. Loads a model that was previously trained (by train_ipip_mnrl.py or similar).
3. Evaluates how well the model clusters test set items by personality construct.
4. Reports comprehensive metrics and visualizations for model performance.

Usage:
    python scripts/evaluate_trained_ipip_model.py [--model_path PATH] [--output_dir DIR]

Arguments:
    --model_path: Path to the trained model directory
    --output_dir: Directory to save evaluation results
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset # Only Dataset is needed from datasets for test_ds
import random
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate a trained model on IPIP test data")
parser.add_argument("--model_path", type=str, default=None, 
                    help="Path to the trained model directory")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save evaluation results")
args = parser.parse_args()

# Find the most recent model if not specified
if args.model_path is None:
    model_dirs = list(Path("models").glob("ipip_*"))
    if model_dirs:
        # Sort by modification time (newest first)
        model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        args.model_path = str(model_dirs[0])
        print(f"Using most recent model: {args.model_path}")
    else:
        args.model_path = "sentence-transformers/all-mpnet-base-v2"
        print(f"No trained models found. Using pre-trained model: {args.model_path}")

# Create output directory
if args.output_dir is None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name = Path(args.model_path).name
    args.output_dir = f"data/visualizations/ipip_evaluation_{model_name}_{timestamp}"

RESULTS_DIR = Path(args.output_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
IPIP_CSV = "data/IPIP.csv" # Source of IPIP items
TRAINED_MODEL_PATH = args.model_path

# Set random seed for reproducibility (important for consistent test set)
random.seed(42)
np.random.seed(42)

def load_ipip_data():
    """Load IPIP data, handling potential encoding issues."""
    print(f"Loading IPIP data from {IPIP_CSV}...")
    try:
        df = pd.read_csv(IPIP_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️  UTF-8 decoding failed. Retrying with latin-1 …")
        df = pd.read_csv(IPIP_CSV, encoding="latin-1")
    
    df = df.dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} valid IPIP items")
    return df

def get_test_dataset(df, test_size=0.2):
    """Shuffle data and return the test set portion as a Hugging Face Dataset."""
    df_shuffled = df.sample(frac=1.0, random_state=42) # Ensure consistent shuffle
    split_idx = int(len(df_shuffled) * (1 - test_size))
    test_df = df_shuffled.iloc[split_idx:]
    print(f"Created test set with {len(test_df)} items")
    return Dataset.from_pandas(test_df)

def evaluate_clustering(model, test_dataset):
    """Evaluate how well the model clusters items by their construct labels."""
    texts = test_dataset["text"]
    true_labels = test_dataset["label"]
    unique_labels = sorted(list(set(true_labels))) # Ensure it's a list for consistent label_to_id mapping
    n_clusters = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    true_label_ids = [label_to_id[label] for label in true_labels]
    
    print(f"Generating embeddings for {len(texts)} test items using model from {TRAINED_MODEL_PATH}...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print(f"Clustering embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Basic clustering metrics
    ari = adjusted_rand_score(true_label_ids, predicted_clusters)
    nmi = normalized_mutual_info_score(true_label_ids, predicted_clusters)
    
    # Calculate silhouette score if we have enough data
    sil_score = 0
    if len(embeddings) > n_clusters * 2:  # Need at least 2 samples per cluster
        try:
            sil_score = silhouette_score(embeddings, predicted_clusters)
        except Exception as e:
            print(f"Could not calculate silhouette score: {e}")
    
    print(f"Clustering metrics:")
    print(f"- Adjusted Rand Index: {ari:.4f}")
    print(f"- Normalized Mutual Information: {nmi:.4f}")
    print(f"- Silhouette Score: {sil_score:.4f}")
    
    # Calculate cluster homogeneity (purity)
    cluster_to_true_labels = defaultdict(list)
    for i, cluster_id in enumerate(predicted_clusters):
        cluster_to_true_labels[cluster_id].append(true_label_ids[i])
    
    purity = 0
    cluster_homogeneity = {}
    for cluster_id, labels_in_cluster in cluster_to_true_labels.items():
        if not labels_in_cluster: continue
        label_counts = Counter(labels_in_cluster)
        most_common_label = label_counts.most_common(1)[0][0]
        correct_assignments = label_counts[most_common_label]
        cluster_homogeneity[cluster_id] = correct_assignments / len(labels_in_cluster)
        purity += correct_assignments
    
    purity /= len(true_label_ids)
    avg_homogeneity = sum(cluster_homogeneity.values()) / len(cluster_homogeneity)
    print(f"- Cluster Purity: {purity:.4f}")
    print(f"- Average Cluster Homogeneity: {avg_homogeneity:.4f}")
    
    # Calculate construct-to-cluster mapping
    construct_to_cluster = {}
    for label in unique_labels:
        label_id = label_to_id[label]
        label_indices = [i for i, l in enumerate(true_label_ids) if l == label_id]
        cluster_assignments = [predicted_clusters[i] for i in label_indices]
        most_common_cluster = Counter(cluster_assignments).most_common(1)[0][0]
        construct_to_cluster[label] = most_common_cluster
    
    # Generate confusion matrix to visualize clustering quality
    cm = confusion_matrix(true_label_ids, predicted_clusters)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    # If too many classes, limit what we show in the confusion matrix
    if n_clusters > 20:
        # Find most populated classes
        label_counts = Counter(true_label_ids)
        top_labels = [label for label, _ in label_counts.most_common(20)]
        
        # Only show top 20 in the confusion matrix
        top_indices = [idx for idx, label in enumerate(range(n_clusters)) if label in top_labels]
        cm_subset = cm_normalized[top_indices, :][:, top_indices]
        
        # Get label names for the top classes
        top_label_names = [unique_labels[i] for i in top_indices]
        
        # Create heatmap
        sns.heatmap(cm_subset, annot=False, cmap='Blues', 
                    xticklabels=top_label_names, 
                    yticklabels=top_label_names)
        plt.title(f"Confusion Matrix (Top 20 Constructs) - {Path(TRAINED_MODEL_PATH).name}")
    else:
        # Show all classes
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                    xticklabels=unique_labels, 
                    yticklabels=unique_labels)
        plt.title(f"Confusion Matrix - {Path(TRAINED_MODEL_PATH).name}")
    
    plt.ylabel('True Construct')
    plt.xlabel('Predicted Cluster')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # Calculate inter-cluster distances
    centroids = kmeans.cluster_centers_
    inter_cluster_distances = {}
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            inter_cluster_distances[(i, j)] = distance
    
    # Find the closest and furthest clusters
    if inter_cluster_distances:
        min_dist_pair = min(inter_cluster_distances.items(), key=lambda x: x[1])
        max_dist_pair = max(inter_cluster_distances.items(), key=lambda x: x[1])
        avg_dist = sum(inter_cluster_distances.values()) / len(inter_cluster_distances)
        
        print(f"Inter-cluster distances:")
        print(f"- Average: {avg_dist:.4f}")
        print(f"- Minimum: {min_dist_pair[1]:.4f} (between clusters {min_dist_pair[0][0]} and {min_dist_pair[0][1]})")
        print(f"- Maximum: {max_dist_pair[1]:.4f} (between clusters {max_dist_pair[0][0]} and {max_dist_pair[0][1]})")
    
    # Generate t-SNE visualization
    try:
        print("Generating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embedded_tsne = tsne.fit_transform(embeddings)
        
        # True labels plot
        plt.figure(figsize=(12, 10))
        # Select top constructs by frequency if too many
        if len(unique_labels) > 20:
            label_counts = Counter(true_labels)
            top_labels = [label for label, _ in label_counts.most_common(20)]
            
            # Plot top labels
            for label in top_labels:
                idx = [j for j, l in enumerate(true_labels) if l == label]
                plt.scatter(embedded_tsne[idx, 0], embedded_tsne[idx, 1], label=label, alpha=0.6, s=20)
            
            # Plot other labels in gray
            other_idx = [j for j, l in enumerate(true_labels) if l not in top_labels]
            plt.scatter(embedded_tsne[other_idx, 0], embedded_tsne[other_idx, 1], 
                      color='gray', alpha=0.3, s=10, label='Other')
        else:
            # Plot all labels
            for label in unique_labels:
                idx = [j for j, l in enumerate(true_labels) if l == label]
                plt.scatter(embedded_tsne[idx, 0], embedded_tsne[idx, 1], label=label, alpha=0.6, s=20)
        
        if len(unique_labels) <= 20:
            plt.legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"t-SNE of IPIP Items (True Labels) - Model: {Path(TRAINED_MODEL_PATH).name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tsne_true_labels.png", dpi=300)
        plt.close()

        # Predicted clusters plot
        plt.figure(figsize=(12, 10))
        for cluster_id in range(min(20, n_clusters)):
            idx = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
            plt.scatter(embedded_tsne[idx, 0], embedded_tsne[idx, 1], 
                      label=f"Cluster {cluster_id}", alpha=0.6, s=20)
        
        plt.legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"t-SNE of IPIP Items (Predicted Clusters) - Model: {Path(TRAINED_MODEL_PATH).name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tsne_predicted_clusters.png", dpi=300)
        plt.close()
        
        # Combined visualization with both true and predicted clusters
        plt.figure(figsize=(14, 12))
        
        # Create a mapping from true label to a color
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        label_to_color = {label_id: cmap(i) for i, label_id in enumerate(range(len(unique_labels)))}
        
        # Plot each point
        for i in range(len(embedded_tsne)):
            true_label = true_label_ids[i]
            color = label_to_color[true_label]
            cluster_id = predicted_clusters[i]
            
            # Mark correctly clustered points with a special marker
            correct_cluster = (construct_to_cluster.get(true_labels[i]) == cluster_id)
            
            plt.scatter(embedded_tsne[i, 0], embedded_tsne[i, 1], 
                      color=color, alpha=0.7, s=40,
                      edgecolors='black' if correct_cluster else 'red')
        
        plt.title(f"t-SNE of IPIP Items (Combined View) - Model: {Path(TRAINED_MODEL_PATH).name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tsne_combined.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    # Save the cluster composition for analysis
    cluster_composition = []
    for cluster_id in sorted(set(predicted_clusters)):
        # Get indices of items in this cluster
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        
        # Get constructs for these items
        cluster_constructs = [true_labels[i] for i in cluster_indices]
        
        # Count constructs in this cluster
        construct_counts = Counter(cluster_constructs)
        
        # Get top constructs
        top_constructs = construct_counts.most_common(3)
        
        # Calculate homogeneity
        homogeneity = cluster_homogeneity.get(cluster_id, 0)
        
        # Create analysis entry
        analysis_entry = {
            "cluster_id": cluster_id,
            "size": len(cluster_indices),
            "homogeneity": homogeneity,
            "most_common_construct": top_constructs[0][0] if top_constructs else "None",
            "most_common_count": top_constructs[0][1] if top_constructs else 0,
            "most_common_pct": (top_constructs[0][1] / len(cluster_indices) * 100) if top_constructs else 0,
            "second_common_construct": top_constructs[1][0] if len(top_constructs) > 1 else "None",
            "second_common_count": top_constructs[1][1] if len(top_constructs) > 1 else 0,
            "second_common_pct": (top_constructs[1][1] / len(cluster_indices) * 100) if len(top_constructs) > 1 else 0,
        }
        
        cluster_composition.append(analysis_entry)
    
    # Save cluster composition to CSV
    composition_df = pd.DataFrame(cluster_composition)
    composition_df.to_csv(RESULTS_DIR / "cluster_composition.csv", index=False)
    
    return {
        "ari": ari, 
        "nmi": nmi, 
        "purity": purity,
        "silhouette": sil_score,
        "avg_homogeneity": avg_homogeneity,
        "inter_cluster_distances": inter_cluster_distances
    }

def main():
    """Main function to load trained model and run evaluation."""
    # Load data and prepare test set
    ipip_df = load_ipip_data()
    test_dataset = get_test_dataset(ipip_df) # Default test_size=0.2
    
    # Load the fine-tuned model
    print(f"Loading fine-tuned model from {TRAINED_MODEL_PATH}...")
    model = SentenceTransformer(TRAINED_MODEL_PATH)
    
    # Evaluate clustering performance
    metrics = evaluate_clustering(model, test_dataset)
    
    # Save metrics to file
    metrics_file = RESULTS_DIR / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        model_name = Path(TRAINED_MODEL_PATH).name
        f.write(f"IPIP Model Evaluation Metrics\n")
        f.write("=========================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Set Size: {len(test_dataset)} items\n")
        f.write(f"Number of Constructs: {test_dataset['label'].nunique()}\n\n")
        
        f.write("Clustering Quality Metrics:\n")
        f.write(f"- Adjusted Rand Index: {metrics['ari']:.4f}\n")
        f.write(f"- Normalized Mutual Information: {metrics['nmi']:.4f}\n")
        f.write(f"- Cluster Purity: {metrics['purity']:.4f}\n")
        f.write(f"- Silhouette Score: {metrics['silhouette']:.4f}\n")
        f.write(f"- Average Cluster Homogeneity: {metrics['avg_homogeneity']:.4f}\n\n")
        
        if 'inter_cluster_distances' in metrics and metrics['inter_cluster_distances']:
            distances = list(metrics['inter_cluster_distances'].values())
            avg_dist = sum(distances) / len(distances)
            min_dist = min(distances)
            max_dist = max(distances)
            
            f.write("Cluster Distance Metrics:\n")
            f.write(f"- Average Distance Between Clusters: {avg_dist:.4f}\n")
            f.write(f"- Minimum Distance Between Clusters: {min_dist:.4f}\n")
            f.write(f"- Maximum Distance Between Clusters: {max_dist:.4f}\n\n")
        
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    print(f"Metrics saved to {metrics_file}")
    
    # Save configuration info
    if Path(TRAINED_MODEL_PATH).is_dir() and (Path(TRAINED_MODEL_PATH) / "training_config.json").exists():
        try:
            with open(Path(TRAINED_MODEL_PATH) / "training_config.json", 'r') as f:
                config = json.load(f)
            
            # Copy the training config to the results directory
            with open(RESULTS_DIR / "training_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Training configuration copied to results directory")
        except Exception as e:
            print(f"Could not load training configuration: {e}")
    
    print(f"\nEvaluation complete! Results saved to {RESULTS_DIR}")
    return metrics

if __name__ == "__main__":
    main() 