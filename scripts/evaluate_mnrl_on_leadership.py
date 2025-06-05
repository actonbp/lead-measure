"""Evaluate the trained IPIP MNRL model on leadership data.

This script:
1. Loads the MNRL model trained on IPIP data
2. Applies it to leadership construct data
3. Evaluates clustering performance and generates visualizations
4. Produces a detailed analysis of construct overlap

Usage:
    python scripts/evaluate_mnrl_on_leadership.py [--model_path MODEL_PATH]

Output:
    data/visualizations/leadership_mnrl_evaluation/
    - leadership_metrics.txt - Text summary of key metrics
    - leadership_confusion_matrix.png - Confusion matrix for leadership constructs
    - leadership_tsne_true_labels.png - t-SNE visualization with true labels
    - leadership_tsne_predicted_clusters.png - t-SNE visualization with predicted clusters
    - leadership_construct_overlap.csv - Analysis of construct overlap
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
import argparse
import datetime

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate trained MNRL model on leadership data")
parser.add_argument("--model_path", type=str, default="models/ipip_mnrl_20250515_1328", 
                    help="Path to the trained model directory")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for results (default: automatically generated)")
args = parser.parse_args()

# Configuration
LEADERSHIP_CSV = "data/processed/leadership_focused_clean.csv"
TRAINED_MODEL_PATH = args.model_path
MODEL_NAME = Path(TRAINED_MODEL_PATH).name

# Define output directory with timestamp if not provided
if args.output_dir is None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    RESULTS_DIR = Path(f"data/visualizations/leadership_mnrl_evaluation_{timestamp}")
else:
    RESULTS_DIR = Path(args.output_dir)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_leadership_data():
    """Load leadership data for evaluation."""
    print(f"Loading leadership data from {LEADERSHIP_CSV}...")
    df = pd.read_csv(LEADERSHIP_CSV)
    
    # Clean data
    df = df.dropna(subset=["Text", "StandardConstruct"])
    print(f"Loaded {len(df)} leadership items across {df['StandardConstruct'].nunique()} constructs")
    
    return df

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
        plt.title(f"Leadership Confusion Matrix (Top 20 Constructs) - {MODEL_NAME}")
    else:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f"Leadership Confusion Matrix - {MODEL_NAME}")
    
    plt.ylabel('True Construct')
    plt.xlabel('Predicted Cluster')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "leadership_confusion_matrix.png", dpi=300)
    plt.close()

def calculate_construct_overlap(df, embeddings):
    """Calculate overlap between leadership constructs using cosine similarity."""
    # Get unique constructs
    constructs = df["StandardConstruct"].unique()
    
    # Calculate centroid for each construct
    construct_centroids = {}
    for construct in constructs:
        indices = df[df["StandardConstruct"] == construct].index
        construct_centroids[construct] = np.mean(embeddings[indices], axis=0)
    
    # Calculate similarity matrix
    similarity_matrix = {}
    for c1 in constructs:
        for c2 in constructs:
            if c1 <= c2:  # Include diagonal and upper triangle
                # Calculate cosine similarity (normalize vectors first)
                v1 = construct_centroids[c1] / np.linalg.norm(construct_centroids[c1])
                v2 = construct_centroids[c2] / np.linalg.norm(construct_centroids[c2])
                similarity = np.dot(v1, v2)
                similarity_matrix[(c1, c2)] = similarity
                similarity_matrix[(c2, c1)] = similarity  # Make symmetric
    
    # Convert to dataframe for easier visualization
    similarity_df = pd.DataFrame(0, index=constructs, columns=constructs)
    for (c1, c2), sim in similarity_matrix.items():
        similarity_df.loc[c1, c2] = sim
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(similarity_df, annot=True, fmt='.2f', cmap='viridis')
    plt.title(f"Leadership Construct Similarity Matrix - {MODEL_NAME}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "leadership_construct_similarity.png", dpi=300)
    plt.close()
    
    # Save similarity matrix to CSV
    similarity_df.to_csv(RESULTS_DIR / "leadership_construct_overlap.csv")
    
    # Identify highly similar construct pairs (above threshold)
    threshold = 0.8
    similar_pairs = []
    for c1 in constructs:
        for c2 in constructs:
            if c1 < c2 and similarity_matrix[(c1, c2)] > threshold:
                similar_pairs.append({
                    "construct1": c1,
                    "construct2": c2,
                    "similarity": similarity_matrix[(c1, c2)]
                })
    
    # Sort by similarity (descending)
    similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "similarity_matrix": similarity_df,
        "similar_pairs": similar_pairs,
        "avg_similarity": np.mean([sim for (c1, c2), sim in similarity_matrix.items() if c1 < c2])
    }

def evaluate_leadership_clustering(model, leadership_df):
    """Evaluate how well the model clusters leadership items by their construct labels."""
    texts = leadership_df["Text"].tolist()
    true_labels = leadership_df["StandardConstruct"].tolist()
    unique_labels = sorted(list(set(true_labels)))
    n_clusters = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    true_label_ids = [label_to_id[label] for label in true_labels]
    
    print(f"Generating embeddings for {len(texts)} leadership items using model from {TRAINED_MODEL_PATH}...")
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
    
    # Calculate construct overlap using embeddings
    overlap_results = calculate_construct_overlap(leadership_df, embeddings)
    
    # Create t-SNE visualization
    try:
        print("Generating t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embedded = tsne.fit_transform(embeddings)
        
        # Plot with true labels
        plt.figure(figsize=(14, 12))
        for label in unique_labels:
            idx = [i for i, l in enumerate(true_labels) if l == label]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.7, s=30)
        
        plt.title(f"t-SNE of Leadership Items (True Constructs) - {MODEL_NAME}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "leadership_tsne_true_labels.png", dpi=300)
        plt.close()
        
        # Plot with predicted clusters
        plt.figure(figsize=(14, 12))
        for cluster_id in range(min(20, n_clusters)):
            idx = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], 
                        label=f"Cluster {cluster_id}", alpha=0.7, s=30)
        
        plt.title(f"t-SNE of Leadership Items (Predicted Clusters) - {MODEL_NAME}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "leadership_tsne_predicted_clusters.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    # Save cluster composition analysis
    cluster_analysis = []
    for cluster_id in sorted(set(predicted_clusters)):
        # Get indices of items in this cluster
        cluster_indices = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        
        # Get constructs for these items
        cluster_constructs = [true_labels[i] for i in cluster_indices]
        
        # Count constructs in this cluster
        construct_counts = Counter(cluster_constructs)
        
        # Get top constructs
        top_constructs = construct_counts.most_common(3)
        
        # Create analysis entry
        analysis_entry = {
            "cluster_id": cluster_id,
            "size": len(cluster_indices),
            "constructs": dict(construct_counts),
            "top_constructs": top_constructs
        }
        
        cluster_analysis.append(analysis_entry)
    
    # Convert to dataframe and save
    cluster_df = pd.DataFrame([
        {
            "ClusterID": item["cluster_id"],
            "Size": item["size"],
            "MostCommonConstruct": item["top_constructs"][0][0] if item["top_constructs"] else "None",
            "MostCommonCount": item["top_constructs"][0][1] if item["top_constructs"] else 0,
            "MostCommonPct": 100 * item["top_constructs"][0][1] / item["size"] if item["top_constructs"] else 0,
            "SecondCommonConstruct": item["top_constructs"][1][0] if len(item["top_constructs"]) > 1 else "None",
            "SecondCommonCount": item["top_constructs"][1][1] if len(item["top_constructs"]) > 1 else 0,
            "SecondCommonPct": 100 * item["top_constructs"][1][1] / item["size"] if len(item["top_constructs"]) > 1 else 0,
            "ThirdCommonConstruct": item["top_constructs"][2][0] if len(item["top_constructs"]) > 2 else "None",
            "ThirdCommonCount": item["top_constructs"][2][1] if len(item["top_constructs"]) > 2 else 0,
            "ThirdCommonPct": 100 * item["top_constructs"][2][1] / item["size"] if len(item["top_constructs"]) > 2 else 0,
        }
        for item in cluster_analysis
    ])
    
    cluster_df.to_csv(RESULTS_DIR / "leadership_cluster_composition.csv", index=False)
    
    # Return metrics and results
    return {
        "ari": ari, 
        "nmi": nmi, 
        "purity": purity,
        "cluster_distances": cluster_distances,
        "model_name": MODEL_NAME,
        "data_size": len(texts),
        "num_constructs": n_clusters,
        "overlap_results": overlap_results
    }

def main():
    """Main function to load trained model and evaluate on leadership data."""
    # Load leadership data
    leadership_df = load_leadership_data()
    
    # Load the fine-tuned model
    print(f"Loading fine-tuned model from {TRAINED_MODEL_PATH}...")
    model = SentenceTransformer(TRAINED_MODEL_PATH)
    
    # Evaluate clustering performance
    metrics = evaluate_leadership_clustering(model, leadership_df)
    
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
    metrics_file = RESULTS_DIR / "leadership_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Leadership Construct Evaluation Metrics\n")
        f.write("=====================================\n\n")
        f.write(f"Model: {metrics['model_name']}\n")
        f.write(f"Data Size: {metrics['data_size']} items\n")
        f.write(f"Number of Constructs: {metrics['num_constructs']}\n\n")
        f.write("Clustering Quality Metrics:\n")
        f.write(f"- Adjusted Rand Index: {metrics['ari']:.4f}\n")
        f.write(f"- Normalized Mutual Information: {metrics['nmi']:.4f}\n")
        f.write(f"- Cluster Purity: {metrics['purity']:.4f}\n\n")
        f.write("Cluster Distance Metrics:\n")
        f.write(f"- Average Distance Between Clusters: {metrics['avg_cluster_distance']:.4f}\n")
        f.write(f"- Minimum Distance Between Clusters: {metrics['min_cluster_distance']:.4f}\n")
        f.write(f"- Maximum Distance Between Clusters: {metrics['max_cluster_distance']:.4f}\n\n")
        f.write("Construct Overlap Analysis:\n")
        f.write(f"- Average Similarity Between Constructs: {metrics['overlap_results']['avg_similarity']:.4f}\n")
        f.write("- Highly Similar Construct Pairs (>0.8 similarity):\n")
        for pair in metrics['overlap_results']['similar_pairs']:
            f.write(f"  - {pair['construct1']} & {pair['construct2']}: {pair['similarity']:.4f}\n")
        f.write(f"\nGenerated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    print(f"Evaluation complete! Results saved to {RESULTS_DIR}")
    print(f"Metrics saved to {metrics_file}")
    
    # Return results path for use in reporting
    return RESULTS_DIR

if __name__ == "__main__":
    main()