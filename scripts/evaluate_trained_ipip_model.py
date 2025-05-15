"""Evaluate a pre-trained GIST model on IPIP personality items.

This script:
1. Loads the IPIP dataset and prepares the test set.
2. Loads a GIST model that was previously trained (e.g., by train_gist_ipip.py).
3. Evaluates how well the model clusters test set items by personality construct.

Usage:
    python scripts/evaluate_trained_ipip_model.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import Dataset # Only Dataset is needed from datasets for test_ds
import random
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer

# Configuration
IPIP_CSV = "data/IPIP.csv" # Source of IPIP items
TRAINED_MODEL_PATH = "models/gist_ipip_snowflake_cosine_100_epochs/checkpoint-5900" # Path to the 100-epoch Snowflake model
RESULTS_DIR = Path("data/visualizations/trained_ipip_snowflake_cosine_100_epochs") # New results dir for 100-epoch run
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    
    ari = adjusted_rand_score(true_label_ids, predicted_clusters)
    nmi = normalized_mutual_info_score(true_label_ids, predicted_clusters)
    
    print(f"Clustering metrics:")
    print(f"- Adjusted Rand Index: {ari:.4f}")
    print(f"- Normalized Mutual Information: {nmi:.4f}")
    
    cluster_to_true_labels = defaultdict(list)
    for i, cluster_id in enumerate(predicted_clusters):
        cluster_to_true_labels[cluster_id].append(true_label_ids[i])
    
    purity = 0
    for cluster_id, labels_in_cluster in cluster_to_true_labels.items(): # Renamed 'labels' to 'labels_in_cluster'
        if not labels_in_cluster: continue # Should not happen with K-Means if all points assigned
        most_common_label_in_cluster = Counter(labels_in_cluster).most_common(1)[0][0] # Renamed
        correct_assignments = sum(1 for label in labels_in_cluster if label == most_common_label_in_cluster)
        purity += correct_assignments
    
    purity /= len(true_label_ids)
    print(f"- Cluster Purity: {purity:.4f}")
    
    try:
        print("Generating t-SNE visualization (skip with Ctrl+C if memory is limited)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1)) # Adjust perplexity
        embedded_tsne = tsne.fit_transform(embeddings) # Renamed 'embedded' to 'embedded_tsne'
        
        plt.figure(figsize=(12, 10))
        unique_label_subset = unique_labels[:20] if len(unique_labels) > 20 else unique_labels
        for i, label_val in enumerate(unique_label_subset): # Renamed 'label' to 'label_val'
            idx = [j for j, l_val in enumerate(true_labels) if l_val == label_val] # Renamed 'l' to 'l_val'
            plt.scatter(embedded_tsne[idx, 0], embedded_tsne[idx, 1], label=label_val, alpha=0.6, s=20)
        
        if len(unique_label_subset) <= 20:
            plt.legend(fontsize=8, markerscale=2)
        plt.title(f"t-SNE of IPIP Items (True Labels) - Model: {Path(TRAINED_MODEL_PATH).name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "trained_ipip_tsne_true_labels.png", dpi=300)
        plt.close() # Close plot to free memory

        plt.figure(figsize=(12, 10))
        # Ensure predicted_clusters is a list/array for consistent iteration
        cluster_ids_to_plot = sorted(list(set(predicted_clusters)))[:20] if n_clusters > 20 else sorted(list(set(predicted_clusters)))

        for cluster_id_val in cluster_ids_to_plot: # Renamed 'cluster_id' to 'cluster_id_val'
            idx = [i for i, c_val in enumerate(predicted_clusters) if c_val == cluster_id_val] # Renamed 'c' to 'c_val'
            plt.scatter(embedded_tsne[idx, 0], embedded_tsne[idx, 1], label=f"Cluster {cluster_id_val}", alpha=0.6, s=20)
        
        if n_clusters <= 20:
            plt.legend(fontsize=8, markerscale=2)
        plt.title(f"t-SNE of IPIP Items (Predicted Clusters) - Model: {Path(TRAINED_MODEL_PATH).name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "trained_ipip_tsne_predicted_clusters.png", dpi=300)
        plt.close() # Close plot
    except (KeyboardInterrupt, MemoryError, ValueError) as e: # Added ValueError for perplexity issues
        print(f"Skipping visualizations: {str(e)}")
    
    return {"ari": ari, "nmi": nmi, "purity": purity}

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
    metrics_file = RESULTS_DIR / "trained_ipip_evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Trained IPIP Model Evaluation Metrics (Model: {TRAINED_MODEL_PATH})\n")
        f.write("===============================================================\n")
        for key, value in metrics.items():
            f.write(f"- {key.replace('_', ' ').capitalize()}: {value:.4f}\n")
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 