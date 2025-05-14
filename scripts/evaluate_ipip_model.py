"""Evaluate GIST model on IPIP personality items.

This script:
1. Loads the IPIP dataset and splits it into train (80%) and test (20%) sets
2. Trains a GIST model on the train set
3. Evaluates how well the model clusters test set items by personality construct

Usage:
    python scripts/evaluate_ipip_model.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
import random
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Configuration
IPIP_CSV = "data/IPIP.csv"
OUTPUT_DIR = "models/ipip_evaluation"
RESULTS_DIR = Path("data/visualizations/ipip_evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
STUDENT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GUIDE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16  # Reduced to save memory
EPOCHS = 2  # Reduced for faster evaluation
LEARNING_RATE = 1e-5

# Set random seed for reproducibility
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
    
    # Ensure we have text and label columns
    df = df.dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} valid IPIP items")
    return df

def create_train_test_dataset(df, test_size=0.2):
    """Split data into train/test sets and prepare pairs for training."""
    # Shuffle and split the dataframe
    df = df.sample(frac=1.0, random_state=42)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Split data: {len(train_df)} train, {len(test_df)} test items")
    
    # Group indices by label for creating pairs
    label_to_indices = {}
    for idx, row in train_df.iterrows():
        label = row["label"]
        label_to_indices.setdefault(label, []).append(idx)
    
    # Create anchor-positive pairs for training
    pairs = []
    for idx, row in train_df.iterrows():
        label = row["label"]
        candidate_indices = [i for i in label_to_indices[label] if i != idx]
        
        if candidate_indices:
            positive_idx = random.choice(candidate_indices)
            pairs.append({
                "anchor": row["text"],
                "positive": train_df.loc[positive_idx, "text"]
            })
    
    train_ds = Dataset.from_pandas(pd.DataFrame(pairs))
    test_ds = Dataset.from_pandas(test_df)
    
    return DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

def train_model(train_dataset):
    """Train a GIST model on the training dataset."""
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    print(f"Initializing models:\n - Student: {STUDENT_MODEL_NAME}\n - Guide: {GUIDE_MODEL_NAME}")
    student = SentenceTransformer(STUDENT_MODEL_NAME)
    guide = SentenceTransformer(GUIDE_MODEL_NAME)
    
    # Setup loss function
    loss = losses.GISTEmbedLoss(
        model=student,
        guide=guide,
        temperature=0.01,
        margin_strategy="absolute",
        margin=0.0,
    )
    
    # Configure training
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch
        learning_rate=LEARNING_RATE,
        logging_steps=0.5,  # log less frequently to reduce overhead
        save_steps=1.0,  # save only at the end
        save_total_limit=1,  # Only keep one checkpoint to save disk space
        no_cuda=False,  # Set to True to force CPU if needed
        seed=42,
    )
    
    # Create and run trainer
    print("Training model...")
    trainer = SentenceTransformerTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    
    trainer.train()
    print(f"Training complete. Model saved to {OUTPUT_DIR}")
    
    return student

def evaluate_clustering(model, test_dataset):
    """Evaluate how well the model clusters items by their construct labels."""
    # Get text and true labels
    texts = test_dataset["text"]
    true_labels = test_dataset["label"]
    unique_labels = sorted(set(true_labels))
    n_clusters = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    true_label_ids = [label_to_id[label] for label in true_labels]
    
    # Get embeddings
    print(f"Generating embeddings for {len(texts)} test items...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Perform K-means clustering
    print(f"Clustering embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Calculate clustering metrics
    ari = adjusted_rand_score(true_label_ids, predicted_clusters)
    nmi = normalized_mutual_info_score(true_label_ids, predicted_clusters)
    
    print(f"Clustering metrics:")
    print(f"- Adjusted Rand Index: {ari:.4f} (higher is better, max=1)")
    print(f"- Normalized Mutual Information: {nmi:.4f} (higher is better, max=1)")
    
    # Calculate purity score (how homogeneous are the clusters)
    cluster_to_true_labels = defaultdict(list)
    for i, cluster_id in enumerate(predicted_clusters):
        cluster_to_true_labels[cluster_id].append(true_label_ids[i])
    
    purity = 0
    for cluster_id, labels in cluster_to_true_labels.items():
        most_common_label = Counter(labels).most_common(1)[0][0]
        correct_assignments = sum(1 for label in labels if label == most_common_label)
        purity += correct_assignments
    
    purity /= len(true_label_ids)
    print(f"- Cluster Purity: {purity:.4f} (higher is better, max=1)")
    
    # Generate t-SNE visualization
    try:
        print("Generating t-SNE visualization (skip with Ctrl+C if memory is limited)...")
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(embeddings)
        
        # Plot with true labels
        plt.figure(figsize=(12, 10))
        
        # Use at most 20 different colors
        unique_label_subset = unique_labels[:20] if len(unique_labels) > 20 else unique_labels
        
        for i, label in enumerate(unique_label_subset):
            idx = [j for j, l in enumerate(true_labels) if l == label]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.6, s=20)
        
        # Only show legend if we have a reasonable number of classes
        if len(unique_label_subset) <= 20:
            plt.legend(fontsize=8, markerscale=2)
        
        plt.title("t-SNE Visualization of IPIP Items (True Labels)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "ipip_tsne_true_labels.png", dpi=300)
        
        # Plot with predicted clusters
        plt.figure(figsize=(12, 10))
        for cluster_id in range(n_clusters):
            if cluster_id >= 20:  # Only show first 20 clusters in visualization
                continue
            idx = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], label=f"Cluster {cluster_id}", alpha=0.6, s=20)
        
        # Only show legend for a reasonable number of clusters
        if n_clusters <= 20:
            plt.legend(fontsize=8, markerscale=2)
        
        plt.title("t-SNE Visualization of IPIP Items (Predicted Clusters)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "ipip_tsne_predicted_clusters.png", dpi=300)
    except (KeyboardInterrupt, MemoryError) as e:
        print(f"Skipping visualizations: {str(e)}")
    
    return {
        "ari": ari,
        "nmi": nmi,
        "purity": purity
    }

def main():
    """Main function to run the evaluation."""
    # Check if output directories exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    ipip_df = load_ipip_data()
    
    # Create train/test datasets
    datasets = create_train_test_dataset(ipip_df)
    
    # Train model on training set
    model = train_model(datasets["train"])
    
    # Evaluate on test set
    metrics = evaluate_clustering(model, datasets["test"])
    
    # Save metrics to file
    with open(RESULTS_DIR / "ipip_evaluation_metrics.txt", "w") as f:
        f.write(f"IPIP Model Evaluation Metrics\n")
        f.write(f"===========================\n")
        f.write(f"Dataset: IPIP with {len(datasets['train'])} training samples, {len(datasets['test'])} test samples\n")
        f.write(f"Model: {STUDENT_MODEL_NAME} trained with GIST loss\n")
        f.write(f"Guide model: {GUIDE_MODEL_NAME}\n\n")
        f.write(f"Adjusted Rand Index: {metrics['ari']:.4f}\n")
        f.write(f"Normalized Mutual Information: {metrics['nmi']:.4f}\n")
        f.write(f"Cluster Purity: {metrics['purity']:.4f}\n")
        f.write(f"\nHigher values indicate better clustering quality (max=1.0).\n")
        f.write(f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

if __name__ == "__main__":
    main() 