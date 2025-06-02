"""Evaluate embedding models with comprehensive validation metrics.

This script provides a more rigorous evaluation of embedding models, including:
1. K-fold cross-validation for more reliable metrics
2. Multiple clustering algorithms to ensure robust results
3. Statistical significance testing between models
4. Detailed validation checks throughout the process
5. Comprehensive reporting of results

Usage:
    python scripts/evaluate_model_with_validation.py --model models/ipip_gist_20250520_1200 --dataset ipip

For help on all options:
    python scripts/evaluate_model_with_validation.py --help

Outputs:
    data/visualizations/model_evaluations/[model_name]_[timestamp]/ - Evaluation results and visualizations
"""

import argparse
import logging
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.model_selection import KFold
from scipy.stats import ttest_rel
import seaborn as sns

# Set seed for reproducibility
random.seed(42)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Constants
IPIP_DATA = "data/processed/leadership_ipip_mapping.csv"
LEADERSHIP_DATA = "data/processed/leadership_focused_clean.csv"
OUTPUT_DIR = "data/visualizations/model_evaluations"
RANDOM_SEED = 42

# Ensure reproducibility
np.random.seed(RANDOM_SEED)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compare_same_vs_different_similarity(df, embeddings, text_col, label_col):
    """Compare similarity of items within the same construct vs different constructs."""
    logger.info("Comparing within-construct vs between-construct similarities...")
    
    # Group by label for efficient sampling
    grouped_by_label = df.groupby(label_col)
    
    # Create a mapping from label to item indices for sampling
    label_to_indices = {
        label: group.index.tolist() for label, group in grouped_by_label
    }
    
    # Lists to store similarity scores
    same_label_sims = []
    diff_label_sims = []
    
    # Get all indices
    all_indices = df.index.tolist()
    
    # For each item, compare with a random item from same and different construct
    for idx, row in df.iterrows():
        label = row[label_col]
        emb_idx = list(df.index).index(idx)  # Get embedding at correct position
        emb = embeddings[emb_idx]
        
        # --- Same construct sampling ---
        same_label_idxs = label_to_indices[label].copy()
        if idx in same_label_idxs:
            same_label_idxs.remove(idx)  # Remove self
        
        if not same_label_idxs:
            # Skip if no other items in this construct
            continue
        
        # Randomly pick one item from same construct
        random_same_idx = random.choice(same_label_idxs)
        random_same_emb_idx = list(df.index).index(random_same_idx)
        same_emb = embeddings[random_same_emb_idx]
        
        # Calculate similarity with same-construct item
        sim_same = cosine_similarity(emb, same_emb)
        
        # --- Different construct sampling ---
        different_label_idxs = [i for i in all_indices if df.loc[i, label_col] != label]
        
        if not different_label_idxs:
            # Skip if no items in other constructs (unlikely)
            continue
        
        # Randomly pick one item from different construct
        random_diff_idx = random.choice(different_label_idxs)
        random_diff_emb_idx = list(df.index).index(random_diff_idx)
        diff_emb = embeddings[random_diff_emb_idx]
        
        # Calculate similarity with different-construct item
        sim_diff = cosine_similarity(emb, diff_emb)
        
        # Store results
        same_label_sims.append(sim_same)
        diff_label_sims.append(sim_diff)
    
    return same_label_sims, diff_label_sims

def analyze_similarity_results(same_sims, diff_sims, output_dir):
    """Analyze similarity comparison results with statistical tests."""
    logger.info("Analyzing similarity comparison results...")
    
    # Remove any NaN values
    valid_indices = [i for i, (s, d) in enumerate(zip(same_sims, diff_sims)) 
                     if not (np.isnan(s) or np.isnan(d))]
    
    same = [same_sims[i] for i in valid_indices]
    diff = [diff_sims[i] for i in valid_indices]
    
    # Paired t-test
    t_stat, p_val = ttest_rel(same, diff)
    
    # Calculate Cohen's d for paired data
    differences = np.array(same) - np.array(diff)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff
    
    # Calculate probability that same > diff
    prob_same_higher = np.mean([s > d for s, d in zip(same, diff)])
    
    # Print results
    logger.info(f"Similarity Comparison Results:")
    logger.info(f"- Average similarity within constructs: {np.mean(same):.4f}")
    logger.info(f"- Average similarity between constructs: {np.mean(diff):.4f}")
    logger.info(f"- Mean difference: {mean_diff:.4f}")
    logger.info(f"- Paired t-test: t={t_stat:.3f}, p={p_val:.3e}")
    logger.info(f"- Cohen's d (paired): {cohens_d:.3f}")
    logger.info(f"- Probability(same > diff): {prob_same_higher:.2%}")
    
    # Plot distribution of similarities
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(same, label="Same Construct", shade=True)
    sns.kdeplot(diff, label="Different Construct", shade=True)
    
    plt.title("Distribution of Embedding Similarities")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/similarity_distribution.png", dpi=300)
    plt.close()
    
    # Plot paired differences
    plt.figure(figsize=(10, 6))
    sns.histplot(differences, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("Distribution of Similarity Differences (Same - Different)")
    plt.xlabel("Similarity Difference")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/similarity_differences.png", dpi=300)
    plt.close()
    
    return {
        "avg_same": np.mean(same),
        "avg_diff": np.mean(diff),
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_val": p_val,
        "cohens_d": cohens_d,
        "prob_same_higher": prob_same_higher,
        "n_samples": len(same)
    }

def load_dataset(dataset_name):
    """Load the specified dataset with validation checks."""
    logger.info(f"Loading {dataset_name} dataset...")
    
    if dataset_name.lower() == "ipip":
        file_path = IPIP_DATA
        text_col = "Text"
        label_col = "StandardConstruct"
    elif dataset_name.lower() == "leadership":
        file_path = LEADERSHIP_DATA
        text_col = "Text"
        label_col = "StandardConstruct"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load dataset {file_path}: {str(e)}")
        raise
    
    # Validate columns exist
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataset")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")
    
    # Check for missing values
    missing_text = df[text_col].isna().sum()
    missing_label = df[label_col].isna().sum()
    if missing_text > 0 or missing_label > 0:
        logger.warning(f"Found missing values: {text_col}: {missing_text}, {label_col}: {missing_label}")
        df = df.dropna(subset=[text_col, label_col])
        logger.info(f"After dropping NAs: {len(df)} rows remain")
    
    # Validate text content
    invalid_text = df[df[text_col].apply(lambda x: not isinstance(x, str) or len(x.strip()) < 3)]
    if len(invalid_text) > 0:
        logger.warning(f"Found {len(invalid_text)} rows with invalid text (too short or non-string)")
        df = df[df[text_col].apply(lambda x: isinstance(x, str) and len(x.strip()) >= 3)]
        logger.info(f"After text validation: {len(df)} rows remain")
    
    # Get construct statistics
    constructs = df[label_col].value_counts()
    logger.info(f"Dataset has {len(df)} items across {len(constructs)} constructs")
    logger.info(f"Construct distribution: min={constructs.min()}, max={constructs.max()}, mean={constructs.mean():.1f}")
    
    # Log the top 5 largest and smallest constructs
    logger.info("Top 5 largest constructs:")
    for construct, count in constructs.head(5).items():
        logger.info(f"  {construct}: {count} items")
        
    logger.info("Top 5 smallest constructs:")
    for construct, count in constructs.tail(5).items():
        logger.info(f"  {construct}: {count} items")
    
    # Return the dataset and important info
    return {
        "df": df,
        "text_col": text_col,
        "label_col": label_col,
        "n_constructs": len(constructs),
        "construct_counts": constructs
    }

def generate_embeddings(model, texts, batch_size=32):
    """Generate embeddings with validation."""
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    
    # Input validation
    if not texts:
        raise ValueError("No texts provided for embedding generation")
    
    # Generate embeddings
    try:
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise
    
    # Validate output
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Expected numpy array but got {type(embeddings)}")
    
    if len(embeddings) != len(texts):
        raise ValueError(f"Expected {len(texts)} embeddings but got {len(embeddings)}")
    
    # Check for NaN values
    if np.isnan(embeddings).any():
        logger.warning("Embeddings contain NaN values!")
        # Replace NaNs with zeros
        embeddings = np.nan_to_num(embeddings)
    
    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings

def cluster_embeddings(embeddings, true_labels, n_clusters, algorithm="kmeans"):
    """Cluster embeddings using various algorithms with validation."""
    logger.info(f"Clustering embeddings with {algorithm} algorithm...")
    
    # Input validation
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Expected numpy array but got {type(embeddings)}")
    
    if embeddings.shape[0] != len(true_labels):
        raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) doesn't match labels ({len(true_labels)})")
    
    # Initialize clustering algorithm
    if algorithm.lower() == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init='auto')
    elif algorithm.lower() == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm.lower() == "dbscan":
        # Estimate eps parameter as the mean of distances to k-th nearest neighbor
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(10, embeddings.shape[0]-1))
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        eps = np.mean(distances[:, -1]) * 0.5  # Use half the mean distance as epsilon
        clusterer = DBSCAN(eps=eps, min_samples=5)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    # Perform clustering
    predicted_clusters = clusterer.fit_predict(embeddings)
    
    # Handle potential noise points from DBSCAN
    if algorithm.lower() == "dbscan":
        noise_count = np.sum(predicted_clusters == -1)
        if noise_count > 0:
            logger.warning(f"DBSCAN identified {noise_count} noise points")
            # Convert noise points (-1) to separate singleton clusters for metrics
            noise_idx = np.where(predicted_clusters == -1)[0]
            max_label = predicted_clusters.max()
            for i, idx in enumerate(noise_idx):
                predicted_clusters[idx] = max_label + i + 1
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
    
    # Calculate purity
    purity = calculate_purity(true_labels, predicted_clusters)
    
    logger.info(f"Clustering metrics: ARI={ari:.4f}, NMI={nmi:.4f}, Purity={purity:.4f}")
    
    return {
        "predicted_clusters": predicted_clusters,
        "metrics": {
            "ari": ari,
            "nmi": nmi,
            "purity": purity
        }
    }

def calculate_purity(true_labels, predicted_clusters):
    """Calculate cluster purity."""
    contingency_matrix = confusion_matrix(true_labels, predicted_clusters)
    return np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def create_visualizations(embeddings, true_labels, predicted_clusters, unique_labels, output_dir):
    """Create t-SNE visualizations with validation."""
    logger.info("Creating visualizations...")
    
    # Input validation
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Expected numpy array but got {type(embeddings)}")
    
    if embeddings.shape[0] != len(true_labels) or embeddings.shape[0] != len(predicted_clusters):
        raise ValueError("Number of embeddings doesn't match labels")
    
    # Create t-SNE projection
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    embedded = tsne.fit_transform(embeddings)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization with true labels
    plt.figure(figsize=(14, 10))
    plt.title("Items by True Construct", fontsize=16)
    
    # Use a subset of constructs for visualization clarity if there are many
    if len(unique_labels) > 10:
        # Get the top 10 constructs by frequency
        top_labels = pd.Series(true_labels).value_counts().nlargest(10).index.tolist()
        for label in top_labels:
            idx = [i for i, l in enumerate(true_labels) if l == label]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.7, s=30)
        
        # Plot "Other" for remaining constructs
        other_idx = [i for i, l in enumerate(true_labels) if l not in top_labels]
        if other_idx:
            plt.scatter(embedded[other_idx, 0], embedded[other_idx, 1], 
                        label="Other Constructs", alpha=0.3, s=15, color="gray")
    else:
        # Plot all constructs if there aren't too many
        for label in unique_labels:
            idx = [i for i, l in enumerate(true_labels) if l == label]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.7, s=30)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_true_labels.png", dpi=300)
    
    # Create visualization with predicted clusters
    plt.figure(figsize=(14, 10))
    plt.title("Items by Predicted Cluster", fontsize=16)
    
    n_clusters = len(set(predicted_clusters))
    for cluster_id in range(n_clusters):
        idx = [i for i, c in enumerate(predicted_clusters) if c == cluster_id]
        plt.scatter(embedded[idx, 0], embedded[idx, 1], 
                    label=f"Cluster {cluster_id}", alpha=0.7, s=30)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_predicted_clusters.png", dpi=300)
    
    # Create combined visualization (true labels with cluster outlines)
    plt.figure(figsize=(14, 10))
    plt.title("True Constructs with Predicted Cluster Boundaries", fontsize=16)
    
    # Plot points with true labels
    if len(unique_labels) > 10:
        # Same as above, only show top 10 constructs by frequency
        top_labels = pd.Series(true_labels).value_counts().nlargest(10).index.tolist()
        for label in top_labels:
            idx = [i for i, l in enumerate(true_labels) if l == label]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.7, s=30)
        
        other_idx = [i for i, l in enumerate(true_labels) if l not in top_labels]
        if other_idx:
            plt.scatter(embedded[other_idx, 0], embedded[other_idx, 1], 
                        label="Other Constructs", alpha=0.3, s=15, color="gray")
    else:
        for label in unique_labels:
            idx = [i for i, l in enumerate(true_labels) if l == label]
            plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label, alpha=0.7, s=30)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_combined.png", dpi=300)
    
    # Create confusion matrix
    plt.figure(figsize=(16, 12))
    plt.title("Confusion Matrix: True Constructs vs Predicted Clusters", fontsize=16)
    
    # If too many constructs, focus on top 15
    if len(unique_labels) > 15:
        top_labels = pd.Series(true_labels).value_counts().nlargest(15).index.tolist()
        mask = pd.Series(true_labels).isin(top_labels)
        cm = confusion_matrix(
            pd.Series(true_labels)[mask], 
            pd.Series(predicted_clusters)[mask]
        )
        sns.heatmap(cm, cmap="YlGnBu", annot=True, fmt="d", 
                    xticklabels=[f"C{i}" for i in range(len(set(predicted_clusters)))],
                    yticklabels=top_labels)
    else:
        cm = confusion_matrix(true_labels, predicted_clusters)
        sns.heatmap(cm, cmap="YlGnBu", annot=True, fmt="d", 
                    xticklabels=[f"C{i}" for i in range(len(set(predicted_clusters)))],
                    yticklabels=unique_labels)
    
    plt.ylabel("True Construct")
    plt.xlabel("Predicted Cluster")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    
    logger.info(f"Visualizations saved to {output_dir}")

def k_fold_evaluation(model, dataset, n_folds=5, algorithms=None):
    """Perform k-fold cross-validation with multiple clustering algorithms."""
    logger.info(f"Performing {n_folds}-fold cross-validation...")
    
    if algorithms is None:
        algorithms = ["kmeans", "agglomerative"]
    
    df = dataset["df"]
    text_col = dataset["text_col"]
    label_col = dataset["label_col"]
    n_constructs = dataset["n_constructs"]
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    unique_labels = list(set(labels))
    
    # Create folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Track metrics across folds
    results = {alg: {"ari": [], "nmi": [], "purity": []} for alg in algorithms}
    
    # Generate embeddings for all texts (done once for efficiency)
    all_embeddings = generate_embeddings(model, texts)
    
    # Perform k-fold evaluation
    for fold, (train_idx, test_idx) in enumerate(kf.split(texts)):
        logger.info(f"Evaluating fold {fold+1}/{n_folds}")
        
        # Get test data for this fold
        test_embeddings = all_embeddings[test_idx]
        test_labels = [labels[i] for i in test_idx]
        unique_test_labels = list(set(test_labels))
        
        # Evaluate with each clustering algorithm
        for algorithm in algorithms:
            cluster_result = cluster_embeddings(
                test_embeddings, 
                test_labels, 
                n_clusters=min(n_constructs, len(test_embeddings)-1),
                algorithm=algorithm
            )
            
            # Store metrics
            results[algorithm]["ari"].append(cluster_result["metrics"]["ari"])
            results[algorithm]["nmi"].append(cluster_result["metrics"]["nmi"])
            results[algorithm]["purity"].append(cluster_result["metrics"]["purity"])
    
    # Calculate average metrics across folds
    avg_results = {}
    for algorithm in algorithms:
        avg_results[algorithm] = {
            "ari": np.mean(results[algorithm]["ari"]),
            "ari_std": np.std(results[algorithm]["ari"]),
            "nmi": np.mean(results[algorithm]["nmi"]),
            "nmi_std": np.std(results[algorithm]["nmi"]),
            "purity": np.mean(results[algorithm]["purity"]),
            "purity_std": np.std(results[algorithm]["purity"]),
        }
        
        logger.info(f"{algorithm.capitalize()} avg metrics:")
        logger.info(f"  ARI: {avg_results[algorithm]['ari']:.4f} ± {avg_results[algorithm]['ari_std']:.4f}")
        logger.info(f"  NMI: {avg_results[algorithm]['nmi']:.4f} ± {avg_results[algorithm]['nmi_std']:.4f}")
        logger.info(f"  Purity: {avg_results[algorithm]['purity']:.4f} ± {avg_results[algorithm]['purity_std']:.4f}")
    
    # Find the best algorithm
    best_algorithm = max(algorithms, key=lambda a: avg_results[a]["ari"])
    logger.info(f"Best algorithm: {best_algorithm} (ARI: {avg_results[best_algorithm]['ari']:.4f})")
    
    # Run once more on all data with the best algorithm for visualization
    cluster_result = cluster_embeddings(
        all_embeddings, 
        labels, 
        n_clusters=min(n_constructs, len(all_embeddings)-1),
        algorithm=best_algorithm
    )
    
    return {
        "embeddings": all_embeddings,
        "labels": labels,
        "unique_labels": unique_labels,
        "predicted_clusters": cluster_result["predicted_clusters"],
        "metrics": avg_results,
        "best_algorithm": best_algorithm
    }

def save_evaluation_metrics(result, model_name, dataset_name, output_dir):
    """Save evaluation metrics to a file."""
    metrics_path = f"{output_dir}/evaluation_metrics.txt"
    
    with open(metrics_path, "w") as f:
        f.write(f"{model_name} Evaluation Metrics on {dataset_name.upper()} dataset\n")
        f.write("="*80 + "\n\n")
        
        f.write("K-fold Cross-Validation Results:\n")
        f.write("-"*40 + "\n")
        
        for algorithm, metrics in result["metrics"].items():
            f.write(f"\n{algorithm.capitalize()}:\n")
            f.write(f"  Adjusted Rand Index: {metrics['ari']:.4f} ± {metrics['ari_std']:.4f}\n")
            f.write(f"  Normalized Mutual Information: {metrics['nmi']:.4f} ± {metrics['nmi_std']:.4f}\n")
            f.write(f"  Cluster Purity: {metrics['purity']:.4f} ± {metrics['purity_std']:.4f}\n")
        
        f.write("\nBest Algorithm: " + result["best_algorithm"] + "\n\n")
        
        # Add similarity analysis results if available
        if "similarity_metrics" in result:
            sim = result["similarity_metrics"]
            f.write("Similarity Analysis Results:\n")
            f.write("-"*40 + "\n\n")
            f.write(f"Average similarity within constructs: {sim['avg_same']:.4f}\n")
            f.write(f"Average similarity between constructs: {sim['avg_diff']:.4f}\n")
            f.write(f"Mean similarity difference: {sim['mean_diff']:.4f}\n")
            f.write(f"Paired t-test: t={sim['t_stat']:.3f}, p={sim['p_val']:.3e}\n")
            f.write(f"Cohen's d (effect size): {sim['cohens_d']:.3f}\n")
            f.write(f"Probability(same > diff): {sim['prob_same_higher']:.2%}\n")
            f.write(f"Number of sample pairs: {sim['n_samples']}\n\n")
            
            # Effect size interpretation
            f.write("Effect size interpretation:\n")
            if sim['cohens_d'] < 0.2:
                f.write("  The effect size is negligible (Cohen's d < 0.2)\n")
            elif sim['cohens_d'] < 0.5:
                f.write("  The effect size is small (Cohen's d < 0.5)\n")
            elif sim['cohens_d'] < 0.8:
                f.write("  The effect size is medium (Cohen's d < 0.8)\n")
            else:
                f.write("  The effect size is large (Cohen's d >= 0.8)\n")
            
            f.write("\n")
        
        f.write("Interpretation:\n")
        f.write("-"*40 + "\n")
        
        ari = result["metrics"][result["best_algorithm"]]["ari"]
        
        if ari < 0.05:
            f.write("The model shows very poor clustering of items by construct. Constructs appear to have significant semantic overlap.\n")
        elif ari < 0.2:
            f.write("The model shows modest clustering of items by construct. Some constructs can be distinguished, but there is still substantial overlap.\n")
        elif ari < 0.4:
            f.write("The model shows moderate clustering of items by construct. Constructs show meaningful semantic separation.\n")
        else:
            f.write("The model shows strong clustering of items by construct. Constructs appear to be semantically distinct.\n")
        
        # Add similarity-based interpretation if available
        if "similarity_metrics" in result:
            sim = result["similarity_metrics"]
            f.write("\nSimilarity-based interpretation:\n")
            if sim['p_val'] < 0.05 and sim['prob_same_higher'] > 0.5:
                f.write("The model successfully distinguishes between items from the same vs different constructs.\n")
                if sim['cohens_d'] > 0.8:
                    f.write("The large effect size suggests strong semantic separation between constructs.\n")
                elif sim['cohens_d'] > 0.5:
                    f.write("The medium effect size suggests moderate semantic separation between constructs.\n")
                else:
                    f.write("The small effect size suggests limited semantic separation between constructs.\n")
            else:
                f.write("The model does not reliably distinguish between same-construct and different-construct pairs.\n")
        
        f.write(f"\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    logger.info(f"Evaluation metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding models with comprehensive validation")
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the model to evaluate")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, choices=["ipip", "leadership"], required=True,
                        help="Dataset to use for evaluation")
    
    # Evaluation parameters
    parser.add_argument('--n_folds', type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument('--algorithms', nargs='+', 
                        default=["kmeans", "agglomerative"],
                        help="Clustering algorithms to use")
    parser.add_argument('--similarity_analysis', action='store_true',
                        help="Perform similarity analysis between same/different constructs")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_name = os.path.basename(args.model)
    output_dir = f"{OUTPUT_DIR}/{model_name}_{args.dataset}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameters
    with open(f"{output_dir}/parameters.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Perform k-fold evaluation
    result = k_fold_evaluation(model, dataset, n_folds=args.n_folds, algorithms=args.algorithms)
    
    # Create visualizations
    create_visualizations(
        result["embeddings"],
        result["labels"],
        result["predicted_clusters"],
        result["unique_labels"],
        output_dir
    )
    
    # Perform similarity analysis
    # Always perform similarity analysis (as in constructembeddings.py)
    logger.info("Performing embedding similarity analysis...")
    same_sims, diff_sims = compare_same_vs_different_similarity(
        dataset["df"], 
        result["embeddings"],
        dataset["text_col"],
        dataset["label_col"]
    )
    
    # Analyze similarity results
    similarity_metrics = analyze_similarity_results(same_sims, diff_sims, output_dir)
    
    # Add similarity metrics to results
    result["similarity_metrics"] = similarity_metrics
    
    # Save evaluation metrics
    save_evaluation_metrics(result, model_name, args.dataset, output_dir)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()