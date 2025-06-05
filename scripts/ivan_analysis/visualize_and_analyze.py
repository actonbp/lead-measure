#!/usr/bin/env python3
"""
Visualization and analysis script using Ivan's approach.

This script:
1. Generates t-SNE visualizations with construct labels
2. Performs similarity analysis comparing same vs different construct pairs
3. Calculates statistical metrics (t-test, Cohen's d, probability)
4. Creates visualizations with median centroids

Based on Ivan Hernandez's analysis methodology (January 2025)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import ttest_rel
from pathlib import Path
import logging
import sys
import random
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('visualize_and_analyze.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
IPIP_CSV = "data/IPIP.csv"
OUTPUT_DIR = "data/visualizations/ivan_analysis"
RANDOM_SEED = 999

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_data_and_model(
    csv_path: str,
    model_path: str
) -> Tuple[pd.DataFrame, SentenceTransformer, np.ndarray]:
    """Load dataset, model, and generate embeddings."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin-1").dropna()
    
    # Standardize column names based on dataset type
    if 'ProcessedText' in df.columns and 'StandardConstruct' in df.columns:
        # Leadership dataset
        df = df.rename(columns={'ProcessedText': 'text', 'StandardConstruct': 'label'})
        logger.info("Detected leadership dataset format")
    elif 'text' not in df.columns or 'label' not in df.columns:
        # Try to infer columns
        text_cols = [col for col in df.columns if 'text' in col.lower()]
        label_cols = [col for col in df.columns if any(name in col.lower() for name in ['label', 'construct', 'factor'])]
        
        if text_cols and label_cols:
            df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
            logger.info(f"Mapped columns: {text_cols[0]} -> text, {label_cols[0]} -> label")
        else:
            raise ValueError(f"Could not find appropriate text and label columns in {df.columns.tolist()}")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Number of unique constructs: {df['label'].nunique()}")
    
    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    logger.info("Generating embeddings...")
    texts = df['text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return df, model, embeddings

def create_tsne_visualization(
    embeddings: np.ndarray,
    labels: List[str],
    output_path: str,
    perplexity: int = 30,
    show_centroids: bool = True
):
    """Create t-SNE visualization with Ivan's approach."""
    logger.info(f"Creating t-SNE visualization (perplexity={perplexity})")
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init='random',
        random_state=RANDOM_SEED
    )
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(
        tsne_embeddings,
        columns=['TSNE Dimension 1', 'TSNE Dimension 2']
    )
    plot_df['label'] = labels
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    scatter = sns.scatterplot(
        x="TSNE Dimension 1",
        y="TSNE Dimension 2",
        hue="label",
        palette='tab20',
        data=plot_df,
        s=20,
        alpha=0.7,
        legend='full'
    )
    
    if show_centroids:
        # Compute median centroids for each label
        centroids = plot_df.groupby('label')[['TSNE Dimension 1', 'TSNE Dimension 2']].median()
        
        # Get color mapping
        unique_labels = plot_df['label'].unique()
        palette = sns.color_palette('tab20', n_colors=len(unique_labels))
        label_color_map = dict(zip(unique_labels, palette))
        
        # Annotate centroids
        for label, (x_med, y_med) in centroids.iterrows():
            plt.text(
                x_med, y_med, label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                color=label_color_map.get(label, 'black'),
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    plt.title(f't-SNE projection of embeddings {"with median centroids" if show_centroids else ""}')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.legend(title='Construct', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved t-SNE visualization to {output_path}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def perform_similarity_analysis(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_path: str
) -> Dict:
    """
    Perform Ivan's similarity analysis comparing same vs different construct pairs.
    """
    logger.info("Performing similarity analysis...")
    
    # Add embeddings to dataframe first
    df['embedding'] = [embeddings[i] for i in range(len(df))]
    
    # Filter to items that belong to only one construct
    text_counts = df.groupby('text')['label'].nunique()
    unique_texts = text_counts[text_counts == 1].index
    df_unique = df[df['text'].isin(unique_texts)].copy()
    
    # Filter to constructs with at least 2 items
    label_counts = df_unique['label'].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    df_unique = df_unique[df_unique['label'].isin(valid_labels)].copy()
    
    # Group by label for sampling
    grouped_by_label = df_unique.groupby('label')
    label_to_indices = {
        label: group.index.tolist() for label, group in grouped_by_label
    }
    
    same_label_sims = []
    diff_label_sims = []
    all_indices = df_unique.index.tolist()
    
    # Calculate similarities
    for idx, row in df_unique.iterrows():
        label = row['label']
        emb = row['embedding']
        
        # Same-label similarity
        same_label_idxs = [i for i in label_to_indices[label] if i != idx]
        if not same_label_idxs:
            continue
        
        random_same_idx = random.choice(same_label_idxs)
        same_emb = df_unique.loc[random_same_idx, 'embedding']
        sim_same = cosine_similarity(emb, same_emb)
        
        # Different-label similarity
        different_label_idxs = list(set(all_indices) - set(label_to_indices[label]))
        if not different_label_idxs:
            continue
        
        random_diff_idx = random.choice(different_label_idxs)
        diff_emb = df_unique.loc[random_diff_idx, 'embedding']
        sim_diff = cosine_similarity(emb, diff_emb)
        
        same_label_sims.append(sim_same)
        diff_label_sims.append(sim_diff)
    
    # Convert to arrays
    same = np.array(same_label_sims)
    diff = np.array(diff_label_sims)
    
    # Statistical tests
    t_stat, p_val = ttest_rel(same, diff)
    
    # Cohen's d
    differences = same - diff
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff
    
    # Probability same > diff
    prob_same_higher = np.mean(same > diff)
    
    # Create results dictionary
    results = {
        'n_comparisons': len(same),
        'same_construct_mean': np.mean(same),
        'same_construct_std': np.std(same),
        'diff_construct_mean': np.mean(diff),
        'diff_construct_std': np.std(diff),
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'probability_same_higher': prob_same_higher,
        'mean_difference': mean_diff
    }
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_path, index=False)
    
    # Print results
    logger.info("\nSimilarity Analysis Results:")
    logger.info(f"Number of comparisons: {results['n_comparisons']}")
    logger.info(f"Same construct similarity: {results['same_construct_mean']:.3f} ± {results['same_construct_std']:.3f}")
    logger.info(f"Different construct similarity: {results['diff_construct_mean']:.3f} ± {results['diff_construct_std']:.3f}")
    logger.info(f"Paired t-test: t={t_stat:.3f}, p={p_val:.3e}")
    logger.info(f"Cohen's d (paired) = {cohens_d:.3f}")
    logger.info(f"Probability(same > diff) = {prob_same_higher:.2%}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    box_data = pd.DataFrame({
        'Same Construct': same,
        'Different Construct': diff
    })
    box_data.boxplot()
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Distribution')
    
    # Histogram of differences
    plt.subplot(1, 2, 2)
    plt.hist(differences, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='No difference')
    plt.xlabel('Similarity Difference (Same - Different)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Differences\nCohen\'s d = {cohens_d:.3f}')
    plt.legend()
    
    plt.tight_layout()
    viz_path = output_path.replace('.csv', '_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved similarity visualization to {viz_path}")
    
    return results

def analyze_model(model_path: str, dataset: str = "ipip"):
    """Analyze a trained model with visualizations and similarity metrics."""
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data and model
    if dataset == "ipip":
        df, model, embeddings = load_data_and_model(IPIP_CSV, model_path)
    elif dataset == "leadership":
        leadership_csv = "data/processed/leadership_focused_clean.csv"
        df, model, embeddings = load_data_and_model(leadership_csv, model_path)
    else:
        raise NotImplementedError(f"Dataset {dataset} not yet supported")
    
    # Create t-SNE visualizations
    labels = df['label'].tolist()
    
    # Standard t-SNE
    create_tsne_visualization(
        embeddings,
        labels,
        f"{OUTPUT_DIR}/tsne_perplexity30.png",
        perplexity=30,
        show_centroids=True
    )
    
    # Alternative perplexity
    create_tsne_visualization(
        embeddings,
        labels,
        f"{OUTPUT_DIR}/tsne_perplexity15.png",
        perplexity=15,
        show_centroids=True
    )
    
    # Perform similarity analysis
    similarity_results = perform_similarity_analysis(
        df,
        embeddings,
        f"{OUTPUT_DIR}/similarity_analysis.csv"
    )
    
    return similarity_results

def compare_baseline_and_trained(
    trained_model_path: str,
    baseline_model_name: str = "BAAI/bge-m3"
):
    """Compare baseline and trained model performance."""
    logger.info("\n=== Analyzing Trained Model ===")
    trained_results = analyze_model(trained_model_path)
    
    logger.info("\n=== Analyzing Baseline Model ===")
    baseline_results = analyze_model(baseline_model_name)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': ['Probability(same > diff)', 'Cohen\'s d', 't-statistic', 'p-value'],
        'Baseline': [
            f"{baseline_results['probability_same_higher']:.2%}",
            f"{baseline_results['cohens_d']:.3f}",
            f"{baseline_results['t_statistic']:.3f}",
            f"{baseline_results['p_value']:.3e}"
        ],
        'Trained': [
            f"{trained_results['probability_same_higher']:.2%}",
            f"{trained_results['cohens_d']:.3f}",
            f"{trained_results['t_statistic']:.3f}",
            f"{trained_results['p_value']:.3e}"
        ]
    })
    
    comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    logger.info("\nModel Comparison:")
    logger.info(comparison.to_string(index=False))

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize and analyze trained embeddings")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="ipip", choices=["ipip", "leadership"])
    parser.add_argument("--compare-baseline", action="store_true", help="Compare with baseline model")
    
    args = parser.parse_args()
    
    if args.compare_baseline:
        compare_baseline_and_trained(args.model)
    else:
        analyze_model(args.model, args.dataset)

if __name__ == "__main__":
    main()