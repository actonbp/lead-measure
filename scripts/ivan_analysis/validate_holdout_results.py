#!/usr/bin/env python3
"""
Validation script to compare holdout IPIP vs leadership performance.

This script:
1. Loads the model trained on 80% IPIP training data
2. Evaluates on 20% IPIP holdout items (novel personality items)
3. Evaluates on leadership items (novel leadership items)
4. Compares performance degradation fairly
5. Generates statistical analysis and visualizations

This addresses the bias issue where the original model was trained and tested
on the same IPIP items.
"""

import os
import json
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
        logging.FileHandler('validate_holdout_results.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
HOLDOUT_ITEMS_FILE = "data/processed/ipip_holdout_items.csv"
HOLDOUT_INFO_FILE = "data/processed/ipip_holdout_info.json"

# NEW: Ivan's construct-level files
CONSTRUCT_HOLDOUT_ITEMS_FILE = "data/processed/ipip_construct_holdout_items.csv"
CONSTRUCT_HOLDOUT_INFO_FILE = "data/processed/ipip_construct_holdout_info.json"

LEADERSHIP_CSV = "data/processed/leadership_focused_clean.csv"
MODEL_PATH = "models/gist_holdout_unified_final"
OUTPUT_DIR = "data/visualizations/holdout_validation"
RANDOM_SEED = 999

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_holdout_ipip() -> pd.DataFrame:
    """Load IPIP holdout items."""
    logger.info(f"Loading IPIP holdout items from {HOLDOUT_ITEMS_FILE}")
    df = pd.read_csv(HOLDOUT_ITEMS_FILE)
    logger.info(f"Loaded {len(df)} holdout items with {df['label'].nunique()} unique constructs")
    return df

def load_leadership_data() -> pd.DataFrame:
    """Load leadership data."""
    logger.info(f"Loading leadership data from {LEADERSHIP_CSV}")
    df = pd.read_csv(LEADERSHIP_CSV)
    
    # Standardize column names
    if 'ProcessedText' in df.columns:
        df = df.rename(columns={'ProcessedText': 'text', 'StandardConstruct': 'label'})
    
    logger.info(f"Loaded {len(df)} leadership items with {df['label'].nunique()} unique constructs")
    return df

def generate_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer
) -> np.ndarray:
    """Generate embeddings for dataset."""
    logger.info("Generating embeddings...")
    texts = df['text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def perform_similarity_analysis(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    dataset_name: str
) -> Dict:
    """Perform Ivan's similarity analysis comparing same vs different construct pairs."""
    logger.info(f"Performing similarity analysis for {dataset_name}...")
    
    # Add embeddings to dataframe
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
    
    # Calculate similarities using Ivan's original methodology
    logger.info("Calculating pairwise similarities...")
    for idx, row in df_unique.iterrows():
        label = row['label']
        emb = row['embedding']
        
        # Same-label similarity (1 random pair per item)
        same_label_idxs = [i for i in label_to_indices[label] if i != idx]
        if not same_label_idxs:
            continue
        
        random_same_idx = random.choice(same_label_idxs)
        same_emb = df_unique.loc[random_same_idx, 'embedding']
        sim_same = cosine_similarity(emb, same_emb)
        
        # Different-label similarity (1 random pair per item)
        different_label_idxs = list(set(all_indices) - set(label_to_indices[label]))
        if not different_label_idxs:
            continue
        
        random_diff_idx = random.choice(different_label_idxs)
        diff_emb = df_unique.loc[random_diff_idx, 'embedding']
        sim_diff = cosine_similarity(emb, diff_emb)
        
        same_label_sims.append(sim_same)
        diff_label_sims.append(sim_diff)
    
    # Convert to arrays
    same_label_sims = np.array(same_label_sims)
    diff_label_sims = np.array(diff_label_sims)
    
    # Statistical analysis using Ivan's method
    t_stat, p_value = ttest_rel(same_label_sims, diff_label_sims)
    
    # Cohen's d using differences between paired comparisons
    differences = same_label_sims - diff_label_sims
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff
    
    # Probability of correct ranking
    prob_correct = np.mean(same_label_sims > diff_label_sims)
    
    results = {
        'dataset': dataset_name,
        'n_items': len(df_unique),
        'n_constructs': df_unique['label'].nunique(),
        'same_label_mean': np.mean(same_label_sims),
        'same_label_std': np.std(same_label_sims),
        'diff_label_mean': np.mean(diff_label_sims),
        'diff_label_std': np.std(diff_label_sims),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'probability_correct': prob_correct,
        'same_label_sims': same_label_sims,
        'diff_label_sims': diff_label_sims
    }
    
    logger.info(f"{dataset_name} Results:")
    logger.info(f"  Same construct similarity: {results['same_label_mean']:.4f} ± {results['same_label_std']:.4f}")
    logger.info(f"  Different construct similarity: {results['diff_label_mean']:.4f} ± {results['diff_label_std']:.4f}")
    logger.info(f"  Cohen's d: {results['cohens_d']:.3f}")
    logger.info(f"  Probability of correct ranking: {results['probability_correct']:.2%}")
    
    return results

def create_comparison_visualization(
    ipip_results: Dict,
    leadership_results: Dict,
    output_path: str
):
    """Create comparison visualization of IPIP holdout vs Leadership performance."""
    logger.info("Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution plots
    ax1 = axes[0, 0]
    ax1.hist(ipip_results['same_label_sims'], bins=50, alpha=0.5, label='Same construct', color='blue')
    ax1.hist(ipip_results['diff_label_sims'], bins=50, alpha=0.5, label='Different construct', color='red')
    ax1.axvline(np.mean(ipip_results['same_label_sims']), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(ipip_results['diff_label_sims']), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'IPIP Holdout - Similarity Distributions\n(Cohen\'s d = {ipip_results["cohens_d"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(leadership_results['same_label_sims'], bins=50, alpha=0.5, label='Same construct', color='blue')
    ax2.hist(leadership_results['diff_label_sims'], bins=50, alpha=0.5, label='Different construct', color='red')
    ax2.axvline(np.mean(leadership_results['same_label_sims']), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(leadership_results['diff_label_sims']), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Leadership - Similarity Distributions\n(Cohen\'s d = {leadership_results["cohens_d"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax3 = axes[1, 0]
    data_to_plot = [
        ipip_results['same_label_sims'],
        ipip_results['diff_label_sims'],
        leadership_results['same_label_sims'],
        leadership_results['diff_label_sims']
    ]
    labels = [
        'IPIP\nSame',
        'IPIP\nDiff',
        'Leadership\nSame',
        'Leadership\nDiff'
    ]
    box_plot = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Similarity Distribution Comparison')
    ax3.grid(True, alpha=0.3)
    
    # 3. Performance metrics comparison
    ax4 = axes[1, 1]
    metrics = ['Probability\nCorrect', 'Cohen\'s d', 'Mean Diff\n(Same-Diff)']
    ipip_values = [
        ipip_results['probability_correct'],
        ipip_results['cohens_d'],
        ipip_results['same_label_mean'] - ipip_results['diff_label_mean']
    ]
    leadership_values = [
        leadership_results['probability_correct'],
        leadership_results['cohens_d'],
        leadership_results['same_label_mean'] - leadership_results['diff_label_mean']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, ipip_values, width, label='IPIP Holdout', color='skyblue')
    bars2 = ax4.bar(x + width/2, leadership_values, width, label='Leadership', color='lightcoral')
    
    ax4.set_ylabel('Value')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison visualization to {output_path}")

def create_tsne_comparison(
    ipip_df: pd.DataFrame,
    ipip_embeddings: np.ndarray,
    leadership_df: pd.DataFrame,
    leadership_embeddings: np.ndarray,
    output_path: str
):
    """Create t-SNE visualizations for both datasets side by side."""
    logger.info("Creating t-SNE comparison visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # IPIP Holdout t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
                init='random', random_state=RANDOM_SEED)
    ipip_tsne = tsne.fit_transform(ipip_embeddings)
    
    # Get unique labels and create color palette
    ipip_labels = ipip_df['label'].values
    unique_labels = np.unique(ipip_labels)
    n_colors = len(unique_labels)
    palette = sns.color_palette('tab20', n_colors=n_colors)
    label_to_color = dict(zip(unique_labels, palette))
    
    # Plot IPIP
    for label in unique_labels:
        mask = ipip_labels == label
        ax1.scatter(ipip_tsne[mask, 0], ipip_tsne[mask, 1], 
                   label=label, s=20, alpha=0.7, 
                   color=label_to_color[label])
    
    ax1.set_title('IPIP Holdout Items - t-SNE Visualization')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.grid(True, alpha=0.3)
    
    # Leadership t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
                init='random', random_state=RANDOM_SEED)
    leadership_tsne = tsne.fit_transform(leadership_embeddings)
    
    # Get unique labels and create color palette
    leadership_labels = leadership_df['label'].values
    unique_labels = np.unique(leadership_labels)
    n_colors = len(unique_labels)
    palette = sns.color_palette('tab10', n_colors=n_colors)
    label_to_color = dict(zip(unique_labels, palette))
    
    # Plot Leadership
    for label in unique_labels:
        mask = leadership_labels == label
        ax2.scatter(leadership_tsne[mask, 0], leadership_tsne[mask, 1],
                   label=label, s=20, alpha=0.7,
                   color=label_to_color[label])
    
    ax2.set_title('Leadership Items - t-SNE Visualization')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved t-SNE comparison to {output_path}")

def save_results_summary(
    ipip_results: Dict,
    leadership_results: Dict,
    output_path: str
):
    """Save summary of results to text file."""
    summary = f"""
Holdout Validation Results Summary
==================================

Training Information:
- Model trained on 80% of IPIP data (training split)
- Evaluated on 20% IPIP holdout (novel personality items)
- Evaluated on leadership items (novel leadership items)

IPIP Holdout Performance:
-------------------------
- Number of items: {ipip_results['n_items']}
- Number of constructs: {ipip_results['n_constructs']}
- Same construct similarity: {ipip_results['same_label_mean']:.4f} ± {ipip_results['same_label_std']:.4f}
- Different construct similarity: {ipip_results['diff_label_mean']:.4f} ± {ipip_results['diff_label_std']:.4f}
- Cohen's d: {ipip_results['cohens_d']:.3f}
- Probability of correct ranking: {ipip_results['probability_correct']:.2%}

Leadership Performance:
----------------------
- Number of items: {leadership_results['n_items']}
- Number of constructs: {leadership_results['n_constructs']}
- Same construct similarity: {leadership_results['same_label_mean']:.4f} ± {leadership_results['same_label_std']:.4f}
- Different construct similarity: {leadership_results['diff_label_mean']:.4f} ± {leadership_results['diff_label_std']:.4f}
- Cohen's d: {leadership_results['cohens_d']:.3f}
- Probability of correct ranking: {leadership_results['probability_correct']:.2%}

Performance Comparison:
----------------------
- IPIP accuracy: {ipip_results['probability_correct']:.2%}
- Leadership accuracy: {leadership_results['probability_correct']:.2%}
- Difference: {(ipip_results['probability_correct'] - leadership_results['probability_correct']) * 100:.2f} percentage points

Key Findings:
-------------
1. The model shows {'similar' if abs(ipip_results['probability_correct'] - leadership_results['probability_correct']) < 0.1 else 'different'} performance on holdout IPIP vs leadership items
2. Effect size (Cohen's d) for IPIP: {ipip_results['cohens_d']:.3f} ({'small' if ipip_results['cohens_d'] < 0.5 else 'medium' if ipip_results['cohens_d'] < 0.8 else 'large'})
3. Effect size (Cohen's d) for Leadership: {leadership_results['cohens_d']:.3f} ({'small' if leadership_results['cohens_d'] < 0.5 else 'medium' if leadership_results['cohens_d'] < 0.8 else 'large'})

Interpretation:
--------------
{"This validates that leadership constructs have significantly more semantic overlap than personality constructs, even when controlling for training bias." if leadership_results['probability_correct'] < ipip_results['probability_correct'] - 0.1 else "The performance difference is minimal, suggesting the original findings may have been influenced by training bias."}
"""
    
    with open(output_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Saved results summary to {output_path}")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate holdout results')
    parser.add_argument('--method', choices=['item_level', 'construct_level'], default='item_level',
                       help='Validation method: item_level (original) or construct_level (Ivan\'s method)')
    parser.add_argument('--model', type=str, help='Model path (overrides default)')
    args = parser.parse_args()
    
    logger.info("🚀 Starting Holdout Validation Analysis")
    logger.info(f"Method: {args.method}")
    
    # Determine model path and files based on method
    if args.method == 'construct_level':
        model_path = args.model or "models/gist_construct_holdout_unified_final"
        holdout_file = CONSTRUCT_HOLDOUT_ITEMS_FILE
        info_file = CONSTRUCT_HOLDOUT_INFO_FILE
        output_dir = "data/visualizations/construct_holdout_validation"
        logger.info("🎯 IVAN'S CONSTRUCT-LEVEL VALIDATION")
    else:
        model_path = args.model or MODEL_PATH
        holdout_file = HOLDOUT_ITEMS_FILE
        info_file = HOLDOUT_INFO_FILE
        output_dir = OUTPUT_DIR
        logger.info("📊 ITEM-LEVEL VALIDATION")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        if args.method == 'construct_level':
            logger.error("Please run training with --method construct_level first")
        else:
            logger.error("Please run train_with_holdout.py first to train the model on holdout data")
        return
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Load datasets using the appropriate files
    ipip_df = pd.read_csv(holdout_file)
    leadership_df = load_leadership_data()
    
    logger.info(f"Loaded {len(ipip_df)} holdout items with {ipip_df['label'].nunique()} unique constructs")
    logger.info(f"Loaded {len(leadership_df)} leadership items with {leadership_df['label'].nunique()} unique constructs")
    
    # Generate embeddings
    ipip_embeddings = generate_embeddings(ipip_df, model)
    leadership_embeddings = generate_embeddings(leadership_df, model)
    
    # Perform similarity analysis
    dataset_name = "IPIP Construct Holdout" if args.method == 'construct_level' else "IPIP Holdout"
    ipip_results = perform_similarity_analysis(ipip_df, ipip_embeddings, dataset_name)
    leadership_results = perform_similarity_analysis(leadership_df, leadership_embeddings, "Leadership")
    
    # Create visualizations
    create_comparison_visualization(
        ipip_results, 
        leadership_results,
        os.path.join(output_dir, "holdout_comparison_analysis.png")
    )
    
    create_tsne_comparison(
        ipip_df,
        ipip_embeddings,
        leadership_df,
        leadership_embeddings,
        os.path.join(output_dir, "holdout_tsne_comparison.png")
    )
    
    # Save results summary
    save_results_summary(
        ipip_results,
        leadership_results,
        os.path.join(output_dir, "holdout_validation_summary.txt")
    )
    
    # Save raw results as JSON (convert numpy types to Python types)
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    all_results = {
        'ipip_holdout': {k: convert_numpy_types(v) for k, v in ipip_results.items() if k not in ['same_label_sims', 'diff_label_sims']},
        'leadership': {k: convert_numpy_types(v) for k, v in leadership_results.items() if k not in ['same_label_sims', 'diff_label_sims']}
    }
    
    with open(os.path.join(output_dir, "holdout_validation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("\n✅ Holdout validation analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Print key comparison
    logger.info("\n📊 Key Comparison:")
    logger.info(f"IPIP Holdout accuracy: {ipip_results['probability_correct']:.2%}")
    logger.info(f"Leadership accuracy: {leadership_results['probability_correct']:.2%}")
    logger.info(f"Difference: {(ipip_results['probability_correct'] - leadership_results['probability_correct']) * 100:.2f} percentage points")

if __name__ == "__main__":
    main()