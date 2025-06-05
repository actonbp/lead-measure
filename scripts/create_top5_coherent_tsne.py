#!/usr/bin/env python3
"""
Create t-SNE visualization showing only top 5 most coherent constructs from each domain.
Uses actual embeddings and calculates within-construct coherence.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def calculate_construct_coherence(embeddings, labels):
    """Calculate within-construct coherence for each unique label."""
    unique_labels = np.unique(labels)
    coherence_scores = {}
    
    for label in unique_labels:
        # Get embeddings for this construct
        mask = labels == label
        construct_embeddings = embeddings[mask]
        
        if len(construct_embeddings) < 2:
            continue
            
        # Calculate pairwise similarities within construct
        similarities = cosine_similarity(construct_embeddings)
        
        # Get upper triangle (excluding diagonal)
        n = len(similarities)
        upper_triangle = similarities[np.triu_indices(n, k=1)]
        
        # Mean coherence score
        coherence_scores[label] = {
            'mean_similarity': np.mean(upper_triangle),
            'std_similarity': np.std(upper_triangle),
            'n_items': len(construct_embeddings),
            'n_pairs': len(upper_triangle)
        }
    
    return coherence_scores

def main():
    """Create t-SNE with top 5 most coherent constructs from each domain."""
    
    # Load data
    logger.info("Loading data...")
    ipip_df = pd.read_csv('data/processed/ipip_construct_holdout_items.csv')
    leadership_df = pd.read_csv('data/processed/leadership_focused_clean.csv')
    
    # Rename columns for consistency
    if 'ProcessedText' in leadership_df.columns:
        leadership_df = leadership_df.rename(columns={'ProcessedText': 'text', 'StandardConstruct': 'label'})
    
    # Load model
    logger.info("Loading model...")
    model_path = 'models/gist_construct_holdout_unified_final'
    if not Path(model_path).exists():
        model_path = 'models/gist_holdout_unified_final'
    
    model = SentenceTransformer(model_path)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    ipip_embeddings = model.encode(ipip_df['text'].tolist(), show_progress_bar=True)
    leadership_embeddings = model.encode(leadership_df['text'].tolist(), show_progress_bar=True)
    
    # Calculate coherence scores
    logger.info("Calculating construct coherence...")
    ipip_coherence = calculate_construct_coherence(ipip_embeddings, ipip_df['label'].values)
    leadership_coherence = calculate_construct_coherence(leadership_embeddings, leadership_df['label'].values)
    
    # Get top 5 most coherent from each domain
    ipip_sorted = sorted(ipip_coherence.items(), key=lambda x: x[1]['mean_similarity'], reverse=True)
    leadership_sorted = sorted(leadership_coherence.items(), key=lambda x: x[1]['mean_similarity'], reverse=True)
    
    top5_ipip = [label for label, _ in ipip_sorted[:5]]
    top5_leadership = [label for label, _ in leadership_sorted[:5]]
    
    logger.info("\nTop 5 most coherent IPIP constructs:")
    for label in top5_ipip:
        stats = ipip_coherence[label]
        logger.info(f"  {label}: {stats['mean_similarity']:.3f} (n={stats['n_items']})")
    
    logger.info("\nTop 5 most coherent leadership constructs:")
    for label in top5_leadership:
        stats = leadership_coherence[label]
        logger.info(f"  {label}: {stats['mean_similarity']:.3f} (n={stats['n_items']})")
    
    # Filter data to only include top 5 constructs
    ipip_mask = ipip_df['label'].isin(top5_ipip)
    leadership_mask = leadership_df['label'].isin(top5_leadership)
    
    ipip_filtered_df = ipip_df[ipip_mask].copy()
    leadership_filtered_df = leadership_df[leadership_mask].copy()
    
    ipip_filtered_embeddings = ipip_embeddings[ipip_mask]
    leadership_filtered_embeddings = leadership_embeddings[leadership_mask]
    
    # Create t-SNE embeddings
    logger.info("Creating t-SNE embeddings...")
    
    # Calculate appropriate perplexity
    ipip_perplexity = min(30, len(ipip_filtered_embeddings) // 3 - 1)
    leadership_perplexity = min(30, len(leadership_filtered_embeddings) // 3 - 1)
    
    logger.info(f"Using perplexity: IPIP={ipip_perplexity}, Leadership={leadership_perplexity}")
    
    tsne = TSNE(n_components=2, perplexity=ipip_perplexity, random_state=42)
    ipip_tsne = tsne.fit_transform(ipip_filtered_embeddings)
    
    tsne = TSNE(n_components=2, perplexity=leadership_perplexity, random_state=42)
    leadership_tsne = tsne.fit_transform(leadership_filtered_embeddings)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # IPIP plot
    palette = sns.color_palette('tab10', n_colors=5)
    label_to_color = dict(zip(top5_ipip, palette))
    
    for i, label in enumerate(top5_ipip):
        mask = ipip_filtered_df['label'] == label
        points = ipip_tsne[mask]
        ax1.scatter(points[:, 0], points[:, 1], 
                   label=label, color=palette[i], 
                   s=50, alpha=0.8, edgecolors='white', linewidth=0.8)
    
    ax1.set_title('IPIP: Top 5 Most Coherent Constructs\n(87.4% overall accuracy, Cohen\'s d = 1.116)', 
                 fontweight='bold', fontsize=13)
    ax1.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Leadership plot
    palette2 = sns.color_palette('Set2', n_colors=5)
    label_to_color = dict(zip(top5_leadership, palette2))
    
    for i, label in enumerate(top5_leadership):
        mask = leadership_filtered_df['label'] == label
        points = leadership_tsne[mask]
        ax2.scatter(points[:, 0], points[:, 1],
                   label=label, color=palette2[i],
                   s=50, alpha=0.8, edgecolors='white', linewidth=0.8)
    
    ax2.set_title('Leadership: Top 5 Most Coherent Constructs\n(62.9% overall accuracy, Cohen\'s d = 0.368)', 
                 fontweight='bold', fontsize=13)
    ax2.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path('data/visualizations/construct_holdout_validation/')
    output_path = output_dir / 'top5_coherent_constructs_tsne.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nVisualization saved to: {output_path}")
    
    # Save coherence summary
    summary_data = []
    for label in top5_ipip:
        stats = ipip_coherence[label]
        summary_data.append({
            'domain': 'IPIP',
            'construct': label,
            'mean_similarity': stats['mean_similarity'],
            'n_items': stats['n_items']
        })
    
    for label in top5_leadership:
        stats = leadership_coherence[label]
        summary_data.append({
            'domain': 'Leadership',
            'construct': label,
            'mean_similarity': stats['mean_similarity'],
            'n_items': stats['n_items']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'top5_coherent_constructs_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")
    
    plt.show()

if __name__ == '__main__':
    main()