#!/usr/bin/env python3
"""
Create visualization showing only the most coherent constructs from each domain.

This script:
1. Loads the trained GIST model
2. Calculates within-construct coherence for all constructs
3. Identifies top 5 most coherent from each domain
4. Creates clean t-SNE visualization showing only these constructs
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_construct_coherence(model, df, text_column='ProcessedText', construct_column='StandardConstruct', max_items_per_construct=15):
    """Calculate within-construct coherence scores."""
    coherence_scores = {}
    
    for construct in df[construct_column].unique():
        if pd.isna(construct):
            continue
            
        construct_items = df[df[construct_column] == construct][text_column].tolist()
        
        if len(construct_items) < 3:  # Need at least 3 items for meaningful coherence
            continue
            
        # Sample items if there are too many to avoid timeout
        if len(construct_items) > max_items_per_construct:
            np.random.seed(42)  # For reproducibility
            construct_items = np.random.choice(construct_items, max_items_per_construct, replace=False).tolist()
        
        print(f"  Processing {construct}: {len(construct_items)} items")
        
        # Get embeddings for all items in this construct
        embeddings = model.encode(construct_items, show_progress_bar=False)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal) for unique pairs
        n = len(similarities)
        upper_triangle = similarities[np.triu_indices(n, k=1)]
        
        # Calculate mean coherence (average within-construct similarity)
        mean_coherence = np.mean(upper_triangle)
        
        coherence_scores[construct] = {
            'coherence': mean_coherence,
            'item_count': len(construct_items),
            'pair_count': len(upper_triangle)
        }
    
    return coherence_scores

def create_coherent_constructs_tsne():
    """Create t-SNE visualization showing most coherent constructs."""
    
    # Load data
    print("Loading data...")
    ipip_df = pd.read_csv('/Users/acton/Documents/GitHub/lead-measure/data/processed/ipip_holdout_items.csv')
    leadership_df = pd.read_csv('/Users/acton/Documents/GitHub/lead-measure/data/processed/leadership_focused_clean.csv')
    
    # Load trained model
    print("Loading trained model...")
    model_path = '/Users/acton/Documents/GitHub/lead-measure/models/gist_holdout_unified_final'
    try:
        model = SentenceTransformer(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")
        print("Using base model instead...")
        model = SentenceTransformer('BAAI/bge-m3')
    
    # Calculate coherence for IPIP constructs
    print("Calculating IPIP construct coherence...")
    ipip_coherence = calculate_construct_coherence(model, ipip_df, text_column='text', construct_column='label')
    
    # Calculate coherence for leadership constructs  
    print("Calculating leadership construct coherence...")
    leadership_coherence = calculate_construct_coherence(model, leadership_df)
    
    # Sort and get top 5 from each domain
    ipip_sorted = sorted(ipip_coherence.items(), key=lambda x: x[1]['coherence'], reverse=True)
    leadership_sorted = sorted(leadership_coherence.items(), key=lambda x: x[1]['coherence'], reverse=True)
    
    top_ipip = ipip_sorted[:5]
    top_leadership = leadership_sorted[:5]
    
    print("\nTop 5 Most Coherent IPIP Constructs:")
    for construct, stats in top_ipip:
        print(f"  {construct}: {stats['coherence']:.3f} (n={stats['item_count']})")
    
    print("\nTop 5 Most Coherent Leadership Constructs:")
    for construct, stats in top_leadership:
        print(f"  {construct}: {stats['coherence']:.3f} (n={stats['item_count']})")
    
    # Prepare data for visualization
    selected_constructs = []
    embeddings_list = []
    construct_labels = []
    domain_labels = []
    colors = []
    
    # Color schemes
    ipip_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue family
    leadership_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']  # Warmer tones
    
    # Process IPIP constructs (limit items for cleaner visualization)
    max_items_per_viz = 10  # Limit items per construct for visualization
    
    for i, (construct, stats) in enumerate(top_ipip):
        construct_items = ipip_df[ipip_df['label'] == construct]['text'].tolist()
        
        # Sample for visualization if too many items
        if len(construct_items) > max_items_per_viz:
            np.random.seed(42)
            construct_items = np.random.choice(construct_items, max_items_per_viz, replace=False).tolist()
        
        construct_embeddings = model.encode(construct_items, show_progress_bar=False)
        
        for embedding in construct_embeddings:
            embeddings_list.append(embedding)
            construct_labels.append(construct)
            domain_labels.append('IPIP')
            colors.append(ipip_colors[i])
    
    # Process leadership constructs
    for i, (construct, stats) in enumerate(top_leadership):
        construct_items = leadership_df[leadership_df['StandardConstruct'] == construct]['ProcessedText'].tolist()
        
        # Sample for visualization if too many items
        if len(construct_items) > max_items_per_viz:
            np.random.seed(42)
            construct_items = np.random.choice(construct_items, max_items_per_viz, replace=False).tolist()
        
        construct_embeddings = model.encode(construct_items, show_progress_bar=False)
        
        for embedding in construct_embeddings:
            embeddings_list.append(embedding)
            construct_labels.append(construct)
            domain_labels.append('Leadership')
            colors.append(leadership_colors[i])
    
    # Create t-SNE
    print("\nGenerating t-SNE embedding...")
    embeddings_array = np.array(embeddings_list)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)//4))
    tsne_results = tsne.fit_transform(embeddings_array)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left panel: IPIP constructs
    ipip_mask = np.array(domain_labels) == 'IPIP'
    ipip_tsne = tsne_results[ipip_mask]
    ipip_constructs = np.array(construct_labels)[ipip_mask]
    ipip_plot_colors = np.array(colors)[ipip_mask]
    
    for i, construct in enumerate([c for c, _ in top_ipip]):
        mask = ipip_constructs == construct
        ax1.scatter(ipip_tsne[mask, 0], ipip_tsne[mask, 1], 
                   c=ipip_colors[i], label=construct, alpha=0.7, s=60)
    
    ax1.set_title('Most Coherent IPIP Constructs\n(Top 5 by Within-Construct Similarity)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Leadership constructs
    leadership_mask = np.array(domain_labels) == 'Leadership'
    leadership_tsne = tsne_results[leadership_mask]
    leadership_constructs = np.array(construct_labels)[leadership_mask]
    leadership_plot_colors = np.array(colors)[leadership_mask]
    
    for i, construct in enumerate([c for c, _ in top_leadership]):
        mask = leadership_constructs == construct
        ax2.scatter(leadership_tsne[mask, 0], leadership_tsne[mask, 1], 
                   c=leadership_colors[i], label=construct, alpha=0.7, s=60)
    
    ax2.set_title('Most Coherent Leadership Constructs\n(Top 5 by Within-Construct Similarity)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path('/Users/acton/Documents/GitHub/lead-measure/data/visualizations/construct_holdout_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'most_coherent_constructs_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Create summary statistics table
    summary_data = []
    
    # Add IPIP data
    for construct, stats in top_ipip:
        summary_data.append({
            'Domain': 'IPIP',
            'Construct': construct,
            'Coherence_Score': stats['coherence'],
            'Item_Count': stats['item_count'],
            'Rank': len(summary_data) + 1
        })
    
    # Add leadership data
    for construct, stats in top_leadership:
        summary_data.append({
            'Domain': 'Leadership', 
            'Construct': construct,
            'Coherence_Score': stats['coherence'],
            'Item_Count': stats['item_count'],
            'Rank': len([x for x in summary_data if x['Domain'] == 'Leadership']) + 1
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'most_coherent_constructs_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Print comparison
    print(f"\nCoherence Comparison:")
    ipip_mean = np.mean([stats['coherence'] for _, stats in top_ipip])
    leadership_mean = np.mean([stats['coherence'] for _, stats in top_leadership])
    print(f"  Top 5 IPIP coherence: {ipip_mean:.3f}")
    print(f"  Top 5 Leadership coherence: {leadership_mean:.3f}")
    print(f"  Difference: {ipip_mean - leadership_mean:.3f}")
    
    plt.show()
    
    return top_ipip, top_leadership, summary_df

if __name__ == '__main__':
    create_coherent_constructs_tsne()