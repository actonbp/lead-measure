#!/usr/bin/env python3
"""
Create simplified visualization showing most coherent constructs using pre-calculated statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_simple_coherent_visualization():
    """Create visualization based on construct statistics."""
    
    # Load leadership data to count constructs
    print("Loading leadership data...")
    leadership_df = pd.read_csv('/Users/acton/Documents/GitHub/lead-measure/data/processed/leadership_focused_clean.csv')
    
    # Count items per leadership construct
    leadership_counts = leadership_df['StandardConstruct'].value_counts()
    print("\nLeadership construct item counts:")
    for construct, count in leadership_counts.head(10).items():
        print(f"  {construct}: {count} items")
    
    # Load IPIP statistics
    print("\nLoading IPIP statistics...")
    ipip_stats = pd.read_csv('/Users/acton/Documents/GitHub/lead-measure/data/processed/ipip_construct_statistics.csv')
    
    # Get top constructs by item count (proxy for stability/coherence)
    top_ipip_by_size = ipip_stats.head(5)
    top_leadership_by_size = leadership_counts.head(5)
    
    print("\nTop 5 IPIP constructs by size:")
    for _, row in top_ipip_by_size.iterrows():
        print(f"  {row['construct']}: {row['item_count']} items")
    
    print("\nTop 5 Leadership constructs by size:")
    for construct, count in top_leadership_by_size.items():
        print(f"  {construct}: {count} items")
    
    # Create visualization comparing construct sizes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: IPIP construct sizes
    ipip_names = [name[:15] + '...' if len(name) > 15 else name for name in top_ipip_by_size['construct']]
    ax1.bar(range(len(ipip_names)), top_ipip_by_size['item_count'], color='steelblue', alpha=0.7)
    ax1.set_title('Top 5 IPIP Constructs by Size\n(Proxy for Stability)', fontweight='bold')
    ax1.set_ylabel('Number of Items')
    ax1.set_xticks(range(len(ipip_names)))
    ax1.set_xticklabels(ipip_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Top right: Leadership construct sizes  
    leadership_names = [name[:15] + '...' if len(name) > 15 else name for name in top_leadership_by_size.index]
    ax2.bar(range(len(leadership_names)), top_leadership_by_size.values, color='coral', alpha=0.7)
    ax2.set_title('Top 5 Leadership Constructs by Size\n(Proxy for Stability)', fontweight='bold')
    ax2.set_ylabel('Number of Items')
    ax2.set_xticks(range(len(leadership_names)))
    ax2.set_xticklabels(leadership_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Bottom left: Distribution of IPIP construct sizes
    ax3.hist(ipip_stats['item_count'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(ipip_stats['item_count'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ipip_stats["item_count"].mean():.1f}')
    ax3.set_title('Distribution of IPIP Construct Sizes', fontweight='bold')
    ax3.set_xlabel('Number of Items per Construct')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Bottom right: Comparison summary
    comparison_data = {
        'Domain': ['IPIP\n(246 constructs)', 'Leadership\n(11 constructs)'],
        'Max Size': [ipip_stats['item_count'].max(), leadership_counts.max()],
        'Mean Size': [ipip_stats['item_count'].mean(), leadership_counts.mean()],
        'Min Size': [ipip_stats['item_count'].min(), leadership_counts.min()]
    }
    
    x = range(len(comparison_data['Domain']))
    width = 0.25
    
    ax4.bar([i - width for i in x], comparison_data['Max Size'], width, label='Max', color='darkblue', alpha=0.8)
    ax4.bar(x, comparison_data['Mean Size'], width, label='Mean', color='steelblue', alpha=0.8)
    ax4.bar([i + width for i in x], comparison_data['Min Size'], width, label='Min', color='lightblue', alpha=0.8)
    
    ax4.set_title('Construct Size Comparison\n(Larger = More Items = Potentially More Stable)', fontweight='bold')
    ax4.set_ylabel('Number of Items')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comparison_data['Domain'])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path('/Users/acton/Documents/GitHub/lead-measure/data/visualizations/construct_holdout_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'construct_size_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Create summary table
    summary_data = []
    
    # Add top IPIP constructs
    for _, row in top_ipip_by_size.iterrows():
        summary_data.append({
            'Domain': 'IPIP',
            'Construct': row['construct'],
            'Item_Count': row['item_count'],
            'Expected_Pairs': row['expected_pairs']
        })
    
    # Add top leadership constructs
    for construct, count in top_leadership_by_size.items():
        expected_pairs = count * (count - 1) // 2  # Combination formula
        summary_data.append({
            'Domain': 'Leadership',
            'Construct': construct,
            'Item_Count': count,
            'Expected_Pairs': expected_pairs
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'largest_constructs_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    plt.show()
    
    # Print insights
    print(f"\nKey Insights:")
    print(f"  • IPIP has much more granular constructs (mean: {ipip_stats['item_count'].mean():.1f} items)")
    print(f"  • Leadership constructs are more densely populated (mean: {leadership_counts.mean():.1f} items)")
    print(f"  • Largest IPIP construct: {top_ipip_by_size.iloc[0]['construct']} ({top_ipip_by_size.iloc[0]['item_count']} items)")
    print(f"  • Largest Leadership construct: {top_leadership_by_size.index[0]} ({top_leadership_by_size.iloc[0]} items)")
    print(f"  • This size difference may contribute to different embedding quality")

if __name__ == '__main__':
    create_simple_coherent_visualization()