#!/usr/bin/env python3
"""
Basic Embedding Analysis for Leadership and Personality Traits

This script analyzes both Big Five personality traits and leadership styles
using TF-IDF vectorization and UMAP visualization.

Usage:
    python basic_embedding_analysis.py

Output:
    - Visualizations of embeddings in the visualizations directory
    - Analysis reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Big Five trait mapping - categories that map to the Big Five
# This is a more comprehensive mapping based on IPIP scales
BIG_FIVE_MAPPING = {
    'Openness': [
        'Complexity', 'Imagination', 'Creativity/Originality', 'Intellect', 
        'Intellectual Openness', 'Understanding', 'Depth', 'Culture', 'Ingenuity',
        'Artistic Interests', 'Adventurousness', 'Liberalism', 'Imagination',
        'Aesthetic Appreciation', 'Introspection', 'Reflection'
    ],
    'Conscientiousness': [
        'Orderliness', 'Dutifulness', 'Achievement-striving', 'Competence', 
        'Organization', 'Efficiency', 'Industriousness/Perseverance/Persistence',
        'Purposefulness', 'Deliberateness', 'Methodicalness', 'Self-Discipline',
        'Cautiousness', 'Purposefulness', 'Perfectionism', 'Rationality'
    ],
    'Extraversion': [
        'Gregariousness', 'Assertiveness', 'Warmth', 'Talkativeness', 
        'Sociability', 'Vitality/Enthusiasm/Zest', 'Exhibitionism',
        'Leadership', 'Friendliness', 'Positive Expressivity', 'Activity Level',
        'Excitement-Seeking', 'Cheerfulness', 'Poise', 'Provocativeness', 'Self-disclosure'
    ],
    'Agreeableness': [
        'Compassion', 'Cooperation', 'Sympathy', 'Empathy', 'Nurturance',
        'Pleasantness', 'Tenderness', 'Morality', 'Docility', 'Trust',
        'Altruism', 'Compliance', 'Modesty', 'Straightforwardness',
        'Adaptability'
    ],
    'Neuroticism': [
        'Anxiety', 'Emotionality', 'Anger', 'Distrust', 'Negative-Valence',
        'Stability', 'Emotional Stability', 'Depression', 'Self-consciousness',
        'Immoderation', 'Vulnerability', 'Impulsiveness', 'Cool-headedness',
        'Tranquility', 'Imperturbability', 'Sensitivity'
    ]
}

# Positive Leadership styles to include
POSITIVE_LEADERSHIP_STYLES = [
    'Authentic', 
    'Charismatic', 
    'Consideration', 
    'Initiating Structure',
    'Empowering', 
    'Ethical', 
    'Instrumental', 
    'Servant', 
    'Transformational'
]

def load_ipip_data():
    """Load and preprocess the IPIP personality items."""
    ipip_path = DATA_DIR / "ipip.csv"
    
    if not ipip_path.exists():
        raise FileNotFoundError(f"IPIP dataset not found at {ipip_path}")
    
    print(f"Loading IPIP dataset from {ipip_path}")
    
    # Load the dataset, handle encoding issues
    try:
        ipip_df = pd.read_csv(ipip_path, encoding='utf-8')
    except UnicodeDecodeError:
        ipip_df = pd.read_csv(ipip_path, encoding='latin1')
    
    print(f"Loaded {len(ipip_df)} personality items")
    return ipip_df

def filter_big_five_traits(ipip_df):
    """Filter items to only include Big Five personality traits."""
    # First, try to map using the label column
    def map_to_big_five(label):
        for big_five, labels in BIG_FIVE_MAPPING.items():
            if label in labels:
                return big_five
        return None
    
    # Map based on known instruments and their scales (like BFAS, IPIP-NEO, BFI)
    def map_by_instrument(row):
        # If already mapped, keep it
        if pd.notna(row['big_five']):
            return row['big_five']
            
        # Try to map based on instrument and label
        instrument = row['instrument'] if 'instrument' in row else None
        label = row['label'] if 'label' in row else None
        
        # Direct mapping for IPIP-based instruments
        if instrument in ['IPIP-NEO', 'BFAS', 'BFI', 'NEO-PI-R', 'IPIP']:
            # These are the most common Big Five instruments
            if 'open' in str(label).lower() or 'intellect' in str(label).lower():
                return 'Openness'
            elif 'conscien' in str(label).lower() or 'order' in str(label).lower():
                return 'Conscientiousness'
            elif 'extra' in str(label).lower() or 'gregarious' in str(label).lower():
                return 'Extraversion'
            elif 'agree' in str(label).lower() or 'altru' in str(label).lower():
                return 'Agreeableness'
            elif 'neuro' in str(label).lower() or 'emotion' in str(label).lower():
                return 'Neuroticism'
        
        # For 16PF, map specific scales to Big Five
        if instrument == '16PF':
            if label in ['Warmth', 'Liveliness', 'Social Boldness', 'Gregariousness', 'Friendliness']:
                return 'Extraversion'
            elif label in ['Emotional Stability', 'Anxiety', 'Tension', 'Emotionality']:
                return 'Neuroticism'
            elif label in ['Rule-Consciousness', 'Perfectionism', 'Orderliness', 'Dutifulness']:
                return 'Conscientiousness'
            elif label in ['Sensitivity', 'Warmth', 'Cooperation']:
                return 'Agreeableness'
            elif label in ['Openness to Change', 'Abstractness', 'Complexity', 'Intellect']:
                return 'Openness'
        
        return None
    
    # First pass - use label mapping
    ipip_df['big_five'] = ipip_df['label'].apply(map_to_big_five)
    
    # Second pass - use instrument and label-based mapping
    ipip_df['big_five'] = ipip_df.apply(map_by_instrument, axis=1)
    
    # Filter to only include items mapped to Big Five
    big_five_df = ipip_df.dropna(subset=['big_five']).copy()  # Create explicit copy to avoid SettingWithCopyWarning
    
    # For reversed items (key=-1), add "not" to the text to preserve semantics
    if 'key' in ipip_df.columns:
        def modify_reversed_text(row):
            if row['key'] == -1 or row['key'] == '-1':
                # Add "not" or "don't" to the beginning of the text if it doesn't already have a negative
                text = str(row['text'])
                if not any(neg in text.lower() for neg in ["not ", "n't ", "never ", "nobody ", "nothing "]):
                    if text.lower().startswith(("i ", "am ", "i am ", "i'm ")):
                        return text.replace("I ", "I don't ", 1).replace("I am ", "I am not ", 1).replace("I'm ", "I'm not ", 1)
                    else:
                        return "Not " + text[0].lower() + text[1:]
            return row['text']
        
        big_five_df.loc[:, 'processed_text'] = big_five_df.apply(modify_reversed_text, axis=1)
    else:
        big_five_df.loc[:, 'processed_text'] = big_five_df['text']
    
    # Balance the dataset to have similar numbers of items per trait
    min_count = big_five_df['big_five'].value_counts().min()
    balanced_df = pd.DataFrame()
    for trait in big_five_df['big_five'].unique():
        trait_items = big_five_df[big_five_df['big_five'] == trait]
        if len(trait_items) > int(min_count * 1.5):  # If we have >50% more than the minimum
            sample_size = int(min_count * 1.5)
            balanced_df = pd.concat([balanced_df, trait_items.sample(sample_size, random_state=42)])
        else:
            balanced_df = pd.concat([balanced_df, trait_items])
    
    print(f"Filtered to {len(big_five_df)} items mapped to Big Five traits")
    print(f"Balanced to {len(balanced_df)} items")
    print(balanced_df['big_five'].value_counts())
    
    return balanced_df

def load_leadership_data(dataset_name="focused_clean"):
    """Load the specified processed leadership dataset."""
    file_path = PROCESSED_DIR / f"leadership_{dataset_name}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Leadership dataset not found at {file_path}")
    
    print(f"Loading leadership dataset from {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} leadership items")
    return df

def filter_positive_leadership_styles(df, construct_column="StandardConstruct"):
    """Filter for only positive leadership styles."""
    # Filter to only positive styles
    positive_df = df[df[construct_column].isin(POSITIVE_LEADERSHIP_STYLES)].copy()
    
    print(f"Filtered to {len(positive_df)} items from positive leadership styles")
    print(positive_df[construct_column].value_counts())
    
    return positive_df

def generate_tfidf_embeddings(df, text_column):
    """Generate TF-IDF embeddings for the given texts."""
    print("Generating TF-IDF embeddings...")
    
    # Create vectorizer with improved parameters for personality/leadership language
    vectorizer = TfidfVectorizer(
        max_features=500,  # Increased feature count
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.9,  # Ignore terms that appear in more than 90% of documents
        use_idf=True,
        sublinear_tf=True  # Apply sublinear tf scaling (1+log(tf))
    )
    
    # Generate embeddings
    if text_column == 'processed_text' and 'processed_text' in df.columns:
        texts = df['processed_text'].fillna("").tolist()
    else:
        texts = df[text_column].fillna("").tolist()
    
    embeddings = vectorizer.fit_transform(texts).toarray()
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Get top features for each category (for interpretation)
    if 'big_five' in df.columns or 'StandardConstruct' in df.columns:
        category_column = 'big_five' if 'big_five' in df.columns else 'StandardConstruct'
        feature_names = vectorizer.get_feature_names_out()
        
        # Print top 10 features for each category
        print("\nTop features by category:")
        
        # Create new indices that match the rows in the embeddings matrix
        df_reset = df.reset_index(drop=True)
        
        for category in df_reset[category_column].unique():
            # Get indices of documents in this category (using reset indices)
            indices = df_reset[df_reset[category_column] == category].index.tolist()
            
            if len(indices) > 0:
                # Calculate average TF-IDF score for each term in this category
                tfidf_means = embeddings[indices].mean(axis=0)
                
                # Get indices of top terms
                top_indices = tfidf_means.argsort()[-10:][::-1]
                
                # Get top terms and their scores
                top_terms = [(feature_names[i], tfidf_means[i]) for i in top_indices]
                
                print(f"{category}: {', '.join([term for term, score in top_terms])}")
    
    return embeddings, vectorizer

def analyze_distances(embeddings, df, label_column):
    """Analyze intra-category and inter-category distances."""
    print(f"Analyzing semantic distances between {label_column} categories")
    
    # Calculate all pairwise distances
    similarities = cosine_similarity(embeddings)
    distances = 1 - similarities
    
    # Get category labels
    categories = df[label_column].values
    
    # Initialize containers for distances
    intra_distances = []
    inter_distances = []
    
    # Collect distances
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):  # Upper triangle only
            if categories[i] == categories[j]:
                intra_distances.append(distances[i, j])
            else:
                inter_distances.append(distances[i, j])
    
    # Convert to numpy arrays
    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)
    
    # Calculate statistics
    intra_mean = np.mean(intra_distances)
    intra_std = np.std(intra_distances)
    inter_mean = np.mean(inter_distances)
    inter_std = np.std(inter_distances)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
    effect_size = (inter_mean - intra_mean) / pooled_std
    
    print(f"Intra-category distances: mean = {intra_mean:.4f}, std = {intra_std:.4f}")
    print(f"Inter-category distances: mean = {inter_mean:.4f}, std = {inter_std:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot histograms of distances
    sns.histplot(intra_distances, alpha=0.5, label=f"Within category: μ={intra_mean:.4f}", kde=True)
    sns.histplot(inter_distances, alpha=0.5, label=f"Between categories: μ={inter_mean:.4f}", kde=True)
    
    plt.axvline(intra_mean, color='blue', linestyle='--')
    plt.axvline(inter_mean, color='orange', linestyle='--')
    
    plt.title(f"{label_column} Semantic Distance Distribution (Effect Size d={effect_size:.4f})", fontsize=14)
    plt.xlabel("Semantic Distance (1 - Cosine Similarity)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / f"{label_column.lower()}_distance_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved distance distribution to {output_file}")
    
    return {
        "intra_mean": intra_mean,
        "intra_std": intra_std,
        "inter_mean": inter_mean,
        "inter_std": inter_std,
        "effect_size": effect_size
    }

def visualize_embeddings_umap(embeddings, df, label_column, name):
    """Create UMAP visualization of embeddings colored by category."""
    print(f"Creating UMAP visualization for {name}")
    
    # Apply UMAP dimensionality reduction with optimized parameters for clustering
    try:
        # Try with densmap if supported
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=30,  # Higher value emphasizes global structure
            min_dist=0.05,   # Lower value for tighter clusters
            random_state=42,
            densmap=True,    # Use densMAP extension which preserves density information
            spread=0.8       # Lower makes the clusters more compact
        )
    except TypeError:
        # Fallback if densmap is not supported
        print("densmap option not supported in this UMAP version, using standard UMAP")
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=30,  # Higher value emphasizes global structure
            min_dist=0.05,   # Lower value for tighter clusters
            random_state=42,
            spread=0.8       # Lower makes the clusters more compact
        )
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    if 'processed_text' in df.columns:
        text_column = 'processed_text'
    elif 'text' in df.columns:
        text_column = 'text'
    elif 'ProcessedText' in df.columns:
        text_column = 'ProcessedText'
    else:
        text_column = df.columns[3]  # Fallback
    
    plot_df = pd.DataFrame({
        "x": embedding_2d[:, 0],
        "y": embedding_2d[:, 1],
        "category": df[label_column],
        "text": df[text_column].astype(str)
    })
    
    # Plot with distinct colors and improved aesthetics
    plt.figure(figsize=(16, 12))
    
    # Get unique categories and set color palette
    categories = plot_df["category"].unique()
    palette = sns.color_palette("tab10", len(categories))  # More distinct colors
    
    # Add background for better cluster visibility
    plt.grid(False)
    plt.gca().set_facecolor('#f8f8f8')
    
    # Create scatter plot for each category
    for i, category in enumerate(sorted(categories)):
        subset = plot_df[plot_df["category"] == category]
        plt.scatter(
            subset["x"], 
            subset["y"], 
            c=[palette[i]] * len(subset),
            label=category,
            alpha=0.8,
            s=70,
            edgecolors="white",
            linewidth=0.6
        )
    
    # Calculate and plot centroids for each category
    centroids = []
    for i, category in enumerate(sorted(categories)):
        subset = plot_df[plot_df["category"] == category]
        centroid_x = subset["x"].mean()
        centroid_y = subset["y"].mean()
        centroids.append((centroid_x, centroid_y, category))
        
        # Highlight centroid with a star marker
        plt.scatter(
            centroid_x, centroid_y, 
            marker='*', 
            s=300, 
            c=[palette[i]], 
            edgecolors='black', 
            linewidth=1.5,
            zorder=100  # Ensure it's on top
        )
        
        # Add category label at centroid
        plt.annotate(
            category,
            (centroid_x, centroid_y),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='bottom',
            xytext=(0, 10),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.title(f"{name} UMAP Projection", fontsize=18, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)
    
    # Create a more compact legend
    legend = plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc="upper left",
        fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    
    # Add a descriptive textbox
    intra_cluster_avg = 0
    for i, category in enumerate(sorted(categories)):
        subset = plot_df[plot_df["category"] == category]
        if len(subset) >= 2:
            # Calculate average distance from points to centroid
            centroid = (subset["x"].mean(), subset["y"].mean())
            distances = np.sqrt((subset["x"] - centroid[0])**2 + (subset["y"] - centroid[1])**2)
            intra_cluster_avg += distances.mean()
    
    if len(categories) > 0:
        intra_cluster_avg /= len(categories)
        
    plt.figtext(
        0.5, 0.01,
        f"UMAP projection of {name} using TF-IDF vectorization\n"
        f"Average intra-cluster spread: {intra_cluster_avg:.2f}",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / f"{name.lower().replace(' ', '_')}_umap_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP visualization to {output_file}")
    
    return plot_df, embedding_2d

def perform_cluster_analysis(embeddings, df, embedding_2d, label_column, name):
    """Perform HDBSCAN clustering and compare to actual category labels."""
    print(f"Performing HDBSCAN clustering for {name}")
    
    # Determine expected number of clusters
    expected_clusters = df[label_column].nunique()
    
    # Apply HDBSCAN clustering with parameters optimized for the expected number of clusters
    # For Big Five, we expect 5 clusters; for leadership styles, we expect around 7-9
    if expected_clusters <= 5:
        # Parameters for smaller number of broader clusters (Big Five)
        min_cluster_size = max(10, int(len(df) / (expected_clusters * 3)))
        min_samples = max(5, int(min_cluster_size / 2))
    else:
        # Parameters for larger number of more specific clusters (Leadership)
        if name == "Leadership Styles":
            # Special handling for leadership styles - use smaller clusters
            min_cluster_size = 10  # Fixed smaller value
            min_samples = 3        # Reduced to allow more clusters
        else:
            min_cluster_size = max(5, int(len(df) / (expected_clusters * 2)))
            min_samples = max(3, int(min_cluster_size / 3))
    
    print(f"Using min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",  # Excess of Mass - better for semantic clusters
        cluster_selection_epsilon=0.5,   # More relaxed cluster selection
        prediction_data=True             # Generate prediction data for potential new points
    )
    
    # Cluster using the 2D UMAP embeddings for better results
    cluster_labels = clusterer.fit_predict(embedding_2d)
    
    # Create result DataFrame
    result_df = df.copy()
    result_df["cluster"] = cluster_labels
    result_df["cluster_label"] = result_df["cluster"].apply(
        lambda x: f"Cluster {x}" if x >= 0 else "Noise"
    )
    
    # Add UMAP coordinates
    result_df["umap_x"] = embedding_2d[:, 0]
    result_df["umap_y"] = embedding_2d[:, 1]
    
    # Calculate cluster metrics
    true_labels = pd.Categorical(df[label_column]).codes
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    ami_score = adjusted_mutual_info_score(true_labels, cluster_labels)
    
    print(f"Number of clusters found: {len(set(cluster_labels) - {-1})}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Adjusted Mutual Information: {ami_score:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Clusters
    cluster_palette = sns.color_palette("husl", len(result_df["cluster_label"].unique()))
    
    for i, cluster in enumerate(sorted(result_df["cluster_label"].unique())):
        mask = result_df["cluster_label"] == cluster
        axes[0].scatter(
            result_df.loc[mask, "umap_x"],
            result_df.loc[mask, "umap_y"],
            label=cluster,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    axes[0].set_title(f"HDBSCAN Clusters ({name})", fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    
    # Plot 2: True categories
    category_palette = sns.color_palette("deep", len(result_df[label_column].unique()))
    
    for i, category in enumerate(sorted(result_df[label_column].unique())):
        mask = result_df[label_column] == category
        axes[1].scatter(
            result_df.loc[mask, "umap_x"],
            result_df.loc[mask, "umap_y"],
            label=category,
            alpha=0.7,
            s=60,
            edgecolors="white",
            linewidth=0.5
        )
    
    axes[1].set_title(f"True {name} Categories", fontsize=14)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    
    # Add metrics to figure
    plt.figtext(
        0.5, 0.01,
        f"Clustering Metrics - ARI: {ari_score:.3f}, AMI: {ami_score:.3f}\n"
        f"Number of clusters found: {len(set(cluster_labels) - {-1})}, "
        f"Noise points: {(cluster_labels == -1).sum()}",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save figure
    output_file = VISUALIZATIONS_DIR / f"{name.lower().replace(' ', '_')}_cluster_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved cluster analysis to {output_file}")
    
    # Create heatmap showing category-cluster relationship
    plt.figure(figsize=(12, 8))
    
    # Create crosstab of category vs cluster
    category_cluster_crosstab = pd.crosstab(
        result_df[label_column],
        result_df["cluster_label"],
        normalize="index"  # Normalize by row (category)
    )
    
    # Plot heatmap
    sns.heatmap(
        category_cluster_crosstab,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    
    plt.title(f"Relationship Between {name} Categories and Clusters", fontsize=14)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file = VISUALIZATIONS_DIR / f"{name.lower().replace(' ', '_')}_category_cluster_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    print(f"Saved category-cluster heatmap to {heatmap_file}")
    
    return {
        "ari_score": ari_score,
        "ami_score": ami_score,
        "num_clusters": len(set(cluster_labels) - {-1}),
        "noise_points": (cluster_labels == -1).sum()
    }

def save_results(data):
    """Save results as a pickle file."""
    output_file = PROCESSED_DIR / "embedding_analysis_results.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved results to {output_file}")

def create_comparison_report(big_five_metrics, leadership_metrics):
    """Create a text report comparing Big Five and leadership results."""
    # Handle clustering comparison, accounting for edge cases
    big_five_clusters = big_five_metrics['clustering']['num_clusters']
    leadership_clusters = leadership_metrics['clustering']['num_clusters']
    
    # Determine which dataset better distinguishes categories via clustering
    if big_five_clusters == 0 and leadership_clusters == 0:
        clustering_comparison = "Neither dataset produced meaningful clusters with HDBSCAN"
    elif big_five_clusters == 0:
        clustering_comparison = "Only leadership styles produced meaningful clusters"
    elif leadership_clusters == 0:
        clustering_comparison = "Only Big Five traits produced meaningful clusters"
    else:
        # If both have clusters, compare metrics
        if big_five_metrics['clustering']['ari_score'] > leadership_metrics['clustering']['ari_score']:
            clustering_comparison = "Big Five categories are more easily distinguished by unsupervised clustering"
        else:
            clustering_comparison = "Leadership categories are more easily distinguished by unsupervised clustering"
    
    # Determine which has stronger effect size 
    leadership_stronger = leadership_metrics['distance']['effect_size'] > big_five_metrics['distance']['effect_size']
    max_effect = max(leadership_metrics['distance']['effect_size'], big_five_metrics['distance']['effect_size'])
    effect_diff = abs(big_five_metrics['distance']['effect_size'] - leadership_metrics['distance']['effect_size'])
    
    # Create report
    report = f"""
    Embedding Analysis Comparison: Big Five vs. Leadership Styles
    ===========================================================
    
    Effect Size Comparison:
    ----------------------
    Big Five Effect Size: {big_five_metrics['distance']['effect_size']:.4f}
    Leadership Effect Size: {leadership_metrics['distance']['effect_size']:.4f}
    
    Intra-Category Distances:
    -----------------------
    Big Five: {big_five_metrics['distance']['intra_mean']:.4f} ± {big_five_metrics['distance']['intra_std']:.4f}
    Leadership: {leadership_metrics['distance']['intra_mean']:.4f} ± {leadership_metrics['distance']['intra_std']:.4f}
    
    Inter-Category Distances:
    -----------------------
    Big Five: {big_five_metrics['distance']['inter_mean']:.4f} ± {big_five_metrics['distance']['inter_std']:.4f}
    Leadership: {leadership_metrics['distance']['inter_mean']:.4f} ± {leadership_metrics['distance']['inter_std']:.4f}
    
    Clustering Metrics:
    -----------------
    Big Five - Adjusted Rand Index: {big_five_metrics['clustering']['ari_score']:.4f}
    Leadership - Adjusted Rand Index: {leadership_metrics['clustering']['ari_score']:.4f}
    
    Big Five - Adjusted Mutual Information: {big_five_metrics['clustering']['ami_score']:.4f}
    Leadership - Adjusted Mutual Information: {leadership_metrics['clustering']['ami_score']:.4f}
    
    Clusters Found:
    -------------
    Big Five: {big_five_clusters} clusters (vs. 5 true categories)
    Leadership: {leadership_clusters} clusters (vs. {len(POSITIVE_LEADERSHIP_STYLES)} true categories)
    
    Interpretation:
    -------------
    {"Leadership styles show stronger separation" if leadership_stronger else "Big Five traits show stronger separation"} with an effect size difference of {effect_diff:.4f}.
    
    The clustering analysis shows that {clustering_comparison}.
    
    Key Findings:
    -----------
    1. TF-IDF vectorization and UMAP visualization reveal substantial overlap between both Big Five 
       traits and leadership styles in semantic space.
       
    2. The effect size between intra-category and inter-category distances is {"larger for leadership styles" if leadership_stronger else "larger for Big Five traits"} ({max_effect:.4f}), suggesting that {"leadership constructs" if leadership_stronger else "personality traits"} show more internal cohesion.
    
    3. The semantic space of both domains shows substantial overlap, suggesting that 
       current measurement approaches may not fully capture distinct constructs.
    
    Questions for further analysis:
    ----------------------------
    1. Why are leadership constructs more similar to each other than personality traits?
    2. Which specific leadership constructs show the greatest overlap?
    3. Would a different embedding approach (e.g., fine-tuning with GIST loss) improve the separation?
    4. Are there alternative language-based methods that could better differentiate these constructs?
    """
    
    report_file = VISUALIZATIONS_DIR / "big_five_leadership_comparison_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"Saved comparison report to {report_file}")
    return report

def main():
    # Part 1: Analyze Big Five personality traits
    print("\n==== Analyzing Big Five Personality Traits ====\n")
    # 1. Load and preprocess data
    ipip_df = load_ipip_data()
    big_five_df = filter_big_five_traits(ipip_df)
    
    # 2. Generate embeddings
    big_five_embeddings, big_five_vectorizer = generate_tfidf_embeddings(big_five_df, text_column="text")
    
    # 3. Analyze distances
    big_five_distance_metrics = analyze_distances(big_five_embeddings, big_five_df, label_column="big_five")
    
    # 4. Visualize with UMAP
    big_five_umap_df, big_five_umap_coords = visualize_embeddings_umap(
        big_five_embeddings, big_five_df, label_column="big_five", name="Big Five Personality Traits"
    )
    
    # 5. Perform cluster analysis
    big_five_cluster_metrics = perform_cluster_analysis(
        big_five_embeddings, big_five_df, big_five_umap_coords, 
        label_column="big_five", name="Big Five Personality Traits"
    )
    
    # Part 2: Analyze leadership styles
    print("\n==== Analyzing Leadership Styles ====\n")
    # 1. Load and preprocess data
    leadership_df = load_leadership_data(dataset_name="focused_clean")
    positive_leadership_df = filter_positive_leadership_styles(leadership_df)
    
    # 2. Generate embeddings
    leadership_embeddings, leadership_vectorizer = generate_tfidf_embeddings(
        positive_leadership_df, text_column="ProcessedText"
    )
    
    # 3. Analyze distances
    leadership_distance_metrics = analyze_distances(
        leadership_embeddings, positive_leadership_df, label_column="StandardConstruct"
    )
    
    # 4. Visualize with UMAP
    leadership_umap_df, leadership_umap_coords = visualize_embeddings_umap(
        leadership_embeddings, positive_leadership_df, 
        label_column="StandardConstruct", name="Leadership Styles"
    )
    
    # 5. Perform cluster analysis
    leadership_cluster_metrics = perform_cluster_analysis(
        leadership_embeddings, positive_leadership_df, leadership_umap_coords,
        label_column="StandardConstruct", name="Leadership Styles"
    )
    
    # Aggregate results
    results = {
        "big_five": {
            "data": big_five_df,
            "embeddings": big_five_embeddings,
            "distance": big_five_distance_metrics,
            "clustering": big_five_cluster_metrics
        },
        "leadership": {
            "data": positive_leadership_df,
            "embeddings": leadership_embeddings,
            "distance": leadership_distance_metrics,
            "clustering": leadership_cluster_metrics
        },
        "method": "TF-IDF"
    }
    
    # Save results
    save_results(results)
    
    # Generate comparison report
    comparison_report = create_comparison_report(
        results["big_five"], results["leadership"]
    )
    
    print("\nAnalysis complete! Results saved to the visualizations directory.")
    print(f"Big Five effect size: {big_five_distance_metrics['effect_size']:.4f}")
    print(f"Leadership effect size: {leadership_distance_metrics['effect_size']:.4f}")
    
    print("\nKey comparison findings:")
    print(f"- {'Big Five traits' if big_five_distance_metrics['effect_size'] > leadership_distance_metrics['effect_size'] else 'Leadership styles'} show stronger categorical separation")
    print(f"- {'Big Five categories' if big_five_cluster_metrics['ari_score'] > leadership_cluster_metrics['ari_score'] else 'Leadership categories'} are more easily distinguished by clustering")

if __name__ == "__main__":
    main()