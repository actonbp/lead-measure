#!/usr/bin/env python3
"""
GIST Loss Implementation for Personality Trait Analysis

This script implements the GIST loss approach to analyze personality traits
from the IPIP dataset, visualizing the clusters with UMAP. It can be run
in either 'big_five' mode or 'ipip_all' mode (using all facets).

Usage:
    python gist_loss_personality.py [--mode ipip_all]

Output:
    - Visualizations of embeddings in the visualizations directory
    - Trained model saved to models directory
    - Analysis reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import argparse
import itertools # Added for combinations

# Import SentenceTransformers
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Big Five trait mapping - categories that map to the Big Five
# Kept for potential future use or reference, but not used if mode='ipip_all'
BIG_FIVE_MAPPING = {
    'Openness': [
        'Complexity', 'Imagination', 'Creativity/Originality', 'Intellect', 
        'Intellectual Openness', 'Understanding', 'Depth', 'Culture', 'Ingenuity'
    ],
    'Conscientiousness': [
        'Orderliness', 'Dutifulness', 'Achievement-striving', 'Competence', 
        'Organization', 'Efficiency', 'Industriousness/Perseverance/Persistence',
        'Purposefulness', 'Deliberateness', 'Methodicalness'
    ],
    'Extraversion': [
        'Gregariousness', 'Assertiveness', 'Warmth', 'Talkativeness', 
        'Sociability', 'Vitality/Enthusiasm/Zest', 'Exhibitionism',
        'Leadership', 'Friendliness', 'Positive Expressivity'
    ],
    'Agreeableness': [
        'Compassion', 'Cooperation', 'Sympathy', 'Empathy', 'Nurturance',
        'Pleasantness', 'Tenderness', 'Morality', 'Docility'
    ],
    'Neuroticism': [
        'Anxiety', 'Emotionality', 'Anger', 'Distrust', 'Negative-Valence',
        'Stability', 'Emotional Stability'
    ]
}

def load_ipip_data():
    """Load and preprocess the IPIP personality items."""
    ipip_path = DATA_DIR / "ipip.csv"
    
    if not ipip_path.exists():
        raise FileNotFoundError(f"IPIP dataset not found at {ipip_path}")
    
    print(f"Loading IPIP dataset from {ipip_path}")
    
    # Load the dataset, handle encoding issues
    try:
        # Assuming 'text' and 'label' are the correct column names
        # Handle potential missing values in text
        ipip_df = pd.read_csv(ipip_path, encoding='utf-8')
    except UnicodeDecodeError:
        ipip_df = pd.read_csv(ipip_path, encoding='latin1')
        
    if 'text' not in ipip_df.columns or 'label' not in ipip_df.columns:
         raise ValueError("Expected columns 'text' and 'label' not found in ipip.csv")

    ipip_df.dropna(subset=['text', 'label'], inplace=True)
    ipip_df['text'] = ipip_df['text'].astype(str)
    ipip_df['label'] = ipip_df['label'].astype(str)

    print(f"Loaded {len(ipip_df)} personality items after cleaning")
    return ipip_df

def filter_big_five_traits(ipip_df):
    """Filter items to only include Big Five personality traits."""
    # Map each item to its Big Five category if applicable
    def map_to_big_five(label):
        for big_five, labels in BIG_FIVE_MAPPING.items():
            if label in labels:
                return big_five
        return None
    
    # Apply the mapping
    ipip_df['big_five'] = ipip_df['label'].apply(map_to_big_five)
    
    # Filter to only include items mapped to Big Five
    big_five_df = ipip_df.dropna(subset=['big_five']).copy() # Use .copy() to avoid SettingWithCopyWarning
    
    print(f"Filtered to {len(big_five_df)} items mapped to Big Five traits")
    print(big_five_df['big_five'].value_counts())
    
    return big_five_df

def prepare_gist_training_data(df, text_column="text", label_column="label"):
    """
    Prepare training data for GIST loss fine-tuning.
    
    GIST loss works with positive pairs (items from the same category).
    This version generates ALL positive pairs within each label group.
    """
    # Ensure label column exists and has enough variance for stratification
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame.")
    if df[label_column].nunique() < 2:
        print(f"Warning: Only {df[label_column].nunique()} unique labels found. Stratification might not be effective.")
        test_size = 0.2
        stratify = None
    else:
        # Check for labels with only one sample - causes issues with stratification
        label_counts = df[label_column].value_counts()
        labels_to_keep = label_counts[label_counts > 1].index
        df_filtered = df[df[label_column].isin(labels_to_keep)]
        
        if len(df_filtered) < len(df):
            print(f"Warning: Removed {len(df) - len(df_filtered)} items belonging to labels with only one sample before train/test split.")
            df = df_filtered

        if df[label_column].nunique() < 2:
             print(f"Warning: After removing single-sample labels, only {df[label_column].nunique()} unique labels remain. Stratification disabled.")
             test_size = 0.2
             stratify = None
        else:
            test_size = 0.2
            stratify = df[label_column]

    # Split data into train/test
    try:
         train_df, test_df = train_test_split(
              df, test_size=test_size, stratify=stratify, random_state=42
         )
    except ValueError as e:
         print(f"Error during train_test_split (likely due to stratification issues): {e}")
         print("Falling back to non-stratified split.")
         train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Group items by trait
    trait_to_items = {}
    for trait in train_df[label_column].unique():
        trait_items = train_df[train_df[label_column] == trait][text_column].tolist()
        trait_to_items[trait] = trait_items
    
    # Create training examples - pairs from the same trait
    train_examples = []
    
    for trait, items in trait_to_items.items():
        # Ensure we have at least 2 items per trait
        if len(items) < 2:
            continue
            
        # Create positive pairs using all combinations
        # FROM ITERTOOLS, WE WANT TO GET EVERY COMBONATION OF PARIS FOR THAT CONSTRUCT (FOR ITEMS); TRAINING ERROR IS HIGHER WHEN WE DO THIS
        for item1, item2 in itertools.combinations(items, 2):
             train_examples.append(InputExample(texts=[item1, item2]))

    print(f"Created {len(train_examples)} training pairs (all combinations) from {len(train_df)} training items.")
    print(f"Using {len(test_df)} items for evaluation.")
    
    return train_examples, train_df, test_df

def create_evaluator(df, trait_column="label", text_column="text"):
    """Create evaluator for measuring model performance during training."""
    # Create arrays for sentences1, sentences2, and scores
    sentences1 = []
    sentences2 = []
    scores = []
    
    traits = df[trait_column].unique()
    if len(traits) < 2:
        print("Warning: Not enough unique traits in evaluation set to create negative pairs. Evaluator might be less effective.")
        
    # Generate positive pairs (same trait)
    for trait in traits:
        trait_items = df[df[trait_column] == trait][text_column].tolist()
        if len(trait_items) >= 2:
            # Limit pairs per trait to avoid huge evaluator data
            max_items_per_trait = 30 
            num_items = min(len(trait_items), max_items_per_trait)
            for i in range(num_items):
                for j in range(i + 1, num_items):
                    sentences1.append(trait_items[i])
                    sentences2.append(trait_items[j])
                    scores.append(1.0)  # 1.0 = similar
    
    # Generate negative pairs (different traits)
    # Aim for roughly equal number of positive and negative pairs if possible
    num_positive_pairs = len(scores)
    num_negative_pairs_to_generate = min(num_positive_pairs * 2, 5000) # Limit total negative pairs

    if len(traits) >= 2:
        generated_neg_pairs = 0
        attempts = 0
        max_attempts = num_negative_pairs_to_generate * 5 # Limit attempts to avoid infinite loop

        while generated_neg_pairs < num_negative_pairs_to_generate and attempts < max_attempts:
            attempts += 1
            try:
                # Randomly select two different traits
                trait1, trait2 = np.random.choice(traits, 2, replace=False)
                
                # Get items from each trait
                items1 = df[df[trait_column] == trait1][text_column].tolist()
                items2 = df[df[trait_column] == trait2][text_column].tolist()

                if not items1 or not items2: # Skip if a trait has no items (shouldn't happen with proper filtering)
                    continue

                item1 = np.random.choice(items1)
                item2 = np.random.choice(items2)
                
                # Avoid adding duplicate pairs (less likely but possible)
                # Simple check for now, could be more robust
                is_duplicate = False
                for k in range(len(sentences1)):
                    if scores[k] == 0.0 and \
                       ((sentences1[k] == item1 and sentences2[k] == item2) or \
                        (sentences1[k] == item2 and sentences2[k] == item1)):
                       is_duplicate = True
                       break
                if not is_duplicate:
                    sentences1.append(item1)
                    sentences2.append(item2)
                    scores.append(0.0)  # 0.0 = dissimilar
                    generated_neg_pairs += 1
            except Exception as e:
                print(f"Warning: Error generating negative pair: {e}")
                continue # Skip this attempt

        print(f"Generated {num_positive_pairs} positive and {generated_neg_pairs} negative pairs for evaluation.")
    else:
        print("Skipping negative pair generation for evaluator due to insufficient unique traits.")

    if not sentences1:
         print("Warning: No evaluation pairs generated. Cannot create evaluator.")
         return None

    return evaluation.EmbeddingSimilarityEvaluator(sentences1=sentences1, sentences2=sentences2, scores=scores)

def train_gist_model(train_examples, evaluator, model_name="all-MiniLM-L6-v2", epochs=2, save_name_prefix="model"):
    """Train a SentenceTransformer model using GIST loss."""
    print(f"Training model with GIST loss using {model_name} as base model")
    
    # Initialize model
    # Use a pre-trained Sentence Transformer model
    model = SentenceTransformer(model_name)
    
    # Initialize a separate guide model (can be the same architecture)
    # TWO MODELS; RELEVANT EXAMPLES (GUIDE; LET ME GET YOU SOME NEGATIVES) & 2nd model IS MODEL (COMPLEX ONE)
    guide_model = SentenceTransformer(model_name) 
    print(f"Initializing guide model for GISTEmbedLoss with {model_name}")

    # Initialize GISTEmbedLoss
    # This loss function works with positive pairs. Negatives are sampled implicitly, guided by the 'guide' model.
    #### WE CAN KEEP THE MULTIPLE NEGATIVE AS ANOTHER OPTION, BUT WE INSTEAD WANT TO USE THE GIST LOSS. 
    train_loss = losses.GISTEmbedLoss(model=model, guide=guide_model) # Use GISTEmbedLoss, providing the guide model
    print("Using GISTEmbedLoss for training.")


    # Initialize MultipleNegativesRankingLoss (serves as GIST loss implementation) -> No longer used as GIST loss here
    # This loss function works with positive pairs and uses in-batch negatives
    # train_loss = losses.MultipleNegativesRankingLoss(model) ## THIS IS THE OLDER STRONG WAY, WE CAN TRY BOTH. CHANGE THE GIST EMBEDD LOSS -> Commented out

    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16) # Batch size might need adjustment
    
    # Set up warmup steps (10% of training steps)
    num_training_steps = len(train_dataloader) * epochs
    warmup_steps = int(num_training_steps * 0.1)
    
    output_save_path = str(MODELS_DIR / f"{save_name_prefix}_gist_{model_name.replace('/', '_')}")
    print(f"Model will be saved to: {output_save_path}")

    # Train the model ## OLDER VERSION: SBERT DOCUMENTATION ON EMBED LOSS, GIST LOSS,
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=int(len(train_dataloader) * 0.1), # Evaluate every 10% of an epoch
        warmup_steps=warmup_steps,
        output_path=output_save_path,
        save_best_model=True, # Save the best model based on evaluator performance
        checkpoint_path=str(MODELS_DIR / "checkpoints" / save_name_prefix), # Save checkpoints
        checkpoint_save_steps=int(len(train_dataloader) * 0.5), # Save checkpoint twice per epoch
        show_progress_bar=True
    )
    
    # Load the best model after training
    print(f"Loading best model from {output_save_path}")
    best_model = SentenceTransformer(output_save_path)

    return best_model # Return the best performing model

def generate_embeddings(model, df, text_column="text"):
    """Generate embeddings for all items."""
    print(f"Generating embeddings for {len(df)} items using column '{text_column}'")
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def analyze_distances(embeddings, df, trait_column="label", save_name_prefix="analysis"):
    """Analyze intra-trait and inter-trait distances."""
    print("Analyzing semantic distances between traits")
    
    # Calculate all pairwise distances (cosine distance = 1 - cosine similarity)
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, 1.0) # Ensure diagonal is exactly 1 ## DO WE NEED TO DO THIS? 
    distances = 1 - similarities
    
    # Get trait labels
    traits = df[trait_column].values
    unique_traits = df[trait_column].unique()
    print(f"Analyzing {len(unique_traits)} unique traits/labels.")

    # Initialize containers for distances
    intra_trait_distances = []
    inter_trait_distances = []
    
    # Collect distances efficiently
    trait_map = {trait: i for i, trait in enumerate(unique_traits)}
    trait_indices = [trait_map[t] for t in traits]
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):  # Upper triangle only, excluding diagonal
            dist = distances[i, j]
            # Handle potential NaN/inf values resulting from embedding issues
            if not np.isfinite(dist):
                print(f"Warning: Non-finite distance encountered between item {i} and {j}. Skipping.")
                continue

            if trait_indices[i] == trait_indices[j]:
                intra_trait_distances.append(dist)
            else:
                inter_trait_distances.append(dist)
    
    # Convert to numpy arrays
    intra_distances = np.array(intra_trait_distances)
    inter_distances = np.array(inter_trait_distances)

    # Handle cases where one type of distance might be empty
    mean_intra = np.mean(intra_distances) if len(intra_distances) > 0 else np.nan
    std_intra = np.std(intra_distances) if len(intra_distances) > 0 else np.nan
    mean_inter = np.mean(inter_distances) if len(inter_distances) > 0 else np.nan
    std_inter = np.std(inter_distances) if len(inter_distances) > 0 else np.nan

    print(f"Intra-trait distances: mean = {mean_intra:.4f}, std = {std_intra:.4f} (n={len(intra_distances)})")
    print(f"Inter-trait distances: mean = {mean_inter:.4f}, std = {std_inter:.4f} (n={len(inter_distances)})")
    
    # Calculate Cohen's d for effect size
    pooled_std = np.sqrt(((len(intra_distances) - 1) * std_intra**2 + (len(inter_distances) - 1) * std_inter**2) / 
                         (len(intra_distances) + len(inter_distances) - 2)) if len(intra_distances) > 1 and len(inter_distances) > 1 else np.nan
    
    cohen_d = (mean_inter - mean_intra) / pooled_std if pooled_std > 0 else np.nan
    print(f"Effect size (Cohen's d): {cohen_d:.4f}")
    
    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    sns.histplot(intra_distances, color="skyblue", label="Intra-trait (Same)", kde=True, stat="density", common_norm=False)
    sns.histplot(inter_distances, color="salmon", label="Inter-trait (Different)", kde=True, stat="density", common_norm=False)
    plt.title(f"Distribution of Cosine Distances ({save_name_prefix})")
    plt.xlabel("Cosine Distance (1 - Similarity)")
    plt.ylabel("Density")
    plt.legend()
    dist_plot_path = VISUALIZATIONS_DIR / f"{save_name_prefix}_distance_distribution.png"
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"Saved distance distribution to {dist_plot_path}")

    metrics = {
        'mean_intra_distance': mean_intra,
        'std_intra_distance': std_intra,
        'mean_inter_distance': mean_inter,
        'std_inter_distance': std_inter,
        'cohen_d': cohen_d
    }
    return metrics
## RESULTS CAN CHANGE ALOT BASED ON UMAP SETTINGS; WE CAN TRY A DIFFERENT ONE 

## UMAP REPULSION_STRENGTH (BY DEFAULT 1.0) & MIN_DIST (BY DEFAULT 0.1), TRY DIFFERENT THINGS TO EMPHASIZE LOCAL STRUCTURE VS GLOBAL STRUCTURE INCREASE FROM 1.0 TO 1.5
def visualize_embeddings_umap(embeddings, df, trait_column="label", save_name_prefix="analysis", n_neighbors=15, min_dist=0.1):
    """Create UMAP visualization of embeddings, colored by trait."""
    print("Creating UMAP visualization")
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_components=2, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Use trait labels for coloring
    labels = df[trait_column].astype('category')
    num_labels = len(labels.cat.categories)
    
    # Use a diverse colormap suitable for many categories
    colors = plt.cm.get_cmap('turbo', num_labels) # turbo is good for many distinct colors
    
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                          c=labels.cat.codes, 
                          cmap=colors, 
                          s=5, # Smaller points for clarity with many items
                          alpha=0.7)
                          
    plt.title(f"UMAP Projection of Item Embeddings ({save_name_prefix})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create a legend - challenging with many labels, show a subset or use alternative annotation?
    # For now, let's add a colorbar as a proxy if too many labels
    if num_labels <= 30: # Arbitrary threshold for showing legend handles
         legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat, 
                                      markerfacecolor=colors(i/num_labels), markersize=5) 
                           for i, cat in enumerate(labels.cat.categories)]
         plt.legend(handles=legend_handles, title=trait_column, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
         # Add a color bar as a proxy legend if too many categories
         cbar = plt.colorbar(scatter, ticks=np.linspace(0, num_labels-1, min(num_labels, 10))) # Show limited ticks
         cbar.set_label(f'{trait_column} (Code)', rotation=270, labelpad=15)
         print(f"Warning: Too many labels ({num_labels}) for a standard legend. Showing color bar instead.")

    umap_plot_path = VISUALIZATIONS_DIR / f"{save_name_prefix}_umap_visualization.png"
    plt.savefig(umap_plot_path, bbox_inches='tight') # Use bbox_inches='tight' for legend
    plt.close()
    print(f"Saved UMAP visualization to {umap_plot_path}")
    
    return embedding_2d

def perform_cluster_analysis(embeddings, df, embedding_2d, trait_column="label", save_name_prefix="analysis", min_cluster_size=5):
    """Perform HDBSCAN clustering and evaluate against true labels."""
    print("Performing HDBSCAN clustering")
    
    # Cluster using HDBSCAN on the original high-dimensional embeddings
    # Using 'euclidean' metric to avoid potential issues with 'cosine' in underlying libraries (e.g., BallTree)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=True)
    print("Using HDBSCAN with metric='euclidean'")
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Add cluster labels to the DataFrame
    df['cluster'] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(df)*100:.2f}%)")

    # Evaluate clustering quality against the true trait labels
    # Exclude noise points from evaluation metrics? Typically yes.
    valid_indices = cluster_labels != -1
    true_labels_valid = df.loc[valid_indices, trait_column].values
    cluster_labels_valid = cluster_labels[valid_indices]

    if len(true_labels_valid) > 0 and len(np.unique(true_labels_valid)) > 1 and len(np.unique(cluster_labels_valid)) > 0:
        ari = adjusted_rand_score(true_labels_valid, cluster_labels_valid)
        ami = adjusted_mutual_info_score(true_labels_valid, cluster_labels_valid)
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Adjusted Mutual Information (AMI): {ami:.4f}")
    else:
        print("Could not calculate ARI/AMI (insufficient data points or clusters after removing noise).")
        ari = np.nan
        ami = np.nan

    # --- Visualization 1: UMAP colored by HDBSCAN clusters ---
    plt.figure(figsize=(14, 10))
    
    # Use a categorical palette, add grey for noise (-1)
    unique_cluster_labels = sorted(list(set(cluster_labels)))
    # Ensure palette size covers the max cluster index, not just the count
    max_cluster_label = max(unique_cluster_labels) if unique_cluster_labels else -1
    num_cluster_colors_needed = max_cluster_label + 1 # Palette needs indices 0 through max_cluster_label
    # num_cluster_colors = len(unique_cluster_labels) -1 if -1 in unique_cluster_labels else len(unique_cluster_labels) # Old calculation
    print(f"Max cluster label: {max_cluster_label}, needing {num_cluster_colors_needed} colors for palette.")
    
    # Create a palette, handling the potential for noise points (-1)
    # Use a cyclical palette if many clusters needed, like 'husl' or 'hls'
    palette_name = 'husl' if num_cluster_colors_needed > 20 else 'deep' # Switch palette for many clusters
    palette = sns.color_palette(palette_name, num_cluster_colors_needed)
    
    # Create color map, assigning grey to noise (-1)
    color_map = {label: palette[label] for label in unique_cluster_labels if label != -1}
    # color_map = {label: palette[i] for i, label in enumerate(unique_cluster_labels) if label != -1} # Old map creation
    color_map[-1] = (0.5, 0.5, 0.5) # Grey for noise

    colors_for_points = [color_map[label] for label in cluster_labels]

    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                c=colors_for_points, 
                s=5, 
                alpha=0.7)
                
    plt.title(f"UMAP Projection Colored by HDBSCAN Clusters ({save_name_prefix})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    # Add legend for clusters? Can be noisy. Maybe just title indicates coloring.
    # Optional: Add a text label for noise points percentage
    plt.text(0.95, 0.01, f'{n_noise} noise points ({n_noise/len(df)*100:.1f}%)',
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, color='gray', fontsize=9)

    cluster_plot_path = VISUALIZATIONS_DIR / f"{save_name_prefix}_cluster_analysis.png"
    plt.savefig(cluster_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster analysis visualization to {cluster_plot_path}")

    # --- Visualization 2: Heatmap of True Traits vs. Clusters ---
    if n_clusters > 0 and len(true_labels_valid) > 0: # Only create heatmap if clusters were found
         contingency_matrix = pd.crosstab(df.loc[valid_indices, trait_column], 
                                          df.loc[valid_indices, 'cluster'])
         
         # Normalize the matrix for better color comparison (optional)
         # contingency_matrix_norm = contingency_matrix.apply(lambda x: x / x.sum(), axis=1)

         plt.figure(figsize=(min(20, n_clusters * 0.5 + 5), min(15, len(contingency_matrix.index) * 0.3 + 3))) # Dynamic sizing
         sns.heatmap(contingency_matrix, annot=True, fmt="d", cmap="viridis")
         plt.title(f"Heatmap of True {trait_column} vs. HDBSCAN Clusters ({save_name_prefix})")
         plt.ylabel(f"True {trait_column}")
         plt.xlabel("HDBSCAN Cluster Label")
         heatmap_plot_path = VISUALIZATIONS_DIR / f"{save_name_prefix}_trait_cluster_heatmap.png"
         try:
              plt.savefig(heatmap_plot_path, bbox_inches='tight')
              print(f"Saved trait-cluster heatmap to {heatmap_plot_path}")
         except Exception as e:
              print(f"Warning: Could not save heatmap plot, possibly due to large size or other issue: {e}")
         plt.close()
         
    metrics = {
        'num_hdbscan_clusters': n_clusters,
        'num_noise_points': n_noise,
        'adjusted_rand_index': ari,
        'adjusted_mutual_info': ami
    }
    return metrics

def save_results(embeddings, df, metrics, model_name, save_name_prefix="results"):
    """Save embeddings, dataframe with clusters, and metrics."""
    results_path = PROCESSED_DIR / f"{save_name_prefix}_gist_results.pkl"
    
    # Ensure metrics includes all analyses performed (distance, cluster, 1-NN accuracy)
    results = {
        'embeddings': embeddings, # Embeddings for the full dataset (used for UMAP/HDBSCAN)
        'dataframe': df, # Contains original data + cluster labels for the full dataset
        'metrics': metrics, # Combined dictionary of distance, cluster, and 1-NN accuracy metrics
        'model_name': model_name
    }
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved results (embeddings, dataframe, metrics) to {results_path}")

def main(mode='big_five'):
    """Main execution function."""
    print(f"--- Running GIST Loss Analysis in '{mode}' mode ---")
    
    # Common settings
    # model_base_name = "all-MiniLM-L6-v2" # Changed default as mpnet can be large -> Reverted to mpnet
    model_base_name = "all-mpnet-base-v2" 
    print(f"Using base model: {model_base_name}")
    epochs = 2
    text_column = "text" # Assuming this is the item text column in ipip.csv

    # 1. Load Data
    ipip_df = load_ipip_data()

    if mode == 'big_five':
        # --- Big Five Mode ---
        analysis_df = filter_big_five_traits(ipip_df)
        label_column = "big_five"
        save_name_prefix = "big_five"
        min_cluster_size_hdbscan = 5 # Default for Big Five might be reasonable
    elif mode == 'ipip_all':
        # --- All IPIP Facets Mode ---
        analysis_df = ipip_df.copy() # Use the full dataframe
        label_column = "label" # Use the original facet labels
        save_name_prefix = "ipip_all"
        # Adjust clustering parameters if needed for many facets
        min_cluster_size_hdbscan = 3 # Might need smaller min size for specific facets
        print(f"Using all {analysis_df[label_column].nunique()} unique labels from the dataset.")
    else:
        raise ValueError("Invalid mode. Choose 'big_five' or 'ipip_all'.")

    if analysis_df.empty:
         print("Error: DataFrame is empty after loading/filtering. Exiting.")
         return
         
    # 2. Prepare Training Data & Evaluator
    print(f"Preparing training data using label column: '{label_column}'")
    train_examples, train_df, test_df = prepare_gist_training_data(
        analysis_df, text_column=text_column, label_column=label_column
    )
    
    print("Creating evaluation set...")
    evaluator = create_evaluator(
         test_df, trait_column=label_column, text_column=text_column
    )
    if evaluator is None:
        print("Warning: Could not create evaluator. Training will proceed without evaluation steps.")

    # 3. Train Model
    model = train_gist_model(
        train_examples, 
        evaluator, 
        model_name=model_base_name, 
        epochs=epochs, 
        save_name_prefix=save_name_prefix
    )
    
    # --- 1-Nearest Neighbor Evaluation --- 
    print("\n--- Performing 1-Nearest Neighbor Evaluation ---")
    # Generate embeddings for train and test sets specifically
    print("Generating embeddings for training set...")
    train_embeddings = generate_embeddings(model, train_df, text_column=text_column)
    print("Generating embeddings for test set...")
    test_embeddings = generate_embeddings(model, test_df, text_column=text_column)

    # Check for non-finite values in train/test embeddings
    if not np.all(np.isfinite(train_embeddings)) or not np.all(np.isfinite(test_embeddings)):
         print("Warning: Non-finite values found in train/test embeddings. Replacing with 0.")
         train_embeddings = np.nan_to_num(train_embeddings)
         test_embeddings = np.nan_to_num(test_embeddings)

    # Calculate cosine similarities between test and train embeddings
    print("Calculating similarities between test and train embeddings...")
    # Resulting shape: (n_test_samples, n_train_samples)
    similarities = cosine_similarity(test_embeddings, train_embeddings)

    # Find the index of the nearest neighbor in the training set for each test item
    print("Finding nearest neighbors...")
    nearest_neighbor_indices = np.argmax(similarities, axis=1)

    # Get the labels of the nearest neighbors
    train_labels = train_df[label_column].values # Get as numpy array for efficient indexing
    predicted_labels = train_labels[nearest_neighbor_indices]

    # Get the true labels of the test set
    true_labels = test_df[label_column].values

    # Calculate accuracy
    print("Calculating 1-NN accuracy...")
    accuracy = np.mean(predicted_labels == true_labels)
    
    print(f"\nOverall 1-Nearest Neighbor Accuracy: {accuracy:.4f}")
    # Store the accuracy metric
    one_nn_metrics = {"one_nn_accuracy": accuracy}
    
    # --- End 1-Nearest Neighbor Evaluation ---

    # 4. Generate Embeddings for ALL relevant items (using the full analysis_df for UMAP/HDBSCAN/Distance analysis)
    print("\nGenerating embeddings for the full dataset (for UMAP/HDBSCAN/Distance analyses)...")
    embeddings = generate_embeddings(model, analysis_df, text_column=text_column)
    
    # Check for NaN/inf values in embeddings
    if not np.all(np.isfinite(embeddings)):
        print("Warning: Non-finite values (NaN or Inf) found in generated embeddings! This may affect analysis.")
        # Optional: Handle them, e.g., replace with zeros or mean, or investigate source.
        embeddings = np.nan_to_num(embeddings) 
        print("Replaced non-finite values with 0.")

    # 5. Analyze Distances (on full dataset)
    distance_metrics = analyze_distances(
        embeddings, analysis_df, trait_column=label_column, save_name_prefix=save_name_prefix
    )
    
    # 6. Visualize Embeddings (UMAP on full dataset)
    # Adjust UMAP parameters potentially based on number of items/labels
    n_neighbors_umap = 15 if len(analysis_df) > 50 else 5 
    min_dist_umap = 0.1 
    embedding_2d = visualize_embeddings_umap(
        embeddings, analysis_df, trait_column=label_column, save_name_prefix=save_name_prefix,
        n_neighbors=n_neighbors_umap, min_dist=min_dist_umap
    )
    
    # 7. Perform Cluster Analysis (HDBSCAN on full dataset)
    cluster_metrics = perform_cluster_analysis(
        embeddings, analysis_df, embedding_2d, trait_column=label_column, 
        save_name_prefix=save_name_prefix, min_cluster_size=min_cluster_size_hdbscan
    )
    
    # 8. Save Combined Results
    # Combine all metrics dictionaries
    all_metrics = {**distance_metrics, **cluster_metrics, **one_nn_metrics}
    save_results(
        embeddings, analysis_df, all_metrics,
        model_name=f"{save_name_prefix}_gist_{model_base_name.replace('/', '_')}", # Use consistent naming
        save_name_prefix=save_name_prefix
    )

    print(f"\n--- Analysis complete for mode '{mode}'! ---")
    print("Results, models, and visualizations saved.")
    if mode == 'big_five':
        print("Next suggested step: Run in 'ipip_all' mode or proceed to leadership analysis.")
    elif mode == 'ipip_all':
         print("Next suggested step: Compare results with 'big_five' mode or proceed to leadership analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GIST Loss Analysis on IPIP Data")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="big_five", 
        choices=["big_five", "ipip_all"],
        help="Analysis mode: 'big_five' for Big Five traits, 'ipip_all' for all IPIP facets."
    )
    args = parser.parse_args()
    
    main(mode=args.mode)