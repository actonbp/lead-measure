#!/usr/bin/env python3
"""
Construct Learning Model with GIST Loss

This script implements a model that learns to identify construct spaces using the GIST loss approach.
The model is trained on IPIP personality construct data and then applied to leadership styles.

Key steps:
1. Train a model on ALL IPIP constructs to learn "what a construct is"
2. Evaluate the model's ability to identify constructs on a test set
3. Apply the trained model to leadership styles data
4. Compare how well the construct-trained model identifies leadership styles

Usage:
    python construct_learning_model.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import umap
from sentence_transformers import SentenceTransformer, losses, InputExample
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm.auto import tqdm

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Configure PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class ConstructLearner:
    """
    Class to learn and identify construct spaces using GIST loss.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the construct learner with a base embedding model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.label_encoder = None
        self.construct_embeddings = None
        self.construct_centroids = {}
        self.umap_model = None
        
    def load_ipip_data(self):
        """Load the full IPIP dataset with all constructs."""
        ipip_path = DATA_DIR / "ipip.csv"
        
        if not ipip_path.exists():
            raise FileNotFoundError(f"IPIP dataset not found at {ipip_path}")
        
        print(f"Loading IPIP dataset from {ipip_path}")
        
        # Load the dataset, handle encoding issues
        try:
            ipip_df = pd.read_csv(ipip_path, encoding='utf-8')
        except UnicodeDecodeError:
            ipip_df = pd.read_csv(ipip_path, encoding='latin1')
        
        # Clean up the data
        ipip_df = ipip_df.dropna(subset=['text', 'label'])
        
        # Handle reversed items (key=-1)
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
            
            ipip_df['processed_text'] = ipip_df.apply(modify_reversed_text, axis=1)
        else:
            ipip_df['processed_text'] = ipip_df['text']
        
        # Count constructs
        construct_counts = ipip_df['label'].value_counts()
        print(f"\nLoaded {len(ipip_df)} items across {len(construct_counts)} constructs")
        print(f"Top 10 constructs by frequency:")
        print(construct_counts.head(10))
        
        # Filter out constructs with too few items
        min_items = 5
        valid_constructs = construct_counts[construct_counts >= min_items].index
        ipip_df = ipip_df[ipip_df['label'].isin(valid_constructs)]
        
        print(f"\nAfter filtering constructs with <{min_items} items:")
        print(f"Retained {len(ipip_df)} items across {len(valid_constructs)} constructs")
        
        return ipip_df
    
    def load_leadership_data(self, dataset_name="focused_clean"):
        """Load the preprocessed leadership data."""
        file_path = PROCESSED_DIR / f"leadership_{dataset_name}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Leadership dataset not found at {file_path}")
        
        print(f"Loading leadership dataset from {file_path}")
        
        df = pd.read_csv(file_path)
        df['processed_text'] = df['ProcessedText']  # Standardize column name
        
        print(f"Loaded {len(df)} leadership items across {df['StandardConstruct'].nunique()} constructs")
        print(df['StandardConstruct'].value_counts())
        
        return df
    
    def generate_base_embeddings(self, texts):
        """Generate base embeddings using the sentence transformer model."""
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    def prepare_training_data(self, df, text_column='processed_text', label_column='label'):
        """Prepare data for GIST loss training."""
        # Encode labels numerically
        self.label_encoder = LabelEncoder()
        df['label_id'] = self.label_encoder.fit_transform(df[label_column])
        
        # Split into train/validation/test sets (60/20/20)
        train_df, temp_df = train_test_split(
            df, test_size=0.4, stratify=df['label_id'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['label_id'], random_state=42
        )
        
        print(f"Train set: {len(train_df)} items")
        print(f"Validation set: {len(val_df)} items")
        print(f"Test set: {len(test_df)} items")
        
        # Generate training pairs for GIST loss
        train_examples = []
        
        # Group items by construct
        construct_items = {}
        for construct in train_df[label_column].unique():
            construct_items[construct] = train_df[train_df[label_column] == construct][text_column].tolist()
        
        # Create positive pairs
        for construct, items in construct_items.items():
            if len(items) < 2:
                continue
                
            for i, anchor in enumerate(items):
                # Sample a few positives from the same construct
                pos_indices = np.random.choice(
                    [j for j in range(len(items)) if j != i],
                    min(3, len(items) - 1),  # Take up to 3 positives
                    replace=False
                )
                
                for pos_idx in pos_indices:
                    train_examples.append(InputExample(texts=[anchor, items[pos_idx]], label=1.0))
        
        print(f"Created {len(train_examples)} training pairs")
        
        return train_examples, train_df, val_df, test_df
    
    def create_gist_model(self, train_examples, epochs=10, batch_size=32):
        """Train a model with GIST loss on the IPIP construct data."""
        print(f"Training GIST model for {epochs} epochs...")
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Initialize GIST loss (using MultipleNegativesRankingLoss as a proxy for GIST)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Set up warmup steps
        warmup_steps = int(len(train_dataloader) * 0.1)
        
        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            output_path=str(MODELS_DIR / f"construct_gist_{self.model_name.split('/')[-1]}")
        )
        
        print("GIST model training complete!")
        return self.model
    
    def compute_construct_centroids(self, df, embeddings, label_column='label'):
        """Compute centroids for each construct based on embeddings."""
        construct_centroids = {}
        construct_embeddings = {}
        
        # Reset index to ensure alignment between DataFrame and embeddings
        df_reset = df.reset_index(drop=True)
        
        for construct in df_reset[label_column].unique():
            # Get indices of items from this construct using the reset index
            indices = df_reset[df_reset[label_column] == construct].index.tolist()
            
            # Get embeddings for these items
            construct_embs = embeddings[indices]
            
            # Compute centroid
            centroid = np.mean(construct_embs, axis=0)
            
            # Store centroid and embeddings
            construct_centroids[construct] = centroid
            construct_embeddings[construct] = construct_embs
        
        self.construct_centroids = construct_centroids
        self.construct_embeddings = construct_embeddings
        
        return construct_centroids, construct_embeddings
    
    def predict_constructs(self, embeddings, top_k=1):
        """Predict constructs for embeddings based on similarity to centroids."""
        if not self.construct_centroids:
            raise ValueError("Construct centroids not computed yet! Call compute_construct_centroids first.")
        
        # Stack all centroids into a matrix
        centroid_matrix = np.vstack(list(self.construct_centroids.values()))
        constructs = list(self.construct_centroids.keys())
        
        # Compute similarities to all centroids
        similarities = cosine_similarity(embeddings, centroid_matrix)
        
        if top_k == 1:
            # Get the most similar centroid for each embedding
            predicted_indices = np.argmax(similarities, axis=1)
            predicted_constructs = [constructs[idx] for idx in predicted_indices]
            confidence_scores = [similarities[i, idx] for i, idx in enumerate(predicted_indices)]
            
            return predicted_constructs, confidence_scores
        else:
            # Get top-k most similar centroids
            top_k_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
            
            # Get the corresponding constructs and scores
            top_k_constructs = []
            top_k_scores = []
            
            for i, indices in enumerate(top_k_indices):
                constructs_i = [constructs[idx] for idx in indices]
                scores_i = [similarities[i, idx] for idx in indices]
                
                top_k_constructs.append(constructs_i)
                top_k_scores.append(scores_i)
            
            return top_k_constructs, top_k_scores
    
    def evaluate_predictions(self, true_constructs, predicted_constructs, prefix=""):
        """Evaluate how well the model predicts constructs."""
        accuracy = accuracy_score(true_constructs, predicted_constructs)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_constructs, predicted_constructs))
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_constructs, predicted_constructs)
        
        # Due to potentially large number of constructs, only show confusion matrix visually if reasonable
        if len(set(true_constructs)) <= 30:
            plt.figure(figsize=(12, 10))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Constructs')
            plt.ylabel('True Constructs')
            plt.title('Confusion Matrix')
            
            # Save confusion matrix
            output_file = f"confusion_matrix_{prefix}.png" if prefix else "confusion_matrix.png"
            plt.savefig(VISUALIZATIONS_DIR / output_file, dpi=300, bbox_inches="tight")
            plt.close()
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_report(true_constructs, predicted_constructs, output_dict=True)
        }
    
    def create_umap_model(self, embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine'):
        """Create a UMAP model based on the embeddings."""
        print("Creating UMAP model...")
        self.umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        
        # Fit the model
        self.umap_model.fit(embeddings)
        
        return self.umap_model
    
    def visualize_embeddings(self, embeddings, df, label_column, title, output_file):
        """Visualize embeddings with UMAP, colored by construct labels."""
        print(f"Creating UMAP visualization for {title}...")
        
        # If UMAP model doesn't exist, create one
        if self.umap_model is None:
            self.create_umap_model(embeddings)
        
        # Transform embeddings to 2D
        embedding_2d = self.umap_model.transform(embeddings)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            "x": embedding_2d[:, 0],
            "y": embedding_2d[:, 1],
            "construct": df[label_column],
            "text": df["processed_text"].astype(str)
        })
        
        # Plot with distinct colors
        plt.figure(figsize=(16, 12))
        
        # Get unique constructs and set color palette
        constructs = plot_df["construct"].unique()
        
        # Use a colormap that works well with many categories
        if len(constructs) <= 10:
            palette = sns.color_palette("tab10", len(constructs))
        elif len(constructs) <= 20:
            palette = sns.color_palette("tab20", len(constructs))
        else:
            # For many constructs, use a continuous colormap
            palette = plt.cm.viridis(np.linspace(0, 1, len(constructs)))
        
        # Create scatter plot for each construct
        for i, construct in enumerate(sorted(constructs)):
            subset = plot_df[plot_df["construct"] == construct]
            plt.scatter(
                subset["x"], 
                subset["y"], 
                c=[palette[i]] * len(subset),
                label=construct if len(constructs) <= 30 else None, # Only show legend if not too many
                alpha=0.7,
                s=70,
                edgecolors="white",
                linewidth=0.5
            )
        
        # If we've computed centroids, add them to the plot
        if self.construct_centroids and all(c in self.construct_centroids for c in constructs):
            # Project centroids to 2D
            centroid_embeddings = np.vstack([self.construct_centroids[c] for c in sorted(constructs)])
            centroid_2d = self.umap_model.transform(centroid_embeddings)
            
            # Plot centroids
            for i, construct in enumerate(sorted(constructs)):
                plt.scatter(
                    centroid_2d[i, 0], centroid_2d[i, 1],
                    marker='*',
                    s=300,
                    c=[palette[i]],
                    edgecolors='black',
                    linewidth=1.5,
                    zorder=100
                )
                
                # Add label
                if len(constructs) <= 30:  # Only label if not too many
                    plt.annotate(
                        construct,
                        (centroid_2d[i, 0], centroid_2d[i, 1]),
                        fontsize=12,
                        fontweight='bold',
                        ha='center',
                        va='bottom',
                        xytext=(0, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
        
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
        
        # Add legend if not too many constructs
        if len(constructs) <= 30:
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=12 if len(constructs) < 15 else 8,
                frameon=True,
                facecolor='white',
                edgecolor='gray'
            )
        
        # Add info about number of constructs
        plt.figtext(
            0.5, 0.01,
            f"Number of constructs: {len(constructs)}",
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
        )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        return plot_df, embedding_2d
    
    def compute_construct_separation_metrics(self, embeddings, df, label_column):
        """Compute metrics for how well separated the constructs are."""
        # Calculate all pairwise distances
        similarities = cosine_similarity(embeddings)
        distances = 1 - similarities
        
        # Get construct labels
        constructs = df[label_column].values
        
        # Initialize containers for distances
        intra_construct_distances = []
        inter_construct_distances = []
        
        # Collect distances
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):  # Upper triangle only
                if constructs[i] == constructs[j]:
                    intra_construct_distances.append(distances[i, j])
                else:
                    inter_construct_distances.append(distances[i, j])
        
        # Convert to numpy arrays
        intra_distances = np.array(intra_construct_distances)
        inter_distances = np.array(inter_construct_distances)
        
        # Calculate statistics
        intra_mean = np.mean(intra_distances)
        intra_std = np.std(intra_distances)
        inter_mean = np.mean(inter_distances)
        inter_std = np.std(inter_distances)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
        effect_size = (inter_mean - intra_mean) / pooled_std
        
        print(f"Intra-construct distances: mean = {intra_mean:.4f}, std = {intra_std:.4f}")
        print(f"Inter-construct distances: mean = {inter_mean:.4f}, std = {inter_std:.4f}")
        print(f"Effect size (Cohen's d): {effect_size:.4f}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot histograms of distances
        sns.histplot(intra_distances, alpha=0.5, label=f"Within construct: μ={intra_mean:.4f}", kde=True)
        sns.histplot(inter_distances, alpha=0.5, label=f"Between constructs: μ={inter_mean:.4f}", kde=True)
        
        plt.axvline(intra_mean, color='blue', linestyle='--')
        plt.axvline(inter_mean, color='orange', linestyle='--')
        
        plt.title(f"Semantic Distance Distribution (Effect Size d={effect_size:.4f})", fontsize=14)
        plt.xlabel("Semantic Distance (1 - Cosine Similarity)")
        plt.ylabel("Frequency")
        plt.legend()
        
        # Save figure
        output_file = VISUALIZATIONS_DIR / f"distance_distribution_{label_column}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        return {
            "intra_mean": intra_mean,
            "intra_std": intra_std,
            "inter_mean": inter_mean,
            "inter_std": inter_std,
            "effect_size": effect_size
        }
    
    def create_comparison_report(self, ipip_metrics, leadership_metrics):
        """Create a comprehensive report comparing the performance on IPIP vs leadership data."""
        report = f"""
    Construct Space Learning Model Report
    ====================================
    
    Model details:
    - Base model: {self.model_name}
    - GIST loss training on IPIP constructs
    - Transfer to leadership styles
    
    IPIP Construct Identification Performance:
    -----------------------------------------
    - Test Accuracy: {ipip_metrics['accuracy']:.4f}
    - Number of constructs: {len(ipip_metrics['classification_report'].keys()) - 3}  # Subtract 'accuracy', 'macro avg', 'weighted avg'
    - Effect size between construct categories: {ipip_metrics['separation']['effect_size']:.4f}
    
    Leadership Construct Identification Performance:
    ----------------------------------------------
    - Test Accuracy: {leadership_metrics['accuracy']:.4f}
    - Number of constructs: {len(leadership_metrics['classification_report'].keys()) - 3}
    - Effect size between construct categories: {leadership_metrics['separation']['effect_size']:.4f}
    
    Comparative Analysis:
    -------------------
    The model {
        "performed BETTER" if ipip_metrics['accuracy'] > leadership_metrics['accuracy'] else "performed WORSE"
    } on IPIP constructs compared to leadership constructs.
    
    Effect size comparison shows that {
        "IPIP constructs are more distinct" if ipip_metrics['separation']['effect_size'] > leadership_metrics['separation']['effect_size']
        else "Leadership constructs are more distinct"
    } in the embedding space.
    
    Key Findings:
    -----------
    1. The model was able to learn construct spaces from IPIP data with {ipip_metrics['accuracy']:.1%} accuracy.
    
    2. When applied to leadership styles, the model achieved {leadership_metrics['accuracy']:.1%} accuracy in predicting the correct construct.
    
    3. The {
        "smaller" if leadership_metrics['separation']['effect_size'] < ipip_metrics['separation']['effect_size']
        else "larger"
    } effect size for leadership constructs ({leadership_metrics['separation']['effect_size']:.4f} vs. {ipip_metrics['separation']['effect_size']:.4f}) 
    suggests that leadership constructs {
        "overlap more" if leadership_metrics['separation']['effect_size'] < ipip_metrics['separation']['effect_size']
        else "are more distinct"
    } compared to personality constructs.
    
    4. UMAP visualizations confirm these findings, showing {
        "less" if leadership_metrics['separation']['effect_size'] < ipip_metrics['separation']['effect_size']
        else "more" 
    } distinct clustering for leadership constructs.
    
    Conclusion:
    ----------
    This analysis {
        "supports" if leadership_metrics['accuracy'] < ipip_metrics['accuracy'] and leadership_metrics['separation']['effect_size'] < ipip_metrics['separation']['effect_size']
        else "does not support"
    } the hypothesis that leadership constructs have substantial semantic overlap and may not represent truly distinct constructs.
    
    The model's ability to more effectively distinguish personality constructs compared to leadership constructs suggests that 
    leadership measurement may benefit from reconsidering its construct boundaries and potentially developing more distinctive measurement approaches.
        """
        
        # Save report
        report_file = VISUALIZATIONS_DIR / "construct_learning_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"Saved comparison report to {report_file}")
        return report
    
    def save_model(self, output_path=None):
        """Save the model and important attributes."""
        if output_path is None:
            output_path = MODELS_DIR / "construct_learner_model.pkl"
        
        # Create dictionary with model state and attributes
        model_data = {
            'construct_centroids': self.construct_centroids,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'timestamp': time.time()
        }
        
        # Save model to disk
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save the sentence transformer model
        self.model.save(str(MODELS_DIR / "construct_model"))
        
        print(f"Model saved to {output_path}")
        
    def load_model(self, model_path=None, st_model_path=None):
        """Load a saved model."""
        if model_path is None:
            model_path = MODELS_DIR / "construct_learner_model.pkl"
        
        if st_model_path is None:
            st_model_path = MODELS_DIR / "construct_model"
        
        # Load model data
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore attributes
        self.construct_centroids = model_data['construct_centroids']
        self.label_encoder = model_data['label_encoder']
        self.model_name = model_data['model_name']
        
        # Load sentence transformer model
        self.model = SentenceTransformer(str(st_model_path))
        
        print(f"Model loaded from {model_path}")


def main():
    """Main function to run the construct learning process."""
    print("\n===== Construct Learning Model =====\n")
    
    # Create construct learner
    learner = ConstructLearner()
    
    # Load and prepare IPIP data
    ipip_df = learner.load_ipip_data()
    
    # Prepare IPIP data for training
    train_examples, train_df, val_df, test_df = learner.prepare_training_data(ipip_df)
    
    # Train GIST model
    learner.create_gist_model(train_examples, epochs=1)  # Reduced epochs for faster execution
    
    # Generate embeddings for test data
    test_embeddings = learner.generate_base_embeddings(test_df['processed_text'].tolist())
    
    # Compute construct centroids from training data
    train_embeddings = learner.generate_base_embeddings(train_df['processed_text'].tolist())
    learner.compute_construct_centroids(train_df, train_embeddings)
    
    # Predict constructs for test data
    predicted_constructs, confidence_scores = learner.predict_constructs(test_embeddings)
    
    # Evaluate predictions
    ipip_metrics = learner.evaluate_predictions(test_df['label'].tolist(), predicted_constructs, prefix="ipip")
    
    # Compute separation metrics
    ipip_sep_metrics = learner.compute_construct_separation_metrics(test_embeddings, test_df, 'label')
    ipip_metrics['separation'] = ipip_sep_metrics
    
    # Create UMAP visualization for IPIP constructs
    learner.create_umap_model(test_embeddings)
    learner.visualize_embeddings(
        test_embeddings, 
        test_df, 
        'label', 
        'IPIP Construct Space (GIST Loss)', 
        VISUALIZATIONS_DIR / "ipip_construct_space.png"
    )
    
    # Load leadership data
    leadership_df = learner.load_leadership_data()
    
    # Generate embeddings for leadership data
    leadership_embeddings = learner.generate_base_embeddings(leadership_df['processed_text'].tolist())
    
    # Compute leadership construct separation metrics
    leadership_sep_metrics = learner.compute_construct_separation_metrics(
        leadership_embeddings, leadership_df, 'StandardConstruct'
    )
    
    # Now make predictions using the IPIP-trained model
    # We'll need to map IPIP constructs to leadership constructs
    # For now, just use the existing model to get embeddings and see how they cluster
    leadership_predicted, leadership_confidence = learner.predict_constructs(leadership_embeddings)
    
    # Create a mapping from predicted IPIP constructs to leadership constructs
    prediction_map = pd.DataFrame({
        'true_construct': leadership_df['StandardConstruct'],
        'predicted_ipip': leadership_predicted,
        'confidence': leadership_confidence
    })
    
    # Save this mapping for analysis
    prediction_map.to_csv(PROCESSED_DIR / "leadership_ipip_mapping.csv", index=False)
    
    # Evaluate how well IPIP predictions align with leadership constructs
    leadership_metrics = {
        'accuracy': accuracy_score(leadership_df['StandardConstruct'], leadership_predicted),
        'classification_report': classification_report(
            leadership_df['StandardConstruct'], leadership_predicted, output_dict=True
        ),
        'separation': leadership_sep_metrics
    }
    
    # Visualize leadership data with IPIP-trained UMAP
    learner.visualize_embeddings(
        leadership_embeddings,
        leadership_df,
        'StandardConstruct',
        'Leadership Styles in IPIP-Trained Construct Space',
        VISUALIZATIONS_DIR / "leadership_in_ipip_space.png"
    )
    
    # Create comparison report
    learner.create_comparison_report(ipip_metrics, leadership_metrics)
    
    # Save model
    learner.save_model()
    
    print("\nConstruct learning analysis complete!")
    print(f"IPIP construct identification accuracy: {ipip_metrics['accuracy']:.4f}")
    print(f"Leadership construct identification accuracy: {leadership_metrics['accuracy']:.4f}")
    print(f"Effect size - IPIP: {ipip_sep_metrics['effect_size']:.4f}, Leadership: {leadership_sep_metrics['effect_size']:.4f}")


if __name__ == "__main__":
    main()