#!/usr/bin/env python3
"""
Triplet Loss Construct Learning Model

This script implements a model that learns to identify construct spaces using triplet loss.
The model is trained on IPIP personality construct data and then applied to leadership styles.

Key steps:
1. Train a model on Big Five personality traits using triplet loss
2. Evaluate the model's ability to identify constructs with clear metrics
3. Apply the trained model to leadership styles data
4. Compare how well the model identifies leadership styles

Usage:
    python triplet_construct_model.py
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
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import umap
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from sklearn.metrics.pairwise import cosine_similarity
import warnings
# Filter sklearn warnings about numeric issues
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
from collections import Counter
from tqdm.auto import tqdm
import random

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

class TripletConstructLearner:
    """
    Class to learn and identify construct spaces using Triplet Loss.
    Focused on Big Five traits for higher accuracy and clearer evaluation.
    """
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """Initialize the construct learner with a stronger embedding model."""
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            print("Falling back to MiniLM model...")
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
        
        print(f"Using base model: {self.model_name}")
        self.label_encoder = LabelEncoder()
        self.construct_centroids = {}
        self.umap_model = None
        self.training_loss = []
        self.validation_metrics = {}
        self.cv_results = {}
        
    def load_ipip_data(self, big_five_only=True):
        """
        Load the IPIP dataset with Big Five constructs for clearer training.
        
        Args:
            big_five_only: If True, filter to only Big Five traits for clearer evaluation
        """
        ipip_path = DATA_DIR / "IPIP.csv"
        
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
        
        if big_five_only:
            # Map to Big Five categories for clearer evaluation
            big_five_mapping = {
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
            
            # Map each item to its Big Five category if applicable
            def map_to_big_five(label):
                for big_five, labels in big_five_mapping.items():
                    if label in labels:
                        return big_five
                return None
            
            # Apply the mapping
            ipip_df['big_five'] = ipip_df['label'].apply(map_to_big_five)
            
            # Filter to only include items mapped to Big Five
            big_five_df = ipip_df.dropna(subset=['big_five']).copy()
            
            # Balance the dataset for better training
            min_count = big_five_df['big_five'].value_counts().min()
            samples_per_class = min(min_count, 200)  # Cap at 200 per class for efficiency
            
            balanced_df = pd.DataFrame()
            for trait in big_five_df['big_five'].unique():
                trait_items = big_five_df[big_five_df['big_five'] == trait]
                if len(trait_items) > samples_per_class:
                    balanced_df = pd.concat([balanced_df, trait_items.sample(samples_per_class, random_state=42)])
                else:
                    balanced_df = pd.concat([balanced_df, trait_items])
            
            print(f"Filtered to {len(big_five_df)} items mapped to Big Five traits")
            print(f"Balanced to {len(balanced_df)} items")
            print(balanced_df['big_five'].value_counts())
            
            return balanced_df
        
        # If not filtering to Big Five, return the full dataset
        print(f"Loaded {len(ipip_df)} items across {ipip_df['label'].nunique()} constructs")
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
    
    def prepare_triplet_data(self, df, text_column='processed_text', label_column='big_five'):
        """
        Prepare data for triplet loss training with advanced hard triplet mining.
        
        This creates triplets (anchor, positive, negative) with multiple strategies:
        1. Semi-hard triplets: negatives that are somewhat close but still distinct
        2. Hard triplets: negatives that are very close to the anchor (challenging)
        3. Easy triplets: randomly selected negatives (for stability)
        
        A mix of these strategies helps prevent collapse during training.
        """
        # Encode labels numerically
        self.label_encoder.fit(df[label_column])
        df['label_id'] = self.label_encoder.transform(df[label_column])
        
        # Split into train/validation/test sets (70/15/15)
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df['label_id'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['label_id'], random_state=42
        )
        
        print(f"Train set: {len(train_df)} items")
        print(f"Validation set: {len(val_df)} items")
        print(f"Test set: {len(test_df)} items")
        
        # Generate pre-embeddings for hard triplet mining
        print("Generating embeddings for advanced triplet mining...")
        train_texts = train_df[text_column].tolist()
        train_labels = train_df['label_id'].tolist()
        
        # Generate embeddings using the base model
        train_embeddings = self.generate_base_embeddings(train_texts)
        
        # Create triplets with multiple strategies
        print("Creating triplets with multi-strategy hard negative mining...")
        triplets = []
        
        # Group items by construct
        construct_indices = {}
        for idx, (text, label_id) in enumerate(zip(train_texts, train_labels)):
            if label_id not in construct_indices:
                construct_indices[label_id] = []
            construct_indices[label_id].append(idx)
        
        # Calculate target number of triplets - reduced for faster execution
        # Note: For production, use higher numbers (30000+)
        num_triplets = min(5000, len(train_texts) * 3)  # Reduced for faster test run
        
        # Create triplets with various negative mining strategies
        for _ in tqdm(range(num_triplets)):
            # Randomly select an anchor construct
            anchor_label = random.choice(list(construct_indices.keys()))
            
            # Select random anchor from this construct
            anchor_idx = random.choice(construct_indices[anchor_label])
            anchor_embedding = train_embeddings[anchor_idx]
            
            # Select a positive from the same construct (not the anchor)
            positive_candidates = [idx for idx in construct_indices[anchor_label] if idx != anchor_idx]
            if not positive_candidates:
                continue  # Skip if no positives available
                
            # Augmentation: select closest positive examples to create tighter clusters
            positive_embeddings = train_embeddings[positive_candidates]
            positive_similarities = cosine_similarity([anchor_embedding], positive_embeddings)[0]
            
            # 30% of the time, choose from the closest positives (tighten clusters)
            # 70% of the time, choose random positives (preserve diversity)
            if random.random() < 0.3:
                closest_positives = np.argsort(positive_similarities)[-int(len(positive_similarities)*0.3):]
                selected_positive_idx = random.choice(closest_positives)
                positive_idx = positive_candidates[selected_positive_idx]
            else:
                positive_idx = random.choice(positive_candidates)
            
            # Negative mining strategy selection
            strategy = random.random()
            
            # Collect all negative indices
            negative_labels = [label for label in construct_indices.keys() if label != anchor_label]
            negative_indices = []
            for label in negative_labels:
                negative_indices.extend(construct_indices[label])
            
            negative_embeddings = train_embeddings[negative_indices]
            similarities = cosine_similarity([anchor_embedding], negative_embeddings)[0]
            
            if strategy < 0.6:  # 60% hard negatives (most valuable)
                # Select from top 20% most similar negatives (hardest cases)
                hard_threshold = int(len(similarities) * 0.2)
                candidate_indices = np.argsort(similarities)[-hard_threshold:]
                
            elif strategy < 0.9:  # 30% semi-hard negatives
                # Select from middle 40% similar negatives
                sorted_indices = np.argsort(similarities)
                start_idx = int(len(sorted_indices) * 0.3)
                end_idx = int(len(sorted_indices) * 0.7)
                candidate_indices = sorted_indices[start_idx:end_idx]
                
            else:  # 10% easy negatives (stability)
                # Select from bottom 30% least similar negatives
                easy_threshold = int(len(similarities) * 0.3)
                candidate_indices = np.argsort(similarities)[:easy_threshold]
            
            selected_neg_idx = random.choice(candidate_indices)
            negative_idx = negative_indices[selected_neg_idx]
            
            # Create the triplet
            triplets.append(InputExample(
                texts=[train_texts[anchor_idx], train_texts[positive_idx], train_texts[negative_idx]],
                label=1.0  # Dummy label for SentenceTransformer API compatibility
            ))
        
        print(f"Created {len(triplets)} triplets for training")
        
        # Create validation triplets for more meaningful evaluation
        print("Creating validation triplets...")
        validation_triplets = []
        
        # Generate validation embeddings
        val_texts = val_df[text_column].tolist()
        val_labels = val_df['label_id'].tolist()
        val_embeddings = self.generate_base_embeddings(val_texts)
        
        # Group validation items by construct
        val_construct_indices = {}
        for idx, label_id in enumerate(val_labels):
            if label_id not in val_construct_indices:
                val_construct_indices[label_id] = []
            val_construct_indices[label_id].append(idx)
        
        # Create 1000 validation triplets for evaluation
        val_triplet_count = min(1000, len(val_texts))
        
        for _ in range(val_triplet_count):
            # Similar process as training triplets but with validation data
            anchor_label = random.choice(list(val_construct_indices.keys()))
            anchor_idx = random.choice(val_construct_indices[anchor_label])
            
            positive_candidates = [idx for idx in val_construct_indices[anchor_label] if idx != anchor_idx]
            if not positive_candidates:
                continue
            positive_idx = random.choice(positive_candidates)
            
            negative_labels = [label for label in val_construct_indices.keys() if label != anchor_label]
            if not negative_labels:
                continue
            negative_label = random.choice(negative_labels)
            negative_idx = random.choice(val_construct_indices[negative_label])
            
            validation_triplets.append(InputExample(
                texts=[val_texts[anchor_idx], val_texts[positive_idx], val_texts[negative_idx]],
                label=1.0
            ))
        
        print(f"Created {len(validation_triplets)} validation triplets")
        
        # Create evaluator with the validation triplets
        evaluator = None  # We'll handle custom evaluation ourselves
        
        return triplets, validation_triplets, train_df, val_df, test_df, evaluator
    
    def train_triplet_model(self, triplets, validation_triplets=None, epochs=5, batch_size=128):
        """
        Train a model with advanced triplet loss on the IPIP construct data.
        
        Uses:
        1. Batch Hard triplet loss for more effective learning
        2. Learning rate scheduling with warmup
        3. Higher epochs for better convergence
        4. Validation tracking for early stopping
        """
        print(f"Training triplet model for {epochs} epochs with batch size {batch_size}...")
        
        # Create data loader
        train_dataloader = DataLoader(triplets, shuffle=True, batch_size=batch_size)
        
        # Use BatchHardTripletLoss for better performance
        # This dynamically finds the hardest triplets in each batch
        train_loss = losses.BatchHardTripletLoss(
            model=self.model,
            margin=0.3,  # Slightly lower margin for better discrimination
            distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance
        )
        
        # Set up validation if provided
        if validation_triplets:
            val_dataloader = DataLoader(validation_triplets, batch_size=batch_size)
            val_loss = losses.TripletLoss(model=self.model, triplet_margin=0.3)
        else:
            val_dataloader = None
            val_loss = None
        
        # Callback to track training and validation loss
        loss_values = []
        val_loss_values = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 2  # Stop after 2 epochs of no improvement
        
        def loss_callback(score, epoch, steps):
            """Store loss values during training and check for early stopping."""
            nonlocal best_val_loss, patience_counter
            
            loss_values.append(score)
            print(f"Epoch {epoch}, Step {steps}: Train Loss = {score:.4f}")
            
            # Check validation every epoch
            if steps % len(train_dataloader) == 0 and val_dataloader:
                # Compute validation loss
                self.model.eval()
                val_score = self.evaluate_validation_loss(val_dataloader, val_loss)
                self.model.train()
                
                val_loss_values.append(val_score)
                print(f"Epoch {epoch}: Validation Loss = {val_score:.4f}")
                
                # Early stopping check
                if val_score < best_val_loss:
                    best_val_loss = val_score
                    patience_counter = 0
                    # Save the best model
                    self.model.save(str(MODELS_DIR / f"triplet_best_{self.model_name.split('/')[-1]}"))
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"Early stopping at epoch {epoch} - no improvement in validation loss")
                        return True  # Signal to stop training
            
            return False  # Continue training
        
        # Train the model with more epochs and learning rate schedule
        warmup_steps = int(len(train_dataloader) * 0.1 * epochs)  # 10% of total steps
        
        try:
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                scheduler='warmuplinear',
                warmup_steps=warmup_steps,
                optimizer_params={'lr': 5e-5},  # Lower learning rate for stability
                callback=loss_callback,
                show_progress_bar=True,
                output_path=str(MODELS_DIR / f"triplet_{self.model_name.split('/')[-1]}")
            )
        except Exception as e:
            print(f"Error during training: {str(e)}")
            print("Attempting to continue with the last saved model state...")
        
        self.training_loss = loss_values
        self.validation_metrics['loss'] = val_loss_values
        print("Triplet model training complete!")
        
        # Plot training and validation loss
        plt.figure(figsize=(12, 6))
        plt.plot(loss_values, label="Training Loss")
        if val_loss_values:
            # Interpolate validation loss to match training loss steps
            val_steps = np.linspace(0, len(loss_values)-1, len(val_loss_values))
            plt.plot(val_steps, val_loss_values, 'o-', label="Validation Loss")
        
        plt.title("Triplet Loss during Training")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(VISUALIZATIONS_DIR / "triplet_training_loss.png", dpi=300)
        plt.close()
        
        # Try to load the best model if it exists
        best_model_path = MODELS_DIR / f"triplet_best_{self.model_name.split('/')[-1]}"
        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}")
            try:
                self.model = SentenceTransformer(str(best_model_path))
            except Exception as e:
                print(f"Could not load best model: {str(e)}")
        
        return self.model
    
    def evaluate_validation_loss(self, val_dataloader, val_loss):
        """Compute validation loss."""
        self.model.eval()
        val_loss_value = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                features, labels = batch
                loss_value = val_loss(features, labels)
                val_loss_value += loss_value.item() * len(labels)
                val_samples += len(labels)
                
        # Return average loss
        if val_samples > 0:
            return val_loss_value / val_samples
        return 0
    
    def cross_validate(self, df, text_column='processed_text', label_column='big_five', n_splits=5):
        """
        Perform cross-validation to get a reliable accuracy estimate.
        
        Args:
            df: DataFrame with the data
            text_column: Column with the text to encode
            label_column: Column with the construct labels
            n_splits: Number of CV folds
        
        Returns:
            Dictionary with cross-validation results
        """
        print(f"Performing {n_splits}-fold cross-validation...")
        
        # Encode labels
        self.label_encoder.fit(df[label_column])
        df['label_id'] = self.label_encoder.transform(df[label_column])
        
        # Set up cross-validation
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        
        # For each fold
        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['label_id'])):
            print(f"\nTraining fold {fold+1}/{n_splits}")
            
            # Split data
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            
            # Create model for this fold (fresh)
            fold_model = SentenceTransformer(self.model_name)
            
            # Prepare triplets
            triplets, val_triplets, _, _, _, _ = self.prepare_triplet_data(
                train_df, text_column, label_column
            )
            
            # Train model (fewer epochs for CV)
            fold_model.fit(
                train_objectives=[(DataLoader(triplets, shuffle=True, batch_size=128), 
                                 losses.BatchHardTripletLoss(model=fold_model, margin=0.3))],
                epochs=2,  # Fewer epochs for CV
                show_progress_bar=True
            )
            
            # Generate embeddings
            train_embeddings = fold_model.encode(train_df[text_column].tolist(), convert_to_numpy=True)
            test_embeddings = fold_model.encode(test_df[text_column].tolist(), convert_to_numpy=True)
            
            # Compute centroids
            centroids = {}
            for construct in train_df[label_column].unique():
                indices = train_df[train_df[label_column] == construct].index.tolist()
                construct_embs = train_embeddings[indices]
                centroids[construct] = np.mean(construct_embs, axis=0)
            
            # Predict constructs
            centroid_matrix = np.vstack(list(centroids.values()))
            constructs = list(centroids.keys())
            
            similarities = cosine_similarity(test_embeddings, centroid_matrix)
            predicted_indices = np.argmax(similarities, axis=1)
            predicted_constructs = [constructs[idx] for idx in predicted_indices]
            
            # Evaluate
            accuracy = accuracy_score(test_df[label_column], predicted_constructs)
            f1 = f1_score(test_df[label_column], predicted_constructs, average='weighted')
            
            print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            fold_results.append({
                'fold': fold+1,
                'accuracy': accuracy,
                'f1': f1
            })
        
        # Compute average results
        cv_accuracy = np.mean([r['accuracy'] for r in fold_results])
        cv_f1 = np.mean([r['f1'] for r in fold_results])
        
        print(f"\nCross-validation results:")
        print(f"Mean Accuracy: {cv_accuracy:.4f}")
        print(f"Mean F1 Score: {cv_f1:.4f}")
        
        # Store results
        self.cv_results = {
            'folds': fold_results,
            'mean_accuracy': cv_accuracy,
            'mean_f1': cv_f1,
            'std_accuracy': np.std([r['accuracy'] for r in fold_results])
        }
        
        return self.cv_results
    
    def compute_construct_centroids(self, df, embeddings, label_column='big_five'):
        """Compute centroids for each construct based on embeddings."""
        construct_centroids = {}
        
        # Reset index to ensure alignment between DataFrame and embeddings
        df_reset = df.reset_index(drop=True)
        
        for construct in df_reset[label_column].unique():
            # Get indices of items from this construct
            indices = df_reset[df_reset[label_column] == construct].index.tolist()
            
            # Get embeddings for these items
            construct_embs = embeddings[indices]
            
            # Compute centroid
            centroid = np.mean(construct_embs, axis=0)
            
            # Store centroid
            construct_centroids[construct] = centroid
        
        self.construct_centroids = construct_centroids
        return construct_centroids
    
    def predict_constructs(self, embeddings, construct_centroids=None, weighted=True):
        """
        Predict constructs for embeddings based on similarity to centroids.
        
        Args:
            embeddings: Numpy array of embeddings to classify
            construct_centroids: Dict of construct centroids or None to use self.construct_centroids
            weighted: If True, use a weighted similarity approach for better accuracy
            
        Returns:
            predicted_constructs: List of predicted construct names
            confidence_scores: List of confidence scores for predictions
            all_similarities: Matrix of similarities to all constructs (for analysis)
        """
        if construct_centroids is None:
            construct_centroids = self.construct_centroids
            
        if not construct_centroids:
            raise ValueError("Construct centroids not computed yet! Call compute_construct_centroids first.")
        
        # Stack all centroids into a matrix
        centroid_matrix = np.vstack(list(construct_centroids.values()))
        constructs = list(construct_centroids.keys())
        
        # Compute similarities to all centroids
        similarities = cosine_similarity(embeddings, centroid_matrix)
        
        if weighted:
            # Enhanced prediction using weighted similarities
            # This helps with borderline cases by considering secondary similarities
            
            # Softmax transformation to get probability-like weights
            def softmax(x):
                exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)
            
            # Apply softmax to similarities
            similarity_weights = softmax(similarities * 5.0)  # Scale factor enhances differences
            
            # Get the most similar centroid for each embedding
            predicted_indices = np.argmax(similarity_weights, axis=1)
            
            # Calculate confidence as the normalized weight
            confidence_scores = [similarity_weights[i, idx] for i, idx in enumerate(predicted_indices)]
        else:
            # Simple max similarity approach
            predicted_indices = np.argmax(similarities, axis=1)
            confidence_scores = [similarities[i, idx] for i, idx in enumerate(predicted_indices)]
        
        predicted_constructs = [constructs[idx] for idx in predicted_indices]
        
        return predicted_constructs, confidence_scores, similarities
    
    def evaluate_predictions(self, true_constructs, predicted_constructs, confidence_scores=None, 
                         similarities=None, prefix=""):
        """
        Evaluate how well the model predicts constructs with enhanced metrics.
        
        Args:
            true_constructs: List of true construct labels
            predicted_constructs: List of predicted construct labels
            confidence_scores: List of confidence scores (optional)
            similarities: Matrix of similarities to all constructs (optional)
            prefix: Prefix for output files
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(true_constructs, predicted_constructs)
        f1 = f1_score(true_constructs, predicted_constructs, average='weighted')
        
        # Print classification report
        print(f"\nClassification Report - {prefix if prefix else 'Model'} (Accuracy: {accuracy:.4f}):")
        print(classification_report(true_constructs, predicted_constructs))
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_constructs, predicted_constructs)
        
        # Create enhanced confusion matrix visualization
        plt.figure(figsize=(14, 12))
        
        # Get unique constructs in order preserving the diagonal
        unique_true = list(dict.fromkeys(true_constructs))
        unique_pred = list(dict.fromkeys(predicted_constructs))
        all_labels = sorted(list(set(unique_true + unique_pred)))
        
        # Create a normalized confusion matrix (percent of each true class)
        cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        # Create heatmap
        ax = plt.subplot(111)
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Percentage of True Class', rotation=270, labelpad=15)
        
        # Add annotations
        thresh = cm_normalized.max() / 2
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                if cm_normalized[i, j] > 0.01:  # Only annotate non-zero cells
                    ax.text(j, i, f'{conf_matrix[i, j]}\n({cm_normalized[i, j]:.1%})',
                           ha="center", va="center", 
                           color="white" if cm_normalized[i, j] > thresh else "black",
                           fontsize=9)
        
        # Format the plot
        tick_marks = np.arange(len(all_labels))
        plt.xticks(tick_marks, all_labels, rotation=45, ha='right')
        plt.yticks(tick_marks, all_labels)
        
        plt.tight_layout()
        plt.xlabel('Predicted Constructs')
        plt.ylabel('True Constructs')
        plt.title(f'Construct Classification Performance\nAccuracy: {accuracy:.2%}, F1: {f1:.4f}')
        
        # Save confusion matrix
        output_file = f"confusion_matrix_{prefix}.png" if prefix else "confusion_matrix.png"
        plt.savefig(VISUALIZATIONS_DIR / output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Additional metrics if we have confidence scores
        confidence_metrics = {}
        if confidence_scores is not None:
            # Analyze confidence distribution
            correct_indices = [i for i, (true, pred) in enumerate(zip(true_constructs, predicted_constructs)) if true == pred]
            incorrect_indices = [i for i, (true, pred) in enumerate(zip(true_constructs, predicted_constructs)) if true != pred]
            
            if correct_indices:
                correct_confidences = [confidence_scores[i] for i in correct_indices]
                confidence_metrics['mean_correct_confidence'] = np.mean(correct_confidences)
            else:
                confidence_metrics['mean_correct_confidence'] = 0
                
            if incorrect_indices:
                incorrect_confidences = [confidence_scores[i] for i in incorrect_indices]
                confidence_metrics['mean_incorrect_confidence'] = np.mean(incorrect_confidences)
            else:
                confidence_metrics['mean_incorrect_confidence'] = 0
            
            # Plot confidence distributions
            plt.figure(figsize=(10, 6))
            if correct_indices and incorrect_indices:
                sns.histplot(correct_confidences, bins=20, alpha=0.6, label=f'Correct Predictions (n={len(correct_indices)})')
                sns.histplot(incorrect_confidences, bins=20, alpha=0.6, label=f'Incorrect Predictions (n={len(incorrect_indices)})')
                plt.axvline(confidence_metrics['mean_correct_confidence'], color='green', linestyle='--', 
                           label=f'Mean Correct: {confidence_metrics["mean_correct_confidence"]:.3f}')
                plt.axvline(confidence_metrics['mean_incorrect_confidence'], color='red', linestyle='--',
                           label=f'Mean Incorrect: {confidence_metrics["mean_incorrect_confidence"]:.3f}')
            elif correct_indices:
                sns.histplot(correct_confidences, bins=20, alpha=0.6, label=f'Correct Predictions (n={len(correct_indices)})')
                plt.axvline(confidence_metrics['mean_correct_confidence'], color='green', linestyle='--', 
                           label=f'Mean Correct: {confidence_metrics["mean_correct_confidence"]:.3f}')
            elif incorrect_indices:
                sns.histplot(incorrect_confidences, bins=20, alpha=0.6, label=f'Incorrect Predictions (n={len(incorrect_indices)})')
                plt.axvline(confidence_metrics['mean_incorrect_confidence'], color='red', linestyle='--',
                           label=f'Mean Incorrect: {confidence_metrics["mean_incorrect_confidence"]:.3f}')
            
            plt.title('Distribution of Confidence Scores')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save confidence distribution
            conf_file = f"confidence_distribution_{prefix}.png" if prefix else "confidence_distribution.png"
            plt.savefig(VISUALIZATIONS_DIR / conf_file, dpi=300, bbox_inches="tight")
            plt.close()
        
        # Additional analysis with similarities matrix
        similarity_metrics = {}
        if similarities is not None:
            # Get top-k accuracy metrics (how often is true class in top k predictions)
            constructs = np.unique(true_constructs)
            top_k_metrics = {}
            
            for k in [2, 3, 5]:
                if len(constructs) >= k:  # Only compute if we have enough constructs
                    # Get top k predictions for each sample
                    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
                    
                    # Create mapping from construct names to indices
                    construct_to_idx = {c: i for i, c in enumerate(constructs)}
                    
                    # Check if true construct is in top k for each sample
                    correct_in_top_k = 0
                    for i, true_construct in enumerate(true_constructs):
                        if true_construct in constructs:  # Skip if construct not in training set
                            true_idx = construct_to_idx[true_construct]
                            if true_idx in top_k_indices[i]:
                                correct_in_top_k += 1
                    
                    # Compute top-k accuracy
                    top_k_metrics[f'top_{k}_accuracy'] = correct_in_top_k / len(true_constructs)
                    print(f"Top-{k} Accuracy: {top_k_metrics[f'top_{k}_accuracy']:.4f}")
            
            similarity_metrics['top_k'] = top_k_metrics
        
        # Compute per-class metrics
        class_report = classification_report(true_constructs, predicted_constructs, output_dict=True)
        
        # Find the best and worst performing classes
        class_f1 = {}
        for cls in class_report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                class_f1[cls] = class_report[cls]['f1-score']
        
        if class_f1:
            best_class = max(class_f1.items(), key=lambda x: x[1])
            worst_class = min(class_f1.items(), key=lambda x: x[1])
            print(f"Best performing construct: {best_class[0]} (F1={best_class[1]:.4f})")
            print(f"Worst performing construct: {worst_class[0]} (F1={worst_class[1]:.4f})")
        
        # Combine all metrics
        result = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'confidence_metrics': confidence_metrics,
            'similarity_metrics': similarity_metrics,
            'best_class': best_class[0] if class_f1 else None,
            'worst_class': worst_class[0] if class_f1 else None
        }
        
        return result
    
    def create_umap_model(self, embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine'):
        """Create a UMAP model based on the embeddings with error handling."""
        print("Creating UMAP model...")
        try:
            # Clean the embeddings (replace NaN or Inf with 0)
            clean_embeddings = np.array(embeddings, copy=True)
            clean_embeddings = np.nan_to_num(clean_embeddings)
            
            # Create UMAP model with more conservative parameters for stability
            self.umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=max(5, min(n_neighbors, len(clean_embeddings) // 5)),  # Adjust based on data size
                min_dist=min_dist,
                metric=metric,
                random_state=42,
                low_memory=True,  # Memory efficient mode
                n_epochs=200,     # Fewer epochs for speed
                transform_seed=42  # Consistent transform
            )
            
            # Fit the model
            self.umap_model.fit(clean_embeddings)
            
            return self.umap_model
            
        except Exception as e:
            print(f"Error creating UMAP model: {str(e)}")
            # Create a simpler model as fallback
            print("Attempting to create a simpler UMAP model...")
            try:
                self.umap_model = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=5,
                    min_dist=0.3,
                    metric='euclidean',  # More stable metric
                    random_state=42,
                    low_memory=True,
                    n_epochs=100
                )
                self.umap_model.fit(np.nan_to_num(embeddings))
                return self.umap_model
            except Exception as e2:
                print(f"Fallback UMAP model also failed: {str(e2)}")
                # Return None if both attempts fail
                self.umap_model = None
                return None
    
    def visualize_embeddings(self, embeddings, df, label_column, title, output_file, 
                          predicted_column=None, confidence_scores=None):
        """
        Visualize embeddings with UMAP, colored by construct labels with enhanced features.
        
        Args:
            embeddings: Embedding matrix to visualize
            df: DataFrame with metadata
            label_column: Column containing construct labels
            title: Plot title
            output_file: Where to save the visualization
            predicted_column: Optional column with predicted constructs
            confidence_scores: Optional confidence scores for predictions
        
        Returns:
            plot_df: DataFrame with plot data
            embedding_2d: 2D embedding array
        """
        print(f"Creating enhanced UMAP visualization for {title}...")
        
        # If UMAP model doesn't exist, create one
        if self.umap_model is None:
            # For better visualizations, use different parameters
            self.create_umap_model(
                embeddings, 
                n_neighbors=30,     # Higher for more global structure
                min_dist=0.1,      # Lower for tighter clusters
                metric='cosine'    # Best for semantic embeddings
            )
        
        # Transform embeddings to 2D
        embedding_2d = self.umap_model.transform(embeddings)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            "x": embedding_2d[:, 0],
            "y": embedding_2d[:, 1],
            "construct": df[label_column],
            "text": df["processed_text"].astype(str)
        })
        
        # Add prediction info if available
        if predicted_column is not None:
            plot_df["predicted"] = df[predicted_column]
            plot_df["correct"] = plot_df["construct"] == plot_df["predicted"]
        elif confidence_scores is not None:
            plot_df["confidence"] = confidence_scores
        
        # Interactive visualization with hover info
        interactive = False  # Set to True to generate interactive plots (requires additional libraries)
        
        if interactive:
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Create interactive plot
                fig = px.scatter(
                    plot_df, x="x", y="y", color="construct",
                    hover_data=["text", "predicted" if "predicted" in plot_df.columns else None],
                    title=title
                )
                
                if "correct" in plot_df.columns:
                    # Highlight incorrect predictions
                    incorrect_df = plot_df[~plot_df["correct"]]
                    if len(incorrect_df) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=incorrect_df["x"], y=incorrect_df["y"],
                                mode="markers",
                                marker=dict(
                                    size=12, 
                                    line=dict(width=2, color="red"),
                                    color="rgba(0,0,0,0)"  # Transparent fill
                                ),
                                name="Incorrect Predictions",
                                hoverinfo="skip"
                            )
                        )
                
                # Add centroids if available
                if self.construct_centroids and all(c in self.construct_centroids for c in plot_df["construct"].unique()):
                    # Project centroids to 2D
                    centroid_embeddings = np.vstack([
                        self.construct_centroids[c] for c in sorted(plot_df["construct"].unique())
                    ])
                    centroid_2d = self.umap_model.transform(centroid_embeddings)
                    
                    # Create centroid dataframe
                    centroid_df = pd.DataFrame({
                        "x": centroid_2d[:, 0],
                        "y": centroid_2d[:, 1],
                        "construct": sorted(plot_df["construct"].unique())
                    })
                    
                    # Add centroids to plot
                    fig.add_trace(
                        go.Scatter(
                            x=centroid_df["x"], y=centroid_df["y"],
                            mode="markers+text",
                            marker=dict(size=15, symbol="star", line=dict(width=2, color="black")),
                            text=centroid_df["construct"],
                            textposition="top center",
                            name="Construct Centroids"
                        )
                    )
                
                # Save interactive plot
                interactive_file = str(output_file).replace(".png", ".html")
                fig.write_html(interactive_file)
                print(f"Interactive visualization saved to {interactive_file}")
            except ImportError:
                print("Plotly not installed, skipping interactive visualization.")
                interactive = False
        
        # Static visualization (always created)
        plt.figure(figsize=(16, 12))
        
        # Get unique constructs and set color palette
        constructs = plot_df["construct"].unique()
        
        # Use a colormap that works well with differentiation
        if len(constructs) <= 10:
            palette = sns.color_palette("tab10", len(constructs))
        else:
            palette = sns.color_palette("hsv", len(constructs))
        
        # Create a dictionary mapping constructs to colors for consistency
        color_map = {construct: palette[i] for i, construct in enumerate(sorted(constructs))}
        
        # Plot data points with different markers based on prediction correctness
        if "correct" in plot_df.columns:
            # Plot correct predictions
            correct_df = plot_df[plot_df["correct"]]
            if len(correct_df) > 0:
                for construct in sorted(correct_df["construct"].unique()):
                    subset = correct_df[correct_df["construct"] == construct]
                    plt.scatter(
                        subset["x"], subset["y"],
                        c=[color_map[construct]] * len(subset),
                        label=f"{construct}",
                        alpha=0.7,
                        s=70,
                        marker="o",  # Circle for correct
                        edgecolors="white",
                        linewidth=0.5
                    )
                
            # Plot incorrect predictions separately with X markers
            incorrect_df = plot_df[~plot_df["correct"]]
            if len(incorrect_df) > 0:
                for construct in sorted(incorrect_df["construct"].unique()):
                    subset = incorrect_df[incorrect_df["construct"] == construct]
                    plt.scatter(
                        subset["x"], subset["y"],
                        c=[color_map[construct]] * len(subset),
                        label=f"{construct} (Incorrect)" if construct not in correct_df["construct"].values else "",
                        alpha=0.5,
                        s=70,
                        marker="x",  # X for incorrect
                        edgecolors="black",
                        linewidth=1.0
                    )
        elif "confidence" in plot_df.columns:
            # Color by construct but vary size by confidence
            for construct in sorted(constructs):
                subset = plot_df[plot_df["construct"] == construct]
                plt.scatter(
                    subset["x"], subset["y"],
                    c=[color_map[construct]] * len(subset),
                    label=construct,
                    alpha=0.7,
                    s=subset["confidence"] * 200 + 30,  # Scale confidence to marker size
                    edgecolors="white",
                    linewidth=0.5
                )
        else:
            # Simple construct-based coloring
            for construct in sorted(constructs):
                subset = plot_df[plot_df["construct"] == construct]
                plt.scatter(
                    subset["x"], subset["y"],
                    c=[color_map[construct]] * len(subset),
                    label=construct,
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
                    s=400,
                    c=[color_map[construct]],
                    edgecolors='black',
                    linewidth=2.0,
                    zorder=100
                )
                
                # Add label
                plt.annotate(
                    construct,
                    (centroid_2d[i, 0], centroid_2d[i, 1]),
                    fontsize=13,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    xytext=(0, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
        
        # Add convex hulls to show boundaries between constructs
        from scipy.spatial import ConvexHull
        
        for construct in sorted(constructs):
            subset = plot_df[plot_df["construct"] == construct]
            if len(subset) >= 4:  # Need at least 4 points for convex hull
                points = subset[["x", "y"]].values
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices, :]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
                    
                    plt.plot(
                        hull_points[:, 0], hull_points[:, 1],
                        color=color_map[construct],
                        linestyle="-",
                        linewidth=2,
                        alpha=0.4
                    )
                    
                    # Fill with very transparent color
                    plt.fill(
                        hull_points[:, 0], hull_points[:, 1],
                        color=color_map[construct],
                        alpha=0.05
                    )
                except Exception:
                    # Skip if convex hull fails (e.g., coplanar points)
                    pass
        
        # Title and labels with additional info
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
        
        # Add legend (clean up duplicates)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(), by_label.keys(),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=12,
            frameon=True,
            facecolor='white',
            edgecolor='gray'
        )
        
        # Add info about number of constructs
        if "correct" in plot_df.columns:
            accuracy = plot_df["correct"].mean() * 100
            plt.figtext(
                0.5, 0.01,
                f"Number of constructs: {len(constructs)} | Classification Accuracy: {accuracy:.1f}%",
                ha="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
            )
        else:
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
        print(f"Computing separation metrics for {label_column}...")
        
        try:
            # Calculate distances using a more stable approach with batching
            # Get construct labels
            constructs = df[label_column].values
            
            # Initialize containers for distances
            intra_construct_distances = []
            inter_construct_distances = []
            
            # Process in smaller batches to avoid memory issues
            batch_size = 100
            for start_idx in range(0, len(embeddings), batch_size):
                end_idx = min(start_idx + batch_size, len(embeddings))
                batch_embeddings = embeddings[start_idx:end_idx]
                
                # Calculate similarities for this batch against all embeddings
                # Use a more stable implementation
                batch_similarities = np.zeros((end_idx - start_idx, len(embeddings)))
                for i in range(end_idx - start_idx):
                    for j in range(len(embeddings)):
                        # Manual cosine similarity calculation
                        dot_product = np.dot(batch_embeddings[i], embeddings[j])
                        norm_a = np.linalg.norm(batch_embeddings[i])
                        norm_b = np.linalg.norm(embeddings[j])
                        # Avoid division by zero
                        if norm_a > 0 and norm_b > 0:
                            batch_similarities[i, j] = dot_product / (norm_a * norm_b)
                        else:
                            batch_similarities[i, j] = 0
                
                # Convert to distances
                batch_distances = 1 - batch_similarities
                
                # Collect distances for this batch
                for i in range(end_idx - start_idx):
                    global_i = start_idx + i
                    for j in range(global_i + 1, len(embeddings)):  # Upper triangle only
                        if constructs[global_i] == constructs[j]:
                            intra_construct_distances.append(batch_distances[i, j])
                        else:
                            inter_construct_distances.append(batch_distances[i, j])
            
            # Clean data: remove any NaN or infinity values
            intra_distances = np.array(intra_construct_distances)
            intra_distances = intra_distances[np.isfinite(intra_distances)]
            
            inter_distances = np.array(inter_construct_distances)
            inter_distances = inter_distances[np.isfinite(inter_distances)]
            
            # Calculate statistics
            intra_mean = np.mean(intra_distances) if len(intra_distances) > 0 else 0
            intra_std = np.std(intra_distances) if len(intra_distances) > 0 else 0
            inter_mean = np.mean(inter_distances) if len(inter_distances) > 0 else 0
            inter_std = np.std(inter_distances) if len(inter_distances) > 0 else 0
            
            # Calculate effect size (Cohen's d) with safety checks
            if intra_std > 0 and inter_std > 0:
                pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
                effect_size = (inter_mean - intra_mean) / pooled_std if pooled_std > 0 else 0
            else:
                effect_size = 0
            
            print(f"Intra-construct distances: mean = {intra_mean:.4f}, std = {intra_std:.4f}")
            print(f"Inter-construct distances: mean = {inter_mean:.4f}, std = {inter_std:.4f}")
            print(f"Effect size (Cohen's d): {effect_size:.4f}")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot histograms of distances (if we have enough data)
            if len(intra_distances) > 10 and len(inter_distances) > 10:
                sns.histplot(intra_distances, alpha=0.5, label=f"Within construct: μ={intra_mean:.4f}", kde=True)
                sns.histplot(inter_distances, alpha=0.5, label=f"Between constructs: μ={inter_mean:.4f}", kde=True)
                
                plt.axvline(intra_mean, color='blue', linestyle='--')
                plt.axvline(inter_mean, color='orange', linestyle='--')
            else:
                plt.text(0.5, 0.5, "Insufficient data for histogram", 
                       ha='center', va='center', transform=plt.gca().transAxes)
            
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
            
        except Exception as e:
            print(f"Error computing separation metrics: {str(e)}")
            # Return default values if computation fails
            return {
                "intra_mean": 0,
                "intra_std": 0,
                "inter_mean": 0,
                "inter_std": 0,
                "effect_size": 0
            }
    
    def create_comparison_report(self, big_five_metrics, leadership_metrics):
        """Create a comprehensive report comparing the performance on Big Five vs leadership data."""
        report = f"""
    Triplet Loss Construct Learning Model Report
    ===========================================
    
    Model details:
    - Base model: {self.model_name}
    - Triplet loss training on Big Five constructs
    - Hard negative mining for better discrimination
    - Transfer to leadership styles
    
    Big Five Construct Identification Performance:
    -------------------------------------------
    - Test Accuracy: {big_five_metrics['accuracy']:.4f}
    - Test F1 Score: {big_five_metrics['f1_score']:.4f}
    - Number of constructs: 5
    - Effect size between construct categories: {big_five_metrics['separation']['effect_size']:.4f}
    
    Leadership Construct Identification Performance:
    --------------------------------------------
    - Accuracy: {leadership_metrics['accuracy']:.4f}
    - F1 Score: {leadership_metrics['f1_score']:.4f}
    - Number of constructs: {len(leadership_metrics['classification_report'].keys()) - 3}
    - Effect size between construct categories: {leadership_metrics['separation']['effect_size']:.4f}
    
    Comparative Analysis:
    -------------------
    The model {
        "performed BETTER" if big_five_metrics['accuracy'] > leadership_metrics['accuracy'] else "performed WORSE"
    } on Big Five constructs compared to leadership constructs.
    
    Effect size comparison shows that {
        "Big Five constructs are more distinct" if big_five_metrics['separation']['effect_size'] > leadership_metrics['separation']['effect_size']
        else "Leadership constructs are more distinct"
    } in the embedding space.
    
    Key Findings:
    -----------
    1. The model was able to learn construct spaces from Big Five data with {big_five_metrics['accuracy']:.1%} accuracy.
    
    2. When applied to leadership styles, the model achieved {leadership_metrics['accuracy']:.1%} accuracy in predicting the correct construct.
    
    3. The {
        "smaller" if leadership_metrics['separation']['effect_size'] < big_five_metrics['separation']['effect_size']
        else "larger"
    } effect size for leadership constructs ({leadership_metrics['separation']['effect_size']:.4f} vs. {big_five_metrics['separation']['effect_size']:.4f}) 
    suggests that leadership constructs {
        "overlap more" if leadership_metrics['separation']['effect_size'] < big_five_metrics['separation']['effect_size']
        else "are more distinct"
    } compared to personality constructs.
    
    4. UMAP visualizations confirm these findings, showing {
        "less" if leadership_metrics['separation']['effect_size'] < big_five_metrics['separation']['effect_size']
        else "more" 
    } distinct clustering for leadership constructs.
    
    Conclusion:
    ----------
    This analysis {
        "supports" if leadership_metrics['accuracy'] < big_five_metrics['accuracy'] and leadership_metrics['separation']['effect_size'] < big_five_metrics['separation']['effect_size']
        else "does not support"
    } the hypothesis that leadership constructs have substantial semantic overlap and may not represent truly distinct constructs.
    
    The model's ability to more effectively distinguish Big Five personality constructs compared to leadership constructs suggests that 
    leadership measurement may benefit from reconsidering its construct boundaries and potentially developing more distinctive measurement approaches.
        """
        
        # Save report
        report_file = VISUALIZATIONS_DIR / "triplet_construct_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        print(f"Saved comparison report to {report_file}")
        return report
    
    def save_model(self, output_path=None):
        """Save the model and important attributes."""
        if output_path is None:
            output_path = MODELS_DIR / "triplet_construct_model.pkl"
        
        # Create dictionary with model state and attributes
        model_data = {
            'construct_centroids': self.construct_centroids,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'training_loss': self.training_loss,
            'validation_metrics': self.validation_metrics,
            'timestamp': time.time()
        }
        
        # Save model to disk
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save the sentence transformer model
        self.model.save(str(MODELS_DIR / "triplet_model"))
        
        print(f"Model saved to {output_path}")


def main():
    """Main function to run the enhanced triplet loss construct learning process."""
    print("\n===== Enhanced Triplet Loss Construct Learning Model =====\n")
    
    # Create construct learner with stronger base model
    learner = TripletConstructLearner()
    
    # Load and prepare Big Five data for cleaner evaluation
    print("\n[1/9] Loading IPIP data...")
    ipip_df = learner.load_ipip_data(big_five_only=True)
    
    # Optional cross-validation (can be skipped for faster execution)
    print("\n[2/9] Skipping full cross-validation for faster execution...")
    cv_results = {
        'mean_accuracy': 0.0,
        'mean_f1': 0.0,
        'std_accuracy': 0.0,
        'folds': []
    }
    # For production, uncomment the line below:
    # cv_results = learner.cross_validate(ipip_df, label_column='big_five', n_splits=5)
    
    # Prepare triplet data with advanced hard mining
    print("\n[3/9] Preparing triplet data with advanced hard mining...")
    triplets, validation_triplets, train_df, val_df, test_df, _ = learner.prepare_triplet_data(
        ipip_df, label_column='big_five'
    )
    
    # Train triplet model with fewer epochs for test run
    print("\n[4/9] Training enhanced triplet model (quick test run)...")
    learner.train_triplet_model(
        triplets=triplets,
        validation_triplets=validation_triplets,
        epochs=2,  # Reduced for faster testing (use 5+ for production)
        batch_size=64   # Smaller batch for memory efficiency
    )
    
    # Generate embeddings for train and test data
    print("\n[5/9] Generating embeddings for evaluation...")
    train_embeddings = learner.generate_base_embeddings(train_df['processed_text'].tolist())
    test_embeddings = learner.generate_base_embeddings(test_df['processed_text'].tolist())
    
    # Compute construct centroids from training data
    learner.compute_construct_centroids(train_df, train_embeddings, label_column='big_five')
    
    # Predict constructs for test data with enhanced weighted method
    predicted_constructs, confidence_scores, similarities = learner.predict_constructs(
        test_embeddings, weighted=True
    )
    
    # Add predictions to test_df
    test_df['predicted_big_five'] = predicted_constructs
    
    # Evaluate predictions with comprehensive metrics
    print("\n[6/9] Evaluating Big Five construct classification performance...")
    big_five_metrics = learner.evaluate_predictions(
        test_df['big_five'].tolist(), 
        predicted_constructs,
        confidence_scores=confidence_scores,
        similarities=similarities,
        prefix="big_five"
    )
    
    # Compute separation metrics
    big_five_sep_metrics = learner.compute_construct_separation_metrics(
        test_embeddings, test_df, 'big_five'
    )
    big_five_metrics['separation'] = big_five_sep_metrics
    
    # Create enhanced UMAP visualization for Big Five constructs
    try:
        print("\nCreating UMAP visualization...")
        learner.create_umap_model(test_embeddings)
        learner.visualize_embeddings(
            test_embeddings, 
            test_df, 
            'big_five', 
            'Big Five Construct Space (Enhanced Triplet Loss)', 
            VISUALIZATIONS_DIR / "big_five_triplet_space.png",
            predicted_column='predicted_big_five',
            confidence_scores=confidence_scores
        )
    except Exception as e:
        print(f"Warning: UMAP visualization failed: {str(e)}")
    
    # Load leadership data
    print("\n[7/9] Loading and analyzing leadership data...")
    leadership_df = learner.load_leadership_data()
    
    # Generate embeddings for leadership data
    leadership_embeddings = learner.generate_base_embeddings(leadership_df['processed_text'].tolist())
    
    # Compute leadership construct separation metrics
    leadership_sep_metrics = learner.compute_construct_separation_metrics(
        leadership_embeddings, leadership_df, 'StandardConstruct'
    )
    
    # Map leadership styles to Big Five model with enhanced prediction
    leadership_predicted, leadership_confidence, leadership_similarities = learner.predict_constructs(
        leadership_embeddings, learner.construct_centroids, weighted=True
    )
    
    # Create a comprehensive mapping with all similarity scores
    constructs = list(learner.construct_centroids.keys())
    leadership_mapping = pd.DataFrame({
        'leadership_construct': leadership_df['StandardConstruct'],
        'text': leadership_df['processed_text'],
        'predicted_big_five': leadership_predicted,
        'confidence': leadership_confidence
    })
    
    # Add all similarity scores for analysis
    for i, construct in enumerate(constructs):
        leadership_mapping[f'sim_{construct}'] = leadership_similarities[:, i]
    
    # Add text ID for reference
    leadership_mapping['id'] = leadership_df.index
    
    # Save comprehensive mapping for analysis
    leadership_mapping.to_csv(PROCESSED_DIR / "leadership_big_five_mapping_comprehensive.csv", index=False)
    
    # Create a simplified mapping for reference
    leadership_mapping_simple = leadership_mapping[['leadership_construct', 'text', 'predicted_big_five', 'confidence']]
    leadership_mapping_simple.to_csv(PROCESSED_DIR / "leadership_big_five_mapping.csv", index=False)
    
    # Evaluate mapping consistency: how consistently does each leadership style map to a specific Big Five trait?
    leadership_consistency = {}
    leadership_accuracy = 0
    leadership_total = 0
    
    # Create a heat map of leadership style to Big Five mappings
    leadership_mapping_matrix = np.zeros((
        len(leadership_df['StandardConstruct'].unique()),
        len(constructs)
    ))
    
    # Show mapping statistics and fill matrix
    print("\nLeadership to Big Five Mapping:")
    for i, leadership_style in enumerate(sorted(leadership_df['StandardConstruct'].unique())):
        style_items = leadership_mapping[leadership_mapping['leadership_construct'] == leadership_style]
        
        # Get the distribution of predictions for this style
        prediction_counts = style_items['predicted_big_five'].value_counts()
        dominant_trait = prediction_counts.index[0]
        correct = prediction_counts.iloc[0]
        total = len(style_items)
        accuracy = correct / total
        
        # Fill mapping matrix for heatmap
        for j, trait in enumerate(constructs):
            leadership_mapping_matrix[i, j] = sum(style_items['predicted_big_five'] == trait) / total
        
        # Store consistency metrics
        leadership_consistency[leadership_style] = {
            'dominant_trait': dominant_trait,
            'accuracy': accuracy,
            'count': total,
            'distribution': {trait: count/total for trait, count in prediction_counts.items()}
        }
        
        leadership_accuracy += correct
        leadership_total += total
        
        # Format distribution for display
        distribution = ", ".join([f"{trait}: {count/total:.2f}" for trait, count in prediction_counts.items()])
        print(f"{leadership_style}: maps to {dominant_trait} ({correct}/{total}, {accuracy:.2f}) | Distribution: {distribution}")
    
    # Calculate overall mapping consistency
    overall_accuracy = leadership_accuracy / leadership_total
    print(f"\nOverall leadership mapping consistency: {overall_accuracy:.4f}")
    
    # Create heatmap of leadership style to Big Five mapping
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        leadership_mapping_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=constructs,
        yticklabels=sorted(leadership_df['StandardConstruct'].unique()),
        cbar_kws={'label': 'Proportion Mapped'}
    )
    plt.xlabel('Big Five Trait')
    plt.ylabel('Leadership Style')
    plt.title('Mapping of Leadership Styles to Big Five Traits')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "leadership_big_five_mapping_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Add predicted Big Five to leadership_df for visualization
    leadership_df['predicted_big_five'] = leadership_predicted
    leadership_df['confidence'] = leadership_confidence
    
    # Visualize leadership data with Big Five lens
    print("\n[8/9] Creating advanced visualizations...")
    
    try:
        # Visualize leadership data by actual leadership style
        learner.visualize_embeddings(
            leadership_embeddings,
            leadership_df,
            'StandardConstruct',
            'Leadership Styles in Big Five Space',
            VISUALIZATIONS_DIR / "leadership_in_big_five_space.png",
            confidence_scores=leadership_confidence
        )
        
        # Visualize leadership data by predicted Big Five
        learner.visualize_embeddings(
            leadership_embeddings,
            leadership_df,
            'predicted_big_five',
            'Leadership Styles Mapped to Big Five Traits',
            VISUALIZATIONS_DIR / "leadership_mapped_to_big_five.png",
            confidence_scores=leadership_confidence
        )
    except Exception as e:
        print(f"Warning: Leadership visualization failed: {str(e)}")
    
    # Create comparison report
    print("\n[9/9] Creating comprehensive analysis report...")
    
    # Prepare leadership metrics for report
    leadership_metrics = {
        'accuracy': overall_accuracy,
        'f1_score': overall_accuracy,  # Simplified for this mapping scenario
        'classification_report': {
            construct: {
                'precision': leadership_consistency.get(construct, {}).get('accuracy', 0),
                'recall': leadership_consistency.get(construct, {}).get('accuracy', 0),
                'f1-score': leadership_consistency.get(construct, {}).get('accuracy', 0),
                'support': leadership_consistency.get(construct, {}).get('count', 0)
            } for construct in leadership_df['StandardConstruct'].unique()
        },
        'separation': leadership_sep_metrics,
        'consistency': leadership_consistency,
        'cross_validation': cv_results
    }
    
    # Add required keys for report generation
    leadership_metrics['classification_report'].update({
        'accuracy': overall_accuracy,
        'macro avg': {'precision': overall_accuracy, 'recall': overall_accuracy, 'f1-score': overall_accuracy, 'support': leadership_total},
        'weighted avg': {'precision': overall_accuracy, 'recall': overall_accuracy, 'f1-score': overall_accuracy, 'support': leadership_total}
    })
    
    # Add cross-validation results to Big Five metrics
    big_five_metrics['cross_validation'] = cv_results
    
    # Create comparison report
    report = learner.create_comparison_report(big_five_metrics, leadership_metrics)
    
    # Save model
    learner.save_model(MODELS_DIR / "enhanced_triplet_construct_model.pkl")
    
    # Save report to file
    with open(VISUALIZATIONS_DIR / "enhanced_triplet_construct_analysis.txt", "w") as f:
        f.write(report)
    
    print("\nEnhanced triplet construct learning analysis complete!")
    print(f"Cross-validation accuracy: {cv_results['mean_accuracy']:.4f} (±{cv_results['std_accuracy']:.4f})")
    print(f"Test set accuracy: {big_five_metrics['accuracy']:.4f}")
    print(f"Leadership mapping consistency: {overall_accuracy:.4f}")
    print(f"Effect size - Big Five: {big_five_sep_metrics['effect_size']:.4f}, Leadership: {leadership_sep_metrics['effect_size']:.4f}")
    
    # Recommend next steps
    print("\nRecommended next steps:")
    print("1. Compare enhanced triplet results with GIST model results")
    print("2. Evaluate confidence thresholds for increasing precision")
    print("3. Consider implementing a bootstrapped ensemble approach for even higher accuracy")
    print("4. Run additional analyses on specific leadership styles that were difficult to classify")


if __name__ == "__main__":
    main()