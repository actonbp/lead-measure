#!/usr/bin/env python3
"""
Flexible Pipeline for Fine-Tuning and Evaluating Embedding Models

This script allows experimenting with different local sentence-transformer models,
loss functions, and evaluation methods on a given dataset.
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
import itertools 
import json
import time
from datetime import datetime

# Added for API key loading and API client
import os
from dotenv import load_dotenv
import openai 

# Import SentenceTransformers
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation

# Placeholder for API calls if needed later
# import openai

# --- Configuration ---

def parse_args():
    parser = argparse.ArgumentParser(description="Run Embedding Model Experiments")
    
    # Data Args
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV data file.")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for item text.")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for construct/facet labels.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for the test set.")
    
    # Embedding Model Args
    parser.add_argument("--embedding_config", type=str, required=True, 
                        help="Embedding model configuration (e.g., 'local:all-mpnet-base-v2', 'api:openai:text-embedding-3-large')")
    
    # Fine-tuning Args (only for local models)
    parser.add_argument("--loss_function", type=str, default="None", 
                        help="Loss function for fine-tuning (e.g., 'GISTEmbedLoss', 'MultipleNegativesRankingLoss', 'None' for no fine-tuning).")
    parser.add_argument("--pairing_strategy", type=str, default="all_pairs", choices=['all_pairs', 'sampled_pairs'],
                        help="How to generate positive pairs for contrastive losses.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for fine-tuning.")
    
    # Evaluation Args
    parser.add_argument("--evaluation_methods", nargs='+', default=['1-NN', 'distance'], 
                        help="List of evaluation methods to run (e.g., '1-NN', 'distance', 'hdbscan', 'umap').")
    
    # Output Args
    parser.add_argument("--output_dir", type=str, default="./experiment_results", help="Directory to save results and models.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Optional name for this experiment run.")

    # API Args
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY", 
                        help="Environment variable name containing the API key for API models.")
    parser.add_argument("--api_batch_size", type=int, default=500, 
                        help="Number of texts to send per OpenAI API batch request.") # Added batch size for API calls

    args = parser.parse_args()

    # Create a unique run ID if no experiment name is given
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"run_{timestamp}"
        
    args.output_path = Path(args.output_dir) / args.experiment_name
    os.makedirs(args.output_path, exist_ok=True)
        
    return args

# --- Helper Functions (to be implemented) ---

def load_and_split_data(args):
    """Loads data, cleans it, and splits into train/test."""
    print(f"Loading data from: {args.data_path}")
    data_file = Path(args.data_path)
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        return None, None
    
    # Load the dataset, handle encoding issues
    try:
        df = pd.read_csv(data_file, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying latin1 encoding.")
        try:
             df = pd.read_csv(data_file, encoding='latin1')
        except Exception as e:
             print(f"Error loading CSV file with latin1 encoding: {e}")
             return None, None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None
        
    # Validate essential columns
    if args.text_column not in df.columns or args.label_column not in df.columns:
         print(f"Error: Required columns '{args.text_column}' or '{args.label_column}' not found in {data_file}")
         return None, None

    # Basic cleaning
    initial_rows = len(df)
    df.dropna(subset=[args.text_column, args.label_column], inplace=True)
    df[args.text_column] = df[args.text_column].astype(str)
    df[args.label_column] = df[args.label_column].astype(str)
    cleaned_rows = len(df)
    if cleaned_rows < initial_rows:
         print(f"Removed {initial_rows - cleaned_rows} rows with missing text or labels.")

    print(f"Loaded {cleaned_rows} items after cleaning.")

    if cleaned_rows == 0:
         print("Error: No valid data remaining after cleaning.")
         return None, None
         
    # --- Train/Test Split ---
    print(f"Splitting data into train/test sets (test_size={args.test_size})")
    
    stratify_col = None
    if df[args.label_column].nunique() >= 2:
        # Check for labels with only one sample before attempting stratification
        label_counts = df[args.label_column].value_counts()
        if (label_counts == 1).any():
            print(f"Warning: Found { (label_counts == 1).sum()} labels with only one sample. Stratification might be unstable or fail.")
            # Optionally filter out single-sample labels if they cause issues, but try without filtering first.
            # labels_to_keep = label_counts[label_counts > 1].index
            # df_filtered = df[df[args.label_column].isin(labels_to_keep)]
            # if len(df_filtered) < len(df):
            #     print(f"Removed {len(df) - len(df_filtered)} items with single-sample labels before split.")
            #     df = df_filtered
            # if df[args.label_column].nunique() >= 2:
            #      stratify_col = df[args.label_column]
            # else:
            #      print("Warning: Not enough unique labels remain after filtering for stratification.")
            # For now, we'll still try to stratify even with single samples, sklearn might handle it.
            stratify_col = df[args.label_column]
        else:
             stratify_col = df[args.label_column]
    else:
        print("Warning: Not enough unique labels for stratification.")

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            stratify=stratify_col,
            random_state=42 # Use a fixed random state for reproducibility
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=42
        )
        
    print(f"Train set size: {len(train_df)} items")
    print(f"Test set size: {len(test_df)} items")
    
    return train_df, test_df

def get_embeddings(args, texts_to_embed):
     """Handles getting embeddings from local or API models."""
     print(f"Getting embeddings using config: {args.embedding_config}")
     # ... (Implementation needed - needs parsing embedding_config)
     # For now, assume local and return base model embeddings
     model_type, model_name = args.embedding_config.split(':', 1)
     if model_type == 'local':
         print(f"Loading local base model: {model_name}")
         model = SentenceTransformer(model_name)
         embeddings = model.encode(texts_to_embed, show_progress_bar=True)
         return embeddings, model # Return model only if local base
     elif model_type == 'api':
         print("API embedding not yet implemented.")
         # ... (API call logic would go here)
         raise NotImplementedError("API embedding retrieval not implemented.")
     else:
         raise ValueError(f"Invalid embedding config format: {args.embedding_config}")


def prepare_training_data(df, strategy, text_col, label_col):
    """Prepares InputExamples based on the pairing strategy."""
    print(f"Preparing training data using strategy: {strategy}")
    
    train_examples = []
    grouped = df.groupby(label_col)
    
    for label, group in grouped:
        items = group[text_col].tolist()
        if len(items) < 2:
            continue # Need at least two items to form a pair
            
        if strategy == 'all_pairs':
            # Generate all unique pairs within the group
            for item1, item2 in itertools.combinations(items, 2):
                train_examples.append(InputExample(texts=[item1, item2]))
        elif strategy == 'sampled_pairs':
            # Sample a fixed number of pairs per anchor (adapted from gist_loss_personality)
            for i, anchor in enumerate(items):
                num_samples = min(5, len(items) - 1) # Sample up to 5 positive pairs
                if num_samples <= 0:
                    continue 
                
                possible_indices = [j for j in range(len(items)) if j != i]
                # Ensure we don't request more samples than available indices
                if num_samples > len(possible_indices):
                    num_samples = len(possible_indices)
                    
                if num_samples > 0:
                     pos_indices = np.random.choice(
                         possible_indices,
                         num_samples,
                         replace=False
                     )
                     for pos_idx in pos_indices:
                         train_examples.append(InputExample(texts=[anchor, items[pos_idx]]))
        else:
            raise ValueError(f"Unknown pairing_strategy: {strategy}")

    if not train_examples:
         print("Warning: No training pairs generated. Check data and label distribution.")
         return None
         
    print(f"Created {len(train_examples)} training pairs using strategy '{strategy}'.")
    return train_examples

def fine_tune_model(base_model, train_examples, loss_name, args):
    """Fine-tunes a local model using the specified loss."""
    print(f"Fine-tuning model '{base_model.config.name_or_path}' with loss: {loss_name}")
    
    # Select and initialize the loss function
    if loss_name == 'GISTEmbedLoss':
        print("Initializing GISTEmbedLoss...")
        # Requires a guide model, typically the same architecture
        try:
            guide_model = SentenceTransformer(base_model.config.name_or_path)
            train_loss = losses.GISTEmbedLoss(model=base_model, guide=guide_model)
        except AttributeError as e:
            print(f"Error: Could not find GISTEmbedLoss. Ensure sentence-transformers version supports it. {e}")
            return None
        except Exception as e:
            print(f"Error initializing GISTEmbedLoss: {e}")
            return None
            
    elif loss_name == 'MultipleNegativesRankingLoss':
        print("Initializing MultipleNegativesRankingLoss...")
        try:
             # This loss works well with (anchor, positive) pairs and uses in-batch negatives
             train_loss = losses.MultipleNegativesRankingLoss(model=base_model)
        except AttributeError as e:
             print(f"Error: Could not find MultipleNegativesRankingLoss. Check sentence-transformers version. {e}")
             return None
        except Exception as e:
             print(f"Error initializing MultipleNegativesRankingLoss: {e}")
             return None
    else:
        print(f"Error: Unknown or unsupported loss function specified: {loss_name}")
        return None
        
    # Create data loader
    # Consider making batch size configurable if needed
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Set up warmup steps (10% of training steps)
    num_training_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(num_training_steps * 0.1)
    
    # Define output path for the fine-tuned model
    # Include model name and loss in the path for clarity
    model_save_name = f"{base_model.config.name_or_path.replace('/', '_')}_tuned_with_{loss_name}"
    output_save_path = args.output_path / model_save_name
    print(f"Fine-tuned model will be saved to: {output_save_path}")

    # Set up evaluator (optional but recommended)
    # For simplicity, we won't create a complex evaluator here, but it could be added
    # evaluator = create_evaluator(...) # Requires test_df and adaptation
    evaluator = None 
    evaluation_steps = 0
    if evaluator:
         evaluation_steps = int(len(train_dataloader) * 0.1) # Evaluate every 10%

    # Train the model
    try:
        base_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': args.learning_rate}, # Pass learning rate
            output_path=str(output_save_path),
            save_best_model= evaluator is not None, # Only save best if evaluator exists
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
            show_progress_bar=True
        )
    except Exception as e:
         print(f"Error during model training: {e}")
         # Attempt to catch ImportError related to missing dependencies
         if isinstance(e, ImportError) and ('datasets' in str(e) or 'accelerate' in str(e)):
              print("Hint: Training dependencies missing. Try `pip install datasets accelerate`")
         return None # Indicate failure

    # If no evaluator, the final model is at output_save_path
    # If evaluator exists, SentenceTransformer saves the best model there automatically.
    print(f"Loading trained model from {output_save_path}")
    try:
        tuned_model = SentenceTransformer(str(output_save_path))
        return tuned_model
    except Exception as e:
        print(f"Error loading trained model from {output_save_path}: {e}")
        return None

def run_1nn_evaluation(train_embeddings, test_embeddings, train_labels, test_labels):
    """Calculates 1-NN accuracy using cosine similarity."""
    print("Running 1-NN evaluation...")
    if train_embeddings is None or test_embeddings is None or train_labels is None or test_labels is None:
        print("Error: Missing embeddings or labels for 1-NN evaluation.")
        return None
        
    if len(train_embeddings) == 0 or len(test_embeddings) == 0:
        print("Error: Empty embeddings provided for 1-NN evaluation.")
        return None
        
    try:
        # Calculate cosine similarities between test and train embeddings
        print("Calculating similarities between test and train embeddings...")
        # Resulting shape: (n_test_samples, n_train_samples)
        similarities = cosine_similarity(test_embeddings, train_embeddings)

        # Find the index of the nearest neighbor in the training set for each test item
        print("Finding nearest neighbors...")
        nearest_neighbor_indices = np.argmax(similarities, axis=1)

        # Get the labels of the nearest neighbors
        # Ensure train_labels is numpy array
        if isinstance(train_labels, pd.Series):
            train_labels = train_labels.values 
        predicted_labels = train_labels[nearest_neighbor_indices]

        # Get the true labels of the test set
        if isinstance(test_labels, pd.Series):
             test_labels = test_labels.values

        # Calculate accuracy
        print("Calculating 1-NN accuracy...")
        accuracy = np.mean(predicted_labels == test_labels)
        
        print(f"Overall 1-Nearest Neighbor Accuracy: {accuracy:.4f}")
        return accuracy
        
    except Exception as e:
        print(f"Error during 1-NN evaluation: {e}")
        # Print shapes for debugging
        print(f"Test embeddings shape: {test_embeddings.shape if test_embeddings is not None else 'None'}")
        print(f"Train embeddings shape: {train_embeddings.shape if train_embeddings is not None else 'None'}")
        print(f"Train labels length: {len(train_labels) if train_labels is not None else 'None'}")
        return None

def run_distance_analysis(embeddings, labels):
    """Calculates intra/inter distances and Cohen's d."""
    print("Running distance analysis...")
    if embeddings is None or labels is None:
         print("Error: Missing embeddings or labels for distance analysis.")
         return None
         
    if len(embeddings) == 0:
         print("Error: Empty embeddings provided for distance analysis.")
         return None
         
    try:
        # Calculate all pairwise distances (cosine distance = 1 - cosine similarity)
        similarities = cosine_similarity(embeddings)
        np.fill_diagonal(similarities, 1.0) # Ensure diagonal is exactly 1
        distances = 1 - similarities
        
        # Get unique labels
        unique_labels = pd.unique(labels) # Use pandas unique for potentially mixed types
        print(f"Analyzing {len(unique_labels)} unique traits/labels.")

        # Initialize containers for distances
        intra_label_distances = []
        inter_label_distances = []
        
        # Map labels to indices for efficient comparison
        label_map = {label: i for i, label in enumerate(unique_labels)}
        label_indices = np.array([label_map.get(l, -1) for l in labels]) # Use .get for safety
        
        # Collect distances efficiently
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):  # Upper triangle only, excluding diagonal
                dist = distances[i, j]
                if not np.isfinite(dist):
                    # print(f"Warning: Non-finite distance encountered between item {i} and {j}. Skipping.")
                    continue

                if label_indices[i] == -1 or label_indices[j] == -1: # Skip if label wasn't in map (shouldn't happen)
                     continue
                     
                if label_indices[i] == label_indices[j]:
                    intra_label_distances.append(dist)
                else:
                    inter_label_distances.append(dist)
        
        # Convert to numpy arrays
        intra_distances = np.array(intra_label_distances)
        inter_distances = np.array(inter_label_distances)

        # Calculate statistics, handle empty arrays
        mean_intra = np.mean(intra_distances) if len(intra_distances) > 0 else np.nan
        std_intra = np.std(intra_distances) if len(intra_distances) > 1 else np.nan # std needs >1 point
        mean_inter = np.mean(inter_distances) if len(inter_distances) > 0 else np.nan
        std_inter = np.std(inter_distances) if len(inter_distances) > 1 else np.nan

        print(f"Intra-label distances: mean = {mean_intra:.4f}, std = {std_intra:.4f} (n={len(intra_distances)})")
        print(f"Inter-label distances: mean = {mean_inter:.4f}, std = {std_inter:.4f} (n={len(inter_distances)})")
        
        # Calculate Cohen's d for effect size
        cohen_d = np.nan # Default to NaN
        if len(intra_distances) > 1 and len(inter_distances) > 1 and std_intra > 0 and std_inter > 0:
             pooled_std = np.sqrt(((len(intra_distances) - 1) * std_intra**2 + (len(inter_distances) - 1) * std_inter**2) / 
                                 (len(intra_distances) + len(inter_distances) - 2))
             if pooled_std > 0:
                 cohen_d = (mean_inter - mean_intra) / pooled_std
        print(f"Effect size (Cohen's d): {cohen_d:.4f}")
        
        metrics = {
            'mean_intra_distance': mean_intra,
            'std_intra_distance': std_intra,
            'mean_inter_distance': mean_inter,
            'std_inter_distance': std_inter,
            'cohen_d': cohen_d
        }
        return metrics
    except Exception as e:
         print(f"Error during distance analysis: {e}")
         print(f"Embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
         print(f"Labels length: {len(labels) if labels is not None else 'None'}")
         return None
    
def save_experiment_results(args, metrics, output_path):
     """Saves configuration and results metrics."""
     print(f"Saving results to {output_path}")
     # Convert Path objects in args to strings for JSON serialization
     config_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
     results = {
         'config': config_to_save, # Save the run configuration
         'metrics': metrics
     }
     results_file = output_path / "results.json"
     try:
          with open(results_file, 'w') as f:
              json.dump(results, f, indent=4)
          print(f"Results saved to {results_file}")
     except Exception as e:
          print(f"Error saving results to JSON: {e}")

def _get_openai_embeddings_batch(client, texts, api_model_name):
     """Helper to get embeddings for a single batch, with basic retry."""
     max_retries = 3
     wait_time = 5 # seconds
     for attempt in range(max_retries):
         try:
             res = client.embeddings.create(input=texts, model=api_model_name)
             embeddings = [item.embedding for item in res.data]
             return np.array(embeddings)
         except openai.RateLimitError as e:
             print(f"Rate limit hit, waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
             time.sleep(wait_time)
         except Exception as e:
             print(f"Error getting OpenAI embeddings (batch size {len(texts)}, attempt {attempt+1}): {e}")
             if attempt == max_retries - 1:
                  print("Max retries reached. Failed to get embeddings for this batch.")
                  return None # Failed after retries
             time.sleep(wait_time) # Wait before retrying other errors too
     return None # Should not be reached if retries fail correctly

def get_api_embeddings(texts_to_embed, api_config, api_key, batch_size=500):
    """Gets embeddings from an API, currently supports OpenAI."""
    if not texts_to_embed:
        return np.array([])

    provider, api_model_name = api_config.split(':', 1)

    if provider.lower() == 'openai':
        if not api_key:
            print("Error: OpenAI API key not provided or found.")
            return None
        try:
             client = openai.OpenAI(api_key=api_key)
        except Exception as e:
             print(f"Error initializing OpenAI client: {e}")
             return None
             
        print(f"Getting OpenAI embeddings for {len(texts_to_embed)} texts using model '{api_model_name}' (batch size: {batch_size})...")
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="OpenAI Embeddings"):
            batch_texts = texts_to_embed[i:min(i + batch_size, len(texts_to_embed))]
            batch_embeddings = _get_openai_embeddings_batch(client, batch_texts, api_model_name)
            if batch_embeddings is None:
                 print(f"Error: Failed to get embeddings for batch starting at index {i}. Aborting.")
                 return None # Indicate failure
            all_embeddings.append(batch_embeddings)
            
        if not all_embeddings:
            print("Warning: No embeddings were generated.")
            return np.array([])
            
        # Check if any batch failed (should be caught earlier, but as a safeguard)
        if any(batch is None for batch in all_embeddings):
             print("Error: Some batches failed during embedding generation.")
             return None
             
        return np.vstack(all_embeddings)
        
    else:
        print(f"Error: Unsupported API provider: {provider}")
        return None

# --- Main Workflow ---

def main():
    # Load .env file variables
    load_dotenv()
    
    args = parse_args()
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Full configuration: {vars(args)}")

    start_time = time.time()

    # 1. Load Data
    train_df, test_df = load_and_split_data(args)
    if train_df is None or test_df is None:
        print("Failed to load or split data. Exiting.")
        return

    # 2. Determine Embedding Strategy & Get Model/Embeddings
    model_type, model_identifier = args.embedding_config.split(':', 1)
    
    final_model = None # Will hold the model object if local & fine-tuned/used directly
    train_embeddings = None
    test_embeddings = None
    all_embeddings = None # For combined dataset evaluations
    
    if model_type == 'local':
        # --- Local Model Path ---
        print(f"Processing local model: {model_identifier}")
        try:
             base_model = SentenceTransformer(model_identifier)
        except Exception as e:
             print(f"Error loading base model '{model_identifier}': {e}")
             return

        if args.loss_function != "None":
            # Fine-tuning path
            train_examples = prepare_training_data(train_df, args.pairing_strategy, args.text_column, args.label_column)
            if not train_examples:
                 print("Failed to prepare training data. Exiting.")
                 return
                 
            final_model = fine_tune_model(base_model, train_examples, args.loss_function, args)
            if final_model is None:
                 print("Fine-tuning failed. Exiting.")
                 return
        else:
            # No fine-tuning, use base model directly
            print("No fine-tuning specified. Using base model.")
            final_model = base_model
            
        # Generate embeddings using the final local model
        print("Generating final embeddings for train set...")
        train_embeddings = final_model.encode(train_df[args.text_column].tolist(), show_progress_bar=True)
        print("Generating final embeddings for test set...")
        test_embeddings = final_model.encode(test_df[args.text_column].tolist(), show_progress_bar=True)
            
    elif model_type == 'api':
        # --- API Model Path ---
        print(f"Processing API model: {model_identifier}")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"Error: API key not found in environment variable '{args.api_key_env}'. Set the variable or use --api_key_env.")
            return
            
        # Split identifier into provider and model name
        try:
             api_provider, api_model_name = model_identifier.split(':', 1)
        except ValueError:
             print(f"Error: Invalid API model identifier format: {model_identifier}. Expected 'provider:model_name'.")
             return
             
        api_config = f"{api_provider}:{api_model_name}" # Pass combined identifier

        print("Generating API embeddings for train set...")
        train_embeddings = get_api_embeddings(
             train_df[args.text_column].tolist(), 
             api_config, 
             api_key, 
             args.api_batch_size
        )
        print("Generating API embeddings for test set...")
        test_embeddings = get_api_embeddings(
             test_df[args.text_column].tolist(), 
             api_config, 
             api_key, 
             args.api_batch_size
        )
        
        if train_embeddings is None or test_embeddings is None:
            print("Failed to retrieve API embeddings. Exiting.")
            return
            
        # Note: final_model remains None for API path
        print("API embeddings generated successfully.")
        
    else:
         raise ValueError(f"Invalid embedding config model type: {model_type}")

    # Check if embeddings were generated successfully before proceeding
    if train_embeddings is None or test_embeddings is None:
         print("Error: Embedding generation failed. Cannot proceed with evaluation.")
         return
         
    # Check for non-finite values (common issue, good to check early)
    if not np.all(np.isfinite(train_embeddings)) or not np.all(np.isfinite(test_embeddings)):
         print("Warning: Non-finite values (NaN or Inf) found in generated train/test embeddings! Evaluation might be affected or fail.")
         # Depending on evaluation, might need to handle (e.g., np.nan_to_num) or investigate source.

    # 4. Run Evaluations
    metrics = {}
    print("\n--- Running Evaluations ---")
    if '1-NN' in args.evaluation_methods:
        nn_acc = run_1nn_evaluation(train_embeddings, test_embeddings, train_df[args.label_column].values, test_df[args.label_column].values)
        if nn_acc is not None:
             metrics['1-NN_accuracy'] = nn_acc
             
    if 'distance' in args.evaluation_methods:
         # Requires embeddings for the full dataset
         print("Generating embeddings for full dataset for distance analysis...")
         all_texts = pd.concat([train_df, test_df])[args.text_column].tolist()
         all_labels = pd.concat([train_df, test_df])[args.label_column].values
         
         if model_type == 'local':
              # Use the final local model (base or fine-tuned)
              all_embeddings = final_model.encode(all_texts, show_progress_bar=True)
         elif model_type == 'api':
              # Get API embeddings for the full dataset
              all_embeddings = get_api_embeddings(all_texts, api_config, api_key, args.api_batch_size)
              if all_embeddings is None:
                  print("Warning: Failed to get API embeddings for full dataset. Skipping distance analysis.")
                  
         if all_embeddings is not None:
             if not np.all(np.isfinite(all_embeddings)):
                   print("Warning: Non-finite values found in full dataset embeddings! Distance analysis might be affected.")
             dist_metrics = run_distance_analysis(all_embeddings, all_labels)
             if dist_metrics:
                  metrics.update(dist_metrics) # Merge distance metrics
         else:
             print("Skipping distance analysis due to embedding generation failure.")

    # Add placeholders/calls for other evaluations (umap, hdbscan) here if needed

    # 5. Save Results
    save_experiment_results(args, metrics, args.output_path)

    end_time = time.time()
    print(f"\nExperiment '{args.experiment_name}' finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main() 