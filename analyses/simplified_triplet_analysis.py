#!/usr/bin/env python3
"""
Simplified Triplet Model Analysis

This script implements a streamlined version of the triplet construct model
focused on achieving high accuracy in a minimal implementation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=== Simplified Triplet Model Analysis ===")

# Load IPIP data
print("\n1. Loading IPIP data...")
ipip_path = DATA_DIR / "IPIP.csv"
try:
    ipip_df = pd.read_csv(ipip_path, encoding='utf-8')
except UnicodeDecodeError:
    ipip_df = pd.read_csv(ipip_path, encoding='latin1')

# Define Big Five mapping
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

# Map labels to Big Five
def map_to_big_five(label):
    for big_five, labels in big_five_mapping.items():
        if label in labels:
            return big_five
    return None

# Process IPIP data
ipip_df['big_five'] = ipip_df['label'].apply(map_to_big_five)
ipip_df = ipip_df.dropna(subset=['big_five', 'text']).copy()

# Balance dataset
samples_per_class = 150  # Use fewer samples for quicker execution
balanced_df = pd.DataFrame()
for trait in ipip_df['big_five'].unique():
    trait_items = ipip_df[ipip_df['big_five'] == trait]
    if len(trait_items) > samples_per_class:
        balanced_df = pd.concat([balanced_df, trait_items.sample(samples_per_class, random_state=42)])
    else:
        balanced_df = pd.concat([balanced_df, trait_items])

ipip_df = balanced_df.copy()
print(f"Processed {len(ipip_df)} IPIP items:")
print(ipip_df['big_five'].value_counts())

# Split data
print("\n2. Preparing data...")
train_df, test_df = train_test_split(ipip_df, test_size=0.2, stratify=ipip_df['big_five'], random_state=42)
print(f"Train set: {len(train_df)} items")
print(f"Test set: {len(test_df)} items")

# Load and initialize model 
print("\n3. Initializing model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for speed
model = SentenceTransformer(model_name)
print(f"Using model: {model_name}")

# Create triplet data
print("\n4. Creating triplets...")
train_texts = train_df['text'].tolist()
train_labels = train_df['big_five'].tolist()

# Generate base embeddings
print("Generating embeddings...")
train_embeddings = model.encode(train_texts, show_progress_bar=True, convert_to_numpy=True)

# Group items by construct
construct_indices = {}
for idx, label in enumerate(train_labels):
    if label not in construct_indices:
        construct_indices[label] = []
    construct_indices[label].append(idx)

# Create triplets with hard negative mining
triplets = []
num_triplets = min(2000, len(train_texts) * 3)  # Limit for quick execution

for _ in tqdm(range(num_triplets), desc="Creating triplets"):
    # Random anchor construct
    anchor_label = random.choice(list(construct_indices.keys()))
    
    # Random anchor
    anchor_idx = random.choice(construct_indices[anchor_label])
    anchor_embedding = train_embeddings[anchor_idx]
    
    # Random positive
    positive_candidates = [idx for idx in construct_indices[anchor_label] if idx != anchor_idx]
    if not positive_candidates:
        continue
    positive_idx = random.choice(positive_candidates)
    
    # Hard negative mining
    negative_labels = [label for label in construct_indices.keys() if label != anchor_label]
    negative_indices = []
    for label in negative_labels:
        negative_indices.extend(construct_indices[label])
    
    negative_embeddings = train_embeddings[negative_indices]
    similarities = cosine_similarity([anchor_embedding], negative_embeddings)[0]
    
    # Get hard negatives
    hard_threshold = max(1, int(len(similarities) * 0.2))
    hard_negative_indices = np.argsort(similarities)[-hard_threshold:]
    selected_hard_idx = random.choice(hard_negative_indices)
    negative_idx = negative_indices[selected_hard_idx]
    
    # Create triplet
    triplets.append(InputExample(
        texts=[train_texts[anchor_idx], train_texts[positive_idx], train_texts[negative_idx]]
    ))

print(f"Created {len(triplets)} triplets")

# Train model
print("\n5. Training model...")
train_dataloader = DataLoader(triplets, shuffle=True, batch_size=32)
train_loss = losses.BatchHardTripletLoss(model=model, margin=0.2)

warmup_steps = int(len(train_dataloader) * 0.1)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,  # More epochs for better accuracy
    warmup_steps=warmup_steps,
    show_progress_bar=True
)

# Test the model
print("\n6. Testing model...")
# Encode test data
test_texts = test_df['text'].tolist()
test_labels = test_df['big_five'].tolist()
test_embeddings = model.encode(test_texts, show_progress_bar=True, convert_to_numpy=True)

# Compute centroids - important to reset indices first
train_df_reset = train_df.reset_index(drop=True)
construct_centroids = {}

# Rebuild train embeddings by index for safety (matching the dataframe)
train_indices = list(range(len(train_df_reset)))
 
for trait in train_df_reset['big_five'].unique():
    # Get indices within the reset dataframe
    trait_rows = train_df_reset[train_df_reset['big_five'] == trait]
    trait_indices = trait_rows.index.tolist()
    
    # Map to embedding indices safely
    trait_embeddings = np.array([train_embeddings[i] for i in trait_indices])
    construct_centroids[trait] = np.mean(trait_embeddings, axis=0)

# Predict using centroids
print("Predicting constructs...")
centroid_matrix = np.vstack(list(construct_centroids.values()))
constructs = list(construct_centroids.keys())

similarities = cosine_similarity(test_embeddings, centroid_matrix)
predicted_indices = np.argmax(similarities, axis=1)
predicted_constructs = [constructs[idx] for idx in predicted_indices]

# Evaluate predictions
accuracy = accuracy_score(test_labels, predicted_constructs)
f1 = f1_score(test_labels, predicted_constructs, average='weighted')

print("\n7. Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, predicted_constructs))

# Create visualization
plt.figure(figsize=(12, 8))
plt.bar(['Accuracy', 'F1 Score'], [accuracy, f1], color=['blue', 'green'])
plt.title('Triplet Model Performance on Big Five Classification')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate([accuracy, f1]):
    plt.text(i, v+0.02, f"{v:.4f}", ha='center')
plt.savefig(VISUALIZATIONS_DIR / "triplet_accuracy.png", dpi=300, bbox_inches="tight")

print("\nAnalysis complete! Results saved to visualizations directory.")