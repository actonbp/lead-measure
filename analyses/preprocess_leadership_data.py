#!/usr/bin/env python3
"""
Leadership Measures Data Preprocessing

This script processes the leadership measures dataset to create multiple versions:
1. Complete dataset (all constructs)
2. Fischer & Sitkin (2023) constructs only
3. Fischer & Sitkin constructs with stems removed
4. Fischer & Sitkin constructs with stems and gendered language removed

Usage:
    python preprocess_leadership_data.py

Output:
    Processed CSV files saved to data/processed/
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define Fischer & Sitkin (2023) leadership constructs
FISCHER_SITKIN_CONSTRUCTS = {
    # 8 positive styles
    'Authentic': ['authentic', 'authenticity'],
    'Charismatic': ['charismatic', 'charisma'],
    'Consideration': ['consideration', 'considerate'],
    'Initiating Structure': ['initiating structure', 'structure', 'structuring'],
    'Empowering': ['empowering', 'empowerment'],
    'Ethical': ['ethical', 'ethics'],
    'Instrumental': ['instrumental'],
    'Servant': ['servant'],
    'Transformational': ['transformational', 'transform'],
    
    # 2 negative styles
    'Abusive': ['abusive', 'abuse'],
    'Destructive': ['destructive', 'destruction', 'Destructive Leadership']
}

# Stem patterns to remove (common leader references)
STEM_PATTERNS = [
    # Basic leader references
    r'^my (leader|manager|supervisor|boss|superior)',
    r'^the (leader|manager|supervisor|boss|superior)',
    r'^our (leader|manager|supervisor|boss|superior)',
    r'^this (leader|manager|supervisor|boss|superior)',
    r'^your (leader|manager|supervisor|boss|superior)',
    r'^this person',
    
    # Department/team variations
    r'^my department\'?s? (leader|manager|supervisor|boss|superior)',
    r'^the department\'?s? (leader|manager|supervisor|boss|superior)',
    r'^our (team\'?s?|department\'?s?) (leader|manager|supervisor|boss|superior)',
    r'^the (team\'?s?|department\'?s?) (leader|manager|supervisor|boss|superior)',
    
    # Pronouns used at start
    r'^he/she',
    r'^s?he',
    r'^they',
    r'^(my|your|the|our) (leader|manager|supervisor|boss|superior) (who|that)',
    
    # Job titles
    r'^(my|the|our) (CEO|CFO|CTO|president|executive|director|head)',
    r'^(my|the|our) (foreman|chairperson|chairman)',
    
    # General starts to remove
    r'^the person (who|that)',
    r'^a (leader|manager) (who|that)',
]

# Gendered terms to replace
GENDERED_TERMS = {
    # Pronouns
    'he': 'they',
    'she': 'they',
    'his': 'their',
    'her': 'their',
    'hers': 'theirs',
    'himself': 'themselves',
    'herself': 'themselves',
    'him': 'them',
    
    # Job titles
    'chairman': 'chairperson',
    'chairwoman': 'chairperson',
    'foreman': 'supervisor',
    'forewoman': 'supervisor',
    'policeman': 'police officer',
    'policewoman': 'police officer',
    'fireman': 'firefighter',
    'firewoman': 'firefighter',
    'businessman': 'business person',
    'businesswoman': 'business person',
    'salesman': 'salesperson',
    'saleswoman': 'salesperson',
    'mailman': 'mail carrier',
    'mailwoman': 'mail carrier',
    'steward': 'flight attendant',
    'stewardess': 'flight attendant',
    'waiter': 'server',
    'waitress': 'server',
    'spokesman': 'spokesperson',
    'spokeswoman': 'spokesperson',
    
    # General terms
    'mankind': 'humanity',
    'man-made': 'artificial',
    'manpower': 'workforce',
    'workmanship': 'craftsmanship',
    'man': 'person',
    'men': 'people',
    'woman': 'person',
    'women': 'people',
    'male': 'individual',
    'female': 'individual',
    'guy': 'person',
    'guys': 'people',
    'boys': 'people',
    'girls': 'people',
    'gentleman': 'person',
    'gentlemen': 'people',
    'lady': 'person',
    'ladies': 'people'
}

# Moved this function definition to top level
def map_to_standard_construct(behavior, dimension):
    """Map behavior (and dimension for LBDQ) to standardized construct name."""
    # Handle LBDQ specifically using the Dimension column
    if isinstance(behavior, str) and 'lbdq' in behavior.lower():
        if isinstance(dimension, str):
            dim_lower = dimension.lower()
            if 'initiating structure' in dim_lower:
                return 'Initiating Structure'
            elif 'consideration' in dim_lower:
                return 'Consideration'
        # Fallback for LBDQ if dimension is missing or doesn't match expected values
        # print(f"Warning: LBDQ item found ('{behavior}') but dimension '{dimension}' not recognized. Returning as 'LBDQ_Unknown'.")
        return 'LBDQ_Unknown' # Assign specific unknown category 
    
    # Original logic for non-LBDQ behaviors
    if isinstance(behavior, str):
        behavior_lower = behavior.lower()
        for construct, terms in FISCHER_SITKIN_CONSTRUCTS.items():
            if any(term.lower() in behavior_lower for term in terms):
                return construct
                
    # Fallback if no match found for non-LBDQ
    return behavior 

def load_leadership_data(file_path=None):
    """Load the leadership measures dataset."""
    if file_path is None:
        file_path = RAW_DATA_DIR / "Measures_text_long.csv"
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {file_path}")
    
    # Load data
    print(f"Loading leadership measures from {file_path}")
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    df = df.dropna(subset=['Text'])  # Remove items with no text
    
    print(f"Loaded {len(df)} leadership items across {df['Behavior'].nunique()} constructs")
    return df

def is_fischer_sitkin_construct(behavior):
    """Check if a behavior belongs to Fischer & Sitkin (2023) constructs OR is LBDQ."""
    if not isinstance(behavior, str):
        return False
        
    behavior_lower = behavior.lower()
    
    # Explicitly include LBDQ
    if 'lbdq' in behavior_lower:
        return True
        
    # Check against keywords in the dictionary for other constructs
    return any(
        term.lower() in behavior_lower
        for construct_terms in FISCHER_SITKIN_CONSTRUCTS.values() 
        for term in construct_terms
    )

def filter_fischer_sitkin_constructs(df):
    """Filter to include only Fischer & Sitkin (2023) leadership constructs."""
    # Create a boolean mask for matching constructs
    mask = df['Behavior'].apply(is_fischer_sitkin_construct)
    
    filtered_df = df[mask].copy()
    
    print(f"Filtered to {len(filtered_df)} items from Fischer & Sitkin (2023) constructs")
    print(f"Constructs included: {filtered_df['Behavior'].unique()}")
    
    return filtered_df

def remove_stems(text):
    """Remove common leadership item stems from text."""
    lower_text = text.lower()
    
    # Check and remove each stem pattern
    for pattern in STEM_PATTERNS:
        match = re.match(pattern, lower_text, re.IGNORECASE)
        if match:
            # Remove the stem and clean up
            text = text[match.end():].strip() # Get text after stem and strip whitespace
            # --- Restore more aggressive leading char cleanup ---
            text = re.sub(r'^[^\w]+\b', '', text).strip() # Remove leading non-alphanumeric (should catch leading '(' if needed)
            # text = re.sub(r'^[\'\"]s\b\s*', '', text).strip() # Keep this commented out for now
            # text = re.sub(r'^[,.:;!?]+\s*', '', text).strip() # Keep this commented out for now
            # --- End Restore --- 
            
            # Capitalize first letter if text remains
            if text:
                text = text[0].upper() + text[1:]
            break # Important: stop after first match
    
    return text

def remove_gendered_language(text):
    """Replace gendered terms with gender-neutral alternatives."""
    # First handle combined forms like "his/her" -> "their"
    combined_forms = {
        "his/her": "their",
        "he/she": "they",
        "him/her": "them",
        "himself/herself": "themselves",
        "s/he": "they"
    }
    
    for pattern, replacement in combined_forms.items():
        text = re.sub(r'\b' + re.escape(pattern) + r'\b', replacement, text, flags=re.IGNORECASE)
    
    # Then handle individual words
    words = text.split()
    result_words = []
    
    for word in words:
        # Extract any punctuation
        prefix_punct = ''
        suffix_punct = ''
        
        match = re.match(r'^([^\w]*)(.+?)([^\w]*)$', word)
        if match:
            prefix_punct, word_core, suffix_punct = match.groups()
        else:
            word_core = word
        
        # Check if the word (case-insensitive) matches any gendered term
        word_lower = word_core.lower()
        if word_lower in GENDERED_TERMS:
            # Replace with gender-neutral term, preserve case
            if word_core.isupper():
                replacement = GENDERED_TERMS[word_lower].upper()
            elif word_core[0].isupper():
                replacement = GENDERED_TERMS[word_lower].capitalize()
            else:
                replacement = GENDERED_TERMS[word_lower]
            
            # Reassemble with punctuation
            result_words.append(f"{prefix_punct}{replacement}{suffix_punct}")
        else:
            # Keep original word
            result_words.append(word)
    
    return ' '.join(result_words)

def remove_specific_words(text, words_to_remove):
    """Remove specific whole words from text (case-insensitive)."""
    if not words_to_remove:
        return text
        
    # Build a regex pattern like r'\b(word1|word2|word3)\b'
    pattern = r'\b(?i)(?:{})'.format('|'.join(map(re.escape, words_to_remove))) + r'\b'
    
    # Replace matched words with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    # Clean up potential extra spaces left by removal
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    
    return cleaned_text

def preprocess_dataset(df):
    """Apply preprocessing to create all dataset variations."""
    # 1. Original dataset (keep a copy)
    original_df = df.copy()
    
    # 2. Fischer & Sitkin constructs only
    fischer_sitkin_df = filter_fischer_sitkin_constructs(df)
    
    # 3. Remove stems
    fischer_sitkin_no_stems_df = fischer_sitkin_df.copy()
    fischer_sitkin_no_stems_df['ProcessedText'] = fischer_sitkin_df['Text'].apply(remove_stems)
    
    # 4. Remove stems and gendered language
    fischer_sitkin_clean_df = fischer_sitkin_no_stems_df.copy()
    fischer_sitkin_clean_df['ProcessedText'] = fischer_sitkin_no_stems_df['ProcessedText'].apply(remove_gendered_language)
    
    return {
        'original': original_df,
        'fischer_sitkin': fischer_sitkin_df,
        'fischer_sitkin_no_stems': fischer_sitkin_no_stems_df,
        'fischer_sitkin_clean': fischer_sitkin_clean_df
    }

def save_datasets(datasets):
    """Save all dataset variations to CSV files."""
    for name, df in datasets.items():
        output_path = PROCESSED_DIR / f"leadership_measures_{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {name} dataset ({len(df)} items) to {output_path}")

def generate_dataset_report(datasets):
    """Generate a markdown report describing the datasets."""
    report = "# Leadership Measures Datasets\n\n"
    report += "## Dataset Variations\n\n"
    
    for name, df in datasets.items():
        report += f"### {name.replace('_', ' ').title()}\n\n"
        report += f"- Items: {len(df)}\n"
        report += f"- Constructs: {df['Behavior'].nunique()}\n"
        
        if 'fischer_sitkin' in name:
            construct_counts = df['Behavior'].value_counts().to_dict()
            report += "- Construct counts:\n"
            for construct, count in construct_counts.items():
                report += f"  - {construct}: {count} items\n"
        
        report += "\n"
    
    # Sample items section
    report += "## Sample Items\n\n"
    
    if 'fischer_sitkin_clean' in datasets:
        df = datasets['fischer_sitkin_clean']
        report += "### Examples of Text Processing\n\n"
        report += "| Original Text | Processed Text |\n"
        report += "|--------------|----------------|\n"
        
        # Get 5 random examples
        sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
        for idx in sample_indices:
            orig_text = datasets['fischer_sitkin']['Text'].iloc[idx]
            proc_text = df['ProcessedText'].iloc[idx]
            report += f"| {orig_text} | {proc_text} |\n"
    
    # Save report
    report_path = PROCESSED_DIR / "dataset_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Saved dataset report to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess leadership measures data")
    parser.add_argument("--data-file", type=str, help="Path to leadership measures data file")
    
    args = parser.parse_args()
    
    # Load data
    data_file = Path(args.data_file) if args.data_file else None
    leadership_df = load_leadership_data(data_file)
    
    # Preprocess data to create all variations
    datasets = preprocess_dataset(leadership_df)
    
    # Save datasets
    save_datasets(datasets)
    
    # Generate report
    generate_dataset_report(datasets)
    
    print("\nData preprocessing completed successfully.")
    print("Next steps:")
    print("1. Review the dataset_report.md file to ensure preprocessing is correct")
    print("2. Use the processed datasets for embedding analysis")

if __name__ == "__main__":
    main() 