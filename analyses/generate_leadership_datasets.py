#!/usr/bin/env python3
"""
Generate Leadership Datasets

This script generates various leadership datasets based on specified criteria:
1. A focused dataset containing only items from 10 specific leadership constructs
2. A complete dataset with all items
3. Clean versions of these datasets with stems removed and gendered language neutralized

Usage:
    python generate_leadership_datasets.py

Output:
    Multiple CSV files saved to data/processed/ with different dataset variations
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from preprocess_leadership_data import (
    remove_stems, 
    remove_gendered_language, 
    FISCHER_SITKIN_CONSTRUCTS,
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DIR
)

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define dataset variations to create
DATASET_VARIATIONS = {
    'original': 'Complete dataset with all items',
    'original_no_stems': 'Complete dataset with stems removed',
    'original_clean': 'Complete dataset with stems removed and gender-neutral language',
    'focused': 'Focused dataset with only Fischer & Sitkin (2023) constructs',
    'focused_no_stems': 'Focused dataset with stems removed',
    'focused_clean': 'Focused dataset with stems removed and gender-neutral language'
}

def load_data():
    """Load the leadership measures dataset."""
    file_path = DATA_DIR / "Measures_text_long.csv"
    
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
    """Check if a behavior belongs to Fischer & Sitkin (2023) constructs."""
    return any(
        term.lower() in behavior.lower() 
        for construct_terms in FISCHER_SITKIN_CONSTRUCTS.values() 
        for term in construct_terms
    )

def map_to_standard_construct(behavior):
    """Map behavior to standardized construct name."""
    for construct, terms in FISCHER_SITKIN_CONSTRUCTS.items():
        if any(term.lower() in behavior.lower() for term in terms):
            return construct
    return behavior  # Fallback to original if no match

def generate_datasets(df):
    """Generate all dataset variations."""
    datasets = {}
    
    # 1. Complete dataset (original)
    datasets['original'] = df.copy()
    
    # 2. Complete dataset with stems removed
    datasets['original_no_stems'] = df.copy()
    datasets['original_no_stems']['ProcessedText'] = df['Text'].apply(remove_stems)
    
    # 3. Complete dataset with stems removed and gender-neutral language
    datasets['original_clean'] = datasets['original_no_stems'].copy()
    datasets['original_clean']['ProcessedText'] = datasets['original_no_stems']['ProcessedText'].apply(remove_gendered_language)
    
    # 4. Create focused dataset with Fischer & Sitkin constructs
    mask = df['Behavior'].apply(is_fischer_sitkin_construct)
    focused_df = df[mask].copy()
    focused_df['StandardConstruct'] = focused_df['Behavior'].apply(map_to_standard_construct)
    datasets['focused'] = focused_df
    
    # 5. Focused dataset with stems removed
    datasets['focused_no_stems'] = focused_df.copy()
    datasets['focused_no_stems']['ProcessedText'] = focused_df['Text'].apply(remove_stems)
    
    # 6. Focused dataset with stems removed and gender-neutral language
    datasets['focused_clean'] = datasets['focused_no_stems'].copy()
    datasets['focused_clean']['ProcessedText'] = datasets['focused_no_stems']['ProcessedText'].apply(remove_gendered_language)
    
    return datasets

def save_datasets(datasets):
    """Save all dataset variations to CSV files."""
    for name, df in datasets.items():
        output_path = PROCESSED_DIR / f"leadership_{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {name} dataset ({len(df)} items) to {output_path}")

def generate_dataset_report(datasets):
    """Generate a markdown report describing the datasets."""
    report = "# Leadership Datasets Report\n\n"
    report += "## Dataset Variations\n\n"
    
    for name, df in datasets.items():
        report += f"### {name.replace('_', ' ').title()}\n\n"
        report += f"- Description: {DATASET_VARIATIONS[name]}\n"
        report += f"- Items: {len(df)}\n"
        report += f"- Constructs: {df['Behavior'].nunique()}\n"
        
        if 'focused' in name:
            construct_counts = df['StandardConstruct'].value_counts().to_dict()
            report += "- Construct counts:\n"
            for construct, count in sorted(construct_counts.items()):
                report += f"  - {construct}: {count} items\n"
        
        report += "\n"
    
    # Sample items section
    report += "## Sample Items\n\n"
    
    # Examples of original vs processed text
    for dataset_name in ['original_clean', 'focused_clean']:
        if dataset_name in datasets:
            df = datasets[dataset_name]
            report += f"### Examples from {dataset_name.replace('_', ' ').title()}\n\n"
            report += "| Original Text | Processed Text |\n"
            report += "|--------------|----------------|\n"
            
            # Get 5 random examples with complete texts
            np.random.seed(42)  # Set seed for reproducibility
            sample_indices = []
            candidates = np.random.choice(len(df), min(20, len(df)), replace=False)
            
            # Select cleaner examples (not too long, not too short)
            for idx in candidates:
                if len(df['Text'].iloc[idx]) > 10 and len(df['Text'].iloc[idx]) < 150:
                    sample_indices.append(idx)
                    if len(sample_indices) >= 5:
                        break
            
            if len(sample_indices) < 5:  # If we didn't get 5 good examples, use the first 5
                sample_indices = candidates[:5]
            
            for idx in sample_indices:
                orig_text = df['Text'].iloc[idx]
                # Check if ProcessedText exists, otherwise use Text
                if 'ProcessedText' in df.columns:
                    proc_text = df['ProcessedText'].iloc[idx]
                else:
                    proc_text = orig_text
                
                # Escape any pipe characters that could break the markdown table
                orig_text = orig_text.replace('|', '\\|')
                proc_text = proc_text.replace('|', '\\|')
                
                report += f"| {orig_text} | {proc_text} |\n"
            
            report += "\n"
    
    # Save report
    report_path = PROCESSED_DIR / "leadership_datasets_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Saved dataset report to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate leadership datasets")
    args = parser.parse_args()
    
    # Load data
    leadership_df = load_data()
    
    # Generate datasets
    datasets = generate_datasets(leadership_df)
    
    # Save datasets
    save_datasets(datasets)
    
    # Generate report
    generate_dataset_report(datasets)
    
    print("\nDataset generation completed successfully.")
    print("Next steps:")
    print("1. Review the leadership_datasets_report.md file")
    print("2. Use the processed datasets for embedding analysis")

if __name__ == "__main__":
    main() 