# Python Analyses Directory

This directory contains Python scripts and notebooks for embedding-based analysis of leadership measurement scales.

## Directory Structure

- `preprocessing/`: Scripts for data cleaning and standardization
- `embedding/`: Code for generating embeddings from various models
- `similarity/`: Analysis of semantic similarities between items and constructs
- `visualization/`: Code for visualizing embedding spaces and relationships
- `clustering/`: Scripts for identifying clusters of similar constructs
- `notebooks/`: Jupyter notebooks with exploratory analyses and results

## Key Features

The Python code in this directory enables:
- Generation of embeddings from leadership measurement items
- Computation of similarity metrics between items and scales
- Identification of semantic clusters among leadership constructs
- Comparison of different embedding models for leadership analysis
- Visualization of semantic relationships between constructs

## Required Dependencies

Main dependencies include:
- numpy, pandas, scikit-learn
- Embedding libraries (transformers, sentence-transformers)
- API clients for commercial embedding services
- Visualization tools (matplotlib, plotly, seaborn)
- Dimensionality reduction (UMAP, t-SNE)

A complete list of dependencies is available in the project's `requirements.txt` file.

## Key Workflows

1. **Preprocessing Pipeline**
   - Load raw data from leadership scales
   - Standardize text formatting
   - Prepare items for embedding generation

2. **Embedding Generation**
   - Generate embeddings using multiple models
   - Store embeddings in standardized formats
   - Compare embedding quality across models

3. **Similarity Analysis**
   - Compute item-to-item similarities
   - Analyze scale-level semantic relationships
   - Identify redundant or overlapping constructs

4. **Visualization and Reporting**
   - Generate visualizations of embedding spaces
   - Create summary reports of findings
   - Prepare figures for publications

## Getting Started

To begin working with the Python code:
1. Set up a virtual environment
2. Install dependencies from `requirements.txt`
3. Configure API keys for commercial embedding services
4. Start with the example notebooks in the `notebooks/` directory 