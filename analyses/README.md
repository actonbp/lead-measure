# Analyses Directory

This directory contains all code related to analyzing leadership measurement scales using embedding approaches.

## Directory Structure

- [`python/`](python/README.md): Python scripts and notebooks
  - Embedding generation code
  - Semantic similarity analysis
  - Dimensionality reduction and visualization
  - Machine learning models

- [`r/`](r/README.md): R scripts for statistical analyses
  - Psychometric analysis
  - Comparative statistical tests
  - Visualization of results
  - Traditional construct validation approaches

## Analysis Workflow

The typical analysis workflow includes:

1. Loading processed leadership scale data
2. Generating embeddings using various models (API-based and local)
3. Computing semantic similarity between items and constructs
4. Identifying patterns and clusters of leadership constructs
5. Comparing embedding-based results with traditional psychometric approaches
6. Visualizing the relationships between leadership constructs

## Embedding Models

The project utilizes multiple embedding models, including:
- Commercial API models (OpenAI, Anthropic, etc.)
- Open-source models (Llama, etc.)
- Specialized language models for organizational/leadership domains

## Key Analysis Techniques

- Cosine similarity measurement
- Dimensionality reduction (t-SNE, UMAP)
- Clustering techniques
- Network analysis of construct relationships
- Comparative analysis with traditional factor structures

## Adding New Analyses

When adding new analyses:
1. Use clear, descriptive filenames
2. Include comprehensive documentation and comments
3. Add requirements or dependencies to the project's requirements file
4. Update this README if you're adding a new category of analysis 