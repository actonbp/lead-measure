# Quarto Documents Directory

This directory contains [Quarto](https://quarto.org/) documents for reproducible research reporting on the leadership measurement embeddings project.

## Purpose

Quarto documents in this directory serve to:
- Document analyses and findings in a reproducible format
- Generate publication-quality figures and tables
- Combine code (Python and R) with narrative explanations
- Prepare content for academic papers and presentations

## Expected Contents

- `.qmd` files with analysis code and narrative
- Output documents (HTML, PDF, Word)
- Supporting materials for documents
- Templates for consistent reporting

## Document Types

The directory includes several types of Quarto documents:

1. **Analysis Reports**
   - Detailed documentation of specific analyses
   - Embedding generation and comparison
   - Similarity analyses between leadership constructs

2. **Visualization Reports**
   - Specialized documents focusing on visualizing relationships
   - Interactive visualizations where appropriate
   - High-quality figures for publications

3. **Methodology Documents**
   - Documentation of analytical approaches
   - Comparison of embedding models
   - Validation of analytical techniques

4. **Paper Drafts**
   - Preliminary drafts of academic papers
   - Supplementary materials for submissions
   - Results summaries for collaborators

## Using Quarto

Quarto enables seamless integration of code (Python, R) with narrative text:
```
---
title: "Leadership Construct Analysis"
format: html
---

## Introduction

This document analyzes semantic similarities between leadership constructs.

```{python}
# Python code for analysis
import pandas as pd
# ... analysis code
```
```

## Getting Started

To work with these Quarto documents:
1. Install Quarto from [quarto.org](https://quarto.org/docs/get-started/)
2. Ensure you have Python and/or R environments configured
3. Render documents with `quarto render document.qmd`
4. For development, use `quarto preview document.qmd`

## Contributing

When contributing Quarto documents:
1. Use consistent formatting and style
2. Ensure all code chunks are properly documented
3. Include appropriate citations for methods and data
4. Make figures and tables publication-ready 