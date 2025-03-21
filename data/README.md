# Data Directory

This directory contains all the datasets used in the leadership measurement embeddings project.

## Directory Structure

- [`raw/`](raw/README.md): Original, unmodified data sources
  - Leadership measurement scales in their original format
  - Original survey items and instruments
  - Source publications and documentation

- [`processed/`](processed/README.md): Cleaned and prepared datasets
  - Standardized format of leadership scales
  - Preprocessed text data ready for embedding generation
  - Combined datasets for analysis

- [`metadata/`](metadata/README.md): Information about the datasets
  - Documentation about each leadership construct
  - Scale properties and psychometric information
  - Citations and sources for each measure

## Data Sources

The project compiles leadership measurement scales from various sources, including:

- Published leadership scales (ethical leadership, transformational leadership, etc.)
- Survey instruments from leadership research
- Item banks from leadership assessment tools

## Data Format

Processed data is primarily stored in structured formats (CSV, JSON) with the following information:
- Scale name/construct
- Item text
- Item number
- Response format
- Source information

## Adding New Data

When adding new leadership measures to the repository:
1. Place original files in the `raw/` directory
2. Document the source in the metadata directory
3. Process the data into standard format and save to `processed/`
4. Update this README if you're adding a new category of data 