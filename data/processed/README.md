# Processed Data Directory

This directory contains standardized and processed versions of leadership measurement scales prepared for analysis.

## Purpose

The processed data directory contains leadership measurement items and scales that have been:
- Converted to standardized formats
- Cleaned and preprocessed for analysis
- Organized for efficient use in embedding generation and analysis

## Expected Contents

- CSV files with standardized leadership scale items
- JSON files with structured measure information
- Combined datasets integrating multiple leadership constructs
- Pre-computed embedding matrices (if applicable)

## Data Structure

Processed data files typically follow this structure:

```
| scale_id | construct | item_id | item_text | reverse_coded | sub_dimension | source |
|----------|-----------|---------|-----------|---------------|---------------|--------|
| ELS      | Ethical   | ELS1    | "Item..." | FALSE         | NULL          | Brown  |
```

## Standardization Process

Data in this directory has undergone:
1. Text standardization (consistent capitalization, punctuation)
2. Format standardization (consistent file formats and data structures)
3. Variable naming standardization (consistent column names)
4. Consolidation of related measures where appropriate

## Data Dictionary

Essential fields in processed data:
- `scale_id`: Unique identifier for the measurement scale
- `construct`: Leadership construct being measured
- `item_id`: Unique identifier for each item
- `item_text`: The actual text of the measurement item
- `reverse_coded`: Indicator if item is reverse-coded
- `sub_dimension`: Sub-factor or dimension (if applicable)
- `source`: Citation information for the source

## Processing Scripts

The scripts used to process raw data into these standardized formats are located in the `analyses/python/preprocessing/` directory. 