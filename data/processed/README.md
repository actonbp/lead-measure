# Processed Leadership Datasets

This directory contains the processed versions of leadership measurement datasets, created through various preprocessing steps to support different analytical approaches.

## Dataset Variations

We created six different dataset variations through combinations of:
1. **Scope**: Complete dataset vs. Fischer & Sitkin (2023) core constructs only
2. **Stem Removal**: Original stems vs. removed stems (e.g., "My supervisor...")
3. **Gender Neutralization**: Original gendered language vs. neutralized language

### Complete Datasets

| Filename | Description | Items | Constructs |
|----------|-------------|-------|------------|
| `leadership_original.csv` | Complete dataset with all items | 829 | 38 |
| `leadership_original_no_stems.csv` | Complete dataset with stems removed | 829 | 38 |
| `leadership_original_clean.csv` | Complete dataset with stems removed and gender-neutral language | 829 | 38 |

### Focused Datasets (Fischer & Sitkin Constructs)

| Filename | Description | Items | Constructs |
|----------|-------------|-------|------------|
| `leadership_focused.csv` | Focused dataset with only core constructs | 340 | 14 |
| `leadership_focused_no_stems.csv` | Focused dataset with stems removed | 340 | 14 |
| `leadership_focused_clean.csv` | Focused dataset with stems removed and gender-neutral language | 340 | 14 |

## Core Leadership Constructs

The focused datasets include items from the following leadership constructs identified by Fischer & Sitkin (2023):

| Construct | Count | Example Item |
|-----------|-------|--------------|
| Abusive | 8 | Ridicules me |
| Authentic | 14 | Says exactly what they mean |
| Charismatic | 25 | Communicates a clear vision of the future |
| Empowering | 17 | Allows me to make important decisions quickly |
| Ethical | 80 | Can be trusted to do the things they promise |
| Instrumental | 16 | Ensures procedures and processes support goal accomplishment |
| Servant | 71 | Puts my best interests ahead of their own |
| Transformational | 109 | Articulates a compelling vision of the future |

## Preprocessing Steps

### Stem Removal
Many leadership items begin with phrases like "My supervisor..." or "The leader...". We removed these stems to focus on the core behavioral content. Examples:

| Original Item | Processed Item |
|---------------|----------------|
| My supervisor encourages me when I encounter problems. | Encourages me when I encounter problems. |
| The leader sets a good example for the team. | Sets a good example for the team. |

### Gender Neutralization
We converted gendered language to gender-neutral alternatives to reduce bias. Examples:

| Original Text | Gender-Neutral Text |
|---------------|---------------------|
| Conducts his/her personal life in an ethical manner | Conducts their personal life in an ethical manner |
| He provides me with assistance in exchange for my efforts | They provide me with assistance in exchange for my efforts |

## Dataset Statistics

| Dataset | Items | Constructs | Gender-Neutral | Stems Removed |
|---------|-------|------------|----------------|---------------|
| Original | 829 | 38 | No | No |
| Original No Stems | 829 | 38 | No | Yes |
| Original Clean | 829 | 38 | Yes | Yes |
| Focused | 340 | 14 | No | No |
| Focused No Stems | 340 | 14 | No | Yes |
| Focused Clean | 340 | 14 | Yes | Yes |

## Usage

These datasets form the foundation for the embedding analysis and triplet loss training described in the main project README. The "clean" versions (with stems removed and gender-neutralized) are recommended for most analyses to reduce noise and bias in the semantic representations. 