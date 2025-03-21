# Leadership Measurement Data Processing

This report documents the data preprocessing steps for analyzing leadership measurement text using embedding models. We focused on standardizing and cleaning leadership measurement items to enable better semantic analysis.

## Research Context

Leadership research has produced numerous overlapping constructs and measurement scales. This proliferation creates challenges for understanding the fundamental dimensions of leadership behavior. Our approach uses natural language processing to analyze the semantic space of leadership measurements, potentially identifying redundancies across constructs.

## Data Source and Processing

The original dataset (`Measures_text_long.csv`) contains 829 leadership measurement items across 38 different leadership constructs. Each item is associated with a leadership behavior/construct (e.g., Transformational, Ethical) and often with sub-dimensions.

### Data Processing Steps

1. **Identification of key constructs**: Based on Fischer & Sitkin (2023), we focused on 10 leadership constructs, including positive styles (Authentic, Charismatic, Empowering, Ethical, etc.) and negative styles (Abusive, Destructive).

2. **Stem removal**: Many leadership items begin with phrases like "My supervisor..." or "The leader...". These stems add noise to semantic analysis, so we systematically removed them using regular expressions.

3. **Gender neutralization**: We converted gendered language to gender-neutral alternatives to reduce potential bias and improve semantic analysis:
   - Replacing combined forms like "his/her" → "their"
   - Converting individual gendered terms (e.g., "chairman" → "chairperson")

4. **Dataset variations**: Multiple dataset versions were created for comparative analysis.

## Dataset Variations

### Original Complete Datasets

- **Original Dataset**: 829 items, 38 constructs - All leadership items
- **Original No Stems**: 829 items, 38 constructs - Leadership items with reference prefixes removed
- **Original Clean**: 829 items, 38 constructs - Items with stems removed and gender-neutral language

### Focused Fischer & Sitkin Datasets

- **Focused Dataset**: 340 items, 14 constructs - Only theoretically significant leadership constructs
- **Focused No Stems**: 340 items, 14 constructs - Key constructs without reference prefixes
- **Focused Clean**: 340 items, 14 constructs - Fully preprocessed key constructs

### Distribution of Items Across Constructs

The Fischer & Sitkin focused datasets contain the following distribution:

| Construct | Count |
|-----------|-------|
| Abusive | 8 items |
| Authentic | 14 items |
| Charismatic | 25 items |
| Empowering | 17 items |
| Ethical | 80 items |
| Instrumental | 16 items |
| Servant | 71 items |
| Transformational | 109 items |

## Text Processing Examples

### Stem Removal Examples

| Original Text | Processed Text |
|--------------|----------------|
| My supervisor encourages me when I encounter arduous problems. | Encourages me when I encounter arduous problems. |
| My department manager holds department employees to high ethical standards. | Holds department employees to high ethical standards. |
| The leader sets a good example for the team. | Sets a good example for the team. |

### Gender Neutralization Examples

| Original Text | Gender-Neutral Text |
|--------------|----------------|
| Conducts his/her personal life in an ethical manner | Conducts their personal life in an ethical manner |
| He provides me with assistance in exchange for my efforts | They provide me with assistance in exchange for my efforts |
| Our supervisor speaks about his vision for the future | Our supervisor speaks about their vision for the future |

## Sample Items From Processed Datasets

### Examples from Original Clean

| Original Text | Processed Text |
|--------------|----------------|
| My supervisor encourages me when I encounter arduous problems. | Encourages me when I encounter arduous problems. |
| I experience the following HR practices as being implemented to support me: appraisal | I experience the following HR practices as being implemented to support me: appraisal |
| My department manager holds department employees to high ethical standards. | Holds department employees to high ethical standards. |
| Talks enthusiastically about what needs to be accomplished by our team. | Talks enthusiastically about what needs to be accomplished by our team. |
| Makes fair and balanced decisions | Makes fair and balanced decisions |

### Examples from Focused Clean

| Original Text | Processed Text |
|--------------|----------------|
| Says positive things about the team. | Says positive things about the team. |
| Explains what is expected of each member of the group | Explains what is expected of each member of the group |
| Does not criticize subordinates without good reason | Does not criticize subordinates without good reason |
| differentiates among us | differentiates among us |
| Conducts his/her personal life in an ethical manner | Conducts their personal life in an ethical manner |

## Technical Implementation

The preprocessing was implemented in Python with:
- Regular expressions for pattern matching and stem removal
- Dictionary-based approach for gender neutralization
- Keyword matching for construct identification

The original texts are preserved alongside processed versions, allowing for comparison and validation. 