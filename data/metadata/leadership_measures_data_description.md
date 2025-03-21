# Leadership Measures Dataset Documentation

## Overview

The leadership measures dataset (`Measures_text_long.csv`) contains a comprehensive collection of leadership measurement scales from various sources. The dataset includes multiple leadership constructs (e.g., Transformational, Transactional, Ethical, Servant, Empowering, Charismatic) with their associated items and sub-dimensions.

## Data Structure

The dataset follows a long-format structure with the following columns:

| Column | Description |
|--------|-------------|
| MeasureID | Unique identifier for the measurement instrument (1-100) |
| Behavior | Primary leadership construct/behavior being measured (e.g., Transformational, Ethical) |
| Dimensions | Sub-dimension or factor within the construct (e.g., "Contingent Reward" within Transactional leadership) |
| Item | Item identifier within the measure (e.g., Item1, Item2) |
| Xnum | Sequential numeric identifier across all items (1-888) |
| Text | The actual text content of the measurement item |

## Dataset Statistics

- Total number of unique measures (MeasureIDs): 100
- Total number of unique leadership behaviors/constructs: 88
- Total number of measurement items: 888
- File format: CSV (Comma-Separated Values)

## Major Leadership Constructs

The dataset includes, but is not limited to, the following leadership constructs:

1. **Transformational Leadership**
   - Sub-dimensions: Articulates Vision, Provide Appropriate Model, Fostering Acceptance of Group Goals, High Performance Expectations, Individual Support, Intellectual Stimulation, Charisma/Inspirational, Individualized Consideration

2. **Transactional Leadership**
   - Sub-dimensions: Contingent Reward, Management-by-Exception-Active

3. **Ethical Leadership**
   - Focuses on ethical standards, fairness, and integrity

4. **Servant Leadership**
   - Emphasizes serving others and supporting followers' growth

5. **Empowering Leadership/Behavior**
   - Sub-dimensions: Delegation of Authority, Accountability, Self-directed Decision Making, Information Sharing, Skill Development, Coaching for Innovative Performance

6. **Charismatic Leadership**
   - Sub-dimensions: Vision and Articulation, Environmental Sensitivity, Unconventional Behavior, Personal Risk

7. **Ambidextrous Leadership**
   - Sub-dimensions: Opening, Closing

8. **People Management**
   - Sub-dimensions: Supportive HR Practices, Implementation Tailor-made Arrangements, Support of Employees' Commitment, Support of Employees' Career Development, Ability, Commitment, Autonomy, Extra-role Behavior

## Usage Notes

- Some items may be reverse-coded (negative wording), which would need to be addressed in analyses
- The dataset combines multiple leadership measurement scales into a unified format
- The Xnum field provides a sequential identifier that can be useful for sorting or tracking items across the entire dataset

## Research Applications

This dataset is particularly useful for:

1. Comparing linguistic similarities and differences between leadership constructs
2. Identifying potential redundancies in leadership measurement
3. Analyzing the semantic space of leadership measurement
4. Developing more parsimonious leadership measurement approaches
5. Exploring how different leadership constructs relate to each other at the item level

## Embedding-Based Analysis Potential

The text of these leadership measurement items can be transformed into embeddings to:

1. Measure semantic similarity between items across different leadership constructs
2. Identify clusters of semantically related leadership dimensions
3. Quantify the degree of construct redundancy based on language use
4. Create visualizations of the leadership construct semantic space
5. Compare embedding-based relationships with traditional factor structures 