# Leadership Measurement Analysis

## Project Overview

This project explores the semantic structure of leadership measurement using natural language processing and embedding techniques. The central research question is: **Do leadership constructs represent genuinely distinct dimensions, or are they largely redundant in how they are measured?**

## Research Approach

This project follows a multi-stage approach to analyzing leadership measurement:

### Phase 1: Data Preprocessing and Initial Exploration (Completed)

- Collecting leadership measurement items from multiple constructs (transformational, ethical, servant, etc.)
- Preprocessing texts to remove leader references (stems) and neutralize gendered language
- Generating embeddings using transformer-based models
- Visualizing semantic relationships through dimensionality reduction
- Calculating similarity metrics between constructs

Initial findings suggest substantial overlap between leadership constructs in semantic space, raising questions about their distinctiveness.

### Phase 2: Comparative Analysis with Triplet Loss Training (Planned)

The next phase will use triplet loss training to directly compare:
1. **Big Five Personality Constructs** (extraversion, neuroticism, agreeableness, conscientiousness, openness)
2. **Leadership Constructs** (transformational, ethical, servant, authentic, etc.)

Triplet loss training will teach models to distinguish between items from different constructs. Our hypothesis is that:
- Triplet loss will successfully separate Big Five personality constructs
- Triplet loss will fail to separate leadership constructs

This will provide a rigorous test of whether leadership constructs represent meaningfully distinct dimensions.

### Phase 3: Alternative Linguistic Properties (Planned)

If leadership constructs don't separate items effectively, what does? The final phase will explore:
- What linguistic properties distinguish leadership items if not construct membership
- Alternative taxonomies based on linguistic features like:
  - Linguistic complexity
  - Positive vs. negative framing
  - Agency vs. communion themes
  - Abstraction levels
  - Target of behavior (leader, follower, organization)
  - Temporal orientation

## Directory Structure

- `analyses/`: Analysis scripts and code
- `data/`: Raw and processed datasets
  - `raw/`: Original data files
  - `processed/`: Preprocessed datasets
  - `embeddings/`: Generated embedding files
  - `visualizations/`: Output visualizations
- `docs/`: Documentation and reports
- `scripts/`: Utility scripts

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with any required API keys
4. Run preprocessing: `python analyses/generate_leadership_datasets.py`
5. Generate embeddings: `python analyses/generate_leadership_embeddings.py`

## Results and Implications

The results of this analysis will have important implications for:
- Leadership theory development
- Measurement practices in leadership research
- Potential redundancies in leadership constructs
- Future directions for leadership assessment

## Contributors

- Bryan Acton
- Steven Zhou

## License

MIT License
