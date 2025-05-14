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

## GIST Loss Training Approach

We're using the GIST (Guided In-batch Similarity Training) loss approach to train our embedding model:

1. First, we train a model on IPIP personality items to learn how personality constructs are structured
2. Then, we apply this pretrained model to leadership items to see if it can effectively distinguish leadership constructs

### Running the Scripts

#### Step 1: Generate IPIP pairs
```bash
python scripts/build_ipip_pairs.py
```
This creates anchor-positive pairs from IPIP personality items for GIST loss training.

#### Step 2: Train and evaluate IPIP model
```bash
python scripts/evaluate_ipip_model.py
```
This script:
- Loads the IPIP dataset and splits it into train (80%) and test (20%) sets
- Trains a GIST model on the train set
- Evaluates how well the model clusters test set items by personality construct
- Generates visualizations and metrics showing clustering quality

#### Step 3: Train full IPIP model
```bash
python scripts/train_gist_ipip.py
```
This trains a model on all IPIP data using GIST loss for later application to leadership items.

#### Step 4: Apply model to leadership items
```bash
python scripts/apply_gist_to_leadership.py
```
This script:
- Applies the trained GIST model to leadership items
- Clusters leadership items and evaluates if constructs are semantically distinct
- Generates visualizations and metrics comparing predicted clusters to true constructs
- Tests the hypothesis that leadership constructs have substantial semantic overlap

### Hardware Requirements

- Training requires at least 16GB RAM
- GPU acceleration is recommended but not required

## Important Note on Large Directories

Several directories contain large files that are not tracked in git:

- `models/` - Contains model checkpoints and saved models (~172GB)
- `leadmeasure_env/` - Python virtual environment 
- `experiment_results/` - Contains experiment output files

These directories are in `.gitignore` and should not be committed. When running the scripts, the required directories will be created automatically if they don't exist.

## Setting Up on a New Computer

Follow these steps to set up and run the project on a new machine:

1. **Clone the repository**
   ```bash
   git clone https://github.com/actonbp/lead-measure.git
   cd lead-measure
   ```

2. **Set up the Python environment**
   ```bash
   # Create a virtual environment
   python -m venv leadmeasure_env
   
   # Activate the virtual environment
   # On Windows:
   leadmeasure_env\Scripts\activate
   # On macOS/Linux:
   source leadmeasure_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the scripts in order** as described in the "Running the Scripts" section above:
   - First generate IPIP pairs
   - Then train and evaluate the model
   - Finally apply to leadership items

All necessary directories will be created automatically by the scripts.

## Results and Implications

The results of this analysis will have important implications for:
- Leadership theory development
- Measurement practices in leadership research
- Potential redundancies in leadership constructs
- Future directions for leadership assessment

## Future Directions / To-Do

- Analyze the potential impact of outdated language (e.g., 'foreman' in LBDQ) on embedding results and construct similarity.

## Contributors

- Bryan Acton
- Steven Zhou

## License

MIT License
