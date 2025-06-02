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

### Phase 2: Comparative Analysis with Contrastive Learning (In Progress)

The current phase uses contrastive learning approaches to directly compare:
1. **Big Five Personality Constructs** (extraversion, neuroticism, agreeableness, conscientiousness, openness)
2. **Leadership Constructs** (transformational, ethical, servant, authentic, etc.)

We've implemented multiple contrastive learning approaches including:
- **Multiple Negatives Ranking Loss (MNRL)**: Treats other batch items as negatives, efficient for small datasets
- **Triplet Loss**: Learns from triplets of anchor, positive, and negative examples
- **Contrastive Loss**: Traditional contrastive learning with positive/negative pairs

Our hypothesis is that:
- Contrastive learning will successfully separate Big Five personality constructs
- Contrastive learning will fail to separate leadership constructs

This provides a rigorous test of whether leadership constructs represent meaningfully distinct dimensions.

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

## 🚀 Getting Started - Ivan's Enhanced Analysis (June 2025)

### Current Status:
- ✅ **Step 1 COMPLETED**: 41,723 randomized pairs generated
- ❌ **Step 2 FAILED**: Training needs machine with 16GB+ GPU memory
- ⏭️ **READY TO CONTINUE**: All setup complete, environment ready

### Quick Start on New Machine:
```bash
# 1. Activate environment
source leadmeasure_env/bin/activate

# 2. Check setup
python scripts/ivan_analysis/run_analysis_steps.py --check

# 3. Run the critical training step
make ivan-step2

# 4. Complete analysis
make ivan-step3
make ivan-step4
```

### Expected Results:
- **99.43%** construct separation accuracy (vs 81.66% baseline)
- **Cohen's d = 2.487** (massive effect size)
- Clear t-SNE visualizations showing construct relationships

## GIST Loss Training Approach

We're using the GIST (Guided In-batch Similarity Training) loss approach to train our embedding model:

1. First, we train a model on IPIP personality items to learn how personality constructs are structured
2. Then, we apply this pretrained model to leadership items to see if it can effectively distinguish leadership constructs

### Running the Scripts (Original Approach)

> **Note**: These scripts are now in the `scripts/archive` folder and are kept for reference only.
> For the current improved workflow, see the "Improved Workflow (2025 Update)" section below.

#### Step 1: Generate IPIP pairs
```bash
python scripts/archive/build_ipip_pairs.py
```
This creates anchor-positive pairs from IPIP personality items for GIST loss training.

#### Step 2: Train and evaluate IPIP model
```bash
python scripts/archive/evaluate_ipip_model.py
```
This script:
- Loads the IPIP dataset and splits it into train (80%) and test (20%) sets
- Trains a GIST model on the train set
- Evaluates how well the model clusters test set items by personality construct
- Generates visualizations and metrics showing clustering quality

#### Step 3: Train full IPIP model
```bash
python scripts/archive/train_gist_ipip.py
```
This trains a model on all IPIP data using GIST loss for later application to leadership items.

#### Step 4: Apply model to leadership items
```bash
python scripts/archive/apply_gist_to_leadership.py
```
This script:
- Applies the trained GIST model to leadership items
- Clusters leadership items and evaluates if constructs are semantically distinct
- Generates visualizations and metrics comparing predicted clusters to true constructs
- Tests the hypothesis that leadership constructs have substantial semantic overlap

## Improved Workflow (2025 Update)

After identifying performance bottlenecks in our initial approach, we've developed an improved workflow that addresses limitations in data quality, balanced training, and loss function selection.

### Key Improvements

1. **Comprehensive Pair Generation**: Generate all possible within-construct pairs
2. **Balanced Training Data**: Prevent over/under-representation of constructs 
3. **Alternative Loss Functions**: Use MultipleNegativesRankingLoss for better results with limited data

### Running the Improved Workflow

Run the complete improved workflow with:

```bash
chmod +x scripts/run_improved_workflow.sh
./scripts/run_improved_workflow.sh
```

Or run the steps individually for more control:

1. **Generate comprehensive pairs**:
   ```bash
   python scripts/build_ipip_pairs_improved.py
   ```
   This creates balanced anchor-positive text pairs from IPIP items.

2. **Train with configurable loss functions**:
   ```bash
   # Default (MNRL - Multiple Negatives Ranking Loss)
   python scripts/train_ipip_mnrl.py
   
   # With different loss function
   python scripts/train_ipip_mnrl.py --loss_fn triplet
   
   # With more epochs
   python scripts/train_ipip_mnrl.py --epochs 15
   
   # Full help
   python scripts/train_ipip_mnrl.py --help
   ```

3. **Evaluate model performance**:
   ```bash
   # Evaluate the most recent model
   python scripts/evaluate_trained_ipip_model.py
   
   # Evaluate specific model
   python scripts/evaluate_trained_ipip_model.py --model_path models/ipip_mnrl_20250515_1328
   ```

4. **Apply to leadership data**:
   ```bash
   python scripts/apply_best_model_to_leadership.py
   ```

The evaluation includes comprehensive metrics and visualizations to analyze model performance.

For detailed documentation on the improved approach, see [docs/improved_workflow.md](docs/improved_workflow.md)

### Hardware Requirements

- Training requires at least 16GB RAM
- GPU acceleration is recommended but not required

## Important Note on Large Directories

Several directories contain large files that are not tracked in git:

- `models/` - Contains model checkpoints and saved models (~172GB)
- `leadmeasure_env/` - Python virtual environment 
- `experiment_results/` - Contains experiment output files

These directories are in `.gitignore` and should not be committed. When running the scripts, the required directories will be created automatically if they don't exist.

To manage disk space, you can use these cleanup scripts:

### Model Cleanup
Keep only the most recent model versions:
```bash
# Show what would be deleted without deleting anything
python scripts/cleanup_models.py --dry-run

# Keep only the most recent model of each type
python scripts/cleanup_models.py

# Keep the 2 most recent models of each type
python scripts/cleanup_models.py --keep 2
```

### Data Cleanup
Remove redundant or temporary data files:
```bash
# Show what would be deleted without deleting anything
python scripts/cleanup_data.py --dry-run

# Perform basic cleanup (keeps essential files)
python scripts/cleanup_data.py

# Perform aggressive cleanup (only keeps raw data and final results)
python scripts/cleanup_data.py --all
```

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
