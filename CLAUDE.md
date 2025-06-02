# Claude Guidance for Leadership Measurement Analysis Project

## Project Goal and Approach

This project aims to determine if leadership measurement constructs are semantically distinct or redundant compared to established personality constructs (like IPIP).

The primary method involves:
1. Generating text embeddings for survey items using Sentence Transformer models (either fine-tuned local models or external API models).
2. Fine-tuning local models using contrastive loss functions (e.g., MultipleNegativesRankingLoss) to enhance construct separation in the embedding space.
3. Evaluating the separability using metrics like Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and cluster purity.

## Key Directories and Files

- `analyses/`: Analysis scripts and code
- `data/`: Raw and processed datasets
  - `raw/`: Original data files
  - `processed/`: Preprocessed datasets
  - `visualizations/`: Output visualizations
- `docs/`: Documentation and reports
- `scripts/`: Utility scripts
- `models/`: Contains model checkpoints (not tracked in git)

## Improved Workflow (2025 Update)

The current improved workflow consists of these steps:

1. **Generate comprehensive anchor-positive pairs**
   - Script: `scripts/build_ipip_pairs_improved.py`
   - Generates all possible within-construct pairs from IPIP items
   - Implements rebalancing to ensure fair representation across constructs
   - Output: `data/processed/ipip_pairs_comprehensive.jsonl`

2. **Train model with MultipleNegativesRankingLoss**
   - Script: `scripts/train_ipip_mnrl.py`
   - Uses the comprehensive pairs dataset
   - Applies MultipleNegativesRankingLoss for more effective training with limited data
   - Output: `models/ipip_mnrl/`

3. **Compare model performance** 
   - Script: `scripts/compare_model_performance.py`
   - Evaluates and compares different models on both IPIP and leadership data
   - Calculates clustering metrics (ARI, NMI, purity)
   - Outputs: Comparison metrics and visualizations

4. **Apply best model to leadership data**
   - Script: `scripts/apply_best_model_to_leadership.py`
   - Automatically selects the best performing model based on IPIP ARI score
   - Applies that model to leadership items
   - Generates visualizations and analysis of leadership construct clustering

## Current Status

- Phase 1 (Data Preprocessing and Initial Exploration) is complete
- Phase 2 (Comparative Analysis with Contrastive Learning) is complete with the improved workflow
- Phase 3 (Alternative Linguistic Properties) is the next focus

## Key Findings

1. Multiple analysis approaches (unsupervised clustering, supervised learning, contrastive learning) consistently show substantial semantic overlap between leadership constructs.
2. Most leadership styles map primarily to conscientiousness-related traits like "Dutifulness."
3. The MNRL model separated personality constructs with moderate success (ARI: 0.25) but performed poorly on leadership constructs (ARI: 0.07).
4. High similarity scores (>0.85) between many leadership construct pairs indicate they may be measuring the same underlying concepts.
5. Leadership styles do not form semantically distinct categories as currently measured in the literature.

## Current Research Focus (June 2025)

### Ivan's Enhanced Analysis Approach - READY TO RUN

We've implemented Ivan Hernandez's improved methodology that achieves dramatically better construct separation:

**Key Improvements:**
- Triple randomization in pair generation (eliminates ordering bias)
- TSDAE pre-training for domain adaptation
- BGE-M3 model specifically optimized for clustering
- Larger batch sizes (96) for better GIST loss performance
- Rigorous statistical validation

**Expected Results:**
- Baseline: 81.66% probability of correct same-construct ranking
- Ivan's approach: 99.43% probability (Cohen's d = 2.487)

### CURRENT STATUS (June 2, 2025):
✅ **Step 1 COMPLETED**: Generated 41,723 randomized pairs (`data/processed/ipip_pairs_randomized.jsonl`)
❌ **Step 2 FAILED**: Training stopped due to GPU memory limitations on current machine
⏭️ **NEXT ACTION REQUIRED**: Run on machine with more GPU memory (16GB+ recommended)

### IMMEDIATE NEXT STEPS FOR AI AGENT:

**Option A: Mac Studio M1/M2 64GB (OPTIMIZED - 4x faster training):**
```bash
source leadmeasure_env/bin/activate
make ivan-check
make ivan-step2-mac-studio    # OPTIMIZED for 64GB unified memory
make ivan-step3
make ivan-step4
```

**Option B: Other machines:**
```bash
source leadmeasure_env/bin/activate
make ivan-check
make ivan-step2    # Standard training
make ivan-step3
make ivan-step4
```

**Option C: Complete workflow:**
```bash
./scripts/run_ivan_analysis.sh
```

**Mac Studio Benefits:**
- **4x faster training**: 128 vs 32 batch size for GIST loss
- **64GB unified memory**: No CPU/GPU memory transfer overhead
- **Expected time**: 15-20 minutes vs 30-60 minutes
- **Same accuracy**: 99.43% construct separation

### Next Research Steps

1. Apply Ivan's enhanced model to leadership data to confirm/refute semantic overlap findings with higher confidence

2. Explore alternative taxonomies based on linguistic features:
   - Analyze linguistic complexity
   - Compare positive vs. negative framing
   - Examine agency vs. communion themes
   - Assess abstraction levels
   - Analyze target of behavior (leader, follower, organization)
   - Investigate temporal orientation

3. Apply dimensional reduction techniques to identify a more parsimonious set of leadership dimensions.

4. Create visualizations that better highlight the overlap patterns between constructs.

## Repository Management Notes

- Several directories contain large files that are not tracked in git:
  - `models/` - Contains model checkpoints and saved models (~172GB)
  - `leadmeasure_env/` - Python virtual environment 
  - `experiment_results/` - Contains experiment output files

- To manage disk space, use these cleanup scripts:
  - `scripts/cleanup_models.py` - Keep only the most recent model versions
  - `scripts/cleanup_data.py` - Remove redundant or temporary data files

## Research Recommendations

1. **Reconsider Measurement Approaches**: Leadership assessment could benefit from acknowledging the redundancy in current constructs and developing more focused, distinctive measurement approaches.

2. **Simplified Framework**: Consider a more parsimonious framework focusing on 2-3 broader leadership dimensions rather than 7-9 theoretically separate styles.

3. **Different Analytical Lens**: Future research should view leadership styles as different lenses or emphases on related underlying traits rather than discrete constructs.