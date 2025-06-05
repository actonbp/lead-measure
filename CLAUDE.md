# Claude Guidance for Leadership Measurement Analysis Project

## AI Development Guidelines

### Repository Management Principles
- **Avoid script proliferation**: Always use existing scripts rather than creating new ones
- **Extend, don't duplicate**: Enhance existing functionality instead of rewriting
- **Maintain unified pipeline**: Prefer the consolidated `scripts/ivan_analysis/` approach
- **Document thoughtfully**: Update existing documentation rather than creating separate files
- **Clean as you go**: Use established cleanup procedures to maintain organization

### Code Development Best Practices
- **Check before creating**: Search for existing implementations before writing new code
- **Preserve functionality**: When modifying scripts, ensure backward compatibility
- **Follow patterns**: Match existing code style, naming conventions, and structure
- **Test incrementally**: Validate changes against existing workflows
- **Minimize complexity**: Choose simple solutions that integrate with current architecture

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

### CURRENT STATUS (June 3, 2025):
✅ **Step 1 COMPLETED**: Generated 41,723 randomized pairs (`data/processed/ipip_pairs_randomized.jsonl`)
✅ **Step 2 COMPLETED**: Ivan's enhanced training completed successfully on Mac Studio
✅ **Step 3 COMPLETED**: Analysis and visualization generation completed  
✅ **Step 4 COMPLETED**: Baseline comparison and model evaluation completed
✅ **Leadership Analysis COMPLETED**: Applied Ivan's model to leadership constructs
✅ **Holdout Validation COMPLETED**: Created proper train/test splits and re-trained model
✅ **Unified Pipeline COMPLETED**: Created platform-agnostic pipeline with background execution

### FINAL VALIDATED FINDINGS (June 4, 2025):

**IPIP Holdout Constructs (Never Seen During Training):**
- **87.4%** accuracy in construct separation (Cohen's d = 1.116)
- 50 holdout constructs tested on trained model
- Statistically significant: t = 23.056, p < 2.22e-16

**Leadership Constructs:**
- **62.9%** accuracy in construct separation (Cohen's d = 0.368) 
- **24.5 percentage point difference** compared to personality constructs
- **Critical insight**: Leadership constructs are significantly more semantically overlapping than personality constructs

**Research Implications:**
- Provides **unbiased empirical evidence** for construct proliferation concerns in leadership research
- Even with rigorous holdout validation, leadership constructs show poor semantic separation
- Suggests many leadership theories may be measuring similar underlying dimensions
- Supports need for construct consolidation rather than continued proliferation

### ENHANCED VALIDATION FRAMEWORK:

## Dual Holdout Validation Approaches

To address potential sample size and construct bias issues, the pipeline now supports two complementary validation methods:

### Approach 1: Stratified Item Holdout (Recommended)
- **Method**: 90-10 split across all constructs (improved from 80-20)
- **Training**: Model sees some items from all 246 IPIP constructs
- **Testing**: ~380 unseen IPIP items vs ~300 leadership items (balanced comparison)
- **Benefits**: Controls for sample size effects, tests item-level generalization

### Approach 2: Complete Construct Holdout (Advanced)
- **Method**: Hold out ~10 entire IPIP constructs from training
- **Training**: Model sees all items from ~236 constructs only
- **Testing**: All items from 10 completely unseen constructs vs leadership items
- **Benefits**: Tests construct-level generalization, more stringent validation

### Implementation Options:
```bash
# Stratified holdout (recommended for sample balance)
python3 scripts/ivan_analysis/create_holdout_splits.py --method stratified --holdout-ratio 0.1

# Complete construct holdout (for construct-level validation)
python3 scripts/ivan_analysis/create_holdout_splits.py --method construct --num-holdout-constructs 10
```

### VALIDATION ANALYSIS COMPLETED (June 4, 2025):

✅ **TRAINING COMPLETE!** Holdout validation model: `models/gist_construct_holdout_unified_final`
✅ **VALIDATION ANALYSIS COMPLETE!** Unbiased comparison completed
✅ **KEY VISUALIZATION CREATED**: `data/visualizations/construct_holdout_validation/top5_coherent_constructs_tsne.png`

**Validation Results:**
- IPIP holdout constructs: 87.4% accuracy (Cohen's d = 1.116)
- Leadership constructs: 62.9% accuracy (Cohen's d = 0.368)
- Statistical significance: t(853.99) = 43.49, p < 2.22e-16

**Key Output Files:**
- `/data/visualizations/construct_holdout_validation/holdout_validation_results.json`
- `/data/visualizations/construct_holdout_validation/top5_coherent_constructs_tsne.png` (CLEANEST VIZ)
- `/data/visualizations/construct_holdout_validation/clean_performance_summary.png`

**Most Coherent Constructs Identified:**
- IPIP: Interest in Collecting (0.720), Interest in Watching Television (0.691)
- Leadership: Abusive (0.576), Instrumental (0.448)

**Performance Optimizations Applied:**
- ✅ `GIST_BATCH_SIZE` increased 32 → 96 (leveraging 64GB memory)
- ✅ `TSDAE_BATCH_SIZE` increased 4 → 16  
- ✅ `TSDAE_EPOCHS` increased 1 → 3 (better domain adaptation)
- ✅ Using `BAAI/bge-m3` as guide model
- ✅ MPS acceleration with proper fallback handling

### Next Research Steps

**PRIMARY FOCUS: Manuscript Development**
1. **Update manuscript** in `/manuscript/leadership_measurement_paper.qmd` with validated findings
2. **Compile manuscript** using Quarto: `quarto render manuscript/leadership_measurement_paper.qmd --to apaquarto-docx`
3. **Incorporate key visualization**: `top5_coherent_constructs_tsne.png` showing cleanest comparison
4. **Finalize statistical reporting** with exact validated metrics (87.4% vs 62.9%)

**SECONDARY: Analysis Refinement**
1. Explore alternative taxonomies based on linguistic features:
   - Analyze linguistic complexity and abstraction levels
   - Compare positive vs. negative framing
   - Examine agency vs. communion themes
2. Apply dimensional reduction techniques to identify parsimonious leadership dimensions
3. Create additional targeted visualizations for manuscript

**FUTURE: Repository Organization**
1. **Create `/archive` folder** for superseded analyses and old scripts
2. **Move visualizations** out of `/data` into proper `/results` or `/outputs` folder  
3. **Consolidate scripts** and reduce duplication in `/scripts` directory
4. **Clean up visualization files** - many redundant files in `construct_holdout_validation/`
5. **Streamline directory structure** for better organization

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