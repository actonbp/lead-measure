# Archive of Original Scripts

This folder contains the original implementation scripts from the initial approach (pre-2025) and intermediate approaches.

These scripts are kept for reference and historical purposes. **Please use the current scripts in the parent directory for all new analyses.**

## Archived Scripts

### Original Implementation (pre-2025)
1. **build_ipip_pairs.py**
   - Original pair generation script
   - Created only one positive pair per anchor
   - Replaced by `build_ipip_pairs_improved.py` which generates all possible within-construct pairs

2. **train_gist_ipip.py**
   - Original training script using GISTEmbedLoss
   - Replaced by improved training with TSDAE pre-training

3. **apply_gist_to_leadership.py**
   - Original script for applying trained model to leadership items
   - Replaced by `apply_best_model_to_leadership.py` which automatically selects the best model
   
4. **evaluate_ipip_model.py**
   - Original evaluation script
   - Replaced by more comprehensive evaluation scripts with similarity analysis

### Intermediate Approaches (2025)
5. **train_ipip_gist.py**
   - Intermediate training script using GISTEmbedLoss with all-mpnet-base-v2
   - Replaced by updated train_ipip_mnrl.py which now supports GIST loss with BGE-M3 model

6. **compare_gist_vs_mnrl.py**
   - Comparison script between GIST and MNRL approaches
   - Analysis complete, results documented in improved_workflow.md

7. **evaluate_mnrl_model.py**
   - Early MNRL evaluation script
   - Superseded by `evaluate_model_with_validation.py` with more comprehensive metrics

8. **evaluate_trained_ipip_model.py**
   - Detailed evaluation of specific models
   - Functionality merged into `evaluate_model_with_validation.py`

9. **generate_mnrl_evaluation_report.py**
   - Report generation for MNRL models
   - Functionality integrated into other evaluation scripts

### Debug and Testing Scripts
10. **debug_train.py**
    - Temporary debugging script for training issues
    - No longer needed

11. **minimal_load_test.py**
    - Simple model loading test
    - Used for debugging model compatibility

12. **download_model.py**
    - Utility for downloading specific models
    - No longer needed

13. **check_training_progress.py**
    - Training progress monitoring script
    - Debugging utility

### Workflow Scripts (Superseded)
14. **run_improved_workflow.sh**
    - Intermediate workflow using MNRL approach
    - Replaced by `run_refined_workflow.sh` with BGE-M3 and TSDAE

15. **run_complete_evaluation.sh**
    - Comprehensive evaluation workflow
    - Functionality merged into refined workflow

16. **run_loss_function_comparison.sh**
    - Comparison workflow for different loss functions
    - Analysis complete

17. **run_with_local_model.sh**
    - Workflow for local model training
    - Debugging workflow

## Why These Were Replaced

Our original approach had several limitations:
- Limited training data (only one pair per anchor)
- Imbalanced representation of constructs
- Suboptimal model selection for clustering tasks
- No domain adaptation for specialized text
- Insufficient batch sizes for contrastive learning

The MultipleNegativesRankingLoss approach (May 2025) was an intermediate improvement, but has now been superseded by our refined approach using:
- BAAI/bge-m3 model optimized for clustering
- TSDAE pre-training for domain adaptation
- Randomized anchor-positive positions
- Improved GISTEmbedLoss with larger batch sizes
- Mixed precision training

For details on the refined approach, see `docs/improved_workflow.md`. 