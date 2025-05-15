# Archive of Original Scripts

This folder contains the original implementation scripts from the initial approach (pre-2025).

These scripts are kept for reference and historical purposes. **Please use the current scripts in the parent directory for all new analyses.**

## Archived Scripts

1. **build_ipip_pairs.py**
   - Original pair generation script
   - Created only one positive pair per anchor
   - Replaced by `build_ipip_pairs_improved.py` which generates all possible within-construct pairs

2. **train_gist_ipip.py**
   - Original training script using GISTEmbedLoss
   - Replaced by `train_ipip_mnrl.py` which uses MultipleNegativesRankingLoss for better results

3. **apply_gist_to_leadership.py**
   - Original script for applying trained model to leadership items
   - Replaced by `apply_best_model_to_leadership.py` which automatically selects the best model
   
4. **evaluate_ipip_model.py**
   - Original evaluation script
   - Replaced by `compare_model_performance.py` which provides more comprehensive evaluation

## Why These Were Replaced

Our original approach had several limitations:
- Limited training data (only one pair per anchor)
- Imbalanced representation of constructs
- GISTEmbedLoss wasn't optimal for our dataset size

For details on the improved approach, see `docs/improved_workflow.md`. 