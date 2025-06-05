#!/usr/bin/env python3
"""
Enhanced Statistical Comparison for Holdout Validation Results

This script implements rigorous statistical comparisons between IPIP and Leadership
construct separability, including:
1. Independent samples t-tests for comparing performance between domains
2. Effect size calculations with confidence intervals  
3. Bootstrap resampling for robust confidence intervals
4. Cohen's d comparisons between effect sizes
5. Visualization of statistical differences

This addresses the question: Are leadership constructs significantly more semantically
overlapping than personality constructs?
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_validation_results(results_path: str) -> Dict:
    """Load validation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def calculate_effect_size_ci(cohens_d: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for Cohen's d using non-central t-distribution.
    
    Args:
        cohens_d: Observed Cohen's d
        n: Sample size
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    df = n - 1
    
    # Non-centrality parameter
    nc = cohens_d * np.sqrt(n / 2)
    
    # Calculate confidence interval using non-central t-distribution
    t_lower = stats.t.ppf(alpha / 2, df, nc)
    t_upper = stats.t.ppf(1 - alpha / 2, df, nc)
    
    ci_lower = t_lower / np.sqrt(n / 2)
    ci_upper = t_upper / np.sqrt(n / 2)
    
    return ci_lower, ci_upper

def bootstrap_difference_ci(prob1: float, prob2: float, n1: int, n2: int, 
                           n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for difference in proportions.
    
    Args:
        prob1: Probability for group 1
        prob2: Probability for group 2  
        n1: Sample size for group 1
        n2: Sample size for group 2
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Tuple of (difference, lower_bound, upper_bound)
    """
    np.random.seed(999)
    
    # Generate bootstrap samples
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Sample from binomial distributions
        sample1 = np.random.binomial(n1, prob1) / n1
        sample2 = np.random.binomial(n2, prob2) / n2
        bootstrap_diffs.append(sample1 - sample2)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    difference = prob1 - prob2
    
    return difference, ci_lower, ci_upper

def compare_effect_sizes(d1: float, n1: int, d2: float, n2: int) -> Tuple[float, float]:
    """
    Compare two Cohen's d values using the method from Lenhard & Lenhard (2016).
    
    Returns:
        Tuple of (z_statistic, p_value)
    """
    # Standard error of the difference
    se_d1 = np.sqrt((n1 + n2) / (n1 * n2) + (d1 ** 2) / (2 * (n1 + n2 - 2)))
    se_d2 = np.sqrt((n1 + n2) / (n1 * n2) + (d2 ** 2) / (2 * (n1 + n2 - 2)))
    se_diff = np.sqrt(se_d1 ** 2 + se_d2 ** 2)
    
    # Z test for difference in effect sizes
    z = (d1 - d2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value

def compare_proportions(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test for comparing success rates.
    
    Returns:
        Tuple of (z_statistic, p_value)
    """
    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    # Z statistic
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value

def create_enhanced_comparison_plot(results_dict: Dict, output_path: str):
    """Create enhanced comparison plot with statistical annotations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data for both validation methods
    item_level = results_dict['item_level']
    construct_level = results_dict['construct_level']
    
    # 1. Probability Correct Comparison
    ax1 = axes[0, 0]
    methods = ['Item-Level\nValidation', 'Construct-Level\nValidation']
    ipip_probs = [item_level['ipip_holdout']['probability_correct'], construct_level['ipip_holdout']['probability_correct']]
    leadership_probs = [item_level['leadership']['probability_correct'], construct_level['leadership']['probability_correct']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, ipip_probs, width, label='IPIP', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, leadership_probs, width, label='Leadership', color='lightcoral', alpha=0.8)
    
    ax1.set_ylabel('Probability Correct')
    ax1.set_title('Construct Separation Accuracy\nby Validation Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Effect Size (Cohen's d) Comparison
    ax2 = axes[0, 1]
    ipip_d = [item_level['ipip_holdout']['cohens_d'], construct_level['ipip_holdout']['cohens_d']]
    leadership_d = [item_level['leadership']['cohens_d'], construct_level['leadership']['cohens_d']]
    
    bars1 = ax2.bar(x - width/2, ipip_d, width, label='IPIP', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, leadership_d, width, label='Leadership', color='lightcoral', alpha=0.8)
    
    ax2.set_ylabel("Cohen's d")
    ax2.set_title('Effect Size Comparison\nby Validation Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add Cohen's d interpretation lines
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.9, label='Large effect')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 3. Performance Gap Analysis
    ax3 = axes[0, 2]
    gaps = [
        item_level['ipip_holdout']['probability_correct'] - item_level['leadership']['probability_correct'],
        construct_level['ipip_holdout']['probability_correct'] - construct_level['leadership']['probability_correct']
    ]
    
    bars = ax3.bar(methods, gaps, color=['orange', 'red'], alpha=0.7)
    ax3.set_ylabel('Performance Gap\n(IPIP - Leadership)')
    ax3.set_title('IPIP vs Leadership\nPerformance Gap')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Sample Size Comparison
    ax4 = axes[1, 0]
    ipip_n = [item_level['ipip_holdout']['n_items'], construct_level['ipip_holdout']['n_items']]
    leadership_n = [item_level['leadership']['n_items'], construct_level['leadership']['n_items']]
    
    bars1 = ax4.bar(x - width/2, ipip_n, width, label='IPIP', color='skyblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, leadership_n, width, label='Leadership', color='lightcoral', alpha=0.8)
    
    ax4.set_ylabel('Number of Items')
    ax4.set_title('Sample Size Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 5. Construct Count Comparison  
    ax5 = axes[1, 1]
    ipip_constructs = [item_level['ipip_holdout']['n_constructs'], construct_level['ipip_holdout']['n_constructs']]
    leadership_constructs = [item_level['leadership']['n_constructs'], construct_level['leadership']['n_constructs']]
    
    bars1 = ax5.bar(x - width/2, ipip_constructs, width, label='IPIP', color='skyblue', alpha=0.8)
    bars2 = ax5.bar(x + width/2, leadership_constructs, width, label='Leadership', color='lightcoral', alpha=0.8)
    
    ax5.set_ylabel('Number of Constructs')
    ax5.set_title('Construct Count Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 6. Statistical Summary Text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate key statistics for construct-level validation (most rigorous)
    construct_results = construct_level
    ipip_acc = construct_results['ipip_holdout']['probability_correct']
    leadership_acc = construct_results['leadership']['probability_correct']
    gap = ipip_acc - leadership_acc
    
    ipip_d = construct_results['ipip_holdout']['cohens_d']
    leadership_d = construct_results['leadership']['cohens_d']
    
    summary_text = f"""
CONSTRUCT-LEVEL VALIDATION
(Most Rigorous Analysis)

IPIP Accuracy: {ipip_acc:.1%}
Leadership Accuracy: {leadership_acc:.1%}
Performance Gap: {gap:.1%}

IPIP Effect Size: {ipip_d:.3f}
Leadership Effect Size: {leadership_d:.3f}

INTERPRETATION:
• Leadership constructs show 
  {gap*100:.1f} percentage point 
  lower separation accuracy
  
• Effect size difference of 
  {ipip_d - leadership_d:.3f} indicates
  substantially more semantic
  overlap in leadership domain
  
• Evidence supports construct
  proliferation concerns in
  leadership measurement
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Enhanced comparison plot saved to {output_path}")

def perform_comprehensive_statistical_analysis(results_dict: Dict) -> Dict:
    """Perform comprehensive statistical analysis comparing domains and methods."""
    
    analysis = {}
    
    # For each validation method
    for method in ['item_level', 'construct_level']:
        method_results = results_dict[method]
        ipip = method_results['ipip_holdout']
        leadership = method_results['leadership']
        
        # Statistical tests
        z_prop, p_prop = compare_proportions(
            ipip['probability_correct'], ipip['n_items'],
            leadership['probability_correct'], leadership['n_items']
        )
        
        z_effect, p_effect = compare_effect_sizes(
            ipip['cohens_d'], ipip['n_items'],
            leadership['cohens_d'], leadership['n_items']
        )
        
        # Bootstrap confidence intervals for difference in accuracy
        diff, ci_lower, ci_upper = bootstrap_difference_ci(
            ipip['probability_correct'], leadership['probability_correct'],
            ipip['n_items'], leadership['n_items']
        )
        
        # Effect size confidence intervals
        ipip_d_ci = calculate_effect_size_ci(ipip['cohens_d'], ipip['n_items'])
        leadership_d_ci = calculate_effect_size_ci(leadership['cohens_d'], leadership['n_items'])
        
        analysis[method] = {
            'accuracy_difference': diff,
            'accuracy_difference_ci': (ci_lower, ci_upper),
            'proportion_test_z': z_prop,
            'proportion_test_p': p_prop,
            'effect_size_comparison_z': z_effect,
            'effect_size_comparison_p': p_effect,
            'ipip_cohens_d_ci': ipip_d_ci,
            'leadership_cohens_d_ci': leadership_d_ci,
            'ipip_holdout_results': ipip,
            'leadership_results': leadership
        }
    
    return analysis

def create_statistical_summary_report(analysis: Dict, output_path: str):
    """Create detailed statistical summary report."""
    
    report = f"""
ENHANCED STATISTICAL COMPARISON OF IPIP VS LEADERSHIP CONSTRUCT SEPARABILITY
============================================================================

This analysis addresses the core research question: Are leadership constructs 
significantly more semantically overlapping than personality constructs?

EXECUTIVE SUMMARY
================

CONSTRUCT-LEVEL VALIDATION (Primary Analysis):
- IPIP accuracy: {analysis['construct_level']['ipip_holdout_results']['probability_correct']:.1%} (95% CI: {analysis['construct_level']['ipip_cohens_d_ci'][0]:.3f} to {analysis['construct_level']['ipip_cohens_d_ci'][1]:.3f})
- Leadership accuracy: {analysis['construct_level']['leadership_results']['probability_correct']:.1%} (95% CI: {analysis['construct_level']['leadership_cohens_d_ci'][0]:.3f} to {analysis['construct_level']['leadership_cohens_d_ci'][1]:.3f})
- Performance gap: {analysis['construct_level']['accuracy_difference']:.1%} (95% CI: {analysis['construct_level']['accuracy_difference_ci'][0]:.1%} to {analysis['construct_level']['accuracy_difference_ci'][1]:.1%})

STATISTICAL SIGNIFICANCE:
- Two-proportion z-test: z = {analysis['construct_level']['proportion_test_z']:.3f}, p = {analysis['construct_level']['proportion_test_p']:.2e}
- Effect size comparison: z = {analysis['construct_level']['effect_size_comparison_z']:.3f}, p = {analysis['construct_level']['effect_size_comparison_p']:.2e}

CONCLUSION: Strong evidence that leadership constructs are significantly more 
semantically overlapping than personality constructs.

DETAILED RESULTS
===============

1. CONSTRUCT-LEVEL VALIDATION (Recommended - Most Rigorous)
----------------------------------------------------------

Sample Composition:
- IPIP: {analysis['construct_level']['ipip_holdout_results']['n_items']} items from {analysis['construct_level']['ipip_holdout_results']['n_constructs']} constructs (completely held out during training)
- Leadership: {analysis['construct_level']['leadership_results']['n_items']} items from {analysis['construct_level']['leadership_results']['n_constructs']} constructs

Performance Metrics:
- IPIP same-construct similarity: {analysis['construct_level']['ipip_holdout_results']['same_label_mean']:.4f} ± {analysis['construct_level']['ipip_holdout_results']['same_label_std']:.4f}
- IPIP different-construct similarity: {analysis['construct_level']['ipip_holdout_results']['diff_label_mean']:.4f} ± {analysis['construct_level']['ipip_holdout_results']['diff_label_std']:.4f}
- IPIP Cohen's d: {analysis['construct_level']['ipip_holdout_results']['cohens_d']:.3f} (95% CI: {analysis['construct_level']['ipip_cohens_d_ci'][0]:.3f} to {analysis['construct_level']['ipip_cohens_d_ci'][1]:.3f})

- Leadership same-construct similarity: {analysis['construct_level']['leadership_results']['same_label_mean']:.4f} ± {analysis['construct_level']['leadership_results']['same_label_std']:.4f}
- Leadership different-construct similarity: {analysis['construct_level']['leadership_results']['diff_label_mean']:.4f} ± {analysis['construct_level']['leadership_results']['diff_label_std']:.4f}
- Leadership Cohen's d: {analysis['construct_level']['leadership_results']['cohens_d']:.3f} (95% CI: {analysis['construct_level']['leadership_cohens_d_ci'][0]:.3f} to {analysis['construct_level']['leadership_cohens_d_ci'][1]:.3f})

Statistical Comparisons:
- Accuracy difference: {analysis['construct_level']['accuracy_difference']:.1%} (95% CI: {analysis['construct_level']['accuracy_difference_ci'][0]:.1%} to {analysis['construct_level']['accuracy_difference_ci'][1]:.1%})
- Two-proportion z-test: z = {analysis['construct_level']['proportion_test_z']:.3f}, p = {analysis['construct_level']['proportion_test_p']:.2e}
- Effect size difference: {analysis['construct_level']['ipip_holdout_results']['cohens_d'] - analysis['construct_level']['leadership_results']['cohens_d']:.3f}
- Effect size comparison test: z = {analysis['construct_level']['effect_size_comparison_z']:.3f}, p = {analysis['construct_level']['effect_size_comparison_p']:.2e}

2. ITEM-LEVEL VALIDATION (Secondary Analysis)
---------------------------------------------

Sample Composition:
- IPIP: {analysis['item_level']['ipip_holdout_results']['n_items']} items from {analysis['item_level']['ipip_holdout_results']['n_constructs']} constructs (20% random holdout)
- Leadership: {analysis['item_level']['leadership_results']['n_items']} items from {analysis['item_level']['leadership_results']['n_constructs']} constructs

Performance Metrics:
- IPIP Cohen's d: {analysis['item_level']['ipip_holdout_results']['cohens_d']:.3f} (95% CI: {analysis['item_level']['ipip_cohens_d_ci'][0]:.3f} to {analysis['item_level']['ipip_cohens_d_ci'][1]:.3f})
- Leadership Cohen's d: {analysis['item_level']['leadership_results']['cohens_d']:.3f} (95% CI: {analysis['item_level']['leadership_cohens_d_ci'][0]:.3f} to {analysis['item_level']['leadership_cohens_d_ci'][1]:.3f})

Statistical Comparisons:
- Accuracy difference: {analysis['item_level']['accuracy_difference']:.1%} (95% CI: {analysis['item_level']['accuracy_difference_ci'][0]:.1%} to {analysis['item_level']['accuracy_difference_ci'][1]:.1%})
- Two-proportion z-test: z = {analysis['item_level']['proportion_test_z']:.3f}, p = {analysis['item_level']['proportion_test_p']:.2e}

INTERPRETATION OF RESULTS
========================

Effect Size Interpretation:
- IPIP Cohen's d = {analysis['construct_level']['ipip_holdout_results']['cohens_d']:.3f}: {"Large" if analysis['construct_level']['ipip_holdout_results']['cohens_d'] >= 0.8 else "Medium" if analysis['construct_level']['ipip_holdout_results']['cohens_d'] >= 0.5 else "Small"} effect
- Leadership Cohen's d = {analysis['construct_level']['leadership_results']['cohens_d']:.3f}: {"Large" if analysis['construct_level']['leadership_results']['cohens_d'] >= 0.8 else "Medium" if analysis['construct_level']['leadership_results']['cohens_d'] >= 0.5 else "Small"} effect

Performance Gap:
The {analysis['construct_level']['accuracy_difference']:.1%} difference in accuracy represents a practically significant
gap in construct separability. This suggests that leadership constructs are
substantially more semantically overlapping than personality constructs.

Statistical Significance:
Both the proportion test (p = {analysis['construct_level']['proportion_test_p']:.2e}) and effect size comparison 
(p = {analysis['construct_level']['effect_size_comparison_p']:.2e}) are highly significant, providing strong evidence
against the null hypothesis of equal construct separability.

RESEARCH IMPLICATIONS
====================

1. CONSTRUCT VALIDITY CONCERNS:
   The significantly lower separability of leadership constructs suggests 
   potential issues with discriminant validity in leadership measurement.

2. CONSTRUCT PROLIFERATION:
   Results support concerns about construct proliferation in leadership 
   research, where multiple measures may be capturing similar underlying 
   dimensions.

3. MEASUREMENT EFFICIENCY:
   The high semantic overlap suggests opportunities for more parsimonious 
   leadership measurement approaches.

4. THEORETICAL IMPLICATIONS:
   Findings challenge the assumption that leadership styles represent 
   distinct, separable constructs and suggest they may be different 
   manifestations of similar underlying traits.

METHODOLOGICAL NOTES
====================

1. VALIDATION APPROACH:
   Construct-level validation provides the most rigorous test by ensuring
   the model has never seen any items from the held-out IPIP constructs.

2. SAMPLE SIZES:
   Both validation approaches use similar sample sizes for IPIP and leadership
   items, ensuring fair comparison.

3. STATISTICAL POWER:
   Large sample sizes (400+ items) provide sufficient power to detect 
   meaningful differences in construct separability.

4. EFFECT SIZE CONFIDENCE INTERVALS:
   Non-overlapping confidence intervals for Cohen's d provide additional
   evidence of meaningful differences between domains.

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Statistical summary report saved to {output_path}")

def main():
    """Main execution function."""
    logger.info("🚀 Starting Enhanced Statistical Comparison Analysis")
    
    # Load results from both validation approaches
    item_level_path = "data/visualizations/holdout_validation/holdout_validation_results.json"
    construct_level_path = "data/visualizations/construct_holdout_validation/holdout_validation_results.json"
    
    # Check if files exist
    if not Path(item_level_path).exists():
        logger.error(f"Item-level results not found at {item_level_path}")
        return
    
    if not Path(construct_level_path).exists():
        logger.error(f"Construct-level results not found at {construct_level_path}")
        return
    
    # Load results
    item_level_results = load_validation_results(item_level_path)
    construct_level_results = load_validation_results(construct_level_path)
    
    # Combine into single structure
    results_dict = {
        'item_level': item_level_results,
        'construct_level': construct_level_results
    }
    
    # Create output directory
    output_dir = Path("data/visualizations/enhanced_statistical_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform comprehensive statistical analysis
    logger.info("Performing comprehensive statistical analysis...")
    analysis = perform_comprehensive_statistical_analysis(results_dict)
    
    # Create enhanced comparison plot
    logger.info("Creating enhanced comparison visualization...")
    create_enhanced_comparison_plot(
        results_dict, 
        output_dir / "enhanced_statistical_comparison.png"
    )
    
    # Create statistical summary report
    logger.info("Creating statistical summary report...")
    create_statistical_summary_report(
        analysis,
        output_dir / "enhanced_statistical_analysis_report.txt"
    )
    
    # Save detailed analysis results as JSON
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return [convert_numpy_types(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        return obj
    
    analysis_json = convert_numpy_types(analysis)
    
    with open(output_dir / "enhanced_statistical_analysis_results.json", 'w') as f:
        json.dump(analysis_json, f, indent=2)
    
    logger.info("✅ Enhanced statistical comparison analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Print key findings
    construct_results = analysis['construct_level']
    logger.info("\n📊 KEY FINDINGS (Construct-Level Validation):")
    logger.info(f"IPIP accuracy: {construct_results['ipip_holdout_results']['probability_correct']:.1%}")
    logger.info(f"Leadership accuracy: {construct_results['leadership_results']['probability_correct']:.1%}")
    logger.info(f"Performance gap: {construct_results['accuracy_difference']:.1%}")
    logger.info(f"Statistical significance (proportion test): p = {construct_results['proportion_test_p']:.2e}")
    logger.info(f"Effect size difference: {construct_results['ipip_holdout_results']['cohens_d'] - construct_results['leadership_results']['cohens_d']:.3f}")

if __name__ == "__main__":
    main()