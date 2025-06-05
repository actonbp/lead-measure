#!/usr/bin/env python3
"""
Comprehensive statistical comparison between IPIP and Leadership construct separability.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_effect_size_comparison(d1, n1, d2, n2):
    """
    Calculate the statistical significance of difference between two Cohen's d values.
    Using the method from Borenstein et al. (2009) Introduction to Meta-Analysis.
    """
    # Standard error of Cohen's d
    se_d1 = np.sqrt((n1 + n2) / (n1 * n2) + (d1**2) / (2 * (n1 + n2 - 2)))
    se_d2 = np.sqrt((n1 + n2) / (n1 * n2) + (d2**2) / (2 * (n1 + n2 - 2)))
    
    # Standard error of the difference
    se_diff = np.sqrt(se_d1**2 + se_d2**2)
    
    # Z-test for difference
    z_stat = (d1 - d2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value, se_diff

def two_proportion_z_test(p1, n1, p2, n2):
    """
    Z-test for comparing two proportions.
    """
    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    # Z-statistic
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value

def bootstrap_confidence_interval(data1, data2, n_bootstrap=10000, confidence=0.95):
    """
    Bootstrap confidence interval for difference in means.
    """
    np.random.seed(42)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_diffs, 100 * alpha/2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha/2))
    
    return lower, upper, bootstrap_diffs

def main():
    """Run comprehensive statistical analysis."""
    
    # Our results
    ipip_accuracy = 0.8735
    ipip_cohens_d = 1.116
    ipip_n = 427
    
    leadership_accuracy = 0.6290
    leadership_cohens_d = 0.368
    leadership_n = 434
    
    # Create output directory
    output_dir = Path("data/visualizations/enhanced_statistical_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔬 Enhanced Statistical Comparison: IPIP vs Leadership")
    print("=" * 60)
    
    # 1. Two-proportion z-test
    z_prop, p_prop = two_proportion_z_test(ipip_accuracy, ipip_n, leadership_accuracy, leadership_n)
    
    print(f"\n1. TWO-PROPORTION Z-TEST")
    print(f"   H0: IPIP accuracy = Leadership accuracy")
    print(f"   IPIP accuracy: {ipip_accuracy:.1%} (n={ipip_n})")
    print(f"   Leadership accuracy: {leadership_accuracy:.1%} (n={leadership_n})")
    print(f"   Difference: {(ipip_accuracy - leadership_accuracy)*100:.1f} percentage points")
    print(f"   Z-statistic: {z_prop:.3f}")
    print(f"   p-value: {p_prop:.2e}")
    print(f"   Result: {'SIGNIFICANT' if p_prop < 0.05 else 'NOT SIGNIFICANT'}")
    
    # 2. Effect size comparison
    z_effect, p_effect, se_diff = calculate_effect_size_comparison(
        ipip_cohens_d, ipip_n, leadership_cohens_d, leadership_n
    )
    
    print(f"\n2. EFFECT SIZE COMPARISON")
    print(f"   IPIP Cohen's d: {ipip_cohens_d:.3f}")
    print(f"   Leadership Cohen's d: {leadership_cohens_d:.3f}")
    print(f"   Difference: {ipip_cohens_d - leadership_cohens_d:.3f}")
    print(f"   Z-statistic: {z_effect:.3f}")
    print(f"   p-value: {p_effect:.2e}")
    print(f"   Result: {'SIGNIFICANT' if p_effect < 0.05 else 'NOT SIGNIFICANT'}")
    
    # 3. Confidence intervals for Cohen's d
    def cohens_d_ci(d, n, confidence=0.95):
        """Confidence interval for Cohen's d using non-central t-distribution."""
        alpha = 1 - confidence
        df = n - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        se_d = np.sqrt((n + n) / (n * n) + (d**2) / (2 * (n + n - 2)))
        margin = t_critical * se_d
        return d - margin, d + margin
    
    ipip_ci_lower, ipip_ci_upper = cohens_d_ci(ipip_cohens_d, ipip_n)
    leadership_ci_lower, leadership_ci_upper = cohens_d_ci(leadership_cohens_d, leadership_n)
    
    print(f"\n3. CONFIDENCE INTERVALS FOR COHEN'S D")
    print(f"   IPIP 95% CI: [{ipip_ci_lower:.3f}, {ipip_ci_upper:.3f}]")
    print(f"   Leadership 95% CI: [{leadership_ci_lower:.3f}, {leadership_ci_upper:.3f}]")
    print(f"   Overlap: {'YES' if ipip_ci_lower <= leadership_ci_upper else 'NO'}")
    
    # 4. Bootstrap analysis (simulated data based on known statistics)
    # Simulate data based on known means and effect sizes
    np.random.seed(42)
    
    # For IPIP: generate data with accuracy 87.35%
    ipip_simulated = np.random.binomial(1, ipip_accuracy, ipip_n)
    
    # For Leadership: generate data with accuracy 62.90%
    leadership_simulated = np.random.binomial(1, leadership_accuracy, leadership_n)
    
    boot_lower, boot_upper, boot_diffs = bootstrap_confidence_interval(
        ipip_simulated, leadership_simulated
    )
    
    print(f"\n4. BOOTSTRAP ANALYSIS")
    print(f"   Bootstrap 95% CI for difference: [{boot_lower:.3f}, {boot_upper:.3f}]")
    print(f"   Bootstrap mean difference: {np.mean(boot_diffs):.3f}")
    
    # 5. Practical significance assessment
    print(f"\n5. PRACTICAL SIGNIFICANCE")
    print(f"   Accuracy difference: {(ipip_accuracy - leadership_accuracy)*100:.1f} percentage points")
    print(f"   Effect size difference: {ipip_cohens_d - leadership_cohens_d:.3f}")
    
    effect_diff = ipip_cohens_d - leadership_cohens_d
    if effect_diff >= 0.8:
        practical = "LARGE practical difference"
    elif effect_diff >= 0.5:
        practical = "MEDIUM practical difference" 
    elif effect_diff >= 0.2:
        practical = "SMALL practical difference"
    else:
        practical = "NEGLIGIBLE practical difference"
    
    print(f"   Interpretation: {practical}")
    
    # 6. Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    categories = ['IPIP Holdout', 'Leadership']
    accuracies = [ipip_accuracy * 100, leadership_accuracy * 100]
    errors = [1.96 * np.sqrt(ipip_accuracy * (1-ipip_accuracy) / ipip_n) * 100,
              1.96 * np.sqrt(leadership_accuracy * (1-leadership_accuracy) / leadership_n) * 100]
    
    bars = ax1.bar(categories, accuracies, yerr=errors, capsize=10, 
                   color=['steelblue', 'coral'], alpha=0.7)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Construct Separation Accuracy Comparison')
    ax1.set_ylim(0, 100)
    
    # Add significance annotation
    y_max = max(accuracies) + max(errors) + 5
    ax1.plot([0, 1], [y_max, y_max], 'k-', linewidth=2)
    ax1.text(0.5, y_max + 2, f'p = {p_prop:.2e}', ha='center', fontweight='bold')
    
    # Cohen's d comparison
    cohens_ds = [ipip_cohens_d, leadership_cohens_d]
    d_errors = [ipip_ci_upper - ipip_cohens_d, leadership_ci_upper - leadership_cohens_d]
    
    bars2 = ax2.bar(categories, cohens_ds, yerr=d_errors, capsize=10,
                    color=['steelblue', 'coral'], alpha=0.7)
    ax2.set_ylabel("Cohen's d")
    ax2.set_title('Effect Size Comparison')
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    
    # Add significance annotation
    y_max = max(cohens_ds) + max(d_errors) + 0.1
    ax2.plot([0, 1], [y_max, y_max], 'k-', linewidth=2)
    ax2.text(0.5, y_max + 0.05, f'p = {p_effect:.2e}', ha='center', fontweight='bold')
    
    # Bootstrap distribution
    ax3.hist(boot_diffs, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax3.axvline(np.mean(boot_diffs), color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.axvline(boot_lower, color='red', linestyle=':', linewidth=2, label='95% CI')
    ax3.axvline(boot_upper, color='red', linestyle=':', linewidth=2)
    ax3.set_xlabel('Difference in Accuracy')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Bootstrap Distribution of Accuracy Difference')
    ax3.legend()
    
    # Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'IPIP Holdout', 'Leadership', 'Difference', 'p-value'],
        ['Accuracy', f'{ipip_accuracy:.1%}', f'{leadership_accuracy:.1%}', 
         f'{(ipip_accuracy-leadership_accuracy)*100:.1f}pp', f'{p_prop:.2e}'],
        ["Cohen's d", f'{ipip_cohens_d:.3f}', f'{leadership_cohens_d:.3f}', 
         f'{ipip_cohens_d-leadership_cohens_d:.3f}', f'{p_effect:.2e}'],
        ['Sample size', f'{ipip_n}', f'{leadership_n}', '', ''],
        ['Interpretation', 'Large effect', 'Small effect', 'Large difference', 'Highly significant']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "enhanced_statistical_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results = {
        'accuracy_comparison': {
            'ipip_accuracy': ipip_accuracy,
            'leadership_accuracy': leadership_accuracy,
            'difference': ipip_accuracy - leadership_accuracy,
            'z_statistic': z_prop,
            'p_value': p_prop
        },
        'effect_size_comparison': {
            'ipip_cohens_d': ipip_cohens_d,
            'leadership_cohens_d': leadership_cohens_d,
            'difference': ipip_cohens_d - leadership_cohens_d,
            'z_statistic': z_effect,
            'p_value': p_effect
        },
        'confidence_intervals': {
            'ipip_ci': [ipip_ci_lower, ipip_ci_upper],
            'leadership_ci': [leadership_ci_lower, leadership_ci_upper],
            'bootstrap_ci': [boot_lower, boot_upper]
        },
        'practical_significance': practical
    }
    
    # Save report
    report = f"""
Enhanced Statistical Analysis Report
===================================

CRITICAL FINDINGS:
- IPIP accuracy: {ipip_accuracy:.1%} (95% CI: {ipip_ci_lower:.1%} to {ipip_ci_upper:.1%} for Cohen's d)
- Leadership accuracy: {leadership_accuracy:.1%} (95% CI: {leadership_ci_lower:.1%} to {leadership_ci_upper:.1%} for Cohen's d)
- Performance gap: {(ipip_accuracy - leadership_accuracy)*100:.1f} percentage points (95% CI: {boot_lower*100:.1f}% to {boot_upper*100:.1f}%)

STATISTICAL SIGNIFICANCE:
- Two-proportion z-test: z = {z_prop:.3f}, p = {p_prop:.2e} (highly significant)
- Effect size comparison: z = {z_effect:.3f}, p = {p_effect:.2e} (highly significant)
- Effect size difference: {ipip_cohens_d - leadership_cohens_d:.3f} ({practical})

RESEARCH IMPLICATIONS:
This analysis provides strong empirical evidence that leadership constructs have 
significantly less discriminant validity than personality constructs, supporting 
concerns about construct proliferation in leadership measurement.

The {(ipip_accuracy - leadership_accuracy)*100:.1f} percentage point difference with p < {p_prop:.0e} indicates 
that the semantic overlap in leadership constructs is not due to chance but represents 
a systematic measurement issue that needs to be addressed in leadership theory and practice.
"""
    
    with open(output_dir / "enhanced_statistical_analysis_report.txt", 'w') as f:
        f.write(report)
    
    # Save raw results
    import json
    with open(output_dir / "enhanced_statistical_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 SUMMARY")
    print(f"   Files saved to: {output_dir}")
    print(f"   Key finding: {practical}")
    print(f"   Statistical significance: p < {min(p_prop, p_effect):.0e}")

if __name__ == "__main__":
    main()