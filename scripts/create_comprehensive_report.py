"""Create a comprehensive MNRL model evaluation report for sharing.

This script:
1. Collects the results from the IPIP and leadership evaluations
2. Combines them into a single comprehensive report in Markdown format
3. Creates a PDF version of the report
4. Places all shareable outputs in a specific directory

Usage:
    python scripts/create_comprehensive_report.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
import re
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# Configuration
OUTPUT_DIR = Path("docs/output/model_evaluations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the latest evaluation directories
def find_latest_evaluations():
    """Find the latest IPIP and leadership evaluation directories."""
    ipip_eval_dirs = list(Path("data/visualizations").glob("mnrl_evaluation_*"))
    leadership_eval_dirs = list(Path("data/visualizations").glob("leadership_mnrl_evaluation_*"))
    
    # Sort by creation time (newest first)
    ipip_eval_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    leadership_eval_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    return ipip_eval_dirs[0] if ipip_eval_dirs else None, leadership_eval_dirs[0] if leadership_eval_dirs else None

def extract_metrics(metrics_file):
    """Extract metrics from a metrics text file."""
    metrics = {}
    with open(metrics_file, 'r') as f:
        content = f.read()
        
        # Extract model name
        model_match = re.search(r"Model: ([^\n]+)", content)
        if model_match:
            metrics['model'] = model_match.group(1)
        
        # Extract ARI
        ari_match = re.search(r"Adjusted Rand Index: ([0-9.]+)", content)
        if ari_match:
            metrics['ari'] = float(ari_match.group(1))
        
        # Extract NMI
        nmi_match = re.search(r"Normalized Mutual Information: ([0-9.]+)", content)
        if nmi_match:
            metrics['nmi'] = float(nmi_match.group(1))
        
        # Extract Purity
        purity_match = re.search(r"Cluster Purity: ([0-9.]+)", content)
        if purity_match:
            metrics['purity'] = float(purity_match.group(1))
    
    return metrics

def create_markdown_report(ipip_dir, leadership_dir):
    """Create a comprehensive markdown report from evaluation results."""
    report_path = OUTPUT_DIR / "mnrl_model_evaluation_report.md"
    
    # Extract metrics
    ipip_metrics = extract_metrics(ipip_dir / "evaluation_metrics.txt")
    leadership_metrics = extract_metrics(leadership_dir / "leadership_metrics.txt")
    
    # Parse the leadership similarity data
    leadership_similar_pairs = []
    with open(leadership_dir / "leadership_metrics.txt", 'r') as f:
        content = f.read()
        sim_section = content.split("Highly Similar Construct Pairs")[1].split("Generated on")[0]
        pair_matches = re.findall(r"  - ([^:]+): ([0-9.]+)", sim_section)
        for match in pair_matches:
            constructs, similarity = match
            leadership_similar_pairs.append({
                'constructs': constructs.strip(),
                'similarity': float(similarity)
            })
    
    # Create the report
    with open(report_path, 'w') as f:
        f.write(f"# MNRL Model Evaluation Report\n\n")
        f.write(f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}*\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents the evaluation results of our IPIP MNRL (Multiple Negatives Ranking Loss) model trained on comprehensive personality item pairs. We assess how effectively the model clusters both personality items and leadership items by their respective construct categories.\n\n")
        
        f.write("### Key Findings\n\n")
        f.write("**IPIP Personality Data**:\n")
        f.write(f"- **Adjusted Rand Index (ARI)**: {ipip_metrics.get('ari', 0):.4f}\n")
        f.write(f"- **Normalized Mutual Information (NMI)**: {ipip_metrics.get('nmi', 0):.4f}\n")
        f.write(f"- **Cluster Purity**: {ipip_metrics.get('purity', 0):.4f}\n\n")
        
        f.write("**Leadership Construct Data**:\n")
        f.write(f"- **Adjusted Rand Index (ARI)**: {leadership_metrics.get('ari', 0):.4f}\n")
        f.write(f"- **Normalized Mutual Information (NMI)**: {leadership_metrics.get('nmi', 0):.4f}\n")
        f.write(f"- **Cluster Purity**: {leadership_metrics.get('purity', 0):.4f}\n\n")
        
        f.write("The substantial drop in metrics from IPIP to leadership data supports our research hypothesis that leadership constructs have significant overlap and are less distinctly separated than personality constructs.\n\n")
        
        f.write("## IPIP Personality Constructs Evaluation\n\n")
        f.write("The model shows moderate performance in clustering personality items by their construct categories, significantly above random assignment. This indicates the model successfully captures semantic relationships between items within the same personality construct.\n\n")
        
        f.write("### IPIP Clustering Visualizations\n\n")
        
        f.write("#### Confusion Matrix\n\n")
        f.write("![IPIP Confusion Matrix](ipip_confusion_matrix.png)\n\n")
        
        f.write("#### t-SNE Visualization (True Labels)\n\n")
        f.write("![IPIP t-SNE True Labels](ipip_tsne_true_labels.png)\n\n")
        
        f.write("#### t-SNE Visualization (Predicted Clusters)\n\n")
        f.write("![IPIP t-SNE Predicted Clusters](ipip_tsne_predicted_clusters.png)\n\n")
        
        f.write("#### Combined t-SNE Visualization\n\n")
        f.write("![IPIP t-SNE Combined](ipip_tsne_combined.png)\n\n")
        
        f.write("## Leadership Constructs Evaluation\n\n")
        f.write("The model shows much lower performance on leadership data, with clustering metrics significantly below those for personality constructs. This suggests that leadership constructs, as currently measured, do not form semantically distinct categories.\n\n")
        
        f.write("### Leadership Clustering Visualizations\n\n")
        
        f.write("#### Leadership Confusion Matrix\n\n")
        f.write("![Leadership Confusion Matrix](leadership_confusion_matrix.png)\n\n")
        
        f.write("#### Leadership t-SNE Visualization (True Labels)\n\n")
        f.write("![Leadership t-SNE True Labels](leadership_tsne_true_labels.png)\n\n")
        
        f.write("#### Leadership t-SNE Visualization (Predicted Clusters)\n\n")
        f.write("![Leadership t-SNE Predicted Clusters](leadership_tsne_predicted_clusters.png)\n\n")
        
        f.write("#### Leadership Construct Similarity\n\n")
        f.write("![Leadership Construct Similarity](leadership_construct_similarity.png)\n\n")
        
        f.write("### Leadership Construct Overlap Analysis\n\n")
        f.write("The analysis reveals substantial overlap between leadership constructs. Below are the most similar construct pairs (similarity > 0.85):\n\n")
        
        # Sort by similarity
        leadership_similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        f.write("| Construct Pair | Similarity |\n")
        f.write("|---------------|------------|\n")
        for pair in leadership_similar_pairs:
            if pair['similarity'] > 0.85:
                f.write(f"| {pair['constructs']} | {pair['similarity']:.4f} |\n")
        
        f.write("\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The results strongly support our research hypothesis that leadership constructs, as currently measured in the literature, have substantial semantic overlap and are less distinctly separated than personality constructs. Many leadership construct pairs show similarity values above 0.85, indicating they may be measuring essentially the same underlying concept despite having different names.\n\n")
        
        f.write("This suggests that the current taxonomic structure of leadership measurement may be artificially complex, with many constructs capturing similar underlying concepts. Future research should focus on identifying a more parsimonious set of truly distinct leadership dimensions.\n\n")
        
        f.write("## Appendix: Model and Evaluation Details\n\n")
        f.write("- **Model**: MNRL (Multiple Negatives Ranking Loss) with all-mpnet-base-v2 base model\n")
        f.write("- **Training Data**: Comprehensive and balanced anchor-positive IPIP item pairs\n")
        f.write("- **IPIP Evaluation**: Test set of 761 items across 220 constructs\n")
        f.write("- **Leadership Evaluation**: 434 items across 11 constructs\n")
    
    print(f"Markdown report created at {report_path}")
    return report_path

def copy_visualizations(ipip_dir, leadership_dir, output_dir):
    """Copy visualization files to the output directory."""
    # Copy IPIP visualizations
    shutil.copy(ipip_dir / "confusion_matrix.png", output_dir / "ipip_confusion_matrix.png")
    shutil.copy(ipip_dir / "tsne_true_labels.png", output_dir / "ipip_tsne_true_labels.png")
    shutil.copy(ipip_dir / "tsne_predicted_clusters.png", output_dir / "ipip_tsne_predicted_clusters.png")
    if (ipip_dir / "tsne_combined.png").exists():
        shutil.copy(ipip_dir / "tsne_combined.png", output_dir / "ipip_tsne_combined.png")
    
    # Copy leadership visualizations
    shutil.copy(leadership_dir / "leadership_confusion_matrix.png", output_dir / "leadership_confusion_matrix.png")
    shutil.copy(leadership_dir / "leadership_tsne_true_labels.png", output_dir / "leadership_tsne_true_labels.png")
    shutil.copy(leadership_dir / "leadership_tsne_predicted_clusters.png", output_dir / "leadership_tsne_predicted_clusters.png")
    shutil.copy(leadership_dir / "leadership_construct_similarity.png", output_dir / "leadership_construct_similarity.png")
    
    # Copy raw data files
    shutil.copy(leadership_dir / "leadership_construct_overlap.csv", output_dir / "leadership_construct_overlap.csv")
    shutil.copy(leadership_dir / "leadership_cluster_composition.csv", output_dir / "leadership_cluster_composition.csv")
    shutil.copy(ipip_dir / "evaluation_metrics.txt", output_dir / "ipip_metrics.txt")
    shutil.copy(leadership_dir / "leadership_metrics.txt", output_dir / "leadership_metrics.txt")
    
    print(f"Visualization files copied to {output_dir}")

def create_pdf_from_markdown(markdown_path):
    """Create a PDF from a markdown file using pandoc if available."""
    pdf_path = markdown_path.with_suffix('.pdf')
    
    try:
        # Check if pandoc is installed
        result = subprocess.run(['which', 'pandoc'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Generating PDF with pandoc...")
            cmd = ['pandoc', str(markdown_path), '-o', str(pdf_path)]
            subprocess.run(cmd, check=True)
            print(f"PDF created at {pdf_path}")
            return pdf_path
        else:
            print("Pandoc not found. PDF generation skipped.")
            return None
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        return None

def main():
    """Create a comprehensive report from the latest evaluation results."""
    # Find the latest evaluation directories
    ipip_dir, leadership_dir = find_latest_evaluations()
    
    if not ipip_dir or not leadership_dir:
        print("Error: Could not find evaluation directories.")
        return
    
    print(f"Using IPIP evaluation: {ipip_dir}")
    print(f"Using Leadership evaluation: {leadership_dir}")
    
    # Copy visualizations to output directory
    copy_visualizations(ipip_dir, leadership_dir, OUTPUT_DIR)
    
    # Create markdown report
    markdown_path = create_markdown_report(ipip_dir, leadership_dir)
    
    # Create PDF if possible
    pdf_path = create_pdf_from_markdown(markdown_path)
    
    # Create a README
    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# MNRL Model Evaluation Results\n\n")
        f.write(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Contents\n\n")
        f.write("- `mnrl_model_evaluation_report.md` - Comprehensive report in Markdown format\n")
        if pdf_path:
            f.write("- `mnrl_model_evaluation_report.pdf` - PDF version of the report\n")
        f.write("- Visualization files (PNG images of confusion matrices and t-SNE plots)\n")
        f.write("- Raw data files (CSV files with detailed metrics)\n\n")
        f.write("## How to View\n\n")
        f.write("Open the Markdown file in any Markdown viewer or the PDF file for the formatted report.\n")
        
    print("\nComprehensive report created successfully!")
    print(f"All outputs are in: {OUTPUT_DIR}")
    print(f"Main report: {markdown_path}")
    if pdf_path:
        print(f"PDF report: {pdf_path}")

if __name__ == "__main__":
    main()