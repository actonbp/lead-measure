"""Generate a PDF report from evaluation results.

This script uses reportlab to create a PDF report from the evaluation results.
"""
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch

# Find the latest evaluation directories
def find_latest_evaluations():
    """Find the latest IPIP and leadership evaluation directories."""
    ipip_eval_dirs = list(Path("data/visualizations").glob("mnrl_evaluation_*"))
    leadership_eval_dirs = list(Path("data/visualizations").glob("leadership_mnrl_evaluation_*"))
    
    # Sort by creation time (newest first)
    ipip_eval_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    leadership_eval_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    return ipip_eval_dirs[0] if ipip_eval_dirs else None, leadership_eval_dirs[0] if leadership_eval_dirs else None

# Extract metrics from a file
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

# Parse leadership similarity data
def parse_leadership_similarity(metrics_file):
    """Parse leadership similarity data from metrics file."""
    similar_pairs = []
    with open(metrics_file, 'r') as f:
        content = f.read()
        
        # Try to find the similarity section
        if "Highly Similar Construct Pairs" in content:
            sim_section = content.split("Highly Similar Construct Pairs")[1].split("Generated on")[0]
            pair_matches = re.findall(r"  - ([^:]+): ([0-9.]+)", sim_section)
            for match in pair_matches:
                constructs, similarity = match
                similar_pairs.append({
                    'constructs': constructs.strip(),
                    'similarity': float(similarity)
                })
    
    # Sort by similarity
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_pairs

def create_pdf_report(ipip_dir, leadership_dir, output_path):
    """Create a PDF report from evaluation results."""
    # Extract metrics
    ipip_metrics = extract_metrics(ipip_dir / "evaluation_metrics.txt")
    leadership_metrics = extract_metrics(leadership_dir / "leadership_metrics.txt")
    leadership_similar_pairs = parse_leadership_similarity(leadership_dir / "leadership_metrics.txt")
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    styles['Title'].fontSize = 18
    styles['Title'].spaceAfter = 12
    
    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceAfter = 10
    
    styles['Heading3'].fontSize = 12
    styles['Heading3'].spaceAfter = 8
    
    story = []
    
    # Title and date
    story.append(Paragraph("MNRL Model Evaluation Report", styles['Title']))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Spacer(1, 0.25*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(
        "This report presents the evaluation results of our IPIP MNRL (Multiple Negatives Ranking Loss) "
        "model trained on comprehensive personality item pairs. We assess how effectively the model clusters "
        "both personality items and leadership items by their respective construct categories.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    # Key Findings
    story.append(Paragraph("Key Findings", styles['Heading3']))
    
    # IPIP Metrics
    story.append(Paragraph("IPIP Personality Data:", styles['Normal']))
    ipip_data = [
        ["Metric", "Value"],
        ["Adjusted Rand Index (ARI)", f"{ipip_metrics.get('ari', 0):.4f}"],
        ["Normalized Mutual Information (NMI)", f"{ipip_metrics.get('nmi', 0):.4f}"],
        ["Cluster Purity", f"{ipip_metrics.get('purity', 0):.4f}"]
    ]
    ipip_table = Table(ipip_data, colWidths=[3*inch, 1*inch])
    ipip_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(ipip_table)
    story.append(Spacer(1, 0.1*inch))
    
    # Leadership Metrics
    story.append(Paragraph("Leadership Construct Data:", styles['Normal']))
    leadership_data = [
        ["Metric", "Value"],
        ["Adjusted Rand Index (ARI)", f"{leadership_metrics.get('ari', 0):.4f}"],
        ["Normalized Mutual Information (NMI)", f"{leadership_metrics.get('nmi', 0):.4f}"],
        ["Cluster Purity", f"{leadership_metrics.get('purity', 0):.4f}"]
    ]
    leadership_table = Table(leadership_data, colWidths=[3*inch, 1*inch])
    leadership_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(leadership_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "The substantial drop in metrics from IPIP to leadership data supports our research hypothesis "
        "that leadership constructs have significant overlap and are less distinctly separated than "
        "personality constructs.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.25*inch))
    
    # IPIP Evaluation
    story.append(Paragraph("IPIP Personality Constructs Evaluation", styles['Heading2']))
    story.append(Paragraph(
        "The model shows moderate performance in clustering personality items by their construct categories, "
        "significantly above random assignment. This indicates the model successfully captures semantic "
        "relationships between items within the same personality construct.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    # IPIP Clustering Visualizations
    story.append(Paragraph("IPIP Clustering Visualizations", styles['Heading3']))
    
    # Confusion Matrix
    story.append(Paragraph("Confusion Matrix", styles['Heading3']))
    if (ipip_dir / "confusion_matrix.png").exists():
        img = Image(str(ipip_dir / "confusion_matrix.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    # t-SNE True Labels
    story.append(Paragraph("t-SNE Visualization (True Labels)", styles['Heading3']))
    if (ipip_dir / "tsne_true_labels.png").exists():
        img = Image(str(ipip_dir / "tsne_true_labels.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    # t-SNE Predicted Clusters
    story.append(Paragraph("t-SNE Visualization (Predicted Clusters)", styles['Heading3']))
    if (ipip_dir / "tsne_predicted_clusters.png").exists():
        img = Image(str(ipip_dir / "tsne_predicted_clusters.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    # Combined visualization if available
    if (ipip_dir / "tsne_combined.png").exists():
        story.append(Paragraph("Combined t-SNE Visualization", styles['Heading3']))
        img = Image(str(ipip_dir / "tsne_combined.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    story.append(Spacer(1, 0.25*inch))
    
    # Leadership Evaluation
    story.append(Paragraph("Leadership Constructs Evaluation", styles['Heading2']))
    story.append(Paragraph(
        "The model shows much lower performance on leadership data, with clustering metrics significantly "
        "below those for personality constructs. This suggests that leadership constructs, as currently "
        "measured, do not form semantically distinct categories.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    # Leadership Clustering Visualizations
    story.append(Paragraph("Leadership Clustering Visualizations", styles['Heading3']))
    
    # Leadership Confusion Matrix
    story.append(Paragraph("Leadership Confusion Matrix", styles['Heading3']))
    if (leadership_dir / "leadership_confusion_matrix.png").exists():
        img = Image(str(leadership_dir / "leadership_confusion_matrix.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    # Leadership t-SNE True Labels
    story.append(Paragraph("Leadership t-SNE Visualization (True Labels)", styles['Heading3']))
    if (leadership_dir / "leadership_tsne_true_labels.png").exists():
        img = Image(str(leadership_dir / "leadership_tsne_true_labels.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    # Leadership t-SNE Predicted Clusters
    story.append(Paragraph("Leadership t-SNE Visualization (Predicted Clusters)", styles['Heading3']))
    if (leadership_dir / "leadership_tsne_predicted_clusters.png").exists():
        img = Image(str(leadership_dir / "leadership_tsne_predicted_clusters.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    # Leadership Construct Similarity
    story.append(Paragraph("Leadership Construct Similarity", styles['Heading3']))
    if (leadership_dir / "leadership_construct_similarity.png").exists():
        img = Image(str(leadership_dir / "leadership_construct_similarity.png"))
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        story.append(img)
    
    story.append(Spacer(1, 0.1*inch))
    
    # Leadership Construct Overlap Analysis
    story.append(Paragraph("Leadership Construct Overlap Analysis", styles['Heading3']))
    story.append(Paragraph(
        "The analysis reveals substantial overlap between leadership constructs. "
        "Below are the most similar construct pairs (similarity > 0.85):",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    # Create a table of similar pairs
    if leadership_similar_pairs:
        high_similarity_pairs = [pair for pair in leadership_similar_pairs if pair['similarity'] > 0.85]
        if high_similarity_pairs:
            pair_data = [["Construct Pair", "Similarity"]]
            for pair in high_similarity_pairs:
                pair_data.append([pair['constructs'], f"{pair['similarity']:.4f}"])
            
            pair_table = Table(pair_data, colWidths=[4*inch, 1*inch])
            pair_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            story.append(pair_table)
    
    story.append(Spacer(1, 0.25*inch))
    
    # Conclusion
    story.append(Paragraph("Conclusion", styles['Heading2']))
    story.append(Paragraph(
        "The results strongly support our research hypothesis that leadership constructs, as currently "
        "measured in the literature, have substantial semantic overlap and are less distinctly separated "
        "than personality constructs. Many leadership construct pairs show similarity values above 0.85, "
        "indicating they may be measuring essentially the same underlying concept despite having different names.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph(
        "This suggests that the current taxonomic structure of leadership measurement may be artificially "
        "complex, with many constructs capturing similar underlying concepts. Future research should focus "
        "on identifying a more parsimonious set of truly distinct leadership dimensions.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.25*inch))
    
    # Appendix
    story.append(Paragraph("Appendix: Model and Evaluation Details", styles['Heading2']))
    appendix_data = [
        ["Model", f"MNRL (Multiple Negatives Ranking Loss) with all-mpnet-base-v2 base model"],
        ["Training Data", "Comprehensive and balanced anchor-positive IPIP item pairs"],
        ["IPIP Evaluation", "Test set of personality items across constructs"],
        ["Leadership Evaluation", "Leadership items across 11 leadership constructs"],
    ]
    
    for row in appendix_data:
        story.append(Paragraph(f"• {row[0]}: {row[1]}", styles['Normal']))
    
    # Build the PDF
    doc.build(story)
    print(f"PDF report created at {output_path}")

def main():
    # Find the latest evaluation directories
    ipip_dir, leadership_dir = find_latest_evaluations()
    
    if not ipip_dir or not leadership_dir:
        print("Error: Could not find evaluation directories.")
        sys.exit(1)
    
    print(f"Using IPIP evaluation: {ipip_dir}")
    print(f"Using Leadership evaluation: {leadership_dir}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("docs/output/model_evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the PDF report
    output_path = str(output_dir / "mnrl_model_evaluation_report.pdf")
    create_pdf_report(ipip_dir, leadership_dir, output_path)

if __name__ == "__main__":
    main()