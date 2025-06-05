#!/usr/bin/env python3
"""
Generate Comprehensive PDF Report for Leadership Analysis

This script combines the markdown reports and visualizations into a single PDF document
using the python-markdown and weasyprint libraries.

Usage:
    python generate_pdf_report.py

Output:
    A PDF file in the docs directory
"""

import os
import markdown
import sys
from pathlib import Path
from datetime import datetime
from weasyprint import HTML, CSS
import shutil
import argparse

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
VISUALIZATIONS_DIR = DATA_DIR / "visualizations"
OUTPUT_DIR = DOCS_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def read_markdown_file(file_path):
    """Read content from a markdown file."""
    with open(file_path, 'r') as f:
        return f.read()

def convert_markdown_to_html(markdown_content):
    """Convert markdown content to HTML."""
    extensions = ['tables', 'fenced_code', 'codehilite']
    return markdown.markdown(markdown_content, extensions=extensions)

def create_combined_html(datasets_report, embeddings_report, visualizations):
    """Create a combined HTML document with all content."""
    
    # Convert markdown to HTML
    datasets_html = convert_markdown_to_html(datasets_report)
    embeddings_html = convert_markdown_to_html(embeddings_report)
    
    # Create HTML for visualizations
    vis_html = "<h1>Visualization Results</h1>\n"
    for i, (title, path) in enumerate(visualizations):
        # Copy visualization to temp directory
        vis_filename = f"visualization_{i}.png"
        shutil.copy(path, TEMP_DIR / vis_filename)
        
        # Add to HTML
        vis_html += f"<h2>{title}</h2>\n"
        vis_html += f"<img src='{vis_filename}' style='max-width:100%; height:auto;' />\n"
        vis_html += "<br/><br/>\n"
    
    # Create the complete HTML document
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Leadership Measurement Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 2em;
                font-size: 12px;
            }}
            h1 {{
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 0.5em;
                font-size: 24px;
            }}
            h2 {{
                color: #444;
                margin-top: 1.5em;
                font-size: 20px;
            }}
            h3 {{
                color: #555;
                font-size: 16px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1em 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 1em auto;
                page-break-inside: avoid;
            }}
            .page-break {{
                page-break-before: always;
            }}
            ul, ol {{
                margin-bottom: 1em;
            }}
            .footer {{
                text-align: center;
                margin-top: 2em;
                font-size: 10px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <h1>Leadership Measurement Analysis: Comprehensive Report</h1>
        <p>Generated on: {datetime.now().strftime('%B %d, %Y')}</p>
        
        {datasets_html}
        
        <div class="page-break"></div>
        
        {embeddings_html}
        
        <div class="page-break"></div>
        
        {vis_html}
        
        <div class="footer">
            <p>Leadership Measurement Analysis Project - {datetime.now().year}</p>
        </div>
    </body>
    </html>
    """
    
    return html

def save_html_to_pdf(html_content, output_path):
    """Convert HTML content to PDF and save to the output path."""
    html = HTML(string=html_content, base_url=str(TEMP_DIR))
    html.write_pdf(output_path)
    print(f"PDF report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive PDF report")
    parser.add_argument('--output', default='leadership_analysis_report.pdf',
                       help='Output filename for the PDF report')
    args = parser.parse_args()
    
    # Read the markdown reports
    datasets_report = read_markdown_file(PROCESSED_DIR / "leadership_datasets_comprehensive_report.md")
    embeddings_report = read_markdown_file(PROCESSED_DIR / "leadership_embeddings_analysis_report.md")
    
    # Find visualizations
    visualizations = []
    for file in VISUALIZATIONS_DIR.glob("*.png"):
        if "umap" in file.name:
            title = "UMAP Visualization of Leadership Constructs"
            visualizations.append((title, file))
        elif "similarity_matrix" in file.name:
            title = "Semantic Similarity Between Leadership Constructs"
            visualizations.append((title, file))
    
    # Sort visualizations (UMAP first, then similarity matrices)
    visualizations.sort(key=lambda x: 0 if "umap" in str(x[1]) else 1)
    
    # Create HTML content
    html_content = create_combined_html(datasets_report, embeddings_report, visualizations)
    
    # Save HTML to temp file for debugging
    with open(TEMP_DIR / "report.html", "w") as f:
        f.write(html_content)
    
    # Generate PDF
    output_path = OUTPUT_DIR / args.output
    save_html_to_pdf(html_content, output_path)
    
    print(f"\nReport generation completed successfully.")
    print(f"HTML preview saved to: {TEMP_DIR / 'report.html'}")
    print(f"PDF report saved to: {output_path}")

if __name__ == "__main__":
    main() 