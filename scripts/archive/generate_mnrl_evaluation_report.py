"""Generate a comprehensive evaluation report for the MNRL model.

This script:
1. Runs the model evaluation to get metrics and visualizations
2. Updates the Quarto report template with the correct paths
3. Renders the final PDF report

Usage:
    python scripts/generate_mnrl_evaluation_report.py [--model_path MODEL_PATH]

Requirements:
    - Quarto CLI installed (https://quarto.org/docs/get-started/)
    - Python packages: pandas, matplotlib, scikit-learn
"""
import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path
import shutil

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate evaluation report for MNRL model")
parser.add_argument("--model_path", type=str, default="models/ipip_mnrl_20250515_1328", 
                    help="Path to the trained model directory")
args = parser.parse_args()

def run_evaluation():
    """Run the evaluation script and return the results directory."""
    # Create results directory name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = f"data/visualizations/mnrl_evaluation_{timestamp}"
    
    # Run the evaluation script
    print(f"Running model evaluation on {args.model_path}...")
    cmd = [sys.executable, "scripts/evaluate_mnrl_model.py", 
           "--model_path", args.model_path, 
           "--output_dir", results_dir]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running evaluation:")
        print(result.stderr)
        sys.exit(1)
    
    print(result.stdout)
    return results_dir

def update_qmd_template(results_dir):
    """Update the Quarto template with the correct results path."""
    template_path = "docs/quarto/ipip_mnrl_evaluation.qmd"
    output_path = f"docs/quarto/ipip_mnrl_evaluation_{Path(results_dir).name}.qmd"
    
    # Read the template
    with open(template_path, "r") as f:
        content = f.read()
    
    # Replace the placeholder with the actual results path
    content = content.replace("RESULTS_PATH_PLACEHOLDER", results_dir)
    
    # Write the updated file
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"Updated Quarto template saved to {output_path}")
    return output_path

def render_pdf(qmd_path):
    """Render the Quarto document to PDF."""
    # Make sure output directory exists
    output_dir = Path("docs/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get filename without extension for output name
    base_name = Path(qmd_path).stem
    output_file = output_dir / f"{base_name}.pdf"
    
    # Run quarto render
    print(f"Rendering PDF report...")
    cmd = ["quarto", "render", qmd_path, "--to", "pdf", "-o", str(output_file)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error rendering PDF:")
            print(result.stderr)
            
            # Try alternative approach using pandoc directly
            print("Attempting alternative rendering method...")
            html_path = output_dir / f"{base_name}.html"
            
            # First render to HTML
            html_cmd = ["quarto", "render", qmd_path, "--to", "html", "-o", str(html_path)]
            html_result = subprocess.run(html_cmd, capture_output=True, text=True)
            
            if html_result.returncode == 0:
                # Then convert HTML to PDF
                pdf_cmd = ["pandoc", str(html_path), "-o", str(output_file)]
                pdf_result = subprocess.run(pdf_cmd, capture_output=True, text=True)
                
                if pdf_result.returncode == 0:
                    print(f"PDF report generated at {output_file}")
                else:
                    print("Alternative rendering failed:")
                    print(pdf_result.stderr)
        else:
            print(f"PDF report generated at {output_file}")
            return str(output_file)
    except Exception as e:
        print(f"Error rendering PDF: {str(e)}")
        print("Quarto render failed. Please make sure Quarto is installed: https://quarto.org/docs/get-started/")
        print("You can manually render the report by running: quarto render", qmd_path, "--to pdf")
    
    return None

def main():
    """Run the entire pipeline from evaluation to report generation."""
    # Run the evaluation
    results_dir = run_evaluation()
    
    # Update the Quarto template
    qmd_path = update_qmd_template(results_dir)
    
    # Render the PDF
    pdf_path = render_pdf(qmd_path)
    
    if pdf_path:
        print("\nEvaluation and reporting complete!")
        print(f"- Evaluation results: {results_dir}")
        print(f"- Quarto document: {qmd_path}")
        print(f"- PDF report: {pdf_path}")
    else:
        print("\nEvaluation complete, but PDF rendering failed.")
        print(f"- Evaluation results: {results_dir}")
        print(f"- Quarto document: {qmd_path}")
        print("You can manually render the PDF using Quarto or another tool.")

if __name__ == "__main__":
    main()