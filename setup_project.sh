#!/bin/bash
# Script to set up the project structure for leadership embedding analysis

echo "Setting up project structure for leadership embedding analysis..."

# Create directory structure
mkdir -p data/raw data/processed data/metadata
mkdir -p analyses
mkdir -p outputs
mkdir -p docs

# Move the measures file to the raw data directory
if [ -f "data/Measures_text_long.csv" ]; then
    echo "Moving Measures_text_long.csv to data/raw directory..."
    mv data/Measures_text_long.csv data/raw/
    echo "Data file moved successfully."
else
    echo "Note: Measures_text_long.csv not found in data/ directory. If it exists elsewhere, please move it to data/raw/ manually."
fi

# Create empty __init__.py files for Python import structure
touch analyses/__init__.py

# Make analyses script executable
if [ -f "analyses/leadership_embedding_analysis.py" ]; then
    chmod +x analyses/leadership_embedding_analysis.py
    echo "Made analysis script executable."
fi

echo "Project structure setup complete. Directory structure:"
ls -la

echo ""
echo "Getting ready to run initial analysis..."
echo "To analyze leadership constructs, run:"
echo "  python analyses/leadership_embedding_analysis.py"
echo ""
echo "Note: This script requires Python packages listed in requirements.txt."
echo "Install them with: pip install -r requirements.txt" 