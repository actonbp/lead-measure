#!/bin/bash
# Script to organize data files into the proper directory structure

# Create necessary directories if they don't exist
mkdir -p data/raw data/processed data/metadata

# Move the measures file to the raw data directory
if [ -f "data/Measures_text_long.csv" ]; then
    echo "Moving Measures_text_long.csv to data/raw directory..."
    mv data/Measures_text_long.csv data/raw/
fi

# Check if the file was moved successfully
if [ -f "data/raw/Measures_text_long.csv" ]; then
    echo "Successfully organized data files."
else
    echo "Warning: Could not find or move Measures_text_long.csv"
fi

echo "Data organization complete." 