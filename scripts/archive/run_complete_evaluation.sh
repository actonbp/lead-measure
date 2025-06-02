#!/bin/bash
# Run the complete evaluation workflow for the newly trained MNRL model

set -e  # Exit immediately if any command fails

# Set the model path
MODEL_PATH="models/ipip_mnrl_20250515_1328"

echo "===== Starting Complete MNRL Model Evaluation ====="
echo "Using model: $MODEL_PATH"
echo

# Step 1: Evaluate model on IPIP data
echo "Step 1: Evaluating MNRL model on IPIP data..."
OUTPUT_DIR_IPIP=$(python scripts/evaluate_mnrl_model.py --model_path "$MODEL_PATH" | grep "Results saved to" | cut -d' ' -f4)
echo "IPIP evaluation complete. Results in: $OUTPUT_DIR_IPIP"
echo

# Step 2: Evaluate model on leadership data
echo "Step 2: Evaluating MNRL model on leadership data..."
OUTPUT_DIR_LEADERSHIP=$(python scripts/evaluate_mnrl_on_leadership.py --model_path "$MODEL_PATH" | grep "Results saved to" | cut -d' ' -f4)
echo "Leadership evaluation complete. Results in: $OUTPUT_DIR_LEADERSHIP"
echo

# Step 3: Generate the IPIP evaluation report
echo "Step 3: Generating IPIP evaluation report..."
python scripts/generate_mnrl_evaluation_report.py --model_path "$MODEL_PATH"
echo

# Step 4: Create a symbolic link to the latest results for easy access
mkdir -p data/visualizations/latest_results
rm -f data/visualizations/latest_results/ipip
rm -f data/visualizations/latest_results/leadership
ln -sf ../"$(basename "$OUTPUT_DIR_IPIP")" data/visualizations/latest_results/ipip
ln -sf ../"$(basename "$OUTPUT_DIR_LEADERSHIP")" data/visualizations/latest_results/leadership

echo "===== Evaluation Workflow Complete ====="
echo "- IPIP evaluation: $OUTPUT_DIR_IPIP"
echo "- Leadership evaluation: $OUTPUT_DIR_LEADERSHIP"
echo "- Symbolic links created in data/visualizations/latest_results/"
echo "- Reports saved in docs/output/"
echo
echo "You can now view the full results in those directories."