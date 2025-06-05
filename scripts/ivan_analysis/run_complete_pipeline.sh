#!/bin/bash
# Complete pipeline runner for Ivan's enhanced analysis.
# Handles platform detection and runs appropriate configuration.

echo "🚀 Ivan's Enhanced Analysis Pipeline"
echo "===================================="

# Detect OS
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected: $OS on $ARCH"

# Activate virtual environment
if [ -d "leadmeasure_env" ]; then
    echo "Activating virtual environment..."
    source leadmeasure_env/bin/activate
else
    echo "❌ Virtual environment not found!"
    echo "Please create it first with: python3 -m venv leadmeasure_env"
    exit 1
fi

# Check if holdout splits exist
if [ ! -f "data/processed/ipip_train_pairs_holdout.jsonl" ]; then
    echo ""
    echo "📊 Creating holdout validation splits..."
    python3 scripts/ivan_analysis/create_holdout_splits.py
    if [ $? -ne 0 ]; then
        echo "❌ Error creating holdout splits"
        exit 1
    fi
fi

# Determine configuration based on platform
echo ""
echo "🖥️  Platform Configuration:"

HIGH_MEMORY_FLAG=""
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "✅ Mac Silicon detected (M1/M2)"
    
    # Check available memory (Mac specific)
    TOTAL_MEM=$(sysctl -n hw.memsize)
    TOTAL_MEM_GB=$((TOTAL_MEM / 1073741824))
    echo "   Memory: ${TOTAL_MEM_GB}GB"
    
    if [ $TOTAL_MEM_GB -ge 32 ]; then
        echo "   Using high-memory optimizations"
        HIGH_MEMORY_FLAG="--high-memory"
    fi
elif command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HIGH_MEMORY_FLAG="--high-memory"
else
    echo "⚠️  No GPU acceleration detected, using CPU"
fi

# Run training
echo ""
echo "🔄 Starting training pipeline..."
echo "   Mode: Holdout validation (80/20 split)"
echo ""

python3 scripts/ivan_analysis/unified_training.py \
    --mode holdout \
    $HIGH_MEMORY_FLAG \
    --skip-tsdae

if [ $? -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi

# Run validation
echo ""
echo "📊 Running validation analysis..."
python3 scripts/ivan_analysis/validate_holdout_results.py

if [ $? -ne 0 ]; then
    echo "❌ Validation failed"
    exit 1
fi

echo ""
echo "✅ Pipeline complete!"
echo ""
echo "📁 Results saved to:"
echo "   - Models: models/gist_holdout_unified_final/"
echo "   - Visualizations: data/visualizations/holdout_validation/"
echo "   - Summary: data/visualizations/holdout_validation/holdout_validation_summary.txt"