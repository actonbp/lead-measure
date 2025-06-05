# Makefile for Leadership Measurement Analysis
# June 2025

.PHONY: help setup test ivan-setup ivan-check ivan-step1 ivan-step2 ivan-step3 ivan-step4 ivan-all ivan-clean clean

# Default target
help:
	@echo "Leadership Measurement Analysis - Make Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          - Set up virtual environment and install dependencies"
	@echo "  make ivan-setup     - Set up Ivan's analysis environment"
	@echo ""
	@echo "Ivan's Analysis Commands:"
	@echo "  make ivan-check           - Check environment for Ivan's analysis"
	@echo "  make ivan-step1           - Generate randomized pairs"
	@echo "  make ivan-step2           - Train model (TSDAE + GIST)"
	@echo "  make ivan-step2-mac-studio - Train optimized for Mac Studio M1/M2 64GB"
	@echo "  make ivan-step3           - Analyze IPIP embeddings"
	@echo "  make ivan-step4           - Compare with baseline"
	@echo "  make ivan-all             - Run complete Ivan pipeline"
	@echo "  make ivan-clean           - Clean Ivan's analysis status"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean          - Clean temporary files and caches"
	@echo "  make test           - Run tests"

# Setup virtual environment
setup:
	@echo "Setting up virtual environment..."
	python3 -m venv leadmeasure_env
	./leadmeasure_env/bin/pip install --upgrade pip
	./leadmeasure_env/bin/pip install -r requirements.txt
	@echo "✓ Setup complete. Activate with: source leadmeasure_env/bin/activate"

# Setup for Ivan's analysis
ivan-setup:
	@echo "Setting up Ivan's analysis environment..."
	@if [ ! -d "leadmeasure_env" ]; then \
		make setup; \
	fi
	./leadmeasure_env/bin/pip install -r scripts/ivan_analysis/requirements.txt
	@echo "✓ Ivan's analysis setup complete"

# Check Ivan's environment
ivan-check:
	@echo "Checking Ivan's analysis environment..."
	python scripts/ivan_analysis/run_analysis_steps.py --check

# Ivan's analysis steps
ivan-step1:
	@echo "Running Step 1: Generate randomized pairs..."
	python scripts/ivan_analysis/run_analysis_steps.py --step 1

ivan-step2:
	@echo "Running Step 2: Train model (this will take time)..."
	python scripts/ivan_analysis/run_analysis_steps.py --step 2

ivan-step2-mac-studio:
	@echo "Running Step 2 optimized for Mac Studio M1/M2 64GB..."
	python scripts/ivan_analysis/train_with_tsdae_mac_studio.py

ivan-step3:
	@echo "Running Step 3: Analyze IPIP embeddings..."
	python scripts/ivan_analysis/run_analysis_steps.py --step 3

ivan-step4:
	@echo "Running Step 4: Compare with baseline..."
	python scripts/ivan_analysis/run_analysis_steps.py --step 4

# Run complete Ivan pipeline
ivan-all:
	@echo "Running complete Ivan analysis pipeline..."
	python scripts/ivan_analysis/run_analysis_steps.py --all

# Clean Ivan's status
ivan-clean:
	@echo "Cleaning Ivan's analysis status..."
	python scripts/ivan_analysis/run_analysis_steps.py --clean

# Run tests
test:
	@echo "Running tests..."
	@if [ -f "scripts/test_preprocessing.py" ]; then \
		python -m pytest scripts/test_preprocessing.py -v; \
	else \
		echo "No tests found"; \
	fi

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -f debug_output.log
	rm -f minimal_test_output.log
	rm -f local_model_training.log
	@echo "✓ Cleanup complete"