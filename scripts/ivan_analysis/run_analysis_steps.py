#!/usr/bin/env python3
"""
Step-by-step runner for Ivan's analysis pipeline.
Allows running each step individually with validation checks.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalysisRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.ivan_scripts = self.project_root / "scripts/ivan_analysis"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.viz_dir = self.data_dir / "visualizations/ivan_analysis"
        
        # Track completed steps
        self.status_file = self.ivan_scripts / ".analysis_status.json"
        self.status = self.load_status()
    
    def load_status(self):
        """Load previous run status."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_status(self):
        """Save current status."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def check_environment(self):
        """Check if environment is properly set up."""
        logger.info("Checking environment setup...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            logger.error(f"Python 3.8+ required. Found: {sys.version}")
            return False
        
        # Check required packages
        required_packages = [
            'sentence_transformers',
            'torch',
            'pandas',
            'numpy',
            'sklearn',
            'nltk',
            'matplotlib',
            'seaborn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Run: pip install -r scripts/ivan_analysis/requirements.txt")
            return False
        
        # Check data files
        required_files = [
            self.data_dir / "IPIP.csv",
            self.data_dir / "processed/ipip_pairs_comprehensive.jsonl"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            logger.warning(f"Missing data files: {missing_files}")
            # First file is critical
            if not (self.data_dir / "IPIP.csv").exists():
                logger.error("Critical file IPIP.csv not found!")
                return False
        
        logger.info("✓ Environment check passed")
        return True
    
    def step1_generate_pairs(self):
        """Step 1: Generate randomized pairs."""
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Generate Randomized Pairs")
        logger.info("="*60)
        
        output_file = self.data_dir / "processed/ipip_pairs_randomized.jsonl"
        
        # Check if already completed
        if output_file.exists() and self.status.get('step1_completed'):
            logger.info("✓ Step 1 already completed. Output exists:")
            logger.info(f"  {output_file}")
            
            # Show statistics
            with open(output_file, 'r') as f:
                line_count = sum(1 for _ in f)
            logger.info(f"  Contains {line_count} pairs")
            
            response = input("\nRe-run this step? (y/N): ")
            if response.lower() != 'y':
                return True
        
        # Run the script
        logger.info("Running pair generation...")
        script_path = self.ivan_scripts / "build_pairs_randomized.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to generate pairs:\n{result.stderr}")
            return False
        
        # Verify output
        if not output_file.exists():
            logger.error("Output file not created!")
            return False
        
        # Check content
        with open(output_file, 'r') as f:
            first_line = f.readline()
            try:
                data = json.loads(first_line)
                if 'anchor' not in data or 'positive' not in data:
                    logger.error("Invalid pair format!")
                    return False
            except:
                logger.error("Failed to parse output file!")
                return False
        
        self.status['step1_completed'] = True
        self.status['step1_timestamp'] = datetime.now().isoformat()
        self.save_status()
        
        logger.info("✓ Step 1 completed successfully!")
        return True
    
    def step2_train_model(self):
        """Step 2: Train with TSDAE and GIST loss."""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Train Model (TSDAE + GIST)")
        logger.info("="*60)
        
        final_model = self.models_dir / "ivan_tsdae_gist_final"
        
        # Check if already completed
        if final_model.exists() and self.status.get('step2_completed'):
            logger.info("✓ Step 2 already completed. Models exist:")
            logger.info(f"  {final_model}")
            
            response = input("\nRe-run training? This will take time! (y/N): ")
            if response.lower() != 'y':
                return True
        
        # Check prerequisites
        pairs_file = self.data_dir / "processed/ipip_pairs_randomized.jsonl"
        if not pairs_file.exists():
            logger.error("Pairs file not found! Run step 1 first.")
            return False
        
        logger.info("Starting training (this may take 30-60 minutes)...")
        logger.info("Training will proceed in phases:")
        logger.info("  - TSDAE pre-training (1 epoch)")
        logger.info("  - GIST training (5 phases x 10 epochs)")
        
        script_path = self.ivan_scripts / "train_with_tsdae.py"
        
        # Run with real-time output
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            logger.error("Training failed!")
            return False
        
        # Verify output
        if not final_model.exists():
            logger.error("Final model not created!")
            return False
        
        self.status['step2_completed'] = True
        self.status['step2_timestamp'] = datetime.now().isoformat()
        self.save_status()
        
        logger.info("✓ Step 2 completed successfully!")
        return True
    
    def step3_analyze_ipip(self):
        """Step 3: Analyze IPIP embeddings."""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Analyze IPIP Embeddings")
        logger.info("="*60)
        
        model_path = self.models_dir / "ivan_tsdae_gist_final"
        
        # Check prerequisites
        if not model_path.exists():
            logger.error("Trained model not found! Run step 2 first.")
            return False
        
        # Expected outputs
        outputs = [
            self.viz_dir / "tsne_perplexity30.png",
            self.viz_dir / "tsne_perplexity15.png",
            self.viz_dir / "similarity_analysis.csv"
        ]
        
        if all(f.exists() for f in outputs) and self.status.get('step3_completed'):
            logger.info("✓ Step 3 already completed. Outputs exist:")
            for f in outputs:
                logger.info(f"  {f.name}")
            
            response = input("\nRe-run analysis? (y/N): ")
            if response.lower() != 'y':
                return True
        
        logger.info("Running IPIP analysis...")
        script_path = self.ivan_scripts / "visualize_and_analyze.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--model", str(model_path),
             "--dataset", "ipip"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Analysis failed:\n{result.stderr}")
            return False
        
        # Show results
        if (self.viz_dir / "similarity_analysis.csv").exists():
            df = pd.read_csv(self.viz_dir / "similarity_analysis.csv")
            logger.info("\nSimilarity Analysis Results:")
            logger.info(f"  Probability(same > diff): {df['probability_same_higher'].iloc[0]:.2%}")
            logger.info(f"  Cohen's d: {df['cohens_d'].iloc[0]:.3f}")
        
        self.status['step3_completed'] = True
        self.status['step3_timestamp'] = datetime.now().isoformat()
        self.save_status()
        
        logger.info("✓ Step 3 completed successfully!")
        return True
    
    def step4_compare_baseline(self):
        """Step 4: Compare with baseline."""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Compare with Baseline")
        logger.info("="*60)
        
        model_path = self.models_dir / "ivan_tsdae_gist_final"
        comparison_file = self.viz_dir / "model_comparison.csv"
        
        if comparison_file.exists() and self.status.get('step4_completed'):
            logger.info("✓ Step 4 already completed.")
            
            # Show comparison
            df = pd.read_csv(comparison_file)
            logger.info("\nModel Comparison:")
            print(df.to_string(index=False))
            
            response = input("\nRe-run comparison? (y/N): ")
            if response.lower() != 'y':
                return True
        
        logger.info("Running baseline comparison...")
        logger.info("This will download BGE-M3 model on first run (~2GB)")
        
        script_path = self.ivan_scripts / "visualize_and_analyze.py"
        
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--model", str(model_path),
             "--compare-baseline"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Comparison failed:\n{result.stderr}")
            return False
        
        self.status['step4_completed'] = True
        self.status['step4_timestamp'] = datetime.now().isoformat()
        self.save_status()
        
        logger.info("✓ Step 4 completed successfully!")
        return True
    
    def run_all(self):
        """Run all steps in sequence."""
        steps = [
            self.step1_generate_pairs,
            self.step2_train_model,
            self.step3_analyze_ipip,
            self.step4_compare_baseline
        ]
        
        for i, step in enumerate(steps, 1):
            if not step():
                logger.error(f"Failed at step {i}. Fix issues and re-run.")
                return False
        
        logger.info("\n" + "="*60)
        logger.info("ALL STEPS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        return True
    
    def show_status(self):
        """Show current analysis status."""
        logger.info("\nAnalysis Status:")
        logger.info("-" * 40)
        
        steps = [
            ('step1_completed', 'Generate Pairs'),
            ('step2_completed', 'Train Model'),
            ('step3_completed', 'Analyze IPIP'),
            ('step4_completed', 'Compare Baseline')
        ]
        
        for key, name in steps:
            if self.status.get(key):
                timestamp = self.status.get(key.replace('_completed', '_timestamp'), 'Unknown')
                logger.info(f"✓ {name}: Completed at {timestamp}")
            else:
                logger.info(f"✗ {name}: Not completed")
    
    def clean_status(self):
        """Clean status to start fresh."""
        if self.status_file.exists():
            self.status_file.unlink()
        self.status = {}
        logger.info("Status cleaned. Ready for fresh run.")

def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Ivan's analysis pipeline step by step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis_steps.py --check        # Check environment
  python run_analysis_steps.py --status       # Show progress
  python run_analysis_steps.py --step 1       # Run specific step
  python run_analysis_steps.py --all          # Run all steps
  python run_analysis_steps.py --clean        # Clean status
        """
    )
    
    parser.add_argument('--check', action='store_true', help='Check environment setup')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--step', type=int, choices=[1,2,3,4], help='Run specific step')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--clean', action='store_true', help='Clean status for fresh start')
    
    args = parser.parse_args()
    
    runner = AnalysisRunner()
    
    # Always check environment first
    if not runner.check_environment():
        logger.error("Environment check failed. Fix issues before proceeding.")
        sys.exit(1)
    
    if args.check:
        # Already checked above
        pass
    elif args.status:
        runner.show_status()
    elif args.clean:
        runner.clean_status()
    elif args.step:
        step_methods = {
            1: runner.step1_generate_pairs,
            2: runner.step2_train_model,
            3: runner.step3_analyze_ipip,
            4: runner.step4_compare_baseline
        }
        step_methods[args.step]()
    elif args.all:
        runner.run_all()
    else:
        # Interactive mode
        runner.show_status()
        print("\nWhat would you like to do?")
        print("1. Run Step 1: Generate Pairs")
        print("2. Run Step 2: Train Model")
        print("3. Run Step 3: Analyze IPIP")
        print("4. Run Step 4: Compare Baseline")
        print("5. Run All Steps")
        print("6. Clean Status")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ")
        
        if choice == '1':
            runner.step1_generate_pairs()
        elif choice == '2':
            runner.step2_train_model()
        elif choice == '3':
            runner.step3_analyze_ipip()
        elif choice == '4':
            runner.step4_compare_baseline()
        elif choice == '5':
            runner.run_all()
        elif choice == '6':
            runner.clean_status()

if __name__ == "__main__":
    main()