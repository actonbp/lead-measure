#!/usr/bin/env python3
"""
Script to check the progress of the model training by looking for 
checkpoints and visualizing the results.
"""
import os
import glob
import argparse
import time
import json
from datetime import datetime

def find_latest_model():
    """Find the most recent model directory."""
    models = glob.glob("models/ipip_gist_*")
    if not models:
        return None
    # Sort by modification time, newest first
    return sorted(models, key=os.path.getmtime, reverse=True)[0]

def check_phase_status(model_dir):
    """Check the status of training phases in the model directory."""
    if not os.path.exists(model_dir):
        print(f"No model directory found at {model_dir}")
        return False
    
    print(f"Checking model: {model_dir}")
    
    # Check for training config
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Training configuration:")
        print(f"  Model: {config.get('base_model', 'unknown')}")
        print(f"  Loss function: {config.get('loss_fn', 'unknown')}")
        print(f"  Phases: {config.get('num_phases', 'unknown')}")
        print(f"  Epochs per phase: {config.get('epochs', 'unknown')}")
        total_epochs = int(config.get('epochs', 0)) * int(config.get('num_phases', 0))
        print(f"  Total epochs: {total_epochs}")
    
    # Check phase directories
    phase_dirs = glob.glob(os.path.join(model_dir, "phase_*"))
    completed_phases = len(phase_dirs)
    
    if completed_phases == 0:
        print("No phases have been completed yet.")
        return False
    
    print(f"Completed phases: {completed_phases}")
    
    # Check the latest phase
    latest_phase = sorted(phase_dirs, key=os.path.getmtime, reverse=True)[0]
    print(f"Latest phase: {os.path.basename(latest_phase)}")
    
    # Check for checkpoints in the latest phase
    checkpoints = glob.glob(os.path.join(latest_phase, "checkpoint-*"))
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=os.path.getmtime, reverse=True)[0]
        print(f"Latest checkpoint: {os.path.basename(last_checkpoint)}")
        checkpoint_time = datetime.fromtimestamp(os.path.getmtime(last_checkpoint))
        print(f"Checkpoint time: {checkpoint_time}")
        
        # Check how long ago the last checkpoint was created
        time_diff = time.time() - os.path.getmtime(last_checkpoint)
        minutes = int(time_diff / 60)
        print(f"Last checkpoint created: {minutes} minutes ago")
    else:
        print(f"No checkpoints found in {latest_phase}")
    
    # Check for the model in the latest phase
    model_path = os.path.join(latest_phase, "model")
    if os.path.exists(model_path):
        print(f"Model saved in the latest phase")
        model_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"Model save time: {model_time}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Check the progress of model training")
    parser.add_argument("--model_dir", type=str, help="Path to the model directory")
    args = parser.parse_args()
    
    model_dir = args.model_dir
    if not model_dir:
        model_dir = find_latest_model()
    
    if not model_dir:
        print("No models found. Training might not have started yet.")
        return
    
    check_phase_status(model_dir)

if __name__ == "__main__":
    main()