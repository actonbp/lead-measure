#!/usr/bin/env python3
"""
Background training runner for long-running experiments.
Launches training in background and provides monitoring capabilities.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import argparse
import signal

def check_running_training():
    """Check if training is already running."""
    pid_file = Path("logs/training.pid")
    if pid_file.exists():
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        try:
            # Check if process is still running
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            return pid
        except ProcessLookupError:
            # Process no longer exists, clean up
            pid_file.unlink()
    return None

def start_background_training(script_args):
    """Start training in background."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "scripts/ivan_analysis/unified_training.py"
    ] + script_args
    
    # Log files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"
    pid_file = "logs/training.pid"
    status_file = "logs/training_status.json"
    
    print(f"🚀 Starting training in background...")
    print(f"📝 Logs will be written to: {log_file}")
    
    # Start process
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from terminal
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Ensure unbuffered output
        )
    
    # Save PID
    with open(pid_file, 'w') as f:
        f.write(str(process.pid))
    
    # Save status info
    status = {
        "pid": process.pid,
        "start_time": timestamp,
        "log_file": log_file,
        "command": " ".join(cmd),
        "args": script_args
    }
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"✅ Training started with PID: {process.pid}")
    
    return process.pid, log_file

def monitor_training(pid=None):
    """Monitor running training."""
    if pid is None:
        pid = check_running_training()
        if pid is None:
            print("❌ No training process found")
            return
    
    # Load status
    status_file = Path("logs/training_status.json")
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        print(f"\n📊 Training Status:")
        print(f"  PID: {status['pid']}")
        print(f"  Started: {status['start_time']}")
        print(f"  Log file: {status['log_file']}")
        
        # Check if still running
        try:
            os.kill(pid, 0)
            print(f"  Status: 🟢 Running")
            
            # Show last few lines of log
            print(f"\n📜 Recent log output:")
            print("-" * 50)
            os.system(f"tail -n 20 {status['log_file']}")
            
        except ProcessLookupError:
            print(f"  Status: 🔴 Stopped")
    else:
        print("❌ No status file found")

def stop_training():
    """Stop running training."""
    pid = check_running_training()
    if pid is None:
        print("❌ No training process found")
        return
    
    print(f"🛑 Stopping training process {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
        print("✅ Training stopped")
        
        # Clean up PID file
        pid_file = Path("logs/training.pid")
        if pid_file.exists():
            pid_file.unlink()
    except ProcessLookupError:
        print("❌ Process already stopped")

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Background training manager')
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start training in background')
    start_parser.add_argument('--mode', choices=['full', 'holdout'], default='holdout',
                            help='Training mode')
    start_parser.add_argument('--method', choices=['item_level', 'construct_level'], default='item_level',
                            help='Holdout method: item_level (original) or construct_level (Ivan\'s method)')
    start_parser.add_argument('--high-memory', action='store_true',
                            help='Use high memory optimizations')
    start_parser.add_argument('--skip-tsdae', action='store_true',
                            help='Skip TSDAE pre-training if model exists')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor running training')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop running training')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        # Check if already running
        pid = check_running_training()
        if pid:
            print(f"⚠️  Training already running with PID {pid}")
            print("Use 'monitor' to check status or 'stop' to terminate")
            return
        
        # Build script arguments
        script_args = ['--mode', args.mode, '--method', args.method]
        if args.high_memory:
            script_args.append('--high-memory')
        if args.skip_tsdae:
            script_args.append('--skip-tsdae')
        
        # Start training
        pid, log_file = start_background_training(script_args)
        
        print("\n📋 Monitoring commands:")
        print(f"  Watch logs: tail -f {log_file}")
        print(f"  Check status: python3 {__file__} monitor")
        print(f"  Stop training: python3 {__file__} stop")
        
    elif args.action == 'monitor':
        monitor_training()
        
    elif args.action == 'stop':
        stop_training()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()