#!/usr/bin/env python3
"""
Repository cleanup script to remove clutter and organize files.
Moves log files, temporary files, and organizes the repository structure.
"""

import os
import shutil
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for organization."""
    directories = [
        "logs",
        "scripts/archive_old",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")

def move_log_files(dry_run=False):
    """Move log files from root to logs directory."""
    log_files = list(Path(".").glob("*.log"))
    temp_files = ["training_pid.txt"]
    
    moved_count = 0
    
    # Move log files
    for log_file in log_files:
        target = Path("logs") / log_file.name
        if dry_run:
            logger.info(f"Would move: {log_file} -> {target}")
        else:
            try:
                shutil.move(str(log_file), str(target))
                logger.info(f"Moved: {log_file} -> {target}")
                moved_count += 1
            except Exception as e:
                logger.warning(f"Could not move {log_file}: {e}")
    
    # Move temporary files
    for temp_file in temp_files:
        if Path(temp_file).exists():
            target = Path("temp") / temp_file
            if dry_run:
                logger.info(f"Would move: {temp_file} -> {target}")
            else:
                try:
                    shutil.move(temp_file, str(target))
                    logger.info(f"Moved: {temp_file} -> {target}")
                    moved_count += 1
                except Exception as e:
                    logger.warning(f"Could not move {temp_file}: {e}")
    
    return moved_count

def archive_old_scripts(dry_run=False):
    """Archive old deprecated scripts."""
    # Scripts that are now deprecated in favor of unified pipeline
    deprecated_scripts = [
        "scripts/train_with_holdout.py",
        "scripts/train_with_tsdae_mac_studio.py",
    ]
    
    archived_count = 0
    
    for script in deprecated_scripts:
        if Path(script).exists():
            target = Path("scripts/archive_old") / Path(script).name
            if dry_run:
                logger.info(f"Would archive: {script} -> {target}")
            else:
                try:
                    shutil.move(script, str(target))
                    logger.info(f"Archived: {script} -> {target}")
                    archived_count += 1
                except Exception as e:
                    logger.warning(f"Could not archive {script}: {e}")
    
    return archived_count

def update_gitignore():
    """Update .gitignore to include new directories."""
    gitignore_entries = [
        "\n# Cleanup directories",
        "logs/",
        "temp/",
        "scripts/archive_old/",
        "\n# Additional patterns",
        "*.log",
        "*.pid",
        "*_output.log"
    ]
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = ""
    
    # Add entries that don't already exist
    new_entries = []
    for entry in gitignore_entries:
        if entry.strip() and entry.strip() not in existing_content:
            new_entries.append(entry)
    
    if new_entries:
        with open(gitignore_path, 'a') as f:
            f.write('\n'.join(new_entries))
        logger.info(f"Updated .gitignore with {len(new_entries)} new entries")
    else:
        logger.info(".gitignore is already up to date")

def clean_empty_directories():
    """Remove empty directories."""
    empty_dirs = []
    
    # Check for empty directories
    for root, dirs, files in os.walk(".", topdown=False):
        for directory in dirs:
            dir_path = Path(root) / directory
            try:
                if not any(dir_path.iterdir()):
                    empty_dirs.append(dir_path)
            except PermissionError:
                continue
    
    removed_count = 0
    for empty_dir in empty_dirs:
        # Skip protected directories
        if any(protected in str(empty_dir) for protected in ['.git', 'leadmeasure_env', '__pycache__']):
            continue
        
        try:
            empty_dir.rmdir()
            logger.info(f"Removed empty directory: {empty_dir}")
            removed_count += 1
        except Exception as e:
            logger.warning(f"Could not remove {empty_dir}: {e}")
    
    return removed_count

def generate_cleanup_report(moved_logs, archived_scripts, removed_dirs):
    """Generate a cleanup report."""
    report = f"""
Repository Cleanup Report
========================

Files moved to logs/: {moved_logs}
Scripts archived: {archived_scripts}
Empty directories removed: {removed_dirs}

Current structure:
├── logs/              # All log files
├── temp/              # Temporary files
├── scripts/
│   ├── ivan_analysis/ # Main pipeline (USE THIS)
│   ├── archive/       # Original deprecated scripts
│   └── archive_old/   # Recently deprecated scripts
└── ...

Next steps:
1. Use unified pipeline: scripts/ivan_analysis/
2. Old scripts preserved in archive directories
3. Logs organized in logs/ directory

Repository is now cleaner and better organized!
"""
    
    print(report)
    
    # Save report
    with open("cleanup_report.txt", 'w') as f:
        f.write(report)
    logger.info("Cleanup report saved to cleanup_report.txt")

def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description='Clean up repository clutter')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    args = parser.parse_args()
    
    logger.info("Starting repository cleanup...")
    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")
    
    # Setup directories
    if not args.dry_run:
        setup_directories()
    
    # Move files
    moved_logs = move_log_files(args.dry_run)
    archived_scripts = archive_old_scripts(args.dry_run)
    
    if not args.dry_run:
        # Update gitignore
        update_gitignore()
        
        # Clean empty directories
        removed_dirs = clean_empty_directories()
        
        # Generate report
        generate_cleanup_report(moved_logs, archived_scripts, removed_dirs)
    else:
        logger.info("Dry run complete. Run without --dry-run to perform cleanup.")

if __name__ == "__main__":
    main()