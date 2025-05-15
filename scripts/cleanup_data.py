#!/usr/bin/env python3
"""
Cleanup script for data directories.

This script identifies and optionally deletes redundant or temporary data files
while preserving essential files needed for the analysis workflow.

Usage:
    python scripts/cleanup_data.py [--dry-run] [--all]

Arguments:
    --dry-run: Show what would be deleted without actually deleting
    --all: More aggressive cleanup, including intermediate results files

Examples:
    # Show what would be deleted without actually deleting
    python scripts/cleanup_data.py --dry-run

    # Perform basic cleanup (keeps essential files)
    python scripts/cleanup_data.py

    # Perform aggressive cleanup (only keeps raw data and final results)
    python scripts/cleanup_data.py --all
"""

import os
import argparse
import logging
import shutil
import fnmatch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define paths relative to project root
PROCESSED_DIR = "data/processed"
VISUALIZATIONS_DIR = "data/visualizations"
TEMP_DIRS = [
    "docs/output/temp",
    "data/visualizations/temp"
]

# Files that are safe to delete in basic cleanup (redundant or intermediate)
BASIC_CLEANUP_PATTERNS = [
    "*.tmp",
    "temp_*",
    "*_backup_*",
    "*.bak",
    "*_old.*",
    "*.pkl.gz",  # Compressed pickle files if uncompressed versions exist
    "*_debug_*",
]

# Additional files/patterns for aggressive cleanup (keep only essential files)
AGGRESSIVE_CLEANUP_PATTERNS = [
    "embedding_analysis_results.pkl",  # Large intermediate results
    "*_intermediate_*.pkl",
    "*_checkpoint_*.pkl",
    "*_analysis_*.pkl",  # Analysis results that can be regenerated
]

# Always preserve these files regardless of cleanup level
PRESERVE_PATTERNS = [
    "ipip_pairs_comprehensive.jsonl",  # Latest training data
    "leadership_focused_clean.csv",    # Latest clean leadership data
    "README.md",                       # Documentation files
    "*.md",                            # All markdown documentation files
    "final_*.csv",                     # Any final result CSV files
    "final_*.json",                    # Any final result JSON files
]

def should_preserve(file_path, preserve_patterns):
    """Check if a file matches any preservation pattern."""
    file_name = os.path.basename(file_path)
    return any(fnmatch.fnmatch(file_name, pattern) for pattern in preserve_patterns)

def should_delete(file_path, delete_patterns, preserve_patterns):
    """Check if a file matches any deletion pattern and doesn't match preservation patterns."""
    if should_preserve(file_path, preserve_patterns):
        return False
        
    file_name = os.path.basename(file_path)
    return any(fnmatch.fnmatch(file_name, pattern) for pattern in delete_patterns)

def format_size(size_bytes):
    """Format file size in a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def list_files_for_cleanup(directory, delete_patterns, preserve_patterns):
    """List files in directory that match deletion patterns but not preservation patterns."""
    files_to_delete = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory '{directory}' does not exist.")
        return files_to_delete
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if should_delete(file_path, delete_patterns, preserve_patterns):
                files_to_delete.append({
                    'path': file_path,
                    'size': os.path.getsize(file_path)
                })
    
    return files_to_delete

def cleanup_temp_dirs(temp_dirs, dry_run=False):
    """Remove temporary directories."""
    deleted_count = 0
    freed_bytes = 0
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            dir_size = sum(f.stat().st_size for f in Path(temp_dir).glob('**/*') if f.is_file())
            
            if dry_run:
                logger.info(f"Would delete temp directory: {temp_dir} ({format_size(dir_size)})")
            else:
                try:
                    logger.info(f"Deleting temp directory: {temp_dir} ({format_size(dir_size)})")
                    shutil.rmtree(temp_dir)
                    deleted_count += 1
                    freed_bytes += dir_size
                except Exception as e:
                    logger.error(f"Error deleting directory {temp_dir}: {e}")
    
    return deleted_count, freed_bytes

def cleanup_data_files(directories, delete_patterns, preserve_patterns, dry_run=False):
    """Clean up data files based on patterns."""
    total_deleted = 0
    total_freed = 0
    
    # Process each directory
    for directory in directories:
        files_to_delete = list_files_for_cleanup(directory, delete_patterns, preserve_patterns)
        
        if not files_to_delete:
            logger.info(f"No files to delete in {directory}")
            continue
            
        logger.info(f"\nFound {len(files_to_delete)} files to delete in {directory}:")
        
        for file_info in files_to_delete:
            file_path = file_info['path']
            file_size = file_info['size']
            
            if dry_run:
                logger.info(f"  Would delete: {file_path} ({format_size(file_size)})")
                total_freed += file_size
            else:
                try:
                    logger.info(f"  Deleting: {file_path} ({format_size(file_size)})")
                    os.remove(file_path)
                    total_deleted += 1
                    total_freed += file_size
                except Exception as e:
                    logger.error(f"  Error deleting {file_path}: {e}")
    
    return total_deleted, total_freed

def main():
    parser = argparse.ArgumentParser(
        description="Clean up temporary and redundant data files while preserving essential ones."
    )
    parser.add_argument('--dry-run', action='store_true', 
                        help="Show what would be deleted without actually deleting")
    parser.add_argument('--all', action='store_true',
                        help="More aggressive cleanup, including intermediate results files")
    
    args = parser.parse_args()
    
    logger.info("Data cleanup script")
    logger.info("==================")
    
    if args.dry_run:
        logger.info("Running in dry-run mode (no files will be deleted)")
    
    # Define cleanup patterns based on aggressiveness level
    delete_patterns = BASIC_CLEANUP_PATTERNS.copy()
    if args.all:
        logger.info("Performing aggressive cleanup")
        delete_patterns.extend(AGGRESSIVE_CLEANUP_PATTERNS)
    else:
        logger.info("Performing basic cleanup")
    
    # Clean up temporary directories
    logger.info("\nChecking temporary directories...")
    temp_dirs_deleted, temp_dirs_freed = cleanup_temp_dirs(TEMP_DIRS, args.dry_run)
    
    # Clean up data files
    data_dirs = [PROCESSED_DIR, VISUALIZATIONS_DIR]
    logger.info("\nChecking data directories...")
    data_deleted, data_freed = cleanup_data_files(
        data_dirs, delete_patterns, PRESERVE_PATTERNS, args.dry_run
    )
    
    # Report results
    total_freed = temp_dirs_freed + data_freed
    total_deleted = temp_dirs_deleted + data_deleted
    
    logger.info("\nCleanup Summary:")
    logger.info("----------------")
    
    if args.dry_run:
        logger.info(f"Would delete {total_deleted} directories and files")
        logger.info(f"Would free up approximately {format_size(total_freed)}")
    else:
        logger.info(f"Deleted {total_deleted} directories and files")
        logger.info(f"Freed up approximately {format_size(total_freed)}")

if __name__ == "__main__":
    main() 