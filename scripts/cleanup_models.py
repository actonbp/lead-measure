#!/usr/bin/env python3
"""
Cleanup script for model directories.

This script finds and deletes old model directories, keeping only the most recent N models 
of each type. Model types are detected by their naming pattern (e.g., ipip_mnrl_*, ipip_triplet_*).

Usage:
    python scripts/cleanup_models.py [--keep 1] [--dry-run] [--path models]

Arguments:
    --keep N: Number of most recent models of each type to keep (default: 1)
    --dry-run: Show what would be deleted without actually deleting
    --path: Directory containing model folders (default: models/)

Examples:
    # Keep only the most recent model of each type
    python scripts/cleanup_models.py

    # Keep the 2 most recent models of each type
    python scripts/cleanup_models.py --keep 2

    # Show what would be deleted without actually deleting
    python scripts/cleanup_models.py --dry-run
"""

import os
import argparse
import re
from datetime import datetime
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_timestamp(folder_name):
    """Extract timestamp from folder name like 'ipip_mnrl_20250515_1303'."""
    match = re.search(r'(\d{8}_\d{4})$', folder_name)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
        except ValueError:
            return None
    return None

def get_model_type(folder_name):
    """Extract model type from folder name (e.g., 'ipip_mnrl' from 'ipip_mnrl_20250515_1303')."""
    match = re.match(r'^(.+)_\d{8}_\d{4}$', folder_name)
    if match:
        return match.group(1)
    return None

def cleanup_models(models_dir, keep=1, dry_run=False):
    """
    Clean up model directories, keeping only the most recent N models of each type.
    
    Args:
        models_dir: Directory containing model folders
        keep: Number of most recent models to keep for each type
        dry_run: If True, just print what would be deleted without actually deleting
    
    Returns:
        tuple: (number of deleted folders, bytes freed)
    """
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory '{models_dir}' does not exist.")
        return 0, 0
    
    # Group folders by model type
    model_folders = {}
    
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        
        # Skip non-directories and special directories
        if not os.path.isdir(item_path) or item in ['__pycache__', '.git']:
            continue
        
        # Try to parse as timestamped model directory
        timestamp = parse_timestamp(item)
        model_type = get_model_type(item)
        
        if timestamp and model_type:
            if model_type not in model_folders:
                model_folders[model_type] = []
            
            model_folders[model_type].append({
                'name': item,
                'path': item_path,
                'timestamp': timestamp,
                'size': sum(f.stat().st_size for f in Path(item_path).glob('**/*') if f.is_file())
            })
    
    # Sort each group by timestamp (newest first) and identify folders to delete
    folders_to_delete = []
    
    for model_type, folders in model_folders.items():
        # Sort by timestamp (newest first)
        sorted_folders = sorted(folders, key=lambda x: x['timestamp'], reverse=True)
        
        # Keep the N most recent, mark the rest for deletion
        if len(sorted_folders) > keep:
            folders_to_delete.extend(sorted_folders[keep:])
            
            logger.info(f"Model type '{model_type}':")
            logger.info(f"  Keeping:")
            for folder in sorted_folders[:keep]:
                logger.info(f"    - {folder['name']} ({format_size(folder['size'])})")
            
            logger.info(f"  Marked for deletion:")
            for folder in sorted_folders[keep:]:
                logger.info(f"    - {folder['name']} ({format_size(folder['size'])})")
        else:
            logger.info(f"Model type '{model_type}': {len(sorted_folders)} folders found, all within keep limit ({keep})")
    
    # Delete folders if not in dry run mode
    deleted_count = 0
    freed_bytes = 0
    
    if folders_to_delete:
        logger.info(f"\nTotal: {len(folders_to_delete)} folders marked for deletion")
        
        if dry_run:
            logger.info("DRY RUN: No files will be deleted")
            
            total_size = sum(folder['size'] for folder in folders_to_delete)
            logger.info(f"Would free up approximately {format_size(total_size)}")
            freed_bytes = total_size
        else:
            for folder in folders_to_delete:
                try:
                    logger.info(f"Deleting {folder['path']}...")
                    shutil.rmtree(folder['path'])
                    deleted_count += 1
                    freed_bytes += folder['size']
                except Exception as e:
                    logger.error(f"Error deleting {folder['path']}: {e}")
            
            logger.info(f"Successfully deleted {deleted_count} folders")
            logger.info(f"Freed up approximately {format_size(freed_bytes)}")
    else:
        logger.info("\nNo folders to delete")
    
    return deleted_count, freed_bytes

def format_size(size_bytes):
    """Format file size in a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def main():
    parser = argparse.ArgumentParser(description="Clean up old model directories, keeping only the most recent N models of each type.")
    parser.add_argument('--keep', type=int, default=1, help="Number of most recent models to keep for each type (default: 1)")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be deleted without actually deleting")
    parser.add_argument('--path', type=str, default="models", help="Directory containing model folders (default: models/)")
    
    args = parser.parse_args()
    
    logger.info(f"Scanning {args.path} directory for model folders...")
    logger.info(f"Will keep the {args.keep} most recent model(s) of each type")
    
    if args.dry_run:
        logger.info("Running in dry-run mode (no files will be deleted)")
    
    cleanup_models(args.path, args.keep, args.dry_run)

if __name__ == "__main__":
    main() 