#!/usr/bin/env python3
"""
Script to download BAAI/bge-m3 model using huggingface_hub explicitly.
This provides a more reliable download than relying on the automatic download.
"""
import os
import logging
import argparse
from huggingface_hub import snapshot_download, hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_name, output_dir=None, local_files_only=False, force_download=False):
    """Download model explicitly using huggingface_hub."""
    try:
        logger.info(f"Starting download of {model_name}")
        
        if output_dir is None:
            # Default to a cache directory within models/
            output_dir = os.path.join("models", model_name.replace("/", "_"))
        
        os.makedirs(output_dir, exist_ok=True)
        
        # First try downloading just the config to see if we can access the model
        logger.info(f"Testing access by downloading config.json first...")
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            local_files_only=local_files_only,
            force_download=force_download
        )
        logger.info(f"Successfully downloaded config to {config_path}")
        
        # Now download the entire model
        logger.info(f"Downloading full model to {output_dir}")
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_files_only=local_files_only,
            force_download=force_download
        )
        
        logger.info(f"Model successfully downloaded to: {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model from Hugging Face Hub")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", 
                      help="Model name/path on Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory (defaults to models/<model_name>)")
    parser.add_argument("--local_only", action="store_true",
                      help="Use only local files (no download)")
    parser.add_argument("--force", action="store_true",
                      help="Force re-download even if files exist")
    
    args = parser.parse_args()
    
    download_model(
        args.model, 
        output_dir=args.output_dir,
        local_files_only=args.local_only,
        force_download=args.force
    )