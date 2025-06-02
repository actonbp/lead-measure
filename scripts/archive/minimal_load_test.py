#!/usr/bin/env python3
"""
Minimal script to test if we can load the model with minimal dependencies.
"""
import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting minimal test")
        
        # Try importing PyTorch first
        logger.info("Importing PyTorch...")
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Try tokenizers directly
        logger.info("Importing tokenizers...")
        import tokenizers
        logger.info(f"Tokenizers version: {tokenizers.__version__}")
        
        # Try loading transformers directly
        logger.info("Importing transformers...")
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Load model using transformers directly
        logger.info("Loading tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Tokenizer loaded successfully")
        
        logger.info("Loading model...")
        model = transformers.AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Model loaded successfully")
        
        # Try a simple encoding
        logger.info("Encoding text...")
        inputs = tokenizer("This is a test sentence", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token embedding
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        logger.info("All tests passed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)