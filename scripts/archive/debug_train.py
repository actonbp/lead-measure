#!/usr/bin/env python3
"""
Script to debug loading the bge-m3 model.
"""
import sys
import os
import logging
from sentence_transformers import SentenceTransformer, models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    model_path = "models/BAAI_bge-m3"
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model path exists: {os.path.exists(model_path)}")
    
    # List model files
    logger.info("Model directory contents:")
    for file in os.listdir(model_path):
        logger.info(f"  - {file}")
    
    # Try loading the model
    logger.info("Attempting to load model...")
    try:
        # First try loading as a sentence transformer directly
        logger.info("Method 1: Loading as SentenceTransformer")
        model = SentenceTransformer(model_path)
        logger.info(f"Successfully loaded model: {model}")
        
        # Test encoding
        test_embedding = model.encode("This is a test sentence")
        logger.info(f"Test embedding shape: {test_embedding.shape}")
        logger.info("Model successfully loaded!")
        return True
    except Exception as e:
        logger.error(f"Error loading model directly: {str(e)}")
        
        # Try alternate method: loading via Transformer + Pooling
        try:
            logger.info("Method 2: Loading as Transformer + Pooling")
            word_embedding_model = models.Transformer(model_path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode="cls")
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            logger.info(f"Successfully loaded model: {model}")
            
            # Test encoding
            test_embedding = model.encode("This is a test sentence")
            logger.info(f"Test embedding shape: {test_embedding.shape}")
            logger.info("Model successfully loaded!")
            return True
        except Exception as e:
            logger.error(f"Error loading model with Transformer + Pooling: {str(e)}")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)