#!/usr/bin/env python3
# capapp/main.py
import sys
from pathlib import Path

# Add the project root to the Python path for correct module resolution
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import config
from orchestration.pipeline import DDoSPipeline
from utils.logger import logger

def main():
    """The main entry point for the DDoS Detection Pipeline application."""
    # Setup directories first to prevent logging or file errors on startup
    config.setup_directories()
    
    logger.info("=========================================")
    logger.info("  Initializing Disk-Based DDoS Pipeline")
    logger.info("=========================================")
    
    pipeline = DDoSPipeline()
    
    try:
        pipeline.run()
    except Exception as e:
        logger.critical(f"💥 A fatal, unhandled error occurred in main: {e}", exc_info=True)
        # Attempt a graceful shutdown even on fatal error
        pipeline.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
