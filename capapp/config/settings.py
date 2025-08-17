# capapp/config/settings.py
import os
from pathlib import Path

class Config:
    """
    Centralized configuration for the disk-based DDoS detection pipeline.
    """
    # 1. DIRECTORY STRUCTURE
    BASE_DIR = Path(__file__).parent.parent
    CAPTURE_DIR = BASE_DIR / "capture_output"
    IN_PROGRESS_DIR = CAPTURE_DIR / "in_progress"
    FEATURES_DIR = BASE_DIR / "features_output"
    ERROR_DIR = CAPTURE_DIR / "error"
    LOG_DIR = BASE_DIR / "logs"

    # 2. CAPTURE & ROTATION SETTINGS
    CAPTURE_INTERFACE = os.getenv("CAPTURE_INTERFACE", "enp0s3")
    CAPTURE_FILTER = os.getenv("CAPTURE_FILTER", "")
    ROTATE_INTERVAL_SECONDS = 30
    ROTATE_MAX_SIZE_MB = 50

    # 3. DISPATCHER & PROCESSING SETTINGS
    DISPATCHER_POLL_INTERVAL_SECONDS = 5
    MAX_PROCESSING_WORKERS = (os.cpu_count() or 1)
    # --- FIX: Added a dedicated, longer timeout for feature extraction ---
    PROCESSING_TIMEOUT_SECONDS = 300  # 5 minutes


    FLOW_TIMEOUT_NS = 1_200_000_000  # 1.2 seconds (inactivity timeout)
    MAX_FLOW_DURATION_NS = 120_000_000_000  # 120 seconds (max flow duration)
    ACTIVE_THRESHOLD_US = 1_000_000


    @classmethod
    def setup_directories(cls):
        """Creates all necessary directories for the pipeline to operate."""
        print("Setting up required directories...")
        for directory in [
            cls.CAPTURE_DIR, cls.IN_PROGRESS_DIR, cls.FEATURES_DIR,
            cls.ERROR_DIR, cls.LOG_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        print("Directories are ready.")

config = Config()
