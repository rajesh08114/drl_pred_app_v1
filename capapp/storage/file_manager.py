# capapp/storage/file_manager.py
import shutil
from pathlib import Path
from capapp.utils.logger import logger
from capapp.config.settings import config

class FileManager:
    """
    Provides static methods for managing the lifecycle of .pcap files
    in a thread-safe manner by moving them between directories.
    """
    @staticmethod
    def move_to_in_progress(pcap_path: Path) -> Path:
        """Atomically moves a file to the 'in_progress' directory."""
        try:
            destination = config.IN_PROGRESS_DIR / pcap_path.name
            shutil.move(str(pcap_path), str(destination))
            return destination
        except FileNotFoundError:
            logger.warning(f"File {pcap_path.name} was moved or deleted before processing could start.")
            return None
        except Exception as e:
            logger.error(f"Failed to move {pcap_path.name} to in_progress: {e}")
            return None

    @staticmethod
    def move_to_processed(pcap_path: Path):
        """Deletes a file from 'in_progress' after it's successfully processed."""
        try:
            pcap_path.unlink()
            logger.info(f"🧹 Deleted successfully processed file: {pcap_path.name}")
        except Exception as e:
            logger.error(f"Failed to delete processed file {pcap_path.name}: {e}")

    @staticmethod
    def move_to_error(pcap_path: Path):
        """Moves a file that failed processing to the 'error' directory."""
        try:
            destination = config.ERROR_DIR / pcap_path.name
            shutil.move(str(pcap_path), str(destination))
            logger.warning(f"Moved failed file to error directory: {pcap_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {pcap_path.name} to error directory: {e}")
