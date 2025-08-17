# capapp/orchestration/pipeline.py

import time
from capapp.utils.logger import logger
from capapp.capture.packet_capture import PacketCapturer
from capapp.processing.dispatcher import FileDispatcher

class DDoSPipeline:
    """
    The master orchestrator for the disk-based DDoS detection pipeline.
    Initializes, starts, and stops all components in the correct order.
    """
    def __init__(self):
        logger.info("Initializing pipeline components...")
        self.capturer = PacketCapturer()
        self.dispatcher = FileDispatcher()
        self.components = [self.capturer, self.dispatcher]

    def start(self):
        """Starts all pipeline components."""
        logger.info("🚀 Starting all pipeline components...")
        for component in self.components:
            component.start()
        logger.info("✅ All pipeline components are running.")

    def stop(self):
        """Stops all pipeline components gracefully in reverse order."""
        logger.info("🛑 Stopping all pipeline components...")
        for component in reversed(self.components):
            try:
                component.stop()
            except Exception as e:
                logger.error(f"Error stopping component {component.__class__.__name__}: {e}")
        logger.info("✅ Pipeline shutdown complete.")

    def run(self):
        """Runs the pipeline indefinitely until a stop signal is received."""
        self.start()
        try:
            # Keep the main thread alive to allow daemon threads to run
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Initiating shutdown...")
        finally:
            self.stop()
