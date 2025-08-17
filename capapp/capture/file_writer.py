# capapp/capture/file_writer.py
import time
import threading
from queue import Queue, Empty
from datetime import datetime
from scapy.all import wrpcap
from utils.logger import logger
from config.settings import config

class PCAPFileWriter:
    """
    Writes packets from a queue to .pcap files, rotating them based on size or time.
    This version is optimized for high traffic by draining the queue in batches.
    """
    def __init__(self, packet_queue: Queue, file_queue: Queue):
        self.packet_queue = packet_queue
        self.file_queue = file_queue
        self.shutdown_event = threading.Event()
        self.writer_thread = None

    def _write_loop(self):
        """
        Main loop that consumes packets in batches and writes them to files.
        This is optimized to handle high-throughput scenarios.
        """
        packets_batch = []
        last_rotation_time = time.time()
        current_batch_size = 0

        while not self.shutdown_event.is_set():
            # --- FIX: Efficiently drain the queue in a non-blocking way ---
            try:
                # Pull all available packets from the queue up to a reasonable limit
                for _ in range(config.PACKET_QUEUE_MAXSIZE):
                    packet = self.packet_queue.get_nowait()
                    packets_batch.append(packet)
                    current_batch_size += len(packet)
                    self.packet_queue.task_done()
            except Empty:
                # This is expected when the queue is empty.
                pass

            time_since_rotation = time.time() - last_rotation_time
            size_exceeded = current_batch_size >= config.MAX_PCAP_SIZE
            time_exceeded = time_since_rotation >= config.CAPTURE_INTERVAL

            # Rotate file if conditions are met AND there's something to write
            if (size_exceeded or time_exceeded) and packets_batch:
                self._write_and_enqueue(packets_batch)
                # Reset for the next batch
                packets_batch = []
                current_batch_size = 0
                last_rotation_time = time.time()
            
            # If no packets were processed, sleep briefly to prevent a busy-wait loop
            if not packets_batch:
                time.sleep(0.05) # 50ms sleep

        # Final write on shutdown for any remaining packets
        if packets_batch:
            logger.info(f"Performing final write of {len(packets_batch)} packets before shutdown.")
            self._write_and_enqueue(packets_batch)

        logger.info("File writer loop has stopped.")

    def _write_and_enqueue(self, packets: list):
        """
        Writes a batch of packets to a new .pcap file and enqueues the file path.
        """
        timestamp = datetime.now()
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = config.PCAP_DIR / f"B_{timestamp}.pcap"
        
        try:
            logger.info(f"Writing {len(packets)} packets to {file_path.name}...")
            wrpcap(str(file_path), packets)
            
            if not self.file_queue.full():
                self.file_queue.put(file_path)
                logger.info(f"📦 File rotated and enqueued: {file_path.name}")
            else:
                logger.error(f"File processing queue is full. Discarding {file_path.name}.")
                file_path.unlink() # Clean up the file that won't be processed
        except Exception as e:
            logger.error(f"Failed to write .pcap file {file_path.name}: {e}", exc_info=True)

    def start(self):
        """Starts the file writer thread."""
        if self.writer_thread and self.writer_thread.is_alive():
            logger.warning("File writer thread is already running.")
            return

        self.shutdown_event.clear()
        self.writer_thread = threading.Thread(target=self._write_loop, name="FileWriterThread")
        self.writer_thread.daemon = True
        self.writer_thread.start()
        logger.info("PCAP file writer started.")

    def stop(self):
        """Stops the file writer thread gracefully."""
        logger.info("Stopping file writer...")
        self.shutdown_event.set()
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5)
        logger.info("File writer stopped.")
