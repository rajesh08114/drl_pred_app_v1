# capapp/capture/packet_capture.py
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path
from scapy.all import sniff, wrpcap, Scapy_Exception, get_if_list

from capapp.config.settings import config
from capapp.utils.logger import logger

class PacketCapturer:
    """
    Captures network traffic directly to disk and rotates .pcap files.
    This component is fully independent and does not use in-memory queues.
    """
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.capture_thread = None
        self.packets = []
        self.current_file_path: Path = None
        self.last_rotation_time = 0
        self.lock = threading.Lock()
        self.interface = self._validate_interface()

    def _validate_interface(self) -> str:
        """
        Validates the configured interface or auto-detects a suitable one.
        """
        available_interfaces = get_if_list()
        configured_iface = config.CAPTURE_INTERFACE

        if configured_iface in available_interfaces:
            logger.info(f"Successfully validated configured interface: {configured_iface}")
            return configured_iface

        logger.warning(f"Configured interface '{configured_iface}' not found.")
        
        # Filter out loopback interfaces
        non_loopback = [iface for iface in available_interfaces if "lo" not in iface]
        if non_loopback:
            auto_selected_iface = non_loopback[0]
            logger.warning(f"Automatically selected the first available non-loopback interface: {auto_selected_iface}")
            return auto_selected_iface

        logger.error(f"No suitable non-loopback network interfaces found. Available: {available_interfaces}")
        raise SystemExit("Fatal: Could not find a network interface to capture on.")


    def _get_new_filepath(self) -> Path:
        """Generates a unique, timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # unique_id = str(uuid.uuid4()).split('-')[0] # Short UUID for uniqueness
        # return config.CAPTURE_DIR / f"capture_{timestamp}_{unique_id}.pcap"
        
        return config.CAPTURE_DIR / f"B_{timestamp}.pcap"

    def _packet_handler(self, packet):
        """Callback for Scapy's sniff. Appends packet to the in-memory list."""
        with self.lock:
            self.packets.append(packet)

    def _rotation_manager(self):
        """
        Periodically checks if the current capture file needs to be rotated
        based on time or size, and writes the batch to disk.
        """
        self.current_file_path = self._get_new_filepath()
        self.last_rotation_time = time.time()
        logger.info(f"Starting new capture file: {self.current_file_path.name}")

        while not self.shutdown_event.is_set():
            time.sleep(1) # Check for rotation every second

            with self.lock:
                current_size_bytes = sum(len(p) for p in self.packets)
                time_elapsed = time.time() - self.last_rotation_time

                size_exceeded = current_size_bytes >= (config.ROTATE_MAX_SIZE_MB * 1024 * 1024)
                time_exceeded = time_elapsed >= config.ROTATE_INTERVAL_SECONDS

                if (size_exceeded or time_exceeded) and self.packets:
                    # Perform rotation
                    packets_to_write = self.packets
                    self.packets = []
                    
                    old_file_path = self.current_file_path
                    self.current_file_path = self._get_new_filepath()
                    self.last_rotation_time = time.time()

                    # Write the captured batch in the background to not block the manager
                    threading.Thread(
                        target=self._write_file,
                        args=(packets_to_write, old_file_path)
                    ).start()
                    
                    logger.info(f"Starting new capture file: {self.current_file_path.name}")

    def _write_file(self, packets: list, filepath: Path):
        """Writes a list of packets to a .pcap file."""
        try:
            wrpcap(str(filepath), packets)
            logger.info(f"📦 Rotated and saved: {filepath.name} ({len(packets)} packets)")
        except Exception as e:
            logger.error(f"Failed to write .pcap file {filepath.name}: {e}")

    def start(self):
        """Starts the packet capture and rotation manager threads."""
        if not self.interface:
            logger.critical("Cannot start capture: No valid network interface was found.")
            return

        if self.capture_thread and self.capture_thread.is_alive():
            logger.warning("Capture is already running.")
            return

        logger.info(f"Starting packet capture on interface '{self.interface}'...")
        self.shutdown_event.clear()

        # Start the rotation manager
        manager_thread = threading.Thread(target=self._rotation_manager, name="RotationManager")
        manager_thread.daemon = True
        manager_thread.start()

        # Start the sniffer
        self.capture_thread = threading.Thread(
            target=lambda: sniff(
                iface=self.interface,
                prn=self._packet_handler,
                filter=config.CAPTURE_FILTER,
                stop_filter=lambda p: self.shutdown_event.is_set()
            ),
            name="PacketSniffer"
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop(self):
        """Stops the capture process gracefully."""
        logger.info("Stopping packet capture...")
        self.shutdown_event.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)

        # Final write for any remaining packets
        with self.lock:
            if self.packets:
                logger.info(f"Performing final write of {len(self.packets)} packets.")
                self._write_file(self.packets, self.current_file_path)
        
        logger.info("Packet capture stopped.")
