from threading import Thread, Event, Lock
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import torch
from detection_module.model_update import ModelUpdater
from detection_module.detection import EnhancedPPOAgent, EnhancedDDoSEnvironment
from capapp.orchestration.pipeline import DDoSPipeline
from detection_module.predict_pipeline import LocalPredictionPipeline
from capapp.utils.logger import logger
from capapp.config.settings import config

config.setup_directories()

class PipelineController:
    def __init__(self):
        self.pipeline_active = Event()
        self.pipeline_threads = []
        self.pipeline: Optional[DDoSPipeline] = None
        self.detect: Optional[LocalPredictionPipeline] = None
        self.lock = Lock()
        
        # Detection tracking
        self.ddos_count = 0
        self.normal_count = 0
        self.suspicious_count = 0
        self.recent_detections: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.model_path="/home/bot/Desktop/drl_app-detect/detection_module/trained_models/final_drl1.pt"

        self.model_updater = ModelUpdater(
                    model_api_url="http://192.168.10.37:5000/api/pipeline/model/download",
                    current_model_path=self.model_path,
                    update_interval_hours=2
                )

        self.model_updater.start_periodic_update()
        
    def initialize_components(self) -> None:
        """Initialize pipeline components with thread-safe protection"""
        with self.lock:
            if self.pipeline is None or self.detect is None:
                self.pipeline = DDoSPipeline()


                
                
                
                self.detect = LocalPredictionPipeline(
                    model_path=self.model_path,
                    processed_dir="/home/bot/Desktop/drl_app-detect/capapp/features_output",
                    output_dir="./data/predictions",
                    force_cpu=True,
                    model_updater= self.model_updater
                )
                
                # Reset counters when initializing new components
                self._reset_counters()
    
    def _reset_counters(self) -> None:
        """Reset all detection counters"""
        self.ddos_count = 0
        self.normal_count = 0
        self.suspicious_count = 0
        self.recent_detections = []
        self.start_time = datetime.now()
    
    def start_all(self) -> bool:
        """Start all pipeline components"""
        if self.pipeline_active.is_set():
            logger.warning("Pipeline already running")
            return False
        
        try:
            self.initialize_components()
            
            # Initialize fresh threads
            threads = [
                Thread(target=self._run_pipeline, name="DDoSPipelineThread"),
                Thread(target=self._run_detection, name="DetectionThread")
            ]
            
            # Start threads
            for t in threads:
                t.daemon = True
                t.start()
            
            with self.lock:
                self.pipeline_threads = threads
                self.pipeline_active.set()
            
            logger.info("Pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.stop_all()
            return False
    
    def _run_pipeline(self) -> None:
        """Wrapper for pipeline execution"""
        try:
            if self.pipeline:
                self.pipeline.run()
        except Exception as e:
            logger.error(f"Pipeline thread failed: {e}")
    
    def _run_detection(self) -> None:
        """Wrapper for detection execution"""
        try:
            if self.detect:
                self.detect.start()
        except Exception as e:
            logger.error(f"Detection thread failed: {e}")
    
    def stop_all(self) -> bool:
        """Stop all pipeline components"""
        if not self.pipeline_active.is_set():
            logger.warning("Pipeline not running")
            return False
        
        try:
            # Stop components
            if self.pipeline:
                self.pipeline.stop()
            if self.detect:
                self.detect.stop()
            
            # Clear threads
            with self.lock:
                self.pipeline_threads = []
                self.pipeline_active.clear()
            
            logger.info("Pipeline stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if pipeline is running"""
        return self.pipeline_active.is_set()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "running": self.is_running(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "processed_files": self.detect.processed_files if self.detect else 0,
            "failed_files": self.detect.failed_files if self.detect else 0,
            "ddos_detections": self.ddos_count,
            "normal_detections": self.normal_count,
            "suspicious_detections": self.suspicious_count,
            "recent_detections_count": len(self.recent_detections),
            "model_loaded": self.detect is not None and self.detect.model is not None,
            "queue_size": self.detect.file_queue.qsize() if self.detect else 0,
            "device": str(self.detect.device) if self.detect else "unknown"
        }
        
        # Add pipeline-specific status if available
        if self.pipeline and hasattr(self.pipeline, 'get_status'):
            status.update({"pipeline": self.pipeline.get_status()})
        
        return status
    
    def get_recent_detections(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent detections for dashboard display"""
        with self.lock:
            return self.recent_detections[-limit:][::-1]  # Return most recent first
    
    def get_detection_details(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific detection"""
        with self.lock:
            for detection in self.recent_detections:
                if detection['id'] == detection_id:
                    return detection
            return None
    
    def record_detection(self, detection_data: Dict[str, Any]) -> None:
        """Record a new detection event"""
        
        
        detection = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'src_ip': detection_data.get('Src IP', 'unknown'),
            'dst_ip': detection_data.get('Dst IP', 'unknown'),
            'protocol': detection_data.get('Protocol', 'unknown'),
            'duration': detection_data.get('Flow Duration', 0),
            'status': 'DDoS' if detection_data.get('prediction', 0) == "DDoS" else 'Normal',
            'severity': 'critical' if detection_data.get('prediction', 0) == "DDoS" else 'normal',
            'confidence': detection_data.get('confidence', 0.0),
            'flow_id': detection_data.get('Flow ID', 'unknown'),
            'packets': detection_data.get('Total Fwd Packets', 0) + detection_data.get('Total Bwd Packets', 0),
            'bytes': detection_data.get('Total Length of Fwd Packets', 0) + detection_data.get('Total Length of Bwd Packets', 0)
        }

       
        
        with self.lock:
            # Update counters
          
            if detection['status'] == 'DDoS':
                self.ddos_count += 1
                
            else:
                self.normal_count += 1
            
            # Store recent detection
            self.recent_detections.append(detection)
            if len(self.recent_detections) > 100:  # Keep maximum 100 detections
                self.recent_detections.pop(0)

# Global controller instance
controller = PipelineController()

# Convenience functions for Flask routes
def start_pipeline() -> bool:
    return controller.start_all()

def stop_pipeline() -> bool:
    return controller.stop_all()

def pipeline_status() -> Dict[str, Any]:
    return controller.get_status()

def is_pipeline_running() -> bool:
    return controller.is_running()

def get_recent_detections(limit: int = 20) -> List[Dict[str, Any]]:
    return controller.get_recent_detections(limit)

def get_detection_details(detection_id: str) -> Optional[Dict[str, Any]]:
    return controller.get_detection_details(detection_id)