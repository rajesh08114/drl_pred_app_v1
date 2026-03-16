import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import os
import uuid
import logging
import queue
import threading
from datetime import datetime
import requests
import time
from detection_module.model_update import ModelUpdater
from typing import Optional, Tuple, Dict, Any,List
from detection_module.detection import EnhancedPPOAgent,EnhancedDDoSEnvironment





# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LocalPredictionPipeline")

class LocalPredictionPipeline:
    def __init__(self, 
                 model_path: str,
                 processed_dir: str,
                 flask_app_url: str = "http://192.168.0.208:8000",
                 output_dir: str = "./predictions",
                 queue_maxsize: int = 10,
                 force_cpu: bool = False,model_updater : ModelUpdater = None):
        """
        Initialize the local prediction pipeline.
        
        Args:
            model_path: Path to the trained model (.pkl file)
            processed_dir: Directory to watch for new CSV files
            flask_app_url: URL of the local Flask application
            output_dir: Directory to save prediction results
            queue_maxsize: Maximum size of the processing queue
            force_cpu: Whether to force CPU usage even if GPU is available

        """


        self.model_updater = model_updater


        self.ddos_count = 0
        self.normal_count = 0
        self.suspicious_count = 0
        self.recent_detections = []

        self.device = self._select_device(force_cpu)
        
        self.model = self._load_model(model_path)
        self.model
        
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.flask_app_url = flask_app_url
        self.file_queue = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        
        # Statistics
        self.processed_files = 0
        self.failed_files = 0
        
        self._setup_directories()
        self._processed_files = set()  # Track processed files
        self._processing_lock = threading.Lock()
        

    
    def _select_device(self, force_cpu: bool) -> torch.device:
        """Select the appropriate device for computation"""
        if force_cpu:
            logger.info("Forcing CPU usage as requested")
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            logger.info("CUDA GPU available, using GPU acceleration")
            return torch.device("cuda")
        
        logger.info("No GPU available, falling back to CPU")
        return torch.device("cpu")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Safely load the model with GPU/CPU compatibility"""
        try:
            # Try loading with default method first
            from detection_module.detection import EnhancedPPOAgent, ImprovedPPONetwork
            with open(model_path, 'rb') as f:
                # model=EnhancedPPOAgent.load_model(model_path,map_location="cpu")
                
                # if hasattr(model, 'policy'):
                #     # logger.info(model.policy.eval())
                #     # model = model.policy.eval()
                #     # logger.info(model)
                #     logger.info("Model loaded successfully to %s", self.device)
                #     return model

               
                with self.model_updater.lock:
                    if self.model_updater.model is not None:
                        return self.model_updater.model
                    model=EnhancedPPOAgent.load_model(model_path,map_location="cpu")
                
                    if hasattr(model, 'policy'):
                        logger.info("Model loaded successfully to %s", self.device)
                        return model
                    
            
                    else:
                        raise ValueError("Loaded agent has no policy attribute")
            
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Could not load model: {str(e)}")

    def _load_model_cpu(self, model_path: str) -> torch.nn.Module:
        """Load model with forced CPU mapping"""
        try:
            # Special handling for GPU-trained models on CPU systems
            with open(model_path, 'rb') as f:
                if torch.cuda.is_available():
                    model = torch.load(f, map_location=lambda storage, loc: storage.cpu())
                else:
                    model = pickle.load(f)
                
                if isinstance(model, torch.nn.Module):
                    model = model.to('cpu')
                    logger.info("Model successfully loaded on CPU")
                    return model
                
                raise ValueError("Loaded object is not a PyTorch model")
        except Exception as e:
            logger.error(f"CPU model load failed: {str(e)}")
            raise RuntimeError("Failed to load model on CPU")

    def _setup_directories(self):
        """Ensure required directories exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir = self.processed_dir / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _get_oldest_file(self) -> Optional[Path]:
        """Get the oldest file based on timestamp in filename"""
        files = []
        for f in self.processed_dir.glob("B_*_features.csv"):
            try:
                # Extract timestamp from filename (B_YYYYMMDD_HHMMSS_features.csv)
                ts_str = "_".join(f.name.split('_')[1:3])
                timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                files.append((timestamp, f))
            except Exception as e:
                logger.warning(f"Skipping file with invalid timestamp {f.name}: {str(e)}")
                continue
        
        return min(files, key=lambda x: x[0])[1] if files else None

    def _send_to_flask_app(self, data: Dict[str, Any], endpoint: str) -> bool:
        """Send data to local Flask application with retry logic"""
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.flask_app_url}/{endpoint}",
                    json=data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logging.info(f"send data succes:{response.status_code}")
                    return True
                
                logger.warning(f"Flask app returned status {response.status_code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
            except Exception as e:
                logger.warning(f"Failed to connect to Flask app (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        return False

    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, pd.DataFrame]:
        """Prepare data for model prediction"""
        # Keep original for output
        output_df = df.copy()
        df = df.drop(columns=['Flow ID','Timestamp','Fwd Header Length.1'])
        # or ['Start Time', 'End Time']

        import ipaddress

        # Convert IPv4 address to int
        df['Src IP'] = df['Src IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Replace NaNs with column mean or 0
        df.fillna(df.mean(), inplace=True)
        X_train = scaler.fit_transform(df)
        
        return X_train, output_df

    def _postprocess_results(self, predictions: Dict[str, np.ndarray], original_df: pd.DataFrame) -> pd.DataFrame:
        """Combine predictions with original data"""
        result_df = original_df.copy()
        result_df['prediction'] = predictions['labels']
        # result_df['confidence'] = predictions['confidences']
        # result_df['anomaly_score'] = predictions['anomaly_scores']
        # result_df['processing_time'] = datetime.utcnow().isoformat()
        return result_df

    def _save_predictions(self, df: pd.DataFrame, input_filename: str):
        """Save prediction results to local CSV"""
        output_file = self.output_dir / f"pred_{input_filename}"
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")

    def _quarantine_file(self, file_path: Path):
        """Move problematic files to quarantine"""
        quarantine_dir = self.processed_dir / "quarantine"
        new_path = quarantine_dir / file_path.name
        
        try:
            os.rename(file_path, new_path)
            logger.warning(f"Moved {file_path.name} to quarantine")
            self.failed_files += 1
        except Exception as e:
            logger.error(f"Failed to quarantine {file_path.name}: {str(e)}")

    def _process_file(self, file_path: Path):
        """Process a single file through the pipeline"""
        with self._processing_lock:
            if file_path.name in self._processed_files:
                logger.debug(f"Skipping already processed file: {file_path.name}")
                return
            self._processed_files.add(file_path.name)

        try:
            logger.info(f"Processing file: {file_path.name}")
            
            # cheack is path exists
            if not file_path.exists():
                logger.error(f"File {file_path.name} does not exist")
                return
                
            
            # 1. Read and validate file
            df = pd.read_csv(file_path)
            
            if df.empty:
                raise ValueError("Empty DataFrame")
            
            # 2. Send raw data to Flask app
            raw_data = df.to_dict(orient='records')
            if not self._send_to_flask_app({
                "filename": file_path.name,
                "data": raw_data,
                "timestamp": datetime.now().isoformat()
            }, "raw_data"):
                # raise RuntimeError("Failed to send raw data to Flask app after retries")
                
                print()
            # 3. Perform prediction
            
            input_tensor, original_df = self._preprocess_data(df)
            
            
            try:
                with torch.no_grad():
                    
                    logger.info(self.model)
                    predictions = self.model.predict_batch(input_tensor)
                    
                    
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning("CUDA error during prediction, retrying on CPU...")
                    input_tensor = input_tensor.to('cpu')
                    self.model = self.model.to('cpu')
                    with torch.no_grad():
                        predictions = self.model.predict_batch(input_tensor)
                else:
                    raise
            
            # 4. Save predictions locally
            result_df = self._postprocess_results(predictions, original_df)
            
            self._save_predictions(result_df, file_path.name)

            #-------------------------------

            for _, row in result_df.iterrows():
                from core.controller import controller
                controller.record_detection(row.to_dict())
            
           
          

            #-------------------------------
            
            
            # 5. Send predictions to Flask app
            if not self._send_to_flask_app({
                "filename": file_path.name,
                "predictions": result_df.to_dict(orient='records'),
                "metadata": {
                    "model_version": "1.0",
                    "processing_time": datetime.now().isoformat(),
                    "device": str(self.device)
                }
            }, "/api/data"):
                logger.warning("Failed to send predictions to Flask app (saved locally)")
            
            # 6. Cleanup
            # os.remove(file_path)
            # self.processed_files += 1
            logger.info(f"Successfully processed {file_path.name}")
            try:
                os.remove(file_path)
                logger.info(f"Successfully deleted {file_path}")
                self.processed_files += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path.name}: {str(e)}")
                self._quarantine_file(file_path)


        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            self._quarantine_file(file_path)
    # def get_recent_detections(self, limit: int = 20) -> List[Dict[str, Any]]:
    #     """
    #     Get recent detections for the dashboard
    #     Args:
    #         limit: Maximum number of detections to return
    #     Returns:
    #         List of detection records, most recent first
    #     """
    #     with self._detection_lock:
    #         return self.recent_detections[-limit:][::-1]

    # def get_detection_details(self, detection_id: str) -> Optional[Dict[str, Any]]:
    #     """
    #     Get details for a specific detection
    #     Args:
    #         detection_id: UUID of the detection
    #     Returns:
    #         Detection details or None if not found
    #     """
    #     with self._detection_lock:
    #         for detection in self.recent_detections:
    #             if detection['id'] == detection_id:
    #                 return detection
    #         return None

    # def get_status(self) -> Dict[str, Any]:
    #     """Get current pipeline status with detection statistics"""
    #     base_status = super().get_status()
    #     base_status.update({
    #         "ddos_count": self.ddos_count,
    #         "normal_count": self.normal_count,
    #         "suspicious_count": self.suspicious_count,
    #         "recent_detections_count": len(self.recent_detections),
    #         "detection_stats": {
    #             "ddos": self.ddos_count,
    #             "normal": self.normal_count,
    #             "suspicious": self.suspicious_count
    #         }
    #     })
    #     return base_status


    def _file_discovery_worker(self):
        """Continuously find and queue files for processing"""
        # logger.info("File discovery worker started")
        
        # while not self._stop_event.is_set():
        #     try:
        #         if not self.file_queue.full():
        #             file_path = self._get_oldest_file()
        #             if file_path:
        #                 self.file_queue.put(file_path)
        #                 logger.debug(f"Queued file: {file_path.name}")
                
        #         time.sleep(1)
                
        #     except Exception as e:
        #         logger.error(f"File discovery error: {str(e)}")
        #         time.sleep(5)
        logger.info("File discovery worker started")
    
        while not self._stop_event.is_set():
            try:
                if not self.file_queue.full():
                    file_path = self._get_oldest_file()
                    if file_path:
                        with self._processing_lock:
                            if file_path.name not in self._processed_files:
                                self.file_queue.put(file_path)
                                logger.debug(f"Queued file: {file_path.name}")
                
                time.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"File discovery error: {str(e)}")
                time.sleep(5)

    def _processing_worker(self):
        """Process files from the queue"""
        logger.info("Processing worker started")
        
        while not self._stop_event.is_set() or not self.file_queue.empty():
            try:
                file_path = self.file_queue.get(timeout=1)
                self._process_file(file_path)
                self.file_queue.task_done()
                
            except queue.Empty:
                continue
                
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                time.sleep(1)

    def start(self):
        """Start the pipeline workers"""
        if hasattr(self, '_discovery_thread') and self._discovery_thread.is_alive():
            logger.warning("Pipeline is already running")
            return
        
        self._stop_event.clear()
        
        self._discovery_thread = threading.Thread(
            target=self._file_discovery_worker,
            name="FileDiscoveryThread",
            daemon=True
        )
        
        self._processing_thread = threading.Thread(
            target=self._processing_worker,
            name="ProcessingWorkerThread",
            daemon=True
        )
        
        self._discovery_thread.start()
        self._processing_thread.start()
        
        logger.info("Pipeline started successfully")

    def stop(self):
        """Stop the pipeline gracefully"""
        self._stop_event.set()
        self.model_updater.stop_periodic_update()
        
        if hasattr(self, '_discovery_thread'):
            self._discovery_thread.join(timeout=5)
            
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=5)
        
        logger.info(f"Pipeline stopped. Processed {self.processed_files} files, {self.failed_files} failed")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "running": not self._stop_event.is_set(),
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "queue_size": self.file_queue.qsize(),
            "device": str(self.device),
            "model_loaded": self.model is not None
        }

