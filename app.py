from flask import Flask, render_template, request, redirect, jsonify
import os
import sys
import socket
from datetime import datetime
from typing import Dict, Any, Optional
from core.controller import start_pipeline, stop_pipeline,pipeline_status,is_pipeline_running,get_recent_detections,get_detection_details,controller
from detection_module.detection import EnhancedPPOAgent, EnhancedDDoSEnvironment
from capapp.config.settings import config

app = Flask(__name__)

# Configuration
DEFAULT_DETECTION_LIMIT = 20
MAX_DETECTION_LIMIT = 100

def check_privileges() -> bool:
    """Check if the application has required network privileges"""
    try:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
        s.close()
        return True
    except PermissionError:
        return False

@app.route('/')
def index() -> str:
    """Render the main dashboard page"""
    return render_template('index.html',
                         has_privileges=check_privileges(),
                         is_running=is_pipeline_running(),
                         current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/start', methods=['POST'])
def start() -> Any:
    """Start the DDoS detection pipeline"""
    if not check_privileges():
        return jsonify({"status": "error", "message": "Insufficient privileges"}), 403
    
    if is_pipeline_running():
        return jsonify({"status": "error", "message": "Pipeline already running"}), 400
    
    if start_pipeline():
        return redirect('/')
    return jsonify({"status": "error", "message": "Failed to start pipeline"}), 500

@app.route('/stop', methods=['POST'])
def stop() -> Any:
    """Stop the DDoS detection pipeline"""
    if not is_pipeline_running():
        return jsonify({"status": "error", "message": "Pipeline not running"}), 400
    
    if stop_pipeline():
        return redirect('/')
    return jsonify({"status": "error", "message": "Failed to stop pipeline"}), 500

@app.route('/api/status')
def api_status() -> Dict[str, Any]:
    """Return comprehensive system status"""
    status = pipeline_status()
    return jsonify(status)

@app.route('/api/detections')
def recent_detections() -> Dict[str, Any]:
    """Return recent detection results with pagination"""
    limit = min(int(request.args.get('limit', DEFAULT_DETECTION_LIMIT)), MAX_DETECTION_LIMIT)
    offset = int(request.args.get('offset', 0))
    
    detections = get_recent_detections(limit=limit)
    total_detections = len(detections)
    
    return jsonify({
        'detections': detections[offset:offset+limit],
        'metadata': {
            'total': total_detections,
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total_detections
        }
    })

@app.route('/api/detections/<detection_id>')
def detection_details(detection_id: str) -> Any:
    """Return detailed information about a specific detection"""
    details = get_detection_details(detection_id)
    if details:
        return jsonify(details)
    return jsonify({"error": "Detection not found"}), 404

@app.route('/api/stats')
def detection_stats() -> Dict[str, Any]:
    """Return detection statistics"""
    status = pipeline_status()
    return jsonify({
        'ddos_count': status.get('ddos_count', 0),
        'normal_count': status.get('normal_count', 0),
        'suspicious_count': status.get('suspicious_count', 0),
        'throughput': {
            'files_processed': status.get('processed_files', 0),
            'files_failed': status.get('failed_files', 0),
            'processing_rate': calculate_processing_rate(status)
        }
    })

def calculate_processing_rate(status: Dict[str, Any]) -> float:
    """Calculate files processed per minute"""
    if 'start_time' not in status or not status['start_time']:
        return 0.0
    
    start_time = datetime.fromisoformat(status['start_time'])
    uptime_minutes = (datetime.now() - start_time).total_seconds() / 60
    if uptime_minutes <= 0:
        return 0.0
    
    return round(status.get('processed_files', 0) / uptime_minutes, 2)


@app.route('/api/model_status')
def model_status():
    """Return current model update status"""
    return jsonify({
        "status": "success",
        "data": {
            "last_update": controller.model_updater.last_update.isoformat() 
                          if controller.model_updater.last_update else None,
            
        }
    })

@app.route('/api/update_model', methods=['POST'])
def trigger_model_update():
    """Manually trigger model update"""
    try:
        success = controller.model_updater.update_model()
        if success:
            return jsonify({
                "status": "success", 
                "message": "Model updated successfully",
                "timestamp": datetime.now().isoformat()
            })
        return jsonify({
            "status": "error",
            "message": "Model update failed"
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
if __name__ == '__main__':
    if not check_privileges():
        print("\nWARNING: Missing packet capture privileges. Try:")
        print("  sudo setcap cap_net_raw,cap_net_admin+eip {}".format(sys.executable))
        print("Or run with: sudo {} {}\n".format(sys.executable, __file__))
    
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
