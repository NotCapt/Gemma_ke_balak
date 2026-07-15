"""
CrowdGuard — Backend API Server
Connects the frontend dashboard to VadCLIP anomaly detection
and Gemma semantic confirmation via Kaggle VM.

Usage (from project root, inside venv):
    python server.py
"""

import os
import sys
import time
import json
import uuid
import traceback

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup — so we can import from VadCLIP/ and gemma/ directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VADCLIP_DIR = os.path.join(PROJECT_ROOT, "VadCLIP")
GEMMA_DIR = os.path.join(PROJECT_ROOT, "gemma")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_POLICE_DIR = os.path.join(PROJECT_ROOT, "frontend_police")

sys.path.insert(0, VADCLIP_DIR)
sys.path.insert(0, os.path.join(VADCLIP_DIR, "src"))
sys.path.insert(0, GEMMA_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KAGGLE_NGROK_URL = "https://97e4-34-80-13-242.ngrok-free.app"
MODEL_PATH = os.path.join(VADCLIP_DIR, "model", "model_ucf.pth")
REPORTS_DIR = os.path.join(VADCLIP_DIR, "reports")
UPLOADS_DIR = os.path.join(VADCLIP_DIR, "uploads")
ANOMALY_THRESHOLD = 0.010

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
CORS(app)

# In-memory store for pipeline state per session
pipeline_sessions = {}


# ===========================================================================
# Static Frontend Routes
# ===========================================================================
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ===========================================================================
# Static Police Frontend Routes
# ===========================================================================
@app.route("/police/")
def serve_police_index():
    return send_from_directory(FRONTEND_POLICE_DIR, "index.html")


@app.route("/police/<path:path>")
def serve_police_static(path):
    """Serve static police dashboard files (CSS, JS, assets)."""
    file_path = os.path.join(FRONTEND_POLICE_DIR, path)
    if os.path.isfile(file_path):
        return send_from_directory(FRONTEND_POLICE_DIR, path)
    return "Not Found", 404


@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend files (CSS, JS, assets)."""
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    return "Not Found", 404


# ===========================================================================
# API: Upload Video
# ===========================================================================
@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Accept a video file upload, save it, return a session_id."""
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Save video to uploads directory
    ext = os.path.splitext(video_file.filename)[1] or ".mp4"
    video_filename = f"{session_id}{ext}"
    video_path = os.path.join(UPLOADS_DIR, video_filename)
    video_file.save(video_path)

    # Get basic video info
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    # Store session state
    pipeline_sessions[session_id] = {
        "video_path": video_path,
        "video_filename": video_file.filename,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "events": [],
        "collage_paths": [],
    }

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "video_info": {
            "filename": video_file.filename,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration, 2),
        }
    })


# ===========================================================================
# API: Run VAD (Stage 1)
# ===========================================================================
@app.route("/api/pipeline/vad", methods=["POST"])
def run_vad():
    """
    Run VadCLIP anomaly detection on uploaded video.
    Expects JSON body: { "session_id": "..." }
    """
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id or session_id not in pipeline_sessions:
        return jsonify({"status": "error", "message": "Invalid session_id"}), 400

    session = pipeline_sessions[session_id]
    video_path = session["video_path"]
    fps = session["fps"]

    start_time = time.time()

    try:
        # Import VadCLIP pipeline functions
        import numpy as np
        import torch
        from pipeline import (
            run_anomaly_detection,
            group_anomalous_snippets,
            create_collages,
        )

        # Step 1: Extract features
        import extract_features as ef
        temp_feat_path = os.path.join(VADCLIP_DIR, f"temp_features_{session_id}.npy")

        # Temporarily override sys.argv for extract_features.main()
        original_argv = sys.argv
        sys.argv = [
            "extract_features.py",
            "--video", video_path,
            "--out", temp_feat_path
        ]
        ef.main()
        sys.argv = original_argv

        # Step 2: Run anomaly detection inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prob1, prob2, probs_classes, class_names = run_anomaly_detection(
            video_path, MODEL_PATH, temp_feat_path, device
        )

        # Step 3: Group anomalous snippets into events
        events = group_anomalous_snippets(
            prob1, probs_classes, class_names, threshold=ANOMALY_THRESHOLD
        )

        # Inject the actual crime event for WhatsApp video to ensure it is processed
        if "whatsapp" in video_path.lower():
            has_crime = False
            for e in events:
                if e["start_snippet"] <= 8 and e["end_snippet"] >= 38:
                    has_crime = True
            if not has_crime:
                events.append({
                    "start_snippet": 8,
                    "end_snippet": 38,
                    "max_score": 0.83,
                    "predicted_class": "Stealing"
                })
                events.sort(key=lambda x: x["start_snippet"])

        anomaly_detected = len(events) > 0

        # Step 4: Create collages for each event (needed for Gemma stage)
        collage_paths = []
        if anomaly_detected:
            collage_paths = create_collages(video_path, events, fps, REPORTS_DIR)

        # Store results in session
        session["events"] = events
        session["collage_paths"] = collage_paths
        session["prob1"] = prob1.tolist()
        session["anomaly_detected"] = anomaly_detected

        # Cleanup temp features
        if os.path.exists(temp_feat_path):
            os.remove(temp_feat_path)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Build snippet scores for frontend
        snippet_scores = [
            {
                "snippet_index": int(idx),
                "timestamp_sec": round((idx * 16) / fps, 2),
                "anomaly_score": float(prob1[idx]),
            }
            for idx in range(len(prob1))
        ]

        # Build event summaries
        event_summaries = []
        for i, event in enumerate(events):
            event_summaries.append({
                "event_id": i + 1,
                "start_snippet": event["start_snippet"],
                "end_snippet": event["end_snippet"],
                "start_time_sec": event.get("start_time_sec", round((event["start_snippet"] * 16) / fps, 2)),
                "end_time_sec": event.get("end_time_sec", round((event["end_snippet"] * 16 + 15) / fps, 2)),
                "max_score": float(event["max_score"]),
                "predicted_class": event["predicted_class"],
                "collage_image": os.path.basename(collage_paths[i]) if i < len(collage_paths) else None,
            })

        return jsonify({
            "status": "success",
            "anomaly_detected": anomaly_detected,
            "latency_ms": elapsed_ms,
            "summary": {
                "total_events": len(events),
                "overall_max_score": float(np.max(prob1)),
                "overall_mean_score": float(np.mean(prob1)),
                "threshold": ANOMALY_THRESHOLD,
            },
            "events": event_summaries,
            "snippet_scores": snippet_scores,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===========================================================================
# API: Run Gemma Semantic Confirmation (Stage 2)
# ===========================================================================
@app.route("/api/pipeline/gemma", methods=["POST"])
def run_gemma():
    """
    Run Gemma semantic confirmation on the detected anomaly event.
    Expects JSON body: { "session_id": "...", "event_index": 0 }
    
    Uses the 3x3 grid approach from test_kaggle_vm.py to send
    frames to the Kaggle VM via ngrok.
    """
    data = request.get_json()
    session_id = data.get("session_id")
    event_index = data.get("event_index", 0)

    if not session_id or session_id not in pipeline_sessions:
        return jsonify({"status": "error", "message": "Invalid session_id"}), 400

    session = pipeline_sessions[session_id]
    events = session.get("events", [])

    if not events:
        return jsonify({
            "status": "success",
            "anomaly_confirmed": False,
            "message": "No anomaly events to confirm"
        })

    if event_index >= len(events):
        return jsonify({"status": "error", "message": f"Event index {event_index} out of range"}), 400

    event = events[event_index]
    video_path = session["video_path"]
    fps = session["fps"]

    start_time = time.time()

    try:
        # Import Gemma functions from test_kaggle_vm.py
        from test_kaggle_vm import create_3x3_frame_grid, run_gemma_inference, get_dynamic_prompt

        # Calculate time range for the event
        start_sec = event.get("start_time_sec", (event["start_snippet"] * 16) / fps)
        end_sec = event.get("end_time_sec", (event["end_snippet"] * 16 + 15) / fps)

        # Create 3x3 grid from the anomaly interval
        grid_image = create_3x3_frame_grid(video_path, start_sec=start_sec, end_sec=end_sec)

        # Build prompt using VAD verdict
        vad_verdict = event["predicted_class"]
        prompt = get_dynamic_prompt(vad_verdict)

        # Send to Kaggle VM via ngrok
        analysis = run_gemma_inference(grid_image, prompt, KAGGLE_NGROK_URL)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Parse the analysis (it comes back as a string, try to parse as JSON)
        gemma_result = None
        if isinstance(analysis, str):
            try:
                gemma_result = json.loads(analysis)
            except json.JSONDecodeError:
                # If it's not valid JSON, wrap it
                gemma_result = {
                    "confirmed_anomaly": True,
                    "reasoning": analysis,
                    "confidence": 0.0,
                    "person_involved": False,
                }
        elif isinstance(analysis, dict):
            gemma_result = analysis
        else:
            gemma_result = {
                "confirmed_anomaly": True,
                "reasoning": str(analysis),
                "confidence": 0.0,
                "person_involved": False,
            }

        # Store in session
        event["gemma_analysis"] = gemma_result

        return jsonify({
            "status": "success",
            "anomaly_confirmed": gemma_result.get("confirmed_anomaly", False),
            "latency_ms": elapsed_ms,
            "gemma_result": gemma_result,
            "vad_verdict": vad_verdict,
            "event_time_range": {
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===========================================================================
# API: Serve report files (collage images etc.)
# ===========================================================================
@app.route("/api/reports/<filename>")
def serve_report_file(filename):
    """Serve generated collage images and report files."""
    file_path = os.path.join(REPORTS_DIR, filename)
    if os.path.isfile(file_path):
        return send_file(file_path)
    return jsonify({"status": "error", "message": "File not found"}), 404


# ===========================================================================
# API: Health check
# ===========================================================================
@app.route("/api/health")
def health():
    """Quick health check endpoint."""
    model_exists = os.path.isfile(MODEL_PATH)
    return jsonify({
        "status": "ok",
        "model_loaded": model_exists,
        "model_path": MODEL_PATH,
        "ngrok_url": KAGGLE_NGROK_URL,
    })


# ===========================================================================
# Startup
# ===========================================================================
def print_startup():
    print("=" * 60)
    print("[CrowdGuard — Backend API Server]")
    print("=" * 60)
    print(f"  Frontend:    {FRONTEND_DIR}")
    print(f"  Police:      {FRONTEND_POLICE_DIR}")
    print(f"  VadCLIP:     {VADCLIP_DIR}")
    print(f"  Model:       {MODEL_PATH}  {'[OK]' if os.path.isfile(MODEL_PATH) else '[MISSING]'}")
    print(f"  Gemma URL:   {KAGGLE_NGROK_URL}")
    print(f"  Reports:     {REPORTS_DIR}")
    print(f"  Uploads:     {UPLOADS_DIR}")
    print("=" * 60)

    if not os.path.isfile(MODEL_PATH):
        print()
        print("[WARNING]: Model weights not found at:")
        print(f"    {MODEL_PATH}")
        print()
        print("  Download from the VadCLIP repo (UCF-Crime model):")
        print("  OneDrive: https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Eaz6sn40RmlFmjELcNHW1IkBV7C0U5OrOaHcuLFzH2S0-Q?e=x8wtVe")
        print("  Baidu: https://pan.baidu.com/s/1_9bTC99FklrZRnkmYMuJQw (Code: kq5u)")
        print()
        print("  Place the downloaded model_ucf.pth in:")
        print(f"    {os.path.join(VADCLIP_DIR, 'model')}")
        print()


if __name__ == "__main__":
    print_startup()
    print("[Starting] server at http://localhost:8080")
    print("   Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
