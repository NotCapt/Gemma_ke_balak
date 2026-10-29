"""
CrowdGuard — Backend API Server
Connects the frontend dashboard to VadCLIP anomaly detection,
Gemma semantic confirmation via Kaggle VM, YOLOv8n + OSNet tracking,
and the hash-chained forensic log.

Usage (from project root, inside venv):
    python server.py

Configuration (environment variables):
    GEMMA_SERVER_URL      ngrok URL of the Kaggle VM serving Gemma (needed for Stage 2)
    VAD_THRESHOLD         VadCLIP anomaly-score threshold (default 0.010)
    VAD_CROP              feature-extraction crop: official | clip (default official)
    GEMMA_MIN_WINDOW_SEC  minimum clip length shown to Gemma (default 6.0)
    YOLO_IMGSZ            YOLO input size; 1280 catches small CCTV figures (default 640)
    YOLO_CONF             minimum person confidence (default 0.4)
    OSNET_WEIGHTS         OSNet Re-ID checkpoint for Stage 4 (see yolo_onset/yolo_osnet.py)
    REID_THRESHOLD        cosine similarity for a cross-camera match (default 0.70)
    MAX_UPLOAD_MB         upload size limit (default 100, matches the dashboard copy)
    PORT                  default 8080
"""

import os
import sys
import time
import uuid
import re
import hashlib
import threading
import functools
import importlib.util
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path setup — so we can import from VadCLIP/, gemma/ and yolo_onset/
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VADCLIP_DIR = os.path.join(PROJECT_ROOT, "VadCLIP")
GEMMA_DIR = os.path.join(PROJECT_ROOT, "gemma")
YOLO_DIR = os.path.join(PROJECT_ROOT, "yolo_onset")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_POLICE_DIR = os.path.join(PROJECT_ROOT, "frontend_police")

sys.path.insert(0, VADCLIP_DIR)
sys.path.insert(0, os.path.join(VADCLIP_DIR, "src"))
sys.path.insert(0, GEMMA_DIR)
sys.path.insert(0, YOLO_DIR)

from forensic_log import ForensicLog
from reid_gallery import EmbeddingGallery

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# ngrok free-tier URLs rotate on every restart — never hardcode them.
KAGGLE_NGROK_URL = os.environ.get("GEMMA_SERVER_URL", "").rstrip("/")
MODEL_PATH = os.path.join(VADCLIP_DIR, "model", "model_ucf.pth")
REPORTS_DIR = os.path.join(VADCLIP_DIR, "reports")   # sample outputs from the CLI pipeline
RUNS_DIR = os.path.join(VADCLIP_DIR, "runs")         # per-session artifacts from this server
UPLOADS_DIR = os.path.join(VADCLIP_DIR, "uploads")
FORENSIC_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "forensic_log.jsonl")

ANOMALY_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.010"))
VAD_CROP = os.environ.get("VAD_CROP", "official")
GEMMA_MIN_WINDOW_SEC = float(os.environ.get("GEMMA_MIN_WINDOW_SEC", "6.0"))
REID_THRESHOLD = float(os.environ.get("REID_THRESHOLD", "0.70"))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.4"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
PORT = int(os.environ.get("PORT", "8080"))

SNIPPET_FRAMES = 16          # VadCLIP scores one feature per 16 frames
TRACK_SAMPLE_FRAMES = 8      # frames sampled from an event for YOLO
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DECISION_ACTIONS = {"confirm", "reject", "override"}

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

# In-memory store for pipeline state per session
pipeline_sessions = {}

# One heavy model call at a time — concurrent CLIP/VadCLIP/YOLO runs just
# thrash memory on a single GPU/CPU.
_inference_lock = threading.Lock()

forensic_log = ForensicLog(FORENSIC_LOG_PATH)
reid_gallery = EmbeddingGallery(threshold=REID_THRESHOLD)


# ===========================================================================
# Helpers
# ===========================================================================
def _error(message, status=400, **extra):
    return jsonify({"status": "error", "message": message, **extra}), status


def _json_body():
    """request.get_json() returns None (→ AttributeError) on a missing/invalid body."""
    return request.get_json(silent=True) or {}


def _lookup_session(data):
    session_id = data.get("session_id")
    if not session_id or session_id not in pipeline_sessions:
        return None, None, _error("Invalid session_id")
    return session_id, pipeline_sessions[session_id], None


def _lookup_event(session, data):
    try:
        event_index = int(data.get("event_index", 0))
    except (TypeError, ValueError):
        return None, None, _error("event_index must be an integer")
    events = session.get("events", [])
    # Reject negatives explicitly — Python would silently index from the end.
    if not 0 <= event_index < len(events):
        return None, None, _error(f"Event index {event_index} out of range")
    return event_index, events[event_index], None


def _sanitize_camera_id(value):
    value = re.sub(r"[^A-Za-z0-9_-]", "", str(value or ""))[:32]
    return value or "CAM-01"


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _severity(event):
    """Severity = VAD score × Gemma confidence (0 until Gemma has confirmed)."""
    gemma = event.get("gemma_analysis") or {}
    if not gemma.get("confirmed_anomaly"):
        return 0.0
    return round(float(event["max_score"]) * float(gemma.get("confidence", 0.0)), 4)


def _event_bounds(event, fps):
    """(start_sec, end_sec) of an event; create_collages normally fills these in."""
    start_sec = event.get("start_time_sec", (event["start_snippet"] * SNIPPET_FRAMES) / fps)
    end_sec = event.get("end_time_sec", (event["end_snippet"] * SNIPPET_FRAMES + 15) / fps)
    return float(start_sec), float(end_sec)


def _gemma_window(event, fps, duration):
    """Event interval, widened to GEMMA_MIN_WINDOW_SEC so a 3x3 grid shows actual progression."""
    start_sec, end_sec = _event_bounds(event, fps)
    if end_sec - start_sec < GEMMA_MIN_WINDOW_SEC and duration > 0:
        center = (start_sec + end_sec) / 2
        start_sec = max(0.0, center - GEMMA_MIN_WINDOW_SEC / 2)
        end_sec = min(duration, start_sec + GEMMA_MIN_WINDOW_SEC)
        start_sec = max(0.0, end_sec - GEMMA_MIN_WINDOW_SEC)
    return start_sec, end_sec


def _tracking_dependencies():
    return {name: importlib.util.find_spec(name) is not None for name in ("ultralytics", "torchreid")}


@functools.lru_cache(maxsize=1)
def _load_tracking_models():
    """YOLOv8n + OSNet, loaded once. Raises ImportError if the libraries are missing."""
    import torch
    from yolo_osnet import load_yolo, load_osnet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = load_yolo()
    osnet_model, reid_weights = load_osnet(device=device)
    return yolo_model, osnet_model, reid_weights


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
        return _error("No video file provided")

    video_file = request.files["video"]
    if video_file.filename == "":
        return _error("Empty filename")

    ext = (os.path.splitext(video_file.filename)[1] or ".mp4").lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return _error(f"Unsupported video type '{ext}'")

    # Generate unique session ID
    session_id = uuid.uuid4().hex[:8]

    # Save video to uploads directory
    video_filename = f"{session_id}{ext}"
    video_path = os.path.join(UPLOADS_DIR, video_filename)
    video_file.save(video_path)

    # Get basic video info
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or total_frames <= 0:
        os.remove(video_path)
        return _error("Could not read the video (unsupported codec or corrupt file)")
    duration = total_frames / fps

    # Store session state
    pipeline_sessions[session_id] = {
        "video_path": video_path,
        "video_filename": video_file.filename,
        "video_sha256": _file_sha256(video_path),
        "camera_id": _sanitize_camera_id(request.form.get("camera_id")),
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "events": [],
        "collage_paths": [],
    }

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "camera_id": pipeline_sessions[session_id]["camera_id"],
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
    session_id, session, err = _lookup_session(_json_body())
    if err:
        return err

    if not os.path.isfile(MODEL_PATH):
        return _error(f"VadCLIP weights not found at {MODEL_PATH} — see README step 1", 500)

    video_path = session["video_path"]
    fps = session["fps"]

    start_time = time.time()

    try:
        # Import VadCLIP pipeline functions
        import numpy as np
        import torch
        import extract_features as ef
        from pipeline import (
            run_anomaly_detection,
            group_anomalous_snippets,
            create_collages,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with _inference_lock:
            # Step 1: Extract features (models are cached after the first request)
            features = ef.extract_video_features(video_path, device=device, crop=VAD_CROP)

            # Step 2: Run anomaly detection inference
            prob1, prob2, probs_classes, class_names = run_anomaly_detection(
                video_path, MODEL_PATH, None, device, features=features
            )

        # Step 3: Group anomalous snippets into events
        events = group_anomalous_snippets(
            prob1, probs_classes, class_names, threshold=ANOMALY_THRESHOLD
        )
        anomaly_detected = len(events) > 0

        # Step 4: Create collages for each event (also sets buffered start/end times)
        collage_paths = []
        if anomaly_detected:
            collage_paths = create_collages(video_path, events, fps, RUNS_DIR, prefix=f"{session_id}_")

        # Store results in session
        session["events"] = events
        session["collage_paths"] = collage_paths
        session["prob1"] = prob1.tolist()
        session["anomaly_detected"] = anomaly_detected

        elapsed_ms = int((time.time() - start_time) * 1000)
        session["vad_latency_ms"] = elapsed_ms

        # Build snippet scores for frontend
        snippet_scores = [
            {
                "snippet_index": int(idx),
                "timestamp_sec": round((idx * SNIPPET_FRAMES) / fps, 2),
                "anomaly_score": float(prob1[idx]),
            }
            for idx in range(len(prob1))
        ]

        # Build event summaries
        event_summaries = []
        for i, event in enumerate(events):
            collage_name = os.path.basename(collage_paths[i]) if i < len(collage_paths) else None
            event_summaries.append({
                "event_id": i + 1,
                "start_snippet": event["start_snippet"],
                "end_snippet": event["end_snippet"],
                "start_time_sec": event.get("start_time_sec", round((event["start_snippet"] * SNIPPET_FRAMES) / fps, 2)),
                "end_time_sec": event.get("end_time_sec", round((event["end_snippet"] * SNIPPET_FRAMES + 15) / fps, 2)),
                "max_score": float(event["max_score"]),
                "predicted_class": event["predicted_class"],
                "collage_image": collage_name,
                "collage_url": f"/api/reports/{collage_name}" if collage_name else None,
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
                "model": "VadCLIP (UCF-Crime)",
                "crop_mode": VAD_CROP,
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
    Run Gemma semantic confirmation on one detected anomaly event.
    Expects JSON body: { "session_id": "...", "event_index": 0 }

    Uses the 3x3 grid approach from test_kaggle_vm.py to send
    frames to the Kaggle VM via ngrok.
    """
    data = _json_body()
    session_id, session, err = _lookup_session(data)
    if err:
        return err

    if not session.get("events"):
        return jsonify({
            "status": "success",
            "anomaly_confirmed": False,
            "message": "No anomaly events to confirm"
        })

    event_index, event, err = _lookup_event(session, data)
    if err:
        return err

    video_path = session["video_path"]
    start_time = time.time()

    try:
        # Import Gemma functions from test_kaggle_vm.py
        from test_kaggle_vm import (
            create_3x3_frame_grid, run_gemma_inference, get_dynamic_prompt, parse_gemma_json,
        )

        start_sec, end_sec = _gemma_window(event, session["fps"], session["duration"])

        # Create 3x3 grid from the anomaly interval and keep a copy for the explainability panel
        grid_image = create_3x3_frame_grid(video_path, start_sec=start_sec, end_sec=end_sec)
        grid_name = f"{session_id}_event_{event_index + 1}_gemma_grid.jpg"
        grid_image.save(os.path.join(RUNS_DIR, grid_name), "JPEG", quality=85)

        # Build prompt using VAD verdict
        vad_verdict = event["predicted_class"]
        prompt = get_dynamic_prompt(vad_verdict)

        # Send to Kaggle VM via ngrok
        analysis = run_gemma_inference(grid_image, prompt, KAGGLE_NGROK_URL)

        elapsed_ms = int((time.time() - start_time) * 1000)

        gemma_result = parse_gemma_json(analysis)
        if gemma_result is None:
            # Unparseable output: escalate to a human instead of silently dropping
            # the event, but make it obvious this wasn't a real confirmation.
            gemma_result = {
                "confirmed_anomaly": True,
                "needs_review": True,
                "parse_error": True,
                "reasoning": str(analysis)[:1000],
                "confidence": 0.0,
                "person_involved": False,
                "objects_detected": [],
            }

        # Store in session
        event["gemma_analysis"] = gemma_result
        event["gemma_latency_ms"] = elapsed_ms
        event["gemma_window"] = [round(start_sec, 2), round(end_sec, 2)]

        return jsonify({
            "status": "success",
            "event_index": event_index,
            "anomaly_confirmed": gemma_result.get("confirmed_anomaly", False),
            "latency_ms": elapsed_ms,
            "gemma_result": gemma_result,
            "vad_verdict": vad_verdict,
            "vad_score": float(event["max_score"]),
            "severity": _severity(event),
            "grid_url": f"/api/reports/{grid_name}",
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
# API: Person Localization + Re-ID + Cross-Camera Matching (Stages 3-5)
# ===========================================================================
@app.route("/api/pipeline/track", methods=["POST"])
def run_tracking():
    """
    Run YOLOv8n on frames from a Gemma-confirmed event, embed each person with
    OSNet and match the embeddings against other cameras' galleries.
    Expects JSON body: { "session_id": "...", "event_index": 0 }
    """
    data = _json_body()
    session_id, session, err = _lookup_session(data)
    if err:
        return err
    event_index, event, err = _lookup_event(session, data)
    if err:
        return err

    gemma = event.get("gemma_analysis") or {}
    if not gemma.get("confirmed_anomaly"):
        return _error("Tracking is gated behind Gemma confirmation; this event is not confirmed")
    if not gemma.get("person_involved"):
        return jsonify({"status": "skipped", "message": "Gemma reported no person involved"})

    try:
        yolo_model, osnet_model, reid_weights = _load_tracking_models()
    except ImportError as e:
        return jsonify({"status": "unavailable", "message": str(e),
                        "dependencies": _tracking_dependencies()}), 503

    try:
        import cv2
        from yolo_osnet import detect_people, crop_boxes, embed_crop

        fps = session["fps"]
        # Scan the same (widened) window Gemma confirmed, not just the raw VAD interval.
        start_sec, end_sec = event.get("gemma_window") or _event_bounds(event, fps)
        start_frame = int(start_sec * fps)
        end_frame = min(session["total_frames"] - 1, int(end_sec * fps))
        span = max(0, end_frame - start_frame)
        frame_indices = sorted({start_frame + round(i * span / max(1, TRACK_SAMPLE_FRAMES - 1))
                                for i in range(TRACK_SAMPLE_FRAMES)})

        # Stage 3: detect people on a few frames, keep the frame with the most detections
        cap = cv2.VideoCapture(session["video_path"])
        best = None
        t0 = time.time()
        with _inference_lock:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                boxes = detect_people(frame, yolo_model, conf_thresh=YOLO_CONF, imgsz=YOLO_IMGSZ)
                score = (len(boxes), sum(b[4] for b in boxes))
                if best is None or score > best[0]:
                    best = (score, idx, frame, boxes)
        cap.release()
        frames_scanned = len(frame_indices)
        yolo_ms_per_frame = int((time.time() - t0) * 1000 / max(1, frames_scanned))

        if best is None:
            return _error("Could not read any frames from the event interval", 500)
        _, best_idx, best_frame, boxes = best

        # Stage 4: OSNet embedding per person crop
        t1 = time.time()
        with _inference_lock:
            embeddings = [embed_crop(crop, osnet_model) for crop in crop_boxes(best_frame, boxes)]
        reid_ms = int((time.time() - t1) * 1000)

        # Stage 5: cross-camera matching against other cameras' embeddings
        event_ref = f"{session_id}#{event_index + 1}"
        matches = reid_gallery.match_and_add(session["camera_id"], embeddings,
                                             meta={"event_ref": event_ref, "session_id": session_id})

        # Annotated keyframe for the dashboard
        annotated = best_frame.copy()
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (246, 92, 139), 2)
            cv2.putText(annotated, f"person {i} {conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (246, 92, 139), 2)
        yolo_name = f"{session_id}_event_{event_index + 1}_yolo.jpg"
        cv2.imwrite(os.path.join(RUNS_DIR, yolo_name), annotated)

        tracking = {
            "keyframe_index": int(best_idx),
            "keyframe_time_sec": round(best_idx / fps, 2),
            "frames_scanned": frames_scanned,
            "detections": [
                {"person": i, "bbox": [x1, y1, x2, y2], "confidence": round(conf, 3)}
                for i, (x1, y1, x2, y2, conf) in enumerate(boxes)
            ],
            "yolo_latency_ms": yolo_ms_per_frame,
            "reid_latency_ms": reid_ms,
            "reid_weights": reid_weights,
            "embedding_dim": int(embeddings[0].shape[0]) if embeddings else 512,
            "embedding_preview": [round(float(v), 4) for v in embeddings[0][:64]] if embeddings else [],
            "cross_camera": {
                "camera_id": session["camera_id"],
                "threshold": REID_THRESHOLD,
                "cameras_in_gallery": reid_gallery.cameras(),
                "matches": matches,
                "any_match": any(m["matched"] for m in matches),
            },
        }
        event["tracking"] = tracking

        return jsonify({"status": "success", "annotated_url": f"/api/reports/{yolo_name}", **tracking})

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ===========================================================================
# API: Forensic Log (Stage 6) + Operator Decisions (Stage 7)
# ===========================================================================
@app.route("/api/pipeline/log", methods=["POST"])
def log_event():
    """
    Append a hash-chained forensic record for one event, built from server-side
    session state (not from client-supplied values).
    Expects JSON body: { "session_id": "...", "event_index": 0 }
    """
    data = _json_body()
    session_id, session, err = _lookup_session(data)
    if err:
        return err
    event_index, event, err = _lookup_event(session, data)
    if err:
        return err

    gemma = event.get("gemma_analysis")
    payload = {
        "event_ref": f"{session_id}#{event_index + 1}",
        "camera_id": session["camera_id"],
        "video": {"filename": session["video_filename"], "sha256": session["video_sha256"]},
        "stages": {
            "vad": {
                "model": "VadCLIP (UCF-Crime)",
                "score": float(event["max_score"]),
                "threshold": ANOMALY_THRESHOLD,
                "crop_mode": VAD_CROP,
                "predicted_class": event["predicted_class"],
                "start_time_sec": event.get("start_time_sec"),
                "end_time_sec": event.get("end_time_sec"),
                "latency_ms": session.get("vad_latency_ms"),
            },
            "gemma": None if gemma is None else {
                **gemma,
                "window_sec": event.get("gemma_window"),
                "latency_ms": event.get("gemma_latency_ms"),
            },
            "tracking": event.get("tracking"),
        },
        "severity_score": _severity(event),
    }
    entry = forensic_log.append("pipeline_event", payload)
    event["forensic_hash"] = entry["hash"]
    return jsonify({"status": "success", "entry": entry})


@app.route("/api/forensic/decision", methods=["POST"])
def log_decision():
    """
    Record the operator's human-in-the-loop decision, chained to the event's log entry.
    Expects JSON body: { "session_id", "event_index", "action": confirm|reject|override, "note"? }
    """
    data = _json_body()
    session_id, session, err = _lookup_session(data)
    if err:
        return err
    event_index, event, err = _lookup_event(session, data)
    if err:
        return err

    action = data.get("action")
    if action not in DECISION_ACTIONS:
        return _error(f"action must be one of {sorted(DECISION_ACTIONS)}")
    if not event.get("forensic_hash"):
        return _error("Log the event (POST /api/pipeline/log) before recording a decision")

    entry = forensic_log.append("operator_decision", {
        "event_ref": f"{session_id}#{event_index + 1}",
        "event_hash": event["forensic_hash"],
        "action": action,
        "note": str(data.get("note", ""))[:500],
    })
    return jsonify({"status": "success", "entry": entry})


@app.route("/api/forensic/log")
def get_forensic_log():
    """Most recent forensic entries (default 50)."""
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 1000))
    except ValueError:
        limit = 50
    return jsonify({"status": "success", "entries": forensic_log.entries(limit)})


@app.route("/api/forensic/verify")
def verify_forensic_log():
    """Re-walk the hash chain and report the first broken entry, if any."""
    return jsonify({"status": "success", **forensic_log.verify()})


# ===========================================================================
# API: Serve report files (collage images etc.)
# ===========================================================================
@app.route("/api/reports/<filename>")
def serve_report_file(filename):
    """Serve generated collage/grid/YOLO images. send_from_directory blocks path traversal."""
    for directory in (RUNS_DIR, REPORTS_DIR):
        if os.path.isfile(os.path.join(directory, os.path.basename(filename))):
            return send_from_directory(directory, os.path.basename(filename))
    return _error("File not found", 404)


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
        "ngrok_url": KAGGLE_NGROK_URL or None,
        "vad_threshold": ANOMALY_THRESHOLD,
        "vad_crop": VAD_CROP,
        "tracking_dependencies": _tracking_dependencies(),
        "yolo": {"imgsz": YOLO_IMGSZ, "conf": YOLO_CONF},
        "osnet_weights": os.environ.get("OSNET_WEIGHTS") or None,
        "forensic_log": forensic_log.verify(),
    })


@app.errorhandler(413)
def upload_too_large(_):
    return _error(f"Upload exceeds {MAX_UPLOAD_MB} MB (set MAX_UPLOAD_MB to raise the limit)", 413)


# ===========================================================================
# Startup
# ===========================================================================
def print_startup():
    deps = _tracking_dependencies()
    print("=" * 60)
    print("[CrowdGuard — Backend API Server]")
    print("=" * 60)
    print(f"  Frontend:    {FRONTEND_DIR}")
    print(f"  Police:      {FRONTEND_POLICE_DIR}")
    print(f"  VadCLIP:     {VADCLIP_DIR}")
    print(f"  Model:       {MODEL_PATH}  {'[OK]' if os.path.isfile(MODEL_PATH) else '[MISSING]'}")
    print(f"  VAD:         threshold={ANOMALY_THRESHOLD} crop={VAD_CROP}")
    print(f"  Gemma URL:   {KAGGLE_NGROK_URL or '[NOT SET] — export GEMMA_SERVER_URL'}")
    print(f"  Tracking:    ultralytics={'[OK]' if deps['ultralytics'] else '[MISSING]'} "
          f"torchreid={'[OK]' if deps['torchreid'] else '[MISSING]'} "
          f"OSNet weights={os.environ.get('OSNET_WEIGHTS') or '[ImageNet only]'}")
    print(f"  Runs:        {RUNS_DIR}")
    print(f"  Uploads:     {UPLOADS_DIR}")
    print(f"  Forensic:    {FORENSIC_LOG_PATH}")
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
    print(f"[Starting] server at http://localhost:{PORT}")
    print("   Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
