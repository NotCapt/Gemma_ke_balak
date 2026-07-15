import io
import cv2
import requests
from PIL import Image
import numpy as np

# =====================================================================
# ⚙️ LOCAL CLIENT CONFIGURATION
# =====================================================================
KAGGLE_NGROK_URL = "https://67e9-34-134-45-216.ngrok-free.app"

# Use the video we have been working with
VIDEO_PATH = r"C:\Users\govin\Downloads\t2\WhatsApp Video 2026-07-15 at 12.12.55 PM.mp4"

def get_dynamic_prompt(vad_verdict: str) -> str:
    return (
        f"You are an expert AI Security Analyst examining a 3x3 grid of chronological CCTV frames. "
        f"The primary Video Anomaly Detection (VAD) model has flagged this sequence with the verdict: '{vad_verdict}'. "
        f"Analyze the progression of events across the grid panels from top-left to bottom-right. "
        f"Verify if the visual evidence aligns with the '{vad_verdict}' verdict. "
        f"Identify any suspicious activities, focusing on vehicle break-ins, forced entry, or theft. "
        f"Provide an extremely short summary of actions observed (less than 40 words).\n\n"
        f"You MUST output your response strictly as a JSON object with this format:\n"
        f"{{\n"
        f"  \"confirmed_anomaly\": true/false,\n"
        f"  \"person_involved\": true/false,\n"
        f"  \"anomaly_type\": \"class_name\",\n"
        f"  \"objects_detected\": [\"object1\", \"object2\"],\n"
        f"  \"reasoning\": \"under 15 words observation\",\n"
        f"  \"confidence\": 0.95,\n"
        f"  \"department_to_notify\": \"Police\" or \"Hospital\" or \"Fire Dept.\"\n"
        f"}}\n"
        f"Do not include any Markdown formatting or extra text outside the JSON object. Keep reasoning extremely short."
    )

# =====================================================================
# 📹 3x3 GRID FRAME STITCHER
# =====================================================================
def create_3x3_frame_grid(video_path: str, start_sec: float=None, end_sec: float=None) -> Image.Image:
    """Extracts 9 chronological frames from a video anomaly interval and stitches them into a 3x3 PIL Canvas."""
    print(f"[Processing] local video stream: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open or find video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_sec is None or end_sec is None:
        start_frame = 0
        end_frame = total_frames - 1
    else:
        # Convert seconds to frames
        start_frame = max(0, int(start_sec * fps))
        end_frame = min(total_frames - 1, int(end_sec * fps))
        
    duration_frames = end_frame - start_frame
    if duration_frames < 8:
        # If interval is very short, spread across the available frames evenly
        step = max(1, duration_frames / 8.0)
    else:
        # 9 evenly spaced frames
        step = duration_frames / 8.0
        
    frame_indices = [int(start_frame + i * step) for i in range(9)]
    frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
    
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            # Fallback to black frame if read fails
            h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if h == 0 or w == 0:
                h, w = 480, 640 # default
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Optional: Resize individual frames slightly to prevent the 3x3 grid from being excessively large (e.g. 3x 1080p = 3240p)
        # Let's resize each frame to a manageable width (e.g. 640px) to ensure smooth network transmission
        h, w = frame.shape[:2]
        new_w = 640
        new_h = int((new_w / w) * h)
        frame = cv2.resize(frame, (new_w, new_h))
            
        frames.append(Image.fromarray(frame))
        
    cap.release()
    
    if len(frames) < 9:
        raise RuntimeError("Failed to extract 9 chronological frames.")
        
    frame_w, frame_h = frames[0].size
    grid_image = Image.new("RGB", (frame_w * 3, frame_h * 3))
    
    for i, img in enumerate(frames):
        grid_image.paste(img, ((i % 3) * frame_w, (i // 3) * frame_h))
        
    print(f"[Success] 3x3 Grid Matrix built successfully ({grid_image.size[0]}x{grid_image.size[1]}px).")
    return grid_image

# =====================================================================
# 🧠 GEMMA INFERENCE EXECUTOR
# =====================================================================
def run_gemma_inference(grid_image: Image.Image, prompt: str, server_url: str) -> str:
    """Sends the processed frame matrix to the remote Gemma-4 multimodal engine."""
    endpoint = f"{server_url}/api/inference"
    print(f"[Dispatching] visual payload to remote Gemma VLM endpoint: {endpoint}")
    
    img_byte_arr = io.BytesIO()
    grid_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    files = {"image": ("grid.png", img_byte_arr, "image/png")}
    data = {"prompt": prompt}
    
    # Required header to bypass ngrok free tier browser warning screen
    headers = {"ngrok-skip-browser-warning": "true"}
    
    response = requests.post(endpoint, files=files, data=data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            return result.get("analysis")
        else:
            raise Exception(f"Remote Gemma Application Exception: {result.get('message', result)}")
    else:
        raise Exception(f"HTTP Connection Failure ({response.status_code}): {response.text}")

# =====================================================================
# MAIN EXECUTION ROUTINE
# =====================================================================
if __name__ == "__main__":
    import sys
    # Try to set stdout to utf-8 if possible, or we just rely on string literals below
    print("=" * 60)
    print("SECURITY CCTV ANALYSIS PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Prepare the video grid locally
        # Testing with an anomaly interval detected by VAD (e.g. 0.0s to 5.0s buffer)
        processed_grid = create_3x3_frame_grid(VIDEO_PATH, start_sec=0.0, end_sec=5.0)
        
        # Simulated VAD verdict for this interval
        vad_verdict = "Stealing"
        dynamic_prompt = get_dynamic_prompt(vad_verdict)
        
        # Step 2: Request analysis from the remote Gemma engine
        analysis_report = run_gemma_inference(processed_grid, dynamic_prompt, KAGGLE_NGROK_URL)
            
        print("\n" + "=" * 60)
        print("ANALYSIS REPORT RECEIVED [ENGINE: GEMMA-4 MULTIMODAL]")
        print("=" * 60)
        print(analysis_report)
        print("=" * 60 + "\n")
        
    except Exception as error:
        print(f"\nPIPELINE ERROR OCCURRED: {error}\n")
