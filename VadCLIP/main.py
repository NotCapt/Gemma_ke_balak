import os
import sys
import time
import threading
import http.server
import socketserver
import webbrowser

def print_startup_sequence():
    """Simulates the loading of the various system components for presentation purposes."""
    print("=" * 70)
    print("🛡️ SEMANTIC ANOMALY DETECTION & CROSS-CAMERA FORENSIC SYSTEM 🛡️")
    print("=" * 70)
    
    steps = [
        ("[SYSTEM] Booting Security Pipeline...", 0.5),
        ("[MODULE 1] Initializing CLIP-TSA Video Anomaly Detection (VAD)...", 1.2),
        ("   └── Loading model weights (model_ucf.pth) to CUDA...", 0.8),
        ("[MODULE 2] Waking up Remote Gemma-4 Multimodal Engine...", 1.0),
        ("   └── Establishing secure ngrok connection (https://5e59-34-134-45-216.ngrok-free.app)...", 0.5),
        ("   └── Gemma API Connection: SUCCESS", 0.3),
        ("[MODULE 3] Initializing YOLOv8n Object Localization...", 1.5),
        ("   └── Loading Person detection classes...", 0.4),
        ("[MODULE 4] Initializing OSNet Re-ID Engine...", 1.2),
        ("   └── Loading Market-1501 feature extractors...", 0.6),
        ("[MODULE 5] Initializing Cross-Camera Embedding Matcher...", 0.4),
        ("[DATABASE] Connecting to Forensic Hash-Chained Log...", 0.5),
        ("[SUCCESS] All AI components loaded successfully. Gated pipeline is ACTIVE.", 0.2)
    ]
    
    for msg, delay in steps:
        print(msg)
        time.sleep(delay)
        
    print("=" * 70)
    print("🌐 Starting Dashboard Frontend Service...")
    print("=" * 70)

def start_frontend_server(port=8000):
    """Starts a simple HTTP server to host the static frontend."""
    # Find the frontend directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.abspath(os.path.join(current_dir, "..", "frontend"))
    
    if not os.path.exists(frontend_dir):
        print(f"[ERROR] Frontend directory not found at {frontend_dir}")
        sys.exit(1)
        
    # Change working directory so the server serves the correct files
    os.chdir(frontend_dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Use socketserver to avoid port collision issues
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"✅ Dashboard is live! Access it at: http://localhost:{port}")
        print("Press Ctrl+C to shut down the system.")
        httpd.serve_forever()

if __name__ == "__main__":
    try:
        # 1. Print the fake loading sequence
        print_startup_sequence()
        
        # 2. Open the browser to the frontend
        port = 8080
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        
        # 3. Start the static file server
        start_frontend_server(port)
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down all AI components gracefully. Goodbye!")
        sys.exit(0)
