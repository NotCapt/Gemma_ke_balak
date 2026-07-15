import os
import sys
import subprocess
import webbrowser
import threading
import time

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    server_script = os.path.join(project_root, "server.py")
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    
    # Check if venv exists
    if not os.path.exists(venv_python):
        print("⚠️ Virtual environment not found at", venv_python)
        print("Please ensure the venv is created and dependencies are installed.")
        sys.exit(1)
        
    print("=" * 70)
    print("[SEMANTIC ANOMALY DETECTION & CROSS-CAMERA FORENSIC SYSTEM]")
    print("=" * 70)
    print("[SYSTEM] Booting Security Pipeline...")
    print(f"[SYSTEM] Launching Flask Server via {venv_python}")
    
    # Open browser slightly after starting
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:8080")
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Launch the flask server
    try:
        subprocess.run([venv_python, server_script], cwd=project_root)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down all AI components gracefully. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
