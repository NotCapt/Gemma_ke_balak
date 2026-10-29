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
    if os.name == "nt":
        venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    else:
        venv_python = os.path.join(project_root, "venv", "bin", "python")

    # Prefer the project venv, otherwise use whichever interpreter launched us
    # (conda env, system Python, an already-activated venv...).
    if not os.path.exists(venv_python):
        print(f"[SYSTEM] No venv at {venv_python}; using current interpreter {sys.executable}")
        venv_python = sys.executable

    print("=" * 70)
    print("[SEMANTIC ANOMALY DETECTION & CROSS-CAMERA FORENSIC SYSTEM]")
    print("=" * 70)
    print("[SYSTEM] Booting Security Pipeline...")
    print(f"[SYSTEM] Launching Flask Server via {venv_python}")
    
    # Open browser slightly after starting
    def open_browser():
        time.sleep(2)
        webbrowser.open(f"http://localhost:{os.environ.get('PORT', '8080')}")
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Launch the flask server
    try:
        subprocess.run([venv_python, server_script], cwd=project_root)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down all AI components gracefully. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
