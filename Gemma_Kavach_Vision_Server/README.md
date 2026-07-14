# Gemma Kavach Vision Server

The Vision Server monitors CCTV/drone feeds to detect crowd density, panic behavior, and potential emergencies in real-time. 

In v2, this server acts as an **API Client** that queries the central `GemmaServer` running on `localhost:8000`. It no longer runs its own separate model.

## Key Features (v2)

- **Dual-Analysis Pipeline:** Analyzes both Crowd Density and Crowd Motion simultaneously.
- **Reasoning-Augmented Classification:** Returns Gemma's step-by-step reasoning for *why* a frame was flagged (e.g., "High density + Chaotic motion detected").
- **Centralized Audit Database:** Uses a shared SQLite database (`kavach.db`) to log all flagged frames and reasoning, enabling forensic audit trails.
- **Cloud-Free:** All GCS dependencies have been removed. Flagged frames are saved locally.

## Setup & Run

1. Ensure `GemmaServer` is running on port 8000.
2. Run the Vision Server:
```bash
cd Gemma_Kavach_Vision_Server
pip install -r requirements.txt
python main.py
```

The server runs on port `8001`.
