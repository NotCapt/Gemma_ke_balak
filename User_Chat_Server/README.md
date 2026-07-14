# User Chat Server (Emergency Reporting)

The Chat Server handles text-based emergency reports submitted by the public via the web interface.

In v2, this server acts as an **API Client** that queries the central `GemmaServer` running on `localhost:8000`.

## Key Features (v2)

- **Server-Side Classification:** Emergency reports are classified server-side using the fine-tuned Gemma model, preventing client-side spoofing.
- **Structured Reasoning:** Returns severity, recommended action, and AI reasoning.
- **Audit Integration:** Logs all emergency reports to the centralized `kavach.db` SQLite database (in the `emergency_reports` and `audit_trail` tables).
- **Cloud-Free:** Image attachments are saved locally. All GCS dependencies have been removed.

## Setup & Run

1. Ensure `GemmaServer` is running on port 8000.
2. Run the Chat Server:
```bash
cd User_Chat_Server
pip install -r requirements.txt
python main.py
```

The server runs on port `8002`.