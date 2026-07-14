# Gemma Kavach Voice Server

The Voice Server handles real-time audio queries from security officers (e.g., "What is the status of Zone B?").

In v2, this server acts as an **API Client** that queries the central `GemmaServer` running on `localhost:8000`.

## Key Features (v2)

- **Local Audio Transcription:** Uses the central GemmaServer for transcription instead of relying on external cloud APIs or RunPod.
- **Natural Hindi Responses:** Uses Gemma to generate natural, conversational Hindi responses based on real-time database metrics.
- **Database Integration:** Queries the shared `kavach.db` SQLite database to get the latest vision analytics for the requested zone.
- **Graceful TTS Fallback:** Uses Google TTS for audio generation, but fails gracefully if the API key is not present.

## Setup & Run

1. Ensure `GemmaServer` is running on port 8000.
2. (Optional) Set `GOOGLE_TEXT_TO_SPEECH` in your `.env` file for audio responses.
3. Run the Voice Server:
```bash
cd Gemma_Kavach_Voice_Server
pip install -r requirements.txt
python main.py
```

The server runs on port `8003`.