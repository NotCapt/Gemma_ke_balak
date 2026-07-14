# GemmaServer (Central Inference Hub)

The `GemmaServer` is the **core inference engine** for the entire Gemma Kavach v2 architecture. It runs the fine-tuned Gemma 4 multimodal model (E2B-it) locally on a single machine, providing inference APIs to all three downstream client services (Vision, Voice, and Chat).

By centralizing inference, we achieve:
1. **On-Device Privacy:** No data (images, voice, text) is ever sent to the cloud.
2. **Resource Efficiency:** We only need to load the 2B model into GPU/RAM once.
3. **Zero Latency/Network Hops:** Services communicate via `localhost:8000`.

## Endpoints Provided

- `POST /ask_image`: Multimodal vision reasoning (used by Vision Server).
- `POST /classify`: Text classification with reasoning (used by Chat Server).
- `POST /ask`: Audio transcription and general query (used by Voice Server).
- `POST /generate`: Basic text generation (used by Voice Server for natural Hindi responses).

## Security

- **CORS Hardened**: Restricted strictly to localhost to prevent unauthorized external access.

## Running the Server

```bash
cd GemmaServer
pip install -r requirements.txt
python gemma_server.py
```
*Note: Make sure this server is running on port 8000 before starting any of the client services.*
