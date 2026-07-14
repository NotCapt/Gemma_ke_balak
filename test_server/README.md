# Test Server

These scripts are used to verify that the central `GemmaServer` is operating correctly on `localhost:8000`.

## Scripts

- `test_text.py`: Tests the text classification endpoint (`/classify`).
- `test_image.py`: Tests the multimodal vision endpoint (`/ask_image`).
- `test_audio.py`: Tests the audio transcription endpoint (`/ask`).

## Usage

```bash
cd test_server
pip install -r requirements.txt

python test_text.py
python test_image.py
python test_audio.py
```
