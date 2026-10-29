# Semantic Anomaly Detection & Cross-Camera Forensic Tracking System

## 1. Problem Statement
Traditional public safety monitoring suffers from **alarm fatigue** due to passive, high-noise surveillance that lacks semantic context, often failing to distinguish between benign activities and actual emergencies. Current systems face two critical failures:
- **Cloud-dependent processing** — creates latency and privacy risk.
- **Black-box pattern matching** — lacks a reasoning layer for reliable, forensic-grade verification.

---

## 2. Solution Summary
A multi-stage, compute-gated pipeline that filters noise cheaply (VAD), verifies semantically on-device (fine-tuned Gemma), and only then spends compute on person localization and cross-camera identity tracking (YOLO + Re-ID). Every stage is logged into an auditable trail for forensic-grade accountability — all without sending raw footage to the cloud.

---

## 3. Core Design Principle
> **Gate expensive computation behind cheap confirmation.**

Every stage only runs if the previous stage says "worth looking at." This is what keeps the system low-latency, on-device, and scalable across camera clusters.

---

## 4. System Architecture

```text
Camera 1 ─┐
Camera 2 ─┼─► [Stage 1: VAD]  (always-on, per stream)
Camera N ─┘         │
                anomaly_score > threshold?
                     │
              ┌──────┴──────┐
              NO             YES
              │              │
           discard   [Stage 2: Fine-tuned Gemma — on-device]
                             │
                    confirmed real anomaly? + person_involved?
                             │
                      ┌──────┴──────┐
                      NO             YES
                      │              │
                   discard   [Stage 3: YOLOv8n — person detect + box]
                                     │
                             [Stage 4: OSNet Re-ID — crop → embedding]
                                     │
                             [Stage 5: Embedding store + cosine matcher]
                                     │
                          matched across cameras?
                                     │
                          ┌──────────┴──────────┐
                        match                 no match
                          │                      │
                          └──────────┬───────────┘
                                     ▼
                     [Stage 6: Forensic log — hash-chained, append-only]
                                     │
                                     ▼
                       [Stage 7: Alert Dashboard — human-in-the-loop]
```

---

## 5. Running the Demo

To launch the end-to-end pipeline and access the live Alert Dashboard:

### Step 1: Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
`ultralytics` and `torchreid` are only needed for Stages 3–5; without them the dashboard marks those stages as unavailable instead of faking results.

### Step 2: Download Model Weights
1. Download the pre-trained Video Anomaly Detection (VAD) model weights from this SharePoint link:  
   [**Download UCF-Crime Model Weights (model_ucf.pth)**](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Eaz6sn40RmlFmjELcNHW1IkBV7C0U5OrOaHcuLFzH2S0-Q?e=x8wtVe)
2. Place the downloaded weight file (`model_ucf.pth`) into the `VadCLIP/model/` directory. (Create the folder if it doesn't exist).
3. *(Optional, for meaningful Re-ID)* Download an OSNet Re-ID checkpoint (e.g. `osnet_x1_0` trained on MSMT17 or Market-1501) from the [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO) and point `OSNET_WEIGHTS` at it. Without it OSNet falls back to ImageNet weights, which are not trained for person matching.

### Step 3: Configure
All settings are environment variables. None are hardcoded, because ngrok URLs change on every restart.

| Variable | Default | Purpose |
|---|---|---|
| `GEMMA_SERVER_URL` | *(required for Stage 2)* | Current ngrok URL of the Kaggle VM serving Gemma |
| `VAD_THRESHOLD` | `0.010` | VadCLIP anomaly-score threshold. Re-tune when you change `VAD_CROP` |
| `VAD_CROP` | `official` | `official` = VadCLIP `src/crop.py` center crop; `clip` = the old CLIP transform |
| `GEMMA_MIN_WINDOW_SEC` | `6.0` | Minimum clip length shown to Gemma (short VAD events give 9 near-identical frames otherwise) |
| `YOLO_IMGSZ` / `YOLO_CONF` | `640` / `0.4` | YOLO input size and person confidence. `1280` catches small CCTV figures |
| `OSNET_WEIGHTS` | *(unset)* | OSNet Re-ID checkpoint path |
| `REID_THRESHOLD` | `0.70` | Cosine similarity for a cross-camera match |
| `MAX_UPLOAD_MB` / `PORT` | `100` / `8080` | Upload limit and server port |

### Step 4: Boot the System
Open your terminal, navigate to the project directory, and execute the main pipeline runner:
```bash
cd VadCLIP
python main.py          # or, from the project root: python server.py
```
This starts the backend (VAD, Gemma client, YOLOv8n, OSNet Re-ID, forensic log) and opens the **Frontend Dashboard** at `http://localhost:8080`.
You can also access the **Police Dispatch Dashboard** by navigating to `http://localhost:8080/police/` in your browser. Alerts reach it only after an operator clicks **Confirm & Dispatch** in Stage 7. Open both pages in the same browser, because they talk over `BroadcastChannel`.

To try cross-camera matching, run the pipeline on two clips with different **Camera ID** values.

### API
| Endpoint | Stage | Notes |
|---|---|---|
| `POST /api/upload` | — | multipart `video`, optional `camera_id`; records the file's SHA-256 |
| `POST /api/pipeline/vad` | 1 | VadCLIP scores, events, collages |
| `POST /api/pipeline/gemma` | 2 | One event per call (`event_index`); returns parsed JSON + the exact grid sent |
| `POST /api/pipeline/track` | 3–5 | Only for Gemma-confirmed, person-involved events; `503` if YOLO/torchreid are missing |
| `POST /api/pipeline/log` | 6 | Appends a SHA-256 hash-chained record built from server-side state |
| `POST /api/forensic/decision` | 7 | Operator confirm / reject / override, chained to the event record |
| `GET /api/forensic/verify` | 6 | Re-walks the chain and reports the first tampered entry |
| `GET /api/forensic/log` | 6 | Most recent entries |

---

## 6. Stage-by-Stage Specification

### Stage 1 — Video Anomaly Detection (VAD)
- **Model:** VadCLIP (CLIP ViT-B/16 features, pretrained on UCF-Crime)
- **Input:** one CLIP feature per 16-frame snippet (currently a whole uploaded clip, not a live stream)
- **Output:** anomaly score (0–1)
- **Runs:** continuously, on every stream
- **Why:** cheapest possible filter — kills ~90%+ of normal footage before anything expensive runs

### Stage 2 — Semantic Confirmation (Fine-tuned Gemma, on-device)
- **Input:** flagged clip/keyframes + anomaly score
- **Output:** confirm/reject, natural-language description, `person_involved: yes/no`, confidence
- **Why on-device:** directly resolves the latency/privacy failure — no raw footage leaves the edge device.
- **Why this stage matters:** provides the essential "reasoning layer" missing from black-box systems.

### Stage 3 — Person Localization (YOLOv8n)
- **Input:** confirmed clip frames (only if `person_involved: yes`)
- **Output:** bounding boxes, class = person
- **Gated:** never runs on non-person anomalies (fire, crowd surge, etc.) to save compute.

### Stage 4 — Re-Identification (OSNet)
- **Input:** YOLO-cropped person image
- **Output:** 512-dim embedding vector
- **Latency:** ~5–10ms/crop (GPU)

### Stage 5 — Cross-Camera Matching
- **Store:** in-memory dict or Redis `{camera_id, timestamp, embedding, anomaly_type, confidence}`
- **Matcher:** cosine similarity, threshold ~0.7
- **Output:** same-person-across-cameras flag, or new-individual flag

### Stage 6 — Forensic Logging
- **Format:** append-only, hash-chained JSON log
- **Contents per event:** VAD score → Gemma reasoning text → YOLO box coords → embedding match confidence → final decision. This ensures a reconstructable, tamper-evident reasoning trail for every alert.

### Stage 7 — Alert Dashboard
- **Severity score:** VAD score × Gemma confidence.
- **Explainability panel:** Live display of Gemma's reasoning text.
- **Comparison mode:** Side-by-side "VAD-only" vs "full pipeline" to visually demonstrate the reduction in alarm fatigue.

### Stage 8 — Police Dispatch Dashboard
- **Access:** Available via the `/police/` endpoint.
- **Forensic Intel:** Displays real-time threat severity, location/sector, camera feeds, VAD score, and Gemma confidence.
- **Department Routing:** Automatically highlights the intended department (e.g., Police, Fire Dept, Hospital) based on Gemma's structured JSON analysis.

---

## 7. Key Differentiators

| Traditional Failure | How this System Resolves It |
|---|---|
| **Cloud latency/privacy risk** | Fine-tuned Gemma runs on-device; no raw footage leaves the edge |
| **Black-box, no reasoning** | Semantic confirmation + natural-language explanation at every alert |
| **No forensic verification** | Hash-chained audit log, every stage's output is traceable |
| **Alarm fatigue** | Compute-gated pipeline; VAD filters ~90%+ before tracking runs |
| **No cross-camera correlation**| OSNet Re-ID + cosine matching links the same person across the cluster |

---

## 8. Known Limitations
- **No public large-scale Indian VAD dataset exists:** The system is currently trained on UCF-Crime; the architecture is designed for future fine-tuning once a localized benchmark becomes available.
- **Re-ID accuracy real-world variance:** Benchmark accuracy (94%) drops in real-world conditions (~75–85%). Mitigated by requiring VAD+Gemma confirmation before Re-ID operates.
- **Person-centric anomaly tracking:** Non-person anomalies (e.g., fires, left objects) are designed to skip Stages 3–5.
- **Gemma is not on-device yet:** Stage 2 sends a 3×3 frame grid to a Kaggle VM over ngrok (`gemma/test_kaggle_vm.py`), and the CLI path (`VadCLIP/pipeline.py`) calls the Gemini API. Both send frames off the device. Running the fine-tuned model locally is what makes the privacy claim true.
- **VAD calibration on custom footage:** On the bundled demo clip, VadCLIP's peak score is 0.025. The only snippets above the 0.010 threshold are the first and last second of the clip, a smooth edge-decay pattern, while the theft itself scores about 0.0001–0.0005 (`VadCLIP/reports/incident_report.json`). Verify feature extraction against the official UCF-Crime features (see `VadCLIP/extract_features.py`) and calibrate the threshold on normal footage before relying on Stage 1.
- **Re-ID needs Re-ID weights:** torchreid's `pretrained=True` loads ImageNet weights. Set `OSNET_WEIGHTS` to a Re-ID checkpoint, or the similarity scores in Stage 5 are not meaningful.
- **Gallery and sessions are in memory:** They reset when the server restarts. The forensic log (`logs/forensic_log.jsonl`) persists.
