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

To launch the simulated end-to-end pipeline and access the live Alert Dashboard:

### Step 1: Download Model Weights
1. Download the pre-trained Video Anomaly Detection (VAD) model weights from this SharePoint link:  
   [**Download UCF-Crime Model Weights (model_ucf.pth)**](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Eaz6sn40RmlFmjELcNHW1IkBV7C0U5OrOaHcuLFzH2S0-Q?e=x8wtVe)
2. Place the downloaded weight file (`model_ucf.pth`) into the `VadCLIP/model/` directory. (Create the folder if it doesn't exist).

### Step 2: Boot the System
Open your terminal, navigate to the project directory, and execute the main pipeline runner:
```bash
cd VadCLIP
python main.py
```
This script will initialize the multi-stage pipeline, including the VAD, Gemma Multimodal Engine, YOLOv8n, and OSNet Re-ID embeddings. Once the backend boots up, it will automatically launch the **Frontend Dashboard** in your default web browser (running locally on port 8080).

---

## 6. Stage-by-Stage Specification

### Stage 1 — Video Anomaly Detection (VAD)
- **Model:** CLIP-TSA or RTFM (pretrained on UCF-Crime)
- **Input:** rolling clip window (16–32 frames) per camera stream
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
