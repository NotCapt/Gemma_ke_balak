# CrowdGuard AI - Public Safety Monitoring System

An intelligent crowd safety monitoring and emergency response suite powered by Google's Gemma AI (v2 Local Inference Architecture).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139+-green.svg)](https://fastapi.tiangolo.com/)

## 🎯 Overview

CrowdGuard AI is a comprehensive AI-powered safety suite designed for real-time monitoring and emergency response in crowded environments like festivals, public gatherings, and events. By running large language models locally, the system guarantees low-latency response times and strict data privacy. It combines computer vision, natural language processing, and voice recognition into one unified ecosystem.

## ✨ Key Features

### 🎥 Vision Server - Real-time Crowd Monitoring
- **Dual Analysis Engine**: Simultaneous crowd density and motion behavior detection.
- **Reasoning Engine**: Generates structured AI reasoning for detected risks in real time.
- **Live Dashboard**: Real-time analytics with visual safety indicators.
- **SQLite Audit Trail**: Robust local database logging for forensic auditing and historical analysis.

### 🎤 Voice Server - Hands-free Command Interface
- **Hindi-English Voice Commands**: Natural language processing optimized for Indian linguistic contexts.
- **Zone Status Queries**: Real-time security data retrieval via voice.
- **Text-to-Speech**: Hindi audio responses with automated Text-To-Speech integrations.

### 📱 Emergency Reporting - Citizen Interface
- **AI Classification**: Fine-tuned Gemma model for emergency categorization (Child Lost, Crowd Panic, Medical Help, etc.).
- **Mobile Interface**: Responsive web application for fast reporting.
- **Image Documentation**: Secure photo upload with emergency reports.
- **Automated Alerts**: Email notifications instantly dispatched to response teams.

### 🤖 Core Inference & Fine-tuning
- **100% Local Inference**: Eliminates third-party cloud dependencies; runs entirely on local GPUs via Unsloth.
- **LoRA Fine-tuning**: Parameter-efficient training pipeline for Gemma.
- **Synthetic Data**: Includes scripts to generate thousands of emergency classification samples.

## 🚀 Quick Start

### Prerequisites
- **Hardware**: CUDA-compatible GPU (e.g., RTX 4050, 4090).
- **Software**: Python 3.10+, CUDA drivers.

### 1. Environment Setup

Install the required dependencies into your system or virtual environment:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` files in their respective directories if required (e.g., for email alerts):

```bash
# Vision and Chat Servers
GOOGLE_APP_PASSWORD=your_gmail_password
```

### 3. Start Services

You can run these components independently. They all connect to the core Inference Server.

#### Core Inference Server (Port 8000)
*Must be running first to handle AI requests.*
```bash
cd GemmaServer
python gemma_server.py
```

#### Vision Monitoring Server (Port 38277)
```bash
cd Gemma_Kavach_Vision_Server
uvicorn main:app --host 0.0.0.0 --port 38277 --reload
```

#### Voice Command Server (Port 7860)
```bash
cd Gemma_Kavach_Voice_Server
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

#### Emergency Reporting Server (Port 8501)
```bash
cd User_Chat_Server
python main.py
```

## 📊 Risk Assessment Matrix

The Vision Server automatically categorizes threats using the following logic:

| Density | Motion | Risk Level |
|---------|--------|------------|
| High | Chaotic | CRITICAL |
| Medium | Chaotic | HIGH |
| Low | Chaotic | MODERATE |
| High | Calm | MODERATE |
| Medium/Low | Calm | SAFE |

## 🛠️ Technology Stack

- **AI Framework**: Google Gemma + Unsloth (LoRA Fine-tuning)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Computer Vision**: OpenCV, Pillow
- **Audio Processing**: Librosa, SoundFile
- **Data Persistence**: Local SQLite

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
