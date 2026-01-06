# Classroom Attention Monitoring System

A real-time multi-person classroom monitoring system using:
- Gaze estimation (CNN + Vision Transformer)
- Facial emotion recognition
- MediaPipe face mesh
- Streamlit dashboard

## Features
- Real-time attention percentage
- Multi-face tracking
- CNN-based gaze classification
- ViT-based gaze regression (pitch & yaw)
- Emotion-aware attention scoring
- CSV logging for analytics

## Tech Stack
- Python 3.10
- Streamlit
- MediaPipe
- TensorFlow / Keras
- PyTorch + timm
- HuggingFace Transformers

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app2.py
