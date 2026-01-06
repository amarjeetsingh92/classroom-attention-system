# app2.py
# Integrated Streamlit app:
# - Emotion detection via Hugging Face transformers pipeline
# - ViT-based gaze regressor (timm backbone + small head) with optional checkpoint loading
# - Mediapipe face mesh for face detection & landmarks
# - Attention scoring & CSV logging

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
from collections import deque
import os
import csv
from datetime import datetime

# ML imports
import torch
import torch.nn as nn
import timm
from transformers import pipeline
from PIL import Image
import torchvision.transforms as T

# -------------------- UI / Page config --------------------
st.set_page_config(page_title="App2 — Gaze + Emotion (ViT)", layout="wide")
st.title("ViT Gaze + HF Emotion Classroom Monitor")

# -------------------- Sidebar: session settings --------------------
st.sidebar.header("Session & Model Settings")
class_name = st.sidebar.text_input("Class Name", value="Class A")
subject_name = st.sidebar.text_input("Subject", value="Math")
start_monitor = st.sidebar.checkbox("Start Monitoring", value=False)

# Gaze model settings
st.sidebar.subheader("Gaze model (ViT regressor)")
vit_model_name = st.sidebar.selectbox(
    "Backbone (timm model)", 
    options=["vit_base_patch16_224", "vit_small_patch16_224"], 
    index=0
)
gaze_checkpoint = st.sidebar.file_uploader("Optional: Upload gaze checkpoint (.pt)", type=["pt"])
yaw_threshold = st.sidebar.slider("Yaw threshold (degrees)", min_value=5.0, max_value=30.0, value=15.0)
pitch_threshold = st.sidebar.slider("Pitch threshold (degrees)", min_value=5.0, max_value=30.0, value=12.0)

# Emotion model settings
st.sidebar.subheader("Emotion model (HF)")
hf_emotion_model_name = st.sidebar.text_input("HF model name", value="trpakov/vit-face-expression")
load_hf_button = st.sidebar.button("(Re)load HF emotion model")

# Misc
max_faces = st.sidebar.slider("Max faces to track", min_value=1, max_value=6, value=4)
log_every = st.sidebar.number_input("Log interval (seconds)", min_value=30, max_value=600, value=60)

CSV_PATH = "app2_attention_log.csv"

# -------------------- Mediapipe setup --------------------
mp_face_mesh = mp.solutions.face_mesh
drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=max_faces,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------------------- UI placeholders --------------------
FRAME_WINDOW = st.image([])
chart_placeholder = st.empty()
status_text = st.empty()

# -------------------- Streaming data --------------------
times = deque(maxlen=600)
attention_scores = deque(maxlen=600)
start_time = time.time()

# -------------------- Helper functions: geometry --------------------
def get_avg_point(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices if i < len(landmarks)]
    if not pts:
        return None
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    return x, y

def get_gaze_direction_from_angles(pitch_deg, yaw_deg, yaw_thresh=15.0, pitch_thresh=12.0):
    """Map continuous angles to discrete labels."""
    if yaw_deg < -yaw_thresh:
        return "right"
    if yaw_deg > yaw_thresh:
        return "left"
    if pitch_deg < -pitch_thresh:
        return "up"
    if pitch_deg > pitch_thresh:
        return "down"
    return "front"

def get_gaze_direction(landmarks, w, h):
    """
    Simple fallback gaze detection from iris position (used only if gaze model is not available).
    Return one of 'left','right','up','down','front'.
    """
    try:
        LEFT_IRIS = [468, 469, 470, 471]
        RIGHT_IRIS = [473, 474, 475, 476]
        LEFT_EYE_L = 33
        LEFT_EYE_R = 133
        RIGHT_EYE_L = 362
        RIGHT_EYE_R = 263
        LEFT_TOP = [159, 160]
        LEFT_BOTTOM = [145, 144]
        RIGHT_TOP = [386, 387]
        RIGHT_BOTTOM = [374, 380]
        left_iris = get_avg_point(landmarks, LEFT_IRIS, w, h)
        right_iris = get_avg_point(landmarks, RIGHT_IRIS, w, h)
        if left_iris is None or right_iris is None:
            return "front"

        left_eye_l = (landmarks[LEFT_EYE_L].x * w, landmarks[LEFT_EYE_L].y * h)
        left_eye_r = (landmarks[LEFT_EYE_R].x * w, landmarks[LEFT_EYE_R].y * h)
        right_eye_l = (landmarks[RIGHT_EYE_L].x * w, landmarks[RIGHT_EYE_L].y * h)
        right_eye_r = (landmarks[RIGHT_EYE_R].x * w, landmarks[RIGHT_EYE_R].y * h)

        left_width = (left_eye_r[0] - left_eye_l[0]) or 1e-6
        right_width = (right_eye_r[0] - right_eye_l[0]) or 1e-6

        left_ratio = (left_iris[0] - left_eye_l[0]) / left_width
        right_ratio = (right_iris[0] - right_eye_l[0]) / right_width
        iris_ratio = (left_ratio + right_ratio) / 2.0

        left_top = get_avg_point(landmarks, LEFT_TOP, w, h)
        left_bottom = get_avg_point(landmarks, LEFT_BOTTOM, w, h)
        right_top = get_avg_point(landmarks, RIGHT_TOP, w, h)
        right_bottom = get_avg_point(landmarks, RIGHT_BOTTOM, w, h)

        if None in (left_top, left_bottom, right_top, right_bottom):
            vert_ratio = 0.5
        else:
            left_h = (left_bottom[1] - left_top[1]) or 1e-6
            right_h = (right_bottom[1] - right_top[1]) or 1e-6
            left_v = (left_iris[1] - left_top[1]) / left_h
            right_v = (right_iris[1] - right_top[1]) / right_h
            vert_ratio = (left_v + right_v) / 2.0

        if iris_ratio < 0.30:
            return "right"
        elif iris_ratio > 0.70:
            return "left"
        elif vert_ratio < 0.35:
            return "up"
        elif vert_ratio > 0.75:
            return "down"
        else:
            return "front"
    except Exception:
        return "front"

# -------------------- HF Emotion model --------------------
@st.cache_resource
def load_hf_emotion_model(model_name: str):
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("image-classification", model=model_name, device=device)
        return pipe
    except Exception as e:
        st.warning(f"HF model load failed: {e}")
        return None

# Load HF model on startup (or when button pressed)
if 'hf_pipeline' not in st.session_state:
    st.session_state.hf_pipeline = load_hf_emotion_model(hf_emotion_model_name)

if load_hf_button:
    st.session_state.hf_pipeline = load_hf_emotion_model(hf_emotion_model_name)

def normalize_label(raw_label: str) -> str:
    if not raw_label:
        return "thinking"
    label = raw_label.lower().strip()
    mapping = {
        "happiness": "happy", "happy": "happy",
        "surprise": "thinking", "surprised": "thinking",
        "sadness": "thinking", "sad": "thinking",
        "neutral": "thinking", "neutrality": "thinking",
        "angry": "angry", "anger": "angry",
        "disgust": "thinking", "fear": "thinking",
    }
    token = "".join([c for c in label if c.isalpha() or c.isspace()]).split()
    base = token[-1] if token else label
    return mapping.get(base, "thinking")

def predict_emotion_from_face(face_rgb):
    pipe = st.session_state.get('hf_pipeline', None)
    if pipe is None:
        return "thinking"
    try:
        pil = Image.fromarray(face_rgb.astype("uint8"), "RGB")
        out = pipe(pil, top_k=1)
        raw = out[0].get('label', '') if isinstance(out, list) and out else ''
        return normalize_label(raw)
    except Exception:
        return "thinking"

# -------------------- ViT gaze regressor --------------------
class ViTGazeRegressor(nn.Module):
    def __init__(self, vit_name='vit_base_patch16_224', pretrained=True, hidden_dim=256, dropout=0.2):
        super().__init__()
        # timm model returning features (no classifier)
        self.backbone = timm.create_model(vit_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # outputs: [pitch_deg, yaw_deg]
        )

    def forward(self, x):
        feat = self.backbone(x)  # (B, feat_dim)
        out = self.head(feat)
        return out

@st.cache_resource
def load_gaze_model(vit_name: str, checkpoint_bytes=None):
    """
    Returns a (model, device, transform) tuple.
    If checkpoint_bytes provided (uploaded .pt file), loads state_dict from it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTGazeRegressor(vit_name=vit_name, pretrained=True).to(device)
    if checkpoint_bytes is not None:
        try:
            # checkpoint_bytes is an UploadedFile (Streaming)
            state = torch.load(checkpoint_bytes, map_location=device)
            # support either state_dict or whole model saved
            if isinstance(state, dict) and any(k.startswith('head') or k.startswith('backbone') for k in state.keys()):
                model.load_state_dict(state, strict=False)
            elif 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'], strict=False)
            else:
                # try strict load if possible
                model.load_state_dict(state, strict=False)
            st.success("Gaze checkpoint loaded.")
        except Exception as e:
            st.warning(f"Failed loading gaze checkpoint: {e}")
    model.eval()
    # transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return model, device, transform

# initialize gaze model
if 'gaze_bundle' not in st.session_state:
    cp_bytes = None
    if gaze_checkpoint is not None:
        cp_bytes = gaze_checkpoint
    st.session_state.gaze_bundle = load_gaze_model(vit_model_name, cp_bytes)

# If user uploads a checkpoint at runtime, reload model
if gaze_checkpoint is not None:
    st.session_state.gaze_bundle = load_gaze_model(vit_model_name, gaze_checkpoint)

def predict_gaze_angles_batch(model, device, transform, faces_rgb_list):
    """
    faces_rgb_list: list of numpy RGB arrays (H,W,3 uint8).
    returns list of (pitch_deg, yaw_deg) tuples.
    """
    if not faces_rgb_list:
        return []
    tensors = []
    for f in faces_rgb_list:
        pil = Image.fromarray(f.astype("uint8"), "RGB")
        tensors.append(transform(pil))
    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy()  # shape (B,2)
    # Assume model outputs degrees (or tune accordingly). If your training used radians, multiply by 180/pi.
    result = [(float(o[0]), float(o[1])) for o in out]
    return result

# -------------------- CSV logging --------------------
def append_to_csv(timestamp, elapsed_sec, attention_level, class_name, subject_name):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "elapsed_seconds", "attention_level", "class_name", "subject_name"])
        writer.writerow([timestamp, elapsed_sec, attention_level, class_name, subject_name])

# -------------------- Main loop --------------------
if start_monitor:
    cap = cv2.VideoCapture(0)
    st.sidebar.success("Camera active")
    PROCESS_EVERY = 1.0
    last_process_time = time.time()
    last_log_time = time.time()
    attention_level = 0
    last_faces_info = []

    # fetch bundle
    gaze_model, gaze_device, gaze_transform = st.session_state.gaze_bundle

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not available.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape

        current_time = time.time()
        should_process = (current_time - last_process_time) >= PROCESS_EVERY

        # draw landmarks if present
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                drawing.draw_landmarks(frame, lm, mp_face_mesh.FACEMESH_CONTOURS)

        if should_process:
            attentive_count = 0
            total_faces = 0
            last_faces_info = []

            faces_rgb_for_model = []
            face_bbox_list = []
            face_landmarks_list = []

            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    xs = [int(p.x * w) for p in lm.landmark]
                    ys = [int(p.y * h) for p in lm.landmark]
                    x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
                    y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    # expand bbox a bit
                    pad_x = int(0.15 * (x_max - x_min))
                    pad_y = int(0.25 * (y_max - y_min))
                    x0 = max(0, x_min - pad_x)
                    x1 = min(w - 1, x_max + pad_x)
                    y0 = max(0, y_min - pad_y)
                    y1 = min(h - 1, y_max + pad_y)

                    face_crop = frame[y0:y1, x0:x1]
                    if face_crop.size == 0:
                        continue
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    faces_rgb_for_model.append(face_rgb)
                    face_bbox_list.append((x0, y0, x1, y1))
                    face_landmarks_list.append(lm)
                    total_faces += 1

            # Gaze inference (batch) if model present
            gaze_angles = []
            try:
                if faces_rgb_for_model and gaze_model is not None:
                    gaze_angles = predict_gaze_angles_batch(gaze_model, gaze_device, gaze_transform, faces_rgb_for_model)
                else:
                    gaze_angles = [(0.0, 0.0)] * len(faces_rgb_for_model)
            except Exception as e:
                # fallback: empty or zeros
                gaze_angles = [(0.0, 0.0)] * len(faces_rgb_for_model)
                st.sidebar.warning(f"Gaze inference error: {e}")

            # Emotion inference (per-face)
            emotions = []
            for f_rgb in faces_rgb_for_model:
                try:
                    emo = predict_emotion_from_face(f_rgb)
                except Exception:
                    emo = "thinking"
                emotions.append(emo)

            # Decide attentiveness per face using rules
            for idx in range(len(faces_rgb_for_model)):
                bbox = face_bbox_list[idx]
                lm = face_landmarks_list[idx]
                emotion_label = emotions[idx]
                pitch_deg, yaw_deg = gaze_angles[idx]
                # map to discrete gaze
                gaze_label = get_gaze_direction_from_angles(pitch_deg, yaw_deg, yaw_thresh=yaw_threshold, pitch_thresh=pitch_threshold)
                # fallback refine using landmarks-only method if needed
                if gaze_label == "front" and lm is not None:
                    # check quick landmark-based
                    fallback_gaze = get_gaze_direction(lm.landmark, w, h)
                    gaze_label = fallback_gaze

                # rules: gaze front/up OR emotion != thinking -> attentive
                gaze_attentive = gaze_label in ["front", "up"]
                if emotion_label == "happy":
                    emotion_attentive = False
                elif emotion_label == "angry":
                    emotion_attentive = False
                else:
                    # thinking/neutral -> attentive only if gaze front/up
                    emotion_attentive = gaze_label not in ["left", "right"]

                if gaze_attentive or emotion_attentive:
                    attention_label = "Attentive"
                else:
                    attention_label = "Distracted"

                if attention_label == "Attentive":
                    attentive_count += 1

                color = (0,255,0) if attention_label=="Attentive" else (0,0,255)
                last_faces_info.append({
                    "bbox": bbox,
                    "attention_label": attention_label,
                    "emotion_label": emotion_label,
                    "gaze": gaze_label,
                    "pitch_yaw": (pitch_deg, yaw_deg),
                    "color": color
                })

            attention_level = int((attentive_count / total_faces) * 100) if total_faces else 0
            elapsed_sec = int(current_time - start_time)
            times.append(elapsed_sec)
            attention_scores.append(attention_level)
            df = pd.DataFrame({"Time (s)": list(times), "Attention": list(attention_scores)})
            chart_placeholder.line_chart(df.set_index("Time (s)"))

            last_process_time = current_time

            # logging
            if (current_time - last_log_time) >= log_every:
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_to_csv(timestamp_str, elapsed_sec, attention_level, class_name, subject_name)
                last_log_time = current_time

        # Draw boxes and labels on frame (every frame)
        for info in last_faces_info:
            x0,y0,x1,y1 = info["bbox"]
            cv2.rectangle(frame, (x0,y0), (x1,y1), info["color"], 2)
            lbl = f"{info['attention_label']} | {info['emotion_label']} | gaze:{info['gaze']}"
            cv2.putText(frame, lbl, (x0, max(0,y0-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info["color"], 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        distracted = 100 - attention_level
        status = "ACCEPTED ✅" if distracted <= 60 else "ALERT ❗"
        status_text.info(
            f"📊 Attention: {attention_level}% | Distracted: {distracted}% | Status: {status} | Class: {class_name} | Subject: {subject_name}"
        )

        # break condition: user toggles off
        if not start_monitor:
            break

    cap.release()
else:
    st.info("Enter Class & Subject, then toggle 'Start Monitoring' to begin.")
