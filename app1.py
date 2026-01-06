import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
from collections import deque
from deepface import DeepFace
import os
import csv
from datetime import datetime

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="🎓 Classroom Attention Dashboard", layout="wide")
st.title("🎓 Classroom Attention Dashboard")

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("Session Details")
class_name = st.sidebar.text_input("Class Name")
subject_name = st.sidebar.text_input("Subject")
run = st.sidebar.checkbox("Start Monitoring")

# -------------------- MediaPipe Setup --------------------
mp_face_mesh = mp.solutions.face_mesh
drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FRAME_WINDOW = st.image([])
chart_placeholder = st.empty()
status_text = st.empty()

# -------------------- Attention Data --------------------
times = deque(maxlen=600)
attention_scores = deque(maxlen=600)
start_time = time.time()

CSV_PATH = "attention_log.csv"

# -------------------- Helper Functions --------------------
def get_avg_point(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    if not pts:
        return None
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    return x, y


def get_gaze_direction(landmarks, w, h):
    LEFT_IRIS = [468, 469, 470, 471]
    RIGHT_IRIS = [473, 474, 475, 476]

    LEFT_EYE_L, LEFT_EYE_R = 33, 133
    RIGHT_EYE_L, RIGHT_EYE_R = 362, 263

    try:
        left_iris = get_avg_point(landmarks, LEFT_IRIS, w, h)
        right_iris = get_avg_point(landmarks, RIGHT_IRIS, w, h)

        if left_iris is None or right_iris is None:
            return "front"

        left_eye_l = landmarks[LEFT_EYE_L].x * w
        left_eye_r = landmarks[LEFT_EYE_R].x * w
        right_eye_l = landmarks[RIGHT_EYE_L].x * w
        right_eye_r = landmarks[RIGHT_EYE_R].x * w

        left_ratio = (left_iris[0] - left_eye_l) / (left_eye_r - left_eye_l + 1e-6)
        right_ratio = (right_iris[0] - right_eye_l) / (right_eye_r - right_eye_l + 1e-6)
        iris_ratio = (left_ratio + right_ratio) / 2

        if iris_ratio < 0.35:
            return "right"
        elif iris_ratio > 0.65:
            return "left"
        else:
            return "front"
    except:
        return "front"


def predict_emotion(face_rgb):
    try:
        result = DeepFace.analyze(
            img_path=face_rgb,
            actions=["emotion"],
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        return result.get("dominant_emotion", "neutral")
    except:
        return "neutral"


def append_to_csv(timestamp, elapsed, attention, class_name, subject_name):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "elapsed_seconds",
                "attention_level",
                "class_name",
                "subject_name"
            ])
        writer.writerow([
            timestamp,
            elapsed,
            attention,
            class_name,
            subject_name
        ])

# -------------------- Main Loop --------------------
if run:
    cap = cv2.VideoCapture(0)
    st.sidebar.success("Camera Active")

    PROCESS_EVERY = 1.0
    LOG_EVERY = 60.0
    last_process_time = time.time()
    last_log_time = time.time()

    last_faces_info = []
    attention_level = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not available")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape
        now = time.time()

        if (now - last_process_time) >= PROCESS_EVERY:
            attentive = 0
            total = 0
            last_faces_info = []

            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    total += 1
                    gaze = get_gaze_direction(lm.landmark, w, h)

                    xs = [int(p.x * w) for p in lm.landmark]
                    ys = [int(p.y * h) for p in lm.landmark]
                    x1, x2 = max(min(xs), 0), min(max(xs), w)
                    y1, y2 = max(min(ys), 0), min(max(ys), h)

                    face_crop = frame[y1:y2, x1:x2]
                    emotion = predict_emotion(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)) if face_crop.size else "neutral"

                    attentive_flag = gaze == "front" and emotion not in ["angry", "sad"]
                    label = "Attentive" if attentive_flag else "Distracted"
                    if attentive_flag:
                        attentive += 1

                    color = (0, 255, 0) if attentive_flag else (0, 0, 255)
                    last_faces_info.append((x1, y1, x2, y2, label, emotion, gaze, color))

            attention_level = int((attentive / total) * 100) if total else 0
            elapsed = int(now - start_time)

            times.append(elapsed)
            attention_scores.append(attention_level)
            df = pd.DataFrame({"Time": list(times), "Attention": list(attention_scores)})
            chart_placeholder.line_chart(df.set_index("Time"))

            if (now - last_log_time) >= LOG_EVERY:
                append_to_csv(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    elapsed,
                    attention_level,
                    class_name,
                    subject_name
                )
                last_log_time = now

            last_process_time = now

        for x1, y1, x2, y2, label, emotion, gaze, color in last_faces_info:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} | {emotion} | gaze:{gaze}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status_text.info(f"Attention: {attention_level}%")

        if not run:
            break

    cap.release()
else:
    st.info("👈 Enter Class & Subject, then start monitoring.")
