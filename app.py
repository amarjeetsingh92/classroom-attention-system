import streamlit as st
import cv2, numpy as np, mediapipe as mp, tensorflow as tf, time, pandas as pd, os
from tensorflow.keras.models import load_model
from collections import deque

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="🎓 Classroom Attention Dashboard", layout="wide")
st.title("🎓 Classroom Attention Dashboard")

# Sidebar Inputs
st.sidebar.header("Session Details")
class_name = st.sidebar.text_input("Class Name")
subject_name = st.sidebar.text_input("Subject")
run = st.sidebar.checkbox("Start Monitoring")

# -------------------- Load Model & Setup --------------------
@st.cache_resource
def load_gaze_model():
    model_path = "gaze_cnn_64x64_gray_v2.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    return load_model(model_path)

model = load_gaze_model()
if model is None:
    st.stop()

mp_face_mesh = mp.solutions.face_mesh
drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)

FRAME_WINDOW = st.image([])
chart_placeholder = st.empty()
status_text = st.empty()

# Initialize attention data
times = deque(maxlen=30)
attention_scores = deque(maxlen=30)
start_time = time.time()

# -------------------- Helper Functions --------------------
def get_eye_region_from_landmarks(frame, landmarks, idx_list):
    """Return normalized 64x64 grayscale eye image or None if invalid."""
    h, w, _ = frame.shape
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
    pts = np.array(pts, dtype=np.int32)
    if pts.size == 0:
        return None
    x, y, w1, h1 = cv2.boundingRect(pts)
    # add small margin
    pad = int(0.2 * max(w1, h1))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + w1 + pad)
    y2 = min(h, y + h1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    eye = frame[y1:y2, x1:x2]
    if eye.size == 0:
        return None
    try:
        eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    eye_resized = cv2.resize(eye_gray, (64, 64))
    eye_norm = eye_resized.astype(np.float32) / 255.0
    # shape (1,64,64,1)
    return np.expand_dims(eye_norm, axis=(0, -1))

def predict_gaze_from_eye(eye_tensor):
    """Return one of ['left','right','up','down','front'] or 'front' on failure."""
    classes = ['left', 'right', 'up', 'down', 'front']
    try:
        pred = model.predict(eye_tensor, verbose=0)[0]
        label = classes[int(np.argmax(pred))]
        return label
    except Exception:
        return "front"

def draw_gaze_pointer(frame, start, end, color=(0,255,0)):
    cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)

# -------------------- Main Loop --------------------
if run:
    cap = cv2.VideoCapture(0)
    st.sidebar.success("Camera Active")

    # Landmarks indices for eyes (same as used in your other script)
    LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 246]
    RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 466]

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Camera not found!")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape
        attentive_count = 0
        total_faces = 0

        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                total_faces += 1

                # draw mesh
                drawing.draw_landmarks(frame, lm, mp_face_mesh.FACEMESH_CONTOURS)

                # get eye crops
                left_eye_tensor = get_eye_region_from_landmarks(frame, lm.landmark, LEFT_EYE_IDX)
                right_eye_tensor = get_eye_region_from_landmarks(frame, lm.landmark, RIGHT_EYE_IDX)

                left_gaze = predict_gaze_from_eye(left_eye_tensor) if left_eye_tensor is not None else "front"
                right_gaze = predict_gaze_from_eye(right_eye_tensor) if right_eye_tensor is not None else "front"

                gaze = left_gaze if left_gaze == right_gaze else "front"

                # Attention classification
                attention_label = "Attentive" if gaze == "front" else "Distracted"
                if attention_label == "Attentive":
                    attentive_count += 1

                # draw bounding box and label
                xs = [int(x.x * w) for x in lm.landmark]
                ys = [int(x.y * h) for x in lm.landmark]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                color = (0, 255, 0) if attention_label == "Attentive" else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, attention_label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # optional: draw small text of gaze for debugging
                cv2.putText(frame, f"{left_gaze}/{right_gaze}", (x_min, y_max + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # compute attention level percent
        attention_level = int((attentive_count / total_faces) * 100) if total_faces > 0 else 0

        # show frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Update chart
        times.append(round(time.time() - start_time, 1))
        attention_scores.append(attention_level)
        df = pd.DataFrame({"Time (s)": list(times), "Attention": list(attention_scores)})
        chart_placeholder.line_chart(df.set_index("Time (s)"))

        status_text.info(f"📊 Attention Level: {attention_level}%")

    cap.release()
    st.sidebar.warning("Monitoring stopped.")
else:
    st.info("👈 Enter Class & Subject, then click 'Start Monitoring' to begin.")
