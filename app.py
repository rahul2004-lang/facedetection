from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Model paths
gender_model_path = 'models/gender_model.h5'
age_model_path = 'models/age_model.h5'
emotion_model_path = 'models/emotion_model.h5'

# Check if model files exist
if not os.path.exists(gender_model_path):
    print(f"Gender model not found at {gender_model_path}")
if not os.path.exists(age_model_path):
    print(f"Age model not found at {age_model_path}")
if not os.path.exists(emotion_model_path):
    print(f"Emotion model not found at {emotion_model_path}")

# Load pre-trained models
gender_model = load_model(gender_model_path) if os.path.exists(gender_model_path) else None
age_model = load_model(age_model_path) if os.path.exists(age_model_path) else None
emotion_model = load_model(emotion_model_path) if os.path.exists(emotion_model_path) else None

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Camera handling
# ...existing code...

# Camera handling
camera = cv2.VideoCapture(0)

def detect_faces_and_hands():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe Hand Detection
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = sum([landmark.y < hand_landmarks.landmark[0].y for i, landmark in enumerate(hand_landmarks.landmark[4:21:4])])
                cv2.putText(frame, f"Fingers: {fingers}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces_and_hands(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close_camera')
def close_camera():
    camera.release()
    return "Camera Closed"

if __name__ == "__main__":
    app.run(debug=True)
