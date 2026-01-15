import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "asl_lstm_model.h5"
LABELS_PATH = "label_classes.npy"

# -------------------------------
# Load model & labels
# -------------------------------
model = load_model(MODEL_PATH)
class_names = np.load(LABELS_PATH)

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extract pose, left hand, right hand keypoints as 1D array"""
    # Pose (33*4)
    pose = np.zeros(33*4)
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in pose_landmarks]).flatten()

    # Left hand (21*3)
    lh = np.zeros(21*3)
    if results.left_hand_landmarks:
        lh_landmarks = results.left_hand_landmarks.landmark
        lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lh_landmarks]).flatten()

    # Right hand (21*3)
    rh = np.zeros(21*3)
    if results.right_hand_landmarks:
        rh_landmarks = results.right_hand_landmarks.landmark
        rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in rh_landmarks]).flatten()

    return np.concatenate([pose, lh, rh])

# -------------------------------
# Real-time capture
# -------------------------------
SEQ_LENGTH = 60  # number of frames per gesture
sequence = []

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False) as holistic:

    predicted_word = ""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Keep sequence length
        if len(sequence) > SEQ_LENGTH:
            sequence.pop(0)

        # Predict when sequence is full
        if len(sequence) == SEQ_LENGTH:
            input_data = np.expand_dims(np.array(sequence), axis=0)  # shape: (1, 60, 1662)
            prediction = model.predict(input_data, verbose=0)
            predicted_word = class_names[np.argmax(prediction)]

        # Display predicted word
        cv2.putText(frame, f"Predicted: {predicted_word}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("ASL Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
