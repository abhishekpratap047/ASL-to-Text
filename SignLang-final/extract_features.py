import cv2
import mediapipe as mp
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------
VIDEO_DIR = "clips"         # Folder containing class subfolders with videos
OUTPUT_DIR = "clips_npy"    # Folder to save extracted .npy files

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# MediaPipe Setup
# -------------------------------
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    """
    Extract pose, left hand, and right hand keypoints from MediaPipe results.
    Missing landmarks are replaced with zeros.
    Returns a 1D array of shape (1662,)
    """
    # Pose (33 landmarks * 4 values: x, y, z, visibility)
    pose = np.zeros(33*4)
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in pose_landmarks]).flatten()

    # Left hand (21 landmarks * 3 values: x, y, z)
    lh = np.zeros(21*3)
    if results.left_hand_landmarks:
        lh_landmarks = results.left_hand_landmarks.landmark
        lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in lh_landmarks]).flatten()

    # Right hand (21 landmarks * 3 values: x, y, z)
    rh = np.zeros(21*3)
    if results.right_hand_landmarks:
        rh_landmarks = results.right_hand_landmarks.landmark
        rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in rh_landmarks]).flatten()

    # Concatenate all features
    return np.concatenate([pose, lh, rh])

# -------------------------------
# Process videos
# -------------------------------
with mp_holistic.Holistic(static_image_mode=False,
                          model_complexity=2,
                          enable_segmentation=False,
                          refine_face_landmarks=False) as holistic:

    for class_name in os.listdir(VIDEO_DIR):
        class_path = os.path.join(VIDEO_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        save_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            cap = cv2.VideoCapture(video_path)
            frames_features = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Process frame
                results = holistic.process(frame_rgb)
                # Extract keypoints
                keypoints = extract_keypoints(results)
                frames_features.append(keypoints)

            cap.release()
            frames_features = np.array(frames_features)  # shape: (num_frames, 1662)

            # Save as .npy
            base_name = os.path.splitext(video_name)[0]
            save_path = os.path.join(save_class_dir, base_name + ".npy")
            np.save(save_path, frames_features)
            print(f"Saved: {class_name}/{base_name}.npy | Shape: {frames_features.shape}")

print("All videos processed and features saved successfully!")
