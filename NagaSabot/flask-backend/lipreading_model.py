import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import requests

# Use gdown for Google Drive direct download
try:
    import gdown
except ImportError:
    gdown = None

# MODEL_PATH = "nagsabot_full_model_onlytwophrase.keras"
# MODEL_URL = "https://drive.google.com/uc?id=1Ubje08hb0aKAZLU5xhw3QTUv5UcUVcpA"  # Direct download link for Google Drive file

def download_model():
    # if not os.path.exists(MODEL_PATH):
    print("Model file not found, downloading...")
    if gdown:
        # gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("gdown not installed. Please install it with 'pip install gdown'.")
        raise ImportError("gdown is required to download the model from Google Drive.")
    print("Model downloaded.")
    # else:
    #     print("Model file already exists, skipping download.")

# Constants for the lip reading model
NUM_FRAMES = 30  # Changed from 75 to 30 frames per sample
HEIGHT = 80       # Height of the lip patch
WIDTH = 112       # Width of the lip patch
CHANNELS = 3      # RGB channels

# Bikol-Naga phrases (must match exactly the labels used for training)
BIKOL_NAGA_PHRASES = [
    "marhay na aldaw",
    "dios mabalos",
    "nagkakan ka na",
    "pasain ka",
    "maray man ako",
    "maduman na ako",
    "nuarin kita mahali",
    "pahagad man ako",
    "tano",
    "iyo na ito",
    "dae man giraray",
    "bako man",
    "sakuya an",
    "magkarigos na ika",
    "madya mabalyo na kita"
]

def is_mouth_open(face_landmarks, threshold=0.018):
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    return abs(upper_lip.y - lower_lip.y) > threshold

class LipReadingModel:
    def __init__(self):
        # Load the model from the specified file
        self.model = tf.keras.models.load_model('nagsabot_full_model_onlytwophrase.keras')
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def preprocess_image(self, img):
        try:
            # Convert to LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
            l_channel_eq = clahe.apply(l_channel)
            # Merge back and convert to RGB
            img_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            img_eq = cv2.cvtColor(img_eq, cv2.COLOR_LAB2RGB)
            return img_eq
        except Exception as e:
            print(f"Error in preprocess_image: {e}")
            return img

    def extract_lip_region(self, frame, face_landmarks):
        img_h, img_w = frame.shape[:2]
        try:
            # Get lip corners
            lip_left = int(face_landmarks.landmark[61].x * img_w)
            lip_right = int(face_landmarks.landmark[291].x * img_w)
            # Get lip top and bottom
            lip_top = int(face_landmarks.landmark[0].y * img_h)
            lip_bottom = int(face_landmarks.landmark[17].y * img_h)
            # Calculate width and height of lip region
            width_diff = WIDTH - (lip_right - lip_left)
            height_diff = HEIGHT - (lip_bottom - lip_top)
            # Calculate padding
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top
            # Ensure padding doesn't extend beyond frame
            pad_left = min(pad_left, lip_left)
            pad_right = min(pad_right, img_w - lip_right)
            pad_top = min(pad_top, lip_top)
            pad_bottom = min(pad_bottom, img_h - lip_bottom)
            # Calculate coordinates with padding
            x1 = max(0, lip_left - pad_left)
            y1 = max(0, lip_top - pad_top)
            x2 = min(img_w, lip_right + pad_right)
            y2 = min(img_h, lip_bottom + pad_bottom)
            # Extract the lip region
            lip_frame = frame[y1:y2, x1:x2]
            if lip_frame.size == 0:
                return None
            # Resize to the standard dimensions
            lip_frame = cv2.resize(lip_frame, (WIDTH, HEIGHT))
            # Apply preprocessing
            lip_frame = self.preprocess_image(lip_frame)
            return lip_frame
        except Exception as e:
            print(f"Error extracting lip region: {e}")
            return None

    def predict(self, video_path):
        try:
            print(f"Starting video processing for: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video file")
                return {"phrase": "Error opening video", "accuracy": 0.0}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames in video: {total_frames}")
            
            frames_buffer = []
            open_mouth_count = 0
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}")
                
                # Flip the frame horizontally for selfie-view
                frame = cv2.flip(frame, 1)
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Process with MediaPipe
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    print(f"Face detected in frame {frame_count}")
                    face_landmarks = results.multi_face_landmarks[0]
                    # Check if mouth is open
                    if is_mouth_open(face_landmarks):
                        open_mouth_count += 1
                    lip_frame = self.extract_lip_region(frame, face_landmarks)
                    if lip_frame is not None:
                        frames_buffer.append(lip_frame)
                        print(f"Lip frame collected. Total lip frames: {len(frames_buffer)}/{NUM_FRAMES}")
                        if len(frames_buffer) >= NUM_FRAMES:
                            print("Required number of frames collected")
                            break
                else:
                    print(f"No face detected in frame {frame_count}")
            
            cap.release()
            print(f"Video processing complete. Collected {len(frames_buffer)} lip frames")
            
            if len(frames_buffer) < NUM_FRAMES:
                print(f"Not enough frames collected. Got {len(frames_buffer)}, need {NUM_FRAMES}")
                if len(frames_buffer) > 0:
                    # Pad with the last frame until we have exactly NUM_FRAMES
                    while len(frames_buffer) < NUM_FRAMES:
                        frames_buffer.append(frames_buffer[-1])
                    print(f"Padded frames to {NUM_FRAMES}")
                else:
                    return {"phrase": "Not enough frames", "accuracy": 0.0}

            # Check if enough frames have open mouth (talking)
            open_mouth_ratio = open_mouth_count / len(frames_buffer) if len(frames_buffer) > 0 else 0
            print(f"Open mouth frames: {open_mouth_count}/{len(frames_buffer)} (ratio: {open_mouth_ratio:.2f})")
            if open_mouth_ratio < 0.3:  # Less than 30% of frames have open mouth
                print("No speech detected (mouth mostly closed)")
                return {"phrase": "No speech detected", "accuracy": 0.0}

            # Convert to numpy array and normalize to [0,1]
            input_data = np.array(frames_buffer[:NUM_FRAMES], dtype=np.float32) / 255.0
            # Reshape to match model input shape
            input_data = input_data.reshape(1, NUM_FRAMES, HEIGHT, WIDTH, CHANNELS)
            print(f"Input data shape: {input_data.shape}")
            
            # Make prediction
            prediction = self.model.predict(input_data, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            print(f"Prediction made: {BIKOL_NAGA_PHRASES[predicted_class]} with confidence {confidence}")
            
            return {
                "phrase": BIKOL_NAGA_PHRASES[predicted_class],
                "accuracy": confidence
            }
        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()
            return {"phrase": "Error in prediction", "accuracy": 0.0} 