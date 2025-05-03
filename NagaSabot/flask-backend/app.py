from flask import Flask, request, jsonify, send_from_directory  # type: ignore
from flask_cors import CORS  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore
import os
from datetime import datetime
import uuid
import cv2
import math
import numpy as np
import mediapipe as mp
import tensorflow as tf
import logging
from typing import Tuple, Dict

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NagaSabot")

# Load model once globally
def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nagsabot_full_model_morecleaner4.keras')
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB max

# Directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Constants
TOTAL_FRAMES = 75
LIP_WIDTH = 112
LIP_HEIGHT = 80
CHANNELS = 3
NUM_FRAMES = 75
HEIGHT = 80
WIDTH = 112

mp_face_mesh = mp.solutions.face_mesh
LIP_OUTER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

BIKOL_NAGA_PHRASES = [
    "marhay na aldaw", "dios mabalos", "padaba taka", "tabi po", "iyo tabi"
]

def enhance_lip_region(lip_frame: np.ndarray) -> np.ndarray:
    try:
        if lip_frame is None or lip_frame.size == 0:
            return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)
        if len(lip_frame.shape) == 2 or lip_frame.shape[2] == 1:
            lip_frame = cv2.cvtColor(lip_frame, cv2.COLOR_GRAY2BGR)
        elif lip_frame.shape[2] != 3:
            return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)

        lip_frame_gray = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        lip_frame_eq = clahe.apply(lip_frame_gray)
        lip_frame_filtered = cv2.bilateralFilter(lip_frame_eq, 5, 35, 35)
        kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        lip_frame_sharp = cv2.filter2D(lip_frame_filtered, -1, kernel)
        lip_frame_final = cv2.GaussianBlur(lip_frame_sharp, (3, 3), 0)
        return cv2.cvtColor(lip_frame_final, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.error(f"Error in enhance_lip_region: {e}")
        return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)

def get_fixed_centered_lip_region(image: np.ndarray, face_landmarks) -> Tuple[np.ndarray, float, Tuple[int, int, int, int], float]:
    h, w, _ = image.shape
    try:
        all_lip_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                          for i in LIP_OUTER_INDICES + LIP_INNER_INDICES]
        all_x, all_y = zip(*all_lip_points)
        center_x, center_y = sum(all_x) / len(all_x), sum(all_y) / len(all_y)
        center_point = (int(center_x), int(center_y))
        lip_width_raw = max(all_x) - min(all_x)
        lip_height_raw = max(all_y) - min(all_y)

        padding_factor = 1.75
        lip_width_padded = int(lip_width_raw * padding_factor)
        lip_height_padded = int(lip_height_raw * padding_factor)
        target_aspect = 4 / 3
        current_aspect = lip_width_padded / lip_height_padded

        if current_aspect > target_aspect:
            lip_height_padded = int(lip_width_padded / target_aspect)
        else:
            lip_width_padded = int(lip_height_padded * target_aspect)

        left_eye = face_landmarks.landmark[LEFT_EYE_OUTER]
        right_eye = face_landmarks.landmark[RIGHT_EYE_OUTER]
        angle_deg = math.degrees(math.atan2((right_eye.y - left_eye.y) * h, (right_eye.x - left_eye.x) * w))
        rot_mat = cv2.getRotationMatrix2D(center_point, angle_deg, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

        x1 = max(0, int(center_point[0] - lip_width_padded // 2))
        x2 = min(w, int(center_point[0] + lip_width_padded // 2))
        y1 = max(0, int(center_point[1] - lip_height_padded // 2))
        y2 = min(h, int(center_point[1] + lip_height_padded // 2))

        lip_crop = rotated[y1:y2, x1:x2]
        if lip_crop.size > 0:
            resized = cv2.resize(lip_crop, (LIP_WIDTH, LIP_HEIGHT))
        else:
            resized = np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)
        return resized, 0, (x1, y1, x2, y2), angle_deg
    except Exception as e:
        logger.error(f"Error in get_fixed_centered_lip_region: {e}")
        return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8), 0, (0, 0, 0, 0), 0

def predict_lipreading(video_path: str) -> Dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                lip_region, *_ = get_fixed_centered_lip_region(frame, landmarks)
                enhanced = enhance_lip_region(lip_region)
                all_frames.append(enhanced)
    cap.release()

    if len(all_frames) >= TOTAL_FRAMES:
        indices = np.linspace(0, len(all_frames) - 1, TOTAL_FRAMES).astype(int)
        frames = [all_frames[i] for i in indices]
    else:
        frames = all_frames + [np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)] * (TOTAL_FRAMES - len(all_frames))

    X = np.expand_dims(np.array(frames, dtype=np.float32) / 255.0, axis=0)
    prediction = model.predict(X)[0]
    idx = int(np.argmax(prediction))
    phrase = BIKOL_NAGA_PHRASES[idx] if idx < len(BIKOL_NAGA_PHRASES) else f"Unknown ({idx})"
    confidence = float(prediction[idx])
    logger.info(f"Predicted: {phrase} (Confidence: {confidence})")
    return {'phrase': phrase, 'confidence': confidence}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_video():
    if request.method == 'OPTIONS':
        return '', 204
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(video.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        original = secure_filename(video.filename)
        ext = original.rsplit('.', 1)[1].lower()
        unique = f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        video.save(path)

        result = predict_lipreading(path)

        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': unique,
            'original_filename': original,
            'videoUrl': f'https://nagasabot.onrender.com/uploads/{unique}',
            'phrase': result['phrase'],
            'accuracy': result['confidence'],
            'timestamp': datetime.now().timestamp()
        }), 200
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_video(filename):
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    temp_path = 'temp_video.mp4'
    file.save(temp_path)
    try:
        result = predict_lipreading(temp_path)
        return jsonify({
            'phrase': result['phrase'],
            'confidence': result['confidence']
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
