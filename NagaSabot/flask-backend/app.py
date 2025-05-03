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
import download_model  # Ensure model is downloaded before loading

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
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "HEAD"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "X-Requested-With"]
    }
})

# Add after_request handler to ensure CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,HEAD')
    return response

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
    logger.info(f"Starting lip reading prediction for video: {video_path}")
    start_time = datetime.now()
    
    # Validate video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return {'error': 'Video file not found', 'phrase': 'Error', 'confidence': 0.0}
    
    # Get video info
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    logger.info(f"Video file size: {file_size_mb:.2f} MB")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return {'error': 'Cannot open video file', 'phrase': 'Error', 'confidence': 0.0}
    
    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Video info - FPS: {fps}, Frames: {frame_count}, Resolution: {video_width}x{video_height}, Duration: {duration:.2f}s")
    
    # Check if video is too short
    if frame_count < 25:
        logger.warning(f"Video is too short: {frame_count} frames, need at least 10 frames")
        return {'error': 'Video is too short', 'phrase': 'Too short', 'confidence': 0.0}
    
    all_frames = []
    processed_frames = 0
    faces_detected = 0
    
    try:
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
                
                processed_frames += 1
                if processed_frames % 10 == 0:
                    logger.info(f"Processed {processed_frames}/{frame_count} frames...")
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                
                if result.multi_face_landmarks:
                    faces_detected += 1
                    landmarks = result.multi_face_landmarks[0]
                    lip_region, *_ = get_fixed_centered_lip_region(frame, landmarks)
                    enhanced = enhance_lip_region(lip_region)
                    all_frames.append(enhanced)
    except Exception as e:
        logger.error(f"Error processing video frames: {str(e)}")
        return {'error': f'Error processing video: {str(e)}', 'phrase': 'Error', 'confidence': 0.0}
    finally:
        cap.release()
    
    # Check if we found any frames with faces
    if not all_frames:
        logger.warning("No faces detected in the video")
        return {'error': 'No faces detected in video', 'phrase': 'No face', 'confidence': 0.0}
    
    logger.info(f"Video processing stats - Processed frames: {processed_frames}, Frames with faces: {faces_detected}, Usable frames: {len(all_frames)}")
    
    # Prepare frames for model input
    try:
        if len(all_frames) >= TOTAL_FRAMES:
            logger.info(f"Sampling {TOTAL_FRAMES} frames from {len(all_frames)} available frames")
            indices = np.linspace(0, len(all_frames) - 1, TOTAL_FRAMES).astype(int)
            frames = [all_frames[i] for i in indices]
        else:
            logger.info(f"Padding with {TOTAL_FRAMES - len(all_frames)} empty frames")
            frames = all_frames + [np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)] * (TOTAL_FRAMES - len(all_frames))
        
        X = np.expand_dims(np.array(frames, dtype=np.float32) / 255.0, axis=0)
        
        # Run prediction
        logger.info("Running model prediction...")
        prediction = model.predict(X)[0]
        idx = int(np.argmax(prediction))
        phrase = BIKOL_NAGA_PHRASES[idx] if idx < len(BIKOL_NAGA_PHRASES) else f"Unknown ({idx})"
        confidence = float(prediction[idx])
        
        # Calculate time taken
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Prediction complete - Phrase: '{phrase}', Confidence: {confidence:.4f}, Processing time: {processing_time:.2f}s")
        
        return {
            'phrase': phrase, 
            'confidence': confidence,
            'processing_time': processing_time,
            'frames_processed': processed_frames,
            'faces_detected': faces_detected,
            'prediction_index': idx
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {'error': f'Prediction error: {str(e)}', 'phrase': 'Error', 'confidence': 0.0}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST', 'OPTIONS'])
def upload_video():
    if request.method == 'OPTIONS':
        return '', 204
        
    # Add support for GET requests
    if request.method == 'GET':
        return jsonify({
            'message': 'This is the video upload endpoint. Please use POST method to upload a video file.',
            'usage': 'Make a POST request with a video file in the "video" field of a multipart/form-data request.',
            'example': 'curl -X POST -F "video=@your_video.mp4" https://nagasabot.onrender.com/upload',
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }), 200
        
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
    try:
        # Check if model is loaded
        model_loaded = model is not None
        
        # Check if upload directory exists and is writable
        upload_dir_exists = os.path.exists(UPLOAD_FOLDER)
        upload_dir_writable = os.access(UPLOAD_FOLDER, os.W_OK)
        
        # Get number of files in upload directory
        upload_files_count = len(os.listdir(UPLOAD_FOLDER)) if upload_dir_exists else 0
        
        # Get memory usage if psutil is available
        memory_info = {}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = {
                "memory_used_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "memory_percent": round(process.memory_percent(), 2)
            }
        except ImportError:
            memory_info = {"message": "psutil not available for memory stats"}
            
        return jsonify({
            'status': 'healthy',
            'server_time': datetime.now().isoformat(),
            'model_loaded': model_loaded,
            'upload_directory': {
                'exists': upload_dir_exists,
                'writable': upload_dir_writable,
                'files_count': upload_files_count
            },
            'memory': memory_info,
            'environment': os.environ.get('FLASK_ENV', 'production'),
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return jsonify({
        'message': 'API is working!',
        'time': datetime.now().isoformat(),
        'endpoints': {
            'health': '/health - GET request to check server health',
            'predict': '/predict - POST request to predict from video',
            'upload': '/upload - POST request to upload and process video',
            'get_video': '/uploads/<filename> - GET request to retrieve a video'
        }
    })

@app.route('/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    # Add support for GET requests
    if request.method == 'GET':
        return jsonify({
            'message': 'This is the prediction endpoint. Please use POST method to upload a video file.',
            'usage': 'Make a POST request with a video file in the "file" field of a multipart/form-data request.',
            'example': 'curl -X POST -F "file=@your_video.mp4" https://nagasabot.onrender.com/predict',
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }), 200
        
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

@app.route('/')
def index():
    """Root endpoint that provides API information"""
    return jsonify({
        'name': 'NagaSabot Lip Reading API',
        'status': 'operational',
        'version': '1.0.0',
        'endpoints': {
            'root': '/ - This information',
            'test': '/test - Basic API test',
            'health': '/health - Server health check',
            'predict': '/predict - Lip reading prediction (POST with video file)',
            'upload': '/upload - Video upload and processing (POST with video file)',
            'uploads': '/uploads/<filename> - Access uploaded files'
        },
        'documentation': 'Visit /test for more endpoint details',
        'frontend': 'https://nagasabot-frontend.onrender.com'
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
