from flask import Flask, request, jsonify, send_from_directory # type: ignore
from flask_cors import CORS # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os
from datetime import datetime
import uuid
from lipreading_model import predict_lipreading, load_model

app = Flask(__name__)
CORS(app, origins=["https://nagasabot-frontend.onrender.com", "http://localhost:4200"])

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'}  # Added webm support
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB max-limit

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Generate unique filename
        original_filename = secure_filename(video_file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
        
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        video_file.save(filepath)
        
        # Get lipreading result from the model
        result = predict_lipreading(filepath, model)
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': unique_filename,
            'original_filename': original_filename,
            'videoUrl': f'http://localhost:5000/uploads/{unique_filename}',
            'phrase': result.get('phrase', 'No phrase detected'),
            'accuracy': float(result.get('confidence', 0.0)),
            'timestamp': datetime.now().timestamp()
        }), 200
        
    except Exception as e:
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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    temp_path = 'temp_video.mp4'
    file.save(temp_path)
    try:
        result = predict_lipreading(temp_path, model)
        return jsonify({
            'phrase': result.get('phrase', 'No phrase detected'),
            'confidence': float(result.get('confidence', 0.0))
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
