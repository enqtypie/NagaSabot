from flask import Flask, request, jsonify, send_from_directory # type: ignore
from flask_cors import CORS # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'}  # Added webm support
MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB max-limit

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

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
        
        # For demo purposes, return mock lip reading results
        # In a real application, this would be processed by your lip reading model
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': unique_filename,
            'original_filename': original_filename,
            'videoUrl': f'http://localhost:5000/uploads/{unique_filename}',
            'phrase': 'Hello, this is a test phrase',  # Mock phrase - replace with actual lip reading result
            'accuracy': 85.5,  # Mock accuracy - replace with actual accuracy
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
