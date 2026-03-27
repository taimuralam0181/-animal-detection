import os
import tempfile
import uuid
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Add safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([DetectionModel])

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILENAME = os.getenv('YOLO_MODEL', 'yolov8m.pt')
FALLBACK_MODEL_FILENAME = 'yolov8n.pt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_IMAGE_SIZE = 960
AGNOSTIC_NMS = True
ANIMAL_CLASSES = {
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe'
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
print("Loading YOLOv8 model...")
try:
    model = YOLO(MODEL_PATH)
    loaded_model_name = MODEL_FILENAME
    print(f"Model loaded successfully: {loaded_model_name}")
except Exception as e:
    fallback_model_path = os.path.join(MODEL_DIR, FALLBACK_MODEL_FILENAME)
    loaded_model_name = FALLBACK_MODEL_FILENAME
    print(
        f"Warning: failed to load {MODEL_PATH} ({e}). "
        f"Falling back to {fallback_model_path}."
    )
    model = YOLO(fallback_model_path)

def save_pil_image(image, filepath):
    """Write images atomically so later reads don't see partial files."""
    directory = os.path.dirname(filepath)
    _, extension = os.path.splitext(filepath)

    with tempfile.NamedTemporaryFile(
        dir=directory,
        delete=False,
        suffix=extension or '.jpg',
    ) as temp_file:
        temp_path = temp_file.name

    try:
        image.save(temp_path, format='JPEG', quality=95)
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def save_uploaded_image(file_storage):
    """Accept any Pillow-readable image and normalize it to JPEG for inference."""
    original_name = secure_filename(file_storage.filename or 'image')
    base_name, _ = os.path.splitext(original_name)
    safe_base_name = base_name or 'image'
    filename = f"{uuid.uuid4()}_{safe_base_name}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as image:
            image = ImageOps.exif_transpose(image)
            image.load()

            if image.mode in ('RGBA', 'LA') or (
                image.mode == 'P' and 'transparency' in image.info
            ):
                rgba_image = image.convert('RGBA')
                normalized_image = Image.new('RGB', rgba_image.size, (255, 255, 255))
                normalized_image.paste(rgba_image, mask=rgba_image.getchannel('A'))
            else:
                normalized_image = image.convert('RGB')

            image_array = np.array(normalized_image)
            save_pil_image(normalized_image, filepath)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('Unsupported or invalid image file') from exc

    return filename, filepath, image_array

def draw_bounding_boxes(image_array, detections, output_path):
    """Draw bounding boxes on the normalized RGB image and persist the result."""
    img = image_array.copy()

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Draw rectangle
        color = (34, 197, 94)  # Green color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{det['class']} {det['confidence']:.1%}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_top = max(0, y1 - label_size[1] - 10)
        cv2.rectangle(img, (x1, label_top), (x1 + label_size[0], y1), color, -1)

        # Draw label text
        text_y = max(label_size[1] + 2, y1 - 5)
        cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    annotated_image = Image.fromarray(img)
    save_pil_image(annotated_image, output_path)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': loaded_model_name,
        'message': 'Animal Detection API is running'
    })

@app.route('/api/detect', methods=['POST'])
def detect_animals():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        # Save the uploaded image in a normalized format that YOLO/OpenCV can read.
        try:
            filename, filepath, image_array = save_uploaded_image(file)
        except ValueError as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400
        
        # Run YOLOv8 detection
        results = model(
            image_array,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=INFERENCE_IMAGE_SIZE,
            agnostic_nms=AGNOSTIC_NMS,
        )
        
        # Parse detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                class_name = result.names[cls_id]

                # Keep only supported animal classes above the confidence threshold.
                if confidence >= CONFIDENCE_THRESHOLD and class_name in ANIMAL_CLASSES:
                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': box.xyxy[0].tolist()
                    })
        
        # Draw bounding boxes on image
        if detections:
            base, _ = os.path.splitext(filepath)
            annotated_path = f"{base}_annotated.jpg"
            draw_bounding_boxes(image_array, detections, annotated_path)
            result_filename = os.path.basename(annotated_path)
        else:
            result_filename = filename
        
        return jsonify({
            'success': True,
            'image_url': f"/api/uploads/{result_filename}",
            'detections': detections,
            'total_detections': len(detections)
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("=" * 50)
    print("Animal Detection API")
    print("=" * 50)
    print("Server running at http://localhost:5000")
    print("API endpoint: http://localhost:5000/api/detect")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
