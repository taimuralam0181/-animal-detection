import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from ultralytics import YOLO, YOLOWorld
from ultralytics.nn.tasks import DetectionModel

# Add safe globals for PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([DetectionModel])

app = Flask(__name__)
CORS(app)

LOCAL_PROVIDER = 'local'
SEEK_INATURALIST_PROVIDER = 'seek-inaturalist'
GOOGLE_CLOUD_VISION_PROVIDER = 'google-cloud-vision'
AWS_REKOGNITION_PROVIDER = 'aws-rekognition'
AZURE_CUSTOM_VISION_PROVIDER = 'azure-custom-vision'

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT_PATH = Path(PROJECT_ROOT)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILENAME = os.getenv('YOLO_MODEL', 'yolov8s-world.pt')
FALLBACK_MODEL_FILENAME = 'yolov8m.pt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
DEFAULT_PROVIDER = os.getenv('DETECTION_PROVIDER', LOCAL_PROVIDER).strip().lower()
MAX_IMAGE_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_DATASET_UPLOAD_BYTES = int(
    os.getenv('MAX_DATASET_UPLOAD_BYTES', str(12 * 1024 * 1024 * 1024))
)
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_IMAGE_SIZE = 960
AGNOSTIC_NMS = True
CLOSED_SET_ANIMAL_CLASSES = {
    'cat',
    'cow',
    'dog',
    'elephant',
    'giraffe',
    'horse',
    'zebra',
}
TRAINING_TARGET_ANIMAL_CLASSES = [
    'dog',
    'cat',
    'cow',
    'horse',
    'deer',
    'elephant',
    'zebra',
    'giraffe',
    'tiger',
    'lion',
]
DETECTION_TARGET_ANIMAL_CLASSES = [
    'lion',
    'tiger',
    'leopard',
    'deer',
    'dog',
    'cat',
    'elephant',
    'giraffe',
    'zebra',
    'horse',
    'snake',
    'cow',
    'bear',
    'fox',
    'monkey',
]
DETECTION_TARGET_CLASS_SET = set(DETECTION_TARGET_ANIMAL_CLASSES)
LABEL_ALIASES = {
    'cattle': 'cow',
    'ox': 'cow',
    'bull': 'cow',
    'equine': 'horse',
    'deers': 'deer',
    'leopards': 'leopard',
    'doe': 'deer',
    'stag': 'deer',
    'fawn': 'deer',
    'snakes': 'snake',
    'serpent': 'snake',
    'serpents': 'snake',
    'bears': 'bear',
    'foxes': 'fox',
    'monkeys': 'monkey',
    'ape': 'monkey',
    'apes': 'monkey',
    'primate': 'monkey',
    'primates': 'monkey',
}
PROVIDER_METADATA = {
    LOCAL_PROVIDER: {
        'label': 'Local YOLO + CLIP',
        'mode': 'builtin',
        'description': 'Runs directly on this machine with the bundled model.',
    },
    SEEK_INATURALIST_PROVIDER: {
        'label': 'Seek / iNaturalist',
        'mode': 'external',
        'description': 'Official public docs expose observations APIs, not a hosted prediction endpoint.',
    },
    GOOGLE_CLOUD_VISION_PROVIDER: {
        'label': 'Google Cloud Vision',
        'mode': 'cloud',
        'description': 'Uses Cloud Vision object localization with Google credentials.',
    },
    AWS_REKOGNITION_PROVIDER: {
        'label': 'AWS Rekognition',
        'mode': 'cloud',
        'description': 'Uses Rekognition labels or Custom Labels, depending on your AWS config.',
    },
    AZURE_CUSTOM_VISION_PROVIDER: {
        'label': 'Azure Custom Vision',
        'mode': 'cloud',
        'description': 'Calls a published Azure Custom Vision object detection model.',
    },
}
ACTIVE_DATASET_DIR = PROJECT_ROOT_PATH / 'datasets' / 'animals10'
USER_DATASET_UPLOADS_DIR = PROJECT_ROOT_PATH / 'datasets' / 'user_uploads'
CUSTOM_MODELS_DIR = PROJECT_ROOT_PATH / 'custom_models'
CUSTOM_MODELS_REGISTRY_PATH = CUSTOM_MODELS_DIR / 'registry.json'
TRAINING_SCRIPT_PATH = Path(__file__).resolve().parent / 'train_animals10.py'
TRAINING_LOG_DIR = PROJECT_ROOT_PATH / 'runs' / 'logs'
NORMALIZER_SCRIPT = PROJECT_ROOT_PATH / 'scripts' / 'normalize_dataset_zip.py'
DATASET_REQUIRED_SPLITS = ('train', 'val')
DATASET_ALL_SPLITS = ('train', 'val', 'test')
CUSTOM_SELECTION_MODES = ('default', 'single', 'selected', 'all')
CUSTOM_MIN_TRAINING_IMAGES = 5
CUSTOM_MODEL_SCORE_FLOOR = 0.18
DATASET_IMAGE_EXTENSIONS = {
    '.avif',
    '.bmp',
    '.gif',
    '.heic',
    '.heif',
    '.jfif',
    '.jpeg',
    '.jpg',
    '.png',
    '.tif',
    '.tiff',
    '.webp',
}
DATASET_CONFIG_PATH = ACTIVE_DATASET_DIR / 'data.yaml'

CACHE_HOME = os.path.join(PROJECT_ROOT, '.cache-home')
CLIP_CACHE_DIR = os.path.join(CACHE_HOME, '.cache', 'clip')
CLIP_MODEL_NAME = 'ViT-B/32'
YOLO_CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'Ultralytics')
SEEK_INATURALIST_URL = (
    'https://help.inaturalist.org/en/support/solutions/articles/'
    '151000169914-what-is-the-difference-between-inaturalist-and-seek-by-inaturalist-'
)
GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '').strip()
AWS_REGION = (
    os.getenv('AWS_REGION')
    or os.getenv('AWS_DEFAULT_REGION')
    or 'us-east-1'
)
AWS_REKOGNITION_PROJECT_VERSION_ARN = os.getenv(
    'AWS_REKOGNITION_PROJECT_VERSION_ARN',
    '',
).strip()
AZURE_CUSTOM_VISION_ENDPOINT = os.getenv(
    'AZURE_CUSTOM_VISION_ENDPOINT',
    '',
).strip().rstrip('/')
AZURE_CUSTOM_VISION_PREDICTION_KEY = os.getenv(
    'AZURE_CUSTOM_VISION_PREDICTION_KEY',
    '',
).strip()
AZURE_CUSTOM_VISION_PROJECT_ID = os.getenv(
    'AZURE_CUSTOM_VISION_PROJECT_ID',
    '',
).strip()
AZURE_CUSTOM_VISION_PUBLISHED_NAME = os.getenv(
    'AZURE_CUSTOM_VISION_PUBLISHED_NAME',
    '',
).strip()
os.makedirs(CACHE_HOME, exist_ok=True)
os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)
os.makedirs(USER_DATASET_UPLOADS_DIR, exist_ok=True)
os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
os.environ['HOME'] = CACHE_HOME
os.environ['USERPROFILE'] = CACHE_HOME
os.environ['YOLO_CONFIG_DIR'] = YOLO_CONFIG_DIR

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = max(MAX_IMAGE_UPLOAD_BYTES, MAX_DATASET_UPLOAD_BYTES)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

for split_name in DATASET_ALL_SPLITS:
    os.makedirs(ACTIVE_DATASET_DIR / 'images' / split_name, exist_ok=True)
    os.makedirs(ACTIVE_DATASET_DIR / 'labels' / split_name, exist_ok=True)

training_job = {
    'status': 'idle',
    'job_id': None,
    'started_at': None,
    'finished_at': None,
    'command': None,
    'run_name': None,
    'log_path': None,
    'return_code': None,
    'error': None,
    'process': None,
}

custom_training_job = {
    'status': 'idle',
    'model_id': None,
    'model_name': None,
    'started_at': None,
    'finished_at': None,
    'progress': 0,
    'processed_images': 0,
    'total_images': 0,
    'error': None,
    'message': None,
    'thread': None,
}
custom_training_lock = threading.Lock()

# Load YOLOv8 model
print("Loading YOLOv8 model...")
try:
    if MODEL_FILENAME.endswith('-world.pt'):
        model = YOLOWorld(MODEL_PATH)
        model.set_classes(DETECTION_TARGET_ANIMAL_CLASSES)
        active_classes = DETECTION_TARGET_ANIMAL_CLASSES
        model_mode = 'open-vocabulary'
    else:
        model = YOLO(MODEL_PATH)
        active_classes = sorted(CLOSED_SET_ANIMAL_CLASSES)
        model_mode = 'closed-set'
    loaded_model_name = MODEL_FILENAME
    print(f"Model loaded successfully: {loaded_model_name} ({model_mode})")
except Exception as e:
    fallback_model_path = os.path.join(MODEL_DIR, FALLBACK_MODEL_FILENAME)
    loaded_model_name = FALLBACK_MODEL_FILENAME
    active_classes = sorted(CLOSED_SET_ANIMAL_CLASSES)
    model_mode = 'closed-set-fallback'
    print(
        f"Warning: failed to load {MODEL_PATH} ({e}). "
        f"Falling back to {fallback_model_path}."
    )
    model = YOLO(fallback_model_path)

clip_module = None
clip_model = None
clip_preprocess = None
clip_text_tokens = None
clip_enabled = False

try:
    import clip as clip_module

    clip_model, clip_preprocess = clip_module.load(
        CLIP_MODEL_NAME,
        device='cpu',
        download_root=CLIP_CACHE_DIR,
    )
    clip_text_tokens = clip_module.tokenize(DETECTION_TARGET_ANIMAL_CLASSES)
    clip_enabled = True
    print(f"CLIP classifier loaded successfully: {CLIP_MODEL_NAME}")
except Exception as e:
    print(f"Warning: CLIP classifier unavailable ({e}). Using detector labels only.")

def module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False

def pil_image_to_jpeg_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()

def save_bytes_atomically(file_bytes, filepath):
    """Write files atomically so later reads don't see partial files."""
    directory = os.path.dirname(filepath)
    _, extension = os.path.splitext(filepath)

    with tempfile.NamedTemporaryFile(
        dir=directory,
        delete=False,
        suffix=extension or '.jpg',
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(file_bytes)
        temp_file.flush()

    try:
        os.replace(temp_path, filepath)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def save_pil_image(image, filepath):
    save_bytes_atomically(pil_image_to_jpeg_bytes(image), filepath)

def format_bytes(byte_count):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(byte_count)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == 'B':
                return f'{int(size)} {unit}'
            return f'{size:.1f} {unit}'
        size /= 1024

def save_uploaded_dataset_archive(file_storage, archive_path):
    archive_path = Path(archive_path)
    file_storage.stream.seek(0)

    with archive_path.open('wb') as output_file:
        shutil.copyfileobj(file_storage.stream, output_file, length=1024 * 1024)

def ensure_active_dataset_config():
    dataset_root = ACTIVE_DATASET_DIR.resolve().as_posix()
    names_block = '\n'.join(
        f'  {index}: {name}'
        for index, name in enumerate(TRAINING_TARGET_ANIMAL_CLASSES)
    )
    config_text = (
        f'path: {dataset_root}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n\n'
        'names:\n'
        f'{names_block}\n'
    )
    DATASET_CONFIG_PATH.write_text(config_text, encoding='utf-8')

ensure_active_dataset_config()

def load_json_file(path, default_value):
    path = Path(path)
    if not path.exists():
        return default_value

    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def save_json_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def slugify_name(value):
    slug = re.sub(r'[^a-z0-9]+', '-', value.strip().lower())
    return slug.strip('-') or 'custom-animal'

def get_custom_model_dir(model_id):
    return CUSTOM_MODELS_DIR / model_id

def get_custom_model_metadata_path(model_id):
    return get_custom_model_dir(model_id) / 'metadata.json'

def get_custom_model_prototype_path(model_id):
    return get_custom_model_dir(model_id) / 'prototype.npy'

def get_custom_model_images_dir(model_id):
    return get_custom_model_dir(model_id) / 'images'

def list_custom_model_images(model_id):
    images_dir = get_custom_model_images_dir(model_id)
    if not images_dir.exists():
        return []

    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in DATASET_IMAGE_EXTENSIONS
    )

def update_custom_registry():
    registry_payload = {'models': []}

    for model_dir in sorted(CUSTOM_MODELS_DIR.iterdir()) if CUSTOM_MODELS_DIR.exists() else []:
        if not model_dir.is_dir():
            continue
        metadata_path = model_dir / 'metadata.json'
        if not metadata_path.exists():
            continue
        registry_payload['models'].append(load_json_file(metadata_path, {}))

    save_json_file(CUSTOM_MODELS_REGISTRY_PATH, registry_payload)
    return registry_payload

def save_custom_model_metadata(model_id, metadata):
    metadata_path = get_custom_model_metadata_path(model_id)
    save_json_file(metadata_path, metadata)
    update_custom_registry()

def list_custom_models():
    registry = load_json_file(CUSTOM_MODELS_REGISTRY_PATH, {'models': []})
    models = registry.get('models', [])
    return sorted(
        models,
        key=lambda item: (
            item.get('trained_at') or '',
            item.get('created_at') or '',
            item.get('name') or '',
        ),
        reverse=True,
    )

def get_custom_model_by_id(model_id):
    for model in list_custom_models():
        if model.get('id') == model_id:
            return model
    return None

def reset_custom_training_job():
    custom_training_job.update({
        'status': 'idle',
        'model_id': None,
        'model_name': None,
        'started_at': None,
        'finished_at': None,
        'progress': 0,
        'processed_images': 0,
        'total_images': 0,
        'error': None,
        'message': None,
        'thread': None,
    })

def set_custom_training_state(**updates):
    with custom_training_lock:
        custom_training_job.update(updates)

def serialize_custom_training_job():
    with custom_training_lock:
        thread = custom_training_job.get('thread')
        if thread is not None and not thread.is_alive():
            custom_training_job['thread'] = None

        return {
            'status': custom_training_job['status'],
            'model_id': custom_training_job['model_id'],
            'model_name': custom_training_job['model_name'],
            'started_at': custom_training_job['started_at'],
            'finished_at': custom_training_job['finished_at'],
            'progress': custom_training_job['progress'],
            'processed_images': custom_training_job['processed_images'],
            'total_images': custom_training_job['total_images'],
            'error': custom_training_job['error'],
            'message': custom_training_job['message'],
            'is_running': custom_training_job['status'] in {'preparing', 'training'},
        }

def custom_training_is_running():
    training_state = serialize_custom_training_job()
    return training_state['status'] in {'preparing', 'training'}

def save_training_image(file_storage, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            normalized_image = image.convert('RGB')
            save_pil_image(normalized_image, str(output_path))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f'Unsupported training image: {file_storage.filename}') from exc

def append_training_images(model_id, image_files):
    images_dir = get_custom_model_images_dir(model_id)
    images_dir.mkdir(parents=True, exist_ok=True)

    existing_images = list_custom_model_images(model_id)
    start_index = len(existing_images) + 1
    new_image_paths = []

    for offset, image_file in enumerate(image_files, start=0):
        extension = Path(secure_filename(image_file.filename or '')).suffix.lower()
        if extension not in DATASET_IMAGE_EXTENSIONS:
            extension = '.jpg'
        output_path = images_dir / f'image-{start_index + offset:04d}{extension}'
        save_training_image(image_file, output_path)
        new_image_paths.append(output_path)

    return existing_images + new_image_paths, new_image_paths

def extract_crop_pil(image_array, bbox=None):
    if bbox is None:
        return Image.fromarray(image_array)

    height, width = image_array.shape[:2]
    clamped_bbox = clamp_bbox(bbox, width, height)
    if clamped_bbox is None:
        return None

    x1, y1, x2, y2 = clamped_bbox
    crop = image_array[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return Image.fromarray(crop)

def encode_pil_image_with_clip(pil_image):
    if not clip_enabled:
        raise RuntimeError('CLIP is not available for custom model training.')

    image_tensor = clip_preprocess(pil_image).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features[0].cpu().numpy().astype(np.float32)

def train_custom_model_worker(model_id, model_name, image_paths):
    metadata = get_custom_model_by_id(model_id) or {
        'id': model_id,
        'name': model_name,
    }

    try:
        set_custom_training_state(
            status='preparing',
            message='Preparing custom animal images...',
        )
        metadata.update({
            'status': 'preparing',
            'error': None,
        })
        save_custom_model_metadata(model_id, metadata)

        embeddings = []
        total_images = len(image_paths)
        set_custom_training_state(total_images=total_images, processed_images=0, progress=0)

        for index, image_path in enumerate(image_paths, start=1):
            if index == 1:
                set_custom_training_state(
                    status='training',
                    message='Building custom model embeddings...',
                )
                metadata['status'] = 'training'
                save_custom_model_metadata(model_id, metadata)

            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                embeddings.append(encode_pil_image_with_clip(image))

            progress = int(round((index / total_images) * 100))
            set_custom_training_state(
                processed_images=index,
                progress=progress,
            )

        if not embeddings:
            raise RuntimeError('No valid images were available for custom model training.')

        embedding_matrix = np.vstack(embeddings).astype(np.float32)
        prototype = embedding_matrix.mean(axis=0)
        prototype /= np.linalg.norm(prototype) + 1e-12
        similarity_scores = embedding_matrix @ prototype
        threshold = float(
            max(
                CUSTOM_MODEL_SCORE_FLOOR,
                min(0.95, float(np.percentile(similarity_scores, 20)) - 0.02),
            )
        )

        np.save(get_custom_model_prototype_path(model_id), prototype.astype(np.float32))
        metadata.update({
            'status': 'completed',
            'image_count': total_images,
            'threshold': round(threshold, 4),
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'embedding_model': CLIP_MODEL_NAME,
            'error': None,
        })
        save_custom_model_metadata(model_id, metadata)
        set_custom_training_state(
            status='completed',
            finished_at=datetime.now().isoformat(timespec='seconds'),
            progress=100,
            processed_images=total_images,
            total_images=total_images,
            message='Custom animal model is ready.',
            error=None,
            thread=None,
        )
    except Exception as exc:
        metadata.update({
            'status': 'failed',
            'error': str(exc),
        })
        save_custom_model_metadata(model_id, metadata)
        set_custom_training_state(
            status='failed',
            finished_at=datetime.now().isoformat(timespec='seconds'),
            error=str(exc),
            message='Custom animal training failed.',
            thread=None,
        )

def start_custom_model_training(model_name, image_files, existing_model_id=None):
    if not clip_enabled:
        raise RuntimeError('Custom animal training requires CLIP, but CLIP is not available.')
    if custom_training_is_running():
        raise RuntimeError('Another custom animal model is already training.')
    if not image_files:
        raise ValueError('Upload at least one image to train a custom animal model.')
    if len(image_files) < CUSTOM_MIN_TRAINING_IMAGES:
        raise ValueError(
            f'Upload at least {CUSTOM_MIN_TRAINING_IMAGES} images for better custom training.'
        )

    existing_model = None
    if existing_model_id:
        existing_model = get_custom_model_by_id(existing_model_id)
        if existing_model is None:
            raise ValueError('Selected custom animal model was not found.')

    cleaned_name = (model_name or '').strip()
    if existing_model is not None:
        if not cleaned_name:
            cleaned_name = existing_model['name']
        model_id = existing_model['id']
        all_image_paths, new_image_paths = append_training_images(model_id, image_files)
        metadata = dict(existing_model)
        metadata.update({
            'name': cleaned_name,
            'slug': slugify_name(cleaned_name),
            'status': 'preparing',
            'image_count': len(all_image_paths),
            'last_appended_at': datetime.now().isoformat(timespec='seconds'),
            'new_images_added': len(new_image_paths),
            'error': None,
        })
        image_paths = all_image_paths
    else:
        if not cleaned_name:
            raise ValueError('Custom animal name is required.')

        model_id = f"{slugify_name(cleaned_name)}-{uuid.uuid4().hex[:6]}"
        all_image_paths, new_image_paths = append_training_images(model_id, image_files)
        metadata = {
            'id': model_id,
            'name': cleaned_name,
            'slug': slugify_name(cleaned_name),
            'status': 'preparing',
            'image_count': len(all_image_paths),
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'trained_at': None,
            'threshold': None,
            'error': None,
            'new_images_added': len(new_image_paths),
        }
        image_paths = all_image_paths

    save_custom_model_metadata(model_id, metadata)

    with custom_training_lock:
        reset_custom_training_job()
        custom_training_job.update({
            'status': 'preparing',
            'model_id': model_id,
            'model_name': cleaned_name,
            'started_at': datetime.now().isoformat(timespec='seconds'),
            'finished_at': None,
            'progress': 0,
            'processed_images': 0,
            'total_images': len(image_paths),
            'error': None,
            'message': 'Preparing custom animal images...',
        })
        thread = threading.Thread(
            target=train_custom_model_worker,
            args=(model_id, cleaned_name, image_paths),
            daemon=True,
        )
        custom_training_job['thread'] = thread
        thread.start()

    return {
        'model': metadata,
        'training': serialize_custom_training_job(),
    }

def get_selected_custom_models(selection_mode, requested_model_ids):
    completed_models = [
        model_data
        for model_data in list_custom_models()
        if model_data.get('status') == 'completed'
    ]
    if not completed_models or selection_mode == 'default':
        return []

    if selection_mode == 'all':
        return completed_models

    requested_lookup = set(requested_model_ids or [])
    selected_models = [
        model_data
        for model_data in completed_models
        if model_data.get('id') in requested_lookup
    ]
    if selection_mode == 'single' and selected_models:
        return selected_models[:1]
    return selected_models

def classify_crop_with_custom_models(image_array, bbox, selected_models):
    if not selected_models or not clip_enabled:
        return []

    crop_image = extract_crop_pil(image_array, bbox)
    if crop_image is None:
        return []

    image_embedding = encode_pil_image_with_clip(crop_image)
    matches = []

    for model_data in selected_models:
        prototype_path = get_custom_model_prototype_path(model_data['id'])
        if not prototype_path.exists():
            continue

        prototype = np.load(prototype_path).astype(np.float32)
        similarity = float(np.dot(image_embedding, prototype))
        threshold = float(model_data.get('threshold') or CUSTOM_MODEL_SCORE_FLOOR)
        score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        matches.append({
            'id': model_data['id'],
            'name': model_data['name'],
            'similarity': round(similarity, 4),
            'score': round(score, 3),
            'threshold': round(threshold, 4),
            'accepted': similarity >= threshold,
        })

    return sorted(matches, key=lambda item: item['similarity'], reverse=True)

def parse_custom_model_request(form_data):
    selection_mode = (form_data.get('custom_mode') or 'default').strip().lower()
    if selection_mode not in CUSTOM_SELECTION_MODES:
        selection_mode = 'default'

    raw_selected_ids = form_data.get('custom_model_ids', '[]')
    selected_ids = []
    if raw_selected_ids:
        try:
            parsed_ids = json.loads(raw_selected_ids)
            if isinstance(parsed_ids, list):
                selected_ids = [
                    str(model_id).strip()
                    for model_id in parsed_ids
                    if str(model_id).strip()
                ]
        except json.JSONDecodeError:
            selected_ids = [
                model_id.strip()
                for model_id in raw_selected_ids.split(',')
                if model_id.strip()
            ]

    if selection_mode == 'single' and len(selected_ids) > 1:
        selected_ids = selected_ids[:1]

    return selection_mode, selected_ids

def apply_custom_models_to_detections(image_array, detections, selection_mode, selected_model_ids):
    selected_models = get_selected_custom_models(selection_mode, selected_model_ids)
    if not selected_models:
        return detections, []

    enriched_detections = []
    for detection in detections:
        updated_detection = dict(detection)
        custom_matches = classify_crop_with_custom_models(
            image_array,
            detection['bbox'],
            selected_models,
        )
        updated_detection['custom_matches'] = custom_matches
        if custom_matches:
            top_match = custom_matches[0]
            updated_detection['custom_match'] = top_match
            if top_match['accepted']:
                updated_detection['resolved_class'] = top_match['name']
                updated_detection['resolved_confidence'] = top_match['score']
            else:
                updated_detection['resolved_class'] = detection['class']
                updated_detection['resolved_confidence'] = detection['confidence']
        else:
            updated_detection['resolved_class'] = detection['class']
            updated_detection['resolved_confidence'] = detection['confidence']

        enriched_detections.append(updated_detection)

    return enriched_detections, [
        {'id': model_data['id'], 'name': model_data['name']}
        for model_data in selected_models
    ]

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

            jpeg_bytes = pil_image_to_jpeg_bytes(normalized_image)
            image_array = np.array(normalized_image)
            save_bytes_atomically(jpeg_bytes, filepath)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('Unsupported or invalid image file') from exc

    return filename, filepath, image_array, jpeg_bytes

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

def clamp_bbox(bbox, width, height):
    x1 = max(0, min(int(round(bbox[0])), max(width - 1, 0)))
    y1 = max(0, min(int(round(bbox[1])), max(height - 1, 0)))
    x2 = max(x1 + 1, min(int(round(bbox[2])), width))
    y2 = max(y1 + 1, min(int(round(bbox[3])), height))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]

def normalize_target_label(raw_label):
    if not raw_label:
        return None

    normalized = raw_label.strip().lower().replace('_', ' ').replace('-', ' ')
    candidates = [normalized]
    if normalized.endswith('es'):
        candidates.append(normalized[:-2])
    if normalized.endswith('s'):
        candidates.append(normalized[:-1])

    for candidate in candidates:
        if candidate in DETECTION_TARGET_CLASS_SET:
            return candidate
        if candidate in LABEL_ALIASES:
            return LABEL_ALIASES[candidate]

        for token in candidate.split():
            if token in DETECTION_TARGET_CLASS_SET:
                return token
            if token in LABEL_ALIASES:
                return LABEL_ALIASES[token]

    return None

def normalized_vertices_to_bbox(vertices, width, height):
    if not vertices:
        return None

    x_values = [float(getattr(vertex, 'x', 0.0) or 0.0) * width for vertex in vertices]
    y_values = [float(getattr(vertex, 'y', 0.0) or 0.0) * height for vertex in vertices]
    return clamp_bbox([min(x_values), min(y_values), max(x_values), max(y_values)], width, height)

def aws_bbox_to_pixels(box, width, height):
    if not box:
        return None

    left = float(box.get('Left', 0.0)) * width
    top = float(box.get('Top', 0.0)) * height
    right = left + (float(box.get('Width', 0.0)) * width)
    bottom = top + (float(box.get('Height', 0.0)) * height)
    return clamp_bbox([left, top, right, bottom], width, height)

def azure_bbox_to_pixels(box, width, height):
    if not box:
        return None

    left = float(box.get('left', 0.0)) * width
    top = float(box.get('top', 0.0)) * height
    right = left + (float(box.get('width', 0.0)) * width)
    bottom = top + (float(box.get('height', 0.0)) * height)
    return clamp_bbox([left, top, right, bottom], width, height)

def deduplicate_detections(detections):
    unique_detections = []
    seen_keys = set()

    for detection in sorted(detections, key=lambda item: item['confidence'], reverse=True):
        bbox_key = tuple(int(round(value / 8.0)) for value in detection['bbox'])
        dedupe_key = (detection['class'], bbox_key)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        unique_detections.append(detection)

    return unique_detections

def ensure_clean_dir(directory):
    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)

def normalize_zip_with_script(input_path, output_path):
    command = [
        sys.executable,
        str(NORMALIZER_SCRIPT),
        str(input_path),
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'Normalizer failed: {result.stderr.strip() or result.stdout.strip()}'
        )
    return result.stdout.strip()

def is_dataset_image(path):
    return path.suffix.lower() in DATASET_IMAGE_EXTENSIONS

def summarize_dataset(dataset_root):
    dataset_root = Path(dataset_root)
    split_summary = {}
    total_images = 0
    total_labels = 0

    for split_name in DATASET_ALL_SPLITS:
        image_dir = dataset_root / 'images' / split_name
        label_dir = dataset_root / 'labels' / split_name
        image_count = 0
        label_count = 0

        if image_dir.exists():
            image_count = sum(1 for path in image_dir.iterdir() if path.is_file() and is_dataset_image(path))
        if label_dir.exists():
            label_count = sum(1 for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == '.txt')

        split_summary[split_name] = {
            'images': image_count,
            'labels': label_count,
        }
        total_images += image_count
        total_labels += label_count

    return {
        'path': str(dataset_root),
        'splits': split_summary,
        'total_images': total_images,
        'total_labels': total_labels,
    }

def find_yolo_dataset_root(search_root):
    search_root = Path(search_root)
    candidate_paths = [search_root] + [path for path in search_root.rglob('*') if path.is_dir()]

    for candidate in candidate_paths:
        if all(
            (candidate / 'images' / split_name).exists()
            and (candidate / 'labels' / split_name).exists()
            for split_name in DATASET_REQUIRED_SPLITS
        ):
            return candidate

    return None

def extract_dataset_zip(archive_path, destination_dir):
    archive_path = Path(archive_path)
    destination_dir = Path(destination_dir)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = destination_dir / member.filename
            resolved_destination = member_path.resolve()
            if not str(resolved_destination).startswith(str(destination_dir.resolve())):
                raise ValueError('Invalid ZIP file structure')

        archive.extractall(destination_dir)

def import_dataset_into_active(source_root, import_prefix):
    source_root = Path(source_root)
    imported_counts = {split_name: 0 for split_name in DATASET_ALL_SPLITS}
    skipped_counts = {split_name: 0 for split_name in DATASET_ALL_SPLITS}

    for split_name in DATASET_ALL_SPLITS:
        source_image_dir = source_root / 'images' / split_name
        source_label_dir = source_root / 'labels' / split_name

        if not source_image_dir.exists() or not source_label_dir.exists():
            continue

        target_image_dir = ACTIVE_DATASET_DIR / 'images' / split_name
        target_label_dir = ACTIVE_DATASET_DIR / 'labels' / split_name

        for image_path in sorted(source_image_dir.iterdir()):
            if not image_path.is_file() or not is_dataset_image(image_path):
                continue

            label_path = source_label_dir / f'{image_path.stem}.txt'
            if not label_path.exists():
                skipped_counts[split_name] += 1
                continue

            target_stem = f'{import_prefix}_{image_path.stem}'
            shutil.copy2(image_path, target_image_dir / f'{target_stem}{image_path.suffix.lower()}')
            shutil.copy2(label_path, target_label_dir / f'{target_stem}.txt')
            imported_counts[split_name] += 1

    return {
        'imported': imported_counts,
        'skipped_without_label': skipped_counts,
    }

def dataset_ready_for_training():
    summary = summarize_dataset(ACTIVE_DATASET_DIR)
    train_split = summary['splits']['train']
    val_split = summary['splits']['val']
    ready = (
        train_split['images'] > 0
        and train_split['labels'] > 0
        and val_split['images'] > 0
        and val_split['labels'] > 0
    )
    return ready, summary

def read_log_tail(log_path, max_lines=20):
    if not log_path:
        return []

    log_file = Path(log_path)
    if not log_file.exists():
        return []

    with log_file.open('r', encoding='utf-8', errors='replace') as handle:
        lines = handle.readlines()

    return [line.rstrip() for line in lines[-max_lines:]]

def sync_training_job_state():
    process = training_job.get('process')
    if process is None:
        return

    return_code = process.poll()
    if return_code is None:
        return

    training_job['process'] = None
    training_job['return_code'] = return_code
    training_job['finished_at'] = datetime.now().isoformat(timespec='seconds')
    if return_code == 0:
        training_job['status'] = 'completed'
        training_job['error'] = None
    else:
        training_job['status'] = 'failed'
        training_job['error'] = f'Training exited with code {return_code}.'

def serialize_training_job():
    sync_training_job_state()
    ready, dataset_summary = dataset_ready_for_training()

    return {
        'status': training_job['status'],
        'job_id': training_job['job_id'],
        'started_at': training_job['started_at'],
        'finished_at': training_job['finished_at'],
        'command': training_job['command'],
        'run_name': training_job['run_name'],
        'log_path': training_job['log_path'],
        'return_code': training_job['return_code'],
        'error': training_job['error'],
        'dataset_ready': ready,
        'dataset': dataset_summary,
        'log_tail': read_log_tail(training_job['log_path']),
    }

def start_training_job():
    sync_training_job_state()
    if training_job['status'] == 'running':
        raise RuntimeError('Training is already running.')

    dataset_ready, dataset_summary = dataset_ready_for_training()
    if not dataset_ready:
        raise ValueError(
            'Training dataset is incomplete. Upload train and val image/label pairs first.'
        )

    job_id = uuid.uuid4().hex[:8]
    run_name = f'animals10-{job_id}'
    log_path = TRAINING_LOG_DIR / f'{run_name}.log'
    command = [
        sys.executable,
        str(TRAINING_SCRIPT_PATH),
        '--name',
        run_name,
    ]

    if not TRAINING_SCRIPT_PATH.exists():
        raise FileNotFoundError(f'Training script not found: {TRAINING_SCRIPT_PATH}')

    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    with log_path.open('w', encoding='utf-8') as log_handle:
        log_handle.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        log_handle.write(f"Command: {' '.join(command)}\n\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            cwd=str(TRAINING_SCRIPT_PATH.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    training_job['status'] = 'running'
    training_job['job_id'] = job_id
    training_job['started_at'] = datetime.now().isoformat(timespec='seconds')
    training_job['finished_at'] = None
    training_job['command'] = command
    training_job['run_name'] = run_name
    training_job['log_path'] = str(log_path)
    training_job['return_code'] = None
    training_job['error'] = None
    training_job['process'] = process

    return {
        'job': serialize_training_job(),
        'dataset': dataset_summary,
    }

def classify_crop_with_clip(image_array, bbox):
    """Refine ambiguous detector labels with a crop-based CLIP classification pass."""
    if not clip_enabled:
        return None, None

    height, width = image_array.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    crop = image_array[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop_image = Image.fromarray(crop)
    image_tensor = clip_preprocess(crop_image).unsqueeze(0)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image_tensor, clip_text_tokens)
        probabilities = logits_per_image.softmax(dim=-1)[0]

    best_index = int(torch.argmax(probabilities).item())
    best_label = DETECTION_TARGET_ANIMAL_CLASSES[best_index]
    best_probability = float(probabilities[best_index].item())
    return best_label, best_probability

def detect_with_local_model(image_array):
    results = model(
        image_array,
        verbose=False,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_IMAGE_SIZE,
        agnostic_nms=AGNOSTIC_NMS,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[cls_id]
            final_class_name = class_name
            final_confidence = confidence

            if model_mode == 'open-vocabulary':
                clip_class_name, clip_probability = classify_crop_with_clip(
                    image_array,
                    box.xyxy[0].tolist(),
                )
                if clip_class_name is not None:
                    final_class_name = clip_class_name
                    final_confidence = (confidence * clip_probability) ** 0.5

            if (
                final_confidence >= CONFIDENCE_THRESHOLD
                and final_class_name in active_classes
            ):
                detections.append({
                    'class': final_class_name,
                    'confidence': round(final_confidence, 3),
                    'bbox': box.xyxy[0].tolist(),
                })

    return deduplicate_detections(detections), None, None

def detect_with_google_cloud_vision(image_bytes, image_array):
    if not module_available('google.cloud.vision'):
        raise RuntimeError(
            'Google Cloud Vision dependency is missing. Run pip install -r backend/requirements.txt.'
        )

    try:
        from google.cloud import vision
    except ImportError as exc:
        raise RuntimeError(
            'Google Cloud Vision dependency is missing. Run pip install -r backend/requirements.txt.'
        ) from exc

    client = vision.ImageAnnotatorClient()
    response = client.object_localization(image=vision.Image(content=image_bytes))
    if response.error.message:
        raise RuntimeError(f"Google Cloud Vision error: {response.error.message}")

    height, width = image_array.shape[:2]
    detections = []

    for annotation in response.localized_object_annotations:
        class_name = normalize_target_label(annotation.name)
        confidence = float(annotation.score)
        bbox = normalized_vertices_to_bbox(
            annotation.bounding_poly.normalized_vertices,
            width,
            height,
        )

        if class_name and bbox and confidence >= CONFIDENCE_THRESHOLD:
            detections.append({
                'class': class_name,
                'confidence': round(confidence, 3),
                'bbox': bbox,
            })

    message = None
    if not detections:
        message = 'Google Cloud Vision did not return any matching target animal boxes for this image.'

    return deduplicate_detections(detections), message, None

def detect_with_aws_rekognition(image_bytes, image_array):
    if not module_available('boto3'):
        raise RuntimeError(
            'AWS SDK dependency is missing. Run pip install -r backend/requirements.txt.'
        )

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            'AWS SDK dependency is missing. Run pip install -r backend/requirements.txt.'
        ) from exc

    client = boto3.Session(region_name=AWS_REGION).client('rekognition')
    height, width = image_array.shape[:2]
    detections = []

    if AWS_REKOGNITION_PROJECT_VERSION_ARN:
        response = client.detect_custom_labels(
            ProjectVersionArn=AWS_REKOGNITION_PROJECT_VERSION_ARN,
            Image={'Bytes': image_bytes},
            MinConfidence=CONFIDENCE_THRESHOLD * 100,
        )

        for label in response.get('CustomLabels', []):
            class_name = normalize_target_label(label.get('Name', ''))
            confidence = float(label.get('Confidence', 0.0)) / 100.0
            bbox = aws_bbox_to_pixels(
                label.get('Geometry', {}).get('BoundingBox'),
                width,
                height,
            )

            if class_name and bbox and confidence >= CONFIDENCE_THRESHOLD:
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence, 3),
                    'bbox': bbox,
                })
    else:
        response = client.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=25,
            MinConfidence=CONFIDENCE_THRESHOLD * 100,
        )

        for label in response.get('Labels', []):
            class_name = normalize_target_label(label.get('Name', ''))
            if not class_name:
                continue

            for instance in label.get('Instances', []):
                confidence = float(instance.get('Confidence', 0.0)) / 100.0
                bbox = aws_bbox_to_pixels(instance.get('BoundingBox'), width, height)
                if bbox and confidence >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': bbox,
                    })

    message = None
    if not detections:
        if AWS_REKOGNITION_PROJECT_VERSION_ARN:
            message = 'AWS Rekognition Custom Labels returned no matching target detections for this image.'
        else:
            message = 'AWS Rekognition returned no matching target animal boxes for this image.'

    return deduplicate_detections(detections), message, None

def detect_with_azure_custom_vision(image_bytes, image_array):
    if not AZURE_CUSTOM_VISION_ENDPOINT:
        raise RuntimeError('Set AZURE_CUSTOM_VISION_ENDPOINT before using Azure Custom Vision.')
    if not AZURE_CUSTOM_VISION_PREDICTION_KEY:
        raise RuntimeError('Set AZURE_CUSTOM_VISION_PREDICTION_KEY before using Azure Custom Vision.')
    if not AZURE_CUSTOM_VISION_PROJECT_ID:
        raise RuntimeError('Set AZURE_CUSTOM_VISION_PROJECT_ID before using Azure Custom Vision.')
    if not AZURE_CUSTOM_VISION_PUBLISHED_NAME:
        raise RuntimeError('Set AZURE_CUSTOM_VISION_PUBLISHED_NAME before using Azure Custom Vision.')

    response = requests.post(
        (
            f"{AZURE_CUSTOM_VISION_ENDPOINT}/customvision/v3.1/prediction/"
            f"{AZURE_CUSTOM_VISION_PROJECT_ID}/detect/iterations/"
            f"{AZURE_CUSTOM_VISION_PUBLISHED_NAME}/image"
        ),
        headers={
            'Prediction-Key': AZURE_CUSTOM_VISION_PREDICTION_KEY,
            'Content-Type': 'application/octet-stream',
        },
        data=image_bytes,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    height, width = image_array.shape[:2]
    detections = []

    for prediction in payload.get('predictions', []):
        class_name = normalize_target_label(prediction.get('tagName', ''))
        confidence = float(prediction.get('probability', 0.0))
        bbox = azure_bbox_to_pixels(prediction.get('boundingBox'), width, height)

        if class_name and bbox and confidence >= CONFIDENCE_THRESHOLD:
            detections.append({
                'class': class_name,
                'confidence': round(confidence, 3),
                'bbox': bbox,
            })

    message = None
    if not detections:
        message = 'Azure Custom Vision returned no matching target detections for this image.'

    return deduplicate_detections(detections), message, None

def detect_with_seek_inaturalist():
    return [], (
        'Seek / iNaturalist was added as an external option, but the official iNaturalist API docs '
        'do not expose a public hosted prediction endpoint for direct app integration.'
    ), SEEK_INATURALIST_URL

def provider_is_configured(provider_id):
    if provider_id == LOCAL_PROVIDER:
        return True
    if provider_id == SEEK_INATURALIST_PROVIDER:
        return True
    if provider_id == GOOGLE_CLOUD_VISION_PROVIDER:
        return module_available('google.cloud.vision') and bool(GOOGLE_CREDENTIALS_PATH)
    if provider_id == AWS_REKOGNITION_PROVIDER:
        return module_available('boto3')
    if provider_id == AZURE_CUSTOM_VISION_PROVIDER:
        return (
            bool(AZURE_CUSTOM_VISION_ENDPOINT)
            and bool(AZURE_CUSTOM_VISION_PREDICTION_KEY)
            and bool(AZURE_CUSTOM_VISION_PROJECT_ID)
            and bool(AZURE_CUSTOM_VISION_PUBLISHED_NAME)
        )
    return False

def get_provider_statuses():
    provider_statuses = []
    for provider_id, metadata in PROVIDER_METADATA.items():
        provider_statuses.append({
            'id': provider_id,
            'label': metadata['label'],
            'mode': metadata['mode'],
            'description': metadata['description'],
            'configured': provider_is_configured(provider_id),
        })
    return provider_statuses

def build_detection_response(
    provider_id,
    filename,
    filepath,
    image_array,
    detections,
    message=None,
    external_url=None,
    custom_selection_mode='default',
    custom_models_used=None,
):
    if detections:
        base, _ = os.path.splitext(filepath)
        annotated_path = f"{base}_annotated.jpg"
        draw_bounding_boxes(image_array, detections, annotated_path)
        result_filename = os.path.basename(annotated_path)
    else:
        result_filename = filename

    payload = {
        'success': True,
        'image_url': f"/api/uploads/{result_filename}",
        'detections': detections,
        'total_detections': len(detections),
        'provider': provider_id,
        'provider_label': PROVIDER_METADATA[provider_id]['label'],
        'provider_mode': PROVIDER_METADATA[provider_id]['mode'],
        'custom_selection_mode': custom_selection_mode,
        'custom_models_used': custom_models_used or [],
    }

    if message:
        payload['message'] = message
    if external_url:
        payload['external_url'] = external_url

    return payload

@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_error):
    return jsonify({
        'success': False,
        'error': (
            'Upload too large. '
            f'Image limit: {format_bytes(MAX_IMAGE_UPLOAD_BYTES)}. '
            f'Dataset ZIP limit: {format_bytes(MAX_DATASET_UPLOAD_BYTES)}.'
        ),
    }), 413

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': loaded_model_name,
        'model_mode': model_mode,
        'clip_enabled': clip_enabled,
        'supported_classes': active_classes,
        'target_classes': DETECTION_TARGET_ANIMAL_CLASSES,
        'training_target_classes': TRAINING_TARGET_ANIMAL_CLASSES,
        'custom_models': list_custom_models(),
        'default_provider': DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDER_METADATA else LOCAL_PROVIDER,
        'providers': get_provider_statuses(),
        'message': 'Animal Detection API is running'
    })

@app.route('/api/custom-models/status', methods=['GET'])
def custom_models_status():
    return jsonify({
        'success': True,
        'clip_enabled': clip_enabled,
        'selection_modes': list(CUSTOM_SELECTION_MODES),
        'models': list_custom_models(),
        'training': serialize_custom_training_job(),
        'message': 'Custom animal models fetched successfully.',
    })

@app.route('/api/custom-models/train', methods=['POST'])
def train_custom_model():
    try:
        if 'images' not in request.files:
            return jsonify({'success': False, 'error': 'No training images were provided.'}), 400

        model_name = (request.form.get('name') or '').strip()
        existing_model_id = (request.form.get('existing_model_id') or '').strip() or None
        image_files = [
            image_file
            for image_file in request.files.getlist('images')
            if image_file and image_file.filename
        ]
        training_state = start_custom_model_training(
            model_name,
            image_files,
            existing_model_id=existing_model_id,
        )
        return jsonify({
            'success': True,
            'model': training_state['model'],
            'training': training_state['training'],
            'message': (
                'Custom animal retraining started.'
                if existing_model_id
                else 'Custom animal training started.'
            ),
        })
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 409
    except Exception as exc:
        print(f"Error: {str(exc)}")
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/custom-models/<model_id>', methods=['DELETE'])
def delete_custom_model(model_id):
    try:
        model_data = get_custom_model_by_id(model_id)
        if model_data is None:
            return jsonify({'success': False, 'error': 'Custom model not found.'}), 404

        if custom_training_is_running() and serialize_custom_training_job().get('model_id') == model_id:
            return jsonify({
                'success': False,
                'error': 'Cannot delete a custom model while it is training.',
            }), 409

        shutil.rmtree(get_custom_model_dir(model_id), ignore_errors=True)
        update_custom_registry()
        return jsonify({
            'success': True,
            'deleted_model_id': model_id,
            'message': 'Custom model deleted successfully.',
        })
    except Exception as exc:
        print(f"Error: {str(exc)}")
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/datasets/status', methods=['GET'])
def dataset_status():
    ensure_active_dataset_config()
    return jsonify({
        'success': True,
        'dataset': summarize_dataset(ACTIVE_DATASET_DIR),
        'target_classes': TRAINING_TARGET_ANIMAL_CLASSES,
        'detection_target_classes': DETECTION_TARGET_ANIMAL_CLASSES,
        'message': 'Upload a YOLO dataset ZIP to merge it into the training dataset.',
    })

@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    try:
        ensure_active_dataset_config()
        if request.content_length and request.content_length > MAX_DATASET_UPLOAD_BYTES:
            return jsonify({
                'success': False,
                'error': (
                    'Dataset ZIP is too large. '
                    f'Current limit: {format_bytes(MAX_DATASET_UPLOAD_BYTES)}.'
                ),
            }), 413

        if 'dataset' not in request.files:
            return jsonify({'success': False, 'error': 'No dataset ZIP provided'}), 400

        dataset_file = request.files['dataset']
        if dataset_file.filename == '':
            return jsonify({'success': False, 'error': 'No dataset file selected'}), 400

        original_name = secure_filename(dataset_file.filename or 'dataset.zip')
        if not original_name.lower().endswith('.zip'):
            return jsonify({'success': False, 'error': 'Dataset file must be a ZIP archive'}), 400

        import_prefix = uuid.uuid4().hex[:8]
        upload_root = USER_DATASET_UPLOADS_DIR / f'{import_prefix}_{Path(original_name).stem}'
        upload_root.mkdir(parents=True, exist_ok=True)
        archive_path = upload_root / 'dataset.zip'
        extract_root = upload_root / 'extracted'
        normalized_archive = upload_root / 'normalized.zip'
        normalization_message = None

        try:
            save_uploaded_dataset_archive(dataset_file, archive_path)
            ensure_clean_dir(extract_root)
            extract_dataset_zip(archive_path, extract_root)
        except zipfile.BadZipFile:
            shutil.rmtree(upload_root, ignore_errors=True)
            return jsonify({'success': False, 'error': 'Invalid ZIP archive'}), 400
        except ValueError as exc:
            shutil.rmtree(upload_root, ignore_errors=True)
            return jsonify({'success': False, 'error': str(exc)}), 400

        dataset_root = find_yolo_dataset_root(extract_root)
        if dataset_root is None:
            try:
                normalization_message = normalize_zip_with_script(archive_path, normalized_archive)
                ensure_clean_dir(extract_root)
                extract_dataset_zip(normalized_archive, extract_root)
                dataset_root = find_yolo_dataset_root(extract_root)
            except RuntimeError as exc:
                shutil.rmtree(upload_root, ignore_errors=True)
                return jsonify({'success': False, 'error': str(exc)}), 400

        if dataset_root is None:
            shutil.rmtree(upload_root, ignore_errors=True)
            return jsonify({
                'success': False,
                'error': (
                    'YOLO dataset layout not found. ZIP must contain images/train, images/val, '
                    'labels/train, and labels/val.'
                ),
            }), 400

        imported_summary = import_dataset_into_active(dataset_root, import_prefix)
        total_imported = sum(imported_summary['imported'].values())
        if total_imported == 0:
            shutil.rmtree(upload_root, ignore_errors=True)
            return jsonify({
                'success': False,
                'error': 'No valid image/label pairs were found in the uploaded dataset ZIP.',
            }), 400

        source_summary = summarize_dataset(dataset_root)
        active_summary = summarize_dataset(ACTIVE_DATASET_DIR)
        shutil.rmtree(upload_root, ignore_errors=True)

        return jsonify({
            'success': True,
            'dataset_name': original_name,
            'source_dataset': source_summary,
            'imported_summary': imported_summary,
            'active_dataset': active_summary,
            'message': 'Dataset imported into datasets/animals10 and ready for training.',
            'normalization': normalization_message,
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/training/status', methods=['GET'])
def training_status():
    return jsonify({
        'success': True,
        'training': serialize_training_job(),
        'message': 'Training status fetched successfully.',
    })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    try:
        ensure_active_dataset_config()
        training_state = start_training_job()
        return jsonify({
            'success': True,
            'training': training_state['job'],
            'dataset': training_state['dataset'],
            'message': 'Training started in the background.',
        })
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 409
    except Exception as exc:
        print(f"Error: {str(exc)}")
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/detect', methods=['POST'])
def detect_animals():
    try:
        if request.content_length and request.content_length > MAX_IMAGE_UPLOAD_BYTES:
            return jsonify({
                'success': False,
                'error': (
                    'Image is too large for detection. '
                    f'Current limit: {format_bytes(MAX_IMAGE_UPLOAD_BYTES)}.'
                ),
            }), 413

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        file = request.files['image']
        provider_id = request.form.get('provider', DEFAULT_PROVIDER).strip().lower()
        custom_selection_mode, selected_custom_model_ids = parse_custom_model_request(request.form)

        if provider_id not in PROVIDER_METADATA:
            return jsonify({'success': False, 'error': 'Unsupported detection provider'}), 400

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        try:
            filename, filepath, image_array, image_bytes = save_uploaded_image(file)
        except ValueError as exc:
            return jsonify({'success': False, 'error': str(exc)}), 400

        if provider_id == LOCAL_PROVIDER:
            detections, message, external_url = detect_with_local_model(image_array)
        elif provider_id == GOOGLE_CLOUD_VISION_PROVIDER:
            detections, message, external_url = detect_with_google_cloud_vision(
                image_bytes,
                image_array,
            )
        elif provider_id == AWS_REKOGNITION_PROVIDER:
            detections, message, external_url = detect_with_aws_rekognition(
                image_bytes,
                image_array,
            )
        elif provider_id == AZURE_CUSTOM_VISION_PROVIDER:
            detections, message, external_url = detect_with_azure_custom_vision(
                image_bytes,
                image_array,
            )
        else:
            detections, message, external_url = detect_with_seek_inaturalist()

        detections, custom_models_used = apply_custom_models_to_detections(
            image_array,
            detections,
            custom_selection_mode,
            selected_custom_model_ids,
        )

        return jsonify(
            build_detection_response(
                provider_id,
                filename,
                filepath,
                image_array,
                detections,
                message=message,
                external_url=external_url,
                custom_selection_mode=custom_selection_mode,
                custom_models_used=custom_models_used,
            )
        )
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
