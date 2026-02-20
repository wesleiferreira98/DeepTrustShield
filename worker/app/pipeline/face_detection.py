import cv2
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")

def download_model():
    """Download the face detection model if not present."""
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading face detection model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, MODEL_PATH)
        logger.info("Model downloaded successfully")


def detect_faces(image_path: str) -> list:
    """
    Detect faces in an image using MediaPipe Tasks API.
    
    Returns:
        List of face dictionaries with x, y, width, height
    """
    # Download model if needed
    download_model()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]
    
    # Create face detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    
    faces = []
    
    with vision.FaceDetector.create_from_options(options) as detector:
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect faces
        detection_result = detector.detect(mp_image)
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            
            # Get coordinates (already in pixels)
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Add padding (20% on each side)
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(w + 2 * padding_x, width - x)
            h = min(h + 2 * padding_y, height - y)
            
            confidence = detection.categories[0].score if detection.categories else 0
            
            faces.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "confidence": float(confidence),
                # Porcentagens relativas para o frontend
                "x_pct": round(x / width * 100, 2),
                "y_pct": round(y / height * 100, 2),
                "width_pct": round(w / width * 100, 2),
                "height_pct": round(h / height * 100, 2),
            })
            
            logger.info(f"Face detected: ({x}, {y}) {w}x{h}, confidence: {confidence:.2f}")
    
    return faces
