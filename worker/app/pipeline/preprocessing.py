import cv2
import os
import numpy as np
import torch
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

# ImageNet normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = 224

# Se o rosto detectado cobre mais que este % da imagem, usa a imagem inteira
FACE_COVERAGE_THRESHOLD = 0.6


def preprocess_face(image_path: str, face: dict, job_id: str, artifacts_path: str):
    """
    Preprocess face for model inference.
    
    1. Crop face region (ou usa imagem inteira se já for um rosto)
    2. Resize to 224x224
    3. Normalize with ImageNet stats
    
    Returns:
        Tuple of (face_crop_path, face_tensor)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_height, img_width = image.shape[:2]
    
    # Extract face region
    x, y, w, h = face["x"], face["y"], face["width"], face["height"]
    
    # Calcular a cobertura do rosto na imagem
    face_area = w * h
    image_area = img_width * img_height
    coverage = face_area / image_area if image_area > 0 else 0
    
    logger.info(f"Face coverage: {coverage*100:.1f}% of image")
    
    # Se o rosto cobre mais de 60% da imagem, a imagem já é um rosto recortado
    # Neste caso, usamos a imagem INTEIRA para não recortar novamente
    if coverage >= FACE_COVERAGE_THRESHOLD:
        logger.info(f"Image appears to be a pre-cropped face (coverage={coverage*100:.1f}%). Using full image.")
        face_crop = image
    else:
        logger.info(f"Cropping face region: ({x}, {y}) {w}x{h}")
        face_crop = image[y:y+h, x:x+w]
    
    # Save face crop for evidence
    job_dir = os.path.join(artifacts_path, job_id)
    os.makedirs(job_dir, exist_ok=True)
    face_crop_path = os.path.join(job_dir, "face_crop.jpg")
    cv2.imwrite(face_crop_path, face_crop)
    logger.info(f"Face crop saved: {face_crop_path}")
    
    # Resize for model
    face_resized = cv2.resize(face_crop, (TARGET_SIZE, TARGET_SIZE))
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    face_tensor = transform(face_rgb)
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    
    logger.info(f"Preprocessed face tensor shape: {face_tensor.shape}")
    
    return face_crop_path, face_tensor
