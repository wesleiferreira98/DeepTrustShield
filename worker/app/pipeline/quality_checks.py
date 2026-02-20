import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def check_quality(image_path: str, face: dict) -> dict:
    """
    Check image quality for the detected face region.
    
    Returns:
        Dictionary with quality metrics:
        - blur_score: Laplacian variance (higher = sharper)
        - brightness: Average brightness (0-255)
        - contrast: Standard deviation of intensity
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Extract face region
    x, y, w, h = face["x"], face["y"], face["width"], face["height"]
    face_region = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Calculate blur score (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    # Calculate brightness (mean of grayscale)
    brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    quality = {
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "contrast": float(contrast)
    }
    
    logger.info(f"Quality check: blur={blur_score:.1f}, brightness={brightness:.1f}, contrast={contrast:.1f}")
    
    return quality
