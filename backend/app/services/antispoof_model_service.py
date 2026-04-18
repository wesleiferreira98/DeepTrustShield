from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sigmoid(value: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-value))


def _default_analysis(enabled: bool, reason: str) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "passed": not enabled,
        "score": 1.0 if not enabled else 0.0,
        "p_spoof": 0.0 if not enabled else 1.0,
        "threshold": None,
        "aggregation": None,
        "frame_scores": [],
        "reasons": [] if not enabled else [reason],
    }


def _load_metadata(metadata_path: str) -> dict[str, Any]:
    path = Path(metadata_path)
    if not metadata_path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_session(model_path: str):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def _preprocess_crop(crop, image_size: int, np, cv2):
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))


def _aggregate_scores(scores: list[float], aggregation: str, np) -> float:
    if not scores:
        return 1.0
    if aggregation == "median":
        return float(np.median(scores))
    if aggregation in {"p75", "percentile75"}:
        return float(np.percentile(scores, 75))
    if aggregation in {"p90", "percentile90"}:
        return float(np.percentile(scores, 90))
    if aggregation == "max":
        return float(max(scores))
    return float(np.percentile(scores, 75))


def analyze_antispoof_crops(crops: list[Any], settings) -> dict[str, Any]:
    """Run the ONNX anti-spoof model on BGR face crops.

    Returns p_spoof where higher means more likely presentation attack.
    """
    if not settings.liveness_onnx_antispoof_enabled:
        return _default_analysis(False, "model_disabled")

    model_path = os.path.abspath(settings.liveness_onnx_antispoof_path)
    if not os.path.exists(model_path):
        analysis = _default_analysis(True, "model_file_missing")
        analysis["model_path"] = model_path
        return analysis

    if not crops:
        return _default_analysis(True, "no_face_crops")

    try:
        import cv2
        import numpy as np
    except ImportError:
        return _default_analysis(True, "preprocess_dependencies_missing")

    try:
        session = _load_session(model_path)
    except ImportError:
        return _default_analysis(True, "onnxruntime_missing")
    except Exception as exc:
        analysis = _default_analysis(True, "model_load_failed")
        analysis["error"] = str(exc)
        return analysis

    metadata = _load_metadata(settings.liveness_onnx_antispoof_metadata_path)
    image_size = int(metadata.get("image_size", 224))
    metadata_threshold = metadata.get("threshold")
    threshold = (
        float(settings.liveness_onnx_antispoof_threshold)
        if settings.liveness_onnx_antispoof_threshold >= 0
        else float(metadata_threshold if metadata_threshold is not None else 0.5)
    )
    aggregation = (settings.liveness_onnx_antispoof_aggregation or "p75").lower()

    batch = np.stack([_preprocess_crop(crop, image_size, np, cv2) for crop in crops]).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: batch})[0].reshape(-1)
    frame_scores = [float(_sigmoid(float(logit))) for logit in output]
    p_spoof = _aggregate_scores(frame_scores, aggregation, np)
    passed = p_spoof < threshold

    return {
        "enabled": True,
        "passed": bool(passed),
        "score": float(1.0 - p_spoof),
        "p_spoof": float(p_spoof),
        "threshold": threshold,
        "aggregation": aggregation,
        "frame_scores": frame_scores,
        "reasons": [] if passed else ["model_spoof_score_high"],
        "model_path": model_path,
        "model_version": metadata.get("model_version", "antispoof_mobilenet_v1"),
        "metadata_threshold": metadata_threshold,
    }
