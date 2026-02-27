from __future__ import annotations

from dataclasses import dataclass
from typing import Any

cv2 = None
mp = None
np = None

BLINK_CLOSE_THRESHOLD_FACTOR = 0.82
BLINK_MIN_CLOSED_FRAMES = 1
DISTANCE_TARGET_RANGE_RATIO = 0.16
SMILE_TARGET_FACTOR = 1.12


@dataclass
class FrameMetrics:
    ear: float
    face_width: float
    smile_ratio: float


def _distance(a, b) -> float:
    if np is None:
        raise RuntimeError("Dependencias de liveness nao carregadas")
    return float(np.hypot((a.x - b.x), (a.y - b.y)))


def _calc_ear(landmarks, p1, p2, p3, p4, p5, p6) -> float | None:
    try:
        a = landmarks[p2]
        b = landmarks[p6]
        c = landmarks[p3]
        d = landmarks[p5]
        e = landmarks[p1]
        f = landmarks[p4]
    except Exception:
        return None

    horizontal = _distance(e, f)
    if not np.isfinite(horizontal) or horizontal < 1e-6:
        return None

    return (_distance(a, b) + _distance(c, d)) / (2.0 * horizontal)


def _extract_metrics(landmarks) -> FrameMetrics | None:
    left_ear = _calc_ear(landmarks, 33, 160, 158, 133, 153, 144)
    right_ear = _calc_ear(landmarks, 362, 385, 387, 263, 373, 380)
    if left_ear is None or right_ear is None:
        return None

    ear = (left_ear + right_ear) / 2.0

    xs = [lm.x for lm in landmarks]
    face_width = float(max(xs) - min(xs))
    if not np.isfinite(face_width) or face_width <= 1e-6:
        return None

    try:
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
    except Exception:
        return None

    eye_dist = _distance(left_eye, right_eye)
    if not np.isfinite(eye_dist) or eye_dist <= 1e-6:
        return None

    mouth_width = _distance(mouth_left, mouth_right)
    smile_ratio = mouth_width / eye_dist

    return FrameMetrics(ear=ear, face_width=face_width, smile_ratio=smile_ratio)


def _count_blinks(ear_values: list[float]) -> tuple[int, float, float]:
    if np is None:
        raise RuntimeError("Dependencias de liveness nao carregadas")
    if not ear_values:
        return 0, 0.0, 0.0

    arr = np.array(ear_values, dtype=np.float32)
    baseline = float(np.percentile(arr, 75))
    threshold = baseline * BLINK_CLOSE_THRESHOLD_FACTOR

    closed_frames = 0
    was_closed = False
    blinks = 0

    for ear in ear_values:
        is_closed = ear < threshold
        if is_closed:
            closed_frames += 1
            was_closed = True
        else:
            if was_closed and closed_frames >= BLINK_MIN_CLOSED_FRAMES:
                blinks += 1
            closed_frames = 0
            was_closed = False

    return blinks, baseline, threshold


def analyze_liveness_frames(frame_bytes_list: list[bytes], required_checks: list[str] | None = None) -> dict[str, Any]:
    global cv2, mp, np
    if cv2 is None or mp is None or np is None:
        try:
            import cv2 as _cv2
            import mediapipe as _mp
            import numpy as _np
        except ImportError as exc:
            raise ImportError(
                "Dependencias de liveness ausentes. Instale: opencv-python-headless, mediapipe, numpy."
            ) from exc
        cv2 = _cv2
        mp = _mp
        np = _np

    if required_checks is None:
        required_checks = ["blink", "distance", "smile"]

    metrics: list[FrameMetrics] = []
    total_frames = len(frame_bytes_list)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    try:
        for frame_bytes in frame_bytes_list:
            img_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            frame_metrics = _extract_metrics(landmarks)
            if frame_metrics:
                metrics.append(frame_metrics)
    finally:
        face_mesh.close()

    valid_frames = len(metrics)
    if valid_frames == 0:
        return {
            "passed": False,
            "checks": {
                "blink": {"passed": False, "count": 0, "target": 1},
                "distance": {"passed": False, "range_ratio": 0.0, "target": DISTANCE_TARGET_RANGE_RATIO},
                "smile": {"passed": False, "max_ratio": 0.0, "target": SMILE_TARGET_FACTOR},
            },
            "summary": {
                "total_frames": total_frames,
                "valid_frames": 0,
                "face_detected_frames": 0,
            },
        }

    ear_values = [m.ear for m in metrics]
    face_widths = [m.face_width for m in metrics]
    smile_ratios = [m.smile_ratio for m in metrics]

    blink_count, ear_baseline, ear_threshold = _count_blinks(ear_values)
    blink_passed = blink_count >= 1

    width_baseline = float(np.median(face_widths)) if face_widths else 0.0
    width_range = float(max(face_widths) - min(face_widths)) if face_widths else 0.0
    range_ratio = (width_range / width_baseline) if width_baseline > 1e-6 else 0.0
    distance_passed = range_ratio >= DISTANCE_TARGET_RANGE_RATIO

    smile_baseline = float(np.median(smile_ratios)) if smile_ratios else 0.0
    smile_max = float(max(smile_ratios)) if smile_ratios else 0.0
    smile_ratio_gain = (smile_max / smile_baseline) if smile_baseline > 1e-6 else 0.0
    smile_passed = smile_ratio_gain >= SMILE_TARGET_FACTOR

    checks = {
        "blink": {
            "passed": blink_passed,
            "count": blink_count,
            "target": 1,
            "ear_baseline": ear_baseline,
            "ear_threshold": ear_threshold,
        },
        "distance": {
            "passed": distance_passed,
            "range_ratio": range_ratio,
            "target": DISTANCE_TARGET_RANGE_RATIO,
        },
        "smile": {
            "passed": smile_passed,
            "max_ratio": smile_ratio_gain,
            "target": SMILE_TARGET_FACTOR,
        },
    }

    required = [c for c in required_checks if c in checks]
    if not required:
        required = ["blink", "distance", "smile"]

    passed = all(checks[check]["passed"] for check in required)

    return {
        "passed": passed,
        "checks": checks,
        "required_checks": required,
        "summary": {
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "face_detected_frames": valid_frames,
        },
    }
