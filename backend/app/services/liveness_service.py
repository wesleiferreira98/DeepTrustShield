from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from app.config import get_settings
from app.services.antispoof_model_service import analyze_antispoof_crops

cv2 = None
mp = None
np = None
mp_face_mesh = None
mp_tasks_vision = None
mp_tasks_base_options = None
landmarker_model_path = None

BLINK_CLOSE_THRESHOLD_FACTOR = 0.85
BLINK_MIN_CLOSED_FRAMES = 1
DISTANCE_TARGET_RANGE_RATIO = 0.16
SMILE_TARGET_FACTOR = 1.10
SMILE_INTERMEDIATE_FACTOR = 1.05
SMILE_MIN_ABS_DELTA = 0.035
SMILE_HOLD_MIN_CONSECUTIVE_FRAMES = 3
MOUTH_OPEN_TARGET_FACTOR = 1.46
MOUTH_OPEN_INTERMEDIATE_FACTOR = 1.24
MOUTH_OPEN_MIN_ABS_DELTA = 0.065
HEAD_TURN_TARGET_DELTA = 0.10
HEAD_PITCH_TARGET_DELTA = 0.07
ANTI_SPOOF_INTERNAL_MOTION_MIN = 0.085
ANTI_SPOOF_COMPENSATED_LOCAL_MOTION_MIN = 0.025
ANTI_SPOOF_GEOMETRIC_MOTION_MIN = 0.13
ANTI_SPOOF_LOW_TEXTURE_LAPLACIAN = 12.0
ANTI_SPOOF_SCREEN_RECTANGLE_MIN = 0.30
ANTI_SPOOF_SCREEN_RECTANGLE_HARD_MIN = 0.30
ANTI_SPOOF_CLOSE_SCREEN_FACE_BACKGROUND_MIN = 0.25
ANTI_SPOOF_CLOSE_SCREEN_LOCAL_MOTION_MAX = 0.012
ANTI_SPOOF_PLANAR_MOTION_PENALTY = 0.30


@dataclass
class FrameMetrics:
    ear: float
    face_width: float
    smile_ratio: float
    mouth_open_ratio: float
    head_yaw: float
    head_pitch: float


SUPPORTED_LIVENESS_REQUIREMENTS = {
    "blink",
    "distance",
    "smile",
    "smile_hold",
    "mouth_open",
    "head_left",
    "head_right",
    "head_up",
    "head_down",
}


class _LegacyFaceMeshWrapper:
    def __init__(self):
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect_landmarks(self, rgb_image):
        result = self._mesh.process(rgb_image)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def close(self):
        self._mesh.close()


class _TasksFaceLandmarkerWrapper:
    def __init__(self, model_path: str):
        options = mp_tasks_vision.FaceLandmarkerOptions(
            base_options=mp_tasks_base_options(model_asset_path=model_path),
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self):
        self._landmarker.close()


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
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
    except Exception:
        return None

    eye_dist = _distance(left_eye, right_eye)
    if not np.isfinite(eye_dist) or eye_dist <= 1e-6:
        return None

    mouth_width = _distance(mouth_left, mouth_right)
    smile_ratio = mouth_width / eye_dist
    mouth_open_ratio = _distance(upper_lip, lower_lip) / eye_dist
    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    eye_mid_y = (left_eye.y + right_eye.y) / 2.0
    head_yaw = (nose_tip.x - eye_mid_x) / eye_dist
    head_pitch = (nose_tip.y - eye_mid_y) / eye_dist

    return FrameMetrics(
        ear=ear,
        face_width=face_width,
        smile_ratio=smile_ratio,
        mouth_open_ratio=mouth_open_ratio,
        head_yaw=head_yaw,
        head_pitch=head_pitch,
    )


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


def _normalize_landmark_points(landmarks, indices: list[int]) -> dict[int, tuple[float, float]]:
    try:
        left_eye = landmarks[33]
        right_eye = landmarks[263]
    except Exception:
        return {}

    eye_dx = right_eye.x - left_eye.x
    eye_dy = right_eye.y - left_eye.y
    eye_dist = float(np.hypot(eye_dx, eye_dy))
    if not np.isfinite(eye_dist) or eye_dist <= 1e-6:
        return {}

    angle = float(np.arctan2(eye_dy, eye_dx))
    cos_a = float(np.cos(-angle))
    sin_a = float(np.sin(-angle))
    origin_x = (left_eye.x + right_eye.x) / 2.0
    origin_y = (left_eye.y + right_eye.y) / 2.0

    normalized: dict[int, tuple[float, float]] = {}
    for index in indices:
        try:
            landmark = landmarks[index]
        except Exception:
            continue
        centered_x = (landmark.x - origin_x) / eye_dist
        centered_y = (landmark.y - origin_y) / eye_dist
        normalized[index] = (
            centered_x * cos_a - centered_y * sin_a,
            centered_x * sin_a + centered_y * cos_a,
        )
    return normalized


def _calculate_compensated_local_motion(all_landmarks: list[Any | None]) -> dict[str, float]:
    local_indices = [
        13,
        14,
        61,
        291,
        78,
        308,
        160,
        158,
        153,
        144,
        385,
        387,
        373,
        380,
    ]
    previous_points: dict[int, tuple[float, float]] | None = None
    local_movements: list[float] = []
    mouth_movements: list[float] = []
    eye_movements: list[float] = []

    for landmarks in all_landmarks:
        if landmarks is None:
            continue

        current_points = _normalize_landmark_points(landmarks, local_indices)
        if not current_points:
            continue

        if previous_points:
            shared_indices = [index for index in local_indices if index in previous_points and index in current_points]
            if shared_indices:
                deltas = [
                    float(
                        np.hypot(
                            current_points[index][0] - previous_points[index][0],
                            current_points[index][1] - previous_points[index][1],
                        )
                    )
                    for index in shared_indices
                ]
                local_movements.append(float(np.median(deltas)))

                mouth_indices = [13, 14, 61, 291, 78, 308]
                mouth_deltas = [deltas[shared_indices.index(index)] for index in mouth_indices if index in shared_indices]
                if mouth_deltas:
                    mouth_movements.append(float(np.median(mouth_deltas)))

                eye_indices = [160, 158, 153, 144, 385, 387, 373, 380]
                eye_deltas = [deltas[shared_indices.index(index)] for index in eye_indices if index in shared_indices]
                if eye_deltas:
                    eye_movements.append(float(np.median(eye_deltas)))

        previous_points = current_points

    return {
        "compensated_local_motion": float(np.percentile(local_movements, 75)) if local_movements else 0.0,
        "compensated_mouth_motion": float(np.percentile(mouth_movements, 75)) if mouth_movements else 0.0,
        "compensated_eye_motion": float(np.percentile(eye_movements, 75)) if eye_movements else 0.0,
    }


def _detect_blink_events(indexed_metrics: list[tuple[int, FrameMetrics]]) -> tuple[list[dict[str, Any]], float, float]:
    if np is None:
        raise RuntimeError("Dependencias de liveness nao carregadas")
    if not indexed_metrics:
        return [], 0.0, 0.0

    ear_values = [metric.ear for _, metric in indexed_metrics]
    arr = np.array(ear_values, dtype=np.float32)
    baseline = float(np.percentile(arr, 75))
    threshold = baseline * BLINK_CLOSE_THRESHOLD_FACTOR

    events: list[dict[str, Any]] = []
    current_start: int | None = None
    current_min_ear: float | None = None

    for frame_index, metric in indexed_metrics:
        is_closed = metric.ear < threshold
        if is_closed:
            if current_start is None:
                current_start = frame_index
                current_min_ear = metric.ear
            else:
                current_min_ear = min(current_min_ear if current_min_ear is not None else metric.ear, metric.ear)
            continue

        if current_start is not None:
            duration_frames = frame_index - current_start
            if duration_frames >= BLINK_MIN_CLOSED_FRAMES:
                events.append(
                    {
                        "start_frame": current_start,
                        "end_frame": frame_index - 1,
                        "duration_frames": duration_frames,
                        "min_ear": current_min_ear,
                    }
                )
            current_start = None
            current_min_ear = None

    if current_start is not None:
        duration_frames = indexed_metrics[-1][0] - current_start + 1
        if duration_frames >= BLINK_MIN_CLOSED_FRAMES:
            events.append(
                {
                    "start_frame": current_start,
                    "end_frame": indexed_metrics[-1][0],
                    "duration_frames": duration_frames,
                    "min_ear": current_min_ear,
                }
            )

    return events, baseline, threshold


def _ensure_liveness_dependencies():
    global cv2, mp, np, mp_face_mesh, mp_tasks_vision, mp_tasks_base_options, landmarker_model_path
    if cv2 is not None and mp is not None and np is not None:
        return

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

    settings = get_settings()
    landmarker_model_path = settings.mediapipe_face_landmarker_path

    if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "face_mesh"):
        mp_face_mesh = _mp.solutions.face_mesh

    try:
        from mediapipe.tasks.python import vision as _vision
        from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions
        mp_tasks_vision = _vision
        mp_tasks_base_options = _BaseOptions
    except Exception:
        mp_tasks_vision = None
        mp_tasks_base_options = None


def _create_face_detector():
    _ensure_liveness_dependencies()

    if mp_tasks_vision is not None and mp_tasks_base_options is not None:
        if landmarker_model_path and os.path.exists(landmarker_model_path):
            return _TasksFaceLandmarkerWrapper(landmarker_model_path)

    if mp_face_mesh is not None:
        return _LegacyFaceMeshWrapper()

    if mp_tasks_vision is not None and mp_tasks_base_options is not None:
        raise ImportError(
            "MediaPipe Tasks encontrado, mas o modelo face_landmarker_v2.task nao foi localizado. "
            f"Defina o arquivo em {landmarker_model_path}."
        )

    raise ImportError(
        "Nao foi possivel inicializar o detector facial do backend com a versao atual do MediaPipe."
    )


def _build_temporal_frame_metrics(all_frame_metrics: list[FrameMetrics | None]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    previous_valid: FrameMetrics | None = None

    for index, frame_metrics in enumerate(all_frame_metrics):
        if frame_metrics is None:
            payload.append(
                {
                    "frame_index": index,
                    "face_detected": False,
                    "ear": None,
                    "face_width": None,
                    "smile_ratio": None,
                    "mouth_open_ratio": None,
                    "head_yaw": None,
                    "head_pitch": None,
                    "delta_ear": None,
                    "delta_face_width": None,
                    "delta_smile_ratio": None,
                    "delta_mouth_open_ratio": None,
                    "delta_head_yaw": None,
                    "delta_head_pitch": None,
                    "movement_score": None,
                }
            )
            continue

        delta_ear = None
        delta_face_width = None
        delta_smile_ratio = None
        delta_mouth_open_ratio = None
        delta_head_yaw = None
        delta_head_pitch = None
        movement_score = 0.0

        if previous_valid is not None:
            delta_ear = float(frame_metrics.ear - previous_valid.ear)
            delta_face_width = float(frame_metrics.face_width - previous_valid.face_width)
            delta_smile_ratio = float(frame_metrics.smile_ratio - previous_valid.smile_ratio)
            delta_mouth_open_ratio = float(frame_metrics.mouth_open_ratio - previous_valid.mouth_open_ratio)
            delta_head_yaw = float(frame_metrics.head_yaw - previous_valid.head_yaw)
            delta_head_pitch = float(frame_metrics.head_pitch - previous_valid.head_pitch)
            movement_score = float(
                abs(delta_ear)
                + abs(delta_face_width)
                + abs(delta_smile_ratio)
                + abs(delta_mouth_open_ratio)
                + abs(delta_head_yaw)
                + abs(delta_head_pitch)
            )

        payload.append(
            {
                "frame_index": index,
                "face_detected": True,
                "ear": frame_metrics.ear,
                "face_width": frame_metrics.face_width,
                "smile_ratio": frame_metrics.smile_ratio,
                "mouth_open_ratio": frame_metrics.mouth_open_ratio,
                "head_yaw": frame_metrics.head_yaw,
                "head_pitch": frame_metrics.head_pitch,
                "delta_ear": delta_ear,
                "delta_face_width": delta_face_width,
                "delta_smile_ratio": delta_smile_ratio,
                "delta_mouth_open_ratio": delta_mouth_open_ratio,
                "delta_head_yaw": delta_head_yaw,
                "delta_head_pitch": delta_head_pitch,
                "movement_score": movement_score,
            }
        )
        previous_valid = frame_metrics

    return payload


def _calculate_temporal_consistency(frame_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    movement_scores = [
        metric["movement_score"]
        for metric in frame_metrics
        if metric.get("movement_score") is not None
    ]
    if not movement_scores:
        return {
            "mean_movement_score": 0.0,
            "max_movement_score": 0.0,
            "low_motion_ratio": 1.0,
            "looks_static": True,
        }

    mean_movement = float(sum(movement_scores) / len(movement_scores))
    max_movement = float(max(movement_scores))
    low_motion_count = sum(score < 0.01 for score in movement_scores)
    low_motion_ratio = float(low_motion_count / len(movement_scores))

    return {
        "mean_movement_score": mean_movement,
        "max_movement_score": max_movement,
        "low_motion_ratio": low_motion_ratio,
        "looks_static": bool(low_motion_ratio >= 0.8 and max_movement < 0.03),
    }


def _extract_face_bbox(image, landmarks, margin_ratio: float = 0.18):
    height, width = image.shape[:2]
    xs = [point.x for point in landmarks if 0.0 <= point.x <= 1.0]
    ys = [point.y for point in landmarks if 0.0 <= point.y <= 1.0]
    if not xs or not ys:
        return None

    raw_left = int(min(xs) * width)
    raw_right = int(max(xs) * width)
    raw_top = int(min(ys) * height)
    raw_bottom = int(max(ys) * height)
    face_width = max(raw_right - raw_left, 1)
    face_height = max(raw_bottom - raw_top, 1)
    margin_x = int(face_width * margin_ratio)
    margin_y = int(face_height * margin_ratio)

    left = max(0, raw_left - margin_x)
    right = min(width, raw_right + margin_x)
    top = max(0, raw_top - margin_y)
    bottom = min(height, raw_bottom + margin_y)
    if right - left < 24 or bottom - top < 24:
        return None
    return left, top, right, bottom


def _extract_face_crop(image, landmarks, margin_ratio: float = 0.18):
    bbox = _extract_face_bbox(image, landmarks, margin_ratio)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return image[top:bottom, left:right]


def _line_overlap_ratio(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    reference = max(1.0, end_b - start_b)
    return float(overlap / reference)


def _score_screen_border_lines(edges, face_bbox: tuple[int, int, int, int], gray=None) -> float:
    height, width = edges.shape[:2]
    face_left, face_top, face_right, face_bottom = face_bbox
    face_width = max(face_right - face_left, 1)
    face_height = max(face_bottom - face_top, 1)
    face_center_x = (face_left + face_right) / 2
    face_center_y = (face_top + face_bottom) / 2
    min_dimension = min(height, width)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(35, int(min_dimension * 0.07)),
        minLineLength=max(36, int(min_dimension * 0.16)),
        maxLineGap=max(8, int(min_dimension * 0.035)),
    )
    if lines is None:
        return 0.0

    left_vertical = 0.0
    right_vertical = 0.0
    top_horizontal = 0.0
    bottom_horizontal = 0.0
    vertical_positions: dict[str, float] = {}
    horizontal_positions: dict[str, float] = {}

    for line in lines[:, 0]:
        x1, y1, x2, y2 = [float(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0:
            continue

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        if abs_dy >= max(abs_dx * 3.5, 1.0):
            line_x = (x1 + x2) / 2
            left_distance = face_left - line_x
            right_distance = line_x - face_right
            is_left_border = face_width * 0.04 <= left_distance <= face_width * 0.90
            is_right_border = face_width * 0.04 <= right_distance <= face_width * 0.90
            if not (is_left_border or is_right_border):
                continue
            overlap = _line_overlap_ratio(y_min, y_max, face_top, face_bottom)
            if overlap < 0.45:
                continue
            length_score = min(1.0, length / max(face_height * 1.35, 1.0))
            edge_strength = 0.55 * overlap + 0.45 * length_score
            if is_left_border:
                if edge_strength > left_vertical:
                    left_vertical = edge_strength
                    vertical_positions["left"] = line_x
            elif edge_strength > right_vertical:
                right_vertical = edge_strength
                vertical_positions["right"] = line_x

        if abs_dx >= max(abs_dy * 3.5, 1.0):
            line_y = (y1 + y2) / 2
            top_distance = face_top - line_y
            bottom_distance = line_y - face_bottom
            is_top_border = face_height * 0.05 <= top_distance <= face_height * 0.90
            is_bottom_border = face_height * 0.05 <= bottom_distance <= face_height * 0.90
            if not (is_top_border or is_bottom_border):
                continue
            overlap = _line_overlap_ratio(x_min, x_max, face_left, face_right)
            if overlap < 0.45:
                continue
            length_score = min(1.0, length / max(face_width * 1.35, 1.0))
            edge_strength = 0.55 * overlap + 0.45 * length_score
            if is_top_border:
                if edge_strength > top_horizontal:
                    top_horizontal = edge_strength
                    horizontal_positions["top"] = line_y
            elif edge_strength > bottom_horizontal:
                bottom_horizontal = edge_strength
                horizontal_positions["bottom"] = line_y

    vertical_pair_score = 0.0
    has_horizontal_support = bool(top_horizontal or bottom_horizontal)
    has_vertical_support = bool(left_vertical or right_vertical)
    vertical_dark_pair = False
    if gray is not None and left_vertical and right_vertical:
        sample_top = max(0, face_top)
        sample_bottom = min(height, face_bottom)

        def _vertical_strip_mean(line_x: float) -> float:
            center = int(round(line_x))
            strip_left = max(0, center - 4)
            strip_right = min(width, center + 5)
            strip = gray[sample_top:sample_bottom, strip_left:strip_right]
            if strip.size == 0:
                return 1.0
            return float(np.mean(strip) / 255.0)

        left_strip_mean = _vertical_strip_mean(vertical_positions["left"])
        right_strip_mean = _vertical_strip_mean(vertical_positions["right"])
        vertical_dark_pair = bool(left_strip_mean <= 0.58 and right_strip_mean <= 0.58)

    if left_vertical and right_vertical:
        span = vertical_positions["right"] - vertical_positions["left"]
        span_ratio = span / max(face_width, 1)
        if span_ratio > 2.85:
            span_score = 0.0
        else:
            span_score = min(1.0, max(0.0, (span_ratio - 1.05) / 1.35))
        if has_horizontal_support or vertical_dark_pair:
            vertical_pair_score = min(1.0, 0.45 + 0.35 * min(left_vertical, right_vertical) + 0.20 * span_score)
        else:
            vertical_pair_score = 0.18 * min(left_vertical, right_vertical)
    else:
        vertical_pair_score = 0.12 * max(left_vertical, right_vertical)

    horizontal_pair_score = 0.0
    if top_horizontal and bottom_horizontal:
        span = horizontal_positions["bottom"] - horizontal_positions["top"]
        span_ratio = span / max(face_height, 1)
        if span_ratio > 2.85:
            span_score = 0.0
        else:
            span_score = min(1.0, max(0.0, (span_ratio - 1.05) / 1.35))
        if has_vertical_support:
            horizontal_pair_score = min(1.0, 0.45 + 0.35 * min(top_horizontal, bottom_horizontal) + 0.20 * span_score)
        else:
            horizontal_pair_score = 0.15 * min(top_horizontal, bottom_horizontal)
    else:
        horizontal_pair_score = 0.10 * max(top_horizontal, bottom_horizontal)

    mixed_score = 0.0
    if max(left_vertical, right_vertical) and max(top_horizontal, bottom_horizontal):
        mixed_score = min(
            1.0,
            0.18
            + 0.32 * max(left_vertical, right_vertical)
            + 0.25 * max(top_horizontal, bottom_horizontal),
        )

    return float(max(vertical_pair_score, horizontal_pair_score, mixed_score))


def _score_screen_rectangle(image, face_bbox: tuple[int, int, int, int]) -> float:
    height, width = image.shape[:2]
    image_area = float(height * width)
    face_left, face_top, face_right, face_bottom = face_bbox
    face_area = float(max(face_right - face_left, 1) * max(face_bottom - face_top, 1))
    face_center_x = (face_left + face_right) / 2
    face_center_y = (face_top + face_bottom) / 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.06 or area > image_area * 0.92:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        x, y, rect_w, rect_h = cv2.boundingRect(approximation)
        if rect_w <= 0 or rect_h <= 0:
            continue
        if rect_w > width * 0.96 and rect_h > height * 0.96:
            continue

        rect_area = float(rect_w * rect_h)
        if rect_area < face_area * 1.5:
            continue

        contains_center = (x <= face_center_x <= x + rect_w) and (y <= face_center_y <= y + rect_h)
        contains_face = (
            x <= face_left
            and y <= face_top
            and x + rect_w >= face_right
            and y + rect_h >= face_bottom
        )
        if not (contains_center and contains_face):
            continue

        aspect_ratio = rect_w / max(rect_h, 1)
        if aspect_ratio < 0.35 or aspect_ratio > 2.9:
            continue

        rectangularity = min(1.0, area / max(rect_area, 1.0))
        corner_score = 1.0 if len(approximation) == 4 else 0.75
        rect_area_ratio = min(1.0, rect_area / max(image_area * 0.45, 1.0))
        edge_roi = edges[y : y + rect_h, x : x + rect_w]
        edge_density = float(np.mean(edge_roi > 0)) if edge_roi.size else 0.0
        edge_score = min(1.0, edge_density / 0.08)
        score = (
            0.35 * rectangularity
            + 0.25 * corner_score
            + 0.20 * rect_area_ratio
            + 0.20 * edge_score
        )
        best_score = max(best_score, float(max(0.0, min(1.0, score))))

    border_line_score = _score_screen_border_lines(edges, face_bbox, gray)
    return float(max(best_score, border_line_score))


def _calculate_background_features(image, face_bbox: tuple[int, int, int, int]) -> dict[str, float]:
    height, width = image.shape[:2]
    left, top, right, bottom = face_bbox
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    mask = np.ones((height, width), dtype=bool)
    pad_x = int((right - left) * 0.15)
    pad_y = int((bottom - top) * 0.15)
    mask[
        max(0, top - pad_y) : min(height, bottom + pad_y),
        max(0, left - pad_x) : min(width, right + pad_x),
    ] = False

    face_values = value[top:bottom, left:right]
    background_values = value[mask]
    if face_values.size == 0 or background_values.size == 0:
        return {
            "face_brightness": 0.0,
            "background_brightness": 0.0,
            "face_background_delta": 0.0,
        }

    face_brightness = float(np.mean(face_values))
    background_brightness = float(np.mean(background_values))
    return {
        "face_brightness": face_brightness,
        "background_brightness": background_brightness,
        "face_background_delta": float(abs(face_brightness - background_brightness)),
    }


def _calculate_frequency_features(gray_crop) -> dict[str, float]:
    if gray_crop is None or gray_crop.size == 0:
        return {
            "frequency_peak_score": 0.0,
            "high_frequency_energy": 0.0,
            "horizontal_band_energy": 0.0,
            "vertical_band_energy": 0.0,
            "isolated_peak_score": 0.0,
            "frequency_grid_score": 0.0,
            "frequency_directional_score": 0.0,
            "frequency_ring_contrast": 0.0,
            "moire_score": 0.0,
        }

    resized = cv2.resize(gray_crop, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    window = np.outer(np.hanning(resized.shape[0]), np.hanning(resized.shape[1])).astype(np.float32)
    normalized = (resized - float(np.mean(resized))) * window
    spectrum = np.fft.fftshift(np.fft.fft2(normalized))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)

    size = magnitude.shape[0]
    center = size // 2
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    high_mask = radius >= 18
    very_high_mask = radius >= 34
    if not np.any(high_mask):
        return {
            "frequency_peak_score": 0.0,
            "high_frequency_energy": 0.0,
            "horizontal_band_energy": 0.0,
            "vertical_band_energy": 0.0,
            "isolated_peak_score": 0.0,
            "frequency_grid_score": 0.0,
            "frequency_directional_score": 0.0,
            "frequency_ring_contrast": 0.0,
            "moire_score": 0.0,
        }

    high_values = magnitude[high_mask]
    very_high_values = magnitude[very_high_mask] if np.any(very_high_mask) else high_values
    frequency_peak_score = float(np.max(high_values) / (np.mean(high_values) + 1e-6))
    high_frequency_energy = float(np.mean(very_high_values) / (np.mean(magnitude) + 1e-6))
    horizontal_mask = np.zeros_like(high_mask, dtype=bool)
    horizontal_mask[center - 2 : center + 3, :] = True
    vertical_mask = np.zeros_like(high_mask, dtype=bool)
    vertical_mask[:, center - 2 : center + 3] = True
    horizontal_values = magnitude[horizontal_mask & high_mask]
    vertical_values = magnitude[vertical_mask & high_mask]
    horizontal_band_energy = float(np.mean(horizontal_values) / (np.mean(high_values) + 1e-6)) if horizontal_values.size else 0.0
    vertical_band_energy = float(np.mean(vertical_values) / (np.mean(high_values) + 1e-6)) if vertical_values.size else 0.0

    local_mean = cv2.blur(magnitude, (9, 9))
    local_peak_map = magnitude / (local_mean + 1e-6)
    isolated_peak_score = float(np.max(local_peak_map[high_mask])) if np.any(high_mask) else 0.0

    mid_mask = (radius >= 12) & (radius < 30)
    ring_mid = magnitude[mid_mask]
    ring_high = magnitude[very_high_mask] if np.any(very_high_mask) else high_values
    frequency_ring_contrast = float(np.mean(ring_high) / (np.mean(ring_mid) + 1e-6)) if ring_mid.size else 0.0
    band_max = max(horizontal_band_energy, vertical_band_energy)
    band_min = min(horizontal_band_energy, vertical_band_energy)
    frequency_directional_score = float(max(0.0, band_max - band_min))
    frequency_grid_score = float(min(horizontal_band_energy, vertical_band_energy))

    peak_component = min(1.0, max(0.0, (frequency_peak_score - 2.2) / 4.8))
    energy_component = min(1.0, max(0.0, (high_frequency_energy - 0.82) / 0.55))
    band_component = min(1.0, max(0.0, (max(horizontal_band_energy, vertical_band_energy) - 0.95) / 0.75))
    isolated_peak_component = min(1.0, max(0.0, (isolated_peak_score - 1.28) / 1.20))
    grid_component = min(1.0, max(0.0, (frequency_grid_score - 0.70) / 0.45))
    ring_component = min(1.0, max(0.0, (frequency_ring_contrast - 0.82) / 0.40))
    moire_score = float(
        0.30 * peak_component
        + 0.20 * energy_component
        + 0.15 * band_component
        + 0.20 * isolated_peak_component
        + 0.10 * grid_component
        + 0.05 * ring_component
    )

    return {
        "frequency_peak_score": frequency_peak_score,
        "high_frequency_energy": high_frequency_energy,
        "horizontal_band_energy": horizontal_band_energy,
        "vertical_band_energy": vertical_band_energy,
        "isolated_peak_score": isolated_peak_score,
        "frequency_grid_score": frequency_grid_score,
        "frequency_directional_score": frequency_directional_score,
        "frequency_ring_contrast": frequency_ring_contrast,
        "moire_score": moire_score,
    }


def _extract_skin_frequency_patches(gray_crop) -> list:
    if gray_crop is None or gray_crop.size == 0:
        return []

    height, width = gray_crop.shape[:2]
    if height < 72 or width < 72:
        return []

    regions = [
        (0.34, 0.12, 0.66, 0.28),  # testa central
        (0.18, 0.45, 0.39, 0.70),  # bochecha esquerda
        (0.61, 0.45, 0.82, 0.70),  # bochecha direita
    ]
    patches = []
    for x1_ratio, y1_ratio, x2_ratio, y2_ratio in regions:
        x1 = max(0, min(width - 1, int(width * x1_ratio)))
        y1 = max(0, min(height - 1, int(height * y1_ratio)))
        x2 = max(x1 + 1, min(width, int(width * x2_ratio)))
        y2 = max(y1 + 1, min(height, int(height * y2_ratio)))
        patch = gray_crop[y1:y2, x1:x2]
        if patch.size and patch.shape[0] >= 24 and patch.shape[1] >= 24:
            patches.append(patch)

    return patches


def _calculate_skin_frequency_features(gray_crop) -> dict[str, float]:
    patches = _extract_skin_frequency_patches(gray_crop)
    if not patches:
        return {
            "skin_patch_count": 0.0,
            "skin_frequency_peak_score_median": 0.0,
            "skin_high_frequency_energy_median": 0.0,
            "skin_moire_score_median": 0.0,
            "skin_moire_score_max": 0.0,
        }

    patch_features = [_calculate_frequency_features(patch) for patch in patches]
    return {
        "skin_patch_count": float(len(patch_features)),
        "skin_frequency_peak_score_median": float(
            np.median([item["frequency_peak_score"] for item in patch_features])
        ),
        "skin_high_frequency_energy_median": float(
            np.median([item["high_frequency_energy"] for item in patch_features])
        ),
        "skin_moire_score_median": float(np.median([item["moire_score"] for item in patch_features])),
        "skin_moire_score_max": float(np.max([item["moire_score"] for item in patch_features])),
    }


def _calculate_lbp_features(gray_crop) -> dict[str, float]:
    if gray_crop is None or gray_crop.size == 0:
        return {
            "lbp_entropy": 0.0,
            "lbp_uniformity": 0.0,
            "lbp_local_contrast": 0.0,
        }

    resized = cv2.resize(gray_crop, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    center = resized[1:-1, 1:-1]
    neighbors = [
        resized[:-2, :-2],
        resized[:-2, 1:-1],
        resized[:-2, 2:],
        resized[1:-1, 2:],
        resized[2:, 2:],
        resized[2:, 1:-1],
        resized[2:, :-2],
        resized[1:-1, :-2],
    ]

    lbp = np.zeros(center.shape, dtype=np.uint8)
    contrast_acc = np.zeros(center.shape, dtype=np.float32)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
        contrast_acc += np.abs(neighbor - center)

    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
    hist_sum = float(np.sum(hist))
    if hist_sum <= 0:
        return {
            "lbp_entropy": 0.0,
            "lbp_uniformity": 0.0,
            "lbp_local_contrast": 0.0,
        }

    probability = hist / hist_sum
    non_zero = probability[probability > 0]
    entropy = float(-np.sum(non_zero * np.log2(non_zero)) / 8.0)
    uniformity = float(np.sum(probability * probability))
    local_contrast = float(np.mean(contrast_acc / 8.0) / 255.0)
    return {
        "lbp_entropy": entropy,
        "lbp_uniformity": uniformity,
        "lbp_local_contrast": local_contrast,
    }


def _calculate_skin_lbp_features(gray_crop) -> dict[str, float]:
    patches = _extract_skin_frequency_patches(gray_crop)
    if not patches:
        return {
            "skin_lbp_entropy": 0.0,
            "skin_lbp_uniformity": 0.0,
            "skin_lbp_local_contrast": 0.0,
        }

    patch_features = [_calculate_lbp_features(patch) for patch in patches]
    return {
        "skin_lbp_entropy": float(np.median([item["lbp_entropy"] for item in patch_features])),
        "skin_lbp_uniformity": float(np.median([item["lbp_uniformity"] for item in patch_features])),
        "skin_lbp_local_contrast": float(np.median([item["lbp_local_contrast"] for item in patch_features])),
    }


def _calculate_spoof_frame_features(image, landmarks) -> dict[str, Any] | None:
    face_bbox = _extract_face_bbox(image, landmarks)
    if face_bbox is None:
        return None

    left, top, right, bottom = face_bbox
    crop = image[top:bottom, left:right]
    if crop is None:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float(np.mean((value > 245) & (saturation < 45)))
    overexposed_ratio = float(np.mean(value > 245))
    edge_density = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
    background_features = _calculate_background_features(image, face_bbox)
    screen_rectangle_score = _score_screen_rectangle(image, face_bbox)
    frequency_features = _calculate_frequency_features(gray)
    skin_frequency_features = _calculate_skin_frequency_features(gray)
    lbp_features = _calculate_lbp_features(gray)
    skin_lbp_features = _calculate_skin_lbp_features(gray)

    return {
        "laplacian_var": laplacian_var,
        "glare_ratio": glare_ratio,
        "overexposed_ratio": overexposed_ratio,
        "brightness_mean": float(np.mean(value) / 255.0),
        "brightness_p95": float(np.percentile(value, 95) / 255.0),
        "contrast_std": float(np.std(gray) / 255.0),
        "edge_density": edge_density,
        "screen_rectangle_score": screen_rectangle_score,
        **frequency_features,
        **skin_frequency_features,
        **lbp_features,
        **skin_lbp_features,
        **background_features,
    }


def _build_default_spoof_analysis(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "passed": not enabled,
        "score": 0.0 if enabled else 1.0,
        "threshold": get_settings().liveness_anti_spoof_threshold,
        "reasons": ["no_valid_face_frames"] if enabled else [],
        "signals": {},
    }


def _combine_spoof_analyses(heuristic_analysis: dict[str, Any], model_analysis: dict[str, Any]) -> dict[str, Any]:
    model_enabled = bool(model_analysis.get("enabled"))
    heuristic_enabled = bool(heuristic_analysis.get("enabled"))
    passed = bool(heuristic_analysis.get("passed", True)) and (
        bool(model_analysis.get("passed", True)) if model_enabled else True
    )
    reasons = [
        *heuristic_analysis.get("reasons", []),
        *model_analysis.get("reasons", []),
    ]
    model_score = float(model_analysis.get("score", 1.0)) if model_enabled else 1.0
    heuristic_score = float(heuristic_analysis.get("score", 1.0)) if heuristic_enabled else 1.0

    return {
        "enabled": heuristic_enabled or model_enabled,
        "passed": passed,
        "score": min(heuristic_score, model_score),
        "threshold": heuristic_analysis.get("threshold"),
        "reasons": reasons,
        "heuristic": heuristic_analysis,
        "model": model_analysis,
    }


def _mark_spoof_analysis_failed(spoof_analysis: dict[str, Any], reason: str) -> None:
    spoof_analysis["passed"] = False
    spoof_analysis["score"] = min(float(spoof_analysis.get("score", 1.0)), 0.0)
    reasons = spoof_analysis.setdefault("reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def _replace_model_failure_with_quality_reason(model_analysis: dict[str, Any], reason: str) -> None:
    model_analysis["passed"] = False
    model_analysis["score"] = min(float(model_analysis.get("score", 1.0)), 0.0)
    model_analysis["reasons"] = [reason]


def _analyze_spoof_signals(
    *,
    settings,
    valid_frames: int,
    total_frames: int,
    ear_values: list[float],
    mouth_open_abs_delta: float,
    smile_abs_delta: float,
    range_ratio: float,
    head_left_delta: float,
    head_right_delta: float,
    head_up_delta: float,
    head_down_delta: float,
    frame_spoof_features: list[dict[str, Any]],
    compensated_motion: dict[str, float] | None = None,
) -> dict[str, Any]:
    threshold = settings.liveness_anti_spoof_threshold
    if not settings.liveness_anti_spoof_enabled:
        return {
            "enabled": False,
            "passed": True,
            "score": 1.0,
            "threshold": threshold,
            "reasons": [],
            "signals": {},
        }

    ear_range = float(max(ear_values) - min(ear_values)) if ear_values else 0.0
    internal_motion = float(ear_range + mouth_open_abs_delta + smile_abs_delta)
    compensated_motion = compensated_motion or {}
    compensated_local_motion = float(compensated_motion.get("compensated_local_motion", 0.0))
    compensated_mouth_motion = float(compensated_motion.get("compensated_mouth_motion", 0.0))
    compensated_eye_motion = float(compensated_motion.get("compensated_eye_motion", 0.0))
    geometric_motion = float(
        range_ratio
        + max(head_left_delta, head_right_delta)
        + max(head_up_delta, head_down_delta)
    )
    median_laplacian = float(np.median([item["laplacian_var"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_glare_ratio = float(np.median([item["glare_ratio"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_edge_density = float(np.median([item["edge_density"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_brightness = float(np.median([item["brightness_mean"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_brightness_p95 = float(np.median([item["brightness_p95"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_contrast = float(np.median([item["contrast_std"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_overexposed_ratio = float(np.median([item["overexposed_ratio"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_screen_rectangle_score = float(np.median([item["screen_rectangle_score"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_face_background_delta = float(np.median([item["face_background_delta"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_moire_score = float(np.median([item["moire_score"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_frequency_peak_score = float(np.median([item["frequency_peak_score"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_high_frequency_energy = float(np.median([item["high_frequency_energy"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_horizontal_band_energy = float(np.median([item["horizontal_band_energy"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_vertical_band_energy = float(np.median([item["vertical_band_energy"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_isolated_peak_score = float(np.median([item["isolated_peak_score"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_frequency_grid_score = float(np.median([item["frequency_grid_score"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_frequency_directional_score = float(np.median([item["frequency_directional_score"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_frequency_ring_contrast = float(np.median([item["frequency_ring_contrast"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_skin_patch_count = float(np.median([item["skin_patch_count"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_skin_moire_score = float(np.median([item["skin_moire_score_median"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    max_skin_moire_score = float(np.max([item["skin_moire_score_max"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_skin_frequency_peak_score = float(np.median([item["skin_frequency_peak_score_median"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_skin_high_frequency_energy = float(np.median([item["skin_high_frequency_energy_median"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_lbp_entropy = float(np.median([item["lbp_entropy"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_lbp_uniformity = float(np.median([item["lbp_uniformity"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_lbp_local_contrast = float(np.median([item["lbp_local_contrast"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_skin_lbp_entropy = float(np.median([item["skin_lbp_entropy"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_skin_lbp_uniformity = float(np.median([item["skin_lbp_uniformity"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_skin_lbp_local_contrast = float(np.median([item["skin_lbp_local_contrast"] for item in frame_spoof_features])) if frame_spoof_features else 0.0

    score = 1.0
    reasons: list[str] = []

    if valid_frames < max(3, int(total_frames * 0.5)):
        score -= 0.25
        reasons.append("few_valid_face_frames")

    planar_motion_suspected = (
        geometric_motion >= ANTI_SPOOF_GEOMETRIC_MOTION_MIN
        and internal_motion < ANTI_SPOOF_INTERNAL_MOTION_MIN
    )
    if planar_motion_suspected:
        score -= ANTI_SPOOF_PLANAR_MOTION_PENALTY
        reasons.append("planar_motion_without_internal_face_change")

    if median_laplacian and median_laplacian < ANTI_SPOOF_LOW_TEXTURE_LAPLACIAN:
        score -= 0.12
        reasons.append("low_face_texture")

    compensated_motion_too_low = compensated_local_motion < ANTI_SPOOF_COMPENSATED_LOCAL_MOTION_MIN
    close_screen_suspected = (
        median_face_background_delta >= ANTI_SPOOF_CLOSE_SCREEN_FACE_BACKGROUND_MIN
        and compensated_local_motion <= ANTI_SPOOF_CLOSE_SCREEN_LOCAL_MOTION_MAX
    )

    if median_screen_rectangle_score >= ANTI_SPOOF_SCREEN_RECTANGLE_HARD_MIN:
        score -= 0.65
        reasons.append("screen_rectangle_detected")
    elif (
        median_screen_rectangle_score >= ANTI_SPOOF_SCREEN_RECTANGLE_MIN
        and (internal_motion < ANTI_SPOOF_INTERNAL_MOTION_MIN or compensated_motion_too_low)
    ):
        score -= 0.55
        reasons.append("screen_rectangle_detected")

    if close_screen_suspected:
        score -= 0.55
        reasons.append("close_screen_low_local_motion")

    score = float(max(0.0, min(1.0, score)))

    return {
        "enabled": True,
        "passed": bool(score >= threshold),
        "score": score,
        "threshold": threshold,
        "reasons": reasons,
        "signals": {
            "internal_motion": internal_motion,
            "internal_motion_min": ANTI_SPOOF_INTERNAL_MOTION_MIN,
            "compensated_local_motion": compensated_local_motion,
            "compensated_local_motion_min": ANTI_SPOOF_COMPENSATED_LOCAL_MOTION_MIN,
            "close_screen_local_motion_max": ANTI_SPOOF_CLOSE_SCREEN_LOCAL_MOTION_MAX,
            "compensated_mouth_motion": compensated_mouth_motion,
            "compensated_eye_motion": compensated_eye_motion,
            "geometric_motion": geometric_motion,
            "geometric_motion_min": ANTI_SPOOF_GEOMETRIC_MOTION_MIN,
            "ear_range": ear_range,
            "mouth_open_abs_delta": mouth_open_abs_delta,
            "smile_abs_delta": smile_abs_delta,
            "median_laplacian_var": median_laplacian,
            "median_glare_ratio": median_glare_ratio,
            "median_overexposed_ratio": median_overexposed_ratio,
            "median_brightness": median_brightness,
            "median_brightness_p95": median_brightness_p95,
            "median_contrast": median_contrast,
            "median_edge_density": median_edge_density,
            "median_screen_rectangle_score": median_screen_rectangle_score,
            "screen_rectangle_min": ANTI_SPOOF_SCREEN_RECTANGLE_MIN,
            "screen_rectangle_hard_min": ANTI_SPOOF_SCREEN_RECTANGLE_HARD_MIN,
            "median_face_background_delta": median_face_background_delta,
            "close_screen_face_background_min": ANTI_SPOOF_CLOSE_SCREEN_FACE_BACKGROUND_MIN,
            "median_moire_score": median_moire_score,
            "median_frequency_peak_score": median_frequency_peak_score,
            "median_high_frequency_energy": median_high_frequency_energy,
            "median_horizontal_band_energy": median_horizontal_band_energy,
            "median_vertical_band_energy": median_vertical_band_energy,
            "median_isolated_peak_score": median_isolated_peak_score,
            "median_frequency_grid_score": median_frequency_grid_score,
            "median_frequency_directional_score": median_frequency_directional_score,
            "median_frequency_ring_contrast": median_frequency_ring_contrast,
            "median_skin_patch_count": median_skin_patch_count,
            "median_skin_moire_score": median_skin_moire_score,
            "max_skin_moire_score": max_skin_moire_score,
            "median_skin_frequency_peak_score": median_skin_frequency_peak_score,
            "median_skin_high_frequency_energy": median_skin_high_frequency_energy,
            "median_lbp_entropy": median_lbp_entropy,
            "median_lbp_uniformity": median_lbp_uniformity,
            "median_lbp_local_contrast": median_lbp_local_contrast,
            "median_skin_lbp_entropy": median_skin_lbp_entropy,
            "median_skin_lbp_uniformity": median_skin_lbp_uniformity,
            "median_skin_lbp_local_contrast": median_skin_lbp_local_contrast,
            "valid_frame_ratio": float(valid_frames / total_frames) if total_frames else 0.0,
        },
    }


def _build_default_checks() -> dict[str, dict[str, Any]]:
    return {
        "blink": {"passed": False, "count": 0, "target": 1},
        "distance": {"passed": False, "range_ratio": 0.0, "target": DISTANCE_TARGET_RANGE_RATIO},
        "smile": {"passed": False, "max_ratio": 0.0, "target": SMILE_TARGET_FACTOR},
        "smile_hold": {"passed": False, "max_ratio": 0.0, "target": SMILE_TARGET_FACTOR},
        "mouth_open": {"passed": False, "max_ratio": 0.0, "target": MOUTH_OPEN_TARGET_FACTOR},
        "head_left": {"passed": False, "delta": 0.0, "target": HEAD_TURN_TARGET_DELTA},
        "head_right": {"passed": False, "delta": 0.0, "target": HEAD_TURN_TARGET_DELTA},
        "head_up": {"passed": False, "delta": 0.0, "target": HEAD_PITCH_TARGET_DELTA},
        "head_down": {"passed": False, "delta": 0.0, "target": HEAD_PITCH_TARGET_DELTA},
    }


def _evaluate_challenges(
    session_definition: dict[str, Any] | None,
    aggregate_checks: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not session_definition or not session_definition.get("challenges"):
        required_checks = [name for name in aggregate_checks.keys() if name in SUPPORTED_LIVENESS_REQUIREMENTS]
        return [], required_checks

    challenge_results: list[dict[str, Any]] = []
    required_union: list[str] = []

    for challenge in session_definition.get("challenges", []):
        requirements = [
            requirement
            for requirement in challenge.get("requirements", [])
            if requirement in SUPPORTED_LIVENESS_REQUIREMENTS and requirement in aggregate_checks
        ]
        targets = challenge.get("targets", {})
        checks: dict[str, dict[str, Any]] = {}

        for requirement in requirements:
            base = dict(aggregate_checks[requirement])
            if requirement in targets:
                base["target"] = targets[requirement]

                if requirement == "blink":
                    base["passed"] = base.get("count", 0) >= targets[requirement]
                elif requirement == "distance":
                    base["passed"] = base.get("range_ratio", 0.0) >= targets[requirement]
                elif requirement == "smile":
                    base["passed"] = base.get("max_ratio", 0.0) >= targets[requirement]
                elif requirement == "smile_hold":
                    base["passed"] = (
                        base.get("max_ratio", 0.0) >= targets[requirement]
                        and base.get("max_consecutive_frames", 0) >= SMILE_HOLD_MIN_CONSECUTIVE_FRAMES
                    )
                elif requirement == "mouth_open":
                    base["passed"] = base.get("max_ratio", 0.0) >= targets[requirement]
                elif requirement in {"head_left", "head_right", "head_up", "head_down"}:
                    base["passed"] = base.get("delta", 0.0) >= targets[requirement]

            checks[requirement] = base
            if requirement not in required_union:
                required_union.append(requirement)

        challenge_results.append(
            {
                "id": challenge.get("id", "unknown"),
                "label": challenge.get("label", challenge.get("id", "Desafio")),
                "requirements": requirements,
                "passed": bool(requirements) and all(check["passed"] for check in checks.values()),
                "checks": checks,
            }
        )

    return challenge_results, required_union


def _should_enforce_backend_challenges(session_definition: dict[str, Any] | None) -> bool:
    """Only enforce challenge replay when the client did not use a signed UI session.

    In the browser flow the user completes the dynamic challenges before the
    short recording starts. The backend recording is then used for anti-spoof
    verification, so re-requiring every challenge in those later frames causes
    false rejections for real users.
    """
    return not (session_definition and session_definition.get("signature"))


def analyze_liveness_frames(
    frame_bytes_list: list[bytes],
    required_checks: list[str] | None = None,
    session_definition: dict[str, Any] | None = None,
    client_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_liveness_dependencies()
    runtime_settings = get_settings()

    if required_checks is None:
        required_checks = ["blink", "distance", "smile"]
    client_context = client_context or {}
    capture_source = client_context.get("capture_source_guess")
    device_class = client_context.get("device_class")
    is_mobile_capture = bool(capture_source == "phone_camera" or device_class in {"mobile", "tablet"})

    all_frame_metrics: list[FrameMetrics | None] = []
    all_frame_landmarks: list[Any | None] = []
    frame_spoof_features: list[dict[str, Any]] = []
    antispoof_crops: list[Any] = []
    total_frames = len(frame_bytes_list)

    face_detector = _create_face_detector()

    try:
        for frame_bytes in frame_bytes_list:
            img_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = face_detector.detect_landmarks(rgb)
            if not landmarks:
                all_frame_metrics.append(None)
                all_frame_landmarks.append(None)
                continue

            frame_metrics = _extract_metrics(landmarks)
            if frame_metrics:
                all_frame_metrics.append(frame_metrics)
                all_frame_landmarks.append(landmarks)
                spoof_features = _calculate_spoof_frame_features(image, landmarks)
                if spoof_features:
                    frame_spoof_features.append(spoof_features)
                face_crop = _extract_face_crop(image, landmarks)
                if face_crop is not None:
                    antispoof_crops.append(face_crop)
            else:
                all_frame_metrics.append(None)
                all_frame_landmarks.append(None)
    finally:
        face_detector.close()

    metrics = [metric for metric in all_frame_metrics if metric is not None]
    valid_frames = len(metrics)
    if valid_frames == 0:
        checks = _build_default_checks()
        challenge_results, resolved_required = _evaluate_challenges(session_definition, checks)
        model_spoof_analysis = analyze_antispoof_crops([], runtime_settings)
        spoof_analysis = _combine_spoof_analyses(
            _build_default_spoof_analysis(runtime_settings.liveness_anti_spoof_enabled),
            model_spoof_analysis,
        )
        return {
            "passed": False,
            "checks": checks,
            "required_checks": resolved_required,
            "session": session_definition,
            "challenge_results": challenge_results,
            "frame_metrics": _build_temporal_frame_metrics(all_frame_metrics),
            "spoof_analysis": spoof_analysis,
            "summary": {
                "total_frames": total_frames,
                "valid_frames": 0,
                "face_detected_frames": 0,
                "anti_spoof": spoof_analysis,
                "temporal_consistency": {
                    "mean_movement_score": 0.0,
                    "max_movement_score": 0.0,
                    "low_motion_ratio": 1.0,
                    "looks_static": True,
                },
            },
        }

    indexed_metrics = [(index, metric) for index, metric in enumerate(all_frame_metrics) if metric is not None]
    ear_values = [m.ear for m in metrics]
    face_widths = [m.face_width for m in metrics]
    smile_ratios = [m.smile_ratio for m in metrics]
    mouth_open_ratios = [m.mouth_open_ratio for m in metrics]
    head_yaws = [m.head_yaw for m in metrics]
    head_pitches = [m.head_pitch for m in metrics]

    blink_events, ear_baseline, ear_threshold = _detect_blink_events(indexed_metrics)
    blink_count = len(blink_events)
    blink_passed = blink_count >= 1

    width_baseline = float(np.median(face_widths)) if face_widths else 0.0
    width_range = float(max(face_widths) - min(face_widths)) if face_widths else 0.0
    range_ratio = (width_range / width_baseline) if width_baseline > 1e-6 else 0.0
    width_peak_frame = indexed_metrics[int(np.argmax(np.array(face_widths, dtype=np.float32)))][0] if face_widths else None
    width_min_frame = indexed_metrics[int(np.argmin(np.array(face_widths, dtype=np.float32)))][0] if face_widths else None
    distance_passed = range_ratio >= DISTANCE_TARGET_RANGE_RATIO

    smile_baseline = float(np.percentile(np.array(smile_ratios, dtype=np.float32), 35)) if smile_ratios else 0.0
    smile_max = float(max(smile_ratios)) if smile_ratios else 0.0
    smile_max_index = int(np.argmax(np.array(smile_ratios, dtype=np.float32))) if smile_ratios else 0
    smile_peak_frame = indexed_metrics[smile_max_index][0] if smile_ratios else None
    smile_abs_delta = float(smile_max - smile_baseline) if smile_ratios else 0.0
    smile_ratio_gain = (smile_max / smile_baseline) if smile_baseline > 1e-6 else 0.0
    intermediate_threshold = smile_baseline * SMILE_INTERMEDIATE_FACTOR if smile_baseline > 1e-6 else 0.0
    intermediate_reached = any(ratio >= intermediate_threshold for ratio in smile_ratios) if smile_ratios else False
    qualifying_smile_frames = 0
    max_qualifying_smile_frames = 0
    for ratio_value in smile_ratios:
        ratio_gain_value = (ratio_value / smile_baseline) if smile_baseline > 1e-6 else 0.0
        abs_delta_value = ratio_value - smile_baseline
        if ratio_gain_value >= SMILE_TARGET_FACTOR and abs_delta_value >= SMILE_MIN_ABS_DELTA:
            qualifying_smile_frames += 1
            max_qualifying_smile_frames = max(max_qualifying_smile_frames, qualifying_smile_frames)
        else:
            qualifying_smile_frames = 0
    smile_passed = bool(
        intermediate_reached
        and smile_ratio_gain >= SMILE_TARGET_FACTOR
        and smile_abs_delta >= SMILE_MIN_ABS_DELTA
    )
    smile_hold_passed = bool(
        smile_passed
        and max_qualifying_smile_frames >= SMILE_HOLD_MIN_CONSECUTIVE_FRAMES
    )

    mouth_open_baseline = float(np.percentile(np.array(mouth_open_ratios, dtype=np.float32), 35)) if mouth_open_ratios else 0.0
    mouth_open_max = float(max(mouth_open_ratios)) if mouth_open_ratios else 0.0
    mouth_open_max_index = int(np.argmax(np.array(mouth_open_ratios, dtype=np.float32))) if mouth_open_ratios else 0
    mouth_open_peak_frame = indexed_metrics[mouth_open_max_index][0] if mouth_open_ratios else None
    mouth_open_abs_delta = float(mouth_open_max - mouth_open_baseline) if mouth_open_ratios else 0.0
    mouth_open_ratio_gain = (mouth_open_max / mouth_open_baseline) if mouth_open_baseline > 1e-6 else 0.0
    mouth_open_intermediate_threshold = (
        mouth_open_baseline * MOUTH_OPEN_INTERMEDIATE_FACTOR if mouth_open_baseline > 1e-6 else 0.0
    )
    mouth_open_intermediate_reached = (
        any(ratio >= mouth_open_intermediate_threshold for ratio in mouth_open_ratios) if mouth_open_ratios else False
    )
    mouth_open_passed = bool(
        mouth_open_intermediate_reached
        and mouth_open_ratio_gain >= MOUTH_OPEN_TARGET_FACTOR
        and mouth_open_abs_delta >= MOUTH_OPEN_MIN_ABS_DELTA
    )

    head_yaw_baseline = float(indexed_metrics[0][1].head_yaw) if indexed_metrics else 0.0
    head_pitch_baseline = float(indexed_metrics[0][1].head_pitch) if indexed_metrics else 0.0
    head_yaw_min = float(min(head_yaws)) if head_yaws else 0.0
    head_yaw_max = float(max(head_yaws)) if head_yaws else 0.0
    head_pitch_min = float(min(head_pitches)) if head_pitches else 0.0
    head_pitch_max = float(max(head_pitches)) if head_pitches else 0.0
    head_left_delta = max(0.0, head_yaw_baseline - head_yaw_min)
    head_right_delta = max(0.0, head_yaw_max - head_yaw_baseline)
    head_up_delta = max(0.0, head_pitch_baseline - head_pitch_min)
    head_down_delta = max(0.0, head_pitch_max - head_pitch_baseline)
    head_left_passed = head_left_delta >= HEAD_TURN_TARGET_DELTA
    head_right_passed = head_right_delta >= HEAD_TURN_TARGET_DELTA
    head_up_passed = head_up_delta >= HEAD_PITCH_TARGET_DELTA
    head_down_passed = head_down_delta >= HEAD_PITCH_TARGET_DELTA
    compensated_motion = _calculate_compensated_local_motion(all_frame_landmarks)
    heuristic_spoof_analysis = _analyze_spoof_signals(
        settings=runtime_settings,
        valid_frames=valid_frames,
        total_frames=total_frames,
        ear_values=ear_values,
        mouth_open_abs_delta=mouth_open_abs_delta,
        smile_abs_delta=smile_abs_delta,
        range_ratio=range_ratio,
        head_left_delta=head_left_delta,
        head_right_delta=head_right_delta,
        head_up_delta=head_up_delta,
        head_down_delta=head_down_delta,
        frame_spoof_features=frame_spoof_features,
        compensated_motion=compensated_motion,
    )
    model_spoof_analysis = analyze_antispoof_crops(antispoof_crops, runtime_settings)
    model_p_spoof_raw = float(model_spoof_analysis.get("p_spoof", 0.0))
    model_p_spoof = model_p_spoof_raw
    review_threshold = float(runtime_settings.liveness_onnx_antispoof_review_threshold)
    mobile_hard_threshold = float(runtime_settings.liveness_mobile_onnx_hard_threshold)
    model_decision_enabled = bool(runtime_settings.liveness_onnx_antispoof_decision_enabled)

    if model_spoof_analysis.get("enabled") and not model_decision_enabled:
        model_spoof_analysis["decision_enabled"] = False
        model_spoof_analysis["diagnostic_passed"] = model_spoof_analysis.get("passed")
        model_spoof_analysis["diagnostic_reasons"] = list(model_spoof_analysis.get("reasons", []))
        model_spoof_analysis["passed"] = True
        model_spoof_analysis["score"] = 1.0
        model_spoof_analysis["reasons"] = []
    elif model_spoof_analysis.get("enabled"):
        model_spoof_analysis["decision_enabled"] = True

    if (
        model_decision_enabled
        and
        is_mobile_capture
        and model_spoof_analysis.get("enabled")
        and model_spoof_analysis.get("threshold") is not None
        and float(model_spoof_analysis["threshold"]) <= model_p_spoof_raw < mobile_hard_threshold
    ):
        _replace_model_failure_with_quality_reason(model_spoof_analysis, "mobile_antispoof_uncertain")
        model_spoof_analysis["mobile_policy"] = {
            "enabled": True,
            "hard_threshold": mobile_hard_threshold,
            "capture_source": capture_source,
            "device_class": device_class,
        }

    median_face_width = float(np.median(face_widths)) if face_widths else 0.0
    median_laplacian = (
        float(np.median([item["laplacian_var"] for item in frame_spoof_features]))
        if frame_spoof_features
        else 0.0
    )
    face_too_small_for_model = median_face_width < float(runtime_settings.liveness_onnx_min_face_width_ratio)
    too_blurry_for_model = bool(median_laplacian and median_laplacian < ANTI_SPOOF_LOW_TEXTURE_LAPLACIAN)
    model_quality_limited = bool(face_too_small_for_model or too_blurry_for_model)
    model_has_strong_spoof_signal = bool(
        model_spoof_analysis.get("enabled")
        and model_spoof_analysis.get("threshold") is not None
        and float(model_spoof_analysis.get("p_spoof", 0.0)) >= float(model_spoof_analysis["threshold"])
    )
    model_has_hard_spoof_signal = bool(
        model_spoof_analysis.get("enabled")
        and float(model_spoof_analysis.get("p_spoof", 0.0))
        >= float(runtime_settings.liveness_onnx_quality_override_threshold)
    )
    quality_details = {
        "face_width": median_face_width,
        "min_face_width": float(runtime_settings.liveness_onnx_min_face_width_ratio),
        "median_laplacian_var": median_laplacian,
        "face_too_small": face_too_small_for_model,
        "too_blurry": too_blurry_for_model,
        "limited": model_quality_limited,
    }
    if model_quality_limited and model_spoof_analysis.get("enabled"):
        model_spoof_analysis["quality"] = quality_details

    quality_should_block = bool(model_decision_enabled and model_quality_limited and model_p_spoof >= review_threshold)
    if quality_should_block and model_spoof_analysis.get("enabled") and not model_has_hard_spoof_signal:
        _replace_model_failure_with_quality_reason(model_spoof_analysis, "insufficient_antispoof_quality")
        model_spoof_analysis["quality"] = quality_details

    spoof_analysis = _combine_spoof_analyses(heuristic_spoof_analysis, model_spoof_analysis)

    frame_metrics_response = _build_temporal_frame_metrics(all_frame_metrics)
    temporal_consistency = _calculate_temporal_consistency(frame_metrics_response)

    checks = {
        "anti_spoof": {
            "passed": spoof_analysis["passed"],
            "score": spoof_analysis["score"],
            "threshold": spoof_analysis["threshold"],
            "reasons": spoof_analysis["reasons"],
        },
        "anti_spoof_model": {
            "passed": model_spoof_analysis["passed"],
            "score": model_spoof_analysis["score"],
            "p_spoof": model_spoof_analysis["p_spoof"],
            "threshold": model_spoof_analysis["threshold"],
            "aggregation": model_spoof_analysis["aggregation"],
            "reasons": model_spoof_analysis["reasons"],
        },
        "blink": {
            "passed": blink_passed,
            "count": blink_count,
            "target": 1,
            "ear_baseline": ear_baseline,
            "ear_threshold": ear_threshold,
            "events": blink_events,
        },
        "distance": {
            "passed": distance_passed,
            "range_ratio": range_ratio,
            "target": DISTANCE_TARGET_RANGE_RATIO,
            "baseline_face_width": width_baseline,
            "max_face_width": float(max(face_widths)) if face_widths else 0.0,
            "min_face_width": float(min(face_widths)) if face_widths else 0.0,
            "peak_frame": width_peak_frame,
            "min_frame": width_min_frame,
        },
        "smile": {
            "passed": smile_passed,
            "max_ratio": smile_ratio_gain,
            "target": SMILE_TARGET_FACTOR,
            "baseline_ratio": smile_baseline,
            "peak_ratio": smile_max,
            "abs_delta": smile_abs_delta,
            "intermediate_reached": intermediate_reached,
            "intermediate_target_ratio": SMILE_INTERMEDIATE_FACTOR,
            "peak_frame": smile_peak_frame,
        },
        "smile_hold": {
            "passed": smile_hold_passed,
            "max_ratio": smile_ratio_gain,
            "target": SMILE_TARGET_FACTOR,
            "baseline_ratio": smile_baseline,
            "peak_ratio": smile_max,
            "abs_delta": smile_abs_delta,
            "intermediate_reached": intermediate_reached,
            "intermediate_target_ratio": SMILE_INTERMEDIATE_FACTOR,
            "max_consecutive_frames": max_qualifying_smile_frames,
            "hold_target_frames": SMILE_HOLD_MIN_CONSECUTIVE_FRAMES,
            "peak_frame": smile_peak_frame,
        },
        "mouth_open": {
            "passed": mouth_open_passed,
            "max_ratio": mouth_open_ratio_gain,
            "target": MOUTH_OPEN_TARGET_FACTOR,
            "baseline_ratio": mouth_open_baseline,
            "peak_ratio": mouth_open_max,
            "abs_delta": mouth_open_abs_delta,
            "intermediate_reached": mouth_open_intermediate_reached,
            "intermediate_target_ratio": MOUTH_OPEN_INTERMEDIATE_FACTOR,
            "peak_frame": mouth_open_peak_frame,
        },
        "head_left": {
            "passed": head_left_passed,
            "delta": head_left_delta,
            "target": HEAD_TURN_TARGET_DELTA,
            "baseline": head_yaw_baseline,
            "min_yaw": head_yaw_min,
            "max_yaw": head_yaw_max,
        },
        "head_right": {
            "passed": head_right_passed,
            "delta": head_right_delta,
            "target": HEAD_TURN_TARGET_DELTA,
            "baseline": head_yaw_baseline,
            "min_yaw": head_yaw_min,
            "max_yaw": head_yaw_max,
        },
        "head_up": {
            "passed": head_up_passed,
            "delta": head_up_delta,
            "target": HEAD_PITCH_TARGET_DELTA,
            "baseline": head_pitch_baseline,
            "min_pitch": head_pitch_min,
            "max_pitch": head_pitch_max,
        },
        "head_down": {
            "passed": head_down_passed,
            "delta": head_down_delta,
            "target": HEAD_PITCH_TARGET_DELTA,
            "baseline": head_pitch_baseline,
            "min_pitch": head_pitch_min,
            "max_pitch": head_pitch_max,
        },
    }

    challenge_results, challenge_required = _evaluate_challenges(session_definition, checks)
    model_frame_scores = [
        float(score)
        for score in model_spoof_analysis.get("frame_scores", [])
        if isinstance(score, (int, float))
    ]
    model_threshold = model_spoof_analysis.get("threshold")
    model_p_spoof = float(model_spoof_analysis.get("p_spoof", 0.0))
    frame_spike_margin = float(runtime_settings.liveness_onnx_antispoof_frame_spike_margin)
    max_model_frame_score = max(model_frame_scores) if model_frame_scores else 0.0
    high_model_frame_count = (
        sum(1 for score in model_frame_scores if model_threshold is not None and score >= float(model_threshold))
        if model_frame_scores
        else 0
    )
    borderline_model_score = bool(
        model_decision_enabled
        and
        not is_mobile_capture
        and
        model_spoof_analysis.get("enabled")
        and model_spoof_analysis.get("passed", True)
        and model_threshold is not None
        and review_threshold <= model_p_spoof < float(model_threshold)
    )
    strong_frame_spike = bool(
        model_threshold is not None
        and max_model_frame_score >= float(model_threshold) + frame_spike_margin
    )
    repeated_frame_spike = high_model_frame_count >= 2
    frame_spike_in_review_zone = bool(
        borderline_model_score
        and (strong_frame_spike or repeated_frame_spike)
    )
    if frame_spike_in_review_zone:
        _mark_spoof_analysis_failed(spoof_analysis, "borderline_spoof_score_with_high_frame")
        checks["anti_spoof"]["passed"] = False
        checks["anti_spoof"]["score"] = spoof_analysis["score"]
        checks["anti_spoof"]["reasons"] = spoof_analysis["reasons"]

    heuristic_signals = spoof_analysis.get("heuristic", {}).get("signals", {})
    fusion_low_internal_high_delta = bool(
        not is_mobile_capture
        and
        borderline_model_score
        and float(heuristic_signals.get("internal_motion", 1.0))
        <= float(runtime_settings.liveness_fusion_internal_motion_max)
        and float(heuristic_signals.get("median_face_background_delta", 0.0))
        >= float(runtime_settings.liveness_fusion_face_background_delta)
    )
    if fusion_low_internal_high_delta:
        _mark_spoof_analysis_failed(spoof_analysis, "fusion_low_internal_motion_high_face_background_delta")
        checks["anti_spoof"]["passed"] = False
        checks["anti_spoof"]["score"] = spoof_analysis["score"]
        checks["anti_spoof"]["reasons"] = spoof_analysis["reasons"]

    required = [c for c in (required_checks or []) if c in checks]
    if not required:
        required = challenge_required or ["blink", "distance", "smile"]

    enforce_challenges = _should_enforce_backend_challenges(session_definition)
    if challenge_results:
        passed = all(challenge["passed"] for challenge in challenge_results) if enforce_challenges else True
    else:
        passed = all(checks[check]["passed"] for check in required)
    if spoof_analysis["enabled"]:
        passed = bool(passed and spoof_analysis["passed"])

    return {
        "passed": passed,
        "checks": checks,
        "required_checks": required,
        "session": session_definition,
        "challenge_results": challenge_results,
        "frame_metrics": frame_metrics_response,
        "spoof_analysis": spoof_analysis,
        "summary": {
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "face_detected_frames": valid_frames,
            "blink_event_count": blink_count,
            "anti_spoof": spoof_analysis,
            "challenge_results_enforced": enforce_challenges,
            "borderline_review": {
                "enabled": True,
                "review_threshold": review_threshold,
                "triggered": frame_spike_in_review_zone,
                "max_model_frame_score": max_model_frame_score,
                "high_model_frame_count": high_model_frame_count,
                "frame_spike_margin": frame_spike_margin,
                "strong_frame_spike": strong_frame_spike,
                "repeated_frame_spike": repeated_frame_spike,
            },
            "fusion_review": {
                "enabled": True,
                "triggered": fusion_low_internal_high_delta,
                "skipped_for_mobile": is_mobile_capture,
                "internal_motion_max": float(runtime_settings.liveness_fusion_internal_motion_max),
                "face_background_delta_min": float(runtime_settings.liveness_fusion_face_background_delta),
            },
            "mobile_policy": {
                "enabled": is_mobile_capture,
                "hard_threshold": mobile_hard_threshold,
                "capture_source": capture_source,
                "device_class": device_class,
            },
            "antispoof_quality": {
                "limited": model_quality_limited,
                "model_has_strong_spoof_signal": model_has_strong_spoof_signal,
                "model_has_hard_spoof_signal": model_has_hard_spoof_signal,
                "quality_override_threshold": float(runtime_settings.liveness_onnx_quality_override_threshold),
                "face_width": median_face_width,
                "min_face_width": float(runtime_settings.liveness_onnx_min_face_width_ratio),
                "median_laplacian_var": median_laplacian,
                "face_too_small": face_too_small_for_model,
                "too_blurry": too_blurry_for_model,
            },
            "temporal_consistency": temporal_consistency,
        },
    }
