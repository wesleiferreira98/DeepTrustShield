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
ANTI_SPOOF_GEOMETRIC_MOTION_MIN = 0.13
ANTI_SPOOF_LOW_TEXTURE_LAPLACIAN = 12.0
ANTI_SPOOF_HIGH_GLARE_RATIO = 0.08


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


def _extract_face_crop(image, landmarks, margin_ratio: float = 0.18):
    height, width = image.shape[:2]
    xs = [point.x for point in landmarks if 0.0 <= point.x <= 1.0]
    ys = [point.y for point in landmarks if 0.0 <= point.y <= 1.0]
    if not xs or not ys:
        return None

    left = max(0, int((min(xs) - margin_ratio) * width))
    right = min(width, int((max(xs) + margin_ratio) * width))
    top = max(0, int((min(ys) - margin_ratio) * height))
    bottom = min(height, int((max(ys) + margin_ratio) * height))
    if right - left < 24 or bottom - top < 24:
        return None
    return image[top:bottom, left:right]


def _calculate_spoof_frame_features(image, landmarks) -> dict[str, Any] | None:
    crop = _extract_face_crop(image, landmarks)
    if crop is None:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float(np.mean((value > 245) & (saturation < 45)))
    edge_density = float(np.mean(cv2.Canny(gray, 80, 160) > 0))

    return {
        "laplacian_var": laplacian_var,
        "glare_ratio": glare_ratio,
        "edge_density": edge_density,
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
    geometric_motion = float(
        range_ratio
        + max(head_left_delta, head_right_delta)
        + max(head_up_delta, head_down_delta)
    )
    median_laplacian = float(np.median([item["laplacian_var"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_glare_ratio = float(np.median([item["glare_ratio"] for item in frame_spoof_features])) if frame_spoof_features else 0.0
    median_edge_density = float(np.median([item["edge_density"] for item in frame_spoof_features])) if frame_spoof_features else 0.0

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
        score -= 0.50
        reasons.append("planar_motion_without_internal_face_change")

    if median_laplacian and median_laplacian < ANTI_SPOOF_LOW_TEXTURE_LAPLACIAN:
        score -= 0.12
        reasons.append("low_face_texture")

    if median_glare_ratio > ANTI_SPOOF_HIGH_GLARE_RATIO:
        score -= 0.15
        reasons.append("screen_or_reflection_glare")

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
            "geometric_motion": geometric_motion,
            "geometric_motion_min": ANTI_SPOOF_GEOMETRIC_MOTION_MIN,
            "ear_range": ear_range,
            "mouth_open_abs_delta": mouth_open_abs_delta,
            "smile_abs_delta": smile_abs_delta,
            "median_laplacian_var": median_laplacian,
            "median_glare_ratio": median_glare_ratio,
            "median_edge_density": median_edge_density,
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


def analyze_liveness_frames(
    frame_bytes_list: list[bytes],
    required_checks: list[str] | None = None,
    session_definition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_liveness_dependencies()
    runtime_settings = get_settings()

    if required_checks is None:
        required_checks = ["blink", "distance", "smile"]

    all_frame_metrics: list[FrameMetrics | None] = []
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
                continue

            frame_metrics = _extract_metrics(landmarks)
            if frame_metrics:
                all_frame_metrics.append(frame_metrics)
                spoof_features = _calculate_spoof_frame_features(image, landmarks)
                if spoof_features:
                    frame_spoof_features.append(spoof_features)
                face_crop = _extract_face_crop(image, landmarks)
                if face_crop is not None:
                    antispoof_crops.append(face_crop)
            else:
                all_frame_metrics.append(None)
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
    )
    model_spoof_analysis = analyze_antispoof_crops(antispoof_crops, runtime_settings)
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

    required = [c for c in (required_checks or []) if c in checks]
    if not required:
        required = challenge_required or ["blink", "distance", "smile"]

    if challenge_results:
        passed = all(challenge["passed"] for challenge in challenge_results)
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
            "temporal_consistency": temporal_consistency,
        },
    }
