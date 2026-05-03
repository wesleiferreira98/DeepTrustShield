from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from app.config import BASE_DIR


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ground_truth_roots() -> list[Path]:
    configured = os.getenv("AUDIO_MANIPULATION_GROUND_TRUTH_DIRS", "")
    if configured.strip():
        return [Path(item).expanduser().resolve() for item in configured.split(os.pathsep) if item.strip()]
    return [Path(BASE_DIR) / "audio" / "data" / "eval_manipulated"]


def _interval_overlap(left: dict, right: dict) -> float:
    start = max(_safe_float(left.get("start")), _safe_float(right.get("start")))
    end = min(_safe_float(left.get("end")), _safe_float(right.get("end")))
    return max(0.0, end - start)


def _find_ground_truth(audio_hash: str) -> tuple[Path | None, dict | None]:
    for root in _ground_truth_roots():
        if not root.exists():
            continue
        for path in root.rglob("*.ground_truth.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("output_audio_sha256") == audio_hash:
                return path, data
            output_audio = data.get("output_audio")
            if output_audio and not data.get("output_audio_sha256"):
                output_path = Path(output_audio)
                if output_path.exists() and _file_sha256(output_path) == audio_hash:
                    return path, data
    return None, None


def load_audio_manipulation_ground_truth(
    audio_path: str,
    segments: list[dict] | None = None,
    threshold: float | None = None,
    memory_candidates: list[dict] | None = None,
) -> dict:
    """Load controlled manipulation metadata for eval files, when available.

    This is intentionally a validation aid. It should not be used to alter the
    production decision because real uploaded audios will not have this label.
    """
    path = Path(audio_path)
    if not path.exists():
        return {"available": False, "status": "audio_not_found"}

    audio_hash = _file_sha256(path)
    source, data = _find_ground_truth(audio_hash)
    if not data or not source:
        return {"available": False, "status": "not_found", "audio_sha256": audio_hash}

    manipulated_segments = data.get("manipulated_segments") or []
    threshold = _safe_float(threshold, 0.5)
    suspicious_segments = [
        segment
        for segment in (segments or [])
        if _safe_float(segment.get("p_fake")) >= threshold
    ]
    memory_candidates = memory_candidates or []

    threshold_overlap_seconds = 0.0
    for manipulated in manipulated_segments:
        for suspicious in suspicious_segments:
            threshold_overlap_seconds += _interval_overlap(manipulated, suspicious)

    memory_overlap_seconds = 0.0
    for manipulated in manipulated_segments:
        for candidate in memory_candidates:
            memory_overlap_seconds += _interval_overlap(manipulated, candidate)

    overlap_seconds = max(threshold_overlap_seconds, memory_overlap_seconds)

    manipulated_seconds = sum(
        max(0.0, _safe_float(item.get("end")) - _safe_float(item.get("start")))
        for item in manipulated_segments
    )
    suspicious_seconds = sum(
        max(0.0, _safe_float(item.get("end")) - _safe_float(item.get("start")))
        for item in suspicious_segments
    )

    manipulated_overlap_ratio = overlap_seconds / max(manipulated_seconds, 0.001)
    suspicious_overlap_ratio = threshold_overlap_seconds / max(suspicious_seconds, 0.001)
    memory_overlap_ratio = memory_overlap_seconds / max(manipulated_seconds, 0.001)

    return {
        "available": True,
        "status": "matched",
        "source": str(source),
        "match_method": "output_audio_sha256",
        "audio_sha256": audio_hash,
        "label": data.get("label"),
        "manipulation_type": data.get("manipulation_type"),
        "manipulation_method": data.get("manipulation_method"),
        "manipulated_segments": manipulated_segments,
        "source_segment": data.get("source_segment"),
        "expected_behavior": data.get("expected_behavior") or {},
        "overlap": {
            "suspicious_count": len(suspicious_segments),
            "memory_candidate_count": len(memory_candidates),
            "overlap_seconds": overlap_seconds,
            "threshold_overlap_seconds": threshold_overlap_seconds,
            "memory_overlap_seconds": memory_overlap_seconds,
            "manipulated_overlap_ratio": min(1.0, manipulated_overlap_ratio),
            "suspicious_overlap_ratio": min(1.0, suspicious_overlap_ratio),
            "memory_overlap_ratio": min(1.0, memory_overlap_ratio),
            "localized_hit": manipulated_overlap_ratio >= 0.25 or suspicious_overlap_ratio >= 0.25,
            "localized_hit_source": "memory" if memory_overlap_seconds > threshold_overlap_seconds else "threshold",
        },
    }
