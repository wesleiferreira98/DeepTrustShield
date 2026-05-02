from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_word(word: Any) -> dict:
    return {
        "word": getattr(word, "word", "").strip(),
        "start": float(getattr(word, "start", 0.0) or 0.0),
        "end": float(getattr(word, "end", 0.0) or 0.0),
        "confidence": float(getattr(word, "probability", 0.0) or 0.0),
    }


def _serialize_segment(segment: Any) -> dict:
    words = [
        _serialize_word(word)
        for word in (getattr(segment, "words", None) or [])
        if getattr(word, "word", "").strip()
    ]
    return {
        "id": int(getattr(segment, "id", 0) or 0),
        "start": float(getattr(segment, "start", 0.0) or 0.0),
        "end": float(getattr(segment, "end", 0.0) or 0.0),
        "text": (getattr(segment, "text", "") or "").strip(),
        "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
        "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
        "words": words,
    }


def _flatten_words(segments: list[dict]) -> list[dict]:
    words = []
    for segment in segments:
        if segment.get("words"):
            words.extend(segment["words"])
    return words


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper nao esta instalado no ambiente do worker") from exc

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda"
        try:
            import torch

            if not torch.cuda.is_available():
                resolved_device = "cpu"
        except Exception:
            resolved_device = "cpu"

    logger.info("Loading ASR model %s on %s (%s)", model_name, resolved_device, compute_type)
    return WhisperModel(model_name, device=resolved_device, compute_type=compute_type), resolved_device


def _sidecar_summary(path: Path, data: dict, status: str) -> dict:
    words = data.get("words") or []
    segments = data.get("segments") or []
    return {
        "available": True,
        "status": status,
        "path": str(path),
        "text": data.get("text") or "",
        "word_count": len(words),
        "segment_count": len(segments),
        "asr": data.get("asr") or {},
    }


def ensure_audio_transcript_sidecar(
    audio_path: str,
    *,
    enabled: bool = True,
    model_name: str = "medium",
    language: str = "pt",
    device: str = "auto",
    compute_type: str = "default",
    beam_size: int = 5,
    vad_filter: bool = False,
    overwrite: bool = False,
) -> dict:
    """Generate a transcript sidecar next to the audio when one is missing."""
    path = Path(audio_path)
    sidecar_path = path.with_suffix(".transcript.json")

    if sidecar_path.exists() and not overwrite:
        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            return _sidecar_summary(sidecar_path, data, "existing")
        except Exception as exc:
            logger.warning("Failed to read existing transcript sidecar %s: %s", sidecar_path, exc)

    if not enabled:
        return {
            "available": False,
            "status": "disabled",
            "path": str(sidecar_path),
            "text": "",
            "word_count": 0,
            "segment_count": 0,
            "asr": {},
        }

    try:
        model, resolved_device = _load_whisper_model(model_name, device, compute_type)
        segments_iter, info = model.transcribe(
            str(path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=True,
        )
        segments = [_serialize_segment(segment) for segment in segments_iter]
        words = _flatten_words(segments)
        data = {
            "text": " ".join(segment["text"] for segment in segments if segment.get("text")).strip(),
            "words": words,
            "segments": segments,
            "asr": {
                "engine": "faster-whisper",
                "model": model_name,
                "language": getattr(info, "language", language),
                "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                "duration_seconds": float(getattr(info, "duration", 0.0) or 0.0),
                "duration_after_vad_seconds": float(getattr(info, "duration_after_vad", 0.0) or 0.0),
                "device": resolved_device,
                "compute_type": compute_type,
                "beam_size": beam_size,
                "vad_filter": bool(vad_filter),
            },
            "source_audio": str(path),
            "source_audio_sha256": _file_sha256(path),
            "sidecar_version": "audio_transcript_sidecar_v1",
        }
        sidecar_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return _sidecar_summary(sidecar_path, data, "generated")
    except Exception as exc:
        logger.warning("Automatic audio ASR failed for %s: %s", path, exc)
        return {
            "available": False,
            "status": "failed",
            "path": str(sidecar_path),
            "text": "",
            "word_count": 0,
            "segment_count": 0,
            "asr": {
                "engine": "faster-whisper",
                "model": model_name,
                "language": language,
                "device": device,
                "compute_type": compute_type,
            },
            "error": str(exc),
        }
