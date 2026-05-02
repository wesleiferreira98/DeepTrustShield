from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

logger = logging.getLogger(__name__)
ASR_SAMPLE_RATE = 16000
ASR_TARGET_RMS = 0.08


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


def _rms(waveform: torch.Tensor) -> float:
    if waveform.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(waveform.float().pow(2))).item())


def _light_stationary_noise_reduction(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Apply a conservative spectral subtraction pass for ASR only."""
    if waveform.numel() < sample_rate:
        return waveform

    frame_size = 512
    hop_size = 160
    if waveform.numel() < frame_size * 2:
        return waveform

    window = torch.hann_window(frame_size, device=waveform.device)
    stft = torch.stft(
        waveform.float(),
        n_fft=frame_size,
        hop_length=hop_size,
        win_length=frame_size,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs()
    frame_energy = magnitude.mean(dim=0)
    if frame_energy.numel() < 4 or frame_energy.max().item() <= 1e-8:
        return waveform

    quiet_count = max(3, int(frame_energy.numel() * 0.12))
    quiet_indices = torch.argsort(frame_energy)[:quiet_count]
    noise_profile = magnitude[:, quiet_indices].median(dim=1).values.unsqueeze(1)

    reduction_strength = 0.35
    spectral_floor = 0.08
    cleaned_magnitude = torch.clamp(
        magnitude - (noise_profile * reduction_strength),
        min=magnitude * spectral_floor,
    )
    phase = stft / torch.clamp(magnitude, min=1e-8)
    cleaned = torch.istft(
        cleaned_magnitude * phase,
        n_fft=frame_size,
        hop_length=hop_size,
        win_length=frame_size,
        window=window,
        length=waveform.numel(),
    )
    return (0.85 * cleaned) + (0.15 * waveform.float())


def _trim_edge_silence(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, dict]:
    """Trim only leading/trailing low-energy regions; internal pauses are preserved."""
    frame_size = int(sample_rate * 0.03)
    hop_size = int(sample_rate * 0.01)
    if waveform.numel() < frame_size * 4:
        return waveform, {"trimmed": False, "trim_start_seconds": 0.0, "trim_end_seconds": 0.0}

    frames = waveform.unfold(0, frame_size, hop_size)
    frame_rms = torch.sqrt(torch.mean(frames.float().pow(2), dim=1))
    global_rms = _rms(waveform)
    threshold = max(0.003, global_rms * 0.18)
    active = frame_rms > threshold
    if not active.any():
        return waveform, {"trimmed": False, "trim_start_seconds": 0.0, "trim_end_seconds": 0.0}

    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    margin = int(sample_rate * 0.25)
    start_sample = max(0, int(active_indices[0].item() * hop_size) - margin)
    end_sample = min(waveform.numel(), int(active_indices[-1].item() * hop_size + frame_size) + margin)
    if end_sample <= start_sample:
        return waveform, {"trimmed": False, "trim_start_seconds": 0.0, "trim_end_seconds": 0.0}

    trimmed_start = start_sample / float(sample_rate)
    trimmed_end = (waveform.numel() - end_sample) / float(sample_rate)
    if trimmed_start < 0.05 and trimmed_end < 0.05:
        return waveform, {"trimmed": False, "trim_start_seconds": 0.0, "trim_end_seconds": 0.0}

    return waveform[start_sample:end_sample], {
        "trimmed": True,
        "trim_start_seconds": trimmed_start,
        "trim_end_seconds": trimmed_end,
    }


def prepare_audio_for_asr(audio_path: str, *, enabled: bool = True) -> tuple[str, dict, Path | None]:
    """Create a temporary ASR input without changing the original analysis audio."""
    path = Path(audio_path)
    if not enabled:
        return str(path), {"enabled": False}, None

    try:
        waveform, source_sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        logger.warning("ASR preprocessing failed to load %s: %s", path, exc)
        return str(path), {"enabled": True, "status": "load_failed", "error": str(exc)}, None

    metadata: dict[str, Any] = {
        "enabled": True,
        "status": "applied",
        "source_sample_rate": int(source_sample_rate),
        "target_sample_rate": ASR_SAMPLE_RATE,
        "source_channels": int(waveform.size(0)),
        "steps": ["mono", "resample_16khz", "dc_remove", "highpass_60hz", "light_stationary_noise_reduction", "edge_silence_trim", "light_rms_normalization"],
    }

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != ASR_SAMPLE_RATE:
        waveform = T.Resample(orig_freq=source_sample_rate, new_freq=ASR_SAMPLE_RATE)(waveform)

    mono = waveform.squeeze(0).float().contiguous()
    original_rms = _rms(mono)
    mono = mono - mono.mean()
    try:
        mono = AF.highpass_biquad(mono.unsqueeze(0), ASR_SAMPLE_RATE, cutoff_freq=60.0).squeeze(0)
    except Exception as exc:
        metadata["highpass_warning"] = str(exc)

    denoised = _light_stationary_noise_reduction(mono, ASR_SAMPLE_RATE)
    trimmed, trim_metadata = _trim_edge_silence(denoised, ASR_SAMPLE_RATE)
    processed = trimmed.contiguous()

    processed_rms = _rms(processed)
    gain = 1.0
    if processed_rms > 1e-6:
        gain = min(3.0, max(0.35, ASR_TARGET_RMS / processed_rms))
        processed = processed * gain

    peak = float(processed.abs().max().item()) if processed.numel() else 0.0
    if peak > 0.98:
        processed = processed * (0.98 / peak)

    final_peak = float(processed.abs().max().item()) if processed.numel() else 0.0
    metadata.update(
        {
            "original_rms": original_rms,
            "processed_rms_before_gain": processed_rms,
            "target_rms": ASR_TARGET_RMS,
            "gain": gain,
            "final_peak": final_peak,
            **trim_metadata,
        }
    )

    temp_file = tempfile.NamedTemporaryFile(prefix="deeptrust_asr_", suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    torchaudio.save(str(temp_path), processed.unsqueeze(0).cpu(), ASR_SAMPLE_RATE)
    metadata["temporary_input"] = str(temp_path)
    return str(temp_path), metadata, temp_path


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
    preprocess_audio: bool = True,
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
        asr_audio_path, preprocessing, temporary_path = prepare_audio_for_asr(str(path), enabled=preprocess_audio)
        segments_iter, info = model.transcribe(
            asr_audio_path,
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
                "preprocessing": preprocessing,
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
    finally:
        if "temporary_path" in locals() and temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to remove temporary ASR input %s", temporary_path, exc_info=True)
