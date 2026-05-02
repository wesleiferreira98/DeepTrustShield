from __future__ import annotations

import json
import logging
import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[3]

WORD_RE = re.compile(r"[\wÀ-ÿ']+", re.UNICODE)
DEFAULT_EVENT_CONFIDENCE = 0.45
EXPLICIT_EVENT_CONFIDENCE = 0.80
ANALYSIS_SAMPLE_RATE = 16000


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_quality_label(event: dict) -> str:
    confidence = _safe_float(event.get("confidence"))
    variant = str(event.get("variant") or "")
    if variant in {"", "unknown"}:
        return "candidate"
    if confidence >= 0.80:
        return "strong"
    if confidence >= 0.70:
        return "moderate"
    return "weak"


def _event_counter(events: list[dict], *keys: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        label = ".".join(str(event.get(key) or "unknown") for key in keys)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def summarize_phonetic_events(events: list[dict]) -> dict:
    classified = [
        event
        for event in events
        if event.get("source") == "acoustic_transcript_candidate" and event.get("variant") != "unknown"
    ]
    strong = [event for event in classified if _event_quality_label(event) == "strong"]
    moderate = [event for event in classified if _event_quality_label(event) == "moderate"]
    weak = [event for event in classified if _event_quality_label(event) == "weak"]
    reliable = [event for event in classified if _event_quality_label(event) in {"strong", "moderate"}]
    unknown = [event for event in events if event.get("variant") in {"", None, "unknown"}]

    return {
        "candidate_count": len(events),
        "classified_candidate_count": len(classified),
        "reliable_event_count": len(reliable),
        "unknown_candidate_count": len(unknown),
        "strong_classified_count": len(strong),
        "moderate_classified_count": len(moderate),
        "weak_classified_count": len(weak),
        "classified_ratio": len(classified) / float(len(events)) if events else 0.0,
        "reliable_event_ratio": len(reliable) / float(len(events)) if events else 0.0,
        "candidate_by_phenomenon": _event_counter(events, "phenomenon"),
        "candidate_by_context": _event_counter(events, "phenomenon", "context"),
        "classified_by_phenomenon": _event_counter(classified, "phenomenon"),
        "classified_by_context": _event_counter(classified, "phenomenon", "context"),
        "classified_by_variant": _event_counter(classified, "phenomenon", "variant"),
        "average_classified_confidence": (
            sum(_safe_float(event.get("confidence")) for event in classified) / float(len(classified))
            if classified
            else 0.0
        ),
    }


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _normalize_word(value: str) -> str:
    return _strip_accents(value or "").lower().strip(".,;:!?()[]{}\"'")


def _phonetic_event(
    *,
    event_id: str,
    phenomenon: str,
    word: str,
    start_time: float,
    end_time: float,
    target_segment: str,
    variant: str,
    context: str,
    position: str,
    confidence: float,
    source: str,
    neural_score_local: float | None = None,
    extra: dict | None = None,
) -> dict:
    event = {
        "event_id": event_id,
        "phenomenon": phenomenon,
        "word": word,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "target_segment": target_segment,
        "variant": variant,
        "position": position,
        "context": context,
        "confidence": float(confidence),
        "source": source,
    }
    if neural_score_local is not None:
        event["neural_score_local"] = float(neural_score_local)
    if extra:
        event.update(extra)
    event["event_quality"] = _event_quality_label(event)
    acoustic_features = event.get("acoustic_features") if isinstance(event.get("acoustic_features"), dict) else {}
    event["candidate"] = {
        "phenomenon": phenomenon,
        "word": word,
        "target_segment": target_segment,
        "expected_context": context,
        "position": position,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "source": source,
    }
    event["evidence"] = {
        "available": bool(acoustic_features),
        "features": acoustic_features,
        "method": event.get("variant_method"),
    }
    event["decision"] = {
        "variant": variant,
        "confidence": float(confidence),
        "quality": event["event_quality"],
        "method": event.get("variant_method"),
    }
    return event


def _segment_score_for_time(segments: list[dict], midpoint: float) -> float | None:
    for segment in segments:
        start = _safe_float(segment.get("start"))
        end = _safe_float(segment.get("end"))
        if start <= midpoint <= end:
            return _safe_float(segment.get("p_fake"))
    return None


def _load_analysis_waveform(audio_path: str) -> tuple[torch.Tensor | None, int]:
    try:
        waveform, source_rate = torchaudio.load(audio_path)
    except Exception as exc:
        logger.warning("Failed to load audio for phonetic event analysis %s: %s", audio_path, exc)
        return None, ANALYSIS_SAMPLE_RATE

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != ANALYSIS_SAMPLE_RATE:
        waveform = T.Resample(orig_freq=source_rate, new_freq=ANALYSIS_SAMPLE_RATE)(waveform)
    return waveform.squeeze(0).contiguous(), ANALYSIS_SAMPLE_RATE


def _slice_waveform(waveform: torch.Tensor, sample_rate: int, start: float, end: float) -> torch.Tensor:
    start_sample = max(0, int(start * sample_rate))
    end_sample = min(int(end * sample_rate), int(waveform.numel()))
    if end_sample <= start_sample:
        return torch.empty(0)
    return waveform[start_sample:end_sample]


def _rms(samples: torch.Tensor) -> float:
    if samples.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(samples.float().pow(2))).item())


def _zero_crossing_rate(samples: torch.Tensor) -> float:
    if samples.numel() < 2:
        return 0.0
    signs = torch.sign(samples.float())
    signs[signs == 0] = 1
    crossings = (signs[1:] != signs[:-1]).float().mean()
    return float(crossings.item())


def _spectral_centroid(samples: torch.Tensor, sample_rate: int) -> float:
    if samples.numel() < 32:
        return 0.0
    window = torch.hann_window(samples.numel(), device=samples.device)
    spectrum = torch.fft.rfft(samples.float() * window).abs()
    if spectrum.sum().item() <= 1e-8:
        return 0.0
    freqs = torch.fft.rfftfreq(samples.numel(), d=1.0 / sample_rate).to(samples.device)
    return float((freqs * spectrum).sum().item() / spectrum.sum().item())


def _classify_s_coda_acoustics(
    waveform: torch.Tensor | None,
    sample_rate: int,
    start: float,
    end: float,
    char_index: int,
    word_length: int,
    word_confidence: float,
    is_external: bool,
) -> dict:
    if waveform is None:
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "audio_unavailable"}

    duration = max(0.0, end - start)
    if duration < 0.12 or word_length <= 0:
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "word_too_short"}

    if is_external:
        window_end = end
        window_start = max(start, end - min(0.18, max(0.07, duration * 0.35)))
    else:
        center = start + (duration * ((char_index + 0.5) / max(word_length, 1)))
        half_window = min(0.07, max(0.04, duration / max(word_length, 1)))
        window_start = max(start, center - half_window)
        window_end = min(end, center + half_window)

    target = _slice_waveform(waveform, sample_rate, window_start, window_end)
    if target.numel() < int(sample_rate * 0.035):
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "window_too_short"}

    target_rms = _rms(target)
    zcr = _zero_crossing_rate(target)
    centroid = _spectral_centroid(target, sample_rate)

    variant = "unknown"
    acoustic_confidence = 0.0

    if target_rms < 0.004:
        variant = "deletion"
        acoustic_confidence = 0.62
    elif zcr >= 0.18 and centroid >= 4300:
        variant = "alveolar"
        acoustic_confidence = 0.74
    elif zcr >= 0.11 and 2300 <= centroid <= 4300:
        variant = "palatal"
        acoustic_confidence = 0.72

    if variant == "unknown":
        confidence = min(word_confidence, DEFAULT_EVENT_CONFIDENCE)
    else:
        confidence = min(0.86, (0.45 * word_confidence) + (0.55 * acoustic_confidence))

    return {
        "variant": variant,
        "confidence": confidence,
        "method": "s_coda_frication_centroid",
        "target_rms": target_rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "analysis_start": window_start,
        "analysis_end": window_end,
    }


def _classify_td_before_i_acoustics(
    waveform: torch.Tensor | None,
    sample_rate: int,
    start: float,
    end: float,
    char_index: int,
    word_length: int,
    word_confidence: float,
) -> dict:
    if waveform is None:
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "audio_unavailable"}

    duration = max(0.0, end - start)
    if duration < 0.12 or word_length <= 0:
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "word_too_short"}

    center = start + (duration * ((char_index + 0.5) / max(word_length, 1)))
    release_start = max(start, center)
    release_end = min(end, center + min(0.16, max(0.07, duration * 0.35)))
    release = _slice_waveform(waveform, sample_rate, release_start, release_end)

    if release.numel() < int(sample_rate * 0.035):
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "window_too_short"}

    release_rms = _rms(release)
    zcr = _zero_crossing_rate(release)
    centroid = _spectral_centroid(release, sample_rate)

    variant = "unknown"
    acoustic_confidence = 0.0

    if release_rms >= 0.006 and zcr >= 0.13 and 2300 <= centroid <= 5200:
        variant = "palatalized"
        acoustic_confidence = 0.74
    elif release_rms >= 0.006 and zcr < 0.09 and centroid < 3000:
        variant = "dentoalveolar"
        acoustic_confidence = 0.68

    if variant == "unknown":
        confidence = min(word_confidence, DEFAULT_EVENT_CONFIDENCE)
    else:
        confidence = min(0.84, (0.45 * word_confidence) + (0.55 * acoustic_confidence))

    return {
        "variant": variant,
        "confidence": confidence,
        "method": "td_before_i_release_frication",
        "release_rms": release_rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "analysis_start": release_start,
        "analysis_end": release_end,
    }


def _classify_final_r_acoustics(
    waveform: torch.Tensor | None,
    sample_rate: int,
    start: float,
    end: float,
    word_confidence: float,
) -> dict:
    """Conservative acoustic hint for final /r/ presence/deletion candidates.

    This is not a full rhotic classifier. It only checks whether the final word
    tail looks more like a smooth vowel tail or a noisy/burst-like consonantal
    tail. Ambiguous cases remain unknown.
    """
    if waveform is None:
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "audio_unavailable"}

    duration = max(0.0, end - start)
    if duration < 0.18:
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "word_too_short"}

    word_audio = _slice_waveform(waveform, sample_rate, start, end)
    if word_audio.numel() < int(sample_rate * 0.12):
        return {"variant": "unknown", "confidence": min(word_confidence, DEFAULT_EVENT_CONFIDENCE), "method": "window_too_short"}

    tail_seconds = min(0.16, max(0.08, duration * 0.35))
    tail_samples = max(1, int(tail_seconds * sample_rate))
    tail = word_audio[-tail_samples:]
    body = word_audio[:-tail_samples] if word_audio.numel() > tail_samples else word_audio

    tail_rms = _rms(tail)
    body_rms = _rms(body)
    rms_ratio = tail_rms / max(body_rms, 1e-6)
    tail_zcr = _zero_crossing_rate(tail)

    variant = "unknown"
    acoustic_confidence = 0.0

    if tail_rms < 0.004:
        variant = "deletion"
        acoustic_confidence = 0.62
    elif tail_zcr <= 0.085 and rms_ratio >= 0.22:
        variant = "deletion"
        acoustic_confidence = 0.72
    elif tail_zcr >= 0.145 and tail_rms >= 0.006:
        variant = "present"
        acoustic_confidence = 0.70

    if variant == "unknown":
        confidence = min(word_confidence, DEFAULT_EVENT_CONFIDENCE)
    else:
        confidence = min(0.86, (0.45 * word_confidence) + (0.55 * acoustic_confidence))

    return {
        "variant": variant,
        "confidence": confidence,
        "method": "final_r_tail_energy_zcr",
        "tail_rms": tail_rms,
        "body_rms": body_rms,
        "tail_body_rms_ratio": rms_ratio,
        "tail_zero_crossing_rate": tail_zcr,
    }


def _word_items_from_text(text: str, duration_seconds: float | None = None) -> list[dict]:
    words = WORD_RE.findall(text or "")
    if not words:
        return []

    duration = max(_safe_float(duration_seconds, len(words) * 0.35), 0.001)
    step = duration / float(len(words))
    items = []
    for index, word in enumerate(words):
        start = index * step
        items.append(
            {
                "word": word,
                "start": start,
                "end": min(duration, start + step * 0.85),
                "confidence": DEFAULT_EVENT_CONFIDENCE,
            }
        )
    return items


def _word_items_from_json(data: Any, duration_seconds: float | None = None) -> list[dict]:
    if isinstance(data, dict):
        for key in ("words", "alignment", "segments"):
            if isinstance(data.get(key), list):
                return _word_items_from_json(data[key], duration_seconds)
        if isinstance(data.get("text"), str):
            return _word_items_from_text(data["text"], duration_seconds)
        return []

    if not isinstance(data, list):
        return []

    words = []
    for item in data:
        if isinstance(item, str):
            words.append({"word": item, "confidence": DEFAULT_EVENT_CONFIDENCE})
        elif isinstance(item, dict):
            text = item.get("word") or item.get("text") or item.get("token")
            if not text:
                continue
            words.append(
                {
                    "word": text,
                    "start": item.get("start", item.get("start_time")),
                    "end": item.get("end", item.get("end_time")),
                    "confidence": item.get("confidence", DEFAULT_EVENT_CONFIDENCE),
                }
            )

    if not words:
        return []

    missing_time = any(word.get("start") is None or word.get("end") is None for word in words)
    if not missing_time:
        return words

    duration = max(_safe_float(duration_seconds, len(words) * 0.35), 0.001)
    step = duration / float(len(words))
    for index, word in enumerate(words):
        start = index * step
        word.setdefault("start", start)
        word.setdefault("end", min(duration, start + step * 0.85))
    return words


def _load_sidecar(audio_path: str) -> tuple[str | None, Any | None]:
    path = Path(audio_path)
    candidates = [
        path.with_suffix(".phonetic.json"),
        path.with_suffix(".events.json"),
        path.with_suffix(".alignment.json"),
        path.with_suffix(".transcript.json"),
        path.with_suffix(".json"),
        path.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".txt":
                return str(candidate), candidate.read_text(encoding="utf-8")
            return str(candidate), json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read phonetic sidecar %s: %s", candidate, exc)
            return str(candidate), None
    return _load_sidecar_by_audio_hash(path)


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as exc:
        logger.warning("Failed to hash audio for sidecar lookup %s: %s", path, exc)
        return None


def _sidecar_search_dirs() -> list[Path]:
    raw_value = os.getenv("AUDIO_TRANSCRIPT_SIDECAR_DIRS")
    if raw_value:
        return [Path(item).expanduser().resolve() for item in raw_value.split(os.pathsep) if item.strip()]
    return [(BASE_DIR / "audio" / "data" / "eval_external").resolve()]


def _load_sidecar_by_audio_hash(audio_path: Path) -> tuple[str | None, Any | None]:
    audio_hash = _file_sha256(audio_path)
    if not audio_hash:
        return None, None

    for search_dir in _sidecar_search_dirs():
        if not search_dir.exists():
            continue
        for candidate in search_dir.rglob("*.transcript.json"):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("source_audio_sha256") == audio_hash:
                return str(candidate), data
    return None, None


def _transcript_summary(data: Any, source: str | None) -> dict:
    if not isinstance(data, dict):
        return {
            "available": False,
            "source": source,
            "text": data if isinstance(data, str) else "",
            "word_count": 0,
            "segment_count": 0,
            "asr": {},
        }
    words = data.get("words") if isinstance(data.get("words"), list) else []
    segments = data.get("segments") if isinstance(data.get("segments"), list) else []
    return {
        "available": bool(data.get("text") or words or segments),
        "source": source,
        "text": data.get("text") or "",
        "word_count": len(words),
        "segment_count": len(segments),
        "asr": data.get("asr") or {},
    }


def _events_from_explicit_data(data: Any, segments: list[dict]) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("phonetic_events"), list):
        return []

    events = []
    for index, raw_event in enumerate(data["phonetic_events"]):
        if not isinstance(raw_event, dict):
            continue

        start = _safe_float(raw_event.get("start_time", raw_event.get("start")))
        end = _safe_float(raw_event.get("end_time", raw_event.get("end")), start)
        midpoint = start + ((end - start) / 2.0)
        event = {
            "event_id": raw_event.get("event_id") or f"ev_{index + 1:04d}",
            "phenomenon": raw_event.get("phenomenon"),
            "word": raw_event.get("word"),
            "start_time": start,
            "end_time": end,
            "target_segment": raw_event.get("target_segment"),
            "variant": raw_event.get("variant"),
            "position": raw_event.get("position"),
            "context": raw_event.get("context"),
            "confidence": _safe_float(raw_event.get("confidence", raw_event.get("variant_confidence")), EXPLICIT_EVENT_CONFIDENCE),
            "source": raw_event.get("source") or "explicit_sidecar",
            "neural_score_local": raw_event.get("neural_score_local"),
        }
        if event["neural_score_local"] is None:
            event["neural_score_local"] = _segment_score_for_time(segments, midpoint)
        event.update({key: value for key, value in raw_event.items() if key not in event})
        events.append(event)
    return events


def _looks_like_infinitive_candidate(norm: str) -> bool:
    if len(norm) < 4:
        return False
    return norm.endswith(("ar", "er", "ir"))


def _starts_with_vowel(norm: str) -> bool:
    return bool(norm) and norm[0] in "aeiou"


def _external_coda_context(item: dict, next_item: dict | None, next_norm: str) -> tuple[str, str]:
    if not next_item or not next_norm:
        return "external_coda_pause", "pause"

    current_end = _safe_float(item.get("end"))
    next_start = _safe_float(next_item.get("start"), current_end)
    if next_start - current_end > 0.30:
        return "external_coda_pause", "pause"

    following_segment = next_norm[0]
    if _starts_with_vowel(next_norm):
        return "external_coda_vowel", following_segment
    return "external_coda_consonant", following_segment


def _is_s_coda_candidate(norm: str, index: int) -> tuple[bool, str]:
    prev_char = norm[index - 1] if index > 0 else ""
    next_char = norm[index + 1] if index + 1 < len(norm) else ""

    # Orthographic "ss" is intervocalic fricative spelling in words like
    # "disse" and "isso"; it is not evidence of /s/ in coda for V1.
    if prev_char == "s" or next_char == "s":
        return False, "orthographic_ss"

    is_external = index == len(norm) - 1
    if is_external:
        return True, "external_coda"

    is_internal_coda = bool(next_char and next_char not in "aeiou")
    if is_internal_coda:
        return True, "internal_coda"

    return False, "not_coda"


def _rotic_events(
    word: str,
    norm: str,
    item: dict,
    event_id: str,
    segments: list[dict],
    waveform: torch.Tensor | None,
    sample_rate: int,
) -> list[dict]:
    if not _looks_like_infinitive_candidate(norm):
        return []
    start = _safe_float(item.get("start"))
    end = _safe_float(item.get("end"), start)
    midpoint = start + ((end - start) / 2.0)
    word_confidence = _safe_float(item.get("confidence"), DEFAULT_EVENT_CONFIDENCE)
    acoustic = _classify_final_r_acoustics(waveform, sample_rate, start, end, word_confidence)
    return [
        _phonetic_event(
            event_id=event_id,
            phenomenon="rotic_coda",
            word=word,
            start_time=start,
            end_time=end,
            target_segment="r",
            variant=acoustic["variant"],
            context="verb_infinitive_external_coda",
            position="external_coda",
            confidence=acoustic["confidence"],
            source="acoustic_transcript_candidate" if acoustic["variant"] != "unknown" else "transcript_candidate",
            neural_score_local=_segment_score_for_time(segments, midpoint),
            extra={
                "lexical_class": "verb_infinitive_candidate",
                "variant_method": acoustic["method"],
                "acoustic_features": {
                    key: value
                    for key, value in acoustic.items()
                    if key not in {"variant", "confidence", "method"}
                },
            },
        )
    ]


def _s_coda_events(
    word: str,
    norm: str,
    item: dict,
    next_item: dict | None,
    next_norm: str,
    event_id: str,
    segments: list[dict],
    waveform: torch.Tensor | None,
    sample_rate: int,
) -> list[dict]:
    if "s" not in norm:
        return []

    events = []
    start = _safe_float(item.get("start"))
    end = _safe_float(item.get("end"), start)
    midpoint = start + ((end - start) / 2.0)
    for index, char in enumerate(norm):
        if char != "s":
            continue
        is_candidate, candidate_reason = _is_s_coda_candidate(norm, index)
        if not is_candidate:
            continue
        next_char = norm[index + 1] if index + 1 < len(norm) else ""
        is_external = index == len(norm) - 1
        if is_external:
            context, following_segment = _external_coda_context(item, next_item, next_norm)
        else:
            context = "internal_coda"
            following_segment = next_char
        word_confidence = _safe_float(item.get("confidence"), DEFAULT_EVENT_CONFIDENCE)
        acoustic = _classify_s_coda_acoustics(
            waveform,
            sample_rate,
            start,
            end,
            index,
            len(norm),
            word_confidence,
            is_external,
        )
        events.append(
            _phonetic_event(
                event_id=f"{event_id}_{len(events) + 1}",
                phenomenon="s_coda",
                word=word,
                start_time=start,
                end_time=end,
                target_segment="s",
                variant=acoustic["variant"],
                context=context,
                position="external_coda" if is_external else "internal_coda",
                confidence=acoustic["confidence"],
                source="acoustic_transcript_candidate" if acoustic["variant"] != "unknown" else "transcript_candidate",
                neural_score_local=_segment_score_for_time(segments, midpoint),
                extra={
                    "following_segment": following_segment,
                    "candidate_method": "orthographic_conservative_s_coda",
                    "candidate_reason": candidate_reason,
                    "variant_method": acoustic["method"],
                    "acoustic_features": {
                        key: value
                        for key, value in acoustic.items()
                        if key not in {"variant", "confidence", "method"}
                    },
                },
            )
        )
    return events


def _td_before_i_events(
    word: str,
    norm: str,
    item: dict,
    event_id: str,
    segments: list[dict],
    waveform: torch.Tensor | None,
    sample_rate: int,
) -> list[dict]:
    events = []
    start = _safe_float(item.get("start"))
    end = _safe_float(item.get("end"), start)
    midpoint = start + ((end - start) / 2.0)
    for match_index, match in enumerate(re.finditer(r"([td])i", norm)):
        target = match.group(1)
        word_confidence = _safe_float(item.get("confidence"), DEFAULT_EVENT_CONFIDENCE)
        acoustic = _classify_td_before_i_acoustics(
            waveform,
            sample_rate,
            start,
            end,
            match.start(1),
            len(norm),
            word_confidence,
        )
        events.append(
            _phonetic_event(
                event_id=f"{event_id}_{match_index + 1}",
                phenomenon="td_before_i",
                word=word,
                start_time=start,
                end_time=end,
                target_segment=target,
                variant=acoustic["variant"],
                context="phonological_i",
                position="onset",
                confidence=acoustic["confidence"],
                source="acoustic_transcript_candidate" if acoustic["variant"] != "unknown" else "transcript_candidate",
                neural_score_local=_segment_score_for_time(segments, midpoint),
                extra={
                    "following_vowel": "i_phonological",
                    "variant_method": acoustic["method"],
                    "acoustic_features": {
                        key: value
                        for key, value in acoustic.items()
                        if key not in {"variant", "confidence", "method"}
                    },
                },
            )
        )
    return events


def extract_phonetic_events_from_words(
    word_items: list[dict],
    segments: list[dict] | None = None,
    waveform: torch.Tensor | None = None,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
) -> list[dict]:
    segments = segments or []
    events = []
    for word_index, item in enumerate(word_items):
        word = str(item.get("word") or "").strip()
        norm = _normalize_word(word)
        if not norm:
            continue
        next_item = word_items[word_index + 1] if word_index + 1 < len(word_items) else None
        next_norm = _normalize_word(str(next_item.get("word") or "")) if next_item else ""
        base_id = f"ev_{len(events) + 1:04d}"
        events.extend(_rotic_events(word, norm, item, base_id, segments, waveform, sample_rate))
        base_id = f"ev_{len(events) + 1:04d}"
        events.extend(_s_coda_events(word, norm, item, next_item, next_norm, base_id, segments, waveform, sample_rate))
        base_id = f"ev_{len(events) + 1:04d}"
        events.extend(_td_before_i_events(word, norm, item, base_id, segments, waveform, sample_rate))
    return events


def extract_phonetic_events(
    audio_path: str,
    *,
    audio_info: dict | None = None,
    segments: list[dict] | None = None,
) -> dict:
    """Extract V1 phonetic events from a sidecar transcript/events file.

    This is a bridge layer. It supports explicit `phonetic_events` today and
    transcript-derived low-confidence candidates while ASR/alignment is not yet
    integrated into the audio pipeline.
    """
    audio_info = audio_info or {}
    segments = segments or []
    sidecar_path, data = _load_sidecar(audio_path)

    if sidecar_path is None:
        return {
            "available": False,
            "status": "no_sidecar",
            "source": None,
            "events": [],
            "event_count": 0,
            "candidate_count": 0,
            "note": "Nenhuma transcricao ou lista de eventos foneticos foi encontrada ao lado do audio.",
        }

    if data is None:
        return {
            "available": False,
            "status": "sidecar_unreadable",
            "source": sidecar_path,
            "events": [],
            "event_count": 0,
            "candidate_count": 0,
            "note": "O arquivo auxiliar foi encontrado, mas nao pode ser lido.",
        }

    explicit_events = _events_from_explicit_data(data, segments)
    if explicit_events:
        event_summary = summarize_phonetic_events(explicit_events)
        return {
            "available": True,
            "status": "explicit_events",
            "source": sidecar_path,
            "events": explicit_events,
            "event_count": len(explicit_events),
            "candidate_count": len(explicit_events),
            "classified_candidate_count": event_summary["classified_candidate_count"],
            "event_summary": event_summary,
            "transcript": _transcript_summary(data, sidecar_path),
            "note": "Eventos foneticos explicitos carregados do arquivo auxiliar.",
        }

    waveform, sample_rate = _load_analysis_waveform(audio_path)

    if isinstance(data, str):
        word_items = _word_items_from_text(data, audio_info.get("duration_seconds"))
    else:
        word_items = _word_items_from_json(data, audio_info.get("duration_seconds"))

    candidate_events = extract_phonetic_events_from_words(word_items, segments, waveform, sample_rate)
    event_summary = summarize_phonetic_events(candidate_events)
    classified_count = event_summary["classified_candidate_count"]
    if classified_count:
        note = "Eventos candidatos foram inferidos da transcricao; alguns receberam classificacao acustica conservadora."
    elif candidate_events:
        note = (
            "Eventos candidatos foram inferidos da transcricao com baixa confianca; "
            "eles ainda nao calibram a decisao sem classificacao fonetica."
        )
    else:
        note = "A transcricao nao gerou eventos foneticos candidatos para a V1."

    return {
        "available": bool(candidate_events),
        "status": "acoustic_classified_candidates" if classified_count else ("transcript_candidates" if candidate_events else "no_candidates"),
        "source": sidecar_path,
        "events": candidate_events,
        "event_count": classified_count,
        "candidate_count": len(candidate_events),
        "classified_candidate_count": classified_count,
        "event_summary": event_summary,
        "transcript": _transcript_summary(data, sidecar_path),
        "word_count": len(word_items),
        "note": note,
    }
