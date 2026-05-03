import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from app.pipeline.audio_phonetic_knowledge import apply_phonetic_rules

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "very_short_seconds": 1.0,
    "short_seconds": 2.0,
    "high_silence_ratio": 0.60,
    "low_volume_rms": 0.015,
    "clipping_ratio": 0.01,
    "silence_amplitude": 0.01,
    "near_threshold_margin": 0.10,
    "strong_margin": 0.25,
    "partial_analysis_ratio": 0.75,
    "segment_disagreement_range": 0.35,
    "consistent_fake_ratio": 0.70,
    "isolated_fake_max_ratio": 0.35,
    "localized_sensitivity_min_fake_ratio": 0.04,
    "localized_sensitivity_delta": 0.05,
    "temporal_attention_margin": 0.05,
    "temporal_min_attention_score": 0.82,
    "temporal_local_jump": 0.16,
    "temporal_mad_multiplier": 2.25,
    "temporal_symbolic_delta": 0.02,
    "domain_warning_real_seconds": 8.0,
    "moderate_silence_ratio": 0.35,
    "terminal_segment_margin_seconds": 1.0,
    "segment_volume_change_ratio": 0.45,
}


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_mono_waveform(audio_path: str, target_sample_rate: int) -> tuple[torch.Tensor, dict]:
    waveform, source_rate = torchaudio.load(audio_path)
    original_channels = int(waveform.size(0))
    original_samples = int(waveform.size(-1))

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_rate != target_sample_rate:
        waveform = T.Resample(orig_freq=source_rate, new_freq=target_sample_rate)(waveform)

    info = {
        "source_sample_rate": int(source_rate),
        "target_sample_rate": int(target_sample_rate),
        "original_channels": original_channels,
        "original_samples": original_samples,
        "duration_seconds": original_samples / float(source_rate) if source_rate else None,
    }
    return waveform, info


def extract_audio_symbolic_features(audio_path: str, audio_info: dict | None = None) -> dict:
    """Extract lightweight, interpretable audio features for symbolic rules."""
    audio_info = audio_info or {}
    target_sample_rate = int(audio_info.get("target_sample_rate") or audio_info.get("sample_rate") or 16000)

    try:
        waveform, loaded_info = _load_mono_waveform(audio_path, target_sample_rate)
    except Exception as exc:
        logger.warning("Failed to extract symbolic audio features from %s: %s", audio_path, exc)
        return {
            "path": str(Path(audio_path).name),
            "available": False,
            "error": str(exc),
        }

    duration_seconds = _safe_float(
        audio_info.get("duration_seconds"),
        default=_safe_float(loaded_info.get("duration_seconds")),
    )
    segment_summary = audio_info.get("segment_summary") or {}
    analyzed_fraction = _safe_float(
        segment_summary.get("coverage_fraction"),
        _safe_float(audio_info.get("clip_seconds"), duration_seconds) / max(duration_seconds, 0.001),
    )
    abs_waveform = waveform.abs()
    rms = torch.sqrt(torch.mean(waveform.pow(2))).item() if waveform.numel() else 0.0
    rms_std = waveform.view(-1).std(unbiased=False).item() if waveform.numel() else 0.0
    silence_ratio = (abs_waveform < DEFAULT_THRESHOLDS["silence_amplitude"]).float().mean().item() if waveform.numel() else 1.0
    clipping_ratio = (abs_waveform >= 0.99).float().mean().item() if waveform.numel() else 0.0

    return {
        "path": str(Path(audio_path).name),
        "available": True,
        "duration_seconds": duration_seconds,
        "clip_seconds": _safe_float(audio_info.get("clip_seconds"), duration_seconds),
        "analyzed_fraction": min(1.0, analyzed_fraction),
        "source_sample_rate": int(audio_info.get("source_sample_rate") or loaded_info["source_sample_rate"]),
        "target_sample_rate": target_sample_rate,
        "channels": int(audio_info.get("channels") or loaded_info["original_channels"]),
        "rms_mean": float(rms),
        "rms_std": float(rms_std),
        "silence_ratio": float(silence_ratio),
        "clipping_ratio": float(clipping_ratio),
    }


def _base_flags() -> dict:
    return {
        "short_audio": False,
        "very_short_audio": False,
        "high_silence": False,
        "low_volume": False,
        "clipped_audio": False,
        "near_threshold": False,
        "strong_fake_score": False,
        "strong_real_score": False,
        "low_confidence_fake": False,
        "feature_extraction_failed": False,
        "partial_audio_analysis": False,
        "segment_score_disagreement": False,
        "isolated_suspicious_segment": False,
        "consistent_suspicious_segments": False,
        "regional_bias_review": False,
        "domain_mismatch_warning": False,
        "phonetic_layer_available": False,
        "phonetic_coverage_insufficient": False,
        "phonetic_consistency_support": False,
        "phonetic_disruption_alert": False,
        "terminal_suspicious_segments": False,
        "possible_end_artifact": False,
        "localized_manipulation_sensitivity": False,
        "temporal_memory_available": False,
        "temporal_memory_rupture_candidate": False,
    }


def _slice_waveform(waveform: torch.Tensor, sample_rate: int, start: float, end: float) -> torch.Tensor:
    start_sample = max(0, int(start * sample_rate))
    end_sample = min(int(end * sample_rate), waveform.size(-1))
    if end_sample <= start_sample:
        return torch.empty(0)
    return waveform[:, start_sample:end_sample]


def _segment_audio_metrics(samples: torch.Tensor) -> dict:
    if samples.numel() == 0:
        return {
            "rms": 0.0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
        }
    abs_samples = samples.abs()
    return {
        "rms": float(torch.sqrt(torch.mean(samples.pow(2))).item()),
        "silence_ratio": float((abs_samples < DEFAULT_THRESHOLDS["silence_amplitude"]).float().mean().item()),
        "clipping_ratio": float((abs_samples >= 0.99).float().mean().item()),
    }


def diagnose_suspicious_audio_segments(
    audio_path: str,
    audio_info: dict,
    segments: list[dict],
    threshold: float,
    thresholds: dict | None = None,
) -> dict:
    """Inspect audio-quality clues around segments above the decision threshold."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    suspicious = [
        segment
        for segment in segments
        if isinstance(segment.get("p_fake"), (int, float)) and _safe_float(segment.get("p_fake")) >= threshold
    ]
    if not suspicious:
        return {
            "available": False,
            "reason": "no_suspicious_segments",
            "segments": [],
            "summary": {
                "suspicious_count": 0,
                "terminal_suspicious_count": 0,
                "possible_end_artifact": False,
            },
        }

    target_sample_rate = int(audio_info.get("target_sample_rate") or audio_info.get("sample_rate") or 16000)
    try:
        waveform, loaded_info = _load_mono_waveform(audio_path, target_sample_rate)
    except Exception as exc:
        logger.warning("Failed to diagnose suspicious audio segments from %s: %s", audio_path, exc)
        return {
            "available": False,
            "reason": "audio_load_failed",
            "error": str(exc),
            "segments": [],
            "summary": {
                "suspicious_count": len(suspicious),
                "terminal_suspicious_count": 0,
                "possible_end_artifact": False,
            },
        }

    duration = _safe_float(audio_info.get("duration_seconds"), _safe_float(loaded_info.get("duration_seconds")))
    diagnostics = []
    terminal_count = 0
    possible_end_artifact = False

    for segment in suspicious[:8]:
        start = _safe_float(segment.get("start"))
        end = _safe_float(segment.get("end"))
        index = int(segment.get("index", -1))
        metrics = _segment_audio_metrics(_slice_waveform(waveform, target_sample_rate, start, end))
        previous = next((item for item in segments if int(item.get("index", -99)) == index - 1), None)
        previous_metrics = None
        volume_change_ratio = 0.0
        if previous:
            prev_start = _safe_float(previous.get("start"))
            prev_end = _safe_float(previous.get("end"))
            previous_metrics = _segment_audio_metrics(_slice_waveform(waveform, target_sample_rate, prev_start, prev_end))
            volume_change_ratio = abs(metrics["rms"] - previous_metrics["rms"]) / max(previous_metrics["rms"], 1e-6)

        is_terminal = end >= max(0.0, duration - thresholds["terminal_segment_margin_seconds"])
        if is_terminal:
            terminal_count += 1

        segment_possible_end_artifact = bool(
            is_terminal
            and (
                metrics["silence_ratio"] >= thresholds["moderate_silence_ratio"]
                or volume_change_ratio >= thresholds["segment_volume_change_ratio"]
                or metrics["clipping_ratio"] >= thresholds["clipping_ratio"]
            )
        )
        possible_end_artifact = possible_end_artifact or segment_possible_end_artifact

        diagnostics.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "p_fake": _safe_float(segment.get("p_fake")),
                "terminal_segment": is_terminal,
                "possible_end_artifact": segment_possible_end_artifact,
                "rms": metrics["rms"],
                "silence_ratio": metrics["silence_ratio"],
                "clipping_ratio": metrics["clipping_ratio"],
                "previous_rms": previous_metrics["rms"] if previous_metrics else None,
                "volume_change_ratio": volume_change_ratio,
            }
        )

    return {
        "available": True,
        "segments": diagnostics,
        "summary": {
            "suspicious_count": len(suspicious),
            "terminal_suspicious_count": terminal_count,
            "terminal_suspicious_ratio": terminal_count / float(len(suspicious)) if suspicious else 0.0,
            "possible_end_artifact": possible_end_artifact,
            "max_volume_change_ratio": max((item["volume_change_ratio"] for item in diagnostics), default=0.0),
            "max_silence_ratio": max((item["silence_ratio"] for item in diagnostics), default=0.0),
            "max_clipping_ratio": max((item["clipping_ratio"] for item in diagnostics), default=0.0),
        },
    }


def _summarize_segments(segments: list[dict], threshold: float) -> dict:
    scores = [
        _safe_float(segment.get("p_fake"))
        for segment in segments
        if isinstance(segment.get("p_fake"), (int, float))
    ]
    if not scores:
        return {
            "count": 0,
            "fake_count": 0,
            "fake_ratio": 0.0,
            "min_score": None,
            "max_score": None,
            "mean_score": None,
            "score_range": 0.0,
            "suspicious_segments": [],
        }

    suspicious_segments = [
        {
            "index": int(segment.get("index", index)),
            "start": _safe_float(segment.get("start")),
            "end": _safe_float(segment.get("end")),
            "p_fake": _safe_float(segment.get("p_fake")),
        }
        for index, segment in enumerate(segments)
        if _safe_float(segment.get("p_fake")) >= threshold
    ]
    min_score = min(scores)
    max_score = max(scores)
    mean_score = sum(scores) / len(scores)
    return {
        "count": len(scores),
        "fake_count": len(suspicious_segments),
        "fake_ratio": len(suspicious_segments) / float(len(scores)),
        "min_score": min_score,
        "max_score": max_score,
        "mean_score": mean_score,
        "score_range": max_score - min_score,
        "suspicious_segments": suspicious_segments[:8],
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def evaluate_audio_temporal_memory(
    audio_path: str,
    neural_result: dict,
    symbolic_features: dict | None = None,
    *,
    enabled: bool = True,
    thresholds: dict | None = None,
) -> dict:
    """Compare each window with the internal behavior of the same audio."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    segments = neural_result.get("segments") or []
    threshold = _safe_float(neural_result.get("threshold"), 0.5)
    if not enabled:
        return {"available": False, "status": "disabled", "rupture_candidates": []}
    if len(segments) < 3:
        return {"available": False, "status": "insufficient_segments", "rupture_candidates": []}

    scores = [_safe_float(segment.get("p_fake")) for segment in segments]
    median_score = _median(scores)
    mad = _median([abs(score - median_score) for score in scores]) or 0.001
    attention_threshold = max(
        thresholds["temporal_min_attention_score"],
        threshold - thresholds["temporal_attention_margin"],
    )

    acoustic_metrics_by_index: dict[int, dict] = {}
    target_sample_rate = int((neural_result.get("audio_info") or {}).get("target_sample_rate") or 16000)
    try:
        waveform, _ = _load_mono_waveform(audio_path, target_sample_rate)
        for index, segment in enumerate(segments):
            start = _safe_float(segment.get("start"))
            end = _safe_float(segment.get("end"))
            acoustic_metrics_by_index[index] = _segment_audio_metrics(
                _slice_waveform(waveform, target_sample_rate, start, end)
            )
    except Exception as exc:
        logger.warning("Temporal memory acoustic load failed for %s: %s", audio_path, exc)

    rms_values = [metrics["rms"] for metrics in acoustic_metrics_by_index.values()]
    silence_values = [metrics["silence_ratio"] for metrics in acoustic_metrics_by_index.values()]
    median_rms = _median(rms_values)
    median_silence = _median(silence_values)

    candidates = []
    for index, segment in enumerate(segments):
        score = scores[index]
        previous_score = scores[index - 1] if index > 0 else score
        next_score = scores[index + 1] if index < len(scores) - 1 else score
        local_delta = max(abs(score - previous_score), abs(score - next_score))
        z_like = abs(score - median_score) / max(mad, 0.001)
        metrics = acoustic_metrics_by_index.get(index) or {}
        rms_delta_ratio = abs(_safe_float(metrics.get("rms")) - median_rms) / max(median_rms, 1e-6) if metrics else 0.0
        silence_delta = abs(_safe_float(metrics.get("silence_ratio")) - median_silence) if metrics else 0.0

        reasons = []
        if score >= attention_threshold:
            reasons.append("near_calibrated_threshold")
        if z_like >= thresholds["temporal_mad_multiplier"]:
            reasons.append("score_outlier_in_audio_memory")
        if local_delta >= thresholds["temporal_local_jump"]:
            reasons.append("local_score_jump")
        if rms_delta_ratio >= 0.35:
            reasons.append("local_energy_shift")
        if silence_delta >= 0.25:
            reasons.append("local_silence_shift")

        if "near_calibrated_threshold" in reasons and (
            len(reasons) >= 2 or score >= threshold
        ):
            candidates.append(
                {
                    "index": int(segment.get("index", index)),
                    "start": _safe_float(segment.get("start")),
                    "end": _safe_float(segment.get("end")),
                    "p_fake": score,
                    "memory_score": min(1.0, (0.45 * (score / max(threshold, 0.001))) + (0.35 * min(1.0, z_like / 4.0)) + (0.20 * min(1.0, local_delta / 0.35))),
                    "score_delta_from_median": score - median_score,
                    "local_delta": local_delta,
                    "z_like": z_like,
                    "rms_delta_ratio": rms_delta_ratio,
                    "silence_delta": silence_delta,
                    "reasons": reasons,
                }
            )

    candidates = sorted(candidates, key=lambda item: item["memory_score"], reverse=True)[:8]
    rupture_score = max((item["memory_score"] for item in candidates), default=0.0)
    return {
        "available": True,
        "status": "applied",
        "enabled": True,
        "attention_threshold": attention_threshold,
        "baseline": {
            "median_score": median_score,
            "mad_score": mad,
            "median_rms": median_rms,
            "median_silence_ratio": median_silence,
        },
        "rupture_candidate_count": len(candidates),
        "rupture_score": rupture_score,
        "rupture_candidates": candidates,
        "explanations": (
            ["A memoria temporal encontrou janelas que destoam do comportamento interno do proprio audio."]
            if candidates
            else ["A memoria temporal nao encontrou ruptura local relevante."]
        ),
    }


def _evaluate_regional_bias_hypothesis(symbolic_features: dict, segment_summary: dict, flags: dict) -> dict:
    duration = _safe_float(symbolic_features.get("duration_seconds"))
    silence_ratio = _safe_float(symbolic_features.get("silence_ratio"))
    source_sample_rate = int(symbolic_features.get("source_sample_rate") or 0)
    path = str(symbolic_features.get("path") or "").lower()
    extension = Path(path).suffix.lower()

    evidence = []
    if extension in {".ogg", ".m4a", ".mp3"}:
        evidence.append("compressed_or_mobile_format")
    if source_sample_rate not in {0, 16000}:
        evidence.append("resampled_from_external_sample_rate")
    if duration >= DEFAULT_THRESHOLDS["domain_warning_real_seconds"]:
        evidence.append("long_form_audio")
    if silence_ratio >= DEFAULT_THRESHOLDS["moderate_silence_ratio"]:
        evidence.append("natural_pause_or_recording_silence")

    should_review = (
        flags.get("consistent_suspicious_segments")
        and segment_summary.get("fake_ratio", 0.0) >= DEFAULT_THRESHOLDS["consistent_fake_ratio"]
        and bool(evidence)
    )
    return {
        "review_needed": bool(should_review),
        "hypothesis": "regional_or_domain_false_positive" if should_review else "not_triggered",
        "evidence": evidence,
        "note": (
            "O padrao pode refletir sotaque, regionalismo, compressao ou dominio fora do treino; "
            "precisa ser comparado com a base brasileira."
            if should_review
            else "Sem evidencia simbolica suficiente para levantar hipotese regional/domino."
        ),
    }


def evaluate_audio_symbolic_rules(
    neural_result: dict,
    symbolic_features: dict,
    segments: list[dict] | None = None,
    thresholds: dict | None = None,
) -> dict:
    """Apply first-pass symbolic rules over neural score and audio quality."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    flags = _base_flags()
    explanations: list[str] = []
    adjustments: list[dict] = []

    p_fake = _safe_float(neural_result.get("p_fake"))
    threshold = _safe_float(neural_result.get("threshold"), 0.5)
    margin = p_fake - threshold
    symbolic_score = 0.50
    segments = segments or neural_result.get("segments") or []
    segment_summary = _summarize_segments(segments, threshold)
    segment_diagnostics = neural_result.get("segment_diagnostics") or {}
    temporal_memory = neural_result.get("temporal_memory") or {}
    flags["temporal_memory_available"] = bool(temporal_memory.get("available"))

    if not symbolic_features.get("available", True):
        flags["feature_extraction_failed"] = True
        explanations.append("Nao foi possivel extrair todos os sinais acusticos interpretaveis.")
        adjustments.append({"reason": "feature_extraction_failed", "delta": -0.10})
        symbolic_score -= 0.10
    else:
        duration = _safe_float(symbolic_features.get("duration_seconds"))
        analyzed_fraction = _safe_float(symbolic_features.get("analyzed_fraction"), 1.0)
        silence_ratio = _safe_float(symbolic_features.get("silence_ratio"))
        rms_mean = _safe_float(symbolic_features.get("rms_mean"))
        clipping_ratio = _safe_float(symbolic_features.get("clipping_ratio"))

        if duration < thresholds["very_short_seconds"]:
            flags["very_short_audio"] = True
            flags["short_audio"] = True
            explanations.append("Audio curto demais para uma analise confiavel.")
            adjustments.append({"reason": "very_short_audio", "delta": -0.30})
            symbolic_score -= 0.30
        elif duration < thresholds["short_seconds"]:
            flags["short_audio"] = True
            explanations.append("Audio curto; a confianca da decisao foi reduzida.")
            adjustments.append({"reason": "short_audio", "delta": -0.20})
            symbolic_score -= 0.20

        if analyzed_fraction < thresholds["partial_analysis_ratio"]:
            flags["partial_audio_analysis"] = True
            explanations.append("O modelo analisou apenas uma janela do audio completo; a conclusao precisa de revisao por segmentos.")
            adjustments.append({"reason": "partial_audio_analysis", "delta": -0.15})
            symbolic_score -= 0.15

        if silence_ratio >= thresholds["high_silence_ratio"]:
            flags["high_silence"] = True
            explanations.append("O arquivo contem muito silencio em relacao ao tempo total.")
            adjustments.append({"reason": "high_silence", "delta": -0.15})
            symbolic_score -= 0.15

        if rms_mean <= thresholds["low_volume_rms"]:
            flags["low_volume"] = True
            explanations.append("O volume medio esta baixo, o que pode reduzir a confiabilidade.")
            adjustments.append({"reason": "low_volume", "delta": -0.10})
            symbolic_score -= 0.10

        if clipping_ratio >= thresholds["clipping_ratio"]:
            flags["clipped_audio"] = True
            explanations.append("Ha sinais de saturacao no audio.")
            adjustments.append({"reason": "clipped_audio", "delta": -0.10})
            symbolic_score -= 0.10

    if abs(margin) <= thresholds["near_threshold_margin"]:
        flags["near_threshold"] = True
        explanations.append("O score ficou proximo ao limite de decisao.")
        adjustments.append({"reason": "near_threshold", "delta": -0.10})
        symbolic_score -= 0.10
    elif margin >= thresholds["strong_margin"]:
        flags["strong_fake_score"] = True
        explanations.append("O modelo neural indicou alta probabilidade de audio sintetico.")
        adjustments.append({"reason": "strong_fake_score", "delta": 0.15})
        symbolic_score += 0.15
    elif margin <= -thresholds["strong_margin"]:
        flags["strong_real_score"] = True
        explanations.append("O modelo neural ficou bem abaixo do limite de audio fake.")
        adjustments.append({"reason": "strong_real_score", "delta": -0.10})
        symbolic_score -= 0.10

    if p_fake >= threshold and margin < thresholds["strong_margin"]:
        flags["low_confidence_fake"] = True
        explanations.append("A classificacao fake ocorreu com margem pequena.")

    if segment_summary["count"] >= 2:
        if segment_summary["score_range"] >= thresholds["segment_disagreement_range"]:
            flags["segment_score_disagreement"] = True
            explanations.append("Os scores por segmento variaram bastante ao longo do audio.")
            adjustments.append({"reason": "segment_score_disagreement", "delta": -0.05})
            symbolic_score -= 0.05

        if (
            0 < segment_summary["fake_ratio"] <= thresholds["isolated_fake_max_ratio"]
            and segment_summary["score_range"] >= thresholds["segment_disagreement_range"]
        ):
            flags["isolated_suspicious_segment"] = True
            explanations.append("A suspeita se concentrou em poucos trechos, nao no audio inteiro.")
            adjustments.append({"reason": "isolated_suspicious_segment", "delta": -0.05})
            symbolic_score -= 0.05

        if segment_summary["fake_ratio"] >= thresholds["consistent_fake_ratio"]:
            flags["consistent_suspicious_segments"] = True
            explanations.append("A maior parte das janelas analisadas ficou acima do limite de audio fake.")
            adjustments.append({"reason": "consistent_suspicious_segments", "delta": 0.10})
            symbolic_score += 0.10

    diagnostic_summary = segment_diagnostics.get("summary") or {}
    if diagnostic_summary.get("terminal_suspicious_count", 0) > 0:
        flags["terminal_suspicious_segments"] = True
        explanations.append("Os trechos acima do limite aparecem no fim do audio; vale revisar encerramento, corte ou compressao.")
    if diagnostic_summary.get("possible_end_artifact"):
        flags["possible_end_artifact"] = True
        explanations.append("Ha sinais acusticos no trecho final que podem explicar pico localizado do detector.")
        adjustments.append({"reason": "possible_end_artifact", "delta": -0.05})
        symbolic_score -= 0.05

    regional_bias_review = _evaluate_regional_bias_hypothesis(symbolic_features, segment_summary, flags)
    if regional_bias_review["review_needed"]:
        flags["regional_bias_review"] = True
        flags["domain_mismatch_warning"] = True
        explanations.append("Ha hipotese alternativa de falso positivo por regionalismo, compressao ou dominio fora do treino.")
        adjustments.append({"reason": "regional_bias_review", "delta": -0.20})
        symbolic_score -= 0.20

    phonetic_review = apply_phonetic_rules(
        neural_result.get("phonetic_events") or symbolic_features.get("phonetic_events") or [],
        neural_result,
    )
    flags["phonetic_layer_available"] = bool(phonetic_review.get("available"))

    phonetic_status = phonetic_review.get("status")
    if phonetic_status == "insufficient_coverage":
        flags["phonetic_coverage_insufficient"] = True
        explanations.append("A camada fonetico-fonologica foi executada, mas a cobertura de eventos confiaveis ainda foi baixa.")
    elif phonetic_status == "applied":
        phonetic_adjustment = _safe_float(phonetic_review.get("adjustment"))
        if phonetic_adjustment < 0:
            flags["phonetic_consistency_support"] = True
        elif phonetic_adjustment > 0:
            flags["phonetic_disruption_alert"] = True

        if phonetic_adjustment:
            adjustments.append({"reason": "phonetic_symbolic_layer", "delta": phonetic_adjustment})
            symbolic_score += phonetic_adjustment
            explanations.extend(phonetic_review.get("explanations") or [])
    elif phonetic_status == "no_rule_triggered":
        event_count = int(phonetic_review.get("event_count") or 0)
        evidence_quality = phonetic_review.get("evidence_quality") or {}
        quality_score = _safe_float(evidence_quality.get("quality_score"))
        if event_count > 0:
            explanations.append(
                "A camada fonetico-fonologica encontrou eventos analisaveis, mas nao detectou variacao forte ou ruptura linguistica."
            )
            if quality_score:
                explanations.append(f"Qualidade da evidencia fonetica: {quality_score:.2f} em uma escala de 0 a 1.")

    no_phonetic_support = (
        flags["phonetic_coverage_insufficient"]
        or phonetic_status in {"insufficient_coverage", "no_rule_triggered", "unavailable"}
        or not flags["phonetic_layer_available"]
    )
    localized_without_artifact = (
        flags["isolated_suspicious_segment"]
        and flags["segment_score_disagreement"]
        and segment_summary["fake_ratio"] >= thresholds["localized_sensitivity_min_fake_ratio"]
        and not flags["possible_end_artifact"]
        and not flags["regional_bias_review"]
        and not flags["phonetic_consistency_support"]
    )
    if localized_without_artifact and no_phonetic_support:
        flags["localized_manipulation_sensitivity"] = True
        explanations.append(
            "A suspeita localizada permaneceu sem evidencia fonetica suficiente para reduzir o alerta."
        )
        delta = thresholds["localized_sensitivity_delta"]
        adjustments.append({"reason": "localized_manipulation_sensitivity", "delta": delta})
        symbolic_score += delta

    temporal_candidates = temporal_memory.get("rupture_candidates") or []
    if temporal_candidates:
        flags["temporal_memory_rupture_candidate"] = True
        explanations.append(
            "A memoria temporal encontrou janela(s) que destoam do comportamento interno do audio."
        )
        if (
            segment_summary["fake_count"] == 0
            and no_phonetic_support
            and not flags["phonetic_consistency_support"]
            and not flags["possible_end_artifact"]
            and not flags["regional_bias_review"]
        ):
            delta = thresholds["temporal_symbolic_delta"]
            adjustments.append({"reason": "temporal_memory_rupture_candidate", "delta": delta})
            symbolic_score += delta

    symbolic_score = max(0.0, min(1.0, symbolic_score))
    active_quality_flags = [
        flags["short_audio"],
        flags["high_silence"],
        flags["low_volume"],
        flags["clipped_audio"],
        flags["feature_extraction_failed"],
        flags["partial_audio_analysis"],
        flags["regional_bias_review"],
        flags["domain_mismatch_warning"],
        flags["possible_end_artifact"],
    ]

    if flags["very_short_audio"] or (flags["high_silence"] and flags["low_volume"]):
        confidence = "low"
    elif (
        flags["near_threshold"]
        or flags["isolated_suspicious_segment"]
        or flags["terminal_suspicious_segments"]
        or flags["phonetic_coverage_insufficient"]
        or flags["temporal_memory_rupture_candidate"]
        or any(active_quality_flags)
    ):
        confidence = "moderate"
    elif flags["phonetic_disruption_alert"]:
        confidence = "high"
    elif flags["consistent_suspicious_segments"] and (flags["strong_fake_score"] or p_fake >= threshold):
        confidence = "high"
    elif flags["strong_fake_score"] or flags["strong_real_score"]:
        confidence = "high"
    else:
        confidence = "moderate"

    if not explanations:
        explanations.append("Nenhum alerta simbolico relevante foi encontrado na analise inicial.")

    return {
        "symbolic_score": symbolic_score,
        "confidence": confidence,
        "flags": flags,
        "explanations": explanations,
        "adjustments": adjustments,
        "segment_summary": segment_summary,
        "segment_diagnostics": segment_diagnostics,
        "regional_bias_review": regional_bias_review,
        "phonetic_review": phonetic_review,
        "temporal_memory": temporal_memory,
        "thresholds": thresholds,
    }


def combine_audio_decision(neural_result: dict, symbolic_result: dict) -> dict:
    """Combine neural score with symbolic review into a final auditable decision."""
    p_fake = _safe_float(neural_result.get("p_fake"))
    threshold = _safe_float(neural_result.get("threshold"), 0.5)
    symbolic_score = _safe_float(symbolic_result.get("symbolic_score"), 0.5)
    flags = symbolic_result.get("flags") or {}

    final_score = max(0.0, min(1.0, (0.70 * p_fake) + (0.30 * symbolic_score)))

    segment_summary = symbolic_result.get("segment_summary") or {}
    segment_mean = segment_summary.get("mean_score")
    decision_score = _safe_float(segment_mean, final_score) if segment_summary.get("count", 0) >= 2 else final_score

    if flags.get("very_short_audio") or (flags.get("high_silence") and flags.get("low_volume")):
        decision = "undetermined"
        risk_level = "undetermined"
    elif flags.get("near_threshold"):
        decision = "inconclusive"
        risk_level = "medium"
    elif flags.get("isolated_suspicious_segment"):
        decision = "suspicious_segment"
        risk_level = "medium"
    elif decision_score >= threshold:
        decision = "fake"
        risk_level = "high" if decision_score >= min(threshold + 0.25, 0.80) else "medium"
        if (
            flags.get("regional_bias_review")
            or (flags.get("partial_audio_analysis") and not flags.get("consistent_suspicious_segments"))
        ):
            risk_level = "medium"
    else:
        decision = "real"
        risk_level = "low"

    return {
        "decision": decision,
        "risk_level": risk_level,
        "confidence": symbolic_result.get("confidence", "moderate"),
        "neural_score": p_fake,
        "symbolic_score": symbolic_score,
        "final_score": final_score,
        "decision_score": decision_score,
        "threshold": threshold,
        "flags": flags,
        "explanations": symbolic_result.get("explanations", []),
        "adjustments": symbolic_result.get("adjustments", []),
        "segment_summary": segment_summary,
        "segment_diagnostics": symbolic_result.get("segment_diagnostics", {}),
        "regional_bias_review": symbolic_result.get("regional_bias_review", {}),
        "phonetic_review": symbolic_result.get("phonetic_review", {}),
        "temporal_memory": symbolic_result.get("temporal_memory", {}),
    }


def generate_audio_executive_summary(audio_summary: dict) -> dict:
    """Build a concise, user-facing explanation for the audio decision."""
    symbolic = audio_summary.get("symbolic") or {}
    decision = symbolic.get("decision") or audio_summary.get("decision") or "undetermined"
    risk_level = symbolic.get("risk_level") or "undetermined"
    confidence = symbolic.get("confidence") or "moderate"
    segment_summary = symbolic.get("segment_summary") or audio_summary.get("segment_summary") or {}
    segment_diagnostics = symbolic.get("segment_diagnostics") or audio_summary.get("segment_diagnostics") or {}
    diagnostic_summary = segment_diagnostics.get("summary") or {}
    phonetic_review = symbolic.get("phonetic_review") or {}
    temporal_memory = symbolic.get("temporal_memory") or audio_summary.get("temporal_memory") or {}
    ground_truth = audio_summary.get("ground_truth") or {}
    flags = symbolic.get("flags") or {}

    decision_labels = {
        "real": "real",
        "fake": "fake",
        "inconclusive": "inconclusivo",
        "suspicious_segment": "com suspeita localizada",
        "undetermined": "indeterminado",
    }
    risk_labels = {
        "low": "baixo",
        "medium": "médio",
        "high": "alto",
        "undetermined": "indeterminado",
    }
    confidence_labels = {
        "low": "baixa",
        "moderate": "moderada",
        "high": "alta",
    }

    title = f"Áudio {decision_labels.get(decision, decision)}"
    if decision == "real" and flags.get("terminal_suspicious_segments"):
        title = "Áudio provavelmente real, com revisão recomendada"
    elif decision == "inconclusive" and flags.get("phonetic_consistency_support"):
        title = "Áudio inconclusivo, com sinais linguísticos compatíveis com fala natural"
    elif decision == "suspicious_segment":
        title = "Suspeita localizada em trecho específico"

    lines = [
        f"O áudio foi classificado como {decision_labels.get(decision, decision)}.",
        (
            "O nível de risco estimado é "
            f"{risk_labels.get(risk_level, risk_level)} e a confiança da leitura simbólica é "
            f"{confidence_labels.get(confidence, confidence)}."
        ),
    ]

    fake_count = int(segment_summary.get("fake_count") or 0)
    count = int(segment_summary.get("count") or 0)
    fake_ratio = _safe_float(segment_summary.get("fake_ratio"))
    if count > 0:
        if fake_count == 0:
            lines.append("Nenhuma janela ultrapassou o limite calibrado de áudio fake.")
        elif diagnostic_summary.get("terminal_suspicious_count", 0) > 0:
            lines.append(
                f"A suspeita neural ficou concentrada em {fake_count} de {count} janelas, principalmente no final do áudio."
            )
            lines.append(
                "Essas janelas são tratadas como indício de possível ruptura localizada, não como prova isolada de manipulação."
            )
        elif fake_ratio <= DEFAULT_THRESHOLDS["isolated_fake_max_ratio"]:
            lines.append(f"A suspeita neural ficou concentrada em poucos trechos: {fake_count} de {count} janelas.")
            lines.append(
                "Essas janelas são tratadas como indício de possível ruptura localizada, não como prova isolada de manipulação."
            )
        else:
            lines.append(f"{fake_count} de {count} janelas ficaram acima do limite calibrado.")

    if phonetic_review.get("status") == "applied":
        adjustment = _safe_float(phonetic_review.get("adjustment"))
        if adjustment < 0:
            lines.append("Foram encontrados padrões fonético-fonológicos compatíveis com fala natural.")
        elif adjustment > 0:
            lines.append("A camada fonético-fonológica encontrou sinais de ruptura que aumentam a suspeita.")
        else:
            lines.append("A camada fonético-fonológica foi aplicada, mas não alterou a decisão.")
    elif phonetic_review.get("status") == "insufficient_coverage":
        lines.append("A camada fonético-fonológica não teve eventos suficientes para calibrar a decisão.")
    elif phonetic_review.get("status") == "no_rule_triggered":
        event_count = int(phonetic_review.get("event_count") or 0)
        evidence_quality = phonetic_review.get("evidence_quality") or {}
        quality_score = _safe_float(evidence_quality.get("quality_score"))
        if event_count > 0:
            lines.append(
                "A camada fonético-fonológica analisou eventos do áudio, "
                "mas não encontrou variação relevante ou ruptura linguística forte para calibrar a decisão."
            )
            if quality_score:
                lines.append(f"A qualidade da evidência fonética ficou em {quality_score:.2f} em uma escala de 0 a 1.")
        else:
            lines.append("Eventos fonéticos foram extraídos, mas nenhuma regra linguística forte foi acionada.")

    if diagnostic_summary.get("terminal_suspicious_count", 0) > 0:
        if diagnostic_summary.get("possible_end_artifact"):
            lines.append("Há sinais acústicos no encerramento que podem explicar parte do pico final.")
        else:
            lines.append("Os picos finais não parecem ser explicados apenas por silêncio, clipping ou salto brusco de volume.")

    if temporal_memory.get("rupture_candidate_count", 0) > 0:
        lines.append(
            "A memória temporal encontrou trecho(s) que destoam do comportamento interno do próprio áudio."
        )

    if ground_truth.get("available"):
        overlap = ground_truth.get("overlap") or {}
        if overlap.get("localized_hit"):
            lines.append("No teste controlado, os picos detectados coincidiram com o trecho manipulado.")
        else:
            lines.append("No teste controlado, os picos detectados não coincidiram claramente com o trecho manipulado.")

    if decision in {"inconclusive", "suspicious_segment"} or flags.get("terminal_suspicious_segments"):
        if flags.get("terminal_suspicious_segments"):
            recommendation = "O sistema mantém a decisão como inconclusiva e destaca automaticamente os trechos finais como possível ruptura para análise contextual."
        else:
            recommendation = "O sistema mantém a decisão como inconclusiva e destaca automaticamente os trechos suspeitos como possível ruptura para análise contextual."
    elif decision == "fake":
        recommendation = "O sistema mantém a classificação como suspeita e prioriza os trechos com maior score para explicação automática."
    elif decision == "real":
        recommendation = "Não há evidência simbólica forte de manipulação, mas a decisão deve ser interpretada dentro do contexto do áudio."
    else:
        recommendation = "Recomenda-se nova análise com áudio de melhor qualidade ou maior duração."

    lines.append(recommendation)

    return {
        "title": title,
        "decision": decision,
        "risk_level": risk_level,
        "confidence": confidence,
        "summary": " ".join(lines),
        "bullets": lines,
        "recommendation": recommendation,
    }
