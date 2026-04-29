import logging
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

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
    "domain_warning_real_seconds": 8.0,
    "moderate_silence_ratio": 0.35,
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
            adjustments.append({"reason": "isolated_suspicious_segment", "delta": -0.10})
            symbolic_score -= 0.10

        if segment_summary["fake_ratio"] >= thresholds["consistent_fake_ratio"]:
            flags["consistent_suspicious_segments"] = True
            explanations.append("A maior parte das janelas analisadas ficou acima do limite de audio fake.")
            adjustments.append({"reason": "consistent_suspicious_segments", "delta": 0.10})
            symbolic_score += 0.10

    regional_bias_review = _evaluate_regional_bias_hypothesis(symbolic_features, segment_summary, flags)
    if regional_bias_review["review_needed"]:
        flags["regional_bias_review"] = True
        flags["domain_mismatch_warning"] = True
        explanations.append("Ha hipotese alternativa de falso positivo por regionalismo, compressao ou dominio fora do treino.")
        adjustments.append({"reason": "regional_bias_review", "delta": -0.20})
        symbolic_score -= 0.20

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
    ]

    if flags["very_short_audio"] or (flags["high_silence"] and flags["low_volume"]):
        confidence = "low"
    elif flags["near_threshold"] or flags["isolated_suspicious_segment"] or any(active_quality_flags):
        confidence = "moderate"
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
        "regional_bias_review": regional_bias_review,
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
        "regional_bias_review": symbolic_result.get("regional_bias_review", {}),
    }
