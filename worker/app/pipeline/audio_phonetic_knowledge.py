from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_KNOWLEDGE_PATH = BASE_DIR / "audio" / "knowledge" / "phonetic_rules_v1.yaml"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=4)
def load_phonetic_knowledge(path: str | None = None) -> dict:
    """Load the phonetic-symbolic knowledge base."""
    knowledge_path = Path(path) if path else DEFAULT_KNOWLEDGE_PATH
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is not installed; phonetic knowledge base cannot be loaded.")
        return {
            "available": False,
            "path": str(knowledge_path),
            "error": "pyyaml_not_installed",
            "thresholds": {},
            "rules": [],
        }

    try:
        with knowledge_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception as exc:
        logger.warning("Failed to load phonetic knowledge base from %s: %s", knowledge_path, exc)
        return {
            "available": False,
            "path": str(knowledge_path),
            "error": str(exc),
            "thresholds": {},
            "rules": [],
        }

    data["available"] = True
    data["path"] = str(knowledge_path)
    data.setdefault("thresholds", {})
    data.setdefault("rules", [])
    return data


def _event_confidence(event: dict) -> float:
    return _safe_float(
        event.get("confidence"),
        _safe_float(event.get("variant_confidence"), 0.0),
    )


def _valid_events(events: list[dict], min_confidence: float) -> list[dict]:
    return [
        event
        for event in events
        if (
            isinstance(event, dict)
            and _event_confidence(event) >= min_confidence
            and str(event.get("variant") or "").lower() not in {"", "unknown"}
        )
    ]


def _classified_events(events: list[dict]) -> list[dict]:
    return [
        event
        for event in events
        if isinstance(event, dict) and str(event.get("variant") or "").lower() not in {"", "unknown"}
    ]


def _average_confidence(events: list[dict]) -> float:
    if not events:
        return 0.0
    return sum(_event_confidence(event) for event in events) / float(len(events))


def _phenomenon_diversity_score(events: list[dict]) -> float:
    if not events:
        return 0.0
    phenomena = {str(event.get("phenomenon") or "") for event in events if event.get("phenomenon")}
    return min(1.0, len(phenomena) / 3.0)


def _group_events(events: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for event in events:
        group_key = tuple(event.get(key) for key in keys)
        groups.setdefault(group_key, []).append(event)
    return groups


def _contextual_key(event: dict) -> tuple:
    return (
        event.get("phenomenon"),
        event.get("context"),
        event.get("position"),
        event.get("target_segment"),
        event.get("following_segment") or event.get("following_vowel") or "",
    )


def _dominant_variant_score(events: list[dict]) -> float:
    if not events:
        return 0.0
    counts: dict[str, int] = {}
    for event in events:
        variant = str(event.get("variant") or "unknown")
        counts[variant] = counts.get(variant, 0) + 1
    return max(counts.values()) / float(len(events))


def _variant_ratio(events: list[dict], variants: set[str]) -> float:
    if not events:
        return 0.0
    count = sum(1 for event in events if str(event.get("variant") or "") in variants)
    return count / float(len(events))


def _has_context(events: list[dict], contexts: set[str]) -> bool:
    return any(str(event.get("context") or "") in contexts for event in events)


def _event_variants(events: list[dict]) -> set[str]:
    return {str(event.get("variant") or "") for event in events if event.get("variant")}


def _has_repeated_variant_contrast(events: list[dict]) -> bool:
    counts: dict[str, int] = {}
    for event in events:
        variant = str(event.get("variant") or "")
        if not variant or variant == "unknown":
            continue
        counts[variant] = counts.get(variant, 0) + 1
    repeated_variants = [variant for variant, count in counts.items() if count >= 2]
    return len(repeated_variants) >= 2


def calculate_coverage(events: list[dict], thresholds: dict) -> tuple[int, float]:
    min_required = max(1, _safe_int(thresholds.get("min_required_events"), 10))
    event_count = len(events)
    return event_count, min(1.0, event_count / float(min_required))


def calculate_phonetic_consistency(events: list[dict]) -> float:
    """Estimate consistency inside equivalent phenomenon/context groups."""
    if not events:
        return 0.0

    groups: dict[tuple, list[dict]] = {}
    for event in events:
        groups.setdefault(_contextual_key(event), []).append(event)

    group_scores = []
    for grouped_events in groups.values():
        if len(grouped_events) >= 2:
            group_scores.append(_dominant_variant_score(grouped_events))

    if not group_scores:
        return _dominant_variant_score(events)
    return sum(group_scores) / float(len(group_scores))


def calculate_localized_disruption(events: list[dict], window_seconds: float) -> float:
    """Detect abrupt changes in nearby events with similar context."""
    if len(events) < 2:
        return 0.0

    sorted_events = sorted(events, key=lambda item: _safe_float(item.get("start_time")))
    comparisons = 0
    disruptions = 0

    for index, current in enumerate(sorted_events[:-1]):
        current_start = _safe_float(current.get("start_time"))
        for candidate in sorted_events[index + 1 :]:
            delta = _safe_float(candidate.get("start_time")) - current_start
            if delta > window_seconds:
                break
            same_context = _contextual_key(current) == _contextual_key(candidate)
            if not same_context:
                continue
            comparisons += 1
            if current.get("variant") != candidate.get("variant"):
                disruptions += 1

    if comparisons == 0:
        return 0.0
    return min(1.0, disruptions / float(comparisons))


def detect_localized_disruptions(
    events: list[dict],
    *,
    window_seconds: float,
    threshold: float,
    peak_margin: float,
) -> dict:
    """Find local variant switches that also coincide with a neural peak.

    A disruption is intentionally stricter than simple variation: it requires
    two classified events with the same contextual key, different variants,
    temporal proximity, and local neural support near the calibrated threshold.
    """
    if len(events) < 2:
        return {
            "available": False,
            "comparison_count": 0,
            "raw_disruption_count": 0,
            "aligned_disruption_count": 0,
            "raw_score": 0.0,
            "aligned_score": 0.0,
            "phenomena": [],
            "items": [],
        }

    peak_cutoff = max(0.0, threshold - peak_margin)
    groups: dict[tuple, list[dict]] = {}
    for event in events:
        groups.setdefault(_contextual_key(event), []).append(event)

    comparison_count = 0
    raw_items = []
    aligned_items = []

    for key, grouped_events in groups.items():
        if len(grouped_events) < 2:
            continue
        sorted_events = sorted(grouped_events, key=lambda item: _safe_float(item.get("start_time")))
        for index, current in enumerate(sorted_events[:-1]):
            current_start = _safe_float(current.get("start_time"))
            for candidate in sorted_events[index + 1 :]:
                delta = _safe_float(candidate.get("start_time")) - current_start
                if delta > window_seconds:
                    break
                comparison_count += 1
                if current.get("variant") == candidate.get("variant"):
                    continue

                local_scores = [
                    _safe_float(item.get("neural_score_local"))
                    for item in (current, candidate)
                    if item.get("neural_score_local") is not None
                ]
                max_local_score = max(local_scores) if local_scores else 0.0
                item = {
                    "phenomenon": current.get("phenomenon"),
                    "context": current.get("context"),
                    "position": current.get("position"),
                    "target_segment": current.get("target_segment"),
                    "following_segment": current.get("following_segment") or current.get("following_vowel"),
                    "start_time": current_start,
                    "end_time": _safe_float(candidate.get("start_time")),
                    "delta_seconds": delta,
                    "from_variant": current.get("variant"),
                    "to_variant": candidate.get("variant"),
                    "from_word": current.get("word"),
                    "to_word": candidate.get("word"),
                    "max_neural_score_local": max_local_score,
                    "aligned_with_neural_peak": max_local_score >= peak_cutoff,
                }
                raw_items.append(item)
                if item["aligned_with_neural_peak"]:
                    aligned_items.append(item)

    raw_score = len(raw_items) / float(comparison_count) if comparison_count else 0.0
    aligned_score = len(aligned_items) / float(comparison_count) if comparison_count else 0.0
    phenomena = sorted({str(item.get("phenomenon")) for item in aligned_items if item.get("phenomenon")})
    aligned_by_phenomenon: dict[str, int] = {}
    for item in aligned_items:
        phenomenon = str(item.get("phenomenon") or "")
        if phenomenon:
            aligned_by_phenomenon[phenomenon] = aligned_by_phenomenon.get(phenomenon, 0) + 1

    return {
        "available": bool(raw_items),
        "comparison_count": comparison_count,
        "raw_disruption_count": len(raw_items),
        "aligned_disruption_count": len(aligned_items),
        "aligned_disruption_count_by_phenomenon": dict(sorted(aligned_by_phenomenon.items())),
        "raw_score": min(1.0, raw_score),
        "aligned_score": min(1.0, aligned_score),
        "phenomena": phenomena,
        "items": aligned_items[:12],
        "unfiltered_items": raw_items[:12],
        "peak_cutoff": peak_cutoff,
        "window_seconds": window_seconds,
    }


def smooth_disruption_score(raw_score: float, event_count: int, min_events_for_full_weight: int = 12) -> float:
    if event_count <= 0:
        return 0.0
    evidence_weight = min(1.0, event_count / float(max(1, min_events_for_full_weight)))
    return raw_score * evidence_weight


def calculate_evidence_quality(events: list[dict], coverage_score: float) -> dict:
    classified = _classified_events(events)
    total_count = len(events)
    classified_count = len(classified)
    avg_confidence = _average_confidence(classified)
    classified_ratio = classified_count / float(total_count) if total_count else 0.0
    diversity_score = _phenomenon_diversity_score(classified)
    quality_score = coverage_score * avg_confidence * (0.60 + 0.40 * diversity_score)
    return {
        "total_event_count": total_count,
        "classified_event_count": classified_count,
        "classified_event_ratio": classified_ratio,
        "average_confidence": avg_confidence,
        "phenomenon_diversity_score": diversity_score,
        "quality_score": max(0.0, min(1.0, quality_score)),
    }


def cap_adjustment_by_evidence(adjustment: float, event_count: int, quality_score: float) -> float:
    """Avoid strong symbolic correction when event evidence is still thin."""
    if adjustment == 0:
        return 0.0

    if event_count < 6 or quality_score < 0.35:
        cap = 0.05
    elif event_count < 10 or quality_score < 0.55:
        cap = 0.08
    elif event_count < 15 or quality_score < 0.70:
        cap = 0.12
    elif event_count < 22 or quality_score < 0.82:
        cap = 0.16
    else:
        cap = 0.20

    return max(-cap, min(cap, adjustment))


def calculate_alignment_score(events: list[dict], threshold: float, peak_margin: float) -> float:
    """Estimate whether linguistic disruptions coincide with high local neural scores."""
    if not events:
        return 0.0
    local_scores = [
        _safe_float(event.get("neural_score_local"))
        for event in events
        if event.get("neural_score_local") is not None
    ]
    if not local_scores:
        return 0.0

    peak_cutoff = max(0.0, threshold - peak_margin)
    peak_count = sum(1 for score in local_scores if score >= peak_cutoff)
    return peak_count / float(len(local_scores))


def _activate_rule(rule: dict, reason: str | None = None) -> dict:
    activated = {
        "rule_id": rule.get("id"),
        "phenomenon": rule.get("phenomenon"),
        "condition": rule.get("condition"),
        "effect": rule.get("effect"),
        "weight": _safe_float(rule.get("weight")),
        "explanation": rule.get("explanation"),
    }
    if reason:
        activated["reason"] = reason
    return activated


def _events_for_rule(events: list[dict], rule: dict) -> list[dict]:
    phenomenon = rule.get("phenomenon")
    if phenomenon == "global":
        return events
    return [event for event in events if event.get("phenomenon") == phenomenon]


def apply_phonetic_rules(
    events: list[dict] | None,
    neural_result: dict,
    knowledge: dict | None = None,
) -> dict:
    """Apply phonetic-symbolic rules over extracted events.

    The current pipeline does not extract ASR/alignment events yet. In that case
    this function returns an auditable "insufficient coverage" report, keeping
    the neural decision unchanged.
    """
    knowledge = knowledge or load_phonetic_knowledge()
    thresholds = knowledge.get("thresholds") or {}
    all_events = events or []
    min_confidence = _safe_float(thresholds.get("min_rule_confidence"), 0.70)
    valid_events = _valid_events(all_events, min_confidence)
    event_count, coverage_score = calculate_coverage(valid_events, thresholds)
    evidence_quality = calculate_evidence_quality(all_events, coverage_score)
    consistency_score = calculate_phonetic_consistency(valid_events)
    neural_score = _safe_float(neural_result.get("p_fake"))
    threshold = _safe_float(neural_result.get("threshold"), 0.5)
    peak_margin = _safe_float(thresholds.get("neural_peak_margin"), 0.08)
    window_seconds = _safe_float(thresholds.get("disruption_window_seconds"), 6.0)
    disruption_details = detect_localized_disruptions(
        valid_events,
        window_seconds=window_seconds,
        threshold=threshold,
        peak_margin=peak_margin,
    )
    raw_disruption_score = disruption_details["aligned_score"]
    disruption_score = smooth_disruption_score(raw_disruption_score, event_count)

    alignment_score = calculate_alignment_score(
        valid_events,
        threshold,
        peak_margin,
    )
    plausibility_score = max(0.0, min(1.0, (0.60 * consistency_score) + (0.40 * (1.0 - disruption_score))))

    activated_rules: list[dict] = []
    explanations: list[str] = []

    if not knowledge.get("available", False):
        return {
            "available": False,
            "status": "knowledge_unavailable",
            "knowledge_path": knowledge.get("path"),
            "error": knowledge.get("error"),
            "event_count": event_count,
            "coverage_score": coverage_score,
            "evidence_quality": evidence_quality,
            "adjustment": 0.0,
            "activated_rules": [],
            "explanations": ["A base de conhecimento fonetico-fonologica nao esta disponivel."],
        }

    rules = knowledge.get("rules") or []
    low_coverage_cutoff = _safe_float(thresholds.get("low_coverage_cutoff"), 0.30)
    normal_coverage_cutoff = _safe_float(thresholds.get("normal_coverage_cutoff"), 0.60)
    localized_disruption_min_pairs = _safe_int(thresholds.get("localized_disruption_min_pairs"), 2)
    aligned_disruption_min_pairs = _safe_int(thresholds.get("aligned_disruption_min_pairs"), 2)
    multi_phenomena_disruption_min_phenomena = _safe_int(thresholds.get("multi_phenomena_disruption_min_phenomena"), 2)
    has_repeated_aligned_disruption = disruption_details["aligned_disruption_count"] >= aligned_disruption_min_pairs
    has_localized_disruption = disruption_details["raw_disruption_count"] >= localized_disruption_min_pairs
    has_multi_phenomena_disruption = (
        has_repeated_aligned_disruption
        and len(disruption_details["phenomena"]) >= multi_phenomena_disruption_min_phenomena
    )
    aligned_disruption_count_by_phenomenon = disruption_details.get("aligned_disruption_count_by_phenomenon") or {}

    if coverage_score < low_coverage_cutoff:
        rule = next((item for item in rules if item.get("id") == "G3"), None)
        if rule:
            activated_rules.append(_activate_rule(rule, "coverage_below_minimum"))
            explanations.append(rule.get("explanation"))

        return {
            "available": True,
            "status": "insufficient_coverage",
            "knowledge_version": knowledge.get("version"),
            "knowledge_path": knowledge.get("path"),
            "event_count": event_count,
            "coverage_score": coverage_score,
            "evidence_quality": evidence_quality,
            "phonetic_consistency_score": consistency_score,
            "linguistic_plausibility_score": plausibility_score,
            "raw_localized_disruption_score": raw_disruption_score,
            "localized_disruption_score": disruption_score,
            "localized_disruption": disruption_details,
            "alignment_score": alignment_score,
            "adjustment": 0.0,
            "activated_rules": activated_rules,
            "explanations": explanations,
        }

    for rule in rules:
        requirements = rule.get("requires") or {}
        rule_events = _events_for_rule(valid_events, rule)
        if len(rule_events) < _safe_int(requirements.get("min_events"), 0):
            continue

        condition = rule.get("condition")
        should_activate = False

        if condition in {
            "dominant_variant_stable",
            "multiple_phenomena_stable",
            "high_neural_score_with_consistent_linguistics",
        }:
            neural_is_high = neural_score >= max(0.0, threshold - _safe_float(thresholds.get("neural_high_margin"), 0.10))
            should_activate = consistency_score >= 0.80 and disruption_score <= 0.25
            if condition == "high_neural_score_with_consistent_linguistics":
                should_activate = should_activate and neural_is_high
        elif condition == "palatal_variant_recurrent_and_stable":
            should_activate = _variant_ratio(rule_events, {"palatal"}) >= 0.70 and disruption_score <= 0.25
        elif condition == "alveolar_variant_recurrent_and_stable":
            should_activate = _variant_ratio(rule_events, {"alveolar"}) >= 0.70 and disruption_score <= 0.25
        elif condition == "palatalized_variant_recurrent":
            should_activate = _variant_ratio(rule_events, {"palatalized", "affricated"}) >= 0.70 and disruption_score <= 0.25
        elif condition == "dentoalveolar_variant_stable":
            should_activate = _variant_ratio(rule_events, {"dentoalveolar"}) >= 0.70 and disruption_score <= 0.25
        elif condition in {
            "abrupt_variant_switch_same_context",
            "abrupt_alveolar_palatal_switch_same_context",
            "abrupt_dentoalveolar_palatalized_switch_same_context",
            "multiple_phenomena_shift_same_interval",
            "high_neural_score_with_aligned_linguistic_disruption",
        }:
            phenomenon = str(rule.get("phenomenon") or "")
            phenomenon_has_disruption = phenomenon == "global" or (
                _safe_int(aligned_disruption_count_by_phenomenon.get(phenomenon), 0) >= aligned_disruption_min_pairs
            )
            should_activate = (
                has_localized_disruption
                and has_repeated_aligned_disruption
                and phenomenon_has_disruption
                and _has_repeated_variant_contrast(rule_events)
            )
            if condition == "high_neural_score_with_aligned_linguistic_disruption":
                should_activate = should_activate and alignment_score >= 0.50
            if condition == "multiple_phenomena_shift_same_interval":
                should_activate = should_activate and has_multi_phenomena_disruption
        elif condition in {
            "verb_infinitive_final_r_deletion_consistent",
        }:
            verb_events = [
                event
                for event in rule_events
                if event.get("context") == "verb_infinitive_external_coda"
            ]
            should_activate = (
                len(verb_events) >= _safe_int(requirements.get("min_events"), 0)
                and _variant_ratio(verb_events, {"deletion"}) >= 0.70
                and disruption_score <= 0.25
            )
        elif condition in {
            "final_r_presence_and_deletion_without_local_disruption",
        }:
            variants = _event_variants(rule_events)
            should_activate = (
                "deletion" in variants
                and len(variants.intersection({"tap", "retroflex", "velar", "glottal", "present"})) > 0
                and disruption_score <= 0.25
            )
        elif condition in {
            "isolated_internal_coda_deletion",
        }:
            internal_deletions = [
                event
                for event in rule_events
                if event.get("context") == "internal_coda" and event.get("variant") == "deletion"
            ]
            should_activate = 0 < len(internal_deletions) <= 2
        elif condition in {
            "palatalization_in_favoring_context",
        }:
            should_activate = _has_context(rule_events, {"external_coda_consonant", "internal_coda"}) and _variant_ratio(
                rule_events,
                {"palatal"},
            ) >= 0.50
        elif condition in {
            "low_palatalization_external_coda_pause",
        }:
            pause_events = [
                event
                for event in rule_events
                if event.get("context") == "external_coda_pause"
            ]
            should_activate = (
                len(pause_events) >= _safe_int(requirements.get("min_events"), 0)
                and _variant_ratio(pause_events, {"palatal"}) <= 0.30
            )
        elif condition in {
            "palatalization_after_palatal_fricative",
        }:
            should_activate = _has_context(rule_events, {"palatal_fricative_preceding"}) and _variant_ratio(
                rule_events,
                {"palatalized", "affricated"},
            ) >= 0.50

        if should_activate:
            activated = _activate_rule(rule)
            activated_rules.append(activated)
            if activated["explanation"]:
                explanations.append(activated["explanation"])

    adjustment = sum(_safe_float(rule.get("weight")) for rule in activated_rules)
    if coverage_score < normal_coverage_cutoff:
        adjustment *= 0.50
    adjustment = cap_adjustment_by_evidence(adjustment, event_count, evidence_quality["quality_score"])

    adjustment = max(
        _safe_float(thresholds.get("max_negative_adjustment"), -0.20),
        min(_safe_float(thresholds.get("max_positive_adjustment"), 0.20), adjustment),
    )

    return {
        "available": True,
        "status": "applied" if activated_rules else "no_rule_triggered",
        "knowledge_version": knowledge.get("version"),
        "knowledge_path": knowledge.get("path"),
        "event_count": event_count,
        "coverage_score": coverage_score,
        "evidence_quality": evidence_quality,
        "phonetic_consistency_score": consistency_score,
        "linguistic_plausibility_score": plausibility_score,
        "raw_localized_disruption_score": raw_disruption_score,
        "localized_disruption_score": disruption_score,
        "localized_disruption": disruption_details,
        "alignment_score": alignment_score,
        "adjustment": adjustment,
        "activated_rules": activated_rules,
        "explanations": explanations,
    }
