#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = REPO_ROOT / "worker"
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

from app.config import config, get_audio_model_config  # noqa: E402
from app.pipeline.audio_inference import run_audio_inference  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "audio" / "reports" / "audio_model_domain_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia checkpoints de áudio em um ou mais manifestos e gera um "
            "comparativo por domínio, fonte e modelo."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Manifesto CSV com coluna audio_path. Pode ser repetido.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fake_braccent_codec", "brspeech_sample"],
        help="Chaves do AUDIO_MODEL_REGISTRY a avaliar.",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Split(s) para avaliar. Use 'all' ou lista separada por vírgula, como test,val.",
    )
    parser.add_argument(
        "--limit-per-manifest-label",
        type=int,
        default=0,
        help="Limite opcional por manifesto e label. 0 avalia tudo.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--predictions-name", default="predictions.csv")
    parser.add_argument("--summary-name", default="summary.json")
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_label(value: str | None) -> tuple[str, int | None]:
    normalized = (value or "").strip().lower()
    if normalized in {"real", "bonafide", "genuine", "0"}:
        return "real", 0
    if normalized in {"fake", "spoof", "synthetic", "1"}:
        return "fake", 1
    return normalized or "unknown", None


def split_is_selected(row_split: str, selected_splits: set[str] | None) -> bool:
    if selected_splits is None:
        return True
    return (row_split or "").strip().lower() in selected_splits


def load_manifest(manifest_path: Path, selected_splits: set[str] | None) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "audio_path" not in reader.fieldnames:
            raise ValueError(f"Manifesto sem coluna audio_path: {manifest_path}")

        for index, row in enumerate(reader):
            row_split = (row.get("split") or "").strip()
            if not split_is_selected(row_split, selected_splits):
                continue

            audio_path = Path(row["audio_path"]).expanduser()
            if not audio_path.is_absolute():
                audio_path = (manifest_path.parent / audio_path).resolve()
            if not audio_path.exists():
                continue

            label_text, label_id = normalize_label(row.get("label"))
            source = row.get("source") or manifest_path.stem
            domain = row.get("domain") or row.get("dataset") or source
            rows.append(
                {
                    "row_index": index,
                    "audio_path": str(audio_path),
                    "manifest": str(manifest_path),
                    "manifest_name": manifest_path.stem,
                    "split": row_split or "unknown",
                    "label": label_text,
                    "label_id": label_id,
                    "source": source,
                    "domain": domain,
                    "accent": row.get("accent") or "unknown",
                    "speaker_id": row.get("speaker_id") or "unknown",
                    "manipulation_type": row.get("manipulation_type") or "none",
                }
            )
    return rows


def apply_limit(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0:
        return rows

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["manifest_name"], row["label"])].append(row)

    rng = random.Random(seed)
    limited: list[dict] = []
    for key in sorted(grouped):
        items = grouped[key]
        rng.shuffle(items)
        limited.extend(items[:limit])
    return sorted(limited, key=lambda item: (item["manifest_name"], item["row_index"]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    if not labels or len(set(labels)) < 2:
        return {
            "count": len(labels),
            "accuracy": None,
            "f1": None,
            "auc": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
        }

    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 1)
    tn = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 0)
    fp = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels)

    positives = [score for y, score in zip(labels, scores) if y == 1]
    negatives = [score for y, score in zip(labels, scores) if y == 0]
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    auc = wins / total if total else None

    return {
        "count": len(labels),
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def summarize_group(rows: list[dict], score_field: str = "p_fake") -> dict:
    labels = [row["label_id"] for row in rows if row.get("label_id") in {0, 1}]
    scores = [float(row[score_field]) for row in rows if row.get("label_id") in {0, 1}]
    threshold = mean([float(row["threshold"]) for row in rows])
    score_values = [float(row[score_field]) for row in rows]
    fake_ratios = [float(row["segment_fake_ratio"]) for row in rows]
    return {
        "count": len(rows),
        "label_counts": dict(sorted((label, sum(1 for row in rows if row["label"] == label)) for label in {row["label"] for row in rows})),
        "mean_score": mean(score_values),
        "p90_score": percentile(score_values, 0.90),
        "mean_segment_fake_ratio": mean(fake_ratios),
        "threshold_mean": threshold,
        "metrics": binary_metrics(labels, scores, threshold),
    }


def evaluate_one(model_key: str, row: dict) -> dict:
    model_config = get_audio_model_config(model_key)
    result = run_audio_inference(
        row["audio_path"],
        model_config["path"],
        model_config.get("summary_path"),
    )
    threshold = float(result.get("threshold") or 0.5)
    segments = result.get("segments") or []
    segment_scores = [float(segment.get("p_fake") or 0.0) for segment in segments]
    segment_fake_count = sum(1 for score in segment_scores if score >= threshold)
    top3_scores = sorted(segment_scores, reverse=True)[:3]
    return {
        **row,
        "model_key": model_config["key"],
        "model_label": model_config["label"],
        "model_version": result.get("model_version") or "",
        "threshold": threshold,
        "p_fake": float(result.get("p_fake") or 0.0),
        "neural_decision": "fake" if float(result.get("p_fake") or 0.0) >= threshold else "real",
        "duration_seconds": result.get("audio_info", {}).get("duration_seconds"),
        "source_sample_rate": result.get("audio_info", {}).get("source_sample_rate"),
        "segment_count": len(segments),
        "segment_mean": mean(segment_scores),
        "segment_max": max(segment_scores) if segment_scores else 0.0,
        "segment_p90": percentile(segment_scores, 0.90),
        "segment_top3_mean": mean(top3_scores),
        "segment_fake_count": segment_fake_count,
        "segment_fake_ratio": segment_fake_count / len(segment_scores) if segment_scores else 0.0,
    }


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_key",
        "model_label",
        "model_version",
        "manifest_name",
        "domain",
        "source",
        "split",
        "label",
        "label_id",
        "accent",
        "speaker_id",
        "manipulation_type",
        "audio_path",
        "threshold",
        "p_fake",
        "neural_decision",
        "duration_seconds",
        "source_sample_rate",
        "segment_count",
        "segment_mean",
        "segment_max",
        "segment_p90",
        "segment_top3_mean",
        "segment_fake_count",
        "segment_fake_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictions(rows: list[dict], args: argparse.Namespace, elapsed_seconds: float) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    by_model_domain: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_model_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model_key"]].append(row)
        by_model_domain[(row["model_key"], row["domain"])].append(row)
        by_model_source[(row["model_key"], row["source"])].append(row)

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed_seconds,
        "models": args.models,
        "manifests": args.manifest,
        "split": args.split,
        "row_count": len(rows),
        "registered_audio_models": {
            key: {
                "label": value.get("label"),
                "path": value.get("path"),
                "summary_path": value.get("summary_path"),
            }
            for key, value in sorted(config.AUDIO_MODEL_REGISTRY.items())
        },
        "by_model": {key: summarize_group(items) for key, items in sorted(by_model.items())},
        "by_model_domain": {
            f"{model_key}::{domain}": summarize_group(items)
            for (model_key, domain), items in sorted(by_model_domain.items())
        },
        "by_model_source": {
            f"{model_key}::{source}": summarize_group(items)
            for (model_key, source), items in sorted(by_model_source.items())
        },
    }


def main() -> None:
    args = parse_args()
    selected_splits = None
    if args.split.strip().lower() != "all":
        selected_splits = {item.strip().lower() for item in args.split.split(",") if item.strip()}

    manifest_paths = [resolve_repo_path(path) for path in args.manifest]
    rows: list[dict] = []
    for manifest_path in manifest_paths:
        rows.extend(load_manifest(manifest_path, selected_splits))
    rows = apply_limit(rows, args.limit_per_manifest_label, args.seed)
    if not rows:
        raise RuntimeError("Nenhum áudio válido encontrado para avaliação.")

    output_dir = resolve_repo_path(args.output_dir)
    predictions: list[dict] = []
    started = time.time()
    total = len(args.models) * len(rows)
    current = 0
    for model_key in args.models:
        for row in rows:
            current += 1
            print(
                f"[{current}/{total}] modelo={model_key} label={row['label']} "
                f"fonte={row['source']} audio={Path(row['audio_path']).name}",
                flush=True,
            )
            try:
                predictions.append(evaluate_one(model_key, row))
            except Exception as exc:
                predictions.append(
                    {
                        **row,
                        "model_key": model_key,
                        "model_label": "",
                        "model_version": "",
                        "threshold": "",
                        "p_fake": "",
                        "neural_decision": "error",
                        "error": str(exc),
                        "segment_fake_ratio": 0.0,
                    }
                )

    elapsed = time.time() - started
    predictions_path = output_dir / args.predictions_name
    summary_path = output_dir / args.summary_name
    write_predictions(predictions_path, predictions)
    summary = summarize_predictions(
        [row for row in predictions if row.get("neural_decision") != "error"],
        args,
        elapsed,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"predictions = {predictions_path}")
    print(f"summary = {summary_path}")


if __name__ == "__main__":
    main()
