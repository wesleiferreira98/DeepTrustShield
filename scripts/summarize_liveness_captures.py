#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "calibration" / "liveness_captures"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "reports" / "liveness_capture_summary"

METRIC_FIELDS = [
    "model_p_spoof",
    "model_threshold",
    "valid_frames",
    "total_frames",
    "screen_rectangle",
    "internal_motion",
    "local_motion",
    "face_background",
    "moire_fft",
    "freq_peak",
    "isolated_peak",
    "fft_grid",
    "fft_direction",
    "fft_ring",
    "fft_energy",
    "skin_moire",
    "skin_peak",
    "lbp_entropy",
    "lbp_uniformity",
    "lbp_contrast",
    "skin_lbp_entropy",
    "skin_lbp_uniformity",
    "skin_lbp_contrast",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera tabela por captura e resumo por rotulo para as capturas de liveness salvas."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_ROOT),
        help="Raiz das capturas de liveness. Padrao: data/calibration/liveness_captures",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="CSV opcional com colunas capture_id,label para sobrescrever/complementar review_label.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Pasta de saida. Padrao: data/reports/liveness_capture_summary",
    )
    return parser.parse_args()


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_label_overrides(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}

    overrides: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            capture_id = (row.get("capture_id") or "").strip()
            label = (row.get("label") or "").strip()
            if capture_id and label:
                overrides[capture_id] = label
    return overrides


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def load_capture_row(metadata_path: Path, label_overrides: dict[str, str]) -> dict | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    result = payload.get("result", {})
    summary = result.get("summary", {})
    spoof_analysis = result.get("spoof_analysis", {})
    model = spoof_analysis.get("model", {})
    heuristic = spoof_analysis.get("heuristic", {})
    signals = heuristic.get("signals", {})
    client_context = payload.get("client_context", {})

    capture_id = payload.get("capture_id")
    if not capture_id:
        return None

    label = (
        label_overrides.get(capture_id)
        or payload.get("review_label")
        or "unlabeled"
    )

    row = {
        "capture_id": capture_id,
        "created_at": payload.get("created_at"),
        "label": label,
        "label_source": (
            "labels_csv"
            if capture_id in label_overrides
            else (payload.get("label_source") or "unlabeled")
        ),
        "passed": result.get("passed"),
        "device_class": (client_context.get("device_class") or "unknown"),
        "capture_source": (client_context.get("capture_source_guess") or "unknown"),
        "stream_width": ((client_context.get("camera") or {}).get("width")),
        "stream_height": ((client_context.get("camera") or {}).get("height")),
        "model_p_spoof": safe_float(model.get("p_spoof")),
        "model_threshold": safe_float(model.get("threshold")),
        "valid_frames": summary.get("valid_frames"),
        "total_frames": summary.get("total_frames"),
        "screen_rectangle": safe_float(signals.get("median_screen_rectangle_score")),
        "internal_motion": safe_float(signals.get("internal_motion")),
        "local_motion": safe_float(signals.get("compensated_local_motion")),
        "face_background": safe_float(signals.get("median_face_background_delta")),
        "moire_fft": safe_float(signals.get("median_moire_score")),
        "freq_peak": safe_float(signals.get("median_frequency_peak_score")),
        "isolated_peak": safe_float(signals.get("median_isolated_peak_score")),
        "fft_grid": safe_float(signals.get("median_frequency_grid_score")),
        "fft_direction": safe_float(signals.get("median_frequency_directional_score")),
        "fft_ring": safe_float(signals.get("median_frequency_ring_contrast")),
        "fft_energy": safe_float(signals.get("median_high_frequency_energy")),
        "skin_moire": safe_float(signals.get("median_skin_moire_score")),
        "skin_peak": safe_float(signals.get("median_skin_frequency_peak_score")),
        "lbp_entropy": safe_float(signals.get("median_lbp_entropy")),
        "lbp_uniformity": safe_float(signals.get("median_lbp_uniformity")),
        "lbp_contrast": safe_float(signals.get("median_lbp_local_contrast")),
        "skin_lbp_entropy": safe_float(signals.get("median_skin_lbp_entropy")),
        "skin_lbp_uniformity": safe_float(signals.get("median_skin_lbp_uniformity")),
        "skin_lbp_contrast": safe_float(signals.get("median_skin_lbp_local_contrast")),
        "reasons": "|".join(spoof_analysis.get("reasons", [])),
        "metadata_path": str(metadata_path),
    }
    return row


def iter_metadata_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/metadata.json"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def coarse_label(label: str) -> str:
    normalized = (label or "").strip().lower()
    if normalized.startswith("real"):
        return "real"
    if normalized.startswith("fake"):
        return "fake"
    return "other"


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["label"], row["device_class"], row["capture_source"])
        grouped[key].append(row)

    summary_rows: list[dict] = []
    for (label, device_class, capture_source), items in sorted(grouped.items()):
        entry = {
            "label": label,
            "device_class": device_class,
            "capture_source": capture_source,
            "captures": len(items),
            "pass_rate": round(sum(1 for item in items if item.get("passed")) / max(len(items), 1), 4),
        }
        for field in METRIC_FIELDS:
            values = [item[field] for item in items if isinstance(item.get(field), (int, float))]
            entry[f"{field}_mean"] = round(sum(values) / len(values), 6) if values else None
            entry[f"{field}_median"] = round(float(median(values)), 6) if values else None
            entry[f"{field}_min"] = round(min(values), 6) if values else None
            entry[f"{field}_max"] = round(max(values), 6) if values else None
        summary_rows.append(entry)
    return summary_rows


def summarize_by_coarse_label(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        family = coarse_label(row["label"])
        if family == "other":
            continue
        key = (family, row["device_class"], row["capture_source"])
        grouped[key].append(row)

    summary_rows: list[dict] = []
    for (family, device_class, capture_source), items in sorted(grouped.items()):
        entry = {
            "family": family,
            "device_class": device_class,
            "capture_source": capture_source,
            "captures": len(items),
            "pass_rate": round(sum(1 for item in items if item.get("passed")) / max(len(items), 1), 4),
        }
        for field in METRIC_FIELDS:
            values = [item[field] for item in items if isinstance(item.get(field), (int, float))]
            entry[f"{field}_mean"] = round(sum(values) / len(values), 6) if values else None
            entry[f"{field}_median"] = round(float(median(values)), 6) if values else None
            entry[f"{field}_min"] = round(min(values), 6) if values else None
            entry[f"{field}_max"] = round(max(values), 6) if values else None
        summary_rows.append(entry)
    return summary_rows


def build_metric_separation(rows: list[dict]) -> list[dict]:
    labeled_rows = [row for row in rows if coarse_label(row["label"]) in {"real", "fake"}]
    real_rows = [row for row in labeled_rows if coarse_label(row["label"]) == "real"]
    fake_rows = [row for row in labeled_rows if coarse_label(row["label"]) == "fake"]

    if not real_rows or not fake_rows:
        return []

    separation_rows: list[dict] = []
    for field in METRIC_FIELDS:
        real_values = [row[field] for row in real_rows if isinstance(row.get(field), (int, float))]
        fake_values = [row[field] for row in fake_rows if isinstance(row.get(field), (int, float))]
        if not real_values or not fake_values:
            continue

        real_mean = sum(real_values) / len(real_values)
        fake_mean = sum(fake_values) / len(fake_values)
        real_median = float(median(real_values))
        fake_median = float(median(fake_values))
        mean_gap = abs(fake_mean - real_mean)
        median_gap = abs(fake_median - real_median)
        direction = "higher_in_fake" if fake_median > real_median else "higher_in_real"

        separation_rows.append(
            {
                "metric": field,
                "real_count": len(real_values),
                "fake_count": len(fake_values),
                "real_mean": round(real_mean, 6),
                "fake_mean": round(fake_mean, 6),
                "real_median": round(real_median, 6),
                "fake_median": round(fake_median, 6),
                "mean_gap_abs": round(mean_gap, 6),
                "median_gap_abs": round(median_gap, 6),
                "direction": direction,
            }
        )

    separation_rows.sort(
        key=lambda item: (item["median_gap_abs"], item["mean_gap_abs"]),
        reverse=True,
    )
    return separation_rows


def build_markdown_table(rows: list[dict]) -> str:
    headers = [
        "capture_id",
        "label",
        "device_class",
        "capture_source",
        "model_p_spoof",
        "screen_rectangle",
        "local_motion",
        "face_background",
        "fft_grid",
        "skin_peak",
        "skin_lbp_uniformity",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        )
    return "\n".join(lines) + "\n"


def build_separation_markdown(separation_rows: list[dict], labeled_rows: list[dict]) -> str:
    real_count = sum(1 for row in labeled_rows if coarse_label(row["label"]) == "real")
    fake_count = sum(1 for row in labeled_rows if coarse_label(row["label"]) == "fake")

    lines = [
        "# Separacao de metricas: real vs fake",
        "",
        f"- Capturas `real_*`: {real_count}",
        f"- Capturas `fake_*`: {fake_count}",
        "",
    ]

    if not separation_rows:
        lines.extend(
            [
                "Ainda nao ha rotulos suficientes para comparar `real` vs `fake`.",
                "",
                "Preencha o arquivo `labels_template.csv` com rotulos como:",
                "",
                "- `real_webcam`",
                "- `real_phone`",
                "- `fake_far_webcam`",
                "- `fake_near_webcam`",
                "- `fake_far_phone`",
                "- `fake_near_phone`",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    headers = [
        "metric",
        "real_median",
        "fake_median",
        "median_gap_abs",
        "direction",
    ]
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    for row in separation_rows[:12]:
        lines.append(
            "| "
            + " | ".join(str(row.get(header, "")) for header in headers)
            + " |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_dir = resolve_path(args.output_dir)
    labels_csv = resolve_path(args.labels_csv)

    if input_root is None or not input_root.exists():
        raise FileNotFoundError(f"Raiz das capturas nao encontrada: {input_root}")
    if output_dir is None:
        raise RuntimeError("Pasta de saida invalida")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_overrides = load_label_overrides(labels_csv)
    rows = []
    for metadata_path in iter_metadata_files(input_root):
        row = load_capture_row(metadata_path, label_overrides)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("Nenhuma captura encontrada para resumir.")

    per_capture_path = output_dir / "per_capture.csv"
    summary_path = output_dir / "summary_by_label.csv"
    family_summary_path = output_dir / "summary_by_family.csv"
    separation_path = output_dir / "metric_separation_real_vs_fake.csv"
    markdown_path = output_dir / "recent_captures.md"
    separation_markdown_path = output_dir / "metric_separation_real_vs_fake.md"

    per_capture_fields = [
        "capture_id",
        "created_at",
        "label",
        "label_source",
        "passed",
        "device_class",
        "capture_source",
        "stream_width",
        "stream_height",
        *METRIC_FIELDS,
        "reasons",
        "metadata_path",
    ]
    write_csv(per_capture_path, rows, per_capture_fields)

    summary_rows = summarize_rows(rows)
    summary_fields = [
        "label",
        "device_class",
        "capture_source",
        "captures",
        "pass_rate",
    ]
    for metric in METRIC_FIELDS:
        summary_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    write_csv(summary_path, summary_rows, summary_fields)

    family_summary_rows = summarize_by_coarse_label(rows)
    family_summary_fields = [
        "family",
        "device_class",
        "capture_source",
        "captures",
        "pass_rate",
    ]
    for metric in METRIC_FIELDS:
        family_summary_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    write_csv(family_summary_path, family_summary_rows, family_summary_fields)

    separation_rows = build_metric_separation(rows)
    separation_fields = [
        "metric",
        "real_count",
        "fake_count",
        "real_mean",
        "fake_mean",
        "real_median",
        "fake_median",
        "mean_gap_abs",
        "median_gap_abs",
        "direction",
    ]
    write_csv(separation_path, separation_rows, separation_fields)

    recent_rows = sorted(rows, key=lambda item: item.get("created_at") or "", reverse=True)[:12]
    markdown_path.write_text(build_markdown_table(recent_rows), encoding="utf-8")
    labeled_rows = [row for row in rows if coarse_label(row["label"]) in {"real", "fake"}]
    separation_markdown_path.write_text(
        build_separation_markdown(separation_rows, labeled_rows),
        encoding="utf-8",
    )

    labels_found = sum(1 for row in rows if row["label"] != "unlabeled")
    print(f"capturas: {len(rows)} | rotuladas: {labels_found} | nao rotuladas: {len(rows) - labels_found}")
    print(f"per_capture: {per_capture_path}")
    print(f"summary_by_label: {summary_path}")
    print(f"summary_by_family: {family_summary_path}")
    print(f"metric_separation: {separation_path}")
    print(f"recent_md: {markdown_path}")
    print(f"metric_separation_md: {separation_markdown_path}")


if __name__ == "__main__":
    main()
