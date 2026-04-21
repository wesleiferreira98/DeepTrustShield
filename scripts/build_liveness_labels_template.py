#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "calibration" / "liveness_captures"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "reports" / "liveness_capture_summary" / "labels_template.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um CSV-base para rotular capturas de liveness manualmente."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT),
        help="Raiz das capturas de liveness. Padrao: data/calibration/liveness_captures",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT),
        help="CSV de saida. Padrao: data/reports/liveness_capture_summary/labels_template.csv",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def iter_metadata_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/metadata.json"))


def load_existing_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    existing: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            capture_id = (row.get("capture_id") or "").strip()
            if not capture_id:
                continue
            existing[capture_id] = {
                "label": (row.get("label") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
    return existing


def load_capture_row(metadata_path: Path, existing_labels: dict[str, dict[str, str]]) -> dict | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    capture_id = payload.get("capture_id")
    if not capture_id:
        return None

    result = payload.get("result", {})
    summary = result.get("summary", {})
    spoof_analysis = result.get("spoof_analysis", {})
    model = spoof_analysis.get("model", {})
    heuristic = spoof_analysis.get("heuristic", {})
    signals = heuristic.get("signals", {})
    client_context = payload.get("client_context", {})
    camera = client_context.get("camera") or {}
    existing = existing_labels.get(capture_id, {})

    return {
        "capture_id": capture_id,
        "label": existing.get("label", ""),
        "notes": existing.get("notes", ""),
        "created_at": payload.get("created_at") or "",
        "passed": result.get("passed"),
        "device_class": (client_context.get("device_class") or "unknown"),
        "capture_source": (client_context.get("capture_source_guess") or "unknown"),
        "stream_width": camera.get("width"),
        "stream_height": camera.get("height"),
        "valid_frames": summary.get("valid_frames"),
        "total_frames": summary.get("total_frames"),
        "model_p_spoof": safe_float(model.get("p_spoof")),
        "screen_rectangle": safe_float(signals.get("median_screen_rectangle_score")),
        "local_motion": safe_float(signals.get("compensated_local_motion")),
        "face_background": safe_float(signals.get("median_face_background_delta")),
        "fft_grid": safe_float(signals.get("median_frequency_grid_score")),
        "skin_peak": safe_float(signals.get("median_skin_frequency_peak_score")),
        "skin_lbp_uniformity": safe_float(signals.get("median_skin_lbp_uniformity")),
        "reasons": "|".join(spoof_analysis.get("reasons", [])),
        "metadata_path": str(metadata_path),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_csv = resolve_path(args.output_csv)

    if not input_root.exists():
        raise FileNotFoundError(f"Raiz das capturas nao encontrada: {input_root}")

    existing_labels = load_existing_labels(output_csv)
    rows: list[dict] = []
    for metadata_path in iter_metadata_files(input_root):
        row = load_capture_row(metadata_path, existing_labels)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("Nenhuma captura encontrada para montar o template.")

    rows.sort(key=lambda item: (item.get("created_at") or "", item["capture_id"]), reverse=True)

    fieldnames = [
        "capture_id",
        "label",
        "notes",
        "created_at",
        "passed",
        "device_class",
        "capture_source",
        "stream_width",
        "stream_height",
        "valid_frames",
        "total_frames",
        "model_p_spoof",
        "screen_rectangle",
        "local_motion",
        "face_background",
        "fft_grid",
        "skin_peak",
        "skin_lbp_uniformity",
        "reasons",
        "metadata_path",
    ]
    write_csv(output_csv, rows, fieldnames)

    labeled = sum(1 for row in rows if row["label"])
    print(f"capturas: {len(rows)} | rotulos preservados: {labeled} | em branco: {len(rows) - labeled}")
    print(f"labels_template: {output_csv}")


if __name__ == "__main__":
    main()
