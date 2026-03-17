#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VALID_LABELS = {
    "0": 0,
    "real": 0,
    "genuine": 0,
    "authentic": 0,
    "1": 1,
    "fake": 1,
    "deepfake": 1,
    "manipulated": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audita vies do modelo usando o mesmo pipeline do worker. "
            "O CSV precisa ter ao menos: image_path,label,group."
        )
    )
    parser.add_argument("--manifest", required=True, help="CSV com image_path,label,group")
    parser.add_argument(
        "--target-model",
        dest="target_models",
        action="append",
        help="Modelo alvo. Pode repetir a flag para comparar varios modelos.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold manual. Se omitido, usa o threshold calibrado de cada modelo.",
    )
    parser.add_argument("--output-json", help="Caminho para salvar o resumo JSON")
    parser.add_argument("--output-csv", help="Prefixo para salvar detalhes por amostra")
    return parser.parse_args()


def normalize_label(raw_label):
    key = str(raw_label).strip().lower()
    if key not in VALID_LABELS:
        raise ValueError(f"label invalido: {raw_label!r}")
    return VALID_LABELS[key]


def resolve_image_path(manifest_path: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def choose_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda face: face.get("confidence", 0.0))


def safe_mean(values):
    return round(statistics.fmean(values), 6) if values else None


def compute_metrics(rows, threshold):
    inferred_rows = [row for row in rows if row["status"] == "ok"]
    real_rows = [row for row in inferred_rows if row["label"] == 0]
    fake_rows = [row for row in inferred_rows if row["label"] == 1]

    tp = sum(1 for row in inferred_rows if row["label"] == 1 and row["predicted_label"] == 1)
    tn = sum(1 for row in inferred_rows if row["label"] == 0 and row["predicted_label"] == 0)
    fp = sum(1 for row in inferred_rows if row["label"] == 0 and row["predicted_label"] == 1)
    fn = sum(1 for row in inferred_rows if row["label"] == 1 and row["predicted_label"] == 0)

    real_scores = [row["p_fake"] for row in real_rows]
    fake_scores = [row["p_fake"] for row in fake_rows]

    return {
        "threshold": threshold,
        "samples_total": len(rows),
        "samples_inferred": len(inferred_rows),
        "samples_skipped": len(rows) - len(inferred_rows),
        "no_face": sum(1 for row in rows if row["status"] == "no_face"),
        "multiple_faces": sum(1 for row in rows if row["status"] == "multiple_faces"),
        "errors": sum(1 for row in rows if row["status"] == "error"),
        "real_count": len(real_rows),
        "fake_count": len(fake_rows),
        "mean_p_fake_real": safe_mean(real_scores),
        "mean_p_fake_fake": safe_mean(fake_scores),
        "accuracy": round((tp + tn) / len(inferred_rows), 6) if inferred_rows else None,
        "fpr": round(fp / len(real_rows), 6) if real_rows else None,
        "fnr": round(fn / len(fake_rows), 6) if fake_rows else None,
        "tpr": round(tp / len(fake_rows), 6) if fake_rows else None,
        "tnr": round(tn / len(real_rows), 6) if real_rows else None,
        "positive_rate": round((tp + fp) / len(inferred_rows), 6) if inferred_rows else None,
    }


def compute_group_gaps(group_metrics):
    comparable_groups = [metrics for metrics in group_metrics.values() if metrics["samples_inferred"] > 0]
    if not comparable_groups:
        return {}

    def gap(metric_name):
        values = [item[metric_name] for item in comparable_groups if item[metric_name] is not None]
        if len(values) < 2:
            return None
        return round(max(values) - min(values), 6)

    return {
        "fpr_gap": gap("fpr"),
        "fnr_gap": gap("fnr"),
        "mean_real_score_gap": gap("mean_p_fake_real"),
        "mean_fake_score_gap": gap("mean_p_fake_fake"),
        "positive_rate_gap": gap("positive_rate"),
    }


def load_manifest_rows(manifest_path: Path):
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"image_path", "label", "group"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV sem colunas obrigatorias: {', '.join(sorted(missing))}")

        for index, raw_row in enumerate(reader, start=1):
            image_path = resolve_image_path(manifest_path, raw_row["image_path"])
            rows.append(
                {
                    "id": raw_row.get("id") or str(index),
                    "image_path": str(image_path),
                    "group": str(raw_row["group"]).strip() or "unknown",
                    "label": normalize_label(raw_row["label"]),
                }
            )
    return rows


def process_samples(manifest_path: Path, target_models, threshold_override):
    try:
        from worker.app.config import config
        from worker.app.pipeline.face_detection import detect_faces
        from worker.app.pipeline.inference import get_threshold, run_inference
        from worker.app.pipeline.preprocessing import preprocess_face
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dependencia ausente para auditoria. Instale o ambiente do worker "
            "(mediapipe, opencv-python-headless, torch, torchvision, timm)."
        ) from exc

    manifest_rows = load_manifest_rows(manifest_path)
    model_details = {model_name: [] for model_name in target_models}
    thresholds = {}

    with tempfile.TemporaryDirectory(prefix="deeptrustshield-bias-audit-") as temp_artifacts:
        for index, base_row in enumerate(manifest_rows, start=1):
            image_path = Path(base_row["image_path"])
            base_detail = {
                "id": base_row["id"],
                "image_path": str(image_path),
                "group": base_row["group"],
                "label": base_row["label"],
                "status": "pending",
                "face_count": 0,
                "error": None,
            }

            try:
                if not image_path.exists():
                    raise FileNotFoundError(f"arquivo nao encontrado: {image_path}")

                faces = detect_faces(str(image_path))
                base_detail["face_count"] = len(faces)
                if not faces:
                    for model_name in target_models:
                        model_details[model_name].append(
                            {
                                **base_detail,
                                "status": "no_face",
                                "p_fake": None,
                                "predicted_label": None,
                            }
                        )
                    continue
                if len(faces) > 1:
                    for model_name in target_models:
                        model_details[model_name].append(
                            {
                                **base_detail,
                                "status": "multiple_faces",
                                "p_fake": None,
                                "predicted_label": None,
                            }
                        )
                    continue

                _, face_tensor = preprocess_face(
                    str(image_path),
                    choose_face(faces),
                    f"audit-{index}",
                    temp_artifacts,
                )

                for model_name in target_models:
                    p_fake = float(run_inference(face_tensor, config.MODEL_PATH, target_model=model_name))
                    model_threshold = threshold_override if threshold_override is not None else get_threshold()
                    thresholds[model_name] = model_threshold
                    model_details[model_name].append(
                        {
                            **base_detail,
                            "status": "ok",
                            "p_fake": round(p_fake, 6),
                            "predicted_label": 1 if p_fake >= model_threshold else 0,
                        }
                    )
            except Exception as exc:
                for model_name in target_models:
                    model_details[model_name].append(
                        {
                            **base_detail,
                            "status": "error",
                            "p_fake": None,
                            "predicted_label": None,
                            "error": str(exc),
                        }
                    )

    return model_details, thresholds


def build_model_report(manifest_path: Path, model_name: str, rows, threshold):
    group_rows = defaultdict(list)
    for row in rows:
        group_rows[row["group"]].append(row)

    group_metrics = {
        group_name: compute_metrics(group_items, threshold)
        for group_name, group_items in sorted(group_rows.items())
    }

    return {
        "manifest": str(manifest_path),
        "target_model": model_name,
        "overall": compute_metrics(rows, threshold),
        "group_metrics": group_metrics,
        "group_gaps": compute_group_gaps(group_metrics),
        "details": rows,
    }


def build_comparison(model_reports):
    comparison_rows = []
    for model_name, report in model_reports.items():
        overall = report["overall"]
        comparison_rows.append(
            {
                "target_model": model_name,
                "accuracy": overall["accuracy"],
                "fpr": overall["fpr"],
                "fnr": overall["fnr"],
                "mean_p_fake_real": overall["mean_p_fake_real"],
                "mean_p_fake_fake": overall["mean_p_fake_fake"],
                "fpr_gap": report["group_gaps"].get("fpr_gap"),
                "fnr_gap": report["group_gaps"].get("fnr_gap"),
                "mean_real_score_gap": report["group_gaps"].get("mean_real_score_gap"),
            }
        )

    def sort_key(item):
        return (
            item["fpr_gap"] is None,
            item["fpr_gap"] if item["fpr_gap"] is not None else 999.0,
            item["fnr_gap"] if item["fnr_gap"] is not None else 999.0,
            -(item["accuracy"] or -1.0),
        )

    return sorted(comparison_rows, key=sort_key)


def audit_manifest(manifest_path: Path, target_models, threshold_override):
    model_details, thresholds = process_samples(manifest_path, target_models, threshold_override)
    model_reports = {
        model_name: build_model_report(
            manifest_path,
            model_name,
            rows,
            thresholds.get(model_name, threshold_override),
        )
        for model_name, rows in model_details.items()
    }
    comparison = build_comparison(model_reports)

    if len(target_models) == 1:
        single_report = next(iter(model_reports.values()))
        single_report["comparison"] = comparison
        single_report["model_reports"] = model_reports
        return single_report

    return {
        "manifest": str(manifest_path),
        "target_models": target_models,
        "model_reports": model_reports,
        "comparison": comparison,
    }


def write_details_csv(details, output_path: Path):
    fieldnames = [
        "id",
        "image_path",
        "group",
        "label",
        "status",
        "face_count",
        "p_fake",
        "predicted_label",
        "error",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in details:
            writer.writerow({name: row.get(name) for name in fieldnames})


def save_details(report, output_csv_prefix: Path):
    if "model_reports" in report and len(report["model_reports"]) > 1:
        for model_name, model_report in report["model_reports"].items():
            write_details_csv(model_report["details"], output_csv_prefix.with_name(f"{output_csv_prefix.name}_{model_name}.csv"))
        return
    if "details" in report:
        write_details_csv(report["details"], output_csv_prefix)
        return
    only_model_report = next(iter(report["model_reports"].values()))
    write_details_csv(only_model_report["details"], output_csv_prefix)


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    target_models = args.target_models or ["auto"]
    report = audit_manifest(manifest_path, target_models, args.threshold)

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.output_csv:
        save_details(report, Path(args.output_csv).resolve())

    if "model_reports" in report and "target_models" in report:
        summary = {
            "manifest": report["manifest"],
            "target_models": report["target_models"],
            "comparison": report["comparison"],
        }
    else:
        summary = {
            "manifest": report["manifest"],
            "target_model": report["target_model"],
            "overall": report["overall"],
            "group_gaps": report["group_gaps"],
            "groups": report["group_metrics"],
            "comparison": report.get("comparison", []),
        }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
