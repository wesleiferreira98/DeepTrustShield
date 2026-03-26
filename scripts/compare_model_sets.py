#!/usr/bin/env python3
import argparse
import csv
import hashlib
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
    "real": 0,
    "genuine": 0,
    "authentic": 0,
    "fake": 1,
    "deepfake": 1,
    "manipulated": 1,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compara varios modelos em blocos de imagens organizados por pasta. "
            "Use --subset nome=caminho[:label], onde label pode ser real ou fake."
        )
    )
    parser.add_argument(
        "--subset",
        dest="subsets",
        action="append",
        required=True,
        help="Formato: nome=caminho[:label]. Ex.: black_real=data/raw/black_real:real",
    )
    parser.add_argument(
        "--target-model",
        dest="target_models",
        action="append",
        required=True,
        help="Modelo alvo. Pode repetir a flag para comparar varios modelos.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold manual. Se omitido, usa o threshold do checkpoint carregado.",
    )
    parser.add_argument(
        "--max-per-subset",
        type=int,
        default=None,
        help="Limita quantas imagens usar por subset para comparacao rapida.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Ignora arquivos duplicados dentro de cada subset com base no hash do conteudo.",
    )
    parser.add_argument("--output-json", help="Caminho do resumo JSON")
    parser.add_argument("--output-csv", help="Caminho do CSV detalhado por imagem")
    return parser.parse_args()


def parse_subset(raw_value: str):
    if "=" not in raw_value:
      raise ValueError(f"subset invalido: {raw_value!r}. Use nome=caminho[:label]")

    name, raw_path = raw_value.split("=", 1)
    label = None
    path_text = raw_path
    if ":" in raw_path:
        path_text, raw_label = raw_path.rsplit(":", 1)
        raw_label = raw_label.strip().lower()
        if raw_label:
            if raw_label not in VALID_LABELS:
                raise ValueError(f"label invalido em subset {raw_value!r}: {raw_label!r}")
            label = VALID_LABELS[raw_label]

    subset = {
        "name": name.strip(),
        "path": Path(path_text).expanduser().resolve(),
        "label": label,
    }
    if not subset["name"]:
        raise ValueError(f"nome invalido em subset {raw_value!r}")
    return subset


def iter_images(root: Path, max_items: int | None = None):
    if not root.exists():
        raise FileNotFoundError(f"pasta nao encontrada: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"nao e diretorio: {root}")

    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
            count += 1
            if max_items is not None and count >= max_items:
                break


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda face: face.get("confidence", 0.0))


def safe_mean(values):
    return round(statistics.fmean(values), 6) if values else None


def safe_median(values):
    return round(statistics.median(values), 6) if values else None


def compute_subset_metrics(rows, threshold, expected_label):
    inferred_rows = [row for row in rows if row["status"] == "ok"]
    scores = [row["p_fake"] for row in inferred_rows]
    positives_05 = sum(1 for row in inferred_rows if row["p_fake"] >= 0.5)
    positives_07 = sum(1 for row in inferred_rows if row["p_fake"] >= 0.7)
    positives_threshold = sum(1 for row in inferred_rows if row["predicted_label"] == 1)

    metrics = {
        "threshold": threshold,
        "samples_total": len(rows),
        "samples_inferred": len(inferred_rows),
        "samples_skipped": len(rows) - len(inferred_rows),
        "no_face": sum(1 for row in rows if row["status"] == "no_face"),
        "multiple_faces": sum(1 for row in rows if row["status"] == "multiple_faces"),
        "errors": sum(1 for row in rows if row["status"] == "error"),
        "mean_p_fake": safe_mean(scores),
        "median_p_fake": safe_median(scores),
        "min_p_fake": round(min(scores), 6) if scores else None,
        "max_p_fake": round(max(scores), 6) if scores else None,
        "over_0_5": positives_05,
        "over_0_7": positives_07,
        "predicted_fake": positives_threshold,
        "predicted_real": len(inferred_rows) - positives_threshold,
    }

    if expected_label == 0:
        metrics["false_positive_rate"] = round(positives_threshold / len(inferred_rows), 6) if inferred_rows else None
    elif expected_label == 1:
        metrics["true_positive_rate"] = round(positives_threshold / len(inferred_rows), 6) if inferred_rows else None

    return metrics


def process_subsets(subsets, target_models, threshold_override, max_per_subset, dedupe):
    try:
        from worker.app.config import config
        from worker.app.pipeline.face_detection import detect_faces
        from worker.app.pipeline.inference import get_threshold, run_inference
        from worker.app.pipeline.preprocessing import preprocess_face
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dependencia ausente para comparacao. Instale o ambiente do worker "
            "(mediapipe, opencv-python-headless, torch, torchvision, timm)."
        ) from exc

    details = {model_name: [] for model_name in target_models}
    thresholds = {}

    with tempfile.TemporaryDirectory(prefix="deeptrustshield-compare-") as temp_artifacts:
        sample_index = 0
        for subset in subsets:
            seen_hashes = set()
            for image_path in iter_images(subset["path"], max_per_subset):
                if dedupe:
                    file_hash = file_sha256(image_path)
                    if file_hash in seen_hashes:
                        continue
                    seen_hashes.add(file_hash)
                sample_index += 1
                base_detail = {
                    "subset": subset["name"],
                    "expected_label": subset["label"],
                    "image_path": str(image_path),
                    "status": "pending",
                    "face_count": 0,
                    "error": None,
                }

                try:
                    faces = detect_faces(str(image_path))
                    base_detail["face_count"] = len(faces)
                    if not faces:
                        for model_name in target_models:
                            details[model_name].append(
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
                            details[model_name].append(
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
                        f"compare-{sample_index}",
                        temp_artifacts,
                    )

                    for model_name in target_models:
                        p_fake = float(run_inference(face_tensor, config.MODEL_PATH, target_model=model_name))
                        model_threshold = threshold_override if threshold_override is not None else get_threshold()
                        thresholds[model_name] = model_threshold
                        details[model_name].append(
                            {
                                **base_detail,
                                "status": "ok",
                                "p_fake": round(p_fake, 6),
                                "predicted_label": 1 if p_fake >= model_threshold else 0,
                            }
                        )
                except Exception as exc:
                    for model_name in target_models:
                        details[model_name].append(
                            {
                                **base_detail,
                                "status": "error",
                                "p_fake": None,
                                "predicted_label": None,
                                "error": str(exc),
                            }
                        )

    return details, thresholds


def build_report(subsets, model_name, rows, threshold):
    by_subset = defaultdict(list)
    for row in rows:
        by_subset[row["subset"]].append(row)

    subset_labels = {subset["name"]: subset["label"] for subset in subsets}
    subset_metrics = {
        subset_name: compute_subset_metrics(subset_rows, threshold, subset_labels.get(subset_name))
        for subset_name, subset_rows in sorted(by_subset.items())
    }

    return {
        "target_model": model_name,
        "threshold": threshold,
        "subset_metrics": subset_metrics,
    }


def print_report(report):
    print(f"\n=== {report['target_model']} ===")
    for subset_name, metrics in report["subset_metrics"].items():
        print(f"[{subset_name}]")
        print(
            "  total={samples_total} inferidas={samples_inferred} mean={mean_p_fake} "
            "median={median_p_fake} over_0_5={over_0_5} over_0_7={over_0_7}".format(**metrics)
        )
        if "false_positive_rate" in metrics:
            print(f"  false_positive_rate={metrics['false_positive_rate']}")
        if "true_positive_rate" in metrics:
            print(f"  true_positive_rate={metrics['true_positive_rate']}")


def write_json(path: Path, reports):
    payload = {"reports": reports}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows_by_model):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_model",
        "subset",
        "expected_label",
        "image_path",
        "status",
        "face_count",
        "p_fake",
        "predicted_label",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, rows in rows_by_model.items():
            for row in rows:
                writer.writerow({"target_model": model_name, **row})


def main():
    args = parse_args()
    subsets = [parse_subset(raw_value) for raw_value in args.subsets]
    details, thresholds = process_subsets(
        subsets,
        args.target_models,
        args.threshold,
        args.max_per_subset,
        args.dedupe,
    )

    reports = []
    for model_name in args.target_models:
        report = build_report(subsets, model_name, details[model_name], thresholds.get(model_name))
        reports.append(report)
        print_report(report)

    if args.output_json:
        write_json(Path(args.output_json), reports)
    if args.output_csv:
        write_csv(Path(args.output_csv), details)


if __name__ == "__main__":
    main()
