#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "model/weights/antispoof_mobilenet_v1_cv/fold_3/antispoof_mobilenet_v1.onnx"
DEFAULT_METADATA = "model/weights/antispoof_mobilenet_v1_cv/fold_3/antispoof_mobilenet_v1_metadata.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibra o threshold do modelo ONNX anti-spoof em uma pasta com live/ e spoof/."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Pasta com subpastas live/ e spoof/, ou use --live-dir/--spoof-dir.",
    )
    parser.add_argument("--live-dir", help="Pasta com amostras live. Sobrescreve input-root/live.")
    parser.add_argument("--spoof-dir", help="Pasta com amostras spoof. Sobrescreve input-root/spoof.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pasta de saida. Padrao: data/reports/antispoof_calibration/<timestamp>.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["image", "folder"],
        default="folder",
        help=(
            "image = cada arquivo e uma amostra; folder = cada subpasta e uma captura "
            "multi-frame, arquivos soltos continuam individuais."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=["median", "p75", "p90", "max", "mean"],
        default="p75",
        help="Como agregar frames quando uma amostra tem varias imagens.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--target-apcer",
        type=float,
        default=0.02,
        help="APCER maximo desejado para recomendacao conservadora.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.005,
        help="Passo da grade de thresholds entre 0.0 e 1.0.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_metadata(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def aggregate_scores(scores: list[float], aggregation: str) -> float:
    if not scores:
        return float("nan")
    if aggregation == "median":
        return float(np.median(scores))
    if aggregation == "p75":
        return float(np.percentile(scores, 75))
    if aggregation == "p90":
        return float(np.percentile(scores, 90))
    if aggregation == "max":
        return float(np.max(scores))
    if aggregation == "mean":
        return float(np.mean(scores))
    return float(np.percentile(scores, 75))


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def build_samples(label_dir: Path, label: str, sample_mode: str) -> list[dict]:
    images = iter_images(label_dir)
    if sample_mode == "image":
        return [
            {
                "sample_id": str(path.relative_to(label_dir)),
                "label": label,
                "label_id": 0 if label == "live" else 1,
                "paths": [path],
            }
            for path in images
        ]

    grouped: dict[str, list[Path]] = {}
    for path in images:
        rel_parent = path.parent.relative_to(label_dir)
        if str(rel_parent) == ".":
            key = path.name
        else:
            key = str(rel_parent)
        grouped.setdefault(key, []).append(path)

    return [
        {
            "sample_id": sample_id,
            "label": label,
            "label_id": 0 if label == "live" else 1,
            "paths": sorted(paths),
        }
        for sample_id, paths in sorted(grouped.items())
    ]


def preprocess_image(path: Path, image_size: int, cv2) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))


def measure_image_quality(path: Path, cv2) -> dict | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    return {
        "brightness_mean": float(np.mean(value) / 255.0),
        "brightness_p95": float(np.percentile(value, 95) / 255.0),
        "contrast_std": float(np.std(gray) / 255.0),
        "overexposed_ratio": float(np.mean(value > 245)),
        "glare_ratio": float(np.mean((value > 245) & (saturation < 45))),
        **calculate_frequency_features(gray, cv2),
        **calculate_skin_frequency_features(gray, cv2),
    }


def calculate_frequency_features(gray_image: np.ndarray, cv2) -> dict[str, float]:
    if gray_image is None or gray_image.size == 0:
        return {
            "frequency_peak_score": 0.0,
            "high_frequency_energy": 0.0,
            "horizontal_band_energy": 0.0,
            "vertical_band_energy": 0.0,
            "moire_score": 0.0,
        }

    resized = cv2.resize(gray_image, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    window = np.outer(np.hanning(resized.shape[0]), np.hanning(resized.shape[1])).astype(np.float32)
    normalized = (resized - float(np.mean(resized))) * window
    spectrum = np.fft.fftshift(np.fft.fft2(normalized))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)

    size = magnitude.shape[0]
    center = size // 2
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    high_mask = radius >= 18
    very_high_mask = radius >= 34
    high_values = magnitude[high_mask]
    very_high_values = magnitude[very_high_mask] if np.any(very_high_mask) else high_values
    if high_values.size == 0:
        return {
            "frequency_peak_score": 0.0,
            "high_frequency_energy": 0.0,
            "horizontal_band_energy": 0.0,
            "vertical_band_energy": 0.0,
            "moire_score": 0.0,
        }

    frequency_peak_score = float(np.max(high_values) / (np.mean(high_values) + 1e-6))
    high_frequency_energy = float(np.mean(very_high_values) / (np.mean(magnitude) + 1e-6))
    horizontal_mask = np.zeros_like(high_mask, dtype=bool)
    horizontal_mask[center - 2 : center + 3, :] = True
    vertical_mask = np.zeros_like(high_mask, dtype=bool)
    vertical_mask[:, center - 2 : center + 3] = True
    horizontal_values = magnitude[horizontal_mask & high_mask]
    vertical_values = magnitude[vertical_mask & high_mask]
    horizontal_band_energy = float(np.mean(horizontal_values) / (np.mean(high_values) + 1e-6)) if horizontal_values.size else 0.0
    vertical_band_energy = float(np.mean(vertical_values) / (np.mean(high_values) + 1e-6)) if vertical_values.size else 0.0
    peak_component = min(1.0, max(0.0, (frequency_peak_score - 2.2) / 4.8))
    energy_component = min(1.0, max(0.0, (high_frequency_energy - 0.82) / 0.55))
    band_component = min(1.0, max(0.0, (max(horizontal_band_energy, vertical_band_energy) - 0.95) / 0.75))
    return {
        "frequency_peak_score": frequency_peak_score,
        "high_frequency_energy": high_frequency_energy,
        "horizontal_band_energy": horizontal_band_energy,
        "vertical_band_energy": vertical_band_energy,
        "moire_score": float(0.45 * peak_component + 0.35 * energy_component + 0.20 * band_component),
    }


def extract_skin_frequency_patches(gray_image: np.ndarray) -> list[np.ndarray]:
    if gray_image is None or gray_image.size == 0:
        return []

    height, width = gray_image.shape[:2]
    if height < 72 or width < 72:
        return []

    regions = [
        (0.34, 0.12, 0.66, 0.28),
        (0.18, 0.45, 0.39, 0.70),
        (0.61, 0.45, 0.82, 0.70),
    ]
    patches = []
    for x1_ratio, y1_ratio, x2_ratio, y2_ratio in regions:
        x1 = max(0, min(width - 1, int(width * x1_ratio)))
        y1 = max(0, min(height - 1, int(height * y1_ratio)))
        x2 = max(x1 + 1, min(width, int(width * x2_ratio)))
        y2 = max(y1 + 1, min(height, int(height * y2_ratio)))
        patch = gray_image[y1:y2, x1:x2]
        if patch.size and patch.shape[0] >= 24 and patch.shape[1] >= 24:
            patches.append(patch)
    return patches


def calculate_skin_frequency_features(gray_image: np.ndarray, cv2) -> dict[str, float]:
    patches = extract_skin_frequency_patches(gray_image)
    if not patches:
        return {
            "skin_patch_count": 0.0,
            "skin_frequency_peak_score_median": 0.0,
            "skin_high_frequency_energy_median": 0.0,
            "skin_moire_score_median": 0.0,
            "skin_moire_score_max": 0.0,
        }

    patch_features = [calculate_frequency_features(patch, cv2) for patch in patches]
    return {
        "skin_patch_count": float(len(patch_features)),
        "skin_frequency_peak_score_median": float(
            np.median([item["frequency_peak_score"] for item in patch_features])
        ),
        "skin_high_frequency_energy_median": float(
            np.median([item["high_frequency_energy"] for item in patch_features])
        ),
        "skin_moire_score_median": float(np.median([item["moire_score"] for item in patch_features])),
        "skin_moire_score_max": float(np.max([item["moire_score"] for item in patch_features])),
    }


def run_onnx(model_path: Path, image_paths: list[Path], image_size: int, batch_size: int) -> dict[Path, float]:
    import cv2
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    scores: dict[Path, float] = {}

    batch: list[np.ndarray] = []
    batch_paths: list[Path] = []
    for path in image_paths:
        tensor = preprocess_image(path, image_size, cv2)
        if tensor is None:
            continue
        batch.append(tensor)
        batch_paths.append(path)
        if len(batch) >= batch_size:
            logits = session.run(None, {input_name: np.stack(batch).astype(np.float32)})[0].reshape(-1)
            scores.update({item_path: sigmoid(float(logit)) for item_path, logit in zip(batch_paths, logits)})
            batch.clear()
            batch_paths.clear()

    if batch:
        logits = session.run(None, {input_name: np.stack(batch).astype(np.float32)})[0].reshape(-1)
        scores.update({item_path: sigmoid(float(logit)) for item_path, logit in zip(batch_paths, logits)})

    return scores


def measure_images(image_paths: list[Path]) -> dict[Path, dict]:
    import cv2

    measurements = {}
    for path in image_paths:
        quality = measure_image_quality(path, cv2)
        if quality is not None:
            measurements[path] = quality
    return measurements


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted_spoof = (scores >= threshold).astype(int)
    live_mask = labels == 0
    spoof_mask = labels == 1

    tn = int(np.sum((predicted_spoof == 0) & live_mask))
    fp = int(np.sum((predicted_spoof == 1) & live_mask))
    fn = int(np.sum((predicted_spoof == 0) & spoof_mask))
    tp = int(np.sum((predicted_spoof == 1) & spoof_mask))

    apcer = fn / max(tp + fn, 1)
    bpcer = fp / max(tn + fp, 1)
    acer = (apcer + bpcer) / 2
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "apcer": float(apcer),
        "bpcer": float(bpcer),
        "acer": float(acer),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_grid(step: float) -> np.ndarray:
    step = max(min(step, 0.1), 0.0005)
    values = np.arange(0.0, 1.0 + step, step)
    return np.clip(values, 0.0, 1.0)


def summarize_thresholds(rows: list[dict], target_apcer: float, step: float) -> tuple[pd.DataFrame, dict]:
    labels = np.array([row["label_id"] for row in rows], dtype=int)
    scores = np.array([row["p_spoof"] for row in rows], dtype=float)
    metrics = [compute_metrics(labels, scores, threshold) for threshold in threshold_grid(step)]
    summary_df = pd.DataFrame(metrics)

    best_acer = min(metrics, key=lambda item: (item["acer"], item["apcer"], item["bpcer"]))
    secure_candidates = [item for item in metrics if item["apcer"] <= target_apcer]
    if secure_candidates:
        secure = min(secure_candidates, key=lambda item: (item["bpcer"], item["acer"], -item["threshold"]))
    else:
        secure = min(metrics, key=lambda item: (item["apcer"], item["bpcer"], item["acer"]))

    report = {
        "best_acer": best_acer,
        "recommended_security": secure,
        "target_apcer": target_apcer,
        "counts": {
            "live": int(np.sum(labels == 0)),
            "spoof": int(np.sum(labels == 1)),
            "total": int(len(labels)),
        },
        "score_distribution": {
            "live_p50": float(np.percentile(scores[labels == 0], 50)) if np.any(labels == 0) else None,
            "live_p75": float(np.percentile(scores[labels == 0], 75)) if np.any(labels == 0) else None,
            "live_p95": float(np.percentile(scores[labels == 0], 95)) if np.any(labels == 0) else None,
            "spoof_p05": float(np.percentile(scores[labels == 1], 5)) if np.any(labels == 1) else None,
            "spoof_p25": float(np.percentile(scores[labels == 1], 25)) if np.any(labels == 1) else None,
            "spoof_p50": float(np.percentile(scores[labels == 1], 50)) if np.any(labels == 1) else None,
        },
    }
    return summary_df, report


def main() -> None:
    args = parse_args()
    input_root = resolve_repo_path(args.input_root)
    live_dir = resolve_repo_path(args.live_dir) or input_root / "live"
    spoof_dir = resolve_repo_path(args.spoof_dir) or input_root / "spoof"
    model_path = resolve_repo_path(args.model)
    metadata_path = resolve_repo_path(args.metadata)
    metadata = load_metadata(metadata_path)
    image_size = int(metadata.get("image_size", 224))

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo ONNX nao encontrado: {model_path}")
    if not live_dir.exists():
        raise FileNotFoundError(f"Pasta live nao encontrada: {live_dir}")
    if not spoof_dir.exists():
        raise FileNotFoundError(f"Pasta spoof nao encontrada: {spoof_dir}")

    output_dir = resolve_repo_path(args.output_dir)
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "data" / "reports" / "antispoof_calibration" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = (
        build_samples(live_dir, "live", args.sample_mode)
        + build_samples(spoof_dir, "spoof", args.sample_mode)
    )
    image_paths = sorted({path for sample in samples for path in sample["paths"]})
    if not samples or not image_paths:
        raise RuntimeError("Nenhuma imagem encontrada para calibracao.")

    print(f"amostras: {len(samples)} | imagens: {len(image_paths)}")
    print(f"modelo: {model_path}")
    print(f"agregacao: {args.aggregation}")

    image_scores = run_onnx(model_path, image_paths, image_size, args.batch_size)
    image_quality = measure_images(image_paths)
    rows = []
    frame_rows = []
    for sample in samples:
        frame_scores = [image_scores[path] for path in sample["paths"] if path in image_scores]
        frame_quality = [image_quality[path] for path in sample["paths"] if path in image_quality]
        if not frame_scores:
            continue
        p_spoof = aggregate_scores(frame_scores, args.aggregation)
        quality_summary = {
            key: float(np.median([item[key] for item in frame_quality])) if frame_quality else None
            for key in [
                "brightness_mean",
                "brightness_p95",
                "contrast_std",
                "overexposed_ratio",
                "glare_ratio",
                "frequency_peak_score",
                "high_frequency_energy",
                "horizontal_band_energy",
                "vertical_band_energy",
                "moire_score",
                "skin_patch_count",
                "skin_frequency_peak_score_median",
                "skin_high_frequency_energy_median",
                "skin_moire_score_median",
                "skin_moire_score_max",
            ]
        }
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "label": sample["label"],
                "label_id": sample["label_id"],
                "p_spoof": p_spoof,
                "aggregation": args.aggregation,
                "frame_count": len(frame_scores),
                "min_frame_score": float(np.min(frame_scores)),
                "max_frame_score": float(np.max(frame_scores)),
                "mean_frame_score": float(np.mean(frame_scores)),
                **quality_summary,
                "paths": "|".join(str(path) for path in sample["paths"]),
            }
        )
        for index, path in enumerate(sample["paths"], start=1):
            if path in image_scores:
                quality = image_quality.get(path, {})
                frame_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "label": sample["label"],
                        "frame_index": index,
                        "p_spoof": image_scores[path],
                        **quality,
                        "path": str(path),
                    }
                )

    if not rows:
        raise RuntimeError("O modelo nao conseguiu processar nenhuma imagem.")

    scores_path = output_dir / "antispoof_scores.csv"
    frames_path = output_dir / "antispoof_frame_scores.csv"
    thresholds_path = output_dir / "threshold_summary.csv"
    report_path = output_dir / "calibration_report.json"

    scores_df = pd.DataFrame(rows)
    frames_df = pd.DataFrame(frame_rows)
    threshold_df, report = summarize_thresholds(rows, args.target_apcer, args.threshold_step)
    report.update(
        {
            "model": str(model_path),
            "metadata": str(metadata_path) if metadata_path else None,
            "metadata_threshold": metadata.get("threshold"),
            "image_size": image_size,
            "sample_mode": args.sample_mode,
            "aggregation": args.aggregation,
            "output_dir": str(output_dir),
        }
    )

    scores_df.to_csv(scores_path, index=False)
    frames_df.to_csv(frames_path, index=False)
    threshold_df.to_csv(thresholds_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    best = report["best_acer"]
    secure = report["recommended_security"]
    print("\n=== calibracao ===")
    print(f"scores: {scores_path}")
    print(f"frames: {frames_path}")
    print(f"thresholds: {thresholds_path}")
    print(f"report: {report_path}")
    print(
        "best_ACER "
        f"thr={best['threshold']:.3f} ACER={best['acer']:.4f} "
        f"APCER={best['apcer']:.4f} BPCER={best['bpcer']:.4f}"
    )
    print(
        "recommended_security "
        f"thr={secure['threshold']:.3f} ACER={secure['acer']:.4f} "
        f"APCER={secure['apcer']:.4f} BPCER={secure['bpcer']:.4f}"
    )


if __name__ == "__main__":
    main()
