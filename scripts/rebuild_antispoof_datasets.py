#!/usr/bin/env python3
"""Build a small face anti-spoofing dataset from public sources.

The script downloads or reads a dataset, extracts frames/images, crops faces,
and writes a manifest suitable for training a binary live/spoof model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

DATASETS = {
    "siw": {
        "slug": "nagatoyuki1218/anti-spoofing-siw-dataset",
        "description": "SiW anti-spoofing sample from Kaggle",
    },
}

LIVE_TOKENS = {
    "live",
    "real",
    "genuine",
    "bonafide",
    "bona",
    "true",
}
SPOOF_TOKENS = {
    "spoof",
    "attack",
    "attacks",
    "fake",
    "print",
    "printed",
    "photo",
    "paper",
    "replay",
    "screen",
    "display",
    "video",
}
NEUTRAL_TOKENS = {
    "anti",
    "antispoof",
    "anti_spoof",
    "anti-spoof",
    "spoofing",
    "dataset",
    "datasets",
    "siw",
    "face",
    "faces",
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "dev",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa/processa datasets publicos de face anti-spoofing, extrai frames/crops "
            "e gera um manifesto live/spoof para treino."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="siw",
        help="Preset de dataset publico. Padrao: siw.",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Slug Kaggle manual, ex.: usuario/dataset. Sobrescreve o slug do preset.",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Pasta local ja extraida. Quando informado, nao baixa do Kaggle.",
    )
    parser.add_argument(
        "--download-root",
        default="data/downloads/antispoof",
        help="Diretorio para zip/extracao de datasets publicos.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/antispoof/processed",
        help="Diretorio de saida com crops processados.",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/antispoof_manifest.csv",
        help="CSV de saida.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=10,
        help="Quantidade maxima de frames extraidos por video.",
    )
    parser.add_argument(
        "--max-items-per-label",
        type=int,
        default=1000,
        help="Limite de imagens/crops processados por label.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Tamanho final do crop quadrado.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para amostragem deterministica.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove a saida processada do dataset antes de recriar.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Baixa novamente mesmo se ja houver zip/extracao.",
    )
    parser.add_argument(
        "--kaggle-json",
        default=None,
        help="Caminho para kaggle.json. Sera copiado para .kaggle/kaggle.json no projeto.",
    )
    parser.add_argument(
        "--kaggle-config-dir",
        default=".kaggle",
        help="Diretorio de config Kaggle usado pelo script.",
    )
    parser.add_argument(
        "--keep-no-face",
        action="store_true",
        help="Mantem frame/imagem usando crop central quando Haar cascade nao achar face.",
    )
    parser.add_argument(
        "--split-train",
        type=float,
        default=0.7,
        help="Proporcao de grupos no split train.",
    )
    parser.add_argument(
        "--split-val",
        type=float,
        default=0.15,
        help="Proporcao de grupos no split val.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def configure_kaggle(args: argparse.Namespace) -> Path:
    config_dir = resolve_repo_path(args.kaggle_config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    if args.kaggle_json:
        source = Path(args.kaggle_json).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"kaggle.json nao encontrado: {source}")
        target = config_dir / "kaggle.json"
        shutil.copy2(source, target)
        target.chmod(0o600)

    return config_dir


def kaggle_download_and_extract(args: argparse.Namespace) -> Path:
    dataset_spec = DATASETS[args.dataset]
    slug = args.slug or dataset_spec["slug"]
    dataset_name = slug.split("/")[-1]
    download_root = resolve_repo_path(args.download_root) / args.dataset
    zip_path = download_root / f"{dataset_name}.zip"
    extract_dir = download_root / "extracted"

    if extract_dir.exists() and any(extract_dir.iterdir()) and not args.force_download:
        print(f"{args.dataset}: extracao existente em {extract_dir}")
        return extract_dir

    config_dir = configure_kaggle(args)
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    download_root.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists() or args.force_download:
        kaggle_bin = Path(sys.executable).with_name("kaggle")
        kaggle_command = str(kaggle_bin) if kaggle_bin.exists() else "kaggle"
        command = [
            kaggle_command,
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(download_root),
        ]
        print(f"{args.dataset}: baixando {slug}")
        subprocess.run(command, check=True, env=env, cwd=REPO_ROOT)

    if extract_dir.exists() and args.force_download:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.dataset}: extraindo {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def tokenize_path(path: Path) -> list[str]:
    tokens: list[str] = []
    for part in path.parts:
        normalized = part.lower().replace(".", "_")
        for token in normalized.replace("-", "_").split("_"):
            token = token.strip()
            if token and token not in NEUTRAL_TOKENS:
                tokens.append(token)
    return tokens


def infer_label(relative_path: Path) -> str | None:
    tokens = tokenize_path(relative_path)
    live_score = sum(1 for token in tokens if token in LIVE_TOKENS)
    spoof_score = sum(1 for token in tokens if token in SPOOF_TOKENS)

    if live_score == 0 and spoof_score == 0:
        return None
    if live_score >= spoof_score:
        return "live"
    return "spoof"


def infer_attack_type(relative_path: Path, label: str) -> str:
    tokens = set(tokenize_path(relative_path))
    if label == "live":
        return "live"
    for attack_type in ("print", "photo", "replay", "screen", "display", "video"):
        if attack_type in tokens:
            return attack_type
    return "spoof"


def collect_media(input_root: Path) -> list[Path]:
    media = []
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            media.append(path)
    return media


def group_id_for_media(path: Path, input_root: Path) -> str:
    relative = path.relative_to(input_root)
    parent = relative.parent.as_posix() or "root"
    stem = path.stem
    return f"{parent}/{stem}"


def infer_subject_id(relative_path: Path) -> str:
    stem = relative_path.stem.lower()
    match = re.search(r"(?:live|spoof|real|fake|attack)[_-](\d+)", stem)
    if match:
        return match.group(1)

    for part in relative_path.parts:
        digit_match = re.search(r"\d{2,}", part)
        if digit_match:
            return digit_match.group(0)

    return "unknown"


def split_for_group(group_id: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def load_cv2():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Dependencias ausentes. Instale opencv-python-headless e numpy no venv de treino."
        ) from exc
    return cv2, np


def make_square_crop(image, box: tuple[int, int, int, int] | None, image_size: int):
    cv2, _np = load_cv2()
    height, width = image.shape[:2]

    if box is None:
        side = min(width, height)
        x1 = max(0, (width - side) // 2)
        y1 = max(0, (height - side) // 2)
        x2 = x1 + side
        y2 = y1 + side
    else:
        x, y, w, h = box
        side = int(max(w, h) * 1.35)
        cx = x + (w // 2)
        cy = y + (h // 2)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(width, x1 + side)
        y2 = min(height, y1 + side)
        x1 = max(0, x2 - side)
        y1 = max(0, y2 - side)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def detect_face_box(image, face_cascade):
    cv2, _np = load_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda item: item[2] * item[3]))


def iter_video_frames(video_path: Path, frames_per_video: int):
    cv2, _np = load_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            indices = list(range(frames_per_video))
        else:
            if frames_per_video <= 1:
                indices = [total_frames // 2]
            else:
                indices = [
                    int(round(i * (total_frames - 1) / (frames_per_video - 1)))
                    for i in range(frames_per_video)
                ]

        for frame_index in indices:
            if total_frames > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                yield frame_index, frame
    finally:
        capture.release()


def iter_media_frames(path: Path, frames_per_video: int):
    cv2, _np = load_cv2()
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            yield 0, image
        return

    yield from iter_video_frames(path, frames_per_video)


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "label",
        "label_id",
        "source",
        "subject_id",
        "session_id",
        "attack_type",
        "split",
        "source_media",
        "frame_index",
        "face_detected",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_dataset(args: argparse.Namespace, input_root: Path, output_root: Path) -> list[dict[str, str]]:
    cv2, _np = load_cv2()
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Haar cascade nao carregou: {cascade_path}")

    media = collect_media(input_root)
    rng = random.Random(args.seed)
    rng.shuffle(media)

    label_counts = {"live": 0, "spoof": 0}
    no_label_skipped = 0
    no_face_skipped = 0
    rows: list[dict[str, str]] = []

    for media_path in media:
        relative = media_path.relative_to(input_root)
        label = infer_label(relative)
        if label not in label_counts:
            no_label_skipped += 1
            continue
        if label_counts[label] >= args.max_items_per_label:
            continue

        group_id = group_id_for_media(media_path, input_root)
        split = split_for_group(group_id, args.split_train, args.split_val)
        attack_type = infer_attack_type(relative, label)
        session_id = hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:12]
        subject_id = infer_subject_id(relative)

        for frame_index, frame in iter_media_frames(media_path, args.frames_per_video):
            if label_counts[label] >= args.max_items_per_label:
                break

            face_box = detect_face_box(frame, face_cascade)
            if face_box is None and not args.keep_no_face:
                no_face_skipped += 1
                continue
            crop = make_square_crop(frame, face_box, args.image_size)
            if crop is None:
                no_face_skipped += 1
                continue

            label_counts[label] += 1
            output_path = (
                output_root
                / split
                / label
                / f"{args.dataset}_{session_id}_{frame_index:06d}_{label_counts[label]:06d}.jpg"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            rows.append(
                {
                    "path": str(output_path.relative_to(REPO_ROOT)),
                    "label": label,
                    "label_id": "0" if label == "live" else "1",
                    "source": args.dataset,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "attack_type": attack_type,
                    "split": split,
                    "source_media": str(media_path.relative_to(input_root)),
                    "frame_index": str(frame_index),
                    "face_detected": "1" if face_box is not None else "0",
                }
            )

        if all(count >= args.max_items_per_label for count in label_counts.values()):
            break

    print(f"midias encontradas: {len(media)}")
    print(f"processadas: live={label_counts['live']} spoof={label_counts['spoof']}")
    print(f"ignoradas sem label inferivel: {no_label_skipped}")
    print(f"frames/imagens ignorados sem face: {no_face_skipped}")
    return rows


def main() -> None:
    args = parse_args()
    if args.split_train <= 0 or args.split_val < 0 or args.split_train + args.split_val >= 1:
        raise ValueError("--split-train + --split-val deve ser menor que 1")

    input_root = resolve_repo_path(args.input_root) if args.input_root else kaggle_download_and_extract(args)
    if not input_root.exists():
        raise FileNotFoundError(f"Dataset de entrada nao encontrado: {input_root}")

    output_root = resolve_repo_path(args.processed_root) / args.dataset
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = process_dataset(args, input_root, output_root)
    manifest_path = resolve_repo_path(args.manifest)
    write_manifest(manifest_path, rows)

    split_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["label"])
        split_counts[key] = split_counts.get(key, 0) + 1

    print(f"manifesto gerado: {manifest_path}")
    for split in ("train", "val", "test"):
        print(
            f"{split}: live={split_counts.get((split, 'live'), 0)} "
            f"spoof={split_counts.get((split, 'spoof'), 0)}"
        )


if __name__ == "__main__":
    main()
