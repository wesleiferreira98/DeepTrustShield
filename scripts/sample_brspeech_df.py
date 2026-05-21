#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "audio" / "data" / "brspeech_df_sample"
DEFAULT_MANIFEST = REPO_ROOT / "audio" / "data" / "manifests" / "brspeech_df_sample_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa uma amostra pequena e balanceada do BRSpeech-DF via streaming do Hugging Face, "
            "salva WAV local e gera manifest compativel com o treino de audio."
        )
    )
    parser.add_argument("--dataset", default="AKCIT-Deepfake/BRSpeech-DF")
    parser.add_argument("--real-config", default="bonafide")
    parser.add_argument("--fake-config", default="spoof")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Manifest CSV com amostras que nao devem ser reutilizadas. Pode ser repetido.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scan-multiplier", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_") or "item"


def target_count_for_split(args: argparse.Namespace, split: str) -> int:
    normalized = split.lower()
    if normalized in {"validation", "valid", "val", "dev"}:
        return args.val_per_class
    if normalized == "test":
        return args.test_per_class
    return args.train_per_class


def manifest_split_name(split: str) -> str:
    normalized = split.lower()
    if normalized in {"validation", "valid", "dev"}:
        return "val"
    return normalized


def stable_id(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_excluded_ids(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for path_text in paths:
        path = resolve_repo_path(path_text)
        if not path.exists():
            print(f"aviso: exclude-manifest nao encontrado: {path}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                config = row.get("brspeech_config") or ("bonafide" if row.get("label") == "real" else "spoof")
                split = row.get("brspeech_split") or row.get("split") or ""
                label = row.get("label") or ""
                source = row.get("brspeech_source") or row.get("speaker_id") or row.get("audio_path") or ""
                excluded.add(stable_id(config, split, label, source))
    return excluded


def extract_audio(example: dict) -> tuple[np.ndarray, int]:
    audio = None
    for key in ("audio", "Audio", "wav", "file"):
        if key in example:
            audio = example[key]
            break
    if audio is None:
        raise ValueError(f"Exemplo sem coluna de audio conhecida. Chaves disponiveis: {sorted(example.keys())}")

    if isinstance(audio, str):
        waveform, sample_rate = sf.read(audio, dtype="float32", always_2d=False)
        return normalize_waveform_array(waveform), int(sample_rate)

    if not isinstance(audio, dict):
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            data = getattr(samples, "data", None)
            sample_rate = getattr(samples, "sample_rate", None)
            if data is not None and sample_rate is not None:
                return normalize_waveform_array(np.asarray(data, dtype=np.float32)), int(sample_rate)
        raise ValueError(
            "Exemplo com coluna de audio em formato inesperado: "
            f"{type(audio).__name__}. Chaves disponiveis: {sorted(example.keys())}"
        )

    array = audio.get("array")
    sample_rate = audio.get("sampling_rate")
    if array is None and audio.get("path"):
        waveform, sample_rate = sf.read(str(audio["path"]), dtype="float32", always_2d=False)
        return normalize_waveform_array(waveform), int(sample_rate)
    if array is None and audio.get("bytes"):
        import io

        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        return normalize_waveform_array(waveform), int(sample_rate)
    if array is None or sample_rate is None:
        raise ValueError(
            "Exemplo de audio sem array/sampling_rate/path/bytes. "
            f"Chaves do audio: {sorted(audio.keys())}; chaves do exemplo: {sorted(example.keys())}"
        )

    return normalize_waveform_array(np.asarray(array, dtype=np.float32)), int(sample_rate)


def normalize_waveform_array(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1 if waveform.shape[0] > waveform.shape[1] else 0)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    return waveform


def example_source(example: dict, fallback: str) -> str:
    for key in ("file", "path", "audio_path", "source", "id", "utt_id", "speaker_id"):
        value = example.get(key)
        if value:
            return str(value)
    audio = example.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return fallback


def iter_stream(dataset_name: str, config: str, split: str) -> Iterable[dict]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dependencia ausente: instale com `.venv/bin/pip install datasets` "
            "ou atualize o ambiente com requirements-training.txt."
        ) from exc

    dataset = load_dataset(dataset_name, name=config, split=split, streaming=True)
    try:
        dataset = dataset.cast_column("audio", Audio())
    except Exception:
        pass
    return dataset


def sample_class(
    *,
    dataset_name: str,
    config: str,
    split: str,
    label: str,
    output_dir: Path,
    target_count: int,
    seed: int,
    max_scan_multiplier: int,
    overwrite: bool,
    excluded_ids: set[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    stream = iter_stream(dataset_name, config, split)
    rng = random.Random(seed)
    max_scan = max(target_count * max_scan_multiplier, target_count)
    indices = list(range(max_scan))
    rng.shuffle(indices)
    wanted = set(indices[:target_count])
    split_name = manifest_split_name(split)
    label_dir = output_dir / split_name / label
    label_dir.mkdir(parents=True, exist_ok=True)
    excluded_ids = excluded_ids or set()

    scanned = 0
    for index, example in enumerate(stream):
        if scanned >= max_scan or len(rows) >= target_count:
            break
        scanned += 1
        if index not in wanted:
            continue

        source = example_source(example, f"{config}_{split}_{index}")
        source_id = stable_id(config, split, label, source)
        if source_id in excluded_ids:
            continue
        item_id = stable_id(config, split, label, source, index)
        audio_path = label_dir / f"{slugify(label)}_{item_id}.wav"
        metadata_path = audio_path.with_suffix(".json")

        if overwrite or not audio_path.exists():
            waveform, sample_rate = extract_audio(example)
            sf.write(str(audio_path), waveform, sample_rate)
            metadata = {
                "dataset": dataset_name,
                "config": config,
                "split": split,
                "label": label,
                "source": source,
                "sample_rate": sample_rate,
                "num_samples": int(waveform.shape[-1]),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        rows.append(
            {
                "audio_path": str(audio_path),
                "split": split_name,
                "label": label,
                "source": "brspeech_df",
                "accent": "unknown",
                "speaker_id": source,
                "brspeech_config": config,
                "brspeech_split": split,
                "brspeech_source": source,
                "brspeech_source_id": source_id,
            }
        )

    if len(rows) < target_count:
        print(f"aviso: {config}/{split}/{label} gerou {len(rows)} de {target_count} amostras solicitadas.")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio_path",
        "split",
        "label",
        "source",
        "accent",
        "speaker_id",
        "brspeech_config",
        "brspeech_split",
        "brspeech_source",
        "brspeech_source_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    manifest_path = resolve_repo_path(args.manifest)
    excluded_ids = load_excluded_ids(args.exclude_manifest)
    if excluded_ids:
        print(f"excluindo amostras ja vistas = {len(excluded_ids)}")
    rows: list[dict] = []

    class_specs = [
        (args.real_config, "real"),
        (args.fake_config, "fake"),
    ]
    for split in args.splits:
        target_count = target_count_for_split(args, split)
        if target_count <= 0:
            continue
        for config, label in class_specs:
            print(f"amostrando {target_count} | split={split} | config={config} | label={label}")
            rows.extend(
                sample_class(
                    dataset_name=args.dataset,
                    config=config,
                    split=split,
                    label=label,
                    output_dir=output_dir,
                    target_count=target_count,
                    seed=args.seed,
                    max_scan_multiplier=args.max_scan_multiplier,
                    overwrite=args.overwrite,
                    excluded_ids=excluded_ids,
                )
            )

    write_manifest(manifest_path, rows)
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = counts.setdefault(row["split"], {"real": 0, "fake": 0})
        bucket[row["label"]] += 1
    print(f"rows = {len(rows)} {counts}")
    print(f"output_dir = {output_dir}")
    print(f"manifest = {manifest_path}")


if __name__ == "__main__":
    main()
