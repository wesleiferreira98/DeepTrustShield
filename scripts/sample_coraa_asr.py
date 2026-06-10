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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "audio" / "data" / "coraa_sample"
DEFAULT_MANIFEST = REPO_ROOT / "audio" / "data" / "manifests" / "coraa_sample_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa uma amostra pequena do CORAA via streaming do Hugging Face para testes de dominio."
    )
    parser.add_argument("--dataset", default="nilc-nlp/CORAA-MUPE-ASR")
    parser.add_argument("--config", default="", help="Config do dataset no Hugging Face, se existir.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--min-seconds", type=float, default=4.0)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--download-shards",
        action="store_true",
        help="Baixa shards parquet completos do Hugging Face antes de amostrar, evitando streaming remoto instavel.",
    )
    parser.add_argument(
        "--parquet-dir",
        default="",
        help="Pasta para salvar/ler shards parquet locais. Padrao: <output-dir>/parquet.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Quantidade de shards parquet a baixar/ler.")
    parser.add_argument("--total-shards", type=int, default=74, help="Total de shards do split no Hugging Face.")
    parser.add_argument(
        "--shard-start",
        type=int,
        default=0,
        help="Indice inicial dos shards. Ex.: 0 baixa train-00000-of-00074.parquet.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Atualiza o manifesto a cada N amostras salvas para preservar progresso em quedas de rede.",
    )
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


def stable_id(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_waveform_array(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1 if waveform.shape[0] > waveform.shape[1] else 0)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    return waveform


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


def iter_stream(dataset_name: str, config: str, split: str) -> Iterable[dict]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dependencia ausente: instale com `.venv/bin/pip install datasets` "
            "ou atualize o ambiente com requirements-training.txt."
        ) from exc

    kwargs = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    dataset = load_dataset(dataset_name, **kwargs)
    try:
        dataset = dataset.cast_column("audio", Audio())
    except Exception:
        pass
    return dataset


def shard_filename(split: str, index: int, total: int) -> str:
    return f"data/{split}-{index:05d}-of-{total:05d}.parquet"


def download_parquet_shards(
    dataset_name: str,
    *,
    split: str,
    shard_start: int,
    num_shards: int,
    total_shards: int,
    parquet_dir: Path,
) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Dependencia ausente: instale huggingface_hub no ambiente.") from exc

    parquet_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for offset in range(num_shards):
        shard_index = shard_start + offset
        filename = shard_filename(split, shard_index, total_shards)
        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=filename,
            local_dir=str(parquet_dir),
        )
        paths.append(Path(local_path).resolve())
    return paths


def iter_local_parquet(paths: list[Path]) -> Iterable[dict]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dependencia ausente: instale com `.venv/bin/pip install datasets` "
            "ou atualize o ambiente com requirements-training.txt."
        ) from exc

    dataset = load_dataset("parquet", data_files=[str(path) for path in paths], split="train")
    try:
        dataset = dataset.cast_column("audio", Audio())
    except Exception:
        pass
    return dataset


def first_text(example: dict) -> str:
    for key in ("text", "sentence", "transcription", "transcript", "normalized_text"):
        value = example.get(key)
        if value:
            return str(value)
    return ""


def first_source(example: dict, fallback: str) -> str:
    for key in ("audio_name", "file_path", "path", "id", "utt_id", "speaker_id", "source"):
        value = example.get(key)
        if value:
            return str(value)
    audio = example.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return fallback


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio_path",
        "split",
        "label",
        "source",
        "accent",
        "speaker_id",
        "dataset",
        "dataset_split",
        "duration_seconds",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = resolve_repo_path(args.output_dir)
    manifest_path = resolve_repo_path(args.manifest)
    parquet_dir = resolve_repo_path(args.parquet_dir) if args.parquet_dir else output_dir / "parquet"
    audio_dir = output_dir / "real"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.download_shards:
        parquet_paths = download_parquet_shards(
            args.dataset,
            split=args.split,
            shard_start=args.shard_start,
            num_shards=args.num_shards,
            total_shards=args.total_shards,
            parquet_dir=parquet_dir,
        )
        stream = iter_local_parquet(parquet_paths)
    else:
        stream = iter_stream(args.dataset, args.config, args.split)
    selected_indices = set(rng.sample(range(args.max_scan), min(args.max_scan, max(args.count * 20, args.count))))

    scanned = 0
    for index, example in enumerate(stream):
        if scanned >= args.max_scan or len(rows) >= args.count:
            break
        scanned += 1
        if index not in selected_indices:
            continue

        waveform, sample_rate = extract_audio(example)
        duration_seconds = waveform.shape[-1] / float(sample_rate) if sample_rate else 0.0
        if duration_seconds < args.min_seconds or duration_seconds > args.max_seconds:
            continue

        source = first_source(example, f"{args.dataset}_{args.split}_{index}")
        item_id = stable_id(args.dataset, args.split, source, index)
        audio_path = audio_dir / f"coraa_real_{item_id}.wav"
        metadata_path = audio_path.with_suffix(".json")
        text = first_text(example)

        if args.overwrite or not audio_path.exists():
            sf.write(str(audio_path), waveform, sample_rate)
            metadata = {
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "label": "real",
                "source": source,
                "text": text,
                "sample_rate": sample_rate,
                "duration_seconds": duration_seconds,
                "raw_keys": sorted(example.keys()),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        rows.append(
            {
                "audio_path": str(audio_path),
                "split": "test",
                "label": "real",
                "source": "coraa",
                "accent": "unknown",
                "speaker_id": source,
                "dataset": args.dataset,
                "dataset_split": args.split,
                "duration_seconds": f"{duration_seconds:.6f}",
                "text": text,
            }
        )
        if args.flush_every > 0 and len(rows) % args.flush_every == 0:
            write_manifest(manifest_path, rows)

    write_manifest(manifest_path, rows)
    print(f"rows = {len(rows)} real")
    print(f"output_dir = {output_dir}")
    print(f"manifest = {manifest_path}")
    if len(rows) < args.count:
        print(f"aviso: foram coletadas {len(rows)} de {args.count} amostras solicitadas.")


if __name__ == "__main__":
    main()
