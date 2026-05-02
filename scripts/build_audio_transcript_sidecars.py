#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.app.pipeline.audio_asr import prepare_audio_for_asr


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
DEFAULT_MODEL = "medium"
DEFAULT_OUTPUT_SUFFIX = ".transcript.json"
DEFAULT_PREPROCESSED_OUTPUT_SUFFIX = ".preprocessed.transcript.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera sidecars de transcricao/alinhamento para audios usando faster-whisper. "
            "O objetivo e alimentar a camada fonetico-fonologica simbolica offline."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Arquivo de audio ou pasta com audios.")
    input_group.add_argument("--manifest", help="CSV com coluna audio_path.")

    parser.add_argument("--audio-column", default="audio_path", help="Coluna do manifesto com caminho do audio.")
    parser.add_argument("--output-suffix", default=None, help="Sufixo do sidecar gerado.")
    parser.add_argument(
        "--preprocessed-output-suffix",
        default=DEFAULT_PREPROCESSED_OUTPUT_SUFFIX,
        help="Sufixo usado quando o pre-tratamento de ASR estiver ativo e --output-suffix nao for informado.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo faster-whisper: tiny, base, small, medium, large-v3 etc.")
    parser.add_argument("--language", default="pt", help="Idioma para ASR. Use 'pt' para portugues.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Dispositivo de inferencia.")
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Tipo de computacao do faster-whisper: default, int8, float16, float32 etc.",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true", help="Ativa VAD do faster-whisper.")
    parser.add_argument(
        "--no-preprocess-asr-audio",
        action="store_true",
        help="Desativa o pre-tratamento leve usado apenas como entrada do ASR.",
    )
    parser.add_argument("--recursive", action="store_true", help="Busca audios recursivamente quando --input for pasta.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescreve sidecars existentes.")
    parser.add_argument("--limit", type=int, default=None, help="Limita a quantidade de audios processados.")
    parser.add_argument("--dry-run", action="store_true", help="Lista arquivos sem transcrever.")
    return parser.parse_args()


def resolve_path(path: str, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base is not None:
        return (base / candidate).resolve()
    return candidate.resolve()


def collect_from_input(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in AUDIO_SUFFIXES:
            raise ValueError(f"Arquivo nao parece audio suportado: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Entrada nao encontrada: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def collect_from_manifest(manifest_path: Path, audio_column: str) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto nao encontrado: {manifest_path}")

    audio_paths = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or audio_column not in reader.fieldnames:
            raise ValueError(f"Manifesto sem coluna obrigatoria: {audio_column}")
        for row in reader:
            value = (row.get(audio_column) or "").strip()
            if not value:
                continue
            audio_paths.append(resolve_path(value, manifest_path.parent))
    return audio_paths


def sidecar_path_for(audio_path: Path, output_suffix: str) -> Path:
    suffix = output_suffix if output_suffix.startswith(".") else f".{output_suffix}"
    return audio_path.with_suffix(suffix)


def output_suffix_for(args: argparse.Namespace) -> str:
    if args.output_suffix:
        return args.output_suffix
    if args.no_preprocess_asr_audio:
        return DEFAULT_OUTPUT_SUFFIX
    return args.preprocessed_output_suffix


def serialize_word(word) -> dict:
    return {
        "word": getattr(word, "word", "").strip(),
        "start": float(getattr(word, "start", 0.0) or 0.0),
        "end": float(getattr(word, "end", 0.0) or 0.0),
        "confidence": float(getattr(word, "probability", 0.0) or 0.0),
    }


def serialize_segment(segment) -> dict:
    words = [serialize_word(word) for word in (getattr(segment, "words", None) or []) if getattr(word, "word", "").strip()]
    return {
        "id": int(getattr(segment, "id", 0) or 0),
        "start": float(getattr(segment, "start", 0.0) or 0.0),
        "end": float(getattr(segment, "end", 0.0) or 0.0),
        "text": (getattr(segment, "text", "") or "").strip(),
        "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
        "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
        "words": words,
    }


def flatten_words(segments: list[dict]) -> list[dict]:
    words = []
    for segment in segments:
        if segment.get("words"):
            words.extend(segment["words"])
        elif segment.get("text"):
            words.append(
                {
                    "word": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": max(0.0, 1.0 - float(segment.get("no_speech_prob") or 0.0)),
                }
            )
    return words


def load_model(args: argparse.Namespace):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Dependencia ausente: instale faster-whisper no ambiente offline. "
            "Exemplo: .venv/bin/pip install faster-whisper"
        ) from exc

    device = args.device
    if device == "auto":
        device = "cuda"
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    print(f"asr_model = {args.model}")
    print(f"device = {device}")
    print(f"compute_type = {args.compute_type}")
    return WhisperModel(args.model, device=device, compute_type=args.compute_type)


def transcribe_audio(model, audio_path: Path, args: argparse.Namespace) -> dict:
    asr_audio_path, preprocessing, temporary_path = prepare_audio_for_asr(
        str(audio_path),
        enabled=not args.no_preprocess_asr_audio,
    )
    try:
        segments_iter, info = model.transcribe(
            asr_audio_path,
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=args.vad_filter,
            word_timestamps=True,
        )
        segments = [serialize_segment(segment) for segment in segments_iter]
        text = " ".join(segment["text"] for segment in segments if segment.get("text")).strip()
        words = flatten_words(segments)
        return {
            "text": text,
            "words": words,
            "segments": segments,
            "asr": {
                "engine": "faster-whisper",
                "model": args.model,
                "language": getattr(info, "language", args.language),
                "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                "duration_seconds": float(getattr(info, "duration", 0.0) or 0.0),
                "duration_after_vad_seconds": float(getattr(info, "duration_after_vad", 0.0) or 0.0),
                "device": args.device,
                "compute_type": args.compute_type,
                "beam_size": args.beam_size,
                "vad_filter": bool(args.vad_filter),
                "preprocessing": preprocessing,
            },
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.input:
        audio_paths = collect_from_input(resolve_path(args.input), args.recursive)
    else:
        audio_paths = collect_from_manifest(resolve_path(args.manifest), args.audio_column)

    if args.limit is not None:
        audio_paths = audio_paths[: max(args.limit, 0)]

    if not audio_paths:
        print("Nenhum audio encontrado.")
        return

    print(f"audios = {len(audio_paths)}")
    for audio_path in audio_paths[:5]:
        print(f"- {audio_path}")
    if len(audio_paths) > 5:
        print(f"... +{len(audio_paths) - 5}")

    if args.dry_run:
        return

    model = load_model(args)
    written = 0
    skipped = 0
    failed = 0

    for index, audio_path in enumerate(audio_paths, start=1):
        output_path = sidecar_path_for(audio_path, output_suffix_for(args))
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(audio_paths)}] pula existente: {output_path}")
            continue

        try:
            print(f"[{index}/{len(audio_paths)}] transcrevendo: {audio_path}")
            transcript = transcribe_audio(model, audio_path, args)
            transcript["source_audio"] = str(audio_path)
            transcript["source_audio_sha256"] = file_sha256(audio_path)
            transcript["sidecar_version"] = "audio_transcript_sidecar_v1"
            write_json(output_path, transcript)
            written += 1
            print(f"  salvo: {output_path} | palavras={len(transcript['words'])}")
        except Exception as exc:
            failed += 1
            print(f"  erro: {exc}", file=sys.stderr)

    print(f"gerados = {written}")
    print(f"ignorados = {skipped}")
    print(f"falhas = {failed}")


if __name__ == "__main__":
    main()
