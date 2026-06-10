#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "audio" / "data" / "manifests" / "audio_v1_1_mixed_manifest.csv"
DEFAULT_BRSPEECH = REPO_ROOT / "audio" / "data" / "manifests" / "brspeech_df_sample_manifest.csv"
DEFAULT_FAKE_BRACCENT = REPO_ROOT / "audio" / "data" / "manifests" / "fake_braccent_d_codec_equalized_manifest.csv"
DEFAULT_CORAA = REPO_ROOT / "audio" / "data" / "manifests" / "coraa_sample_manifest.csv"
DEFAULT_COLETA_REAL = REPO_ROOT / "audio" / "data" / "manifests" / "coleta_manifest.csv"
DEFAULT_COLETA_FAKE = REPO_ROOT / "audio" / "data" / "coleta_manipulated_strong" / "manifest.csv"

FIELDNAMES = [
    "audio_path",
    "split",
    "label",
    "source",
    "domain",
    "accent",
    "speaker_id",
    "start",
    "end",
    "file_label",
    "manipulation_type",
    "overlap_ratio",
    "ground_truth_path",
    "source_audio",
    "insert_audio",
    "original_split",
    "original_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria um manifesto misto para treino V1.1 de áudio com domínios variados."
    )
    parser.add_argument("--brspeech-manifest", default=str(DEFAULT_BRSPEECH))
    parser.add_argument("--fake-braccent-manifest", default=str(DEFAULT_FAKE_BRACCENT))
    parser.add_argument("--coraa-manifest", default=str(DEFAULT_CORAA))
    parser.add_argument("--coleta-real-manifest", default=str(DEFAULT_COLETA_REAL))
    parser.add_argument("--coleta-fake-manifest", default=str(DEFAULT_COLETA_FAKE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brspeech-train-per-label", type=int, default=500)
    parser.add_argument("--brspeech-val-per-label", type=int, default=100)
    parser.add_argument("--brspeech-test-per-label", type=int, default=100)
    parser.add_argument("--fake-braccent-train-per-label", type=int, default=300)
    parser.add_argument("--fake-braccent-val-per-label", type=int, default=80)
    parser.add_argument("--fake-braccent-test-per-label", type=int, default=100)
    parser.add_argument("--coraa-train-ratio", type=float, default=0.70)
    parser.add_argument("--coraa-val-ratio", type=float, default=0.15)
    parser.add_argument("--coleta-train-ratio", type=float, default=0.70)
    parser.add_argument("--coleta-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--coleta-fake-train-repeat",
        type=int,
        default=1,
        help="Quantidade total de repeticoes das amostras fake da coleta no treino. Nao afeta val/test.",
    )
    parser.add_argument(
        "--coleta-real-train-repeat",
        type=int,
        default=1,
        help="Quantidade total de repeticoes das amostras reais da coleta no treino. Nao afeta val/test.",
    )
    parser.add_argument(
        "--coleta-fake-train-operations",
        default="",
        help=(
            "Lista separada por virgula de manipulacoes da coleta usadas no treino. "
            "Val/test continuam com todas."
        ),
    )
    parser.add_argument(
        "--coleta-real-window-seconds",
        type=float,
        default=0.0,
        help="Se maior que 0, cria janelas temporais para audios reais da coleta nos splits configurados.",
    )
    parser.add_argument(
        "--coleta-real-window-stride-seconds",
        type=float,
        default=4.0,
        help="Passo entre janelas reais da coleta quando --coleta-real-window-seconds estiver ativo.",
    )
    parser.add_argument(
        "--coleta-real-window-splits",
        default="train",
        help="Splits da coleta real que devem ser fatiados em janelas. Ex.: train ou train,val.",
    )
    parser.add_argument(
        "--coleta-real-max-windows-per-file",
        type=int,
        default=0,
        help="Limite opcional de janelas reais por arquivo da coleta. 0 nao limita.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_audio_path(path_text: str, manifest_path: Path) -> str:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_split(split: str) -> str:
    split = (split or "").strip().lower()
    if split in {"validation", "valid"}:
        return "val"
    return split


def slug_from_path(path_text: str) -> str:
    stem = Path(path_text).stem.lower()
    chars = []
    for char in stem:
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "unknown"


def base_row(
    row: dict[str, str],
    *,
    manifest_path: Path,
    split: str | None = None,
    source: str | None = None,
    domain: str | None = None,
    start: str = "",
    end: str = "",
    file_label: str | None = None,
    manipulation_type: str | None = None,
    speaker_id: str | None = None,
) -> dict[str, str]:
    label = (row.get("label") or "").strip().lower()
    output = {
        "audio_path": resolve_audio_path(row["audio_path"], manifest_path),
        "split": split if split is not None else clean_split(row.get("split", "")),
        "label": label,
        "source": source if source is not None else (row.get("source") or "unknown"),
        "domain": domain if domain is not None else (row.get("domain") or row.get("source") or "unknown"),
        "accent": row.get("accent") or "unknown",
        "speaker_id": speaker_id if speaker_id is not None else (row.get("speaker_id") or "unknown"),
        "start": start,
        "end": end,
        "file_label": file_label if file_label is not None else label,
        "manipulation_type": manipulation_type if manipulation_type is not None else (row.get("manipulation_type") or "none"),
        "overlap_ratio": row.get("overlap_ratio") or "",
        "ground_truth_path": row.get("ground_truth_path") or "",
        "source_audio": row.get("source_audio") or "",
        "insert_audio": row.get("insert_audio") or "",
        "original_split": clean_split(row.get("split", "")),
        "original_source": row.get("source") or "",
    }
    return output


def sample_by_split_label(
    rows: list[dict[str, str]],
    *,
    manifest_path: Path,
    rng: random.Random,
    caps: dict[tuple[str, str], int],
    source: str | None = None,
    domain: str | None = None,
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = (row.get("label") or "").strip().lower()
        split = clean_split(row.get("split", ""))
        if label in {"real", "fake"} and split in {"train", "val", "test"}:
            grouped[(split, label)].append(row)

    output = []
    for key, bucket in sorted(grouped.items()):
        bucket = bucket[:]
        rng.shuffle(bucket)
        cap = caps.get(key)
        selected = bucket[:cap] if cap is not None else bucket
        for row in selected:
            output.append(base_row(row, manifest_path=manifest_path, source=source, domain=domain))
    return output


def split_items(items: list[str], *, rng: random.Random, train_ratio: float, val_ratio: float) -> dict[str, str]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if total >= 3:
        train_count = min(max(train_count, 1), total - 2)
        val_count = min(max(val_count, 1), total - train_count - 1)
    split_map = {}
    for index, item in enumerate(shuffled):
        if index < train_count:
            split_map[item] = "train"
        elif index < train_count + val_count:
            split_map[item] = "val"
        else:
            split_map[item] = "test"
    return split_map


def audio_duration_seconds(path_text: str) -> float:
    return float(sf.info(path_text).duration)


def temporal_windows(
    *,
    duration_seconds: float,
    window_seconds: float,
    stride_seconds: float,
) -> list[tuple[float, float]]:
    if window_seconds <= 0 or duration_seconds <= 0:
        return []
    if duration_seconds <= window_seconds:
        return [(0.0, duration_seconds)]

    stride_seconds = max(stride_seconds, 0.1)
    starts: list[float] = []
    start = 0.0
    last_start = max(duration_seconds - window_seconds, 0.0)
    while start < last_start:
        starts.append(start)
        start += stride_seconds
    if not starts or abs(starts[-1] - last_start) > 1e-3:
        starts.append(last_start)
    return [(start, min(start + window_seconds, duration_seconds)) for start in starts]


def limited_windows(
    windows: list[tuple[float, float]],
    *,
    max_windows: int,
    rng: random.Random,
) -> list[tuple[float, float]]:
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows
    selected = windows[:]
    rng.shuffle(selected)
    return sorted(selected[:max_windows])


def build_coraa_rows(
    rows: list[dict[str, str]],
    *,
    manifest_path: Path,
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> list[dict[str, str]]:
    indexed = [str(index) for index, row in enumerate(rows) if (row.get("label") or "").strip().lower() == "real"]
    split_map = split_items(indexed, rng=rng, train_ratio=train_ratio, val_ratio=val_ratio)
    output = []
    for index_text, split in split_map.items():
        row = rows[int(index_text)]
        output.append(
            base_row(
                row,
                manifest_path=manifest_path,
                split=split,
                source="coraa",
                domain="fala_espontanea",
                speaker_id=row.get("speaker_id") or f"coraa_{index_text}",
            )
        )
    return output


def build_coleta_rows(
    real_rows: list[dict[str, str]],
    fake_rows: list[dict[str, str]],
    *,
    real_manifest_path: Path,
    fake_manifest_path: Path,
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
    fake_train_repeat: int = 1,
    real_train_repeat: int = 1,
    fake_train_operations: set[str] | None = None,
    real_window_seconds: float = 0.0,
    real_window_stride_seconds: float = 4.0,
    real_window_splits: set[str] | None = None,
    real_max_windows_per_file: int = 0,
) -> list[dict[str, str]]:
    groups = sorted(
        {
            slug_from_path(resolve_audio_path(row["audio_path"], real_manifest_path))
            for row in real_rows
            if (row.get("label") or "").strip().lower() == "real"
        }
    )
    split_map = split_items(groups, rng=rng, train_ratio=train_ratio, val_ratio=val_ratio)

    output = []
    for row in real_rows:
        if (row.get("label") or "").strip().lower() != "real":
            continue
        resolved_path = resolve_audio_path(row["audio_path"], real_manifest_path)
        group = slug_from_path(resolved_path)
        split = split_map.get(group, "test")
        item = base_row(
            row,
            manifest_path=real_manifest_path,
            split=split,
            source="coleta_real",
            domain=row.get("domain") or "coleta_celular",
            speaker_id=group,
            manipulation_type="none",
        )
        should_window = (
            real_window_seconds > 0
            and split in (real_window_splits or set())
        )
        if should_window:
            windows = temporal_windows(
                duration_seconds=audio_duration_seconds(resolved_path),
                window_seconds=real_window_seconds,
                stride_seconds=real_window_stride_seconds,
            )
            windows = limited_windows(windows, max_windows=real_max_windows_per_file, rng=rng)
            for window_index, (start, end) in enumerate(windows):
                window_item = dict(item)
                window_item["start"] = f"{start:.6f}"
                window_item["end"] = f"{end:.6f}"
                window_item["speaker_id"] = group
                window_item["manipulation_type"] = "none_windowed_real"
                window_item["source_audio"] = resolved_path
                output.append(window_item)
        else:
            repeats = max(real_train_repeat, 1) if split == "train" else 1
            output.extend(dict(item) for _ in range(repeats))

    for row in fake_rows:
        if (row.get("label") or "").strip().lower() != "fake":
            continue
        operation = row.get("manipulation_type") or "unknown"
        source_audio = row.get("source_audio") or row.get("audio_path") or ""
        group = slug_from_path(source_audio)
        split = split_map.get(group, "test")
        if split == "train" and fake_train_operations and operation not in fake_train_operations:
            continue
        item = base_row(
            row,
            manifest_path=fake_manifest_path,
            split=split,
            source="coleta_manipulated_strong",
            domain="coleta_celular_manipulado",
            start=row.get("manipulated_start") or "",
            end=row.get("manipulated_end") or "",
            file_label="fake",
            manipulation_type=operation,
            speaker_id=group,
        )
        repeats = max(fake_train_repeat, 1) if split == "train" else 1
        output.extend(dict(item) for _ in range(repeats))
    return output


def validate_rows(rows: list[dict[str, str]]) -> None:
    missing = [row["audio_path"] for row in rows if not Path(row["audio_path"]).exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} arquivo(s) de audio nao encontrados:\n{preview}")

    counts = Counter((row["split"], row["label"]) for row in rows)
    for split in ("train", "val"):
        if counts.get((split, "real"), 0) == 0 or counts.get((split, "fake"), 0) == 0:
            raise RuntimeError(f"Split {split} precisa ter amostras real e fake. Contagem: {dict(counts)}")


def print_counts(rows: list[dict[str, str]]) -> None:
    print("contagem por split/label")
    for key, count in sorted(Counter((row["split"], row["label"]) for row in rows).items()):
        print(f"  {key[0]:5s} {key[1]:4s} {count}")
    print("contagem por split/source/label")
    for key, count in sorted(Counter((row["split"], row["source"], row["label"]) for row in rows).items()):
        print(f"  {key[0]:5s} {key[1]:30s} {key[2]:4s} {count}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    brspeech_path = resolve_path(args.brspeech_manifest)
    fake_braccent_path = resolve_path(args.fake_braccent_manifest)
    coraa_path = resolve_path(args.coraa_manifest)
    coleta_real_path = resolve_path(args.coleta_real_manifest)
    coleta_fake_path = resolve_path(args.coleta_fake_manifest)
    output_path = resolve_path(args.output)
    fake_train_operations = {
        item.strip()
        for item in args.coleta_fake_train_operations.split(",")
        if item.strip()
    } or None
    real_window_splits = {
        item.strip()
        for item in args.coleta_real_window_splits.split(",")
        if item.strip()
    }

    rows: list[dict[str, str]] = []
    rows.extend(
        sample_by_split_label(
            read_csv(brspeech_path),
            manifest_path=brspeech_path,
            rng=rng,
            caps={
                ("train", "real"): args.brspeech_train_per_label,
                ("train", "fake"): args.brspeech_train_per_label,
                ("val", "real"): args.brspeech_val_per_label,
                ("val", "fake"): args.brspeech_val_per_label,
                ("test", "real"): args.brspeech_test_per_label,
                ("test", "fake"): args.brspeech_test_per_label,
            },
            source="brspeech_df",
            domain="tts_formal_limpo",
        )
    )
    rows.extend(
        sample_by_split_label(
            read_csv(fake_braccent_path),
            manifest_path=fake_braccent_path,
            rng=rng,
            caps={
                ("train", "real"): args.fake_braccent_train_per_label,
                ("train", "fake"): args.fake_braccent_train_per_label,
                ("val", "real"): args.fake_braccent_val_per_label,
                ("val", "fake"): args.fake_braccent_val_per_label,
                ("test", "real"): args.fake_braccent_test_per_label,
                ("test", "fake"): args.fake_braccent_test_per_label,
            },
            source="fake_braccent_d_codec_equalized",
            domain="braccent_codec_equalizado",
        )
    )
    rows.extend(
        build_coraa_rows(
            read_csv(coraa_path),
            manifest_path=coraa_path,
            rng=rng,
            train_ratio=args.coraa_train_ratio,
            val_ratio=args.coraa_val_ratio,
        )
    )
    rows.extend(
        build_coleta_rows(
            read_csv(coleta_real_path),
            read_csv(coleta_fake_path),
            real_manifest_path=coleta_real_path,
            fake_manifest_path=coleta_fake_path,
            rng=rng,
            train_ratio=args.coleta_train_ratio,
            val_ratio=args.coleta_val_ratio,
            fake_train_repeat=args.coleta_fake_train_repeat,
            real_train_repeat=args.coleta_real_train_repeat,
            fake_train_operations=fake_train_operations,
            real_window_seconds=args.coleta_real_window_seconds,
            real_window_stride_seconds=args.coleta_real_window_stride_seconds,
            real_window_splits=real_window_splits,
            real_max_windows_per_file=args.coleta_real_max_windows_per_file,
        )
    )

    rows.sort(key=lambda row: (row["split"], row["source"], row["label"], row["audio_path"]))
    validate_rows(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest = {output_path}")
    print(f"rows = {len(rows)}")
    print_counts(rows)


if __name__ == "__main__":
    main()
