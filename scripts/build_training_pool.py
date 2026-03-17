#!/usr/bin/env python3
import argparse
import csv
import os
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Monta um train_pool balanceado a partir de varias fontes locais. "
            "Aceita repeticao de --real-source e --fake-source no formato nome=caminho[:limite]."
        )
    )
    parser.add_argument("--output-root", required=True, help="Diretorio de saida do train_pool")
    parser.add_argument(
        "--real-source",
        action="append",
        default=[],
        help="Fonte real no formato nome=caminho[:limite]",
    )
    parser.add_argument(
        "--fake-source",
        action="append",
        default=[],
        help="Fonte fake no formato nome=caminho[:limite]",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para amostragem")
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="copy duplica arquivos; hardlink economiza espaco local",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove a saida anterior antes de gerar o novo pool",
    )
    return parser.parse_args()


def parse_source_spec(spec: str):
    if "=" not in spec:
        raise ValueError(f"Fonte invalida: {spec!r}. Use nome=caminho[:limite]")

    name, remainder = spec.split("=", 1)
    if ":" in remainder:
        path_text, limit_text = remainder.rsplit(":", 1)
        limit = int(limit_text)
    else:
        path_text = remainder
        limit = None

    name = name.strip()
    path = Path(path_text).resolve()
    if not name:
        raise ValueError(f"Nome de fonte invalido: {spec!r}")
    return {"name": name, "path": path, "limit": limit}


def collect_images(root: Path):
    files = []
    for current_root, _, filenames in os.walk(root, followlinks=True):
        current_root_path = Path(current_root)
        for filename in filenames:
            path = current_root_path / filename
            if path.suffix.lower() in IMAGE_SUFFIXES:
                files.append(path)
    return sorted(files)


def prepare_output_root(output_root: Path, clean: bool):
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def materialize_sampled_files(class_name: str, source_name: str, sampled_files, output_root: Path, mode: str, manifest_rows):
    destination_dir = output_root / class_name / source_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(sampled_files, start=1):
        target_path = destination_dir / f"{index:06d}{source_path.suffix.lower()}"
        if mode == "hardlink":
            if target_path.exists():
                target_path.unlink()
            target_path.hardlink_to(source_path.resolve())
        else:
            shutil.copy2(source_path, target_path)

        manifest_rows.append(
            {
                "label": class_name,
                "source": source_name,
                "original_path": str(source_path.resolve()),
                "pool_path": str(target_path.resolve()),
            }
        )


def sample_source(source_spec, seed):
    if not source_spec["path"].exists():
        raise FileNotFoundError(f"Diretorio de fonte nao encontrado: {source_spec['path']}")

    files = collect_images(source_spec["path"])
    rnd = random.Random(f"{seed}:{source_spec['name']}")
    rnd.shuffle(files)

    if source_spec["limit"] is not None:
        files = files[:source_spec["limit"]]

    return files


def write_manifest(manifest_rows, output_root: Path):
    manifest_path = output_root / "pool_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "source", "original_path", "pool_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def process_sources(source_specs, class_name: str, output_root: Path, seed: int, mode: str, manifest_rows):
    summary_rows = []
    for source_spec in source_specs:
        sampled_files = sample_source(source_spec, seed)
        materialize_sampled_files(
            class_name=class_name,
            source_name=source_spec["name"],
            sampled_files=sampled_files,
            output_root=output_root,
            mode=mode,
            manifest_rows=manifest_rows,
        )
        summary_rows.append(
            {
                "label": class_name,
                "source": source_spec["name"],
                "count": len(sampled_files),
                "path": str(source_spec["path"]),
            }
        )
    return summary_rows


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    prepare_output_root(output_root, clean=args.clean)

    real_sources = [parse_source_spec(spec) for spec in args.real_source]
    fake_sources = [parse_source_spec(spec) for spec in args.fake_source]

    if not real_sources:
        raise ValueError("Informe ao menos uma --real-source")
    if not fake_sources:
        raise ValueError("Informe ao menos uma --fake-source")

    manifest_rows = []
    summary_rows = []
    summary_rows.extend(process_sources(real_sources, "real", output_root, args.seed, args.mode, manifest_rows))
    summary_rows.extend(process_sources(fake_sources, "fake", output_root, args.seed, args.mode, manifest_rows))
    manifest_path = write_manifest(manifest_rows, output_root)

    print(f"train_pool gerado em: {output_root}")
    print(f"manifesto: {manifest_path}")
    for item in summary_rows:
        print(f"{item['label']} / {item['source']}: {item['count']} imagens")


if __name__ == "__main__":
    main()
