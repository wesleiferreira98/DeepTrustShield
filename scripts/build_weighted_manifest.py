#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera um manifesto CSV com pesos por amostra a partir do pool_manifest.csv "
            "para treinos com WeightedRandomSampler."
        )
    )
    parser.add_argument(
        "--pool-manifest",
        default="data/source/train_pool/pool_manifest.csv",
        help="CSV de entrada gerado pelo build_training_pool.py",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/train_pool_weighted_manifest_v8.csv",
        help="CSV de saida com pesos por amostra",
    )
    parser.add_argument(
        "--default-real-weight",
        type=float,
        default=1.0,
        help="Peso padrao para fontes reais",
    )
    parser.add_argument(
        "--default-fake-weight",
        type=float,
        default=1.0,
        help="Peso padrao para fontes fake",
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        help="Override por fonte no formato nome=valor. Pode repetir a flag.",
    )
    return parser.parse_args()


def parse_source_weights(raw_values):
    weights = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise ValueError(f"Peso invalido: {raw_value!r}. Use fonte=valor")
        source, raw_weight = raw_value.split("=", 1)
        weights[source.strip()] = float(raw_weight)
    return weights


def infer_group(source: str, label: str) -> str:
    source_lower = source.lower()
    if label == "fake":
        return "fake"
    if "black" in source_lower or "preto" in source_lower:
        return "black"
    if "indigenous" in source_lower or "indigena" in source_lower:
        return "indigenous"
    if "pardo" in source_lower:
        return "pardo"
    if "fairface" in source_lower:
        return "mixed"
    if "capture" in source_lower:
        return "own_capture"
    return "unspecified"


def infer_context_tag(source: str, original_path: str) -> str:
    haystack = f"{source} {original_path}".lower()
    if "paint" in haystack or "pintura" in haystack:
        return "face_paint"
    if "screenshot" in haystack or "captura de tela" in haystack:
        return "screenshot"
    return "default"


def main():
    args = parse_args()
    pool_manifest = Path(args.pool_manifest).resolve()
    output_path = Path(args.output).resolve()
    source_weights = parse_source_weights(args.source_weight)

    if not pool_manifest.exists():
        raise FileNotFoundError(f"Manifesto do train_pool nao encontrado: {pool_manifest}")

    rows = []
    with pool_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"label", "source", "original_path", "pool_path"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"pool_manifest sem colunas obrigatorias: {', '.join(sorted(missing))}")

        for raw_row in reader:
            label = raw_row["label"].strip().lower()
            source = raw_row["source"].strip()
            if label not in {"real", "fake"}:
                raise ValueError(f"label invalido no pool_manifest: {label!r}")

            base_weight = args.default_real_weight if label == "real" else args.default_fake_weight
            sample_weight = source_weights.get(source, base_weight)
            original_path = raw_row["original_path"].strip()
            pool_path = raw_row["pool_path"].strip()

            rows.append(
                {
                    "image_path": pool_path,
                    "label": label,
                    "source": source,
                    "group": infer_group(source, label),
                    "context_tag": infer_context_tag(source, original_path),
                    "sample_weight": f"{sample_weight:.6f}",
                    "original_path": original_path,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "label",
                "source",
                "group",
                "context_tag",
                "sample_weight",
                "original_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["label"], row["source"]) for row in rows)
    weights_by_source = defaultdict(set)
    for row in rows:
        weights_by_source[(row["label"], row["source"])].add(row["sample_weight"])

    print(f"manifesto ponderado salvo em: {output_path}")
    print(f"total de amostras: {len(rows)}")
    for (label, source), count in sorted(counts.items()):
        weights = ", ".join(sorted(weights_by_source[(label, source)]))
        print(f"{label} / {source}: {count} imagens | peso={weights}")


if __name__ == "__main__":
    main()
