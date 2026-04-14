#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
import signal
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DATASETS = {
    "140k": {
        "slug": "xhlulu/140k-real-and-fake-faces",
        "source_prefix": "kaggle_140k",
        "labels": ("real", "fake"),
    },
    "manjilkarki": {
        "slug": "manjilkarki/deepfake-and-real-images",
        "source_prefix": "kaggle_manjilkarki",
        "labels": ("real", "fake"),
    },
    "deepfake_faces": {
        "slug": "dagnelies/deepfake-faces",
        "source_prefix": "kaggle_deepfake_faces",
        "labels": ("real", "fake"),
    },
    "celeba": {
        "slug": "jessicali9530/celeba-dataset",
        "source_prefix": "kaggle_celeba",
        "labels": ("real",),
    },
}

REAL_TOKENS = {
    "real",
    "reals",
    "real_images",
    "realimages",
    "original",
    "originals",
    "authentic",
    "genuine",
}
FAKE_TOKENS = {
    "fake",
    "fakes",
    "fake_images",
    "fakeimages",
    "deepfake",
    "deepfakes",
    "manipulated",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstrói fontes locais a partir de datasets do Kaggle. "
            "Baixa, extrai, separa real/fake e gera manifesto para montar train_pool."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS.keys()),
        help="Dataset a reconstruir. Pode repetir. Padrao: 140k + manjilkarki.",
    )
    parser.add_argument(
        "--download-root",
        default="data/downloads/kaggle",
        help="Diretorio para zips e extrações baixadas do Kaggle.",
    )
    parser.add_argument(
        "--materialized-root",
        default="data/raw/rebuilt_kaggle",
        help="Diretorio com fontes real/fake reconstruidas para uso no train_pool.",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/kaggle_rebuild_manifest.csv",
        help="Manifesto CSV de saida.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink economiza disco; copy duplica arquivos.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Limite por label em cada dataset. Em --sample-download, define quantos arquivos baixar por label.",
    )
    parser.add_argument(
        "--sample-download",
        action="store_true",
        help="Lista os arquivos no Kaggle e baixa apenas uma amostra por label, sem baixar o zip completo.",
    )
    parser.add_argument(
        "--sample-label",
        action="append",
        choices=("real", "fake"),
        help=(
            "Label a baixar em --sample-download. Pode repetir. "
            "Padrao: todos os labels do dataset. Use para evitar varrer muitas paginas."
        ),
    )
    parser.add_argument(
        "--sample-download-root",
        default="data/downloads/kaggle_samples",
        help="Diretorio para arquivos individuais baixados por --sample-download.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed da amostragem em --sample-download.",
    )
    parser.add_argument(
        "--kaggle-retries",
        type=int,
        default=5,
        help="Tentativas para chamadas Kaggle quando houver erro temporario/rate limit.",
    )
    parser.add_argument(
        "--kaggle-retry-wait",
        type=int,
        default=30,
        help="Espera inicial em segundos entre tentativas Kaggle.",
    )
    parser.add_argument(
        "--kaggle-timeout",
        type=int,
        default=120,
        help="Timeout em segundos para cada chamada Kaggle.",
    )
    parser.add_argument(
        "--kaggle-page-size",
        type=int,
        default=1000,
        help="Quantidade de arquivos por pagina ao listar amostras no Kaggle.",
    )
    parser.add_argument(
        "--kaggle-max-pages",
        type=int,
        default=10,
        help="Maximo de paginas a listar por dataset em --sample-download.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Nao baixa; apenas processa extrações existentes em --download-root.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Baixa novamente mesmo se ja houver zip/extracao.",
    )
    parser.add_argument(
        "--allow-full-download",
        action="store_true",
        help=(
            "Permite baixar o arquivo completo do dataset no Kaggle. "
            "Use apenas quando quiser mesmo baixar tudo; o Kaggle geralmente nao baixa amostras parciais."
        ),
    )
    parser.add_argument(
        "--clean-materialized",
        action="store_true",
        help="Remove apenas --materialized-root antes de recriar as fontes.",
    )
    parser.add_argument(
        "--kaggle-json",
        default=None,
        help="Caminho para kaggle.json. O arquivo sera copiado para .kaggle/kaggle.json dentro do projeto.",
    )
    parser.add_argument(
        "--kaggle-config-dir",
        default=".kaggle",
        help="Diretorio de config Kaggle usado pelo script. Padrao fica dentro do projeto.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def configure_kaggle(args) -> Path:
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


def run_kaggle_download(dataset_key: str, dataset_spec: dict, download_dir: Path, env: dict, force: bool) -> Path:
    zip_path = download_dir / f"{dataset_spec['slug'].split('/')[-1]}.zip"
    extract_dir = download_dir / dataset_key

    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        print(f"{dataset_key}: extracao existente em {extract_dir}")
        return extract_dir

    if not zip_path.exists() or force:
        download_dir.mkdir(parents=True, exist_ok=True)
        kaggle_bin = Path(sys.executable).with_name("kaggle")
        kaggle_command = str(kaggle_bin) if kaggle_bin.exists() else "kaggle"
        command = [
            kaggle_command,
            "datasets",
            "download",
            "-d",
            dataset_spec["slug"],
            "-p",
            str(download_dir),
        ]
        print(f"{dataset_key}: baixando {dataset_spec['slug']}")
        subprocess.run(command, check=True, env=env, cwd=REPO_ROOT)

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"{dataset_key}: extraindo {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def get_kaggle_api(retries: int, wait_seconds: int, timeout_seconds: int):
    def create_and_authenticate_api():
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api

    return with_kaggle_retry(
        "autenticando Kaggle",
        retries,
        wait_seconds,
        timeout_seconds,
        create_and_authenticate_api,
    )


class KaggleCallTimeout(TimeoutError):
    pass


def _timeout_handler(_signum, _frame):
    raise KaggleCallTimeout("tempo limite atingido")


def with_timeout(timeout_seconds: int, func):
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def with_kaggle_retry(description: str, retries: int, wait_seconds: int, timeout_seconds: int, func):
    for attempt in range(1, retries + 1):
        try:
            return with_timeout(timeout_seconds, func)
        except Exception as exc:
            if attempt >= retries:
                raise
            sleep_for = wait_seconds * attempt
            print(f"{description}: tentativa {attempt}/{retries} falhou ({type(exc).__name__}: {exc}). Aguardando {sleep_for}s...")
            time.sleep(sleep_for)


def sample_labels_complete(grouped, labels, max_per_label: int) -> bool:
    return all(len(grouped[label]) >= max_per_label for label in labels)


def list_kaggle_sample_files(
    api,
    dataset_slug: str,
    dataset_spec: dict,
    max_per_label: int,
    requested_labels,
    retries: int,
    wait_seconds: int,
    timeout_seconds: int,
    page_size: int,
    max_pages: int,
):
    labels = tuple(label for label in (requested_labels or dataset_spec["labels"]) if label in dataset_spec["labels"])
    if not labels:
        raise ValueError(f"Nenhum label valido para {dataset_slug}. Labels do dataset: {dataset_spec['labels']}")

    grouped = {label: [] for label in labels}
    unknown = 0
    other_label_skipped = 0
    unknown_examples = []
    page_count = 0
    page_token = None

    while True:
        page_count += 1
        response = with_kaggle_retry(
            f"listando pagina {page_count} de {dataset_slug}",
            retries,
            wait_seconds,
            timeout_seconds,
            lambda: api.dataset_list_files(dataset_slug, page_token=page_token, page_size=page_size),
        )

        for file_info in response.files or []:
            name = getattr(file_info, "name", "")
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = infer_label(Path(name), dataset_spec)
            if label in grouped and len(grouped[label]) < max_per_label:
                grouped[label].append(name)
            elif label in dataset_spec["labels"]:
                other_label_skipped += 1
            else:
                unknown += 1
                if label is None and len(unknown_examples) < 5:
                    unknown_examples.append(name)

        summary = " ".join(f"{label}={len(items)}/{max_per_label}" for label, items in grouped.items())
        print(f"{dataset_slug}: pagina {page_count} processada ({summary})")

        if sample_labels_complete(grouped, labels, max_per_label):
            break

        if page_count >= max_pages:
            print(
                f"{dataset_slug}: limite de paginas atingido ({max_pages}). "
                "Use --sample-label para baixar uma classe especifica ou aumente --kaggle-max-pages."
            )
            break

        page_token = getattr(response, "next_page_token", None)
        if not page_token:
            break

    selected = []
    for label, names in grouped.items():
        selected.extend((label, name) for name in names)

    print(
        f"{dataset_slug}: amostra selecionada "
        + " ".join(f"{label}={len(items)}" for label, items in grouped.items())
        + f" other_label_skipped={other_label_skipped} unknown_skipped={unknown}"
    )
    if unknown_examples:
        print(f"{dataset_slug}: exemplos sem label reconhecido: {', '.join(unknown_examples)}")
    return selected


def download_kaggle_sample(
    dataset_key: str,
    dataset_spec: dict,
    sample_download_root: Path,
    materialized_root: Path,
    mode: str,
    max_per_label: int,
    requested_labels,
    seed: int,
    retries: int,
    wait_seconds: int,
    timeout_seconds: int,
    page_size: int,
    max_pages: int,
):
    api = get_kaggle_api(retries, wait_seconds, timeout_seconds)
    dataset_slug = dataset_spec["slug"]
    selected = list_kaggle_sample_files(
        api=api,
        dataset_slug=dataset_slug,
        dataset_spec=dataset_spec,
        max_per_label=max_per_label,
        requested_labels=requested_labels,
        retries=retries,
        wait_seconds=wait_seconds,
        timeout_seconds=timeout_seconds,
        page_size=page_size,
        max_pages=max_pages,
    )
    random.Random(f"{seed}:{dataset_key}:download-order").shuffle(selected)
    download_dir = sample_download_root / dataset_key
    download_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counters = {"real": 0, "fake": 0}
    for label, file_name in selected:
        counters[label] += 1
        print(f"{dataset_key}: baixando {label} {counters[label]}/{max_per_label}: {file_name}")
        with_kaggle_retry(
            f"baixando {file_name}",
            retries,
            wait_seconds,
            timeout_seconds,
            lambda: api.dataset_download_file(
                dataset_slug,
                file_name,
                path=str(download_dir),
                force=True,
                quiet=True,
            ),
        )

        downloaded_path = download_dir / Path(file_name).name
        if not downloaded_path.exists():
            matches = sorted(download_dir.glob(Path(file_name).name))
            if not matches:
                raise FileNotFoundError(f"Arquivo baixado nao encontrado para {file_name} em {download_dir}")
            downloaded_path = matches[0]

        source_name = f"{dataset_spec['source_prefix']}_{label}"
        target_path = materialized_root / source_name / f"{counters[label]:08d}{downloaded_path.suffix.lower()}"
        materialize(downloaded_path, target_path, mode)
        rows.append(
            {
                "dataset": dataset_key,
                "source": source_name,
                "label": label,
                "original_path": file_name,
                "rebuilt_path": str(target_path.resolve()),
            }
        )

    print(f"{dataset_key}: amostra baixada real={counters['real']} fake={counters['fake']}")
    return rows


def infer_label(path: Path, dataset_spec: dict) -> str | None:
    labels = dataset_spec["labels"]
    if labels == ("real",):
        return "real"

    parts = [part.lower().replace("-", "_").replace(" ", "_") for part in path.parts]
    for part in reversed(parts):
        if part in FAKE_TOKENS:
            return "fake"
        if part in REAL_TOKENS:
            return "real"

        tokens = {token for token in re.split(r"[^a-z0-9]+", Path(part).stem.lower()) if token}
        if tokens & FAKE_TOKENS:
            return "fake"
        if tokens & REAL_TOKENS:
            return "real"
    return None


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def materialize(source_path: Path, target_path: Path, mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    if mode == "hardlink":
        target_path.hardlink_to(source_path.resolve())
    else:
        shutil.copy2(source_path, target_path)


def process_dataset(dataset_key: str, dataset_spec: dict, extract_dir: Path, materialized_root: Path, mode: str, max_per_label: int | None):
    rows = []
    counters = {"real": 0, "fake": 0, "unknown": 0}
    source_prefix = dataset_spec["source_prefix"]

    for source_path in iter_images(extract_dir):
        label = infer_label(source_path, dataset_spec)
        if label not in {"real", "fake"}:
            counters["unknown"] += 1
            continue

        if max_per_label is not None and counters[label] >= max_per_label:
            continue

        counters[label] += 1
        source_name = f"{source_prefix}_{label}"
        target_path = materialized_root / source_name / f"{counters[label]:08d}{source_path.suffix.lower()}"
        materialize(source_path, target_path, mode)
        rows.append(
            {
                "dataset": dataset_key,
                "source": source_name,
                "label": label,
                "original_path": str(source_path.resolve()),
                "rebuilt_path": str(target_path.resolve()),
            }
        )

    print(
        f"{dataset_key}: real={counters['real']} fake={counters['fake']} "
        f"unknown_skipped={counters['unknown']}"
    )
    return rows


def write_manifest(rows, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "source", "label", "original_path", "rebuilt_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifesto gerado: {manifest_path}")


def print_next_steps(materialized_root: Path) -> None:
    sources = sorted(path for path in materialized_root.iterdir() if path.is_dir()) if materialized_root.exists() else []
    real_sources = [path for path in sources if path.name.endswith("_real")]
    fake_sources = [path for path in sources if path.name.endswith("_fake")]

    print("\nFontes reconstruidas:")
    for path in sources:
        print(f"  {path}")

    print("\nExemplo para montar train_pool:")
    command = [
        "python scripts/build_training_pool.py",
        "  --output-root data/source/train_pool_rebuilt",
    ]
    for path in real_sources:
        command.append(f"  --real-source {path.name}={path}")
    for path in fake_sources:
        command.append(f"  --fake-source {path.name}={path}")
    command.append("  --mode hardlink")
    command.append("  --clean")
    print(" \\\n".join(command))


def main():
    args = parse_args()
    dataset_keys = args.dataset or ["140k", "manjilkarki"]
    download_root = resolve_repo_path(args.download_root)
    sample_download_root = resolve_repo_path(args.sample_download_root)
    materialized_root = resolve_repo_path(args.materialized_root)
    manifest_path = resolve_repo_path(args.manifest)
    kaggle_config_dir = configure_kaggle(args)

    if args.clean_materialized and materialized_root.exists():
        shutil.rmtree(materialized_root)

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)

    if args.sample_download and args.max_per_label is None:
        raise RuntimeError("Use --max-per-label com --sample-download para definir o tamanho da amostra.")

    if not args.skip_download and not args.allow_full_download and not args.sample_download:
        raise RuntimeError(
            "Download bloqueado: o Kaggle normalmente baixa o dataset inteiro. "
            "Para trabalhar so com amostras, use --sample-download, --skip-download em uma extracao local existente "
            "ou gere uma amostra local com scripts/build_dataset_sample.py. "
            "Se quiser baixar tudo mesmo assim, rode com --allow-full-download."
        )

    all_rows = []
    for dataset_key in dataset_keys:
        spec = DATASETS[dataset_key]
        extract_dir = download_root / dataset_key

        if args.sample_download:
            all_rows.extend(
                download_kaggle_sample(
                    dataset_key=dataset_key,
                    dataset_spec=spec,
                    sample_download_root=sample_download_root,
                    materialized_root=materialized_root,
                    mode=args.mode,
                    max_per_label=args.max_per_label,
                    requested_labels=args.sample_label,
                    seed=args.seed,
                    retries=args.kaggle_retries,
                    wait_seconds=args.kaggle_retry_wait,
                    timeout_seconds=args.kaggle_timeout,
                    page_size=args.kaggle_page_size,
                    max_pages=args.kaggle_max_pages,
                )
            )
            continue

        if not args.skip_download:
            extract_dir = run_kaggle_download(dataset_key, spec, download_root, env, args.force_download)
        elif not extract_dir.exists():
            raise FileNotFoundError(f"Extracao ausente para {dataset_key}: {extract_dir}")

        all_rows.extend(
            process_dataset(
                dataset_key=dataset_key,
                dataset_spec=spec,
                extract_dir=extract_dir,
                materialized_root=materialized_root,
                mode=args.mode,
                max_per_label=args.max_per_label,
            )
        )

    write_manifest(all_rows, manifest_path)
    print_next_steps(materialized_root)


if __name__ == "__main__":
    main()
