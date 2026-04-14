# Reconstrucao dos datasets Kaggle

Este fluxo reconstrói as fontes brutas perdidas em `data/raw/rebuilt_kaggle/` sem apagar os datasets prontos que ja existem, como `data/dataset_colab` e `data/dataset_colab_v8`.

## 1. Trabalhar apenas com amostra local

Se a ideia e nao baixar tudo agora, use o dataset local que ainda existe e gere uma amostra:

```bash
python scripts/build_dataset_sample.py \
  --input-root data/dataset_colab_v8 \
  --output-root data/dataset_colab_sample \
  --max-per-class 1000 \
  --mode hardlink \
  --clean
```

Isso preserva `data/dataset_colab_v8` e cria uma base menor em `data/dataset_colab_sample`.

## 2. Baixar amostras direto do Kaggle

Com `kaggle.json` configurado, baixe apenas arquivos individuais:

```bash
worker/venv/bin/python scripts/rebuild_kaggle_datasets.py \
  --kaggle-json kaggle.json \
  --dataset 140k \
  --dataset manjilkarki \
  --sample-download \
  --max-per-label 1000 \
  --kaggle-retries 3 \
  --kaggle-retry-wait 60 \
  --clean-materialized
```

Isso lista os arquivos do dataset, escolhe ate 1000 reais e 1000 fakes por dataset e baixa apenas esses arquivos para:

```text
data/downloads/kaggle_samples/
data/raw/rebuilt_kaggle/
```

No dataset `xhlulu/140k-real-and-fake-faces`, a listagem da API do Kaggle pode vir com muitas imagens `fake` antes das `real`. Para nao gastar varias paginas procurando a outra classe e bater em `429 Too Many Requests`, baixe uma classe por vez:

```bash
worker/venv/bin/python scripts/rebuild_kaggle_datasets.py \
  --kaggle-json kaggle.json \
  --dataset 140k \
  --sample-download \
  --sample-label fake \
  --max-per-label 50 \
  --clean-materialized \
  --kaggle-retries 3 \
  --kaggle-retry-wait 60 \
  --kaggle-page-size 1000 \
  --kaggle-max-pages 2
```

Para buscar `real` no `140k`, troque para `--sample-label real` e aumente `--kaggle-max-pages` com calma. Se a API continuar limitando, use outro dataset como fonte de `real` nessa rodada, por exemplo `--dataset celeba`, ou processe uma amostra local ja extraida. O `manjilkarki` tambem pode vir com muitas paginas de `fake` antes das `real`, entao use o mesmo cuidado com `--kaggle-max-pages`.

Use `--clean-materialized` apenas no primeiro comando da rodada. Nos proximos downloads separados por label, remova esse flag para nao apagar a amostra que ja foi materializada.

## 3. Processar amostra a partir de uma extracao Kaggle local

Se voce ja tiver uma extracao local do Kaggle em `data/downloads/kaggle/`, processe so uma amostra:

```bash
worker/venv/bin/python scripts/rebuild_kaggle_datasets.py \
  --dataset 140k \
  --dataset manjilkarki \
  --skip-download \
  --max-per-label 1000 \
  --clean-materialized
```

Importante: `--max-per-label` limita a quantidade copiada/linkada para `data/raw/rebuilt_kaggle`, mas nao faz o Kaggle baixar parcialmente.

## 4. Preparar credencial Kaggle

Baixe o `kaggle.json` na sua conta Kaggle. O token pode ser copiado para `.kaggle/kaggle.json` dentro do projeto assim:

```bash
worker/venv/bin/python scripts/rebuild_kaggle_datasets.py \
  --kaggle-json /caminho/para/kaggle.json \
  --skip-download \
  --max-per-label 100
```

Esse comando nao baixa nada por causa do `--skip-download`; ele apenas deixa a credencial no lugar e tenta processar extracoes locais, se existirem.

## 5. Reconstruir completo, somente se for intencional

Para baixar datasets completos do Kaggle, e necessario passar `--allow-full-download` de proposito:

```bash
worker/venv/bin/python scripts/rebuild_kaggle_datasets.py \
  --dataset 140k \
  --dataset manjilkarki \
  --allow-full-download \
  --clean-materialized
```

Datasets padrao:

- `xhlulu/140k-real-and-fake-faces`
- `manjilkarki/deepfake-and-real-images`

Saidas principais:

```text
data/downloads/kaggle/
data/raw/rebuilt_kaggle/
data/manifests/kaggle_rebuild_manifest.csv
```

## 6. Adicionar fontes extras do Kaggle

Tambem ha suporte para:

```bash
worker/venv/bin/python scripts/rebuild_kaggle_datasets.py \
  --dataset deepfake_faces \
  --dataset celeba
```

Use `celeba` como fonte real e `deepfake_faces` como fonte fake/real quando a estrutura baixada permitir inferir o label por pasta.

## 7. Montar novo train_pool

Depois de reconstruir as fontes, o proprio script imprime um comando pronto para `scripts/build_training_pool.py`.

Exemplo:

```bash
python scripts/build_training_pool.py \
  --output-root data/source/train_pool_rebuilt \
  --real-source kaggle_140k_real=data/raw/rebuilt_kaggle/kaggle_140k_real \
  --real-source kaggle_manjilkarki_real=data/raw/rebuilt_kaggle/kaggle_manjilkarki_real \
  --fake-source kaggle_140k_fake=data/raw/rebuilt_kaggle/kaggle_140k_fake \
  --fake-source kaggle_manjilkarki_fake=data/raw/rebuilt_kaggle/kaggle_manjilkarki_fake \
  --mode hardlink \
  --clean
```

## 8. Gerar dataset de treino local

```bash
python scripts/prepare_colab_dataset.py \
  --real-dir data/source/train_pool_rebuilt/real \
  --fake-dir data/source/train_pool_rebuilt/fake \
  --output-root data/dataset_colab_rebuilt \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --balance-to-min-class \
  --mode hardlink \
  --clean
```

Para preservar o que ja existe, use `data/dataset_colab_rebuilt` ou outro nome novo em vez de sobrescrever `data/dataset_colab_v8`.
