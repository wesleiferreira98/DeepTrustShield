# Anti-Spoof Training

Este fluxo monta uma base inicial para treinar um detector separado de prova de vida:

- `live`: pessoa real na camera
- `spoof`: foto, tela, replay ou outro ataque de apresentacao

## Baixar e Processar SiW

Com `kaggle.json` na raiz:

```bash
worker/venv/bin/python scripts/rebuild_antispoof_datasets.py \
  --kaggle-json kaggle.json \
  --dataset siw \
  --frames-per-video 10 \
  --max-items-per-label 1000 \
  --clean
```

Saidas:

```text
data/antispoof/processed/siw/
data/manifests/antispoof_manifest.csv
```

O manifesto gerado tem colunas para split, label, origem, sessao e tipo de ataque:

```csv
path,label,label_id,source,subject_id,session_id,attack_type,split,source_media,frame_index,face_detected
```

## Processar Uma Pasta Local

```bash
worker/venv/bin/python scripts/rebuild_antispoof_datasets.py \
  --input-root data/downloads/antispoof/siw/extracted \
  --dataset siw \
  --frames-per-video 10 \
  --max-items-per-label 1000 \
  --clean
```

## Notas

- O split e feito por grupo de arquivo/sessao para reduzir vazamento entre treino e validacao.
- O script tenta inferir `live` ou `spoof` pelos nomes de pastas/arquivos.
- Se muitos frames forem ignorados por falta de face, rode com `--keep-no-face` para usar crop central como fallback.
- Para um modelo de producao, combine mais de uma base publica e valide em datasets/pessoas/dispositivos fora do treino.

## Folds Estratificados

Para treinar com validacao cruzada sem misturar o mesmo sujeito entre treino e validacao:

```bash
worker/venv/bin/python scripts/build_antispoof_folds.py \
  --manifest data/manifests/antispoof_manifest.csv \
  --output data/manifests/antispoof_manifest_folds.csv \
  --n-splits 5 \
  --group-column subject_id
```

Use `fold == k` como validacao e os demais folds como treino. Isso e melhor que `StratifiedKFold`
simples porque reduz vazamento entre imagens do mesmo sujeito.

## Treinar MobileNetV3 Small

Abra no VS Code:

```text
notebooks/train_antispoof_mobilenet_v1.ipynb
```

Use o kernel `DeepTrustShield Training (.venv-training)`. O notebook le:

```text
data/manifests/antispoof_manifest_folds.csv
```

e salva:

```text
model/weights/antispoof_mobilenet_v1/antispoof_mobilenet_v1_best.pth
model/weights/antispoof_mobilenet_v1/antispoof_mobilenet_v1.onnx
model/weights/antispoof_mobilenet_v1/antispoof_mobilenet_v1_metadata.json
```

Para rodar todos os folds sem ficar alterando `VAL_FOLD` no notebook:

```bash
.venv-training/bin/python scripts/train_antispoof_mobilenet_cv.py \
  --manifest data/manifests/antispoof_manifest_folds.csv \
  --fold 0 \
  --fold 1 \
  --fold 2 \
  --fold 3 \
  --fold 4 \
  --epochs 8
```

O resumo final fica em:

```text
model/weights/antispoof_mobilenet_v1_cv/cv_summary.csv
```

## Integracao No Backend

O backend usa por padrao o ONNX do fold 3:

```text
model/weights/antispoof_mobilenet_v1_cv/fold_3/antispoof_mobilenet_v1.onnx
```

Configuracoes opcionais no `.env`:

```env
LIVENESS_ONNX_ANTISPOOF_ENABLED=true
LIVENESS_ONNX_ANTISPOOF_AGGREGATION=p75
LIVENESS_ONNX_ANTISPOOF_THRESHOLD=-1.0
```

`LIVENESS_ONNX_ANTISPOOF_AGGREGATION` aceita `median`, `p75`, `p90` ou `max`.
Com `LIVENESS_ONNX_ANTISPOOF_THRESHOLD=-1.0`, o backend usa o threshold calibrado no metadata.
