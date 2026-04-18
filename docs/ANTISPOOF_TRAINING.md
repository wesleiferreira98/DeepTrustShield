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
LIVENESS_ONNX_ANTISPOOF_REVIEW_THRESHOLD=0.095
LIVENESS_ONNX_ANTISPOOF_FRAME_SPIKE_MARGIN=0.035
LIVENESS_ONNX_QUALITY_OVERRIDE_THRESHOLD=0.20
LIVENESS_MOBILE_ONNX_HARD_THRESHOLD=0.20
LIVENESS_FUSION_FACE_BACKGROUND_DELTA=0.34
LIVENESS_FUSION_INTERNAL_MOTION_MAX=0.06
LIVENESS_ONNX_MIN_FACE_WIDTH_RATIO=0.24
```

`LIVENESS_ONNX_ANTISPOOF_AGGREGATION` aceita `median`, `p75`, `p90` ou `max`.
Com `LIVENESS_ONNX_ANTISPOOF_THRESHOLD=-1.0`, o backend usa o threshold calibrado no metadata.
`LIVENESS_ONNX_ANTISPOOF_REVIEW_THRESHOLD` cria uma zona cinza para inspecionar frames individuais.
`LIVENESS_ONNX_ANTISPOOF_FRAME_SPIKE_MARGIN` exige que um pico individual fique bem acima do limite final para reprovar sozinho nessa zona; dois ou mais frames acima do limite tambem reprovam.
`LIVENESS_ONNX_QUALITY_OVERRIDE_THRESHOLD` permite manter a classificacao de spoof mesmo com qualidade limitada apenas quando o ONNX estiver muito alto; abaixo disso a captura vira qualidade insuficiente.
`LIVENESS_MOBILE_ONNX_HARD_THRESHOLD` deixa a camera do celular mais conservadora: abaixo desse valor, ONNX acima do threshold base vira captura inconclusiva em vez de acusacao de spoof.
`LIVENESS_FUSION_FACE_BACKGROUND_DELTA` e `LIVENESS_FUSION_INTERNAL_MOTION_MAX` combinam evidencias fracas: ONNX em zona cinza, pouco movimento interno e face muito diferente do fundo.
`LIVENESS_ONNX_MIN_FACE_WIDTH_RATIO` evita acusar fake quando a face esta pequena demais para o anti-spoof; nesse caso a captura e bloqueada como qualidade insuficiente.

## Calibrar ONNX Com Capturas Locais

Para medir se o ONNX esta separando bem rosto real e ataque de foto/tela, monte uma pasta assim:

```text
data/calibration/antispoof_local/
  live/
    tentativa_001/
      frame_1.jpg
      frame_2.jpg
  spoof/
    celular_foto_001/
      frame_1.jpg
      frame_2.jpg
```

Cada subpasta vira uma tentativa multi-frame. Arquivos soltos dentro de `live/` ou `spoof/` tambem funcionam como amostras individuais.

```bash
.venv-training/bin/python scripts/calibrate_antispoof_onnx.py \
  --input-root data/calibration/antispoof_local \
  --sample-mode folder \
  --aggregation p75 \
  --target-apcer 0.02
```

Saidas:

```text
data/reports/antispoof_calibration/<timestamp>/antispoof_scores.csv
data/reports/antispoof_calibration/<timestamp>/antispoof_frame_scores.csv
data/reports/antispoof_calibration/<timestamp>/threshold_summary.csv
data/reports/antispoof_calibration/<timestamp>/calibration_report.json
```

Use:

- `best_ACER`: threshold com melhor equilibrio entre aceitar live e bloquear spoof.
- `recommended_security`: threshold mais conservador respeitando o `--target-apcer`.
- `antispoof_frame_scores.csv`: diagnostico por frame, bom para ver se uma captura teve apenas um frame estranho ou se toda a tentativa foi suspeita.
