# Auditoria de Vies e Fairness

## Objetivo

Medir se o modelo esta errando de forma desigual entre grupos antes de trocar dataset, recalibrar threshold ou publicar uma versao nova.

O foco aqui nao e discutir identidade racial por inferencia automatica. O foco e avaliar grupos ja rotulados no seu conjunto de validacao.

## Regra de Operacao

Cada rodada de treinamento deve passar por uma auditoria separando ao menos:

- `group`: grupo do dataset que voce quer comparar
- `label`: `real` ou `fake`
- `image_path`: caminho da amostra

O modelo so deve ser comparado com outro depois de olhar:

- `mean_p_fake_real` por grupo
- `mean_p_fake_fake` por grupo
- `fpr` por grupo
- `fnr` por grupo
- `fpr_gap`
- `fnr_gap`

Se o score medio das imagens reais sobe de forma consistente em um grupo, o modelo esta aprendendo um atalho ruim.

## Tamanho Minimo Recomendado Do Holdout

Minimo operacional para uma primeira auditoria:

- `50` imagens `real` por grupo
- `50` imagens `fake` por grupo

Melhor ponto de partida:

- `100` imagens `real` por grupo
- `100` imagens `fake` por grupo

Ideal para comparar modelos com mais confianca:

- `200+` imagens `real` por grupo
- `200+` imagens `fake` por grupo

Se um grupo tiver poucas amostras, a auditoria ainda pode ser rodada, mas a leitura fica muito mais instavel.

## Script de Auditoria

Arquivo: `scripts/audit_bias.py`

Ele usa o mesmo pipeline do worker:

- deteccao facial
- crop/preprocessamento
- inferencia do modelo
- threshold calibrado do checkpoint

Arquivo auxiliar: `scripts/build_bias_manifest.py`

Ele monta o CSV a partir de uma estrutura de pastas para voce nao precisar catalogar amostra por amostra na mao.

## Formato do CSV

Exemplo:

```csv
id,image_path,label,group
1,data/real/black/person_001.jpg,real,black
2,data/real/white/person_002.jpg,real,white
3,data/fake/black/person_003.jpg,fake,black
4,data/fake/white/person_004.jpg,fake,white
```

Campos obrigatorios:

- `image_path`
- `label`
- `group`

Campos aceitos em `label`:

- `real`, `genuine`, `authentic`, `0`
- `fake`, `deepfake`, `manipulated`, `1`

## Como Rodar

### 1. Gerar o manifesto a partir de pastas

Layout `root/group/label/imagens`:

```bash
python scripts/build_bias_manifest.py \
  --root data/bias_holdout \
  --layout group-label \
  --output data/bias_audit_manifest.csv \
  --max-per-cell 300
```

Layout `root/label/group/imagens`:

```bash
python scripts/build_bias_manifest.py \
  --root data/bias_holdout \
  --layout label-group \
  --output data/bias_audit_manifest.csv \
  --max-per-cell 300
```

### 2. Auditar um modelo

```bash
python scripts/audit_bias.py \
  --manifest data/bias_audit_manifest.csv \
  --target-model efficientnet_b0_v5 \
  --output-json artifacts/bias_audit_v5.json \
  --output-csv artifacts/bias_audit_v5
```

### 3. Comparar varios modelos no mesmo holdout

```bash
python scripts/audit_bias.py \
  --manifest data/bias_audit_manifest.csv \
  --target-model efficientnet_b0_v4 \
  --target-model efficientnet_b0_v5 \
  --target-model alexnet \
  --output-json artifacts/bias_compare.json \
  --output-csv artifacts/bias_compare
```

### 4. Auditoria do AlexNet v7

Antes de rodar:

- coloque `alexnet_deepfake_v7.pth` em `model/weights/`
- preencha `data/fairness_holdout/` sem reutilizar imagens do treino

Gerar manifesto:

```bash
python scripts/build_bias_manifest.py \
  --root data/fairness_holdout \
  --layout group-label \
  --output data/manifests/bias_audit_manifest.csv \
  --max-per-cell 100
```

Rodar auditoria:

```bash
python scripts/audit_bias.py \
  --manifest data/manifests/bias_audit_manifest.csv \
  --target-model alexnet \
  --output-json artifacts/bias_alexnet_v7.json \
  --output-csv artifacts/bias_alexnet_v7
```

## Como Ler o Resultado

### `mean_p_fake_real`

Media do score `p_fake` apenas nas imagens reais. Se um grupo tiver esse valor claramente mais alto que outro, o modelo esta mais inclinado a chamar esse grupo de fake.

### `fpr`

Taxa de falso positivo em imagens reais. Esse e o indicador mais importante para o problema que voce observou.

### `fnr`

Taxa de falso negativo em imagens fake. Tambem importa, mas no seu caso o sinal mais preocupante e o falso positivo em pessoas reais.

### `fpr_gap` e `fnr_gap`

Diferenca entre o maior e o menor valor entre grupos. Quanto maior, pior a consistencia.

## Boas Praticas de Curadoria

- Balancear `real` e `fake` por grupo.
- Balancear iluminacao, resolucao, camera e compressao por grupo.
- Nao misturar um grupo com fotos limpas e outro com fotos ruins.
- Nao usar apenas datasets de um pais ou de uma plataforma.
- Medir fairness com o detector facial atual, nao so com o dataset cru.

## Datasets a Priorizar

### Para diversidade de rostos reais

- `FairFace`
- `RFW`
- capturas proprias do seu fluxo real, com celular e webcam, em iluminacoes diferentes

### Para deepfakes com mais variedade

- `FaceForensics++`
- `Celeb-DF`
- `DFDC`
- `DeeperForensics-1.0`

## Mistura Recomendada Para o Proximo Treino

### Reais

- `35%` capturas proprias alinhadas ao seu produto
- `35%` `FairFace`
- `15%` `RFW`
- `15%` reais extraidos dos datasets de deepfake acima

### Fakes

- `30%` `FaceForensics++`
- `25%` `Celeb-DF`
- `25%` `DFDC`
- `20%` `DeeperForensics-1.0`

Essa mistura nao deve ser tratada como receita fixa. Ela e um ponto de partida para reduzir dependencia de um unico dominio.

## Estrategia Recomendada

1. Montar um conjunto de validacao estratificado por grupo.
2. Gerar o CSV com `scripts/build_bias_manifest.py`.
3. Rodar `scripts/audit_bias.py` no modelo atual.
4. Treinar de novo com a mistura revisada de datasets.
5. Comparar modelos no mesmo holdout.
6. Repetir a auditoria usando o mesmo CSV de validacao.
7. So promover o modelo se a melhora geral vier junto com reducao de `fpr_gap`.

## Relacao Com O Colab

- treino no Colab: ok
- auditoria final de fairness: local e fixa
- holdout de fairness: nao deve entrar no treino

Fluxo recomendado:

1. preparar dataset de treino localmente
2. subir para o Drive
3. treinar no Colab
4. baixar o `.pth`
5. auditar localmente com o mesmo holdout
