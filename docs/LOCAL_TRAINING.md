# Treino local com RTX 4050

Este guia deixa o DeepTrustShield pronto para rodar os notebooks localmente, sem depender do Colab a cada treino.

## Estado esperado

- GPU: RTX 4050 visivel em `lspci`.
- Driver NVIDIA funcional em `nvidia-smi`.
- PyTorch com CUDA no ambiente de treino.
- Notebooks abertos a partir da raiz do projeto.

## Verificacao rapida

Use o ambiente dedicado de treino:

```bash
.venv-training/bin/python scripts/check_training_env.py --load-models
```

O campo importante e:

```text
cuda_available=True
cuda_device_0=NVIDIA GeForce RTX 4050...
```

Se aparecer `cuda_available=False`, as analises e os notebooks vao cair para CPU.

## Ambiente de notebook

O ambiente recomendado para notebooks locais e `.venv-training`. Ele tem a stack do worker, PyTorch com CUDA, Jupyter e bibliotecas de treino:

```bash
python -m venv .venv-training
.venv-training/bin/python -m pip install --upgrade pip setuptools wheel
.venv-training/bin/python -m pip install -r worker/requirements.txt -r requirements-training.txt
.venv-training/bin/python -m ipykernel install --user --name deeptrustshield-training --display-name "DeepTrustShield Training (.venv-training)"
```

No VS Code, abra o notebook e selecione o kernel `DeepTrustShield Training (.venv-training)`.

## Notebooks principais

- AlexNet mais recente: `notebooks/train_deepfake_detector_v82.ipynb`
- MobileNetV3 Small: `notebooks/train_deepfake_detector_mobilenet_v1.ipynb`

Esses notebooks ainda tem celulas de Colab, como `from google.colab import drive`. Em treino local, pule essas celulas e aponte os caminhos para os datasets em `data/`.

## Modelos suportados na API

O backend/worker aceita estes identificadores:

```text
auto
alexnet_v82
alexnet_v81
alexnet_v8
alexnet_v7
alexnet_v6
alexnet
mobilenet_v3_small_v1
mobilenet_v3_small
mobilenet
efficientnet_b0_v5
efficientnet_b0_v4
efficientnet_b0
efficientnet_b3
efficientnet_b4
```

Pesos esperados:

```text
model/weights/alexnet_deepfake_v82.pth
model/weights/mobilenet_v3_small_deepfake_v1.pth
```

## Observacao sobre CUDA

O DeepTrustShield usa CUDA automaticamente quando o PyTorch do ambiente retorna `cuda_available=True`. Para conferir o caminho completo do projeto, rode:

```bash
.venv-training/bin/python scripts/check_training_env.py --load-models
```

Com a RTX 4050 ativa, o smoke test deve mostrar os modelos carregando em `device=cuda:0`.
