# Model Weights

Este diretório contém os pesos do modelo EfficientNet-B0 treinado para detecção de deepfake.

## Arquivo esperado

```
efficientnet_b0_deepfake.pth
```

## Formato

- **Arquitetura**: EfficientNet-B0 (timm)
- **Cabeça personalizada**: GAP → Dropout(0.3) → Dense(1) → Sigmoid
- **Entrada**: 224×224×3 (normalizado com ImageNet stats)
- **Saída**: p_fake (0-1)

## Modo Demo

Se os pesos não forem encontrados, o sistema funciona em **modo demo** com pesos aleatórios.
Para produção, treine um modelo com datasets como:

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

## Treinamento

Exemplo de código para treinar:

```python
import timm
import torch.nn as nn

# Criar modelo
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier.in_features, 1),
    nn.Sigmoid()
)

# Treinar...

# Salvar pesos
torch.save(model.state_dict(), 'efficientnet_b0_deepfake.pth')
```
