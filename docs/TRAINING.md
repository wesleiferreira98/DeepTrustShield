# 🎓 Treinamento de Modelos de Detecção de Deepfake

## Visão Geral

O DeepTrustShield utiliza modelos de deep learning baseados em EfficientNet para detectar imagens manipuladas (deepfakes). Os notebooks de treinamento estão localizados em `notebooks/`.

## Notebooks Disponíveis

| Versão | Arquivo | Dataset | Características |
|--------|---------|---------|-----------------|
| v1 | `train_deepfake_detector.ipynb` | 140k Faces | Baseline |
| v2 | `train_deepfake_detector_v2.ipynb` | 140k Faces | Melhorias na arquitetura |
| v3 | `train_deepfake_detector_v3.ipynb` | 140k + FaceForensics | Multi-dataset |
| v4 | `train_deepfake_detector_v4.ipynb` | manjilkarki | BatchNorm, dropout melhorado |
| **v5** | `train_deepfake_detector_v5.ipynb` | **Combinado** | **Recomendado - Melhor generalização** |
| v6 | `train_deepfake_detector_v6.ipynb` | Combinado | AlexNet (experimental) |

---

## Datasets Recomendados

### 1. 140k Real and Fake Faces (Kaggle) ⭐ MVP
```
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
```
- 140.000 imagens (~3GB)
- Balanceado: 100k train, 20k valid, 20k test
- Faces já recortadas

### 2. manjilkarki/deepfake-and-real-images
```python
import kagglehub
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
```
- Deepfakes de alta qualidade (face-swap)
- Reais de vídeos

### 3. LFW (Labeled Faces in the Wild)
```python
from sklearn.datasets import fetch_lfw_people
lfw_data = fetch_lfw_people(min_faces_per_person=1)
```
- Fotos reais diversas
- Boa para combater vieses

### 4. FaceForensics++
- Estado da arte
- Múltiplas técnicas: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- Requer solicitação de acesso

---

## Arquitetura do Modelo

### EfficientNet + Custom Head

```
┌─────────────────────────────────────────┐
│          Input Image (224×224×3)        │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│     EfficientNet-B0/B3/B4 Backbone      │
│          (Pretrained ImageNet)          │
│                                          │
│    ┌─────────────────────────────────┐  │
│    │  Conv Layers + SE Blocks        │  │
│    │  Global Average Pooling         │  │
│    └─────────────────────────────────┘  │
│                                          │
│    Output: Feature Vector (1280 dims)   │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Classification Head            │
│                                          │
│    ┌─────────────────────────────────┐  │
│    │  Dropout (0.5)                   │  │
│    │  Linear (1280 → 256)             │  │
│    │  BatchNorm1d (256)               │  │
│    │  ReLU                            │  │
│    │  Dropout (0.25)                  │  │
│    │  Linear (256 → 1)                │  │
│    └─────────────────────────────────┘  │
│                                          │
│    Output: Logit (1 dim)                │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Sigmoid Activation             │
│                                          │
│    Output: p_fake ∈ [0, 1]              │
│                                          │
│    0 = Real    |    1 = Fake            │
└─────────────────────────────────────────┘
```

### Código da Arquitetura (v4/v5)

```python
import torch
import torch.nn as nn
import timm

class DeepfakeDetectorV4(nn.Module):
    def __init__(self, model_name='efficientnet_b0', dropout=0.5):
        super().__init__()
        
        # Backbone EfficientNet (sem classifier)
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,  # ImageNet pretrained
            num_classes=0,    # Remove head
            global_pool='avg'
        )
        
        num_features = self.backbone.num_features  # 1280 para B0
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(-1)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
```

---

## Processo de Treinamento

### 1. Preparação dos Dados

```python
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Augmentações de treino
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validação/Teste (sem augmentação)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### 2. Dataset Customizado

```python
class DeepfakeDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data: Lista de tuplas (path, label)
                  label: 0 = real, 1 = fake
        """
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
```

### 3. Loop de Treinamento

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configuração
model = DeepfakeDetectorV4(model_name='efficientnet_b0', dropout=0.5)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=20)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validação
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        for images, labels in val_loader:
            outputs = torch.sigmoid(model(images.to(device)))
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.numpy())
        
        auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch}: AUC = {auc:.4f}")
```

### 4. Calibração de Threshold

```python
from sklearn.metrics import precision_recall_curve

# Encontrar melhor threshold
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Melhor threshold: {best_threshold:.3f}")
print(f"F1-Score: {f1_scores[best_idx]:.4f}")
```

### 5. Salvando o Modelo

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_auc': best_auc,
    'best_threshold': best_threshold,
    'test_f1': test_f1,
    'config': {
        'model_name': 'efficientnet_b0',
        'dropout': 0.5,
        'input_size': 224
    }
}

torch.save(checkpoint, 'efficientnet_b0_deepfake_v5.pth')
```

---

## Treinamento v5 (Recomendado)

O notebook v5 combina múltiplos datasets para melhor generalização:

### Estratégia

```
Fakes:  manjilkarki/deepfake-and-real-images (face-swap)
        └── ~70.000 imagens

Reais:  50% manjilkarki (extraídos de vídeo)
        50% LFW (fotos diversas)
        └── ~70.000 imagens balanceadas

Total:  ~140.000 imagens
Split:  80% train / 10% val / 10% test
```

### Benefícios

1. **Reduz viés de câmera**: LFW tem fotos de diversas fontes
2. **Generalização**: Não "memoriza" padrões específicos de um dataset
3. **Robustez**: Funciona com selfies, fotos de celular, etc.

---

## Executando no Google Colab

### 1. Upload do Notebook

Faça upload de `train_deepfake_detector_v5.ipynb` para o Google Colab.

### 2. Configurar GPU

```
Runtime → Change runtime type → GPU (T4 recomendado)
```

### 3. Instalar Dependências

```python
!pip install -q torch torchvision timm scikit-learn pillow tqdm kagglehub
```

### 4. Baixar Dataset

```python
import kagglehub
manjil_path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
```

### 5. Treinar

Execute todas as células do notebook. O treinamento leva ~2-4 horas com GPU T4.

### 6. Baixar Modelo

```python
from google.colab import files
files.download('efficientnet_b0_deepfake_v5.pth')
```

---

## Métricas de Avaliação

### Métricas Principais

| Métrica | Descrição | Target |
|---------|-----------|--------|
| **AUC-ROC** | Área sob curva ROC | > 0.95 |
| **F1-Score** | Média harmônica precisão/recall | > 0.90 |
| **Accuracy** | Taxa de acerto | > 0.90 |
| **Precision** | TP / (TP + FP) | > 0.85 |
| **Recall** | TP / (TP + FN) | > 0.90 |

### Matriz de Confusão

```
                   Predicted
                 Real    Fake
Actual  Real    [TN]    [FP]
        Fake    [FN]    [TP]
```

---

## Resultados dos Modelos

### Benchmarks (Test Set)

| Modelo | AUC | F1 | Accuracy | Threshold |
|--------|-----|-----|----------|-----------|
| EfficientNet-B0 v1 | 0.92 | 0.88 | 0.87 | 0.50 |
| EfficientNet-B0 v4 | 0.95 | 0.91 | 0.90 | 0.50 |
| **EfficientNet-B0 v5** | **0.97** | **0.93** | **0.92** | **0.50** |
| EfficientNet-B3 | 0.96 | 0.92 | 0.91 | 0.50 |
| EfficientNet-B4 | 0.96 | 0.92 | 0.91 | 0.50 |

---

## Dicas de Treinamento

### 1. Overfitting
```python
# Usar dropout agressivo
dropout = 0.5

# Early stopping
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= 5:
        break

# Data augmentation forte
transforms.RandomErasing(p=0.1)
```

### 2. Desbalanceamento
```python
# Weighted sampler
class_weights = [1.0, 1.5]  # Mais peso para fakes
sampler = WeightedRandomSampler(weights, num_samples)

# Ou loss ponderada
pos_weight = torch.tensor([1.5])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 3. Learning Rate
```python
# Warmup + Cosine annealing
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# Ou learning rate finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1e-1, num_iter=100)
lr_finder.plot()
```

### 4. Mixed Precision
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Colocando Modelo em Produção

1. **Copie o arquivo `.pth`** para o diretório `model/weights/`:
   ```bash
   cp efficientnet_b0_deepfake_v5.pth /path/to/DeepTrustShield/model/weights/
   ```

2. **Reinicie o worker** para carregar o novo modelo:
   ```bash
   ./run_podman.sh restart
   ```

3. **Verifique os logs** para confirmar o carregamento:
   ```
   [INFO] Loading model weights from: efficientnet_b0_deepfake_v5.pth
   [INFO] Using DeepfakeDetectorV4 architecture
   [INFO] Using calibrated threshold: 0.500
   ```

---

## Referências

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
- [FaceForensics++ Benchmark](https://github.com/ondyari/FaceForensics)
- [Kaggle 140k Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
