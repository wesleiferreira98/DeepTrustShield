# 📚 Bases de Dados Públicas para Detecção de Deepfake

## Recomendadas para Treinamento

### 1. 140k Real and Fake Faces (Kaggle) ⭐ Recomendado para MVP
- **Link:** https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- **Tamanho:** 140.000 imagens (~3GB)
- **Formato:** Imagens JPG já recortadas (faces)
- **Divisão:** 100k train, 20k valid, 20k test
- **Prós:** Fácil de usar, já está em formato de imagens, balanceado
- **Contras:** Menor diversidade de técnicas de deepfake

### 2. FaceForensics++ ⭐ Estado da Arte
- **Link:** https://github.com/ondyari/FaceForensics
- **Tamanho:** 1000 vídeos originais + 4000 manipulados (~500GB full)
- **Técnicas:** Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter
- **Compressões:** c0 (raw), c23 (HQ), c40 (LQ)
- **Acesso:** Requer preenchimento de formulário
- **Prós:** Múltiplas técnicas, qualidade acadêmica, benchmark padrão
- **Contras:** Precisa extrair frames, download grande

### 3. Celeb-DF (Celeb DeepFake)
- **Link:** https://github.com/yuezunli/celeb-deepfakeforensics
- **Tamanho:** 590 vídeos reais + 5639 deepfakes (~15GB)
- **Características:** Alta qualidade, celebridades, desafiador
- **Acesso:** Requer solicitação via formulário
- **Prós:** Deepfakes de alta qualidade, mais desafiador
- **Contras:** Menor escala, acesso restrito

### 4. DFDC - DeepFake Detection Challenge (Kaggle)
- **Link:** https://www.kaggle.com/c/deepfake-detection-challenge
- **Tamanho:** 100.000+ vídeos (~470GB)
- **Características:** Maior dataset público, diverso
- **Prós:** Escala massiva, variedade de identidades e técnicas
- **Contras:** Download enorme, muitos falsos positivos difíceis

### 5. DeeperForensics-1.0
- **Link:** https://github.com/EndlessSora/DeeperForensics-1.0
- **Tamanho:** 60.000 vídeos (~200GB)
- **Características:** Perturbações controladas (blur, compressão, etc.)
- **Prós:** Testa robustez a degradações
- **Contras:** Acesso restrito

### 6. DeepFake Faces (Kaggle)
- **Link:** https://www.kaggle.com/datasets/dagnelies/deepfake-faces
- **Tamanho:** 191.000 faces (~4GB)
- **Formato:** Imagens extraídas de vídeos
- **Prós:** Pronto para uso, grande escala
- **Contras:** Qualidade variável

---

## Como Baixar

### Kaggle Datasets
```bash
# Instalar kaggle CLI
pip install kaggle

# Configurar credenciais (baixe kaggle.json da sua conta)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download 140k Real and Fake Faces
kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Download DeepFake Faces
kaggle datasets download -d dagnelies/deepfake-faces
```

### FaceForensics++
1. Acesse: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
2. Preencha o formulário
3. Aguarde aprovação (geralmente 1-2 dias)
4. Use o script de download:
```bash
git clone https://github.com/ondyari/FaceForensics.git
python FaceForensics/dataset/download_dataset.py \
    ./faceforensics \
    -d all \
    -c c23 \
    -t videos
```

### Celeb-DF
1. Acesse: https://github.com/yuezunli/celeb-deepfakeforensics
2. Envie email para os autores
3. Aguarde link de download

---

## Estrutura Recomendada

```
data/
├── train/
│   ├── real/
│   │   ├── img001.jpg
│   │   └── ...
│   └── fake/
│       ├── img001.jpg
│       └── ...
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

## Métricas de Referência (FaceForensics++ c23)

| Modelo | Accuracy | AUC |
|--------|----------|-----|
| XceptionNet | 95.7% | 0.981 |
| EfficientNet-B0 | 93.2% | 0.968 |
| EfficientNet-B4 | 96.1% | 0.985 |
| ResNet-50 | 91.5% | 0.952 |

---

## Dicas de Treinamento

1. **Balanceamento:** Mantenha ~50/50 real/fake
2. **Augmentation:** Use compressão JPEG, blur, noise
3. **Multi-dataset:** Treine em múltiplos datasets para generalização
4. **Face crops:** Sempre use apenas a região da face
5. **Transfer learning:** Comece com pesos ImageNet

---

## Uso no Colab Pro

O notebook `train_deepfake_detector.ipynb` está configurado para:
- GPU A100 (40GB) ou V100 (16GB)
- Batch size adaptável
- Mixed precision (AMP)
- Early stopping
- Checkpointing automático
