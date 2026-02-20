"""
Script para testar o modelo v4 diretamente, sem passar pelo pipeline de detecção de rosto.
Isso simula exatamente o que acontece durante o treino.
"""
import sys
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# Configuração
MODEL_PATH = '/home/wesleiferreira/Documentos/detectar/model/weights/efficientnet_b0_deepfake_v4.pth'
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


class DeepfakeDetectorV4(nn.Module):
    """Modelo v4 - mesma arquitetura do notebook de treino."""
    
    def __init__(self, model_name='efficientnet_b0', dropout=0.5):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        num_features = self.backbone.num_features
        
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


def load_model():
    """Carrega o modelo v4."""
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    model = DeepfakeDetectorV4(
        model_name=checkpoint['config']['model_name'],
        dropout=0.5
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    threshold = checkpoint['best_threshold']
    print(f"Modelo carregado. Threshold: {threshold:.4f}")
    
    return model, threshold


def preprocess_image(image_path):
    """Preprocessa a imagem EXATAMENTE como no treino."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor


def predict(model, image_tensor, threshold):
    """Faz a predição."""
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()
    
    is_fake = prob >= threshold
    
    return {
        'probability': prob,
        'is_fake': is_fake,
        'label': 'FAKE' if is_fake else 'REAL',
        'threshold': threshold
    }


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_model_direct.py <caminho_da_imagem>")
        print("\nExemplo:")
        print("  python test_model_direct.py /caminho/para/imagem_fake.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"\nTestando: {image_path}")
    print("="*50)
    
    # Carregar modelo
    model, threshold = load_model()
    
    # Preprocessar imagem
    image_tensor = preprocess_image(image_path)
    print(f"Tensor shape: {image_tensor.shape}")
    
    # Predição
    result = predict(model, image_tensor, threshold)
    
    print(f"\n{'='*50}")
    print(f"RESULTADO: {result['label']}")
    print(f"{'='*50}")
    print(f"Probabilidade de FAKE: {result['probability']*100:.2f}%")
    print(f"Threshold usado: {result['threshold']*100:.2f}%")
    print(f"{'='*50}")
    
    if result['is_fake']:
        print("🚨 DEEPFAKE DETECTADO!")
    else:
        print("✅ Imagem parece REAL")


if __name__ == "__main__":
    main()
