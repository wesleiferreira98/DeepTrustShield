import os
import glob
import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)

# Global model cache
_model = None
_device = None
_best_threshold = 0.5  # Pode ser sobrescrito pelo checkpoint
_loaded_model_name = None
_loaded_weights_path = None
_loaded_model_version = None
_loaded_target_model = None


def get_device():
    """Get the best available device."""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            logger.info("Using CUDA for inference")
        else:
            _device = torch.device("cpu")
            logger.info("Using CPU for inference")
    return _device


class DeepfakeDetector(nn.Module):
    """Modelo de detecção de deepfake baseado em EfficientNet (v1/v2)."""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=1):
        super().__init__()
        
        # Backbone EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0  # Remove classifier
        )
        
        # Obter dimensão do feature
        self.num_features = self.backbone.num_features
        
        # Classifier head (v1/v2 architecture)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)
    
    def predict_proba(self, x):
        """Retorna probabilidade de ser fake."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs


class DeepfakeDetectorV4(nn.Module):
    """Modelo de detecção de deepfake v4 - arquitetura melhorada."""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=1, dropout=0.5):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        num_features = self.backbone.num_features
        
        # Classifier head (v4 architecture - matches training notebook)
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
        """Retorna probabilidade de ser fake."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs


class DeepfakeDetectorAlexNet(nn.Module):
    """Modelo de detecção de deepfake v6 - AlexNet backbone."""
    
    def __init__(self, dropout=0.5):
        super().__init__()
        
        import torchvision.models as models
        
        # Carregar AlexNet sem pretrained (vamos carregar os pesos do checkpoint)
        self.backbone = models.alexnet(weights=None)
        
        # Extrair features (conv + avgpool)
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Número de features após flatten: 256 * 6 * 6 = 9216
        num_features = 256 * 6 * 6
        
        # Classifier para detecção de deepfake (binário)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(-1)
    
    def predict_proba(self, x):
        """Retorna probabilidade de ser fake."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs


def _infer_model_name_from_filename(weights_path: str) -> str:
    filename = os.path.basename(weights_path).lower()
    for name in ("efficientnet_b4", "efficientnet_b3", "efficientnet_b0", "alexnet"):
        if name in filename:
            return name
    return "efficientnet_b0"


def _infer_model_version(model_name: str, weights_path: str) -> str:
    filename = os.path.basename(weights_path).lower()
    if "alexnet" in filename and ("_v7" in filename or "v7" in filename):
        return "alexnet_v7"
    if "alexnet" in filename and ("_v6" in filename or "v6" in filename):
        return "alexnet_v6"
    if "efficientnet_b0" in filename and ("_v5" in filename or "v5" in filename):
        return "efficientnet_b0_v5"
    if "efficientnet_b0" in filename and ("_v4" in filename or "v4" in filename):
        return "efficientnet_b0_v4"
    return model_name


def _candidate_weight_dirs(model_path: str) -> list[str]:
    base = os.path.abspath(model_path)
    parent = os.path.dirname(base)
    grandparent = os.path.dirname(parent)
    dirs = [base, parent, grandparent]
    # Remove duplicados mantendo ordem
    return list(dict.fromkeys(dirs))


def _discover_weight_files(model_path: str) -> list[str]:
    all_candidates = []
    for weight_dir in _candidate_weight_dirs(model_path):
        all_candidates.extend(glob.glob(os.path.join(weight_dir, "*.pth")))

    # Remove duplicados mantendo ordem
    all_candidates = list(dict.fromkeys(all_candidates))

    # Prioriza checkpoints mais novos do comparativo
    priority_exact = {
        "alexnet_deepfake_v7.pth": 0,    # v7 AlexNet
        "alexnet_deepfake_v6.pth": 1,    # v6 AlexNet
        "efficientnet_b0_deepfake_v5.pth": 2,  # v5 é prioridade máxima para EfficientNet
        "efficientnet_b0_deepfake_v4.pth": 3,
        "best_model_efficientnet_b4.pth": 4,
        "best_model_efficientnet_b3.pth": 5,
        "best_model_efficientnet_b0.pth": 6,
        "best_model_alexnet.pth": 7,
        "best_model.pth": 8,
        "efficientnet_b0_deepfake_v2.pth": 9,
        "efficientnet_b0_deepfake_v1.pth": 10,
        "efficientnet_b0_deepfake.pth": 11,
    }

    def rank(path: str) -> tuple[int, str]:
        name = os.path.basename(path)
        if name in priority_exact:
            return (priority_exact[name], name)
        if name.startswith("best_model_"):
            return (8, name)
        if name.endswith("_deepfake_v2.pth"):
            return (9, name)
        if name.endswith(".pth"):
            return (10, name)
        return (99, name)

    return sorted(all_candidates, key=rank)


def _select_candidates(candidate_weights: list[str], target_model: str) -> list[str]:
    if not target_model or target_model == "auto":
        return candidate_weights

    alias_map = {
        "efficientnet_b0_v4": ("efficientnet_b0_v4", "_v4", "deepfake_v4", "final_model_efficientnet_b0_v4"),
        "efficientnet_b0_v5": ("efficientnet_b0_v5", "_v5", "deepfake_v5", "efficientnet_b0_deepfake_v5"),
        "efficientnet_b0": ("efficientnet_b0",),
        "efficientnet_b3": ("efficientnet_b3",),
        "efficientnet_b4": ("efficientnet_b4",),
        "alexnet_v7": ("alexnet_v7", "_v7", "alexnet_deepfake_v7"),
        "alexnet_v6": ("alexnet_v6", "_v6", "alexnet_deepfake_v6"),
        "alexnet": ("alexnet",),
    }
    aliases = alias_map.get(target_model, (target_model,))

    preferred = [
        p for p in candidate_weights
        if any(alias in os.path.basename(p).lower() for alias in aliases)
    ]
    fallback = [p for p in candidate_weights if p not in preferred]
    return preferred + fallback


def load_model(model_path: str, target_model: str = "auto"):
    """
    Load EfficientNet-B0 model for deepfake detection.
    
    Supports multiple weight formats:
    - efficientnet_b0_deepfake_v2.pth (trained with improved notebook)
    - efficientnet_b0_deepfake_v1.pth (trained checkpoint from Colab)
    - efficientnet_b0_deepfake.pth (legacy format)
    """
    global _model, _best_threshold, _loaded_model_name, _loaded_weights_path, _loaded_model_version, _loaded_target_model
    
    # Reuse cache only if loaded model already matches requested target
    if _model is not None and (
        target_model == "auto" or
        _loaded_target_model == target_model or
        _loaded_model_name == target_model or
        _loaded_model_version == target_model
    ):
        return _model
    
    device = get_device()
    
    model = None
    # Try to load weights from model_path and project root
    candidate_weights = _discover_weight_files(model_path)
    candidate_weights = _select_candidates(candidate_weights, target_model)
    weights_loaded = False
    for weights_path in candidate_weights:
        logger.info(f"Loading model weights from: {weights_path}")
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

            # Determine model name from checkpoint metadata first
            model_name = None
            if isinstance(checkpoint, dict):
                cfg = checkpoint.get("config", {})
                model_name = cfg.get("model_name")
            if not model_name:
                model_name = _infer_model_name_from_filename(weights_path)

            # Use DeepfakeDetectorAlexNet for v6/alexnet, DeepfakeDetectorV4 for v4/v5, otherwise regular
            filename = os.path.basename(weights_path).lower()
            if "_v7" in filename or "_v6" in filename or "alexnet" in filename:
                logger.info(f"Using DeepfakeDetectorAlexNet architecture for {filename}")
                model = DeepfakeDetectorAlexNet(dropout=0.5)
            elif "_v4" in filename or "_v5" in filename or "v4" in filename or "v5" in filename:
                logger.info(f"Using DeepfakeDetectorV4 architecture for {filename}")
                model = DeepfakeDetectorV4(model_name=model_name, num_classes=1, dropout=0.5)
            else:
                model = DeepfakeDetector(model_name=model_name, num_classes=1)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if "best_threshold" in checkpoint:
                    _best_threshold = checkpoint["best_threshold"]
                    logger.info(f"Using calibrated threshold: {_best_threshold:.3f}")
                logger.info(
                    "Loaded checkpoint - model: %s, AUC: %s, F1: %s",
                    model_name,
                    checkpoint.get("best_auc", checkpoint.get("test_auc", "N/A")),
                    checkpoint.get("test_f1", "N/A"),
                )
            else:
                model.load_state_dict(checkpoint, strict=False)

            _loaded_model_name = model_name
            _loaded_weights_path = weights_path
            _loaded_model_version = _infer_model_version(model_name, weights_path)
            _loaded_target_model = target_model
            weights_loaded = True
            break
        except Exception as e:
            logger.warning(f"Failed to load {weights_path}: {e}")
            continue
    
    if not weights_loaded:
        logger.warning(f"Model weights not found in {model_path} or parent directories.")
        logger.warning("Using random initialization (DEMO MODE).")
        logger.warning("Train a model with the Colab notebook and place weights here.")
        model = DeepfakeDetector(model_name="efficientnet_b0", num_classes=1)
    
    model = model.to(device)
    model.eval()
    
    _model = model
    return model


def get_threshold() -> float:
    """Retorna o threshold calibrado do modelo."""
    global _best_threshold
    return _best_threshold


def get_loaded_model_name() -> str:
    return _loaded_model_name or "unknown"


def get_loaded_model_version() -> str:
    return _loaded_model_version or get_loaded_model_name()


def run_inference(face_tensor: torch.Tensor, model_path: str, target_model: str = "auto") -> float:
    """
    Run deepfake detection inference.
    
    Args:
        face_tensor: Preprocessed face tensor (1, 3, 224, 224)
        model_path: Path to model weights directory
    
    Returns:
        p_fake: Probability of being fake (0-1)
    """
    device = get_device()
    model = load_model(model_path, target_model=target_model)
    
    # Move tensor to device
    face_tensor = face_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(face_tensor)
        p_fake = torch.sigmoid(logits).item()
    
    threshold = get_threshold()
    is_fake = p_fake > threshold
    
    logger.info(
        "Inference result: p_fake = %.4f (threshold: %.3f, is_fake: %s, model: %s, weights: %s)",
        p_fake, threshold, is_fake, _loaded_model_name, _loaded_weights_path
    )
    
    return p_fake
