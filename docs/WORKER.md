# Worker - Pipeline de Detecção de Deepfake

## Visão Geral

O Worker é o componente responsável pelo processamento de imagens e detecção de deepfakes. Ele consome tarefas da fila Redis (RQ) e executa um pipeline de 7 etapas para analisar cada imagem.

---

## Por que o Worker é Importante?

O Worker é o **coração do DeepTrustShield** — sem ele, não há detecção de deepfakes. Aqui estão os motivos pelos quais esse componente é essencial:

### 1. Processamento Assíncrono

- **Não bloqueia a API**: Enquanto o Worker processa uma imagem (que pode levar segundos), a API continua respondendo a novas requisições
- **Experiência do usuário**: O frontend pode mostrar progresso em tempo real sem travar
- **Resiliência**: Se o Worker falhar, os jobs permanecem na fila e são reprocessados automaticamente

### 2. Escalabilidade Horizontal

- **Múltiplos workers**: Em picos de demanda, basta iniciar mais instâncias do Worker
- **Distribuição de carga**: Redis distribui automaticamente os jobs entre workers disponíveis
- **Zero downtime**: Workers podem ser adicionados/removidos sem parar o sistema

### 3. Isolamento de Recursos

- **Processamento pesado separado**: As operações de ML (inferência, GPU) não impactam a API
- **Memória dedicada**: Modelos carregados em memória ficam isolados
- **Segurança**: Erros no processamento não derrubam a API

### 4. Pipeline Modular

- **Debug facilitado**: Cada etapa pode ser testada/debugada independentemente
- **Rastreabilidade**: Progresso salvo a cada etapa permite identificar onde falhas ocorrem
- **Extensibilidade**: Novas etapas (ex: detecção de manipulação de áudio) podem ser adicionadas sem refatoração

### 5. Flexibilidade de Modelos

- **Hot-swap**: Modelos podem ser atualizados sem reiniciar o sistema
- **A/B testing**: Diferentes versões de modelo podem rodar em paralelo
- **Fallback**: Se um modelo falhar, outro pode assumir automaticamente

### 6. Evidências e Auditoria

- **Artefatos persistentes**: Face crops, bounding boxes salvos para revisão humana
- **Reprodutibilidade**: Com a imagem original e versão do modelo, o resultado pode ser replicado
- **Compliance**: Logs detalhados para auditoria e explicabilidade da decisão

### Comparação: Com vs Sem Worker

| Cenário                       | Sem Worker            | Com Worker                        |
| ------------------------------ | --------------------- | --------------------------------- |
| 100 requisições simultâneas | API trava             | API responde, jobs enfileirados   |
| Modelo crasheia                | API cai               | Worker reinicia, outros continuam |
| Atualização de modelo        | Downtime necessário  | Zero downtime                     |
| Tempo de resposta API          | 2-5 segundos          | < 100ms (job_id retornado)        |
| Uso de GPU                     | Conflitos de memória | Isolado por worker                |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                           WORKER                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Redis      │───▶│   RQ Worker  │───▶│   Pipeline   │      │
│  │   Queue      │    │   Consumer   │    │   Executor   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │               │
│         ┌────────────────────────────────────────┘               │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    PIPELINE STAGES                       │    │
│  │                                                          │    │
│  │  1. validating_image    → Validar formato e integridade │    │
│  │  2. detecting_face      → MediaPipe BlazeFace            │    │
│  │  3. quality_checks      → Blur, brilho, contraste        │    │
│  │  4. preprocessing       → Crop, resize 224×224           │    │
│  │  5. model_inference     → EfficientNet-B0/B3/B4          │    │
│  │  6. calibrating_score   → Aplicar thresholds             │    │
│  │  7. generating_evidence → Salvar artefatos               │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Estrutura do Projeto

```
worker/
├── app/
│   ├── __init__.py
│   ├── config.py            # Configurações e thresholds
│   ├── db_models.py         # Modelos SQLAlchemy
│   ├── tasks.py             # Task principal do RQ
│   └── pipeline/
│       ├── __init__.py
│       ├── face_detection.py    # Detecção de face (MediaPipe)
│       ├── quality_checks.py    # Verificações de qualidade
│       ├── preprocessing.py     # Pré-processamento para modelo
│       ├── inference.py         # Inferência com PyTorch
│       └── blaze_face_short_range.tflite  # Modelo de detecção
├── Dockerfile
├── requirements.txt
└── venv/
```

---

## Configuração

### `app/config.py`

```python
class Config:
    # Database
    DATABASE_URL = "sqlite:///./detectar.db"
  
    # Redis
    REDIS_URL = "redis://localhost:6379/0"
  
    # Paths
    ARTIFACTS_PATH = "./artifacts"
    MODEL_PATH = "./model/weights"
  
    # Thresholds de risco
    THRESHOLD_LOW = 0.50   # Abaixo = baixo risco (real)
    THRESHOLD_HIGH = 0.80  # Acima = alto risco (fake)
  
    # Quality checks
    MIN_FACE_SIZE = 80     # pixels mínimos
    BLUR_THRESHOLD = 100   # Laplacian variance mínima
    BRIGHTNESS_MIN = 40    # 0-255
    BRIGHTNESS_MAX = 220   # 0-255
```

---

## Pipeline de Processamento

### 1. Validação de Imagem (`validating_image`)

Verifica se o arquivo é uma imagem válida.

```python
# Formatos aceitos
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"]
```

### 2. Detecção de Face (`detecting_face`)

Usa MediaPipe Tasks API com BlazeFace para detectar faces.

**Arquivo:** `pipeline/face_detection.py`

```python
def detect_faces(image_path: str) -> list:
    """
    Detecta faces usando MediaPipe BlazeFace.
  
    Returns:
        Lista de dicionários com:
        - x, y, width, height (pixels absolutos)
        - x_pct, y_pct, width_pct, height_pct (porcentagens)
        - confidence (0-1)
    """
```

**Características:**

- Modelo: `blaze_face_short_range.tflite`
- Confiança mínima: 0.5
- Padding automático: 20% em cada lado
- Suporta múltiplas faces (usa a maior)

**Outputs possíveis:**

- 0 faces → Flag `no_face`, resultado `undetermined`
- 1 face → Processamento normal
- 2+ faces → Flag `multiple_faces`, usa a maior face

### 3. Verificação de Qualidade (`quality_checks`)

Analisa a qualidade da região da face.

**Arquivo:** `pipeline/quality_checks.py`

```python
def check_quality(image_path: str, face: dict) -> dict:
    """
    Calcula métricas de qualidade.
  
    Returns:
        {
            "blur_score": float,   # Laplacian variance (maior = mais nítido)
            "brightness": float,   # Média de brilho (0-255)
            "contrast": float      # Desvio padrão da intensidade
        }
    """
```

**Flags geradas:**

| Flag               | Condição               | Impacto            |
| ------------------ | ------------------------ | ------------------ |
| `blur_high`      | blur_score < 100         | Imagem borrada     |
| `low_light`      | brightness < 40 ou > 220 | Iluminação ruim  |
| `face_too_small` | width ou height < 80px   | Face muito pequena |

### 4. Pré-processamento (`preprocessing`)

Prepara a imagem para inferência do modelo.

**Arquivo:** `pipeline/preprocessing.py`

```python
def preprocess_face(image_path, face, job_id, artifacts_path):
    """
    1. Crop da região da face
    2. Resize para 224×224
    3. Conversão BGR → RGB
    4. Normalização ImageNet
  
    Returns:
        (face_crop_path, face_tensor)
    """
```

**Normalização ImageNet:**

```python
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
```

**Detecção de face pré-cropada:**
Se a face detectada cobre mais de 60% da imagem, assume que a imagem já é um face crop e usa a imagem inteira.

### 5. Inferência do Modelo (`model_inference`)

Executa o modelo de deep learning.

**Arquivo:** `pipeline/inference.py`

```python
def run_inference(face_tensor, model_path, target_model="auto") -> float:
    """
    Executa inferência com EfficientNet.
  
    Args:
        face_tensor: Tensor (1, 3, 224, 224)
        model_path: Diretório dos pesos
        target_model: "auto" ou nome específico
  
    Returns:
        p_fake: Probabilidade de ser fake (0-1)
    """
```

#### Modelos Suportados

| Modelo             | Arquivo                                | Arquitetura                 | Descrição        |
| ------------------ | -------------------------------------- | --------------------------- | ------------------ |
| EfficientNet-B0 v5 | `efficientnet_b0_deepfake_v5.pth`    | `DeepfakeDetectorV4`      | Dataset combinado  |
| EfficientNet-B0 v4 | `final_model_efficientnet_b0_v4.pth` | `DeepfakeDetectorV4`      | Treinamento v4     |
| EfficientNet-B0    | `best_model_efficientnet_b0.pth`     | `DeepfakeDetector`        | Modelo padrão     |
| EfficientNet-B3    | `best_model_efficientnet_b3.pth`     | `DeepfakeDetector`        | Maior capacidade   |
| EfficientNet-B4    | `best_model_efficientnet_b4.pth`     | `DeepfakeDetector`        | Máxima capacidade |
| AlexNet v6         | `alexnet_deepfake_v6.pth`            | `DeepfakeDetectorAlexNet` | Modelo legado      |

#### Arquiteturas

**DeepfakeDetector (v1/v2/v3):**

```python
class DeepfakeDetector(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        self.backbone = timm.create_model(model_name, pretrained=False)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
```

**DeepfakeDetectorV4 (v4/v5):**

```python
class DeepfakeDetectorV4(nn.Module):
    def __init__(self, model_name='efficientnet_b0', dropout=0.5):
        self.backbone = timm.create_model(model_name, pretrained=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
```

#### Seleção Automática de Modelo

Quando `target_model="auto"`:

1. Busca pesos em `MODEL_PATH` e diretório pai
2. Ordena por prioridade (v5 > v4 > b4 > b3 > b0)
3. Carrega o primeiro disponível

### 6. Calibração do Score (`calibrating_score`)

Converte `p_fake` em nível de risco.

```python
def calculate_risk_level(p_fake: float, flags: dict) -> str:
    if flags.get("no_face"):
        return "undetermined"
  
    if p_fake is None:
        return "undetermined"
  
    if p_fake < 0.50:
        return "low"       # Provavelmente real
    elif p_fake < 0.80:
        return "medium"    # Inconclusivo
    else:
        return "high"      # Provavelmente fake
```

### 7. Geração de Evidências (`generating_evidence`)

Salva artefatos para visualização:

- `face_crop.jpg` - Recorte da face
- Coordenadas do bounding box em JSON

---

## Task Principal

**Arquivo:** `app/tasks.py`

```python
def process_deepfake_image(job_id: str, image_path: str, target_model: str = "auto"):
    """
    Pipeline completo de processamento.
  
    1. Atualiza estágio no banco a cada etapa
    2. Detecta e processa face
    3. Executa inferência
    4. Salva resultado
    5. Trata erros graciosamente
    """
```

---

## Executando

### Local (Desenvolvimento)

```bash
cd worker

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis
export DATABASE_URL="sqlite:///../detectar.db"
export REDIS_URL="redis://localhost:6379/0"
export MODEL_PATH="../model/weights"

# Iniciar worker
rq worker deepfake --url redis://localhost:6379/0 --with-scheduler
```

### Múltiplos Workers

```bash
# Terminal 1
rq worker deepfake --url redis://localhost:6379/0 &

# Terminal 2
rq worker deepfake --url redis://localhost:6379/0 &

# Terminal 3
rq worker deepfake --url redis://localhost:6379/0 &
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglx-mesa0 libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["rq", "worker", "deepfake", "--url", "redis://redis:6379/0", "--with-scheduler"]
```

---

## Dependências

```txt
rq>=1.16.0
redis>=5.0.1
sqlalchemy>=2.0.25
pillow>=10.2.0
numpy>=1.26.0
opencv-python-headless>=4.9.0
mediapipe>=0.10.9
torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.12
python-dotenv>=1.0.0
```

### Dependências de Sistema (Linux)

```bash
# Debian/Ubuntu
apt-get install libgl1 libglx-mesa0 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Fedora
dnf install mesa-libGL mesa-libGLX glib2 libSM libXext libXrender libgomp
```

---

## GPU Support

O worker detecta automaticamente CUDA:

```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

### Verificar GPU

```python
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

## Logging

O worker usa logging padrão do Python:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exemplos de logs
logger.info(f"[{job_id}] Stage: model_inference")
logger.info(f"Face detected: ({x}, {y}) {w}x{h}, confidence: {conf:.2f}")
logger.info(f"Inference result: p_fake = {p_fake:.4f}")
```

---

## Métricas de Performance

| Etapa                 | Tempo Médio (CPU) | Tempo Médio (GPU) |
| --------------------- | ------------------ | ------------------ |
| Validação           | ~10ms              | ~10ms              |
| Detecção de face    | ~100ms             | ~50ms              |
| Quality checks        | ~20ms              | ~20ms              |
| Preprocessing         | ~30ms              | ~30ms              |
| **Inferência** | ~500ms             | ~50ms              |
| Calibração          | ~5ms               | ~5ms               |
| Evidências           | ~20ms              | ~20ms              |
| **Total**       | ~700ms             | ~200ms             |

---

## Troubleshooting

### Modelo não encontrado

```
WARNING: Model weights not found. Using random initialization (DEMO MODE).
```

**Solução:** Coloque um arquivo `.pth` em `model/weights/` ou treine um modelo com os notebooks.

### Erro de memória GPU

```
RuntimeError: CUDA out of memory
```

**Solução:** Use um modelo menor (EfficientNet-B0 em vez de B4) ou reduza o batch size.

### MediaPipe falha

```
Error: Could not find file blaze_face_short_range.tflite
```

**Solução:** O modelo é baixado automaticamente na primeira execução. Verifique conexão com internet.
