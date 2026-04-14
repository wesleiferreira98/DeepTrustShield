# Backend - API FastAPI

## Visão Geral

O backend é uma API REST construída com FastAPI que gerencia o ciclo de vida dos jobs de análise de deepfake. Ele recebe uploads de imagens, enfileira tarefas para processamento assíncrono e serve os resultados.

## Estrutura do Projeto

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Entry point e configuração do FastAPI
│   ├── config.py            # Configurações (Pydantic Settings)
│   ├── api/
│   │   ├── __init__.py
│   │   └── deepfake.py      # Endpoints da API de deepfake
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py      # Conexão com banco de dados
│   │   └── models.py        # Modelos SQLAlchemy
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Schemas Pydantic (request/response)
│   └── services/
│       ├── __init__.py
│       └── job_service.py   # Lógica de negócio
├── Dockerfile
├── requirements.txt
└── venv/                    # Ambiente virtual (local)
```

---

## Configuração

### `app/config.py`

```python
class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./detectar.db"
  
    # Redis
    redis_url: str = "redis://localhost:6379/0"
  
    # Artifacts
    artifacts_path: str = "./artifacts"
  
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
  
    # Model
    model_version: str = "efficientnet-b0-v1"
```

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
DATABASE_URL=sqlite:///./detectar.db
REDIS_URL=redis://localhost:6379/0
ARTIFACTS_PATH=./artifacts
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

---

## API Endpoints

### `POST /v1/deepfake/image/analyze`

Inicia uma análise de deepfake em uma imagem.

**Request:**

```http
POST /v1/deepfake/image/analyze
Content-Type: multipart/form-data

file: <binary>
target_model: auto | efficientnet_b0 | efficientnet_b0_v4 | efficientnet_b0_v5 | efficientnet_b3 | efficientnet_b4 | alexnet_v82 | alexnet_v81 | alexnet_v8 | alexnet_v7 | alexnet_v6 | alexnet | mobilenet_v3_small_v1 | mobilenet_v3_small | mobilenet
```

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Análise iniciada. Use o endpoint de status para acompanhar."
}
```

**Validações:**

- Tipos aceitos: `image/jpeg`, `image/png`, `image/webp`
- Tamanho máximo: configurável (padrão do servidor)

---

### `GET /v1/deepfake/image/status/{job_id}`

Consulta o status atual de um job.

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "stage": "model_inference",
  "progress": 70,
  "message": "Analisando com modelo de IA...",
  "steps": [
    {"id": "validating_image", "label": "Validando imagem", "state": "completed"},
    {"id": "detecting_face", "label": "Detectando face", "state": "completed"},
    {"id": "quality_checks", "label": "Verificando qualidade", "state": "completed"},
    {"id": "preprocessing", "label": "Pré-processando", "state": "completed"},
    {"id": "model_inference", "label": "Analisando com IA", "state": "running"},
    {"id": "calibrating_score", "label": "Calibrando resultado", "state": "pending"},
    {"id": "generating_evidence", "label": "Gerando evidências", "state": "pending"}
  ],
  "preview": {
    "bbox": {"x": 25.5, "y": 10.2, "width": 45.0, "height": 60.5},
    "face_crop_url": "/artifacts/550e8400.../face_crop.jpg"
  }
}
```

**Estágios do Pipeline:**

| Stage                   | Progress | Descrição                    |
| ----------------------- | -------- | ------------------------------ |
| `pending`             | 0%       | Aguardando na fila             |
| `validating_image`    | 10%      | Validando formato              |
| `detecting_face`      | 25%      | Detectando face com MediaPipe  |
| `quality_checks`      | 40%      | Verificando blur, brilho       |
| `preprocessing`       | 55%      | Crop + resize + normalização |
| `model_inference`     | 70%      | Inferência com EfficientNet   |
| `calibrating_score`   | 85%      | Aplicando thresholds           |
| `generating_evidence` | 95%      | Salvando artefatos             |
| `done`                | 100%     | Concluído                     |
| `error`               | -        | Erro no processamento          |

---

### `GET /v1/deepfake/image/result/{job_id}`

Obtém o resultado final da análise.

**Response (sucesso):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "p_fake": 0.8723,
  "risk_level": "high",
  "flags": {
    "no_face": false,
    "multiple_faces": false,
    "blur_high": false,
    "low_light": false,
    "face_too_small": false
  },
  "evidence": {
    "face_crop_url": "/artifacts/550e8400.../face_crop.jpg",
    "bbox": {"x": 25.5, "y": 10.2, "width": 45.0, "height": 60.5}
  },
  "model_version": "efficientnet_b0_v5",
  "created_at": "2026-02-26T10:30:00Z",
  "processing_time_ms": 2340
}
```

**Níveis de Risco:**

| Risk Level       | p_fake Range | Interpretação    |
| ---------------- | ------------ | ------------------ |
| `low`          | < 0.50       | Provavelmente real |
| `medium`       | 0.50 - 0.80  | Inconclusivo       |
| `high`         | ≥ 0.80      | Provavelmente fake |
| `undetermined` | N/A          | Sem face detectada |

---

### `GET /v1/deepfake/image/artifacts/{job_id}/{name}`

Baixa um artefato específico de um job.

**Exemplos:**

```
GET /v1/deepfake/image/artifacts/550e8400.../face_crop.jpg
GET /v1/deepfake/image/artifacts/550e8400.../original.jpg
```

---

### `GET /health`

Health check do serviço.

```json
{"status": "healthy"}
```

---

## Schemas (Pydantic)

### `app/models/schemas.py`

```python
class JobStage(str, Enum):
    PENDING = "pending"
    VALIDATING_IMAGE = "validating_image"
    DETECTING_FACE = "detecting_face"
    QUALITY_CHECKS = "quality_checks"
    PREPROCESSING = "preprocessing"
    MODEL_INFERENCE = "model_inference"
    CALIBRATING_SCORE = "calibrating_score"
    GENERATING_EVIDENCE = "generating_evidence"
    DONE = "done"
    ERROR = "error"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNDETERMINED = "undetermined"

class Flags(BaseModel):
    no_face: bool = False
    multiple_faces: bool = False
    blur_high: bool = False
    low_light: bool = False
    face_too_small: bool = False

class ResultResponse(BaseModel):
    job_id: str
    p_fake: Optional[float]
    risk_level: RiskLevel
    flags: Flags
    evidence: Evidence
    model_version: str
    created_at: datetime
    processing_time_ms: Optional[int]
```

---

## Banco de Dados

### Tabelas SQLAlchemy

```python
# Job - Estado do processamento
class Job(Base):
    __tablename__ = "jobs"
  
    id = Column(String, primary_key=True)
    stage = Column(String, default="pending")
    progress = Column(Integer, default=0)
    original_path = Column(String)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

# Result - Resultado da análise
class Result(Base):
    __tablename__ = "results"
  
    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    p_fake = Column(Float, nullable=True)
    risk_level = Column(String)
    flags_json = Column(String)  # JSON serializado
    model_version = Column(String)

# Artifact - Artefatos gerados
class Artifact(Base):
    __tablename__ = "artifacts"
  
    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"))
    face_crop_path = Column(String, nullable=True)
    bbox_json = Column(String, nullable=True)  # JSON com coordenadas
```

---

## Executando

### Desenvolvimento

```bash
cd backend

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Produção

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Dependências

```txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
sqlalchemy>=2.0.25
aiosqlite>=0.19.0
redis>=5.0.1
rq>=1.16.0
pydantic>=2.5.3
pydantic-settings>=2.1.0
pillow>=10.2.0
python-dotenv>=1.0.0
aiofiles>=23.2.1
```

---

## Testes

```bash
# Instalar pytest
pip install pytest httpx

# Executar testes
pytest tests/ -v
```

### Exemplo de teste:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_analyze_invalid_type():
    response = client.post(
        "/v1/deepfake/image/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
```
