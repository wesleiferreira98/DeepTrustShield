# DeepTrustShield

Sistema de detecção de deepfake por foto (selfie) usando EfficientNet-B0.

## Quick Start

```bash
# 1. Clone e entre no diretório
cd detectar

# 2. Copie o arquivo de ambiente
cp .env.example .env

# 3. Suba todos os serviços
docker compose up --build

# 4. Acesse
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

## Estrutura do Projeto

```
detectar/
├── backend/          # FastAPI - API REST
├── worker/           # Processamento assíncrono (RQ + Redis)
├── frontend/         # React - Interface do usuário
├── model/            # Pesos do modelo EfficientNet-B0
└── docker-compose.yml
```

## API Endpoints

| Método | Endpoint                                         | Descrição               |
| ------- | ------------------------------------------------ | ------------------------- |
| POST    | `/v1/deepfake/image/analyze`                   | Inicia análise de imagem |
| GET     | `/v1/deepfake/image/status/{job_id}`           | Status do job             |
| GET     | `/v1/deepfake/image/result/{job_id}`           | Resultado final           |
| GET     | `/v1/deepfake/image/artifacts/{job_id}/{name}` | Baixar artefatos          |

## Pipeline de Processamento

1. **validating_image** - Validação do arquivo
2. **detecting_face** - Detecção facial (MediaPipe)
3. **quality_checks** - Verificação de qualidade
4. **preprocessing** - Crop + Resize 224x224
5. **model_inference** - EfficientNet-B0
6. **calibrating_score** - Aplicação de thresholds
7. **generating_evidence** - Geração de evidências
8. **done** - Concluído

## Níveis de Risco

| Score (p_fake)         | Nível          |
| ---------------------- | --------------- |
| < 0.35                 | 🟢 LOW          |
| 0.35 - 0.65            | 🟡 MEDIUM       |
| ≥ 0.65                | 🔴 HIGH         |
| no_face/multiple_faces | ⚪ UNDETERMINED |

## Desenvolvimento

```bash
# Backend apenas
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload

# Worker apenas
cd worker && pip install -r requirements.txt
rq worker --url redis://localhost:6379/0

# Frontend apenas
cd frontend && npm install && npm run dev
```

## Modelo

O MVP usa EfficientNet-B0 com cabeça customizada:

- Input: 224×224×3 (face crop normalizada)
- Output: sigmoid → p_fake (0-1)

Para usar um modelo treinado, coloque os pesos em:

```
model/weights/efficientnet_b0_deepfake.pth
```

## Roadmap

- [ ] Suporte a vídeo (extração de frames + agregação)
- [ ] Múltiplos modelos (ensemble)
- [ ] API de feedback para retreino
- [ ] Dashboard de métricas
