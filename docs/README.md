# 📚 Documentação DeepTrustShield

Sistema de detecção de deepfake em imagens usando deep learning.

## Índice

| Documento | Descrição |
|-----------|-----------|
| [🏗️ ARCHITECTURE.md](ARCHITECTURE.md) | Visão geral da arquitetura, fluxo de dados e componentes |
| [🖥️ BACKEND.md](BACKEND.md) | API FastAPI, endpoints, configuração e banco de dados |
| [⚙️ WORKER.md](WORKER.md) | Pipeline de processamento, detecção facial e inferência |
| [🎓 TRAINING.md](TRAINING.md) | Notebooks de treinamento, datasets e metodologia |
| [📡 API.md](API.md) | Referência completa da API REST com exemplos |
| [⚖️ FAIRNESS.md](FAIRNESS.md) | Auditoria de viés, fairness e protocolo de avaliação por grupo |
| [🎙️ AUDIO_PLAN.md](AUDIO_PLAN.md) | Planejamento técnico do MVP de detecção de fraude com áudio |

## Quick Start

### 1. Instalação Rápida

```bash
./install.sh
```

### 2. Execução com Podman

```bash
./run_podman.sh start
```

### 3. Acesso

- **Frontend:** http://localhost:3000
- **API:** http://localhost:8000
- **Swagger:** http://localhost:8000/docs

## Stack Tecnológica

| Componente | Tecnologias |
|------------|-------------|
| **Frontend** | React 18, Vite, Nginx |
| **Backend** | FastAPI, SQLAlchemy, Redis |
| **Worker** | PyTorch, timm, MediaPipe, OpenCV |
| **Container** | Podman, podman-compose |
| **ML** | EfficientNet-B0/B3/B4 |

## Licença

MIT
