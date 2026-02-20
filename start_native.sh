#!/bin/bash

# ==============================================================================
# Detectar - Script de Inicialização Nativo (sem Docker)
# ==============================================================================
# Este script inicia todos os serviços necessários para a aplicação web:
# - Redis (banco de dados em memória para filas)
# - Backend FastAPI (API REST)
# - Worker RQ (processamento de tarefas em background)
# - Frontend Vite (interface web)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PIDs para cleanup
REDIS_PID=""
BACKEND_PID=""
WORKER_PID=""
FRONTEND_PID=""

# Função de limpeza
cleanup() {
    echo -e "\n${YELLOW}Encerrando serviços...${NC}"
    
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "Parando Frontend (PID: $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    
    if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "Parando Worker (PID: $WORKER_PID)..."
        kill "$WORKER_PID" 2>/dev/null || true
    fi
    
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Parando Backend (PID: $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    
    # Não paramos o Redis se já estava rodando antes
    if [ "$REDIS_STARTED_HERE" = "true" ] && [ -n "$REDIS_PID" ]; then
        echo "Parando Redis (PID: $REDIS_PID)..."
        kill "$REDIS_PID" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Todos os serviços encerrados.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       DETECTAR - Inicialização            ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Verificar dependências
echo -e "${YELLOW}Verificando dependências...${NC}"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Erro: Python 3 não encontrado. Instale com: sudo apt install python3${NC}"
    exit 1
fi
echo -e "  ✓ Python3 encontrado"

# Verificar Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Erro: Node.js não encontrado. Instale com: sudo apt install nodejs npm${NC}"
    exit 1
fi
echo -e "  ✓ Node.js encontrado"

# Verificar Redis
if ! command -v redis-server &> /dev/null && ! command -v redis-cli &> /dev/null; then
    # Detectar o gerenciador de pacotes
    if command -v dnf &> /dev/null; then
        echo -e "${RED}Erro: Redis não encontrado. Instale com: sudo dnf install redis && sudo systemctl start redis${NC}"
    elif command -v apt &> /dev/null; then
        echo -e "${RED}Erro: Redis não encontrado. Instale com: sudo apt install redis-server${NC}"
    elif command -v pacman &> /dev/null; then
        echo -e "${RED}Erro: Redis não encontrado. Instale com: sudo pacman -S redis && sudo systemctl start redis${NC}"
    else
        echo -e "${RED}Erro: Redis não encontrado. Por favor, instale o Redis para sua distribuição.${NC}"
    fi
    exit 1
fi
echo -e "  ✓ Redis encontrado"

# Verificar se os modelos existem
echo ""
echo -e "${YELLOW}Verificando modelos de IA...${NC}"
MODEL_FOUND=false

if [ -d "model/weights" ]; then
    for model in model/weights/*.pth; do
        if [ -f "$model" ]; then
            MODEL_FOUND=true
            echo -e "  ✓ Modelo encontrado: $(basename "$model")"
        fi
    done
fi

# Verificar modelos na raiz também
for model in *.pth; do
    if [ -f "$model" ]; then
        MODEL_FOUND=true
        echo -e "  ✓ Modelo encontrado (raiz): $model"
    fi
done

if [ "$MODEL_FOUND" = false ]; then
    echo -e "${YELLOW}  ⚠ Nenhum modelo .pth encontrado. A análise de deepfake não funcionará.${NC}"
    echo -e "${YELLOW}  Por favor, treine um modelo usando os notebooks em notebooks/${NC}"
fi

# Criar diretórios necessários
echo ""
echo -e "${YELLOW}Criando diretórios necessários...${NC}"
mkdir -p artifacts
mkdir -p model/weights
echo -e "  ✓ Diretórios criados"

# Verificar/Criar arquivo .env se não existir
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo ""
    echo -e "${YELLOW}Criando arquivo .env a partir do exemplo...${NC}"
    cp .env.example .env
    echo -e "  ✓ .env criado"
fi

# Iniciar Redis
echo ""
echo -e "${YELLOW}Iniciando Redis...${NC}"
REDIS_STARTED_HERE=false

if redis-cli ping &> /dev/null; then
    echo -e "  ✓ Redis já está rodando"
else
    # Tentar iniciar via systemctl primeiro (Fedora, Ubuntu systemd, etc)
    if command -v systemctl &> /dev/null; then
        if sudo systemctl start redis 2>/dev/null || sudo systemctl start redis-server 2>/dev/null; then
            sleep 1
            if redis-cli ping &> /dev/null; then
                echo -e "  ✓ Redis iniciado via systemctl"
            else
                echo -e "${RED}  ✗ Redis iniciado mas não respondendo${NC}"
                exit 1
            fi
        else
            # Fallback: tentar iniciar diretamente
            redis-server --daemonize yes --port 6379
            REDIS_PID=$(pgrep -f "redis-server.*6379" | head -1)
            REDIS_STARTED_HERE=true
            sleep 1
            if redis-cli ping &> /dev/null; then
                echo -e "  ✓ Redis iniciado (PID: $REDIS_PID)"
            else
                echo -e "${RED}  ✗ Falha ao iniciar Redis${NC}"
                exit 1
            fi
        fi
    else
        # Sistema sem systemd
        redis-server --daemonize yes --port 6379
        REDIS_PID=$(pgrep -f "redis-server.*6379" | head -1)
        REDIS_STARTED_HERE=true
        sleep 1
        if redis-cli ping &> /dev/null; then
            echo -e "  ✓ Redis iniciado (PID: $REDIS_PID)"
        else
            echo -e "${RED}  ✗ Falha ao iniciar Redis${NC}"
            exit 1
        fi
    fi
fi

# Iniciar Backend
echo ""
echo -e "${YELLOW}Iniciando Backend (FastAPI)...${NC}"

if [ ! -d "backend/venv" ]; then
    echo -e "  Criando ambiente virtual do backend..."
    python3 -m venv backend/venv
    source backend/venv/bin/activate
    pip install -r backend/requirements.txt
    deactivate
fi

cd backend
source venv/bin/activate
export DATABASE_URL="sqlite:///../detectar.db"
export REDIS_URL="redis://localhost:6379/0"
export ARTIFACTS_PATH="../artifacts"
export MODEL_PATH="../model/weights"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
deactivate
cd ..

sleep 2
if kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo -e "  ✓ Backend iniciado (PID: $BACKEND_PID) em http://localhost:8000"
else
    echo -e "${RED}  ✗ Falha ao iniciar Backend${NC}"
    cleanup
fi

# Iniciar Worker
echo ""
echo -e "${YELLOW}Iniciando Worker (RQ)...${NC}"

if [ ! -d "worker/venv" ]; then
    echo -e "  Criando ambiente virtual do worker..."
    python3 -m venv worker/venv
    source worker/venv/bin/activate
    pip install -r worker/requirements.txt
    deactivate
fi

cd worker
source venv/bin/activate
export DATABASE_URL="sqlite:///../detectar.db"
export REDIS_URL="redis://localhost:6379/0"
export ARTIFACTS_PATH="../artifacts"
export MODEL_PATH="../model/weights"

rq worker deepfake --url redis://localhost:6379/0 --with-scheduler &
WORKER_PID=$!
deactivate
cd ..

sleep 2
if kill -0 "$WORKER_PID" 2>/dev/null; then
    echo -e "  ✓ Worker iniciado (PID: $WORKER_PID)"
else
    echo -e "${RED}  ✗ Falha ao iniciar Worker${NC}"
    cleanup
fi

# Iniciar Frontend
echo ""
echo -e "${YELLOW}Iniciando Frontend (Vite)...${NC}"

if [ ! -d "frontend/node_modules" ]; then
    echo -e "  Instalando dependências do frontend..."
    cd frontend
    npm install
    cd ..
fi

cd frontend
npm run dev -- --host 0.0.0.0 --port 3000 &
FRONTEND_PID=$!
cd ..

sleep 3
if kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo -e "  ✓ Frontend iniciado (PID: $FRONTEND_PID) em http://localhost:3000"
else
    echo -e "${RED}  ✗ Falha ao iniciar Frontend${NC}"
    cleanup
fi

# Resumo final
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}       TODOS OS SERVIÇOS INICIADOS!        ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  ${BLUE}Frontend:${NC}  http://localhost:3000"
echo -e "  ${BLUE}Backend:${NC}   http://localhost:8000"
echo -e "  ${BLUE}API Docs:${NC}  http://localhost:8000/docs"
echo -e "  ${BLUE}Redis:${NC}     localhost:6379"
echo ""
echo -e "${YELLOW}Pressione Ctrl+C para encerrar todos os serviços${NC}"
echo ""

# Aguardar
wait
