#!/bin/bash

# ==============================================================================
# DeepTrustShield - Deploy Rápido
# ==============================================================================
# Este script permite instalar e executar o DeepTrustShield em qualquer máquina
# que tenha Podman ou Docker instalado.
#
# Uso:
#   curl -fsSL https://raw.githubusercontent.com/SEU_USUARIO/DeepTrustShield/main/deploy.sh | bash
#
# Ou baixe e execute:
#   wget https://raw.githubusercontent.com/SEU_USUARIO/DeepTrustShield/main/deploy.sh
#   chmod +x deploy.sh
#   ./deploy.sh
# ==============================================================================

set -e

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configurações - ALTERE PARA SEU REGISTRY
REGISTRY="${REGISTRY:-docker.io/SEU_USUARIO}"
IMAGE_PREFIX="${IMAGE_PREFIX:-deeptrust}"
TAG="${TAG:-latest}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/deeptrust}"

# Funções de log
log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              DeepTrustShield - Deploy Rápido                 ║"
    echo "║           Sistema de Detecção de Deepfake                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Detectar container runtime (Podman ou Docker)
detect_runtime() {
    log_info "Detectando container runtime..."
    
    if command -v podman &> /dev/null; then
        RUNTIME="podman"
        COMPOSE_CMD="podman-compose"
        log_success "Podman encontrado"
    elif command -v docker &> /dev/null; then
        RUNTIME="docker"
        COMPOSE_CMD="docker compose"
        log_success "Docker encontrado"
    else
        log_error "Nenhum container runtime encontrado!"
        echo ""
        echo "Por favor, instale Podman ou Docker:"
        echo ""
        echo "  Fedora/RHEL:"
        echo "    sudo dnf install podman podman-compose"
        echo ""
        echo "  Ubuntu/Debian:"
        echo "    sudo apt install podman podman-compose"
        echo "    # ou Docker:"
        echo "    curl -fsSL https://get.docker.com | sh"
        echo ""
        exit 1
    fi
    
    # Verificar compose
    if [ "$RUNTIME" = "podman" ]; then
        if ! command -v podman-compose &> /dev/null; then
            log_info "Instalando podman-compose..."
            pip install --user podman-compose
        fi
    fi
}

# Criar diretório de instalação
setup_directory() {
    log_info "Criando diretório de instalação: $INSTALL_DIR"
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$INSTALL_DIR/model/weights"
    mkdir -p "$INSTALL_DIR/artifacts"
    mkdir -p "$INSTALL_DIR/data"
    
    cd "$INSTALL_DIR"
    
    log_success "Diretórios criados"
}

# Criar docker-compose.yml
create_compose_file() {
    log_info "Criando arquivo de configuração..."
    
    cat > "$INSTALL_DIR/docker-compose.yml" << EOF
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: deeptrust-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  backend:
    image: ${REGISTRY}/${IMAGE_PREFIX}-backend:${TAG}
    container_name: deeptrust-backend
    environment:
      - DATABASE_URL=sqlite:////app/data/detectar.db
      - REDIS_URL=redis://redis:6379/0
      - ARTIFACTS_PATH=/app/artifacts
      - CORS_ORIGINS=http://localhost:3000,http://localhost:5173
    volumes:
      - ./artifacts:/app/artifacts
      - ./data:/app/data
    ports:
      - "8000:8000"
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  worker:
    image: ${REGISTRY}/${IMAGE_PREFIX}-worker:${TAG}
    container_name: deeptrust-worker
    environment:
      - DATABASE_URL=sqlite:////app/data/detectar.db
      - REDIS_URL=redis://redis:6379/0
      - ARTIFACTS_PATH=/app/artifacts
      - MODEL_PATH=/app/model/weights
    volumes:
      - ./artifacts:/app/artifacts
      - ./data:/app/data
      - ./model/weights:/app/model/weights:ro
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  frontend:
    image: ${REGISTRY}/${IMAGE_PREFIX}-frontend:${TAG}
    container_name: deeptrust-frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  redis_data:
EOF

    log_success "Arquivo docker-compose.yml criado"
}

# Baixar imagens
pull_images() {
    log_info "Baixando imagens do registry..."
    
    for img in backend worker frontend; do
        local name="${REGISTRY}/${IMAGE_PREFIX}-${img}:${TAG}"
        log_info "Baixando $name..."
        $RUNTIME pull "$name"
    done
    
    log_success "Todas as imagens baixadas"
}

# Iniciar serviços
start_services() {
    log_info "Iniciando serviços..."
    
    cd "$INSTALL_DIR"
    
    if [ "$RUNTIME" = "podman" ]; then
        podman-compose up -d
    else
        docker compose up -d
    fi
    
    # Aguardar serviços iniciarem
    log_info "Aguardando serviços iniciarem..."
    sleep 5
    
    log_success "Serviços iniciados"
}

# Verificar saúde
check_health() {
    log_info "Verificando saúde dos serviços..."
    
    local healthy=true
    
    # Redis
    if $RUNTIME exec deeptrust-redis redis-cli ping &>/dev/null; then
        echo -e "  Redis:    ${GREEN}●${NC} OK"
    else
        echo -e "  Redis:    ${RED}●${NC} Falha"
        healthy=false
    fi
    
    # Backend
    sleep 2
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs | grep -q "200\|302"; then
        echo -e "  Backend:  ${GREEN}●${NC} OK"
    else
        echo -e "  Backend:  ${YELLOW}●${NC} Iniciando..."
    fi
    
    # Frontend
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200"; then
        echo -e "  Frontend: ${GREEN}●${NC} OK"
    else
        echo -e "  Frontend: ${YELLOW}●${NC} Iniciando..."
    fi
}

# Criar script de controle
create_control_script() {
    cat > "$INSTALL_DIR/deeptrust" << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"

case "$1" in
    start)
        echo "Iniciando DeepTrustShield..."
        if command -v podman-compose &>/dev/null; then
            podman-compose up -d
        else
            docker compose up -d
        fi
        echo "Acesse: http://localhost:3000"
        ;;
    stop)
        echo "Parando DeepTrustShield..."
        if command -v podman-compose &>/dev/null; then
            podman-compose down
        else
            docker compose down
        fi
        ;;
    restart)
        $0 stop
        $0 start
        ;;
    logs)
        if command -v podman-compose &>/dev/null; then
            podman-compose logs -f
        else
            docker compose logs -f
        fi
        ;;
    status)
        if command -v podman &>/dev/null; then
            podman ps --filter "name=deeptrust"
        else
            docker ps --filter "name=deeptrust"
        fi
        ;;
    *)
        echo "Uso: $0 {start|stop|restart|logs|status}"
        exit 1
        ;;
esac
SCRIPT

    chmod +x "$INSTALL_DIR/deeptrust"
    log_success "Script de controle criado: $INSTALL_DIR/deeptrust"
}

# Resumo final
print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         INSTALAÇÃO CONCLUÍDA COM SUCESSO!                    ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Acesse:${NC}"
    echo "  Frontend:  ${BLUE}http://localhost:3000${NC}"
    echo "  Backend:   ${BLUE}http://localhost:8000${NC}"
    echo "  API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${BOLD}Diretório de instalação:${NC} $INSTALL_DIR"
    echo ""
    echo -e "${BOLD}Comandos úteis:${NC}"
    echo "  $INSTALL_DIR/deeptrust start    # Iniciar"
    echo "  $INSTALL_DIR/deeptrust stop     # Parar"
    echo "  $INSTALL_DIR/deeptrust logs     # Ver logs"
    echo "  $INSTALL_DIR/deeptrust status   # Ver status"
    echo ""
    echo -e "${BOLD}Modelos:${NC}"
    echo "  Coloque seus modelos .pth em: $INSTALL_DIR/model/weights/"
    echo ""
}

# Main
main() {
    print_banner
    
    echo -e "${BOLD}Configurações:${NC}"
    echo "  Registry:   $REGISTRY"
    echo "  Tag:        $TAG"
    echo "  Diretório:  $INSTALL_DIR"
    echo ""
    
    read -p "Deseja continuar com a instalação? [S/n]: " confirm
    confirm=${confirm:-S}
    
    if [[ ! "$confirm" =~ ^[Ss]$ ]]; then
        log_info "Instalação cancelada"
        exit 0
    fi
    
    detect_runtime
    setup_directory
    create_compose_file
    pull_images
    start_services
    create_control_script
    check_health
    print_summary
}

# Executar
main "$@"
