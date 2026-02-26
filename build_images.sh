#!/bin/bash

# ==============================================================================
# DeepTrustShield - Build de Imagens com Podman
# ==============================================================================
# Este script constrói todas as imagens do sistema usando Podman
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

# Configurações
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Nome base das imagens
IMAGE_PREFIX="${IMAGE_PREFIX:-deeptrust}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

# Função para log
log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         DeepTrustShield - Build de Imagens                   ║"
    echo "║                  Powered by Podman                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Verificar Podman
check_podman() {
    log_info "Verificando Podman..."
    
    if ! command -v podman &> /dev/null; then
        log_error "Podman não encontrado!"
        log_info "Instalando Podman..."
        
        if command -v dnf &> /dev/null; then
            sudo dnf install -y podman podman-compose
        elif command -v apt &> /dev/null; then
            sudo apt install -y podman podman-compose
        else
            log_error "Por favor, instale o Podman manualmente"
            exit 1
        fi
    fi
    
    log_success "Podman $(podman --version | awk '{print $3}') encontrado"
    
    # Verificar podman-compose
    if ! command -v podman-compose &> /dev/null; then
        log_warning "podman-compose não encontrado, instalando..."
        pip install podman-compose
    fi
    log_success "podman-compose disponível"
}

# Construir imagem
build_image() {
    local name=$1
    local context=$2
    local dockerfile=$3
    local full_name="${IMAGE_PREFIX}-${name}:${IMAGE_TAG}"
    
    echo ""
    log_info "Construindo imagem: ${BOLD}${full_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ -n "$REGISTRY" ]; then
        full_name="${REGISTRY}/${full_name}"
    fi
    
    podman build \
        -t "$full_name" \
        -f "$dockerfile" \
        --layers=true \
        --format=docker \
        "$context"
    
    log_success "Imagem ${full_name} construída com sucesso"
    
    # Criar tag latest se não for
    if [ "$IMAGE_TAG" != "latest" ]; then
        local latest_name="${IMAGE_PREFIX}-${name}:latest"
        if [ -n "$REGISTRY" ]; then
            latest_name="${REGISTRY}/${latest_name}"
        fi
        podman tag "$full_name" "$latest_name"
        log_info "Tag latest criada: ${latest_name}"
    fi
}

# Mostrar ajuda
show_help() {
    echo "Uso: $0 [opções] [componente]"
    echo ""
    echo "Componentes:"
    echo "  all       Constrói todas as imagens (padrão)"
    echo "  backend   Constrói apenas o backend"
    echo "  worker    Constrói apenas o worker"
    echo "  frontend  Constrói apenas o frontend"
    echo ""
    echo "Opções:"
    echo "  -t, --tag TAG        Define a tag das imagens (padrão: latest)"
    echo "  -r, --registry REG   Define o registry (ex: docker.io/usuario)"
    echo "  -p, --prefix PREFIX  Define o prefixo das imagens (padrão: deeptrust)"
    echo "  --no-cache           Constrói sem usar cache"
    echo "  -h, --help           Mostra esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0                          # Constrói todas com tag latest"
    echo "  $0 -t v1.0.0                # Constrói todas com tag v1.0.0"
    echo "  $0 -r docker.io/user all    # Constrói com registry"
    echo "  $0 backend                  # Constrói apenas backend"
}

# Parse de argumentos
NO_CACHE=""
COMPONENT="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -p|--prefix)
            IMAGE_PREFIX="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        all|backend|worker|frontend)
            COMPONENT="$1"
            shift
            ;;
        *)
            log_error "Argumento desconhecido: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main
print_banner

echo -e "${BOLD}Configurações:${NC}"
echo "  Prefixo:  ${IMAGE_PREFIX}"
echo "  Tag:      ${IMAGE_TAG}"
echo "  Registry: ${REGISTRY:-local}"
echo "  Build:    ${COMPONENT}"
echo ""

check_podman

START_TIME=$(date +%s)

# Construir imagens
case $COMPONENT in
    all)
        build_image "backend" "./backend" "./backend/Dockerfile"
        build_image "worker" "./worker" "./worker/Dockerfile"
        build_image "frontend" "./frontend" "./frontend/Dockerfile"
        ;;
    backend)
        build_image "backend" "./backend" "./backend/Dockerfile"
        ;;
    worker)
        build_image "worker" "./worker" "./worker/Dockerfile"
        ;;
    frontend)
        build_image "frontend" "./frontend" "./frontend/Dockerfile"
        ;;
esac

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              BUILD CONCLUÍDO COM SUCESSO!                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Tempo total:${NC} ${DURATION} segundos"
echo ""

# Listar imagens criadas
echo -e "${BOLD}Imagens criadas:${NC}"
podman images | grep "${IMAGE_PREFIX}" | head -10
echo ""

echo -e "${BOLD}Próximos passos:${NC}"
echo "  1. Para executar localmente:"
echo "     ./run_podman.sh"
echo ""
echo "  2. Para publicar no registry:"
echo "     ./push_images.sh -r docker.io/seu-usuario"
echo ""
