#!/bin/bash

# ==============================================================================
# DeepTrustShield - Push de Imagens para Registry
# ==============================================================================
# Este script publica as imagens do sistema para um registry remoto
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

IMAGE_PREFIX="${IMAGE_PREFIX:-deeptrust}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

# Funções de log
log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         DeepTrustShield - Push de Imagens                    ║"
    echo "║              Publicação no Registry                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_help() {
    echo "Uso: $0 [opções]"
    echo ""
    echo "Opções:"
    echo "  -r, --registry REG   Registry de destino (obrigatório)"
    echo "                       Exemplos: docker.io/usuario"
    echo "                                 ghcr.io/usuario"
    echo "                                 quay.io/usuario"
    echo "  -t, --tag TAG        Tag das imagens (padrão: latest)"
    echo "  -p, --prefix PREFIX  Prefixo das imagens (padrão: deeptrust)"
    echo "  --login              Fazer login no registry antes do push"
    echo "  --all-tags           Enviar todas as tags locais"
    echo "  -h, --help           Mostra esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 -r docker.io/meuusuario"
    echo "  $0 -r ghcr.io/meuorg -t v1.0.0"
    echo "  $0 -r quay.io/meuusuario --login"
}

# Parse de argumentos
DO_LOGIN=false
ALL_TAGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -p|--prefix)
            IMAGE_PREFIX="$2"
            shift 2
            ;;
        --login)
            DO_LOGIN=true
            shift
            ;;
        --all-tags)
            ALL_TAGS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Argumento desconhecido: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validar registry
if [ -z "$REGISTRY" ]; then
    log_error "Registry não especificado!"
    echo ""
    show_help
    exit 1
fi

print_banner

echo -e "${BOLD}Configurações:${NC}"
echo "  Registry: ${REGISTRY}"
echo "  Prefixo:  ${IMAGE_PREFIX}"
echo "  Tag:      ${IMAGE_TAG}"
echo ""

# Login se solicitado
if [ "$DO_LOGIN" = true ]; then
    log_info "Fazendo login no registry..."
    
    # Extrair hostname do registry
    REGISTRY_HOST=$(echo "$REGISTRY" | cut -d'/' -f1)
    
    case "$REGISTRY_HOST" in
        docker.io)
            podman login docker.io
            ;;
        ghcr.io)
            echo "Para GitHub Container Registry, use um Personal Access Token"
            podman login ghcr.io
            ;;
        quay.io)
            podman login quay.io
            ;;
        *)
            podman login "$REGISTRY_HOST"
            ;;
    esac
    
    log_success "Login realizado com sucesso"
fi

# Lista de imagens para push
IMAGES=("backend" "worker" "frontend")

# Push das imagens
for img in "${IMAGES[@]}"; do
    local_name="${IMAGE_PREFIX}-${img}:${IMAGE_TAG}"
    remote_name="${REGISTRY}/${IMAGE_PREFIX}-${img}:${IMAGE_TAG}"
    
    echo ""
    log_info "Publicando ${BOLD}${img}${NC}..."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Verificar se a imagem local existe
    if ! podman image exists "$local_name" 2>/dev/null; then
        log_warning "Imagem local não encontrada: $local_name"
        log_info "Execute ./build_images.sh primeiro"
        continue
    fi
    
    # Tag para o registry remoto
    log_info "Criando tag: $remote_name"
    podman tag "$local_name" "$remote_name"
    
    # Push
    log_info "Enviando para $REGISTRY..."
    podman push "$remote_name"
    
    log_success "Imagem $img publicada com sucesso"
    
    # Push tag latest também se não for latest
    if [ "$IMAGE_TAG" != "latest" ]; then
        remote_latest="${REGISTRY}/${IMAGE_PREFIX}-${img}:latest"
        log_info "Enviando tag latest..."
        podman tag "$local_name" "$remote_latest"
        podman push "$remote_latest"
        log_success "Tag latest publicada"
    fi
done

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              PUBLICAÇÃO CONCLUÍDA COM SUCESSO!               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BOLD}Imagens publicadas:${NC}"
for img in "${IMAGES[@]}"; do
    echo "  ${REGISTRY}/${IMAGE_PREFIX}-${img}:${IMAGE_TAG}"
done
echo ""

echo -e "${BOLD}Para usar em outro computador:${NC}"
echo ""
echo "  1. Baixar e executar:"
echo "     export REGISTRY=${REGISTRY}"
echo "     export TAG=${IMAGE_TAG}"
echo "     podman-compose -f podman-compose.yml up -d"
echo ""
echo "  2. Ou usar diretamente:"
echo "     podman pull ${REGISTRY}/${IMAGE_PREFIX}-backend:${IMAGE_TAG}"
echo "     podman pull ${REGISTRY}/${IMAGE_PREFIX}-worker:${IMAGE_TAG}"
echo "     podman pull ${REGISTRY}/${IMAGE_PREFIX}-frontend:${IMAGE_TAG}"
echo ""
