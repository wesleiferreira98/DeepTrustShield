#!/bin/bash

# ==============================================================================
# DeepTrustShield - Executar com Podman
# ==============================================================================
# Este script inicia todos os serviços do sistema usando Podman
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
IMAGE_TAG="${TAG:-latest}"
REGISTRY="${REGISTRY:-}"
COMPOSE_FILE="podman-compose.yml"

# Funções de log
log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              DeepTrustShield - Podman                        ║"
    echo "║            Sistema de Detecção de Deepfake                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_help() {
    echo "Uso: $0 [comando] [opções]"
    echo ""
    echo "Comandos:"
    echo "  start       Inicia todos os serviços (padrão)"
    echo "  stop        Para todos os serviços"
    echo "  restart     Reinicia todos os serviços"
    echo "  logs        Mostra logs de todos os serviços"
    echo "  status      Mostra status dos containers"
    echo "  build       Constrói e inicia os serviços"
    echo "  pull        Baixa imagens do registry"
    echo "  clean       Remove containers e volumes"
    echo ""
    echo "Opções:"
    echo "  -d, --detach     Executa em background"
    echo "  -f, --follow     Segue os logs (com comando logs)"
    echo "  -r, --registry   Usa imagens de um registry remoto"
    echo "  -t, --tag        Tag das imagens (padrão: latest)"
    echo "  -h, --help       Mostra esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0                     # Inicia em foreground"
    echo "  $0 start -d            # Inicia em background"
    echo "  $0 logs -f             # Segue os logs"
    echo "  $0 stop                # Para tudo"
    echo "  $0 -r docker.io/user   # Usa imagens remotas"
}

# Verificar Podman
check_requirements() {
    if ! command -v podman &> /dev/null; then
        log_error "Podman não encontrado!"
        log_info "Instale com: sudo dnf install podman podman-compose"
        exit 1
    fi
    
    if ! command -v podman-compose &> /dev/null; then
        log_warning "podman-compose não encontrado, tentando instalar..."
        pip install podman-compose || {
            log_error "Falha ao instalar podman-compose"
            log_info "Instale manualmente: pip install podman-compose"
            exit 1
        }
    fi
}

# Verificar se imagens existem
check_images() {
    local missing=false
    
    for img in backend worker frontend; do
        local name="${IMAGE_PREFIX}-${img}:${IMAGE_TAG}"
        if [ -n "$REGISTRY" ]; then
            name="${REGISTRY}/${name}"
        fi
        
        if ! podman image exists "$name" 2>/dev/null; then
            missing=true
            log_warning "Imagem não encontrada: $name"
        fi
    done
    
    if [ "$missing" = true ]; then
        echo ""
        if [ -n "$REGISTRY" ]; then
            log_info "Execute './run_podman.sh pull' para baixar as imagens"
        else
            log_info "Execute './build_images.sh' para construir as imagens"
        fi
        echo ""
        
        read -p "Deseja construir as imagens agora? [S/n]: " build_now
        build_now=${build_now:-S}
        if [[ "$build_now" =~ ^[Ss]$ ]]; then
            ./build_images.sh
        else
            exit 1
        fi
    fi
}

# Verificar modelos
check_models() {
    local model_found=false
    
    # Verificar na pasta model/weights
    if ls model/weights/*.pth 1> /dev/null 2>&1; then
        model_found=true
    fi
    
    # Verificar na raiz
    if ls *.pth 1> /dev/null 2>&1; then
        model_found=true
    fi
    
    if [ "$model_found" = false ]; then
        log_warning "Nenhum modelo .pth encontrado!"
        log_info "O sistema precisa de modelos para funcionar corretamente."
        log_info "Coloque arquivos .pth em ./model/weights/ ou na raiz do projeto."
        echo ""
    fi
}

# Comandos
cmd_start() {
    check_images
    check_models
    
    log_info "Iniciando serviços..."
    
    export TAG="${IMAGE_TAG}"
    export REGISTRY="${REGISTRY:-localhost}"
    
    if [ "$DETACH" = true ]; then
        podman-compose -f "$COMPOSE_FILE" up -d
    else
        podman-compose -f "$COMPOSE_FILE" up
    fi
}

cmd_stop() {
    log_info "Parando serviços..."
    podman-compose -f "$COMPOSE_FILE" down
    log_success "Serviços parados"
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_logs() {
    local follow_flag=""
    if [ "$FOLLOW" = true ]; then
        follow_flag="-f"
    fi
    
    podman-compose -f "$COMPOSE_FILE" logs $follow_flag
}

cmd_status() {
    echo -e "${BOLD}Status dos Containers:${NC}"
    echo ""
    podman ps --filter "name=deeptrust" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    
    # Verificar saúde
    echo -e "${BOLD}Verificação de Saúde:${NC}"
    
    # Redis
    if podman exec deeptrust-redis redis-cli ping &>/dev/null; then
        echo -e "  Redis:    ${GREEN}●${NC} Saudável"
    else
        echo -e "  Redis:    ${RED}●${NC} Indisponível"
    fi
    
    # Backend
    if curl -s http://localhost:8000/health &>/dev/null; then
        echo -e "  Backend:  ${GREEN}●${NC} Saudável"
    else
        echo -e "  Backend:  ${RED}●${NC} Indisponível"
    fi
    
    # Frontend
    if curl -s http://localhost:3000 &>/dev/null; then
        echo -e "  Frontend: ${GREEN}●${NC} Saudável"
    else
        echo -e "  Frontend: ${RED}●${NC} Indisponível"
    fi
    
    echo ""
}

cmd_build() {
    log_info "Construindo e iniciando serviços..."
    
    export TAG="${IMAGE_TAG}"
    export REGISTRY="${REGISTRY:-localhost}"
    
    if [ "$DETACH" = true ]; then
        podman-compose -f "$COMPOSE_FILE" up --build -d
    else
        podman-compose -f "$COMPOSE_FILE" up --build
    fi
}

cmd_pull() {
    if [ -z "$REGISTRY" ]; then
        log_error "Registry não especificado!"
        log_info "Use: $0 pull -r docker.io/usuario"
        exit 1
    fi
    
    log_info "Baixando imagens de $REGISTRY..."
    
    for img in backend worker frontend; do
        local name="${REGISTRY}/${IMAGE_PREFIX}-${img}:${IMAGE_TAG}"
        log_info "Baixando $name..."
        podman pull "$name"
    done
    
    log_success "Imagens baixadas com sucesso"
}

cmd_clean() {
    log_warning "Isso irá remover todos os containers e volumes do DeepTrustShield!"
    read -p "Tem certeza? [s/N]: " confirm
    
    if [[ "$confirm" =~ ^[Ss]$ ]]; then
        log_info "Removendo containers..."
        podman-compose -f "$COMPOSE_FILE" down -v
        
        log_info "Removendo imagens locais..."
        podman rmi $(podman images | grep "${IMAGE_PREFIX}" | awk '{print $3}') 2>/dev/null || true
        
        log_success "Limpeza concluída"
    else
        log_info "Operação cancelada"
    fi
}

# Parse de argumentos
COMMAND="start"
DETACH=false
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|logs|status|build|pull|clean)
            COMMAND="$1"
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
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

# Main
print_banner

check_requirements

case $COMMAND in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    logs)
        cmd_logs
        ;;
    status)
        cmd_status
        ;;
    build)
        cmd_build
        ;;
    pull)
        cmd_pull
        ;;
    clean)
        cmd_clean
        ;;
esac

# Mostrar informações se iniciou
if [ "$COMMAND" = "start" ] || [ "$COMMAND" = "build" ]; then
    if [ "$DETACH" = true ]; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║           SERVIÇOS INICIADOS COM SUCESSO!                    ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BOLD}Acesse:${NC}"
        echo "  Frontend:  ${BLUE}http://localhost:3000${NC}"
        echo "  Backend:   ${BLUE}http://localhost:8000${NC}"
        echo "  API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
        echo ""
        echo -e "${BOLD}Comandos úteis:${NC}"
        echo "  ./run_podman.sh logs -f    # Ver logs em tempo real"
        echo "  ./run_podman.sh status     # Ver status dos serviços"
        echo "  ./run_podman.sh stop       # Parar serviços"
        echo ""
    fi
fi
