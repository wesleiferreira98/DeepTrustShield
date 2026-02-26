#!/bin/bash

# ==============================================================================
# DeepTrustShield - Script de Instalação Assistida (Linux/MacOS)
# ==============================================================================
# Este script instala todas as dependências e configura o sistema de detecção
# de deepfake DeepTrustShield de forma totalmente assistida.
# ==============================================================================

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Diretório do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Variáveis de controle
INSTALL_DOCKER=false
INSTALL_NATIVE=false
PYTHON_CMD=""
PIP_CMD=""
OS_TYPE=""
PKG_MANAGER=""

# ==============================================================================
# Funções Auxiliares
# ==============================================================================

print_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║       ██████╗ ███████╗███████╗██████╗ ████████╗██████╗ ██╗   ██╗███████╗████████╗  ║"
    echo "║       ██╔══██╗██╔════╝██╔════╝██╔══██╗╚══██╔══╝██╔══██╗██║   ██║██╔════╝╚══██╔══╝  ║"
    echo "║       ██║  ██║█████╗  █████╗  ██████╔╝   ██║   ██████╔╝██║   ██║███████╗   ██║     ║"
    echo "║       ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝    ██║   ██╔══██╗██║   ██║╚════██║   ██║     ║"
    echo "║       ██████╔╝███████╗███████╗██║        ██║   ██║  ██║╚██████╔╝███████║   ██║     ║"
    echo "║       ╚═════╝ ╚══════╝╚══════╝╚═╝        ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝     ║"
    echo "║                                                                  ║"
    echo "║              Sistema de Detecção de Deepfake                     ║"
    echo "║                   Instalação Assistida                           ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${MAGENTA}➤ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

ask_yes_no() {
    local prompt="$1"
    local default="${2:-y}"
    local response
    
    if [ "$default" = "y" ]; then
        prompt="$prompt [S/n]: "
    else
        prompt="$prompt [s/N]: "
    fi
    
    read -p "$prompt" response
    response=${response:-$default}
    
    case "$response" in
        [sS]|[yY]|[sS][iI][mM]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

press_enter() {
    echo ""
    read -p "Pressione ENTER para continuar..." 
}

# ==============================================================================
# Detecção do Sistema Operacional
# ==============================================================================

detect_os() {
    print_step "Detectando Sistema Operacional"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        
        # Detectar distribuição e gerenciador de pacotes
        if command -v apt-get &> /dev/null; then
            PKG_MANAGER="apt"
            print_info "Sistema detectado: Debian/Ubuntu (apt)"
        elif command -v dnf &> /dev/null; then
            PKG_MANAGER="dnf"
            print_info "Sistema detectado: Fedora/RHEL (dnf)"
        elif command -v yum &> /dev/null; then
            PKG_MANAGER="yum"
            print_info "Sistema detectado: CentOS/RHEL (yum)"
        elif command -v pacman &> /dev/null; then
            PKG_MANAGER="pacman"
            print_info "Sistema detectado: Arch Linux (pacman)"
        elif command -v zypper &> /dev/null; then
            PKG_MANAGER="zypper"
            print_info "Sistema detectado: openSUSE (zypper)"
        else
            print_warning "Gerenciador de pacotes não identificado"
            PKG_MANAGER="unknown"
        fi
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        PKG_MANAGER="brew"
        print_info "Sistema detectado: macOS"
        
        if ! command -v brew &> /dev/null; then
            print_warning "Homebrew não encontrado. Será instalado automaticamente."
        fi
    else
        print_error "Sistema operacional não suportado: $OSTYPE"
        exit 1
    fi
    
    print_success "Detecção concluída"
}

# ==============================================================================
# Instalação do Homebrew (macOS)
# ==============================================================================

install_homebrew() {
    if [[ "$OS_TYPE" == "macos" ]] && ! command -v brew &> /dev/null; then
        print_step "Instalando Homebrew"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        print_success "Homebrew instalado"
    fi
}

# ==============================================================================
# Verificação e Instalação de Dependências
# ==============================================================================

check_install_python() {
    print_step "Verificando Python"
    
    # Tentar encontrar Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        PIP_CMD="pip"
    fi
    
    if [ -n "$PYTHON_CMD" ]; then
        local version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
        local major=$(echo $version | cut -d. -f1)
        local minor=$(echo $version | cut -d. -f2)
        
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            print_success "Python $version encontrado"
            return 0
        else
            print_warning "Python $version encontrado, mas versão 3.8+ é recomendada"
        fi
    fi
    
    if ask_yes_no "Python 3.8+ não encontrado. Deseja instalar agora?"; then
        case "$PKG_MANAGER" in
            apt)
                print_info "Executando: sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
                sudo apt update && sudo apt install -y python3 python3-pip python3-venv
                ;;
            dnf)
                print_info "Executando: sudo dnf install -y python3 python3-pip python3-virtualenv"
                sudo dnf install -y python3 python3-pip python3-virtualenv
                ;;
            yum)
                print_info "Executando: sudo yum install -y python3 python3-pip"
                sudo yum install -y python3 python3-pip
                ;;
            pacman)
                print_info "Executando: sudo pacman -Sy --noconfirm python python-pip"
                sudo pacman -Sy --noconfirm python python-pip
                ;;
            zypper)
                print_info "Executando: sudo zypper install -y python3 python3-pip"
                sudo zypper install -y python3 python3-pip
                ;;
            brew)
                print_info "Executando: brew install python@3.11"
                brew install python@3.11
                ;;
            *)
                print_error "Não foi possível instalar Python automaticamente."
                print_info "Por favor, instale Python 3.8+ manualmente e execute este script novamente."
                exit 1
                ;;
        esac
        
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
        print_success "Python instalado com sucesso"
    else
        print_error "Python é necessário para continuar."
        exit 1
    fi
}

check_install_nodejs() {
    print_step "Verificando Node.js"
    
    if command -v node &> /dev/null; then
        local version=$(node --version | sed 's/v//')
        local major=$(echo $version | cut -d. -f1)
        
        if [ "$major" -ge 18 ]; then
            print_success "Node.js v$version encontrado"
            
            if command -v npm &> /dev/null; then
                print_success "npm $(npm --version) encontrado"
                return 0
            fi
        else
            print_warning "Node.js v$version encontrado, mas versão 18+ é recomendada"
        fi
    fi
    
    if ask_yes_no "Node.js 18+ não encontrado. Deseja instalar agora?"; then
        case "$PKG_MANAGER" in
            apt)
                print_info "Instalando Node.js via NodeSource..."
                curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
                sudo apt install -y nodejs
                ;;
            dnf)
                print_info "Executando: sudo dnf install -y nodejs npm"
                sudo dnf install -y nodejs npm
                ;;
            yum)
                print_info "Instalando Node.js via NodeSource..."
                curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
                sudo yum install -y nodejs
                ;;
            pacman)
                print_info "Executando: sudo pacman -Sy --noconfirm nodejs npm"
                sudo pacman -Sy --noconfirm nodejs npm
                ;;
            zypper)
                print_info "Executando: sudo zypper install -y nodejs npm"
                sudo zypper install -y nodejs npm
                ;;
            brew)
                print_info "Executando: brew install node"
                brew install node
                ;;
            *)
                print_error "Não foi possível instalar Node.js automaticamente."
                print_info "Por favor, instale Node.js 18+ manualmente e execute este script novamente."
                exit 1
                ;;
        esac
        
        print_success "Node.js instalado com sucesso"
    else
        print_error "Node.js é necessário para o frontend."
        exit 1
    fi
}

check_install_redis() {
    print_step "Verificando Redis"
    
    if command -v redis-server &> /dev/null || command -v redis-cli &> /dev/null; then
        print_success "Redis encontrado"
        return 0
    fi
    
    if ask_yes_no "Redis não encontrado. Deseja instalar agora?"; then
        case "$PKG_MANAGER" in
            apt)
                print_info "Executando: sudo apt install -y redis-server"
                sudo apt install -y redis-server
                ;;
            dnf)
                print_info "Executando: sudo dnf install -y redis"
                sudo dnf install -y redis
                ;;
            yum)
                print_info "Executando: sudo yum install -y redis"
                sudo yum install -y redis
                ;;
            pacman)
                print_info "Executando: sudo pacman -Sy --noconfirm redis"
                sudo pacman -Sy --noconfirm redis
                ;;
            zypper)
                print_info "Executando: sudo zypper install -y redis"
                sudo zypper install -y redis
                ;;
            brew)
                print_info "Executando: brew install redis"
                brew install redis
                ;;
            *)
                print_error "Não foi possível instalar Redis automaticamente."
                print_info "Por favor, instale Redis manualmente e execute este script novamente."
                exit 1
                ;;
        esac
        
        print_success "Redis instalado com sucesso"
    else
        print_error "Redis é necessário para o sistema de filas."
        exit 1
    fi
}

check_install_docker() {
    print_step "Verificando Docker (opcional)"
    
    if command -v docker &> /dev/null; then
        local version=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_success "Docker $version encontrado"
        
        if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
            print_success "Docker Compose encontrado"
            INSTALL_DOCKER=true
            return 0
        fi
    fi
    
    print_info "Docker não é obrigatório, mas facilita a execução."
    
    if ask_yes_no "Deseja instalar Docker?" "n"; then
        case "$PKG_MANAGER" in
            apt)
                print_info "Instalando Docker via repositório oficial..."
                sudo apt update
                sudo apt install -y ca-certificates curl gnupg
                sudo install -m 0755 -d /etc/apt/keyrings
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
                sudo chmod a+r /etc/apt/keyrings/docker.gpg
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
                sudo apt update
                sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
                sudo usermod -aG docker $USER
                ;;
            dnf)
                print_info "Executando: sudo dnf install -y docker docker-compose"
                sudo dnf install -y docker docker-compose
                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker $USER
                ;;
            pacman)
                print_info "Executando: sudo pacman -Sy --noconfirm docker docker-compose"
                sudo pacman -Sy --noconfirm docker docker-compose
                sudo systemctl enable docker
                sudo systemctl start docker
                sudo usermod -aG docker $USER
                ;;
            brew)
                print_info "Por favor, baixe Docker Desktop de https://docker.com/products/docker-desktop"
                ;;
            *)
                print_warning "Instalação manual de Docker necessária."
                ;;
        esac
        
        INSTALL_DOCKER=true
        print_success "Docker instalado"
    else
        print_info "Pulando instalação do Docker"
    fi
}

# ==============================================================================
# Configuração do Ambiente
# ==============================================================================

setup_directories() {
    print_step "Criando Diretórios Necessários"
    
    mkdir -p artifacts
    print_success "Diretório artifacts/ criado"
    
    mkdir -p model/weights
    print_success "Diretório model/weights/ criado"
    
    mkdir -p backend/data
    print_success "Diretório backend/data/ criado"
}

setup_env_file() {
    print_step "Configurando Variáveis de Ambiente"
    
    if [ -f ".env" ]; then
        print_info "Arquivo .env já existe"
        if ask_yes_no "Deseja recriá-lo?" "n"; then
            rm -f .env
        else
            print_info "Mantendo .env existente"
            return 0
        fi
    fi
    
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success "Arquivo .env criado a partir do exemplo"
    else
        # Criar .env padrão
        cat > .env << EOF
# DeepTrustShield - Configurações de Ambiente

# Banco de dados
DATABASE_URL=sqlite:///./detectar.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Diretórios
ARTIFACTS_PATH=./artifacts
MODEL_PATH=./model/weights

# CORS (separado por vírgula)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Servidor
HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=3000
EOF
        print_success "Arquivo .env criado com configurações padrão"
    fi
}

# ==============================================================================
# Instalação do Backend
# ==============================================================================

install_backend() {
    print_step "Instalando Backend (FastAPI)"
    
    cd "$SCRIPT_DIR/backend"
    
    # Criar ambiente virtual
    if [ -d "venv" ]; then
        print_info "Ambiente virtual já existe"
        if ask_yes_no "Deseja recriá-lo?" "n"; then
            rm -rf venv
        fi
    fi
    
    if [ ! -d "venv" ]; then
        print_info "Criando ambiente virtual..."
        $PYTHON_CMD -m venv venv
        print_success "Ambiente virtual criado"
    fi
    
    # Ativar e instalar dependências
    print_info "Instalando dependências do backend..."
    source venv/bin/activate
    
    # Atualizar pip
    pip install --upgrade pip
    
    # Instalar dependências
    pip install -r requirements.txt
    
    deactivate
    
    print_success "Backend instalado com sucesso"
    cd "$SCRIPT_DIR"
}

# ==============================================================================
# Instalação do Worker
# ==============================================================================

install_worker() {
    print_step "Instalando Worker (Processamento de IA)"
    
    cd "$SCRIPT_DIR/worker"
    
    # Criar ambiente virtual
    if [ -d "venv" ]; then
        print_info "Ambiente virtual já existe"
        if ask_yes_no "Deseja recriá-lo?" "n"; then
            rm -rf venv
        fi
    fi
    
    if [ ! -d "venv" ]; then
        print_info "Criando ambiente virtual..."
        $PYTHON_CMD -m venv venv
        print_success "Ambiente virtual criado"
    fi
    
    # Ativar e instalar dependências
    print_info "Instalando dependências do worker (pode demorar devido ao PyTorch)..."
    source venv/bin/activate
    
    # Atualizar pip
    pip install --upgrade pip
    
    # Instalar dependências (PyTorch pode demorar)
    pip install -r requirements.txt
    
    deactivate
    
    print_success "Worker instalado com sucesso"
    cd "$SCRIPT_DIR"
}

# ==============================================================================
# Instalação do Frontend
# ==============================================================================

install_frontend() {
    print_step "Instalando Frontend (React + Vite)"
    
    cd "$SCRIPT_DIR/frontend"
    
    # Verificar se node_modules existe
    if [ -d "node_modules" ]; then
        print_info "Dependências do frontend já instaladas"
        if ask_yes_no "Deseja reinstalar?" "n"; then
            rm -rf node_modules
        fi
    fi
    
    if [ ! -d "node_modules" ]; then
        print_info "Instalando dependências do frontend..."
        npm install
        print_success "Dependências instaladas"
    fi
    
    # Build de produção (opcional)
    if ask_yes_no "Deseja fazer build de produção do frontend?" "n"; then
        print_info "Executando build de produção..."
        npm run build
        print_success "Build de produção concluído (pasta dist/)"
    fi
    
    print_success "Frontend instalado com sucesso"
    cd "$SCRIPT_DIR"
}

# ==============================================================================
# Verificação de Modelos
# ==============================================================================

check_models() {
    print_step "Verificando Modelos de IA"
    
    MODEL_FOUND=false
    
    # Verificar modelos na pasta model/weights
    if [ -d "model/weights" ]; then
        for model in model/weights/*.pth; do
            if [ -f "$model" ]; then
                MODEL_FOUND=true
                print_success "Modelo encontrado: $(basename "$model")"
            fi
        done
    fi
    
    # Verificar modelos na raiz
    for model in *.pth; do
        if [ -f "$model" ]; then
            MODEL_FOUND=true
            print_success "Modelo encontrado (raiz): $model"
        fi
    done
    
    if [ "$MODEL_FOUND" = false ]; then
        print_warning "Nenhum modelo .pth encontrado!"
        print_info "O sistema precisa de um modelo treinado para funcionar."
        print_info "Opções:"
        echo "  1. Coloque um modelo .pth na pasta model/weights/"
        echo "  2. Treine um modelo usando os notebooks em notebooks/"
        echo "  3. Baixe um modelo pré-treinado"
    fi
}

# ==============================================================================
# Resumo e Próximos Passos
# ==============================================================================

print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                  ║${NC}"
    echo -e "${GREEN}║            ✓ INSTALAÇÃO CONCLUÍDA COM SUCESSO!                   ║${NC}"
    echo -e "${GREEN}║                                                                  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${BOLD}Componentes Instalados:${NC}"
    echo -e "  ${GREEN}✓${NC} Python $(${PYTHON_CMD} --version 2>&1 | awk '{print $2}')"
    echo -e "  ${GREEN}✓${NC} Node.js $(node --version)"
    echo -e "  ${GREEN}✓${NC} Redis"
    echo -e "  ${GREEN}✓${NC} Backend (FastAPI)"
    echo -e "  ${GREEN}✓${NC} Worker (PyTorch + MediaPipe)"
    echo -e "  ${GREEN}✓${NC} Frontend (React + Vite)"
    
    echo ""
    echo -e "${BOLD}Para iniciar o sistema:${NC}"
    echo ""
    echo -e "  ${CYAN}Modo Nativo (desenvolvimento):${NC}"
    echo -e "    ./start_native.sh"
    echo ""
    
    if [ "$INSTALL_DOCKER" = true ]; then
        echo -e "  ${CYAN}Modo Docker (produção):${NC}"
        echo -e "    docker compose up --build"
        echo ""
    fi
    
    echo -e "${BOLD}Endpoints após iniciar:${NC}"
    echo -e "  Frontend:  ${BLUE}http://localhost:3000${NC}"
    echo -e "  Backend:   ${BLUE}http://localhost:8000${NC}"
    echo -e "  API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
    echo ""
}

start_system_prompt() {
    if ask_yes_no "Deseja iniciar o sistema agora?"; then
        echo ""
        print_info "Iniciando DeepTrustShield..."
        echo ""
        ./start_native.sh
    else
        echo ""
        print_info "Para iniciar mais tarde, execute: ./start_native.sh"
        echo ""
    fi
}

# ==============================================================================
# Menu Principal
# ==============================================================================

show_menu() {
    echo -e "${BOLD}Escolha o tipo de instalação:${NC}"
    echo ""
    echo "  [1] Instalação Completa (Recomendado)"
    echo "      Instala todas as dependências e configura o ambiente"
    echo ""
    echo "  [2] Instalação Mínima (Apenas dependências Python)"
    echo "      Útil se Node.js e Redis já estão instalados"
    echo ""
    echo "  [3] Atualizar Dependências"
    echo "      Reinstala as dependências sem verificar sistema"
    echo ""
    echo "  [4] Verificar Sistema"
    echo "      Apenas verifica se tudo está instalado corretamente"
    echo ""
    echo "  [0] Sair"
    echo ""
    
    read -p "Opção: " choice
    
    case "$choice" in
        1) INSTALL_MODE="full" ;;
        2) INSTALL_MODE="minimal" ;;
        3) INSTALL_MODE="update" ;;
        4) INSTALL_MODE="check" ;;
        0) exit 0 ;;
        *) 
            print_error "Opção inválida"
            show_menu
            ;;
    esac
}

# ==============================================================================
# Execução Principal
# ==============================================================================

main() {
    print_banner
    
    echo -e "${BOLD}Bem-vindo ao instalador do DeepTrustShield!${NC}"
    echo ""
    echo "Este script irá guiá-lo através da instalação do sistema de"
    echo "detecção de deepfake. Siga as instruções na tela."
    echo ""
    
    press_enter
    
    show_menu
    
    # Detectar SO
    detect_os
    
    # Instalar Homebrew se necessário (macOS)
    install_homebrew
    
    case "$INSTALL_MODE" in
        full)
            check_install_python
            check_install_nodejs
            check_install_redis
            check_install_docker
            setup_directories
            setup_env_file
            install_backend
            install_worker
            install_frontend
            check_models
            ;;
        minimal)
            check_install_python
            setup_directories
            setup_env_file
            install_backend
            install_worker
            ;;
        update)
            # Pegar Python existente
            check_install_python
            install_backend
            install_worker
            install_frontend
            ;;
        check)
            check_install_python
            check_install_nodejs
            check_install_redis
            check_models
            echo ""
            print_success "Verificação concluída!"
            exit 0
            ;;
    esac
    
    print_summary
    start_system_prompt
}

# Executar
main "$@"
