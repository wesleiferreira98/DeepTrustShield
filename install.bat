@echo off
SETLOCAL EnableDelayedExpansion

:: ==============================================================================
:: DeepTrustShield - Script de Instalacao Assistida (Windows)
:: ==============================================================================
:: Este script instala todas as dependencias e configura o sistema de deteccao
:: de deepfake DeepTrustShield de forma totalmente assistida.
:: ==============================================================================

title DeepTrustShield - Instalacao Assistida

:: Cores para output (Windows 10+)
set "ESC="
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "CYAN=[96m"
set "MAGENTA=[95m"
set "NC=[0m"
set "BOLD=[1m"

:: Diretorio do script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Variaveis de controle
set "PYTHON_CMD="
set "INSTALL_MODE="
set "MODEL_FOUND=false"

:: ==============================================================================
:: Funcoes de Output
:: ==============================================================================

:print_banner
cls
echo %CYAN%
echo ======================================================================
echo                    DeepTrustShield
echo              Sistema de Deteccao de Deepfake
echo                   Instalacao Assistida
echo ======================================================================
echo %NC%
echo.
goto :eof

:print_step
echo.
echo %BLUE%-----------------------------------------------------------------------%NC%
echo %MAGENTA%^> %~1%NC%
echo %BLUE%-----------------------------------------------------------------------%NC%
echo.
goto :eof

:print_success
echo %GREEN%[OK] %~1%NC%
goto :eof

:print_warning
echo %YELLOW%[!] %~1%NC%
goto :eof

:print_error
echo %RED%[X] %~1%NC%
goto :eof

:print_info
echo %CYAN%[i] %~1%NC%
goto :eof

:: ==============================================================================
:: Menu Principal
:: ==============================================================================

:show_menu
call :print_banner

echo %BOLD%Bem-vindo ao instalador do DeepTrustShield!%NC%
echo.
echo Este script ira guia-lo atraves da instalacao do sistema de
echo deteccao de deepfake. Siga as instrucoes na tela.
echo.
echo %BOLD%Escolha o tipo de instalacao:%NC%
echo.
echo   [1] Instalacao Completa (Recomendado)
echo       Instala todas as dependencias e configura o ambiente
echo.
echo   [2] Instalacao Minima (Apenas dependencias Python)
echo       Util se Node.js e Redis ja estao instalados
echo.
echo   [3] Atualizar Dependencias
echo       Reinstala as dependencias sem verificar sistema
echo.
echo   [4] Verificar Sistema
echo       Apenas verifica se tudo esta instalado corretamente
echo.
echo   [0] Sair
echo.

set /p "choice=Opcao: "

if "%choice%"=="1" set "INSTALL_MODE=full" & goto :main_install
if "%choice%"=="2" set "INSTALL_MODE=minimal" & goto :main_install
if "%choice%"=="3" set "INSTALL_MODE=update" & goto :main_install
if "%choice%"=="4" set "INSTALL_MODE=check" & goto :main_install
if "%choice%"=="0" exit /b 0

call :print_error "Opcao invalida"
timeout /t 2 >nul
goto :show_menu

:: ==============================================================================
:: Verificacao de Python
:: ==============================================================================

:check_python
call :print_step "Verificando Python"

:: Tentar python
where python >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VERSION=%%v"
    set "PYTHON_CMD=python"
    call :print_success "Python !PY_VERSION! encontrado"
    goto :check_python_done
)

:: Tentar python3
where python3 >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "tokens=2" %%v in ('python3 --version 2^>^&1') do set "PY_VERSION=%%v"
    set "PYTHON_CMD=python3"
    call :print_success "Python !PY_VERSION! encontrado"
    goto :check_python_done
)

:: Tentar py
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "tokens=2" %%v in ('py --version 2^>^&1') do set "PY_VERSION=%%v"
    set "PYTHON_CMD=py"
    call :print_success "Python !PY_VERSION! encontrado"
    goto :check_python_done
)

:: Nao encontrado
call :print_error "Python nao encontrado!"
echo.
call :print_info "Por favor, instale Python 3.8+ de https://python.org"
call :print_info "IMPORTANTE: Marque 'Add Python to PATH' durante a instalacao!"
echo.
set /p "dummy=Pressione ENTER apos instalar Python..."
goto :check_python

:check_python_done
goto :eof

:: ==============================================================================
:: Verificacao de Node.js
:: ==============================================================================

:check_nodejs
call :print_step "Verificando Node.js"

where node >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f %%v in ('node --version') do set "NODE_VERSION=%%v"
    call :print_success "Node.js !NODE_VERSION! encontrado"
    
    where npm >nul 2>&1
    if %ERRORLEVEL%==0 (
        for /f %%v in ('npm --version') do set "NPM_VERSION=%%v"
        call :print_success "npm !NPM_VERSION! encontrado"
        goto :check_nodejs_done
    )
)

:: Nao encontrado
call :print_warning "Node.js nao encontrado!"
echo.
call :print_info "Por favor, instale Node.js 18+ de https://nodejs.org"
echo.
set /p "install_node=Deseja abrir a pagina de download? [S/n]: "
if /i "!install_node!"=="" set "install_node=S"
if /i "!install_node!"=="S" start https://nodejs.org
if /i "!install_node!"=="s" start https://nodejs.org
echo.
set /p "dummy=Pressione ENTER apos instalar Node.js..."
goto :check_nodejs

:check_nodejs_done
goto :eof

:: ==============================================================================
:: Verificacao de Redis
:: ==============================================================================

:check_redis
call :print_step "Verificando Redis"

:: Verificar se Redis esta instalado via WSL ou como servico
where redis-cli >nul 2>&1
if %ERRORLEVEL%==0 (
    call :print_success "Redis encontrado"
    goto :check_redis_done
)

:: Verificar Redis via Docker
where docker >nul 2>&1
if %ERRORLEVEL%==0 (
    call :print_info "Redis nao instalado localmente, mas Docker esta disponivel"
    call :print_info "O sistema pode usar Redis via Docker"
    goto :check_redis_done
)

:: Nao encontrado
call :print_warning "Redis nao encontrado!"
echo.
echo Opcoes para instalar Redis no Windows:
echo.
echo   1. Via Docker (recomendado):
echo      docker run -d --name redis -p 6379:6379 redis:7-alpine
echo.
echo   2. Via WSL (Windows Subsystem for Linux):
echo      wsl sudo apt install redis-server
echo.
echo   3. Via Memurai (alternativa nativa ao Redis):
echo      https://www.memurai.com/
echo.
echo   4. Via Chocolatey:
echo      choco install redis-64
echo.
set /p "dummy=Pressione ENTER apos instalar Redis..."
goto :check_redis

:check_redis_done
goto :eof

:: ==============================================================================
:: Configuracao de Diretorios
:: ==============================================================================

:setup_directories
call :print_step "Criando Diretorios Necessarios"

if not exist "artifacts" mkdir "artifacts"
call :print_success "Diretorio artifacts\ criado"

if not exist "model\weights" mkdir "model\weights"
call :print_success "Diretorio model\weights\ criado"

if not exist "backend\data" mkdir "backend\data"
call :print_success "Diretorio backend\data\ criado"

goto :eof

:: ==============================================================================
:: Configuracao do Arquivo .env
:: ==============================================================================

:setup_env
call :print_step "Configurando Variaveis de Ambiente"

if exist ".env" (
    call :print_info "Arquivo .env ja existe"
    set /p "recreate_env=Deseja recria-lo? [s/N]: "
    if /i "!recreate_env!"=="s" del /f ".env"
    if /i not "!recreate_env!"=="s" goto :setup_env_done
)

if exist ".env.example" (
    copy ".env.example" ".env" >nul
    call :print_success "Arquivo .env criado a partir do exemplo"
) else (
    (
        echo # DeepTrustShield - Configuracoes de Ambiente
        echo.
        echo # Banco de dados
        echo DATABASE_URL=sqlite:///./detectar.db
        echo.
        echo # Redis
        echo REDIS_URL=redis://localhost:6379/0
        echo.
        echo # Diretorios
        echo ARTIFACTS_PATH=./artifacts
        echo MODEL_PATH=./model/weights
        echo.
        echo # CORS ^(separado por virgula^)
        echo CORS_ORIGINS=http://localhost:3000,http://localhost:5173
        echo.
        echo # Servidor
        echo HOST=0.0.0.0
        echo BACKEND_PORT=8000
        echo FRONTEND_PORT=3000
    ) > ".env"
    call :print_success "Arquivo .env criado com configuracoes padrao"
)

:setup_env_done
goto :eof

:: ==============================================================================
:: Instalacao do Backend
:: ==============================================================================

:install_backend
call :print_step "Instalando Backend (FastAPI)"

cd /d "%SCRIPT_DIR%backend"

if exist "venv" (
    call :print_info "Ambiente virtual ja existe"
    set /p "recreate_venv=Deseja recria-lo? [s/N]: "
    if /i "!recreate_venv!"=="s" rmdir /s /q "venv"
)

if not exist "venv" (
    call :print_info "Criando ambiente virtual..."
    %PYTHON_CMD% -m venv venv
    call :print_success "Ambiente virtual criado"
)

call :print_info "Instalando dependencias do backend..."
call venv\Scripts\activate.bat

:: Atualizar pip
python -m pip install --upgrade pip

:: Instalar dependencias
pip install -r requirements.txt

call venv\Scripts\deactivate.bat

call :print_success "Backend instalado com sucesso"
cd /d "%SCRIPT_DIR%"
goto :eof

:: ==============================================================================
:: Instalacao do Worker
:: ==============================================================================

:install_worker
call :print_step "Instalando Worker (Processamento de IA)"

cd /d "%SCRIPT_DIR%worker"

if exist "venv" (
    call :print_info "Ambiente virtual ja existe"
    set /p "recreate_venv=Deseja recria-lo? [s/N]: "
    if /i "!recreate_venv!"=="s" rmdir /s /q "venv"
)

if not exist "venv" (
    call :print_info "Criando ambiente virtual..."
    %PYTHON_CMD% -m venv venv
    call :print_success "Ambiente virtual criado"
)

call :print_info "Instalando dependencias do worker (pode demorar devido ao PyTorch)..."
call venv\Scripts\activate.bat

:: Atualizar pip
python -m pip install --upgrade pip

:: Instalar dependencias
pip install -r requirements.txt

call venv\Scripts\deactivate.bat

call :print_success "Worker instalado com sucesso"
cd /d "%SCRIPT_DIR%"
goto :eof

:: ==============================================================================
:: Instalacao do Frontend
:: ==============================================================================

:install_frontend
call :print_step "Instalando Frontend (React + Vite)"

cd /d "%SCRIPT_DIR%frontend"

if exist "node_modules" (
    call :print_info "Dependencias do frontend ja instaladas"
    set /p "reinstall=Deseja reinstalar? [s/N]: "
    if /i "!reinstall!"=="s" rmdir /s /q "node_modules"
)

if not exist "node_modules" (
    call :print_info "Instalando dependencias do frontend..."
    call npm install
    call :print_success "Dependencias instaladas"
)

set /p "build_prod=Deseja fazer build de producao do frontend? [s/N]: "
if /i "!build_prod!"=="s" (
    call :print_info "Executando build de producao..."
    call npm run build
    call :print_success "Build de producao concluido (pasta dist\)"
)

call :print_success "Frontend instalado com sucesso"
cd /d "%SCRIPT_DIR%"
goto :eof

:: ==============================================================================
:: Verificacao de Modelos
:: ==============================================================================

:check_models
call :print_step "Verificando Modelos de IA"

set "MODEL_FOUND=false"

:: Verificar modelos na pasta model/weights
if exist "model\weights\*.pth" (
    for %%f in (model\weights\*.pth) do (
        set "MODEL_FOUND=true"
        call :print_success "Modelo encontrado: %%~nxf"
    )
)

:: Verificar modelos na raiz
if exist "*.pth" (
    for %%f in (*.pth) do (
        set "MODEL_FOUND=true"
        call :print_success "Modelo encontrado (raiz): %%~nxf"
    )
)

if "!MODEL_FOUND!"=="false" (
    call :print_warning "Nenhum modelo .pth encontrado!"
    call :print_info "O sistema precisa de um modelo treinado para funcionar."
    echo Opcoes:
    echo   1. Coloque um modelo .pth na pasta model\weights\
    echo   2. Treine um modelo usando os notebooks em notebooks\
    echo   3. Baixe um modelo pre-treinado
)

goto :eof

:: ==============================================================================
:: Resumo Final
:: ==============================================================================

:print_summary
echo.
echo %GREEN%======================================================================%NC%
echo %GREEN%           INSTALACAO CONCLUIDA COM SUCESSO!%NC%
echo %GREEN%======================================================================%NC%
echo.

echo %BOLD%Componentes Instalados:%NC%
echo   %GREEN%[OK]%NC% Python
echo   %GREEN%[OK]%NC% Node.js
echo   %GREEN%[OK]%NC% Backend (FastAPI)
echo   %GREEN%[OK]%NC% Worker (PyTorch + MediaPipe)
echo   %GREEN%[OK]%NC% Frontend (React + Vite)
echo.

echo %BOLD%Para iniciar o sistema:%NC%
echo.
echo   %CYAN%Modo Docker (recomendado):*NC%
echo     docker compose up --build
echo.
echo   %CYAN%Modo Nativo (manual):%NC%
echo     1. Iniciar Redis (via Docker ou WSL)
echo     2. Abrir 3 terminais e executar:
echo.
echo        Terminal 1 - Backend:
echo          cd backend ^&^& venv\Scripts\activate ^&^& uvicorn app.main:app --reload
echo.
echo        Terminal 2 - Worker:
echo          cd worker ^&^& venv\Scripts\activate ^&^& rq worker deepfake --url redis://localhost:6379/0
echo.
echo        Terminal 3 - Frontend:
echo          cd frontend ^&^& npm run dev
echo.

echo %BOLD%Endpoints apos iniciar:%NC%
echo   Frontend:  %BLUE%http://localhost:3000%NC%
echo   Backend:   %BLUE%http://localhost:8000%NC%
echo   API Docs:  %BLUE%http://localhost:8000/docs%NC%
echo.

goto :eof

:: ==============================================================================
:: Criar Script de Inicializacao para Windows
:: ==============================================================================

:create_start_script
call :print_step "Criando Script de Inicializacao"

(
    echo @echo off
    echo SETLOCAL EnableDelayedExpansion
    echo.
    echo :: DeepTrustShield - Script de Inicializacao
    echo title DeepTrustShield
    echo.
    echo set "SCRIPT_DIR=%%~dp0"
    echo cd /d "%%SCRIPT_DIR%%"
    echo.
    echo echo ======================================
    echo echo    DeepTrustShield - Iniciando...
    echo echo ======================================
    echo echo.
    echo.
    echo :: Verificar Redis
    echo echo Verificando Redis...
    echo docker ps ^| findstr redis ^>nul 2^>^&1
    echo if %%ERRORLEVEL%% NEQ 0 ^(
    echo     echo Iniciando Redis via Docker...
    echo     docker run -d --name deeptrust-redis -p 6379:6379 redis:7-alpine
    echo     timeout /t 3 ^>nul
    echo ^)
    echo echo [OK] Redis rodando
    echo.
    echo :: Iniciar Backend
    echo echo.
    echo echo Iniciando Backend...
    echo start "DeepTrustShield - Backend" cmd /k "cd /d %%SCRIPT_DIR%%backend ^&^& call venv\Scripts\activate.bat ^&^& set DATABASE_URL=sqlite:///../detectar.db ^&^& set REDIS_URL=redis://localhost:6379/0 ^&^& set ARTIFACTS_PATH=../artifacts ^&^& uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
    echo timeout /t 3 ^>nul
    echo echo [OK] Backend iniciado
    echo.
    echo :: Iniciar Worker
    echo echo.
    echo echo Iniciando Worker...
    echo start "DeepTrustShield - Worker" cmd /k "cd /d %%SCRIPT_DIR%%worker ^&^& call venv\Scripts\activate.bat ^&^& set DATABASE_URL=sqlite:///../detectar.db ^&^& set REDIS_URL=redis://localhost:6379/0 ^&^& set ARTIFACTS_PATH=../artifacts ^&^& rq worker deepfake --url redis://localhost:6379/0 --with-scheduler"
    echo timeout /t 2 ^>nul
    echo echo [OK] Worker iniciado
    echo.
    echo :: Iniciar Frontend
    echo echo.
    echo echo Iniciando Frontend...
    echo start "DeepTrustShield - Frontend" cmd /k "cd /d %%SCRIPT_DIR%%frontend ^&^& npm run dev -- --host 0.0.0.0 --port 3000"
    echo timeout /t 3 ^>nul
    echo echo [OK] Frontend iniciado
    echo.
    echo echo.
    echo echo ======================================
    echo echo   TODOS OS SERVICOS INICIADOS!
    echo echo ======================================
    echo echo.
    echo echo   Frontend:  http://localhost:3000
    echo echo   Backend:   http://localhost:8000
    echo echo   API Docs:  http://localhost:8000/docs
    echo echo.
    echo echo Para encerrar, feche as janelas de terminal abertas.
    echo echo.
    echo pause
) > "start_windows.bat"

call :print_success "Script start_windows.bat criado"
goto :eof

:: ==============================================================================
:: Fluxo Principal de Instalacao
:: ==============================================================================

:main_install
if "%INSTALL_MODE%"=="check" goto :check_only

:: Instalacao Full
if "%INSTALL_MODE%"=="full" (
    call :check_python
    call :check_nodejs
    call :check_redis
    call :setup_directories
    call :setup_env
    call :install_backend
    call :install_worker
    call :install_frontend
    call :check_models
    call :create_start_script
    goto :finish
)

:: Instalacao Minima
if "%INSTALL_MODE%"=="minimal" (
    call :check_python
    call :setup_directories
    call :setup_env
    call :install_backend
    call :install_worker
    call :check_models
    goto :finish
)

:: Atualizar
if "%INSTALL_MODE%"=="update" (
    call :check_python
    call :install_backend
    call :install_worker
    call :install_frontend
    goto :finish
)

:check_only
call :check_python
call :check_nodejs
call :check_redis
call :check_models
echo.
call :print_success "Verificacao concluida!"
echo.
pause
exit /b 0

:finish
call :print_summary

set /p "start_now=Deseja executar o script de inicializacao? [S/n]: "
if /i "!start_now!"=="" set "start_now=S"
if /i "!start_now!"=="S" (
    echo.
    call :print_info "Iniciando DeepTrustShield..."
    call start_windows.bat
) else (
    echo.
    call :print_info "Para iniciar mais tarde, execute: start_windows.bat"
    echo.
)

pause
exit /b 0
