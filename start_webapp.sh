#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Erro: Docker Compose não encontrado. Instale docker compose ou docker-compose."
  exit 1
fi

required_files=(
  "docker-compose.yml"
  "best_model_efficientnet_b0.pth"
  "best_model_efficientnet_b3.pth"
  "best_model_efficientnet_b4.pth"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Erro: arquivo obrigatório não encontrado: $file"
    exit 1
  fi
done

echo "Subindo serviços do web app..."
"${COMPOSE_CMD[@]}" up --build -d

echo
echo "Serviços iniciados com sucesso."
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000/docs"
echo
echo "Para acompanhar logs:"
echo "  ${COMPOSE_CMD[*]} logs -f"
