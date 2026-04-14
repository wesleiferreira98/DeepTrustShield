# 📡 API Reference - DeepTrustShield

## Base URL

```
http://localhost:8000
```

## Autenticação

Atualmente a API não requer autenticação. Em produção, recomenda-se implementar JWT ou API Keys.

---

## Endpoints

### POST `/v1/deepfake/image/analyze`

Inicia uma análise de deepfake em uma imagem.

#### Request

```http
POST /v1/deepfake/image/analyze HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="photo.jpg"
Content-Type: image/jpeg

<binary data>
--boundary
Content-Disposition: form-data; name="target_model"

auto
--boundary--
```

#### Parâmetros

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `file` | File | ✅ | Imagem para análise |
| `target_model` | string | ❌ | Modelo a usar (default: "auto") |

#### Formatos Aceitos

- `image/jpeg`
- `image/png`
- `image/webp`

#### Modelos Disponíveis

| Valor | Descrição |
|-------|-----------|
| `auto` | Seleciona automaticamente o melhor modelo disponível |
| `efficientnet_b0` | EfficientNet-B0 padrão |
| `efficientnet_b0_v4` | EfficientNet-B0 versão 4 |
| `efficientnet_b0_v5` | EfficientNet-B0 versão 5 (recomendado) |
| `efficientnet_b3` | EfficientNet-B3 (maior) |
| `efficientnet_b4` | EfficientNet-B4 (maior) |
| `mobilenet_v3_small_v1` | MobileNetV3 Small versão 1 |
| `mobilenet_v3_small` | MobileNetV3 Small |
| `mobilenet` | MobileNet automático |
| `alexnet_v82` | AlexNet versão 8.2 |
| `alexnet_v81` | AlexNet versão 8.1 |
| `alexnet_v8` | AlexNet versão 8 |
| `alexnet_v7` | AlexNet versão 7 |
| `alexnet_v6` | AlexNet versão 6 |
| `alexnet` | AlexNet (experimental) |

#### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Análise iniciada. Use o endpoint de status para acompanhar."
}
```

#### Erros

| Status | Código | Descrição |
|--------|--------|-----------|
| 400 | BAD_REQUEST | Tipo de arquivo não suportado |
| 400 | BAD_REQUEST | Modelo inválido |
| 413 | PAYLOAD_TOO_LARGE | Arquivo muito grande |
| 500 | INTERNAL_ERROR | Erro interno |

---

### GET `/v1/deepfake/image/status/{job_id}`

Consulta o status atual de um job de análise.

#### Request

```http
GET /v1/deepfake/image/status/550e8400-e29b-41d4-a716-446655440000 HTTP/1.1
Host: localhost:8000
```

#### Parâmetros de URL

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `job_id` | string (UUID) | ID do job retornado pelo `/analyze` |

#### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "stage": "model_inference",
  "progress": 70,
  "message": "Analisando com modelo de IA...",
  "steps": [
    {
      "id": "validating_image",
      "label": "Validando imagem",
      "state": "completed"
    },
    {
      "id": "detecting_face",
      "label": "Detectando face",
      "state": "completed"
    },
    {
      "id": "quality_checks",
      "label": "Verificando qualidade",
      "state": "completed"
    },
    {
      "id": "preprocessing",
      "label": "Pré-processando",
      "state": "completed"
    },
    {
      "id": "model_inference",
      "label": "Analisando com IA",
      "state": "running"
    },
    {
      "id": "calibrating_score",
      "label": "Calibrando resultado",
      "state": "pending"
    },
    {
      "id": "generating_evidence",
      "label": "Gerando evidências",
      "state": "pending"
    }
  ],
  "preview": {
    "bbox": {
      "x": 25.5,
      "y": 10.2,
      "width": 45.0,
      "height": 60.5
    },
    "face_crop_url": "/artifacts/550e8400.../face_crop.jpg"
  }
}
```

#### Estágios (stage)

| Stage | Progress | Descrição |
|-------|----------|-----------|
| `pending` | 0% | Aguardando na fila |
| `validating_image` | 10% | Validando formato |
| `detecting_face` | 25% | Detectando face |
| `quality_checks` | 40% | Verificando qualidade |
| `preprocessing` | 55% | Pré-processando |
| `model_inference` | 70% | Executando modelo |
| `calibrating_score` | 85% | Calibrando resultado |
| `generating_evidence` | 95% | Gerando evidências |
| `done` | 100% | Concluído |
| `error` | - | Erro |

#### Estados dos Steps (state)

| State | Descrição |
|-------|-----------|
| `pending` | Aguardando |
| `running` | Em execução |
| `completed` | Concluído |
| `error` | Erro |

#### Erros

| Status | Descrição |
|--------|-----------|
| 404 | Job não encontrado |

---

### GET `/v1/deepfake/image/result/{job_id}`

Obtém o resultado final de uma análise concluída.

#### Request

```http
GET /v1/deepfake/image/result/550e8400-e29b-41d4-a716-446655440000 HTTP/1.1
Host: localhost:8000
```

#### Response (Sucesso)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "p_fake": 0.8723,
  "risk_level": "high",
  "flags": {
    "no_face": false,
    "multiple_faces": false,
    "blur_high": false,
    "low_light": false,
    "face_too_small": false
  },
  "evidence": {
    "face_crop_url": "/artifacts/550e8400.../face_crop.jpg",
    "bbox": {
      "x": 25.5,
      "y": 10.2,
      "width": 45.0,
      "height": 60.5
    }
  },
  "model_version": "efficientnet_b0_v5",
  "created_at": "2026-02-26T10:30:00Z",
  "processing_time_ms": 2340
}
```

#### Response (Sem Face)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "p_fake": null,
  "risk_level": "undetermined",
  "flags": {
    "no_face": true,
    "multiple_faces": false,
    "blur_high": false,
    "low_light": false,
    "face_too_small": false
  },
  "evidence": {},
  "model_version": "efficientnet_b0_v5",
  "created_at": "2026-02-26T10:30:00Z",
  "processing_time_ms": 850
}
```

#### Campos

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `job_id` | string | ID do job |
| `p_fake` | float \| null | Probabilidade de ser fake (0-1) |
| `risk_level` | string | Nível de risco |
| `flags` | object | Flags de qualidade |
| `evidence` | object | Evidências visuais |
| `model_version` | string | Versão do modelo usado |
| `created_at` | datetime | Data de criação |
| `processing_time_ms` | int | Tempo de processamento em ms |

#### Níveis de Risco (risk_level)

| Valor | Range p_fake | Interpretação |
|-------|--------------|---------------|
| `low` | < 0.50 | 🟢 Provavelmente real |
| `medium` | 0.50 - 0.80 | 🟡 Inconclusivo |
| `high` | ≥ 0.80 | 🔴 Provavelmente fake |
| `undetermined` | N/A | ⚪ Sem face detectada |

#### Flags

| Flag | Descrição |
|------|-----------|
| `no_face` | Nenhuma face detectada |
| `multiple_faces` | Múltiplas faces (usou a maior) |
| `blur_high` | Imagem borrada |
| `low_light` | Iluminação inadequada |
| `face_too_small` | Face muito pequena (<80px) |

#### Erros

| Status | Descrição |
|--------|-----------|
| 400 | Análise ainda em andamento |
| 404 | Job não encontrado |

---

### GET `/v1/deepfake/image/artifacts/{job_id}/{name}`

Baixa um artefato específico de um job.

#### Request

```http
GET /v1/deepfake/image/artifacts/550e8400.../face_crop.jpg HTTP/1.1
Host: localhost:8000
```

#### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `job_id` | string | ID do job |
| `name` | string | Nome do arquivo |

#### Artefatos Disponíveis

| Nome | Descrição |
|------|-----------|
| `original.jpg` | Imagem original enviada |
| `face_crop.jpg` | Recorte da face detectada |

#### Response

Binário do arquivo (Content-Type apropriado)

#### Erros

| Status | Descrição |
|--------|-----------|
| 404 | Job ou artefato não encontrado |

---

### GET `/health`

Health check do serviço.

#### Response

```json
{
  "status": "healthy"
}
```

---

### GET `/`

Informações básicas da API.

#### Response

```json
{
  "name": "Detectar API",
  "version": "1.0.0",
  "status": "running"
}
```

---

## Exemplos de Uso

### cURL

```bash
# Enviar imagem para análise
curl -X POST http://localhost:8000/v1/deepfake/image/analyze \
  -F "file=@photo.jpg" \
  -F "target_model=auto"

# Verificar status
curl http://localhost:8000/v1/deepfake/image/status/{job_id}

# Obter resultado
curl http://localhost:8000/v1/deepfake/image/result/{job_id}
```

### Python (requests)

```python
import requests
import time

# Upload e análise
with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/v1/deepfake/image/analyze',
        files={'file': f},
        data={'target_model': 'auto'}
    )
    job_id = response.json()['job_id']

# Polling para resultado
while True:
    status = requests.get(
        f'http://localhost:8000/v1/deepfake/image/status/{job_id}'
    ).json()
    
    if status['stage'] == 'done':
        break
    elif status['stage'] == 'error':
        raise Exception(status['message'])
    
    time.sleep(0.5)

# Obter resultado final
result = requests.get(
    f'http://localhost:8000/v1/deepfake/image/result/{job_id}'
).json()

print(f"Probabilidade de fake: {result['p_fake']:.2%}")
print(f"Nível de risco: {result['risk_level']}")
```

### JavaScript (fetch)

```javascript
// Upload e análise
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('target_model', 'auto');

const response = await fetch('/v1/deepfake/image/analyze', {
  method: 'POST',
  body: formData
});
const { job_id } = await response.json();

// Polling
const pollStatus = async () => {
  const res = await fetch(`/v1/deepfake/image/status/${job_id}`);
  const status = await res.json();
  
  if (status.stage === 'done') {
    const result = await fetch(`/v1/deepfake/image/result/${job_id}`);
    return result.json();
  } else if (status.stage === 'error') {
    throw new Error(status.message);
  }
  
  await new Promise(r => setTimeout(r, 500));
  return pollStatus();
};

const result = await pollStatus();
console.log(`Fake: ${(result.p_fake * 100).toFixed(1)}%`);
```

---

## Rate Limiting

Não há rate limiting implementado atualmente. Em produção, recomenda-se:

- 100 requests/minuto por IP
- 10 uploads/minuto por IP
- Fila máxima de 1000 jobs

---

## OpenAPI / Swagger

Documentação interativa disponível em:

```
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/redoc     # ReDoc
http://localhost:8000/openapi.json  # OpenAPI Schema
```
