
## Objetivo da V1.1 experimental

```text
Adicionar novas regras simbólicas sem comprometer a decisão principal da V1.
```

A V1 atual já tem um fluxo estável: neural em log-Mel, inferência por janelas, transcrição, extração fonético-fonológica, regras simbólicas, memória temporal e decisão final. A camada simbólica também já está bem delimitada: ela calibra, contesta ou reforça a suspeita neural, mas não substitui o modelo.

Então o planejamento deve proteger isso.

## Estrutura que eu usaria

```text
V1.0 — Regras principais
/r/ final
/s/ coda
/t,d/ + i
G1–G6
impactam score

V1.1 experimental — Novas regras
novos fenômenos ou refinamentos
aparecem no relatório
não impactam score inicialmente

V2 — Expansão consolidada
novas regras validadas
pesos calibrados
possível fuzzy/prosódia/vogais
```

## Como inserir as regras novas

Cada regra nova deveria ter alguns campos obrigatórios:

```yaml
id:
title:
phenomenon:
condition:
description:
evidence_type:
requires:
effect_label:
weight:
affects_score:
experimental:
confidence_policy:
```

Exemplo:

```yaml
- id: "PX1"
  title: "Padrão prosódico estável"
  phenomenon: "prosody"
  condition: "stable_pause_energy_rhythm_pattern"
  description: "A distribuição de pausas, energia e ritmo permaneceu estável ao longo do áudio."
  evidence_type: "stability"
  effect_label: "Observacional"
  weight: 0.00
  affects_score: false
  experimental: true
  confidence_policy: "requires_minimum_duration_and_segment_coverage"
```

O campo mais importante é:

```yaml
affects_score: false
```

Assim a regra aparece, mas não altera a decisão.

## Separação no front

Eu colocaria duas áreas:

```text
Regras ativadas
→ regras que alteraram o score

Regras observadas
→ regras experimentais detectadas, mas sem impacto na decisão
```

Exemplo:

```text
Regras ativadas
G1 — Múltiplos fenômenos estáveis       -0.10
G5 — Score neural alto com consistência -0.10

Regras observadas
PX1 — Padrão prosódico estável          observacional
VX1 — Vogais pretônicas consistentes    observacional
QX1 — Compressão pode afetar leitura    observacional
```

Isso evita que o usuário pense que toda regra exibida mexeu na decisão.

## Critérios para uma regra sair do modo experimental

Uma regra só deveria ganhar peso depois de passar por alguns testes:

```text
1. ativa em casos onde faz sentido;
2. não ativa demais em áudios aleatórios;
3. não reduz suspeita em deepfakes óbvios;
4. não aumenta suspeita em fala regional natural;
5. melhora explicação sem contradizer as métricas principais;
6. tem pelo menos alguns exemplos reais documentados.
```

A regra pode receber três destinos:

```text
Promover:
vira regra oficial com peso.

Manter observacional:
útil para relatório, mas sem peso.

Remover:
gera ruído ou ativa de forma inconsistente.
```

## Planejamento em fases

### Fase 1 — Catálogo das novas regras

Antes de codar, montar uma tabela:

| ID  | Fenômeno | Condição                     | Efeito esperado  | Peso inicial | Status       |
| --- | --------- | ------------------------------ | ---------------- | -----------: | ------------ |
| PX1 | Prosódia | ritmo estável                 | reduz cautela    |         0.00 | experimental |
| PX2 | Prosódia | quebra brusca de energia/pausa | aumenta suspeita |         0.00 | experimental |
| VX1 | Vogais    | padrão vocálico consistente  | reduz cautela    |         0.00 | experimental |
| VX2 | Vogais    | mudança localizada            | aumenta suspeita |         0.00 | experimental |

### Fase 2  YAML experimental

Criar um arquivo separado:

```text
phonetic_rules_v1.yaml
experimental_rules_v1_1.yaml
```

Ou manter no mesmo YAML, mas com:

```yaml
ruleset: "v1_1_experimental"
experimental: true
affects_score: false
```

Eu prefiro separado no começo, porque fica mais fácil desligar.

### Fase 3  Backend

O worker deve retornar algo assim:

```json
{
  "activated_rules": [],
  "observed_rules": [],
  "experimental_rules": []
}
```

Ou:

```json
{
  "rules": [
    {
      "id": "G1",
      "title": "Múltiplos fenômenos estáveis",
      "weight": -0.10,
      "affects_score": true,
      "experimental": false
    },
    {
      "id": "PX1",
      "title": "Padrão prosódico estável",
      "weight": 0.00,
      "affects_score": false,
      "experimental": true
    }
  ]
}
```

### Fase 4 Front

O front não recalcula nada. Ele só exibe:

```text
impactou score?
experimental?
qual fenômeno?
qual explicação?
qual evidência?
```

Isso está alinhado com a arquitetura atual, em que o front apenas traduz e organiza o resultado produzido pelo worker.

### Fase 5  Avaliação

Criar uma planilha simples de testes:

| Áudio              | Tipo          | Neural | Simbólico V1 | Regras V1   | Regras exp. | Observação                        |
| ------------------- | ------------- | -----: | ------------: | ----------- | ----------- | ----------------------------------- |
| Real Salvador 1     | real regional |   alto |         baixo | G1, G5      | PX1         | reduziu bem                         |
| Real Salvador 2     | real regional |   alto |        médio | R2, TD2, G1 | PX2         | alinhamento alto, mas baixa ruptura |
| Manipulado 1        | fake          |   alto |             ? | ?           | ?           | verificar se regras exp. ajudam     |
| WhatsApp comprimido | real          |   alto |             ? | ?           | QX1         | alerta de domínio                  |

Essa tabela vai te dizer se as novas regras merecem peso.

## Minha recomendação de prioridade

Eu começaria com regras que não exigem análise fonética muito fina.

Ordem segura:

```text
1. regras de qualidade/domínio;
2. regras temporais/prosódicas simples;
3. regras de distribuição de picos;
4. só depois regras vocálicas;
5. por último regras fonéticas mais finas.
```

Porque vogais e formantes são mais complicados. Prosódia e qualidade podem entrar mais rápido como observacionais.

## Definição da V1.1

A definição oficial poderia ser:

```text
A V1.1 introduz um conjunto experimental de regras simbólicas observacionais. Essas regras ampliam a explicabilidade do sistema, mas não alteram inicialmente o score final. O objetivo é avaliar sua estabilidade em áudios reais e manipulados antes de promovê-las para a camada de calibração.
```
