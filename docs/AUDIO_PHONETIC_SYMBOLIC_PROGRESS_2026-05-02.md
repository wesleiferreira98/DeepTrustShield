# Resumo do avanço da camada fonético-fonológica

Data: 2026-05-02

## Contexto

O objetivo do trabalho de hoje foi avançar a camada neuro-simbólica de áudio para que ela deixe de ser apenas uma análise acústica geral e passe a incorporar sinais fonético-fonológicos do português brasileiro.

A motivação principal continua sendo reduzir falsos positivos em áudios reais com variação regional legítima, especialmente em casos em que o modelo neural apresenta score alto por causa de domínio, compressão, sotaque, prosódia ou características fora do conjunto de treino.

## O Que Foi Implementado

### 1. Sidecars de transcrição

Foi criado um fluxo offline para gerar arquivos auxiliares de transcrição com `faster-whisper`.

Script:

```text
scripts/build_audio_transcript_sidecars.py
```

Ele gera arquivos como:

```text
audio.transcript.json
```

Esses sidecars guardam:

- texto transcrito;
- palavras com tempo inicial e final;
- confiança por palavra;
- segmentos do ASR;
- metadados do modelo usado;
- hash SHA-256 do áudio original.

O hash permite que o worker encontre o sidecar mesmo quando o áudio é enviado pela web e copiado para a pasta de artefatos com nome genérico, como `audio.ogg`.

### 2. Extração de eventos fonéticos

Foi criada a primeira versão do extrator:

```text
worker/app/pipeline/audio_phonetic_events.py
```

Ele lê sidecars e gera eventos no formato `PhoneticEvent`, por exemplo:

```json
{
  "phenomenon": "rotic_coda",
  "word": "gravar",
  "variant": "deletion",
  "context": "verb_infinitive_external_coda",
  "confidence": 0.84
}
```

Nesta V1, os eventos cobrem:

- róticos em coda externa de verbo;
- `/s/` em coda;
- `/t,d/` diante de `[i]`.

### 3. Heurística acústica para `/r/` final

Foi implementada uma heurística conservadora para detectar presença ou apagamento de `/r/` final em candidatos de verbo no infinitivo.

Variantes possíveis:

```text
deletion
present
unknown
```

A heurística usa:

- energia RMS na cauda da palavra;
- razão entre energia da cauda e corpo da palavra;
- taxa de cruzamento por zero.

Isso permitiu detectar casos como:

```text
gravar  -> deletion
poder   -> deletion
entrar  -> deletion
```

Também foi corrigido um falso candidato: palavras curtas terminadas em `or`, como `Por`, não devem ser tratadas automaticamente como infinitivos.

### 4. Heurísticas para `/s/` em coda e `/t,d/` diante de `[i]`

Foram adicionadas heurísticas iniciais para classificar:

`/s/` em coda:

```text
alveolar
palatal
deletion
unknown
```

`/t,d/` diante de `[i]`:

```text
dentoalveolar
palatalized
unknown
```

Essas heurísticas usam sinais simples:

- energia;
- taxa de cruzamento por zero;
- centro espectral.

Elas ainda não são classificadores fonéticos robustos, mas já permitem extrair evidência inicial para a camada simbólica.

### 5. Integração com a camada simbólica

A camada simbólica passou a receber os eventos fonéticos extraídos e aplicar a base de conhecimento:

```text
audio/knowledge/phonetic_rules_v1.yaml
```

Hoje foram acionadas regras como:

```text
R1  apagamento de /r/ final em verbos é plausível
R2  alternância de /r/ sem ruptura localizada clara
TD2 realização dentoalveolar de /t,d/ diante de [i] estável
G1  fenômenos fonético-fonológicos estáveis
G5  score neural alto, mas linguística consistente
```

Também foi corrigido um problema importante: eventos com `variant: unknown` não devem contar como evidência linguística forte.

## Teste Principal: Salvador Real Audio

Foi testado um áudio real, sem edição, de uma pessoa nascida e criada em Salvador.

O modelo neural apresentou scores altos em várias janelas, mas apenas duas janelas ultrapassaram o threshold calibrado.

Resumo:

```text
threshold: 0.9744
janelas analisadas: 25
janelas acima do threshold: 2
fake_ratio: 0.08
decisão: inconclusive
```

As janelas suspeitas ficaram concentradas no final:

```text
46s - 50s
46.4s - 50.4s
```

Isso sugere uma suspeita localizada, não uma inconsistência no áudio inteiro.

## Leitura do Resultado

O resultado atual é tecnicamente melhor que uma decisão binária simples.

O sistema conseguiu dizer:

```text
O score neural está alto em alguns trechos,
mas a suspeita não é consistente no áudio inteiro.
Além disso, há padrões fonético-fonológicos plausíveis e estáveis.
```

Isso é exatamente o papel esperado da camada neuro-simbólica: não substituir o modelo neural, mas interpretar o score com conhecimento linguístico e reduzir o risco de falso positivo em fala regional legítima.

## Estado Atual

A arquitetura já está funcionando em fluxo completo:

```text
áudio
  -> transcrição offline
  -> sidecar por hash
  -> extração de eventos fonéticos
  -> regras simbólicas fonético-fonológicas
  -> decisão auditável
```

O sistema já distingue três situações:

- sem sidecar: não há cobertura fonética;
- sidecar com poucos eventos classificados: não aplica regra forte;
- sidecar com eventos classificados suficientes: aplica regras simbólicas com trava de ajuste.

## Limitações Atuais

A camada ainda é experimental.

As principais limitações são:

- as heurísticas acústicas ainda são simples;
- a classificação de `/s/` e `/t,d/` precisa ser validada com mais exemplos;
- os tempos por palavra vêm do ASR, não de alinhamento fonético fino;
- alguns eventos podem ser classificados incorretamente por ruído, compressão ou alinhamento impreciso;
- o ajuste máximo de `-0.20` pode ser forte se a cobertura parecer alta, mas a qualidade dos eventos ainda for limitada.

## Próximos Passos

### 1. Criar uma decisão intermediária

Adicionar uma saída mais precisa que `inconclusive`, por exemplo:

```text
real_with_review
```

ou:

```text
low_risk_review
```

Esse caso representa melhor áudios reais com picos localizados, mas com padrões linguísticos consistentes.

### 2. Melhorar a qualidade dos eventos fonéticos

Próximos refinamentos:

- melhorar a classificação de `/s/` em coda;
- melhorar a classificação de `/t,d/` diante de `[i]`;
- suavizar métricas quando houver poucos eventos;
- separar melhor eventos classificados por heurística forte e eventos candidatos.

### 3. Avaliar mais áudios reais externos

Criar uma pequena base de avaliação externa com:

- áudios reais de diferentes regiões;
- WhatsApp/OGG;
- MP3;
- gravações com ruído;
- falas longas e curtas;
- diferentes sotaques e estilos de fala.

Essa base não deve ser usada para treino. Ela deve servir como termômetro de falso positivo.

### 4. Inspecionar picos finais

No áudio de Salvador, os únicos segmentos acima do threshold estão no final. Devemos verificar:

- silêncio final;
- corte abrupto;
- compressão;
- ruído;
- mudança de volume;
- efeito do encerramento do áudio.

### 5. Criar relatório automático mais claro

Transformar a saída simbólica em texto amigável:

```text
O áudio foi classificado como inconclusivo.
A suspeita neural ficou concentrada no final.
Foram encontrados padrões linguísticos compatíveis com fala natural.
Recomenda-se revisão humana do trecho final.
```

### 6. Validar as regras contra o Atlas Linguístico Brasileiro

Manter a base simbólica alinhada com fenômenos documentados, sem transformar a camada em classificador regional.

O foco deve continuar sendo:

```text
consistência interna, plausibilidade linguística e mitigação de falso positivo
```

e não:

```text
identificação de origem regional do falante
```

## Conclusão

Hoje a frente de áudio passou de uma camada simbólica genérica para uma primeira camada fonético-fonológica operacional.

Ainda é uma V1, mas já há um fluxo completo e auditável combinando:

- modelo neural;
- inferência por janelas;
- transcrição offline;
- eventos fonéticos;
- regras linguísticas;
- explicações simbólicas.

O teste com áudio real de Salvador mostrou exatamente o caso de uso desejado: o neural levantou suspeita, mas a camada simbólica trouxe contexto linguístico para evitar uma conclusão agressiva.
