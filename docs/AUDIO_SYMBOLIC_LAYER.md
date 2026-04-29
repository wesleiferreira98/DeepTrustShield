# Camada Simbolica Regional para Audio

## Objetivo

A camada simbolica de audio existe para tratar o problema central do MVP:
o modelo neural pode confundir variacoes legitimas do portugues brasileiro com
sinais de audio sintetico.

Ela deve funcionar como uma camada de hipoteses concorrentes. Quando o neural
marca um audio como fake, a simbolica nao deve apenas confirmar ou negar. Ela
deve perguntar:

- isso parece fake ou parece fala regional fora do dominio do treino?
- o score alto pode ser explicado por sotaque, prosodia, compressao ou canal?
- a suspeita aparece no audio inteiro ou so em alguns trechos?
- existe evidencia suficiente para risco alto ou o correto e revisao/inconclusivo?

## Hipoteses

O resultado final deve considerar pelo menos quatro hipoteses:

| Hipotese | Descricao |
|----------|-----------|
| `synthetic_audio` | o audio parece sintetico/clonado |
| `regional_or_domain_false_positive` | o modelo pode estar punindo regionalismo, sotaque, canal ou compressao |
| `localized_suspicious_segment` | a suspeita aparece em poucos trechos |
| `insufficient_evidence` | qualidade, duracao ou margem nao sustentam conclusao |

## Papel do Modelo Neural

O modelo neural continua sendo o detector estatistico principal. Ele fornece:

- `p_fake` global;
- `threshold`;
- scores por janelas;
- versao do modelo.

Mas a decisao final nao deve ser uma copia simples do score neural. A camada
simbolica deve revisar o score sob a perspectiva de regionalismo, dominio e
evidencia por segmentos.

## Papel da Simbolica

A simbolica deve:

- detectar risco de falso positivo regional/domino;
- preservar explicacoes auditaveis;
- rebaixar decisoes `high` quando houver hipotese regional plausivel;
- marcar suspeita localizada quando poucos segmentos forem suspeitos;
- marcar suspeita consistente apenas quando varias janelas sustentarem a mesma conclusao;
- preparar o projeto para auditoria por sotaque usando FakeBrAccent-D.

## Pipeline

```text
audio bruto
  -> normalizacao para 16 kHz mono
  -> inferencia neural global
  -> inferencia neural por janelas
  -> features simbolicas globais
  -> revisao de qualidade
  -> revisao por segmentos
  -> revisao regional/dominio
  -> decisao final + explicacoes
```

## Saida Esperada

```json
{
  "decision": "fake",
  "risk_level": "medium",
  "confidence": "moderate",
  "neural_score": 0.93,
  "symbolic_score": 0.55,
  "final_score": 0.81,
  "decision_score": 0.91,
  "flags": {
    "consistent_suspicious_segments": true,
    "regional_bias_review": true,
    "domain_mismatch_warning": true
  },
  "regional_bias_review": {
    "review_needed": true,
    "hypothesis": "regional_or_domain_false_positive",
    "evidence": [
      "compressed_or_mobile_format",
      "resampled_from_external_sample_rate",
      "long_form_audio"
    ],
    "note": "O padrao pode refletir sotaque, regionalismo, compressao ou dominio fora do treino; precisa ser comparado com a base brasileira."
  },
  "explanations": [
    "A maior parte das janelas analisadas ficou acima do limite de audio fake.",
    "Ha hipotese alternativa de falso positivo por regionalismo, compressao ou dominio fora do treino."
  ]
}
```

## Regras V1

### Qualidade

| Flag | Intencao |
|------|----------|
| `short_audio` | audio curto reduz confianca |
| `very_short_audio` | audio curto demais torna a analise inconclusiva |
| `high_silence` | muito silencio pode distorcer o score |
| `low_volume` | volume baixo pode distorcer o score |
| `clipped_audio` | saturacao reduz confianca |

### Segmentos

| Flag | Intencao |
|------|----------|
| `segment_score_disagreement` | janelas discordam bastante |
| `isolated_suspicious_segment` | poucos trechos suspeitos |
| `consistent_suspicious_segments` | muitas janelas suspeitas |

### Regionalismo e Dominio

| Flag | Intencao |
|------|----------|
| `regional_bias_review` | existe hipotese de falso positivo regional/domino |
| `domain_mismatch_warning` | formato, canal ou duracao parecem fora do treino original |

A V1 nao detecta sotaque diretamente. Ela apenas abre uma hipotese simbolica
quando o modelo fica muito confiante em audio que tem sinais de dominio diferente
do treino, como formato comprimido, sample rate externo, audio longo ou pausas
naturais.

Depois do treino com FakeBrAccent-D, essa etapa deve passar a usar metricas por
grupo regional.

## FakeBrAccent-D

A base FakeBrAccent-D deve ser usada para dar conteudo empirico a camada regional.
Ela deve permitir:

- medir falso positivo por sotaque/regiao;
- calibrar thresholds por grupo;
- verificar se um sotaque recebe scores fake maiores que outros;
- gerar um relatorio de viés regional;
- atualizar as regras simbolicas com evidencias reais.

Manifesto recomendado:

```csv
audio_path,relative_path,filename,extension,split,label,source,accent
```

Campos importantes:

- `label`: `real` ou `fake`;
- `accent`: `baiano`, `carioca`, `fluminense`, `nordestino`, `sulista` ou outro valor presente na base;
- `source`: `fake_braccent_d`.

## Proximos Passos

1. Adaptar/gerar manifesto da FakeBrAccent-D preservando `accent`.
2. Treinar o baseline atual com a base brasileira.
3. Avaliar metricas por sotaque:
   - AUC;
   - F1;
   - falso positivo em reais;
   - falso negativo em fakes.
4. Alimentar a camada simbolica com estatisticas por grupo.
5. Recalibrar o `regional_bias_review` com base empirica.
