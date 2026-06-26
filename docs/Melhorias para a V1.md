Sim, dá para melhorar bastante  mas primeiro:  **não acho que você esteja perdido por falta de direção** . Você está naquele ponto normal em que a ideia já saiu do papel e bateu na realidade feia da implementação: ASR imperfeito, janela acústica aproximada, regra linguística ainda grosseira e score neural meio instável.

A V1 está no caminho certo. O arquivo `audio_phonetic_events.py` já tem uma arquitetura útil: ele carrega sidecars de transcrição, extrai candidatos, tenta classificar acusticamente `/s/` em coda, `/t,d/` diante de `i` e `/r/` final, associa cada evento ao score neural local e gera um resumo dos eventos classificados. Isso é exatamente o esqueleto que a camada precisa. O problema é que, do jeito que está, ela ainda mistura  **candidato ortográfico** , **evidência acústica** e **evento fonético confiável** como se fossem quase a mesma coisa.

A melhoria principal é separar essas três camadas.

## 1. Separe “candidato”, “evidência” e “decisão fonética”

Hoje, quando o código encontra uma palavra, ele já tenta produzir um evento com `variant`, `confidence`, `source`, `event_quality` etc. Isso é bom para protótipo, mas pode te confundir.

Eu faria assim:

```text
CandidateEvent:
  "essa palavra pode conter um fenômeno relevante"

AcousticEvidence:
  "nesse trecho acústico foram medidas tais características"

PhoneticDecision:
  "com base nas evidências, a variante provável é X, com confiança Y"
```

Exemplo:

```json
{
  "candidate": {
    "phenomenon": "td_before_i",
    "word": "dia",
    "target_segment": "d",
    "expected_context": "phonological_i",
    "start_time": 1.56,
    "end_time": 1.80
  },
  "evidence": {
    "release_rms": 0.029,
    "zcr": 0.066,
    "spectral_centroid_hz": 1305
  },
  "decision": {
    "variant": "dentoalveolar",
    "confidence": 0.82,
    "method": "td_before_i_release_frication"
  }
}
```

Isso deixa claro que o sistema não “sabe” a variante; ele  **estima** .

## 2. Corrija falsos candidatos ortográficos

Esse é um ponto crítico. Hoje o código usa a escrita para achar fenômenos. Isso é normal na V1, mas gera erros.

Um exemplo: a palavra **“disse”** pode acabar gerando evento de `/s/` em coda porque o código vê `ss`. Mas em português, `ss` normalmente representa uma fricativa intervocálica, não um `/s/` em coda. Então esse candidato é linguisticamente ruim.

O mesmo vale para vários casos:

| Caso                                       | Problema                                                       |
| ------------------------------------------ | -------------------------------------------------------------- |
| `ss`em “disse”, “isso”               | Não deve ser tratado como `/s/`em coda                      |
| `s`entre vogais                          | Pode representar /z/ ou /s/, mas não necessariamente coda     |
| `x`,`z`,`ç`,`sc`                  | Podem representar sons relacionados, mas exigem regra própria |
| palavra terminada em `s`seguida de vogal | Pode haver ressilabação, não pausa                          |
| palavras terminadas em `r`               | Nem toda palavra terminada em `r`é verbo no infinitivo      |

Então, antes da acústica, você precisa melhorar o módulo de  **candidatura linguística** .

Para V1, eu faria regras conservadoras:

```text
/s/ em coda interna:
  aceitar apenas s seguido de consoante real, exceto "ss".

/s/ externo:
  aceitar s no final da palavra, mas marcar contexto:
  - diante de pausa;
  - diante de consoante;
  - diante de vogal.

/t,d/ diante de i:
  aceitar "ti" e "di" como /i/ fonológico;
  deixar [i] derivado para V2.

/r/ final:
  aceitar verbos terminados em -ar, -er, -ir,
  mas marcar como "infinitive_candidate", não como infinitivo confirmado.
```

A V1 fica mais confiável se ela for  **menos ambiciosa** .

## 3. Não tente classificar tudo: use “unknown” sem culpa

Uma coisa boa no seu código é que ele já retorna `unknown` em vários casos. Mantenha isso. Na verdade, eu aumentaria ainda mais o uso de `unknown`.

Para essa camada, é melhor dizer:

```text
não tenho evidência suficiente
```

do que classificar errado.

Eu usaria uma política assim:

```text
confidence < 0.70 → não entra em regra fonética forte
0.70–0.80 → evidência moderada
> 0.80 → evidência forte
```

E mais importante:

```text
evento unknown conta para cobertura candidata,
mas não conta para consistência fonética.
```

Isso evita que a camada simbólica tome decisão em cima de lixo.

## 4. Calibre os limiares com poucos áudios reais

Hoje seus limiares acústicos são aproximações manuais. Isso é normal, mas você precisa tratá-los como  **heurísticas iniciais** , não como conhecimento consolidado.

Por exemplo, no código, `/s/` palatal e alveolar são definidos por faixas de `zero_crossing_rate` e `spectral_centroid`. Para `/t,d/`, a decisão também depende de energia, ZCR e centroide. Para `/r/`, você usa energia da cauda e taxa de cruzamento por zero.

Isso é um bom começo, mas precisa de calibração.

Eu faria um mini-conjunto manual:

```text
10 exemplos de /r/ apagado
10 exemplos de /r/ presente
10 exemplos de /s/ alveolar
10 exemplos de /s/ palatal
10 exemplos de /t,d/ dentoalveolar
10 exemplos de /t,d/ palatalizado
```

Mesmo que seja pequeno, já ajuda você a ajustar os limiares olhando os valores reais de:

```text
rms
zcr
spectral_centroid
duração
energia relativa
```

A pergunta deixa de ser “qual limiar eu acho que faz sentido?” e vira:

```text
os exemplos reais estão caindo em quais faixas?
```

## 5. Use o Atlas para interpretar, não para classificar

A parte simbólica está indo bem, mas precisa continuar com essa filosofia:

```text
ALiB não vira regra dura.
ALiB vira justificativa de plausibilidade.
```

Então, em vez de:

```text
Salvador permite X, logo é real.
```

Use:

```text
o fenômeno observado é compatível com variação documentada no PB;
não há ruptura localizada;
portanto, reduzir confiança de acusação automática.
```

Isso é o que torna o projeto defensável.

## 6. Melhore a noção de “ruptura localizada”

Acho que essa é a melhoria mais importante da camada simbólica.

Hoje você já associa evento com `neural_score_local`. Isso é excelente. O próximo passo é criar uma função clara para detectar ruptura.

Algo como:

```python
def detect_local_disruption(events, window_seconds=6):
    """
    Detecta se há mudança concentrada de variantes em uma janela curta,
    comparando eventos antes, durante e depois.
    """
```

A lógica pode ser simples:

```text
1. agrupar eventos por fenômeno e contexto;
2. ordenar por tempo;
3. procurar mudança de variante em contextos semelhantes;
4. verificar se a mudança aparece concentrada em poucos segundos;
5. verificar se coincide com pico neural.
```

A camada só deve aumentar suspeita quando houver:

```text
mudança de variante
+ mesmo contexto fonológico
+ concentração temporal
+ score neural local alto
```

Sem esses quatro elementos, eu não aumentaria suspeita.

## 7. Trate “consistência” por contexto, não globalmente

Cuidado com isto:

```text
/s/ palatal em alguns casos
/s/ alveolar em outros casos
```

Isso pode parecer inconsistência, mas pode ser perfeitamente normal se os contextos forem diferentes.

Então a consistência precisa ser calculada assim:

```text
fenômeno + contexto + posição
```

Exemplo:

```text
s_coda + internal_coda + antes de /t/
s_coda + external_coda + pausa
td_before_i + phonological_i
rotic_coda + verb_infinitive_external_coda
```

Não misture tudo no mesmo balde.

## 8. Faça a camada produzir relatório de incerteza

A saída não deve ser só `score_calibrado`.

Ela deve dizer algo assim:

```json
{
  "symbolic_layer_status": "usable_with_caution",
  "coverage": "moderate",
  "reliable_events": 12,
  "candidate_events": 37,
  "main_limitation": "phonetic classification is heuristic",
  "calibration_action": "reduce_suspicion",
  "reason": [
    "score neural alto próximo ao limiar",
    "eventos fonéticos classificados não indicam ruptura localizada",
    "padrão de /t,d/ diante de [i] apareceu estável",
    "apagamento de /r/ final em infinitivos é plausível no PB"
  ]
}
```

Isso é cientificamente mais honesto do que dizer “real” ou “fake”.

## 9. Ajuste o YAML para incluir força da evidência

Sua tabela de regras está boa, mas eu colocaria três campos novos em cada regra:

```yaml
evidence_type: "plausibility" | "disruption" | "coverage" | "quality"
requires_alignment_with_neural_peak: true/false
applies_to_score: true/false
```

Exemplo:

```yaml
- id: "G6"
  phenomenon: "global"
  condition: "high_neural_score_with_aligned_linguistic_disruption"
  evidence_type: "disruption"
  requires_alignment_with_neural_peak: true
  applies_to_score: true
  weight: 0.12
```

E para uma regra de plausibilidade:

```yaml
- id: "R1"
  phenomenon: "rotic_coda"
  condition: "verb_infinitive_final_r_deletion_consistent"
  evidence_type: "plausibility"
  requires_alignment_with_neural_peak: false
  applies_to_score: true
  weight: -0.05
```

Isso ajuda muito depois para explicar por que uma regra mexeu ou não mexeu no score.

## 10. Minha sugestão de prioridade agora

Eu faria nesta ordem:

```text
1. Corrigir a extração de candidatos ruins, principalmente /s/ em "ss".
2. Separar CandidateEvent, AcousticEvidence e PhoneticDecision.
3. Criar métrica de cobertura confiável:
   reliable_events / candidate_events.
4. Criar consistência por fenômeno + contexto.
5. Criar detecção simples de ruptura localizada.
6. Só depois mexer nos pesos do YAML.
```

O erro mais comum agora seria ficar tentando refinar peso de regra enquanto a extração fonética ainda está ruidosa. Primeiro melhora a qualidade dos eventos; depois calibra os pesos.

## Avaliação direta da sua V1

Eu diria:

```text
Arquitetura: boa.
Ideia da camada: muito boa.
Implementação atual: protótipo funcional.
Principal fragilidade: eventos fonéticos ainda muito aproximados.
Melhoria mais urgente: reduzir falso candidato e separar incerteza.
```

Você não precisa transformar isso agora em um classificador fonético perfeito. Para uma V1 acadêmica, basta deixar claro:

```text
A camada usa eventos candidatos e classificações acústicas conservadoras para estimar consistência fonético-fonológica, com foco em reduzir decisões precipitadas quando o detector neural apresenta scores altos em fala real e regionalmente variável.
```

Esse é um escopo muito bom. O próximo salto de qualidade é fazer a V1 ser  **mais conservadora e mais explícita sobre incerteza** .
