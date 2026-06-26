
Exatamente. Essa tabela é o primeiro artefato concreto da camada neuro-simbólica. Ela vira uma ponte entre:

```text
conhecimento linguístico → regra interpretável → efeito computacional
```

Eu montaria a V1 sem tentar cobrir tudo. Começaria com regras  **prudentes** , focadas em consistência e não em “adivinhar sotaque”.

## Tabela de regras V1 — Camada neuro-simbólica fonético-fonológica

| ID  | Fenômeno                        | Condição observada                                                                                                | Interpretação                                 | Efeito no score                                             | Justificativa linguística                                                                                                                  |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | Rótico em coda externa de verbo | Apagamento de /r/ final em verbos no infinitivo, como “falá”, “trabalhá”, “perguntá”                       | Variação legítima e muito comum no PB falado | Reduzir suspeita ou manter score                            | O apagamento do /r/ final em infinitivos é frequente no português brasileiro e não deve ser tratado como artefato                        |
| R2  | Rótico em coda externa de verbo | Presença e ausência de /r/ alternam de forma estável, dependendo do estilo ou contexto                           | Variação possível, mas exige cautela         | Manter score ou reduzir levemente                           | A realização do /r/ pode variar por monitoramento de fala, classe lexical e contexto                                                      |
| R3  | Rótico em coda                  | Alternância abrupta entre apagamento, glotal, velar, retroflexo ou tepe em contextos semelhantes e no mesmo trecho | Possível ruptura fonético-fonológica         | Aumentar suspeita                                           | Mudanças bruscas de variante rótica sem motivação contextual podem indicar descontinuidade no sinal ou composição de fontes distintas |
| R4  | Rótico em coda interna          | Apagamento de /r/ interno em poucas ocorrências isoladas                                                           | Evidência fraca; não concluir sozinho         | Manter score                                                | O apagamento em coda interna é menos geral que em verbos no infinitivo e precisa de mais contexto                                          |
| R5  | Rótico em coda                  | Variante rótica dominante consistente ao longo do áudio                                                           | Padrão linguístico estável                   | Reduzir suspeita se o neural estiver alto sem outra ruptura | Consistência interna sugere padrão individual legítimo, mesmo que regionalmente marcado                                                  |
| S1  | /s/ em coda                      | /s/ palatal [ʃ, ʒ] recorrente e consistente                                                                       | Variação legítima                            | Reduzir suspeita ou manter score                            | Realizações palatais do /s/ em coda são documentadas em várias capitais e não são exclusivas de uma única região                    |
| S2  | /s/ em coda                      | /s/ alveolar [s, z] recorrente e consistente                                                                        | Variação legítima                            | Reduzir suspeita ou manter score                            | Realizações alveolares também são legítimas e predominam em várias áreas                                                             |
| S3  | /s/ em coda                      | Mudança localizada de alveolar para palatal, ou o inverso, sem mudança clara de contexto fonético                | Possível inconsistência                       | Aumentar suspeita se coincidir com score neural alto        | O /s/ varia, mas sua realização tende a obedecer padrões de localidade, posição e segmento seguinte                                    |
| S4  | /s/ em coda                      | Palatalização ocorre principalmente antes de contextos favorecedores                                              | Padrão plausível                              | Manter score                                                | A consoante seguinte influencia a realização do /s/; palatalização condicionada pelo contexto não deve ser tratada como anomalia       |
| S5  | /s/ em coda externa              | Baixa ou nenhuma palatalização em final de palavra diante de pausa                                                | Variação legítima                            | Manter score                                                | Em várias capitais, a palatalização em coda externa é baixa ou ausente                                                                  |
| TD1 | /t, d/ diante de [i]             | Realização palatalizada/africada [tʃ, dʒ], como “tchi”, “dji”                                               | Variação majoritária no PB urbano            | Reduzir suspeita ou manter score                            | A palatalização de /t, d/ diante de [i] é majoritária em muitas capitais brasileiras                                                    |
| TD2 | /t, d/ diante de [i]             | Realização dentoalveolar [t, d] consistente                                                                       | Variação legítima conservadora               | Manter score ou reduzir suspeita                            | A realização dentoalveolar permanece em algumas áreas e não deve ser tratada como erro                                                  |
| TD3 | /t, d/ diante de [i]             | Alternância abrupta entre [t, d] e [tʃ, dʒ] no mesmo tipo de contexto                                            | Possível ruptura                               | Aumentar suspeita se coincidir com score neural alto        | Mudança brusca entre variantes em contexto equivalente pode indicar inconsistência fonética                                              |
| TD4 | /t, d/ diante de [i]             | Palatalização após fricativa palatal antecedente                                                                 | Padrão fonético plausível por assimilação  | Manter score                                                | Sons vizinhos com traços palatais podem favorecer palatalização                                                                          |
| TD5 | /t, d/ diante de [i]             | Padrão mais palatalizado em fala de pessoa jovem, quando esse dado estiver disponível                             | Padrão sociolinguisticamente plausível        | Não aumentar suspeita                                      | O capítulo aponta maior favorecimento da palatalização entre falantes mais jovens                                                        |
| G1  | Múltiplos fenômenos            | Róticos, /s/ e /t,d/ apresentam padrão estável ao longo do áudio                                                | Alta consistência intra-áudio                 | Reduzir suspeita se o neural estiver alto sem ruptura local | Variação linguística legítima tende a ser sistemática                                                                                  |
| G2  | Múltiplos fenômenos            | Um ou mais fenômenos mudam abruptamente no mesmo intervalo temporal                                                | Ruptura localizada                              | Aumentar suspeita                                           | Manipulações podem gerar descontinuidade localizada entre trechos                                                                         |
| G3  | Múltiplos fenômenos            | Poucos eventos linguísticos detectados                                                                             | Cobertura insuficiente                          | Não calibrar ou calibrar minimamente                       | Sem eventos suficientes, a camada simbólica não deve tomar decisão forte                                                                 |
| G4  | Múltiplos fenômenos            | Áudio com baixa qualidade, ruído intenso ou baixa confiança na extração fonética                              | Evidência linguística fraca                   | Marcar como inconclusivo ou reduzir peso da camada          | A análise fonético-fonológica depende de qualidade mínima do sinal                                                                      |
| G5  | Múltiplos fenômenos            | Score neural alto, mas padrões linguísticos consistentes e plausíveis                                            | Possível falso positivo do neural              | Reduzir score final moderadamente                           | O modelo pode estar reagindo a variação regional legítima                                                                                |
| G6  | Múltiplos fenômenos            | Score neural alto e ruptura fonético-fonológica localizada no mesmo trecho                                        | Evidência convergente de suspeita              | Aumentar score final                                        | Convergência entre detector neural e ruptura linguística aumenta a confiabilidade da suspeita                                             |

## Como transformar isso em lógica computacional

Eu colocaria cada regra com um peso, mas sem pesos agressivos no começo.

Exemplo:

| Efeito                          |             Ajuste inicial sugerido |
| ------------------------------- | ----------------------------------: |
| Reduzir suspeita levemente      |                       -0,03 a -0,07 |
| Reduzir suspeita moderadamente  |                       -0,08 a -0,15 |
| Manter score                    |                                   0 |
| Aumentar suspeita levemente     |                       +0,03 a +0,07 |
| Aumentar suspeita moderadamente |                       +0,08 a +0,15 |
| Marcar inconclusivo             | não altera score, reduz confiança |

A fórmula inicial poderia ser:

```text
score_calibrado = score_neural + soma_ajustes
```

Com travas:

```text
score_calibrado ∈ [0, 1]
ajuste máximo positivo: +0,20
ajuste máximo negativo: -0,20
```

Isso evita que a camada simbólica “mande mais” que o detector neural. Ela calibra, não substitui.

## Exemplo prático

Caso 1:

```text
score_neural = 0,82

Eventos:
- apagamento de /r/ final em verbos: consistente
- /s/ em coda: palatal consistente
- /t,d/ diante de [i]: palatalizado consistente
- sem ruptura localizada
```

Regras ativadas:

```text
R1: -0,05
S1: -0,04
TD1: -0,04
G5: -0,10
```

Resultado:

```text
score_calibrado = 0,82 - 0,23
```

Mas com trava de ajuste máximo negativo:

```text
score_calibrado = 0,82 - 0,20 = 0,62
```

Saída:

```text
classe: inconclusivo
justificativa: score neural alto, mas padrões fonético-fonológicos consistentes e plausíveis no pt-BR
```

Caso 2:

```text
score_neural = 0,71

Eventos:
- áudio quase todo com /s/ alveolar
- trecho 00:18–00:23 muda para /s/ palatal
- no mesmo trecho, /t,d/ também mudam de dentoalveolar para africado
- score neural local sobe nesse trecho
```

Regras ativadas:

```text
S3: +0,08
TD3: +0,08
G2: +0,10
G6: +0,12
```

Com trava de ajuste máximo positivo:

```text
score_calibrado = 0,71 + 0,20 = 0,91
```

Saída:

```text
classe: suspeito
justificativa: ruptura fonético-fonológica localizada coincidente com aumento do score neural
```

## O ponto mais importante

Essa tabela não deve virar “regra de sotaque”. Ela deve virar regra de  **consistência** .

A pergunta central da camada é:

```text
O padrão fonético-fonológico observado é estável, plausível e coerente com o próprio áudio?
```

Não é:

```text
De qual região essa pessoa é?
```

Para uma primeira versão V1

A primeira tabela respondeu:

```text
fenômeno → condição → interpretação → efeito no score
```

Agora esta responde:

```text
regra → quais dados preciso ter para aplicar essa regra?
```

# Tabela V1 — Variáveis computacionais necessárias

| ID  | Fenômeno           | Eventos necessários                                         | Variáveis extraídas                                                               | Condição computacional                                                          | Saída esperada                         |
| --- | ------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------- |
| R1  | /r/ final em verbo  | Palavras terminadas em /r/ verbal, especialmente infinitivos | palavra, classe lexical, presença/ausência do /r/, posição temporal, confiança | Se verbo no infinitivo apresenta apagamento de /r/ com confiança alta            | Marcar como variação plausível       |
| R2  | /r/ final em verbo  | Múltiplas ocorrências de verbos no infinitivo              | variante por ocorrência, trecho, estilo/contexto, confiança                       | Se há variação entre presença e ausência, mas sem ruptura abrupta            | Manter score ou reduzir pouco           |
| R3  | Rótico em coda     | Ocorrências de /r/ em coda interna e externa                | variante rótica, posição, palavra, tempo, score neural local                     | Se variantes mudam abruptamente em janelas próximas e contextos semelhantes      | Aumentar suspeita                       |
| R4  | /r/ em coda interna | Palavras com /r/ em coda interna                             | presença/ausência, variante, palavra, confiança                                  | Se há apagamento isolado em coda interna, mas pouca cobertura                    | Evidência fraca                        |
| R5  | Rótico em coda     | Conjunto de ocorrências de /r/ no áudio                    | variante dominante, dispersão, consistência, confiança média                    | Se uma variante domina e aparece de modo estável                                 | Aumentar índice de consistência       |
| S1  | /s/ em coda         | Ocorrências de /s/ em coda interna/externa                  | variante alveolar/palatal, posição, segmento seguinte, tempo                      | Se /s/ palatal aparece de modo recorrente e estável                              | Marcar como plausível                  |
| S2  | /s/ em coda         | Ocorrências de /s/ em coda                                  | variante alveolar/palatal, posição, confiança                                    | Se /s/ alveolar aparece de modo recorrente e estável                             | Marcar como plausível                  |
| S3  | /s/ em coda         | Sequência temporal de ocorrências de /s/                   | variante, tempo, segmento seguinte, score local                                     | Se há troca abrupta de variante sem mudança de contexto e com score neural alto | Aumentar suspeita                       |
| S4  | /s/ em coda         | /s/ seguido por diferentes segmentos                         | segmento seguinte, variante realizada, posição silábica                          | Se palatalização ocorre diante de contexto favorecedor                          | Manter score                            |
| S5  | /s/ em coda externa | /s/ final diante de pausa                                    | posição externa, pausa seguinte, variante                                         | Se há baixa palatalização em final de palavra diante de pausa                  | Não penalizar                          |
| TD1 | /t,d/ diante de [i] | Palavras com /t,d/ antes de /i/ fonológico ou [i] derivado  | segmento alvo, tipo de [i], variante [t,d]/[tʃ,dʒ], tempo                         | Se ocorre variante palatalizada com confiança alta                               | Marcar como plausível                  |
| TD2 | /t,d/ diante de [i] | Ocorrências de /t,d/ antes de [i]                           | variante dentoalveolar, consistência, confiança                                   | Se [t,d] aparece de modo estável                                                 | Marcar como variação legítima        |
| TD3 | /t,d/ diante de [i] | Sequência temporal de ocorrências /t,d/ + [i]              | variante, tipo de [i], contexto, score local                                        | Se alternância abrupta ocorre no mesmo tipo de contexto                          | Aumentar suspeita                       |
| TD4 | /t,d/ diante de [i] | Ocorrências com consoante antecedente                       | consoante antecedente, variante de /t,d/, confiança                                | Se fricativa palatal antecedente favorece [tʃ,dʒ]                               | Manter score                            |
| TD5 | /t,d/ diante de [i] | Ocorrências de /t,d/ + [i] e metadado etário, se existir   | idade/faixa etária, variante, consistência                                        | Se falante jovem usa mais palatalização                                         | Não penalizar                          |
| G1  | Geral               | Eventos de múltiplos fenômenos                             | consistência por fenômeno, consistência global                                   | Se róticos, /s/ e /t,d/ mantêm padrões estáveis                               | Reduzir suspeita se neural estiver alto |
| G2  | Geral               | Eventos em janelas temporais                                 | mudança de variante, janela, score local                                           | Se vários fenômenos mudam no mesmo trecho                                       | Aumentar suspeita                       |
| G3  | Geral               | Contagem de eventos                                          | número de eventos válidos, confiança média                                      | Se há poucos eventos detectáveis                                                | Reduzir peso da camada                  |
| G4  | Geral               | Métricas de qualidade do áudio                             | SNR, ruído, clipping, confiança de ASR/alinhamento                                | Se qualidade é baixa                                                             | Marcar como inconclusivo                |
| G5  | Geral               | Score neural + consistência linguística                    | score global, score local, plausibilidade, consistência                            | Se neural alto, mas linguística consistente                                      | Reduzir score moderadamente             |
| G6  | Geral               | Score neural local + ruptura linguística                    | score por trecho, ruptura por trecho, alinhamento temporal                          | Se ruptura coincide com pico neural                                               | Aumentar score moderadamente            |

# Variáveis centrais da camada

Agora dá para definir as variáveis principais do sistema.

## 1. Variáveis por evento

Cada evento fonético-fonológico detectado poderia ter esta estrutura:

```json
{
  "event_id": "ev_001",
  "phenomenon": "s_coda",
  "word": "mesmo",
  "start_time": 12.35,
  "end_time": 12.58,
  "position": "internal_coda",
  "target_segment": "s",
  "following_segment": "m",
  "variant": "palatal",
  "variant_confidence": 0.84,
  "neural_score_local": 0.37
}
```

Para /r/:

```json
{
  "event_id": "ev_002",
  "phenomenon": "rotic_coda",
  "word": "trabalhar",
  "start_time": 21.10,
  "end_time": 21.62,
  "position": "external_coda",
  "lexical_class": "verb_infinitive",
  "variant": "deletion",
  "variant_confidence": 0.88,
  "neural_score_local": 0.42
}
```

Para /t,d/:

```json
{
  "event_id": "ev_003",
  "phenomenon": "td_before_i",
  "word": "dia",
  "start_time": 31.20,
  "end_time": 31.43,
  "target_segment": "d",
  "following_vowel": "i_phonological",
  "preceding_segment": "pause",
  "variant": "palatalized",
  "variant_confidence": 0.91,
  "neural_score_local": 0.39
}
```

# Variáveis globais da camada

Depois de extrair os eventos, a camada calcularia índices globais.

| Variável                         | Descrição                                            |         Escala | Uso                                  |
| --------------------------------- | ------------------------------------------------------ | -------------: | ------------------------------------ |
| `event_count`                   | Número de eventos linguísticos válidos              |        inteiro | Mede cobertura                       |
| `coverage_score`                | Grau de cobertura linguística                         |          0 a 1 | Decide se a camada pode atuar        |
| `audio_quality_score`           | Qualidade do áudio                                    |          0 a 1 | Reduz peso se áudio ruim            |
| `asr_confidence`                | Confiança da transcrição/alinhamento                |          0 a 1 | Evita análise em transcrição ruim |
| `phonetic_consistency_score`    | Consistência fonético-fonológica geral              |          0 a 1 | Reduz suspeita quando alta           |
| `localized_disruption_score`    | Grau de ruptura localizada                             |          0 a 1 | Aumenta suspeita quando alta         |
| `linguistic_plausibility_score` | Plausibilidade dos padrões encontrados                |          0 a 1 | Reduz falso positivo                 |
| `neural_global_score`           | Score global do detector neural                        |          0 a 1 | Entrada principal                    |
| `neural_local_peaks`            | Trechos com score neural alto                          | lista temporal | Comparar com rupturas                |
| `alignment_score`               | Coincidência entre ruptura linguística e pico neural |          0 a 1 | Aumenta suspeita se alto             |

# Como calcular os índices

## 1. Cobertura linguística

A camada só deve atuar se houver eventos suficientes.

```text
coverage_score = min(event_count / min_required_events, 1.0)
```

Exemplo:

```text
min_required_events = 10
event_count = 3  → coverage_score = 0.3
event_count = 10 → coverage_score = 1.0
event_count = 25 → coverage_score = 1.0
```

Na prática:

```text
coverage_score < 0.3 → não calibrar
0.3 a 0.6 → calibrar com peso baixo
> 0.6 → calibrar normalmente
```

## 2. Consistência por fenômeno

Para cada fenômeno, você calcula se o padrão é estável.

Exemplo para /s/ em coda:

```text
/s/ palatal: 18 ocorrências
/s/ alveolar: 2 ocorrências

consistência alta
```

Exemplo:

```text
/s/ palatal: 10 ocorrências
/s/ alveolar: 9 ocorrências

consistência baixa ou média
```

Uma forma simples:

```text
consistency = frequência_da_variante_dominante
```

Então:

```text
18/20 = 0.90
10/19 = 0.52
```

Mas o ideal é ponderar por contexto. Não é inconsistente variar se o contexto muda.

Por isso, melhor:

```text
consistency = média da consistência dentro de cada contexto equivalente
```

Exemplo:

```text
/s/ em coda externa diante de pausa
/s/ em coda interna antes de /t/
/s/ em coda interna antes de /m/
```

Cada contexto precisa ser avaliado separadamente.

## 3. Ruptura localizada

A ruptura localizada aparece quando há mudança forte em um trecho curto.

Exemplo:

```text
0s–30s: /s/ alveolar dominante
30s–35s: /s/ palatal dominante
35s–60s: /s/ alveolar dominante
```

Isso é mais suspeito do que:

```text
/s/ alveolar em alguns contextos
/s/ palatal em outros contextos
```

Uma métrica simples:

```text
localized_disruption_score =
mudança_de_variante * concentração_temporal * ausência_de_motivação_contextual
```

Onde:

```text
mudança_de_variante: 0 a 1
concentração_temporal: 0 a 1
ausência_de_motivação_contextual: 0 a 1
```

## 4. Alinhamento com o score neural

Esse é um dos pontos mais importantes.

A ruptura só deve pesar muito se coincidir com o detector neural.

```text
alignment_score =
sobreposição temporal entre picos neurais e rupturas linguísticas
```

Exemplo:

```text
pico neural: 00:18–00:24
ruptura /s/: 00:19–00:23

alignment_score alto
```

Outro exemplo:

```text
pico neural: 00:18–00:24
mudança linguística: 00:45–00:50

alignment_score baixo
```

# Estrutura de saída da camada

A saída da camada poderia ser assim:

```json
{
  "neural_global_score": 0.78,
  "calibrated_score": 0.64,
  "final_label": "inconclusive",
  "confidence": "medium",
  "coverage_score": 0.82,
  "phonetic_consistency_score": 0.88,
  "linguistic_plausibility_score": 0.91,
  "localized_disruption_score": 0.18,
  "alignment_score": 0.12,
  "activated_rules": ["R1", "S1", "TD1", "G5"],
  "explanation": [
    "O detector neural produziu score alto.",
    "Os padrões fonético-fonológicos observados foram consistentes ao longo do áudio.",
    "O apagamento de /r/ em verbos no infinitivo é plausível no português brasileiro falado.",
    "Não houve ruptura linguística localizada coincidente com os picos do detector neural."
  ]
}
```

Outro exemplo:

```json
{
  "neural_global_score": 0.72,
  "calibrated_score": 0.89,
  "final_label": "suspect",
  "confidence": "high",
  "coverage_score": 0.76,
  "phonetic_consistency_score": 0.41,
  "linguistic_plausibility_score": 0.52,
  "localized_disruption_score": 0.86,
  "alignment_score": 0.91,
  "activated_rules": ["S3", "TD3", "G2", "G6"],
  "explanation": [
    "Houve ruptura fonético-fonológica localizada.",
    "A mudança de variante em /s/ em coda coincide temporalmente com pico do detector neural.",
    "A alternância de /t,d/ diante de [i] ocorreu em contexto semelhante no mesmo trecho.",
    "A convergência entre score neural e ruptura linguística aumentou a suspeita."
  ]
}
```

# Primeira versão do algoritmo

Em pseudocódigo:

```text
entrada:
  audio

1. neural_scores = detector_neural(audio)

2. transcript = ASR(audio)

3. alignment = alinhar_audio_texto(audio, transcript)

4. events = extrair_eventos_foneticos(audio, transcript, alignment)

5. coverage_score = calcular_cobertura(events)

6. se coverage_score < limite_minimo:
       retornar score_neural sem calibração forte

7. consistency_score = calcular_consistencia(events)

8. plausibility_score = calcular_plausibilidade(events, base_linguistica)

9. disruption_score = detectar_rupturas(events)

10. alignment_score = comparar_rupturas_com_picos_neurais(
        disruption_score,
        neural_scores
    )

11. activated_rules = aplicar_regras(
        events,
        consistency_score,
        plausibility_score,
        disruption_score,
        alignment_score
    )

12. adjustment = somar_ajustes(activated_rules)

13. adjustment = limitar(adjustment, -0.20, +0.20)

14. calibrated_score = neural_global_score + adjustment

15. retornar:
       score calibrado,
       classe,
       confiança,
       regras ativadas,
       explicação
```

# Tabela de entidades mínimas

Para implementar depois, eu dividiria em quatro entidades.

## `PhoneticEvent`

```text
Representa um fenômeno fonético-fonológico detectado.
```

Campos:

```text
event_id
phenomenon
word
start_time
end_time
target_segment
variant
position
context
confidence
neural_score_local
```

## `LinguisticRule`

```text
Representa uma regra simbólica.
```

Campos:

```text
rule_id
phenomenon
condition
effect
weight
explanation_template
```

## `CalibrationReport`

```text
Representa o resultado da camada.
```

Campos:

```text
neural_score
calibrated_score
coverage_score
consistency_score
plausibility_score
disruption_score
alignment_score
activated_rules
final_label
confidence
explanation
```

## `KnowledgeBase`

```text
Representa o conhecimento linguístico usado pela camada.
```

Campos:

```text
phenomena
variants
contexts
plausibility_rules
rule_weights
references
```

# Escopo realista para V1

Para não explodir o projeto, eu faria a V1 assim:

```text
V1:
- entrada: áudio + score neural;
- usar ASR e alinhamento aproximado;
- detectar eventos candidatos;
- aplicar regras só para:
  1. /r/ em coda externa de verbo;
  2. /s/ em coda;
  3. /t,d/ diante de [i];
- gerar relatório explicável;
- calibrar score com limite máximo de ±0,20.
```

E deixaria fora da V1:

```text
- inferência regional direta;
- classificação de sotaque;
- regras complexas para vogais;
- decisão automática definitiva;
- pesos muito agressivos;
- tentativa de provar origem do falante.
```

Essa tabela já te dá um primeiro desenho de implementação. A próxima etapa natural seria transformar isso em um **modelo de dados em JSON/YAML** para representar as regras simbólicas.
