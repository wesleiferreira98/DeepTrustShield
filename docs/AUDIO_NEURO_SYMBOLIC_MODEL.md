# Modelo Neuro-Simbólico de Áudio

## Objetivo

O modelo neuro-simbólico de áudio combina duas leituras diferentes sobre o mesmo arquivo:

1. uma leitura neural, baseada em espectrogramas log-Mel e scores por janela;
2. uma leitura simbólica, baseada em qualidade acústica, comportamento temporal e padrões fonético-fonológicos do português brasileiro.

A camada neural responde à pergunta: “este áudio se parece com os exemplos de áudio fake vistos no treino?”.

A camada simbólica responde a outra pergunta: “a suspeita neural é coerente com o comportamento interno do áudio ou pode ser explicada por variação linguística, domínio fora do treino, baixa cobertura fonética, compressão, silêncio ou artefatos de encerramento?”.

Na V1, a camada simbólica não substitui o modelo neural. Ela calibra, contesta, reforça ou torna inconclusiva a decisão quando a evidência não é forte o suficiente.

## Visão Geral do Fluxo

```text
áudio enviado pelo usuário
  -> normalização técnica para análise neural
  -> inferência neural global
  -> inferência neural por janelas
  -> transcrição automática, quando disponível
  -> extração de eventos fonético-fonológicos
  -> aplicação da base simbólica
  -> memória temporal
  -> camada simbólica independente
  -> decisão final + relatório amigável
```

O modelo neural e a camada simbólica trabalham sobre informações diferentes. O neural trabalha com espectrograma. A simbólica trabalha com metadados, segmentos, diagnósticos acústicos, transcrição, eventos fonéticos e padrões temporais.

## Entrada Neural

O modelo neural não recebe `.mp3`, `.ogg` ou `.wav` diretamente como entrada matemática. O arquivo é carregado e transformado antes da inferência.

O fluxo atual é:

1. carregar o áudio com `torchaudio`;
2. converter para mono;
3. reamostrar para 16 kHz;
4. dividir em janelas de 4 segundos;
5. converter cada janela em espectrograma log-Mel;
6. normalizar a representação numérica;
7. enviar o tensor para a CNN de áudio.

A saída neural principal é `p_fake`, isto é, uma probabilidade estimada de o áudio pertencer à classe fake. Além do score global, o sistema calcula scores por janelas temporais. Essas janelas permitem identificar se a suspeita está espalhada no áudio inteiro ou concentrada em poucos trechos.

## Entrada Simbólica

A camada simbólica recebe:

- `p_fake` global;
- threshold calibrado do modelo selecionado;
- lista de janelas com score neural;
- duração do áudio;
- taxa de silêncio;
- volume médio;
- clipping;
- diagnóstico dos trechos acima do limite;
- transcrição automática, quando disponível;
- palavras alinhadas no tempo, quando disponíveis;
- eventos fonético-fonológicos extraídos da transcrição;
- memória temporal, quando ativada.

Essa camada é chamada de simbólica porque usa regras explícitas, auditáveis e legíveis. Ela não tenta “aprender tudo” de forma opaca. Ela aplica condições como: “houve alternância abrupta de variante no mesmo contexto?”, “há poucos eventos confiáveis?”, “a suspeita está só no final?”, “a qualidade do áudio reduz a confiança?”.

## Fenômenos Fonético-Fonológicos

A base V1 observa três grupos principais de fenômenos.

### `/r/` em coda

No sistema, esse fenômeno aparece como `rotic_coda`.

A coda é a posição final de uma sílaba. No caso de `/r/`, a camada observa realizações como:

- apagamento do `/r/`;
- presença do `/r/`;
- variante tepe;
- variante retroflexa;
- variante velar;
- variante glotal;
- outras realizações.

Um exemplo importante é o `/r/` final em verbos no infinitivo, como `gravar`, `falar`, `poder`, `entrar` e `solucionar`.

No português brasileiro falado, o apagamento do `/r/` final em infinitivos é comum e não deve ser tratado automaticamente como indício de manipulação. Por isso, quando o sistema encontra esse padrão de forma consistente, a suspeita pode ser reduzida.

O front exibe esse grupo como:

```text
/r/ final
```

### `/s/` em coda

No sistema, esse fenômeno aparece como `s_coda`.

A camada observa o `/s/` em final de sílaba ou final de palavra, em contextos como:

- coda interna;
- coda externa diante de pausa;
- coda externa diante de consoante;
- coda externa diante de vogal.

As variantes previstas incluem:

- alveolar;
- palatal;
- assimilação de vozeamento;
- apagamento;
- outras realizações.

A palatalização de `/s/` em coda pode ser uma variação natural do português brasileiro. Em alguns contextos, uma realização como `[ʃ]` ou `[ʒ]` pode ser esperada. A regra não tenta identificar sotaque. Ela observa se o padrão é estável dentro do próprio áudio.

O front exibe esse grupo como:

```text
/s/ coda
```

### `/t,d/` diante de `[i]`

No sistema, esse fenômeno aparece como `td_before_i`.

A camada observa a realização de `/t/` e `/d/` antes de vogal alta anterior, como em palavras do tipo:

- `dia`;
- `tipo`;
- `direto`;
- `definitivo`;
- `áudio`.

As variantes previstas incluem:

- dentoalveolar;
- palatalizada;
- africada;
- outras realizações.

Em muitas variedades do português brasileiro, `/t/` e `/d/` diante de `[i]` podem ser palatalizados ou africados. Em outras, podem permanecer dentoalveolares. O ponto central da V1 não é decidir qual variante é “correta”, mas observar se a variante aparece com estabilidade ou se muda abruptamente no mesmo contexto.

O front exibe esse grupo como:

```text
/t,d/ + i
```

## Extração dos Eventos Fonéticos

A camada fonética depende de transcrição e, idealmente, de tempos por palavra. Quando o usuário envia um áudio novo pela plataforma web, o worker tenta gerar a transcrição automaticamente. A partir dessa transcrição, o sistema cria candidatos fonéticos.

Um candidato pode virar evento confiável quando há evidência acústica suficiente. Por isso, a V1 separa:

- candidato ortográfico;
- evento com evidência acústica;
- evento confiável;
- evento fraco ou inconclusivo.

Essa separação é importante porque a camada não deve forçar uma leitura fonética quando o áudio, o alinhamento ou a janela são insuficientes.

## Métricas Simbólicas

### Cobertura fonética

Indica se há eventos suficientes para aplicar a camada fonético-fonológica. Um áudio curto ou com poucas palavras úteis pode ter cobertura baixa.

Quando a cobertura é baixa, a camada evita fazer afirmações fortes. O front mostra esse caso como:

```text
Cobertura fonética baixa
```

### Qualidade da evidência

Resume a confiabilidade dos eventos classificados. Ela considera quantidade de eventos, confiança média, diversidade de fenômenos e qualidade dos candidatos classificados.

Uma qualidade baixa não significa que o áudio é fake. Significa que a camada simbólica tem pouca base para calibrar a decisão.

### Consistência

Mede se os padrões fonético-fonológicos aparecem de forma estável dentro do próprio áudio.

Alta consistência tende a favorecer fala natural, principalmente quando o modelo neural marcou score alto sem encontrar uma ruptura fonética clara.

### Ruptura localizada

Mede se houve mudança abrupta de variante em contexto semelhante.

Exemplo: o áudio usa uma variante estável de `/t,d/` diante de `[i]`, mas em um trecho específico muda de forma brusca, no mesmo tipo de contexto. Isso pode sugerir troca de falante, recorte, inserção ou outra manipulação localizada.

Na V1, ruptura localizada é um indício. Ela não é prova isolada de manipulação.

### Alinhamento com pico

Mede se a ruptura linguística coincide temporalmente com um pico neural. Quando a mudança fonética acontece no mesmo trecho em que o modelo neural marcou alta suspeita, a evidência fica mais forte.

### Memória temporal

A memória temporal compara cada janela com o comportamento interno do próprio áudio. Ela procura trechos que destoam do padrão global do arquivo.

Ela pode marcar candidatos mesmo quando a camada fonética não tem cobertura suficiente. No front, esse recurso aparece como:

```text
Memória temporal
Variação temporal
```

O objetivo da memória não é dizer “fake”. O objetivo é destacar trechos que merecem explicação automática por destoarem do restante do áudio.

### Simbólico independente

O `independent_symbolic` é uma leitura simbólica que busca reduzir a dependência do modelo neural selecionado.

Ele considera principalmente:

- cobertura fonética;
- qualidade da evidência;
- consistência;
- ruptura fonética bruta;
- plausibilidade linguística;
- avisos de qualidade do áudio.

Essa camada pode classificar sua leitura como:

- `natural`;
- `inconclusive`;
- `suspicious`;
- `insufficient`.

Ela ajuda a contestar falsos positivos neurais, principalmente em casos de domínio fora do treino.

## Regras Simbólicas da Base Fonética

As regras abaixo vêm da base `audio/knowledge/phonetic_rules_v1.yaml`. Cada regra tem um identificador, um fenômeno, uma condição e um peso. Pesos negativos reduzem suspeita. Pesos positivos aumentam suspeita. Peso zero mantém a camada neutra, mas registra uma explicação.

### R1 Apagamento de `/r/` final em infinitivos

Fenômeno: `rotic_coda`.

Condição: `verb_infinitive_final_r_deletion_consistent`.

A regra é ativada quando há apagamento consistente de `/r/` final em verbos no infinitivo.

Interpretação: esse apagamento é frequente no português brasileiro falado. Portanto, quando aparece como padrão estável, ele reduz a suspeita.

Peso: `-0.05`.

Texto exibido:

```text
O apagamento de /r/ final em verbos no infinitivo é plausível no português brasileiro falado.
```

### R2 Alternância de `/r/` sem ruptura clara

Fenômeno: `rotic_coda`.

Condição: `final_r_presence_and_deletion_without_local_disruption`.

A regra é ativada quando há presença e apagamento de `/r/`, mas sem ruptura temporal localizada.

Interpretação: a alternância pode refletir estilo, contexto, monitoramento de fala ou variação natural.

Peso: `-0.03`.

Texto exibido:

```text
A alternância de /r/ final apareceu sem ruptura localizada clara.
```

### R3  Mudança abrupta de `/r/` no mesmo contexto

Fenômeno: `rotic_coda`.

Condição: `abrupt_variant_switch_same_context`.

A regra é ativada quando há troca abrupta de variante rótica em contexto semelhante e alinhada a uma ruptura localizada.

Interpretação: pode indicar troca de fonte, edição, inserção ou quebra de continuidade.

Peso: `+0.08`.

Texto exibido:

```text
Houve mudança abrupta de variante rótica em contexto semelhante.
```

### R4  Apagamento isolado de `/r/` interno

Fenômeno: `rotic_coda`.

Condição: `isolated_internal_coda_deletion`.

A regra trata apagamento isolado de `/r/` em coda interna como evidência fraca.

Interpretação: não deve ser usado isoladamente para aumentar suspeita.

Peso: `0.00`.

Texto exibido:

```text
O apagamento isolado de /r/ em coda interna não foi usado como evidência forte.
```

### R5  Variante dominante estável

Fenômeno: `rotic_coda`.

Condição: `dominant_variant_stable`.

A regra é ativada quando uma variante rótica dominante permanece estável no áudio.

Interpretação: estabilidade interna favorece fala natural ou padrão individual consistente.

Peso: `-0.04`.

Texto exibido:

```text
A variante rótica dominante permaneceu estável ao longo do áudio.
```

### S1  `/s/` palatal recorrente e estável

Fenômeno: `s_coda`.

Condição: `palatal_variant_recurrent_and_stable`.

A regra é ativada quando a realização palatal de `/s/` em coda aparece de forma recorrente e estável.

Interpretação: é uma variação legítima no português brasileiro e não deve ser penalizada quando aparece como padrão.

Peso: `-0.04`.

Texto exibido:

```text
A realização palatal de /s/ em coda apareceu de modo recorrente e estável.
```

### S2  `/s/` alveolar recorrente e estável

Fenômeno: `s_coda`.

Condição: `alveolar_variant_recurrent_and_stable`.

A regra é ativada quando a realização alveolar de `/s/` em coda aparece como padrão recorrente.

Interpretação: também é uma variação legítima do português brasileiro.

Peso: `-0.03`.

Texto exibido:

```text
A realização alveolar de /s/ em coda apareceu de modo recorrente e estável.
```

### S3  Troca abrupta entre `/s/` alveolar e palatal

Fenômeno: `s_coda`.

Condição: `abrupt_alveolar_palatal_switch_same_context`.

A regra é ativada quando há mudança abrupta entre `/s/` alveolar e palatal em contexto semelhante.

Interpretação: pode indicar ruptura fonética localizada.

Peso: `+0.08`.

Texto exibido:

```text
Houve troca abrupta entre /s/ alveolar e palatal em contexto semelhante.
```

### S4  Palatalização em contexto favorecedor

Fenômeno: `s_coda`.

Condição: `palatalization_in_favoring_context`.

A regra é ativada quando a palatalização de `/s/` ocorre em contexto fonético favorecedor.

Interpretação: o padrão é plausível e não deve aumentar suspeita.

Peso: `0.00`.

Texto exibido:

```text
A palatalização de /s/ ocorreu em contexto fonético favorecedor.
```

### S5 Baixa palatalização diante de pausa

Fenômeno: `s_coda`.

Condição: `low_palatalization_external_coda_pause`.

A regra é ativada quando há baixa palatalização de `/s/` em final de palavra diante de pausa.

Interpretação: essa ausência de palatalização também pode ser natural, dependendo do falante e do contexto.

Peso: `0.00`.

Texto exibido:

```text
A baixa palatalização de /s/ final diante de pausa não foi penalizada.
```

### TD1  Palatalização de `/t,d/` diante de `[i]`

Fenômeno: `td_before_i`.

Condição: `palatalized_variant_recurrent`.

A regra é ativada quando `/t,d/` diante de `[i]` aparece de forma palatalizada ou africada de maneira recorrente.

Interpretação: esse padrão é plausível em muitas variedades urbanas do português brasileiro.

Peso: `-0.04`.

Texto exibido:

```text
A palatalização de /t,d/ diante de [i] apareceu como padrão plausível.
```

### TD2  Realização dentoalveolar estável

Fenômeno: `td_before_i`.

Condição: `dentoalveolar_variant_stable`.

A regra é ativada quando `/t,d/` diante de `[i]` permanece dentoalveolar de forma estável.

Interpretação: é uma variante conservadora legítima. A estabilidade interna reduz a suspeita.

Peso: `-0.03`.

Texto exibido:

```text
A realização dentoalveolar de /t,d/ diante de [i] apareceu de modo estável.
```

### TD3 Alternância abrupta de `/t,d/` diante de `[i]`

Fenômeno: `td_before_i`.

Condição: `abrupt_dentoalveolar_palatalized_switch_same_context`.

A regra é ativada quando há alternância abrupta entre realização dentoalveolar e palatalizada em contexto semelhante.

Interpretação: pode indicar ruptura fonético-fonológica localizada.

Peso: `+0.08`.

Texto exibido:

```text
Houve alternância abrupta de /t,d/ diante de [i] em contexto semelhante.
```

### TD4  Palatalização após contexto palatal

Fenômeno: `td_before_i`.

Condição: `palatalization_after_palatal_fricative`.

A regra é ativada quando a palatalização ocorre após contexto palatal favorecedor.

Interpretação: a realização é plausível por assimilação e não deve aumentar suspeita.

Peso: `0.00`.

Texto exibido:

```text
A palatalização de /t,d/ ocorreu após contexto palatal favorecedor.
```

### G1  Múltiplos fenômenos estáveis

Fenômeno: `global`.

Condição: `multiple_phenomena_stable`.

A regra é ativada quando diferentes fenômenos fonético-fonológicos apresentam estabilidade no mesmo áudio.

Interpretação: estabilidade em múltiplos fenômenos favorece fala natural.

Peso: `-0.10`.

Texto exibido:

```text
Os fenômenos fonético-fonológicos observados mantiveram padrão estável.
```

### G2  Múltiplos fenômenos mudaram no mesmo intervalo

Fenômeno: `global`.

Condição: `multiple_phenomena_shift_same_interval`.

A regra é ativada quando mais de um fenômeno muda no mesmo intervalo temporal.

Interpretação: isso é uma evidência mais forte de possível ruptura localizada, porque não depende de apenas um evento isolado.

Peso: `+0.10`.

Texto exibido:

```text
Múltiplos fenômenos mudaram no mesmo intervalo temporal.
```

### G3  Cobertura linguística insuficiente

Fenômeno: `global`.

Condição: `insufficient_linguistic_coverage`.

A regra é ativada quando há poucos eventos confiáveis para aplicar a camada fonético-fonológica.

Interpretação: o sistema deve reduzir o peso simbólico e evitar conclusão forte.

Peso: `0.00`.

Texto exibido:

```text
Há poucos eventos linguístico-fonéticos confiáveis para aplicar a camada.
```

### G4  Qualidade de áudio ou alinhamento baixa

Fenômeno: `global`.

Condição: `low_audio_or_alignment_quality`.

A regra representa casos em que a qualidade do áudio ou do alinhamento reduz a confiabilidade da leitura fonética.

Interpretação: a evidência linguística deve ser tratada como fraca.

Peso: `0.00`.

Texto exibido:

```text
A qualidade do áudio ou do alinhamento reduz a confiabilidade da análise fonética.
```

### G5  Score neural alto com consistência linguística

Fenômeno: `global`.

Condição: `high_neural_score_with_consistent_linguistics`.

A regra é ativada quando o score neural está alto, mas os padrões fonético-fonológicos aparecem consistentes.

Interpretação: pode haver falso positivo neural por domínio fora do treino, variação legítima ou características de canal.

Peso: `-0.10`.

Texto exibido:

```text
O score neural está alto, mas os padrões fonético-fonológicos foram consistentes.
```

### G6  Ruptura linguística alinhada ao pico neural

Fenômeno: `global`.

Condição: `high_neural_score_with_aligned_linguistic_disruption`.

A regra é ativada quando há ruptura linguística localizada e ela coincide com pico do detector neural.

Interpretação: essa é uma das evidências convergentes mais relevantes da V1.

Peso: `+0.12`.

Texto exibido:

```text
A ruptura linguística localizada coincidiu com pico do detector neural.
```

## Regras e Flags Aplicadas no Front End

O front end não recalcula as regras. Ele exibe, traduz e organiza o resultado produzido pelo worker.

As principais flags exibidas são:

| Flag técnica                         | Nome amigável no front    | Sentido                                                                          |
| ------------------------------------- | -------------------------- | -------------------------------------------------------------------------------- |
| `short_audio`                       | Áudio curto               | A duração reduz a confiança da análise.                                      |
| `very_short_audio`                  | Áudio muito curto         | O áudio é curto demais para uma conclusão forte.                              |
| `high_silence`                      | Muito silêncio            | Silêncio excessivo pode distorcer a leitura.                                    |
| `low_volume`                        | Volume baixo               | Volume baixo reduz confiabilidade.                                               |
| `clipped_audio`                     | Áudio saturado            | Há clipping ou saturação.                                                     |
| `near_threshold`                    | Score perto do limite      | O score neural ficou próximo do threshold calibrado.                            |
| `strong_fake_score`                 | Score fake forte           | O score neural ficou fortemente acima do limite.                                 |
| `strong_real_score`                 | Score real forte           | O score neural favoreceu áudio real.                                            |
| `segment_score_disagreement`        | Segmentos divergentes      | As janelas discordaram bastante entre si.                                        |
| `isolated_suspicious_segment`       | Suspeita localizada        | Poucas janelas ficaram suspeitas.                                                |
| `consistent_suspicious_segments`    | Suspeita consistente       | Muitas janelas sustentaram suspeita.                                             |
| `domain_mismatch_warning`           | Domínio fora do treino    | Há hipótese de falso positivo por domínio.                                    |
| `phonetic_layer_available`          | Base fonética carregada   | A base fonético-fonológica foi executada.                                      |
| `phonetic_coverage_insufficient`    | Cobertura fonética baixa  | Não houve eventos suficientes.                                                  |
| `phonetic_consistency_support`      | Consistência fonética    | A fonética favoreceu estabilidade.                                              |
| `phonetic_disruption_alert`         | Ruptura fonética          | A fonética encontrou ruptura relevante.                                         |
| `terminal_suspicious_segments`      | Pico no final              | A suspeita está no encerramento do áudio.                                      |
| `possible_end_artifact`             | Possível artefato final   | O pico pode ser explicado por características do final.                         |
| `temporal_memory_available`         | Memória temporal          | A análise temporal interna foi executada.                                       |
| `temporal_memory_rupture_candidate` | Variação temporal        | A memória encontrou trecho destoante.                                           |
| `symbolic_contestation`             | Contestação simbólica   | A camada simbólica contestou parte da suspeita neural.                          |
| `neural_suspicion_contested`        | Suspeita neural contestada | O neural marcou suspeita, mas a simbólica não confirmou de forma independente. |
| `low_temporal_coverage`             | Cobertura temporal baixa   | Há poucas janelas para sustentar análise temporal forte.                       |
| `domain_shift_warning`              | Alerta de domínio         | O áudio pode estar fora do domínio do treino.                                  |

## Como a Decisão Final é Construída

A decisão final considera:

- score neural;
- score simbólico;
- score final combinado;
- quantidade de janelas acima do limite;
- distribuição temporal da suspeita;
- cobertura fonética;
- qualidade da evidência;
- consistência fonética;
- ruptura localizada;
- memória temporal;
- contestação simbólica independente;
- sinais de domínio fora do treino.

Quando o neural acusa fake, mas a simbólica encontra consistência linguística forte, o sistema pode contestar a suspeita e retornar `inconclusive`.

Quando o neural acusa fake e a simbólica encontra ruptura localizada alinhada ao pico neural, o sistema pode reforçar a suspeita.

Quando não há cobertura suficiente, a camada simbólica evita afirmar demais. Nesses casos, a decisão pode continuar dependendo mais do neural, mas o relatório informa a limitação.

## O Que a Camada Não Faz

A V1 não identifica sotaque do falante.

A V1 não afirma região de origem.

A V1 não trata variação linguística como erro.

A V1 não transforma uma regra isolada em prova de manipulação.

A V1 não deve ser lida como ferramenta pericial conclusiva. Ela é uma camada de apoio à decisão, explicação e controle de falso positivo.

## Limitações Atuais

As heurísticas acústicas ainda são aproximações.

Áudios curtos reduzem muito a utilidade da camada simbólica.

A qualidade da transcrição afeta a extração dos eventos.

O alinhamento por palavra ainda não é um alinhamento fonético fino.

Modelos neurais de áudio são sensíveis a domínio. Áudios muito limpos, formais ou diferentes do treino podem elevar a suspeita mesmo sendo reais.

## Conclusão

O modelo neuro-simbólico da V1 foi desenhado para tornar a análise de áudio mais explicável e menos dependente de um único score neural.

A parte neural detecta padrões estatísticos de áudio fake. A parte simbólica verifica se essa suspeita faz sentido diante da estrutura temporal, da qualidade acústica e dos padrões fonético-fonológicos do próprio áudio.

O ponto mais importante da V1 é a postura crítica: a camada simbólica pode reforçar a suspeita, mas também pode contestá-la quando o áudio apresenta sinais compatíveis com fala natural, domínio fora do treino ou cobertura insuficiente.
