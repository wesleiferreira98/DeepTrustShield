# Relatório de Achados da V1: Domínio, BRSpeech-DF, CORAA e Camada Neuro-Simbólica

## Resumo

Durante os testes da V1 do detector de deepfake de áudio, observamos que o principal gargalo atual não é apenas a arquitetura do modelo neural, mas a sensibilidade ao domínio dos dados de treinamento.

O modelo treinado com uma amostra do BRSpeech-DF apresentou desempenho alto quando avaliado em áudios do próprio domínio da base. No entanto, ao ser testado com áudios reais do CORAA e com áudios reais enviados pela plataforma web, o modelo elevou a suspeita em vários casos que não eram falsos. Isso indica que o detector pode estar aprendendo características do domínio de gravação, e não apenas sinais de manipulação.

## Contexto das Bases de Dados

### FakeBrAccent

O FakeBrAccent foi usado como uma das bases principais nos experimentos iniciais. Ele é voltado para detecção de áudio falso em português brasileiro com atenção a sotaques e variações regionais.

No contexto da V1, essa base foi importante porque trouxe amostras mais alinhadas ao problema de sotaque brasileiro. O modelo treinado com esse domínio se mostrou mais conservador em alguns áudios reais testados manualmente, especialmente quando comparado ao modelo treinado com BRSpeech-DF.

Limitação principal:

- volume reduzido de dados em relação a bases maiores;
- risco de o modelo aprender características específicas da própria base;
- necessidade de maior variedade de canais, microfones, codecs e condições reais de gravação.

### BRSpeech-DF

O BRSpeech-DF é uma base de áudio real e sintético em português usada para detecção de deepfake de voz. Ela foi escolhida porque oferece maior volume de dados e permite treinar um modelo supervisionado com mais exemplos reais e falsos.

Nos testes da V1, usamos apenas uma amostra do BRSpeech-DF, não a base completa. Mesmo assim, o modelo treinado com essa amostra apresentou desempenho alto no teste interno derivado da própria base.

Limitação observada:

- os áudios escutados manualmente apresentam fala muito limpa;
- a fala parece formal em várias amostras;
- há pouca semelhança com áudio cotidiano de celular ou WhatsApp;
- o modelo pode aprender o domínio da base, e não apenas sinais de falsificação.

Por isso, o BRSpeech-DF passou a ser tratado como modelo experimental na interface.

### CORAA

O CORAA é uma base de português brasileiro voltada para reconhecimento automático de fala. Ela reúne fala espontânea e preparada de diferentes fontes, aproximando-se mais do português falado em situações reais do que bases excessivamente limpas ou formais.

Na V1, o CORAA foi usado como base real externa para testar sensibilidade de domínio. Como os áudios do CORAA não são, em princípio, deepfakes, eles ajudam a verificar se o modelo neural acusa falsos positivos quando encontra fala mais espontânea.

Limitação principal:

- não é uma base de deepfake;
- serve melhor como teste de robustez em áudio real;
- ainda não representa totalmente áudio de WhatsApp/celular;
- pode ter áudios curtos, o que reduz a força da camada simbólica.

Mesmo com essas limitações, o CORAA foi útil para mostrar que o modelo BRSpeech-DF pode elevar suspeita em fala real espontânea.

### Áudios próprios e manipulados controlados

Além das bases públicas, também foram usados áudios reais consentidos e manipulações controladas. Esses exemplos são importantes porque se aproximam mais do uso esperado da plataforma.

Eles permitem avaliar:

- áudio real de celular;
- compressão;
- ruído;
- fala espontânea;
- troca localizada de falante;
- cortes e inserções;
- comportamento da memória temporal;
- capacidade da camada simbólica de contestar ou confirmar o neural.

Esses dados ainda são poucos, mas são fundamentais para calibrar a V1 em direção ao uso real.

## Achados Principais

### 1. BRSpeech-DF funciona bem dentro do próprio domínio

O modelo treinado com a amostra do BRSpeech-DF apresentou bons resultados no conjunto de teste derivado da própria base.

Resultado observado:

```text
best_val_auc = 0.995614
test_auc = 0.9791
test_f1 = 0.9041
test_acc = 0.8727
test_f1_calibrado = 0.9252
test_acc_calibrado = 0.9030
```

Esses números indicam que o modelo consegue separar bem áudios reais e falsos quando os exemplos de teste estão próximos do domínio usado no treinamento.

Contudo, a escuta manual de algumas amostras mostrou que a base possui características bastante específicas:

- fala limpa;
- pouca ou nenhuma presença de ruído;
- português formal;
- leitura ou fala com estilo pouco espontâneo;
- gravações distantes do uso cotidiano em celular ou aplicativos de mensagem.

Portanto, o bom desempenho interno não garante robustez em áudios reais coletados em condições comuns.

### 2. BRSpeech-DF elevou suspeita em áudios reais do CORAA

Ao testar áudios reais do CORAA na plataforma web, o modelo BRSpeech-DF frequentemente elevou a suspeita neural.

O padrão observado foi:

- score neural alto;
- poucas janelas temporais analisáveis;
- baixa cobertura fonético-fonológica;
- ausência de confirmação independente pela camada simbólica;
- decisão final rebaixada para inconclusiva.

Esse comportamento sugere que o modelo pode estar reagindo à diferença de domínio entre BRSpeech-DF e CORAA. O CORAA contém fala mais espontânea e mais próxima de português falado em contexto real, enquanto o BRSpeech-DF testado parece estar mais próximo de fala limpa e formal.

### 3. O CORAA evidenciou sensibilidade a fala espontânea

Os testes com CORAA foram importantes porque trouxeram um tipo de fala mais natural, com maior proximidade de uso cotidiano.

Mesmo sendo áudio real, alguns exemplos foram interpretados pelo modelo neural como suspeitos. Isso reforça que o sistema ainda pode confundir:

- informalidade;
- variação natural de fala;
- pausas;
- redução articulatória;
- diferenças de gravação;
- domínio acústico diferente;
- áudio curto;
- ausência de padrões fonéticos suficientes.

Esse achado é crítico para a V1, pois o uso real esperado envolve áudios de celular, WhatsApp, ambientes variados e fala espontânea.

### 4. A camada simbólica passou a contestar falso positivo neural

Após os ajustes, a camada neuro-simbólica deixou de apenas explicar o score neural e passou a atuar como uma camada crítica.

Foram adicionados mecanismos para:

- separar o score neural do score simbólico independente;
- identificar baixa cobertura fonética;
- identificar baixa cobertura temporal;
- detectar possível diferença de domínio;
- rebaixar decisões quando a suspeita neural não é confirmada por evidência independente.

Agora, em cenários como os testes com CORAA, o sistema pode produzir uma saída mais adequada:

```text
Suspeita neural contestada pela revisão simbólica.
```

Ou:

```text
Suspeita neural com alerta de domínio.
```

Isso é importante porque reduz o risco de falso positivo convincente. O sistema deixa de tratar todo score neural alto como prova de manipulação.

### 5. Áudio curto reduz a utilidade da camada simbólica

A camada simbólica depende de material suficiente para extrair evidências.

Ela precisa de:

- transcrição;
- palavras suficientes;
- eventos fonético-fonológicos;
- repetição de contextos;
- variação temporal;
- múltiplas janelas;
- sinais acústicos comparáveis entre trechos.

Quando o áudio é curto ou possui poucas janelas, a camada simbólica não consegue sustentar uma conclusão forte. Nesses casos, a saída correta deve ser:

```text
Evidência simbólica insuficiente.
```

e não:

```text
Áudio fake.
```

Por isso, foi adicionada a flag:

```json
"low_temporal_coverage": true
```

Essa flag impede que o sistema trate poucas janelas suspeitas como localização confiável de manipulação.

## Interpretação Técnica

Os testes indicam que o detector neural de áudio ainda é fortemente afetado por domínio.

Na prática, um score alto pode significar:

- presença real de manipulação;
- áudio fora do domínio de treinamento;
- codec diferente;
- fala espontânea;
- gravação mais ruidosa;
- estilo linguístico diferente;
- baixa qualidade de alinhamento;
- curta duração.

Portanto, a decisão final não deve depender apenas do score neural. A camada simbólica independente e os avisos de domínio são necessários para reduzir falsos positivos.

## Decisão Atual Para a V1

Para a V1, a estratégia mais segura é:

- manter o FakeBrAccent como modelo padrão mais conservador;
- disponibilizar o BRSpeech-DF como modelo experimental;
- mostrar divergências entre modelos;
- rebaixar para inconclusivo quando houver alerta de domínio;
- evitar decisão forte em áudios curtos;
- usar a camada simbólica como contestação, não apenas como explicação.

## Limitações Atuais

As principais limitações observadas são:

- o BRSpeech-DF parece representar um domínio formal e limpo;
- o CORAA evidencia fala mais espontânea, mas ainda não resolve o domínio de celular/WhatsApp;
- a camada simbólica depende da qualidade da transcrição;
- áudios curtos reduzem a força da análise fonético-fonológica;
- o sistema ainda precisa de bases mais próximas do uso real.

## Próximos Passos Recomendados

1. Ampliar a avaliação com áudios reais do CORAA.
2. Coletar ou montar uma base realista com áudio de celular e fala espontânea.
3. Gerar manipulações controladas sobre áudios reais do domínio desejado.
4. Avaliar separadamente:
   - BRSpeech-DF;
   - FakeBrAccent;
   - CORAA;
   - áudios próprios reais;
   - manipulados controlados.
5. Criar métricas por domínio, não apenas métricas globais.
6. Manter alerta de domínio como parte central da decisão da V1.

## Conclusão

A V1 mostrou que o sistema já é capaz de detectar sinais neurais fortes e organizar uma explicação simbólica. No entanto, os testes também mostraram que modelos de áudio são altamente sensíveis ao domínio dos dados.

O BRSpeech-DF melhorou o desempenho dentro de um domínio limpo e formal, mas não resolveu a generalização para fala espontânea. O CORAA evidenciou essa limitação e mostrou que a camada simbólica é essencial para contestar suspeitas neurais frágeis.

O próximo avanço real deve vir menos de trocar arquitetura e mais de aproximar o treinamento do uso esperado: fala cotidiana, celular, ruído, compressão e variação regional.
