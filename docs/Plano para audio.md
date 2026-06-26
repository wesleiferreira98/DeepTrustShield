Dá sim para fazer um MVP, e acho que esse é exatamente o caminho certo.

Uma base pequena, com menos de 1 GB, provavelmente não sustenta um detector neural “forte” generalista sozinha. Mas ela pode sustentar muito bem um **protótipo acadêmico bem formulado**, especialmente se o objetivo inicial for provar a arquitetura:

```text
detector neural simples + regras simbólicas + saída calibrada/explicável
```

Eu pensaria no MVP assim:

```text
1. Usar a base PT-BR pequena como corpus inicial.
2. Treinar ou ajustar um baseline simples real/manipulado.
3. Segmentar os áudios em janelas.
4. Extrair features acústicas básicas:
   - MFCC
   - pitch
   - energia
   - duração de pausas
   - variação de ritmo
5. Criar regras simbólicas fracas:
   - mudança abrupta entre segmentos;
   - trecho suspeito isolado;
   - inconsistência prosódica;
   - baixa confiança do modelo neural;
   - áudio fora da distribuição do treino.
6. Combinar tudo em uma decisão final:
   - autêntico;
   - manipulado;
   - inconclusivo;
   - baixa confiança.
```

Com base pequena, eu evitaria prometer:

```text
“detecção robusta de deepfakes em português brasileiro”
```

E venderia como:

```text
“MVP de arquitetura híbrida/neuro-simbólica para apoio à detecção de deepfake de áudio em português brasileiro.”
```

Isso é honesto e forte.

As regras simbólicas ajudam justamente porque você não depende só de volume de dados. Elas podem funcionar como uma camada de segurança em cima de um modelo ainda limitado. Por exemplo:

```text
Se o modelo neural marcar manipulado,
mas não houver ruptura acústica interna
e o áudio estiver fora da distribuição conhecida,
então reduzir confiança.
```

Ou:

```text
Se o modelo neural marcar manipulado
e o trecho suspeito coincidir com grande variação de pitch, energia e timbre,
então reforçar suspeita.
```

Para um MVP, eu faria uma versão enxuta:

```text
Baseline neural:
CNN simples com espectrograma ou embeddings pré-treinados.

Módulo simbólico:
regras sobre distância entre segmentos, energia, pitch e MFCC.

Saída:
score neural + score simbólico + explicação curta.
```

O mais importante é estruturar a avaliação com humildade:

```text
- dataset pequeno;
- experimento inicial;
- foco em viabilidade;
- comparação baseline vs baseline + regras;
- análise qualitativa dos casos.
```

Mesmo que a acurácia não seja espetacular, se você mostrar que o módulo ajuda a identificar incerteza e reduz decisões precipitadas, já tem uma contribuição interessante.

Então sim: com essa base pequena + regras simbólicas, dá para fazer um MVP perfeitamente defensável. O segredo é não vender como produto final, mas como **prova de conceito robusta e extensível**.
