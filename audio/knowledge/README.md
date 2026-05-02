# Base de conhecimento simbolica de audio

Esta pasta guarda regras linguisticas interpretaveis para a camada
neuro-simbolica de audio.

## V1

O arquivo `phonetic_rules_v1.yaml` descreve uma primeira base de regras
fonetico-fonologicas para o portugues brasileiro. O foco nao e classificar
sotaque nem inferir regiao do falante. O foco e medir consistencia interna:

```text
O padrao observado e estavel, plausivel e coerente dentro do proprio audio?
```

As regras cobrem inicialmente:

- roticos em coda, especialmente /r/ final em verbos;
- /s/ em coda;
- /t,d/ diante de [i];
- regras globais de cobertura, consistencia, ruptura localizada e alinhamento
  com picos do detector neural.

## Estado atual

A base ja e carregada pelo worker e aparece no relatorio simbolico do audio.
Como ainda nao ha ASR/alinhamento fonetico no pipeline, a camada retorna
`insufficient_coverage` ate existirem eventos foneticos confiaveis.

Esse comportamento e proposital: a camada simbolica nao deve inventar
evidencia linguistica quando nao ha eventos suficientes.

## Arquivos auxiliares para testes

Enquanto o ASR/alinhamento nao esta integrado, o worker procura arquivos ao
lado do audio com estes nomes:

```text
audio.phonetic.json
audio.events.json
audio.alignment.json
audio.transcript.json
audio.json
audio.txt
```

Um arquivo com eventos explicitos pode usar:

```json
{
  "phonetic_events": [
    {
      "phenomenon": "rotic_coda",
      "word": "trabalhar",
      "start_time": 1.2,
      "end_time": 1.6,
      "target_segment": "r",
      "variant": "deletion",
      "position": "external_coda",
      "context": "verb_infinitive_external_coda",
      "confidence": 0.9
    }
  ]
}
```

Um arquivo apenas com texto tambem e aceito, mas gera somente candidatos de
baixa confianca. Esses candidatos ajudam a medir cobertura futura, mas ainda
nao devem calibrar a decisao sem classificacao fonetica real.

Quando o audio e enviado pela interface web, ele e copiado para a pasta de
artefatos com nome generico (`audio.ogg`, por exemplo). Para esse caso, o
worker tambem consegue procurar sidecars por `source_audio_sha256`. Por padrao
ele busca em:

```text
audio/data/eval_external
```

Esse comportamento pode ser alterado com:

```text
AUDIO_TRANSCRIPT_SIDECAR_DIRS=/caminho/a:/outro/caminho
```
