Agora sim ficou bem mais saudável.

O mais importante: o modelo novo carregou mesmo. Dá para ver pelo `threshold = 0.9744173288345337`, que é o threshold calibrado do modelo codec-equalizado. Antes o threshold era `0.2501`, então esse resultado já confirma que a web/worker está usando o modelo novo.

Sobre o resultado: ele classificou como `real`, mas ainda com scores altos por janela. Isso é interessante:

```text
threshold calibrado: 0.9744
maior janela:       0.9727
média das janelas:  0.8724
decisão final:      real
```

Ou seja, ele ficou “quase lá” em uma janela, mas nenhuma passou do limite calibrado. Isso parece exatamente o comportamento que queríamos depois da calibração: o modelo ainda percebe características suspeitas no OGG, provavelmente compressão/mobile/resampling, mas não transforma isso automaticamente em fake.

A camada simbólica também se comportou melhor que antes: não marcou `consistent_suspicious_segments`, não marcou `regional_bias_review`, e manteve `risk_level: low` com `confidence: moderate`. Eu só ajustaria uma coisa depois: quando os scores ficam altos mas abaixo do threshold, como aqui, talvez valha criar um alerta simbólico do tipo `high_but_below_threshold` ou `borderline_compression_sensitivity`, porque esse caso merece revisão leve mesmo sendo `real`.

Minha leitura: avanço real. O modelo novo não zerou a suspeita, mas parou de condenar o áudio por causa do domínio OGG/compressão. Isso é exatamente o tipo de melhoria que a calibração e o treino com base BR deveriam trazer.

```
{
  "threshold": 0.9744173288345337,
  "decision": "real",
  "audio_info": {
    "source_sample_rate": 48000,
    "target_sample_rate": 16000,
    "channels": 1,
    "original_samples": 983896,
    "duration_seconds": 20.497833333333332,
    "clip_seconds": 4,
    "n_mels": 80
  },
  "segment_summary": {
    "count": 10,
    "window_seconds": 4,
    "stride_seconds": 2,
    "coverage_fraction": 1,
    "max_segments": 32
  },
  "segments": [
    {
      "index": 0,
      "start": 0,
      "end": 4,
      "duration_seconds": 4,
      "p_fake": 0.9727113246917725
    },
    {
      "index": 1,
      "start": 2,
      "end": 6,
      "duration_seconds": 4,
      "p_fake": 0.9258960485458374
    },
    {
      "index": 2,
      "start": 4,
      "end": 8,
      "duration_seconds": 4,
      "p_fake": 0.9123868346214294
    },
    {
      "index": 3,
      "start": 6,
      "end": 10,
      "duration_seconds": 4,
      "p_fake": 0.8006048798561096
    },
    {
      "index": 4,
      "start": 8,
      "end": 12,
      "duration_seconds": 4,
      "p_fake": 0.8217347264289856
    },
    {
      "index": 5,
      "start": 10,
      "end": 14,
      "duration_seconds": 4,
      "p_fake": 0.9291577339172363
    },
    {
      "index": 6,
      "start": 12,
      "end": 16,
      "duration_seconds": 4,
      "p_fake": 0.797521710395813
    },
    {
      "index": 7,
      "start": 14,
      "end": 18,
      "duration_seconds": 4,
      "p_fake": 0.862112820148468
    },
    {
      "index": 8,
      "start": 16,
      "end": 20,
      "duration_seconds": 4,
      "p_fake": 0.8717238903045654
    },
    {
      "index": 9,
      "start": 16.497875,
      "end": 20.497875,
      "duration_seconds": 4,
      "p_fake": 0.8308390378952026
    }
  ],
  "symbolic": {
    "decision": "real",
    "risk_level": "low",
    "confidence": "moderate",
    "neural_score": 0.8225064277648926,
    "symbolic_score": 0.5,
    "final_score": 0.7257544994354248,
    "decision_score": 0.872468900680542,
    "threshold": 0.9744173288345337,
    "flags": {
      "short_audio": false,
      "very_short_audio": false,
      "high_silence": false,
      "low_volume": false,
      "clipped_audio": false,
      "near_threshold": false,
      "strong_fake_score": false,
      "strong_real_score": false,
      "low_confidence_fake": false,
      "feature_extraction_failed": false,
      "partial_audio_analysis": false,
      "segment_score_disagreement": false,
      "isolated_suspicious_segment": false,
      "consistent_suspicious_segments": false,
      "regional_bias_review": false,
      "domain_mismatch_warning": false
    },
    "explanations": [
      "Nenhum alerta simbolico relevante foi encontrado na analise inicial."
    ],
    "adjustments": [],
    "segment_summary": {
      "count": 10,
      "fake_count": 0,
      "fake_ratio": 0,
      "min_score": 0.797521710395813,
      "max_score": 0.9727113246917725,
      "mean_score": 0.872468900680542,
      "score_range": 0.17518961429595947,
      "suspicious_segments": []
    },
    "regional_bias_review": {
      "review_needed": false,
      "hypothesis": "not_triggered",
      "evidence": [
        "compressed_or_mobile_format",
        "resampled_from_external_sample_rate",
        "long_form_audio",
        "natural_pause_or_recording_silence"
      ],
      "note": "Sem evidencia simbolica suficiente para levantar hipotese regional/domino."
    },
    "features": {
      "path": "audio.ogg",
      "available": true,
      "duration_seconds": 20.497833333333332,
      "clip_seconds": 4,
      "analyzed_fraction": 1,
      "source_sample_rate": 48000,
      "target_sample_rate": 16000,
      "channels": 1,
      "rms_mean": 0.09049420803785324,
      "rms_std": 0.09049413353204727,
      "silence_ratio": 0.43494144082069397,
      "clipping_ratio": 0.0000884238033904694
    }
  }
}
```


1. **Melhorar a interpretação simbólica**

   Esse teste mostrou um caso importante: `real`, mas com scores altos perto do threshold. Eu criaria flags novas como:

   - `high_but_below_threshold`
   - `near_calibrated_threshold`
   - `compression_sensitive_audio`
   - `review_recommended`

   Assim a saída poderia dizer: “classificado como real, mas com sinais altos; revisar por possível compressão/domínio fora do treino”.
2. **Validar com mais áudios reais externos**

   Agora precisamos testar contra áudios reais que não vieram da base: WhatsApp/Telegram, OGG, MP3, gravações de celular, sotaques diferentes, ruído, pausas, ambiente. O objetivo é medir falsos positivos no mundo real, porque esse é o risco mais perigoso para o produto.
3. **Criar uma base de avaliação separada**

   Não treinar nela. Só avaliar. Uma pastinha tipo:

   ```text
   audio/data/eval_external/
   ├── real/
   ├── fake/
   └── manifest.csv
   ```

   Essa base vira nosso “termômetro honesto”. Toda vez que treinarmos outro modelo, comparamos nela.

Depois disso, eu faria:

4. **Adicionar relatório automático por áudio**

   Em vez de só JSON, gerar um resumo amigável:

   > Decisão: real.
   > Nenhuma janela ultrapassou o threshold calibrado.
   > O áudio tem compressão/resampling e scores elevados, então recomenda-se revisão leve.
   >
5. **Testar robustez contra codec**

   Pegar os mesmos áudios e converter para `.wav`, `.mp3`, `.ogg`, talvez com bitrates diferentes. Se a decisão muda demais, o modelo ainda está aprendendo codec.
6. **Treino mais criterioso**

   Para a próxima versão:

   - manter split por locutor;
   - balancear codec entre real/fake;
   - balancear duração;
   - balancear sotaque;
   - guardar uma base externa intocada;
   - usar label shuffle como teste obrigatório;
   - comparar por sotaque, gênero, codec e duração.

O próximo passo prático que eu faria agora é implementar as novas flags simbólicas para esse caso “real, mas com score alto”. Esse resultado que você trouxe é perfeito para calibrar essa camada.
