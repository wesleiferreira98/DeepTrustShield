# Processo de Recebimento de Áudio e Arquitetura Neural

## Objetivo

Este documento descreve como o sistema recebe um arquivo de áudio, como esse arquivo é transformado antes de chegar ao modelo neural e qual arquitetura está sendo usada na V1 do detector de deepfake de áudio.

O ponto principal é que o modelo não recebe diretamente um arquivo `.mp3`, `.ogg` ou `.wav` como entrada bruta. O arquivo é primeiro decodificado, padronizado e convertido em uma representação numérica chamada log-Mel spectrogram. Só depois disso ele é enviado para a rede neural.

## Recebimento do Áudio na Plataforma Web

No fluxo web, o usuário envia um arquivo de áudio pela interface. O backend aceita arquivos como:

- `.wav`;
- `.flac`;
- `.mp3`;
- `.m4a`;
- `.ogg`.

Depois do upload, o backend salva o arquivo na pasta de artefatos do job:

```text
artifacts/<job_id>/audio.ext
```

Esse arquivo salvo é então enviado para o worker, que executa a análise.

## O Modelo Recebe o Arquivo de Áudio Diretamente? Mudança do dia 18/05

Não. O modelo neural não recebe o arquivo de áudio diretamente.

O arquivo enviado pelo usuário é apenas o formato de transporte. Antes da inferência, o sistema executa as seguintes etapas:

1. Decodifica o arquivo com `torchaudio`.
2. Converte o áudio para mono, caso ele tenha mais de um canal.
3. Reamostra o áudio para `16 kHz`.
4. Seleciona uma janela de análise de `4 segundos`.
5. Converte a forma de onda para log-Mel spectrogram.
6. Normaliza os valores para a faixa usada no treino.
7. Envia o tensor resultante para a rede neural.

Portanto, a entrada real do modelo é um tensor derivado do áudio, não o arquivo original.

## Padronização da Forma de Onda

A leitura inicial é feita por:

```python
torchaudio.load(audio_path)
```

Essa função retorna:

- a forma de onda;
- a taxa de amostragem original.

Em seguida, o sistema aplica:

### Conversão para mono

Se o áudio tiver mais de um canal, os canais são somados por média:

```text
áudio estéreo -> áudio mono
```

### Reamostragem

Se o áudio não estiver em `16 kHz`, ele é reamostrado:

```text
taxa original -> 16 kHz
```

Essa etapa é importante porque o modelo foi treinado esperando sempre a mesma taxa de amostragem.

### Janela de 4 segundos

O modelo principal analisa uma janela de `4 segundos`.

Se o áudio for maior que isso, o sistema usa um recorte central para a predição global. Se o áudio for menor, ele é preenchido com zeros até completar o tamanho esperado.

Em termos práticos:

```text
16.000 amostras por segundo × 4 segundos = 64.000 amostras
```

Assim, a entrada temporal fica padronizada.

## Conversão Para Log-Mel Spectrogram

Depois da padronização da forma de onda, o sistema calcula um espectrograma Mel:

```python
T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
)
```

Os principais parâmetros são:

- `sample_rate = 16000`;
- `n_fft = 1024`;
- `hop_length = 256`;
- `win_length = 1024`;
- `n_mels = 80`.

Depois disso, o espectrograma é convertido para escala logarítmica em decibéis:

```python
T.AmplitudeToDB()
```

Os valores são limitados entre `-80 dB` e `0 dB` e normalizados para a faixa entre `0` e `1`.

O formato final esperado pela rede é aproximadamente:

```text
[batch, 1, 80, tempo]
```

Para uma janela de 4 segundos com os parâmetros atuais, o tensor costuma ter formato semelhante a:

```text
[1, 1, 80, 251]
```

## Inferência Por Segmentos

Além da predição global, o sistema também executa inferência por janelas temporais.

O áudio completo é dividido em janelas de `4 segundos`, com passo de `2 segundos`. Para evitar custo excessivo, o sistema limita o número máximo de segmentos a `32`.

Cada segmento recebe um score `p_fake`.

Exemplo de saída:

```json
{
  "index": 0,
  "start": 0,
  "end": 4,
  "duration_seconds": 4,
  "p_fake": 0.42
}
```

Esses scores por segmento são usados para:

- visualizar a linha do tempo da suspeita;
- detectar picos localizados;
- alimentar a camada simbólica;
- avaliar baixa cobertura temporal;
- apoiar a memória temporal.

## Arquitetura Neural Padrão: AudioCNN

A arquitetura padrão usada na V1 é uma CNN leve aplicada sobre log-Mel spectrograms.

Ela recebe um tensor com um canal e `80` bandas Mel. A estrutura é:

```text
Entrada: [batch, 1, 80, tempo]

Conv2d(1 -> 16, kernel 3, padding 1)
BatchNorm2d(16)
ReLU
MaxPool2d(2)

Conv2d(16 -> 32, kernel 3, padding 1)
BatchNorm2d(32)
ReLU
MaxPool2d(2)

Conv2d(32 -> 64, kernel 3, padding 1)
BatchNorm2d(64)
ReLU
AdaptiveAvgPool2d(1, 1)

Linear(64 -> 1)
Sigmoid
```

A saída antes do `sigmoid` é um logit. Depois do `sigmoid`, temos:

```text
p_fake
```

Esse valor representa a probabilidade estimada de o áudio ser falso, segundo o modelo selecionado.

## Arquitetura Experimental: AudioResNetSmall

Também existe uma arquitetura experimental chamada `resnet_small`.

Ela foi adicionada para comparar se uma rede residual pequena conseguiria melhorar a detecção por janelas ou a generalização. Sua estrutura é:

```text
Entrada: [batch, 1, 80, tempo]

Stem:
Conv2d(1 -> 32, kernel 5)
BatchNorm2d(32)
ReLU
MaxPool2d(2)

Blocos residuais:
ResidualBlock(32 -> 32)
ResidualBlock(32 -> 64, stride 2)
ResidualBlock(64 -> 64)
ResidualBlock(64 -> 128, stride 2)
ResidualBlock(128 -> 128)

AdaptiveAvgPool2d(1, 1)
Dropout
Linear(128 -> 1)
Sigmoid
```

Nos testes realizados, essa arquitetura não superou de forma consistente a CNN simples. Por isso, ela permanece como opção experimental, não como modelo padrão da V1.

## Modelos Disponíveis na Interface

Atualmente, a interface web permite selecionar mais de um modelo de áudio:

### FakeBrAccent codec equalizado

É o modelo padrão mais conservador. Ele foi treinado com a base FakeBrAccent ajustada por codec/equalização.

Uso recomendado:

- análise padrão;
- menor risco de falso positivo em alguns áudios reais testados;
- comparação com outros modelos.

### BRSpeech-DF amostra (experimental)

É um modelo treinado com uma amostra do BRSpeech-DF.

Ele apresentou bom desempenho dentro do próprio domínio da base, mas demonstrou sensibilidade a áudios reais fora desse domínio, como exemplos do CORAA e gravações mais espontâneas.

Uso recomendado:

- comparação experimental;
- avaliação de sensibilidade;
- testes de domínio.

Ele não deve ser interpretado como o modelo mais confiável para todos os tipos de áudio.

## Threshold Calibrado

Cada modelo possui um threshold calibrado, carregado a partir do arquivo `summary.json`.

Esse threshold define o limite acima do qual o score neural é considerado suspeito.

Exemplo:

```text
p_fake >= threshold -> suspeita neural
p_fake < threshold  -> sem suspeita neural pelo modelo
```

É importante notar que thresholds diferentes podem gerar decisões diferentes para o mesmo áudio. Isso reforça a necessidade da camada simbólica e dos avisos de domínio.

## Relação Com a Camada Neuro-Simbólica

A saída neural não é a decisão final isolada.

Depois da inferência, o sistema usa:

- score global;
- scores por segmento;
- threshold calibrado;
- transcrição;
- eventos fonético-fonológicos;
- diagnóstico acústico;
- memória temporal;
- score simbólico independente;
- alerta de domínio.

A camada simbólica pode confirmar, rebaixar ou contestar a suspeita neural.

Exemplo:

```text
score neural alto + baixa cobertura fonética + áudio curto = inconclusivo
```

ou:

```text
score neural alto + ruptura fonética independente = suspeita mais forte
```

## Limitações Atuais

As principais limitações observadas são:

- sensibilidade dos modelos neurais ao domínio de gravação;
- desempenho melhor em bases limpas e formais do que em fala cotidiana;
- dependência de transcrição para a camada fonético-fonológica;
- redução de confiabilidade em áudios curtos;
- risco de falso positivo quando o áudio está fora do domínio de treino.

## Conclusão

O modelo neural de áudio não recebe arquivos `.mp3`, `.ogg` ou `.wav` diretamente. Ele recebe uma representação log-Mel padronizada, calculada a partir do áudio decodificado, convertido para mono, reamostrado para `16 kHz` e recortado em janelas de `4 segundos`.

A arquitetura padrão da V1 é uma CNN leve sobre log-Mel spectrograms. A arquitetura residual pequena existe como alternativa experimental, mas ainda não substitui a CNN padrão.

Na V1, a decisão mais confiável surge da combinação entre modelo neural, análise por segmentos, camada simbólica independente e alerta de domínio.
