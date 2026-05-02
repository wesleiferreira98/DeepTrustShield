import { useMemo, useState } from 'react';
import { Shield, ShieldAlert, ShieldCheck, ShieldQuestion, AlertTriangle, RefreshCw } from 'lucide-react';
import LivenessAnalysisCard from './LivenessAnalysisCard';
import './Result.css';

const MODEL_OPTIONS = [
  { value: 'auto', label: 'Automatico (melhor disponivel)' },
  { value: 'efficientnet_b0_v5', label: 'EfficientNet-B0 v5' },
  { value: 'efficientnet_b0_v4', label: 'EfficientNet-B0 v4' },
  { value: 'efficientnet_b0', label: 'EfficientNet-B0 (generico)' },
  { value: 'efficientnet_b3', label: 'EfficientNet-B3' },
  { value: 'efficientnet_b4', label: 'EfficientNet-B4' },
  { value: 'mobilenet_v3_small_v1', label: 'MobileNetV3 Small v1' },
  { value: 'mobilenet', label: 'MobileNet (auto)' },
  { value: 'alexnet_v82', label: 'AlexNet v8.2' },
  { value: 'alexnet_v81', label: 'AlexNet v8.1' },
  { value: 'alexnet_v8', label: 'AlexNet v8' },
  { value: 'alexnet_v7', label: 'AlexNet v7' },
  { value: 'alexnet_v6', label: 'AlexNet v6' },
  { value: 'alexnet', label: 'AlexNet (auto)' },
];

function Result({
  result,
  previewUrl,
  livenessAnalysis = null,
  analysisMode = 'image',
  onReset,
  onReanalyze,
  comparisonResults = [],
  reanalysisLoading = false,
  reanalysisError = null,
}) {
  const [selectedModel, setSelectedModel] = useState('auto');

  const getRiskConfig = (level, pFake) => {
    const percent = pFake !== null ? pFake * 100 : null;

    switch (level) {
      case 'low':
        return {
          icon: ShieldCheck,
          color: 'success',
          label: 'Provavelmente Real',
          description: analysisMode === 'audio' ? 'O audio parece autentico' : 'A imagem parece ser autentica',
        };
      case 'medium':
        if (percent !== null && percent >= 70) {
          return {
            icon: Shield,
            color: 'warning',
            label: 'Incerto',
            description: 'Nao temos certeza. Se tivessemos que classificar, seria fake.',
          };
        }
        return {
          icon: ShieldCheck,
          color: 'success-light',
          label: 'Provavelmente Real',
          description: analysisMode === 'audio'
            ? 'O audio aparenta ser autentico, mas com ressalvas'
            : 'A imagem aparenta ser autentica, mas com ressalvas',
        };
      case 'high':
        return {
          icon: ShieldAlert,
          color: 'danger',
          label: analysisMode === 'audio' ? 'Audio Fake Detectado' : 'Deepfake Detectado',
          description: analysisMode === 'audio'
            ? 'Alta probabilidade de audio sintetico ou clonado'
            : 'Alta probabilidade de manipulacao detectada',
        };
      default:
        return {
          icon: ShieldQuestion,
          color: 'info',
          label: 'Indeterminado',
          description: 'Nao foi possivel concluir a analise',
        };
    }
  };

  const config = getRiskConfig(result.risk_level, result.p_fake);
  const RiskIcon = config.icon;

  const activeFlags = Object.entries(result.flags || {})
    .filter(([_, value]) => value)
    .map(([key]) => key);

  const audioSymbolic = result.audio_summary?.symbolic || null;
  const audioSegmentSummary = audioSymbolic?.segment_summary || result.audio_summary?.segment_summary || null;
  const audioTranscript = result.audio_summary?.transcript || null;
  const activeAudioFlags = Object.entries(audioSymbolic?.flags || {})
    .filter(([_, value]) => value)
    .map(([key]) => key);

  const flagLabels = {
    no_face: 'Nenhuma face detectada',
    multiple_faces: 'Multiplas faces detectadas',
    blur_high: 'Imagem borrada',
    low_light: 'Iluminacao inadequada',
    face_too_small: 'Face muito pequena',
  };

  const audioFlagLabels = {
    short_audio: 'Audio curto',
    very_short_audio: 'Audio muito curto',
    high_silence: 'Muito silencio',
    low_volume: 'Volume baixo',
    clipped_audio: 'Audio saturado',
    near_threshold: 'Score perto do limite',
    strong_fake_score: 'Score fake forte',
    strong_real_score: 'Score real forte',
    low_confidence_fake: 'Fake com baixa margem',
    feature_extraction_failed: 'Features incompletas',
    partial_audio_analysis: 'Analise parcial',
    segment_score_disagreement: 'Segmentos divergentes',
    isolated_suspicious_segment: 'Suspeita localizada',
    consistent_suspicious_segments: 'Suspeita consistente',
    regional_bias_review: 'Revisao regional',
    domain_mismatch_warning: 'Dominio fora do treino',
    phonetic_layer_available: 'Base fonetica carregada',
    phonetic_coverage_insufficient: 'Cobertura fonetica baixa',
    phonetic_consistency_support: 'Consistencia fonetica',
    phonetic_disruption_alert: 'Ruptura fonetica',
    terminal_suspicious_segments: 'Pico no final',
    possible_end_artifact: 'Possivel artefato final',
  };

  const confidenceLabels = {
    low: 'Baixa',
    moderate: 'Moderada',
    high: 'Alta',
  };

  const sortedFrameResults = [...(result.frame_results || [])]
    .sort((a, b) => a.index - b.index)
    .slice(0, 5);

  const sortedComparison = useMemo(() => (
    [...comparisonResults].sort((a, b) => {
      const aScore = typeof a.p_fake === 'number' ? a.p_fake : Number.POSITIVE_INFINITY;
      const bScore = typeof b.p_fake === 'number' ? b.p_fake : Number.POSITIVE_INFINITY;
      if (aScore !== bScore) {
        return aScore - bScore;
      }
      return (a.created_at > b.created_at ? 1 : -1);
    })
  ), [comparisonResults]);

  const scoreExtremes = useMemo(() => {
    const scored = sortedComparison.filter((item) => typeof item.p_fake === 'number');
    if (scored.length < 2) {
      return { bestModel: null, worstModel: null };
    }

    let best = scored[0];
    let worst = scored[0];
    for (const item of scored) {
      if (item.p_fake < best.p_fake) {
        best = item;
      }
      if (item.p_fake > worst.p_fake) {
        worst = item;
      }
    }

    if (best.model_version === worst.model_version) {
      return { bestModel: null, worstModel: null };
    }

    return { bestModel: best.model_version, worstModel: worst.model_version };
  }, [sortedComparison]);

  return (
    <div className="result-container">
      <div className="result-preview">
        <div className="preview-wrapper">
          {analysisMode === 'audio' && previewUrl ? (
            <audio src={previewUrl} controls className="audio-preview" />
          ) : result.evidence?.face_crop_url ? (
            <img
              src={result.evidence.face_crop_url}
              alt="Face analisada"
              className="preview-image"
            />
          ) : previewUrl ? (
            <img src={previewUrl} alt="Imagem original" className="preview-image" />
          ) : null}
        </div>
        <p className="preview-label">
          {analysisMode === 'video' ? 'Frame de referencia' : analysisMode === 'audio' ? 'Audio analisado' : 'Face analisada'}
        </p>
      </div>

      <div className="result-panel">
        <div className={`risk-badge ${config.color}`}>
          <RiskIcon size={32} />
          <div className="risk-info">
            <span className="risk-label">{config.label}</span>
            <span className="risk-description">{config.description}</span>
          </div>
        </div>

        {result.p_fake !== null && (
          <div className="score-section">
            <div className="score-header">
              <span>Probabilidade de Fake</span>
              <span className="score-value">{(result.p_fake * 100).toFixed(1)}%</span>
            </div>
            <div className="score-bar">
              <div
                className={`score-fill ${config.color}`}
                style={{ width: `${result.p_fake * 100}%` }}
              />
              <div className="score-markers">
                <span style={{ left: '50%' }}>50%</span>
                <span style={{ left: '80%' }}>80%</span>
              </div>
            </div>
            <div className="score-labels">
              <span>Autentico</span>
              <span>Suspeito</span>
              <span>Fake</span>
            </div>
          </div>
        )}

        {activeFlags.length > 0 && (
          <div className="flags-section">
            <h3>
              <AlertTriangle size={16} />
              Alertas
            </h3>
            <div className="flags-list">
              {activeFlags.map((flag) => (
                <span key={flag} className="flag-tag">
                  {flagLabels[flag] || flag}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="meta-section">
          <div className="meta-item">
            <span className="meta-label">Modelo</span>
            <span className="meta-value">{result.model_version}</span>
          </div>
          {result.processing_time_ms && (
            <div className="meta-item">
              <span className="meta-label">Tempo</span>
              <span className="meta-value">{(result.processing_time_ms / 1000).toFixed(2)}s</span>
            </div>
          )}
          {result.video_summary && (
            <>
              <div className="meta-item">
                <span className="meta-label">Frames validos</span>
                <span className="meta-value">
                  {result.video_summary.valid_frames}/{result.video_summary.total_frames}
                </span>
              </div>
              {result.video_summary.p_fake_mean !== null && (
                <div className="meta-item">
                  <span className="meta-label">Media (frames)</span>
                  <span className="meta-value">{(result.video_summary.p_fake_mean * 100).toFixed(1)}%</span>
                </div>
              )}
              {result.video_summary.p_fake_median !== null && (
                <div className="meta-item">
                  <span className="meta-label">Mediana (frames)</span>
                  <span className="meta-value">{(result.video_summary.p_fake_median * 100).toFixed(1)}%</span>
                </div>
              )}
              {result.video_summary.p_fake_max !== null && (
                <div className="meta-item">
                  <span className="meta-label">Pior frame (max)</span>
                  <span className="meta-value">{(result.video_summary.p_fake_max * 100).toFixed(1)}%</span>
                </div>
              )}
            </>
          )}
          {result.audio_summary && (
            <>
              <div className="meta-item">
                <span className="meta-label">Limite calibrado</span>
                <span className="meta-value">{(result.audio_summary.threshold * 100).toFixed(1)}%</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">Decisao audio</span>
                <span className="meta-value">{result.audio_summary.decision}</span>
              </div>
              {audioSymbolic?.confidence && (
                <div className="meta-item">
                  <span className="meta-label">Confianca simbolica</span>
                  <span className="meta-value">{confidenceLabels[audioSymbolic.confidence] || audioSymbolic.confidence}</span>
                </div>
              )}
              {typeof audioSymbolic?.final_score === 'number' && (
                <div className="meta-item">
                  <span className="meta-label">Score final</span>
                  <span className="meta-value">{(audioSymbolic.final_score * 100).toFixed(1)}%</span>
                </div>
              )}
              {audioSegmentSummary?.count > 0 && (
                <div className="meta-item">
                  <span className="meta-label">Janelas analisadas</span>
                  <span className="meta-value">{audioSegmentSummary.count}</span>
                </div>
              )}
              {typeof audioSegmentSummary?.fake_ratio === 'number' && (
                <div className="meta-item">
                  <span className="meta-label">Janelas suspeitas</span>
                  <span className="meta-value">{(audioSegmentSummary.fake_ratio * 100).toFixed(1)}%</span>
                </div>
              )}
              {result.audio_summary.audio_info?.duration_seconds && (
                <div className="meta-item">
                  <span className="meta-label">Duracao</span>
                  <span className="meta-value">{result.audio_summary.audio_info.duration_seconds.toFixed(2)}s</span>
                </div>
              )}
            </>
          )}
        </div>

        {analysisMode === 'audio' && result.audio_summary?.executive_summary && (
          <div className="audio-executive-summary">
            <h3>{result.audio_summary.executive_summary.title}</h3>
            <p>{result.audio_summary.executive_summary.summary}</p>
          </div>
        )}

        {analysisMode === 'audio' && audioTranscript?.text && (
          <div className="audio-transcript-section">
            <div className="audio-transcript-header">
              <h3>Transcricao automatica</h3>
              <span>
                {audioTranscript.word_count || 0} palavras
                {audioTranscript.asr?.model ? ` · ${audioTranscript.asr.model}` : ''}
              </span>
            </div>
            <p>{audioTranscript.text}</p>
          </div>
        )}

        {analysisMode === 'audio' && audioSymbolic && (
          <div className="audio-symbolic-section">
            <h3>Revisao simbolica</h3>
            {activeAudioFlags.length > 0 && (
              <div className="flags-list audio-flags-list">
                {activeAudioFlags.map((flag) => (
                  <span key={flag} className="flag-tag">
                    {audioFlagLabels[flag] || flag}
                  </span>
                ))}
              </div>
            )}
            {Array.isArray(audioSymbolic.explanations) && audioSymbolic.explanations.length > 0 && (
              <ul className="audio-explanations">
                {audioSymbolic.explanations.map((explanation) => (
                  <li key={explanation}>{explanation}</li>
                ))}
              </ul>
            )}
            {audioSymbolic.regional_bias_review?.review_needed && (
              <div className="regional-review-box">
                <strong>Hipotese alternativa</strong>
                <span>{audioSymbolic.regional_bias_review.note}</span>
              </div>
            )}
            {audioSymbolic.phonetic_review && (
              <div className="regional-review-box">
                <strong>Camada fonetico-fonologica</strong>
                <span>
                  {audioSymbolic.phonetic_review.status === 'insufficient_coverage'
                    ? 'Base simbolica carregada, mas ainda sem eventos foneticos suficientes para calibracao linguistica.'
                    : `Status: ${audioSymbolic.phonetic_review.status || 'indisponivel'}`}
                </span>
              </div>
            )}
            {audioSymbolic.segment_diagnostics?.summary?.terminal_suspicious_count > 0 && (
              <div className="regional-review-box">
                <strong>Diagnostico dos picos</strong>
                <span>
                  {audioSymbolic.segment_diagnostics.summary.terminal_suspicious_count} trecho(s) acima do limite aparecem no final do audio.
                  {audioSymbolic.segment_diagnostics.summary.possible_end_artifact
                    ? ' Ha sinais de possivel artefato acustico no encerramento.'
                    : ' Nao ha artefato final evidente pelas metricas simples.'}
                </span>
              </div>
            )}
            {audioSegmentSummary?.suspicious_segments?.length > 0 && (
              <div className="audio-segments-list">
                {audioSegmentSummary.suspicious_segments.slice(0, 4).map((segment) => (
                  <div key={`${segment.index}-${segment.start}`} className="audio-segment-item">
                    <span>{segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s</span>
                    <strong>{(segment.p_fake * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {analysisMode === 'audio' && result.audio_summary && (
          <details className="audio-debug-section">
            <summary>Dados simbolicos retornados</summary>
            <pre>{JSON.stringify(result.audio_summary, null, 2)}</pre>
          </details>
        )}

        {analysisMode === 'video' && livenessAnalysis && (
          <LivenessAnalysisCard livenessAnalysis={livenessAnalysis} />
        )}

        {analysisMode !== 'audio' && (
        <div className="reanalyze-section">
          <h3>Reanalisar com outro modelo</h3>
          <div className="reanalyze-controls">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={reanalysisLoading}
            >
              {MODEL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => onReanalyze(selectedModel)}
              disabled={reanalysisLoading}
            >
              {reanalysisLoading ? 'Reanalisando...' : 'Reanalisar'}
            </button>
          </div>
          {reanalysisError && <p className="reanalyze-error">{reanalysisError}</p>}
        </div>
        )}

        {sortedComparison.length > 0 && (
          <div className="comparison-section">
            <h3>Comparativo por modelo</h3>
            <div className="comparison-table">
              <div className="comparison-row comparison-head">
                <span>#</span>
                <span>Modelo</span>
                <span>Score</span>
                <span>Risco</span>
                <span>Tempo</span>
              </div>
              {sortedComparison.map((item, index) => (
                <div
                  key={item.model_version}
                  className={`comparison-row ${
                    item.model_version === scoreExtremes.bestModel
                      ? 'comparison-best'
                      : item.model_version === scoreExtremes.worstModel
                        ? 'comparison-worst'
                        : ''
                  }`}
                >
                  <span>{index + 1}</span>
                  <span>{item.model_version}</span>
                  <span className="comparison-score-cell">
                    {item.p_fake !== null ? `${(item.p_fake * 100).toFixed(1)}%` : '--'}
                    {item.model_version === scoreExtremes.bestModel && (
                      <span className="score-badge best">Melhor</span>
                    )}
                    {item.model_version === scoreExtremes.worstModel && (
                      <span className="score-badge worst">Pior</span>
                    )}
                  </span>
                  <span>{item.risk_level}</span>
                  <span>{item.processing_time_ms ? `${(item.processing_time_ms / 1000).toFixed(2)}s` : '--'}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {analysisMode === 'video' && sortedFrameResults.length > 0 && (
          <div className="frames-section">
            <h3>Frames analisados</h3>
            <div className="frames-list">
              {sortedFrameResults.map((frame) => (
                <div key={frame.index} className="frame-item">
                  <span>Frame {frame.index + 1}</span>
                  <strong>{(frame.p_fake * 100).toFixed(1)}%</strong>
                </div>
              ))}
            </div>
          </div>
        )}

        <button onClick={onReset} className="btn-reset">
          <RefreshCw size={18} />
          Nova analise
        </button>
      </div>
    </div>
  );
}

export default Result;
