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

const AUDIO_MODEL_OPTIONS = [
  { value: 'auto', label: 'Automatico (padrao atual)' },
  { value: 'default', label: 'FakeBrAccent codec equalizado' },
  { value: 'audio_v1_1_domain_adv', label: 'Áudio V1.1 domínio adversarial' },
  { value: 'brspeech_sample', label: 'BRSpeech-DF amostra (experimental)' },
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
  const [rulesExpanded, setRulesExpanded] = useState(false);
  const reanalysisModelOptions = analysisMode === 'audio' ? AUDIO_MODEL_OPTIONS : MODEL_OPTIONS;

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
  const audioIndependentSymbolic = audioSymbolic?.independent_symbolic || result.audio_summary?.independent_symbolic || null;
  const audioSegmentSummary = audioSymbolic?.segment_summary || result.audio_summary?.segment_summary || null;
  const audioTranscript = result.audio_summary?.transcript || null;
  const audioSegments = Array.isArray(result.audio_summary?.segments) ? result.audio_summary.segments : [];
  const audioPhoneticEvents = Array.isArray(result.audio_summary?.phonetic_events)
    ? result.audio_summary.phonetic_events
    : [];
  const audioGroundTruth = result.audio_summary?.ground_truth || null;
  const audioTemporalMemory = audioSymbolic?.temporal_memory || result.audio_summary?.temporal_memory || null;
  const audioTimeline = useMemo(() => {
    if (!audioSegments.length) {
      return null;
    }

    const duration = result.audio_summary?.audio_info?.duration_seconds
      || Math.max(...audioSegments.map((segment) => Number(segment.end) || 0));
    const threshold = typeof result.audio_summary?.threshold === 'number'
      ? result.audio_summary.threshold
      : 0.5;
    const width = 720;
    const height = 220;
    const padding = { top: 18, right: 22, bottom: 58, left: 38 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const xForTime = (time) => padding.left + ((Math.max(0, Math.min(duration, time)) / Math.max(duration, 0.001)) * plotWidth);
    const yForScore = (score) => padding.top + ((1 - Math.max(0, Math.min(1, score))) * plotHeight);
    const points = audioSegments
      .map((segment) => {
        const mid = ((Number(segment.start) || 0) + (Number(segment.end) || 0)) / 2;
        return `${xForTime(mid).toFixed(2)},${yForScore(Number(segment.p_fake) || 0).toFixed(2)}`;
      })
      .join(' ');
    const suspiciousSegments = audioSegments.filter((segment) => (Number(segment.p_fake) || 0) >= threshold);
    const groundTruthSegments = audioGroundTruth?.available && Array.isArray(audioGroundTruth.manipulated_segments)
      ? audioGroundTruth.manipulated_segments
      : [];
    const memorySegments = audioTemporalMemory?.available && Array.isArray(audioTemporalMemory.rupture_candidates)
      ? audioTemporalMemory.rupture_candidates
      : [];
    const reliableEvents = audioPhoneticEvents.filter((event) => (
      event.event_quality === 'strong' || event.event_quality === 'moderate'
    ));
    const eventCounts = reliableEvents.reduce((acc, event) => {
      const key = event.phenomenon || 'outro';
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});

    return {
      width,
      height,
      padding,
      plotWidth,
      plotHeight,
      duration,
      threshold,
      points,
      xForTime,
      yForScore,
      suspiciousSegments,
      groundTruthSegments,
      memorySegments,
      reliableEvents,
      eventCounts,
      maxScore: Math.max(...audioSegments.map((segment) => Number(segment.p_fake) || 0)),
      minScore: Math.min(...audioSegments.map((segment) => Number(segment.p_fake) || 0)),
    };
  }, [audioSegments, audioPhoneticEvents, audioGroundTruth, audioTemporalMemory, result.audio_summary]);
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
    localized_manipulation_sensitivity: 'Sensibilidade localizada',
    temporal_memory_available: 'Memoria temporal',
    temporal_memory_rupture_candidate: 'Variacao temporal',
    symbolic_contestation: 'Contestacao simbolica',
    neural_suspicion_contested: 'Suspeita neural contestada',
    low_temporal_coverage: 'Cobertura temporal baixa',
    domain_shift_warning: 'Alerta de dominio',
  };

  const confidenceLabels = {
    low: 'Baixa',
    moderate: 'Moderada',
    high: 'Alta',
  };

  const eventLabels = {
    rotic_coda: '/r/ final',
    s_coda: '/s/ coda',
    td_before_i: '/t,d/ + i',
  };
  const ruleLabels = {
    R1: 'Apagamento de /r/ final em infinitivos',
    R2: 'Alternancia de /r/ sem ruptura clara',
    R3: 'Mudanca abrupta de /r/ no mesmo contexto',
    R4: 'Apagamento isolado de /r/ interno',
    R5: 'Variante dominante estavel',
    S1: '/s/ palatal recorrente e sistematico',
    S2: '/s/ alveolar recorrente e sistematico',
    S3: 'Troca abrupta entre /s/ alveolar e palatal',
    S4: 'Palatalizacao em contexto favorecedor',
    S5: 'Baixa palatalizacao diante de pausa',
    TD1: 'Palatalizacao de /t,d/ diante de i',
    TD2: 'Realizacao dentoalveolar estavel',
    TD3: 'Alternancia abrupta de /t,d/ diante de i',
    TD4: 'Palatalizacao apos contexto palatal',
    G1: 'Multiplos fenomenos estaveis',
    G2: 'Multiplos fenomenos mudaram no mesmo intervalo',
    G3: 'Cobertura linguistica insuficiente',
    G4: 'Qualidade de audio ou alinhamento baixa',
    G5: 'Score neural alto com consistencia linguistica',
    G6: 'Ruptura linguistica alinhada ao pico neural',
  };
  const audioVisualSummary = useMemo(() => {
    const phoneticReview = audioSymbolic?.phonetic_review || null;
    const eventSummary = result.audio_summary?.phonetic_event_extraction?.event_summary || null;
    const threshold = typeof result.audio_summary?.threshold === 'number'
      ? result.audio_summary.threshold
      : null;

    const metrics = phoneticReview ? [
      ...(audioIndependentSymbolic ? [{
        label: 'Simbolico independente',
        value: Number(audioIndependentSymbolic.score) || 0,
        inverted: true,
      }] : []),
      {
        label: 'Cobertura fonetica',
        value: Number(phoneticReview.coverage_score) || 0,
      },
      {
        label: 'Qualidade da evidencia',
        value: Number(phoneticReview.evidence_quality?.quality_score) || 0,
      },
      {
        label: 'Consistencia',
        value: Number(phoneticReview.phonetic_consistency_score) || 0,
      },
      {
        label: 'Ruptura localizada',
        value: Number(phoneticReview.localized_disruption_score) || 0,
        inverted: true,
      },
      {
        label: 'Alinhamento com pico',
        value: Number(phoneticReview.alignment_score) || 0,
      },
    ] : [];

    const scoreRows = [
      { label: 'Neural', value: Number(audioSymbolic?.neural_score), type: 'neural' },
      { label: 'Simbolico', value: Number(audioSymbolic?.symbolic_score), type: 'symbolic' },
      { label: 'Final', value: Number(audioSymbolic?.final_score), type: 'final' },
    ].filter((item) => Number.isFinite(item.value));

    const phenomenonCounts = eventSummary?.candidate_by_phenomenon || {};
    const classifiedCounts = eventSummary?.classified_by_variant || {};
    const maxPhenomenonCount = Math.max(1, ...Object.values(phenomenonCounts).map((value) => Number(value) || 0));
    const maxClassifiedCount = Math.max(1, ...Object.values(classifiedCounts).map((value) => Number(value) || 0));
    const rules = Array.isArray(phoneticReview?.activated_rules) ? phoneticReview.activated_rules : [];
    const validation = result.audio_summary?.ground_truth?.available
      ? {
          manipulationType: result.audio_summary.ground_truth.manipulation_type,
          manipulationMethod: result.audio_summary.ground_truth.manipulation_method,
          overlap: result.audio_summary.ground_truth.overlap || {},
          segments: result.audio_summary.ground_truth.manipulated_segments || [],
        }
      : null;
    const temporalMemory = audioSymbolic?.temporal_memory || result.audio_summary?.temporal_memory || null;

    return {
      phoneticReview,
      eventSummary,
      threshold,
      metrics,
      scoreRows,
      phenomenonCounts,
      classifiedCounts,
      maxPhenomenonCount,
      maxClassifiedCount,
      rules,
      validation,
      temporalMemory,
    };
  }, [audioIndependentSymbolic, audioSymbolic, result.audio_summary]);

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

        {analysisMode === 'audio' && audioTimeline && (
          <div className="audio-timeline-section">
            <div className="audio-timeline-header">
              <div>
                <h3>Linha do tempo da analise</h3>
                <span>
                  {audioTimeline.suspiciousSegments.length} janela(s) acima do limite · {audioTimeline.reliableEvents.length} evento(s) fonetico(s) confiavel(is)
                </span>
              </div>
              <div className="audio-timeline-score-range">
                {(audioTimeline.minScore * 100).toFixed(1)}% - {(audioTimeline.maxScore * 100).toFixed(1)}%
              </div>
            </div>

            <svg
              className="audio-timeline-chart"
              viewBox={`0 0 ${audioTimeline.width} ${audioTimeline.height}`}
              role="img"
              aria-label="Grafico dos scores por janela de audio"
            >
              <line
                x1={audioTimeline.padding.left}
                y1={audioTimeline.yForScore(1)}
                x2={audioTimeline.padding.left + audioTimeline.plotWidth}
                y2={audioTimeline.yForScore(1)}
                className="audio-chart-grid"
              />
              <line
                x1={audioTimeline.padding.left}
                y1={audioTimeline.yForScore(0.5)}
                x2={audioTimeline.padding.left + audioTimeline.plotWidth}
                y2={audioTimeline.yForScore(0.5)}
                className="audio-chart-grid"
              />
              <line
                x1={audioTimeline.padding.left}
                y1={audioTimeline.yForScore(0)}
                x2={audioTimeline.padding.left + audioTimeline.plotWidth}
                y2={audioTimeline.yForScore(0)}
                className="audio-chart-axis"
              />
              <line
                x1={audioTimeline.padding.left}
                y1={audioTimeline.yForScore(audioTimeline.threshold)}
                x2={audioTimeline.padding.left + audioTimeline.plotWidth}
                y2={audioTimeline.yForScore(audioTimeline.threshold)}
                className="audio-chart-threshold"
              />
              <text
                x={audioTimeline.padding.left + audioTimeline.plotWidth - 4}
                y={audioTimeline.yForScore(audioTimeline.threshold) - 6}
                className="audio-chart-threshold-label"
                textAnchor="end"
              >
                limite {(audioTimeline.threshold * 100).toFixed(1)}%
              </text>

              {audioTimeline.groundTruthSegments.map((segment, index) => {
                const x = audioTimeline.xForTime(Number(segment.start) || 0);
                const endX = audioTimeline.xForTime(Number(segment.end) || 0);
                return (
                  <rect
                    key={`gt-${index}-${segment.start}`}
                    x={x}
                    y={audioTimeline.padding.top}
                    width={Math.max(2, endX - x)}
                    height={audioTimeline.plotHeight}
                    className="audio-chart-ground-truth-band"
                  >
                    <title>
                      {`Trecho manipulado: ${(Number(segment.start) || 0).toFixed(1)}s - ${(Number(segment.end) || 0).toFixed(1)}s`}
                    </title>
                  </rect>
                );
              })}

              {audioTimeline.suspiciousSegments.map((segment) => {
                const x = audioTimeline.xForTime(Number(segment.start) || 0);
                const endX = audioTimeline.xForTime(Number(segment.end) || 0);
                return (
                  <rect
                    key={`sus-${segment.index}-${segment.start}`}
                    x={x}
                    y={audioTimeline.padding.top}
                    width={Math.max(2, endX - x)}
                    height={audioTimeline.plotHeight}
                    className="audio-chart-suspicious-band"
                  >
                    <title>
                      {`${(Number(segment.start) || 0).toFixed(1)}s - ${(Number(segment.end) || 0).toFixed(1)}s · ${((Number(segment.p_fake) || 0) * 100).toFixed(1)}%`}
                    </title>
                  </rect>
                );
              })}

              {audioTimeline.memorySegments.map((segment) => {
                const x = audioTimeline.xForTime(Number(segment.start) || 0);
                const endX = audioTimeline.xForTime(Number(segment.end) || 0);
                return (
                  <rect
                    key={`mem-${segment.index}-${segment.start}`}
                    x={x}
                    y={audioTimeline.padding.top}
                    width={Math.max(2, endX - x)}
                    height={audioTimeline.plotHeight}
                    className="audio-chart-memory-band"
                  >
                    <title>
                      {`Memoria temporal: ${(Number(segment.start) || 0).toFixed(1)}s - ${(Number(segment.end) || 0).toFixed(1)}s · ${(Number(segment.memory_score || segment.p_fake || 0) * 100).toFixed(1)}%`}
                    </title>
                  </rect>
                );
              })}

              <polyline className="audio-chart-line" points={audioTimeline.points} />

              {audioSegments.map((segment) => {
                const mid = ((Number(segment.start) || 0) + (Number(segment.end) || 0)) / 2;
                const score = Number(segment.p_fake) || 0;
                return (
                  <circle
                    key={`pt-${segment.index}-${segment.start}`}
                    cx={audioTimeline.xForTime(mid)}
                    cy={audioTimeline.yForScore(score)}
                    r={score >= audioTimeline.threshold ? 4.5 : 3}
                    className={score >= audioTimeline.threshold ? 'audio-chart-point suspicious' : 'audio-chart-point'}
                  >
                    <title>
                      {`Janela ${segment.index}: ${(Number(segment.start) || 0).toFixed(1)}s - ${(Number(segment.end) || 0).toFixed(1)}s · ${(score * 100).toFixed(1)}%`}
                    </title>
                  </circle>
                );
              })}

              {audioTimeline.reliableEvents.slice(0, 48).map((event, index) => {
                const startTime = Number(event.start_time) || 0;
                const phenomenon = event.phenomenon || 'other';
                const x = audioTimeline.xForTime(startTime);
                const y = audioTimeline.padding.top + audioTimeline.plotHeight + 18 + ((index % 3) * 10);
                return (
                  <g key={`${event.event_id}-${index}`}>
                    <line
                      x1={x}
                      y1={audioTimeline.padding.top + audioTimeline.plotHeight + 4}
                      x2={x}
                      y2={y}
                      className={`audio-event-marker ${phenomenon}`}
                    />
                    <circle
                      cx={x}
                      cy={y}
                      r={3.5}
                      className={`audio-event-dot ${phenomenon}`}
                    >
                      <title>
                        {`${eventLabels[phenomenon] || phenomenon}: ${event.word || ''} · ${event.variant || 'unknown'} · ${((Number(event.confidence) || 0) * 100).toFixed(1)}%`}
                      </title>
                    </circle>
                  </g>
                );
              })}

              <text x={audioTimeline.padding.left} y={audioTimeline.height - 8} className="audio-chart-time-label">0s</text>
              <text
                x={audioTimeline.padding.left + audioTimeline.plotWidth}
                y={audioTimeline.height - 8}
                className="audio-chart-time-label"
                textAnchor="end"
              >
                {audioTimeline.duration.toFixed(1)}s
              </text>
            </svg>

            <div className="audio-timeline-legend">
              <span><i className="legend-line" />Score neural</span>
              <span><i className="legend-band" />Acima do limite</span>
              {audioTimeline.memorySegments.length > 0 && (
                <span><i className="legend-memory" />Variacao temporal</span>
              )}
              {audioTimeline.groundTruthSegments.length > 0 && (
                <span><i className="legend-ground-truth" />Trecho manipulado</span>
              )}
              <span><i className="legend-dot rotic_coda" />/r/ final {audioTimeline.eventCounts.rotic_coda || 0}</span>
              <span><i className="legend-dot s_coda" />/s/ coda {audioTimeline.eventCounts.s_coda || 0}</span>
              <span><i className="legend-dot td_before_i" />/t,d/ + i {audioTimeline.eventCounts.td_before_i || 0}</span>
            </div>
          </div>
        )}

        {analysisMode === 'audio' && result.audio_summary && (
          <div className="audio-visual-dashboard">
            {audioVisualSummary.metrics.length > 0 && (
              <div className="audio-metric-grid">
                {audioVisualSummary.metrics.map((metric) => (
                  <div key={metric.label} className="audio-metric-card">
                    <div className="audio-metric-topline">
                      <span>{metric.label}</span>
                      <strong>{(metric.value * 100).toFixed(0)}%</strong>
                    </div>
                    <div className="audio-mini-bar">
                      <div
                        className={metric.inverted ? 'audio-mini-fill inverted' : 'audio-mini-fill'}
                        style={{ width: `${Math.max(0, Math.min(1, metric.value)) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {audioSegments.length > 0 && (
              <div className="audio-heatmap-box">
                <div className="audio-visual-title-row">
                  <h3>Mapa de janelas</h3>
                  <span>{audioSegments.length} janelas</span>
                </div>
                <div className="audio-window-heatmap" aria-label="Mapa dos scores por janela">
                  {audioSegments.map((segment) => {
                    const score = Number(segment.p_fake) || 0;
                    const threshold = audioVisualSummary.threshold ?? 0.5;
                    const hasGroundTruthOverlap = audioTimeline?.groundTruthSegments?.some((truth) => (
                      Math.max(Number(segment.start) || 0, Number(truth.start) || 0)
                      < Math.min(Number(segment.end) || 0, Number(truth.end) || 0)
                    ));
                    const heatClass = score >= threshold
                      ? 'critical'
                      : score >= Math.max(0, threshold - 0.08)
                        ? 'near'
                        : score >= 0.70
                          ? 'warm'
                          : 'cool';
                    return (
                      <span
                        key={`heat-${segment.index}-${segment.start}`}
                        className={`audio-window-cell ${heatClass}${hasGroundTruthOverlap ? ' ground-truth' : ''}`}
                        title={`Janela ${segment.index}: ${(Number(segment.start) || 0).toFixed(1)}s - ${(Number(segment.end) || 0).toFixed(1)}s · ${(score * 100).toFixed(1)}%`}
                      />
                    );
                  })}
                </div>
              </div>
            )}

            {audioVisualSummary.validation && (
              <div className="audio-validation-panel">
                <div className="audio-visual-title-row">
                  <h3>Validacao controlada</h3>
                  <span>{audioVisualSummary.validation.manipulationType || 'manipulacao'}</span>
                </div>
                <div className="audio-validation-grid">
                  <div>
                    <span>Coincidencia com pico</span>
                    <strong>
                      {audioVisualSummary.validation.overlap.localized_hit
                        ? `Detectada${audioVisualSummary.validation.overlap.localized_hit_source === 'memory' ? ' pela memoria' : ''}`
                        : 'Nao clara'}
                    </strong>
                  </div>
                  <div>
                    <span>Cobertura do trecho</span>
                    <strong>
                      {((Number(audioVisualSummary.validation.overlap.manipulated_overlap_ratio) || 0) * 100).toFixed(0)}%
                    </strong>
                  </div>
                  <div>
                    <span>Picos / memoria</span>
                    <strong>
                      {audioVisualSummary.validation.overlap.suspicious_count || 0}
                      {' / '}
                      {audioVisualSummary.validation.overlap.memory_candidate_count || 0}
                    </strong>
                  </div>
                </div>
              </div>
            )}

            {audioVisualSummary.temporalMemory?.available && (
              <div className="audio-memory-panel">
                <div className="audio-visual-title-row">
                  <h3>Memoria temporal</h3>
                  <span>{audioVisualSummary.temporalMemory.rupture_candidate_count || 0} candidato(s)</span>
                </div>
                <p>
                  {audioVisualSummary.temporalMemory.rupture_candidate_count > 0
                    ? 'Foram encontrados trechos que destoam do comportamento interno do proprio audio.'
                    : 'Nenhuma variacao temporal relevante foi encontrada.'}
                </p>
                {Array.isArray(audioVisualSummary.temporalMemory.rupture_candidates)
                  && audioVisualSummary.temporalMemory.rupture_candidates.length > 0 && (
                  <div className="audio-memory-list">
                    {audioVisualSummary.temporalMemory.rupture_candidates.slice(0, 4).map((candidate) => (
                      <div key={`memory-${candidate.index}-${candidate.start}`}>
                        <span>{(Number(candidate.start) || 0).toFixed(1)}s - {(Number(candidate.end) || 0).toFixed(1)}s</span>
                        <strong>{((Number(candidate.memory_score) || 0) * 100).toFixed(0)}%</strong>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {audioVisualSummary.scoreRows.length > 0 && (
              <div className="audio-score-ladder">
                <div className="audio-visual-title-row">
                  <h3>Neural → Simbolico → Final</h3>
                  {typeof audioVisualSummary.threshold === 'number' && (
                    <span>Limite {(audioVisualSummary.threshold * 100).toFixed(1)}%</span>
                  )}
                </div>
                {audioVisualSummary.scoreRows.map((row) => (
                  <div key={row.label} className="audio-score-row">
                    <span>{row.label}</span>
                    <div className="audio-score-track">
                      {typeof audioVisualSummary.threshold === 'number' && (
                        <i style={{ left: `${audioVisualSummary.threshold * 100}%` }} />
                      )}
                      <b
                        className={`audio-score-fill ${row.type}`}
                        style={{ width: `${Math.max(0, Math.min(1, row.value)) * 100}%` }}
                      />
                    </div>
                    <strong>{(row.value * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            )}

            {audioVisualSummary.eventSummary && (
              <div className="audio-events-visual">
                <div className="audio-visual-title-row">
                  <h3>Eventos foneticos</h3>
                  <span>
                    {audioVisualSummary.eventSummary.reliable_event_count || 0} confiaveis de {audioVisualSummary.eventSummary.candidate_count || 0}
                  </span>
                </div>
                <div className="audio-event-bars">
                  {Object.entries(audioVisualSummary.phenomenonCounts).map(([key, count]) => (
                    <div key={key} className="audio-event-bar-row">
                      <span>{eventLabels[key] || key}</span>
                      <div>
                        <b style={{ width: `${((Number(count) || 0) / audioVisualSummary.maxPhenomenonCount) * 100}%` }} />
                      </div>
                      <strong>{count}</strong>
                    </div>
                  ))}
                </div>
                {Object.keys(audioVisualSummary.classifiedCounts).length > 0 && (
                  <div className="audio-variant-bars">
                    {Object.entries(audioVisualSummary.classifiedCounts).slice(0, 6).map(([key, count]) => (
                      <div key={key} className="audio-variant-row">
                        <span>{key.replace('.', ' · ')}</span>
                        <div>
                          <b style={{ width: `${((Number(count) || 0) / audioVisualSummary.maxClassifiedCount) * 100}%` }} />
                        </div>
                        <strong>{count}</strong>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {audioVisualSummary.rules.length > 0 && (
              <div className="audio-rules-panel">
                <div className="audio-visual-title-row">
                  <h3>Regras ativadas</h3>
                  <button
                    type="button"
                    className="audio-inline-toggle"
                    onClick={() => setRulesExpanded((current) => !current)}
                  >
                    {rulesExpanded ? 'Recolher' : `Expandir ${audioVisualSummary.rules.length} regra(s)`}
                  </button>
                </div>
                <div className="audio-rule-list">
                  {audioVisualSummary.rules.map((rule) => {
                    const weight = Number(rule.weight) || 0;
                    const polarity = weight < 0 ? 'reduce' : weight > 0 ? 'increase' : 'neutral';
                    const friendlyLabel = ruleLabels[rule.rule_id] || rule.condition;
                    return (
                      <div
                        key={`${rule.rule_id}-${rule.condition}`}
                        className={`audio-rule-card ${polarity}${rulesExpanded ? ' expanded' : ''}`}
                      >
                        <div>
                          <strong><small>{rule.rule_id}</small>{friendlyLabel}</strong>
                          <span>{rule.explanation || rule.condition}</span>
                        </div>
                        <b>{weight > 0 ? '+' : ''}{weight.toFixed(2)}</b>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
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
                    <div>
                      <span>{segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s</span>
                      <small>Possível ruptura localizada no áudio</small>
                    </div>
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

        <div className="reanalyze-section">
          <h3>Reanalisar com outro modelo</h3>
          <div className="reanalyze-controls">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={reanalysisLoading}
            >
              {reanalysisModelOptions.map((option) => (
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
