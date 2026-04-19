import './Result.css';

const LIVENESS_REASON_LABELS = {
  model_spoof_score_high: 'Modelo ONNX detectou padrao de foto/tela/replay',
  planar_motion_without_internal_face_change: 'Movimento plano sem mudanca interna do rosto',
  screen_rectangle_detected: 'Retangulo de tela detectado ao redor da face',
  borderline_spoof_score_without_liveness_confirmation: 'Score ONNX em zona cinza sem confirmacao forte de vida',
  borderline_spoof_score_with_high_frame: 'Score ONNX em zona cinza com frame individual suspeito',
  fusion_low_internal_motion_high_face_background_delta: 'Sinais combinados de tela: pouco movimento interno e face muito diferente do fundo',
  low_face_texture: 'Textura facial baixa',
  few_valid_face_frames: 'Poucos frames validos com face',
  no_face_crops: 'Nenhum recorte facial valido para o modelo ONNX',
  insufficient_antispoof_quality: 'Qualidade insuficiente para anti-spoof: aproxime o rosto e mantenha a camera estavel',
  mobile_antispoof_uncertain: 'Anti-spoof inconclusivo no celular: refaca em luz mais uniforme',
  model_file_missing: 'Modelo ONNX nao encontrado no backend',
  onnxruntime_missing: 'ONNX Runtime nao esta instalado no backend',
};

const formatPercent = (value) => (
  typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '--'
);

function LivenessAnalysisCard({ livenessAnalysis }) {
  if (!livenessAnalysis) {
    return null;
  }

  const livenessModel = livenessAnalysis?.spoof_analysis?.model || livenessAnalysis?.checks?.anti_spoof_model || null;
  const livenessHeuristic = livenessAnalysis?.spoof_analysis?.heuristic || livenessAnalysis?.checks?.anti_spoof || null;
  const livenessStatusLabel = livenessAnalysis?.passed ? 'Aprovada' : 'Reprovada';
  const livenessStatusClass = livenessAnalysis?.passed ? 'passed' : 'failed';
  const livenessReasons = [
    ...(livenessAnalysis?.spoof_analysis?.reasons || []),
    ...(livenessModel?.reasons || []),
  ].filter(Boolean);
  const uniqueLivenessReasons = [...new Set(livenessReasons)];
  const challengeEnforced = livenessAnalysis?.summary?.challenge_results_enforced;
  const clientContext = livenessAnalysis?.summary?.client_context || null;
  const heuristicSignals = livenessHeuristic?.signals || {};
  const deviceLabels = {
    desktop: 'PC',
    mobile: 'Celular',
    tablet: 'Tablet',
  };
  const captureLabels = {
    webcam: 'Webcam',
    phone_camera: 'Camera do celular',
  };

  return (
    <div className={`liveness-analysis-section ${livenessStatusClass}`}>
      <div className="liveness-analysis-header">
        <h3>Prova de vida e anti-spoof</h3>
        <span>{livenessStatusLabel}</span>
      </div>

      <div className="liveness-analysis-grid">
        {livenessModel && (
          <>
            <div className="liveness-analysis-item">
              <span className="meta-label">ONNX p-spoof</span>
              <strong>{formatPercent(livenessModel.p_spoof)}</strong>
            </div>
            <div className="liveness-analysis-item">
              <span className="meta-label">Limite</span>
              <strong>{formatPercent(livenessModel.threshold)}</strong>
            </div>
            <div className="liveness-analysis-item">
              <span className="meta-label">Agregacao</span>
              <strong>{livenessModel.aggregation || '--'}</strong>
            </div>
          </>
        )}
        {livenessHeuristic && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Heuristica</span>
            <strong>{livenessHeuristic.passed ? 'OK' : 'Alerta'}</strong>
          </div>
        )}
        <div className="liveness-analysis-item">
          <span className="meta-label">Frames validos</span>
          <strong>
            {livenessAnalysis.summary?.valid_frames ?? '--'}/{livenessAnalysis.summary?.total_frames ?? '--'}
          </strong>
        </div>
        <div className="liveness-analysis-item">
          <span className="meta-label">Desafios no backend</span>
          <strong>{challengeEnforced ? 'Revalidados' : 'Diagnostico'}</strong>
        </div>
        {clientContext && (
          <>
            <div className="liveness-analysis-item">
              <span className="meta-label">Dispositivo</span>
              <strong>{deviceLabels[clientContext.device_class] || clientContext.device_class || '--'}</strong>
            </div>
            <div className="liveness-analysis-item">
              <span className="meta-label">Origem provavel</span>
              <strong>
                {captureLabels[clientContext.capture_source_guess] || clientContext.capture_source_guess || '--'}
              </strong>
            </div>
            {clientContext.camera?.width && clientContext.camera?.height && (
              <div className="liveness-analysis-item">
                <span className="meta-label">Stream</span>
                <strong>{clientContext.camera.width}x{clientContext.camera.height}</strong>
              </div>
            )}
          </>
        )}
        {typeof heuristicSignals.median_screen_rectangle_score === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Tela/retangulo</span>
            <strong>{formatPercent(heuristicSignals.median_screen_rectangle_score)}</strong>
          </div>
        )}
        {typeof heuristicSignals.internal_motion === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Mov. interno</span>
            <strong>{formatPercent(heuristicSignals.internal_motion)}</strong>
          </div>
        )}
        {typeof heuristicSignals.median_face_background_delta === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Face x fundo</span>
            <strong>{formatPercent(heuristicSignals.median_face_background_delta)}</strong>
          </div>
        )}
        {typeof heuristicSignals.median_moire_score === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Moire/FFT</span>
            <strong>{formatPercent(heuristicSignals.median_moire_score)}</strong>
          </div>
        )}
        {typeof heuristicSignals.median_frequency_peak_score === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Pico freq.</span>
            <strong>{heuristicSignals.median_frequency_peak_score.toFixed(2)}</strong>
          </div>
        )}
        {typeof heuristicSignals.median_high_frequency_energy === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Energia freq.</span>
            <strong>{heuristicSignals.median_high_frequency_energy.toFixed(2)}</strong>
          </div>
        )}
        {typeof heuristicSignals.median_skin_moire_score === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Moire pele</span>
            <strong>{formatPercent(heuristicSignals.median_skin_moire_score)}</strong>
          </div>
        )}
        {typeof heuristicSignals.median_skin_frequency_peak_score === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Pico pele</span>
            <strong>{heuristicSignals.median_skin_frequency_peak_score.toFixed(2)}</strong>
          </div>
        )}
        {typeof heuristicSignals.median_skin_patch_count === 'number' && (
          <div className="liveness-analysis-item">
            <span className="meta-label">Patches pele</span>
            <strong>{heuristicSignals.median_skin_patch_count.toFixed(0)}</strong>
          </div>
        )}
      </div>

      {Array.isArray(livenessModel?.frame_scores) && livenessModel.frame_scores.length > 0 && (
        <div className="liveness-frame-scores">
          {livenessModel.frame_scores.slice(0, 5).map((score, index) => (
            <div key={`onnx-frame-${index}`} className="frame-item">
              <span>ONNX frame {index + 1}</span>
              <strong>{formatPercent(score)}</strong>
            </div>
          ))}
        </div>
      )}

      {uniqueLivenessReasons.length > 0 && (
        <p className="liveness-analysis-note">
          {uniqueLivenessReasons.map((reason) => LIVENESS_REASON_LABELS[reason] || reason).join('; ')}
        </p>
      )}
    </div>
  );
}

export default LivenessAnalysisCard;
