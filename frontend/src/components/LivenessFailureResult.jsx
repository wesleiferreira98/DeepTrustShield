import { RefreshCw, ShieldAlert } from 'lucide-react';
import LivenessAnalysisCard from './LivenessAnalysisCard';
import './Result.css';

function LivenessFailureResult({ previewUrl, livenessAnalysis, message, onReset }) {
  const reasons = [
    ...(livenessAnalysis?.spoof_analysis?.reasons || []),
    ...(livenessAnalysis?.spoof_analysis?.model?.reasons || []),
  ];
  const spoofReasons = new Set([
    'model_spoof_score_high',
    'borderline_spoof_score_with_high_frame',
    'screen_rectangle_detected',
    'planar_motion_without_internal_face_change',
  ]);
  const hasSpoofEvidence = reasons.some((reason) => spoofReasons.has(reason));
  const qualityLimited = reasons.includes('insufficient_antispoof_quality')
    || reasons.includes('mobile_antispoof_uncertain')
    || (
      livenessAnalysis?.summary?.antispoof_quality?.limited
      && !livenessAnalysis?.summary?.antispoof_quality?.model_has_hard_spoof_signal
    );
  const title = qualityLimited ? 'Captura insuficiente' : 'Prova de vida reprovada';
  const description = qualityLimited
    ? 'Aproxime o rosto e mantenha a camera estavel'
    : 'A captura parece foto, tela ou replay';

  return (
    <div className="result-container">
      <div className="result-preview">
        <div className="preview-wrapper">
          {previewUrl ? (
            <img src={previewUrl} alt="Captura reprovada" className="preview-image" />
          ) : null}
        </div>
        <p className="preview-label">Captura reprovada</p>
      </div>

      <div className="result-panel">
        <div className="risk-badge danger">
            <ShieldAlert size={32} />
            <div className="risk-info">
            <span className="risk-label">{title}</span>
            <span className="risk-description">{description}</span>
            </div>
          </div>

        {message && (
          <div className="liveness-failure-message">
            {message}
          </div>
        )}

        <LivenessAnalysisCard livenessAnalysis={livenessAnalysis} />

        <button type="button" className="btn-reset" onClick={onReset}>
          <RefreshCw size={18} />
          Nova tentativa
        </button>
      </div>
    </div>
  );
}

export default LivenessFailureResult;
