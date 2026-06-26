import { Loader2, Check, Circle, AlertCircle } from 'lucide-react';
import './AnalysisPanel.css';

function AnalysisPanel({ status, previewUrl, isUploading, analysisMode = 'image' }) {
  const isAudio = analysisMode === 'audio';
  const modeLabel = analysisMode === 'video' ? 'video curto' : isAudio ? 'audio' : 'imagem';
  const defaultSteps = isAudio
    ? [
      { id: 'validating_image', label: 'Validando audio', state: 'pending' },
      { id: 'preprocessing', label: 'Pre-processando audio', state: 'pending' },
      { id: 'model_inference', label: 'Analisando voz com IA', state: 'pending' },
      { id: 'calibrating_score', label: 'Aplicando limite calibrado', state: 'pending' },
      { id: 'generating_evidence', label: 'Gerando diagnostico', state: 'pending' },
    ]
    : [
    { id: 'validating_image', label: 'Validando imagem', state: 'pending' },
    { id: 'detecting_face', label: 'Detectando face', state: 'pending' },
    { id: 'quality_checks', label: 'Verificando qualidade', state: 'pending' },
    { id: 'preprocessing', label: 'Pré-processando', state: 'pending' },
    { id: 'model_inference', label: 'Analisando com IA', state: 'pending' },
    { id: 'calibrating_score', label: 'Calibrando resultado', state: 'pending' },
    { id: 'generating_evidence', label: 'Gerando evidências', state: 'pending' },
  ];
  const steps = status?.steps || defaultSteps;

  const getStepIcon = (state) => {
    switch (state) {
      case 'completed':
        return <Check size={16} className="step-icon completed" />;
      case 'running':
        return <Loader2 size={16} className="step-icon running" />;
      case 'error':
        return <AlertCircle size={16} className="step-icon error" />;
      default:
        return <Circle size={16} className="step-icon pending" />;
    }
  };

  return (
    <div className="analysis-container">
      <div className="analysis-preview">
        {isAudio && previewUrl ? (
          <audio src={previewUrl} controls className="audio-preview" />
        ) : previewUrl && (
          <div className="preview-wrapper">
            <img src={previewUrl} alt="Preview" className="preview-image" />
            {status?.preview?.bbox && (
              <div 
                className="bbox-overlay"
                style={{
                  left: `${status.preview.bbox.x}%`,
                  top: `${status.preview.bbox.y}%`,
                  width: `${status.preview.bbox.width}%`,
                  height: `${status.preview.bbox.height}%`,
                }}
              />
            )}
          </div>
        )}
      </div>

      <div className="analysis-panel">
        <div className="panel-header">
          <h2>Analisando {modeLabel}...</h2>
          <p className="status-message">
            {isUploading
              ? analysisMode === 'video'
                ? 'Enviando frames do video...'
                : isAudio
                  ? 'Enviando audio...'
                : 'Enviando arquivo...'
              : status?.message || 'Processando...'}
          </p>
        </div>

        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${status?.progress || 0}%` }}
          />
        </div>

        <div className="steps-list">
          {steps.map((step) => (
            <div key={step.id} className={`step-item ${step.state}`}>
              {getStepIcon(step.state)}
              <span className="step-label">{step.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AnalysisPanel;
