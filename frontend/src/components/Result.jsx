import { Shield, ShieldAlert, ShieldCheck, ShieldQuestion, AlertTriangle, RefreshCw } from 'lucide-react';
import './Result.css';

function Result({ result, previewUrl, status, onReset }) {
  const getRiskConfig = (level, pFake) => {
    // Usar porcentagem para mensagens mais precisas
    const percent = pFake !== null ? pFake * 100 : null;
    
    switch (level) {
      case 'low':
        return {
          icon: ShieldCheck,
          color: 'success',
          label: 'Provavelmente Real',
          description: 'A imagem parece ser autêntica'
        };
      case 'medium':
        // Mensagem diferente baseada na porcentagem
        if (percent !== null && percent >= 70) {
          return {
            icon: Shield,
            color: 'warning',
            label: 'Incerto',
            description: 'Não temos certeza. Se tivéssemos que classificar, seria fake.'
          };
        } else {
          return {
            icon: ShieldCheck,
            color: 'success-light',
            label: 'Provavelmente Real',
            description: 'A imagem aparenta ser autêntica, mas com ressalvas'
          };
        }
      case 'high':
        return {
          icon: ShieldAlert,
          color: 'danger',
          label: 'Deepfake Detectado',
          description: 'Alta probabilidade de manipulação detectada'
        };
      default:
        return {
          icon: ShieldQuestion,
          color: 'info',
          label: 'Indeterminado',
          description: 'Não foi possível concluir a análise'
        };
    }
  };

  const config = getRiskConfig(result.risk_level, result.p_fake);
  const RiskIcon = config.icon;
  
  const activeFlags = Object.entries(result.flags || {})
    .filter(([_, value]) => value)
    .map(([key]) => key);

  const flagLabels = {
    no_face: 'Nenhuma face detectada',
    multiple_faces: 'Múltiplas faces detectadas',
    blur_high: 'Imagem borrada',
    low_light: 'Iluminação inadequada',
    face_too_small: 'Face muito pequena'
  };

  return (
    <div className="result-container">
      <div className="result-preview">
        <div className="preview-wrapper">
          {result.evidence?.face_crop_url ? (
            <img 
              src={result.evidence.face_crop_url} 
              alt="Face analisada" 
              className="preview-image"
            />
          ) : previewUrl ? (
            <img src={previewUrl} alt="Imagem original" className="preview-image" />
          ) : null}
        </div>
        <p className="preview-label">Face analisada</p>
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
              <span>Autêntico</span>
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
        </div>

        <button onClick={onReset} className="btn-reset">
          <RefreshCw size={18} />
          Nova análise
        </button>
      </div>
    </div>
  );
}

export default Result;
