import { useCallback, useState } from 'react';
import { Upload as UploadIcon, Image } from 'lucide-react';
import './Upload.css';

function Upload({ onUpload }) {
  const [isDragging, setIsDragging] = useState(false);
  const [targetModel, setTargetModel] = useState('auto');

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      onUpload(file, targetModel);
    }
  }, [onUpload, targetModel]);

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file, targetModel);
    }
  }, [onUpload, targetModel]);

  return (
    <div className="upload-container">
      <div 
        className={`upload-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="upload-icon">
          <UploadIcon size={48} />
        </div>
        <h2>Arraste uma selfie aqui</h2>
        <p>ou clique para selecionar</p>

        <div className="model-target">
          <label htmlFor="target-model">Modelo alvo</label>
          <select
            id="target-model"
            value={targetModel}
            onChange={(e) => setTargetModel(e.target.value)}
            onClick={(e) => e.stopPropagation()}
          >
            <option value="auto">Automático (melhor disponível)</option>
            <option value="efficientnet_b0">EfficientNet-B0</option>
            <option value="efficientnet_b3">EfficientNet-B3</option>
            <option value="efficientnet_b4">EfficientNet-B4</option>
            <option value="alexnet">AlexNet</option>
          </select>
        </div>
        
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileSelect}
          className="upload-input"
        />

        <div className="upload-formats">
          <Image size={16} />
          <span>JPG, PNG ou WebP</span>
        </div>
      </div>

      <div className="upload-info">
        <h3>Como funciona?</h3>
        <ol>
          <li>Envie uma selfie ou foto com rosto</li>
          <li>Nossa IA analisa a imagem em tempo real</li>
          <li>Receba o resultado com nível de risco</li>
        </ol>
      </div>
    </div>
  );
}

export default Upload;
