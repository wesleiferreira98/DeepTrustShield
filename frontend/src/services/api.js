const API_BASE = '/v1/deepfake/image';

export async function analyzeImage(file, targetModel = 'auto') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('target_model', targetModel);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao enviar imagem');
  }

  return response.json();
}

export async function pollStatus(jobId) {
  const response = await fetch(`${API_BASE}/status/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao obter status');
  }

  return response.json();
}

export async function getResult(jobId) {
  const response = await fetch(`${API_BASE}/result/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao obter resultado');
  }

  return response.json();
}

export function getArtifactUrl(jobId, name) {
  return `${API_BASE}/artifacts/${jobId}/${name}`;
}
