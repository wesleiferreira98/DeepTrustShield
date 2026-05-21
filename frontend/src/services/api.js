const IMAGE_API_BASE = '/v1/deepfake/image';
const VIDEO_API_BASE = '/v1/deepfake/video';
const AUDIO_API_BASE = '/v1/deepfake/audio';
const LIVENESS_API_BASE = '/v1/liveness';

export async function analyzeImage(file, targetModel = 'auto') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('target_model', targetModel);

  const response = await fetch(`${IMAGE_API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao enviar imagem');
  }

  return response.json();
}

export async function analyzeVideo(frames, targetModel = 'auto') {
  const formData = new FormData();
  frames.forEach((frame, index) => {
    formData.append('frames', frame, frame.name || `frame_${index}.jpg`);
  });
  formData.append('target_model', targetModel);

  const response = await fetch(`${VIDEO_API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao enviar frames de video');
  }

  return response.json();
}

export async function analyzeAudio(file, options = {}) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('use_temporal_memory', options.useTemporalMemory ? 'true' : 'false');
  formData.append('target_audio_model', options.targetAudioModel || 'default');

  const response = await fetch(`${AUDIO_API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao enviar audio');
  }

  return response.json();
}

export async function analyzeLiveness(frames, sessionDefinition = null, requiredChecks = [], clientContext = null) {
  const formData = new FormData();
  frames.forEach((frame, index) => {
    formData.append('frames', frame, frame.name || `frame_${index}.jpg`);
  });

  if (Array.isArray(requiredChecks) && requiredChecks.length > 0) {
    formData.append('required_checks', requiredChecks.join(','));
  }

  if (sessionDefinition) {
    formData.append('session_definition', JSON.stringify(sessionDefinition));
  }

  if (clientContext) {
    formData.append('client_context', JSON.stringify(clientContext));
  }

  const response = await fetch(`${LIVENESS_API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao analisar prova de vida');
  }

  return response.json();
}

export async function createLivenessSession(detectorMode = 'mediapipe') {
  const formData = new FormData();
  formData.append('detector_mode', detectorMode);

  const response = await fetch(`${LIVENESS_API_BASE}/session`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao criar sessao de prova de vida');
  }

  return response.json();
}

export async function pollStatus(jobId, mode = 'image') {
  const apiBase = mode === 'video' ? VIDEO_API_BASE : mode === 'audio' ? AUDIO_API_BASE : IMAGE_API_BASE;
  const response = await fetch(`${apiBase}/status/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao obter status');
  }

  return response.json();
}

export async function getResult(jobId, mode = 'image') {
  const apiBase = mode === 'video' ? VIDEO_API_BASE : mode === 'audio' ? AUDIO_API_BASE : IMAGE_API_BASE;
  const response = await fetch(`${apiBase}/result/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Erro ao obter resultado');
  }

  return response.json();
}

export function getArtifactUrl(jobId, name) {
  return `${IMAGE_API_BASE}/artifacts/${jobId}/${name}`;
}
