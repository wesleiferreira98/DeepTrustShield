const IMAGE_API_BASE    = '/v1/deepfake/image';
const VIDEO_API_BASE    = '/v1/deepfake/video';
const AUDIO_API_BASE    = '/v1/deepfake/audio';
const LIVENESS_API_BASE = '/v1/liveness';

function authHeader() {
  const token = localStorage.getItem('dts-token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

/**
 * Safely extract an error message from any HTTP response.
 * Works even when the server returns an HTML page (nginx error pages, 502, 413, etc.)
 * instead of JSON — which would cause response.json() to throw.
 */
async function parseErrorMessage(response) {
  if (response.status === 401) {
    localStorage.removeItem('dts-token')
    localStorage.removeItem('dts-expires-at')
    window.dispatchEvent(new CustomEvent('dts:logout', { detail: { expired: true } }))
    return 'Sessão expirada. Faça login novamente.'
  }
  const text = await response.text().catch(() => '')
  try {
    const json = JSON.parse(text)
    return json.detail || json.message || `Erro ${response.status}`
  } catch {
    switch (response.status) {
      case 413: return 'Arquivo muito grande. Reduza o tamanho e tente novamente (limite: 25 MB).'
      case 502:
      case 503: return 'Servidor temporariamente indisponível. Tente novamente em instantes.'
      case 504: return 'O servidor demorou demais para responder (timeout). Tente novamente.'
      case 404: return 'Rota não encontrada no servidor. Verifique se o backend está rodando.'
      case 429: return 'Muitas requisições. Aguarde alguns segundos e tente novamente.'
      default:  return `Erro ${response.status}${response.statusText ? ` — ${response.statusText}` : '. Resposta inesperada do servidor.'}`
    }
  }
}

export async function analyzeImage(file, targetModel = 'auto') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('target_model', targetModel);

  const response = await fetch(`${IMAGE_API_BASE}/analyze`, {
    method: 'POST',
    headers: { ...authHeader() },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
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
    headers: { ...authHeader() },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
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
    headers: { ...authHeader() },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
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
    headers: { ...authHeader() },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.json();
}

export async function createLivenessSession(detectorMode = 'mediapipe') {
  const formData = new FormData();
  formData.append('detector_mode', detectorMode);

  const response = await fetch(`${LIVENESS_API_BASE}/session`, {
    method: 'POST',
    headers: { ...authHeader() },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.json();
}

export async function pollStatus(jobId, mode = 'image') {
  const apiBase = mode === 'video' ? VIDEO_API_BASE : mode === 'audio' ? AUDIO_API_BASE : IMAGE_API_BASE;
  const response = await fetch(`${apiBase}/status/${jobId}`, { headers: { ...authHeader() } });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.json();
}

export async function getResult(jobId, mode = 'image') {
  const apiBase = mode === 'video' ? VIDEO_API_BASE : mode === 'audio' ? AUDIO_API_BASE : IMAGE_API_BASE;
  const response = await fetch(`${apiBase}/result/${jobId}`, { headers: { ...authHeader() } });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.json();
}

export function getArtifactUrl(jobId, name) {
  return `${IMAGE_API_BASE}/artifacts/${jobId}/${name}`;
}

export async function getHistory({ mode, days, limit = 20, offset = 0 } = {}) {
  const params = new URLSearchParams({ limit, offset })
  if (mode)     params.set('mode', mode)
  if (days > 0) params.set('days', days)

  const response = await fetch(`/v1/history?${params}`, { headers: { ...authHeader() } })
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response))
  }
  return response.json()
}

export async function getStats() {
  const response = await fetch('/v1/stats/summary', { headers: { ...authHeader() } });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return response.json();
}

export async function getModelStats() {
  const response = await fetch('/v1/stats/models', { headers: { ...authHeader() } });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return response.json();
}
