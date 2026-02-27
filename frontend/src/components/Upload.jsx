import { useCallback, useEffect, useRef, useState } from 'react';
import { Camera, Image, Upload as UploadIcon, Video } from 'lucide-react';
import './Upload.css';

const MAX_RECORDING_MS = 5000;
const FRAME_COUNT = 7;
const LIVENESS_REQUIRED_BLINKS = 1;
const LIVENESS_TIMEOUT_MS = 8000;
const BLINK_CLOSE_THRESHOLD_FACTOR = 0.82;
const BLINK_MIN_CLOSED_FRAMES = 1;

const SCRIPT_CACHE = {};

function loadScript(src) {
  if (SCRIPT_CACHE[src]) {
    return SCRIPT_CACHE[src];
  }

  SCRIPT_CACHE[src] = new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Falha ao carregar script: ${src}`));
    document.head.appendChild(script);
  });

  return SCRIPT_CACHE[src];
}

async function loadMediaPipeFaceMesh() {
  await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js');
  await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js');

  if (typeof window === 'undefined' || !window.FaceMesh) {
    throw new Error('MediaPipe FaceMesh indisponivel');
  }
}

function Upload({ onImageUpload, onVideoUpload }) {
  const [mode, setMode] = useState('image');
  const [isDragging, setIsDragging] = useState(false);
  const [targetModel, setTargetModel] = useState('auto');
  const [autoStartTried, setAutoStartTried] = useState(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [videoStep, setVideoStep] = useState('Aguardando modo de captura');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingMsLeft, setRecordingMsLeft] = useState(MAX_RECORDING_MS);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceValidationMode, setFaceValidationMode] = useState('none');
  const [livenessStatus, setLivenessStatus] = useState('pending'); // pending | passed | timeout | unavailable
  const [livenessProgress, setLivenessProgress] = useState(0);
  const [earDebug, setEarDebug] = useState({
    ear: null,
    baseline: null,
    threshold: null,
    closedFrames: 0,
    blinks: 0,
    isClosed: false,
  });

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectorRef = useRef(null);
  const detectorIntervalRef = useRef(null);
  const detectorBusyRef = useRef(false);
  const recordingTickerRef = useRef(null);
  const livenessStatusRef = useRef('pending');

  const blinkCountRef = useRef(0);
  const eyeClosedFramesRef = useRef(0);
  const eyeWasClosedRef = useRef(false);
  const eyeBaselineRef = useRef(0.28);
  const livenessStartRef = useRef(null);

  const stopFaceDetection = useCallback(() => {
    if (detectorIntervalRef.current) {
      clearInterval(detectorIntervalRef.current);
      detectorIntervalRef.current = null;
    }
    detectorBusyRef.current = false;
  }, []);

  const stopRecordingTicker = useCallback(() => {
    if (recordingTickerRef.current) {
      clearInterval(recordingTickerRef.current);
      recordingTickerRef.current = null;
    }
    setRecordingMsLeft(MAX_RECORDING_MS);
  }, []);

  const resetLiveness = useCallback((detectorMode = faceValidationMode) => {
    blinkCountRef.current = 0;
    eyeClosedFramesRef.current = 0;
    eyeWasClosedRef.current = false;
    eyeBaselineRef.current = 0.28;
    livenessStartRef.current = detectorMode === 'none' ? null : Date.now();
    setLivenessProgress(0);
    setEarDebug({
      ear: null,
      baseline: eyeBaselineRef.current,
      threshold: eyeBaselineRef.current * BLINK_CLOSE_THRESHOLD_FACTOR,
      closedFrames: 0,
      blinks: 0,
      isClosed: false,
    });
    const nextStatus = detectorMode === 'none' ? 'unavailable' : 'pending';
    livenessStatusRef.current = nextStatus;
    setLivenessStatus(nextStatus);
  }, [faceValidationMode]);

  const calcEAR = useCallback((landmarks, [p1, p2, p3, p4, p5, p6]) => {
    const a = landmarks[p2];
    const b = landmarks[p6];
    const c = landmarks[p3];
    const d = landmarks[p5];
    const e = landmarks[p1];
    const f = landmarks[p4];
    if (!a || !b || !c || !d || !e || !f) {
      return null;
    }
    const dist = (u, v) => Math.hypot((u.x - v.x), (u.y - v.y));
    const horizontal = dist(e, f);
    if (!Number.isFinite(horizontal) || horizontal < 1e-4) {
      return null;
    }
    return (dist(a, b) + dist(c, d)) / (2 * horizontal);
  }, []);

  const updateLivenessFromLandmarks = useCallback((landmarks) => {
    const currentStatus = livenessStatusRef.current;
    if (currentStatus === 'passed' || currentStatus === 'unavailable') {
      return;
    }

    if (livenessStartRef.current && Date.now() - livenessStartRef.current > LIVENESS_TIMEOUT_MS) {
      if (livenessStatusRef.current !== 'timeout') {
        livenessStatusRef.current = 'timeout';
        setLivenessStatus('timeout');
        setVideoStep('Nao detectamos o piscar ainda. Tente novamente piscando de forma intencional.');
      }
    }

    const leftEAR = calcEAR(landmarks, [33, 160, 158, 133, 153, 144]);
    const rightEAR = calcEAR(landmarks, [362, 385, 387, 263, 373, 380]);
    if (leftEAR === null || rightEAR === null) {
      return;
    }

    const ear = (leftEAR + rightEAR) / 2;

    if (ear > 0.18) {
      eyeBaselineRef.current = (eyeBaselineRef.current * 0.9) + (ear * 0.1);
    }

    const closeThreshold = eyeBaselineRef.current * BLINK_CLOSE_THRESHOLD_FACTOR;
    const isClosed = ear < closeThreshold;

    if (isClosed) {
      eyeClosedFramesRef.current += 1;
      eyeWasClosedRef.current = true;
    } else {
      if (eyeWasClosedRef.current && eyeClosedFramesRef.current >= BLINK_MIN_CLOSED_FRAMES) {
        blinkCountRef.current += 1;
      }
      eyeWasClosedRef.current = false;
      eyeClosedFramesRef.current = 0;
    }

    const progress = Math.max(0, Math.min(100, (blinkCountRef.current / LIVENESS_REQUIRED_BLINKS) * 100));
    setLivenessProgress(progress);
    setEarDebug({
      ear,
      baseline: eyeBaselineRef.current,
      threshold: closeThreshold,
      closedFrames: eyeClosedFramesRef.current,
      blinks: blinkCountRef.current,
      isClosed,
    });

    if (blinkCountRef.current >= LIVENESS_REQUIRED_BLINKS) {
      livenessStatusRef.current = 'passed';
      setLivenessStatus('passed');
      setVideoStep('Prova de vida aprovada. Pronto para gravar.');
    }
  }, [calcEAR]);

  const stopCamera = useCallback(() => {
    stopFaceDetection();

    if (detectorRef.current && typeof detectorRef.current.close === 'function') {
      try {
        detectorRef.current.close();
      } catch {
        // noop
      }
    }
    detectorRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setFaceValidationMode('none');
    setCameraReady(false);
    setFaceDetected(false);
    setIsRecording(false);
    livenessStatusRef.current = 'pending';
    setLivenessStatus('pending');
    setLivenessProgress(0);
    setEarDebug({
      ear: null,
      baseline: null,
      threshold: null,
      closedFrames: 0,
      blinks: 0,
      isClosed: false,
    });
    blinkCountRef.current = 0;
    eyeClosedFramesRef.current = 0;
    eyeWasClosedRef.current = false;
    eyeBaselineRef.current = 0.28;
    livenessStartRef.current = null;
    stopRecordingTicker();
    setVideoStep('Camera desligada');
  }, [stopFaceDetection, stopRecordingTicker]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  useEffect(() => {
    if (mode !== 'video') {
      stopCamera();
      setAutoStartTried(false);
    }
  }, [mode, stopCamera]);

  useEffect(() => {
    const video = videoRef.current;
    const stream = streamRef.current;

    if (!video || !stream || !cameraReady || mode !== 'video') {
      return;
    }

    if (video.srcObject !== stream) {
      video.srcObject = stream;
    }

    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(() => {});
    }
  }, [cameraReady, mode]);

  const startFaceDetection = useCallback(() => {
    if (!videoRef.current || !detectorRef.current) {
      setFaceDetected(true);
      return;
    }

    stopFaceDetection();
    detectorIntervalRef.current = setInterval(async () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2 || detectorBusyRef.current) {
        return;
      }

      detectorBusyRef.current = true;
      try {
        await detectorRef.current.send({ image: video });
      } catch {
        setFaceDetected(false);
      } finally {
        detectorBusyRef.current = false;
      }
    }, 250);
  }, [stopFaceDetection]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setVideoStep('Solicitando permissao da camera...');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false,
      });

      streamRef.current = stream;
      setCameraReady(true);
      setFaceDetected(false);
      setVideoStep('Camera ativa. Preparando detector de face...');

      try {
        await loadMediaPipeFaceMesh();

        const mpDetector = new window.FaceMesh({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        });

        mpDetector.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        mpDetector.onResults((results) => {
          const hasFaces = Array.isArray(results?.multiFaceLandmarks) && results.multiFaceLandmarks.length > 0;
          setFaceDetected(hasFaces);

          if (hasFaces) {
            if (!livenessStartRef.current) {
              livenessStartRef.current = Date.now();
            }
            updateLivenessFromLandmarks(results.multiFaceLandmarks[0]);
          }
        });

        detectorRef.current = mpDetector;
        setFaceValidationMode('mediapipe');
        resetLiveness('mediapipe');
        setVideoStep('MediaPipe ativo. Aguarde e pisque os olhos.');
      } catch {
        detectorRef.current = null;
        setFaceValidationMode('none');
        setFaceDetected(true);
        resetLiveness('none');
        setCameraError('Deteccao local indisponivel. Gravacao liberada sem pre-validacao.');
        setVideoStep('Sem detector local. Validacao ocorrera no backend.');
      }

      setTimeout(() => startFaceDetection(), 300);
    } catch {
      setCameraError('Nao foi possivel acessar a camera. Verifique a permissao no navegador.');
      setVideoStep('Falha ao abrir camera');
      stopCamera();
    }
  }, [resetLiveness, startFaceDetection, stopCamera, updateLivenessFromLandmarks]);

  useEffect(() => {
    if (mode === 'video' && !cameraReady && !autoStartTried) {
      setAutoStartTried(true);
      startCamera();
    }
  }, [autoStartTried, cameraReady, mode, startCamera]);

  const captureFramesFromLiveVideo = useCallback(async (count = FRAME_COUNT, durationMs = MAX_RECORDING_MS) => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      throw new Error('Preview da camera nao esta pronta para captura.');
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    const frames = [];
    const stepMs = Math.max(Math.floor(durationMs / count), 250);

    for (let i = 0; i < count; i += 1) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameBlob = await new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', 0.9);
      });

      if (frameBlob) {
        frames.push(new File([frameBlob], `frame_${i + 1}.jpg`, { type: 'image/jpeg' }));
      }

      if (i < count - 1) {
        await new Promise((resolve) => setTimeout(resolve, stepMs));
      }
    }

    return frames;
  }, []);

  const handleRecord = useCallback(async () => {
    if (!streamRef.current || isRecording) {
      return;
    }

    const shouldRequireFace = faceValidationMode !== 'none';
    if (shouldRequireFace && !faceDetected) {
      setCameraError('Posicione o rosto no centro para iniciar a gravacao.');
      setVideoStep('Aguardando face para iniciar');
      return;
    }

    const shouldRequireLiveness = faceValidationMode !== 'none' && livenessStatus !== 'unavailable';
    if (shouldRequireLiveness && livenessStatus !== 'passed') {
      setCameraError('Conclua a prova de vida antes de gravar.');
      setVideoStep('Aguardando prova de vida');
      return;
    }

    setCameraError(null);

    try {
      setVideoStep('Capturando frames direto da camera...');
      setIsRecording(true);
      setRecordingMsLeft(MAX_RECORDING_MS);
      recordingTickerRef.current = setInterval(() => {
        setRecordingMsLeft((prev) => Math.max(0, prev - 100));
      }, 100);

      const frames = await captureFramesFromLiveVideo(FRAME_COUNT, MAX_RECORDING_MS);
      if (!frames.length) {
        throw new Error('Nao foi possivel capturar frames da camera.');
      }

      const previewUrl = URL.createObjectURL(frames[0]);
      setVideoStep('Enviando frames para analise...');
      onVideoUpload(frames, targetModel, previewUrl);
    } catch (error) {
      setCameraError(error.message || 'Falha na captura direta de frames.');
      setVideoStep('Falha ao capturar/enviar frames');
    } finally {
      setIsRecording(false);
      stopRecordingTicker();
    }
  }, [captureFramesFromLiveVideo, faceDetected, faceValidationMode, isRecording, livenessStatus, onVideoUpload, stopRecordingTicker, targetModel]);

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
      onImageUpload(file, targetModel);
    }
  }, [onImageUpload, targetModel]);

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      onImageUpload(file, targetModel);
    }
  }, [onImageUpload, targetModel]);

  const detectorLabelMap = {
    mediapipe: 'MediaPipe',
    none: 'Backend',
  };

  return (
    <div className="upload-container">
      <div className="mode-switch" role="tablist" aria-label="Modo de analise">
        <button
          type="button"
          className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
          onClick={() => setMode('image')}
        >
          <Image size={16} />
          Foto
        </button>
        <button
          type="button"
          className={`mode-btn ${mode === 'video' ? 'active' : ''}`}
          onClick={() => setMode('video')}
        >
          <Video size={16} />
          Video curto (5s)
        </button>
      </div>

      <div className="model-target-card">
        <label htmlFor="target-model">Modelo alvo</label>
        <select
          id="target-model"
          value={targetModel}
          onChange={(e) => setTargetModel(e.target.value)}
        >
          <option value="auto">Automatico (melhor disponivel)</option>
          <option value="efficientnet_b0_v5">EfficientNet-B0 v5</option>
          <option value="efficientnet_b0_v4">EfficientNet-B0 v4</option>
          <option value="efficientnet_b0">EfficientNet-B0 (generico)</option>
          <option value="efficientnet_b3">EfficientNet-B3</option>
          <option value="efficientnet_b4">EfficientNet-B4</option>
          <option value="alexnet">AlexNet</option>
        </select>
      </div>

      {mode === 'image' ? (
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
      ) : (
        <div className="camera-zone">
          <div className="camera-preview">
            {cameraReady && (
              <span className={`detector-badge ${faceValidationMode}`}>
                Detector: {detectorLabelMap[faceValidationMode] || 'Backend'}
              </span>
            )}
            {cameraReady && isRecording && (
              <div className="recording-overlay">
                <span className="recording-pill">
                  <span className="recording-dot" />
                  REC {(recordingMsLeft / 1000).toFixed(1)}s
                </span>
                <div className="recording-progress">
                  <div
                    className="recording-progress-fill"
                    style={{ width: `${((MAX_RECORDING_MS - recordingMsLeft) / MAX_RECORDING_MS) * 100}%` }}
                  />
                </div>
              </div>
            )}
            {cameraReady ? (
              <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
            ) : (
              <div className="camera-placeholder">
                <Camera size={42} />
                <p>Ative a camera frontal para gravar ate 5s</p>
              </div>
            )}
          </div>

          <div className="camera-controls">
            {!cameraReady ? (
              <button type="button" className="btn-primary" onClick={startCamera}>
                Ativar camera
              </button>
            ) : (
              <>
                <span className={`face-status ${faceDetected ? 'ok' : 'wait'}`}>
                  {faceDetected ? 'Face detectada: pronto para gravar' : 'Aguardando face no enquadramento'}
                </span>
                <div className={`liveness-box ${livenessStatus}`}>
                  <span className="liveness-title">
                    Prova de vida: pisque os olhos uma vez
                  </span>
                  <div className="liveness-progress">
                    <div className="liveness-progress-fill" style={{ width: `${livenessProgress}%` }} />
                  </div>
                  <span className="liveness-status">
                    {livenessStatus === 'passed'
                      ? 'Aprovado'
                      : livenessStatus === 'timeout'
                        ? 'Tempo excedido (tente piscar novamente)'
                      : livenessStatus === 'unavailable'
                        ? 'Indisponivel no navegador'
                        : 'Pendente'}
                  </span>
                </div>
                <button
                  type="button"
                  className="btn-primary"
                  disabled={
                    isRecording
                    || (
                      faceValidationMode !== 'none'
                      && (
                        !faceDetected
                        || (livenessStatus !== 'passed' && livenessStatus !== 'unavailable')
                      )
                    )
                  }
                  onClick={handleRecord}
                >
                  {isRecording ? 'Gravando (5s)...' : 'Gravar e analisar'}
                </button>
                <button type="button" className="btn-secondary" onClick={stopCamera}>
                  Desligar camera
                </button>
              </>
            )}

            {faceValidationMode === 'mediapipe' && cameraReady && (
              <p className="camera-note">Usando MediaPipe FaceMesh no navegador para pre-validacao de prova de vida.</p>
            )}
            {faceValidationMode === 'none' && cameraReady && (
              <p className="camera-note">Deteccao local indisponivel. Validacao de face ocorrera no backend.</p>
            )}
            {cameraError && <p className="camera-error">{cameraError}</p>}
            <p className="camera-step">Etapa: {videoStep}</p>
            {cameraReady && (
              <div className="ear-debug">
                <span className="ear-debug-title">DEBUG EAR</span>
                <span>EAR: {earDebug.ear !== null ? earDebug.ear.toFixed(3) : '--'}</span>
                <span>Baseline: {earDebug.baseline !== null ? earDebug.baseline.toFixed(3) : '--'}</span>
                <span>Threshold: {earDebug.threshold !== null ? earDebug.threshold.toFixed(3) : '--'}</span>
                <span>Fechado: {earDebug.isClosed ? 'sim' : 'nao'}</span>
                <span>Frames fechados: {earDebug.closedFrames}</span>
                <span>Blinks: {earDebug.blinks}</span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="upload-info">
        <h3>Como funciona?</h3>
        <ol>
          <li>{mode === 'image' ? 'Envie uma selfie ou foto com rosto' : 'Grave ate 5s com o rosto centralizado'}</li>
          <li>{mode === 'image' ? 'Nossa IA analisa a imagem em tempo real' : 'Extraimos frames do video em memoria'}</li>
          <li>Receba o resultado com nivel de risco</li>
        </ol>
      </div>
    </div>
  );
}

export default Upload;
