import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Camera, Image, Upload as UploadIcon, Video } from 'lucide-react';
import './Upload.css';

const MAX_RECORDING_MS = 5000;
const FRAME_COUNT = 7;
const LIVENESS_TIMEOUT_MS = 60000;

const BLINK_CLOSE_THRESHOLD_FACTOR = 0.82;
const BLINK_MIN_CLOSED_FRAMES = 1;
const DISTANCE_TARGET_RANGE_RATIO = 0.16;
const SMILE_TARGET_FACTOR = 1.08;
const SMILE_MIN_ABS_DELTA = 0.025;
const SMILE_CONSECUTIVE_FRAMES_REQUIRED = 1;
const MIN_CHALLENGES_PER_SESSION = 3;
const MAX_CHALLENGES_PER_SESSION = 5;

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

function shuffle(items) {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
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
  const [challengeLabel, setChallengeLabel] = useState('');
  const [challengeStep, setChallengeStep] = useState({ index: 0, total: 0 });
  const [challengeHint, setChallengeHint] = useState('');
  const [challengeList, setChallengeList] = useState([]);

  const [earDebug, setEarDebug] = useState({
    ear: null,
    baseline: null,
    threshold: null,
    closedFrames: 0,
    blinks: 0,
    faceWidth: null,
    smileScore: null,
  });

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectorRef = useRef(null);
  const detectorIntervalRef = useRef(null);
  const detectorBusyRef = useRef(false);
  const recordingTickerRef = useRef(null);

  const livenessStatusRef = useRef('pending');
  const livenessStartRef = useRef(null);
  const challengesRef = useRef([]);
  const challengeIndexRef = useRef(0);

  const blinkCountRef = useRef(0);
  const eyeClosedFramesRef = useRef(0);
  const eyeWasClosedRef = useRef(false);
  const eyeBaselineRef = useRef(0.28);

  const distanceMinRef = useRef(null);
  const distanceMaxRef = useRef(null);
  const distanceBaselineRef = useRef(null);

  const smileBaselineRef = useRef(null);
  const smileConsecutiveRef = useRef(0);

  const detectorLabelMap = useMemo(() => ({ mediapipe: 'MediaPipe', none: 'Backend' }), []);

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

  const resetChallengeInternals = useCallback(() => {
    blinkCountRef.current = 0;
    eyeClosedFramesRef.current = 0;
    eyeWasClosedRef.current = false;
    eyeBaselineRef.current = 0.28;

    distanceMinRef.current = null;
    distanceMaxRef.current = null;
    distanceBaselineRef.current = null;

    smileBaselineRef.current = null;
    smileConsecutiveRef.current = 0;

    setEarDebug({
      ear: null,
      baseline: null,
      threshold: null,
      closedFrames: 0,
      blinks: 0,
      faceWidth: null,
      smileScore: null,
    });
  }, []);

  const applyChallengeUi = useCallback((index) => {
    const all = challengesRef.current;
    const challenge = all[index];
    if (!challenge) {
      return;
    }

    setChallengeStep({ index: index + 1, total: all.length });
    setChallengeLabel(challenge.label);
    setChallengeHint(challenge.hint);
    setLivenessProgress(0);
  }, []);

  const buildChallenges = useCallback(() => {
    const blinkTarget = Math.random() > 0.5 ? 2 : 1;
    const pool = [
      {
        id: 'blink',
        requirements: ['blink'],
        label: `Piscar ${blinkTarget}x`,
        hint: `Piscar ${blinkTarget} vez(es) de forma intencional.`,
        targets: { blink: blinkTarget },
      },
      {
        id: 'distance',
        requirements: ['distance'],
        label: 'Aproximar/Afastar',
        hint: 'Aproxime e afaste o rosto da camera.',
        targets: { distance: DISTANCE_TARGET_RANGE_RATIO },
      },
      {
        id: 'smile',
        requirements: ['smile'],
        label: 'Sorrir',
        hint: 'Sorria claramente por um instante.',
        targets: { smile: SMILE_TARGET_FACTOR },
      },
      {
        id: 'blink_smile',
        requirements: ['blink', 'smile'],
        label: `Piscar ${blinkTarget}x + Sorrir`,
        hint: `Nesta prova combinada, pisque ${blinkTarget} vez(es) e sorria.`,
        targets: { blink: blinkTarget, smile: SMILE_TARGET_FACTOR },
      },
      {
        id: 'distance_smile',
        requirements: ['distance', 'smile'],
        label: 'Aproximar/Afastar + Sorrir',
        hint: 'Aproxime/afaste o rosto e sorria na mesma prova.',
        targets: { distance: DISTANCE_TARGET_RANGE_RATIO, smile: SMILE_TARGET_FACTOR },
      },
      {
        id: 'blink_distance',
        requirements: ['blink', 'distance'],
        label: `Piscar ${blinkTarget}x + Aproximar/Afastar`,
        hint: `Pisque ${blinkTarget} vez(es) e aproxime/afaste o rosto na mesma prova.`,
        targets: { blink: blinkTarget, distance: DISTANCE_TARGET_RANGE_RATIO },
      },
    ];

    const total = MIN_CHALLENGES_PER_SESSION
      + Math.floor(Math.random() * (MAX_CHALLENGES_PER_SESSION - MIN_CHALLENGES_PER_SESSION + 1));

    return shuffle(pool).slice(0, total);
  }, []);

  const resetLiveness = useCallback((detectorMode = faceValidationMode) => {
    resetChallengeInternals();

    if (detectorMode === 'none') {
      livenessStatusRef.current = 'unavailable';
      setLivenessStatus('unavailable');
      setChallengeStep({ index: 0, total: 0 });
      setChallengeLabel('Prova de vida indisponivel');
      setChallengeHint('Navegador sem suporte local.');
      setChallengeList([]);
      setLivenessProgress(0);
      livenessStartRef.current = null;
      return;
    }

    challengesRef.current = buildChallenges();
    setChallengeList(challengesRef.current.map((item) => item.label));
    challengeIndexRef.current = 0;
    livenessStartRef.current = Date.now();
    livenessStatusRef.current = 'pending';
    setLivenessStatus('pending');
    applyChallengeUi(0);
  }, [applyChallengeUi, buildChallenges, faceValidationMode, resetChallengeInternals]);

  const toDistance = useCallback((a, b) => Math.hypot((a.x - b.x), (a.y - b.y)), []);

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

    const horizontal = toDistance(e, f);
    if (!Number.isFinite(horizontal) || horizontal < 1e-4) {
      return null;
    }

    return (toDistance(a, b) + toDistance(c, d)) / (2 * horizontal);
  }, [toDistance]);

  const computeMetrics = useCallback((landmarks) => {
    if (!Array.isArray(landmarks) || landmarks.length < 387) {
      return null;
    }

    const leftEAR = calcEAR(landmarks, [33, 160, 158, 133, 153, 144]);
    const rightEAR = calcEAR(landmarks, [362, 385, 387, 263, 373, 380]);
    if (leftEAR === null || rightEAR === null) {
      return null;
    }
    const ear = (leftEAR + rightEAR) / 2;

    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    for (const lm of landmarks) {
      minX = Math.min(minX, lm.x);
      maxX = Math.max(maxX, lm.x);
    }
    const faceWidth = maxX - minX;

    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const eyeDist = toDistance(leftEye, rightEye);
    if (!Number.isFinite(eyeDist) || eyeDist < 1e-4) {
      return null;
    }

    const mouthLeft = landmarks[61];
    const mouthRight = landmarks[291];
    const mouthWidth = toDistance(mouthLeft, mouthRight);
    const smileScore = mouthWidth / eyeDist;

    return { ear, faceWidth, smileScore };
  }, [calcEAR, toDistance]);

  const completeCurrentChallenge = useCallback(() => {
    const nextIndex = challengeIndexRef.current + 1;
    if (nextIndex >= challengesRef.current.length) {
      livenessStatusRef.current = 'passed';
      setLivenessStatus('passed');
      setLivenessProgress(100);
      setVideoStep('Prova de vida aprovada. Pronto para gravar.');
      return;
    }

    challengeIndexRef.current = nextIndex;
    resetChallengeInternals();
    applyChallengeUi(nextIndex);
  }, [applyChallengeUi, resetChallengeInternals]);

  const updateLivenessFromLandmarks = useCallback((landmarks) => {
    const currentStatus = livenessStatusRef.current;
    if (currentStatus === 'passed' || currentStatus === 'unavailable') {
      return;
    }

    if (livenessStartRef.current && Date.now() - livenessStartRef.current > LIVENESS_TIMEOUT_MS) {
      if (livenessStatusRef.current !== 'timeout') {
        livenessStatusRef.current = 'timeout';
        setLivenessStatus('timeout');
        setVideoStep('Tempo excedido para prova de vida. Reative a camera e tente novamente.');
      }
      return;
    }

    const metrics = computeMetrics(landmarks);
    if (!metrics) {
      return;
    }

    const { ear, faceWidth, smileScore } = metrics;

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

    if (distanceBaselineRef.current === null) {
      distanceBaselineRef.current = faceWidth;
      distanceMinRef.current = faceWidth;
      distanceMaxRef.current = faceWidth;
    } else {
      distanceMinRef.current = Math.min(distanceMinRef.current, faceWidth);
      distanceMaxRef.current = Math.max(distanceMaxRef.current, faceWidth);
    }

    if (smileBaselineRef.current === null) {
      smileBaselineRef.current = smileScore;
    } else if (smileScore < smileBaselineRef.current * 1.2) {
      smileBaselineRef.current = (smileBaselineRef.current * 0.95) + (smileScore * 0.05);
    }

    const currentChallenge = challengesRef.current[challengeIndexRef.current];
    if (!currentChallenge) {
      return;
    }

    const blinkTarget = currentChallenge.targets?.blink ?? 1;
    const blinkProgress = Math.max(0, Math.min(100, (blinkCountRef.current / blinkTarget) * 100));
    const blinkPassed = blinkCountRef.current >= blinkTarget;

    const distanceTarget = currentChallenge.targets?.distance ?? DISTANCE_TARGET_RANGE_RATIO;
    const base = distanceBaselineRef.current || faceWidth;
    const range = Math.max(0, (distanceMaxRef.current - distanceMinRef.current) / Math.max(base, 1e-4));
    const distanceProgress = Math.max(0, Math.min(100, (range / distanceTarget) * 100));
    const distancePassed = range >= distanceTarget;

    const smileTarget = currentChallenge.targets?.smile ?? SMILE_TARGET_FACTOR;
    const baseline = smileBaselineRef.current || smileScore;
    const ratio = smileScore / Math.max(baseline, 1e-4);
    const absDelta = smileScore - baseline;
    const smilePassedFrame = ratio >= smileTarget || absDelta >= SMILE_MIN_ABS_DELTA;
    if (smilePassedFrame) {
      smileConsecutiveRef.current += 1;
    } else {
      smileConsecutiveRef.current = 0;
    }
    const ratioProgress = ratio / smileTarget;
    const deltaProgress = absDelta / SMILE_MIN_ABS_DELTA;
    const smileProgress = Math.max(0, Math.min(100, Math.max(ratioProgress, deltaProgress) * 100));
    const smilePassed = smileConsecutiveRef.current >= SMILE_CONSECUTIVE_FRAMES_REQUIRED;

    const progressByType = {
      blink: blinkProgress,
      distance: distanceProgress,
      smile: smileProgress,
    };
    const passedByType = {
      blink: blinkPassed,
      distance: distancePassed,
      smile: smilePassed,
    };
    const required = Array.isArray(currentChallenge.requirements) ? currentChallenge.requirements : [];
    const requirementProgress = required.map((key) => progressByType[key] ?? 0);
    const combinedProgress = requirementProgress.length
      ? requirementProgress.reduce((acc, value) => acc + value, 0) / requirementProgress.length
      : 0;

    setLivenessProgress(Math.max(0, Math.min(100, combinedProgress)));

    const passed = required.length
      ? required.every((key) => Boolean(passedByType[key]))
      : false;

    if (passed) {
      completeCurrentChallenge();
    }

    setEarDebug({
      ear,
      baseline: eyeBaselineRef.current,
      threshold: closeThreshold,
      closedFrames: eyeClosedFramesRef.current,
      blinks: blinkCountRef.current,
      faceWidth,
      smileScore,
    });
  }, [completeCurrentChallenge, computeMetrics]);

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
    setChallengeLabel('');
    setChallengeHint('');
    setChallengeStep({ index: 0, total: 0 });
    setChallengeList([]);

    livenessStartRef.current = null;
    challengesRef.current = [];
    challengeIndexRef.current = 0;

    resetChallengeInternals();
    stopRecordingTicker();
    setVideoStep('Camera desligada');
  }, [resetChallengeInternals, stopFaceDetection, stopRecordingTicker]);

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
        setVideoStep('MediaPipe ativo. Siga os desafios de prova de vida.');
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
      // drawImage usa os pixels originais do stream (nao aplica transform CSS da preview).
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
                  <div className="liveness-header">
                    <span className="liveness-title">Prova de vida</span>
                    {challengeStep.total > 0 && (
                      <span className="liveness-step-badge">{challengeStep.index}/{challengeStep.total}</span>
                    )}
                  </div>
                  {challengeStep.total > 0 && (
                    <p className="liveness-challenge">{challengeLabel}</p>
                  )}
                  <div className="liveness-progress">
                    <div className="liveness-progress-fill" style={{ width: `${livenessProgress}%` }} />
                  </div>
                  <span className="liveness-percent">{Math.round(livenessProgress)}%</span>
                  <span className="liveness-status">
                    {livenessStatus === 'passed'
                      ? 'Aprovado'
                      : livenessStatus === 'timeout'
                        ? 'Tempo excedido (reative a camera)'
                        : livenessStatus === 'unavailable'
                          ? 'Indisponivel no navegador'
                          : challengeHint || 'Pendente'}
                  </span>
                  {challengeList.length > 0 && (
                    <div className="liveness-checklist">
                      <span className="liveness-checklist-title">Checklist da prova</span>
                      <ul>
                        {challengeList.map((item, idx) => {
                          const done = livenessStatus === 'passed' || idx < challengeIndexRef.current;
                          const current = livenessStatus !== 'passed' && idx === challengeIndexRef.current;
                          return (
                            <li
                              key={`${item}-${idx}`}
                              className={`${done ? 'done' : ''} ${current ? 'current' : ''}`.trim()}
                            >
                              <span className="check-icon">{done ? 'OK' : current ? '>' : '...'}</span>
                              <span className="check-label">{item}</span>
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
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
              <p className="camera-note">Usando MediaPipe FaceMesh para prova de vida com desafios dinamicos.</p>
            )}
            {faceValidationMode === 'none' && cameraReady && (
              <p className="camera-note">Deteccao local indisponivel. Validacao de face ocorrera no backend.</p>
            )}
            {cameraError && <p className="camera-error">{cameraError}</p>}
            <p className="camera-step">Etapa: {videoStep}</p>
            {cameraReady && (
              <div className="ear-debug">
                <span className="ear-debug-title">DEBUG EAR/CHALLENGE</span>
                <span>EAR: {earDebug.ear !== null ? earDebug.ear.toFixed(3) : '--'}</span>
                <span>Baseline: {earDebug.baseline !== null ? earDebug.baseline.toFixed(3) : '--'}</span>
                <span>Threshold: {earDebug.threshold !== null ? earDebug.threshold.toFixed(3) : '--'}</span>
                <span>Frames fechados: {earDebug.closedFrames}</span>
                <span>Blinks: {earDebug.blinks}</span>
                <span>Face width: {earDebug.faceWidth !== null ? earDebug.faceWidth.toFixed(3) : '--'}</span>
                <span>Smile score: {earDebug.smileScore !== null ? earDebug.smileScore.toFixed(3) : '--'}</span>
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
