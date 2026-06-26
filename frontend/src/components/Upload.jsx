import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Camera, FileAudio, Image, Upload as UploadIcon, Video } from 'lucide-react';
import { analyzeLiveness, createLivenessSession } from '../services/api';
import './Upload.css';

const MAX_RECORDING_MS = 5000;
const FRAME_COUNT = 7;
const LIVENESS_TIMEOUT_MS = 40000;

const BLINK_CLOSE_THRESHOLD_FACTOR = 0.85;
const BLINK_MIN_CLOSED_FRAMES = 2;
const BLINK_REOPEN_THRESHOLD_FACTOR = 0.93;
const BLINK_MIN_DEPTH_FACTOR = 0.95;
const BLINK_MAX_FACE_WIDTH_VARIATION_RATIO = 0.12;
const DISTANCE_TARGET_RANGE_RATIO = 0.16;
const SMILE_TARGET_FACTOR = 1.1;
const SMILE_MIN_ABS_DELTA = 0.035;
const SMILE_CONSECUTIVE_FRAMES_REQUIRED = 1;
const SMILE_INTERMEDIATE_FACTOR = 1.05;
const SMILE_BASELINE_UPDATE_MAX_RATIO = 1.015;
const SMILE_HOLD_CONSECUTIVE_FRAMES_REQUIRED = 3;
const MOUTH_OPEN_TARGET_FACTOR = 1.46;
const MOUTH_OPEN_INTERMEDIATE_FACTOR = 1.24;
const MOUTH_OPEN_MIN_ABS_DELTA = 0.065;
const MOUTH_OPEN_CONSECUTIVE_FRAMES_REQUIRED = 2;
const HEAD_TURN_TARGET_DELTA = 0.10;
const HEAD_PITCH_TARGET_DELTA = 0.07;
const MIXED_DISTANCE_PROGRESS_WEIGHT = 0.18;
const MIXED_BLINK_PROGRESS_WEIGHT = 1.25;
const MIXED_SMILE_PROGRESS_WEIGHT = 1.15;
const MIXED_HEAD_POSE_PROGRESS_WEIGHT = 1.35;
const MIN_CHALLENGES_PER_SESSION = 3;
const MAX_CHALLENGES_PER_SESSION = 5;
const CAMERA_CONSTRAINTS = [
  {
    label: 'HD',
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 24, max: 30 },
    },
  },
  {
    label: 'qHD',
    video: {
      facingMode: 'user',
      width: { ideal: 960 },
      height: { ideal: 540 },
      frameRate: { ideal: 24, max: 30 },
    },
  },
  {
    label: 'VGA',
    video: {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 20, max: 30 },
    },
  },
  {
    label: 'Padrao',
    video: { facingMode: 'user' },
  },
];

const AUDIO_MODEL_OPTIONS = [
  { value: 'default', label: 'FakeBrAccent codec equalizado' },
  { value: 'audio_v1_1_domain_adv', label: 'Áudio V1.1 domínio adversarial' },
  { value: 'brspeech_sample', label: 'BRSpeech-DF amostra (experimental)' },
];

const SPOOF_REASON_LABELS = {
  planar_motion_without_internal_face_change: 'movimento parece plano, sem mudanca interna do rosto',
  screen_or_reflection_glare: 'brilho/reflexo suspeito de tela',
  low_face_texture: 'textura facial baixa',
  screen_rectangle_detected: 'retangulo de tela detectado ao redor da face',
  few_valid_face_frames: 'poucos frames validos com face',
  no_valid_face_frames: 'nenhum frame valido com face',
  model_spoof_score_high: 'modelo anti-spoof detectou foto/tela/replay',
  model_file_missing: 'modelo anti-spoof nao encontrado no backend',
  onnxruntime_missing: 'onnxruntime nao esta instalado no backend',
  no_face_crops: 'nenhum recorte facial valido para o modelo anti-spoof',
  insufficient_antispoof_quality: 'qualidade insuficiente para anti-spoof, aproxime o rosto e mantenha a camera estavel',
  mobile_antispoof_uncertain: 'anti-spoof inconclusivo no celular, refaca em luz mais uniforme',
  borderline_spoof_score_with_high_frame: 'score anti-spoof em zona cinza com frame individual suspeito',
  fusion_low_internal_motion_high_face_background_delta: 'sinais combinados de tela: pouco movimento interno e face muito diferente do fundo',
};

const formatPercent = (value) => {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null;
  }
  return `${Math.round(value * 100)}%`;
};

const detectClientDevice = () => {
  const uaData = navigator.userAgentData;
  const userAgent = navigator.userAgent || '';
  const platform = uaData?.platform || navigator.platform || '';
  const maxTouchPoints = navigator.maxTouchPoints || 0;
  const width = window.innerWidth || 0;
  const height = window.innerHeight || 0;
  const mobilePattern = /Android|iPhone|iPod|IEMobile|Opera Mini|Mobile/i;
  const tabletPattern = /iPad|Tablet|PlayBook|Silk/i;
  const uaMobile = Boolean(uaData?.mobile) || mobilePattern.test(userAgent);
  const uaTablet = tabletPattern.test(userAgent) || (maxTouchPoints > 1 && Math.min(width, height) >= 700);
  const deviceClass = uaTablet ? 'tablet' : uaMobile ? 'mobile' : 'desktop';

  return {
    device_class: deviceClass,
    capture_source_guess: deviceClass === 'desktop' ? 'webcam' : 'phone_camera',
    platform,
    touch_points: maxTouchPoints,
    viewport: { width, height },
  };
};

const buildClientCaptureContext = (stream) => {
  const device = detectClientDevice();
  const videoTrack = stream?.getVideoTracks?.()[0] || null;
  const settings = videoTrack?.getSettings?.() || {};

  return {
    ...device,
    camera: {
      label: videoTrack?.label || null,
      facing_mode: settings.facingMode || null,
      width: settings.width || null,
      height: settings.height || null,
      frame_rate: settings.frameRate || null,
    },
  };
};

const openBestCameraStream = async () => {
  let lastError = null;

  for (const option of CAMERA_CONSTRAINTS) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: option.video,
        audio: false,
      });
      return { stream, profile: option.label };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error('Nao foi possivel acessar a camera.');
};

const getLivenessFailureDetail = (livenessResult) => {
  const details = [];
  const failedChallenges = (livenessResult.challenge_results || [])
    .filter((challenge) => !challenge.passed)
    .map((challenge) => challenge.label || challenge.id)
    .filter(Boolean);

  if (failedChallenges.length > 0) {
    details.push(`desafio backend nao confirmado: ${failedChallenges.slice(0, 2).join(', ')}`);
  }

  const spoofReasons = [
    ...(livenessResult.spoof_analysis?.reasons || []),
    ...(livenessResult.spoof_analysis?.model?.reasons || []),
    ...(livenessResult.checks?.anti_spoof_model?.reasons || []),
  ];
  const uniqueReasons = [...new Set(spoofReasons)];
  if (uniqueReasons.length > 0) {
    details.push(uniqueReasons.map((reason) => SPOOF_REASON_LABELS[reason] || reason).join('; '));
  }

  const modelResult = livenessResult.spoof_analysis?.model || livenessResult.checks?.anti_spoof_model;
  const pSpoof = formatPercent(modelResult?.p_spoof);
  const threshold = formatPercent(modelResult?.threshold);
  if (pSpoof && threshold) {
    details.push(`anti-spoof p75=${pSpoof} limite=${threshold}`);
  }

  return details.join('. ');
};

const SCRIPT_CACHE = {};

function loadScript(src) {
  if (SCRIPT_CACHE[src]) {
    return SCRIPT_CACHE[src];
  }

  SCRIPT_CACHE[src] = new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === 'true') {
        resolve();
        return;
      }
      if (existing.dataset.error === 'true') {
        reject(new Error(`Falha ao carregar script: ${src}`));
        return;
      }
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(`Falha ao carregar script: ${src}`)), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => {
      script.dataset.loaded = 'true';
      resolve();
    };
    script.onerror = () => {
      script.dataset.error = 'true';
      reject(new Error(`Falha ao carregar script: ${src}`));
    };
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

function pickWeightedChallenges(pool, total) {
  const available = pool.map((item) => ({ ...item }));
  const selected = [];

  while (available.length > 0 && selected.length < total) {
    const totalWeight = available.reduce((sum, item) => sum + (item.weight ?? 1), 0);
    let ticket = Math.random() * totalWeight;
    let chosenIndex = 0;

    for (let i = 0; i < available.length; i += 1) {
      ticket -= available[i].weight ?? 1;
      if (ticket <= 0) {
        chosenIndex = i;
        break;
      }
    }

    selected.push(available[chosenIndex]);
    available.splice(chosenIndex, 1);
  }

  return shuffle(selected);
}

function Upload({ onImageUpload, onAudioUpload, onVideoUpload, onLivenessRejected }) {
  const [mode, setMode] = useState('image');
  const [isDragging, setIsDragging] = useState(false);
  const [targetModel, setTargetModel] = useState('auto');
  const [targetAudioModel, setTargetAudioModel] = useState('default');
  const [useTemporalMemory, setUseTemporalMemory] = useState(true);
  const [autoStartTried, setAutoStartTried] = useState(false);

  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [videoStep, setVideoStep] = useState('Aguardando modo de captura');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingMsLeft, setRecordingMsLeft] = useState(MAX_RECORDING_MS);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceValidationMode, setFaceValidationMode] = useState('none');
  const [useBackendLiveness, setUseBackendLiveness] = useState(true);
  const [skipLocalLiveness, setSkipLocalLiveness] = useState(false);
  const [backendLivenessStatus, setBackendLivenessStatus] = useState('idle');

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
    mouthOpenScore: null,
    headYaw: null,
    headPitch: null,
  });

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectorRef = useRef(null);
  const detectorIntervalRef = useRef(null);
  const detectorBusyRef = useRef(false);
  const recordingTickerRef = useRef(null);
  const lastDetectorActivityRef = useRef(0);
  const lastChallengeProgressRef = useRef(0);
  const lastChallengeProgressAtRef = useRef(0);
  const lastFaceSeenAtRef = useRef(0);

  const livenessStatusRef = useRef('pending');
  const livenessStartRef = useRef(null);
  const livenessSessionRef = useRef(null);
  const challengesRef = useRef([]);
  const challengeIndexRef = useRef(0);

  const blinkCountRef = useRef(0);
  const eyeClosedFramesRef = useRef(0);
  const eyeWasClosedRef = useRef(false);
  const eyeBaselineRef = useRef(0.28);
  const blinkMinEarRef = useRef(null);
  const blinkStartFaceWidthRef = useRef(null);

  const distanceMinRef = useRef(null);
  const distanceMaxRef = useRef(null);
  const distanceBaselineRef = useRef(null);

  const smileBaselineRef = useRef(null);
  const smileIntermediateReachedRef = useRef(false);
  const smilePeakRatioRef = useRef(1);
  const smileConsecutiveRef = useRef(0);
  const mouthOpenBaselineRef = useRef(null);
  const mouthOpenIntermediateReachedRef = useRef(false);
  const mouthOpenPeakRatioRef = useRef(1);
  const mouthOpenConsecutiveRef = useRef(0);
  const headPoseBaselineRef = useRef(null);
  const headYawMinRef = useRef(null);
  const headYawMaxRef = useRef(null);
  const headPitchMinRef = useRef(null);
  const headPitchMaxRef = useRef(null);

  const detectorLabelMap = useMemo(() => ({ mediapipe: 'MediaPipe', none: 'Backend' }), []);

  const buildBackendLivenessSession = useCallback(() => {
    if (livenessSessionRef.current) {
      return livenessSessionRef.current;
    }

    return {
      detector_mode: faceValidationMode,
      challenges: challengesRef.current.map((challenge) => ({
        id: challenge.id,
        label: challenge.label,
        hint: challenge.hint,
        requirements: Array.isArray(challenge.requirements) ? challenge.requirements : [],
        targets: challenge.targets || {},
      })),
    };
  }, [faceValidationMode]);

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
    blinkMinEarRef.current = null;
    blinkStartFaceWidthRef.current = null;

    distanceMinRef.current = null;
    distanceMaxRef.current = null;
    distanceBaselineRef.current = null;

    smileBaselineRef.current = null;
    smileIntermediateReachedRef.current = false;
    smilePeakRatioRef.current = 1;
    smileConsecutiveRef.current = 0;
    mouthOpenBaselineRef.current = null;
    mouthOpenIntermediateReachedRef.current = false;
    mouthOpenPeakRatioRef.current = 1;
    mouthOpenConsecutiveRef.current = 0;
    headPoseBaselineRef.current = null;
    headYawMinRef.current = null;
    headYawMaxRef.current = null;
    headPitchMinRef.current = null;
    headPitchMaxRef.current = null;

    setEarDebug({
      ear: null,
      baseline: null,
      threshold: null,
      closedFrames: 0,
      blinks: 0,
      faceWidth: null,
      smileScore: null,
      mouthOpenScore: null,
      headYaw: null,
      headPitch: null,
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
    lastChallengeProgressRef.current = 0;
    lastChallengeProgressAtRef.current = Date.now();
  }, []);

  const buildChallenges = useCallback(() => {
    const blinkTarget = Math.random() > 0.5 ? 2 : 1;
    const mouthPool = [
      {
        id: 'mouth_head_left',
        requirements: ['mouth_open', 'head_left'],
        label: 'Abrir a boca + Virar para a esquerda',
        hint: 'Abra a boca e vire levemente o rosto para a esquerda.',
        targets: { mouth_open: MOUTH_OPEN_TARGET_FACTOR, head_left: HEAD_TURN_TARGET_DELTA },
        weight: 1.2,
      },
      {
        id: 'mouth_head_right',
        requirements: ['mouth_open', 'head_right'],
        label: 'Abrir a boca + Virar para a direita',
        hint: 'Abra a boca e vire levemente o rosto para a direita.',
        targets: { mouth_open: MOUTH_OPEN_TARGET_FACTOR, head_right: HEAD_TURN_TARGET_DELTA },
        weight: 1.2,
      },
    ];
    const supportPool = [
      {
        id: 'mouth_hold',
        requirements: ['mouth_open'],
        label: 'Abrir a boca',
        hint: 'Olhe para a camera e abra a boca gradualmente ate ficar bem visivel.',
        targets: { mouth_open: MOUTH_OPEN_TARGET_FACTOR },
        weight: 0.75,
      },
      {
        id: 'blink_head',
        requirements: ['blink', 'head_left'],
        label: `Piscar ${blinkTarget}x + Virar`,
        hint: `Olhando para a camera, pisque ${blinkTarget} vez(es) e vire levemente para a esquerda.`,
        targets: { blink: blinkTarget, head_left: HEAD_TURN_TARGET_DELTA },
        weight: 0.5,
      },
      {
        id: 'distance_head',
        requirements: ['distance', 'head_right'],
        label: 'Aproximar/Afastar + Virar',
        hint: 'Aproxime/afaste o rosto devagar e vire levemente para a direita.',
        targets: { distance: DISTANCE_TARGET_RANGE_RATIO, head_right: HEAD_TURN_TARGET_DELTA },
        weight: 0.55,
      },
    ];

    const total = MIN_CHALLENGES_PER_SESSION
      + Math.floor(Math.random() * (MAX_CHALLENGES_PER_SESSION - MIN_CHALLENGES_PER_SESSION + 1));
    const mandatory = pickWeightedChallenges(mouthPool, 1);
    const remainingPool = [...mouthPool, ...supportPool].filter((challenge) => (
      challenge.id !== mandatory[0].id
    ));

    return [...mandatory, ...pickWeightedChallenges(remainingPool, total - mandatory.length)]
      .sort(() => Math.random() - 0.5);
  }, []);

  const resetLiveness = useCallback((detectorMode = faceValidationMode, serverSession = null) => {
    resetChallengeInternals();

    if (detectorMode === 'none') {
      livenessSessionRef.current = null;
      livenessStatusRef.current = 'unavailable';
      setLivenessStatus('unavailable');
      setChallengeStep({ index: 0, total: 0 });
      setChallengeLabel('Prova de vida indisponivel');
      setChallengeHint('Navegador sem suporte local.');
      setChallengeList([]);
      setLivenessProgress(0);
      livenessStartRef.current = null;
      lastDetectorActivityRef.current = Date.now();
      lastChallengeProgressRef.current = 0;
      lastChallengeProgressAtRef.current = Date.now();
      lastFaceSeenAtRef.current = 0;
      return;
    }

    if (serverSession && Array.isArray(serverSession.challenges) && serverSession.challenges.length > 0) {
      livenessSessionRef.current = serverSession;
      challengesRef.current = serverSession.challenges;
    } else {
      livenessSessionRef.current = null;
      challengesRef.current = buildChallenges();
    }
    setChallengeList(challengesRef.current.map((item) => item.label));
    challengeIndexRef.current = 0;
    livenessStartRef.current = Date.now();
    lastDetectorActivityRef.current = Date.now();
    lastChallengeProgressRef.current = 0;
    lastChallengeProgressAtRef.current = Date.now();
    lastFaceSeenAtRef.current = 0;
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
    const upperLip = landmarks[13];
    const lowerLip = landmarks[14];
    const mouthOpenScore = toDistance(upperLip, lowerLip) / eyeDist;
    const noseTip = landmarks[1];
    if (!noseTip) {
      return null;
    }
    const eyeMidX = (leftEye.x + rightEye.x) / 2;
    const eyeMidY = (leftEye.y + rightEye.y) / 2;
    const headYaw = (noseTip.x - eyeMidX) / eyeDist;
    const headPitch = (noseTip.y - eyeMidY) / eyeDist;

    return { ear, faceWidth, smileScore, mouthOpenScore, headYaw, headPitch };
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
    lastDetectorActivityRef.current = Date.now();

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

    const { ear, faceWidth, smileScore, mouthOpenScore, headYaw, headPitch } = metrics;

    if (ear > 0.18) {
      eyeBaselineRef.current = (eyeBaselineRef.current * 0.9) + (ear * 0.1);
    }

    const closeThreshold = eyeBaselineRef.current * BLINK_CLOSE_THRESHOLD_FACTOR;
    const reopenThreshold = eyeBaselineRef.current * BLINK_REOPEN_THRESHOLD_FACTOR;
    const isClosed = ear < closeThreshold;

    if (isClosed) {
      if (!eyeWasClosedRef.current) {
        blinkStartFaceWidthRef.current = faceWidth;
        blinkMinEarRef.current = ear;
      } else if (blinkMinEarRef.current === null || ear < blinkMinEarRef.current) {
        blinkMinEarRef.current = ear;
      }
      eyeClosedFramesRef.current += 1;
      eyeWasClosedRef.current = true;
    } else {
      const blinkDepthOk = blinkMinEarRef.current !== null
        && blinkMinEarRef.current <= closeThreshold * BLINK_MIN_DEPTH_FACTOR;
      const blinkReopenOk = ear >= reopenThreshold;
      const startFaceWidth = blinkStartFaceWidthRef.current ?? faceWidth;
      const faceWidthVariation = Math.abs(faceWidth - startFaceWidth) / Math.max(startFaceWidth, 1e-4);
      const blinkStableEnough = faceWidthVariation <= BLINK_MAX_FACE_WIDTH_VARIATION_RATIO;

      if (
        eyeWasClosedRef.current
        && eyeClosedFramesRef.current >= BLINK_MIN_CLOSED_FRAMES
        && blinkDepthOk
        && blinkReopenOk
        && blinkStableEnough
      ) {
        blinkCountRef.current += 1;
      }
      eyeWasClosedRef.current = false;
      eyeClosedFramesRef.current = 0;
      blinkMinEarRef.current = null;
      blinkStartFaceWidthRef.current = null;
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
    } else if (
      !smileIntermediateReachedRef.current
      && smileScore <= smileBaselineRef.current * SMILE_BASELINE_UPDATE_MAX_RATIO
    ) {
      smileBaselineRef.current = (smileBaselineRef.current * 0.98) + (smileScore * 0.02);
    }

    if (mouthOpenBaselineRef.current === null) {
      mouthOpenBaselineRef.current = mouthOpenScore;
    } else if (
      !mouthOpenIntermediateReachedRef.current
      && mouthOpenScore <= mouthOpenBaselineRef.current * 1.03
    ) {
      mouthOpenBaselineRef.current = (mouthOpenBaselineRef.current * 0.98) + (mouthOpenScore * 0.02);
    }

    if (headPoseBaselineRef.current === null) {
      headPoseBaselineRef.current = { yaw: headYaw, pitch: headPitch };
      headYawMinRef.current = headYaw;
      headYawMaxRef.current = headYaw;
      headPitchMinRef.current = headPitch;
      headPitchMaxRef.current = headPitch;
    } else {
      headYawMinRef.current = Math.min(headYawMinRef.current, headYaw);
      headYawMaxRef.current = Math.max(headYawMaxRef.current, headYaw);
      headPitchMinRef.current = Math.min(headPitchMinRef.current, headPitch);
      headPitchMaxRef.current = Math.max(headPitchMaxRef.current, headPitch);
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
    if (ratio >= SMILE_INTERMEDIATE_FACTOR || absDelta >= SMILE_MIN_ABS_DELTA * 0.5) {
      smileIntermediateReachedRef.current = true;
    }
    smilePeakRatioRef.current = Math.max(smilePeakRatioRef.current, ratio);

    const smilePassedFrame = (
      smileIntermediateReachedRef.current
      && ratio >= smileTarget
      && absDelta >= SMILE_MIN_ABS_DELTA
      && smilePeakRatioRef.current >= smileTarget
    );
    if (smilePassedFrame) {
      smileConsecutiveRef.current += 1;
    } else {
      smileConsecutiveRef.current = 0;
    }
    const intermediateProgress = smileIntermediateReachedRef.current
      ? 0.35
      : Math.max(0, (ratio - 1) / Math.max(SMILE_INTERMEDIATE_FACTOR - 1, 1e-4)) * 0.35;
    const ratioProgress = Math.max(0, (ratio - 1) / Math.max(smileTarget - 1, 1e-4));
    const deltaProgress = Math.max(0, absDelta / SMILE_MIN_ABS_DELTA);
    const smileBaseProgress = Math.max(intermediateProgress, ratioProgress, deltaProgress);
    const smileHoldTargetFrames = currentChallenge.requirements?.includes('smile_hold')
      ? SMILE_HOLD_CONSECUTIVE_FRAMES_REQUIRED
      : SMILE_CONSECUTIVE_FRAMES_REQUIRED;
    const smileConsecutiveProgress = smileConsecutiveRef.current / Math.max(smileHoldTargetFrames, 1);
    const smileProgress = Math.max(
      0,
      Math.min(100, Math.min(smileBaseProgress, smileConsecutiveProgress) * 100),
    );
    const smilePassed = smileConsecutiveRef.current >= SMILE_CONSECUTIVE_FRAMES_REQUIRED;
    const smileHoldPassed = smileConsecutiveRef.current >= SMILE_HOLD_CONSECUTIVE_FRAMES_REQUIRED;

    const mouthOpenTarget = currentChallenge.targets?.mouth_open ?? MOUTH_OPEN_TARGET_FACTOR;
    const mouthOpenBaseline = mouthOpenBaselineRef.current || mouthOpenScore;
    const mouthOpenRatio = mouthOpenScore / Math.max(mouthOpenBaseline, 1e-4);
    const mouthOpenAbsDelta = mouthOpenScore - mouthOpenBaseline;
    if (
      mouthOpenRatio >= MOUTH_OPEN_INTERMEDIATE_FACTOR
      || mouthOpenAbsDelta >= MOUTH_OPEN_MIN_ABS_DELTA * 0.5
    ) {
      mouthOpenIntermediateReachedRef.current = true;
    }
    mouthOpenPeakRatioRef.current = Math.max(mouthOpenPeakRatioRef.current, mouthOpenRatio);

    const mouthOpenPassedFrame = (
      mouthOpenIntermediateReachedRef.current
      && mouthOpenRatio >= mouthOpenTarget
      && mouthOpenAbsDelta >= MOUTH_OPEN_MIN_ABS_DELTA
      && mouthOpenPeakRatioRef.current >= mouthOpenTarget
    );
    if (mouthOpenPassedFrame) {
      mouthOpenConsecutiveRef.current += 1;
    } else {
      mouthOpenConsecutiveRef.current = 0;
    }
    const mouthOpenIntermediateProgress = mouthOpenIntermediateReachedRef.current
      ? 0.35
      : Math.max(0, (mouthOpenRatio - 1) / Math.max(MOUTH_OPEN_INTERMEDIATE_FACTOR - 1, 1e-4)) * 0.35;
    const mouthOpenRatioProgress = Math.max(0, (mouthOpenRatio - 1) / Math.max(mouthOpenTarget - 1, 1e-4));
    const mouthOpenDeltaProgress = Math.max(0, mouthOpenAbsDelta / MOUTH_OPEN_MIN_ABS_DELTA);
    const mouthOpenBaseProgress = Math.max(
      mouthOpenIntermediateProgress,
      mouthOpenRatioProgress,
      mouthOpenDeltaProgress,
    );
    const mouthOpenConsecutiveProgress = (
      mouthOpenConsecutiveRef.current / Math.max(MOUTH_OPEN_CONSECUTIVE_FRAMES_REQUIRED, 1)
    );
    const mouthOpenProgress = Math.max(
      0,
      Math.min(100, Math.min(mouthOpenBaseProgress, mouthOpenConsecutiveProgress) * 100),
    );
    const mouthOpenPassed = mouthOpenConsecutiveRef.current >= MOUTH_OPEN_CONSECUTIVE_FRAMES_REQUIRED;

    const headBaseline = headPoseBaselineRef.current || { yaw: headYaw, pitch: headPitch };
    const headLeftTarget = currentChallenge.targets?.head_left ?? HEAD_TURN_TARGET_DELTA;
    const headRightTarget = currentChallenge.targets?.head_right ?? HEAD_TURN_TARGET_DELTA;
    const headUpTarget = currentChallenge.targets?.head_up ?? HEAD_PITCH_TARGET_DELTA;
    const headDownTarget = currentChallenge.targets?.head_down ?? HEAD_PITCH_TARGET_DELTA;
    const headLeftDelta = Math.max(0, headBaseline.yaw - headYawMinRef.current);
    const headRightDelta = Math.max(0, headYawMaxRef.current - headBaseline.yaw);
    const headUpDelta = Math.max(0, headBaseline.pitch - headPitchMinRef.current);
    const headDownDelta = Math.max(0, headPitchMaxRef.current - headBaseline.pitch);
    const headLeftProgress = Math.min(100, (headLeftDelta / Math.max(headLeftTarget, 1e-4)) * 100);
    const headRightProgress = Math.min(100, (headRightDelta / Math.max(headRightTarget, 1e-4)) * 100);
    const headUpProgress = Math.min(100, (headUpDelta / Math.max(headUpTarget, 1e-4)) * 100);
    const headDownProgress = Math.min(100, (headDownDelta / Math.max(headDownTarget, 1e-4)) * 100);
    const headLeftPassed = headLeftDelta >= headLeftTarget;
    const headRightPassed = headRightDelta >= headRightTarget;
    const headUpPassed = headUpDelta >= headUpTarget;
    const headDownPassed = headDownDelta >= headDownTarget;

    const progressByType = {
      blink: blinkProgress,
      distance: distanceProgress,
      smile: smileProgress,
      smile_hold: smileProgress,
      mouth_open: mouthOpenProgress,
      head_left: headLeftProgress,
      head_right: headRightProgress,
      head_up: headUpProgress,
      head_down: headDownProgress,
    };
    const progressWeightByType = {
      blink: MIXED_BLINK_PROGRESS_WEIGHT,
      distance: MIXED_DISTANCE_PROGRESS_WEIGHT,
      smile: MIXED_SMILE_PROGRESS_WEIGHT,
      smile_hold: MIXED_SMILE_PROGRESS_WEIGHT,
      mouth_open: MIXED_HEAD_POSE_PROGRESS_WEIGHT,
      head_left: MIXED_HEAD_POSE_PROGRESS_WEIGHT,
      head_right: MIXED_HEAD_POSE_PROGRESS_WEIGHT,
      head_up: MIXED_HEAD_POSE_PROGRESS_WEIGHT,
      head_down: MIXED_HEAD_POSE_PROGRESS_WEIGHT,
    };
    const passedByType = {
      blink: blinkPassed,
      distance: distancePassed,
      smile: smilePassed,
      smile_hold: smileHoldPassed,
      mouth_open: mouthOpenPassed,
      head_left: headLeftPassed,
      head_right: headRightPassed,
      head_up: headUpPassed,
      head_down: headDownPassed,
    };
    const required = Array.isArray(currentChallenge.requirements) ? currentChallenge.requirements : [];
    const requirementProgress = required.map((key) => ({
      value: progressByType[key] ?? 0,
      weight: required.length > 1 ? (progressWeightByType[key] ?? 1) : 1,
    }));
    const combinedProgress = requirementProgress.length
      ? requirementProgress.reduce((acc, item) => acc + (item.value * item.weight), 0)
        / requirementProgress.reduce((acc, item) => acc + item.weight, 0)
      : 0;

    const passed = required.length
      ? required.every((key) => Boolean(passedByType[key]))
      : false;

    const nextProgress = passed ? 100 : Math.max(0, Math.min(100, combinedProgress));
    setLivenessProgress(nextProgress);
    if (Math.abs(nextProgress - lastChallengeProgressRef.current) >= 1) {
      lastChallengeProgressRef.current = nextProgress;
      lastChallengeProgressAtRef.current = Date.now();
    }

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
      mouthOpenScore,
      headYaw,
      headPitch,
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
    lastDetectorActivityRef.current = 0;
    lastChallengeProgressRef.current = 0;
    lastChallengeProgressAtRef.current = 0;
    lastFaceSeenAtRef.current = 0;
    challengesRef.current = [];
    challengeIndexRef.current = 0;
    setBackendLivenessStatus('idle');

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
        setVideoStep('Falha temporaria no detector. Tentando recuperar...');
      } finally {
        detectorBusyRef.current = false;
      }
    }, 250);
  }, [stopFaceDetection]);

  useEffect(() => {
    if (mode !== 'video' || !cameraReady) {
      return undefined;
    }

    const watchdog = setInterval(() => {
      const now = Date.now();

      if (
        faceValidationMode === 'mediapipe'
        && livenessStatusRef.current === 'pending'
        && lastDetectorActivityRef.current
        && now - lastDetectorActivityRef.current > 6000
      ) {
        detectorBusyRef.current = false;
        setVideoStep('Detector sem resposta. Reiniciando monitoramento...');
        startFaceDetection();
        lastDetectorActivityRef.current = now;
      }

      if (
        faceValidationMode === 'mediapipe'
        && livenessStatusRef.current === 'pending'
        && lastFaceSeenAtRef.current
        && now - lastFaceSeenAtRef.current > 5000
      ) {
        setVideoStep('Face fora do quadro. Reposicione o rosto para continuar.');
        return;
      }

      if (
        livenessStatusRef.current === 'pending'
        && faceDetected
        && challengeIndexRef.current < challengesRef.current.length
        && lastChallengeProgressAtRef.current
        && now - lastChallengeProgressAtRef.current > 12000
      ) {
        resetChallengeInternals();
        applyChallengeUi(challengeIndexRef.current);
        setVideoStep('Desafio reiniciado por falta de progresso. Tente novamente.');
      }
    }, 2000);

    return () => clearInterval(watchdog);
  }, [
    applyChallengeUi,
    cameraReady,
    faceDetected,
    faceValidationMode,
    mode,
    resetChallengeInternals,
    startFaceDetection,
  ]);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setVideoStep('Solicitando permissao da camera...');

    try {
      const { stream, profile } = await openBestCameraStream();
      const videoTrack = stream.getVideoTracks?.()[0] || null;
      const cameraSettings = videoTrack?.getSettings?.() || {};

      streamRef.current = stream;
      setCameraReady(true);
      setFaceDetected(false);
      const streamSize = cameraSettings.width && cameraSettings.height
        ? ` (${cameraSettings.width}x${cameraSettings.height}, ${profile})`
        : '';
      setVideoStep(`Camera ativa${streamSize}. Preparando detector de face...`);

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
          try {
            lastDetectorActivityRef.current = Date.now();
            const hasFaces = Array.isArray(results?.multiFaceLandmarks) && results.multiFaceLandmarks.length > 0;
            setFaceDetected(hasFaces);

            if (hasFaces) {
              lastFaceSeenAtRef.current = Date.now();
              if (!livenessStartRef.current) {
                livenessStartRef.current = Date.now();
              }
              updateLivenessFromLandmarks(results.multiFaceLandmarks[0]);
            } else if (livenessStatusRef.current === 'pending') {
              setVideoStep('Face detectada anteriormente. Reposicione o rosto para continuar a prova de vida.');
            }
          } catch (error) {
            setCameraError(error instanceof Error ? error.message : 'Falha no detector de prova de vida.');
            setVideoStep('Falha no detector. Reinicie a camera se o problema persistir.');
          }
        });

        detectorRef.current = mpDetector;
        setFaceValidationMode('mediapipe');
        const serverSession = await createLivenessSession('mediapipe');
        resetLiveness('mediapipe', serverSession);
        setVideoStep('Sessao segura criada. Siga os desafios de prova de vida.');
      } catch (error) {
        detectorRef.current = null;
        setFaceValidationMode('none');
        setFaceDetected(true);
        resetLiveness('none');
        const message = error instanceof Error ? error.message : 'Erro desconhecido ao iniciar detector local';
        setCameraError(`Deteccao local indisponivel (${message}). Gravacao liberada sem pre-validacao.`);
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
        canvas.toBlob(resolve, 'image/jpeg', 0.98);
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

    const shouldRequireLiveness = (
      faceValidationMode !== 'none'
      && livenessStatus !== 'unavailable'
      && !skipLocalLiveness
    );
    if (shouldRequireLiveness && livenessStatus !== 'passed') {
      setCameraError('Conclua a prova de vida antes de gravar.');
      setVideoStep('Aguardando prova de vida');
      return;
    }

    setCameraError(null);

    try {
      setVideoStep('Capturando frames direto da camera...');
      setIsRecording(true);
      setBackendLivenessStatus('idle');
      setRecordingMsLeft(MAX_RECORDING_MS);
      recordingTickerRef.current = setInterval(() => {
        setRecordingMsLeft((prev) => Math.max(0, prev - 100));
      }, 100);

      const frames = await captureFramesFromLiveVideo(FRAME_COUNT, MAX_RECORDING_MS);
      if (!frames.length) {
        throw new Error('Nao foi possivel capturar frames da camera.');
      }

      const previewUrl = URL.createObjectURL(frames[0]);
      let backendLivenessResult = null;
      if (useBackendLiveness) {
        setVideoStep('Validando prova de vida no backend...');
        setBackendLivenessStatus('running');
        const sessionDefinition = buildBackendLivenessSession();
        const requiredChecks = Array.from(
          new Set(
            sessionDefinition.challenges.flatMap((challenge) => (
              Array.isArray(challenge.requirements) ? challenge.requirements : []
            )),
          ),
        );

        try {
          const clientContext = buildClientCaptureContext(streamRef.current);
          const livenessResult = await analyzeLiveness(frames, sessionDefinition, requiredChecks, clientContext);
          backendLivenessResult = livenessResult;
          console.debug('Backend liveness result', livenessResult);
          if (!livenessResult.passed) {
            setBackendLivenessStatus('failed');
            const failureDetail = getLivenessFailureDetail(livenessResult);
            const detail = failureDetail ? ` Motivo: ${failureDetail}.` : '';
            const message = `Backend de liveness reprovou esta captura.${detail}`;
            setCameraError(`${message} Refaça a prova de vida com os movimentos pedidos.`);
            setVideoStep('Prova de vida reprovada');
            if (onLivenessRejected) {
              onLivenessRejected(frames, previewUrl, livenessResult, message);
            }
            return;
          } else {
            setBackendLivenessStatus('passed');
          }
        } catch (error) {
          setBackendLivenessStatus('failed');
          const message = error instanceof Error ? error.message : 'Falha no backend de liveness';
          setCameraError(`${message}. Refaça a captura para validar a prova de vida.`);
          setVideoStep('Prova de vida reprovada');
          return;
        }
      }
      setVideoStep('Enviando frames para analise...');
      onVideoUpload(frames, targetModel, previewUrl, backendLivenessResult);
    } catch (error) {
      if (useBackendLiveness) {
        setBackendLivenessStatus('failed');
      }
      setCameraError(error.message || 'Falha na captura direta de frames.');
      setVideoStep('Falha ao capturar/enviar frames');
    } finally {
      setIsRecording(false);
      stopRecordingTicker();
    }
  }, [
    buildBackendLivenessSession,
    captureFramesFromLiveVideo,
    faceDetected,
    faceValidationMode,
    isRecording,
    livenessStatus,
    onVideoUpload,
    onLivenessRejected,
    stopRecordingTicker,
    targetModel,
    useBackendLiveness,
    skipLocalLiveness,
  ]);

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
    if (file && mode === 'audio' && (file.type.startsWith('audio/') || /\.(flac|wav|mp3|m4a|ogg)$/i.test(file.name))) {
      onAudioUpload(file, { useTemporalMemory, targetAudioModel });
    } else if (file && file.type.startsWith('image/')) {
      onImageUpload(file, targetModel);
    }
  }, [mode, onAudioUpload, onImageUpload, targetAudioModel, targetModel, useTemporalMemory]);

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0];
    if (file && mode === 'audio') {
      onAudioUpload(file, { useTemporalMemory, targetAudioModel });
    } else if (file) {
      onImageUpload(file, targetModel);
    }
  }, [mode, onAudioUpload, onImageUpload, targetAudioModel, targetModel, useTemporalMemory]);

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
        <button
          type="button"
          className={`mode-btn ${mode === 'audio' ? 'active' : ''}`}
          onClick={() => setMode('audio')}
        >
          <FileAudio size={16} />
          Audio
        </button>
      </div>

      {mode !== 'audio' && (
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
          <option value="mobilenet_v3_small_v1">MobileNetV3 Small v1</option>
          <option value="mobilenet">MobileNet (auto)</option>
          <option value="alexnet_v82">AlexNet v8.2</option>
          <option value="alexnet_v81">AlexNet v8.1</option>
          <option value="alexnet_v8">AlexNet v8</option>
          <option value="alexnet_v7">AlexNet v7</option>
          <option value="alexnet_v6">AlexNet v6</option>
          <option value="alexnet">AlexNet (auto)</option>
        </select>
      </div>
      )}

      {mode === 'audio' && (
        <div className="model-target-card">
          <label htmlFor="target-audio-model">Modelo de audio</label>
          <select
            id="target-audio-model"
            value={targetAudioModel}
            onChange={(e) => setTargetAudioModel(e.target.value)}
          >
            {AUDIO_MODEL_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <p className="toggle-help">
            Escolha qual detector neural sera usado antes da revisao simbolica.
          </p>
          <label htmlFor="temporal-memory-toggle">Memoria temporal simbolica</label>
          <div className="inline-toggle-row">
            <input
              id="temporal-memory-toggle"
              type="checkbox"
              checked={useTemporalMemory}
              onChange={(e) => setUseTemporalMemory(e.target.checked)}
            />
            <span title="Compara as janelas do proprio audio entre si para procurar quebras locais de comportamento neural, acustico e fonetico. Ajuda a destacar possiveis rupturas, mas nao e prova isolada de manipulacao.">
              {useTemporalMemory
                ? 'Ativada para procurar rupturas locais'
                : 'Desativada nesta analise'}
            </span>
          </div>
          <p className="toggle-help">
            Ao ativar, o sistema compara trechos do proprio audio para buscar mudancas locais que destoam do restante.
          </p>
        </div>
      )}

      {mode === 'video' && (
        <div className="model-target-card">
          <label htmlFor="backend-liveness-toggle">Prova de vida no backend</label>
          <div className="inline-toggle-row">
            <input
              id="backend-liveness-toggle"
              type="checkbox"
              checked={useBackendLiveness}
              onChange={(e) => {
                setUseBackendLiveness(e.target.checked);
                setBackendLivenessStatus('idle');
              }}
            />
            <span>
              {useBackendLiveness
                ? 'Ativada para esta analise'
                : 'Desativada para esta analise'}
            </span>
          </div>
          <p className="toggle-help">
            Quando ativada, os frames capturados tambem passam pela API de liveness antes da analise do modelo.
          </p>
          <label htmlFor="skip-local-liveness-toggle">Pular prova de vida local</label>
          <div className="inline-toggle-row">
            <input
              id="skip-local-liveness-toggle"
              type="checkbox"
              checked={skipLocalLiveness}
              onChange={(e) => {
                setSkipLocalLiveness(e.target.checked);
                setCameraError(null);
                setVideoStep(
                  e.target.checked
                    ? 'Prova de vida local ignorada para este teste.'
                    : 'Prova de vida local reativada.',
                );
              }}
            />
            <span>
              {skipLocalLiveness
                ? 'Ignorada para este teste'
                : 'Obrigatoria antes da gravacao'}
            </span>
          </div>
          <p className="toggle-help">
            Use isso apenas para testar o fluxo com webcam enquanto calibramos os desafios.
          </p>
        </div>
      )}

      {mode === 'audio' ? (
        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="upload-icon">
            <FileAudio size={48} />
          </div>
          <h2>Arraste um audio aqui</h2>
          <p>ou clique para selecionar</p>

          <input
            type="file"
            accept="audio/flac,audio/x-flac,audio/wav,audio/x-wav,audio/mpeg,audio/mp3,audio/mp4,audio/ogg,.flac,.wav,.mp3,.m4a,.ogg"
            onChange={handleFileSelect}
            className="upload-input"
          />

          <div className="upload-formats">
            <FileAudio size={16} />
            <span>FLAC, WAV, MP3, M4A ou OGG</span>
          </div>
        </div>
      ) : mode === 'image' ? (
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
                <div className={`liveness-box ${skipLocalLiveness ? 'skipped' : livenessStatus}`}>
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
                    {skipLocalLiveness
                      ? 'Ignorada manualmente para este teste'
                      : livenessStatus === 'passed'
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
                        || (
                          !skipLocalLiveness
                          && livenessStatus !== 'passed'
                          && livenessStatus !== 'unavailable'
                        )
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
            {skipLocalLiveness && cameraReady && (
              <p className="camera-note liveness-skip-note">
                Prova de vida local ignorada manualmente. A gravacao esta liberada para teste do modelo.
              </p>
            )}
            {faceValidationMode === 'none' && cameraReady && (
              <p className="camera-note">Deteccao local indisponivel. Validacao de face ocorrera no backend.</p>
            )}
            {useBackendLiveness && cameraReady && (
              <p className={`camera-note backend-liveness ${backendLivenessStatus}`}>
                {backendLivenessStatus === 'running'
                  ? 'Backend de liveness validando a captura.'
                  : backendLivenessStatus === 'passed'
                    ? 'Backend de liveness aprovou esta captura.'
                    : backendLivenessStatus === 'failed'
                      ? 'Backend de liveness reprovou esta captura.'
                      : backendLivenessStatus === 'warning'
                        ? 'Backend de liveness alertou um problema, mas a analise seguira.'
                      : 'Backend de liveness sera usado antes do envio da analise.'}
              </p>
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
                <span>Mouth open: {earDebug.mouthOpenScore !== null ? earDebug.mouthOpenScore.toFixed(3) : '--'}</span>
                <span>Head yaw: {earDebug.headYaw !== null ? earDebug.headYaw.toFixed(3) : '--'}</span>
                <span>Head pitch: {earDebug.headPitch !== null ? earDebug.headPitch.toFixed(3) : '--'}</span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="upload-info">
        <h3>Como funciona?</h3>
        <ol>
          <li>{mode === 'image' ? 'Envie uma selfie ou foto com rosto' : mode === 'audio' ? 'Envie uma amostra curta de voz' : 'Grave ate 5s com o rosto centralizado'}</li>
          <li>{mode === 'image' ? 'Nossa IA analisa a imagem em tempo real' : mode === 'audio' ? 'O modelo de audio calcula a probabilidade de deepfake' : 'Extraimos frames do video em memoria'}</li>
          <li>Receba o resultado com nivel de risco</li>
        </ol>
      </div>
    </div>
  );
}

export default Upload;
