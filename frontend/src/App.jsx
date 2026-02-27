import { useState, useCallback } from 'react'
import Upload from './components/Upload'
import AnalysisPanel from './components/AnalysisPanel'
import Result from './components/Result'
import { analyzeImage, analyzeVideo, pollStatus, getResult } from './services/api'
import './App.css'

const POLL_INTERVAL_MS = 700

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function App() {
  const [state, setState] = useState('idle') // idle | uploading | analyzing | done | error
  const [jobId, setJobId] = useState(null)
  const [analysisMode, setAnalysisMode] = useState('image')
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [error, setError] = useState(null)

  const [sourceImageFile, setSourceImageFile] = useState(null)
  const [sourceVideoFrames, setSourceVideoFrames] = useState([])

  const [comparisonResults, setComparisonResults] = useState([])
  const [reanalysisLoading, setReanalysisLoading] = useState(false)
  const [reanalysisError, setReanalysisError] = useState(null)

  const upsertComparison = useCallback((newResult, requestedModel = 'auto') => {
    const entry = {
      requested_model: requestedModel,
      model_version: newResult.model_version,
      p_fake: newResult.p_fake,
      risk_level: newResult.risk_level,
      processing_time_ms: newResult.processing_time_ms,
      created_at: newResult.created_at,
    }

    setComparisonResults((prev) => {
      const filtered = prev.filter((item) => item.model_version !== entry.model_version)
      return [...filtered, entry]
    })
  }, [])

  const runAnalysisJob = useCallback(async (mode, targetModel, payload, isReanalysis = false) => {
    setError(null)
    setReanalysisError(null)

    if (!isReanalysis) {
      setResult(null)
      setStatus(null)
      setComparisonResults([])
      setState('uploading')
    } else {
      setReanalysisLoading(true)
    }

    try {
      const response = mode === 'video'
        ? await analyzeVideo(payload, targetModel)
        : await analyzeImage(payload, targetModel)

      setJobId(response.job_id)
      if (!isReanalysis) {
        setState('analyzing')
      }

      let finalStage = 'done'
      while (true) {
        const statusData = await pollStatus(response.job_id, mode)
        setStatus(statusData)

        if (statusData.stage === 'done' || statusData.stage === 'error') {
          finalStage = statusData.stage
          break
        }

        await sleep(POLL_INTERVAL_MS)
      }

      const resultData = await getResult(response.job_id, mode)
      setResult(resultData)
      upsertComparison(resultData, targetModel)

      if (!isReanalysis) {
        setState(finalStage === 'error' ? 'error' : 'done')
      }
    } catch (err) {
      if (isReanalysis) {
        setReanalysisError(err.message || 'Erro ao reanalisar com o modelo selecionado')
      } else {
        setError(err.message || 'Erro ao enviar para análise')
        setState('error')
      }
    } finally {
      if (isReanalysis) {
        setReanalysisLoading(false)
      }
    }
  }, [upsertComparison])

  const handleImageUpload = useCallback(async (file, targetModel = 'auto') => {
    setAnalysisMode('image')
    setSourceImageFile(file)
    setSourceVideoFrames([])

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)

    await runAnalysisJob('image', targetModel, file, false)
  }, [runAnalysisJob])

  const handleVideoUpload = useCallback(async (frames, targetModel = 'auto', preview = null) => {
    setAnalysisMode('video')
    setSourceImageFile(null)
    setSourceVideoFrames(frames)

    if (preview) {
      setPreviewUrl(preview)
    } else if (frames.length > 0) {
      setPreviewUrl(URL.createObjectURL(frames[0]))
    } else {
      setPreviewUrl(null)
    }

    await runAnalysisJob('video', targetModel, frames, false)
  }, [runAnalysisJob])

  const handleReanalyze = useCallback(async (targetModel) => {
    if (analysisMode === 'image' && sourceImageFile) {
      await runAnalysisJob('image', targetModel, sourceImageFile, true)
      return
    }

    if (analysisMode === 'video' && sourceVideoFrames.length > 0) {
      await runAnalysisJob('video', targetModel, sourceVideoFrames, true)
      return
    }

    setReanalysisError('Nao encontramos o arquivo original para reanalise. Faça uma nova análise.')
  }, [analysisMode, runAnalysisJob, sourceImageFile, sourceVideoFrames])

  const handleReset = useCallback(() => {
    setState('idle')
    setJobId(null)
    setAnalysisMode('image')
    setStatus(null)
    setResult(null)
    setPreviewUrl(null)
    setError(null)
    setSourceImageFile(null)
    setSourceVideoFrames([])
    setComparisonResults([])
    setReanalysisLoading(false)
    setReanalysisError(null)
  }, [])

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">�️</span>
          <span className="logo-text">DeepTrustShield</span>
        </div>
        <p className="tagline">Proteção contra Deepfakes com IA</p>
      </header>

      <main className="main">
        {state === 'idle' && (
          <Upload onImageUpload={handleImageUpload} onVideoUpload={handleVideoUpload} />
        )}

        {(state === 'uploading' || state === 'analyzing') && (
          <AnalysisPanel
            status={status}
            previewUrl={previewUrl}
            isUploading={state === 'uploading'}
            analysisMode={analysisMode}
          />
        )}

        {(state === 'done' || state === 'error') && result && (
          <Result
            result={result}
            previewUrl={previewUrl}
            status={status}
            analysisMode={analysisMode}
            onReset={handleReset}
            onReanalyze={handleReanalyze}
            comparisonResults={comparisonResults}
            reanalysisLoading={reanalysisLoading}
            reanalysisError={reanalysisError}
          />
        )}

        {state === 'error' && !result && (
          <div className="error-card">
            <h2>❌ Erro</h2>
            <p>{error || 'Ocorreu um erro inesperado'}</p>
            <button onClick={handleReset} className="btn-primary">
              Tentar novamente
            </button>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>MVP v1.0 • EfficientNet-B0</p>
      </footer>
    </div>
  )
}

export default App
