import { useState, useCallback } from 'react'
import Upload from './components/Upload'
import AnalysisPanel from './components/AnalysisPanel'
import Result from './components/Result'
import { analyzeImage, pollStatus, getResult } from './services/api'
import './App.css'

function App() {
  const [state, setState] = useState('idle') // idle | uploading | analyzing | done | error
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [error, setError] = useState(null)

  const handleUpload = useCallback(async (file, targetModel = 'auto') => {
    // Reset state
    setError(null)
    setResult(null)
    setStatus(null)
    
    // Create preview URL
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    
    // Start upload
    setState('uploading')
    
    try {
      // Send to API
      const response = await analyzeImage(file, targetModel)
      setJobId(response.job_id)
      setState('analyzing')
      
      // Start polling
      const pollInterval = setInterval(async () => {
        try {
          const statusData = await pollStatus(response.job_id)
          setStatus(statusData)
          
          if (statusData.stage === 'done' || statusData.stage === 'error') {
            clearInterval(pollInterval)
            
            // Get final result
            const resultData = await getResult(response.job_id)
            setResult(resultData)
            setState(statusData.stage === 'done' ? 'done' : 'error')
          }
        } catch (err) {
          console.error('Polling error:', err)
        }
      }, 700)
      
    } catch (err) {
      console.error('Upload error:', err)
      setError(err.message || 'Erro ao enviar imagem')
      setState('error')
    }
  }, [])

  const handleReset = useCallback(() => {
    setState('idle')
    setJobId(null)
    setStatus(null)
    setResult(null)
    setPreviewUrl(null)
    setError(null)
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
          <Upload onUpload={handleUpload} />
        )}

        {(state === 'uploading' || state === 'analyzing') && (
          <AnalysisPanel 
            status={status} 
            previewUrl={previewUrl}
            isUploading={state === 'uploading'}
          />
        )}

        {(state === 'done' || state === 'error') && result && (
          <Result 
            result={result} 
            previewUrl={previewUrl}
            status={status}
            onReset={handleReset}
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
