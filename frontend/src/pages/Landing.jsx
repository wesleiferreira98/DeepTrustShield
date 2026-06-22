import { ArrowRight, Eye, ImageIcon, Mic, Shield, Video } from 'lucide-react'
import './Landing.css'

const FEATURES = [
  {
    icon: ImageIcon,
    color: 'var(--color-primary)',
    title: 'Análise de Imagem',
    desc: 'Detecta rostos gerados por IA com modelos EfficientNet B0, B3 e B4 treinados em conjuntos de dados curados.',
  },
  {
    icon: Mic,
    color: '#a855f7',
    title: 'Análise de Áudio',
    desc: 'Identifica clonagem de voz e síntese de fala em português com modelo neuro-simbólico especializado.',
  },
  {
    icon: Video,
    color: '#f59e0b',
    title: 'Análise de Vídeo',
    desc: 'Processa frames individualmente e agrega os resultados para detectar manipulações em vídeos curtos.',
  },
  {
    icon: Eye,
    color: '#10b981',
    title: 'Verificação de Vivacidade',
    desc: 'Confirma presença de pessoa real em tempo real por análise de landmarks faciais e anti-spoofing.',
  },
]

export default function Landing({ onLogin }) {
  return (
    <div className="landing">
      {/* Nav */}
      <header className="ln-nav">
        <div className="ln-nav-logo">
          <div className="ln-nav-mark">
            <Shield size={17} />
          </div>
          <span className="ln-nav-brand">DeepTrustShield</span>
        </div>
        <button className="ln-nav-btn" onClick={onLogin}>
          Entrar
        </button>
      </header>

      {/* Hero */}
      <section className="ln-hero">
        <span className="ln-badge">
          <Shield size={11} />
          Powered by Deep Learning
        </span>
        <h1 className="ln-h1">
          Detecte Deepfakes<br />com Precisão Cirúrgica
        </h1>
        <p className="ln-subtitle">
          Plataforma de análise de mídia com inteligência artificial para identificar
          conteúdo sintético em imagens, áudio e vídeo — em segundos.
        </p>
        <button className="ln-cta" onClick={onLogin}>
          Acessar o Sistema
          <ArrowRight size={16} />
        </button>
      </section>

      {/* Features */}
      <section className="ln-features">
        <h2 className="ln-features-title">Quatro modalidades de análise</h2>
        <div className="ln-grid">
          {FEATURES.map(({ icon: Icon, color, title, desc }) => (
            <div key={title} className="ln-card">
              <div className="ln-card-icon" style={{ background: `color-mix(in srgb, ${color} 15%, transparent)` }}>
                <Icon size={20} color={color} />
              </div>
              <h3 className="ln-card-title">{title}</h3>
              <p className="ln-card-desc">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="ln-footer">
        © {new Date().getFullYear()} DeepTrustShield · Tecnologia de detecção de deepfake
      </footer>
    </div>
  )
}
