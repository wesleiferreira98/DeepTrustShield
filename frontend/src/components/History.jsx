import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronRight, Clock, Cpu, Filter, ImageIcon, Mic, RotateCcw, Search, Video } from 'lucide-react'
import Result from './Result'
import { getHistory, getResult } from '../services/api'
import './History.css'

const MODE_LABEL = { image: 'Imagem', audio: 'Áudio', video: 'Vídeo' }
const MODE_ICONS = { image: ImageIcon, audio: Mic, video: Video }

function ModeIcon({ mode, size = 15 }) {
  const Icon = MODE_ICONS[mode] || Video
  return <Icon size={size} />
}
const RISK_LABEL = { HIGH: 'Fake detectado', MEDIUM: 'Suspeito', LOW: 'Autêntico', UNDETERMINED: 'Indeterminado' }
const RISK_SHORT = { HIGH: 'Alto', MEDIUM: 'Médio', LOW: 'Baixo', UNDETERMINED: 'N/D' }
const RISK_CLASS = { HIGH: 'risk-high', MEDIUM: 'risk-medium', LOW: 'risk-low', UNDETERMINED: 'risk-nd' }

const PERIOD_OPTIONS = [
  { value: 0,  label: 'Todos' },
  { value: 1,  label: 'Hoje' },
  { value: 7,  label: '7 dias' },
  { value: 30, label: '30 dias' },
]

const PAGE_SIZE = 15

/* ── Score circle SVG ── */
function ScoreCircle({ value, riskClass }) {
  if (value == null) return <span className="score-nd">N/D</span>
  const r   = 22
  const circ = 2 * Math.PI * r
  const pct  = Math.round(value * 100)
  const dash = circ * (1 - value)
  return (
    <div className={`score-ring ${riskClass}`}>
      <svg width="60" height="60" viewBox="0 0 60 60">
        <circle cx="30" cy="30" r={r} className="ring-track" strokeWidth="5" fill="none" />
        <circle
          cx="30" cy="30" r={r}
          className="ring-fill"
          strokeWidth="5"
          fill="none"
          strokeDasharray={circ}
          strokeDashoffset={dash}
          strokeLinecap="round"
          transform="rotate(-90 30 30)"
        />
      </svg>
      <span className="ring-label">{pct}%</span>
    </div>
  )
}

/* ── Skeleton card ── */
function SkeletonCard() {
  return (
    <div className="hcard hcard-skeleton" aria-hidden="true">
      <div className="hcard-accent" />
      <div className="sk-icon" />
      <div className="hcard-body">
        <div className="sk-line sk-lg" />
        <div className="sk-line sk-sm" />
      </div>
      <div className="sk-circle" />
    </div>
  )
}

/* ── Empty state ── */
function EmptyState({ hasFilters, onClear }) {
  return (
    <div className="hist-empty">
      <div className="hist-empty-icon"><Search size={40} strokeWidth={1.5} /></div>
      <h3>{hasFilters ? 'Nenhum resultado para os filtros aplicados' : 'Nenhuma análise encontrada'}</h3>
      <p>
        {hasFilters
          ? 'Tente remover os filtros ou ampliar o período de busca.'
          : 'As análises concluídas aparecerão aqui automaticamente.'}
      </p>
      {hasFilters && (
        <button className="hist-clear-btn" onClick={onClear}>
          Limpar filtros
        </button>
      )}
    </div>
  )
}

/* ── History card ── */
function HistoryCard({ item, index, onClick }) {
  const riskClass = RISK_CLASS[item.risk_level] || 'risk-nd'
  const dateStr = item.created_at
    ? new Date(item.created_at).toLocaleString('pt-BR', {
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      })
    : '—'
  const procStr = item.processing_time_ms
    ? item.processing_time_ms >= 1000
      ? `${(item.processing_time_ms / 1000).toFixed(1)}s`
      : `${item.processing_time_ms}ms`
    : null

  return (
    <button
      className={`hcard ${riskClass}`}
      style={{ animationDelay: `${index * 40}ms` }}
      onClick={onClick}
      type="button"
    >
      <div className="hcard-accent" aria-hidden="true" />

      {/* Mode icon */}
      <div className="hcard-mode">
        <span className="hcard-icon" aria-label={MODE_LABEL[item.mode]}>
          <ModeIcon mode={item.mode} size={20} />
        </span>
        <span className="hcard-mode-label">{MODE_LABEL[item.mode] || item.mode}</span>
      </div>

      {/* Main info */}
      <div className="hcard-body">
        <div className="hcard-risk-label">{RISK_LABEL[item.risk_level]}</div>
        <div className="hcard-meta">
          {item.model_version && (
            <span className="hcard-meta-item">
              <Cpu size={11} />
              {item.model_version}
            </span>
          )}
          {procStr && (
            <span className="hcard-meta-item">
              <Clock size={11} />
              {procStr}
            </span>
          )}
          <span className="hcard-meta-item hcard-date">
            {dateStr}
          </span>
        </div>
      </div>

      {/* Score ring */}
      <ScoreCircle value={item.p_fake} riskClass={riskClass} />

      {/* Risk chip + chevron */}
      <div className="hcard-right">
        <span className={`hchip ${riskClass}`}>{RISK_SHORT[item.risk_level]}</span>
        <ChevronRight size={16} className="hcard-chevron" />
      </div>
    </button>
  )
}

/* ── Detail overlay (reuses Result component) ── */
function HistoryDetail({ item, onBack }) {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  const previewUrl = item.mode === 'image'
    ? `/artifacts/${item.job_id}/face_crop.jpg`
    : null

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    getResult(item.job_id, item.mode)
      .then((r) => { if (!cancelled) { setResult(r); setLoading(false) } })
      .catch((e) => { if (!cancelled) { setError(e.message); setLoading(false) } })
    return () => { cancelled = true }
  }, [item.job_id, item.mode])

  return (
    <div className="hist-detail">
      <div className="hist-detail-bar">
        <button className="hist-back-btn" onClick={onBack}>
          ← Voltar ao histórico
        </button>
        <div className="hist-detail-meta">
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
            <ModeIcon mode={item.mode} size={13} /> {MODE_LABEL[item.mode]}
          </span>
          <span className={`hchip ${RISK_CLASS[item.risk_level]}`}>
            {RISK_SHORT[item.risk_level]}
          </span>
          {item.created_at && (
            <span className="hist-detail-date">
              {new Date(item.created_at).toLocaleString('pt-BR', {
                day: '2-digit', month: '2-digit', year: 'numeric',
                hour: '2-digit', minute: '2-digit',
              })}
            </span>
          )}
        </div>
      </div>

      {loading && (
        <div className="hist-detail-loading">
          <div className="hist-spin" />
          <span>Carregando resultado...</span>
        </div>
      )}

      {error && (
        <div className="hist-detail-error">
          <p>Erro ao carregar: {error}</p>
          <button className="hist-clear-btn" onClick={onBack}>Voltar</button>
        </div>
      )}

      {result && !loading && (
        <div className="hist-result-wrap">
          <Result
            result={result}
            previewUrl={previewUrl}
            analysisMode={item.mode}
            onReset={onBack}
            onReanalyze={() => {}}
            comparisonResults={[]}
            reanalysisLoading={false}
            reanalysisError={null}
          />
        </div>
      )}
    </div>
  )
}

/* ── Main History component ── */
export default function History() {
  const [items,      setItems]      = useState([])
  const [total,      setTotal]      = useState(0)
  const [loading,    setLoading]    = useState(true)
  const [loadingMore,setLoadingMore]= useState(false)
  const [error,      setError]      = useState(null)
  const [modeFilter, setModeFilter] = useState('')
  const [days,       setDays]       = useState(0)
  const [selected,   setSelected]   = useState(null)
  const offsetRef = useRef(0)

  const hasFilters = modeFilter !== '' || days !== 0

  const fetchPage = useCallback(async (reset = false) => {
    const off = reset ? 0 : offsetRef.current
    if (reset) {
      setLoading(true)
      setItems([])
    } else {
      setLoadingMore(true)
    }
    setError(null)

    try {
      const data = await getHistory({ mode: modeFilter || undefined, days, limit: PAGE_SIZE, offset: off })
      offsetRef.current = off + data.items.length
      setTotal(data.total)
      setItems((prev) => reset ? data.items : [...prev, ...data.items])
    } catch (e) {
      setError(e.message || 'Erro ao carregar histórico')
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }, [modeFilter, days])

  useEffect(() => {
    offsetRef.current = 0
    fetchPage(true)
  }, [fetchPage])

  const clearFilters = useCallback(() => {
    setModeFilter('')
    setDays(0)
  }, [])

  const hasMore = items.length < total

  /* Detail view */
  if (selected) {
    return (
      <div className="history">
        <HistoryDetail item={selected} onBack={() => setSelected(null)} />
      </div>
    )
  }

  return (
    <div className="history">

      {/* Header */}
      <div className="hist-topbar">
        <div>
          <h2 className="hist-heading">Histórico de Análises</h2>
          <p className="hist-sub">
            {total > 0
              ? `${total.toLocaleString('pt-BR')} análise${total !== 1 ? 's' : ''} concluída${total !== 1 ? 's' : ''}`
              : 'Análises concluídas ficam salvas aqui'}
          </p>
        </div>
        <button className="hist-refresh" onClick={() => fetchPage(true)} title="Atualizar">
          <RotateCcw size={14} />
          Atualizar
        </button>
      </div>

      {/* Filters */}
      <div className="hist-filters">
        <div className="hist-filter-group">
          <Filter size={13} className="hist-filter-icon" />
          {['', 'image', 'audio', 'video'].map((m) => (
            <button
              key={m}
              className={`hist-pill ${modeFilter === m ? 'active' : ''}`}
              onClick={() => setModeFilter(m)}
            >
              {m === '' ? 'Todos' : <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}><ModeIcon mode={m} size={12} />{MODE_LABEL[m]}</span>}
            </button>
          ))}
        </div>
        <div className="hist-filter-group hist-period">
          {PERIOD_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              className={`hist-pill ${days === opt.value ? 'active' : ''}`}
              onClick={() => setDays(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div className="hcard-list">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      ) : error ? (
        <div className="hist-error">
          <p>{error}</p>
          <button className="hist-clear-btn" onClick={() => fetchPage(true)}>Tentar novamente</button>
        </div>
      ) : items.length === 0 ? (
        <EmptyState hasFilters={hasFilters} onClear={clearFilters} />
      ) : (
        <>
          <div className="hcard-list">
            {items.map((item, i) => (
              <HistoryCard
                key={item.job_id}
                item={item}
                index={i}
                onClick={() => setSelected(item)}
              />
            ))}
          </div>

          {hasMore && (
            <div className="hist-load-more">
              <button
                className="hist-more-btn"
                onClick={() => fetchPage(false)}
                disabled={loadingMore}
              >
                {loadingMore ? (
                  <><span className="hist-spin-sm" /> Carregando...</>
                ) : (
                  <>Ver mais <span className="hist-more-count">({total - items.length} restantes)</span></>
                )}
              </button>
            </div>
          )}

          {!hasMore && items.length > 0 && (
            <p className="hist-end-msg">
              <Search size={13} /> Todos os {total} registros exibidos
            </p>
          )}
        </>
      )}
    </div>
  )
}
