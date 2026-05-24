import { useEffect, useState, useCallback, useRef } from 'react'
import { Check, Eye, EyeOff, GripVertical, ImageIcon, Mic, Pencil, RotateCcw, Video } from 'lucide-react'
import { getStats, getModelStats } from '../services/api'
import './Dashboard.css'

/* ─────────────── constants ─────────────── */
const MODE_LABEL = { image: 'Imagem', audio: 'Áudio', video: 'Vídeo' }
const MODE_ICONS = { image: ImageIcon, audio: Mic, video: Video }
const RISK_LABEL = { HIGH: 'Alto', MEDIUM: 'Médio', LOW: 'Baixo', UNDETERMINED: 'Indeterminado' }
const RISK_CLASS = { HIGH: 'risk-high', MEDIUM: 'risk-medium', LOW: 'risk-low', UNDETERMINED: 'risk-nd' }
const RISK_COLOR = {
  HIGH: 'var(--color-danger)', MEDIUM: 'var(--color-warning)',
  LOW: 'var(--color-success)', UNDETERMINED: 'var(--color-text-secondary)',
}

const SECTION_META = {
  kpis:   { label: 'KPIs Gerais' },
  midrow: { label: 'Modalidade & Tendência' },
  models: { label: 'Desempenho por Modelo' },
  roc:    { label: 'Curvas ROC por Modelo' },
  recent: { label: 'Análises Recentes' },
}
const SECTION_IDS = Object.keys(SECTION_META)
const DEFAULT_LAYOUT = SECTION_IDS.map((id, i) => ({ id, order: i, visible: true }))
const LAYOUT_KEY = 'dts-dashboard-layout'

function loadLayout() {
  try {
    const raw = localStorage.getItem(LAYOUT_KEY)
    if (raw) {
      const saved = JSON.parse(raw)
      const savedIds = new Set(saved.map((s) => s.id))
      const merged = [
        ...saved,
        ...DEFAULT_LAYOUT
          .filter((s) => !savedIds.has(s.id))
          .map((s, i) => ({ ...s, order: saved.length + i })),
      ]
      return merged.sort((a, b) => a.order - b.order)
    }
  } catch {}
  return DEFAULT_LAYOUT
}

function saveLayout(layout) {
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout))
}

/* ─────────────── tiny helpers ─────────────── */
function ModeIcon({ mode, size = 15 }) {
  const Icon = MODE_ICONS[mode] || Video
  return <Icon size={size} />
}

/* ─────────────── data viz components ─────────────── */
function KpiCard({ label, value, sub, variant }) {
  return (
    <div className={`kpi-card ${variant || ''}`}>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

function ModeBreakdown({ byMode, total }) {
  return (
    <div className="mode-breakdown">
      {['image', 'audio', 'video'].map((m) => {
        const count = byMode[m] || 0
        const pct   = total > 0 ? Math.round((count / total) * 100) : 0
        return (
          <div key={m} className="mode-row">
            <span className="mode-row-icon"><ModeIcon mode={m} size={14} /></span>
            <span className="mode-row-label">{MODE_LABEL[m]}</span>
            <div className="mode-row-bar">
              <div className="mode-row-fill" style={{ width: `${pct}%` }} />
            </div>
            <span className="mode-row-count">{count.toLocaleString('pt-BR')}</span>
            <span className="mode-row-pct">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

function TrendChart({ trend }) {
  if (!trend || trend.length === 0) return <p className="dash-empty">Sem dados nos últimos 14 dias.</p>
  const W = 600, H = 110
  const PAD = { top: 8, right: 8, bottom: 28, left: 28 }
  const cW = W - PAD.left - PAD.right
  const cH = H - PAD.top  - PAD.bottom
  const max   = Math.max(...trend.map((d) => d.count), 1)
  const barW  = Math.max(4, Math.floor(cW / trend.length) - 4)
  const groupW = cW / trend.length
  const toY   = (v) => cH - (v / max) * cH
  const yLines = [0.25, 0.5, 0.75, 1].map((f) => Math.round(f * max))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="trend-svg">
      <g transform={`translate(${PAD.left},${PAD.top})`}>
        {yLines.map((v) => <line key={v} x1={0} y1={toY(v)} x2={cW} y2={toY(v)} className="trend-grid" />)}
        {yLines.map((v) => <text key={v} x={-4} y={toY(v)+4} className="trend-axis-label" textAnchor="end">{v}</text>)}
        {trend.map((day, i) => {
          const cx = i * groupW + groupW / 2
          const rv = day.count - day.fake_count
          const h  = barW / 2
          return (
            <g key={day.date}>
              {rv > 0 && <rect x={cx-h} y={toY(rv)} width={barW} height={cH-toY(rv)} rx={2} className="trend-bar-real"><title>Real: {rv}</title></rect>}
              {day.fake_count > 0 && <rect x={cx-h} y={toY(day.count)} width={barW} height={toY(rv)-toY(day.count)} rx={2} className="trend-bar-fake"><title>Fake: {day.fake_count}</title></rect>}
              <text x={cx} y={cH+18} className="trend-axis-label" textAnchor="middle">{day.date.slice(5)}</text>
            </g>
          )
        })}
      </g>
    </svg>
  )
}

function ScoreBar({ value }) {
  const pct = Math.round(value * 100)
  const cls = value >= 0.65 ? 'danger' : value >= 0.35 ? 'warning' : 'success'
  return (
    <div className="score-inline">
      <span className="score-inline-num">{pct}%</span>
      <div className="score-inline-track">
        <div className={`score-inline-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function RecentTable({ recent }) {
  if (!recent || recent.length === 0) return <p className="dash-empty">Nenhuma análise concluída ainda.</p>
  return (
    <div className="dash-table-wrap">
      <table className="dash-table">
        <thead><tr><th>Modalidade</th><th>Risco</th><th>Score</th><th>Modelo</th><th>Data</th></tr></thead>
        <tbody>
          {recent.map((row) => (
            <tr key={row.job_id}>
              <td><span className="mode-chip"><ModeIcon mode={row.mode} size={13} />{MODE_LABEL[row.mode]||row.mode}</span></td>
              <td><span className={`risk-chip ${RISK_CLASS[row.risk_level]||''}`}>{RISK_LABEL[row.risk_level]||row.risk_level}</span></td>
              <td className="td-score">{row.p_fake!=null?<ScoreBar value={row.p_fake}/>:<span className="td-muted">—</span>}</td>
              <td className="td-model">{row.model_version||'—'}</td>
              <td className="td-date">{row.created_at?new Date(row.created_at).toLocaleString('pt-BR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}):'—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RiskDistBar({ distribution }) {
  return (
    <div className="mdl-dist-wrap">
      {['HIGH','MEDIUM','LOW','UNDETERMINED'].map((k) => {
        const pct = distribution[k] || 0
        if (!pct) return null
        return (
          <div key={k} className="mdl-dist-row">
            <span className={`mdl-dist-label ${RISK_CLASS[k]}`}>{RISK_LABEL[k]}</span>
            <div className="mdl-dist-track"><div className="mdl-dist-fill" style={{width:`${pct}%`,background:RISK_COLOR[k]}}/></div>
            <span className="mdl-dist-pct">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}

function ModelCard({ model }) {
  const det  = Math.round(model.detection_rate * 100)
  const conf = model.avg_p_fake != null ? Math.round(model.avg_p_fake * 100) : null
  const proc = model.avg_processing_ms != null
    ? model.avg_processing_ms >= 1000 ? `${(model.avg_processing_ms/1000).toFixed(1)}s` : `${model.avg_processing_ms}ms`
    : '—'
  const last = model.last_used ? new Date(model.last_used).toLocaleDateString('pt-BR',{day:'2-digit',month:'2-digit',year:'numeric'}) : '—'
  return (
    <div className="mdl-card">
      <div className="mdl-card-header">
        <div className="mdl-name-row"><span className="mdl-icon"><ModeIcon mode={model.mode} size={14}/></span><span className="mdl-name" title={model.model_version}>{model.model_version}</span></div>
        <span className="mdl-mode-badge">{MODE_LABEL[model.mode]||model.mode}</span>
      </div>
      <div className="mdl-kpi-row">
        <div className="mdl-kpi"><span className="mdl-kpi-val">{model.total.toLocaleString('pt-BR')}</span><span className="mdl-kpi-label">Análises</span></div>
        <div className="mdl-kpi"><span className="mdl-kpi-val mdl-kpi-danger">{det}%</span><span className="mdl-kpi-label">Detecção</span></div>
        <div className="mdl-kpi"><span className="mdl-kpi-val">{conf!=null?`${conf}%`:'—'}</span><span className="mdl-kpi-label">Confiança</span></div>
        <div className="mdl-kpi"><span className="mdl-kpi-val">{proc}</span><span className="mdl-kpi-label">Tempo médio</span></div>
      </div>
      <div className="mdl-section-title">Distribuição de risco</div>
      <RiskDistBar distribution={model.risk_distribution}/>
      <div className="mdl-footer"><span>Último uso: {last}</span><span>{model.total} análise{model.total!==1?'s':''}</span></div>
    </div>
  )
}

function ScoreDistChart({ distribution }) {
  if (!distribution || distribution.every((b) => b.count === 0)) return <p className="roc-empty">Sem dados de p_fake para exibir.</p>
  const W = 500, H = 160
  const PAD = { top: 12, right: 12, bottom: 34, left: 40 }
  const cW = W-PAD.left-PAD.right, cH = H-PAD.top-PAD.bottom
  const max  = Math.max(...distribution.map((b) => b.count), 1)
  const slot = cW / 10
  const color = (i) => i<=2?'var(--color-success)':i<=4?'var(--color-info,var(--color-primary))':i<=6?'var(--color-warning)':'var(--color-danger)'
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="roc-svg-large">
      <g transform={`translate(${PAD.left},${PAD.top})`}>
        {[0.25,0.5,0.75,1].map((f)=>{const y=cH-f*cH;return(<g key={f}><line x1={0} y1={y} x2={cW} y2={y} className="roc-grid"/><text x={-4} y={y+4} className="roc-tick" textAnchor="end">{Math.round(f*max)}</text></g>)})}
        {distribution.map((b,i)=>{
          const bH=(b.count/max)*cH, x=i*slot+2, w=slot-4, y=cH-bH
          return(<g key={i}>
            <rect x={x} y={y} width={w} height={bH} rx={2} fill={color(i)} opacity="0.8"><title>{b.label}: {b.count}</title></rect>
            <text x={x+w/2} y={cH+14} className="roc-tick" textAnchor="middle" fontSize={8}>{(i/10).toFixed(1)}</text>
            {b.count>0&&<text x={x+w/2} y={y-3} className="roc-tick" textAnchor="middle" fontSize={8}>{b.count}</text>}
          </g>)
        })}
        <text x={cW/2} y={cH+28} className="roc-axis-title" textAnchor="middle">Score p_fake</text>
        <text x={-cH/2} y={-28} className="roc-axis-title" textAnchor="middle" transform="rotate(-90)">Análises</text>
        <line x1={0} y1={0} x2={0} y2={cH} className="roc-axis-line"/>
        <line x1={0} y1={cH} x2={cW} y2={cH} className="roc-axis-line"/>
      </g>
    </svg>
  )
}

function RocSvg({ points, aucColor, gradientId }) {
  const W=500,H=300,PAD={top:16,right:16,bottom:46,left:50}
  const cW=W-PAD.left-PAD.right,cH=H-PAD.top-PAD.bottom
  const toX=(f)=>f*cW, toY=(t)=>cH-t*cH
  const pathD=points.map((p,i)=>`${i===0?'M':'L'} ${toX(p.fpr).toFixed(2)} ${toY(p.tpr).toFixed(2)}`).join(' ')
  const fillD=`${pathD} L ${toX(1).toFixed(2)} ${toY(0).toFixed(2)} L ${toX(0).toFixed(2)} ${toY(0).toFixed(2)} Z`
  const ticks=[0,0.25,0.5,0.75,1]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="roc-svg-large">
      <defs><linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={aucColor} stopOpacity="0.20"/><stop offset="100%" stopColor={aucColor} stopOpacity="0.02"/></linearGradient></defs>
      <g transform={`translate(${PAD.left},${PAD.top})`}>
        {ticks.map((v)=>(<g key={v}><line x1={0} y1={toY(v)} x2={cW} y2={toY(v)} className="roc-grid"/><line x1={toX(v)} y1={0} x2={toX(v)} y2={cH} className="roc-grid"/></g>))}
        {ticks.map((v)=>(<g key={`t${v}`}><text x={toX(v)} y={cH+16} className="roc-tick" textAnchor="middle">{v.toFixed(2)}</text><text x={-8} y={toY(v)+4} className="roc-tick" textAnchor="end">{v.toFixed(2)}</text></g>))}
        <text x={cW/2} y={cH+38} className="roc-axis-title" textAnchor="middle">Taxa de Falso Positivo (FPR)</text>
        <text x={-cH/2} y={-38} className="roc-axis-title" textAnchor="middle" transform="rotate(-90)">Taxa de Verdadeiro Positivo (TPR)</text>
        <path d={fillD} fill={`url(#${gradientId})`}/>
        <line x1={0} y1={cH} x2={cW} y2={0} className="roc-diagonal"/>
        <path d={pathD} fill="none" stroke={aucColor} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"/>
        <line x1={0} y1={0} x2={0} y2={cH} className="roc-axis-line"/>
        <line x1={0} y1={cH} x2={cW} y2={cH} className="roc-axis-line"/>
      </g>
    </svg>
  )
}

function RocChart({ model, index }) {
  const { roc_curve: pts, auc, model_version, mode, roc_high_count, roc_neg_count, score_distribution } = model
  const aucColor = !auc?'var(--color-text-secondary)':auc>=0.8?'var(--color-success)':auc>=0.6?'var(--color-warning)':'var(--color-danger)'
  const aucLabel = !auc?null:auc>=0.8?'Bom discriminador':auc>=0.6?'Discriminação moderada':'Discriminação fraca'
  const hasRoc   = pts && pts.length >= 2
  return (
    <div className="roc-card">
      <div className="roc-card-header">
        <div className="roc-card-title-row">
          <span className="roc-card-icon"><ModeIcon mode={mode} size={15}/></span>
          <span className="roc-card-name" title={model_version}>{model_version}</span>
          <span className="mdl-mode-badge">{MODE_LABEL[mode]||mode}</span>
        </div>
        {auc!=null&&(
          <div className="roc-auc-pill" style={{color:aucColor,borderColor:aucColor}}>
            AUC = {auc.toFixed(3)}{aucLabel&&<span className="roc-auc-pill-label">{aucLabel}</span>}
          </div>
        )}
      </div>
      <div className="roc-subsection">
        <div className="roc-subsection-header">
          <span className="roc-subsection-title">Curva ROC</span>
          <span className="roc-subsection-meta">{hasRoc?`${roc_high_count} HIGH · ${roc_neg_count} MEDIUM+LOW`:'dados insuficientes'}</span>
        </div>
        {hasRoc?<RocSvg points={pts} auc={auc} aucColor={aucColor} gradientId={`rocFill-${index}`}/>:(
          <div className="roc-no-data">
            <p>Curva não disponível ainda.</p>
            <p className="roc-no-data-hint">Necessário ter resultados <strong>HIGH</strong> e não-HIGH (MEDIUM ou LOW) para o mesmo modelo.<br/>Atualmente {roc_high_count??0} HIGH e {roc_neg_count??0} MEDIUM+LOW disponíveis.</p>
          </div>
        )}
      </div>
      <div className="roc-subsection">
        <div className="roc-subsection-header">
          <span className="roc-subsection-title">Distribuição de scores p_fake</span>
          <span className="roc-subsection-meta">{model.total} análises</span>
        </div>
        <ScoreDistChart distribution={score_distribution}/>
      </div>
      <p className="roc-note">ROC: HIGH (positivo) vs MEDIUM+LOW (negativo) · UNDETERMINED excluído · score = p_fake</p>
    </div>
  )
}

/* ─────────────── section content wrappers ─────────────── */
function SectionKpis({ data }) {
  const totalDone = data.fake_count + data.real_count
  const fakeRate  = totalDone > 0 ? ((data.fake_count / totalDone) * 100).toFixed(1) : '0.0'
  return (
    <div className="kpi-row">
      <KpiCard label="Total de análises"       value={data.total.toLocaleString('pt-BR')}/>
      <KpiCard label="Classificados como Fake" value={data.fake_count.toLocaleString('pt-BR')} sub={`${fakeRate}% das concluídas`} variant="kpi-danger"/>
      <KpiCard label="Classificados como Real" value={data.real_count.toLocaleString('pt-BR')} sub={totalDone>0?`${(100-parseFloat(fakeRate)).toFixed(1)}% das concluídas`:undefined} variant="kpi-success"/>
      <KpiCard label="Taxa de detecção"        value={`${fakeRate}%`} sub="Fake / total concluído" variant="kpi-primary"/>
    </div>
  )
}

function SectionMidrow({ data }) {
  return (
    <div className="dash-mid-row">
      <div className="dash-card">
        <div className="dash-card-header"><h3 className="dash-card-title">Por Modalidade</h3></div>
        <ModeBreakdown byMode={data.by_mode} total={data.total}/>
      </div>
      <div className="dash-card dash-card-wide">
        <div className="dash-card-header">
          <h3 className="dash-card-title">Tendência — últimos 14 dias</h3>
          <div className="trend-legend">
            <span><i className="legend-dot legend-real"/>Real</span>
            <span><i className="legend-dot legend-fake"/>Fake</span>
          </div>
        </div>
        <TrendChart trend={data.trend}/>
      </div>
    </div>
  )
}

function SectionModels({ models, loading, error, onRetry }) {
  const body = loading
    ? <div className="mdl-grid">{[1,2].map((i)=><div key={i} className="mdl-card mdl-card-skeleton" aria-hidden="true"/>)}</div>
    : error
      ? <div className="dash-empty"><p style={{color:'var(--color-danger)',marginBottom:'0.75rem'}}>{error}</p><button className="dash-retry" onClick={onRetry}>Tentar novamente</button></div>
      : !models||models.length===0
        ? <p className="dash-empty">Nenhum modelo utilizado ainda.</p>
        : <div className="mdl-grid">{models.map((m)=><ModelCard key={`${m.model_version}-${m.mode}`} model={m}/>)}</div>
  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3 className="dash-card-title">Desempenho por Modelo</h3>
        {models&&models.length>0&&<span className="dash-card-badge">{models.length} modelo{models.length!==1?'s':''}</span>}
      </div>
      {body}
    </div>
  )
}

function SectionRoc({ models, loading, error, onRetry }) {
  const body = loading
    ? <div className="roc-grid">{[1,2].map((i)=><div key={i} className="roc-card roc-card-skeleton" aria-hidden="true"/>)}</div>
    : error
      ? <div className="dash-empty"><p style={{color:'var(--color-danger)',marginBottom:'0.75rem'}}>{error}</p><button className="dash-retry" onClick={onRetry}>Tentar novamente</button></div>
      : !models||models.length===0
        ? <p className="dash-empty">Nenhum modelo utilizado ainda. As curvas aparecerão conforme as análises forem concluídas.</p>
        : <div className="roc-grid">{models.map((m,i)=><RocChart key={`${m.model_version}-${m.mode}`} model={m} index={i}/>)}</div>
  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3 className="dash-card-title">Curvas ROC por Modelo</h3>
        <span className="dash-card-badge-info">HIGH vs MEDIUM+LOW · UNDETERMINED excluído</span>
      </div>
      {body}
    </div>
  )
}

function SectionRecent({ data }) {
  return (
    <div className="dash-card">
      <div className="dash-card-header">
        <h3 className="dash-card-title">Análises recentes</h3>
        <span className="dash-card-badge">{data.recent.length} registros</span>
      </div>
      <RecentTable recent={data.recent}/>
    </div>
  )
}

/* ─────────────── drag-and-drop section wrapper ─────────────── */
function DashSection({ section, editMode, onToggle, onDragStart, onDragOver, onDrop, dragOver, children }) {
  const isHidden = !section.visible
  const meta     = SECTION_META[section.id] || {}

  if (!editMode && isHidden) return null

  return (
    <div
      className={[
        'dash-section-wrap',
        editMode        && 'in-edit',
        editMode && dragOver && 'drag-target',
        editMode && isHidden && 'is-hidden',
      ].filter(Boolean).join(' ')}
      draggable={editMode}
      onDragStart={editMode ? () => onDragStart(section.id) : undefined}
      onDragOver={editMode  ? (e) => { e.preventDefault(); onDragOver(section.id) } : undefined}
      onDragLeave={editMode ? () => onDragOver(null) : undefined}
      onDrop={editMode      ? (e) => { e.preventDefault(); onDrop(section.id) } : undefined}
    >
      {editMode && (
        <div className="section-edit-bar">
          <GripVertical size={15} className="section-grip" />
          <span className="section-edit-label">{meta.label}</span>
          <button
            className={`section-toggle ${isHidden ? 'toggle-hidden' : ''}`}
            onClick={() => onToggle(section.id)}
            title={isHidden ? 'Mostrar seção' : 'Ocultar seção'}
          >
            {isHidden ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      )}
      <div className={isHidden ? 'section-body section-body-dim' : 'section-body'}>
        {children}
      </div>
    </div>
  )
}

/* ─────────────── main Dashboard ─────────────── */
export default function Dashboard() {
  const [data,          setData]          = useState(null)
  const [models,        setModels]        = useState(null)
  const [loading,       setLoading]       = useState(true)
  const [modelsLoading, setModelsLoading] = useState(true)
  const [error,         setError]         = useState(null)
  const [modelsError,   setModelsError]   = useState(null)

  const [editMode,  setEditMode]  = useState(false)
  const [layout,    setLayout]    = useState(loadLayout)
  const [dragOver,  setDragOver]  = useState(null)
  const dragId = useRef(null)

  const loadSummary = useCallback(async () => {
    setLoading(true); setError(null)
    try { setData(await getStats()) }
    catch (err) { setError(err.message || 'Erro ao carregar estatísticas') }
    finally { setLoading(false) }
  }, [])

  const loadModels = useCallback(async () => {
    setModelsLoading(true); setModelsError(null)
    try { setModels((await getModelStats()).models) }
    catch (err) { setModelsError(err.message || 'Erro ao carregar métricas') }
    finally { setModelsLoading(false) }
  }, [])

  const loadAll = useCallback(() => { loadSummary(); loadModels() }, [loadSummary, loadModels])
  useEffect(() => { loadAll() }, [loadAll])

  /* layout ops */
  const toggleSection = useCallback((id) => {
    setLayout((prev) => {
      const next = prev.map((s) => s.id === id ? { ...s, visible: !s.visible } : s)
      saveLayout(next)
      return next
    })
  }, [])

  const handleDragStart = useCallback((id) => { dragId.current = id }, [])
  const handleDragOver  = useCallback((id) => { setDragOver(id) }, [])

  const handleDrop = useCallback((targetId) => {
    const from = dragId.current
    if (!from || from === targetId) { setDragOver(null); return }
    setLayout((prev) => {
      const next  = [...prev]
      const fromI = next.findIndex((s) => s.id === from)
      const toI   = next.findIndex((s) => s.id === targetId)
      const [moved] = next.splice(fromI, 1)
      next.splice(toI, 0, moved)
      const reordered = next.map((s, i) => ({ ...s, order: i }))
      saveLayout(reordered)
      return reordered
    })
    setDragOver(null)
    dragId.current = null
  }, [])

  const resetLayout = useCallback(() => {
    saveLayout(DEFAULT_LAYOUT)
    setLayout(DEFAULT_LAYOUT)
  }, [])

  /* section renderer */
  const renderContent = (id) => {
    if (!data) return null
    switch (id) {
      case 'kpis':   return <SectionKpis data={data}/>
      case 'midrow': return <SectionMidrow data={data}/>
      case 'models': return <SectionModels models={models} loading={modelsLoading} error={modelsError} onRetry={loadModels}/>
      case 'roc':    return <SectionRoc    models={models} loading={modelsLoading} error={modelsError} onRetry={loadModels}/>
      case 'recent': return <SectionRecent data={data}/>
      default: return null
    }
  }

  if (loading) {
    return (
      <div className="dash-state">
        <div className="dash-spinner"/>
        <span>Carregando estatísticas...</span>
      </div>
    )
  }
  if (error) {
    return (
      <div className="dash-state dash-error">
        <p>{error}</p>
        <button className="dash-retry" onClick={loadAll}>Tentar novamente</button>
      </div>
    )
  }

  return (
    <div className="dashboard">

      {/* Top bar */}
      <div className="dash-topbar">
        <div>
          <h2 className="dash-heading">Painel de Monitoramento</h2>
          <p className="dash-sub">Estatísticas acumuladas de detecções por modalidade</p>
        </div>
        <div className="dash-topbar-actions">
          {!editMode && (
            <button className="dash-refresh" onClick={loadAll}>↻ Atualizar</button>
          )}
          {editMode && (
            <button className="dash-action-btn dash-action-reset" onClick={resetLayout} title="Restaurar layout padrão">
              <RotateCcw size={13}/>
              Restaurar padrão
            </button>
          )}
          <button
            className={`dash-action-btn ${editMode ? 'dash-action-done' : 'dash-action-edit'}`}
            onClick={() => setEditMode((v) => !v)}
          >
            {editMode ? <><Check size={13}/> Concluir</> : <><Pencil size={13}/> Personalizar</>}
          </button>
        </div>
      </div>

      {/* Edit mode banner */}
      {editMode && (
        <div className="edit-mode-banner">
          <GripVertical size={14}/>
          Arraste as seções para reordenar · clique no ícone de olho para mostrar ou ocultar
        </div>
      )}

      {/* Sections */}
      <div className="dash-sections">
        {layout.map((section) => (
          <DashSection
            key={section.id}
            section={section}
            editMode={editMode}
            onToggle={toggleSection}
            onDragStart={handleDragStart}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            dragOver={dragOver === section.id}
          >
            {renderContent(section.id)}
          </DashSection>
        ))}
      </div>
    </div>
  )
}
