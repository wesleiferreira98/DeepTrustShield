import { LayoutDashboard, ClipboardList, ScanFace, Shield } from 'lucide-react'
import './Sidebar.css'

const NAV = [
  { id: 'analysis',  label: 'Análise',   icon: ScanFace },
  { id: 'dashboard', label: 'Painel',    icon: LayoutDashboard },
  { id: 'history',   label: 'Histórico', icon: ClipboardList },
]

export default function Sidebar({ activeTab, onTabChange }) {
  return (
    <>
      {/* ── Desktop sidebar ── */}
      <aside className="sidebar">
        <div className="sb-logo">
          <div className="sb-logo-mark">
            <Shield size={17} />
          </div>
          <div className="sb-logo-text">
            <span className="sb-brand">DeepTrustShield</span>
            <span className="sb-tagline">Detecção de deepfakes com IA</span>
          </div>
        </div>

        <nav className="sb-nav">
          <span className="sb-section-label">Menu</span>
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`sb-item ${activeTab === id ? 'active' : ''}`}
              onClick={() => onTabChange(id)}
              type="button"
            >
              <Icon size={16} className="sb-item-icon" />
              <span>{label}</span>
              {activeTab === id && <span className="sb-item-bar" aria-hidden="true" />}
            </button>
          ))}
        </nav>

        <div className="sb-footer">
          <div className="sb-status">
            <span className="sb-dot" />
            <span>Sistema online</span>
          </div>
          <span className="sb-version">v1.0</span>
        </div>
      </aside>

      {/* ── Mobile bottom nav ── */}
      <nav className="mob-nav" aria-label="Navegação principal">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`mob-nav-item ${activeTab === id ? 'active' : ''}`}
            onClick={() => onTabChange(id)}
            type="button"
          >
            <Icon size={20} className="mob-nav-icon" />
            <span className="mob-nav-label">{label}</span>
          </button>
        ))}
      </nav>
    </>
  )
}
