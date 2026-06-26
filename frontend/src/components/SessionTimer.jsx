import { useEffect, useState } from 'react'
import { Clock } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import './SessionTimer.css'

export default function SessionTimer({ mobile = false }) {
  const { expiresAt, logout } = useAuth()

  const getRemainingSeconds = () =>
    expiresAt ? Math.max(0, Math.floor((expiresAt - Date.now()) / 1000)) : 0

  const [remaining, setRemaining] = useState(getRemainingSeconds)

  useEffect(() => {
    if (!expiresAt) return
    const tick = () => {
      const rem = getRemainingSeconds()
      setRemaining(rem)
      if (rem === 0) logout({ expired: true })
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [expiresAt]) // eslint-disable-line react-hooks/exhaustive-deps

  const minutes   = Math.floor(remaining / 60)
  const seconds   = remaining % 60
  const formatted = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`

  const isWarning = remaining > 0 && remaining < 300  // < 5 min
  const isDanger  = remaining > 0 && remaining < 60   // < 1 min

  // Mobile banner: only render when time is running low
  if (mobile && !isWarning) return null

  const cls = [
    'session-timer',
    mobile    && 'st-mobile',
    isWarning && 'st-warning',
    isDanger  && 'st-danger',
  ].filter(Boolean).join(' ')

  return (
    <div className={cls} title="Tempo restante da sessão">
      <Clock size={mobile ? 13 : 11} />
      <span className="st-label">{mobile ? 'Sessão expira em ' : ''}</span>
      <span className="st-time">{formatted}</span>
    </div>
  )
}
