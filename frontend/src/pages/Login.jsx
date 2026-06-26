import { useState } from 'react'
import { Shield } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import './Login.css'

export default function Login({ onBack, expired = false }) {
  const { login } = useAuth()
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(email.trim(), password)
    } catch (err) {
      setError(err.message || 'Erro ao fazer login.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">
          <div className="login-logo-mark">
            <Shield size={20} />
          </div>
          <span className="login-logo-name">DeepTrustShield</span>
        </div>

        <h1 className="login-title">Bem-vindo de volta</h1>
        <p className="login-subtitle">Entre com suas credenciais para acessar o sistema</p>

        <form onSubmit={handleSubmit} noValidate>
          {expired && !error && (
            <div className="login-expired" role="alert">
              Sua sessão expirou. Faça login para continuar.
            </div>
          )}
          {error && <div className="login-error" role="alert">{error}</div>}

          <div className="login-field">
            <label className="login-label" htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              className="login-input"
              placeholder="seu@email.com"
              value={email}
              onChange={e => setEmail(e.target.value)}
              autoComplete="email"
              required
            />
          </div>

          <div className="login-field">
            <label className="login-label" htmlFor="password">Senha</label>
            <input
              id="password"
              type="password"
              className="login-input"
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </div>

          <button type="submit" className="login-btn" disabled={loading || !email || !password}>
            {loading ? 'Entrando…' : 'Entrar'}
          </button>
        </form>

        <button className="login-back" type="button" onClick={onBack}>
          ← Voltar para a página inicial
        </button>
      </div>
    </div>
  )
}
