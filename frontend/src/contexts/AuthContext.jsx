import { createContext, useCallback, useContext, useEffect, useState } from 'react'

const TOKEN_KEY   = 'dts-token'
const EXPIRES_KEY = 'dts-expires-at'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser]                   = useState(null)
  const [loading, setLoading]             = useState(true)
  const [expiresAt, setExpiresAt]         = useState(null)
  const [sessionExpired, setSessionExpired] = useState(false)

  const logout = useCallback((options = {}) => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(EXPIRES_KEY)
    setUser(null)
    setExpiresAt(null)
    if (options.expired) setSessionExpired(true)
  }, [])

  // Restore session from localStorage on mount
  useEffect(() => {
    const token      = localStorage.getItem(TOKEN_KEY)
    const storedExp  = localStorage.getItem(EXPIRES_KEY)

    if (!token) { setLoading(false); return }

    // Already expired before even hitting the network
    if (storedExp && Date.now() > Number(storedExp)) {
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(EXPIRES_KEY)
      setSessionExpired(true)
      setLoading(false)
      return
    }

    fetch('/v1/auth/me', { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data) {
          setUser(data)
          if (storedExp) setExpiresAt(Number(storedExp))
        } else {
          localStorage.removeItem(TOKEN_KEY)
          localStorage.removeItem(EXPIRES_KEY)
        }
      })
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY)
        localStorage.removeItem(EXPIRES_KEY)
      })
      .finally(() => setLoading(false))
  }, [])

  // Listen for 401 events dispatched by api.js
  useEffect(() => {
    const handle = (e) => logout(e.detail || {})
    window.addEventListener('dts:logout', handle)
    return () => window.removeEventListener('dts:logout', handle)
  }, [logout])

  const login = useCallback(async (email, password) => {
    const r = await fetch('/v1/auth/login', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ email, password }),
    })
    if (!r.ok) {
      const err = await r.json().catch(() => ({}))
      throw new Error(err.detail || 'Credenciais inválidas.')
    }
    const data = await r.json()
    const exp  = Date.now() + data.expires_in * 1000
    localStorage.setItem(TOKEN_KEY,   data.access_token)
    localStorage.setItem(EXPIRES_KEY, String(exp))
    setUser(data.user)
    setExpiresAt(exp)
    setSessionExpired(false)
    return data.user
  }, [])

  return (
    <AuthContext.Provider value={{
      user, loading, login, logout,
      expiresAt, sessionExpired,
      isAuthenticated: !!user,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
