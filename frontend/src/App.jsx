import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'
const WS_URL = 'ws://127.0.0.1:8000/api/live-feed'

function App() {
  const [stats, setStats] = useState(null)
  const [leaderboard, setLeaderboard] = useState([])
  const [liveFeed, setLiveFeed] = useState([])
  const [isOnline, setIsOnline] = useState(false)
  const [loading, setLoading] = useState(true)
  const wsRef = useRef(null)

  // ─── Fetch stats & leaderboard ───
  const fetchData = async () => {
    try {
      const [statsRes, lbRes, activityRes] = await Promise.all([
        fetch(`${API_BASE}/api/stats`),
        fetch(`${API_BASE}/api/leaderboard?limit=20`),
        fetch(`${API_BASE}/api/recent-activity?limit=20`),
      ])

      if (statsRes.ok) setStats(await statsRes.json())
      if (lbRes.ok) setLeaderboard(await lbRes.json())
      if (activityRes.ok) {
        const activity = await activityRes.json()
        setLiveFeed(activity.map(a => ({
          type: 'detection',
          plate_number: a.plate_number,
          timestamp: a.timestamp,
          safety_score: a.safety_score,
        })))
      }
      setIsOnline(true)
      setLoading(false)
    } catch {
      setIsOnline(false)
      setLoading(false)
    }
  }

  // ─── WebSocket for live updates ───
  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => setIsOnline(true)
      ws.onclose = () => {
        setIsOnline(false)
        setTimeout(connectWebSocket, 3000) // auto-reconnect
      }
      ws.onerror = () => setIsOnline(false)

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        // Add to live feed (top of list, max 30 items)
        setLiveFeed(prev => [data, ...prev].slice(0, 30))
        // Refresh stats & leaderboard on new data
        fetchData()
      }
    } catch {
      setTimeout(connectWebSocket, 3000)
    }
  }

  useEffect(() => {
    fetchData()
    connectWebSocket()
    // Periodic polling fallback every 8s
    const interval = setInterval(fetchData, 8000)
    return () => {
      clearInterval(interval)
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  const getScoreClass = (score) => {
    if (score >= 85) return 'excellent'
    if (score >= 65) return 'good'
    if (score >= 40) return 'warning'
    return 'danger'
  }

  const getRankClass = (index) => {
    if (index === 0) return 'gold'
    if (index === 1) return 'silver'
    if (index === 2) return 'bronze'
    return ''
  }

  const getRankLabel = (index) => {
    if (index === 0) return '🥇'
    if (index === 1) return '🥈'
    if (index === 2) return '🥉'
    return `#${index + 1}`
  }

  const getViolationClass = (count) => {
    if (count === 0) return 'clean'
    if (count <= 2) return 'warn'
    return 'danger'
  }

  const formatTime = (isoString) => {
    if (!isoString) return '--:--'
    const date = new Date(isoString)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="logo-icon">🛡️</div>
          <h1>Driver<span>Guard</span></h1>
        </div>
        <div className="header-status">
          <div className={`status-badge ${isOnline ? 'online' : 'offline'}`}>
            <span className="status-dot"></span>
            {isOnline ? 'Pipeline Active' : 'Pipeline Offline'}
          </div>
        </div>
      </header>

      {/* Dashboard */}
      <main className="dashboard">
        {/* Stats Grid */}
        <section className="stats-grid" id="stats-panel">
          <div className="stat-card">
            <div className="stat-icon">🚗</div>
            <div className="stat-label">Tracked Drivers</div>
            <div className="stat-value">{stats?.total_drivers ?? '—'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">📡</div>
            <div className="stat-label">Total Detections</div>
            <div className="stat-value">{stats?.total_events ?? '—'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">⚠️</div>
            <div className="stat-label">Violations</div>
            <div className="stat-value">{stats?.total_violations ?? '—'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">⭐</div>
            <div className="stat-label">Avg Safety Score</div>
            <div className={`stat-value score`}>{stats?.avg_safety_score ?? '—'}</div>
          </div>
        </section>

        {/* Main Panels */}
        <section className="panels">
          {/* Leaderboard */}
          <div className="panel" id="leaderboard-panel">
            <div className="panel-header">
              <h2><span className="panel-icon">🏆</span> Safety Leaderboard</h2>
              <span className="panel-badge">{leaderboard.length} drivers</span>
            </div>
            <div className="panel-body">
              {loading ? (
                <>
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="skeleton skeleton-row" />
                  ))}
                </>
              ) : leaderboard.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">🚦</div>
                  <p>No drivers detected yet. Start the vision pipeline to begin tracking.</p>
                </div>
              ) : (
                leaderboard.map((driver, index) => (
                  <div className="leaderboard-row" key={driver.id}>
                    <span className={`rank ${getRankClass(index)}`}>{getRankLabel(index)}</span>
                    <div className="plate-info">
                      <span className="plate-number">{driver.plate_number}</span>
                      <span className="plate-meta">Last seen {formatTime(driver.last_seen_at)}</span>
                    </div>
                    <span className="sightings">👁️ {driver.total_sightings} seen</span>
                    <span className={`violations ${getViolationClass(driver.violation_count)}`}>
                      {driver.violation_count === 0 ? '✓ Clean' : `${driver.violation_count} flags`}
                    </span>
                    <span className={`score ${getScoreClass(driver.safety_score)}`}>
                      {driver.safety_score}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Live Feed */}
          <div className="panel" id="live-feed-panel">
            <div className="panel-header">
              <h2><span className="panel-icon">📡</span> Live Detections</h2>
              <span className="panel-badge">LIVE</span>
            </div>
            <div className="panel-body">
              {liveFeed.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">📷</div>
                  <p>Waiting for detections from the ANPR Vision Pipeline...</p>
                </div>
              ) : (
                liveFeed.map((item, index) => (
                  <div className="live-feed-item" key={index}>
                    <div className={`feed-icon ${item.type === 'violation' ? 'violation' : 'detection'}`}>
                      {item.type === 'violation' ? '🚨' : '🔍'}
                    </div>
                    <div className="feed-content">
                      <div className="feed-plate">{item.plate_number}</div>
                      <div className="feed-desc">
                        {item.type === 'violation'
                          ? `${item.violation_type} — ${item.points_deducted} pts deducted`
                          : `Detected • Score: ${item.safety_score ?? item.new_score ?? '100'}`}
                      </div>
                    </div>
                    <span className="feed-time">{formatTime(item.timestamp)}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
