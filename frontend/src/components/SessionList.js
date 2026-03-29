import React, { useEffect, useState } from 'react';

export default function SessionList({ sessionId, onSelect }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    console.log('[DEBUG] Fetching sessions for', sessionId);
    fetch(`http://localhost:8000/api/sessions/${sessionId}`)
      .then(res => res.json())
      .then(data => {
        setSessions(Array.isArray(data) ? data : []);
        console.log('[DEBUG] Sessions fetched:', data);
      })
      .catch((err) => { 
        setSessions([]);
        console.log('[DEBUG] Session fetch error:', err);
      })
      .finally(() => setLoading(false));
  }, [sessionId]);

  if (!sessionId) return null;

  return (
    <div className="session-list-panel" style={{marginBottom: 12}}>
      {loading && <div>Loading...</div>}
      <ul className="session-list" style={{listStyle: 'none', padding: 0, margin: 0}}>
        {sessions.map((item) => (
          <li
            key={item._id}
            className={item.session_id === sessionId ? 'active' : ''}
            style={{
              background: item.session_id === sessionId ? '#23263a' : 'transparent',
              borderRadius: 6,
              padding: '8px 10px',
              marginBottom: 4,
              cursor: 'pointer',
              fontWeight: item.session_id === sessionId ? 600 : 400,
              color: item.session_id === sessionId ? '#6cf' : '#fff',
            }}
            onClick={() => onSelect && onSelect(item.session_id)}
          >
            <span style={{fontSize: 13}}><b>Session:</b> {item.session_id.slice(0, 8)}</span>
            <div className="session-list-meta" style={{fontSize: 11, color: '#aaa'}}>
              {item.fir_summary && <span>{item.fir_summary.slice(0, 40)}...</span>}
              <span style={{float: 'right'}}>{item.qa_count} Q/A</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
