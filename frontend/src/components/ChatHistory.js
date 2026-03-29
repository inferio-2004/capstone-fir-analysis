import React, { useEffect, useState } from 'react';

export default function ChatHistory({ sessionId, onSelect }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    console.log('[DEBUG] Fetching chat history for', sessionId);
    fetch(`http://localhost:8000/api/chat_history/${sessionId}`)
      .then(res => res.json())
      .then(data => {
        setHistory(Array.isArray(data) ? data : []);
        console.log('[DEBUG] Chat history fetched:', data);
      })
      .catch((err) => {
        setHistory([]);
        console.log('[DEBUG] Chat history fetch error:', err);
      })
      .finally(() => setLoading(false));
  }, [sessionId]);

  if (!sessionId) return null;

  return (
    <div className="chat-history-panel">
      {loading && <div>Loading...</div>}
      <ul className="chat-history-list" style={{listStyle: 'none', padding: 0, margin: 0}}>
        {history.length === 0 && !loading && <li style={{color: '#888', fontSize: 13}}>No Q/A yet for this session.</li>}
        {history.map((item, idx) => (
          <li key={item._id || idx} style={{marginBottom: 8}}>
            <b>Q:</b> {item.question}<br />
            <b>A:</b> {item.answer ? item.answer.slice(0, 60) : ''}{item.answer && item.answer.length > 60 ? '...' : ''}
            <div className="chat-history-meta" style={{fontSize: 11, color: '#aaa'}}>{item.timestamp ? new Date(item.timestamp).toLocaleString() : ''}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
