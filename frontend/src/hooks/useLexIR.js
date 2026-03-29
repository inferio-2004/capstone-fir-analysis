/**
 * useLexIR — React hook for the LexIR WebSocket backend.
 * Manages connection, message routing, chat history, and streaming state.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

const initialPipeline = () => ({
  s1: 'running',   // running | done
  s2: 'pending',   // pending | running | done
  s3: 'pending',   // pending | ready
});

export function useLexIR(url = 'ws://localhost:8000/ws') {
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);
  /** Keeps handleMsg in sync with whether we are viewing a loaded history session (Issue 1). */
  const loadedSessionRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('');
  const [currentStage, setCurrentStage] = useState(0);
  const [fir, setFir] = useState(null);
  const [stage1, setStage1] = useState(null);
  const [stage2, setStage2] = useState(null);
  const [qaAnswers, setQaAnswers] = useState([]);
  const [chatMessages, setChatMessages] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [loadedSession, setLoadedSession] = useState(null);
  const [pipelineProgress, setPipelineProgress] = useState(null);
  /** Increment to remount / reset FIRForm (Issue 2). */
  const [formResetKey, setFormResetKey] = useState(0);
  /** Default to expanded for new sessions. */
  const [isFIRExpanded, setIsFIRExpanded] = useState(true);

  useEffect(() => {
    loadedSessionRef.current = loadedSession;
  }, [loadedSession]);

  const addChat = useCallback((role, content, meta = {}) => {
    setChatMessages(prev => [...prev, {
      id: Date.now() + Math.random(),
      role,
      content,
      meta,
      timestamp: new Date(),
    }]);
  }, []);

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setError(null);
        try {
          ws.send(JSON.stringify({ type: 'list_sessions' }));
        } catch { /* ignore */ }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectRef.current = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        setError('Cannot reach backend — is the server running?');
        setConnected(false);
      };

      ws.onmessage = (event) => {
        try { handleMsg(JSON.parse(event.data)); } catch { /* ignore */ }
      };
    }

    function handleMsg(msg) {
      switch (msg.type) {
        case 'status': {
          setStatus(msg.message);
          if (typeof msg.stage === 'number') setCurrentStage(msg.stage);
          if (msg.stage === 3 && msg.message.includes('Ready')) setLoading(false);
          const st = msg.stage;
          if (typeof st === 'number' && st >= 1 && st <= 3) {
            break;
          }
          addChat('system', msg.message, { stage: msg.stage });
          break;
        }

        case 'fir_loaded':
          setFir(msg.fir);
          if (msg.session_id) setSessionId(msg.session_id);
          break;

        case 'stage1_result':
          setStage1(msg.data);
          setCurrentStage(1);
          setPipelineProgress({ s1: 'done', s2: 'running', s3: 'pending' });
          addChat('stage', null, { stage: 1, data: msg.data });
          break;

        case 'stage2_result':
          setStage2(msg.data);
          setCurrentStage(2);
          setLoading(false);
          setPipelineProgress({ s1: 'done', s2: 'done', s3: 'ready' });
          addChat('stage', null, { stage: 2, data: msg.data });
          break;

        case 'qa_answer': {
          const d = msg.data || {};
          setQaAnswers(prev => [...prev, d]);
          setCurrentStage(3);
          setLoading(false);

          if (loadedSessionRef.current) {
            const ts = new Date().toISOString();
            setLoadedSession(prev => {
              if (!prev) return prev;
              return {
                ...prev,
                messages: [
                  ...(prev.messages || []),
                  { role: 'assistant', content: d.answer, timestamp: ts },
                ],
              };
            });
          } else {
            addChat('assistant', d.answer, {
              stage: 3,
              question: d.question,
              is_no_match: d.is_no_match,
              precedents_used: d.precedents_used,
            });
          }
          break;
        }

        case 'sessions_list':
          setSessions(msg.sessions || []);
          break;

        case 'history':
          setIsFIRExpanded(false);
          setLoadedSession({
            sessionId: msg.session_id,
            fir: msg.fir,
            stage1: msg.stage1_data,
            stage2: msg.stage2_data,
            messages: msg.messages || [],
          });
          if (msg.session_id) setSessionId(msg.session_id);
          setChatMessages([]);
          setPipelineProgress(null);
          setStage1(null);
          setStage2(null);
          setFir(msg.fir || null);
          if (msg.stage2_data) setCurrentStage(3);
          else if (msg.stage1_data) setCurrentStage(2);
          else setCurrentStage(1);
          break;

        case 'session_cleared': {
          const clearedId = msg.session_id;
          if (loadedSessionRef.current?.sessionId === clearedId) {
            setLoadedSession(null);
            setChatMessages([]);
            setStage1(null);
            setStage2(null);
            setFir(null);
            setSessionId(null);
            setCurrentStage(0);
            setPipelineProgress(null);
            setIsFIRExpanded(false);
            setFormResetKey(k => k + 1);
          }
          try {
            wsRef.current?.send(JSON.stringify({ type: 'list_sessions' }));
          } catch { /* ignore */ }
          break;
        }

        case 'error':
          setError(msg.message);
          setLoading(false);
          setPipelineProgress(null);
          addChat('system', `Error: ${msg.message}`, { isError: true });
          break;

        default:
          break;
      }
    }

    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  const send = useCallback((payload) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  const startAnalysis = useCallback((firData = null) => {
    setLoading(true);
    setError(null);
    setStage1(null);
    setStage2(null);
    setQaAnswers([]);
    setLoadedSession(null);
    setIsFIRExpanded(false);
    setPipelineProgress(initialPipeline());
    const payload = { type: 'start_analysis' };
    if (firData) payload.fir = firData;
    addChat('user',
      firData
        ? `Submitted FIR case: ${firData.incident_description?.slice(0, 150)}...`
        : 'Starting analysis with sample FIR...',
      { isFirSubmit: true },
    );
    send(payload);
  }, [send, addChat]);

  const askQuestion = useCallback((question) => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);

    if (loadedSessionRef.current) {
      const ts = new Date().toISOString();
      setLoadedSession(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          messages: [...(prev.messages || []), { role: 'user', content: question, timestamp: ts }],
        };
      });
    } else {
      addChat('user', question);
    }

    const payload = { type: 'ask_question', question };
    if (sessionId) payload.session_id = sessionId;
    send(payload);
  }, [send, addChat, sessionId]);

  const showCases = useCallback(() => { send({ type: 'show_cases' }); }, [send]);

  const loadSession = useCallback((id) => {
    send({ type: 'get_history', session_id: id });
  }, [send]);

  const resetChat = useCallback(() => {
    setChatMessages([]);
    setStage1(null);
    setStage2(null);
    setQaAnswers([]);
    setFir(null);
    setCurrentStage(0);
    setLoading(false);
    setError(null);
    setStatus('');
    setSessionId(null);
    setLoadedSession(null);
    setIsFIRExpanded(true);
    setPipelineProgress(null);
    setFormResetKey(k => k + 1);
  }, []);

  const deleteSession = useCallback((id) => {
    setSessions(prev => prev.filter(s => s.id !== id));
    if (loadedSessionRef.current?.sessionId === id) {
      setLoadedSession(null);
      setChatMessages([]);
      setStage1(null);
      setStage2(null);
      setFir(null);
      setSessionId(null);
      setCurrentStage(0);
      setIsFIRExpanded(false);
      setPipelineProgress(null);
      setFormResetKey(k => k + 1);
    }
    send({ type: 'clear_session', session_id: id });
  }, [send]);

  return {
    connected,
    status,
    currentStage,
    loading,
    error,
    fir,
    stage1,
    stage2,
    qaAnswers,
    chatMessages,
    sessionId,
    sessions,
    loadedSession,
    pipelineProgress,
    formResetKey,
    isFIRExpanded,
    setIsFIRExpanded,
    startAnalysis,
    askQuestion,
    showCases,
    resetChat,
    loadSession,
    deleteSession,
  };
}
