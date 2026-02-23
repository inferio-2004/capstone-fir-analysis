/**
 * useLexIR — React hook for the LexIR WebSocket backend.
 * Manages connection, message routing, chat history, and streaming state.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

export function useLexIR(url = 'ws://localhost:8000/ws') {
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);
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

  /* ---- helper: push a chat bubble ---- */
  const addChat = useCallback((role, content, meta = {}) => {
    setChatMessages(prev => [...prev, {
      id: Date.now() + Math.random(),
      role,       // 'system' | 'user' | 'assistant' | 'stage'
      content,
      meta,
      timestamp: new Date(),
    }]);
  }, []);

  /* ---- WebSocket lifecycle ---- */
  useEffect(() => {
    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => { setConnected(true); setError(null); };

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
        case 'status':
          setStatus(msg.message);
          setCurrentStage(msg.stage);
          addChat('system', msg.message, { stage: msg.stage });
          if (msg.stage === 3 && msg.message.includes('Ready')) setLoading(false);
          break;

        case 'fir_loaded':
          setFir(msg.fir);
          break;

        case 'stage1_result':
          setStage1(msg.data);
          setCurrentStage(1);
          addChat('stage', null, { stage: 1, data: msg.data });
          break;

        case 'stage2_result':
          setStage2(msg.data);
          setCurrentStage(2);
          setLoading(false);
          addChat('stage', null, { stage: 2, data: msg.data });
          break;

        case 'qa_answer':
          setQaAnswers(prev => [...prev, msg.data]);
          setCurrentStage(3);
          setLoading(false);
          addChat('assistant', msg.data.answer, {
            stage: 3,
            question: msg.data.question,
            is_no_match: msg.data.is_no_match,
            precedents_used: msg.data.precedents_used,
          });
          break;

        case 'error':
          setError(msg.message);
          setLoading(false);
          addChat('system', `Error: ${msg.message}`, { isError: true });
          break;

        default: break;
      }
    }

    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  /* ---- send helper ---- */
  const send = useCallback((payload) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  /* ---- public actions ---- */
  const startAnalysis = useCallback((firData = null) => {
    setLoading(true); setError(null);
    setStage1(null); setStage2(null); setQaAnswers([]);
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

  const runFullAnalysis = useCallback((firData = null) => {
    setLoading(true); setError(null);
    setStage1(null); setStage2(null); setQaAnswers([]);
    const payload = { type: 'run_full_analysis' };
    if (firData) payload.fir = firData;
    addChat('user', 'Running full RAG analysis (may take 30-60 s)...', { isFirSubmit: true });
    send(payload);
  }, [send, addChat]);

  const askQuestion = useCallback((question) => {
    if (!question.trim()) return;
    setLoading(true); setError(null);
    addChat('user', question);
    send({ type: 'ask_question', question });
  }, [send, addChat]);

  const showCases = useCallback(() => { send({ type: 'show_cases' }); }, [send]);

  const resetChat = useCallback(() => {
    setChatMessages([]); setStage1(null); setStage2(null);
    setQaAnswers([]); setFir(null); setCurrentStage(0);
    setLoading(false); setError(null); setStatus('');
  }, []);

  return {
    connected, status, currentStage, loading, error,
    fir, stage1, stage2, qaAnswers, chatMessages,
    startAnalysis, runFullAnalysis, askQuestion, showCases, resetChat,
  };
}
