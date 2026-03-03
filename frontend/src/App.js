import React, { useState, useCallback } from 'react';
import { Download } from 'lucide-react';
import './App.css';
import { useLexIR } from './hooks/useLexIR';
import Sidebar from './components/Sidebar';
import FIRForm from './components/FIRForm';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';

function App() {
  const {
    connected, currentStage, loading, error,
    fir, stage1, stage2, chatMessages,
    startAnalysis, askQuestion, showCases, resetChat,
  } = useLexIR('ws://localhost:8000/ws');

  const hasAnalysis = !!(stage1 || stage2);
  const qaReady = currentStage >= 3;

  /* ---- PDF download ---- */
  const [pdfLoading, setPdfLoading] = useState(false);

  const handleDownloadPDF = useCallback(async () => {
    setPdfLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/fir/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fir: fir || {},
          analysis: stage1?._raw_analysis || null,
        }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `FIR_${fir?.fir_id || 'report'}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('PDF download failed:', err);
      alert('Failed to generate PDF. Is the server running?');
    } finally {
      setPdfLoading(false);
    }
  }, [fir, stage1]);

  return (
    <div className="app-layout">
      <Sidebar
        connected={connected}
        currentStage={currentStage}
        hasAnalysis={hasAnalysis}
        onShowCases={showCases}
        onReset={resetChat}
      />

      <main className="main-panel">
        {/* Top bar */}
        <header className="top-bar">
          <h2>
            {currentStage === 0 && 'Submit a Case'}
            {currentStage === 1 && 'Stage 1 — FIR Analysis'}
            {currentStage === 2 && 'Stage 2 — Similar Cases'}
            {currentStage >= 3 && 'Stage 3 — Precedent Q&A'}
          </h2>
          <div className="top-bar-actions">
            {stage1 && (
              <button
                className="btn btn-pdf-download"
                onClick={handleDownloadPDF}
                disabled={pdfLoading}
                title="Download filled FIR form as PDF"
              >
                <Download size={16} />
                {pdfLoading ? 'Generating…' : 'Download FIR PDF'}
              </button>
            )}
            {error && <span className="top-error">{error}</span>}
          </div>
        </header>

        {/* FIR Form (shown when no analysis yet) */}
        {!hasAnalysis && (
          <div className="form-container">
            <FIRForm onSubmit={startAnalysis} disabled={loading || !connected} />
          </div>
        )}

        {/* Chat area */}
        <ChatArea messages={chatMessages} loading={loading} />

        {/* Chat input (shown once Q&A stage is reached) */}
        {qaReady && (
          <ChatInput
            onSend={askQuestion}
            disabled={loading || !connected}
            placeholder="Ask about sections, precedents, punishments, bail..."
          />
        )}
      </main>
    </div>
  );
}

export default App;
