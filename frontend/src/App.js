import React from 'react';
import './App.css';
import { useLexIR } from './hooks/useLexIR';
import Sidebar from './components/Sidebar';
import FIRForm from './components/FIRForm';
import ChatArea from './components/ChatArea';
import ChatInput from './components/ChatInput';

function App() {
  const {
    connected, currentStage, loading, error,
    stage1, stage2, chatMessages,
    startAnalysis, askQuestion, showCases, resetChat,
  } = useLexIR('ws://localhost:8000/ws');

  const hasAnalysis = !!(stage1 || stage2);
  const qaReady = currentStage >= 3;

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
          {error && <span className="top-error">{error}</span>}
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
