import React from 'react';
import { Wifi, WifiOff, RotateCcw } from 'lucide-react';

export default function Sidebar({ connected, currentStage, onReset, onShowCases, hasAnalysis }) {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="brand-icon">⚖</div>
        <h1>LexIR</h1>
        <p>Legal Intelligence & Retrieval</p>
      </div>

      {/* Connection status */}
      <div className={`connection-status ${connected ? 'online' : 'offline'}`}>
        {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {/* Pipeline stages */}
      <div className="pipeline-stages">
        <h4>Analysis Pipeline</h4>
        <div className={`pipeline-step ${currentStage >= 1 ? 'active' : ''} ${currentStage > 1 ? 'done' : ''}`}>
          <div className="step-dot">1</div>
          <div className="step-info">
            <span className="step-name">FIR Analysis</span>
            <span className="step-desc">Section mapping</span>
          </div>
        </div>
        <div className={`pipeline-step ${currentStage >= 2 ? 'active' : ''} ${currentStage > 2 ? 'done' : ''}`}>
          <div className="step-dot">2</div>
          <div className="step-info">
            <span className="step-name">Case Similarity</span>
            <span className="step-desc">Past precedents</span>
          </div>
        </div>
        <div className={`pipeline-step ${currentStage >= 3 ? 'active' : ''}`}>
          <div className="step-dot">3</div>
          <div className="step-info">
            <span className="step-name">Precedent Q&A</span>
            <span className="step-desc">Ask questions</span>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="sidebar-actions">
        {hasAnalysis && (
          <button className="btn btn-outline" onClick={onShowCases}>
            Show Similar Cases
          </button>
        )}
        <button className="btn btn-outline btn-danger" onClick={onReset}>
          <RotateCcw size={14} /> New Session
        </button>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <small>Capstone Project</small>
        <small>RAG + LLM Pipeline</small>
      </div>
    </aside>
  );
}
