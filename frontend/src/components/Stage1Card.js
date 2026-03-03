import React from 'react';
import { Shield, Scale, FileText, AlertTriangle } from 'lucide-react';
import './Stage1Card.css';

export default function Stage1Card({ data }) {
  if (!data) return null;

  const { fir_summary, intent, severity, legal_basis, statutes, mapped_sections,
          chunks_retrieved, chunks_after_filtering, confidence } = data;

  const severityColor = {
    high: '#ef4444',
    medium: '#f59e0b',
    low: '#22c55e',
  }[severity?.toLowerCase()] || '#94a3b8';

  return (
    <div className="stage-card stage1">
      <div className="stage-card-header">
        <div className="stage-badge stage1-badge">
          <Shield size={14} /> Stage 1
        </div>
        <span className="stage-title">FIR Analysis & Section Mapping</span>
      </div>

      {/* FIR Summary */}
      <div className="stage-section">
        <h4><FileText size={14} /> Case Summary</h4>
        <div className="info-grid">
          <span className="info-label">FIR ID</span>
          <span>{fir_summary?.fir_id}</span>
          <span className="info-label">Date</span>
          <span>{fir_summary?.date}</span>
          <span className="info-label">Complainant</span>
          <span>{fir_summary?.complainant || 'N/A'}</span>
          <span className="info-label">Accused</span>
          <span>{fir_summary?.accused?.join(', ') || 'N/A'}</span>
          <span className="info-label">Victim</span>
          <span>{fir_summary?.victim || 'N/A'}</span>
          <span className="info-label">Location</span>
          <span>{fir_summary?.location || 'N/A'}</span>
        </div>
      </div>

      {/* Intent */}
      <div className="stage-section">
        <h4><AlertTriangle size={14} /> Intent Identification</h4>
        <div className="intent-row">
          <span className="intent-primary">{intent?.primary}</span>
          <span className="confidence-badge">
            {Math.round((intent?.confidence || 0) * 100)}% confident
          </span>
        </div>
        {intent?.secondary?.length > 0 && (
          <div className="intent-secondary">
            Secondary: {intent.secondary.join(', ')}
          </div>
        )}
        <div className="severity-row">
          Severity:
          <span className="severity-badge" style={{ background: severityColor }}>
            {severity?.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Statutes */}
      <div className="stage-section">
        <h4><Scale size={14} /> Applicable Sections ({statutes?.length || 0})</h4>
        <div className="statutes-list">
          {statutes?.map((s, i) => (
            <div key={i} className="statute-item">
              <div className="statute-primary">
                <strong>{s.primary.law} {s.primary.section}</strong>
                {s.primary.title && <span className="statute-title"> — {s.primary.title}</span>}
              </div>
              {s.primary.reasoning && (
                <p className="statute-reasoning">{s.primary.reasoning}</p>
              )}
              {s.corresponding_sections?.length > 0 && (
                <div className="statute-corresponding">
                  {s.corresponding_sections.map((c, j) => (
                    <span key={j} className="corr-badge">
                      → {c.law} {c.section}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Legal Basis */}
      {legal_basis && (
        <div className="stage-section">
          <h4>Legal Basis</h4>
          <p className="legal-basis-text">{legal_basis}</p>
        </div>
      )}

      {/* Footer stats */}
      <div className="stage-footer">
        <span>Chunks: {chunks_retrieved} retrieved → {chunks_after_filtering} filtered</span>
        <span>Confidence: {typeof confidence === 'number' ? Math.round(confidence * 100) + '%' : confidence}</span>
      </div>
    </div>
  );
}
