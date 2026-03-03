import React from 'react';
import { Gavel, ShieldAlert, AlertTriangle, Scale, CheckCircle2 } from 'lucide-react';

export default function VerdictCard({ verdict }) {
  if (!verdict?.predicted_verdict) return null;

  const verdictColor =
    verdict.predicted_verdict.toLowerCase().includes('guilty') ? '#ef4444' :
    verdict.predicted_verdict.toLowerCase().includes('acquittal') ? '#22c55e' :
    '#f59e0b';

  const bailColor =
    verdict.bail_likelihood?.toLowerCase().includes('high') ? '#22c55e' :
    verdict.bail_likelihood?.toLowerCase().includes('low') ? '#ef4444' :
    '#f59e0b';

  const confidencePct = typeof verdict.confidence === 'number'
    ? Math.round(verdict.confidence * 100)
    : null;

  return (
    <div className="verdict-card">
      <div className="verdict-header">
        <Gavel size={18} />
        <h4>Verdict Prediction</h4>
        {confidencePct !== null && (
          <span className="verdict-confidence">{confidencePct}% confidence</span>
        )}
      </div>

      <div className="verdict-grid">
        <div className="verdict-item">
          <ShieldAlert size={16} style={{ color: verdictColor }} />
          <div>
            <span className="verdict-label">Predicted Verdict</span>
            <span className="verdict-value" style={{ color: verdictColor }}>
              {verdict.predicted_verdict}
            </span>
          </div>
        </div>

        <div className="verdict-item">
          <AlertTriangle size={16} style={{ color: '#f59e0b' }} />
          <div>
            <span className="verdict-label">Likely Punishment</span>
            <span className="verdict-value">{verdict.predicted_punishment}</span>
          </div>
        </div>

        <div className="verdict-item">
          <Scale size={16} style={{ color: '#6366f1' }} />
          <div>
            <span className="verdict-label">Punishment Range</span>
            <span className="verdict-value">{verdict.punishment_range}</span>
          </div>
        </div>

        <div className="verdict-item">
          <CheckCircle2 size={16} style={{ color: bailColor }} />
          <div>
            <span className="verdict-label">Bail Likelihood</span>
            <span className="verdict-value" style={{ color: bailColor }}>
              {verdict.bail_likelihood}
            </span>
          </div>
        </div>
      </div>

      {verdict.reasoning && (
        <div className="verdict-reasoning">
          <strong>Reasoning:</strong> {verdict.reasoning}
        </div>
      )}
    </div>
  );
}
