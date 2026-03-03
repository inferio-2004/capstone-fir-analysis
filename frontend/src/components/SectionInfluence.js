import React from 'react';
import { TrendingUp } from 'lucide-react';

export default function SectionInfluence({ sections }) {
  const visible = sections?.filter(s => s.influence_score > 0);
  if (!visible?.length) return null;

  return (
    <div className="influence-card">
      <div className="influence-header">
        <TrendingUp size={18} />
        <h4>Section Influence on Verdict</h4>
      </div>
      <p className="influence-subtitle">
        Which sections are most likely to determine the outcome
      </p>
      <div className="influence-list">
        {visible.map((s, i) => {
          const barColor =
            s.influence_score >= 75 ? '#ef4444' :
            s.influence_score >= 50 ? '#f59e0b' :
            s.influence_score >= 25 ? '#6366f1' : '#94a3b8';
          const levelBadge =
            s.influence_level === 'Primary' ? 'influence-badge-primary' :
            s.influence_level === 'Supporting' ? 'influence-badge-supporting' :
            'influence-badge-minor';
          return (
            <div key={i} className="influence-item">
              <div className="influence-item-top">
                <span className="influence-rank">#{i + 1}</span>
                <span className="influence-section">{s.section}</span>
                <span className={`influence-level-badge ${levelBadge}`}>
                  {s.influence_level}
                </span>
                <span className="influence-pct">{s.influence_score}%</span>
              </div>
              <div className="influence-bar-track">
                <div
                  className="influence-bar-fill"
                  style={{ width: `${s.influence_score}%`, background: barColor }}
                />
              </div>
              {s.reasoning && (
                <p className="influence-reasoning">{s.reasoning}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
