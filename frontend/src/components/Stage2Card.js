import React, { useState } from 'react';
import {
  Scale, ExternalLink, ChevronDown, ChevronUp, Search,
  Gavel, ShieldAlert, AlertTriangle, CheckCircle2, TrendingUp,
} from 'lucide-react';

export default function Stage2Card({ data }) {
  const [expanded, setExpanded] = useState(false);

  if (!data) return null;

  const { status, sections_searched, cases, verdict_prediction,
          section_influence, api_calls_used, error } = data;

  /* ---------- error state ---------- */
  if (status === 'error') {
    return (
      <div className="stage-card stage2">
        <div className="stage-card-header">
          <div className="stage-badge stage2-badge"><Scale size={14} /> Stage 2</div>
          <span className="stage-title">Case Law &amp; Verdict Prediction</span>
        </div>
        <div className="no-match-msg">
          <p>Could not search Indian Kanoon: {error}</p>
        </div>
      </div>
    );
  }

  const statusLabel = {
    success: 'Cases Found',
    partial: 'Partial Results',
    no_results: 'No Cases Found',
  }[status] || status;

  const statusColor = {
    success: '#22c55e',
    partial: '#f59e0b',
    no_results: '#94a3b8',
  }[status] || '#94a3b8';

  const displayCases = expanded ? cases : cases?.slice(0, 3);

  /* ---------- verdict helpers ---------- */
  const vp = verdict_prediction || {};
  const verdictColor =
    vp.predicted_verdict?.toLowerCase().includes('guilty') ? '#ef4444' :
    vp.predicted_verdict?.toLowerCase().includes('acquittal') ? '#22c55e' :
    '#f59e0b';

  const bailColor =
    vp.bail_likelihood?.toLowerCase().includes('high') ? '#22c55e' :
    vp.bail_likelihood?.toLowerCase().includes('low') ? '#ef4444' :
    '#f59e0b';

  const confidencePct = typeof vp.confidence === 'number'
    ? Math.round(vp.confidence * 100)
    : null;

  return (
    <div className="stage-card stage2">
      {/* ---- Header ---- */}
      <div className="stage-card-header">
        <div className="stage-badge stage2-badge"><Scale size={14} /> Stage 2</div>
        <span className="stage-title">Indian Kanoon Case Law &amp; Verdict Prediction</span>
        <span className="match-badge" style={{ background: statusColor }}>
          {statusLabel}
        </span>
      </div>

      {sections_searched?.length > 0 && (
        <div className="kanoon-sections-searched">
          <Search size={12} />
          <span>Searched: {sections_searched.join(', ')}</span>
        </div>
      )}

      {/* ---- No results ---- */}
      {status === 'no_results' ? (
        <div className="no-match-msg">
          <p>No case law found on Indian Kanoon for the identified IPC sections.</p>
          <p className="hint">You can still ask legal questions in Stage 3.</p>
        </div>
      ) : (
        <>
          {/* ============ VERDICT PREDICTION CARD ============ */}
          {vp.predicted_verdict && (
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
                      {vp.predicted_verdict}
                    </span>
                  </div>
                </div>

                <div className="verdict-item">
                  <AlertTriangle size={16} style={{ color: '#f59e0b' }} />
                  <div>
                    <span className="verdict-label">Likely Punishment</span>
                    <span className="verdict-value">{vp.predicted_punishment}</span>
                  </div>
                </div>

                <div className="verdict-item">
                  <Scale size={16} style={{ color: '#6366f1' }} />
                  <div>
                    <span className="verdict-label">Punishment Range</span>
                    <span className="verdict-value">{vp.punishment_range}</span>
                  </div>
                </div>

                <div className="verdict-item">
                  <CheckCircle2 size={16} style={{ color: bailColor }} />
                  <div>
                    <span className="verdict-label">Bail Likelihood</span>
                    <span className="verdict-value" style={{ color: bailColor }}>
                      {vp.bail_likelihood}
                    </span>
                  </div>
                </div>
              </div>

              {vp.reasoning && (
                <div className="verdict-reasoning">
                  <strong>Reasoning:</strong> {vp.reasoning}
                </div>
              )}
            </div>
          )}

          {/* ============ SECTION INFLUENCE ON VERDICT ============ */}
          {section_influence?.length > 0 && (
            <div className="influence-card">
              <div className="influence-header">
                <TrendingUp size={18} />
                <h4>Section Influence on Verdict</h4>
              </div>
              <p className="influence-subtitle">
                Which sections are most likely to determine the outcome
              </p>
              <div className="influence-list">
                {section_influence.filter(s => s.influence_score > 0).map((s, i) => {
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
          )}

          {/* ============ CASE SUMMARIES ============ */}
          <div className="kanoon-cases-list">
            <h4 className="cases-section-title">
              Relevant Court Judgments ({cases?.length || 0})
            </h4>

            {displayCases?.map((c, i) => (
              <div key={c.tid || i} className="kanoon-case-item">
                <div className="kanoon-case-header">
                  <span className="case-number">{i + 1}.</span>
                  <span className="kanoon-case-title">{c.title}</span>
                </div>

                <div className="kanoon-case-meta">
                  <span className="kanoon-court">{c.court}</span>
                  {c.date && <span className="kanoon-date">{c.date}</span>}
                  <span className="kanoon-section-tag">{c.section}</span>
                </div>

                {/* LLM Summary */}
                {c.summary && (
                  <div className="kanoon-summary">
                    {c.summary}
                  </div>
                )}

                {/* Raw snippet fallback */}
                {!c.summary && c.snippet && (
                  <div className="kanoon-snippet">
                    {c.snippet.length > 300 ? c.snippet.slice(0, 300) + '...' : c.snippet}
                  </div>
                )}

                {c.url && (
                  <a href={c.url} target="_blank" rel="noopener noreferrer" className="kanoon-link">
                    <ExternalLink size={12} /> Read full judgment on Indian Kanoon
                  </a>
                )}
              </div>
            ))}
          </div>

          {cases?.length > 3 && (
            <button className="kanoon-toggle" onClick={() => setExpanded(prev => !prev)}>
              {expanded
                ? <><ChevronUp size={14} /> Show fewer</>
                : <><ChevronDown size={14} /> Show all {cases.length} cases</>
              }
            </button>
          )}
        </>
      )}

      {api_calls_used > 0 && (
        <div className="kanoon-footer">
          <span className="kanoon-api-note">
            {cases?.length || 0} case(s) summarized · {api_calls_used} API call(s) · Source: Indian Kanoon
          </span>
        </div>
      )}
    </div>
  );
}
