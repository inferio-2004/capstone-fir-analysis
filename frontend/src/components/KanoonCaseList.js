import React from 'react';
import { ExternalLink } from 'lucide-react';

export default function KanoonCaseList({ cases }) {
  if (!cases?.length) return null;

  return (
    <div className="kanoon-cases-list">
      <h4 className="cases-section-title">
        Relevant Court Judgments ({cases.length})
      </h4>

      {cases.map((c, i) => (
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

          {c.summary && (
            <div className="kanoon-summary">{c.summary}</div>
          )}

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
  );
}
