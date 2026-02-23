import React, { useState } from 'react';
import { Send } from 'lucide-react';

const EMPTY = {
  complainant_name: '',
  accused_names_str: '',
  victim_name: '',
  incident_description: '',
  victim_impact: '',
  evidence: '',
  location: '',
  police_station: '',
  date: new Date().toISOString().slice(0, 10),
};

export default function FIRForm({ onSubmit, disabled }) {
  const [form, setForm] = useState(EMPTY);
  const [expanded, setExpanded] = useState(true);

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: e.target.value }));

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!form.incident_description.trim()) return;

    const fir = {
      fir_id: `FIR-${Date.now()}`,
      date: form.date,
      complainant_name: form.complainant_name || null,
      accused_names: form.accused_names_str
        ? form.accused_names_str.split(',').map(s => s.trim()).filter(Boolean)
        : [],
      victim_name: form.victim_name || null,
      incident_description: form.incident_description,
      victim_impact: form.victim_impact,
      evidence: form.evidence,
      location: form.location,
      police_station: form.police_station,
    };

    onSubmit(fir);
    setExpanded(false);
  };

  const handleSampleFIR = () => {
    onSubmit(null); // null = use sample FIR on backend
    setExpanded(false);
  };

  if (!expanded) {
    return (
      <div className="fir-form-collapsed" onClick={() => setExpanded(true)}>
        <span className="fir-form-collapsed-text">
          FIR submitted — click to edit & resubmit
        </span>
      </div>
    );
  }

  return (
    <form className="fir-form" onSubmit={handleSubmit}>
      <div className="fir-form-header">
        <h3>Submit a Case for Analysis</h3>
        <p>Fill in the case details below, or use the sample FIR to try the system.</p>
      </div>

      <div className="fir-form-grid">
        {/* Row 1 */}
        <div className="fir-field">
          <label>Complainant Name</label>
          <input type="text" placeholder="e.g. Rajesh Kumar" value={form.complainant_name} onChange={set('complainant_name')} />
        </div>
        <div className="fir-field">
          <label>Victim Name</label>
          <input type="text" placeholder="e.g. Priya Sharma" value={form.victim_name} onChange={set('victim_name')} />
        </div>

        {/* Row 2 */}
        <div className="fir-field">
          <label>Accused Names <span className="hint">(comma-separated)</span></label>
          <input type="text" placeholder="e.g. Amit Singh, Vikram Patel" value={form.accused_names_str} onChange={set('accused_names_str')} />
        </div>
        <div className="fir-field">
          <label>Date of Incident</label>
          <input type="date" value={form.date} onChange={set('date')} />
        </div>

        {/* Row 3 */}
        <div className="fir-field">
          <label>Location</label>
          <input type="text" placeholder="e.g. Sector 45, Chandigarh" value={form.location} onChange={set('location')} />
        </div>
        <div className="fir-field">
          <label>Police Station</label>
          <input type="text" placeholder="e.g. Sector 41 PS" value={form.police_station} onChange={set('police_station')} />
        </div>
      </div>

      {/* Full-width fields */}
      <div className="fir-field full">
        <label>Incident Description <span className="required">*</span></label>
        <textarea
          rows={4}
          placeholder="Describe what happened in detail — this is the most important field for analysis..."
          value={form.incident_description}
          onChange={set('incident_description')}
          required
        />
      </div>

      <div className="fir-field full">
        <label>Victim Impact</label>
        <textarea
          rows={2}
          placeholder="Physical injuries, psychological trauma, financial loss..."
          value={form.victim_impact}
          onChange={set('victim_impact')}
        />
      </div>

      <div className="fir-field full">
        <label>Evidence</label>
        <textarea
          rows={2}
          placeholder="CCTV footage, witnesses, medical reports, physical evidence..."
          value={form.evidence}
          onChange={set('evidence')}
        />
      </div>

      <div className="fir-form-actions">
        <button type="submit" className="btn btn-primary" disabled={disabled || !form.incident_description.trim()}>
          <Send size={16} />
          Analyze Case
        </button>
        <button type="button" className="btn btn-secondary" onClick={handleSampleFIR} disabled={disabled}>
          Use Sample FIR
        </button>
      </div>
    </form>
  );
}
