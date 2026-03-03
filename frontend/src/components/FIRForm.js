import React, { useState } from 'react';
import { Send, ChevronDown, ChevronUp } from 'lucide-react';
import './FIRForm.css';

const EMPTY = {
  complainant_name: '',
  father_husband_name: '',
  complainant_dob: '',
  nationality: 'Indian',
  occupation: '',
  complainant_address: '',
  accused_names_str: '',
  victim_name: '',
  incident_description: '',
  victim_impact: '',
  evidence: '',
  location: '',
  district: '',
  police_station: '',
  date: new Date().toISOString().slice(0, 10),
  delay_reason: '',
  properties_stolen: '',
  property_value: '',
};

export default function FIRForm({ onSubmit, disabled, hasAnalysis }) {
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
      father_husband_name: form.father_husband_name || null,
      complainant_dob: form.complainant_dob || null,
      nationality: form.nationality || 'Indian',
      occupation: form.occupation || null,
      complainant_address: form.complainant_address || null,
      accused_names: form.accused_names_str
        ? form.accused_names_str.split(',').map(s => s.trim()).filter(Boolean)
        : [],
      victim_name: form.victim_name || null,
      incident_description: form.incident_description,
      victim_impact: form.victim_impact,
      evidence: form.evidence,
      location: form.location,
      district: form.district || null,
      police_station: form.police_station,
      delay_reason: form.delay_reason || null,
      properties_stolen: form.properties_stolen || null,
      property_value: form.property_value || null,
    };

    onSubmit(fir);
    setExpanded(false);
  };

  const handleSampleFIR = () => {
    onSubmit(null);
    setExpanded(false);
  };

  if (!expanded) {
    return (
      <div className="fir-form-collapsed" onClick={() => setExpanded(true)}>
        <span className="fir-form-collapsed-text">
          {hasAnalysis ? 'Edit FIR & re-analyze' : 'FIR submitted — click to edit & resubmit'}
        </span>
        <ChevronDown size={14} style={{ marginLeft: 6, opacity: 0.5 }} />
      </div>
    );
  }

  return (
    <form className="fir-form" onSubmit={handleSubmit}>
      <div className="fir-form-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h3>{hasAnalysis ? 'Edit & Re-analyze Case' : 'Submit a Case for Analysis'}</h3>
          {hasAnalysis && (
            <button type="button" className="btn btn-secondary" style={{ padding: '4px 10px', fontSize: 12 }} onClick={() => setExpanded(false)}>
              <ChevronUp size={14} /> Collapse
            </button>
          )}
        </div>
        <p>Fill in the case details below, or use the sample FIR to try the system.</p>
      </div>

      {/* ---- Case core ---- */}
      <div className="fir-form-grid">
        <div className="fir-field">
          <label>Complainant Name</label>
          <input type="text" placeholder="e.g. Rajesh Kumar" value={form.complainant_name} onChange={set('complainant_name')} />
        </div>
        <div className="fir-field">
          <label>Father / Husband Name</label>
          <input type="text" placeholder="e.g. Suresh Kumar" value={form.father_husband_name} onChange={set('father_husband_name')} />
        </div>

        <div className="fir-field">
          <label>Complainant DOB</label>
          <input type="date" value={form.complainant_dob} onChange={set('complainant_dob')} />
        </div>
        <div className="fir-field">
          <label>Nationality</label>
          <input type="text" placeholder="Indian" value={form.nationality} onChange={set('nationality')} />
        </div>

        <div className="fir-field">
          <label>Occupation</label>
          <input type="text" placeholder="e.g. Software Engineer" value={form.occupation} onChange={set('occupation')} />
        </div>
        <div className="fir-field">
          <label>Complainant Address</label>
          <input type="text" placeholder="e.g. 12, MG Road, New Delhi" value={form.complainant_address} onChange={set('complainant_address')} />
        </div>
      </div>

      <div className="fir-form-grid">
        <div className="fir-field">
          <label>Victim Name</label>
          <input type="text" placeholder="e.g. Priya Sharma" value={form.victim_name} onChange={set('victim_name')} />
        </div>
        <div className="fir-field">
          <label>Accused Names <span className="hint">(comma-separated)</span></label>
          <input type="text" placeholder="e.g. Amit Singh, Vikram Patel" value={form.accused_names_str} onChange={set('accused_names_str')} />
        </div>

        <div className="fir-field">
          <label>Date of Incident</label>
          <input type="date" value={form.date} onChange={set('date')} />
        </div>
        <div className="fir-field">
          <label>Location / Place of Occurrence</label>
          <input type="text" placeholder="e.g. Sector 45, Chandigarh" value={form.location} onChange={set('location')} />
        </div>

        <div className="fir-field">
          <label>District</label>
          <input type="text" placeholder="e.g. Central Delhi" value={form.district} onChange={set('district')} />
        </div>
        <div className="fir-field">
          <label>Police Station</label>
          <input type="text" placeholder="e.g. Sector 41 PS" value={form.police_station} onChange={set('police_station')} />
        </div>
      </div>

      {/* ---- Full-width fields ---- */}
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

      {/* ---- Property & delay ---- */}
      <div className="fir-form-grid">
        <div className="fir-field">
          <label>Properties Stolen / Involved</label>
          <input type="text" placeholder="e.g. Gold chain, mobile phone" value={form.properties_stolen} onChange={set('properties_stolen')} />
        </div>
        <div className="fir-field">
          <label>Total Value of Property</label>
          <input type="text" placeholder="e.g. ₹50,000" value={form.property_value} onChange={set('property_value')} />
        </div>
      </div>

      <div className="fir-field full">
        <label>Reason for Delay in Reporting <span className="hint">(if any)</span></label>
        <input type="text" placeholder="e.g. Complainant was hospitalised" value={form.delay_reason} onChange={set('delay_reason')} />
      </div>

      <div className="fir-form-actions">
        <button type="submit" className="btn btn-primary" disabled={disabled || !form.incident_description.trim()}>
          <Send size={16} />
          {hasAnalysis ? 'Re-analyze Case' : 'Analyze Case'}
        </button>
        <button type="button" className="btn btn-secondary" onClick={handleSampleFIR} disabled={disabled}>
          Use Sample FIR
        </button>
      </div>
    </form>
  );
}
