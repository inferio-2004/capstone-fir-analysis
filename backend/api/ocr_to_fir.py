#!/usr/bin/env python3
"""
ocr_to_fir.py
Simple OCR utility that accepts an image or PDF file and attempts to parse
FIR fields into the same JSON schema as `src/fir_sample.json`.

Notes:
- Requires Tesseract OCR installed on the system and `pytesseract` Python package.
- For PDFs, it uses `pdf2image` to convert pages to images (optional but recommended).
- Fallback: if fields can't be parsed, the full OCR text is placed into
  `incident_description` and other fields left empty or defaults applied.

Usage:
  python ocr_to_fir.py /path/to/fir_image.jpg output.json

Output: JSON with keys matching `src/fir_sample.json`.
"""

import sys
import json
import re
from pathlib import Path

# Optional imports (try to import; script will show informative error if missing)
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None

# pdf conversion optional
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None


FIELD_KEYS = [
    'fir_id', 'date', 'complainant_name', 'accused_names', 'victim_name',
    'incident_description', 'victim_impact', 'evidence', 'location', 'police_station'
]

# Simple helper: run OCR on an image (PIL Image)
def ocr_image_to_text(img):
    if pytesseract is None:
        raise RuntimeError('pytesseract is not installed. Install with `pip install pytesseract pillow` and ensure tesseract binary is installed on your system.')
    return pytesseract.image_to_string(img)


# Try to parse common FIR field labels from raw OCR text
def parse_fir_from_text(text):
    # Normalize whitespace
    txt = re.sub(r"\r", "", text)
    txt = re.sub(r"\n\s+", "\n", txt)

    # Heuristics: look for lines starting with label words
    fields = {k: None for k in FIELD_KEYS}

    # Common label patterns (case-insensitive)
    patterns = {
        'fir_id': r'\b(FIR[-_ ]?ID|FIR No\.|FIR No|FIR)\s*[:\-]?\s*(\S+)',
        'date': r'\b(Date|Dated)\s*[:\-]?\s*([0-3]?\d[\-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})[\-/ ]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})',
        'complainant_name': r'\b(Complainant|Complainant Name)\s*[:\-]?\s*([A-Z][A-Za-z \'`.-]{1,60})',
        'accused_names': r'\b(Accused|Accused Names?)\s*[:\-]?\s*(.+)',
        'victim_name': r'\b(Victim|Victim Name)\s*[:\-]?\s*([A-Z][A-Za-z \'`.-]{1,60})',
        'location': r'\b(Location|Place of Occurrence)\s*[:\-]?\s*(.+)',
        'police_station': r'\b(Police Station|PS)\s*[:\-]?\s*(.+)'
    }

    # Search for patterns
    for key, pat in patterns.items():
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            # last captured group should be the value
            val = m.groups()[-1].strip()
            if key == 'accused_names':
                # split common separators
                names = re.split(r'[,;]| and ', val)
                fields[key] = [n.strip() for n in names if n.strip()]
            else:
                fields[key] = val

    # For long fields like incident_description, victim_impact, evidence—try to find by keywords
    # Attempt to find a block starting at 'Incident' or 'Description' and ending before next label
    incident_match = re.search(r'(?:Incident(?: Description)?|Details)\s*[:\-]?\s*(.+?)(?:\n\s*(?:Victim Impact|Evidence|Location|Police Station|Complainant|Accused)\b)', txt, flags=re.IGNORECASE | re.DOTALL)
    if incident_match:
        fields['incident_description'] = incident_match.group(1).strip()

    victim_match = re.search(r'(?:Victim Impact|Impact)\s*[:\-]?\s*(.+?)(?:\n\s*(?:Evidence|Location|Police Station|Complainant|Accused)\b)', txt, flags=re.IGNORECASE | re.DOTALL)
    if victim_match:
        fields['victim_impact'] = victim_match.group(1).strip()

    evidence_match = re.search(r'(?:Evidence|Evidences)\s*[:\-]?\s*(.+?)(?:\n\s*(?:Location|Police Station|Complainant|Accused)\b)', txt, flags=re.IGNORECASE | re.DOTALL)
    if evidence_match:
        fields['evidence'] = evidence_match.group(1).strip()

    # Fallback: if incident_description still empty, take first 500-1000 chars as incident
    if not fields['incident_description']:
        # Try to find a paragraph with verbs like 'stole', 'attacked', 'assaulted'
        sent_match = re.search(r'(.{100,800}?\b(stole|robbed|assaulted|attacked|stolen|threatened|kidnapped)\b.{0,400})', txt, flags=re.IGNORECASE | re.DOTALL)
        if sent_match:
            fields['incident_description'] = sent_match.group(1).strip()
        else:
            fields['incident_description'] = txt.strip()[:1200]

    # Ensure accused_names is list
    if not fields['accused_names']:
        # try to find line starting with 'Accused' or 'Accused Names' in the text
        m = re.search(r'Accused\s*[:\-]?\s*(.+)', txt, flags=re.IGNORECASE)
        if m:
            fields['accused_names'] = [n.strip() for n in re.split(r'[,;]| and ', m.group(1)) if n.strip()]
        else:
            fields['accused_names'] = []

    # Post-process: trim values
    for k, v in list(fields.items()):
        if isinstance(v, str):
            fields[k] = v.strip()

    # Set defaults for missing keys
    if not fields.get('complainant_name'):
        fields['complainant_name'] = None
    if not fields.get('victim_name'):
        fields['victim_name'] = None

    return fields


def process_path_to_fir(input_path: Path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = ''
    # If PDF and pdf2image available
    if input_path.suffix.lower() in ['.pdf'] and convert_from_path is not None:
        images = convert_from_path(str(input_path))
        parts = []
        for img in images:
            parts.append(ocr_image_to_text(img))
        text = '\n'.join(parts)
    else:
        # Try to open as image
        try:
            if pytesseract is None:
                raise RuntimeError('pytesseract not installed')
            img = Image.open(str(input_path))
            text = ocr_image_to_text(img)
        except Exception as e:
            raise RuntimeError(f'Unable to OCR input file. Ensure it is an image or install pdf2image for PDFs. ({e})')

    # Parse fields
    parsed = parse_fir_from_text(text)
    # Build final JSON matching src/fir_sample.json schema
    fir_json = {
        'fir_id': parsed.get('fir_id') or 'UNKNOWN',
        'date': parsed.get('date') or None,
        'complainant_name': parsed.get('complainant_name'),
        'accused_names': parsed.get('accused_names') or [],
        'victim_name': parsed.get('victim_name'),
        'incident_description': parsed.get('incident_description'),
        'victim_impact': parsed.get('victim_impact') or '',
        'evidence': parsed.get('evidence') or '',
        'location': parsed.get('location') or '',
        'police_station': parsed.get('police_station') or ''
    }

    return fir_json


def main():
    if len(sys.argv) < 3:
        print('Usage: python ocr_to_fir.py <input-image-or-pdf> <output-json>')
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])

    try:
        fir = process_path_to_fir(inp)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(fir, f, indent=2, ensure_ascii=False)
        print(f'✓ FIR JSON written to: {out}')
    except Exception as e:
        print(f'✗ Error: {e}')
        sys.exit(2)


if __name__ == '__main__':
    main()
