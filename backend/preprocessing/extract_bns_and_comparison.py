#!/usr/bin/env python3
"""
Extract BNS 2023 sections and BNS↔IPC comparative table.
Saves:
- output/bns_sections_extracted.json (dict: section_id -> text)
- output/bns_ipc_comparative.json (list of mappings)

Run: python backend/preprocessing/extract_bns_and_comparison.py
"""
import re
import json
from pathlib import Path
from PyPDF2 import PdfReader

repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / 'src_dataset_files'
output_dir = repo_root / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

bns_pdf = src_dir / 'bns.pdf'
comparison_pdf = src_dir / 'COMPARISON SUMMARY BNS to IPC .pdf'

# Read BNS PDF
print('Reading BNS PDF:', bns_pdf)
reader = PdfReader(str(bns_pdf))
all_text = ''
for p in reader.pages:
    t = p.extract_text() or ''
    all_text += '\n' + t

# Normalize whitespace but keep line breaks
all_text = re.sub(r'\r', '', all_text)
# Ensure consistent newlines
all_text = re.sub(r'\n[ \t]+', '\n', all_text)
# Join digit runs that PDF extraction split (e.g. '15 1' -> '151')
# This helps avoid splitting section numbers like 151 into 15 and 1
all_text = re.sub(r'(?<=\d)\s+(?=\d)', '', all_text)

# Find potential section starts: newline + optional space + number (1-3 digits)
starts = list(re.finditer(r'\n\s*(\d{1,3})(?=[\.\(\s])', all_text))

sections = {}
for i, m in enumerate(starts):
    sec_id = m.group(1)
    start_idx = m.start()
    end_idx = starts[i+1].start() if i+1 < len(starts) else len(all_text)
    sec_text = all_text[start_idx:end_idx].strip()
    # Clean: remove leading number token
    sec_text = re.sub(r'^\s*' + re.escape(sec_id) + r'\s*[\)\.\(\-:\s]*', '', sec_text.strip(), count=1)
    # Collapse multiple spaces
    sec_text = re.sub(r'\s+', ' ', sec_text).strip()
    sections[sec_id] = sec_text

print('Found possible section starts:', len(sections))

# Heuristic: keep only sections with reasonable length
valid_sections = {k:v for k,v in sections.items() if len(v) > 30}
print('Valid sections (len>30):', len(valid_sections))

# Save extracted sections
bns_out = output_dir / 'bns_sections_extracted.json'
with open(bns_out, 'w', encoding='utf-8') as f:
    json.dump(valid_sections, f, indent=2, ensure_ascii=False)
print('Saved BNS sections to', bns_out)

# Now parse comparison PDF for BNS↔IPC table
print('Reading comparison PDF:', comparison_pdf)
reader2 = PdfReader(str(comparison_pdf))
comp_text = ''
for p in reader2.pages:
    comp_text += '\n' + (p.extract_text() or '')

# Normalize
comp_text = re.sub(r'\r','', comp_text)
# Join digit runs that PDF extraction split (e.g. '1 24' -> '124')
comp_text = re.sub(r'(?<=\d)\s+(?=\d)', '', comp_text)

# Heuristic: collapse wrapped lines belonging to the same table row.
# We'll join consecutive lines until the buffer ends with an IPC-like number (1-3 digits).
raw_lines = [ln.strip() for ln in comp_text.split('\n') if ln.strip()]
lines = []
buf = ''
for ln in raw_lines:
    if not buf:
        buf = ln
    else:
        buf = buf + ' ' + ln
    # If buffer ends with a 1-3 digit number (optionally followed by punctuation), treat as row end
    if re.search(r'\b\d{1,3}[\.,:]?$', buf):
        lines.append(buf.strip())
        buf = ''
# If any remainder, append it
if buf:
    lines.append(buf.strip())

mappings = []
# Try to find lines starting with BNS section like '2(2)' or '2(3)' or '23'
for ln in lines:
    # Pattern: BNS id then maybe title then IPC numbers
    # Examples in PDF: '2(2)  "animal" . 47' or '1(1) Short title... 1'
    m = re.match(r'^(\d+(?:\(\d+\))?)\s+(.+?)\s+(\d{1,3}(?:[ ,]\s*\d{1,3})*)$', ln)
    if m:
        bns_id = m.group(1)
        summary = m.group(2).strip()
        ipc_str = m.group(3)
        ipc_numbers = [x.strip() for x in re.split(r'[ ,]+', ipc_str) if x.strip()]
        mappings.append({'bns': bns_id, 'ipc': ipc_numbers, 'summary': summary, 'source_line': ln})

# If above didn't capture enough, fallback: search for patterns within lines
if len(mappings) < 50:
    # Find any occurrence of pattern like '2(2)  ...  47'
    for ln in lines:
        m = re.search(r'(\d+(?:\(\d+\))?).{0,80}?(\d{1,3}(?:[ ,]\s*\d{1,3})*)', ln)
        if m:
            bns_id = m.group(1)
            ipc_str = m.group(2)
            ipc_numbers = [x.strip() for x in re.split(r'[ ,]+', ipc_str) if x.strip()]
            mappings.append({'bns': bns_id, 'ipc': ipc_numbers, 'summary': ln, 'source_line': ln})

# Deduplicate mappings by bns+ipc
uniq = {}
for entry in mappings:
    key = (entry['bns'], ','.join(entry['ipc']))
    if key not in uniq:
        uniq[key] = entry
mappings = list(uniq.values())

comp_out = output_dir / 'bns_ipc_comparative.json'
with open(comp_out, 'w', encoding='utf-8') as f:
    json.dump(mappings, f, indent=2, ensure_ascii=False)

print('Saved comparative mappings to', comp_out)
print('Total comparative mappings extracted:', len(mappings))

# Report counts
print('\nREPORT:')
print('Total section starts found:', len(starts))
print('Valid extracted sections:', len(valid_sections))
print('Comparative mappings extracted:', len(mappings))
