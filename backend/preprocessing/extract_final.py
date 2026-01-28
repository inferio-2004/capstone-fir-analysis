"""
Complete extraction of BNS->IPC mappings with proper handling of:
1. Subsection formats: 70(2), 8(6)(a), 2(1), etc.
2. IPC formats: 376DB, 171-I, 489A, 376(A), etc.
3. Ranges: "237 to 241"
4. Lists: "242, 243, 252"
5. Slash-separated: "233 / 234 / 235"
"""
import re
import json
from pathlib import Path
import PyPDF2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "src" / "COMPARISON SUMMARY BNS to IPC .pdf"
OUTPUT_PATH = PROJECT_ROOT / "output" / "bns_ipc_mappings_final.json"

def expand_range(start, end):
    """Expand 237-241 to [237,238,239,240,241]."""
    try:
        s_match = re.match(r'(\d+)', start)
        e_match = re.match(r'(\d+)', end)
        if s_match and e_match:
            s = int(s_match.group(1))
            e = int(e_match.group(1))
            if s <= e and e - s < 20:
                return [str(i) for i in range(s, e + 1)]
    except:
        pass
    return [start, end]

def parse_ipc_cell(text):
    """Parse IPC cell text, handling all formats."""
    if not text:
        return []
    
    text = text.strip()
    result = []
    
    # Handle slash-separated: "233 / 234 / 235 / 489D"
    if '/' in text and 'to' not in text.lower():
        parts = re.split(r'\s*/\s*', text)
        for p in parts:
            p = p.strip()
            # Match: 489D, 376DB, 171-I, 376(A), 8(6)(a), etc.
            m = re.match(r'^(\d+(?:\(\d+\))?(?:\([a-zA-Z]\))?[A-Z]*(?:-[A-Z])?)$', p)
            if m:
                result.append(m.group(1))
        return result
    
    # Handle comma-separated with potential ranges
    parts = re.split(r',\s*', text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check for range: "237 to 241"
        range_match = re.match(r'^(\d+[A-Z]*)\s*to\s*(\d+[A-Z]*)$', part, re.IGNORECASE)
        if range_match:
            result.extend(expand_range(range_match.group(1), range_match.group(2)))
        else:
            # Single section: 489D, 376DB, 171-I, 376(A), 197(1)(d), etc.
            m = re.match(r'^(\d+(?:\(\d+\))*(?:\([a-zA-Z]\))?[A-Z]*(?:-[A-Z])?)$', part)
            if m:
                result.append(m.group(1))
    
    return result

def extract_from_pdf():
    """Extract all BNS->IPC mappings from PDF."""
    
    with open(PDF_PATH, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"
    
    mappings = []
    
    # BNS section pattern: matches 70(2), 2(1), 178, etc.
    # Format in PDF: "BNS_SECTION Title. IPC_SECTION(S) Description"
    bns_pattern = r'^(\d+(?:\(\d+\))*(?:\([a-zA-Z]\))?)\s+([A-Z][^\.]*(?:\.[^\d])?)'
    
    lines = full_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty, copyright, chapter headers
        if not line or '©' in line or 'Anil Kishore' in line:
            i += 1
            continue
        if re.match(r'^(CHAPTER|PART)', line, re.IGNORECASE):
            i += 1
            continue
        
        # Match BNS entry
        bns_match = re.match(bns_pattern, line)
        if bns_match:
            bns_num = bns_match.group(1)
            rest = bns_match.group(2)
            
            # Accumulate text until next BNS entry
            block_text = rest + " " + line[len(bns_match.group(0)):]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                # Check if next line starts new BNS entry
                if re.match(r'^\d+(?:\(\d+\))*(?:\([a-zA-Z]\))?\s+[A-Z]', next_line):
                    break
                if next_line and '©' not in next_line:
                    block_text += " " + next_line
                j += 1
            
            # Extract IPC references from block
            # IPC pattern: numbers optionally followed by letters, parentheses, hyphens
            # Look for: "376DB", "171-I", "489A", "237 to 241", "242, 243"
            
            # Find IPC field - typically appears after title, before description
            # Pattern: sequence of IPC refs separated by to/,//
            ipc_pattern = r'(\d+(?:\(\d+\))*(?:\([a-zA-Z]\))?[A-Z]*(?:-[A-Z])?(?:\s*(?:to|/|,)\s*\d+(?:\(\d+\))*(?:\([a-zA-Z]\))?[A-Z]*(?:-[A-Z])?)*)'
            
            matches = re.findall(ipc_pattern, block_text)
            
            all_ipcs = []
            low_num_ipcs = []  # Collect low numbers separately
            
            for m in matches:
                m = m.strip()
                # Skip if it's exactly the full BNS number (e.g., "1(1)")
                if m == bns_num:
                    continue
                
                parsed = parse_ipc_cell(m)
                for ipc in parsed:
                    # For subsections like 1(1), don't skip IPC 1
                    # Only skip if ipc equals the FULL bns_num
                    if ipc == bns_num:
                        continue
                    
                    # Validate: IPC sections are 1-511 (with possible suffixes)
                    num_match = re.match(r'(\d+)', ipc)
                    if num_match:
                        num = int(num_match.group(1))
                        if 1 <= num <= 511:
                            # Low numbers (1-52) are valid IPC definitions
                            # but can also be false positives from text
                            if num <= 52:
                                low_num_ipcs.append(ipc)
                            else:
                                all_ipcs.append(ipc)
            
            # Include low numbers only if:
            # 1. There are no high numbers (entry is about definitions)
            # 2. Or the low number has a suffix (like 52A)
            if not all_ipcs and low_num_ipcs:
                # Likely a definitions section, include all
                all_ipcs = low_num_ipcs
            else:
                # Only include low numbers with suffixes
                for ipc in low_num_ipcs:
                    if not re.match(r'^\d+$', ipc):  # Has suffix
                        all_ipcs.append(ipc)
            
            # Deduplicate and sort
            if all_ipcs:
                unique = list(dict.fromkeys(all_ipcs))  # Preserve order, remove dupes
                # Sort by numeric value then suffix
                def sort_key(x):
                    m = re.match(r'(\d+)', x)
                    return (int(m.group(1)) if m else 999, x)
                sorted_ipcs = sorted(set(unique), key=sort_key)
                
                # Extract title (text before IPC numbers)
                title_match = re.match(r'^([A-Za-z][^0-9]*)', block_text)
                title = title_match.group(1).strip() if title_match else ""
                # Clean up title
                title = re.sub(r'\s+', ' ', title).strip()
                if title.endswith('.'):
                    title = title[:-1]
                
                mappings.append({
                    'bns': bns_num,
                    'title': title,
                    'ipc': sorted_ipcs
                })
            else:
                # No IPC found - might be a new section
                title_match = re.match(r'^([A-Za-z][^0-9]*)', block_text)
                title = title_match.group(1).strip() if title_match else ""
                title = re.sub(r'\s+', ' ', title).strip()
                if title.endswith('.'):
                    title = title[:-1]
                
                mappings.append({
                    'bns': bns_num,
                    'title': title,
                    'ipc': []
                })
            
            i = j
        else:
            i += 1
    
    return mappings

def main():
    print("Extracting BNS->IPC mappings with subsection support...")
    
    mappings = extract_from_pdf()
    
    # Deduplicate by BNS (keep entry with most IPCs)
    bns_best = {}
    for m in mappings:
        bns = m['bns']
        if bns not in bns_best or len(m['ipc']) > len(bns_best[bns]['ipc']):
            bns_best[bns] = m
    
    final_mappings = list(bns_best.values())
    
    # Sort by BNS number
    def bns_sort_key(entry):
        bns = entry['bns']
        m = re.match(r'(\d+)', bns)
        num = int(m.group(1)) if m else 999
        return (num, bns)
    
    final_mappings.sort(key=bns_sort_key)
    
    print(f"Total unique BNS sections: {len(final_mappings)}")
    
    # Stats
    empty = [m for m in final_mappings if not m['ipc']]
    multi = [m for m in final_mappings if len(m['ipc']) > 3]
    subsec = [m for m in final_mappings if '(' in m['bns']]
    
    print(f"BNS with subsections (like 70(2)): {len(subsec)}")
    print(f"Empty IPC: {len(empty)}")
    print(f">3 IPC: {len(multi)}")
    
    # Show subsection examples
    print("\n=== BNS Subsection Examples ===")
    for m in final_mappings:
        if '(' in m['bns']:
            print(f"  BNS {m['bns']} -> IPC {m['ipc']}")
            if len([x for x in final_mappings if '(' in x['bns'] and final_mappings.index(x) <= final_mappings.index(m)]) >= 10:
                break
    
    # Verify key mappings
    print("\n=== Key Mappings (178-181) ===")
    for m in final_mappings:
        if m['bns'] in ['178', '179', '180', '181']:
            print(f"  BNS {m['bns']}: {m['ipc']}")
    
    # Check 70(1) and 70(2)
    print("\n=== BNS 70 subsections ===")
    for m in final_mappings:
        if m['bns'].startswith('70'):
            print(f"  BNS {m['bns']}: {m['ipc']}")
    
    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_mappings, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
