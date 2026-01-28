import json
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
chunks_file = repo_root / 'output' / 'statute_chunks_complete.jsonl'
chunks = {}
with open(chunks_file,'r',encoding='utf-8') as f:
    for line in f:
        c=json.loads(line)
        chunks[c['chunk_id']]=c

ipc = [c for c in chunks.values() if c.get('law')=='IPC']
bns = [c for c in chunks.values() if c.get('law')=='BNS']
print(f'IPC chunks: {len(ipc)}')
print(f'BNS chunks: {len(bns)}')

ipc_ids = [int(c['section_id']) for c in ipc if c['section_id'].isdigit()]
print(f'Max numeric IPC section id: {max(ipc_ids) if ipc_ids else None}')

exists_390 = any(c['section_id']=='390' and c['law']=='IPC' for c in chunks.values())
exists_394 = any(c['section_id']=='394' and c['law']=='IPC' for c in chunks.values())
print('IPC 390 present:', exists_390)
print('IPC 394 present:', exists_394)

mappings = json.load(open(repo_root / 'output' / 'ipc_bns_mappings_complete.json','r',encoding='utf-8'))
bns_ids = set([c['section_id'] for c in bns])
mapped_bns = [m for m in mappings if m.get('bns') in bns_ids]
print('Total mappings:', len(mappings))
print('Mappings where BNS text exists:', len(mapped_bns))
print('Mappings where BNS text missing:', len(mappings)-len(mapped_bns))
