import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
chunks = [json.loads(l) for l in open(REPO / "output" / "statute_chunks_complete.jsonl")]
keywords = ["murder", "culpable homicide", "hurt", "grievous hurt", "death caused", "voluntarily causing"]
relevant = [c for c in chunks if any(kw in c.get("full_text", "").lower()[:300] for kw in keywords)]
print(f"Relevant chunks: {len(relevant)}")
for c in relevant[:15]:
    print(f"  {c['chunk_id']}: {c['full_text'][:80]}")
