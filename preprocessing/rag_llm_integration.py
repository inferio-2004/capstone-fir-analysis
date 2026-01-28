#!/usr/bin/env python3
"""
Statute-Aware RAG with Groq LLM Integration
FIR Analysis → Intent Identification → Legal Reasoning → Applicable Statutes
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
load_dotenv()

class StatuteRAGSystem:
    def __init__(self):
        """Initialize RAG system with vector DB, LLM, and statute data"""
        
        # Initialize Groq
        self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        print("✓ Groq client initialized")
        
        # Initialize Pinecone
        try:
            from pinecone import Pinecone
            api_key = os.environ.get('PINECONE_API_KEY')
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index('statute-embeddings')
            print("✓ Pinecone vector DB connected")
        except Exception as e:
            print(f"✗ Pinecone connection failed: {e}")
            raise
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        # Load statute chunks
        self.statute_chunks = self._load_statute_chunks()
        print(f"✓ Loaded {len(self.statute_chunks)} statute chunks")
        
        # Load mappings
        self.statute_mappings = self._load_statute_mappings()
        print(f"✓ Loaded {len(self.statute_mappings)} statute mappings")
        
        # Load negative rules
        self.negative_rules = self._load_negative_rules()
        print(f"✓ Loaded {len(self.negative_rules)} negative rules")
    
    def _load_statute_chunks(self):
        """Load statute chunks from JSONL file"""
        chunks = {}
        chunks_file = Path("output/statute_chunks_complete.jsonl")
        
        with open(chunks_file, 'r') as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk['chunk_id']] = chunk
        
        return chunks
    
    def _load_statute_mappings(self):
        """Load IPC-BNS statute mappings"""
        mappings_file = Path("output/ipc_bns_mappings_complete.json")
        
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        
        return mappings
    
    def _load_negative_rules(self):
        """Load negative rules for filtering"""
        rules_file = Path("output/negative_rules_comprehensive.json")
        
        with open(rules_file, 'r') as f:
            rules = json.load(f)
        
        return {rule['offence']: rule for rule in rules}
    
    def retrieve_relevant_statutes(self, query_text, top_k=8):
        """Retrieve relevant statute chunks from vector DB"""
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        retrieved_chunks = []
        for match in results['matches']:
            chunk_id = match['id']
            score = match['score']
            
            if chunk_id in self.statute_chunks:
                chunk = self.statute_chunks[chunk_id]
                retrieved_chunks.append({
                    'chunk_id': chunk_id,
                    'law': chunk['law'],
                    'section_id': chunk['section_id'],
                    'section_text': chunk['section_text'][:200],  # Preview
                    'full_text': chunk['section_text'],
                    'offence_type': chunk.get('offence_type', 'unknown'),
                    'similarity_score': score
                })
        
        return retrieved_chunks
    
    def apply_negative_rules_filter(self, chunks, case_facts):
        """Filter chunks using negative rules"""
        
        filtered_chunks = []
        
        for chunk in chunks:
            offence_type = chunk['offence_type']
            
            # Check if offence has negative rules
            for rule_name, rule in self.negative_rules.items():
                # Simple heuristic: if chunk text mentions rule name, apply rule
                if rule_name.lower() in chunk['full_text'].lower():
                    # Check against failure examples
                    failure_match = False
                    for failure_example in rule.get('failure_examples', []):
                        if any(fact.lower() in failure_example.lower() 
                               for fact in case_facts.values() if isinstance(fact, str)):
                            failure_match = True
                            break
                    
                    if not failure_match:
                        filtered_chunks.append(chunk)
                        break
            else:
                # No matching rule, keep chunk
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def find_corresponding_sections(self, law, section_id):
        """Find corresponding IPC/BNS section using statute mappings"""
        
        corresponding = []
        
        for mapping in self.statute_mappings:
            if law == 'IPC' and mapping['ipc'] == section_id:
                # This is an IPC section, find its BNS counterpart
                corresponding.append({
                    'law': 'BNS',
                    'section_id': mapping['bns']
                })
            elif law == 'BNS' and mapping['bns'] == section_id:
                # This is a BNS section, find its IPC counterpart
                corresponding.append({
                    'law': 'IPC',
                    'section_id': mapping['ipc']
                })
        
        return corresponding
    
    def get_section_extract(self, law, section_id):
        """Get text extract from a statute section"""
        
        # Try different chunk ID formats
        chunk_id_formats = [
            f"{law.lower()}_{section_id}",
            f"{law}_{section_id}",
            f"{law.lower()}{section_id}",
            f"{law}{section_id}"
        ]
        
        for chunk_id in chunk_id_formats:
            if chunk_id in self.statute_chunks:
                chunk = self.statute_chunks[chunk_id]
                # Return first 150 chars of section text as extract
                return chunk['section_text'][:150].replace('\n', ' ')
        
        # If not found, try to search by section ID in all chunks
        for cid, chunk in self.statute_chunks.items():
            if chunk['section_id'] == section_id and chunk['law'] == law:
                return chunk['section_text'][:150].replace('\n', ' ')
        
        # Return message indicating section not in extracted database
        return f"[{law} {section_id} not in extracted database - check official government source]"
    
    def identify_intent(self, fir_data):
        """Use Groq llama model for intent identification"""
        
        # Prepare FIR context
        fir_summary = f"""
Case Details:
- Complainant: {fir_data.get('complainant_name', 'Unknown')}
- Accused: {fir_data.get('accused_names', 'Unknown')}
- Incident: {fir_data.get('incident_description', 'Unknown')}
- Victim Impact: {fir_data.get('victim_impact', 'Unknown')}
- Evidence: {fir_data.get('evidence', 'Unknown')}
"""
        
        print("\n[1] Identifying Intent (llama-3.1-8b-instant)...")
        
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a legal analyst. Identify the PRIMARY intent/crime from the FIR.
                    Return JSON with: primary_intent, confidence (0-1), secondary_intents (list)"""
                },
                {
                    "role": "user",
                    "content": fir_summary
                }
            ],
            temperature=0.3,
            max_completion_tokens=500,
            response_format={"type": "json_object"}
        )
        
        intent_result = json.loads(response.choices[0].message.content or "{}")
        print(f"✓ Intent identified: {intent_result.get('primary_intent', 'Unknown')}")
        
        return intent_result
    
    def legal_reasoning(self, fir_data, retrieved_chunks, intent_result):
        """Use Groq gpt model for legal reasoning"""
        
        # Prepare context
        chunks_context = "\n".join([
            f"[{chunk['law']} {chunk['section_id']}] {chunk['section_text']}"
            for chunk in retrieved_chunks[:5]  # Top 5 chunks
        ])
        
        fir_summary = f"""
Incident: {fir_data.get('incident_description', 'Unknown')}
Accused: {fir_data.get('accused_names', 'Unknown')}
Victim: {fir_data.get('victim_name', 'Unknown')}
Identified Intent: {intent_result.get('primary_intent', 'Unknown')}
"""
        
        print("\n[2] Legal Reasoning (openai/gpt-oss-120b)...")
        
        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a legal expert analyzing applicable criminal statutes.
                    Given case facts and relevant statute sections, determine which statutes are APPLICABLE.
                    Return JSON with:
                    - applicable_statutes: [{section, law, reasoning}]
                    - legal_basis: explanation of why these apply
                    - severity_assessment: high/medium/low
                    - confidence: 0-1"""
                },
                {
                    "role": "user",
                    "content": f"""
CASE FACTS:
{fir_summary}

RELEVANT STATUTE SECTIONS:
{chunks_context}

Analyze which statutes apply to this case and explain your reasoning.
"""
                }
            ],
            temperature=0.4,
            max_completion_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        reasoning_result = json.loads(response.choices[0].message.content or "{}")
        print(f"✓ Legal reasoning completed")
        
        return reasoning_result
    
    def analyze_fir(self, fir_data):
        """Complete FIR analysis pipeline"""
        
        print("\n" + "="*80)
        print("STATUTE-AWARE RAG ANALYSIS PIPELINE")
        print("="*80)
        
        # Extract case facts for retrieval
        case_facts = {
            'incident': fir_data.get('incident_description', ''),
            'victim_impact': fir_data.get('victim_impact', ''),
            'evidence': fir_data.get('evidence', '')
        }
        
        # Step 1: Retrieve relevant statutes
        print("\n[0] Retrieving relevant statute sections from vector DB...")
        query_text = f"{fir_data.get('incident_description', '')} {fir_data.get('victim_impact', '')}"
        retrieved_chunks = self.retrieve_relevant_statutes(query_text, top_k=10)
        print(f"✓ Retrieved {len(retrieved_chunks)} statute chunks")
        
        # Step 2: Apply negative rules filter
        print("\n[1.5] Applying negative rules filter...")
        filtered_chunks = self.apply_negative_rules_filter(retrieved_chunks, case_facts)
        print(f"✓ Filtered to {len(filtered_chunks)} applicable chunks")
        
        # Step 3: Identify intent
        intent_result = self.identify_intent(fir_data)
        
        # Step 4: Legal reasoning
        reasoning_result = self.legal_reasoning(fir_data, filtered_chunks, intent_result)
        
        # Step 5: Enrich applicable statutes with mappings and extracts
        enriched_statutes = []
        for statute in reasoning_result.get('applicable_statutes', []):
            # Extract section number from statute['section'] (e.g., "IPC 394" → "394")
            section_parts = statute.get('section', '').strip().split()
            if len(section_parts) >= 2:
                law = section_parts[0]
                section_id = section_parts[1]
            else:
                # Fallback: try to parse "IPC 394"
                section_str = statute.get('section', 'IPC 394')
                parts = section_str.split()
                law = parts[0] if parts else 'IPC'
                section_id = parts[-1] if parts else '394'
            
            # Find corresponding section
            corresponding = self.find_corresponding_sections(law, section_id)
            
            # Get extract from current section
            extract = self.get_section_extract(law, section_id)
            
            enriched_statute = {
                'primary': {
                    'law': law,
                    'section': section_id,
                    'title': statute.get('law', ''),
                    'reasoning': statute.get('reasoning', ''),
                    'extract': extract
                },
                'corresponding_sections': []
            }
            
            # Add corresponding sections with extracts
            for corr in corresponding:
                corr_extract = self.get_section_extract(corr['law'], corr['section_id'])
                enriched_statute['corresponding_sections'].append({
                    'law': corr['law'],
                    'section': corr['section_id'],
                    'extract': corr_extract
                })
            
            enriched_statutes.append(enriched_statute)
        
        # Step 6: Format final output
        final_output = {
            'status': 'success',
            'timestamp': str(Path('.').absolute()),
            'fir_summary': {
                'complainant': fir_data.get('complainant_name'),
                'accused': fir_data.get('accused_names'),
                'incident': fir_data.get('incident_description')[:150]
            },
            'analysis': {
                'intent_identification': intent_result,
                'legal_reasoning': reasoning_result
            },
            'retrieved_data': {
                'total_chunks_retrieved': len(retrieved_chunks),
                'chunks_after_filtering': len(filtered_chunks),
                'top_chunks_used': [
                    {
                        'law': chunk['law'],
                        'section': chunk['section_id'],
                        'score': round(chunk['similarity_score'], 4),
                        'preview': chunk['section_text'][:100]
                    }
                    for chunk in filtered_chunks[:3]
                ]
            },
            'applicable_statutes': enriched_statutes,
            'severity': reasoning_result.get('severity_assessment', 'unknown'),
            'confidence': reasoning_result.get('confidence', 0)
        }
        
        return final_output


def create_sample_fir():
    """Create sample FIR for testing"""
    
    return {
        'fir_id': 'FIR-2024-00123',
        'date': '2024-01-10',
        'complainant_name': 'Rajesh Kumar',
        'accused_names': ['Amit Singh', 'Vikram Patel'],
        'victim_name': 'Priya Sharma',
        'incident_description': '''
        On 2024-01-09 at approximately 11:30 PM, the victim was returning home from her office
        when she was approached by two armed men. They forcefully grabbed her, threatened her with
        a knife, and stole her mobile phone and gold jewelry worth approximately Rs. 50,000.
        The victim suffered multiple cuts and bruises. The perpetrators fled the scene in a motorcycle.
        CCTV footage from nearby shops shows the incident clearly.
        ''',
        'victim_impact': '''
        Victim suffered physical injuries (multiple cuts, contusions on arms and face).
        Psychological trauma and fear. Loss of valuables including jewelry and phone.
        Unable to work for 3 days due to injuries.
        ''',
        'evidence': '''
        - CCTV footage showing incident (2 minutes, clear faces)
        - Medical report showing injuries
        - Complaint to mobile phone provider for phone tracking
        - FIR witness: street vendor present at scene
        - Physical evidence: torn portion of victim's dupatta found at scene
        ''',
        'location': 'Sector 45, Chandigarh',
        'police_station': 'Sector 41 Police Station'
    }


def main():
    """Main execution"""
    
    print("="*80)
    print("STATUTE-AWARE RAG SYSTEM WITH GROQ LLM INTEGRATION")
    print("="*80)
    
    try:
        # Initialize system
        print("\n[INIT] Initializing RAG System...")
        rag_system = StatuteRAGSystem()
        
        # Create sample FIR
        print("\n[INPUT] Loading sample FIR...")
        sample_fir = create_sample_fir()
        
        print("\nSample FIR Details:")
        print(f"  Complainant: {sample_fir['complainant_name']}")
        print(f"  Accused: {', '.join(sample_fir['accused_names'])}")
        print(f"  Incident: {sample_fir['incident_description'][:80]}...")
        
        # Run analysis
        print("\n[ANALYSIS] Running FIR Analysis Pipeline...")
        result = rag_system.analyze_fir(sample_fir)
        
        # Save result
        output_file = Path("output/fir_analysis_result.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Analysis saved to: {output_file}")
        
        # Display result
        print("\n" + "="*80)
        print("FINAL ANALYSIS RESULT")
        print("="*80)
        
        print(f"\n[Primary Intent] {result['analysis']['intent_identification'].get('primary_intent', 'Unknown')}")
        print(f"[Confidence] {result['analysis']['intent_identification'].get('confidence', 0)}")
        
        print(f"\n[Legal Basis] {result['analysis']['legal_reasoning'].get('legal_basis', 'N/A')}")
        print(f"[Severity] {result['severity']}")
        
        print(f"\n[Applicable Statutes]")
        for statute in result.get('applicable_statutes', [])[:3]:
            primary = statute.get('primary', {})
            print(f"\n  {primary.get('law')} {primary.get('section')}:")
            print(f"    Title: {primary.get('title', 'N/A')}")
            print(f"    Reasoning: {primary.get('reasoning', 'N/A')[:120]}...")
            print(f"    Extract: {primary.get('extract', 'N/A')[:120]}...")
            
            # Show corresponding sections
            corresponding = statute.get('corresponding_sections', [])
            if corresponding:
                print(f"    Corresponding Sections:")
                for corr in corresponding:
                    print(f"      • {corr['law']} {corr['section']}: {corr['extract'][:100]}...")
        
        print(f"\n[Chunks Retrieved] {result['retrieved_data']['total_chunks_retrieved']}")
        print(f"[Chunks After Filtering] {result['retrieved_data']['chunks_after_filtering']}")
        
        print(f"\n[Top Matching Statutes]")
        for chunk in result['retrieved_data']['top_chunks_used'][:3]:
            print(f"  - {chunk['law']} {chunk['section']} (score: {chunk['score']})")
            print(f"    {chunk['preview']}...")
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
