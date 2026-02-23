#!/usr/bin/env python3
"""
Statute-Aware RAG with LangChain Chain Prompting
FIR Analysis → Intent Classification → Legal Reasoning → Applicable Statutes
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

class StatuteRAGChainSystem:
    def __init__(self):
        """Initialize RAG system with LangChain chain prompting"""
        
        # Initialize Groq LLMs for different stages
        self.llm_intent = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.environ.get('GROQ_API_KEY')
        )
        
        self.llm_reasoning = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.4,
            api_key=os.environ.get('GROQ_API_KEY')
        )
        
        print("✓ Groq LLMs initialized (intent + reasoning)")
        
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
        
        # Create chain prompts
        self._create_chain_prompts()
    
    def _load_statute_chunks(self):
        """Load statute chunks from JSONL file"""
        chunks = {}
        repo_root = Path(__file__).resolve().parents[2]
        chunks_file = repo_root / 'output' / 'statute_chunks_complete.jsonl'

        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk['chunk_id']] = chunk
        
        return chunks
    
    def _load_statute_mappings(self):
        """Load IPC-BNS statute mappings"""
        repo_root = Path(__file__).resolve().parents[2]
        mappings_file = repo_root / 'output' / 'ipc_bns_mappings_complete.json'

        with open(mappings_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        return mappings
    
    def _load_negative_rules(self):
        """Load negative rules for filtering"""
        repo_root = Path(__file__).resolve().parents[2]
        rules_file = repo_root / 'output' / 'negative_rules_comprehensive.json'

        with open(rules_file, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        return {rule['offence']: rule for rule in rules}
    
    def _create_chain_prompts(self):
        """Create LangChain prompts for each stage"""
        
        # STAGE 1: Intent Identification Prompt
        self.intent_prompt = PromptTemplate(
            input_variables=["complainant", "accused", "incident", "victim_impact", "evidence"],
            template="""You are a legal analyst. Analyze the FIR case details and identify the PRIMARY crime/intent.

CASE DETAILS:
Complainant: {complainant}
Accused: {accused}
Incident: {incident}
Victim Impact: {victim_impact}
Evidence: {evidence}

Based on the facts presented, identify:
1. The PRIMARY intent/crime type
2. Confidence level (0-1)
3. Any secondary intents

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "primary_intent": "string",
  "confidence": 0.95,
  "secondary_intents": ["string"]
}}"""
        )
        
        # STAGE 2: Legal Reasoning Prompt
        self.reasoning_prompt = PromptTemplate(
            input_variables=["incident", "accused", "victim", "primary_intent", "statute_context"],
            template="""You are a legal expert analyzing applicable criminal statutes.

CASE FACTS:
Incident: {incident}
Accused: {accused}
Victim: {victim}
Identified Intent: {primary_intent}

RELEVANT STATUTE SECTIONS:
{statute_context}

Determine which statutes are APPLICABLE to this case:
- Consider each statute's elements and how they relate to the case facts
- Explain WHY each statute applies
- Assess overall severity (high/medium/low)
- Provide confidence in your analysis

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "applicable_statutes": [
    {{"section": "string", "law": "string", "reasoning": "string"}}
  ],
  "legal_basis": "string",
  "severity_assessment": "string",
  "confidence": 0.95
}}"""
        )
    
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
                    'section_text': chunk['section_text'][:200],
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
        """Find corresponding IPC/BNS section using statute mappings, with LLM fallback."""
        
        corresponding = []
        
        for mapping in self.statute_mappings:
            if law == 'IPC' and mapping['ipc'] == section_id:
                corresponding.append({
                    'law': 'BNS',
                    'section_id': mapping['bns']
                })
            elif law == 'BNS' and mapping['bns'] == section_id:
                corresponding.append({
                    'law': 'IPC',
                    'section_id': mapping['ipc']
                })
        
        # LLM fallback when static mapping has no entry
        if not corresponding:
            target_law = 'BNS' if law == 'IPC' else 'IPC'
            llm_section = self._llm_map_section(law, section_id, target_law)
            if llm_section:
                corresponding.append({
                    'law': target_law,
                    'section_id': llm_section
                })
        
        return corresponding

    def _llm_map_section(self, source_law, section_id, target_law):
        """Use Groq LLM to find the corresponding BNS/IPC section when static mapping is missing."""
        try:
            prompt = f"""What is the corresponding {target_law} (Bharatiya Nyaya Sanhita) section number for {source_law} Section {section_id} (Indian Penal Code)?
Reply with ONLY the section number (e.g. "309"). If unsure, reply "unknown"."""

            result = self.llm_intent.invoke(prompt).content.strip()
            # Extract just the number
            import re
            m = re.search(r'\b(\d+[A-Za-z]*)\b', result)
            if m and result.lower() != 'unknown':
                mapped = m.group(1)
                print(f"  [LLM Mapping] {source_law} {section_id} → {target_law} {mapped}")
                return mapped
        except Exception as e:
            print(f"  [LLM Mapping] Failed for {source_law} {section_id}: {e}")
        return None
    
    def get_section_extract(self, law, section_id):
        """Get text extract from a statute section"""
        
        chunk_id_formats = [
            f"{law.lower()}_{section_id}",
            f"{law}_{section_id}",
            f"{law.lower()}{section_id}",
            f"{law}{section_id}"
        ]
        
        for chunk_id in chunk_id_formats:
            if chunk_id in self.statute_chunks:
                chunk = self.statute_chunks[chunk_id]
                return chunk['section_text'][:150].replace('\n', ' ')
        
        for cid, chunk in self.statute_chunks.items():
            if chunk['section_id'] == section_id and chunk['law'] == law:
                return chunk['section_text'][:150].replace('\n', ' ')
        
        return f"[{law} {section_id} not in extracted database - check official government source]"
    
    def create_chains(self):
        """Create LangChain chains for each analysis stage"""
        
        # Note: LLMChain is not needed for basic operation
        # We'll use prompts directly with the LLMs
        print("✓ LangChain prompt templates created (intent + reasoning)")
    
    def analyze_fir_with_chains(self, fir_data):
        """Analyze FIR using sequential chain prompting"""
        
        print("\n" + "="*80)
        print("STATUTE-AWARE RAG WITH LANGCHAIN CHAIN PROMPTING")
        print("="*80)
        
        # Extract case facts for retrieval
        case_facts = {
            'incident': fir_data.get('incident_description', ''),
            'victim_impact': fir_data.get('victim_impact', ''),
            'evidence': fir_data.get('evidence', '')
        }
        
        # STEP 0: Retrieve relevant statutes from vector DB
        print("\n[STEP 0] Retrieving relevant statute sections from vector DB...")
        query_text = f"{fir_data.get('incident_description', '')} {fir_data.get('victim_impact', '')}"
        retrieved_chunks = self.retrieve_relevant_statutes(query_text, top_k=10)
        print(f"✓ Retrieved {len(retrieved_chunks)} statute chunks")
        
        # Apply negative rules filter
        print("\n[STEP 1.5] Applying negative rules filter...")
        filtered_chunks = self.apply_negative_rules_filter(retrieved_chunks, case_facts)
        print(f"✓ Filtered to {len(filtered_chunks)} applicable chunks")
        
        # STEP 1: Chain - Intent Identification
        print("\n" + "-"*80)
        print("[CHAIN PROMPT 1] INTENT IDENTIFICATION")
        print("-"*80)
        
        intent_input = {
            "complainant": fir_data.get('complainant_name', 'Unknown'),
            "accused": ', '.join(fir_data.get('accused_names', [])),
            "incident": fir_data.get('incident_description', 'Unknown'),
            "victim_impact": fir_data.get('victim_impact', 'Unknown'),
            "evidence": fir_data.get('evidence', 'Unknown')
        }
        
        print("\nInput Variables:")
        for key, value in intent_input.items():
            preview = value[:100] if isinstance(value, str) else str(value)
            print(f"  {key}: {preview}...")
        
        # Format and run intent chain
        intent_prompt_str = self.intent_prompt.format(**intent_input)
        intent_result_str = self.llm_intent.invoke(intent_prompt_str).content
        
        print("\nIntent Chain Output:")
        print(intent_result_str)
        
        # Parse intent result
        try:
            intent_result = json.loads(intent_result_str)
        except:
            intent_result = {"primary_intent": "Unknown", "confidence": 0, "secondary_intents": []}
        
        print(f"\n✓ Primary Intent: {intent_result.get('primary_intent', 'Unknown')}")
        print(f"✓ Confidence: {intent_result.get('confidence', 0)}")
        
        # STEP 2: Chain - Legal Reasoning (uses output from Step 1)
        print("\n" + "-"*80)
        print("[CHAIN PROMPT 2] LEGAL REASONING (Using Intent from Chain 1)")
        print("-"*80)
        
        # Prepare statute context from retrieved chunks
        statute_context = "\n".join([
            f"[{chunk['law']} {chunk['section_id']}] {chunk['section_text']}"
            for chunk in filtered_chunks[:5]
        ])
        
        reasoning_input = {
            "incident": fir_data.get('incident_description', 'Unknown'),
            "accused": ', '.join(fir_data.get('accused_names', [])),
            "victim": fir_data.get('victim_name', 'Unknown'),
            "primary_intent": intent_result.get('primary_intent', 'Unknown'),  # From Chain 1
            "statute_context": statute_context
        }
        
        print("\nInput Variables (with Chain 1 Output):")
        print(f"  primary_intent: {reasoning_input['primary_intent']} [FROM CHAIN 1]")
        print(f"  statute_context: {len(statute_context)} characters from {len(filtered_chunks)} filtered chunks")
        
        # Format and run reasoning chain
        reasoning_prompt_str = self.reasoning_prompt.format(**reasoning_input)
        reasoning_result_str = self.llm_reasoning.invoke(reasoning_prompt_str).content
        
        print("\nLegal Reasoning Chain Output:")
        print(reasoning_result_str)
        
        # Parse reasoning result
        try:
            reasoning_result = json.loads(reasoning_result_str)
        except:
            reasoning_result = {"applicable_statutes": [], "legal_basis": "Error parsing", "severity_assessment": "unknown", "confidence": 0}
        
        print(f"\n✓ Applicable Statutes: {len(reasoning_result.get('applicable_statutes', []))} found")
        print(f"✓ Severity: {reasoning_result.get('severity_assessment', 'unknown')}")
        
        # STEP 3: Enrich applicable statutes with mappings and extracts
        print("\n" + "-"*80)
        print("[STEP 3] ENRICHING STATUTES WITH MAPPINGS & EXTRACTS")
        print("-"*80)
        
        enriched_statutes = []
        for statute in reasoning_result.get('applicable_statutes', []):
            # Parse section
            section_parts = statute.get('section', '').strip().split()
            if len(section_parts) >= 2:
                law = section_parts[0]
                section_id = section_parts[1]
            else:
                section_str = statute.get('section', 'IPC 394')
                parts = section_str.split()
                law = parts[0] if parts else 'IPC'
                section_id = parts[-1] if parts else '394'
            
            # Find corresponding section
            corresponding = self.find_corresponding_sections(law, section_id)
            
            # Get extract
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
        
        print(f"✓ Enriched {len(enriched_statutes)} statutes with mappings & extracts")
        
        # Format final output
        final_output = {
            'status': 'success',
            'timestamp': str(Path('.').absolute()),
            'chain_prompting_stages': 2,
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
    """Load sample FIR from JSON file"""
    repo_root = Path(__file__).resolve().parents[2]
    fir_path = repo_root / 'src_dataset_files' / 'fir_sample.json'

    with open(fir_path, 'r', encoding='utf-8') as f:
        fir_data = json.load(f)

    return fir_data


def main():
    """Main execution"""
    
    print("="*80)
    print("STATUTE-AWARE RAG WITH LANGCHAIN CHAIN PROMPTING")
    print("="*80)
    
    try:
        # Initialize system
        print("\n[INIT] Initializing RAG System with LangChain...")
        rag_system = StatuteRAGChainSystem()
        
        # Create chains
        rag_system.create_chains()
        
        # Create sample FIR
        print("\n[INPUT] Loading sample FIR...")
        sample_fir = create_sample_fir()
        
        print("\nSample FIR Details:")
        print(f"  Complainant: {sample_fir['complainant_name']}")
        print(f"  Accused: {', '.join(sample_fir['accused_names'])}")
        print(f"  Incident: {sample_fir['incident_description'][:80]}...")
        
        # Run analysis with chains
        print("\n[ANALYSIS] Running FIR Analysis with Chain Prompting...")
        result = rag_system.analyze_fir_with_chains(sample_fir)
        
        # Save result
        repo_root = Path(__file__).resolve().parents[2]
        output_file = repo_root / 'output' / 'fir_analysis_result_chains.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Analysis saved to: {output_file}")
        
        # Display result
        print("\n" + "="*80)
        print("FINAL ANALYSIS RESULT (WITH CHAIN PROMPTING)")
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
        
        print(f"\n[Chain Prompting Stages] {result['chain_prompting_stages']}")
        print("  Stage 1: Intent Identification (llama-3.1-8b-instant)")
        print("  Stage 2: Legal Reasoning (gpt-oss-120b) [using Stage 1 output]")
        
        print("\n" + "="*80)
        print("✓ CHAIN PROMPTING ANALYSIS COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
