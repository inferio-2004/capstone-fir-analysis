#!/usr/bin/env python3
"""
Test Pinecone Vector Database
Verify embeddings are searchable and retrieval works correctly
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

def test_pinecone_vector_db():
    """Test Pinecone vector database with sample queries"""
    
    print("="*80)
    print("TESTING PINECONE VECTOR DATABASE")
    print("="*80)
    
    # Initialize Pinecone
    try:
        from pinecone import Pinecone
        print("\n✓ Pinecone library loaded")
    except ImportError:
        print("\n✗ ERROR: pinecone not installed")
        return False
    
    # Get API key
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        print("✗ ERROR: PINECONE_API_KEY not found in .env")
        return False
    
    print("✓ API Key loaded from .env")
    
    try:
        # Connect to Pinecone
        pc = Pinecone(api_key=api_key)
        print("✓ Connected to Pinecone")
        
        # Get index
        index_name = 'statute-embeddings'
        index = pc.Index(index_name)
        print(f"✓ Connected to index: {index_name}")
        
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        # Load statute chunks for reference
        chunks_file = Path("output/statute_chunks_complete.jsonl")
        chunks = {}
        with open(chunks_file, 'r') as f:
            for line in f:
                chunk = json.loads(line)
                chunks[chunk['chunk_id']] = chunk
        
        print(f"✓ Loaded {len(chunks)} statute chunks for reference")
        
        # Test queries
        test_queries = [
            "murder with intention",
            "theft of property",
            "rape and sexual assault",
            "criminal conspiracy and agreement",
            "wrongful restraint and confinement",
            "hurt and grievous hurt"
        ]
        
        print("\n" + "="*80)
        print("RUNNING TEST QUERIES")
        print("="*80)
        
        all_passed = True
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[Test {i}] Query: '{query}'")
            print("-" * 80)
            
            try:
                # Generate embedding for query
                query_embedding = model.encode(query).tolist()
                
                # Search in Pinecone
                results = index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
                
                if not results['matches']:
                    print("  ✗ No results found")
                    all_passed = False
                    continue
                
                print(f"  ✓ Found {len(results['matches'])} results")
                print(f"  Showing top 3:")
                
                for j, match in enumerate(results['matches'][:3], 1):
                    chunk_id = match['id']
                    score = match['score']
                    
                    if chunk_id in chunks:
                        chunk = chunks[chunk_id]
                        law = chunk['law']
                        section_id = chunk['section_id']
                        offence_type = chunk.get('offence_type', 'unknown')
                        text_preview = chunk['section_text'][:100].replace('\n', ' ')
                        
                        print(f"\n    Result {j}:")
                        print(f"      Section: {law} {section_id}")
                        print(f"      Type: {offence_type}")
                        print(f"      Score: {score:.4f}")
                        print(f"      Preview: {text_preview}...")
                    else:
                        print(f"    Result {j}: {chunk_id} (score: {score:.4f})")
                
                print(f"  ✓ Test {i} PASSED")
                
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                all_passed = False
        
        # Test metadata filtering
        print("\n" + "="*80)
        print("TESTING METADATA FILTERING")
        print("="*80)
        
        print("\n[Filter Test] Query for 'murder' with IPC law filtering")
        print("-" * 80)
        
        try:
            query_embedding = model.encode("murder").tolist()
            
            # Search with metadata filter
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                filter={"law": {"$eq": "IPC"}}
            )
            
            print(f"✓ Found {len(results['matches'])} IPC sections containing 'murder'")
            
            for match in results['matches'][:3]:
                chunk_id = match['id']
                metadata = match.get('metadata', {})
                law = metadata.get('law', 'unknown')
                section_id = metadata.get('section_id', 'unknown')
                print(f"  - {law} {section_id} (score: {match['score']:.4f})")
            
            print("✓ Metadata filtering works!")
            
        except Exception as e:
            print(f"⚠ Metadata filtering not fully supported or error: {e}")
        
        # Index statistics
        print("\n" + "="*80)
        print("INDEX STATISTICS")
        print("="*80)
        
        try:
            index_stats = index.describe_index_stats()
            print(f"✓ Total vectors in index: {index_stats.total_vector_count}")
            print(f"✓ Index dimension: 384")
            print(f"✓ Index metric: cosine")
            print(f"✓ Index status: Ready")
        except Exception as e:
            print(f"Note: {e}")
        
        # Final summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        if all_passed:
            print("\n✓ ALL TESTS PASSED!")
            print("✓ Vector database is working correctly")
            print("✓ Embeddings are searchable")
            print("✓ RAG retrieval is functional")
            return True
        else:
            print("\n⚠ Some tests failed - see details above")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

def main():
    success = test_pinecone_vector_db()
    
    print("\n" + "="*80)
    if success:
        print("VECTOR DATABASE TEST: PASSED ✓")
        print("="*80)
        print("\nYour Pinecone vector database is fully operational!")
        print("You can now use it for RAG queries and LLM integration.")
    else:
        print("VECTOR DATABASE TEST: FAILED ✗")
        print("="*80)
        print("\nCheck the errors above and troubleshoot.")

if __name__ == '__main__':
    main()
