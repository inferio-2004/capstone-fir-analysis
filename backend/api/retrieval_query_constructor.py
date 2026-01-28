"""
Statute Retrieval Query Constructor
This module builds retrieval queries from extracted facts and performs vector search.
"""

import json
from typing import List, Dict, Any

# Keyword synonyms for query expansion
KEYWORD_SYNONYMS = {
    "hit": ["hurt", "injury", "harm", "damage", "assault", "battery"],
    "steal": ["theft", "robbery", "taking", "possession"],
    "rape": ["sexual assault", "sexual abuse", "penetration", "non-consensual"],
    "extort": ["threat", "menace", "coercion", "demand", "inducement"],
    "murder": ["homicide", "killing", "death", "unlawful death"],
    "cheat": ["fraud", "deception", "dishonest", "induced delivery"],
    "kidnap": ["abduction", "unlawful restraint", "wrongful confinement"],
    "property": ["movable", "valuable thing", "goods", "money"]
}

def expand_query_with_synonyms(facts: Dict[str, Any]) -> str:
    """
    Convert extracted facts to retrieval query string.
    
    Args:
        facts: Structured facts JSON from fact extractor
    
    Returns:
        Expanded query string with synonyms
    """
    query_parts = []
    
    # 1-line fact summary
    if 'fact_summary' in facts:
        query_parts.append(facts['fact_summary'])
    
    # Extract keywords and expand
    keywords = facts.get('keywords', [])
    expanded_keywords = set(keywords)
    
    for keyword in keywords:
        if keyword in KEYWORD_SYNONYMS:
            expanded_keywords.update(KEYWORD_SYNONYMS[keyword])
    
    # Build final query
    if expanded_keywords:
        query_parts.append(" ".join(expanded_keywords))
    
    final_query = "Statute search for matching offence ingredients: " + " ".join(query_parts)
    return final_query

def compute_match_score(vector_similarity: float, metadata_match: bool = False, 
                       negative_rule_match: bool = False) -> float:
    """
    Compute final match score from components.
    
    Args:
        vector_similarity: Vector similarity score (0-1)
        metadata_match: Boolean if offence_type/mapped_sections match
        negative_rule_match: Boolean if negative rule failure detected
    
    Returns:
        Final match score (0-1)
    """
    score = vector_similarity
    
    if metadata_match:
        score += 0.1
    
    if negative_rule_match:
        score -= 0.15
    
    return max(0.0, min(1.0, score))

def construct_retrieval_payload(facts: Dict[str, Any], vector_db_results: List[Dict]) -> Dict:
    """
    Package retrieved statute chunks for downstream LLM reasoning.
    
    Args:
        facts: Original extracted facts
        vector_db_results: Results from vector DB with metadata
    
    Returns:
        Packaged payload for LLM
    """
    query = expand_query_with_synonyms(facts)
    
    # Score and rank results
    scored_results = []
    for result in vector_db_results:
        match_score = compute_match_score(
            result['similarity_score'],
            metadata_match=result.get('offence_type') in facts.get('suspected_offences', []),
            negative_rule_match=result.get('negative_rule_flagged', False)
        )
        result['match_score'] = match_score
        scored_results.append(result)
    
    # Sort by match score
    scored_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return {
        "query": query,
        "facts": facts,
        "retrieved_statutes": scored_results[:5],  # Top 5
        "retrieval_metadata": {
            "total_candidates": len(vector_db_results),
            "returned": min(5, len(vector_db_results))
        }
    }
