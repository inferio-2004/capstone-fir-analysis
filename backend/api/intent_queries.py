"""
Intent-to-retrieval-query mapping for the RAG pipeline.

Maps identified criminal intents to focused legal terminology queries
that produce higher cosine-similarity matches in the Pinecone vector DB.
"""


# Maps a keyword fragment to a list of targeted search queries
INTENT_QUERY_MAP: dict[str, list[str]] = {
    "dowry":              ["cruelty by husband demanding dowry Section 498A",
                           "dowry death dowry prohibition harassment married woman"],
    "domestic violence":  ["cruelty by husband wife subjected to cruelty Section 498A",
                           "voluntarily causing hurt assault"],
    "sexual":             ["sexual assault outraging modesty of woman",
                           "rape sexual offence against women"],
    "murder":             ["murder culpable homicide causing death"],
    "kidnap":             ["kidnapping abduction wrongful confinement"],
    "robbery":            ["robbery extortion dacoity snatching"],
    "theft":              ["theft dishonest misappropriation criminal breach of trust"],
    "cheat":              ["cheating fraud dishonestly inducing delivery"],
    "assault":            ["voluntarily causing hurt grievous hurt"],
    "hurt":               ["voluntarily causing hurt grievous hurt simple"],
    "defam":              ["defamation imputation harm reputation"],
    "intimidat":          ["criminal intimidation threat"],
    "cruelty":            ["cruelty by husband demanding dowry Section 498A"],
    "harassment":         ["cruelty by husband demanding dowry harassment married woman"],
}


def intent_to_retrieval_queries(
    primary_intent: str,
    secondary_intents: list[str] | None = None,
) -> list[str]:
    """Convert identified criminal intents into targeted vector-DB queries.

    The raw FIR text often has low cosine similarity with legal statute
    language. This function generates focused queries using legal terminology
    to improve retrieval accuracy.
    """
    queries: list[str] = []
    intent_lower = primary_intent.lower() if primary_intent else ""
    all_intents = [intent_lower] + [s.lower() for s in (secondary_intents or [])]

    for keyword, qs in INTENT_QUERY_MAP.items():
        for intent_text in all_intents:
            if keyword in intent_text:
                queries.extend(qs)

    # Fallback: use the raw intent string
    if not queries:
        queries.append(f"{primary_intent} Indian Penal Code section")

    return list(dict.fromkeys(queries))  # deduplicate, preserve order
