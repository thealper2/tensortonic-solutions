import re

ABSTENTION_ANSWER = "I don't know based on the provided context."
CITATION_PATTERN = r"\[([A-Za-z0-9][A-Za-z0-9_-]*)\]"


def validate_citations(answer, retrieved_ids):
    answer = answer.strip()
    
    if not answer:
        return {
            "answer": "I don't know based on the provided context.",
            "citations": []
        }
    
    pattern = r'\[([A-Za-z0-9][A-Za-z0-9_\-]*)\]'
    matches = re.findall(pattern, answer)
    
    if not matches:
        return {
            "answer": "I don't know based on the provided context.",
            "citations": []
        }
    
    seen = set()
    unique_citations = []
    for citation in matches:
        if citation not in seen:
            seen.add(citation)
            unique_citations.append(citation)
    
    retrieved_set = set(retrieved_ids)
    for citation in unique_citations:
        if citation not in retrieved_set:
            return {
                "answer": "I don't know based on the provided context.",
                "citations": []
            }
    
    return {
        "answer": answer,
        "citations": unique_citations
    }