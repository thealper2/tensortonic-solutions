import numpy as np
from typing import List, Tuple

def create_nsp_pairs(
    documents: List[List[str]],
    pair_specs: List[dict]
) -> List[Tuple[str, str, int]]:
    """
    Returns: list of (sentence_A, sentence_B, is_next_label) tuples
    """
    nsp_pairs = []

    for spec in pair_specs:
        doc_a_idx = spec['doc_a']
        doc_b_idx = spec['doc_b']
        sent_a_idx = spec['sent_a']
        sent_b_idx = spec['sent_b']

        sentence_a = documents[doc_a_idx][sent_a_idx]
        sentence_b = documents[doc_b_idx][sent_b_idx]

        is_next = 1 if (doc_a_idx == doc_b_idx and sent_b_idx == sent_a_idx + 1) else 0
        nsp_pairs.append([sentence_a, sentence_b, is_next])

    return nsp_pairs

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext logits: cls_hidden @ W + b
        """
        logits = cls_hidden @ self.W + self.b
        return logits        

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
