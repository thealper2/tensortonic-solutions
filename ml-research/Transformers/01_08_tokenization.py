import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = id
            self.id_to_word[id] = token

        unique_words = set()
        for text in texts:
            words = text.strip().split()
            unique_words.update(words)

        next_id = len(special_tokens)
        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = next_id
                self.id_to_word[next_id] = word
                next_id += 1

        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.strip().split()
        token_ids = []
        unk_id = self.word_to_id.get(self.unk_token, 1)
        for word in words:
            token_id = self.word_to_id.get(word, unk_id)
            token_ids.append(token_id)

        return token_ids
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for token_id in ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)

        return ' '.join(words)
