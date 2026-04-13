from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        if not word:
            return []

        if len(word) > self.max_word_len:
            return [self.unk_token]

        tokens = []
        start = 0

        while start < len(word):
            matched = False
            end = len(word)

            while end > start:
                if start == 0:
                    candidate = word[start:end]
                else:
                    candidate = "##" + word[start:end]

                if candidate in self.vocab:
                    tokens.append(candidate)
                    start = end
                    matched = True
                    break

                end -= 1

            if not matched:
                return [self.unk_token]

        return tokens
