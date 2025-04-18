"""Text preprocessing for summarization."""
import re
import nltk
from typing import List

class TextPreprocessor:
    def __init__(self, max_length=1024):
        self.max_length = max_length
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\(CNN\)\s*--?\s*", "", text)
        text = re.sub(r"https?://\S+", "", text)
        return text

    def truncate(self, text: str) -> str:
        words = text.split()
        if len(words) > self.max_length:
            words = words[:self.max_length]
        return " ".join(words)

    def split_sentences(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)

    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        text = self.truncate(text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(t) for t in texts]
