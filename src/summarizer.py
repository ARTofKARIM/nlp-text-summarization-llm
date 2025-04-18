"""Summarization engine using transformer models."""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List
from tqdm import tqdm

class TransformerSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.pipe = None

    def load_model(self):
        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    def summarize(self, text, max_length=150, min_length=30, num_beams=4,
                  length_penalty=2.0, no_repeat_ngram_size=3):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024,
                                 truncation=True).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"], max_length=max_length, min_length=min_length,
                num_beams=num_beams, length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def summarize_batch(self, texts: List[str], **kwargs) -> List[str]:
        summaries = []
        for text in tqdm(texts, desc="Summarizing"):
            summaries.append(self.summarize(text, **kwargs))
        return summaries

class ExtractiveSummarizer:
    def __init__(self, num_sentences=3):
        self.num_sentences = num_sentences

    def summarize(self, text):
        import nltk
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= self.num_sentences:
            return text
        word_freq = {}
        for sent in sentences:
            for word in sent.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        scores = []
        for sent in sentences:
            score = sum(word_freq.get(w.lower(), 0) for w in sent.split())
            scores.append(score)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.num_sentences]
        top_indices.sort()
        return " ".join(sentences[i] for i in top_indices)
