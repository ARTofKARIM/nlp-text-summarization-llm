"""Dataset loading for text summarization."""
import pandas as pd
from typing import List, Tuple

class SummarizationDataLoader:
    def __init__(self, config):
        self.config = config
        self.texts = []
        self.summaries = []

    def load_from_csv(self, filepath):
        df = pd.read_csv(filepath)
        text_col = self.config["data"]["text_column"]
        summary_col = self.config["data"]["summary_column"]
        self.texts = df[text_col].tolist()
        self.summaries = df[summary_col].tolist()
        max_s = self.config["data"].get("max_samples")
        if max_s:
            self.texts = self.texts[:max_s]
            self.summaries = self.summaries[:max_s]
        print(f"Loaded {len(self.texts)} text-summary pairs")
        return self.texts, self.summaries

    def load_from_huggingface(self):
        try:
            from datasets import load_dataset
            dataset_name = self.config["data"]["dataset"]
            subset = self.config["data"].get("subset")
            ds = load_dataset(dataset_name, subset, split="test")
            max_s = self.config["data"].get("max_samples", len(ds))
            ds = ds.select(range(min(max_s, len(ds))))
            self.texts = ds[self.config["data"]["text_column"]]
            self.summaries = ds[self.config["data"]["summary_column"]]
            print(f"Loaded {len(self.texts)} samples from {dataset_name}")
        except Exception as e:
            print(f"Could not load from HuggingFace: {e}")
        return self.texts, self.summaries

    def get_stats(self):
        text_lens = [len(t.split()) for t in self.texts]
        sum_lens = [len(s.split()) for s in self.summaries]
        return {
            "num_samples": len(self.texts),
            "avg_text_words": sum(text_lens) / len(text_lens),
            "avg_summary_words": sum(sum_lens) / len(sum_lens),
            "compression_ratio": sum(text_lens) / (sum(sum_lens) + 1),
        }
