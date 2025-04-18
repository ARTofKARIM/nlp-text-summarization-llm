"""ROUGE-based evaluation for summarization."""
from rouge_score import rouge_scorer
import numpy as np
from typing import List, Dict

class SummarizationEvaluator:
    def __init__(self, metrics=None):
        self.metrics = metrics or ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)

    def score_single(self, prediction, reference):
        return self.scorer.score(reference, prediction)

    def score_batch(self, predictions: List[str], references: List[str]) -> Dict:
        all_scores = {m: {"precision": [], "recall": [], "fmeasure": []} for m in self.metrics}
        for pred, ref in zip(predictions, references):
            scores = self.score_single(pred, ref)
            for m in self.metrics:
                all_scores[m]["precision"].append(scores[m].precision)
                all_scores[m]["recall"].append(scores[m].recall)
                all_scores[m]["fmeasure"].append(scores[m].fmeasure)
        avg_scores = {}
        for m in self.metrics:
            avg_scores[m] = {
                "precision": np.mean(all_scores[m]["precision"]),
                "recall": np.mean(all_scores[m]["recall"]),
                "fmeasure": np.mean(all_scores[m]["fmeasure"]),
            }
        return avg_scores

    def print_results(self, scores: Dict):
        print("\nROUGE Evaluation Results:")
        print(f"{'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 44)
        for m, s in scores.items():
            print(f"{m:<12} {s['precision']:>10.4f} {s['recall']:>10.4f} {s['fmeasure']:>10.4f}")
