"""Visualization for summarization results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

class SummarizationVisualizer:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_rouge_scores(self, scores, save=True):
        metrics = list(scores.keys())
        f1_scores = [scores[m]["fmeasure"] for m in metrics]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(metrics, f1_scores, color=["#2ecc71", "#3498db", "#e74c3c"])
        ax.set_ylabel("F1 Score")
        ax.set_title("ROUGE Scores")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center")
        if save:
            fig.savefig(f"{self.output_dir}rouge_scores.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_length_comparison(self, originals, summaries, save=True):
        orig_lens = [len(t.split()) for t in originals]
        sum_lens = [len(s.split()) for s in summaries]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(orig_lens, bins=30, color="steelblue", alpha=0.7, label="Original")
        ax1.hist(sum_lens, bins=30, color="coral", alpha=0.7, label="Summary")
        ax1.set_title("Length Distribution")
        ax1.legend()
        ratios = [s / (o + 1) for s, o in zip(sum_lens, orig_lens)]
        ax2.hist(ratios, bins=30, color="green", alpha=0.7)
        ax2.set_title("Compression Ratio")
        if save:
            fig.savefig(f"{self.output_dir}length_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
