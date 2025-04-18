"""Tests for evaluation module."""
import unittest
from src.evaluation import SummarizationEvaluator

class TestEvaluation(unittest.TestCase):
    def test_score_single(self):
        evaluator = SummarizationEvaluator()
        scores = evaluator.score_single("the cat sat on the mat", "the cat sat on a mat")
        self.assertGreater(scores["rouge1"].fmeasure, 0.5)

    def test_perfect_score(self):
        evaluator = SummarizationEvaluator()
        text = "hello world this is a test"
        scores = evaluator.score_single(text, text)
        self.assertAlmostEqual(scores["rouge1"].fmeasure, 1.0, places=2)

if __name__ == "__main__":
    unittest.main()
