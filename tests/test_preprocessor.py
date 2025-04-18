"""Tests for text preprocessor."""
import unittest
from src.preprocessor import TextPreprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.pp = TextPreprocessor(max_length=50)

    def test_clean(self):
        text = "(CNN) -- Some   news  text  https://t.co/abc"
        result = self.pp.clean_text(text)
        self.assertNotIn("CNN", result)
        self.assertNotIn("https", result)

    def test_truncate(self):
        text = " ".join(["word"] * 100)
        result = self.pp.truncate(text)
        self.assertEqual(len(result.split()), 50)

if __name__ == "__main__":
    unittest.main()
