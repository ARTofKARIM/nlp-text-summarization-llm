"""Main pipeline for text summarization."""
import argparse
import yaml
from src.data_loader import SummarizationDataLoader
from src.preprocessor import TextPreprocessor
from src.summarizer import TransformerSummarizer, ExtractiveSummarizer
from src.evaluation import SummarizationEvaluator

def main():
    parser = argparse.ArgumentParser(description="Text Summarization Pipeline")
    parser.add_argument("--data", help="CSV data path")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--method", choices=["abstractive", "extractive", "both"], default="abstractive")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = SummarizationDataLoader(config)
    if args.data:
        texts, refs = loader.load_from_csv(args.data)
    else:
        texts, refs = loader.load_from_huggingface()

    pp = TextPreprocessor(config["model"]["max_input_length"])
    texts_clean = pp.preprocess_batch(texts)
    evaluator = SummarizationEvaluator(config["evaluation"]["metrics"])

    if args.method in ["abstractive", "both"]:
        summarizer = TransformerSummarizer(config["model"]["name"])
        summarizer.load_model()
        preds = summarizer.summarize_batch(texts_clean[:10],
            max_length=config["model"]["max_output_length"],
            min_length=config["model"]["min_output_length"])
        scores = evaluator.score_batch(preds, refs[:10])
        evaluator.print_results(scores)

    if args.method in ["extractive", "both"]:
        ext = ExtractiveSummarizer(num_sentences=3)
        preds = [ext.summarize(t) for t in texts_clean[:10]]
        scores = evaluator.score_batch(preds, refs[:10])
        print("\nExtractive:")
        evaluator.print_results(scores)

if __name__ == "__main__":
    main()
