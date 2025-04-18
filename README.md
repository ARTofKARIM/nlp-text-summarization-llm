# Text Summarization with LLMs

A text summarization pipeline supporting both abstractive (BART, T5) and extractive approaches with ROUGE evaluation metrics.

## Architecture
```
nlp-text-summarization-llm/
├── src/
│   ├── data_loader.py      # HuggingFace/CSV data loading
│   ├── preprocessor.py     # Text cleaning and truncation
│   ├── summarizer.py       # Transformer & extractive summarizers
│   ├── evaluation.py       # ROUGE metrics computation
│   └── visualization.py    # Score and length analysis plots
├── config/config.yaml
├── tests/
└── main.py
```

## Models
- **BART-large-CNN**: Facebook's abstractive summarization model
- **Extractive**: Frequency-based sentence scoring

## Installation
```bash
git clone https://github.com/mouachiqab/nlp-text-summarization-llm.git
cd nlp-text-summarization-llm
pip install -r requirements.txt
```

## Usage
```bash
python main.py --method abstractive --data data/articles.csv
python main.py --method extractive --data data/articles.csv
```

## Technologies
- Python 3.9+, HuggingFace Transformers, PyTorch, NLTK, rouge-score








