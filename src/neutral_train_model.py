import pickle
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
import os

# =====================
# Configuration
# =====================

SOURCE_PATH = "data/processed_comments/wikipedia/wikipedia.pkl"
BIGRAM_MODEL_PATH = "data/models/bigram/bigram.phr"
OUTPUT_DIR = "data/models/neutral"
OUTPUT_FILENAME = "neutral.model"

# Model Hyperparameters
W2V_PARAMS = {
    "vector_size": 300,
    "window": 5,
    "min_count": 10,
    "workers": 16,
    "sg": 0,
    "epochs": 5,
}


def load_corpus(source_path):
    """Load the processed corpus from a pickle file."""
    with open(source_path, "rb") as f:
        return pickle.load(f)


def apply_bigrams(sentences, bigram_path):
    """Load bigram model and transform sentences."""
    # Convert Path to str because gensim expects a filename string
    bigram_model = Phraser.load(str(bigram_path))
    return [bigram_model[sent] for sent in sentences]


def train_model(sentences, params):
    """Initialize and train the Word2Vec model."""
    return Word2Vec(sentences=sentences, **params)


def save_model(model, output_dir, filename):
    """Save the trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    # convert Path to str because gensim expects a filename string
    save_path = output_dir / filename
    model.save(str(save_path))


def main():
    """Main execution function."""
    
    # 1. Load Data
    sentences = load_corpus(SOURCE_PATH)

    # 2. Transform Data
    sentences = apply_bigrams(sentences, BIGRAM_MODEL_PATH)

    # 3. Train Model
    model = train_model(sentences, W2V_PARAMS)

    # 4. Save Model
    save_model(model, OUTPUT_DIR, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()