import html
import os
import pickle
import re
import unicodedata

import nltk
from datasets import load_from_disk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =====================
# Configuration
# =====================
TARGET_SIZE = 1_000_000
SEED = 23
SOURCE_PATH = "data/raw/wikipedia-20231101-en"
OUTPUT_DIR = "data/processed_comments/wikipedia"
OUTPUT_FILE = "wikipedia.pkl"

# =====================
# Global Resources
# =====================
# Initialize global resources once
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# POS tag cache to avoid redundant tagging
POS_CACHE = {}
# Lemma cache to avoid redundant lemmatization
LEMMA_CACHE = {}


def get_wordnet_pos(tag):
    """Convert NLTK POS tag to WordNet POS tag."""
    if tag.startswith("J"):
        return "a"  # adjective
    if tag.startswith("V"):
        return "v"  # verb
    if tag.startswith("N"):
        return "n"  # noun
    if tag.startswith("R"):
        return "r"  # adverb
    return "n"


def normalize_text(text):
    """Normalize and clean raw text."""
    # Handle HTML entities
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Remove URLs and Markdown formatting
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Basic text cleaning: Keep only letters and spaces, convert to lower
    return re.sub("[^A-Za-z]+", " ", text).lower()


def lemmatize_words(words):
    """Lemmatize words with cached POS tags and lemmas."""
    # POS tagging (with cache)
    uncached = [w for w in words if w not in POS_CACHE]
    if uncached:
        tagged_uncached = nltk.pos_tag(uncached)
        for word, tag in tagged_uncached:
            POS_CACHE[word] = tag

    processed = []
    for word in words:
        tag = POS_CACHE[word]
        wordnet_pos = get_wordnet_pos(tag)
        key = (word, wordnet_pos)
        
        if key in LEMMA_CACHE:
            lemma = LEMMA_CACHE[key]
        else:
            lemma = LEMMATIZER.lemmatize(word, pos=wordnet_pos)
            LEMMA_CACHE[key] = lemma
        processed.append(lemma)
        
    return processed


def preprocess_text(text, lemmatize=True, without_stopwords=False):
    """Preprocess text content with optimized NLTK operations."""
    text = normalize_text(text)
    words = text.split()
    
    if not words:
        return []

    # Lemmatization first
    if lemmatize:
        words = lemmatize_words(words)

    # Remove stopwords after lemmatization
    if without_stopwords:
        words = [w for w in words if w not in STOP_WORDS]

    return [w for w in words if 2 < len(w) <= 15]


def load_and_sample_data(source_path, target_size, seed):
    """Load dataset from disk and sample it."""
    source_ds = load_from_disk(source_path)
    sample_size = min(target_size, len(source_ds))
    return source_ds.shuffle(seed=seed).select(range(sample_size))


def save_corpus(corpus, output_dir, filename):
    """Save the processed corpus to disk."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(corpus, f)


def main():
    """Main execution function."""
    # Load Data
    sampled_ds = load_and_sample_data(SOURCE_PATH, TARGET_SIZE, SEED)

    # Process Data
    # Note: Using list comprehension here to maintain exact behavior of original script
    # which processed everything into memory before saving.
    corpus = [preprocess_text(text, lemmatize=True, without_stopwords=False) 
              for text in sampled_ds["text"]]

    # Save Data
    save_corpus(corpus, OUTPUT_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()