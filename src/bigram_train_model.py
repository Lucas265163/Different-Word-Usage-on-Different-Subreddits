import glob
import os
import pickle
from gensim.models.phrases import Phrases, Phraser


DEFAULT_SUBREDDITS = [
    "democrats",
    "republican",
]

DEFAULT_BASE_DATA_DIR = "data/processed_comments"
DEFAULT_OUTPUT_DIR = "data/models/bigram"
DEFAULT_OUTPUT_FILE = "bigram.phr"
DEFAULT_MIN_COUNT = 10
DEFAULT_THRESHOLD = 10


def get_batch_files(base_data_dir, subreddit):
    pattern = f"{base_data_dir}/{subreddit}/{subreddit}_batch*.pkl"
    files = sorted(glob.glob(pattern))
    print(f"Pattern: {files}")
    print(f"Loading {len(files)} files for subreddit: {subreddit}")
    return files


def load_batch_sentences(file_path):
    try:
        with open(file_path, "rb") as f:
            comments = pickle.load(f)
        return [
            comment["processed_text"]
            for comment in comments
            if "processed_text" in comment
        ]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def train_and_save_global_bigram_model(
    subreddits,
    base_data_dir,
    output_path,
    min_count=DEFAULT_MIN_COUNT,
    threshold=DEFAULT_THRESHOLD,
):
    phrases = Phrases(min_count=min_count, threshold=threshold)
    total_sentences = 0

    for subreddit in subreddits:
        files = get_batch_files(base_data_dir, subreddit)
        for file_path in files:
            batch_sentences = load_batch_sentences(file_path)
            phrases.add_vocab(batch_sentences)
            total_sentences += len(batch_sentences)

    print(f"Total sentences for bigram training: {total_sentences}")
    bigram_model = Phraser(phrases)
    bigram_model.save(output_path)
    print(f"Global bigram model saved to {output_path}")


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    output_dir = DEFAULT_OUTPUT_DIR
    ensure_output_dir(output_dir)

    output_path = f"{output_dir}/{DEFAULT_OUTPUT_FILE}"

    train_and_save_global_bigram_model(
        DEFAULT_SUBREDDITS,
        base_data_dir=DEFAULT_BASE_DATA_DIR,
        output_path=output_path,
        min_count=DEFAULT_MIN_COUNT,
        threshold=DEFAULT_THRESHOLD,
    )


if __name__ == "__main__":
    main()