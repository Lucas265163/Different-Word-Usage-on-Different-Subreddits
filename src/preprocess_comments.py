import datetime
import html
import os
import pickle
import random
import re
import sys
import unicodedata

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from fileStreams import getFileJsonStream


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
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)

    # Remove URLs and Markdown formatting
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Remove subreddit and user references
    text = re.sub(r"/r/\w+", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"/u/\w+", "", text)
    text = re.sub(r"u/\w+", "", text)

    # Keep only letters and spaces
    return re.sub("[^A-Za-z]+", " ", text).lower()


def lemmatize_words(words):
    """Lemmatize words with cached POS tags and lemmas."""
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


def preprocess_text(text, lemmatize=True, without_stopwords=True):
    """Preprocess Reddit text content with optimized NLTK operations."""
    text = normalize_text(text)
    words = text.split()
    if not words:
        return []

    if lemmatize:
        words = lemmatize_words(words)

    if without_stopwords:
        words = [w for w in words if w not in STOP_WORDS]

    # Keep words with length 3..15
    return [w for w in words if 2 < len(w) <= 15]


def build_comment_data(row, processed_words):
    """Build the output record for a single comment."""
    created_timestamp = row["created_utc"]
    date = datetime.datetime.fromtimestamp(int(created_timestamp))
    return {
        "comment_id": row["id"],
        "author": row["author"],
        "date": date.strftime("%Y-%m-%d"),
        "timestamp": created_timestamp,
        "processed_text": processed_words,
        "original": row["body"],
    }


def save_batch(comments_batch, output_dir, subreddit, batch_number):
    """Save one batch to disk with the exact same output format."""
    save_path = f"{output_dir}/{subreddit}_batch{batch_number}.pkl"
    with open(save_path, "wb") as out_file:
        pickle.dump(comments_batch, out_file)
    print(f"Saved {len(comments_batch)} comments to {save_path}")


def process_and_save_comments(path, subreddit, output_dir, without_stopwords=True, batch_size=1000000):
    """Process comments and save in batches."""
    print(f"Processing file: {path}")

    batch_count = 0
    batch_number = 1
    total_count = 0
    comments_batch = []

    with open(path, "rb") as f:
        jsonStream = getFileJsonStream(path, f)
        if jsonStream is None:
            print(f"Unable to read file {path}")
            return

        for row in tqdm(jsonStream, desc=f"Processing {subreddit} comments"):
            if "body" not in row or "created_utc" not in row or "author" not in row or "id" not in row:
                continue

            author = row["author"]
            if author in {"AutoModerator", "election_info_bot"}:
                continue

            processed_words = preprocess_text(
                row["body"],
                lemmatize=True,
                without_stopwords=without_stopwords,
            )

            if processed_words:
                comments_batch.append(build_comment_data(row, processed_words))
                batch_count += 1

            if batch_count >= batch_size:
                print(f"\nReached {batch_size} comments, saving batch {batch_number}...")
                save_batch(comments_batch, output_dir, subreddit, batch_number)
                comments_batch = []
                batch_count = 0
                batch_number += 1
                total_count += batch_size

    if batch_count > 0:
        print(f"\nSaving remaining {batch_count} comments...")
        save_batch(comments_batch, output_dir, subreddit, batch_number)
        total_count += batch_count

    print(f"\nCompleted processing {subreddit} comments!")
    print(f"Total comments saved: {total_count}")


def main():
    """Main function."""
    random.seed(23)
    np.random.seed(23)

    files = {
        "democrats": r"data/raw/democrats_comments.zst",
        "republican": r"data/raw/Republican_comments.zst",
        "liberal": r"data/raw/Liberal_comments.zst"
    }

    subreddits_to_process = list(files.keys())

    for subreddit in subreddits_to_process:
        output_dir = f"data/processed_comments/{subreddit}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing subreddit: {subreddit}")
        process_and_save_comments(
            files[subreddit],
            subreddit,
            output_dir,
            without_stopwords=False,
            batch_size=1_000_000
        )


if __name__ == "__main__":
    main()