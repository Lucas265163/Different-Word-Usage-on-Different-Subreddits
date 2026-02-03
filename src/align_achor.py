"""
Align embeddings using anchor words selected from a global vocabulary.
Outputs:
1) A cosine distance CSV for analyzed words (sorted high -> low).
2) A CSV listing the anchor words used for alignment.

This method:
- Builds a global vocab (overlap across all word_frequency CSVs + Google 10k list).
- Uses the provided anchors directly (capped at 1500 if larger).
- Aligns with anchors using orthogonal Procrustes.
- Computes cosine distance for analysis vocabulary.
"""

import os
import glob
import numpy as np
import pandas as pd
from alignment_utils import (
    load_models,
    mean_center_and_normalize,
    procrustes_rotation,
    cosine_distance,
    save_distance_csv,
    save_word_list_csv,
    ensure_dir
)

# =====================
# Configuration
# =====================
MODEL_DIR = "data/models"
OUTPUT_DIR = "data/output/anchor_words"
WORD_FREQ_DIR = "data/word_frequency"
GOOGLE_VOCAB_PATH = os.path.join(WORD_FREQ_DIR, "google-10000-english-usa.txt")

SUBREDDITS = ("republican", "democrats")
PERIODS = ["before_2016", "2017_2020", "2021_2024"]

TOP_FREQ_RATIO = 0.6
ANCHOR_LIMIT = 1500

# =====================
# Core Logic
# =====================

def load_word_freq_overlap(csv_dir: str) -> set:
    """
    Compute overlap across all *_word_freq.csv files in csv_dir.
    Each CSV is expected to have a 'word' column.
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    word_sets = []
    for path in csv_files:
        df = pd.read_csv(path)
        if "word" not in df.columns:
            continue
        word_sets.append(set(df["word"].astype(str).tolist()))

    if not word_sets:
        return set()

    return set.intersection(*word_sets)

def load_google_vocab(filepath: str) -> set:
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def build_global_vocab() -> set:
    overlap_vocab = load_word_freq_overlap(WORD_FREQ_DIR)
    google_vocab = load_google_vocab(GOOGLE_VOCAB_PATH)
    return overlap_vocab | google_vocab

def get_top_vocab(model, ratio):
    vocab_sorted = sorted(
        model.wv.index_to_key,
        key=lambda w: model.wv.get_vecattr(w, "count"),
        reverse=True
    )
    return vocab_sorted[:int(ratio * len(vocab_sorted))]

def align_and_measure(model_a, model_b, global_vocab: set):
    vocab_a = set(get_top_vocab(model_a, TOP_FREQ_RATIO))
    vocab_b = set(get_top_vocab(model_b, TOP_FREQ_RATIO))

    # Anchors: global vocab intersected with both models
    anchors = [w for w in global_vocab if w in vocab_a and w in vocab_b]

    if len(anchors) > ANCHOR_LIMIT:
        anchors = anchors[:ANCHOR_LIMIT]

    vecs_a = mean_center_and_normalize(np.array([model_a.wv[w] for w in anchors]))
    vecs_b = mean_center_and_normalize(np.array([model_b.wv[w] for w in anchors]))

    rotation = procrustes_rotation(vecs_a, vecs_b)

    # Analysis vocabulary: full intersection of the two models
    common_vocab = sorted(vocab_a.intersection(vocab_b))

    vecs_a_all = mean_center_and_normalize(np.array([model_a.wv[w] for w in common_vocab]))
    vecs_b_all = mean_center_and_normalize(np.array([model_b.wv[w] for w in common_vocab])) @ rotation

    distances_all = cosine_distance(vecs_a_all, vecs_b_all)
    return common_vocab, distances_all, anchors

def main():
    ensure_dir(OUTPUT_DIR)
    models = load_models(MODEL_DIR, SUBREDDITS, PERIODS)

    global_vocab = build_global_vocab()

    sub1, sub2 = SUBREDDITS
    for period in PERIODS:
        print(f"Processing {period}...")

        vocab, distances, anchors = align_and_measure(
            models[sub1][period],
            models[sub2][period],
            global_vocab
        )

        dist_path = os.path.join(OUTPUT_DIR, f"{sub1}_{sub2}_{period}_distances.csv")
        anchor_path = os.path.join(OUTPUT_DIR, f"{sub1}_{sub2}_{period}_calibration_words.csv")

        save_distance_csv(dist_path, vocab, distances)
        save_word_list_csv(anchor_path, anchors, column_name="anchor_words")

if __name__ == "__main__":
    main()