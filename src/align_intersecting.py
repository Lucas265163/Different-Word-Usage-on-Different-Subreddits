"""
Align embeddings using all shared vocabulary (intersection) between two models.
Outputs:
1) A cosine distance CSV for all aligned shared words (sorted high -> low).
2) A CSV listing all shared words used for alignment.

This method:
- Uses the full vocabulary intersection as alignment anchors.
- Mean-centers and normalizes vectors.
- Applies orthogonal Procrustes rotation.
- Computes cosine distance for each shared word after alignment.
"""


import os
import numpy as np
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
OUTPUT_DIR = "data/output/intersecting_words"
SUBREDDITS = ("republican", "democrats")
PERIODS = ["before_2016", "2017_2020", "2021_2024"]
TOP_FREQ_RATIO = 0.6

# =====================
# Core Logic
# =====================
def get_top_vocab(model, ratio):
    vocab_sorted = sorted(
        model.wv.index_to_key,
        key=lambda w: model.wv.get_vecattr(w, "count"),
        reverse=True
    )
    return vocab_sorted[:int(ratio * len(vocab_sorted))]

def align_and_measure(model_a, model_b):
    vocab_a = set(get_top_vocab(model_a, TOP_FREQ_RATIO))
    vocab_b = set(get_top_vocab(model_b, TOP_FREQ_RATIO))
    common_vocab = sorted(vocab_a.intersection(vocab_b))

    vecs_a = np.array([model_a.wv[w] for w in common_vocab])
    vecs_b = np.array([model_b.wv[w] for w in common_vocab])

    vecs_a = mean_center_and_normalize(vecs_a)
    vecs_b = mean_center_and_normalize(vecs_b)

    rotation = procrustes_rotation(vecs_a, vecs_b)
    vecs_b_aligned = vecs_b @ rotation

    distances = cosine_distance(vecs_a, vecs_b_aligned)
    return common_vocab, distances

def main():
    ensure_dir(OUTPUT_DIR)
    models = load_models(MODEL_DIR, SUBREDDITS, PERIODS)

    sub1, sub2 = SUBREDDITS
    for period in PERIODS:
        print(f"Processing {period}...")

        common_vocab, distances = align_and_measure(
            models[sub1][period],
            models[sub2][period]
        )

        dist_path = os.path.join(OUTPUT_DIR, f"{sub1}_{sub2}_{period}_distances.csv")
        vocab_path = os.path.join(OUTPUT_DIR, f"{sub1}_{sub2}_{period}_calibration_words.csv")

        save_distance_csv(dist_path, common_vocab, distances)
        save_word_list_csv(vocab_path, common_vocab, column_name="common_vocab")

if __name__ == "__main__":
    main()