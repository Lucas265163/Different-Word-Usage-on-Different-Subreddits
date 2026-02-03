"""
Iterative alignment between two embedding models.
Outputs:
1) A cosine distance CSV for analyzed words (sorted high -> low).
2) A CSV listing the second-round alignment words (used in iteration).

This method:
- Starts with a large shared vocabulary.
- Aligns once, evaluates alignment quality.
- Selects a refined word set for a second alignment.
- Aligns again and measures cosine distance.
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
OUTPUT_DIR = "data/output/iterative"
SUBREDDITS = ("liberal", "democrats")
PERIODS = ["before_2016", "2017_2020", "2021_2024"]
TOP_FREQ_RATIO = 0.6
INITIAL_ANCHORS = 3000
REFINED_ANCHORS = 1500

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
    top_a = set(get_top_vocab(model_a, TOP_FREQ_RATIO))
    top_b = set(get_top_vocab(model_b, TOP_FREQ_RATIO))

    common_vocab = list(top_a.intersection(top_b))
    common_vocab.sort(
        key=lambda w: model_a.wv.get_vecattr(w, "count") + model_b.wv.get_vecattr(w, "count"),
        reverse=True
    )

    anchors_round1 = common_vocab[:INITIAL_ANCHORS]
    vecs_a1 = mean_center_and_normalize(np.array([model_a.wv[w] for w in anchors_round1]))
    vecs_b1 = mean_center_and_normalize(np.array([model_b.wv[w] for w in anchors_round1]))

    rotation1 = procrustes_rotation(vecs_a1, vecs_b1)
    vecs_b1_rot = vecs_b1 @ rotation1

    distances1 = cosine_distance(vecs_a1, vecs_b1_rot)
    anchor_scores = sorted(zip(anchors_round1, distances1), key=lambda x: x[1])
    anchors_round2 = [w for w, _ in anchor_scores[:REFINED_ANCHORS]]

    vecs_a2 = mean_center_and_normalize(np.array([model_a.wv[w] for w in anchors_round2]))
    vecs_b2 = mean_center_and_normalize(np.array([model_b.wv[w] for w in anchors_round2]))

    rotation2 = procrustes_rotation(vecs_a2, vecs_b2)

    vecs_a_all = mean_center_and_normalize(np.array([model_a.wv[w] for w in common_vocab]))
    vecs_b_all = mean_center_and_normalize(np.array([model_b.wv[w] for w in common_vocab])) @ rotation2

    distances_all = cosine_distance(vecs_a_all, vecs_b_all)
    return common_vocab, distances_all, anchors_round2

def main():
    ensure_dir(OUTPUT_DIR)
    models = load_models(MODEL_DIR, SUBREDDITS, PERIODS)

    sub1, sub2 = SUBREDDITS
    for period in PERIODS:
        print(f"Processing {period}...")

        vocab, distances, anchors = align_and_measure(
            models[sub1][period],
            models[sub2][period]
        )

        dist_path = os.path.join(OUTPUT_DIR, f"{sub1}_{sub2}_{period}_distances.csv")
        anchors_path = os.path.join(OUTPUT_DIR, f"{sub1}_{sub2}_{period}_calibration_words.csv")

        save_distance_csv(dist_path, vocab, distances)
        save_word_list_csv(anchors_path, anchors, column_name="iterative_anchor_words")

if __name__ == "__main__":
    main()