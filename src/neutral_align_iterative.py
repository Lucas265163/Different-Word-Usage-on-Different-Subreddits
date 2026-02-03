"""
Iterative alignment between Political models and a single Neutral model.
Outputs:
1) Cosine distance CSVs for Dem vs Neutral and Rep vs Neutral per period.
"""

import os
import numpy as np
import gensim
from alignment_utils import (
    load_models, # Assuming this helper exists from your previous context
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
NEUTRAL_MODEL_PATH = "data/models/neutral/neutral.model"
OUTPUT_DIR = "data/output/neutral_comparison"
SUBREDDITS = ["democrats", "republican"]
PERIODS = ["before_2016", "2017_2020", "2021_2024"]

# Alignment Hyperparameters
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

def align_and_measure(model_political, model_neutral):
    """
    Aligns a specific political model to the static neutral model using
    iterative Procrustes alignment.
    """
    top_pol = set(get_top_vocab(model_political, TOP_FREQ_RATIO))
    top_neu = set(get_top_vocab(model_neutral, TOP_FREQ_RATIO))

    # Identify Common Vocabulary
    common_vocab = list(top_pol.intersection(top_neu))
    
    # Sort by combined frequency (approximation using count)
    common_vocab.sort(
        key=lambda w: model_political.wv.get_vecattr(w, "count") + model_neutral.wv.get_vecattr(w, "count"),
        reverse=True
    )

    # First Round Alignment
    anchors_round1 = common_vocab[:INITIAL_ANCHORS]
    
    # Get vectors for anchors
    vecs_pol1 = mean_center_and_normalize(np.array([model_political.wv[w] for w in anchors_round1]))
    vecs_neu1 = mean_center_and_normalize(np.array([model_neutral.wv[w] for w in anchors_round1]))

    # Align Political to Neutral (Neutral is the reference)
    rotation1 = procrustes_rotation(vecs_pol1, vecs_neu1)
    vecs_pol1_rot = vecs_pol1 @ rotation1

    # Measure distance to find best anchors
    distances1 = cosine_distance(vecs_pol1_rot, vecs_neu1)
    anchor_scores = sorted(zip(anchors_round1, distances1), key=lambda x: x[1])
    
    # Select best anchors (lowest distance)
    anchors_round2 = [w for w, _ in anchor_scores[:REFINED_ANCHORS]]

    # Second Round Alignment (Refined)
    vecs_pol2 = mean_center_and_normalize(np.array([model_political.wv[w] for w in anchors_round2]))
    vecs_neu2 = mean_center_and_normalize(np.array([model_neutral.wv[w] for w in anchors_round2]))

    rotation2 = procrustes_rotation(vecs_pol2, vecs_neu2)

    # Final Application to All Common Words
    vecs_pol_all = mean_center_and_normalize(np.array([model_political.wv[w] for w in common_vocab])) @ rotation2
    vecs_neu_all = mean_center_and_normalize(np.array([model_neutral.wv[w] for w in common_vocab]))

    distances_all = cosine_distance(vecs_pol_all, vecs_neu_all)
    
    return common_vocab, distances_all, anchors_round2

def main():
    ensure_dir(OUTPUT_DIR)
    
    # Load the single Neutral Model
    print(f"Loading Neutral Model from {NEUTRAL_MODEL_PATH}...")
    neutral_model = gensim.models.Word2Vec.load(NEUTRAL_MODEL_PATH)
    
    # Load Political Models
    print("Loading Political Models...")
    # Assuming load_models returns dict[subreddit][period] -> model
    political_models = load_models(MODEL_DIR, SUBREDDITS, PERIODS)

    # Iterate and Process
    for sub in SUBREDDITS:
        for period in PERIODS:
            print(f"Processing {sub} vs Neutral ({period})...")
            
            curr_pol_model = political_models[sub][period]
            
            vocab, distances, anchors = align_and_measure(
                curr_pol_model, 
                neutral_model
            )

            # Save filenames: e.g., "democrats_vs_neutral_before_2016.csv"
            base_name = f"{sub}_vs_neutral_{period}"
            dist_path = os.path.join(OUTPUT_DIR, f"{base_name}_distances.csv")
            anchors_path = os.path.join(OUTPUT_DIR, f"{base_name}_anchors.csv")

            save_distance_csv(dist_path, vocab, distances)
            save_word_list_csv(anchors_path, anchors, column_name="anchor_words")
            
            print(f"Saved: {dist_path}")

if __name__ == "__main__":
    main()