import os
import glob
import numpy as np
import pandas as pd
import gensim

# =====================
# Configuration Helpers
# =====================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def build_model_path(model_dir: str, subreddit: str, period: str) -> str:
    return os.path.join(model_dir, f"{subreddit}_{period}.model")

# =====================
# Model Loading
# =====================

def load_models(model_dir, subreddits, periods):
    """Load models into a nested dict: models[subreddit][period]."""
    models = {sub: {} for sub in subreddits}
    for sub in subreddits:
        for period in periods:
            path = build_model_path(model_dir, sub, period)
            models[sub][period] = gensim.models.Word2Vec.load(path)
    return models

# =====================
# Vector Processing
# =====================

def mean_center_and_normalize(vectors: np.ndarray) -> np.ndarray:
    """Mean-center and L2-normalize vectors."""
    vectors = vectors - vectors.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def procrustes_rotation(vecs_ref: np.ndarray, vecs_target: np.ndarray) -> np.ndarray:
    """Compute orthogonal Procrustes rotation matrix."""
    m = vecs_ref.T @ vecs_target
    u, _, vt = np.linalg.svd(m)
    return u @ vt

def cosine_distance(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine similarity."""
    return 1 - np.sum(vecs_a * vecs_b, axis=1)

# =====================
# Helper Functions
# =====================

def get_word_frequency(word, model_rep, model_dem):
    """
    Looks up the count of a word in both models and returns the sum.
    Returns 0 if the word is missing in the model.
    """
    count_rep = model_rep.wv.get_vecattr(word, "count") if word in model_rep.wv else 0
    count_dem = model_dem.wv.get_vecattr(word, "count") if word in model_dem.wv else 0
    return count_rep + count_dem

# =====================
# CSV Output
# =====================

def save_distance_csv(output_path: str, words, distances):
    df = pd.DataFrame({"word": words, "cosine_distance": distances})
    df = df.sort_values("cosine_distance", ascending=False)
    df.to_csv(output_path, index=False)

def save_word_list_csv(output_path: str, words, column_name="word"):
    df = pd.DataFrame({column_name: words})
    df.to_csv(output_path, index=False)