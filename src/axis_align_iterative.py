import os
import numpy as np
import pandas as pd
import gensim
from typing import List, Tuple

# Use shared utilities for file handling
# Added get_word_frequency to imports
from alignment_utils import load_models, ensure_dir, get_word_frequency

# =====================
# Configuration
# =====================
MODEL_DIR = "data/models"
OUTPUT_DIR = "data/output/axis"

SUBREDDITS = ["republican", "democrats"]
PERIODS = ["before_2016", "2017_2020", "2021_2024"]

# Canonical Ideological Seeds
AXIS_SEEDS = [
    ("conservative", "liberal"),
    ("republican", "democrat")
]

# Hyperparameters
ALIGN_VOCAB_RATIO = 0.6
INITIAL_ANCHORS = 3000
REFINED_ANCHORS = 1500
ANALYSIS_VOCAB_RATIO = 1  # Top 80% of common vocab used for final scoring

# Seed Expansion Config
EXPAND_K = 100
EXPAND_PAIRS = 10
EXPAND_MIN_COUNT = 50
EXPAND_THRESHOLD = 0.3


# =====================
# Vector Helper Functions
# =====================

def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a single vector (L2)."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _get_neighbors(model, token: str, topn: int) -> List[str]:
    """Safely retrieve neighbors for a token."""
    try:
        return [w for w, _ in model.wv.most_similar(token, topn=topn)]
    except KeyError:
        return []


def get_processed_vectors(model, words, center=True):
    """Batch retrieve, mean-center, and normalize vectors."""
    vecs = np.array([model.wv[word] for word in words])
    if center:
        vecs = vecs - vecs.mean(axis=0)
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1 
    return vecs / norm


# =====================
# Seed Expansion Logic
# =====================

def expand_seeds(
    model,
    human_seeds: List[Tuple[str, str]],
    top_k: int = 50,
    num_pairs: int = 10,
    min_count: int = 50,
    agg: str = "max",
    score_threshold: float = 0.3,
    verbose: bool = True
) -> List[Tuple[str, str]]:
    """
    Expands human seeds by finding similar pairs in the embedding space.
    """
    # 1. Build allowed vocab
    vocab_sorted = sorted(model.wv.index_to_key, key=lambda w: model.wv.get_vecattr(w, "count"), reverse=True)
    allowed_vocab = set(vocab_sorted[:int(0.6 * len(vocab_sorted))])

    # 2. Validate & build canonical seed directions
    seed_dirs = []
    valid_human_seeds = []
    for r_seed, l_seed in human_seeds:
        if r_seed in model.wv and l_seed in model.wv:
            v_r = _normalize(model.wv[r_seed])
            v_l = _normalize(model.wv[l_seed])
            seed_dirs.append(_normalize(v_r - v_l))
            valid_human_seeds.append((r_seed, l_seed))
        else:
            if verbose:
                print(f"[expand_seeds_matched] Warning: skipping seed ({r_seed}, {l_seed}) — missing from vocab")

    if len(seed_dirs) == 0:
        raise ValueError("No valid human seeds found in model vocabulary.")

    seed_dirs = np.stack(seed_dirs, axis=0)

    # 3. Retrieve neighbors for each seed pair
    right_neighbors = {}
    left_neighbors = {}
    for r_seed, l_seed in valid_human_seeds:
        r_neigh = _get_neighbors(model, r_seed, top_k)
        l_neigh = _get_neighbors(model, l_seed, top_k)
        
        if allowed_vocab is not None:
            r_neigh = [w for w in r_neigh if w in allowed_vocab]
            l_neigh = [w for w in l_neigh if w in allowed_vocab]
        
        if min_count is not None:
            r_neigh = [w for w in r_neigh if model.wv.get_vecattr(w, "count") >= min_count]
            l_neigh = [w for w in l_neigh if model.wv.get_vecattr(w, "count") >= min_count]
            
        right_neighbors[(r_seed, l_seed)] = r_neigh
        left_neighbors[(r_seed, l_seed)] = l_neigh

    # 4. Build candidate pairs
    candidate_pairs = set()
    for key in right_neighbors:
        for r in right_neighbors[key]:
            for l in left_neighbors[key]:
                if r == l: continue
                if min_count is not None:
                    if model.wv.get_vecattr(r, "count") < min_count or model.wv.get_vecattr(l, "count") < min_count:
                        continue
                candidate_pairs.add((r, l))
    candidate_pairs = list(candidate_pairs)
    
    if len(candidate_pairs) == 0:
        if verbose: print("No candidate pairs generated.")
        return valid_human_seeds[:num_pairs]

    # 5. Precompute normalized vectors for scoring
    vocab_for_cache = set([w for pair in candidate_pairs for w in pair] + [w for sd in valid_human_seeds for w in sd])
    vec_cache = {w: _normalize(model.wv[w]) for w in vocab_for_cache if w in model.wv}

    # 6. Score candidates
    records = []
    for (r, l) in candidate_pairs:
        vr = vec_cache.get(r)
        vl = vec_cache.get(l)
        if vr is None or vl is None: continue
        
        d_cand = _normalize(vr - vl)
        sims = seed_dirs.dot(d_cand)
        score = float(np.max(sims)) if agg == "max" else float(np.mean(sims))
        
        records.append({
            "r": r, "l": l, "score": score,
            "freq_r": model.wv.get_vecattr(r, "count"),
            "freq_l": model.wv.get_vecattr(l, "count")
        })

    df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

    # 7. Greedy selection
    ban_list = [s for sd in valid_human_seeds for s in sd]
    selected = []
    i = 0
    while len(selected) < (num_pairs - len(valid_human_seeds)) and i < len(df):
        row = df.iloc[i]
        if row['r'] in ban_list or row['l'] in ban_list:
            i += 1
            continue
        if row['score'] < score_threshold:
            break
        selected.append((row['r'], row['l'], row['score'], row['freq_r'], row['freq_l']))
        ban_list.extend([row['r'], row['l']])
        i += 1

    final_pairs = list(valid_human_seeds) + [(r, l) for r, l, _, _, _ in selected]
    print(f'final pairs: {final_pairs}')
    return final_pairs[:num_pairs]


# =====================
# Alignment Logic
# =====================

def align_models(model_base, model_target):
    """
    Aligns model_target to model_base using Iterative Procrustes.
    (Logic copied exactly from provided snippet)
    """
    # 1. Identify Common Vocabulary
    rep_vocab_sorted = sorted(model_base.wv.index_to_key, key=lambda w: model_base.wv.get_vecattr(w, "count"), reverse=True)
    dem_vocab_sorted = sorted(model_target.wv.index_to_key, key=lambda w: model_target.wv.get_vecattr(w, "count"), reverse=True)
    
    num_rep = int(ALIGN_VOCAB_RATIO * len(rep_vocab_sorted))
    num_dem = int(ALIGN_VOCAB_RATIO * len(dem_vocab_sorted))
    
    top_rep = set(rep_vocab_sorted[:num_rep])
    top_dem = set(dem_vocab_sorted[:num_dem])
    
    common_vocab = list(top_rep.intersection(top_dem))
    common_vocab.sort(key=lambda w: model_base.wv.get_vecattr(w, "count") + model_target.wv.get_vecattr(w, "count"), reverse=True)
    
    # 2. Rough Alignment
    initial_anchors = common_vocab[:INITIAL_ANCHORS]
    vecs_base_rough = get_processed_vectors(model_base, initial_anchors, center=True)
    vecs_target_rough = get_processed_vectors(model_target, initial_anchors, center=True)
    
    m = vecs_target_rough.T @ vecs_base_rough
    u, _, vt = np.linalg.svd(m)
    rotation_1 = u @ vt
    vecs_target_rotated = vecs_target_rough @ rotation_1
    
    # 3. Filter Anchors
    similarities = np.sum(vecs_base_rough * vecs_target_rotated, axis=1)
    distances = 1 - similarities
    anchor_scores = sorted(zip(initial_anchors, distances), key=lambda x: x[1])
    
    refined_anchors = [w for w, d in anchor_scores[:REFINED_ANCHORS]]
    
    # 4. Final Alignment
    vecs_base_final = get_processed_vectors(model_base, refined_anchors, center=True)
    vecs_target_final = get_processed_vectors(model_target, refined_anchors, center=True)
    
    m_final = vecs_target_final.T @ vecs_base_final
    u_final, _, vt_final = np.linalg.svd(m_final)
    rotation_final = u_final @ vt_final
    
    model_target.wv.vectors = model_target.wv.vectors @ rotation_final
    
    # 5. Center Adjustment (Crucial step from your snippet)
    mean_base = np.mean(model_base.wv[refined_anchors], axis=0)
    mean_target = np.mean(model_target.wv[refined_anchors], axis=0)
    model_target.wv.vectors = model_target.wv.vectors + (mean_base - mean_target)
    
    if hasattr(model_target.wv, 'fill_norms'):
        model_target.wv.fill_norms(force=True)
        
    return model_target, common_vocab


# =====================
# Axis Logic
# =====================

def construct_semantic_axis(model, seeds):
    """Construct semantic axis from seed pairs."""
    axis_vectors = []
    for right, left in seeds:
        if right in model.wv and left in model.wv:
            v_r = model.wv[right]
            v_l = model.wv[left]
            v_r = v_r / np.linalg.norm(v_r)
            v_l = v_l / np.linalg.norm(v_l)
            axis_vectors.append(v_r - v_l)
            
    if not axis_vectors: 
        return None
    
    final_axis = np.mean(axis_vectors, axis=0)
    final_axis = final_axis / np.linalg.norm(final_axis)
    return final_axis


def calculate_polarization_scores(model_rep, model_dem, vocab, axis_vector):
    """Project words onto axis and calculate polarization."""
    results = []
    for word in vocab:
        v_rep = model_rep.wv[word]
        v_dem = model_dem.wv[word]
        
        v_rep = v_rep / np.linalg.norm(v_rep)
        v_dem = v_dem / np.linalg.norm(v_dem)
        
        proj_rep = np.dot(v_rep, axis_vector)
        proj_dem = np.dot(v_dem, axis_vector)
        polarization = abs(proj_rep - proj_dem)
        
        # Calculate frequency using helper from alignment_utils
        total_freq = get_word_frequency(word, model_rep, model_dem)

        results.append({
            "word": word,
            "rep_score": proj_rep,
            "dem_score": proj_dem,
            "polarization": polarization,
            "total_freq": total_freq  # Added to results
        })
    return results


def main():
    ensure_dir(OUTPUT_DIR)
    
    # Set display options
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    # Load Models
    print(f"Loading models from {MODEL_DIR}...")
    models = load_models(MODEL_DIR, SUBREDDITS, PERIODS)

    for period in PERIODS:
        print(f"\n=== Processing Period: {period} ===")
        
        # 1. Get models
        try:
            model_rep = models["republican"][period]
            model_dem = models["democrats"][period]
        except KeyError:
            print(f"Skipping {period} - models not found.")
            continue
        
        # 2. Align Models
        print("Aligning models...")
        model_dem, common_vocab = align_models(model_rep, model_dem)
        
        # 3. Construct Axis with Expansion
        current_seeds = expand_seeds(
            model=model_rep, 
            human_seeds=AXIS_SEEDS, 
            top_k=EXPAND_K, 
            num_pairs=EXPAND_PAIRS, 
            min_count=EXPAND_MIN_COUNT,
            agg='max', 
            score_threshold=EXPAND_THRESHOLD
        )
        
        axis_vector = construct_semantic_axis(model_rep, current_seeds)
        if axis_vector is None:
            print("Error: Seeds not found.")
            continue
            
        # 4. Analyze Polarization
        num_anchors = int(ANALYSIS_VOCAB_RATIO * len(common_vocab))
        core_vocab = common_vocab[:num_anchors]
        
        results = calculate_polarization_scores(model_rep, model_dem, core_vocab, axis_vector)
        
        # 5. Save and Print
        df = pd.DataFrame(results).sort_values("polarization", ascending=False)
        filename = f"{period}_polarization.csv"
        df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
        
        print(f"\n--- Top 15 Polarized Words ({period}) ---")
        # Added total_freq to print output
        print(df.head(15)[['word', 'polarization', 'rep_score', 'dem_score', 'total_freq']])


if __name__ == "__main__":
    main()