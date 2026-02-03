import os
import numpy as np
import matplotlib.pyplot as plt
import umap
import gensim
import copy

# Import shared utilities and alignment logic
from align_achor import OUTPUT_DIR
from alignment_utils import load_models, ensure_dir, get_word_frequency
from axis_align_iterative import (
    MODEL_DIR, 
    SUBREDDITS, 
    PERIODS,
    align_models,       
    get_processed_vectors 
)

# =====================
# Configuration
# =====================
OUTPUT_DIR = "data/umap_visualizations"

HIGHLIGHT_WORDS = [
    'remove', 'trump', 'people', 'get', 'like', 
    'vote', 'say', 'would', 'think', 'make', 
    'freedom', 'rights', 'america'
]

UMAP_PARAMS = {
    'n_neighbors': 15,
    'n_components': 2,
    'min_dist': 0.1,
    'metric': 'cosine',
    'random_state': 42 
}

# =====================
# UMAP Helper Functions
# =====================

def prepare_vectors_and_freqs(model):
    """
    Extracts vectors, words, and log-frequencies from a model.
    """
    words = list(model.wv.index_to_key)
    vectors = get_processed_vectors(model, words, center=True)
    freqs = np.array([model.wv.get_vecattr(w, "count") for w in words])
    log_freqs = np.log1p(freqs)
    return words, vectors, log_freqs

def plot_umap_comparison(reducer, vectors, log_freqs, words, title, color_map, label_prefix, ax=None):
    """
    Plots a single model's vectors using a pre-fitted reducer.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Transform vectors using the shared reducer
    coords = reducer.transform(vectors)
    
    # Plot background points (Higher Alpha and Size)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=log_freqs, 
        cmap=color_map, 
        alpha=0.6,   # Darker/More visible
        s=10,        # Larger points
        label=f'{label_prefix} Vocab'
    )
    
    # Highlight specific words
    for hw in HIGHLIGHT_WORDS:
        if hw in words:
            idx = words.index(hw)
            # Add black edge and zorder to make text pop
            ax.scatter(coords[idx, 0], coords[idx, 1], color='red', s=80, edgecolors='black', linewidth=1, zorder=10)
            ax.text(
                coords[idx, 0], coords[idx, 1], 
                hw, 
                color='black', 
                fontsize=11, 
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'),
                zorder=11
            )
            
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return sc

# =====================
# Main Visualization Logic
# =====================

def visualize_period_alignment(period, models_dict):
    print(f"\n--- Visualizing Period: {period} ---")
    
    # 1. Load Original Models
    try:
        model_rep = models_dict["republican"][period]
        model_dem = models_dict["democrats"][period]
    except KeyError:
        print(f"Skipping {period}, models not found.")
        return

    # Deep copy for unaligned version
    model_dem_unaligned = copy.deepcopy(model_dem)
    
    # 2. Prepare Data
    print("Preparing vectors...")
    words_rep, vecs_rep, freq_rep = prepare_vectors_and_freqs(model_rep)
    words_dem_raw, vecs_dem_raw, freq_dem_raw = prepare_vectors_and_freqs(model_dem_unaligned)

    # 3. Fit UMAP on Republican Model (The "Anchor")
    print("Fitting UMAP on Republican model (Reference)...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    reducer.fit(vecs_rep)

    # 4. Plot 1: Unaligned Comparison
    # Using 'Blues' for Rep and 'Greens' for Dem (Consistent with "After")
    print("Plotting Unaligned State...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    plot_umap_comparison(reducer, vecs_rep, freq_rep, words_rep, 
                         f"Republican (Reference) - {period}", "Blues", "Rep", ax=axes[0])
    
    plot_umap_comparison(reducer, vecs_dem_raw, freq_dem_raw, words_dem_raw, 
                         f"Democrat (Unaligned) - {period}", "Greens", "Dem", ax=axes[1])
    
    plt.suptitle(f"Before Alignment: {period}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{period}_umap_before_align.png"), dpi=150)
    
    # 5. Perform Alignment
    print("Running Iterative Procrustes Alignment...")
    model_dem_aligned, common_vocab = align_models(model_rep, model_dem)
    
    # 6. Prepare Aligned Vectors
    words_dem_aligned, vecs_dem_aligned, freq_dem_aligned = prepare_vectors_and_freqs(model_dem_aligned)
    
    # 7. Plot 2: Aligned Comparison
    # Using 'Blues' for Rep and 'Greens' for Dem
    print("Plotting Aligned State...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    
    plot_umap_comparison(reducer, vecs_rep, freq_rep, words_rep, 
                         f"Republican (Reference) - {period}", "Blues", "Rep", ax=axes2[0])
    
    plot_umap_comparison(reducer, vecs_dem_aligned, freq_dem_aligned, words_dem_aligned, 
                         f"Democrat (Aligned) - {period}", "Greens", "Dem", ax=axes2[1])
    
    plt.suptitle(f"After Alignment: {period}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{period}_umap_after_align.png"), dpi=150)
    
    # 8. Plot 3: Overlay
    print("Plotting Overlay...")
    plt.figure(figsize=(12, 10))
    
    coords_rep = reducer.transform(vecs_rep)
    coords_dem = reducer.transform(vecs_dem_aligned)
    
    # Solid Blue and Solid Green for overlay clarity
    plt.scatter(coords_rep[:, 0], coords_rep[:, 1], c='blue', alpha=0.3, s=10, label='Republican')
    plt.scatter(coords_dem[:, 0], coords_dem[:, 1], c='green', alpha=0.3, s=10, label='Democrat (Aligned)')
    
    # Highlight common words
    for hw in HIGHLIGHT_WORDS:
        if hw in common_vocab:
            idx_r = words_rep.index(hw)
            idx_d = words_dem_aligned.index(hw)
            
            # Rep Word (Blue)
            plt.scatter(coords_rep[idx_r, 0], coords_rep[idx_r, 1], c='blue', s=80, edgecolors='black', zorder=10)
            
            # Dem Word (Green)
            plt.scatter(coords_dem[idx_d, 0], coords_dem[idx_d, 1], c='green', s=80, edgecolors='black', zorder=10)
            
            # Connection Line
            plt.plot([coords_rep[idx_r, 0], coords_dem[idx_d, 0]], 
                     [coords_rep[idx_r, 1], coords_dem[idx_d, 1]], 
                     color='black', alpha=0.6, linewidth=1, linestyle='--')
            
            # Single Label (Midpoint or just one)
            mid_x = (coords_rep[idx_r, 0] + coords_dem[idx_d, 0]) / 2
            mid_y = (coords_rep[idx_r, 1] + coords_dem[idx_d, 1]) / 2
            plt.text(mid_x, mid_y, hw, color='black', fontsize=12, weight='bold', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), zorder=11)

    plt.title(f"Overlay of Aligned Spaces: {period}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{period}_umap_overlay.png"), dpi=150)


def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Loading models from {MODEL_DIR}...")
    models = load_models(MODEL_DIR, SUBREDDITS, PERIODS)
    
    # Loop through all periods
    for period in PERIODS:
        visualize_period_alignment(period, models)

if __name__ == "__main__":
    main()