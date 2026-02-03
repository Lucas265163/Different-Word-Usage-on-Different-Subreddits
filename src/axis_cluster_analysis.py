import os
import numpy as np
import pandas as pd
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import shared utilities
from alignment_utils import load_models, ensure_dir
from axis_align_iterative import (
    MODEL_DIR, 
    OUTPUT_DIR, 
    SUBREDDITS, 
    PERIODS
)

# =====================
# Configuration
# =====================
# Clustering Parameters
MIN_CLUSTER_SIZE = 20
MIN_SAMPLES = 10
NOISE_LABEL = -1

# Visualization Config
FIG_SIZE = (12, 10)
PALETTE_NAME = 'tab10'  # Matplotlib Tableau 10 palette
PLOT_ALPHA_NOISE = 0.3
PLOT_SIZE_NOISE = 10
PLOT_SIZE_CLUSTER = 20


# =====================
# Helper: Color Mapping
# =====================

def get_tab10_color_name(cluster_id: int) -> str:
    """
    Maps a cluster ID to its approximate color name in the Tab10 palette.
    Tab10 Order: Blue, Orange, Green, Red, Purple, Brown, Pink, Gray, Olive, Cyan
    """
    if cluster_id == NOISE_LABEL:
        return "Noise (Gray)"
    
    # Standard Tableau 10 color names
    colors = [
        "Blue", "Orange", "Green", "Red", "Purple", 
        "Brown", "Pink", "Gray", "Olive", "Cyan"
    ]
    
    # Cycle through colors if we have more than 10 clusters
    return colors[cluster_id % len(colors)]


# =====================
# Clustering Logic
# =====================

def run_hdbscan_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies HDBSCAN clustering to the semantic scores.
    """
    data = df[['rep_score', 'dem_score']].values
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, 
        min_samples=MIN_SAMPLES
    )
    
    df = df.copy()
    df['cluster'] = clusterer.fit_predict(data)
    return df


def visualize_and_save_clusters(df: pd.DataFrame, period_name: str, output_path: str) -> None:
    """
    Generates, shows, and saves the scatter plot for clusters.
    """
    plt.figure(figsize=FIG_SIZE)
    
    # 1. Plot Noise (Background)
    noise = df[df['cluster'] == NOISE_LABEL]
    plt.scatter(
        noise['rep_score'], 
        noise['dem_score'], 
        c='lightgray', 
        s=PLOT_SIZE_NOISE, 
        alpha=PLOT_ALPHA_NOISE, 
        label='Noise'
    )
    
    # 2. Plot Clusters
    clustered = df[df['cluster'] != NOISE_LABEL]
    if not clustered.empty:
        sns.scatterplot(
            data=clustered,
            x='rep_score',
            y='dem_score',
            hue='cluster',
            palette=PALETTE_NAME,
            s=PLOT_SIZE_CLUSTER,
            legend='full'
        )
    
    # 3. Styling
    plt.title(f'Semantic Clustering: {period_name}')
    plt.xlabel('Republican Alignment (Right -> Conservative)')
    plt.ylabel('Democrat Alignment (Right -> Conservative)')
    
    # Custom Legend with Color Names
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in labels:
        try:
            cid = int(label)
            new_labels.append(f"{cid} ({get_tab10_color_name(cid)})")
        except ValueError:
            new_labels.append(label) # Keep 'Noise' or other text
            
    plt.legend(handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster ID (Color)")
    
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.tight_layout()
    
    # Save before show
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# =====================
# Statistical Analysis
# =====================

def analyze_cluster_frequencies_log(df_clustered: pd.DataFrame, model_rep, model_dem) -> pd.DataFrame:
    """
    Calculates log-frequency per cluster for Reps vs Dems.
    Identifies clusters by color name.
    """
    stats = []
    
    # Group by cluster, excluding noise
    valid_clusters = df_clustered[df_clustered['cluster'] != NOISE_LABEL]
    
    for cluster_id, group in valid_clusters.groupby('cluster'):
        words = group['word'].tolist()
        
        # Calculate raw counts for words in this cluster
        raw_rep = sum(model_rep.wv.get_vecattr(w, "count") for w in words if w in model_rep.wv)
        raw_dem = sum(model_dem.wv.get_vecattr(w, "count") for w in words if w in model_dem.wv)
        
        # Log Scale (using log1p to handle zeros safely)
        # We sum the logs of individual words, or log the sum? 
        # Usually log of the total frequency sum is better for "cluster magnitude"
        log_rep = np.log1p(raw_rep)
        log_dem = np.log1p(raw_dem)
        
        stats.append({
            "color_label": get_tab10_color_name(cluster_id), # Use Color Name
            "cluster_id": cluster_id,
            "size": len(words),
            "top_examples": ", ".join(words[:5]),
            "log_freq_rep": round(log_rep, 2),
            "log_freq_dem": round(log_dem, 2),
            "diff_log": round(log_rep - log_dem, 2)
        })
        
    return pd.DataFrame(stats).sort_values("diff_log", ascending=False)


# =====================
# Main Execution
# =====================

def main():
    # Set display options
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.float_format', '{:.3f}'.format)

    # Note: We still need models for frequency count
    print(f"Loading models from {MODEL_DIR} for frequency analysis...")
    models = load_models(MODEL_DIR, SUBREDDITS, PERIODS)

    for period in PERIODS:
        print(f"\n=== Processing Period: {period} ===")
        
        # 1. Load CSV (Instead of calculating scores)
        csv_filename = f"{period}_polarization.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Error: CSV not found at {csv_path}. Please run axis_align_iterative.py first.")
            continue
            
        print(f"Loading data from {csv_path}...")
        df_scores = pd.read_csv(csv_path)
        
        # 2. Get Models (for Stats only)
        try:
            model_rep = models["republican"][period]
            model_dem = models["democrats"][period]
        except KeyError:
            print(f"Skipping {period} - models not found for stats.")
            continue
        
        # 3. Clustering
        print(f"Running HDBSCAN for {period}...")
        df_clustered = run_hdbscan_clustering(df_scores)
        
        # 4. Visualization & Saving Plot
        plot_filename = f"{period}_clusters.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        visualize_and_save_clusters(df_clustered, period, plot_path)
        
        # 5. Frequency Analysis (Log Scale + Colors)
        print(f"Analyzing Cluster Frequencies for {period}...")
        cluster_stats = analyze_cluster_frequencies_log(df_clustered, model_rep, model_dem)
        
        # 6. Save Stats to CSV
        stats_filename = f"{period}_cluster_stats.csv"
        stats_path = os.path.join(OUTPUT_DIR, stats_filename)
        cluster_stats.to_csv(stats_path, index=False)
        print(f"Cluster statistics saved to {stats_path}")

        print("\nCluster Statistics (Positive Diff = Higher Republican Log Usage):")
        print(cluster_stats[['color_label', 'size', 'log_freq_rep', 'log_freq_dem', 'diff_log']])

if __name__ == "__main__":
    main()