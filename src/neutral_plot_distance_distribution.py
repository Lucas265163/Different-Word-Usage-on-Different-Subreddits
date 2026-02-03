import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Configuration
# =====================
INPUT_DIR = "data/output/neutral_comparison"
SUBREDDITS = ["democrats", "republican"]
PERIODS = ["before_2016", "2017_2020", "2021_2024"]
PERIOD_LABELS = ["Before 2016", "2017-2020", "2021-2024"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

# =====================
# Plotting Logic
# =====================

def load_csvs(subreddit):
    data = {}
    for period in PERIODS:
        # Pattern: democrats_vs_neutral_before_2016_distances.csv
        filename = f"{subreddit}_vs_neutral_{period}_distances.csv"
        filepath = os.path.join(INPUT_DIR, filename)
        
        if os.path.exists(filepath):
            data[period] = pd.read_csv(filepath)
        else:
            print(f"Warning: File not found {filepath}")
    return data

def plot_subreddit_vs_neutral(subreddit):
    data = load_csvs(subreddit)
    
    if not data:
        print(f"No data found for {subreddit}")
        return

    # Create 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    # 1. Plot Individual Periods (First 3 subplots)
    for i, period in enumerate(PERIODS):
        ax = axes[i]
        
        if period in data:
            df = data[period]
            dist_data = df["cosine_distance"]
            
            # Histogram
            ax.hist(dist_data, bins=40, alpha=0.6, density=True, 
                   color=COLORS[i], label=PERIOD_LABELS[i])
            
            # Statistics
            mean_val = dist_data.mean()
            median_val = dist_data.median()
            
            # Add vertical lines for stats
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='purple', linestyle=':', linewidth=2,
                      label=f'Median: {median_val:.3f}')
            
            ax.set_title(f"{PERIOD_LABELS[i]} ({subreddit.capitalize()} vs Neutral)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Cosine Distance (from Neutral)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

    # 2. Plot Combined Comparison (4th subplot)
    ax_combined = axes[3]
    for i, period in enumerate(PERIODS):
        if period in data:
            df = data[period]
            ax_combined.hist(df["cosine_distance"], bins=40, alpha=0.4, density=True, 
                            label=PERIOD_LABELS[i], color=COLORS[i])

    ax_combined.set_title(f"Combined Comparison - {subreddit.capitalize()} vs Neutral", fontsize=12, fontweight='bold')
    ax_combined.set_xlabel("Cosine Distance (from Neutral)")
    ax_combined.set_ylabel("Density")
    ax_combined.legend()
    ax_combined.grid(True, alpha=0.3)

    plt.suptitle(f"Divergence from Neutral: {subreddit.capitalize()}", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(INPUT_DIR, f"{subreddit}_vs_neutral_distribution.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    # plt.show()

def main():
    for sub in SUBREDDITS:
        plot_subreddit_vs_neutral(sub)

if __name__ == "__main__":
    main()