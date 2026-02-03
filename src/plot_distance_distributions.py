import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Configuration
# =====================
BASE_OUTPUT_DIR = "data/output"
METHODS = ["intersecting_words", "anchor_words", "iterative"]
PERIODS = ["before_2016", "2017_2020", "2021_2024"]
PERIOD_LABELS = ["Before 2016", "2017-2020", "2021-2024"]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

# =====================
# Plotting
# =====================

def load_csvs(method_dir):
    data = {}
    for period in PERIODS:
        # Warning: This assumes one file per period or picks the first one found.
        # If you have multiple subreddit pairs, you might need to iterate over them specifically.
        pattern = os.path.join(method_dir, f"*_{period}_distances.csv")
        matches = glob.glob(pattern)
        if not matches:
            continue
        data[period] = pd.read_csv(matches[0])
    return data

def plot_method(method):
    method_dir = os.path.join(BASE_OUTPUT_DIR, method)
    data = load_csvs(method_dir)
    
    if not data:
        print(f"No data found for method: {method}")
        return

    # Create 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten() # Flatten 2D array to 1D for easy iteration

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
            
            ax.set_title(f"{PERIOD_LABELS[i]} Distribution", fontsize=12, fontweight='bold')
            ax.set_xlabel("Cosine Distance")
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

    ax_combined.set_title(f"Combined Comparison - {method}", fontsize=12, fontweight='bold')
    ax_combined.set_xlabel("Cosine Distance")
    ax_combined.set_ylabel("Density")
    ax_combined.legend()
    ax_combined.grid(True, alpha=0.3)

    plt.suptitle(f"Cosine Distance Analysis: {method}", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    plt.savefig(os.path.join(method_dir, f"{method}_distribution.png"))

def main():
    for method in METHODS:
        plot_method(method)

if __name__ == "__main__":
    main()