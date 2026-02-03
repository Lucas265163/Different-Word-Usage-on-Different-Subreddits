import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
from typing import Tuple

# Import shared configuration from previous script
from axis_align_iterative import (
    OUTPUT_DIR, 
    PERIODS
)

# =====================
# Configuration
# =====================
# Heatmap Grid
GRID_STEP_SIZE = 0.05
GRID_DIRECTION_FLOOR = 'floor'
GRID_DIRECTION_CEIL = 'ceil'

# Visualization
FIG_SIZE = (10, 8)
CMAP = 'YlOrRd'
DENSITY_THRESHOLD = 0.05  # Bottom 5% density is filtered out


# =====================
# Helper Functions
# =====================

def snap_to_grid(value: float, step: float = 0.05, direction: str = 'floor') -> float:
    """
    Snaps a value to the nearest step (e.g., 0.05).
    direction='floor' -> rounds down (for min bounds)
    direction='ceil' -> rounds up (for max bounds)
    """
    if direction == 'floor':
        return np.floor(value / step) * step
    else:
        return np.ceil(value / step) * step


def calculate_95_percent_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe to keep only the 95% most dense points.
    Uses Gaussian KDE to estimate density.
    """
    # Ensure we are working with clean data
    df = df.dropna(subset=['rep_score', 'dem_score'])
    
    x = df['rep_score']
    y = df['dem_score']
    
    # 1. Calculate Point Density (Unweighted)
    # stacking x and y to create a shape of (2, N) for the KDE
    try:
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        density = kernel(values)
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix in KDE. Returning original dataframe.")
        return df
    
    # 2. Find the threshold for top 95%
    # We sort the density values and find the value at the 5th percentile
    sorted_density = np.sort(density)
    threshold_idx = int(len(density) * DENSITY_THRESHOLD) # Cut off bottom 5%
    threshold = sorted_density[threshold_idx]
    
    # 3. Filter Data
    mask = density > threshold
    df_core = df[mask].copy()
    
    print(f"Filtered {len(df) - len(df_core)} outlier points (Bottom {DENSITY_THRESHOLD*100}% density).")
    return df_core


def plot_constrained_heatmap(df: pd.DataFrame, period_name: str, output_path: str, step_size: float = 0.05):
    """
    Plots a heatmap only for the 95% density region with fixed 0.05 bin sizes.
    Saves the plot to disk.
    """
    # 1. Get the 95% Core Data
    print(f"Calculating core density for {period_name}...")
    df_core = calculate_95_percent_core(df)
    
    if df_core.empty:
        print(f"Warning: No data remaining for {period_name} after filtering.")
        return

    # 2. Calculate "Snapped" Boundaries
    x_min_raw, x_max_raw = df_core['rep_score'].min(), df_core['rep_score'].max()
    y_min_raw, y_max_raw = df_core['dem_score'].min(), df_core['dem_score'].max()
    
    # Round bounds to nearest 0.05
    x_start = snap_to_grid(x_min_raw, step_size, GRID_DIRECTION_FLOOR)
    x_end = snap_to_grid(x_max_raw, step_size, GRID_DIRECTION_CEIL)
    y_start = snap_to_grid(y_min_raw, step_size, GRID_DIRECTION_FLOOR)
    y_end = snap_to_grid(y_max_raw, step_size, GRID_DIRECTION_CEIL)
    
    # 3. Create Explicit Bins
    # np.arange excludes the stop value, so we add a tiny epsilon or step_size
    x_bins = np.arange(x_start, x_end + step_size/1000, step_size)
    y_bins = np.arange(y_start, y_end + step_size/1000, step_size)
    
    print(f"Plotting range: X[{x_start:.2f}, {x_end:.2f}], Y[{y_start:.2f}, {y_end:.2f}]")
    
    # 4. Plot
    plt.figure(figsize=FIG_SIZE)
    
    # Check if 'total_freq' exists, otherwise default to count=1
    weights = df_core['total_freq'] if 'total_freq' in df_core.columns else None
    
    h = plt.hist2d(
        df_core['rep_score'], 
        df_core['dem_score'], 
        bins=[x_bins, y_bins],  # Pass specific bin edges
        range=[[x_start, x_end], [y_start, y_end]],
        weights=weights, 
        cmap=CMAP, 
        norm=colors.LogNorm() 
    )
    
    plt.colorbar(h[3], label='Total Word Frequency (Log Scale)')
    plt.title(f'Word Frequency Heatmap (95% Core): {period_name}')
    plt.xlabel('Republican Alignment Score')
    plt.ylabel('Democrat Alignment Score')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    
    # Force the axis ticks to match our grid steps for readability
    # Show every 2nd tick to avoid crowding if dense
    plt.xticks(np.round(x_bins[::2], 2)) 
    plt.yticks(np.round(y_bins[::2], 2))
    
    plt.grid(False)
    plt.tight_layout()
    
    # Save Logic
    print(f"Saving heatmap to {output_path}...")
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# =====================
# Main Execution
# =====================

def main():
    print(f"Output Directory: {OUTPUT_DIR}")
    
    for period in PERIODS:
        print(f"\n{'='*60}")
        print(f"=== Processing Period: {period} ===")
        print(f"{'='*60}")
        
        # 1. Load CSV
        # We assume axis_align_iterative.py has already run and generated this file
        csv_filename = f"{period}_polarization.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Error: CSV not found at {csv_path}. Please run axis_align_iterative.py first.")
            continue
            
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # 2. Validation
        if 'total_freq' not in df.columns:
            print("Warning: 'total_freq' column missing. Heatmap will be unweighted (counts only).")
        
        # 3. Generate and Save Heatmap
        plot_filename = f"{period}_heatmap.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        
        plot_constrained_heatmap(
            df, 
            period_name=period, 
            output_path=plot_path, 
            step_size=GRID_STEP_SIZE
        )
    
if __name__ == "__main__":
    main()