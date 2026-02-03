import os
import pandas as pd
from typing import Tuple, List, Dict

# Import shared configuration
from axis_align_iterative import (
    OUTPUT_DIR, 
    PERIODS
)

# =====================
# Configuration
# =====================

# Specific coordinate ranges for each period as requested
BIN_CONFIGS = {
    "before_2016": [
        {"label": "Center / Neutral", "x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        {"label": "Extreme Republican", "x": (0.20, 0.30), "y": (0.25, 0.35)},
        {"label": "Extreme Democrat", "x": (-0.45, -0.35), "y": (-0.45, -0.35)},
        {"label": "Divergent (Rep High / Dem Low)", "x": (0.10, 0.20), "y": (-0.20, -0.10)},
        {"label": "Divergent (Rep Low / Dem High)", "x": (-0.15, -0.05), "y": (0.05, 0.15)},
    ],
    "2017_2020": [
        {"label": "Center / Neutral", "x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        {"label": "Extreme Republican", "x": (0.30, 0.40), "y": (0.30, 0.40)},
        {"label": "Extreme Democrat", "x": (-0.35, -0.25), "y": (-0.35, -0.25)},
        {"label": "Divergent (Rep High / Dem Low)", "x": (0.05, 0.15), "y": (-0.15, -0.05)},
        {"label": "Divergent (Rep Low / Dem High)", "x": (-0.15, -0.05), "y": (0.05, 0.15)},
    ],
    "2021_2024": [
        {"label": "Center / Neutral", "x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        {"label": "Extreme Republican", "x": (0.25, 0.35), "y": (0.25, 0.35)},
        {"label": "Extreme Democrat", "x": (-0.35, -0.25), "y": (-0.35, -0.25)},
        {"label": "Divergent (Rep High / Dem Low)", "x": (0.05, 0.15), "y": (-0.15, -0.05)},
        {"label": "Divergent (Rep Low / Dem High)", "x": (-0.15, -0.05), "y": (0.10, 0.20)},
    ]
}

TOP_N_WORDS = 15

# =====================
# Inspection Logic
# =====================

def inspect_bin(df: pd.DataFrame, label: str, x_range: Tuple[float, float], y_range: Tuple[float, float], top_n: int = 15) -> str:
    """
    Filters the dataframe for a specific square (bin) and generates a report string.
    """
    subset = df[
        (df['rep_score'] >= x_range[0]) & (df['rep_score'] <= x_range[1]) &
        (df['dem_score'] >= y_range[0]) & (df['dem_score'] <= y_range[1])
    ]
    
    # Build the report string
    lines = []
    lines.append(f"\n>>> Analyzing Bin: {label}")
    lines.append(f"    Range: X[{x_range[0]}, {x_range[1]}], Y[{y_range[0]}, {y_range[1]}]")
    lines.append(f"    Total words in bin: {len(subset)}")
    
    if not subset.empty:
        # Check if total_freq exists, otherwise fallback to index order (unlikely if pipeline ran correctly)
        sort_col = 'total_freq' if 'total_freq' in subset.columns else 'polarization'
        
        top_words = subset.sort_values(sort_col, ascending=False).head(top_n)['word'].tolist()
        lines.append(f"    Top {top_n} words by {sort_col}: {', '.join(top_words)}")
    else:
        lines.append("    (No words found in this range)")
        
    return "\n".join(lines)


def save_report(output_path: str, content: str):
    """Saves the report content to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Report saved to: {output_path}")


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
        csv_filename = f"{period}_polarization.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Error: CSV not found at {csv_path}. Please run axis_align_iterative.py first.")
            continue
            
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # 2. Retrieve Config
        if period not in BIN_CONFIGS:
            print(f"Warning: No bin configuration found for period '{period}'. Skipping inspection.")
            continue
            
        period_configs = BIN_CONFIGS[period]
        
        # 3. Run Inspections
        full_report = f"Inspection Report for Period: {period}\n{'='*40}\n"
        
        for config in period_configs:
            report_segment = inspect_bin(
                df, 
                label=config['label'], 
                x_range=config['x'], 
                y_range=config['y'], 
                top_n=TOP_N_WORDS
            )
            
            # Print to console
            print(report_segment)
            
            # Add to file content
            full_report += report_segment + "\n"
            
        # 4. Save to File
        report_filename = f"{period}_inspection_report.txt"
        report_path = os.path.join(OUTPUT_DIR, report_filename)
        save_report(report_path, full_report)

if __name__ == "__main__":
    main()