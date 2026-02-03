#!/usr/bin/env python3
"""
ANOVA on simulation run data: bin by ESI, compute mean position/yaw error per group,
run one-way ANOVA, and plot 8 subplots (ESI vs absolute error) with mean annotations.
Expects 30 CSV files with columns: time, esi, position_error, yaw_error.
Reads first 50 seconds from each file.
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "data"  # directory containing run CSVs
FILE_PATTERN = "run_*.csv"  # glob for 30 files, e.g. run_01.csv, run_02.csv, ...
MAX_TIME = 50.0  # use only first 50 seconds
# ESI bins (same order as in paper: No, Light, Moderate, Severe)
ESI_BINS = [
    (0.95, 1.0),   # [1.0-0.95]  -> No
    (0.66, 0.94),  # [0.95-0.66] -> Light
    (0.33, 0.65),  # [0.66-0.33] -> Moderate
    (0.0, 0.32),   # [0.33-0.0]  -> Severe
]
BIN_LABELS = ["No\n[1.0-0.95]", "Light\n[0.94-0.66]", "Moderate\n[0.65-0.33]", "Severe\n[0.32-0.0]"]

# Column names (adjust if your CSVs differ)
COL_TIME = "time"
COL_ESI = "esi"
COL_POS_ERROR = "position_error"  # or "pos_error"
COL_YAW_ERROR = "yaw_error"


def find_run_files():
    """Return list of paths to run CSV files (expect 30)."""
    path = os.path.join(DATA_DIR, FILE_PATTERN)
    files = sorted(glob.glob(path))
    if len(files) < 30:
        raise FileNotFoundError(
            f"Expected at least 30 files matching {path}, found {len(files)}. "
            f"Create a 'data' directory and add run_01.csv ... run_30.csv with columns: "
            f"{COL_TIME}, {COL_ESI}, {COL_POS_ERROR}, {COL_YAW_ERROR}"
        )
    return files[:30]


def load_first_50_seconds(file_paths):
    """Load first 50 seconds from each file; return single DataFrame."""
    dfs = []
    for p in file_paths:
        df = pd.read_csv(p)
        # Normalize column names to lowercase
        df.columns = df.columns.str.strip().str.lower()
        df.columns = df.columns.str.replace(" ", "_")
        # Map common alternatives
        time_col = COL_TIME if COL_TIME in df.columns else next(
            (c for c in df.columns if "time" in c), None
        )
        esi_col = COL_ESI if COL_ESI in df.columns else next(
            (c for c in df.columns if "esi" in c), None
        )
        pos_col = COL_POS_ERROR if COL_POS_ERROR in df.columns else next(
            (c for c in df.columns if "position" in c or "pos" in c), None
        )
        yaw_col = COL_YAW_ERROR if COL_YAW_ERROR in df.columns else next(
            (c for c in df.columns if "yaw" in c), None
        )
        if time_col is None or esi_col is None or pos_col is None or yaw_col is None:
            raise ValueError(
                f"File {p} must have columns for time, esi, position_error, yaw_error. "
                f"Found: {list(df.columns)}"
            )
        df = df.rename(columns={
            time_col: COL_TIME,
            esi_col: COL_ESI,
            pos_col: COL_POS_ERROR,
            yaw_col: COL_YAW_ERROR,
        })
        df = df[df[COL_TIME] <= MAX_TIME][[COL_TIME, COL_ESI, COL_POS_ERROR, COL_YAW_ERROR]]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def bin_by_esi(df):
    """
    Assign each row to an ESI bin. Returns list of 4 DataFrames (one per bin).
    Bins: [0.95-1.0], (0.66-0.94], (0.33-0.65], [0-0.32].
    """
    groups = []
    for i, (low, high) in enumerate(ESI_BINS):
        if i == 0:
            # First bin: include 0.95 [0.95, 1.0]
            mask = (df[COL_ESI] >= low) & (df[COL_ESI] <= high)
        elif low == 0:
            # Severe bin: include 0 [0, 0.32]
            mask = (df[COL_ESI] >= low) & (df[COL_ESI] <= high)
        else:
            mask = (df[COL_ESI] > low) & (df[COL_ESI] <= high)
        groups.append(df.loc[mask].copy())
    return groups


def main():
    file_paths = find_run_files()
    df = load_first_50_seconds(file_paths)
    groups = bin_by_esi(df)

    # Mean position and yaw error per group
    pos_means = [g[COL_POS_ERROR].mean() for g in groups]
    yaw_means = [g[COL_YAW_ERROR].mean() for g in groups]
    pos_stds = [g[COL_POS_ERROR].std(ddof=1) for g in groups]
    yaw_stds = [g[COL_YAW_ERROR].std(ddof=1) for g in groups]

    print("Group means (first 50 s, binned by ESI):")
    for i, label in enumerate(BIN_LABELS):
        print(
            f"  {label.replace(chr(10), ' ')}: "
            f"position_error mean = {pos_means[i]:.4f} (std {pos_stds[i]:.4f}), "
            f"yaw_error mean = {yaw_means[i]:.4f} (std {yaw_stds[i]:.4f}), n = {len(groups[i])}"
        )

    # One-way ANOVA: position error across 4 groups
    pos_arrays = [g[COL_POS_ERROR].values for g in groups]
    yaw_arrays = [g[COL_YAW_ERROR].values for g in groups]
    f_pos, p_pos = stats.f_oneway(*pos_arrays)
    f_yaw, p_yaw = stats.f_oneway(*yaw_arrays)
    print("\nOne-way ANOVA (4 ESI groups):")
    print(f"  Position error: F = {f_pos:.4f}, p = {p_pos:.4e}")
    print(f"  Yaw error:      F = {f_yaw:.4f}, p = {p_yaw:.4e}")

    # 8 subplots: 2 rows (position, yaw) x 4 columns (ESI bins)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex="col")
    fig.suptitle("Absolute error vs ESI by condition (first 50 s, 30 runs)")

    for col, (label, g) in enumerate(zip(BIN_LABELS, groups)):
        # Row 0: position error vs ESI
        ax_pos = axes[0, col]
        ax_pos.scatter(g[COL_ESI], g[COL_POS_ERROR], alpha=0.3, s=8)
        ax_pos.axhline(pos_means[col], color="C1", ls="--", linewidth=1.5)
        x_max = g[COL_ESI].max() if len(g) else 1.0
        ax_pos.annotate(
            f"mean = {pos_means[col]:.3f}",
            xy=(x_max, pos_means[col]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
        )
        ax_pos.set_ylabel("Position error")
        ax_pos.set_title(label.replace("\n", " "))
        ax_pos.grid(True, alpha=0.3)

        # Row 1: yaw error vs ESI
        ax_yaw = axes[1, col]
        ax_yaw.scatter(g[COL_ESI], g[COL_YAW_ERROR], alpha=0.3, s=8)
        ax_yaw.axhline(yaw_means[col], color="C1", ls="--", linewidth=1.5)
        ax_yaw.annotate(
            f"mean = {yaw_means[col]:.3f}",
            xy=(x_max, yaw_means[col]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
        )
        ax_yaw.set_xlabel("ESI")
        ax_yaw.set_ylabel("Yaw error")
        ax_yaw.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "figures/anova_esi_bins.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
