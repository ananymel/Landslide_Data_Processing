# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 16:40:50 2025

@author: melis
"""
#%% Not RAM friendly
import polars as pl
from pathlib import Path

# === Directories ===
unfiltered_dir = Path(r"D:\06042025\09_19_25_gdf_parquet_version_unfiltered")  # full data
filtered_dir   = Path(r"D:\06042025\09_19_25_gdf_parquet_version")             # missing slope
output_dir     = Path(r"D:\06042025\09_19_25_gdf_parquet_version_repaired")    # save repaired
output_dir.mkdir(exist_ok=True)

# === Loop through filtered parquet files ===
for f in sorted(filtered_dir.glob("*.parquet")):
    fname = f.name
    unfiltered_file = unfiltered_dir / fname
    
    if not unfiltered_file.exists():
        print(f"❌ No matching unfiltered file for {fname}")
        continue
    
    print(f"Processing {fname} ...")
    
    # Load filtered (missing slope)
    df_filtered = pl.read_parquet(f)
    
    # Load unfiltered (has slope)
    df_unfiltered = pl.read_parquet(unfiltered_file)
    
    # Keep only needed join keys + slope
    df_unfiltered_slope = df_unfiltered.select([
        "X_min", "X_max", "Y_min", "Y_max", "Slope Value"
    ])
    
    # Join: filtered + slope (matching on 4 columns)
    df_merged = df_filtered.join(
        df_unfiltered_slope,
        on=["X_min", "X_max", "Y_min", "Y_max"],
        how="left"
    )
    
    # Warn if some slopes couldn't be matched
    missing_count = df_merged["Slope Value"].null_count()
    if missing_count > 0:
        print(f"⚠️ {missing_count} rows missing slope in {fname}")
    
    # Save repaired parquet
    out_path = output_dir / fname
    df_merged.write_parquet(out_path)
    print(f"✅ Saved repaired file: {out_path}")


#%% CPU Parallel

import os
import polars as pl

# Force Polars to only use 1 thread per process
os.environ["POLARS_MAX_THREADS"] = "1"

# Optional: formatting config
pl.Config.set_tbl_formatting("UTF8_FULL")

# Verify
print("Polars threads per worker:", pl.thread_pool_size())



from pathlib import Path
from joblib import Parallel, delayed

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import re

# === Directories ===
unfiltered_dir = Path(r"C:\Users\melis\Desktop\add_slope_parquet\10_07_25_gdf_parquet_version_unfiltered_slim")
filtered_dir   = Path(r"C:\Users\melis\Desktop\add_slope_parquet\correct_parquet_filtered_ones")
output_dir     = Path(r"C:\Users\melis\Desktop\add_slope_parquet\10_07_25_gdf_parquet_version_repaired")
output_dir.mkdir(exist_ok=True)

# === Helper: extract rect number ===
def get_rect_number(path: Path):
    m = re.search(r"rect(\d+)", path.stem)
    return int(m.group(1)) if m else None

def repair_rect(rect_num, filtered_file, unfiltered_file):
    try:
        out_path = output_dir / filtered_file.name

        # Unfiltered: only slope + keys
        df_unfiltered = (
            pl.scan_parquet(unfiltered_file)
              .select(["X_min", "X_max", "Y_min", "Y_max", "Slope Value"])
        )

        # Filtered: only keys (no geometry)
        df_filtered = (
            pl.scan_parquet(filtered_file)
              .select(["X_min", "X_max", "Y_min", "Y_max"])
        )

        # Lazy join
        df_merged = df_filtered.join(
            df_unfiltered,
            on=["X_min", "X_max", "Y_min", "Y_max"],
            how="left"
        )

        # Stream directly to disk
        df_merged.sink_parquet(out_path)

        return f"✅ rect{rect_num}: repaired and saved → {out_path.name}"

    except Exception as e:
        return f"⚠️ rect{rect_num}: error {e}"


# === Build rect → file maps ===
filtered_map   = {get_rect_number(f): f for f in filtered_dir.glob("*.parquet")}
unfiltered_map = {get_rect_number(f): f for f in unfiltered_dir.glob("*.parquet")}

# Common rects (skip ones that don't exist in both dirs, e.g., rect10)
common_rects = sorted(set(filtered_map.keys()) & set(unfiltered_map.keys()))
print(f"Found {len(common_rects)} rects in both dirs: {common_rects}")

# === Run in parallel ===
n_jobs = min(len(common_rects), multiprocessing.cpu_count())  # don’t oversubscribe
results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
    delayed(repair_rect)(rect, filtered_map[rect], unfiltered_map[rect])
    for rect in common_rects
)


# === Print summary ===
for r in results:
    print(r)
