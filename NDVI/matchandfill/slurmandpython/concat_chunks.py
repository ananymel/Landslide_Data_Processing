#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob
import pandas as pd
from datetime import datetime

base_dir = "/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

for i in range(1, 19):
    if i == 10:  # skip rect10
        continue

    rect_dir = os.path.join(base_dir, f"rect{i}")
    merged_dir = os.path.join(rect_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # === Concat full rect files ===
    all_with_nan_dir = os.path.join(rect_dir, f"all_with_nan_rect{i}")
    full_out = os.path.join(merged_dir, f"gdf_rect{i}_concat_full.parquet")

    full_files = sorted(glob.glob(os.path.join(all_with_nan_dir, f"gdf_rect{i}_slope_geology_part*.parquet")))
    df_full = None
    if full_files:
        df_full = pd.concat([pd.read_parquet(f) for f in full_files], ignore_index=True)
        df_full.to_parquet(full_out, index=False)
        log(f"✅ Saved full concat for rect{i} with {len(df_full)} rows → {full_out}")
    else:
        log(f"⚠️ No files found in {all_with_nan_dir}")

    # === Concat missing files ===
    missing_dir = os.path.join(rect_dir, f"missing_ndvi_rows_rect{i}")
    missing_out = os.path.join(merged_dir, f"gdf_rect{i}_missing_concat.parquet")

    missing_files = sorted(glob.glob(os.path.join(missing_dir, f"missing_NDVI_ID_rect{i}_part*.parquet")))
    df_missing = None
    if missing_files:
        df_missing = pd.concat([pd.read_parquet(f) for f in missing_files], ignore_index=True)
        df_missing.to_parquet(missing_out, index=False)
        log(f"✅ Saved missing concat for rect{i} with {len(df_missing)} rows → {missing_out}")
    else:
        log(f"⚠️ No files found in {missing_dir}")

    # === Sanity check ===
    if df_full is not None and df_missing is not None:
        missing_pct = (len(df_missing) / len(df_full) * 100) if len(df_full) > 0 else 0
        log(f"📊 Rect{i} summary: Total rows = {len(df_full)}, Missing rows = {len(df_missing)} "
            f"({missing_pct:.2f}% missing)")
