#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import polars as pl
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import gc
import pyarrow.parquet as pq


sys.path.append("/home1/10725/mfidansoy1777/.local/lib/python3.12/site-packages")
# === CONFIG ===
i = 1   # change for rect number
rects_file = Path(f"/scratch/10725/mfidansoy1777/13_10_25_geology_joined/slurm/gdf_rect{i}_slope_geology.parquet")
json_file  = Path("/scratch/10725/mfidansoy1777/bbox_master_map.json")
output_dir = Path(f"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/rect{i}/all_with_nan_rect{i}")
missing_dir = Path(f"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/rect{i}/missing_ndvi_rows_rect{i}")
output_dir.mkdir(parents=True, exist_ok=True)
missing_dir.mkdir(parents=True, exist_ok=True)


''' This code does not allow any partial overlap'''
# === Logging helper ===
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# === Load JSON once as DataFrame ===
def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    json_df = pd.DataFrame([
        {
            "NDVI_ID": v,
            "X_min": float(parts[0]),
            "Y_max": float(parts[1]),
            "X_max": float(parts[2]),
            "Y_min": float(parts[3]),
        }
        for k, v in data.items()
        for parts in [k.split("|")]
    ])
    # Pre-build interval indices for fast lookup
    json_df = json_df.reset_index(drop=True)
    x_intervals = pd.IntervalIndex.from_arrays(json_df["X_min"], json_df["X_max"], closed="both")
    y_intervals = pd.IntervalIndex.from_arrays(json_df["Y_min"], json_df["Y_max"], closed="both")
    return json_df, x_intervals, y_intervals


def process_chunk_rect(chunk: pd.DataFrame, json_df, x_intervals, y_intervals):
    rect_ids = []
    for xmin, xmax, ymin, ymax in zip(chunk["X_min"], chunk["X_max"], chunk["Y_min"], chunk["Y_max"]):
        # Candidate matches: interval contains BOTH bounds
        #This requires both the left and right edge of the grid cell (xmin, xmax) to be inside the same NDVI box’s x-range.
        #Same for y_max
        x_mask = x_intervals.contains(xmin) & x_intervals.contains(xmax)
        y_mask = y_intervals.contains(ymin) & y_intervals.contains(ymax)
        candidates = json_df[x_mask & y_mask]

        if not candidates.empty:
            rect_ids.append(candidates.iloc[0]["NDVI_ID"])  # take first match
        else:
            rect_ids.append(None)

    chunk["NDVI_ID"] = rect_ids
    return chunk

def process_file(parq_path, json_df, x_intervals, y_intervals, chunk_size=200_000):
    log(f"Processing {parq_path.name} with Arrow streaming...")
    parquet_file = pq.ParquetFile(parq_path)

    for idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size, 
                                                          columns=["X_min","X_max","Y_min","Y_max","PTYPE","Slope Value"])):
        #only for trial just get the first chunk
        # if idx > 0:
        #     break

        df_chunk = batch.to_pandas()
        processed = process_chunk_rect(df_chunk, json_df, x_intervals, y_intervals)

        # === Save full processed chunk ===
        out_path = output_dir / f"{parq_path.stem}_part{idx}.parquet"
        processed.to_parquet(out_path, index=False)
        log(f"✅ Wrote {len(processed)} rows to {out_path}")

        # === Save NaN rows separately ===
        missing = processed[processed["NDVI_ID"].isna()]
        if not missing.empty:
            miss_path = missing_dir / f"missing_NDVI_ID_rect{i}_part{idx}.parquet"
            missing.to_parquet(miss_path, index=False)
            log(f"⚠️  Found {len(missing)} missing NDVI_IDs. Saved to {miss_path}")
        else:
            log("🎉 No missing NDVI_IDs in this chunk")

        del processed, df_chunk, batch, missing
        gc.collect()



# def process_file(parq_path, json_df, x_intervals, y_intervals, chunk_size=200_000):
#     log(f"Processing {parq_path.name} with Arrow streaming...")
#     parquet_file = pq.ParquetFile(parq_path)

#     for idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size, columns=["X_min","X_max","Y_min","Y_max","PTYPE","Slope Value"])):
#         #only for trial just get the first chunk
#         if idx>0:
#             break
#         df_chunk = batch.to_pandas()
#         processed = process_chunk_rect(df_chunk, json_df, x_intervals, y_intervals)

#         out_path = output_dir / f"{parq_path.stem}_part{idx}.parquet"
#         processed.to_parquet(out_path, index=False)

#         log(f"✅ Wrote {len(processed)} rows to {out_path}")
#         del processed, df_chunk, batch
#         gc.collect()



if __name__ == "__main__":
    log(f"Loading JSON: {json_file}")
    json_df, x_intervals, y_intervals = load_json(json_file)
    log(f"Loaded {len(json_df)} JSON tiles")

    process_file(rects_file, json_df, x_intervals, y_intervals, chunk_size=200_000)


#%%




