#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import json
from datetime import datetime
i=2
# === Logging helper ===
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# === Paths ===

import glob
missing_files = glob.glob(f"/scratch/.../missing_ndvi_rows/rect{i}/missing_NDVI_ID_part*.parquet")

# missing_file = f"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/missing_ndvi_rows/rect{i}/missing_NDVI_ID.parquet"
json_file    = "/scratch/10725/mfidansoy1777/bbox_master_map.json"
orig_file    = f"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/rect{i}/gdf_rect{i}_slope_geology_part0.parquet"
out_file     = f"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/rect{i}/nans_filled/gdf_rect{i}_slope_geology_part0_filled.parquet"
# Make sure directory exists
os.makedirs(os.path.dirname(out_file), exist_ok=True)
# === Step 0. Load original file and count NaNs ===
log(f"Loading original rect{i} file...")
df_orig = pd.read_parquet(orig_file)
orig_nans = df_orig["NDVI_ID"].isna().sum()
log(f"Original file had {orig_nans} rows with missing NDVI_ID")

# === Step 1. Load missing NDVI rows ===
log("Loading missing NDVI rows parquet...")
df_missing = pd.read_parquet(missing_files)
log(f"Extracted file has {len(df_missing)} missing rows")

df_missing["geometry"] = df_missing.apply(lambda r: box(r["X_min"], r["Y_min"], r["X_max"], r["Y_max"]), axis=1)
grid_gdf = gpd.GeoDataFrame(df_missing, geometry="geometry", crs="EPSG:4326")

# === Step 2. Load NDVI bounding boxes ===
log("Loading NDVI bounding boxes...")
with open(json_file) as f:
    ndvi_map = json.load(f)

ndvi_boxes = []
for k, v in ndvi_map.items():
    xmin, ymax, xmax, ymin = map(float, k.split("|"))
    ndvi_boxes.append({"NDVI_ID": v, "geometry": box(xmin, ymin, xmax, ymax)})

ndvi_gdf = gpd.GeoDataFrame(ndvi_boxes, crs="EPSG:4326").set_index("NDVI_ID")
log(f"Loaded {len(ndvi_gdf)} NDVI bounding boxes")

# === Step 3. Spatial join (find all candidate NDVI IDs) ===
log("Finding partial overlaps...")
joined = gpd.sjoin(grid_gdf, ndvi_gdf, how="left", predicate="intersects")

if "NDVI_ID_right" in joined.columns:
    joined = joined.rename(columns={"NDVI_ID_right": "NDVI_ID"})
if "index_right" in joined.columns:
    joined = joined.drop(columns=["index_right"])

joined = joined.groupby(joined.index).agg({
    "X_min": "first", "X_max": "first", "Y_min": "first", "Y_max": "first",
    "PTYPE": "first", "Slope Value": "first",
    "geometry": "first",
    "NDVI_ID": lambda x: list(x.dropna().unique()) if not x.isna().all() else None
}).reset_index(drop=True)

# === Step 4. Pick NDVI_ID with largest overlap ===
def pick_largest_overlap(cell, candidate_ids, ndvi_gdf):
    overlaps = []
    for nid in candidate_ids:
        try:
            poly = ndvi_gdf.loc[nid, "geometry"]
            area = cell.intersection(poly).area
            overlaps.append((nid, area))
        except Exception:
            continue
    if overlaps:
        return max(overlaps, key=lambda x: x[1])[0]
    return None

log("Selecting best NDVI_ID based on overlap area...")
joined["Best_NDVI_ID"] = joined.apply(
    lambda r: pick_largest_overlap(r["geometry"], r["NDVI_ID"], ndvi_gdf),
    axis=1
)

df_best = joined.drop(columns=["geometry", "NDVI_ID"])
df_best = df_best.rename(columns={"Best_NDVI_ID": "NDVI_ID"})

resolved_count = df_best["NDVI_ID"].notna().sum()
log(f"Resolved {resolved_count} of {len(df_best)} missing rows")

# === Step 5. Merge back into original full dataset ===
log("Merging corrections back into original file...")
df_filled = df_orig.merge(
    df_best,
    on=["X_min", "X_max", "Y_min", "Y_max"],
    how="left",
    suffixes=("", "_fix")
)

df_filled["NDVI_ID"] = df_filled["NDVI_ID"].combine_first(df_filled["NDVI_ID_fix"])
df_filled = df_filled.drop(columns=["NDVI_ID_fix"])

final_nans = df_filled["NDVI_ID"].isna().sum()
resolved_total = orig_nans - final_nans

log(f"✅ Total originally missing: {orig_nans}")
log(f"✅ Filled via overlap: {resolved_total}")
log(f"❌ Still missing after fix: {final_nans}")

# === Save final ===
df_filled.to_parquet(out_file, index=False)
log(f"Saved updated rect{i} file to {out_file}")

# Quick sample output
print(df_filled.head())

assert len(df_filled) == len(df_orig), "Row count mismatch after merge!"