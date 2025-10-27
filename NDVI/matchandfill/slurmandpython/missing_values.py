#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''IMPORTANT: Get the ones that does not match because we look for exact match'''
''' These missing ones are matching with multiple NDVI_ID'''
'''Find which IDs they match with'''
''' By using the overlap area select the best ID'''

''' Use shapely to match the missing values'''
''' Ssave the missing values assinged ndvi in another file'''
''' Then check for original ndvi_rect{i}_part(k) and fill out the NaN NDVI ID s'''
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import box
# import json
# from datetime import datetime
# i=1
# # === Logging helper ===
# def log(msg):
#     print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# # === Paths ===
# in_file   = fr"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/missing_ndvi_rows/rect{i}/missing_NDVI_ID.parquet"
# json_file = fr"/scratch/10725/mfidansoy1777/bbox_master_map.json"
# out_file  = fr"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/missing_ndvi_rows/rect2/missing_NDVI_ID_with_best_match/missing_NDVI_ID_with_best_match_rect{i}.parquet"

# original_file = "/scratch/10725/mfidansoy1777/grid_numbers_added/grid_numbers_fixed/gdf_rect6_slope_inside.parquet"
# # === Load parquet with multiple NDVI candidates ===
# df = pd.read_parquet(in_file)

# # Build geometry for each grid cell first
# df["geometry"] = df.apply(lambda r: box(r["X_min"], r["Y_min"], r["X_max"], r["Y_max"]), axis=1)

# # === Load NDVI bounding boxes as polygons ===
# with open(json_file) as f:
#     ndvi_map = json.load(f)

# ndvi_boxes = []
# for k, v in ndvi_map.items():
#     xmin, ymax, xmax, ymin = map(float, k.split("|"))
#     ndvi_boxes.append({"NDVI_ID": v, "geometry": box(xmin, ymin, xmax, ymax)})

# ndvi_gdf = gpd.GeoDataFrame(ndvi_boxes, crs="EPSG:4326").set_index("NDVI_ID")

# # === Helper to pick NDVI_ID with largest overlap ===
# def pick_largest_overlap(cell, candidate_ids, ndvi_gdf):
#     overlaps = []
#     for nid in candidate_ids:
#         try:
#             poly = ndvi_gdf.loc[nid, "geometry"]
#             area = cell.intersection(poly).area
#             overlaps.append((nid, area))
#         except KeyError:
#             continue
#     if overlaps:
#         return max(overlaps, key=lambda x: x[1])[0]
#     return None

# # === Apply selection ===
# log("Selecting NDVI_ID with largest overlap...")
# df["Best_NDVI_ID"] = df.apply(
#     lambda r: pick_largest_overlap(r["geometry"], r["NDVI_ID"], ndvi_gdf),
#     axis=1
# )

# ##################

# #Drop the NDVI_ID List and change name of Best_NDVI with NDVI_ID

# #################
# # Drop geometry before saving
# df_out = df.drop(columns=["geometry"])
# df_out.to_parquet(out_file, index=False)

# # === Additional Checks ===
# try:
#     orig = pd.read_parquet(original_file, columns=["X_min"])  # read minimal cols for speed
#     orig_len = len(orig)
# except Exception as e:
#     orig_len = None
#     log(f"⚠️ Could not read original file: {e}")

# log(f"✅ Saved with best NDVI_ID assigned to {out_file}")
# log(f"Original file length: {orig_len}")
# log(f"df_out length: {len(df_out)}")
# log(f"df_out columns: {list(df_out.columns)}")

# missing_count = df_out["NDVI_ID"].isna().sum()
# if missing_count > 0:
#     log(f"⚠️ Found {missing_count} rows still missing NDVI_ID")
# else:
#     log("✅ No missing NDVI_ID values remain")
# log(f"✅ Saved with best NDVI_ID assigned to {out_file}")
# print(df_out.head())




##############2ND SCRIPT#############

#df[filter] =- filter[NDVI_ID][ONLY[Nans]]

#check how many of them changed --> should be 3750

#update the /scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/rect2/gdf_rect2_slope_geology_part0.parquet 
# based on new NDVI values from previous one







# -*- coding: utf-8 -*-
"""
Parallelized NDVI missing-value filler for rect1..18 (excluding rect10).
- For each rect: merge missing files, calculate overlaps, assign best NDVI_ID
- Save fixed parquet under base_dir
- Write per-rect logs
- Write global intersections log with detailed overlap areas
- Aggressively frees RAM with del + gc.collect()
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime
import os, glob, json, gc
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Paths ===
BASE = "/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology"
JSON_FILE = "/scratch/10725/mfidansoy1777/bbox_master_map.json"
INTERSECTION_LOG = os.path.join(BASE, "all_rects_intersections.log")

# === Load NDVI bounding boxes once globally ===
with open(JSON_FILE) as f:
    ndvi_map = json.load(f)

ndvi_boxes = []
for k, v in ndvi_map.items():
    xmin, ymax, xmax, ymin = map(float, k.split("|"))
    ndvi_boxes.append({"NDVI_ID": v, "geometry": box(xmin, ymin, xmax, ymax)})

ndvi_gdf = gpd.GeoDataFrame(ndvi_boxes, crs="EPSG:4326").set_index("NDVI_ID")
del ndvi_boxes, ndvi_map
gc.collect()

# === Helper: pick NDVI with largest overlap ===
def pick_largest_overlap(cell, candidate_ids, ndvi_gdf):
    # Ensure candidate_ids is a list
    if pd.isna(candidate_ids):
        return None, []
    if not isinstance(candidate_ids, (list, tuple)):
        candidate_ids = [candidate_ids]

    overlaps = []
    for nid in candidate_ids:
        try:
            poly = ndvi_gdf.loc[nid, "geometry"]
            area = cell.intersection(poly).area
            overlaps.append((nid, area))
        except KeyError:
            continue

    if overlaps:
        return max(overlaps, key=lambda x: x[1])[0], overlaps
    return None, []


# === Worker for each rect ===
def process_rect(rect):
    rect_dir = os.path.join(BASE, f"rect{rect}")
    missing_dir = os.path.join(rect_dir, f"missing_ndvi_rows_rect{rect}")
    merged_dir  = os.path.join(rect_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    log_file = os.path.join(rect_dir, f"missing_fill_rect{rect}.log")
    out_file = os.path.join(rect_dir, f"missing_NDVI_ID_rect{rect}_with_best_match.parquet")

    with open(log_file, "w") as lf:
        def log(msg):
            lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

        # Load merged original file
        orig_file = os.path.join(merged_dir, f"gdf_rect{rect}_concat_full.parquet")
        if not os.path.exists(orig_file):
            log(f"❌ Missing original merged file: {orig_file}")
            return f"Rect {rect} skipped (missing merged file)"

        df_orig = pd.read_parquet(orig_file, columns=["X_min"])
        orig_len = len(df_orig)
        del df_orig
        gc.collect()

        # Load missing files
        missing_files = sorted(glob.glob(os.path.join(missing_dir, "missing_NDVI_ID_rect*.parquet")))
        if not missing_files:
            log("⚠️ No missing NDVI parquet files found.")
            return f"Rect {rect}: no missing files"

        df_missing = pd.concat([pd.read_parquet(f) for f in missing_files], ignore_index=True)
        df_missing["geometry"] = df_missing.apply(
            lambda r: box(r["X_min"], r["Y_min"], r["X_max"], r["Y_max"]), axis=1
        )

        log(f"Loaded {orig_len} original rows and {len(df_missing)} missing rows")

        # Apply best NDVI_ID selection with overlaps recorded
        all_intersections = []
        best_ids = []
        for _, r in df_missing.iterrows():
            best, overlaps = pick_largest_overlap(r["geometry"], r["NDVI_ID"], ndvi_gdf)
            best_ids.append(best)
            all_intersections.append(overlaps)

        df_missing["NDVI_ID"] = best_ids

        df_out = df_missing.drop(columns=["geometry"])
        df_out.to_parquet(out_file, index=False)

        log(f"✅ Saved fixed missing NDVI file to {out_file}")
        log(f"Original merged length: {orig_len}")
        log(f"Fixed missing length: {len(df_out)}")
        log(f"Columns: {list(df_out.columns)}")

        missing_count = df_out["NDVI_ID"].isna().sum()
        filled_count = len(df_out) - missing_count

        log(f"✅ Filled {filled_count} NDVI_ID values")
        if missing_count > 0:
            log(f"⚠️ Still {missing_count} rows without NDVI_ID")

        # Free memory
        del df_missing, df_out, best_ids
        gc.collect()

    # Append intersections to global log
    with open(INTERSECTION_LOG, "a") as intf:
        for idx, overlaps in enumerate(all_intersections):
            intf.write(f"rect{rect},row{idx}: {overlaps}\n")

    # Free after writing
    del all_intersections
    gc.collect()

    return f"Rect {rect} done. Log: {log_file}"

# === Run in parallel for rect1..18 (excluding 10) ===
if __name__ == "__main__":
    rects = [i for i in range(1, 19) if i != 10]
    with ProcessPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(process_rect, r): r for r in rects}
        for fut in as_completed(futures):
            print(fut.result())
