#!/usr/bin/env python3




import os, gc, json
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime

# === CONFIG ===
i = 3  # rect number to process

base_dir     = f"/scratch/10725/mfidansoy1777/ndvi_index_added_slope_geology/rect{i}/merged"
json_file    = "/scratch/10725/mfidansoy1777/bbox_master_map.json"
full_file    = os.path.join(base_dir, f"gdf_rect{i}_concat_full.parquet")
missing_file = os.path.join(base_dir, f"gdf_rect{i}_missing_concat.parquet")
out_file     = os.path.join(base_dir, f"gdf_rect{i}_concat_full_filled.parquet")
log_file     = os.path.join(base_dir, f"gdf_rect{i}_fill_log.txt")

def log(msg):
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

# === Load NDVI bounding boxes ===
log("Loading NDVI bounding boxes...")
with open(json_file) as f:
    ndvi_map = json.load(f)

ndvi_boxes = []
for k, v in ndvi_map.items():
    xmin, ymax, xmax, ymin = map(float, k.split("|"))
    ndvi_boxes.append({"NDVI_ID": v, "geometry": box(xmin, ymin, xmax, ymax)})
ndvi_gdf = gpd.GeoDataFrame(ndvi_boxes, crs="EPSG:4326").set_index("NDVI_ID")
log(f"Loaded {len(ndvi_gdf)} NDVI bounding boxes")

# === Load full + missing datasets ===
df_full = pd.read_parquet(full_file)
df_missing = pd.read_parquet(missing_file)

total_rows = len(df_full)
orig_missing = df_full["NDVI_ID"].isna().sum()

log(f"Full dataset rows: {total_rows}")
log(f"Originally missing NDVI rows: {orig_missing}")

if len(df_missing) > 0:
    # Build geometries for missing rows
    df_missing["geometry"] = df_missing.apply(
        lambda r: box(r["X_min"], r["Y_min"], r["X_max"], r["Y_max"]), axis=1
    )
    grid_gdf = gpd.GeoDataFrame(df_missing, geometry="geometry", crs="EPSG:4326")

    # Spatial join (intersects)
    joined = gpd.sjoin(grid_gdf, ndvi_gdf, how="left", predicate="intersects")
    if "NDVI_ID_right" in joined.columns:
        joined = joined.rename(columns={"NDVI_ID_right": "NDVI_ID"})
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    # Aggregate possible NDVI IDs per cell
    joined = joined.groupby(joined.index).agg({
        "X_min": "first", "X_max": "first", "Y_min": "first", "Y_max": "first",
        "PTYPE": "first", "Slope Value": "first", "geometry": "first",
        "NDVI_ID": lambda x: list(x.dropna().unique()) if not x.isna().all() else None
    }).reset_index(drop=True)

    # Counters for diagnostics
    no_match_count = 0
    single_match_count = 0
    multi_match_count = 0
    sample_no_match = []
    sample_multi = []



    def pick_best(row):
        global no_match_count, single_match_count, multi_match_count, sample_no_match, sample_multi
        if not isinstance(row["NDVI_ID"], list) or row["NDVI_ID"] is None:
            no_match_count += 1
            if len(sample_no_match) < 5:
                sample_no_match.append((row["X_min"], row["Y_min"]))
            return None

        if len(row["NDVI_ID"]) == 1:
            single_match_count += 1
            return row["NDVI_ID"][0]

        multi_match_count += 1
        if len(sample_multi) < 5:
            sample_multi.append((row["X_min"], row["Y_min"], row["NDVI_ID"]))

        overlaps = []
        for nid in row["NDVI_ID"]:
            try:
                area = row["geometry"].intersection(ndvi_gdf.loc[nid, "geometry"]).area
                overlaps.append((nid, area))
            except Exception:
                continue
        return max(overlaps, key=lambda x: x[1])[0] if overlaps else None


    joined["Best_NDVI_ID"] = joined.apply(pick_best, axis=1)

    resolved = joined["Best_NDVI_ID"].notna().sum()
    log(f"Resolved {resolved} of {len(joined)} missing rows")
    log(f"  • No overlaps: {no_match_count}")
    log(f"  • Single overlaps: {single_match_count}")
    log(f"  • Multiple overlaps: {multi_match_count}")
    if sample_no_match:
        log(f"  🔎 Example no-match cells: {sample_no_match}")
    if sample_multi:
        log(f"  🔎 Example multi-match cells: {sample_multi}")

    # Merge back into full dataset
    df_fix = joined.drop(columns=["geometry","NDVI_ID"]).rename(
        columns={"Best_NDVI_ID":"NDVI_ID"}
    )
    df_full = df_full.merge(
        df_fix, on=["X_min","X_max","Y_min","Y_max"], how="left", suffixes=("","_fix")
    )
    df_full["NDVI_ID"] = df_full["NDVI_ID"].combine_first(df_full["NDVI_ID_fix"])
    df_full = df_full.drop(columns=["NDVI_ID_fix"])

final_missing = df_full["NDVI_ID"].isna().sum()
filled_total = orig_missing - final_missing

log(f"✅ Total rows: {total_rows}")
log(f"✅ Originally missing: {orig_missing}")
log(f"✅ Filled successfully: {filled_total}")
log(f"❌ Still missing after fix: {final_missing}")

# Save
df_full.to_parquet(out_file, index=False)
log(f"✅ Saved filled file: {out_file}")

# Cleanup
del df_full, df_missing, ndvi_gdf, ndvi_boxes
gc.collect()
