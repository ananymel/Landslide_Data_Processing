import sys
from pathlib import Path
sys.path.append("/home1/10725/mfidansoy1777/.local/lib/python3.12/site-packages")

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import box

from datetime import datetime
import multiprocessing as mp
from functools import partial
import gc
import sys
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

i=11
# === Logging helper ===
def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)
    
# === CONFIG ===
output_dir = Path("/scratch/10725/mfidansoy1777/13_10_25_geology_joined/slurm")
output_dir.mkdir(exist_ok=True)

geology_shp = "/scratch/10725/mfidansoy1777/shapefiles/shapefiles/GMC_geo_poly.shp"
slope_crs = "EPSG:4326"

log("Reading geology polygons...")
geology_gdf = gpd.read_file(geology_shp)
if geology_gdf.crs is None or geology_gdf.crs.to_string() != slope_crs:
    log("Reprojecting geology shapefile...")
    geology_gdf = geology_gdf.to_crs(slope_crs)
log(f"Loaded geology with {len(geology_gdf)} polygons.")


# === Chunk generator ===
def chunk_generator(parquet_path, chunk_size, columns):
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
        df_chunk = batch.to_pandas()
        # for col in ["X_min", "Y_min", "X_max", "Y_max"]:
        #     df_chunk[col] = pd.to_numeric(df_chunk[col], downcast="float")
        yield df_chunk


def process_chunk_full(chunk, slope_crs, geology_gdf):
    # gdf = gpd.GeoDataFrame(
    #     chunk,
    #     geometry=[box(xmin, ymin, xmax, ymax)
    #               for xmin, ymin, xmax, ymax in zip(
    #                   chunk["X_min"], chunk["Y_min"], chunk["X_max"], chunk["Y_max"]
    #               )],
    #     crs=slope_crs
    # )

    gdf = gpd.GeoDataFrame(
    chunk.copy(),   # ✅ copy all slope columns
    geometry=[box(xmin, ymin, xmax, ymax)
              for xmin, ymin, xmax, ymax in zip(
                  chunk["X_min"], chunk["Y_min"], chunk["X_max"], chunk["Y_max"]
              )],
    crs=slope_crs
    )


    # Spatial join
    candidates = gpd.sjoin(
        gdf, geology_gdf[["geometry", "PTYPE"]],
        how="left", predicate="intersects"
    )
    # print("After join:", candidates.columns.tolist())


    # Compute intersection areas ONCE
    candidates["intersection_area"] = candidates.apply(
        lambda row: row.geometry.intersection(
            geology_gdf.loc[int(row.index_right), "geometry"]
        ).area if pd.notna(row.index_right) else 0,
        axis=1
    )

    # Deduplicate inside this chunk
    candidates_resolved = (
        candidates.sort_values("intersection_area", ascending=False)
        .drop_duplicates(subset=["X_min", "Y_min", "X_max", "Y_max"])
    )

    #log(f"column names of candidates_resolved: {candidates_resolved.columns.tolist()}")



    # Drop heavy geometry column
    #candidates_resolved = candidates_resolved.drop(columns=["geometry", "index_right"], errors="ignore")

    return candidates_resolved

    


# === Process one parquet file (entrypoint) ===
def process_one_file(parq_path: Path, chunk_size=200000):
    try:
        log(f"Reading {parq_path.name}...")
        parquet_file = pq.ParquetFile(parq_path)
        total_rows = parquet_file.metadata.num_rows
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        log(f"{parq_path.name}: {total_rows} rows in {total_chunks} chunks")

        num_procs = max(1, mp.cpu_count() // 2)

        # Process chunks fully in parallel
        with mp.Pool(processes=num_procs) as pool:
            results = list(tqdm(
                pool.imap(
                    partial(process_chunk_full, slope_crs=slope_crs, geology_gdf=geology_gdf),
                    chunk_generator(parq_path, chunk_size,
                                    ["X_max", "X_min", "Y_max", "Y_min", "Slope Value"])
                ),
                total=total_chunks,
                desc=f"Processing {parq_path.name}",
                dynamic_ncols=True,
                position=0,
                mininterval=5
            ))

        

        # Combine all chunk results

        candidates_resolved = pd.concat(results, ignore_index=True)
        log(f"{parq_path.name}: concatenated {len(candidates_resolved)} rows after per-chunk dedup")

        # === FINAL DEDUP across chunks ===
        # candidates_resolved = (
        #     candidates_resolved.sort_values("intersection_area", ascending=False)
        #     .drop_duplicates(subset=["X_min", "Y_min", "X_max", "Y_max"])
        # )

        log(f"Column names:  {candidates_resolved.columns.tolist()}")

        candidates_resolved = candidates_resolved.drop(columns=["geometry", "index_right", "intersection_area"], errors="ignore")
        log(f"column names of candidates_resolved: {candidates_resolved.columns.tolist()}")

        # Sav
        out_path = output_dir / parq_path.name.replace("_slope_inside", "_slope_geology")
        candidates_resolved.to_parquet(out_path, index=False)
        print(candidates_resolved.head(3))

        log(f"✅ Saved {out_path} with {len(candidates_resolved)} rows")

        del candidates_resolved, results
        return f"✅ Done {parq_path.name}"

    except Exception as e:
        return f"❌ Error {parq_path.name}: {e}"


from pathlib import Path



parq_file = Path(f"/scratch/10725/mfidansoy1777/grid_numbers_added/grid_numbers_fixed/gdf_rect{i}_slope_inside.parquet")
# log(f"Saving file to {output_dir} …")
process_one_file(parq_file)