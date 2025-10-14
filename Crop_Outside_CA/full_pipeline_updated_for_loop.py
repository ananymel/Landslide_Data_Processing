import geopandas as gpd
import polars as pl
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm
import gc

BATCH_SIZE = 1_000_000

ca_shp = r"C:\Users\melis\Downloads\CA_Boundary\CA_Boundary\California_Boundary.shp"
ca = gpd.read_file(ca_shp).to_crs("EPSG:4326")
ca_union = ca.unary_union

parq_dir = Path(r"D:\06042025\09_19_25_gdf_parquet_version")
out_dir = Path(r"D:\06042025\09_19_25_gdf_SHAPE_FILES")
sample_out_dir = Path(r"C:\Users\melis\Desktop\09_19_2025_pythonfiles\shape_files_09_26_25")
sample_out_dir.mkdir(parents=True, exist_ok=True)


def process_z(z: int):
    print(f"\n=== Processing rect{z} ===")

    # Input/output paths
    parq_in = parq_dir / f"gdf_rect{z}_slope_inside.parquet"
    parq_out = parq_dir / f"09_20_25_gdf_rect{z}_slope_inside_updated.parquet"
    gpkg_out = out_dir / f"09_20_25_gdf_rect{z}_slope_inside_updated.gpkg"
    shp_sample_out = sample_out_dir / f"rect{z}_sample.shp"

    # ------------------- Step 1: Filter by CA -------------------
    lf = pl.scan_parquet(parq_in).select(["X_min", "X_max", "Y_min", "Y_max"])
    n_total = lf.select(pl.count()).collect().item()
    print(f"  → Total rows: {n_total:,}")

    kept_parts = []
    total_kept = 0

    total_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=total_batches, desc=f"Filtering rect{z}", unit="batch")

    for offset in range(0, n_total, BATCH_SIZE):
        batch = (
            lf.slice(offset, BATCH_SIZE)
              .collect(streaming=True)
              .to_pandas()
        )
        batch["geometry"] = batch.apply(
            lambda r: box(r["X_min"], r["Y_min"], r["X_max"], r["Y_max"]),
            axis=1
        )
        gdf = gpd.GeoDataFrame(batch, geometry="geometry", crs="EPSG:4326")
        gdf_filtered = gdf[gdf.within(ca_union)]
        total_kept += len(gdf_filtered)
        df_filtered = pl.from_pandas(gdf_filtered.drop(columns="geometry"))
        kept_parts.append(df_filtered)
        pbar.update(1)
        del batch, gdf, gdf_filtered, df_filtered  # free RAM

    pbar.close()

    if kept_parts:
        df_all = pl.concat(kept_parts, how="vertical")
        df_all.write_parquet(parq_out, compression="zstd")
        del df_all
    else:
        pl.DataFrame(schema={"X_min": pl.Float64, "X_max": pl.Float64,
                             "Y_min": pl.Float64, "Y_max": pl.Float64}) \
          .write_parquet(parq_out)

    n_removed = n_total - total_kept
    print(f"✔ Saved filtered file: {parq_out}")
    print(f"  Kept rows: {total_kept:,}, Removed rows: {n_removed:,}")

    del kept_parts

    # ------------------- Step 2: Convert to GPKG -------------------
    print(f"Converting {parq_out.name} → GPKG")
    lf2 = pl.scan_parquet(parq_out).select(["X_min", "X_max", "Y_min", "Y_max"])
    nrows = lf2.select(pl.count()).collect().item()
    total_batches = (nrows + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=total_batches, desc=f"Writing GPKG rect{z}", unit="batch")

    first_batch = True
    for offset in range(0, nrows, BATCH_SIZE):
        batch = (
            lf2.slice(offset, BATCH_SIZE)
               .collect(streaming=True)
               .to_pandas()
        )
        batch["geometry"] = batch.apply(
            lambda r: box(r["X_min"], r["Y_min"], r["X_max"], r["Y_max"]), axis=1
        )
        gdf = gpd.GeoDataFrame(batch, geometry="geometry", crs="EPSG:4326")
        gdf.to_file(
            gpkg_out,
            layer="grid",
            driver="GPKG",
            mode="w" if first_batch else "a"
        )
        first_batch = False
        pbar.update(1)
        del batch, gdf

    pbar.close()
    print(f"✔ Saved GeoPackage: {gpkg_out}")

    # ------------------- Step 3: Sample 1% → Shapefile -------------------
    gdf_final = gpd.read_file(gpkg_out)
    sample = gdf_final.sample(frac=0.01, random_state=42)
    sample.to_file(shp_sample_out, driver="ESRI Shapefile")
    print(f"✅ Saved sample shapefile: {shp_sample_out}")

    # cleanup everything
    del lf, lf2, gdf_final, sample
    gc.collect()


# === Loop over all z values ===
for z in [13, 14]:
    process_z(z)
    gc.collect()

print("🎉 All files processed and sampled successfully.")
