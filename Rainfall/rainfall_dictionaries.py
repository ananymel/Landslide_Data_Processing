import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from pathlib import Path
import json


def process_rainfall_yearly_nc(nc_path, output_dir, boundary_path, var_name="precipitation_amount"):
    """
    Processes one yearly rainfall .nc file:
    - Extract each daily layer
    - Crop to California boundary
    - Save each day as single-band GeoTIFF
    - Compute daily stats and missing-day report
    """
    nc_path = Path(nc_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = xr.open_dataset(nc_path)
    if var_name not in ds:
        raise ValueError(f"{var_name} not found in {nc_path}. Available: {list(ds.keys())}")
    rain = ds[var_name]

    # Get coordinate arrays
    if "lat" in ds:
        lat = ds["lat"]
    elif "latitude" in ds:
        lat = ds["latitude"]
    else:
        raise KeyError("No latitude variable found")

    if "lon" in ds:
        lon = ds["lon"]
    elif "longitude" in ds:
        lon = ds["longitude"]
    else:
        raise KeyError("No longitude variable found")

    # Convert to rioxarray and assign CRS
    rain = rain.rio.write_crs("EPSG:4326")

    # Load CA boundary and ensure CRS match
    ca_boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")

    # Get time dimension (days)
    times = pd.to_datetime(ds["day"].values)
    print(f"\n=== Processing {nc_path.name} ===")
    print(f"Year: {times[0].year}, Days: {len(times)}")

    valid_dates = []
    for i, day in enumerate(times):
        daily = rain.isel(day=i)  # select ith time slice
        daily_clipped = daily.rio.clip(ca_boundary.geometry, ca_boundary.crs, drop=True)
        data = daily_clipped.values.astype("float32")

        # Compute stats
        total = data.size
        nan_ct = np.isnan(data).sum()
        valid = total - nan_ct
        if valid > 0:
            mean_val = np.nanmean(data)
            print(f"{day.date()}: {valid:,}/{total:,} valid ({valid/total*100:.2f}%), mean={mean_val:.3f}")
            valid_dates.append(day)
        else:
            print(f"{day.date()}: all NaN")

        # Save daily GeoTIFF
        out_name = f"Rainfall_{day.strftime('%Y_%m_%d')}.tif"
        out_path = output_dir / out_name
        daily_clipped.rio.to_raster(out_path, nodata=np.nan)

    # --- Missing day check ---
    all_days = pd.date_range(f"{times[0].year}-01-01", f"{times[0].year}-12-31", freq="D")
    missing_days = all_days.difference(valid_dates)

    print(f"\n=== Summary for {times[0].year} ===")
    print(f"Total days: {len(all_days)}, Available: {len(valid_dates)}, Missing: {len(missing_days)}")
    if len(missing_days) > 0:
        for d in missing_days:
            print("  -", d.strftime("%Y-%m-%d"))

    return {
        "year": times[0].year,
        "total_days": len(all_days),
        "valid_days": len(valid_dates),
        "missing_days": len(missing_days),
        "missing_list": [d.strftime("%Y-%m-%d") for d in missing_days]
    }


def process_rainfall_folder(input_dir, output_dir, boundary_path, var_name="precipitation_amount"):
    """
    Loop over all yearly rainfall .nc files (e.g. pr_2017.nc, pr_2018.nc, ...)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(input_dir.glob("pr_*.nc"))
    if not nc_files:
        print(f"❌ No rainfall .nc files found in {input_dir}")
        return

    reports = []
    for nc_file in nc_files:
        year_output = output_dir / nc_file.stem.replace(".nc", "")
        year_output.mkdir(exist_ok=True)
        report = process_rainfall_yearly_nc(nc_file, year_output, boundary_path, var_name)
        reports.append(report)

    # Save global missing-days report
    rows = []
    for r in reports:
        for d in r["missing_list"]:
            rows.append({"Year": r["year"], "MissingDate": d})

    if rows:
        report_path = output_dir / "missing_days_report.csv"
        pd.DataFrame(rows).to_csv(report_path, index=False)
        print(f"\n📄 Missing-days report saved: {report_path}")
    else:
        print("\n✅ No missing days detected across all years.")
    return reports


# === MAIN ===
if __name__ == "__main__":
    boundary_path = "/Users/melis/Downlaods/hilsside_only _figure/only_hillside_landslide/Fixed_geometries/CA_Boundary/California_Boundary.shp"
    input_dir = "/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/not_cropped_nc_files"
    output_dir = "/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/cropped_ca_daily_tifs"

    reports = process_rainfall_folder(input_dir, output_dir, boundary_path)


#%%


import rasterio
from pathlib import Path
import json

# Pick any daily rainfall GeoTIFF
tif_path = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/"
                "rainfall_updated_oct_2025/cropped_ca_daily_tifs/pr_2017/"
                "Rainfall_2017_01_01.tif")

with rasterio.open(tif_path) as src:
    print("\n=== GeoTIFF Metadata ===")
    print(f"File: {tif_path.name}")
    print(f"CRS: {src.crs}")
    print(f"Width × Height: {src.width} × {src.height}")
    print(f"Data type: {src.dtypes}")
    print(f"Bands: {src.count}")
    print(f"Transform: {src.transform}")
    print(f"Bounds: {src.bounds}")
    print(f"Resolution: {src.res}")

    # Optional: print all metadata tags
    meta = src.meta
    print("\n--- Full Metadata ---")
    print(json.dumps(meta, indent=4, default=str))

# %%
import pandas as pd
import json
from pathlib import Path

# === Input paths ===
full_csv = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/"
                "rainfall_updated_oct_2025/cropped_ca_tif/bbox_master_from_tif.csv")

# === Output paths ===
filtered_csv = full_csv.parent / "bbox_master_CA_only_filtered.csv"
filtered_json = full_csv.parent / "bbox_master_CA_only_filtered.json"

# === Load the full 56,069-grid file ===
df_full = pd.read_csv(full_csv)
print(f"Full grid: {len(df_full):,} boxes")

# === Load the list of valid (inside CA) grid numbers ===
ca_boxes_csv = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/"
                    "rainfall_updated_oct_2025/cropped_ca_tif/bbox_master_CA_only.csv")
df_ca = pd.read_csv(ca_boxes_csv)
valid_ids = set(df_ca["grid_number"])
print(f"Valid California boxes: {len(valid_ids):,}")

# === Filter only California boxes ===
df_filtered = df_full[df_full["grid_number"].isin(valid_ids)].copy()
print(f"✅ Filtered rows kept: {len(df_filtered):,}")

# === Save filtered CSV and JSON ===
df_filtered.to_csv(filtered_csv, index=False)
df_filtered.to_json(filtered_json, orient="records", indent=4)
#%%
print(f"💾 Saved filtered CSV → {filtered_csv}")
print(f"💾 Saved filtered JSON → {filtered_json}")





# %%
import geopandas as gpd
import pandas as pd
import json
from pathlib import Path

# === Paths ===
base_dir = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/cropped_ca_tif")
geojson_in = base_dir / "bbox_master_CA_only.geojson"
csv_out = base_dir / "bbox_master_CA_only_filtered.csv"
json_out = base_dir / "bbox_master_CA_only_filtered.json"

# === Read GeoJSON ===
gdf = gpd.read_file(geojson_in)
print(f"Loaded {len(gdf):,} California boxes")

# === Drop geometry and export clean coordinate table ===
df = gdf.drop(columns="geometry")

df.to_csv(csv_out, index=False)
df.to_json(json_out, orient="records", indent=4)

print(f"✅ Saved filtered CSV → {csv_out}")
print(f"✅ Saved filtered JSON → {json_out}")

# %%
import geopandas as gpd
import pandas as pd
import json
from pathlib import Path

# === Paths ===
base_dir = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/cropped_ca_tif")
geojson_in = base_dir / "bbox_master_CA_only.geojson"
csv_out = base_dir / "bbox_master_CA_only_filtered_2.csv"
json_out = base_dir / "bbox_master_CA_only_filtered_2.json"

# === Read the *GeoJSON*, not the big CSV ===
gdf = gpd.read_file(geojson_in)
print(f"Loaded {len(gdf):,} California boxes")

# === Drop geometry column and re-save ===
df = gdf.drop(columns="geometry").copy()
df.to_csv(csv_out, index=False)
df.to_json(json_out, orient="records", indent=4)

print(f"✅ Exported {len(df):,} rows")
print(f"💾 Saved filtered CSV → {csv_out}")
print(f"💾 Saved filtered JSON → {json_out}")

# %%


import json
import pandas as pd
from pathlib import Path

json_path = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/"
                 "rainfall_updated_oct_2025/cropped_ca_tif/"
                 "bbox_master_CA_only_filtered_2.json")

csv_out = json_path.with_name("bbox_master_CA_only_filtered_final.csv")

# --- Load clean list ---
with open(json_path) as f:
    data = json.load(f)

# Should be a list of dicts
assert isinstance(data, list), "JSON is not a list"
print(f"✅ JSON contains {len(data):,} bounding boxes")

# --- Convert to DataFrame and save ---
df = pd.DataFrame(data)
cols = ["x_min", "y_max", "x_max", "y_min", "grid_number"]
df = df[[c for c in cols if c in df.columns]]

df.to_csv(csv_out, index=False)
print(f"💾 Saved final CSV → {csv_out}")
print(f"✅ CSV rows: {len(df):,}")


# %%
import json
from pathlib import Path

json_path = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/cropped_ca_tif/bbox_master_CA_only_filtered_2.json")

with open(json_path) as f:
    data = json.load(f)

print(f"Entries in JSON: {len(data):,}")




# %%


import json
from pathlib import Path

json_path = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/"
                 "rainfall_updated_oct_2025/cropped_ca_tif/"
                 "bbox_master_CA_only_filtered_2.json")

with open(json_path) as f:
    data = json.load(f)

print(type(data))
print(f"Top-level keys: {list(data)[:5]}")
print("Example element:", data[0] if isinstance(data, list) else "Not a list")



# %%

#Change the grid number column 

import pandas as pd
from pathlib import Path

csv_path = Path("/Users/melis/Desktop/rainfallandndvi_october_updated/"
                "rainfall_updated_oct_2025/cropped_ca_tif/"
                "bbox_master_CA_only_filtered_final_renumbered.csv")

# Load the existing CSV
df = pd.read_csv(csv_path)

# Renumber grid_number from 1 to len(df)
df["grid_number"] = range(0, len(df))

# Save as a new version (or overwrite)
out_path = csv_path.with_name("bbox_master_CA_only_filtered_final_renumbered.csv")
df.to_csv(out_path, index=False)

print(f"✅ Renumbered {len(df):,} grid cells from 1 to {len(df):,}")
print(f"💾 Saved updated file → {out_path}")





# %%
