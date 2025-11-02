#%%
"""
Rainfall single year NetCDF → Daily GeoTIFF Converter
----------------------------------------------
Processes only single year NetCDF file (pr_{i}.nc):
- Crops each day to California
- Saves one GeoTIFF/day as float32
- Sets NoData = -9999
- Uses LZW compression
"""
i= 2021
import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

def process_rainfall(nc_path, output_dir, boundary_path, var_name="precipitation_amount"):
    # --- Paths ---
    nc_path = Path(nc_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    ds = xr.open_dataset(nc_path)
    if var_name not in ds:
        raise ValueError(f"{var_name} not found. Available variables: {list(ds.keys())}")
    rain = ds[var_name]

    # --- Write CRS ---
    rain = rain.rio.write_crs("EPSG:4326")

    # --- Load California boundary ---
    ca_boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")

    # --- Extract time dimension ---
    times = pd.to_datetime(ds["day"].values)
    year = times[0].year  # Store year before loop to avoid variable collision
    print(f"\n=== Processing {nc_path.name} ===")
    print(f"Year: {year}, Days: {len(times)}")

    valid_dates = []

    # --- Loop over days ---
    for day_idx, day in enumerate(times):  # Changed 'i' to 'day_idx'
        daily = rain.isel(day=day_idx)
        try:
            # Clip to California
            daily_clipped = daily.rio.clip(ca_boundary.geometry, ca_boundary.crs, drop=True)

            # Convert to float32
            data = daily_clipped.values.astype("float32")

            # Compute statistics
            total = data.size
            nan_ct = np.isnan(data).sum()
            valid = total - nan_ct
            if valid > 0:
                mean_val = np.nanmean(data)
                print(f"{day.date()}: {valid:,}/{total:,} valid ({valid/total*100:.2f}%), mean={mean_val:.3f}")
                valid_dates.append(day)
            else:
                print(f"{day.date()}: all NaN")

            # Save GeoTIFF
            out_name = f"Rainfall_{day.strftime('%Y_%m_%d')}.tif"
            out_path = output_dir / out_name
            daily_clipped.rio.to_raster(
                out_path,
                dtype="float32",
                nodata=-9999,
                compress="LZW"
            )

        except Exception as e:
            print(f"⚠️ Skipped {day.date()} ({e})")

    # --- Missing day summary ---
    all_days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")  # Use 'year' and 'f' prefix
    missing_days = all_days.difference(valid_dates)

    print(f"\n=== Summary for {year} ===")  # Show actual year
    print(f"Total days: {len(all_days)}, Available: {len(valid_dates)}, Missing: {len(missing_days)}")
    if len(missing_days) > 0:
        print("Missing dates:")
        for d in missing_days:
            print("  -", d.strftime("%Y-%m-%d"))

    # --- Save missing-day report ---
    if len(missing_days) > 0:
        report_path = output_dir / f"missing_days_{year}.csv"  # Use 'year' and 'f' prefix
        pd.DataFrame({"MissingDate": [d.strftime("%Y-%m-%d") for d in missing_days]}).to_csv(report_path, index=False)
        print(f"📄 Missing-day report saved: {report_path}")

# ==========================================================
# === MAIN EXECUTION =======================================
# ==========================================================
if __name__ == "__main__":
    boundary_path = fr"/Users/melis/Downlaods/hilsside_only _figure/only_hillside_landslide/Fixed_geometries/CA_Boundary/California_Boundary.shp"
    nc_path = fr"/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/not_cropped_nc_files/pr_{i}.nc"
    output_dir = fr"/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/cropped_ca_daily_tifs/pr_{i}"

    process_rainfall(nc_path, output_dir, boundary_path)


#%% .tif to csv individual days

import rasterio
import numpy as np
import pandas as pd
import os
from glob import glob



# === Paths ===
input_dir = f"/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/cropped_ca_daily_tifs/pr_{i}"
output_dir = os.path.join(input_dir, "CSV_FILES")
os.makedirs(output_dir, exist_ok=True)

# === Find all GeoTIFFs ===
tif_files = sorted(glob(os.path.join(input_dir, "Rainfall_*.tif")))
print(f"📂 Found {len(tif_files)} TIFF files in {input_dir}\n")

if len(tif_files) == 0:
    print(f"ERROR: No TIF files found in {input_dir}")
    print("Please check the path!")
    exit(1)

# === Loop over each file ===
for idx, tif_path in enumerate(tif_files, 1):
    file_name = os.path.basename(tif_path).replace(".tif", "_values.csv")
    out_path = os.path.join(output_dir, file_name)

    if idx % 50 == 0 or idx <= 3:  # Print first 3 and every 50
        print(f"[{idx}/{len(tif_files)}] Processing: {os.path.basename(tif_path)}")

    with rasterio.open(tif_path) as src:
        # Get metadata
        nodata = src.nodata  # Should be 32767.0
        scale = src.scales[0] if src.scales[0] is not None else 1.0  # Should be 0.1
        offset = src.offsets[0] if src.offsets[0] is not None else 0.0  # Should be 0.0

        if idx == 1:
            print(f"  Metadata: NoData={nodata}, Scale={scale}, Offset={offset}")

        # Read the array
        arr = src.read(1).astype(float)

        # Mask NoData FIRST (before applying scale)
        arr[arr == nodata] = np.nan

        # Apply scale and offset AFTER masking NoData
        # Formula: actual_value = (stored_value * scale) + offset
        arr = (arr * scale) + offset

        if idx == 1:
            valid = arr[~np.isnan(arr)]
            print(f"  After scaling: Min={valid.min():.2f}, Max={valid.max():.2f}, Mean={valid.mean():.2f}")

        # Get transform (maps pixel -> geographic coordinates)
        transform = src.transform
        height, width = arr.shape

        # Collect data
        rows = []
        for r in range(height):
            for c in range(width):
                val = arr[r, c]
                if np.isnan(val):
                    continue

                # Compute bounding box for pixel
                x_min, y_max = transform * (c, r)
                x_max, y_min = transform * (c + 1, r + 1)

                rows.append([y_min, x_min, y_max, x_max, val])

    # === Create DataFrame ===
    df = pd.DataFrame(rows, columns=["latitude_min", "longitude_min", "latitude_max", "longitude_max", "rainfall_mm"])

    # Show stats for first 3 files
    if idx <= 3:
        if len(df) > 0:
            stats = df['rainfall_mm'].describe()
            print(f"  ✅ Extracted {len(df):,} pixels")
            print(f"     Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f} mm")
            print(f"     First 5 values: {df['rainfall_mm'].head(5).tolist()}")
        else:
            print(f"  ⚠️  WARNING: No valid pixels found!")

    # === Save CSV ===
    df.to_csv(out_path, index=False)

print(f"\n🎯 All .tif files processed successfully!")
print(f"📁 Output directory: {output_dir}")
print(f"\n✅ Values should now be CORRECT (divided by 10)")
print(f"   Jan 3 should have mean ~2.5 mm (not 25 mm)")

#%%

import os
import pandas as pd
from glob import glob


starting_number = 1462  # Start from 1 for 2017, 366 for 2018, 731 for 2019, 1096 for 2020, 1461 for 2021, 1826 for 2022, 2191 for 2023, 2556 for 2024

# === Paths ===
csv_dir = f"/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/cropped_ca_daily_tifs/pr_{i}/CSV_FILES"
out_path = os.path.join(csv_dir, f"Rainfall_CA_{i}_full_timeseries.csv")

# === Get all daily CSV files ===
csv_files = sorted(glob(os.path.join(csv_dir, f"Rainfall_{i}_*.csv")))
print(f"📂 Found {len(csv_files)} CSV files")

if not csv_files:
    raise FileNotFoundError("No CSV files found — check your path pattern.")

# === Read the first CSV as the base (keep first 4 columns) ===
base_df = pd.read_csv(csv_files[0])
base_coords = base_df.iloc[:, 0:4].copy()   # first four columns

# Column number starts from starting_number (366 for 2018)
col_number = starting_number

# Add first day's rainfall with numbered column
base_coords[str(col_number)] = base_df.iloc[:, 4]
print(f"✅ Added column {col_number} from {os.path.basename(csv_files[0])}")
col_number += 1

# === Loop through remaining files and append the 5th column ===
for csv_path in csv_files[1:]:
    df = pd.read_csv(csv_path)
    base_coords[str(col_number)] = df.iloc[:, 4]   # add rainfall values with number as column name

    if col_number % 50 == 0 or col_number == starting_number + len(csv_files) - 1:  # Print every 50 and last
        print(f"✅ Added column {col_number} from {os.path.basename(csv_path)}")

    col_number += 1

# === Save merged DataFrame ===
base_coords.to_csv(out_path, index=False)
print(f"\n💾 Saved merged rainfall table → {out_path}")
print(f"📊 Final shape: {base_coords.shape}")
print(f"📊 Columns: {starting_number} to {col_number - 1} ({col_number - starting_number} days)")
print(f"\nColumn names: {list(base_coords.columns[:10])}...")
print(f"\nFirst few rows:")
print(base_coords.head())
