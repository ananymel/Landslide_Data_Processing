"""
Rainfall single year NetCDF → Daily GeoTIFF Converter
----------------------------------------------
Processes only single year NetCDF file (pr_{i}.nc):
- Crops each day to California
- Saves one GeoTIFF/day as float32
- Sets NoData = -9999
- Uses LZW compression
"""
i= 2019
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