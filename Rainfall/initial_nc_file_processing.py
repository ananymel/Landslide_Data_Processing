import os
import glob
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import numpy as np

# === Paths ===
input_dir = "/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/pr_2019.nc"
output_dir = "/Users/melis/Desktop/rainfallandndvi_october_updated/rainfall_updated_oct_2025/pr_2019_cropped.nc"
os.makedirs(output_dir, exist_ok=True)

# === Load California Boundary ===
boundary_path = "/Users/melis/Downlaods/hilsside_only _figure/only_hillside_landslide/Fixed_geometries/CA_Boundary/California_Boundary.shp"
ca_boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")  # ensure lat/lon

# === Loop over all .nc files ===
# nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
nc_files = input_dir
missing_data_counts = []


file_name = os.path.basename(nc_files)
print(f"📂 Processing {file_name}...")


# === Load NetCDF ===
ds = xr.open_dataset(file_name)
#print(ds.variables)
# === Check NDVI variable ===
if "precipitation_amount" not in ds.variables:
    print(f"⚠️ Skipping {file_name}: 'Rainfall' variable not found.")

#%%
# rain_var = None
# for var in ds.data_vars:
#     if any(key in var.lower() for key in ["rain", "precip", "pr"]):
#         rain_var = var
#         break

# if rain_var is None:
#     print(f"⚠️ Skipping {file_name}: rainfall variable not found.")
# else:
#     print(f"🌧 Found rainfall variable: {rain_var}")
#     rain = ds[rain_var]

#variable name is: precipitation amount
#%%
# === Load Rainfall and Set CRS ===
rainfall = ds["precipitation_amount"].rio.write_crs("EPSG:4326")

# === Clip using geometry (with drop=True to remove outer bounds) ===
rainfall_clipped = rainfall.rio.clip(ca_boundary.geometry, ca_boundary.crs, drop=True)

# === Optional: replace any -9999 with np.nan before counting
nodata_val = -9999
rainfall_clipped = rainfall_clipped.where(rainfall_clipped != nodata_val)

# === Count missing (NaN) values ===
missing_count = int(np.isnan(rainfall_clipped.values).sum())
missing_data_counts.append((file_name, missing_count))
print(f"❗ Missing values: {missing_count}")

# === Export as GeoTIFF with nodata defined ===
output_path = os.path.join(output_dir, file_name.replace(".nc", "_cropped.tif"))
rainfall_clipped.rio.to_raster(output_path, nodata=nodata_val)
print(f"✅ Saved: {output_path}")

#%%
# === Count missing (NaN) values ===
data = rainfall_clipped.values
missing_count = int(np.isnan(data).sum())
total_count = data.size
missing_percent = (missing_count / total_count) * 100

missing_data_counts.append((file_name, missing_count, total_count, missing_percent))

print(f"❗ Missing values: {missing_count:,} / {total_count:,} "
      f"({missing_percent:.2f}% missing)")

# === Export as GeoTIFF with nodata defined ===
output_path = os.path.join(output_dir, file_name.replace(".nc", "_cropped.tif"))
rainfall_clipped.rio.to_raster(output_path, nodata=nodata_val)
print(f"✅ Saved: {output_path}")
