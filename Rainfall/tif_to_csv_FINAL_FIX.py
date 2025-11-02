"""
Generate individual CSV files from TIF files - FINAL CORRECTED VERSION
This version properly handles the scale factor (0.1) and NoData (32767)
"""

import rasterio
import numpy as np
import pandas as pd
import os
from glob import glob

# Set year
i = 2019

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
