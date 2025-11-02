import os
import pandas as pd
from glob import glob

i = 2019
starting_number = 761  # Start from 366 for 2018

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
