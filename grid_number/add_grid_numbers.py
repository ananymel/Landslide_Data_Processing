#%% Assign Grid Number

import pandas as pd
import numpy as np
from pathlib import Path
import gc

# === CONFIG ===
input_dir = Path("/mnt/c/Users/melis/Desktop/add_slope_parquet/10_07_25_gdf_parquet_version_repaired_serial/grid_numbers_added")
output_dir = input_dir / "grid_numbers_fixed"
output_dir.mkdir(exist_ok=True)

# === Collect parquet files ===
files = sorted(input_dir.glob("*.parquet"), key=lambda f: int("".join(filter(str.isdigit, f.stem))))

# === Recalculate Grid Numbers ===
current_start = 1  # Start GridNumber from 1
for file in files:
    print(f"Processing: {file.name}", flush=True)
    try:
        # Read the file
        df = pd.read_parquet(file)

        # Recalculate GridNumber
        nrows = len(df)
        df["GridNumber"] = np.arange(current_start, current_start + nrows, dtype=np.int64)

        # Save the updated file
        out_path = output_dir / file.name
        df.to_parquet(out_path, index=False)

        # Update the starting index for the next file
        current_start += nrows
        del df
        

        print(f"✅ Fixed GridNumber for: {file.name} ({nrows} rows)", flush=True)

        gc.collect()
    except Exception as e:
        print(f"❌ Failed to process {file.name}: {e}", flush=True)





#%%Sanity Check

import pandas as pd
from pathlib import Path

# === CONFIG ===
input_dir = Path("/mnt/c/Users/melis/Desktop/add_slope_parquet/10_07_25_gdf_parquet_version_repaired_serial/grid_numbers_added/grid_numbers_fixed")

# === Collect parquet files ===
files = sorted(input_dir.glob("*.parquet"), key=lambda f: int("".join(filter(str.isdigit, f.stem))))

# === Calculate First and Last Index ===
for file in files:
    try:
        # Read only the GridNumber column to minimize memory usage
        df = pd.read_parquet(file, columns=["GridNumber"])

        # Get the first and last index
        first_index = df["GridNumber"].iloc[0]
        last_index = df["GridNumber"].iloc[-1]

        print(f"{file.name}: First Index = {first_index}, Last Index = {last_index}", flush=True)
    except Exception as e:
        print(f"❌ Failed to process {file.name}: {e}", flush=True)
