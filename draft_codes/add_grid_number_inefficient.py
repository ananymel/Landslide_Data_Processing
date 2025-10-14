# import pandas as pd
# import numpy as np
# from pathlib import Path
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm

# # === CONFIG ===
# input_dir = Path("/mnt/c/Users/melis/Desktop/add_slope_parquet/10_07_25_gdf_parquet_version_repaired_serial")
# print(f"Input dir: {input_dir}")

# output_dir = input_dir / "grid_numbers_added"
# output_dir.mkdir(exist_ok=True)

# # === Collect parquet files (rect1..rect24) ===
# files = sorted(input_dir.glob("*.parquet"), key=lambda f: int("".join(filter(str.isdigit, f.stem))))

# # === Precompute starting grid index for each rect ===
# row_counts = []
# for f in files:
#     try:
#         n = pd.read_parquet(f, columns=["Slope Value"]).shape[0]  # read only one column for speed
#     except Exception:
#         n = pd.read_parquet(f).shape[0]
#     row_counts.append(n)

# start_indices = []
# current_start = 1
# for n in row_counts:
#     start_indices.append(current_start)
#     current_start += n

# # === Worker function with tqdm ===
# def add_grid_numbers(args):
#     file, start = args
#     try:
#         df = pd.read_parquet(file)
#         nrows = len(df)
        
#         # tqdm bar for this file
#         pbar = tqdm(total=nrows, desc=f"{file.name}", position=0, leave=True, mininterval=1)
        
#         # Assign GridNumber in chunks to update tqdm every 1M rows
#         chunk = 1_000_000
#         grid_numbers = np.empty(nrows, dtype=np.int64)
#         for i in range(0, nrows, chunk):
#             grid_numbers[i:i+chunk] = np.arange(i, min(i+chunk, nrows)) + start
#             pbar.update(min(chunk, nrows - i))
#         df["GridNumber"] = grid_numbers
#         pbar.close()

#         # Save output
#         out_path = output_dir / file.name
#         df.to_parquet(out_path, index=False)

#         return f"✅ {file.name}: {nrows} rows, start={start}, end={start+nrows-1}"
#     except Exception as e:
#         return f"❌ {file.name} failed: {e}"

# # === Parallel execution ===
# if __name__ == "__main__":
#     jobs = list(zip(files, start_indices))
#     with Pool(processes=min(cpu_count(), len(jobs))) as pool:
#         for r in pool.imap_unordered(add_grid_numbers, jobs, chunksize=1):
#             print(r)


################################################################################################################################################


import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import gc
import traceback
from multiprocessing import TimeoutError

# === CONFIG ===
input_dir = Path("/mnt/c/Users/melis/Desktop/add_slope_parquet/10_07_25_gdf_parquet_version_repaired_serial")
output_dir = input_dir / "grid_numbers_added"
output_dir.mkdir(exist_ok=True)

# === Collect parquet files ===
files = sorted(
    [f for f in input_dir.glob("*.parquet") if f"gdf_rect{int(''.join(filter(str.isdigit, f.stem)))}_slope_inside.parquet" and int("".join(filter(str.isdigit, f.stem))) not in {1, 2,3,4,5,6, 7,8, 9,11,12, 13,14, 15,16, 17, 18, 19, 20, 21,22,23, 24}],
    key=lambda f: int("".join(filter(str.isdigit, f.stem)))
)

# === Precompute starting grid index ===
row_counts = [pd.read_parquet(f, columns=["Slope Value"]).shape[0] for f in files]

start_indices, current_start = [], 1
for n in row_counts:
    start_indices.append(current_start)
    current_start += n


# === Worker function with percentage updates ===
def add_grid_numbers(args):
    file, start = args
    print(f"Worker started for: {file.name}", flush=True)  # Debugging output
    try:
        df = pd.read_parquet(file)
        nrows = len(df)
        chunk = 500_000

        grid_numbers = np.empty(nrows, dtype=np.int64)

        for i in range(0, nrows, chunk):
            end = min(i + chunk, nrows)
            grid_numbers[i:end] = np.arange(start + i, start + end, dtype=np.int64)

            # print percentage update
            percent = (end / nrows) * 100
            # print(f"{file.name}: {percent:.2f}% completed", flush=True)

        df["GridNumber"] = grid_numbers

        out_path = output_dir / file.name
        df.to_parquet(out_path, index=False)  # No compression
        del df, grid_numbers
        gc.collect()

        print(f"Worker finished for: {file.name}", flush=True)  # Debugging output
        return f"✅ {file.name}: finished ({nrows} rows, start={start}, end={start+nrows-1})"

    except Exception as e:
        error_message = f"❌ {file.name} failed: {e}\n{traceback.format_exc()}"
        print(error_message, flush=True)
        return f"❌ {file.name} failed: {e}"


# === Parallel execution ===
if __name__ == "__main__":
    jobs = list(zip(files, start_indices))
    with Pool(processes=cpu_count() // 5, maxtasksperchild=10) as pool:
        try:
            for r in pool.imap_unordered(add_grid_numbers, jobs, chunksize=1):
                print(r, flush=True)
        except TimeoutError:
            print("A worker process timed out.", flush=True)
