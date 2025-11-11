import json
import pandas as pd
import os
from tqdm import tqdm
import sys

sys.path.append("/home1/10725/mfidansoy1777/.local/lib/python3.12/site-packages")
os.makedirs("/scratch/10725/mfidansoy1777/divide_ndvi_to_rects/ndvi_split_excels", exist_ok=True)

with open("/scratch/10725/mfidansoy1777/divide_ndvi_to_rects/ndvi_dict.json", "r") as f:
    ndvi_dict = json.load(f)

with open("/scratch/10725/mfidansoy1777/divide_ndvi_to_rects/bbox_master_map.json", "r") as f:
    bbox_map = json.load(f)

bbox_df = pd.DataFrame([
    {
        "bbox_str": k,
        "ndvi_id": v,
        "X_min": float(k.split("|")[0]),
        "Y_min": float(k.split("|")[1]),
        "X_max": float(k.split("|")[2]),
        "Y_max": float(k.split("|")[3])
    }
    for k, v in bbox_map.items()
])

rect_files = [f for f in os.listdir("/scratch/10725/mfidansoy1777/divide_ndvi_to_rects/rect_parquets") if f.endswith(".parquet")]

for rect_file in tqdm(rect_files):
    rect_path = os.path.join("/scratch/10725/mfidansoy1777/divide_ndvi_to_rects/rect_parquets", rect_file)
    rect_df = pd.read_parquet(rect_path)

    rect_xmin = rect_df["X_min"].min()
    rect_xmax = rect_df["X_max"].max()
    rect_ymin = rect_df["Y_min"].min()
    rect_ymax = rect_df["Y_max"].max()

    inside_mask = (
        (bbox_df["X_min"] >= rect_xmin) &
        (bbox_df["X_max"] <= rect_xmax) &
        (bbox_df["Y_min"] >= rect_ymin) &
        (bbox_df["Y_max"] <= rect_ymax)
    )
    ndvi_inside = bbox_df[inside_mask]

    if ndvi_inside.empty:
        continue

    rows = []
    for _, row in ndvi_inside.iterrows():
        ndvi_id = str(row["ndvi_id"])
        ndvi_values = ndvi_dict.get(ndvi_id, [])
        ndvi_values = [0 if v is None or pd.isna(v) else v for v in ndvi_values]
        row_data = [ndvi_id, row["X_min"], row["Y_min"], row["X_max"], row["Y_max"]] + ndvi_values
        rows.append(row_data)

    # First 5 columns are ID and bbox, rest are NDVI days
    columns = [None]*len(rows[0])
    columns[0:5] = ["NDVI_ID","X_min","Y_min","X_max","Y_max"]

    rect_ndvi_df = pd.DataFrame(rows, columns=columns)

    output_path = os.path.join("/scratch/10725/mfidansoy1777/divide_ndvi_to_rects/ndvi_split_excels", rect_file.replace(".parquet", "_ndvi_matrix.xlsx"))
    rect_ndvi_df.to_excel(output_path, index=False)
