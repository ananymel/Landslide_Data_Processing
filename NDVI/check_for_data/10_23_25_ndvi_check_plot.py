#%%
import json
import plotly.graph_objects as go

# --- rectangle helper ---
def rect_polygon(x_min, y_min, x_max, y_max):
    """Return polygon (lon, lat) lists for a rectangle."""
    lons = [x_min, x_max, x_max, x_min, x_min]
    lats = [y_min, y_min, y_max, y_max, y_min]
    return lons, lats

# --- load NDVI JSON ---
def load_ndvi_map(json_file):
    """
    JSON keys are 'x_min|y_max|x_max|y_min'.
    Convert to dict {NDVI_ID: (x_min, y_min, x_max, y_max)}.
    """
    with open(json_file, "r") as f:
        ndvi_map = json.load(f)

    id_to_bbox = {}
    for k, v in ndvi_map.items():
        x_min, y_max, x_max, y_min = map(float, k.split("|"))
        id_to_bbox[v] = (x_min, y_min, x_max, y_max)  # reorder
    return id_to_bbox

# --- plotting ---
def plot_rectangles(gridcells, ndvi_boxes, out_file="rectangles_plot.html", zoom=12):
    fig = go.Figure()

    # add gridcells (red)
    for i, (x_min, y_min, x_max, y_max) in enumerate(gridcells):
        lons, lats = rect_polygon(x_min, y_min, x_max, y_max)
        fig.add_trace(go.Scattermap(
            lon=lons, lat=lats, mode="lines", fill="toself",
            name=f"Gridcell {i+1}", line=dict(width=2, color="red")
        ))

    # add NDVI boxes (green)
    for i, (x_min, y_min, x_max, y_max) in enumerate(ndvi_boxes):
        lons, lats = rect_polygon(x_min, y_min, x_max, y_max)
        fig.add_trace(go.Scattermap(
            lon=lons, lat=lats, mode="lines", fill="toself",
            name=f"NDVI {i+1}", line=dict(width=2, color="green")
        ))

    # center view on all boxes
    all_boxes = gridcells + ndvi_boxes
    all_lons = [x for b in all_boxes for x in (b[0], b[2])]
    all_lats = [y for b in all_boxes for y in (b[1], b[3])]
    center_lon = (min(all_lons) + max(all_lons)) / 2
    center_lat = (min(all_lats) + max(all_lats)) / 2

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center={"lat": center_lat, "lon": center_lon},
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.write_html(out_file, include_plotlyjs="cdn")
    print(f"✅ Saved plot to {out_file}")


#%%
# Example usage
json_file = "/Users/melis/Desktop/rainfallandndvi_october_updated/ndvi_updated_oct_2025/NDVI_Download_09_22_2025/ndvi_dictionaries_10_02_25/bbox_master_map.json"
id_to_bbox = load_ndvi_map(json_file)

ndvi_id = 529
ndvi_rect = id_to_bbox[ndvi_id]

gridcells = [
    (-123.606389, 38.847963, -123.605463, 38.848889),
    (-123.604259, 38.847963, -123.603333, 38.848889),
    (-123.599722, 38.847963, -123.598796, 38.848889),
]

plot_rectangles(gridcells, [ndvi_rect], "grid_vs_ndvi.html", zoom=13)

# %%
