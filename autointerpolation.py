# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import psycopg2
import subprocess
import os




# =========================================================
# PARAMETERS
# =========================================================
REFERENCE_RASTER = "land_use_simpl_32633.tif"
OUTPUT_RASTER = "idw_precip_norm_0_1_32633.tif"

P_MAX = 100.0

DB_NAME = "postgis_35_sample"
DB_USER = "postgres"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"

RASTER_TABLE = "precipitation_idw_norm_32633"

RASTER2PGSQL = r"C:\Program Files\PostgreSQL\17\bin\raster2pgsql.exe"
PSQL = r"C:\Program Files\PostgreSQL\17\bin\psql.exe"

# =========================================================
# 1. READ REFERENCE RASTER
# =========================================================

def mask_with_reference(reference_raster_path, raster_array):
    """
    Mask raster_array using NoData of reference raster.
    Cells where reference raster is NoData will be set to NoData.
    """
    with rasterio.open(reference_raster_path) as ref:
        ref_data = ref.read(1)
        ref_nodata = ref.nodata

    if ref_nodata is None:
        raise RuntimeError("Reference raster has no NoData value defined.")

    masked = raster_array.copy()
    masked[ref_data == ref_nodata] = ref_nodata

    return masked, ref_nodata



with rasterio.open(REFERENCE_RASTER) as src:
    transform = src.transform
    width = src.width
    height = src.height
    profile = src.profile

# Build interpolation grid explicitly (robust)
x0 = transform.c
y0 = transform.f
dx = transform.a
dy = transform.e  # negative in north-up rasters

xs = x0 + dx * (np.arange(width) + 0.5)
ys = y0 + dy * (np.arange(height) + 0.5)

grid_x, grid_y = np.meshgrid(xs, ys)
pixel_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))

print("Pixel coords sample:", pixel_coords[:5])
print("Pixel coords unique (x):", np.unique(pixel_coords[:,0]).size)
print("Pixel coords unique (y):", np.unique(pixel_coords[:,1]).size)
# =========================================================
# 2. LOAD POINTS FROM POSTGIS
# =========================================================
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

query = """
SELECT
    ST_X(geom) AS x,
    ST_Y(geom) AS y,
    value
FROM precipitation_points;
"""

points_df = pd.read_sql(query, conn)
conn.close()

if points_df.empty:
    raise RuntimeError("No data found in precipitation_points table.")

known_coords = points_df[["x", "y"]].values
known_values = points_df["value"].values

# =========================================================
# 3. IDW INTERPOLATION
# =========================================================
def idw(coords, values, query_coords, power=2, k=4):
    k = min(k, len(coords))
    tree = cKDTree(coords)
    dists, idxs = tree.query(query_coords, k=k)

    if k == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    weights = 1.0 / (dists ** power + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    return np.sum(values[idxs] * weights, axis=1)

interp_values = idw(known_coords, known_values, pixel_coords)

# =========================================================
# 4. PHYSICAL NORMALIZATION [0-1]
# =========================================================
normalized = interp_values / P_MAX
normalized = np.clip(normalized, 0.0, 1.0)

raster_2d = normalized.reshape((height, width))

# =========================================================
# 5. OPTIONAL SMOOTHING
# =========================================================
raster_smoothed = gaussian_filter(raster_2d, sigma=1)

raster_masked, ref_nodata = mask_with_reference(
    REFERENCE_RASTER,
    raster_smoothed
)


# =========================================================
# 6. SAVE RASTER TO FILE
# =========================================================
from rasterio.crs import CRS

profile.update(
    dtype="float32",
    count=1,
    nodata=ref_nodata
)

with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
    dst.write(raster_masked.astype("float32"), 1)


# =========================================================
# 7. LOAD RASTER INTO POSTGIS
# =========================================================
os.environ["PGPASSWORD"] = DB_PASSWORD

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

cur.execute(f"DROP TABLE IF EXISTS {RASTER_TABLE};")
conn.commit()

cur.close()
conn.close()

raster2pgsql_cmd = [
    RASTER2PGSQL,
    "-s", "32633",
    "-I", "-C", "-M",
    OUTPUT_RASTER,
    RASTER_TABLE
]

psql_cmd = [
    PSQL,
    "-d", DB_NAME,
    "-U", DB_USER,
    "-h", DB_HOST,
    "-p", DB_PORT
]

print("Importing raster into PostGIS...")

p1 = subprocess.Popen(raster2pgsql_cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(psql_cmd, stdin=p1.stdout)

p1.stdout.close()
p2.communicate()

print("DONE. Raster created and stored in PostGIS.")

print("Number of points:", len(known_coords))
print("Unique values:", np.unique(known_values))
print("Interpolated min/max:", interp_values.min(), interp_values.max())

