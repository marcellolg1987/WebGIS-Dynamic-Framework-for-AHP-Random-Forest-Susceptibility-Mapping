# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import LeaveOneOut


# =========================================================
# CONFIG
# =========================================================

# --- Cartelle (tutto salvato in locale qui)
WORKDIR = r""  # <- CAMBIA
DATA_DIR = os.path.join(WORKDIR, "00_input_from_postgis")
AHP_DIR  = os.path.join(WORKDIR, "01_ahp")
RF_DIR   = os.path.join(WORKDIR, "02_rf")
FIG_DIR  = os.path.join(WORKDIR, "99_figures")

for d in [DATA_DIR, AHP_DIR, RF_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Executables (se non sono nel PATH, metti percorso completo)
GDAL_TRANSLATE = "gdal_translate"  # oppure r"C:\OSGeo4W\bin\gdal_translate.exe"
RASTER2PGSQL   = "raster2pgsql"    # oppure r"C:\OSGeo4W\bin\raster2pgsql.exe"
PSQL           = "psql"            # oppure r"C:\Program Files\PostgreSQL\16\bin\psql.exe"

# --- Connessione DB
PG_HOST = "localhost"
PG_PORT = "5432"
PG_DB   = "postgis_35_sample"
PG_USER = "postgres"
PG_PWD  = "password"

# Schema/colonna raster
PG_SCHEMA = "public"
PG_RAST_COL = "rast"

# --- Sorgenti raster in PostGIS (tabella -> nome file locale)
# ATTENZIONE: qui metti i NOMI REALI delle tabelle PostGIS raster
POSTGIS_RASTERS = {
    "rain":       ("idw_precip_norm_0_1_aligned_32633", "idw_precip_norm_0_1_aligned_32633.tif"),
    "slope":      ("slope_normalized_aligned2_32633",   "slope_normalized_aligned2_32633.tif"),
    "lithology":  ("litologia_final_aligned2_32633",    "litologia_final_aligned2_32633.tif"),
    "landuse":    ("land_use_simpl_aligned2_32633",      "land_use_simpl_aligned2_32633.tif"),
    "aspect":     ("aspect_classified_aligned2_32633",  "aspect_classified_aligned2_32633.tif"),
    "dist_river": ("proximity_fiumi_masked_aligned_32633", "proximity_fiumi_masked_aligned_32633.tif"),
    "dist_road":  ("proximity_strade_masked_aligned_32633", "proximity_strade_masked_aligned_32633.tif"),
}

# --- Inventario frane (punti) in locale
FRANE_SHP = os.path.join(WORKDIR, "landslides_4_roc.shp")  # <- CAMBIA

# --- CRS e nodata
EPSG = 32633
NODATA = -9999.0

# --- AHP: pesi (devono sommare a 1)
AHP_WEIGHTS = {
    "rain":       0.22,
    "slope":      0.26,
    "lithology":  0.14,
    "landuse":    0.10,
    "aspect":     0.06,
    "dist_river": 0.12,
    "dist_road":  0.10,
}

# --- AHP: soglie per classi 1..5 (se i fattori sono 1..5, questa è ok)
AHP_BREAKS = [1.5, 2.5, 3.5, 4.5]  # -> classi 1..5

# --- RF: iperparametri “safe” per pochi positivi (come nel tuo rf.py) :contentReference[oaicite:2]{index=2}
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=2,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

# --- Negativi RF: presi dalle classi AHP 1–2 (come fai già) :contentReference[oaicite:3]{index=3}
NEG_FROM_AHP_CLASSES = (1, 2)

# --- Output tabelle in PostGIS
OUT_SCHEMA = "public"
OUT_AHP_TABLE = "susc_ahp_rast"
OUT_RF_TABLE  = "susc_ahp_rf_rast"

# Tile size per raster2pgsql (ottimizza ingest)
TILE = "128x128"

AHP_MATRIX = np.array([
    [1.00, 0.50, 3.00, 5.00, 3.00, 3.00, 7.00],
    [2.00, 1.00, 7.00, 7.00, 5.00, 4.00, 9.00],
    [0.33, 0.14, 1.00, 0.25, 2.00, 0.25, 3.00],
    [0.20, 0.14, 4.00, 1.00, 2.00, 0.33, 5.00],
    [0.33, 0.20, 0.50, 0.50, 1.00, 0.50, 1.00],
    [0.33, 0.25, 4.00, 3.00, 2.00, 1.00, 5.00],
    [0.14, 0.11, 0.33, 0.20, 1.00, 0.20, 1.00],
], dtype="float64")


# =========================================================
# HELPERS: comandi esterni
# =========================================================

def run_cmd(cmd, env=None):
    print("[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError(f"Comando fallito: {' '.join(cmd)}")
    return p.stdout

def pg_conninfo_for_gdal():
    # stringa GDAL PG:
    return f"PG:host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PWD}"

def pg_env():
    # per psql/raster2pgsql: passiamo password via env
    e = os.environ.copy()
    e["PGPASSWORD"] = PG_PWD
    return e

def export_postgis_raster_to_tif(table_name, out_tif):
    """
    Esporta un raster da PostGIS a GeoTIFF usando gdal_translate.
    Richiede che il raster sia in una tabella PostGIS (colonna rast).
    """
    conn = pg_conninfo_for_gdal()
    src = f"{conn} schema={PG_SCHEMA} table={table_name} column={PG_RAST_COL}"
    run_cmd([GDAL_TRANSLATE, "-of", "GTiff", src, out_tif], env=pg_env())

def import_tif_to_postgis(in_tif, out_table):
    """
    Carica un GeoTIFF in PostGIS usando raster2pgsql + psql.
    Sovrascrive la tabella (drop & create).
    """
    full_table = f"{OUT_SCHEMA}.{out_table}"

    # drop table
    drop_sql = f"DROP TABLE IF EXISTS {full_table};"
    run_cmd([PSQL, "-h", PG_HOST, "-p", PG_PORT, "-U", PG_USER, "-d", PG_DB, "-c", drop_sql], env=pg_env())

    # raster2pgsql pipe -> psql
    # -s EPSG, -I index, -C add constraints, -M vacuum analyze, -t tiling
    r2p = subprocess.Popen(
        [RASTER2PGSQL, "-s", str(EPSG), "-I", "-C", "-M", "-t", TILE, in_tif, full_table],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=pg_env()
    )
    psql = subprocess.Popen(
        [PSQL, "-h", PG_HOST, "-p", PG_PORT, "-U", PG_USER, "-d", PG_DB],
        stdin=r2p.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=pg_env()
    )
    r2p.stdout.close()
    out, err = psql.communicate()
    r2p_err = r2p.stderr.read()
    r2p.stderr.close()

    if psql.returncode != 0 or r2p.returncode not in (0, None):
        print(out)
        print(err)
        print(r2p_err)
        raise RuntimeError("Import raster in PostGIS fallito.")

    print(f"[OK] Caricato in PostGIS: {full_table}")

# =========================================================
# AHP + ROC (logica basata sul tuo somma_pesata_def_new.py) :contentReference[oaicite:4]{index=4}
# =========================================================

def compute_auc_from_scores(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    tpr = np.cumsum(labels == 1) / P
    fpr = np.cumsum(labels == 0) / N
    return fpr, tpr, float(np.trapezoid(tpr, fpr))

def sample_raster_at_points(src, xy_list):
    vals = []
    band = src.read(1)
    for x, y in xy_list:
        r, c = rowcol(src.transform, x, y)
        if 0 <= r < src.height and 0 <= c < src.width:
            vals.append(band[r, c])
        else:
            vals.append(np.nan)
    return np.array(vals, dtype="float64")

def ahp_run(feature_paths, out_cont, out_class):
    # somma pesata
    arrays = []
    mask = None
    meta = None

    for k, path in feature_paths.items():
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
            if meta is None:
                meta = src.meta.copy()
                if src.nodata is not None:
                    mask = (data != src.nodata)
                else:
                    mask = np.ones_like(data, dtype=bool)
            else:
                if src.nodata is not None:
                    mask &= (data != src.nodata)
            arrays.append(data * float(AHP_WEIGHTS[k]))

    susc = np.sum(arrays, axis=0).astype("float32")
    susc[~mask] = NODATA

    meta.update(dtype="float32", nodata=NODATA, count=1)

    with rasterio.open(out_cont, "w", **meta) as dst:
        dst.write(susc, 1)

    # classi 1..5
    b1, b2, b3, b4 = AHP_BREAKS
    cls = np.full_like(susc, NODATA, dtype="float32")
    valid = susc != NODATA

    cls[valid & (susc <= b1)] = 1
    cls[valid & (susc > b1) & (susc <= b2)] = 2
    cls[valid & (susc > b2) & (susc <= b3)] = 3
    cls[valid & (susc > b3) & (susc <= b4)] = 4
    cls[valid & (susc > b4)] = 5

    with rasterio.open(out_class, "w", **meta) as dst:
        dst.write(cls, 1)

    return susc, cls, meta

def ahp_roc(out_cont_tif, out_png):
    with rasterio.open(out_cont_tif) as src:
        susc = src.read(1)
        valid = np.isfinite(susc) & (susc != NODATA)

        gdf = gpd.read_file(FRANE_SHP).to_crs(src.crs)
        pts = [(p.x, p.y) for p in gdf.geometry]

        pos = sample_raster_at_points(src, pts)
        pos = pos[np.isfinite(pos)]

        # negativi random su celle valide (1:1), escludendo 2 pixel (20m) attorno alle frane
        pixel_size = abs(src.transform.a)
        exclude_buffer_m = 2 * pixel_size
        exclusion = gdf.buffer(exclude_buffer_m).unary_union

        rows, cols = np.where(valid)
        neg = []
        tries = 0
        while len(neg) < len(pos) and tries < 200000:
            tries += 1
            i = np.random.randint(0, len(rows))
            r, c = rows[i], cols[i]
            x, y = rasterio.transform.xy(src.transform, r, c, offset="center")
            if exclusion.contains(Point(x, y)):
                continue
            neg.append(susc[r, c])
        neg = np.array(neg, dtype="float64")

        scores = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)

        fpr, tpr, auc_val = compute_auc_from_scores(scores, labels)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AHP AUC = {auc_val:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC_AHP")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    return auc_val

# =========================================================
# RF + ROC + MAPPA (logica basata sul tuo rf.py) :contentReference[oaicite:5]{index=5}
# =========================================================

def sample_rasters_at_points(raster_paths, points_geom):
    data = []
    # usa il primo raster come riferimento griglia
    first = next(iter(raster_paths.values()))
    with rasterio.open(first) as ref:
        ref_band = ref.read(1)
        for p in points_geom:
            r, c = rowcol(ref.transform, p.x, p.y)
            if not (0 <= r < ref.height and 0 <= c < ref.width):
                continue
            row = {}
            ok = True
            for k, rp in raster_paths.items():
                with rasterio.open(rp) as src:
                    v = src.read(1)[r, c]
                    if src.nodata is not None and v == src.nodata:
                        ok = False
                        break
                    row[k] = v
            if ok:
                data.append(row)
    return pd.DataFrame(data)

def rf_run(feature_paths, ahp_cont_tif, ahp_class_tif, out_prob_tif, out_class_tif, out_roc_png):
    # positivi
    gdf_pos = gpd.read_file(FRANE_SHP).to_crs(f"EPSG:{EPSG}")
    df_pos = sample_rasters_at_points(feature_paths, gdf_pos.geometry)
    df_pos["label"] = 1

    # negativi dalle classi AHP 1-2
    with rasterio.open(ahp_class_tif) as src:
        ahp_cls = src.read(1)
        tfm = src.transform
        nod = src.nodata if src.nodata is not None else NODATA

    neg_mask = np.isin(ahp_cls, list(NEG_FROM_AHP_CLASSES))
    rr, cc = np.where(neg_mask)
    n_neg = len(df_pos)  # 1:1 come nel tuo rf.py :contentReference[oaicite:6]{index=6}
    if len(rr) < n_neg:
        raise ValueError("Pochi pixel disponibili per negativi (classi AHP 1-2).")

    idx = np.random.choice(len(rr), n_neg, replace=False)
    neg_points = [Point(*rasterio.transform.xy(tfm, rr[i], cc[i], offset="center")) for i in idx]
    df_neg = sample_rasters_at_points(feature_paths, neg_points)
    df_neg["label"] = 0

    df = pd.concat([df_pos, df_neg], ignore_index=True).dropna()
    X = df.drop(columns=["label"])
    y = df["label"].values.astype(int)

    # RF + LOOCV ROC
    rf = RandomForestClassifier(**RF_PARAMS)
    loo = LeaveOneOut()

    y_true, y_score = [], []
    for tr, te in loo.split(X):
        rf.fit(X.iloc[tr], y[tr])
        y_true.append(y[te][0])
        y_score.append(rf.predict_proba(X.iloc[te])[0, 1])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_rf = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"RF AUC = {auc_rf:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC_RF (LOOCV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_roc_png, dpi=200)
    plt.close()

    # fit finale su tutto
    rf.fit(X, y)

    # mappa probabilità su griglia del raster AHP continuo
# predizione raster (robusta: esclude NoData di ogni raster)
    with rasterio.open(ahp_cont_tif) as ref:
        meta = ref.meta.copy()
        out = np.full((ref.height, ref.width), meta.get("nodata", NODATA), dtype="float32")

# Leggi tutte le bande + costruisci mask valida usando nodata per ciascun raster
    bands = []
    valid_mask = None

    for name in X.columns:
        rp = feature_paths[name]
        with rasterio.open(rp) as src:
            b = src.read(1)
            nd = src.nodata

        # mask di validitÃ  per questo raster
            if nd is None:
                m = np.isfinite(b)
            else:
                m = np.isfinite(b) & (b != nd)

            bands.append(b)
            valid_mask = m if valid_mask is None else (valid_mask & m)

    stack = np.stack(bands, axis=-1)

# Predici solo sui pixel veramente validi
    Xpix = pd.DataFrame(stack[valid_mask], columns=X.columns)
    out[valid_mask] = rf.predict_proba(Xpix)[:, 1].astype("float32")

# Scrivi raster probabilitÃ 
    meta.update(dtype="float32", nodata=meta.get("nodata", NODATA), count=1)
    with rasterio.open(out_prob_tif, "w", **meta) as dst:
        dst.write(out, 1)


    # classi 1..5 per quantili (più stabile per RF)
    vals = out[valid_mask]
    q = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    cls = np.full_like(out, meta["nodata"], dtype="float32")
    cls[(out <= q[0]) & valid_mask] = 1
    cls[(out > q[0]) & (out <= q[1]) & valid_mask] = 2
    cls[(out > q[1]) & (out <= q[2]) & valid_mask] = 3
    cls[(out > q[2]) & (out <= q[3]) & valid_mask] = 4
    cls[(out > q[3]) & valid_mask] = 5

    with rasterio.open(out_class_tif, "w", **meta) as dst:
        dst.write(cls, 1)

    return auc_rf

# =========================================================
# MAIN PIPELINE
# =========================================================

def ahp_consistency_ratio(A):
    n = A.shape[0]

    eigvals, _ = np.linalg.eig(A)
    lambda_max = np.max(np.real(eigvals))

    CI = (lambda_max - n) / (n - 1)

    RI_TABLE = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
        5: 1.12, 6: 1.24, 7: 1.32,
        8: 1.41, 9: 1.45, 10: 1.49
    }
    RI = RI_TABLE[n]

    CR = CI / RI if RI != 0 else 0.0
    return lambda_max, CI, CR




def main():
    
        # --- AHP Consistency Ratio ---
    lambda_max, CI, CR = ahp_consistency_ratio(AHP_MATRIX)

    print("\n=== AHP CONSISTENCY CHECK ===")
    print(f"lambda_max = {lambda_max:.3f}")
    print(f"Consistency Index (CI) = {CI:.3f}")
    print(f"Consistency Ratio (CR) = {CR:.3f}")

    if CR < 0.10:
        print(">>> AHP matrix is CONSISTENT (CR < 0.10)")
    else:
        print(">>> WARNING: AHP matrix is NOT consistent (CR >= 0.10)")

    # 1) Export raster da PostGIS -> GeoTIFF locali
    print("=== EXPORT PostGIS -> GeoTIFF ===")
    local_paths = {}
    for key, (table, tif_name) in POSTGIS_RASTERS.items():
        out_tif = os.path.join(DATA_DIR, tif_name)
        export_postgis_raster_to_tif(table, out_tif)
        local_paths[key] = out_tif
        print("[OK]", key, "->", out_tif)

    # 2) AHP (continua + classi) + ROC
    print("\n=== AHP ===")
    ahp_cont = os.path.join(AHP_DIR, "suscettibilita_AHP_continua.tif")
    ahp_cls  = os.path.join(AHP_DIR, "suscettibilita_AHP_5classi.tif")
    susc, cls, meta = ahp_run(
        feature_paths={
            "rain": local_paths["rain"],
            "slope": local_paths["slope"],
            "lithology": local_paths["lithology"],
            "landuse": local_paths["landuse"],
            "aspect": local_paths["aspect"],
            "dist_river": local_paths["dist_river"],
            "dist_road": local_paths["dist_road"],
        },
        out_cont=ahp_cont,
        out_class=ahp_cls
    )
    ahp_auc = ahp_roc(ahp_cont, os.path.join(FIG_DIR, "ROC_AHP.png"))
    print(f"[AHP] AUC = {ahp_auc:.3f}")

    # 3) RF (AHP come feature) + ROC + mappe
    print("\n=== RF (AHP+RF) ===")
    rf_prob = os.path.join(RF_DIR, "suscettibilita_RF_prob.tif")
    rf_cls  = os.path.join(RF_DIR, "suscettibilita_RF_5classi.tif")

    # Feature RF: tutti i raster + AHP continuo
    feature_paths_rf = {
        "slope": local_paths["slope"],
        "rain": local_paths["rain"],
        "lithology": local_paths["lithology"],
        "landuse": local_paths["landuse"],
        "aspect": local_paths["aspect"],
        "dist_river": local_paths["dist_river"],
        "dist_road": local_paths["dist_road"],
        "ahp": ahp_cont,  # AHP come feature
    }

    rf_auc = rf_run(
        feature_paths=feature_paths_rf,
        ahp_cont_tif=ahp_cont,
        ahp_class_tif=ahp_cls,
        out_prob_tif=rf_prob,
        out_class_tif=rf_cls,
        out_roc_png=os.path.join(FIG_DIR, "ROC_RF.png")
    )
    print(f"[RF] AUC (LOOCV) = {rf_auc:.3f}")

    # 4) Import risultati in PostGIS (AHP continua + RF prob)
    print("\n=== IMPORT GeoTIFF -> PostGIS ===")
    import_tif_to_postgis(ahp_cont, OUT_AHP_TABLE)
    import_tif_to_postgis(rf_prob, OUT_RF_TABLE)

    print("\nFATTO.")
    print("Output locale:")
    print(" -", ahp_cont)
    print(" -", ahp_cls)
    print(" -", rf_prob)
    print(" -", rf_cls)
    print(" -", os.path.join(FIG_DIR, "ROC_AHP.png"))
    print(" -", os.path.join(FIG_DIR, "ROC_RF.png"))

if __name__ == "__main__":
    main()
