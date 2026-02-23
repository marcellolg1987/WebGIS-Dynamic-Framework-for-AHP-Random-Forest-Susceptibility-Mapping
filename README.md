# WebGIS Dynamic Framework for AHP + Random Forest Susceptibility Mapping

## Overview

This repository provides a dynamic, open-source WebGIS framework for
automated landslide susceptibility mapping.\
The system integrates knowledge-driven (AHP) and data-driven (Random
Forest) approaches within a PostGIS-based geospatial infrastructure.

The framework is designed to support: - Dynamic rainfall ingestion -
Automated raster processing - Susceptibility modeling (AHP + RF) -
ROC/AUC validation - WMS publication - 3D WebGIS visualization
(CesiumJS)

The architecture is modular, scalable, and Digital Twin--ready.

------------------------------------------------------------------------

## Scientific Background

### Analytic Hierarchy Process (AHP)

AHP is used to compute a weighted linear combination of seven
conditioning factors:

-   Rainfall
-   Slope
-   Lithology
-   Land Use
-   Aspect
-   Distance to Rivers
-   Distance to Roads

The model includes:

-   Pairwise comparison matrix
-   Eigenvector-based weight derivation
-   Consistency Index (CI)
-   Consistency Ratio (CR \< 0.10 threshold)
-   Continuous susceptibility index generation

### Random Forest (RF)

A Random Forest classifier refines susceptibility prediction using:

-   300 decision trees
-   max_depth = 2
-   min_samples_leaf = 3
-   class_weight = "balanced"
-   random_state = 42

Validation strategy: - Leave-One-Out Cross Validation (LOOCV) - ROC
curve - Area Under Curve (AUC)

The RF model produces: - Probability raster - Quantile-based classified
raster (5 classes)

------------------------------------------------------------------------

## System Architecture

Weather Stations → PostgreSQL/PostGIS → Python Processing (AHP + RF) →
PostGIS Raster Storage → MapServer (WMS) → CesiumJS Viewer

### Core Modules

1.  **Data Storage Layer**
    -   PostgreSQL + PostGIS
    -   Vector and raster storage
    -   EPSG:32633 (UTM WGS84 Zone 33N)
    -   10 × 10 m resolution
2.  **Processing Layer (Python / Anaconda)**
    -   IDW rainfall interpolation
    -   Raster normalization
    -   AHP weighted overlay
    -   RF training and prediction
    -   ROC generation
3.  **Publication Layer**
    -   MapServer WMS service
    -   CesiumJS 3D visualization

------------------------------------------------------------------------

## Repository Structure

. ├── scripts/ │ ├── ahp_rf.py │ ├── autointerpolation.py │ ├──
load_csv_in_postgres.py │ ├── raster_to_postgres.py │ ├── data/ │ ├──
landslides_4\_roc.zip │ ├── stazioni_meteo.dbf.csv │ ├── rasters/ │ ├──
slope_normalized_aligned2_32633.tif │ ├──
aspect_classified_aligned2_32633.tif │ ├──
litologia_final_aligned2_32633.tif │ ├──
land_use_simpl_aligned_32633.tif │ ├──
proximity_fiumi_masked_aligned_32633.tif │ ├──
proximity_strade_masked_aligned_32633.tif │ └── docs/

------------------------------------------------------------------------

## Requirements

### Database

-   PostgreSQL ≥ 14
-   PostGIS extension enabled

### Python

-   Python 3.12 (Anaconda recommended)

Required libraries: numpy, pandas, geopandas, rasterio, shapely,
matplotlib, scikit-learn, psycopg2, scipy

### External Tools

-   GDAL (gdal_translate)
-   raster2pgsql
-   psql
-   MapServer
-   CesiumJS
-   Apache (or equivalent web server)

------------------------------------------------------------------------

## Processing Workflow

### Step 1 --- Load Rainfall Stations

python load_csv_in_postgres.py

Creates: precipitation_points table (PostGIS geometry SRID 32633)

### Step 2 --- Generate Rainfall Raster (IDW)

python autointerpolation.py

-   Extracts 2-day cumulative rainfall
-   Performs IDW interpolation
-   Normalizes rainfall to \[0--1\] using fixed physical threshold
    (0--100 mm)
-   Masks with reference raster
-   Stores raster in PostGIS

### Step 3 --- Import Static Conditioning Factors

python raster_to_postgres.py

Loads static rasters into PostGIS.

### Step 4 --- Run Susceptibility Model

python ahp_rf.py

Generates: - Continuous AHP raster - Classified AHP raster - RF
probability raster - Classified RF raster - ROC_AHP.png - ROC_RF.png

------------------------------------------------------------------------

## Reproducibility

-   Fixed CRS (EPSG:32633)
-   Fixed AHP weights
-   Random seed = 42
-   Deterministic rainfall normalization (0--100 mm)
-   Explicit raster alignment
-   LOOCV validation strategy

------------------------------------------------------------------------

## Digital Twin Perspective

The framework supports dynamic geospatial data ingestion and automated
model updating.\
While not a full Digital Twin implementation, it represents an
early-stage Digital Twin--ready infrastructure capable of integrating
sensor-derived inputs, database storage, predictive modeling, and
real-time web visualization.

------------------------------------------------------------------------

## License

This project is released under the Creative Commons CC0 1.0 Universal
(CC0-1.0) Public Domain Dedication.

You are free to copy, modify, distribute and use the work, even for
commercial purposes, without asking permission.
