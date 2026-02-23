# WebGIS Dynamic Framework for AHP + Random Forest Susceptibility Mapping

## Overview

This repository implements a dynamic, open-source WebGIS framework for
automated landslide susceptibility mapping. The system integrates AHP
(Analytic Hierarchy Process) and Random Forest models within a
PostGIS-based geospatial infrastructure, supporting dynamic rainfall
ingestion, raster processing, and WMS publication.

## Architecture

Weather Stations → PostgreSQL/PostGIS → Python Processing (AHP + RF) →
PostGIS Raster Storage → MapServer (WMS) → CesiumJS Viewer

## Core Components

-   Rainfall interpolation (IDW, normalized 0--1)
-   AHP weighted overlay with consistency ratio validation
-   Random Forest (LOOCV validation)
-   ROC/AUC performance evaluation
-   Automated raster storage in PostGIS
-   WMS publication via MapServer

## Requirements

-   PostgreSQL + PostGIS
-   Python 3.12 (Anaconda recommended)
-   GDAL tools (gdal_translate, raster2pgsql)
-   MapServer
-   CesiumJS

Required Python libraries: numpy, pandas, geopandas, rasterio, shapely,
matplotlib, scikit-learn, psycopg2, scipy

## Workflow

1.  Load rainfall stations into PostGIS
2.  Generate interpolated rainfall raster (IDW)
3.  Import static conditioning factors
4.  Run AHP + Random Forest model
5.  Publish susceptibility raster via WMS

## Reproducibility

-   Fixed CRS: EPSG:32633
-   Fixed AHP weights
-   Random seed = 42
-   Explicit raster alignment
-   Deterministic rainfall normalization (0--100 mm)

## License

Specify your preferred license (e.g., MIT).
