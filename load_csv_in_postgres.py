import pandas as pd
import psycopg2
from io import StringIO

# Connect once
conn = psycopg2.connect(
    "dbname=postgis_35_sample user=postgres password=password host=localhost port=5432"
)
cur = conn.cursor()

# 1) Ensure PostGIS is enabled
cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

# 2) Create target table
cur.execute("""
    CREATE TABLE IF NOT EXISTS precipitation_points (
        id SERIAL PRIMARY KEY,
        station_name TEXT NOT NULL,
        value REAL,
        geom geometry(Point, 32633)
    );
""")

# 3) Create spatial index
cur.execute("""
    CREATE INDEX IF NOT EXISTS precipitation_points_geom_idx
    ON precipitation_points
    USING GIST (geom);
""")

# Load CSV (lon, lat, station_name, value)
df = pd.read_csv('stazioni_meteo.dbf.csv', sep=';')

# Temp table
cur.execute("DROP TABLE IF EXISTS temp_precip;")
cur.execute("""
    CREATE TEMP TABLE temp_precip (
        lon DOUBLE PRECISION,
        lat DOUBLE PRECISION,
        station_name TEXT,
        value FLOAT
    );
""")

# Copy into temp table
buffer = StringIO()
df.to_csv(buffer, index=False, header=False, sep=';')
buffer.seek(0)
cur.copy_from(buffer, 'temp_precip', sep=';')

# Insert with CRS transform 4326 -> 32633
cur.execute("""
    INSERT INTO precipitation_points (geom, station_name, value)
    SELECT 
        ST_Transform(ST_SetSRID(ST_MakePoint(lon, lat), 4326), 32633),
        station_name,
        value
    FROM temp_precip;
""")

# Commit everything (table + data)
conn.commit()

cur.close()
conn.close()

print("OK: tabella precipitation_points creata/aggiornata e dati inseriti.")


