import subprocess
import os

# Define your variables
raster_path = "proximity_strade_masked_aligned2_32633.tif"
table_name = "proximity_strade_masked_aligned_32633"
srid = "32633"
database_name = "postgis_35_sample"
user = "postgres"
host = "localhost"
port = "5432"
password = "password"  # ?? Set your actual password here

# Set PGPASSWORD for authentication
os.environ["password"] = password

# Build the raster2pgsql command
raster2pgsql_cmd = [
    r"C:\Program Files\PostgreSQL\17\bin\raster2pgsql.exe",
    "-s", srid,
    "-I",  # create spatial index
    "-C",  # apply constraints
    "-M",  # vacuum analyze
    raster_path,
    table_name
]

# Build the psql command
psql_cmd = [
    r"C:\Program Files\PostgreSQL\17\bin\psql.exe",  # Use full path to avoid ambiguity
    "-d", database_name,
    "-U", user,
    "-h", host,
    "-p", port
]

# Pipe raster2pgsql into psql
raster2pgsql = subprocess.Popen(raster2pgsql_cmd, stdout=subprocess.PIPE)
psql = subprocess.Popen(psql_cmd, stdin=raster2pgsql.stdout)

# Finalize
raster2pgsql.stdout.close()
psql.communicate()
