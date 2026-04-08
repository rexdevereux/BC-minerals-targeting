import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# --- paths ---
data_dir  = Path("data")
in_path   = data_dir / "geochem_03_validated.parquet"
out_path  = data_dir / "geochem_04_spatial.parquet"

geology_gpkg  = data_dir / "BC_digital_geology.gpkg"
terranes_gpkg = data_dir / "BC_terranes.gpkg"

# --- load samples ---
logger.info("loading validated samples...")
gdf = gpd.read_parquet(in_path)
logger.info(f"loaded {len(gdf):,} samples — crs: {gdf.crs}")

# --- load spatial layers ---
logger.info("loading bedrock geology...")
bedrock = gpd.read_file(geology_gpkg, layer="Bedrock_ll83_poly")
logger.info(f"bedrock: {len(bedrock):,} polygons — crs: {bedrock.crs}")

logger.info("loading faults...")
faults = gpd.read_file(geology_gpkg, layer="Faults_ll83_sp")
logger.info(f"faults: {len(faults):,} features — crs: {faults.crs}")

logger.info("loading terranes...")
terranes = gpd.read_file(terranes_gpkg, layer="terranes")
logger.info(f"terranes: {len(terranes):,} polygons — crs: {terranes.crs}")

# --- reproject all to BC Albers (EPSG:3005) for accurate distance calculations ---
# WGS84 degrees are not equal-area — distances in degrees are meaningless
# BC Albers is the standard projection for BC provincial data
logger.info("reprojecting all layers to BC Albers (EPSG:3005)...")
gdf      = gdf.to_crs("EPSG:3005")
bedrock  = bedrock.to_crs("EPSG:3005")
faults   = faults.to_crs("EPSG:3005")
terranes = terranes.to_crs("EPSG:3005")
logger.info("reprojection done")

# --- 10km grid assignment ---
# grid cells aggregate nearby samples into regional zones for ML features
# using projected coordinates (metres) for clean 10km cells
grid_size_m = 10_000  # 10km in metres
gdf["grid_col"] = np.floor(gdf.geometry.x / grid_size_m).astype(int)
gdf["grid_row"] = np.floor(gdf.geometry.y / grid_size_m).astype(int)
gdf["grid_id"]  = gdf["grid_row"].astype(str) + "_" + gdf["grid_col"].astype(str)

n_cells      = gdf["grid_id"].nunique()
avg_per_cell = len(gdf) / n_cells
logger.info(f"grid assigned — {n_cells:,} cells, avg {avg_per_cell:.1f} samples/cell")

# --- spatial join: bedrock geology ---
# assigns rock_class, rock_type, terrane, era to each sample point
# using the polygon each point falls within
logger.info("joining bedrock geology to samples...")
bedrock_cols = bedrock[["rock_class", "rock_type", "rock_code",
                         "terrane", "era", "strat_age", "geometry"]]

gdf = gpd.sjoin(gdf, bedrock_cols, how="left", predicate="within")
gdf = gdf.drop(columns=["index_right"], errors="ignore")

n_matched = gdf["rock_class"].notna().sum()
logger.info(f"bedrock join: {n_matched:,} samples matched ({n_matched/len(gdf)*100:.1f}%)")

print(f"\n--- rock class distribution ---")
print(gdf["rock_class"].value_counts().to_string())

print(f"\n--- terrane distribution (top 15) ---")
print(gdf["terrane"].value_counts().head(15).to_string())

# --- spatial join: terranes ---
# assigns terrane name and tectonic setting from the dedicated terrane layer
logger.info("joining terranes...")
terrane_cols = terranes[["TERRANE", "T_NAME", "TGP_SIMPLE", "TECT_SET", "geometry"]]
terrane_cols = terrane_cols.rename(columns={
    "TERRANE":    "terrane_code",
    "T_NAME":     "terrane_name",
    "TGP_SIMPLE": "terrane_group",
    "TECT_SET":   "tectonic_setting",
})

gdf = gpd.sjoin(gdf, terrane_cols, how="left", predicate="within")
gdf = gdf.drop(columns=["index_right"], errors="ignore")

n_terrane = gdf["terrane_name"].notna().sum()
logger.info(f"terrane join: {n_terrane:,} samples matched ({n_terrane/len(gdf)*100:.1f}%)")

print(f"\n--- terrane group distribution ---")
print(gdf["terrane_group"].value_counts().to_string())

# --- distance to nearest fault (km) ---
# ore deposits cluster near faults — this is a key exploration feature
# unary_union merges all fault lines into one geometry for efficient distance calc
logger.info("computing distance to nearest fault...")
faults_union = faults.geometry.unary_union
gdf["dist_to_fault_m"]  = gdf.geometry.distance(faults_union)
gdf["dist_to_fault_km"] = (gdf["dist_to_fault_m"] / 1000).round(2)
gdf = gdf.drop(columns=["dist_to_fault_m"])

print(f"\n--- distance to nearest fault ---")
print(f"  min:    {gdf['dist_to_fault_km'].min():.1f} km")
print(f"  median: {gdf['dist_to_fault_km'].median():.1f} km")
print(f"  max:    {gdf['dist_to_fault_km'].max():.1f} km")
print(f"  within 10km of fault: {(gdf['dist_to_fault_km'] < 10).sum():,} samples")

# --- distance to nearest terrane boundary (km) ---
# porphyry Cu-Au deposits in BC cluster along terrane boundaries
# (e.g. Stikinia/Quesnellia boundary hosts Highland Valley, Mount Polley etc.)
logger.info("computing distance to nearest terrane boundary...")
terrane_bounds  = terranes.geometry.boundary.unary_union
gdf["dist_to_terrane_boundary_m"]  = gdf.geometry.distance(terrane_bounds)
gdf["dist_to_terrane_boundary_km"] = (gdf["dist_to_terrane_boundary_m"] / 1000).round(2)
gdf = gdf.drop(columns=["dist_to_terrane_boundary_m"])

print(f"\n--- distance to nearest terrane boundary ---")
print(f"  min:    {gdf['dist_to_terrane_boundary_km'].min():.1f} km")
print(f"  median: {gdf['dist_to_terrane_boundary_km'].median():.1f} km")
print(f"  max:    {gdf['dist_to_terrane_boundary_km'].max():.1f} km")
print(f"  within 25km of boundary: {(gdf['dist_to_terrane_boundary_km'] < 25).sum():,} samples")

# --- reproject back to WGS84 for storage ---
logger.info("reprojecting back to WGS84...")
gdf = gdf.to_crs("EPSG:4326")

# --- summary ---
new_cols = [
    "grid_row", "grid_col", "grid_id",
    "rock_class", "rock_type", "rock_code", "terrane", "era", "strat_age",
    "terrane_code", "terrane_name", "terrane_group", "tectonic_setting",
    "dist_to_fault_km", "dist_to_terrane_boundary_km",
]
print(f"\n--- new columns added ---")
for col in new_cols:
    if col in gdf.columns:
        print(f"  {col}")

# --- save ---
logger.info(f"saving to {out_path}...")
gdf.to_parquet(out_path, index=False)
logger.info(f"saved — {len(gdf):,} rows x {len(gdf.columns)} columns")
print(f"\n--- done --- geochem_04_spatial.parquet ready")