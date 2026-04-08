import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from loguru import logger

# --- paths ---
data_dir = Path("data")
raw_path  = data_dir / "rgs2020_data.csv"
out_path  = data_dir / "geochem_01_raw.parquet"

# --- load ---
logger.info(f"reading csv from {raw_path}...")
df = pd.read_csv(raw_path)
logger.info(f"loaded {len(df):,} rows x {len(df.columns)} columns")

# --- basic inspection ---
print(f"\n--- shape ---")
print(f"  {df.shape}")

print(f"\n--- sample media types (MAT) ---")
print(df["MAT"].value_counts().to_string())

print(f"\n--- coordinate nulls ---")
print(f"  LAT nulls:  {df['LAT'].isna().sum()}")
print(f"  LONG nulls: {df['LONG'].isna().sum()}")

print(f"\n--- year range ---")
print(f"  {df['YEAR'].min()} – {df['YEAR'].max()}")

print(f"\n--- first 3 rows (key cols) ---")
print(df[["MASTERID", "LAT", "LONG", "MAT", "YEAR", "ROCK_LITHO"]].head(3).to_string())

# --- drop rows with no coordinates ---
n_before = len(df)
df = df.dropna(subset=["LAT", "LONG"])
n_dropped = n_before - len(df)
logger.info(f"dropped {n_dropped} rows with missing coordinates — {len(df):,} remain")

# --- build geodataframe ---
logger.info("building geodataframe...")
geometry = [Point(lon, lat) for lon, lat in zip(df["LONG"], df["LAT"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
logger.info(f"geodataframe built — crs: {gdf.crs}")

print(f"\n--- bounds ---")
print(f"  lat:  {gdf['LAT'].min():.3f} – {gdf['LAT'].max():.3f}")
print(f"  long: {gdf['LONG'].min():.3f} – {gdf['LONG'].max():.3f}")

# --- save ---
logger.info(f"saving to {out_path}...")
out_path.parent.mkdir(parents=True, exist_ok=True)
gdf.to_parquet(out_path, index=False)
logger.info(f"saved successfully")

# --- verify ---
logger.info("verifying parquet...")
verify = gpd.read_parquet(out_path)
logger.info(f"verified: {len(verify):,} rows — crs: {verify.crs}")

print(f"\n--- done --- 01_raw.parquet ready")