import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from loguru import logger

# --- config ---
WFS_URL = "https://openmaps.gov.bc.ca/geo/pub/WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE/ows"
LAYER_NAME = "pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE"
PAGE_SIZE = 10_000  # DataBC's GeoServer caps a single WFS response at 10,000 features

# Red Chris deposit, MINFILE 104H 005 — v1 AOI anchor point
RED_CHRIS_LAT = 57.699722
RED_CHRIS_LON = -129.805278
AOI_RADIUS_M = 50_000

# Porphyry Cu-Au deposit type codes (Red Chris is coded L04 primary, L03 secondary)
PORPHYRY_CUAU_CODES = {"L03", "L04"}

data_dir = Path(__file__).resolve().parent.parent / "data"
raw_path = data_dir / "minfile_bc_raw.parquet"
aoi_path = data_dir / "minfile_aoi.parquet"

# --- build AOI ---
logger.info(f"building {AOI_RADIUS_M / 1000:.0f} km AOI buffer around Red Chris (BC Albers, EPSG:3005)...")
anchor = gpd.GeoSeries([Point(RED_CHRIS_LON, RED_CHRIS_LAT)], crs="EPSG:4326").to_crs("EPSG:3005")
aoi_geom = anchor.buffer(AOI_RADIUS_M).iloc[0]
logger.info(f"AOI bounds (EPSG:3005): {tuple(round(v) for v in aoi_geom.bounds)}")

# --- fetch full MINFILE layer from BC Data Catalogue (paginated) ---
logger.info("fetching MINFILE spatial layer from DataBC WFS...")
all_features = []
start_index = 0
while True:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": LAYER_NAME,
        "outputFormat": "application/json",
        "count": PAGE_SIZE,
        "startIndex": start_index,
    }
    resp = requests.get(WFS_URL, params=params, timeout=60)
    resp.raise_for_status()
    page = resp.json()
    all_features.extend(page["features"])
    total = page["totalFeatures"]
    logger.info(f"  fetched {len(all_features):,} / {total:,} features...")
    if len(all_features) >= total:
        break
    start_index += PAGE_SIZE

minfile = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:3005")
logger.info(f"loaded {len(minfile):,} MINFILE occurrences province-wide")

# --- save raw province-wide layer (cache — avoids re-querying the WFS on rerun) ---
data_dir.mkdir(parents=True, exist_ok=True)
minfile.to_parquet(raw_path, index=False)
logger.info(f"saved raw province-wide layer to {raw_path}")

# --- clip to AOI ---
logger.info("clipping to AOI...")
aoi_gdf = minfile[minfile.geometry.within(aoi_geom)].copy()
logger.info(f"{len(aoi_gdf):,} occurrences within {AOI_RADIUS_M / 1000:.0f} km of Red Chris")

# --- flag positive class: porphyry Cu-Au deposit types (MINFILE codes L03/L04) ---
deposit_code_cols = [c for c in aoi_gdf.columns if c.startswith("DEPOSIT_TYPE_CODE")]

def is_porphyry_cuau(row):
    return any(
        str(row[c]).strip() in PORPHYRY_CUAU_CODES
        for c in deposit_code_cols
        if pd.notna(row[c])
    )

aoi_gdf["is_porphyry_cuau"] = aoi_gdf.apply(is_porphyry_cuau, axis=1)

n_pos = int(aoi_gdf["is_porphyry_cuau"].sum())
logger.info(f"{n_pos} positive (porphyry Cu-Au, MINFILE L03/L04) occurrences in AOI")

print(f"\n--- positive class: porphyry Cu-Au occurrences in AOI ---")
print(aoi_gdf.loc[
    aoi_gdf["is_porphyry_cuau"],
    ["MINFILE_NUMBER", "MINFILE_NAME1", "STATUS_DESCRIPTION", "DEPOSIT_TYPE_DESCRIPTION1", "DEPOSIT_TYPE_DESCRIPTION2"],
].to_string())

print(f"\n--- all AOI occurrences by deposit class ---")
print(aoi_gdf["DEPOSIT_CLASS_DESCRIPTION1"].value_counts().to_string())

# --- save AOI-clipped layer ---
# NOTE: only `is_porphyry_cuau` rows are usable as positive labels. Every other row here
# is an occurrence of SOME mineralization (not confirmed barren) — it must not be treated
# as a negative/background example when pseudo-absence sampling is built in a later step.
aoi_gdf.to_parquet(aoi_path, index=False)
logger.info(f"saved AOI-clipped layer to {aoi_path}")

# --- verify ---
verify = gpd.read_parquet(aoi_path)
logger.info(f"verified: {len(verify):,} rows — crs: {verify.crs}")

print(f"\n--- done --- minfile_aoi.parquet ready ({n_pos} positive porphyry Cu-Au labels)")
