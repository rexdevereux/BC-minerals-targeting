"""
07_get_minfile.py

Fetches every BC MINFILE mineral occurrence province-wide from the DataBC WFS.
Kept raw and unfiltered — deposit-type / commodity labelling happens
downstream (10_build_training_table.py) so other commodity models can
derive their own labels from the same file.

Usage:
    pixi run python scripts/07_get_minfile.py

Outputs:
    data/minfile_bc_raw.parquet
"""

from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger

# --- config ---
WFS_URL = "https://openmaps.gov.bc.ca/geo/pub/WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE/ows"
LAYER_NAME = "pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE"
PAGE_SIZE = 10_000  # DataBC's GeoServer caps a single WFS response at 10,000 features

root_dir = Path(__file__).resolve().parents[1]
data_dir = root_dir / "data"
raw_path = data_dir / "minfile_bc_raw.parquet"

# --- fetch full MINFILE layer (paginated) ---
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

# --- save ---
data_dir.mkdir(parents=True, exist_ok=True)
minfile.to_parquet(raw_path, index=False)

verify = gpd.read_parquet(raw_path)
logger.info(f"verified: {len(verify):,} rows — crs: {verify.crs}")

print("\n--- occurrences by deposit class ---")
print(minfile["DEPOSIT_CLASS_DESCRIPTION1"].value_counts().head(15).to_string())
print(f"\n--- done --- minfile_bc_raw.parquet ready ({len(minfile):,} occurrences)")
