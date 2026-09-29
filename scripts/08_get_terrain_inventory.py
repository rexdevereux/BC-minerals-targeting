"""
08_get_terrain_inventory.py

Fetches BC Terrain Inventory Mapping (TIM) polygons province-wide from the
DataBC WFS — the landform classification layer (surficial material, surface
expression, geomorphological process) requested for the regional model.

Coverage note: TIM was mapped project-by-project from the 1970s-80s onward
and does NOT cover all of BC — ~50% of BC's area and ~57% of RGS geochem
samples fall inside a TIM polygon. Downstream joins must treat "no TIM
polygon" as its own category rather than dropping those rows.

Usage:
    pixi run python scripts/08_get_terrain_inventory.py

Outputs:
    data/terrain_inventory_bc_raw.parquet
"""

from pathlib import Path

import geopandas as gpd
import requests
from loguru import logger

# --- config ---
WFS_URL = "https://openmaps.gov.bc.ca/geo/pub/WHSE_TERRESTRIAL_ECOLOGY.STE_TER_INVENTORY_POLYS_SVW/ows"
LAYER_NAME = "pub:WHSE_TERRESTRIAL_ECOLOGY.STE_TER_INVENTORY_POLYS_SVW"
PAGE_SIZE = 10_000  # DataBC's GeoServer caps a single WFS response at 10,000 features
# this layer has no primary key for natural ordering — GeoServer refuses
# startIndex paging without an explicit sort, so OBJECTID is forced here
SORT_BY = "OBJECTID"

root_dir = Path(__file__).resolve().parents[1]
data_dir = root_dir / "data"
raw_path = data_dir / "terrain_inventory_bc_raw.parquet"

# --- fetch full province-wide TIM layer (paginated) ---
logger.info("fetching TIM spatial layer from DataBC WFS...")
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
        "sortBy": SORT_BY,
    }
    resp = requests.get(WFS_URL, params=params, timeout=120)
    resp.raise_for_status()
    page = resp.json()
    all_features.extend(page["features"])
    total = page["totalFeatures"]
    logger.info(f"  fetched {len(all_features):,} / {total:,} features...")
    if len(all_features) >= total:
        break
    start_index += PAGE_SIZE

tim = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:3005")
logger.info(f"loaded {len(tim):,} TIM polygons province-wide")

# --- save ---
data_dir.mkdir(parents=True, exist_ok=True)
tim.to_parquet(raw_path, index=False)
logger.info(f"saved raw province-wide TIM layer to {raw_path}")

# --- verify ---
verify = gpd.read_parquet(raw_path)
logger.info(f"verified: {len(verify):,} rows — crs: {verify.crs}")

print("\n--- coverage by project type ---")
print(tim["PROJECT_TYPE"].value_counts().to_string())
print("\n--- top dominant surficial materials ---")
print(tim["DOMINANT_SURFICIAL_MATERIAL"].value_counts().head(15).to_string())
print(f"\n--- done --- terrain_inventory_bc_raw.parquet ready ({len(tim):,} polygons)")
