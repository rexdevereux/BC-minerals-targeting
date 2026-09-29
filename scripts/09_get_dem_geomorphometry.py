"""
09_get_dem_geomorphometry.py

Fetches a province-wide Copernicus DEM (GLO-90) mosaic via Microsoft Planetary
Computer STAC and derives the two geomorphometric covariates wanted for the
regional prospectivity model: TPI (continuous ridge/valley position) and the
Weiss (2001) landform classification (ridge/valley/plain/canyon/etc.) that's
built on top of it — distinct from the field-mapped TIM surficial-material
polygons fetched in 08_get_terrain_inventory.py.

90m resolution is a deliberate choice for this province-wide pass — full BC
at native 30m would be ~2.4 billion pixels (~10GB/band), too large for this
machine's RAM, and finer than the eventual hex-grid cells need anyway.
Re-fetch at 30m (cop-dem-glo-30) for any future site-specific model.

Usage:
    pixi run python scripts/09_get_dem_geomorphometry.py

Peak RAM is ~12GB (landforms step) — run it detached (nohup ... &) on WSL so a
dropped VS Code connection doesn't kill it mid-run.

Outputs:
    outputs/geomorphometry/dem.tif
    outputs/geomorphometry/tpi.tif
    outputs/geomorphometry/landforms.tif
"""

from pathlib import Path

import geopandas as gpd
import odc.stac
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401 - registers the .rio accessor
import xrspatial
from loguru import logger

RESOLUTION = 90  # metres — province-wide pass; see module docstring
OUTPUT_CRS = "EPSG:3005"

root_dir = Path(__file__).resolve().parents[1]
boundary_path = root_dir / "data" / "BC_boundary.gpkg"
output_dir = root_dir / "outputs" / "geomorphometry"
output_dir.mkdir(parents=True, exist_ok=True)

logger.info("loading BC boundary...")
boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")

logger.info("searching planetary computer for Copernicus DEM GLO-90 tiles...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
# bbox, not the full coastline polygon via `intersects` — BC's coastline has
# enough vertices to blow past the STAC API's request body size limit.
# Precise clipping still happens below via `geopolygon=` at load time.
search = catalog.search(collections=["cop-dem-glo-90"], bbox=list(boundary.total_bounds))
items = list(search.items())
logger.info(f"{len(items)} DEM tiles found covering BC")

logger.info(f"loading + reprojecting mosaic to {OUTPUT_CRS} at {RESOLUTION}m...")
ds = odc.stac.load(
    items,
    bands=["data"],
    geopolygon=boundary.geometry.iloc[0],
    crs=OUTPUT_CRS,
    resolution=RESOLUTION,
    chunks={"x": 1024, "y": 1024},
)
elevation = ds["data"].isel(time=0).squeeze(drop=True)
logger.info("compositing tiles into memory (dask -> numpy)...")
elevation = elevation.compute()
elevation = elevation.where(elevation > -1000)  # drop DEM nodata / ocean fill
logger.info(f"mosaic shape: {elevation.shape} ({elevation.size:,} pixels)")


def save(da, name):
    da = da.rio.write_crs(OUTPUT_CRS)
    path = output_dir / f"{name}.tif"
    da.rio.to_raster(path, compress="deflate")
    logger.info(f"  saved {path}")


save(elevation, "dem")
save(xrspatial.tpi(elevation), "tpi")
save(xrspatial.landforms(elevation), "landforms")

logger.info("done — geomorphometric covariates ready in outputs/geomorphometry/")
