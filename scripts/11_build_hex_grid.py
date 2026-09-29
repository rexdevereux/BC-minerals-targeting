"""
11_build_hex_grid.py

Builds the province-wide modelling table on an H3 hexagon grid: every covariate
layer is summarised per hex, with neighbour-ring context and porphyry Cu-Au
labels. This is the table the models train on and predict over — predicting
every hex gives the continuous prospectivity map.

H3 resolution 7 (~5.2 km² hexes, ~2.4 km centre to centre) matches the
footprint of a porphyry system. RGS density is ~1 sample per 19 km², so most
hexes have no geochem sample; those are filled from neighbouring hexes and
flagged in `geochem_source`.

Columns per hex:
    raster covariates   {elevation,topo_position,mag_field,mag_vertical_deriv,
                         mag_analytic_signal,mag_tilt_deriv}_mean / _std
                        landform_{class}_frac — share of each Weiss landform class
                        mag_coverage — share of the hex with magnetic data
    neighbour context   {layer}_ring1_mean — mean over the hex and its 6 neighbours
    geology (at centre) rock_class, rock_type, era, terrane_name, terrane_group,
                        tectonic_setting, dist_to_fault_km, dist_to_terrane_boundary_km
    terrain inventory   tim_surficial_material ("not mapped" outside TIM coverage)
    geochemistry        n_geochem_samples, {element}_log_mean, {element}_zlocal_max,
                        score_porphyry_max, geochem_source (sampled / ring1 / ring2 / none)
    labels              n_minfile, n_porphyry_cuau, label_porphyry:
                          1   hex contains a porphyry Cu-Au occurrence (MINFILE L03/L04)
                          0   no MINFILE occurrence of any type within 2 rings (~5 km)
                          NaN otherwise — mineralised but not porphyry, can't be background

Usage:
    pixi run python scripts/11_build_hex_grid.py

Outputs:
    data/hex_grid.parquet     hex polygons (EPSG:3005) + all columns above
    data/minfile_h3.parquet   MINFILE occurrences tagged with their hex, for
                              deriving labels for other commodities
"""

import warnings
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon

H3_RES = 7
CRS = "EPSG:3005"
PORPHYRY_CUAU_CODES = {"L03", "L04"}
BOUNDARY_SIMPLIFY_M = 200
COLUMN_PREFIX = {  # raster file stem -> hex_grid column prefix
    "dem": "elevation",
    "tpi": "topo_position",
    "rtf": "mag_field",
    "1vd": "mag_vertical_deriv",
    "as": "mag_analytic_signal",
    "tdr": "mag_tilt_deriv",
}
LANDFORM_CLASSES = {
    1: "canyon", 2: "midslope_drainage", 3: "upland_drainage", 4: "u_valley", 5: "plain",
    6: "open_slope", 7: "upper_slope", 8: "local_ridge", 9: "midslope_ridge", 10: "mountain_top",
}

root_dir = Path(__file__).resolve().parents[1]
data_dir = root_dir / "data"
geomorph_dir = root_dir / "outputs" / "geomorphometry"
geophys_dir = root_dir / "outputs" / "geophysics"
out_path = data_dir / "hex_grid.parquet"
minfile_out_path = data_dir / "minfile_h3.parquet"


def nanmean_rows(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN is intended
        return np.nanmean(matrix, axis=1)


# --- hex grid over BC ---
logger.info(f"building H3 resolution {H3_RES} grid over BC...")
boundary = gpd.read_file(data_dir / "BC_boundary.gpkg").to_crs(CRS)
boundary = boundary.simplify(BOUNDARY_SIMPLIFY_M).to_crs("EPSG:4326").union_all()
cells = sorted(h3.geo_to_cells(boundary, H3_RES))
n = len(cells)
cell_index = {c: i for i, c in enumerate(cells)}
logger.info(f"  {n:,} hexes")

hex_polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
centre_latlng = np.array([h3.cell_to_latlng(c) for c in cells])
hexes = gpd.GeoDataFrame(
    {"h3_cell": cells, "centroid_lat": centre_latlng[:, 0], "centroid_lon": centre_latlng[:, 1]},
    geometry=hex_polys,
    crs="EPSG:4326",
).to_crs(CRS)
centres = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(centre_latlng[:, 1], centre_latlng[:, 0]), crs="EPSG:4326"
).to_crs(CRS)


def neighbour_matrix(k):
    """Row i holds the grid indices of hex i's k-disk; -1 where a neighbour is outside BC."""
    disks = [[cell_index.get(nb, -1) for nb in h3.grid_disk(c, k)] for c in cells]
    matrix = np.full((n, max(len(d) for d in disks)), -1, dtype=np.int64)
    for i, disk in enumerate(disks):
        matrix[i, : len(disk)] = disk
    return matrix


def disk_values(values, neighbours):
    return np.append(values, np.nan)[neighbours]  # index -1 picks the NaN pad


logger.info("  building neighbour rings...")
ring1 = neighbour_matrix(1)
ring2 = neighbour_matrix(2)


# --- raster zonal statistics ---
def zones_for(raster_path):
    with rasterio.open(raster_path) as src:
        zones = rasterize(
            zip(hexes.geometry, range(1, n + 1)),
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="int32",
        )
        grid_key = (src.shape, src.transform)
    return zones, grid_key


def read_on_grid(raster_path, grid_key):
    with rasterio.open(raster_path) as src:
        if (src.shape, src.transform) != grid_key:
            raise ValueError(f"{raster_path.name} is not on the same pixel grid as its siblings")
        return src.read(1).astype("float32")


def zonal_mean_std(values, zones):
    valid = np.isfinite(values) & (zones > 0)
    z, v = zones[valid], values[valid].astype("float64")
    count = np.bincount(z, minlength=n + 1)
    total = np.bincount(z, weights=v, minlength=n + 1)
    total_sq = np.bincount(z, weights=v * v, minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean**2, 0))
    return mean[1:], std[1:], count[1:]


def add_mean_std(name, values, zones):
    mean, std, count = zonal_mean_std(values, zones)
    hexes[f"{name}_mean"] = mean
    hexes[f"{name}_std"] = std
    return count


logger.info("zonal stats on the DEM grid (dem, tpi, landforms)...")
zones, grid_key = zones_for(geomorph_dir / "dem.tif")
for name in ["dem", "tpi"]:
    add_mean_std(COLUMN_PREFIX[name], read_on_grid(geomorph_dir / f"{name}.tif", grid_key), zones)
landforms = read_on_grid(geomorph_dir / "landforms.tif", grid_key)
valid = np.isfinite(landforms) & (zones > 0)
z, cls = zones[valid], landforms[valid]
landform_total = np.bincount(z, minlength=n + 1)
for code, label in LANDFORM_CLASSES.items():
    with np.errstate(invalid="ignore", divide="ignore"):
        hexes[f"landform_{label}_frac"] = (np.bincount(z[cls == code], minlength=n + 1) / landform_total)[1:]
del zones, landforms, valid, z, cls

logger.info("zonal stats on the magnetics grid (rtf, 1vd, as, tdr)...")
zones, grid_key = zones_for(geophys_dir / "rtf.tif")
pixels_per_hex = np.bincount(zones.ravel(), minlength=n + 1)[1:]
valid_counts = {
    name: add_mean_std(COLUMN_PREFIX[name], read_on_grid(geophys_dir / f"{name}.tif", grid_key), zones)
    for name in ["rtf", "1vd", "as", "tdr"]
}
with np.errstate(invalid="ignore", divide="ignore"):
    hexes["mag_coverage"] = np.where(pixels_per_hex > 0, valid_counts["rtf"] / pixels_per_hex, 0.0)
del zones

logger.info("neighbour-ring means...")
for prefix in COLUMN_PREFIX.values():
    hexes[f"{prefix}_ring1_mean"] = nanmean_rows(disk_values(hexes[f"{prefix}_mean"].to_numpy(), ring1))


# --- geology, terranes, structure at hex centres ---
def first_match(joined):
    return joined[~joined.index.duplicated(keep="first")].reindex(centres.index)


logger.info("joining bedrock geology and terranes at hex centres...")
geology_gpkg = data_dir / "BC_digital_geology.gpkg"
bedrock = gpd.read_file(geology_gpkg, layer="Bedrock_ll83_poly").to_crs(CRS)
joined = first_match(gpd.sjoin(centres, bedrock[["rock_class", "rock_type", "era", "geometry"]], how="left", predicate="within"))
for col in ["rock_class", "rock_type", "era"]:
    hexes[col] = joined[col].to_numpy()

terranes = gpd.read_file(data_dir / "BC_terranes.gpkg", layer="terranes").to_crs(CRS)
terrane_cols = terranes[["T_NAME", "TGP_SIMPLE", "TECT_SET", "geometry"]].rename(
    columns={"T_NAME": "terrane_name", "TGP_SIMPLE": "terrane_group", "TECT_SET": "tectonic_setting"}
)
joined = first_match(gpd.sjoin(centres, terrane_cols, how="left", predicate="within"))
for col in ["terrane_name", "terrane_group", "tectonic_setting"]:
    hexes[col] = joined[col].to_numpy()

logger.info("distance to nearest fault and terrane boundary...")
faults = gpd.read_file(geology_gpkg, layer="Faults_ll83_sp").to_crs(CRS)
nearest = first_match(gpd.sjoin_nearest(centres, faults[["geometry"]], distance_col="d"))
hexes["dist_to_fault_km"] = (nearest["d"] / 1000).round(2).to_numpy()
boundaries = gpd.GeoDataFrame(geometry=terranes.geometry.boundary, crs=CRS).explode(index_parts=False)
nearest = first_match(gpd.sjoin_nearest(centres, boundaries[["geometry"]], distance_col="d"))
hexes["dist_to_terrane_boundary_km"] = (nearest["d"] / 1000).round(2).to_numpy()

logger.info("joining terrain inventory (TIM) at hex centres...")
tim = gpd.read_parquet(data_dir / "terrain_inventory_bc_raw.parquet", columns=["DOMINANT_SURFICIAL_MATERIAL", "geometry"])
joined = first_match(gpd.sjoin(centres, tim, how="left", predicate="within"))
hexes["tim_surficial_material"] = joined["DOMINANT_SURFICIAL_MATERIAL"].fillna("not mapped").to_numpy()
del tim


# --- geochemistry aggregated per hex, gaps filled from neighbours ---
logger.info("aggregating geochem samples per hex...")
geochem = pd.read_parquet(data_dir / "geochem_05_features.parquet")
elements = sorted(c.removesuffix("_zscore_local") for c in geochem.columns if c.endswith("_zscore_local"))
geochem["idx"] = [
    cell_index.get(h3.latlng_to_cell(lat, lon, H3_RES), -1) for lat, lon in zip(geochem["latitude"], geochem["longitude"])
]
outside = int((geochem["idx"] < 0).sum())
geochem = geochem[geochem["idx"] >= 0]
aggregations = {f"{el}_log_mean": (f"{el}_log", "mean") for el in elements}
aggregations |= {f"{el}_zlocal_max": (f"{el}_zscore_local", "max") for el in elements}
aggregations |= {"score_porphyry_max": ("score_porphyry", "max"), "n_geochem_samples": ("sample_id", "size")}
per_hex = geochem.groupby("idx").agg(**aggregations).reindex(range(n))
logger.info(f"  {int(per_hex['n_geochem_samples'].notna().sum()):,} hexes sampled ({outside} samples fell outside the grid)")

sampled = per_hex["n_geochem_samples"].notna().to_numpy()
source = np.where(sampled, "sampled", "none").astype(object)
for col in [c for c in per_hex.columns if c != "n_geochem_samples"]:
    values = per_hex[col].to_numpy(dtype="float64")
    filled = values.copy()
    from_ring1 = nanmean_rows(disk_values(values, ring1))
    from_ring2 = nanmean_rows(disk_values(values, ring2))
    filled = np.where(np.isnan(filled), from_ring1, filled)
    filled = np.where(np.isnan(filled), from_ring2, filled)
    hexes[col] = filled
    if col == "score_porphyry_max":
        source = np.where(~sampled & np.isfinite(from_ring1), "ring1", source)
        source = np.where((source == "none") & np.isfinite(from_ring2), "ring2", source)
hexes["n_geochem_samples"] = per_hex["n_geochem_samples"].fillna(0).astype(int).to_numpy()
hexes["geochem_source"] = source


# --- labels from MINFILE ---
logger.info("assigning MINFILE occurrences to hexes and labelling...")
minfile = gpd.read_parquet(data_dir / "minfile_bc_raw.parquet").to_crs("EPSG:4326")
deposit_cols = [c for c in minfile.columns if c.startswith("DEPOSIT_TYPE_CODE")]
minfile["is_porphyry_cuau"] = (
    minfile[deposit_cols].apply(lambda col: col.astype("string").str.strip().isin(PORPHYRY_CUAU_CODES)).any(axis=1)
)
minfile["h3_cell"] = [h3.latlng_to_cell(p.y, p.x, H3_RES) for p in minfile.geometry]
minfile.to_crs(CRS).to_parquet(minfile_out_path, index=False)

idx = minfile["h3_cell"].map(cell_index).dropna().astype(int).to_numpy()
is_porphyry = minfile.loc[minfile["h3_cell"].isin(cell_index), "is_porphyry_cuau"].to_numpy()
hexes["n_minfile"] = np.bincount(idx, minlength=n)
hexes["n_porphyry_cuau"] = np.bincount(idx[is_porphyry], minlength=n)

has_occurrence = np.append(hexes["n_minfile"].to_numpy() > 0, False)
occurrence_nearby = has_occurrence[ring2].any(axis=1)
label = np.full(n, np.nan)
label[~occurrence_nearby] = 0
label[hexes["n_porphyry_cuau"].to_numpy() > 0] = 1
hexes["label_porphyry"] = label


# --- save ---
hexes.to_parquet(out_path, index=False)
logger.info(f"saved {out_path} ({len(hexes):,} hexes x {hexes.shape[1]} columns)")

print("\n--- label_porphyry ---")
print(pd.Series(label).value_counts(dropna=False).rename({1.0: "positive", 0.0: "background"}).to_string())
print("\n--- coverage ---")
print(f"  magnetics (any):     {(hexes['mag_coverage'] > 0).mean():.1%}")
print(f"  TIM mapped:          {(hexes['tim_surficial_material'] != 'not mapped').mean():.1%}")
print(f"  rock_class matched:  {hexes['rock_class'].notna().mean():.1%}")
print("  geochem source:      " + ", ".join(f"{k} {v:.1%}" for k, v in hexes["geochem_source"].value_counts(normalize=True).items()))
print(f"\n--- done --- hex_grid.parquet ready")
