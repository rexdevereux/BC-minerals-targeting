import json
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

# --- paths ---
data_dir   = Path("data")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

in_path      = data_dir / "geochem_02_standardised.parquet"
out_parquet  = data_dir / "geochem_03_validated.parquet"
out_geojson  = output_dir / "bc_geochem_samples.geojson"
out_report   = output_dir / "geochem_validation_report.json"

# --- load ---
logger.info(f"loading {in_path}...")
gdf = gpd.read_parquet(in_path)
logger.info(f"loaded {len(gdf):,} rows x {len(gdf.columns)} columns")

element_cols = [c for c in gdf.columns if c.endswith("_ppm") or c.endswith("_pct")]

# --- validation framework ---
checks = []

def check(name, level, passed, detail=""):
    status = "PASS" if passed else ("WARNING" if level == "warning" else "FAIL")
    checks.append({
        "check": name,
        "level": level,
        "status": status,
        "detail": detail,
    })
    icon = "✓" if passed else ("⚠" if level == "warning" else "✗")
    print(f"  {icon}  [{status:<7}] {name}: {detail}")
    return passed

print("\n--- running validation checks ---\n")

# 1. row count
n = len(gdf)
check("minimum row count", "critical",
      n >= 10_000,
      f"{n:,} rows")

# 2. no duplicate sample IDs
n_dupes = gdf["sample_id"].duplicated().sum()
check("no duplicate sample IDs", "critical",
      n_dupes == 0,
      f"{n_dupes} duplicates found")

# 3. coordinates within BC bounding box
# BC approx bounds: lat 48.3–60.0, lon -139.1–-114.0
lat_ok  = gdf["latitude"].between(48.0, 60.5).all()
lon_ok  = gdf["longitude"].between(-139.5, -113.5).all()
check("coordinates within BC bounds", "critical",
      lat_ok and lon_ok,
      f"lat range {gdf['latitude'].min():.3f}–{gdf['latitude'].max():.3f}, "
      f"lon range {gdf['longitude'].min():.3f}–{gdf['longitude'].max():.3f}")

# 4. no negative values remaining after BDL substitution
any_negative = False
for col in element_cols:
    n_neg = (gdf[col] < 0).sum()
    if n_neg > 0:
        any_negative = True
        logger.warning(f"  {col} still has {n_neg} negative values")
check("no negative element values", "critical",
      not any_negative,
      "all BDL substitutions applied correctly")

# 5. CRS is WGS84
correct_crs = gdf.crs is not None and gdf.crs.to_epsg() == 4326
check("CRS is EPSG:4326 (WGS84)", "critical",
      correct_crs,
      str(gdf.crs) if gdf.crs else "no CRS set")

# 6. geometry is valid and non-null
n_null_geom    = gdf.geometry.isna().sum()
n_invalid_geom = (~gdf.geometry.is_valid).sum()
check("geometry non-null and valid", "critical",
      n_null_geom == 0 and n_invalid_geom == 0,
      f"{n_null_geom} null, {n_invalid_geom} invalid geometries")

# 7. core element null rates under 25%
# lithium/cerium are sparse by design — exempt from this check
exempt = {"lithium_ppm", "cerium_ppm"}
high_null_cols = []
for col in element_cols:
    if col in exempt:
        continue
    null_pct = gdf[col].isna().mean() * 100
    if null_pct > 25:
        high_null_cols.append(f"{col} ({null_pct:.1f}%)")
check("core element null rates < 25%", "warning",
      len(high_null_cols) == 0,
      f"high null cols: {high_null_cols}" if high_null_cols else "all within threshold")

# 8. geologically plausible value ranges
# upper bounds based on known ore-grade maxima — extreme outliers suggest data error
plausibility = {
    "copper_ppm":     10_000,
    "nickel_ppm":     10_000,
    "zinc_ppm":       20_000,
    "lead_ppm":       20_000,
    "gold_ppm":       100,
    "arsenic_ppm":    10_000,
    "uranium_ppm":    1_000,
}
implausible = []
for col, max_val in plausibility.items():
    if col not in gdf.columns:
        continue
    n_exceed = (gdf[col] > max_val).sum()
    if n_exceed > 0:
        implausible.append(f"{col} ({n_exceed} values > {max_val})")
check("element values within plausible range", "warning",
      len(implausible) == 0,
      f"exceeds max: {implausible}" if implausible else "all within expected range")

# 9. year range sensible
yr_min, yr_max = gdf["year"].min(), gdf["year"].max()
check("year range sensible", "warning",
      1970 <= yr_min and yr_max <= 2025,
      f"{yr_min}–{yr_max}")

# 10. audit columns present
audit_cols = ["_pipeline_version", "_processed_at", "_source"]
missing_audit = [c for c in audit_cols if c not in gdf.columns]
check("audit columns present", "warning",
      len(missing_audit) == 0,
      f"missing: {missing_audit}" if missing_audit else "all present")

# --- summary ---
n_fail    = sum(1 for c in checks if c["status"] == "FAIL")
n_warn    = sum(1 for c in checks if c["status"] == "WARNING")
n_pass    = sum(1 for c in checks if c["status"] == "PASS")
pipeline_ok = n_fail == 0

print(f"\n--- validation summary ---")
print(f"  PASS:    {n_pass}")
print(f"  WARNING: {n_warn}")
print(f"  FAIL:    {n_fail}")
print(f"  pipeline ok: {pipeline_ok}")

if not pipeline_ok:
    raise RuntimeError("critical validation checks failed — pipeline halted")

# --- save validation report ---
report = {
    "run_at":      datetime.now(timezone.utc).isoformat(),
    "input_rows":  n,
    "pipeline_ok": pipeline_ok,
    "checks":      checks,
}
with open(out_report, "w") as f:
    json.dump(report, f, indent=2)
logger.info(f"validation report saved to {out_report}")

# --- save validated parquet ---
logger.info(f"saving {out_parquet}...")
gdf.to_parquet(out_parquet, index=False)
logger.info("parquet saved")

# --- export geojson for qgis ---
# subset to key columns only — geojson with 30 cols is unwieldy in qgis
geojson_cols = (
    ["sample_id", "year", "sample_media", "rock_litho",
     "latitude", "longitude"]
    + element_cols
    + ["geometry"]
)
geojson_cols = [c for c in geojson_cols if c in gdf.columns]
gdf_export = gdf[geojson_cols].copy()

logger.info(f"exporting geojson to {out_geojson}...")
gdf_export.to_file(out_geojson, driver="GeoJSON")
logger.info(f"geojson saved — {len(gdf_export):,} features, {len(geojson_cols)-1} attributes")

print(f"\n--- done ---")
print(f"  geochem_03_validated.parquet → next pipeline stage")
print(f"  bc_geochem_samples.geojson   → drag into QGIS to inspect")
print(f"  geochem_validation_report.json → audit trail")