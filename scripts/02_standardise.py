import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

# --- paths ---
data_dir = Path("data")
in_path  = data_dir / "geochem_01_raw.parquet"
out_path = data_dir / "geochem_02_standardised.parquet"

# --- load ---
logger.info(f"loading {in_path}...")
gdf = gpd.read_parquet(in_path)
logger.info(f"loaded {len(gdf):,} rows x {len(gdf.columns)} columns")

# --- media filter ---
# keep stream sediment only — most appropriate for regional mineral exploration
# targeting and most comparable to USGS NGS soil/sediment data
# "Stream Sediment and Water" includes paired water samples at same sites — keep
stream_types = ["Stream Sediment and Water", "Stream Sediment"]
before = len(gdf)
gdf = gdf[gdf["MAT"].isin(stream_types)].copy()
logger.info(f"media filter: kept {len(gdf):,} stream sediment rows (dropped {before - len(gdf):,})")

print(f"\n--- media types retained ---")
print(gdf["MAT"].value_counts().to_string())

# --- element column selection ---
# METHOD CHOICE: prefer ICP (ICP-MS / ICP-OES) as primary method
# ICP gives the most complete multi-element coverage in this dataset
# where multiple methods exist for the same element (e.g. Cu_AAS_PPM,
# Cu_ICP_PPM, Cu_INA_PPM) we take ICP only and drop the rest
# exception: Au_FA_PPB (fire assay) is the gold standard for gold — kept alongside ICP

element_mapping = {
    # target critical minerals
    "Cu_ICP_PPM": "copper_ppm",
    "Ni_ICP_PPM": "nickel_ppm",
    "Co_ICP_PPM": "cobalt_ppm",
    "Mo_ICP_PPM": "molybdenum_ppm",
    "Zn_ICP_PPM": "zinc_ppm",
    "Pb_ICP_PPM": "lead_ppm",
    "Li_ICP_PPM": "lithium_ppm",
    # gold — fire assay most accurate but sparse (13.5% coverage)
    # fallback chain: FA → ICP → INA to maximise coverage
    # handled separately below after main rename
    "Au_FA_PPB":  "gold_fa_ppb",
    "Au_ICP_PPB": "gold_icp_ppb",
    # Au_INA_PPB excluded — 23,160 BDL substitutions introduces too much
    # imputed noise; FA→ICP alone gives ~80% coverage, consistent with other elements
    # pathfinders
    "As_ICP_PPM": "arsenic_ppm",
    "Cr_ICP_PPM": "chromium_ppm",
    "Mn_ICP_PPM": "manganese_ppm",
    "Sb_ICP_PPM": "antimony_ppm",
    "Ba_ICP_PPM": "barium_ppm",
    # rare earth / critical minerals bonus
    "La_ICP_PPM": "lanthanum_ppm",
    "Ce_ICP_PPM": "cerium_ppm",
    "U_ICP_PPM":  "uranium_ppm",
}

# metadata columns to keep
meta_cols = [
    "MASTERID", "LAT", "LONG", "MAT", "YEAR", "DATE",
    "ROCK_LITHO", "STRAT", "SORC", "UTMZ", "UTME83", "UTMN83",
    "geometry"
]

# check which element cols actually exist
available = {k: v for k, v in element_mapping.items() if k in gdf.columns}
missing   = [k for k in element_mapping if k not in gdf.columns]
if missing:
    logger.warning(f"columns not found in data (skipped): {missing}")

logger.info(f"retaining {len(available)} element columns")

# --- subset and rename ---
keep_cols = [c for c in meta_cols if c in gdf.columns] + list(available.keys())
gdf = gdf[keep_cols].copy()
gdf = gdf.rename(columns=available)
gdf = gdf.rename(columns={
    "MASTERID":  "sample_id",
    "LAT":       "latitude",
    "LONG":      "longitude",
    "MAT":       "sample_media",
    "YEAR":      "year",
    "DATE":      "date",
    "ROCK_LITHO":"rock_litho",
    "STRAT":     "stratigraphy",
    "SORC":      "source_lab",
    "UTMZ":      "utm_zone",
    "UTME83":    "utm_easting",
    "UTMN83":    "utm_northing",
})

print(f"\n--- columns after rename ---")
print([c for c in gdf.columns if c != "geometry"])

# --- BDL handling ---
# negative values = below detection limit, not real negatives
# industry standard: substitute with abs(value) / 2
# preserves information that element was present at low level
element_cols = list(available.values())
print(f"\n--- below detection limit substitution ---")

for col in element_cols:
    if col not in gdf.columns:
        continue
    # coerce to numeric first — some cols may have string artefacts
    gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    n_bdl = (gdf[col] < 0).sum()
    gdf[col] = gdf[col].apply(
        lambda x: abs(x) / 2.0 if pd.notna(x) and x < 0 else x
    )
    print(f"  {col:<20} BDL values substituted: {n_bdl:>5,}")

# --- unit standardisation ---
# gold fallback chain: FA (most accurate) → ICP → INA (most coverage)
# all three methods in PPB — convert final column to PPM
# this gets us from 13.5% (FA only) to ~86%+ coverage
for col in ["gold_fa_ppb", "gold_icp_ppb", "gold_ina_ppb"]:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

gdf["gold_ppm"] = (
    gdf.get("gold_fa_ppb", pd.Series(dtype=float))
    .combine_first(gdf.get("gold_icp_ppb", pd.Series(dtype=float)))
) * 0.001

gdf = gdf.drop(columns=["gold_fa_ppb", "gold_icp_ppb"], errors="ignore")
n_gold = gdf["gold_ppm"].notna().sum()
logger.info(f"gold fallback chain applied — {n_gold:,} values ({n_gold/len(gdf)*100:.1f}% coverage)")

# refresh element cols list after gold rename
element_cols = [c for c in gdf.columns
                if c.endswith("_ppm") or c.endswith("_pct")]

# --- basic stats post-cleaning ---
print(f"\n--- element summary (post BDL substitution) ---")
for col in element_cols:
    if col in gdf.columns:
        vals = gdf[col].dropna()
        print(f"  {col:<20} n={len(vals):>6,}  "
              f"min={vals.min():.3f}  "
              f"median={vals.median():.3f}  "
              f"max={vals.max():.1f}")

# --- null summary ---
print(f"\n--- null counts per element ---")
for col in element_cols:
    if col in gdf.columns:
        n_null = gdf[col].isna().sum()
        pct    = n_null / len(gdf) * 100
        print(f"  {col:<20} nulls: {n_null:>5,}  ({pct:.1f}%)")

# --- audit columns ---
gdf["_pipeline_version"] = "1.0.0"
gdf["_processed_at"]     = datetime.now(timezone.utc).isoformat()
gdf["_source"]           = "BC RGS 2020 — BCGS GeoFile 2020-08"
gdf["_media_filter"]     = "stream sediment only"
gdf[    "_method_choice"]    = "ICP primary; Au fallback chain FA→ICP only (INA excluded — too many BDL substitutions)"

# --- save ---
logger.info(f"saving to {out_path}...")
gdf.to_parquet(out_path, index=False)
logger.info("saved successfully")

# --- verify ---
verify = gpd.read_parquet(out_path)
logger.info(f"verified: {len(verify):,} rows x {len(verify.columns)} columns")
print(f"\n--- done --- 02_standardised.parquet ready")