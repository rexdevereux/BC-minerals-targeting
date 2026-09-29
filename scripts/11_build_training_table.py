"""
10_build_training_table.py

Joins province-wide geochemical features with MINFILE occurrence labels into a
single training table for a regional (BC-wide) porphyry Cu-Au prospectivity
model. Satellite indices are deliberately excluded here — they only exist for
one candidate AOI (not province-wide coverage) and belong to a separate,
site-specific model later.

Label scheme (pseudo-absence sampling):
    positive (1)  — geochem sample within POSITIVE_RADIUS_M of a known
                    porphyry Cu-Au occurrence (MINFILE L03/L04)
    excluded (NaN)— sample within EXCLUSION_RADIUS_M of ANY MINFILE occurrence
                    (positive or not) — mineralization present but not
                    confirmed porphyry, so it can't be trusted as background
    background (0)— sample beyond EXCLUSION_RADIUS_M of every occurrence

Usage:
    pixi run python scripts/10_build_training_table.py

Outputs:
    data/training_table.parquet
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

# --- config ---
POSITIVE_RADIUS_M = 2_000
EXCLUSION_RADIUS_M = 5_000

# Porphyry Cu-Au deposit type codes (Red Chris is coded L04 primary, L03 secondary)
PORPHYRY_CUAU_CODES = {"L03", "L04"}

root_dir = Path(__file__).resolve().parents[1]
data_dir = root_dir / "data"
geochem_path = data_dir / "geochem_05_features.parquet"
minfile_path = data_dir / "minfile_bc_raw.parquet"
out_path = data_dir / "training_table.parquet"

# --- load ---
logger.info("loading geochem features (province-wide)...")
geochem = gpd.read_parquet(geochem_path).to_crs("EPSG:3005")
logger.info(f"{len(geochem):,} geochem samples")

logger.info("loading MINFILE occurrences (province-wide)...")
minfile = gpd.read_parquet(minfile_path).to_crs("EPSG:3005")

deposit_code_cols = [c for c in minfile.columns if c.startswith("DEPOSIT_TYPE_CODE")]


def is_porphyry_cuau(row):
    return any(
        str(row[c]).strip() in PORPHYRY_CUAU_CODES
        for c in deposit_code_cols
        if pd.notna(row[c])
    )


minfile["is_porphyry_cuau"] = minfile.apply(is_porphyry_cuau, axis=1)
positives = minfile[minfile["is_porphyry_cuau"]]
logger.info(f"{len(minfile):,} occurrences province-wide, {len(positives):,} porphyry Cu-Au positives")

# --- nearest-occurrence distances via spatial join (fast — avoids O(n*m) loop) ---
logger.info("computing nearest-occurrence distances...")
nearest_positive = gpd.sjoin_nearest(
    geochem[["sample_id", "geometry"]], positives[["geometry"]], distance_col="dist_to_positive_m"
).drop_duplicates("sample_id")[["sample_id", "dist_to_positive_m"]]

nearest_any = gpd.sjoin_nearest(
    geochem[["sample_id", "geometry"]], minfile[["geometry"]], distance_col="dist_to_any_occurrence_m"
).drop_duplicates("sample_id")[["sample_id", "dist_to_any_occurrence_m"]]

geochem = geochem.merge(nearest_positive, on="sample_id").merge(nearest_any, on="sample_id")
geochem["dist_to_positive_km"] = geochem["dist_to_positive_m"] / 1_000
geochem["dist_to_any_occurrence_km"] = geochem["dist_to_any_occurrence_m"] / 1_000

# --- labels ---
is_positive = geochem["dist_to_positive_m"] <= POSITIVE_RADIUS_M
is_background = geochem["dist_to_any_occurrence_m"] > EXCLUSION_RADIUS_M

geochem["label"] = np.nan
geochem.loc[is_background, "label"] = 0
geochem.loc[is_positive, "label"] = 1  # positive wins if a sample somehow satisfies both

n_pos = int((geochem["label"] == 1).sum())
n_neg = int((geochem["label"] == 0).sum())
n_excl = int(geochem["label"].isna().sum())
logger.info(f"labels — positive: {n_pos}, background: {n_neg}, excluded (ambiguous): {n_excl}")

# --- save ---
geochem = geochem.drop(columns=["dist_to_positive_m", "dist_to_any_occurrence_m"])
geochem.to_parquet(out_path, index=False)
logger.info(f"saved training table to {out_path} ({geochem.shape[0]:,} rows, {geochem.shape[1]} cols)")

print("\n--- label distribution ---")
print(geochem["label"].value_counts(dropna=False).to_string())
