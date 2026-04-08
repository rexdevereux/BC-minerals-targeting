import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

# --- paths ---
data_dir  = Path("data")
in_path   = data_dir / "geochem_04_spatial.parquet"
out_path  = data_dir / "geochem_05_features.parquet"
meta_path = Path("outputs") / "geochem_metadata.yaml"

# --- load ---
logger.info(f"loading {in_path}...")
gdf = gpd.read_parquet(in_path)
logger.info(f"loaded {len(gdf):,} rows x {len(gdf.columns)} columns")

element_cols = [c for c in gdf.columns if c.endswith("_ppm")]
print(f"\nelement columns: {element_cols}")

# --- log transforms ---
# geochemical concentrations are log-normally distributed
# log1p(x) = log(x + 1) — handles zeros safely
# compresses 5 orders of magnitude into a manageable range
logger.info("applying log1p transforms...")
for col in element_cols:
    out_col       = col.replace("_ppm", "_log")
    gdf[out_col]  = np.log1p(gdf[col].clip(lower=0))

log_cols = [c for c in gdf.columns if c.endswith("_log")]
logger.info(f"created {len(log_cols)} log columns")

# show effect on copper
cu_raw = gdf["copper_ppm"].dropna()
cu_log = gdf["copper_log"].dropna()
print(f"\n--- log transform effect (copper) ---")
print(f"  raw: min={cu_raw.min():.2f}  max={cu_raw.max():.0f}  std={cu_raw.std():.1f}")
print(f"  log: min={cu_log.min():.2f}  max={cu_log.max():.2f}  std={cu_log.std():.2f}")

# --- global z-scores ---
# measures how many std deviations above/below the province-wide mean
# z > 2 = notable anomaly, z > 3 = strong, z > 4 = exceptional
logger.info("computing global z-scores...")
for col in log_cols:
    out_col      = col.replace("_log", "_zscore")
    mu           = gdf[col].mean()
    sigma        = gdf[col].std()
    gdf[out_col] = (gdf[col] - mu) / sigma if sigma > 0 else 0.0

zscore_cols = [c for c in gdf.columns if c.endswith("_zscore")]
logger.info(f"created {len(zscore_cols)} global z-score columns")

cu_z = gdf["copper_zscore"].dropna()
print(f"\n--- copper anomaly distribution ---")
print(f"  z > 2 (notable):     {(cu_z > 2).sum():>6,}")
print(f"  z > 3 (strong):      {(cu_z > 3).sum():>6,}")
print(f"  z > 4 (exceptional): {(cu_z > 4).sum():>6,}")

# --- local z-scores stratified by rock class ---
# controls for lithological background variation
# e.g. 100ppm Cu in sedimentary rock is more anomalous than in intrusive rock
# uses real rock_class from BCGS bedrock geology — better than longitude bands
logger.info("computing rock-class stratified z-scores...")
for col in log_cols:
    out_col            = col.replace("_log", "_zscore_local")
    group_mean         = gdf.groupby("rock_class")[col].transform("mean")
    group_std          = gdf.groupby("rock_class")[col].transform("std").fillna(1)
    gdf[out_col]       = (gdf[col] - group_mean) / group_std

local_zscore_cols = [c for c in gdf.columns if c.endswith("_zscore_local")]
logger.info(f"created {len(local_zscore_cols)} local z-score columns")

# --- pathfinder element ratios ---
# diagnostic of specific deposit types based on economic geology research
ratio_pairs = [
    # porphyry indicators
    ("copper_ppm",  "molybdenum_ppm", "ratio_cu_mo", "porphyry Cu-Mo indicator"),
    ("copper_ppm",  "gold_ppm",       "ratio_cu_au", "porphyry Cu-Au indicator"),
    ("arsenic_ppm", "gold_ppm",       "ratio_as_au", "epithermal gold pathfinder"),
    # battery metals indicators
    ("cobalt_ppm",  "nickel_ppm",     "ratio_co_ni", "magmatic Ni-Co indicator"),
    ("copper_ppm",  "zinc_ppm",       "ratio_cu_zn", "VMS deposit indicator"),
    ("lithium_ppm", "manganese_ppm",  "ratio_li_mn", "Li pegmatite indicator"),
]

logger.info("computing pathfinder element ratios...")
for elem_a, elem_b, out_col, desc in ratio_pairs:
    if elem_a in gdf.columns and elem_b in gdf.columns:
        gdf[out_col] = gdf[elem_a] / gdf[elem_b].replace(0, np.nan)
        n_valid      = gdf[out_col].notna().sum()
        print(f"  {out_col:<20} {desc:<40} n={n_valid:,}")

# --- grid cell aggregates ---
# smooths individual sample noise into regional signals
# high cell mean = sustained anomaly across multiple samples = stronger signal
logger.info("computing grid cell aggregates...")
agg_cols = ["copper_ppm", "nickel_ppm", "cobalt_ppm",
            "gold_ppm", "molybdenum_ppm", "lithium_ppm"]

for col in agg_cols:
    if col not in gdf.columns:
        continue
    base = col.replace("_ppm", "")
    gdf[f"{base}_cell_mean"] = gdf.groupby("grid_id")[col].transform("mean")
    gdf[f"{base}_cell_max"]  = gdf.groupby("grid_id")[col].transform("max")
    gdf[f"{base}_cell_std"]  = gdf.groupby("grid_id")[col].transform("std").fillna(0)

cell_cols = [c for c in gdf.columns if "_cell_" in c]
logger.info(f"created {len(cell_cols)} grid aggregate columns")

# --- score 1: porphyry cu-au-mo ---
# targets large disseminated Cu-Au-Mo systems (highland valley, mount polley style)
# weights based on deposit-type importance for BC porphyry systems
# fillna(0) treats nulls as background (z=0) — missing data doesn't penalise samples
porphyry_weights = {
    "copper_zscore":     0.35,
    "gold_zscore":       0.25,
    "molybdenum_zscore": 0.20,
    "arsenic_zscore":    0.10,  # pathfinder for porphyry Au
    "cobalt_zscore":     0.10,  # common porphyry byproduct
}

logger.info("computing porphyry Cu-Au-Mo score...")
porphyry_score = sum(
    gdf[col].fillna(0) * weight
    for col, weight in porphyry_weights.items()
    if col in gdf.columns
)
gdf["score_porphyry"] = porphyry_score
gdf["rank_porphyry"]  = gdf["score_porphyry"].rank(ascending=False, method="min").astype(int)

# top 2% flagged as priority targets
threshold_p              = gdf["score_porphyry"].quantile(0.98)
gdf["target_porphyry"]   = gdf["score_porphyry"] >= threshold_p
n_porphyry               = gdf["target_porphyry"].sum()
logger.info(f"porphyry targets (top 2%): {n_porphyry:,}")

# --- score 2: battery metals (li-co-ni) ---
# targets li pegmatites, magmatic ni-co, sediment-hosted co
# lithium weighted highest — BC has active Li pegmatite exploration
# fillna(0) same rationale as above
battery_weights = {
    "lithium_zscore": 0.45,  # primary target — Li pegmatites
    "cobalt_zscore":  0.30,  # Co-Ni magmatic/sedimentary
    "nickel_zscore":  0.25,  # Ni sulphide / laterite
}

logger.info("computing battery metals score...")
battery_score = sum(
    gdf[col].fillna(0) * weight
    for col, weight in battery_weights.items()
    if col in gdf.columns
)
gdf["score_battery"]   = battery_score
gdf["rank_battery"]    = gdf["score_battery"].rank(ascending=False, method="min").astype(int)

threshold_b            = gdf["score_battery"].quantile(0.98)
gdf["target_battery"]  = gdf["score_battery"] >= threshold_b
n_battery              = gdf["target_battery"].sum()
logger.info(f"battery metals targets (top 2%): {n_battery:,}")

# --- results summary ---
print(f"\n--- porphyry cu-au-mo score ---")
print(f"  range:   {gdf['score_porphyry'].min():.3f} – {gdf['score_porphyry'].max():.3f}")
print(f"  median:  {gdf['score_porphyry'].median():.3f}")
print(f"  targets: {n_porphyry:,} samples (top 2%)")

print(f"\n--- battery metals score ---")
print(f"  range:   {gdf['score_battery'].min():.3f} – {gdf['score_battery'].max():.3f}")
print(f"  median:  {gdf['score_battery'].median():.3f}")
print(f"  targets: {n_battery:,} samples (top 2%)")

print(f"\n--- top 10 porphyry targets ---")
top_porphyry = gdf.nsmallest(10, "rank_porphyry")[
    ["sample_id", "latitude", "longitude", "rock_class", "terrane_name",
     "copper_ppm", "gold_ppm", "molybdenum_ppm",
     "score_porphyry", "rank_porphyry"]
].round(3)
print(top_porphyry.to_string())

print(f"\n--- top 10 battery metals targets ---")
top_battery = gdf.nsmallest(10, "rank_battery")[
    ["sample_id", "latitude", "longitude", "rock_class", "terrane_name",
     "lithium_ppm", "cobalt_ppm", "nickel_ppm",
     "score_battery", "rank_battery"]
].round(3)
print(top_battery.to_string())

# --- feature inventory ---
feature_categories = {
    "raw concentrations":      [c for c in gdf.columns if c.endswith("_ppm")],
    "log transforms":          [c for c in gdf.columns if c.endswith("_log")],
    "global z-scores":         [c for c in gdf.columns if c.endswith("_zscore") and "local" not in c],
    "local z-scores":          [c for c in gdf.columns if c.endswith("_zscore_local")],
    "pathfinder ratios":       [c for c in gdf.columns if c.startswith("ratio_")],
    "grid aggregates":         [c for c in gdf.columns if "_cell_" in c],
    "spatial features":        ["grid_id", "rock_class", "terrane_name",
                                 "dist_to_fault_km", "dist_to_terrane_boundary_km"],
    "targeting outputs":       ["score_porphyry", "rank_porphyry", "target_porphyry",
                                 "score_battery",  "rank_battery",  "target_battery"],
}

total_features = sum(
    len([c for c in cols if c in gdf.columns])
    for cols in feature_categories.values()
)

print(f"\n--- feature summary ---")
for category, cols in feature_categories.items():
    existing = [c for c in cols if c in gdf.columns]
    print(f"  {category:<25} {len(existing):>3} features")
print(f"  {'total':<25} {total_features:>3} features")

# --- save parquet ---
logger.info(f"saving to {out_path}...")
gdf.to_parquet(out_path, index=False)
logger.info(f"saved — {len(gdf):,} rows x {len(gdf.columns)} columns")

# --- save metadata yaml ---
metadata = {
    "schema_version":   "1.0",
    "generated_at":     datetime.now(timezone.utc).isoformat(),
    "pipeline_version": "1.0.0",
    "source":           "BC RGS 2020 — BCGS GeoFile 2020-08",
    "total_rows":       len(gdf),
    "total_features":   int(total_features),
    "porphyry_targets": int(n_porphyry),
    "battery_targets":  int(n_battery),
    "feature_groups":   {
        k: [c for c in v if c in gdf.columns]
        for k, v in feature_categories.items()
    }
}
with open(meta_path, "w") as f:
    yaml.dump(metadata, f, default_flow_style=False)
logger.info(f"metadata saved to {meta_path}")

print(f"\n--- done --- geochem_05_features.parquet ready")