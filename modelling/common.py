"""Shared helpers for train.py and predict.py: config, data, features, provenance."""

import hashlib
import subprocess
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
MAX_CATEGORIES = 250  # HistGradientBoosting accepts at most 255 categories per feature


def _deposit_points(hexes, known, ax, colour):
    hexes[known].centroid.plot(ax=ax, color=colour, markersize=1.5)
    return Line2D([], [], marker="o", linestyle="", color=colour, markersize=4, label="known deposits")


def plot_prospectivity_map(hexes, percentile, known, title):
    """Hex polygons (not centroid dots, which leave moiré gaps) coloured by percentile."""
    fig, ax = plt.subplots(figsize=(9, 8))
    hexes.assign(percentile=percentile).plot(
        column="percentile", cmap="magma", linewidth=0, antialiased=False, ax=ax, rasterized=True,
        legend=True, legend_kwds={"shrink": 0.6, "label": "prospectivity percentile"},
    )
    ax.legend(handles=[_deposit_points(hexes, known, ax, "cyan")], loc="lower left")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_split_map(hexes, region, known, title):
    colours = {"train": "#d9d9d9", "test": "#fd8d3c"}
    fig, ax = plt.subplots(figsize=(9, 8))
    hexes.plot(color=[colours[r] for r in region], linewidth=0, antialiased=False, ax=ax, rasterized=True)
    handles = [Patch(color=c, label=name) for name, c in colours.items()]
    handles.append(_deposit_points(hexes, known, ax, "black"))
    ax.legend(handles=handles, loc="lower left")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


_LAYER_NAMES = {
    "elevation": "elevation",
    "topo_position": "topographic position (TPI)",
    "mag_field": "magnetic field (RTF)",
    "mag_vertical_deriv": "magnetic vertical derivative (1VD)",
    "mag_analytic_signal": "magnetic edges (analytic signal)",
    "mag_tilt_deriv": "magnetic tilt derivative",
}
_STAT_NAMES = {"ring1_mean": "mean incl. neighbours", "mean": "mean in hex", "std": "variability in hex"}
_FIXED_NAMES = {
    "dist_to_fault_km": "distance to fault (km)",
    "dist_to_terrane_boundary_km": "distance to terrane boundary (km)",
    "mag_coverage": "magnetic data coverage",
    "rock_class": "rock class",
    "era": "geological era",
    "terrane_name": "terrane",
    "terrane_group": "terrane group",
    "tectonic_setting": "tectonic setting",
    "tim_surficial_material": "surficial material (TIM)",
    "geochem_source": "geochem sample source",
}


def readable_feature_name(column):
    """Display label for a hex_grid column; the column names themselves are what models are trained on."""
    if column in _FIXED_NAMES:
        return _FIXED_NAMES[column]
    for layer, label in _LAYER_NAMES.items():
        stat = column.removeprefix(f"{layer}_")
        if stat != column and stat in _STAT_NAMES:
            return f"{label} — {_STAT_NAMES[stat]}"
    if column.startswith("landform_") and column.endswith("_frac"):
        return f"landform: {column.removeprefix('landform_').removesuffix('_frac').replace('_', ' ')} (share of hex)"
    if column.endswith("_log_mean"):
        return f"{column.removesuffix('_log_mean')} concentration (log, mean)"
    if column.endswith("_zlocal_max"):
        return f"{column.removesuffix('_zlocal_max')} anomaly vs local rock (max)"
    return column.replace("_", " ")


def load_config(path):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def load_hex_grid(path):
    return gpd.read_parquet(ROOT / path)


def feature_columns(df, exclude):
    """Split every non-excluded hex_grid column into (categorical, numeric)."""
    candidates = [c for c in df.columns if c not in set(exclude) and c != "geometry"]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numeric]
    return categorical, numeric


def model_input(df, categorical, numeric):
    X = df[categorical + numeric].copy()
    X[categorical] = X[categorical].astype(object)
    return X


def sha256(path):
    digest = hashlib.sha256()
    with open(ROOT / path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "diff": f"# git status --porcelain\n{status}\n\n# git diff HEAD\n{git('diff', 'HEAD')}" if status else "",
    }


def provenance_tags(data_path):
    git = git_info()
    tags = {
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": str(git["dirty"]).lower(),
        "data_path": str(data_path),
        "data_sha256": sha256(data_path)[:16],
        "pixi_lock_sha256": sha256("pixi.lock")[:16],
    }
    return tags, git["diff"]


def flatten(d, parent=""):
    out = {}
    for key, value in d.items():
        name = f"{parent}.{key}" if parent else key
        if isinstance(value, dict):
            out |= flatten(value, name)
        else:
            out[name] = ",".join(map(str, value)) if isinstance(value, list) else value
    return out


def build_model(categorical, numeric, params, seed):
    encode = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    max_categories=MAX_CATEGORIES,
                ),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ]
    )
    classifier = HistGradientBoostingClassifier(
        categorical_features=list(range(len(categorical))), random_state=seed, **params
    )
    return Pipeline([("encode", encode), ("classifier", classifier)])


def ranking_metrics(scores, labels, capture_pcts):
    """PR/ROC on labelled hexes; capture-by-area over every hex in the region."""
    labelled = ~np.isnan(labels)
    y, s = labels[labelled].astype(int), scores[labelled]
    ranked_positive = labels[np.argsort(-scores, kind="stable")] == 1
    curve = np.cumsum(ranked_positive) / ranked_positive.sum()
    metrics = {
        "pr_auc": average_precision_score(y, s),
        "roc_auc": roc_auc_score(y, s),
        "capture_auc": curve.mean(),
        "positive_rate": y.mean(),
    }
    for pct in capture_pcts:
        metrics[f"capture_{pct}pct_area"] = curve[int(np.ceil(pct / 100 * len(scores))) - 1]
    return metrics, curve


def probability_metrics(probabilities, labels):
    """Brier score on labelled hexes, plus the score of always predicting the base rate."""
    labelled = ~np.isnan(labels)
    y = labels[labelled].astype(int)
    return {
        "brier": brier_score_loss(y, probabilities[labelled]),
        "brier_reference": y.mean() * (1 - y.mean()),
    }


def spatial_blocks(df, resolution):
    return np.array([h3.cell_to_parent(c, resolution) for c in df["h3_cell"]])


def spatial_holdout(blocks, labels, test_fraction, seed):
    """Test-region mask: whole blocks, stratified so the held-out blocks contain deposits."""
    labelled_idx = np.flatnonzero(~np.isnan(labels))
    splitter = StratifiedGroupKFold(n_splits=round(1 / test_fraction), shuffle=True, random_state=seed)
    _, test_pos = next(splitter.split(labelled_idx, labels[labelled_idx], groups=blocks[labelled_idx]))
    return np.isin(blocks, np.unique(blocks[labelled_idx[test_pos]]))


def spatial_cv_folds(blocks, labels, train_pool, n_folds, seed):
    """Yield (fit_idx, val_region) per spatial-CV fold, using only the training pool."""
    labelled = ~np.isnan(labels)
    train_idx = np.flatnonzero(train_pool & labelled)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for _, val_pos in splitter.split(train_idx, labels[train_idx], groups=blocks[train_idx]):
        val_region = train_pool & np.isin(blocks, np.unique(blocks[train_idx[val_pos]]))
        yield np.flatnonzero(train_pool & labelled & ~val_region), val_region
