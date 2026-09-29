"""Shared helpers for train.py and predict.py: config, data, features, provenance."""

import hashlib
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


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
