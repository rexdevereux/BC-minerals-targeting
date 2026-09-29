"""
train.py — train and evaluate a prospectivity model on the hex grid, tracked in MLflow.

Usage (MLflow server must be running: pixi run mlflow-ui):
    pixi run train
    pixi run train --config configs/<your_variant>.yaml

Evaluation is spatial: hexes are grouped into large H3 blocks and whole blocks
are held out, so the test set is geographically separate from training. A
random hex split would leak — neighbouring hexes share neighbour-ring features,
gap-filled geochemistry, and the same mineral districts.

Logged per run:
    tags       commodity, deposit codes, model type, git commit/branch/dirty,
               data + lockfile hashes (uncommitted diff saved as an artifact)
    params     the whole config, flattened, plus split and feature counts
    metrics    spatial-CV (per fold + mean/std) and held-out test: PR-AUC,
               ROC-AUC, and share of deposits captured in the top 1/5/10/20%
               of area — each also for the rule-based baseline score
    artifacts  config, resolved feature list, split summary, capture curve,
               split map, permutation importance
    model      sklearn pipeline, registered as <commodity>_prospectivity
"""

import argparse
import math

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import common

MAX_CATEGORIES = 250  # HistGradientBoosting accepts at most 255 categories per feature


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


def evaluate(scores, labels, capture_pcts, prefix):
    """Ranking metrics on labelled hexes; capture-by-area over every hex in the region."""
    labelled = ~np.isnan(labels)
    y, s = labels[labelled].astype(int), scores[labelled]
    metrics = {
        f"{prefix}pr_auc": average_precision_score(y, s),
        f"{prefix}roc_auc": roc_auc_score(y, s),
        f"{prefix}positive_rate": y.mean(),
    }
    metrics[f"{prefix}pr_auc_lift"] = metrics[f"{prefix}pr_auc"] / metrics[f"{prefix}positive_rate"]

    ranked_positive = labels[np.argsort(-scores, kind="stable")] == 1
    capture_curve = np.cumsum(ranked_positive) / ranked_positive.sum()
    for pct in capture_pcts:
        metrics[f"{prefix}capture_at_{pct}pct_area"] = capture_curve[math.ceil(pct / 100 * len(scores)) - 1]
    return metrics, capture_curve


def fill_missing_scores(scores):
    return np.where(np.isnan(scores), np.nanmin(scores) - 1, scores)


def flatten(d, parent=""):
    out = {}
    for key, value in d.items():
        name = f"{parent}.{key}" if parent else key
        if isinstance(value, dict):
            out |= flatten(value, name)
        else:
            out[name] = ",".join(map(str, value)) if isinstance(value, list) else value
    return out


def plot_capture(curves):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, curve in curves.items():
        ax.plot(np.linspace(0, 100, len(curve)), curve * 100, label=label)
    ax.plot([0, 100], [0, 100], "k--", lw=0.8, label="random")
    ax.set(xlabel="% of test area explored (ranked by score)", ylabel="% of known deposits captured",
           title="Held-out test blocks — capture curve", xlim=(0, 100), ylim=(0, 100))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_split(df, test_region, label_col):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["centroid_lon"], df["centroid_lat"], c=np.where(test_region, "tab:orange", "lightgrey"),
               s=0.2, marker=",", linewidths=0)
    positives = df[label_col] == 1
    ax.scatter(df.loc[positives, "centroid_lon"], df.loc[positives, "centroid_lat"], c="black", s=2)
    ax.set(title="Spatial split — orange: held-out test blocks, black: known deposits",
           xlabel="longitude", ylabel="latitude")
    ax.set_aspect(1.6)
    fig.tight_layout()
    return fig


def plot_importance(importance, top=25):
    top_features = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top_features.index, top_features["mean"], xerr=top_features["std"])
    ax.set(xlabel="drop in test PR-AUC when shuffled", title=f"Permutation importance (top {top})")
    fig.tight_layout()
    return fig


def log_figure(fig, artifact_path):
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/porphyry_cuau.yaml")
args = parser.parse_args()

cfg = common.load_config(args.config)
target, split_cfg, eval_cfg = cfg["target"], cfg["split"], cfg["evaluation"]
label_col, seed = target["label_column"], split_cfg["seed"]

df = common.load_hex_grid(cfg["data"]["path"])
categorical, numeric = common.feature_columns(df, cfg["features"]["exclude"])
X = common.model_input(df, categorical, numeric)
labels = df[label_col].to_numpy(dtype="float64")
labelled = ~np.isnan(labels)

# --- spatial split: hold out whole blocks, stratified so test blocks contain deposits ---
blocks = df["h3_cell"].map(lambda c: h3.cell_to_parent(c, split_cfg["block_h3_resolution"])).to_numpy()
holdout = StratifiedGroupKFold(n_splits=round(1 / split_cfg["test_fraction"]), shuffle=True, random_state=seed)
labelled_idx = np.flatnonzero(labelled)
_, test_pos = next(holdout.split(labelled_idx, labels[labelled_idx], groups=blocks[labelled_idx]))
test_region = np.isin(blocks, np.unique(blocks[labelled_idx[test_pos]]))
train_idx = np.flatnonzero(~test_region & labelled)

split_summary = {
    region: {
        "blocks": int(len(np.unique(blocks[mask]))),
        "hexes": int(mask.sum()),
        "labelled_hexes": int((mask & labelled).sum()),
        "positive_hexes": int((labels[mask] == 1).sum()),
    }
    for region, mask in {"train": ~test_region, "test": test_region}.items()
}

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment"])

with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]) as run:
    tags, uncommitted_diff = common.provenance_tags(cfg["data"]["path"])
    mlflow.set_tags(tags | {
        "stage": "train",
        "commodity": target["commodity"],
        "deposit_type_codes": ",".join(target["deposit_type_codes"]),
        "label_column": label_col,
        "model_type": cfg["model"]["type"],
        "config_file": args.config,
    })
    if uncommitted_diff:
        mlflow.log_text(uncommitted_diff, "provenance/uncommitted.diff")
    mlflow.log_params(flatten({k: v for k, v in cfg.items() if k != "mlflow"}))
    mlflow.log_params({
        "features.n_categorical": len(categorical),
        "features.n_numeric": len(numeric),
        **{f"split.{region}_{k}": v for region, s in split_summary.items() for k, v in s.items()},
    })
    mlflow.log_artifact(str(common.ROOT / args.config), "config")
    mlflow.log_dict({"categorical": categorical, "numeric": numeric}, "config/features.json")
    mlflow.log_dict(split_summary, "evaluation/split_summary.json")
    mlflow.log_input(
        mlflow.data.from_pandas(pd.DataFrame(df.drop(columns="geometry")), source=cfg["data"]["path"],
                                name="hex_grid", targets=label_col),
        context="training",
    )

    # --- spatial cross-validation within the training blocks ---
    cv = StratifiedGroupKFold(n_splits=split_cfg["cv_folds"], shuffle=True, random_state=seed)
    fold_metrics = []
    for fold, (fit_pos, val_pos) in enumerate(cv.split(train_idx, labels[train_idx], groups=blocks[train_idx])):
        val_region = ~test_region & np.isin(blocks, np.unique(blocks[train_idx[val_pos]]))
        fit_idx = train_idx[fit_pos]
        model = build_model(categorical, numeric, cfg["model"]["params"], seed)
        model.fit(X.iloc[fit_idx], labels[fit_idx].astype(int))
        scores = model.predict_proba(X[val_region])[:, 1]
        metrics, _ = evaluate(scores, labels[val_region], eval_cfg["capture_area_pct"], "cv_")
        mlflow.log_metrics(metrics, step=fold)
        fold_metrics.append(metrics)
        print(f"fold {fold}: PR-AUC {metrics['cv_pr_auc']:.3f}, ROC-AUC {metrics['cv_roc_auc']:.3f}")
    fold_frame = pd.DataFrame(fold_metrics)
    mlflow.log_metrics({f"{k}_mean": v for k, v in fold_frame.mean().items()})
    mlflow.log_metrics({f"{k}_std": v for k, v in fold_frame.std().items()})

    # --- final model on all training blocks, scored once on the held-out test blocks ---
    model = build_model(categorical, numeric, cfg["model"]["params"], seed)
    model.fit(X.iloc[train_idx], labels[train_idx].astype(int))
    test_scores = model.predict_proba(X[test_region])[:, 1]
    test_metrics, test_curve = evaluate(test_scores, labels[test_region], eval_cfg["capture_area_pct"], "test_")
    baseline = fill_missing_scores(df.loc[test_region, target["baseline_score"]].to_numpy(dtype="float64"))
    baseline_metrics, baseline_curve = evaluate(baseline, labels[test_region], eval_cfg["capture_area_pct"], "test_baseline_")
    mlflow.log_metrics(test_metrics | baseline_metrics)

    log_figure(plot_capture({"model": test_curve, f"baseline ({target['baseline_score']})": baseline_curve}), "evaluation/capture_curve.png")
    log_figure(plot_split(df, test_region, label_col), "evaluation/split_map.png")

    test_labelled = test_region & labelled
    result = permutation_importance(
        model, X[test_labelled], labels[test_labelled].astype(int), scoring="average_precision",
        n_repeats=eval_cfg["permutation_repeats"], random_state=seed,
    )
    importance = pd.DataFrame(
        {"mean": result.importances_mean, "std": result.importances_std}, index=X.columns
    ).sort_values("mean", ascending=False)
    mlflow.log_text(importance.to_csv(), "evaluation/permutation_importance.csv")
    log_figure(plot_importance(importance), "evaluation/permutation_importance.png")

    sample = X.iloc[train_idx[:2000]]
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(sample, model.predict_proba(sample)[:, 1]),
        registered_model_name=f"{target['commodity']}_prospectivity",
        # skops refuses types it can't vet; these come from our own HistGradientBoosting pipeline
        skops_trusted_types=[
            "functools.partial",
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            "sklearn.utils.validation.check_array",
        ],
    )

    print(f"\nrun {run.info.run_id}")
    print(f"  spatial CV  PR-AUC {fold_frame['cv_pr_auc'].mean():.3f} ± {fold_frame['cv_pr_auc'].std():.3f}")
    print(f"  test        PR-AUC {test_metrics['test_pr_auc']:.3f} (baseline {baseline_metrics['test_baseline_pr_auc']:.3f}, "
          f"random {test_metrics['test_positive_rate']:.3f})")
    for pct in eval_cfg["capture_area_pct"]:
        print(f"  top {pct:>2}% of test area captures {test_metrics[f'test_capture_at_{pct}pct_area']:.0%} of deposits "
              f"(baseline {baseline_metrics[f'test_baseline_capture_at_{pct}pct_area']:.0%})")
    print(f"  predict with: pixi run predict --model-uri runs:/{run.info.run_id}/model")
