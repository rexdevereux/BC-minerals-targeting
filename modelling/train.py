"""
train.py — train and evaluate a prospectivity model on the hex grid, tracked in MLflow.

Usage (MLflow server must be running: pixi run mlflow-ui):
    pixi run train
    pixi run train --config configs/<your_variant>.yaml

Evaluation is spatial: hexes are grouped into large H3 blocks and whole blocks
are held out (stratified so the test set contains deposits). The same blocking
is used inside every spatial-CV fold.

Headline metrics (the rest are in evaluation/metrics_detail.json):
    test_pr_auc                        main score — ranking quality with ~1% positives
    test_pr_auc_baseline               the rule-based score the model has to beat
    test_roc_auc
    test_capture_auc                   area under the capture curve (0.5 random, 1 perfect)
    test_capture_5pct_area             share of test deposits in the top 5% of test area
    test_capture_10pct_area
    test_capture_10pct_area_baseline
    cv_pr_auc_mean, cv_pr_auc_std      spatial-CV skill, and how much it swings by region
    train_pr_auc                       in-sample; far above test_pr_auc means overfitting
    test_brier_score                   probability accuracy (lower is better); class balancing
                                       inflates probabilities, so expect it above the reference

Artifacts: config, resolved feature list, split summary, split map, scorecard
(headline metrics on fixed 0-1 axes vs baseline and random), capture curve,
province prospectivity map, permutation importance, per-fold metrics, and the
model (registered as <commodity>_prospectivity).
"""

import argparse

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

import common

HEADLINE_CAPTURE_PCTS = (5, 10)


def fill_missing_scores(scores):
    return np.where(np.isnan(scores), np.nanmin(scores) - 1, scores)


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


def plot_scorecard(test, baseline, folds, train_pr_auc, run_name):
    """Headline metrics on a fixed 0-1 axis, each against the baseline score and a random ranking."""
    rows = [  # label, model value, baseline value, random value, error bar
        ("Test PR-AUC", test["pr_auc"], baseline["pr_auc"], test["positive_rate"], None),
        ("Test ROC-AUC", test["roc_auc"], baseline["roc_auc"], 0.5, None),
        ("Test capture AUC", test["capture_auc"], baseline["capture_auc"], 0.5, None),
        ("Deposits in top 5% of area", test["capture_5pct_area"], baseline["capture_5pct_area"], 0.05, None),
        ("Deposits in top 10% of area", test["capture_10pct_area"], baseline["capture_10pct_area"], 0.10, None),
        ("Spatial-CV PR-AUC (mean ± sd)", folds["pr_auc"].mean(), None, folds["positive_rate"].mean(), folds["pr_auc"].std()),
        ("Train PR-AUC (in-sample)", train_pr_auc, None, test["positive_rate"], None),
        ("Test Brier score (lower = better)", test["brier"], None, test["brier_reference"], None),
    ]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y, [r[1] for r in rows], xerr=[r[4] or 0 for r in rows], color="tab:blue", height=0.6, label="model")
    for yi, (_, value, base, rand, _) in zip(y, rows):
        ax.text(min(value, 0.9) + 0.02, yi, f"{value:.3f}", va="center", fontsize=9)
        if base is not None:
            ax.plot(base, yi, "D", color="black", markersize=6)
        ax.plot(rand, yi, "|", color="grey", markersize=18, markeredgewidth=2)
    ax.plot([], [], "D", color="black", label="rule-based baseline")
    ax.plot([], [], "|", color="grey", markersize=12, markeredgewidth=2, label="random / base-rate reference")
    ax.set(yticks=y, yticklabels=[r[0] for r in rows], xlim=(0, 1), title=f"Scorecard — {run_name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_importance(importance, top=25):
    top_features = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top_features["label"], top_features["mean"], xerr=top_features["std"])
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
capture_pcts = sorted(set(eval_cfg["capture_area_pct"]) | set(HEADLINE_CAPTURE_PCTS))

df = common.load_hex_grid(cfg["data"]["path"])
categorical, numeric = common.feature_columns(df, cfg["features"]["exclude"])
X = common.model_input(df, categorical, numeric)
labels = df[label_col].to_numpy(dtype="float64")
labelled = ~np.isnan(labels)
known = labels == 1

# --- spatial holdout: whole blocks, stratified so test blocks contain deposits ---
blocks = common.spatial_blocks(df, split_cfg["block_h3_resolution"])
test_region = common.spatial_holdout(blocks, labels, split_cfg["test_fraction"], seed)
train_pool = ~test_region
train_idx = np.flatnonzero(train_pool & labelled)

region_name = np.where(test_region, "test", "train")
split_summary = {
    name: {
        "blocks": int(len(np.unique(blocks[mask]))),
        "hexes": int(mask.sum()),
        "labelled_hexes": int((mask & labelled).sum()),
        "positive_hexes": int((mask & known).sum()),
    }
    for name, mask in {"train": train_pool, "test": test_region}.items()
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
    mlflow.log_params(common.flatten({k: v for k, v in cfg.items() if k != "mlflow"}))
    mlflow.log_params({
        "features.n_categorical": len(categorical),
        "features.n_numeric": len(numeric),
        **{f"split.{name}_{k}": v for name, s in split_summary.items() for k, v in s.items()},
    })
    mlflow.log_artifact(str(common.ROOT / args.config), "config")
    mlflow.log_dict({"categorical": categorical, "numeric": numeric}, "config/features.json")
    mlflow.log_dict(split_summary, "evaluation/split_summary.json")
    mlflow.log_input(
        mlflow.data.from_pandas(pd.DataFrame(df.drop(columns="geometry")), source=cfg["data"]["path"],
                                name="hex_grid", targets=label_col),
        context="training",
    )
    log_figure(
        common.plot_split_map(df, region_name, known,
                              f"Spatial split — H3 res-{split_cfg['block_h3_resolution']} blocks"),
        "evaluation/split_map.png",
    )

    # --- spatial CV within the training pool, same blocking as the holdout ---
    fold_metrics = []
    cv_folds = common.spatial_cv_folds(blocks, labels, train_pool, split_cfg["cv_folds"], seed)
    for fold, (fit_idx, val_region) in enumerate(cv_folds):
        model = common.build_model(categorical, numeric, cfg["model"]["params"], seed)
        model.fit(X.iloc[fit_idx], labels[fit_idx].astype(int))
        val_scores = model.predict_proba(X[val_region])[:, 1]
        metrics, _ = common.ranking_metrics(val_scores, labels[val_region], capture_pcts)
        metrics |= common.probability_metrics(val_scores, labels[val_region])
        fold_metrics.append(metrics | {"fold": fold, "fit_hexes": len(fit_idx)})
        print(f"fold {fold}: PR-AUC {metrics['pr_auc']:.3f}, ROC-AUC {metrics['roc_auc']:.3f}")
    folds = pd.DataFrame(fold_metrics).set_index("fold")

    # --- final model on the whole training pool, scored once on the held-out test blocks ---
    model = common.build_model(categorical, numeric, cfg["model"]["params"], seed)
    model.fit(X.iloc[train_idx], labels[train_idx].astype(int))
    test_scores = model.predict_proba(X[test_region])[:, 1]
    test, test_curve = common.ranking_metrics(test_scores, labels[test_region], capture_pcts)
    test |= common.probability_metrics(test_scores, labels[test_region])
    baseline_scores = fill_missing_scores(df.loc[test_region, target["baseline_score"]].to_numpy(dtype="float64"))
    baseline, baseline_curve = common.ranking_metrics(baseline_scores, labels[test_region], capture_pcts)
    train_pr_auc = average_precision_score(
        labels[train_idx].astype(int), model.predict_proba(X.iloc[train_idx])[:, 1]
    )

    headline = {
        "test_pr_auc": test["pr_auc"],
        "test_pr_auc_baseline": baseline["pr_auc"],
        "test_roc_auc": test["roc_auc"],
        "test_capture_auc": test["capture_auc"],
        "test_capture_5pct_area": test["capture_5pct_area"],
        "test_capture_10pct_area": test["capture_10pct_area"],
        "test_capture_10pct_area_baseline": baseline["capture_10pct_area"],
        "cv_pr_auc_mean": folds["pr_auc"].mean(),
        "cv_pr_auc_std": folds["pr_auc"].std(),
        "train_pr_auc": train_pr_auc,
        "test_brier_score": test["brier"],
    }
    mlflow.log_metrics(headline)
    mlflow.log_dict(
        {"test": test, "test_baseline": baseline, "train_pr_auc": train_pr_auc,
         "cv_folds": folds.reset_index().to_dict(orient="records")},
        "evaluation/metrics_detail.json",
    )
    mlflow.log_text(folds.to_csv(), "evaluation/cv_folds.csv")
    log_figure(plot_capture({"model": test_curve, f"baseline ({target['baseline_score']})": baseline_curve}),
               "evaluation/capture_curve.png")
    log_figure(plot_scorecard(test, baseline, folds, train_pr_auc, run.info.run_name), "evaluation/scorecard.png")

    all_scores = model.predict_proba(X)[:, 1]
    percentile = pd.Series(all_scores).rank(pct=True).mul(100).to_numpy()
    log_figure(
        common.plot_prospectivity_map(df, percentile, known, f"{target['commodity']} prospectivity — {run.info.run_name}"),
        "evaluation/prospectivity_map.png",
    )

    test_labelled = test_region & labelled
    result = permutation_importance(
        model, X[test_labelled], labels[test_labelled].astype(int), scoring="average_precision",
        n_repeats=eval_cfg["permutation_repeats"], random_state=seed,
    )
    importance = pd.DataFrame(
        {"label": [common.readable_feature_name(c) for c in X.columns],
         "mean": result.importances_mean, "std": result.importances_std},
        index=pd.Index(X.columns, name="feature"),
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

    print(f"\nrun {run.info.run_id} ({run.info.run_name})")
    print(f"  split: {split_summary['test']['blocks']} test blocks, {split_summary['test']['positive_hexes']} test deposits")
    for name, value in headline.items():
        print(f"  {name:<34} {value:.3f}")
    print(f"  predict with: pixi run predict --model-uri runs:/{run.info.run_id}/model")
