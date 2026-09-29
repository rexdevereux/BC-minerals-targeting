"""
tune.py — Optuna hyperparameter search scored by spatial CV, tracked in MLflow.

Usage (MLflow server must be running: pixi run mlflow-ui):
    pixi run tune
    pixi run tune --config configs/<tuning_config>.yaml

Uses exactly train.py's split: the held-out test blocks are removed first and
never touched, so the final training run's test score stays honest. Every
trial is scored on the same spatial-CV folds train.py uses.

MLflow layout: one parent run (search space, best settings, optimisation
history, parameter importance, trials table) with a nested child run per trial
(settings + per-fold CV metrics; pruned trials tagged). The best settings are
written to the config's `output_config` — train that for the final model.
"""

import argparse
import copy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import yaml

import common

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/porphyry_cuau_tuning.yaml")
args = parser.parse_args()

tuning = common.load_config(args.config)
cfg = common.load_config(tuning["base_config"])
study_cfg, space = tuning["study"], tuning["search_space"]
target, split_cfg = cfg["target"], cfg["split"]
label_col, seed, objective_metric = target["label_column"], split_cfg["seed"], study_cfg["objective"]

df = common.load_hex_grid(cfg["data"]["path"])
categorical, numeric = common.feature_columns(df, cfg["features"]["exclude"])
X = common.model_input(df, categorical, numeric)
labels = df[label_col].to_numpy(dtype="float64")

blocks = common.spatial_blocks(df, split_cfg["block_h3_resolution"])
test_region = common.spatial_holdout(blocks, labels, split_cfg["test_fraction"], seed)
folds = list(common.spatial_cv_folds(blocks, labels, ~test_region, split_cfg["cv_folds"], seed))


def suggest(trial):
    params = {}
    for name, spec in space.items():
        options = {k: spec[k] for k in ("step", "log") if k in spec}
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], **options)
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], **options)
    return params


def objective(trial):
    params = cfg["model"]["params"] | suggest(trial)
    mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True)
    mlflow.set_tags({"stage": "tune-trial", "commodity": target["commodity"], "optuna_trial": trial.number})
    mlflow.log_params(params)
    fold_scores = []
    try:
        for fold, (fit_idx, val_region) in enumerate(folds):
            model = common.build_model(categorical, numeric, params, seed)
            model.fit(X.iloc[fit_idx], labels[fit_idx].astype(int))
            metrics, _ = common.ranking_metrics(model.predict_proba(X[val_region])[:, 1], labels[val_region], [10])
            fold_scores.append(metrics)
            mlflow.log_metrics({f"cv_{objective_metric}": metrics[objective_metric],
                                "cv_capture_10pct_area": metrics["capture_10pct_area"]}, step=fold)
            trial.report(np.mean([s[objective_metric] for s in fold_scores]), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    except optuna.TrialPruned:
        mlflow.set_tag("optuna_state", "pruned")
        mlflow.end_run(status="KILLED")
        raise
    except Exception:
        mlflow.end_run(status="FAILED")
        raise

    scores = np.array([s[objective_metric] for s in fold_scores])
    mlflow.log_metrics({
        f"cv_{objective_metric}_mean": scores.mean(),
        f"cv_{objective_metric}_std": scores.std(ddof=1),
        "cv_capture_10pct_area_mean": np.mean([s["capture_10pct_area"] for s in fold_scores]),
    })
    mlflow.set_tag("optuna_state", "complete")
    mlflow.end_run()
    return scores.mean()


def log_axes(ax, artifact_path):
    fig = ax.figure
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment"])

with mlflow.start_run(run_name=tuning["mlflow"]["run_name"]) as parent:
    tags, uncommitted_diff = common.provenance_tags(cfg["data"]["path"])
    mlflow.set_tags(tags | {
        "stage": "tune",
        "commodity": target["commodity"],
        "label_column": label_col,
        "model_type": cfg["model"]["type"],
        "config_file": args.config,
        "base_config": tuning["base_config"],
    })
    if uncommitted_diff:
        mlflow.log_text(uncommitted_diff, "provenance/uncommitted.diff")
    mlflow.log_params(common.flatten({"study": study_cfg, "search_space": space, "split": split_cfg}))
    mlflow.log_artifact(str(common.ROOT / args.config), "config")
    mlflow.log_artifact(str(common.ROOT / tuning["base_config"]), "config")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=study_cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=study_cfg["pruning_startup_trials"], n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=study_cfg["n_trials"], timeout=study_cfg["timeout_minutes"] * 60)

    states = [t.state for t in study.trials]
    best = study.best_trial
    mlflow.log_metrics({
        f"best_cv_{objective_metric}_mean": best.value,
        "n_trials_complete": states.count(optuna.trial.TrialState.COMPLETE),
        "n_trials_pruned": states.count(optuna.trial.TrialState.PRUNED),
    })
    mlflow.log_params({f"best.{k}": v for k, v in best.params.items()} | {"best.trial": best.number})
    mlflow.log_text(study.trials_dataframe().to_csv(index=False), "tuning/trials.csv")
    log_axes(optuna.visualization.matplotlib.plot_optimization_history(study), "tuning/optimisation_history.png")
    log_axes(optuna.visualization.matplotlib.plot_param_importances(study), "tuning/param_importance.png")

    tuned = copy.deepcopy(cfg)
    tuned["model"]["params"] = cfg["model"]["params"] | best.params
    tuned["mlflow"]["run_name"] = f"{cfg['mlflow']['run_name']}-tuned"
    tuned["provenance"] = {"tuning_run_id": parent.info.run_id, "tuning_config": args.config,
                           "base_config": tuning["base_config"]}
    out_path = common.ROOT / tuning["output_config"]
    header = (f"# Generated by modelling/tune.py — best of {len(study.trials)} trials "
              f"(MLflow run {parent.info.run_id}).\n# Edit {tuning['base_config']} instead; re-run tuning to refresh.\n\n")
    out_path.write_text(header + yaml.safe_dump(tuned, sort_keys=False))
    mlflow.log_artifact(str(out_path), "config")

print(f"\ntuning run {parent.info.run_id}")
print(f"  trials: {states.count(optuna.trial.TrialState.COMPLETE)} complete, "
      f"{states.count(optuna.trial.TrialState.PRUNED)} pruned")
print(f"  best cv_{objective_metric}_mean {best.value:.3f} (trial {best.number})")
for k, v in best.params.items():
    print(f"    {k}: {v}")
print(f"  train the tuned model: pixi run train --config {tuning['output_config']}")
