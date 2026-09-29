"""
predict.py — score every hex with a model logged by train.py, tracked as its own MLflow run.

Usage (MLflow server must be running: pixi run mlflow-ui):
    pixi run predict --model-uri runs:/<run_id>/model
    pixi run predict --model-uri models:/porphyry_cuau_prospectivity/1
    pixi run predict --model-uri models:/porphyry_cuau_prospectivity@champion

The prediction run is logged in the same experiment as the model it used,
tagged with that model's run id, so every map traces back to the exact
training run, config, code commit, and data it came from.

Outputs:
    outputs/predictions/<commodity>_<model run id>.gpkg   hex polygons + probability + percentile, for QGIS
    MLflow: predictions parquet + prospectivity map PNG
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

import common

parser = argparse.ArgumentParser()
parser.add_argument("--model-uri", required=True, help="runs:/<run_id>/model, models:/<name>/<version> or models:/<name>@<alias>")
parser.add_argument("--data", default="data/hex_grid.parquet")
parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
args = parser.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)
model_run_id = mlflow.models.get_model_info(args.model_uri).run_id
model_run = mlflow.get_run(model_run_id)
commodity = model_run.data.tags["commodity"]
label_col = model_run.data.tags["label_column"]
with open(mlflow.artifacts.download_artifacts(run_id=model_run_id, artifact_path="config/features.json")) as f:
    features = json.load(f)

df = common.load_hex_grid(args.data)
X = common.model_input(df, features["categorical"], features["numeric"])
model = mlflow.sklearn.load_model(args.model_uri)
probability = model.predict_proba(X)[:, 1]

predictions = df[["h3_cell", "centroid_lat", "centroid_lon", label_col, "geometry"]].copy()
predictions["probability"] = probability
predictions["percentile"] = pd.Series(probability).rank(pct=True).mul(100).to_numpy()

out_dir = common.ROOT / "outputs" / "predictions"
out_dir.mkdir(parents=True, exist_ok=True)
gpkg_path = out_dir / f"{commodity}_{model_run_id[:8]}.gpkg"
predictions.to_file(gpkg_path, driver="GPKG")

known = predictions[label_col] == 1
top10 = predictions["percentile"] >= 90

fig, ax = plt.subplots(figsize=(8, 8))
points = ax.scatter(predictions["centroid_lon"], predictions["centroid_lat"], c=predictions["percentile"],
                    cmap="magma", s=0.3, marker=",", linewidths=0)
ax.scatter(predictions.loc[known, "centroid_lon"], predictions.loc[known, "centroid_lat"],
           s=3, facecolors="none", edgecolors="cyan", linewidths=0.4, label="known deposits")
fig.colorbar(points, ax=ax, shrink=0.6, label="prospectivity percentile")
ax.set(title=f"{commodity} prospectivity — model run {model_run_id[:8]}", xlabel="longitude", ylabel="latitude")
ax.set_aspect(1.6)
ax.legend(loc="lower left")
fig.tight_layout()

with mlflow.start_run(experiment_id=model_run.info.experiment_id, run_name=f"predict-{commodity}-{model_run_id[:8]}"):
    tags, uncommitted_diff = common.provenance_tags(args.data)
    mlflow.set_tags(tags | {
        "stage": "predict",
        "commodity": commodity,
        "model_uri": args.model_uri,
        "model_run_id": model_run_id,
    })
    if uncommitted_diff:
        mlflow.log_text(uncommitted_diff, "provenance/uncommitted.diff")
    mlflow.log_metrics({
        "n_hexes": len(predictions),
        # in-sample: includes deposits the model trained on, so optimistic — use the train run's test metrics for skill
        "known_deposits_in_top10pct_insample": float(top10[known].mean()),
    })
    mlflow.log_figure(fig, "prospectivity_map.png")
    plt.close(fig)
    table_path = out_dir / f"{commodity}_{model_run_id[:8]}.parquet"
    pd.DataFrame(predictions.drop(columns="geometry")).to_parquet(table_path, index=False)
    mlflow.log_artifact(str(table_path), "predictions")

print(f"scored {len(predictions):,} hexes with {args.model_uri}")
print(f"  QGIS layer: {gpkg_path}")
print(f"  known deposits in the top 10% (in-sample): {top10[known].mean():.0%}")
