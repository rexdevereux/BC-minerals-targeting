import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from pathlib import Path
from loguru import logger

# --- paths ---
data_dir   = Path("data")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

in_path = data_dir / "geochem_05_features.parquet"

# --- style ---
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f5f5f3",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
})

# bc map bounds
bc_lon = (-139.5, -113.5)
bc_lat = (48.0,    60.5)

# load bc boundary for map overlays
logger.info("loading BC boundary...")
bc_boundary = gpd.read_file(
    data_dir / "BC_boundary.gpkg",
    layer="geoboundariescanadm1"
).to_crs("EPSG:4326")

def add_bc_boundary(ax):
    """overlay BC boundary outline on a map axes"""
    bc_boundary.boundary.plot(
        ax=ax, color="white", linewidth=0.8, zorder=10
    )

# --- load ---
logger.info(f"loading {in_path}...")
gdf = gpd.read_parquet(in_path)
df  = pd.DataFrame(gdf)
logger.info(f"loaded {len(df):,} rows x {len(df.columns)} columns")

# ── plot 1: element distributions ────────────────────────────────────
logger.info("plot 1: element distributions...")

elements = [
    ("copper_ppm",     "copper_log",     "Copper",     "#e74c3c"),
    ("nickel_ppm",     "nickel_log",     "Nickel",     "#3498db"),
    ("cobalt_ppm",     "cobalt_log",     "Cobalt",     "#9b59b6"),
    ("gold_ppm",       "gold_log",       "Gold",       "#f1c40f"),
    ("molybdenum_ppm", "molybdenum_log", "Molybdenum", "#1abc9c"),
    ("lithium_ppm",    "lithium_log",    "Lithium",    "#e67e22"),
]
elements = [(a, b, n, c) for a, b, n, c in elements if a in df.columns]

fig, axes = plt.subplots(2, len(elements), figsize=(20, 8))
fig.suptitle(
    "BC Regional Geochemical Survey 2020 — Element Distributions\n"
    "Raw concentrations (top) vs log-transformed (bottom) — 50,990 stream sediment samples",
    fontsize=13, fontweight="bold"
)

for i, (raw_col, log_col, name, color) in enumerate(elements):
    raw_vals = df[raw_col].dropna()
    log_vals = df[log_col].dropna()

    # raw — clip at 99th percentile so extreme outliers don't squash the plot
    axes[0, i].hist(raw_vals.clip(upper=raw_vals.quantile(0.99)),
                    bins=50, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.3)
    axes[0, i].set_title(f"{name}\n(raw ppm)", fontsize=10)
    axes[0, i].set_xlabel("ppm")
    if i == 0:
        axes[0, i].set_ylabel("Count")

    # log transformed + normal curve overlay
    axes[1, i].hist(log_vals, bins=50, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.3, density=True)
    mu, sigma = log_vals.mean(), log_vals.std()
    x = np.linspace(log_vals.min(), log_vals.max(), 200)
    axes[1, i].plot(x, stats.norm.pdf(x, mu, sigma),
                    color="black", linewidth=1.5,
                    linestyle="--", label="Normal fit")
    axes[1, i].set_title(f"log({name}+1)", fontsize=10)
    axes[1, i].set_xlabel("log(ppm+1)")
    if i == 0:
        axes[1, i].set_ylabel("Density")
    axes[1, i].legend(fontsize=7)

plt.tight_layout()
out = output_dir / "01_element_distributions.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"saved {out}")

# ── plot 2: copper anomaly map ────────────────────────────────────────
logger.info("plot 2: copper anomaly map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    "Copper Geochemical Anomaly — British Columbia\n"
    "BC Regional Geochemical Survey 2020 — stream sediment samples",
    fontsize=13, fontweight="bold"
)

# raw concentration
ax1 = axes[0]
clip_val = df["copper_ppm"].quantile(0.98)
sc1 = ax1.scatter(
    df["longitude"], df["latitude"],
    c=df["copper_ppm"].clip(upper=clip_val),
    cmap="YlOrRd",
    norm=mcolors.LogNorm(
        vmin=max(df["copper_ppm"].quantile(0.02), 0.1),
        vmax=clip_val
    ),
    s=2, alpha=0.6, linewidths=0
)
ax1.set_xlim(*bc_lon)
ax1.set_ylim(*bc_lat)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Raw Copper Concentration (ppm)", fontsize=11)
ax1.set_facecolor("#e8f4f8")
plt.colorbar(sc1, ax=ax1, shrink=0.7).set_label("Cu (ppm)")

# z-score anomaly
ax2 = axes[1]
zvals = df["copper_zscore"].clip(-3, 5)
sc2 = ax2.scatter(
    df["longitude"], df["latitude"],
    c=zvals,
    cmap="RdYlBu_r",
    norm=mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=5),
    s=2, alpha=0.6, linewidths=0
)
ax2.set_xlim(*bc_lon)
ax2.set_ylim(*bc_lat)
ax2.set_xlabel("Longitude")
ax2.set_title("Copper Z-score Anomaly\n(red = exploration target)", fontsize=11)
ax2.set_facecolor("#e8f4f8")
plt.colorbar(sc2, ax=ax2, shrink=0.7).set_label("Z-score")

top1 = df[df["copper_zscore"] >= df["copper_zscore"].quantile(0.99)]
ax2.scatter(top1["longitude"], top1["latitude"],
            s=15, facecolors="none", edgecolors="black",
            linewidths=0.6, zorder=5,
            label=f"Top 1% (n={len(top1):,})")
ax2.legend(fontsize=9, loc="lower right")

add_bc_boundary(ax1)
add_bc_boundary(ax2)
plt.tight_layout()
out = output_dir / "02_copper_anomaly_map.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"saved {out}")

# ── plot 3: porphyry targeting map ────────────────────────────────────
logger.info("plot 3: porphyry targeting map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    "Porphyry Cu-Au-Mo Targeting — British Columbia\n"
    "Weighted score: Cu(35%) + Au(25%) + Mo(20%) + As(10%) + Co(10%)",
    fontsize=13, fontweight="bold"
)

score = df["score_porphyry"]
ax1   = axes[0]
sc1   = ax1.scatter(
    df["longitude"], df["latitude"],
    c=score,
    cmap="inferno",
    norm=mcolors.TwoSlopeNorm(
        vmin=score.quantile(0.02),
        vcenter=score.median(),
        vmax=score.quantile(0.98)
    ),
    s=2, alpha=0.7, linewidths=0
)
ax1.set_xlim(*bc_lon)
ax1.set_ylim(*bc_lat)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Porphyry Score (all samples)", fontsize=11)
ax1.set_facecolor("#1a1a2e")
plt.colorbar(sc1, ax=ax1, shrink=0.7).set_label("Score")

ax2         = axes[1]
background  = df[~df["target_porphyry"]]
targets     = df[df["target_porphyry"]]
ax2.scatter(background["longitude"], background["latitude"],
            c="#2c3e50", s=1, alpha=0.3, linewidths=0, label="Background")
sc2 = ax2.scatter(targets["longitude"], targets["latitude"],
                  c=targets["score_porphyry"],
                  cmap="YlOrRd", s=15, alpha=0.9,
                  linewidths=0, zorder=5,
                  label=f"Drill targets (n={len(targets):,})")
ax2.set_xlim(*bc_lon)
ax2.set_ylim(*bc_lat)
ax2.set_xlabel("Longitude")
ax2.set_title("Priority Porphyry Targets (top 2%)\nColour = score", fontsize=11)
ax2.set_facecolor("#1a1a2e")
ax2.legend(fontsize=9, loc="lower right",
           facecolor="#2c3e50", labelcolor="white")
plt.colorbar(sc2, ax=ax2, shrink=0.7).set_label("Score")

add_bc_boundary(ax1)
add_bc_boundary(ax2)
plt.tight_layout()
out = output_dir / "03_porphyry_targets.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"saved {out}")

# ── plot 4: battery metals targeting map ─────────────────────────────
logger.info("plot 4: battery metals targeting map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    "Battery Metals Targeting — British Columbia\n"
    "Weighted score: Li(45%) + Co(30%) + Ni(25%)",
    fontsize=13, fontweight="bold"
)

score = df["score_battery"]
ax1   = axes[0]
sc1   = ax1.scatter(
    df["longitude"], df["latitude"],
    c=score,
    cmap="viridis",
    norm=mcolors.TwoSlopeNorm(
        vmin=score.quantile(0.02),
        vcenter=score.median(),
        vmax=score.quantile(0.98)
    ),
    s=2, alpha=0.7, linewidths=0
)
ax1.set_xlim(*bc_lon)
ax1.set_ylim(*bc_lat)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Battery Metals Score (all samples)", fontsize=11)
ax1.set_facecolor("#1a1a2e")
plt.colorbar(sc1, ax=ax1, shrink=0.7).set_label("Score")

ax2        = axes[1]
background = df[~df["target_battery"]]
targets    = df[df["target_battery"]]
ax2.scatter(background["longitude"], background["latitude"],
            c="#2c3e50", s=1, alpha=0.3, linewidths=0, label="Background")
sc2 = ax2.scatter(targets["longitude"], targets["latitude"],
                  c=targets["score_battery"],
                  cmap="YlGn", s=15, alpha=0.9,
                  linewidths=0, zorder=5,
                  label=f"Drill targets (n={len(targets):,})")
ax2.set_xlim(*bc_lon)
ax2.set_ylim(*bc_lat)
ax2.set_xlabel("Longitude")
ax2.set_title("Priority Battery Metals Targets (top 2%)\nColour = score", fontsize=11)
ax2.set_facecolor("#1a1a2e")
ax2.legend(fontsize=9, loc="lower right",
           facecolor="#2c3e50", labelcolor="white")
plt.colorbar(sc2, ax=ax2, shrink=0.7).set_label("Score")

add_bc_boundary(ax1)
add_bc_boundary(ax2)
plt.tight_layout()
out = output_dir / "04_battery_targets.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"saved {out}")

# ── plot 5: correlation heatmap + cu vs mo scatter ────────────────────
logger.info("plot 5: correlation heatmap...")

log_cols = [c for c in df.columns if c.endswith("_log")]
corr     = df[log_cols].dropna().corr()
labels   = [c.replace("_log", "").replace("_", " ").title()
            for c in corr.columns]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Element Correlation Analysis — Deposit Type Signatures",
             fontsize=13, fontweight="bold")

im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axes[0].set_xticks(range(len(labels)))
axes[0].set_yticks(range(len(labels)))
axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
axes[0].set_yticklabels(labels, fontsize=9)
axes[0].set_title("Pearson Correlation\n(log-transformed concentrations)", fontsize=11)
plt.colorbar(im, ax=axes[0], shrink=0.8).set_label("Correlation")

for i in range(len(corr)):
    for j in range(len(corr)):
        val   = corr.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        axes[0].text(j, i, f"{val:.2f}", ha="center",
                     va="center", fontsize=7, color=color)

# cu vs mo scatter — porphyry pathfinder
if "copper_log" in df.columns and "molybdenum_log" in df.columns:
    data = df[["copper_log", "molybdenum_log", "score_porphyry"]].dropna()
    sc   = axes[1].scatter(
        data["copper_log"], data["molybdenum_log"],
        c=data["score_porphyry"],
        cmap="plasma", alpha=0.4, s=3, linewidths=0
    )
    m, b, r, p, _ = stats.linregress(
        data["copper_log"], data["molybdenum_log"]
    )
    x_line = np.linspace(data["copper_log"].min(),
                         data["copper_log"].max(), 100)
    axes[1].plot(x_line, m * x_line + b, color="black",
                 linewidth=1.5, linestyle="--", label=f"r = {r:.2f}")
    axes[1].set_xlabel("log(Cu + 1)", fontsize=11)
    axes[1].set_ylabel("log(Mo + 1)", fontsize=11)
    axes[1].set_title(
        "Cu vs Mo — Porphyry Copper Pathfinder\nColour = porphyry score",
        fontsize=11
    )
    axes[1].legend(fontsize=10)
    plt.colorbar(sc, ax=axes[1], shrink=0.8).set_label("Porphyry score")

plt.tight_layout()
out = output_dir / "05_correlation_heatmap.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"saved {out}")

# --- final summary ---
print(f"\n{'='*60}")
print(f"BC CRITICAL MINERALS PIPELINE — COMPLETE")
print(f"{'='*60}")
print(f"  source:             BC RGS 2020 (BCGS GeoFile 2020-08)")
print(f"  samples:            {len(df):,}")
print(f"  features:           {len(df.columns)}")
print(f"  porphyry targets:   {df['target_porphyry'].sum():,} (top 2%)")
print(f"  battery targets:    {df['target_battery'].sum():,} (top 2%)")
print(f"\n  output maps:")
for f in sorted(output_dir.glob("*.png")):
    size_kb = f.stat().st_size / 1024
    print(f"    {f.name:<45} {size_kb:.0f} KB")
print(f"{'='*60}")