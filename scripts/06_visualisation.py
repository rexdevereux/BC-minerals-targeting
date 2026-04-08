import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from scipy import stats
from pathlib import Path
from loguru import logger

# --- paths ---
data_dir   = Path("data")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

in_path    = data_dir / "geochem_05_features.parquet"
font_path  = data_dir / "Inter" / "Inter-VariableFont_opsz,wght.ttf"

# --- load inter font ---
fm.fontManager.addfont(str(font_path))
inter = fm.FontProperties(fname=str(font_path))
font_name = inter.get_name()

# --- global style ---
black  = "#0a0a0a"
white  = "#ffffff"
grey   = "#888888"

plt.rcParams.update({
    "figure.facecolor":   black,
    "axes.facecolor":     black,
    "axes.edgecolor":     grey,
    "axes.labelcolor":    white,
    "axes.grid":          False,
    "text.color":         white,
    "xtick.color":        white,
    "ytick.color":        white,
    "legend.facecolor":   "#1a1a1a",
    "legend.edgecolor":   grey,
    "legend.labelcolor":  white,
    "font.family":        font_name,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# --- bc bounds and boundary ---
bc_lon = (-139.5, -113.5)
bc_lat = (48.0,    60.5)

logger.info("loading BC boundary...")
bc_boundary = gpd.read_file(
    data_dir / "BC_boundary.gpkg",
    layer="geoboundariescanadm1"
).to_crs("EPSG:4326")

def add_bc_boundary(ax):
    bc_boundary.boundary.plot(ax=ax, color=white, linewidth=0.8, zorder=10)

def style_map_ax(ax):
    ax.set_facecolor(black)
    ax.set_xlim(*bc_lon)
    ax.set_ylim(*bc_lat)
    ax.tick_params(colors=white)

# --- custom colormaps ---
def make_cmap(colors, name):
    return mcolors.LinearSegmentedColormap.from_list(name, colors)

cu_raw_cmap   = make_cmap(["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026"], "cu_raw")
cu_zscore_cmap= make_cmap(["#ffffcc","#c2e699","#78c679","#31a354","#006837"], "cu_z")
target_cmap   = make_cmap(["#edf8fb","#b2e2e2","#66c2a4","#2ca25f","#006d2c"], "targets")
corr_cmap     = make_cmap(["#d7191c","#fdae61","#ffffbf","#a6d96a","#1a9641"], "corr")

# element colours
elem_colors = {
    "copper":     "#ff8b47",
    "nickel":     "#cea2fa",
    "cobalt":     "#48bcfd",
    "gold":       "#ffb914",
    "molybdenum": "#bff102",
    "lithium":    "#f20c59",
}

# --- load data ---
logger.info(f"loading {in_path}...")
gdf = gpd.read_parquet(in_path)
df  = pd.DataFrame(gdf)
logger.info(f"loaded {len(df):,} rows x {len(df.columns)} columns")

# ── plot 1: element distributions ────────────────────────────────────
logger.info("plot 1: element distributions...")

elements = [
    ("copper_ppm",     "copper_log",     "Copper",     elem_colors["copper"]),
    ("nickel_ppm",     "nickel_log",     "Nickel",     elem_colors["nickel"]),
    ("cobalt_ppm",     "cobalt_log",     "Cobalt",     elem_colors["cobalt"]),
    ("gold_ppm",       "gold_log",       "Gold",       elem_colors["gold"]),
    ("molybdenum_ppm", "molybdenum_log", "Molybdenum", elem_colors["molybdenum"]),
    ("lithium_ppm",    "lithium_log",    "Lithium",    elem_colors["lithium"]),
]
elements = [(a, b, n, c) for a, b, n, c in elements if a in df.columns]

fig, axes = plt.subplots(2, len(elements), figsize=(20, 8))
fig.patch.set_facecolor(black)
fig.suptitle(
    "BC Regional Geochemical Survey 2020 — Element Distributions\n"
    "Raw concentrations (top) vs log-transformed (bottom) — 50,990 stream sediment samples",
    fontsize=13, fontweight="bold", color=white
)

for i, (raw_col, log_col, name, color) in enumerate(elements):
    raw_vals = df[raw_col].dropna()
    log_vals = df[log_col].dropna()

    for ax in [axes[0, i], axes[1, i]]:
        ax.set_facecolor(black)
        ax.tick_params(colors=white)
        ax.xaxis.label.set_color(white)
        ax.yaxis.label.set_color(white)
        for spine in ax.spines.values():
            spine.set_edgecolor(grey)

    axes[0, i].hist(raw_vals.clip(upper=raw_vals.quantile(0.99)),
                    bins=50, color=color, alpha=0.85, edgecolor="none")
    axes[0, i].set_title(f"{name}\n(raw ppm)", fontsize=10, color=white)
    axes[0, i].set_xlabel("ppm", color=white)
    if i == 0:
        axes[0, i].set_ylabel("Count", color=white)

    axes[1, i].hist(log_vals, bins=50, color=color, alpha=0.85,
                    edgecolor="none", density=True)
    mu, sigma = log_vals.mean(), log_vals.std()
    x = np.linspace(log_vals.min(), log_vals.max(), 200)
    axes[1, i].plot(x, stats.norm.pdf(x, mu, sigma),
                    color=white, linewidth=1.5, linestyle="--", label="Normal fit")
    axes[1, i].set_title(f"log({name}+1)", fontsize=10, color=white)
    axes[1, i].set_xlabel("log(ppm+1)", color=white)
    if i == 0:
        axes[1, i].set_ylabel("Density", color=white)
    axes[1, i].legend(fontsize=7)

plt.tight_layout()
out = output_dir / "01_element_distributions.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=black)
plt.close()
logger.info(f"saved {out}")

# ── plot 2: copper anomaly map ────────────────────────────────────────
logger.info("plot 2: copper anomaly map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(black)
fig.suptitle(
    "Copper Geochemical Anomaly — British Columbia\n"
    "BC Regional Geochemical Survey 2020 — stream sediment samples",
    fontsize=13, fontweight="bold", color=white
)

ax1 = axes[0]
clip_val = df["copper_ppm"].quantile(0.98)
sc1 = ax1.scatter(
    df["longitude"], df["latitude"],
    c=df["copper_ppm"].clip(upper=clip_val),
    cmap=cu_raw_cmap,
    norm=mcolors.LogNorm(
        vmin=max(df["copper_ppm"].quantile(0.02), 0.1),
        vmax=clip_val
    ),
    s=2, alpha=0.7, linewidths=0
)
style_map_ax(ax1)
add_bc_boundary(ax1)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Raw Copper Concentration (ppm)", fontsize=11, color=white)
cb1 = plt.colorbar(sc1, ax=ax1, shrink=0.7)
cb1.set_label("Cu (ppm)", color=white)
cb1.ax.yaxis.set_tick_params(color=white)
plt.setp(cb1.ax.yaxis.get_ticklabels(), color=white)

ax2 = axes[1]
zvals = df["copper_zscore"].clip(-3, 5)
sc2 = ax2.scatter(
    df["longitude"], df["latitude"],
    c=zvals,
    cmap=cu_zscore_cmap,
    norm=mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=5),
    s=2, alpha=0.7, linewidths=0
)
style_map_ax(ax2)
add_bc_boundary(ax2)
ax2.set_xlabel("Longitude")
ax2.set_title("Copper Z-score Anomaly\n(bright = exploration target)", fontsize=11, color=white)
cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.7)
cb2.set_label("Z-score", color=white)
cb2.ax.yaxis.set_tick_params(color=white)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=white)

top1 = df[df["copper_zscore"] >= df["copper_zscore"].quantile(0.99)]
ax2.scatter(top1["longitude"], top1["latitude"],
            s=15, facecolors="none", edgecolors=white,
            linewidths=0.6, zorder=5,
            label=f"Top 1% (n={len(top1):,})")
ax2.legend(fontsize=9, loc="lower right")

plt.tight_layout()
out = output_dir / "02_copper_anomaly_map.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=black)
plt.close()
logger.info(f"saved {out}")

# ── plot 3: porphyry targeting map ────────────────────────────────────
logger.info("plot 3: porphyry targeting map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(black)
fig.suptitle(
    "Porphyry Cu-Au-Mo Targeting — British Columbia\n"
    "Weighted score: Cu(35%) + Au(25%) + Mo(20%) + As(10%) + Co(10%)",
    fontsize=13, fontweight="bold", color=white
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
style_map_ax(ax1)
add_bc_boundary(ax1)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Porphyry Score (all samples)", fontsize=11, color=white)
cb1 = plt.colorbar(sc1, ax=ax1, shrink=0.7)
cb1.set_label("Score", color=white)
cb1.ax.yaxis.set_tick_params(color=white)
plt.setp(cb1.ax.yaxis.get_ticklabels(), color=white)

ax2        = axes[1]
targets    = df[df["target_porphyry"]]
sc2 = ax2.scatter(targets["longitude"], targets["latitude"],
                  c=targets["score_porphyry"],
                  cmap=target_cmap, s=15, alpha=0.9,
                  linewidths=0, zorder=5,
                  label=f"Drill targets (n={len(targets):,})")
style_map_ax(ax2)
add_bc_boundary(ax2)
ax2.set_xlabel("Longitude")
ax2.set_title("Priority Porphyry Targets (top 2%)\nColour = score", fontsize=11, color=white)
ax2.legend(fontsize=9, loc="lower right")
cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.7)
cb2.set_label("Score", color=white)
cb2.ax.yaxis.set_tick_params(color=white)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=white)

plt.tight_layout()
out = output_dir / "03_porphyry_targets.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=black)
plt.close()
logger.info(f"saved {out}")

# ── plot 4: battery metals targeting map ─────────────────────────────
logger.info("plot 4: battery metals targeting map...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(black)
fig.suptitle(
    "Battery Metals Targeting — British Columbia\n"
    "Weighted score: Li(45%) + Co(30%) + Ni(25%)",
    fontsize=13, fontweight="bold", color=white
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
style_map_ax(ax1)
add_bc_boundary(ax1)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Battery Metals Score (all samples)", fontsize=11, color=white)
cb1 = plt.colorbar(sc1, ax=ax1, shrink=0.7)
cb1.set_label("Score", color=white)
cb1.ax.yaxis.set_tick_params(color=white)
plt.setp(cb1.ax.yaxis.get_ticklabels(), color=white)

ax2        = axes[1]
targets    = df[df["target_battery"]]
sc2 = ax2.scatter(targets["longitude"], targets["latitude"],
                  c=targets["score_battery"],
                  cmap=target_cmap, s=15, alpha=0.9,
                  linewidths=0, zorder=5,
                  label=f"Drill targets (n={len(targets):,})")
style_map_ax(ax2)
add_bc_boundary(ax2)
ax2.set_xlabel("Longitude")
ax2.set_title("Priority Battery Metals Targets (top 2%)\nColour = score", fontsize=11, color=white)
ax2.legend(fontsize=9, loc="lower right")
cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.7)
cb2.set_label("Score", color=white)
cb2.ax.yaxis.set_tick_params(color=white)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=white)

plt.tight_layout()
out = output_dir / "04_battery_targets.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=black)
plt.close()
logger.info(f"saved {out}")

# ── plot 5: correlation heatmap + cu vs mo scatter ────────────────────
logger.info("plot 5: correlation heatmap...")

log_cols = [c for c in df.columns if c.endswith("_log")]
corr     = df[log_cols].dropna().corr()
labels   = [c.replace("_log", "").replace("_", " ").title()
            for c in corr.columns]
n        = len(labels)

# mask upper triangle — keep lower triangle only
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr_masked = corr.copy()
corr_masked[mask] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(black)
fig.suptitle("Element Correlation Analysis — Deposit Type Signatures",
             fontsize=13, fontweight="bold", color=white)

ax = axes[0]
ax.set_facecolor(black)
im = ax.imshow(corr_masked.values, cmap=corr_cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color=white)
ax.set_yticklabels(labels, fontsize=9, color=white)
ax.set_title("Pearson Correlation\n(log-transformed concentrations)", fontsize=11, color=white)
for spine in ax.spines.values():
    spine.set_edgecolor(grey)
cb = plt.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Correlation", color=white)
cb.ax.yaxis.set_tick_params(color=white)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=white)

# annotate lower triangle only
for i in range(n):
    for j in range(n):
        if mask[i, j] or i == j:
            continue
        val   = corr.values[i, j]
        color = white if abs(val) > 0.6 else black
        ax.text(j, i, f"{val:.2f}", ha="center",
                va="center", fontsize=7, color=color)

# cu vs mo scatter
ax2 = axes[1]
ax2.set_facecolor(black)
if "copper_log" in df.columns and "molybdenum_log" in df.columns:
    data = df[["copper_log", "molybdenum_log", "score_porphyry"]].dropna()
    sc   = ax2.scatter(
        data["copper_log"], data["molybdenum_log"],
        c=data["score_porphyry"],
        cmap="plasma", alpha=0.4, s=3, linewidths=0
    )
    m, b, r, p, _ = stats.linregress(
        data["copper_log"], data["molybdenum_log"]
    )
    x_line = np.linspace(data["copper_log"].min(),
                         data["copper_log"].max(), 100)
    ax2.plot(x_line, m * x_line + b, color=white,
             linewidth=1.5, linestyle="--", label=f"r = {r:.2f}")
    ax2.set_xlabel("log(Cu + 1)", color=white)
    ax2.set_ylabel("log(Mo + 1)", color=white)
    ax2.set_title(
        "Cu vs Mo — Porphyry Copper Pathfinder\nColour = porphyry score",
        fontsize=11, color=white
    )
    ax2.tick_params(colors=white)
    for spine in ax2.spines.values():
        spine.set_edgecolor(grey)
    ax2.legend(fontsize=10)
    cb2 = plt.colorbar(sc, ax=ax2, shrink=0.8)
    cb2.set_label("Porphyry score", color=white)
    cb2.ax.yaxis.set_tick_params(color=white)
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color=white)

plt.tight_layout()
out = output_dir / "05_correlation_heatmap.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=black)
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