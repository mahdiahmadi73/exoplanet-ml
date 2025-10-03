
#!/usr/bin/env python3
"""
PCA and Correlation EDA

- Loads the data (z-scored features + labels)
- Plots:
    1) Feature correlation heatmap (using matplotlib only)
    2) PCA 2D scatter (PC1 vs PC2) colored by binary label
    3) PCA 2D scatter colored by mission
- Saves figures to reports/figures/

Run:
    python pca_and_eda.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PROC = PROJECT_ROOT / "data" / "processed"
OUTDIR = PROJECT_ROOT / "reports" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Prefer CSV since it's guaranteed by the pipeline; parquet is optional
unified_csv = PROC / "unified_clean.csv"
if not unified_csv.exists():
    raise SystemExit(f"Missing file: {unified_csv}. Run prepare_exoplanet_data.py first.")

df = pd.read_csv(unified_csv)

# Columns expected from the pipeline
Z_FEATS = ["z_period_days", "z_transit_depth_ppm", "z_planet_radius_re", "z_stellar_radius_rs", "z_snr"]
LABEL_BIN = "label_binary"
LABEL_MULTI = "label_multiclass"
MISSION = "mission"

# Basic sanity check
missing_cols = [c for c in Z_FEATS + [LABEL_BIN, LABEL_MULTI, MISSION] if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Missing expected columns: {missing_cols} in {unified_csv}")

# 1) Correlation Heatmap (matplotlib only) 
corr = df[Z_FEATS].corr().values
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
im = ax.imshow(corr, interpolation='nearest', aspect='auto')
ax.set_xticks(range(len(Z_FEATS)))
ax.set_yticks(range(len(Z_FEATS)))
ax.set_xticklabels(Z_FEATS, rotation=45, ha='right')
ax.set_yticklabels(Z_FEATS)
ax.set_title("Feature Correlation (z-scored)")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r")
fig.tight_layout()
heatmap_path = OUTDIR / "correlation_heatmap.png"
fig.savefig(heatmap_path, bbox_inches="tight")
plt.close(fig)

#  PCA (2D) colored by binary label ===
X = df[Z_FEATS].to_numpy()
# drop rows with NaNs just in case
mask = ~np.isnan(X).any(axis=1) & df[LABEL_BIN].notna()
X = X[mask]
y_bin = df.loc[mask, LABEL_BIN].to_numpy()
missions = df.loc[mask, MISSION].astype(str).to_numpy()

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

def scatter_by_labels(Xp, labels, label_names, title, outfile):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    # Unique order preserves 0 then 1
    uniq = [l for l in label_names if l in np.unique(labels)]
    for l in uniq:
        sel = labels == l
        ax.scatter(Xp[sel, 0], Xp[sel, 1], alpha=0.5, label=str(l), s=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend(title="Label", loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

scatter_by_labels(
    X_pca, y_bin, label_names=np.array([0, 1]),
    title="PCA (PC1 vs PC2) — colored by label_binary (0=non-planet, 1=planet)",
    outfile=OUTDIR / "pca_binary.png",
)

# PCA (2D) colored by mission 
# Reuse same PCA projection so axes are identical
def scatter_by_mission(Xp, missions_arr, title, outfile):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    uniq = np.unique(missions_arr)
    for m in uniq:
        sel = missions_arr == m
        ax.scatter(Xp[sel, 0], Xp[sel, 1], alpha=0.5, label=str(m), s=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend(title="Mission", loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

scatter_by_mission(
    X_pca, missions,
    title="PCA (PC1 vs PC2) — colored by mission",
    outfile=OUTDIR / "pca_mission.png",
)

