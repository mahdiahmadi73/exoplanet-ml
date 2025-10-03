import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROC = PROJECT_ROOT / "data" / "processed"
OUTDIR = PROJECT_ROOT / "reports" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

unified_csv = PROC / "unified_clean.csv"
if not unified_csv.exists():
    raise SystemExit(f"Missing file: {unified_csv}. Run prepare_exoplanet_data.py first.")

df = pd.read_csv(unified_csv)
X = df[["z_period_days","z_transit_depth_ppm","z_planet_radius_re","z_stellar_radius_rs","z_snr"]].to_numpy()
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = df.loc[mask, "label_binary"].to_numpy()
missions = df.loc[mask, "mission"].astype(str).to_numpy()

um = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
X_um = um.fit_transform(X)

def scatter(X2, color, legend_vals, title, outpath):
    plt.figure(figsize=(7,6), dpi=140)
    for val in legend_vals:
        sel = color == val
        plt.scatter(X2[sel,0], X2[sel,1], s=10, alpha=0.6, label=str(val))
    plt.title(title); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(outpath); plt.close()

scatter(X_um, y, np.array([0,1]), "UMAP — colored by label_binary", "reports/figures/umap_binary.png")
scatter(X_um, missions, np.unique(missions), "UMAP — colored by mission", "reports/figures/umap_mission.png")