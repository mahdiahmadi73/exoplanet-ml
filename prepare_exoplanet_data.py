#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

import csv


BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
ART = BASE / "artifacts"

# Create directories if they don't exist
for p in [RAW, PROC, ART]:
    p.mkdir(parents=True, exist_ok=True)

FEATURES_ORDER = ["period_days", "transit_depth_ppm", "planet_radius_re", "stellar_radius_rs", "snr"]
KOI_COLMAP = {
    "source_id": "kepoi_name",
    "period_days": "koi_period",
    "transit_depth_ppm": "koi_depth",
    "planet_radius_re": "koi_prad",
    "stellar_radius_rs": "koi_srad",
    "snr": "koi_model_snr",      # prefer this
    "epoch": "koi_time0bk",
    "label_raw": "koi_disposition",  # prefer this according  to Markus' dataset Validataion.
}
K2_COLMAP = {
    "source_id": "pl_name",           # fallback to 'hostname'
    "period_days": "pl_orbper",
    "transit_depth_ppm": "pl_trandep",
    "planet_radius_re": "pl_rade",
    "stellar_radius_rs": "st_rad",
    "snr": "k2_snr",                  # often absent; NaN is fine
    "epoch": "pl_tranmid",
    "label_raw": "disposition",       # per your header
}
TOI_COLMAP = {
    "source_id": "toi",
    "period_days": "pl_orbper",
    "transit_depth_ppm": "pl_trandep",
    "planet_radius_re": "pl_rade",
    "stellar_radius_rs": "st_rad",
    "snr": "snr",                 # often missing; OK to be NaN
    "epoch": "pl_tranmid",
    "label_raw": "tfopwg_disp",
}
LABEL_MAP = {
    "CONFIRMED": ["CONFIRMED", "CONF", "CANDIDATE CONFIRMED", "KP", "CP", "KNOWN PLANET"],
    "CANDIDATE": ["CANDIDATE", "PC", "APC", "AMBIGUOUS", "CAND", "CANDIDATE AMBIGUOUS"],
    "FALSE POSITIVE": [
        "FALSE POSITIVE", "FP", "FALSE-POSITIVE", "NOT DISPOSITIONED - FALSE POSITIVE", "FA"  
    ],
}

#To ignore comment lines and stuff at top of the csv file
def robust_read_table(path):

    from pandas.errors import ParserError

    # Sniff delimiter; default to comma
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        sample = f.read(65536)
        f.seek(0)
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=[',', '\t', ';', '|']).delimiter
        except csv.Error:
            sep = ','

    common_kwargs = dict(
        sep=sep,
        comment='#',        # ignore NASA metadata/header lines
        encoding='utf-8',
        on_bad_lines='error',
        compression='infer',
    )

    # 1) Try fast C engine (supports low_memory)
    try:
        return pd.read_csv(
            path,
            engine='c',
            low_memory=False,
            **common_kwargs
        )
    except (ParserError, ValueError):
        # 2) Fallback: python engine (must drop low_memory)
        return pd.read_csv(
            path,
            engine='python',
            **common_kwargs
        )

def normalize_label(value: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().upper()
    for k, vals in LABEL_MAP.items():
        if s in vals:
            return k
    if "FALSE" in s and "POS" in s:
        return "FALSE POSITIVE"
    if s == "KP" or "KNOWN" in s or "CONF" in s:
        return "CONFIRMED"
    if s == "PC" or "CAND" in s or "APC" in s or "AMBIG" in s:
        return "CANDIDATE"
    if s == "CP":
        return "CONFIRMED"
    return s

def load_and_standardize(path: Path, mission: str, colmap: dict) -> pd.DataFrame:
    df = robust_read_table(path)
    out = pd.DataFrame()
    for unified_col, src_col in colmap.items():
        out[unified_col] = df[src_col] if src_col in df.columns else np.nan
    out["mission"] = mission

    # ---- KOI / Kepler fallbacks ----
    if mission.upper() in ("KOI", "KEPLER"):
        # Primary label is koi_disposition; if missing, fallback to koi_pdisposition
        if out["label_raw"].isna().all() and "koi_pdisposition" in df.columns:
            out["label_raw"] = df["koi_pdisposition"]
        # SNR fallback
        if out["snr"].isna().all() and "koi_snr" in df.columns:
            out["snr"] = df["koi_snr"]

    # ---- K2 fallbacks ----
    if mission.upper() == "K2":
        if out["source_id"].isna().all():
            if "hostname" in df.columns and "pl_letter" in df.columns:
                out["source_id"] = df["hostname"].astype(str) + "-" + df["pl_letter"].astype(str)
            elif "hostname" in df.columns:
                out["source_id"] = df["hostname"].astype(str)
        # Transit depth fallback
        if out["transit_depth_ppm"].isna().all():
            for alt in ("tran_depth", "k2_trandep"):
                if alt in df.columns:
                    out["transit_depth_ppm"] = pd.to_numeric(df[alt], errors="coerce")
                    break

    # ---- TESS fallbacks ----
    if mission.upper() == "TESS":
        if out["period_days"].isna().all():
            for c in ["orbital_period", "toi_period"]:
                if c in df.columns:
                    out["period_days"] = pd.to_numeric(df[c], errors="coerce"); break
        if out["transit_depth_ppm"].isna().all():
            for c in ["transit_depth", "toi_transit_depth"]:
                if c in df.columns:
                    out["transit_depth_ppm"] = pd.to_numeric(df[c], errors="coerce"); break

    # Now map the (possibly updated) raw label → coarse label
    out["label_coarse"] = out["label_raw"].apply(normalize_label)

    # Ensure numeric casting
    for c in FEATURES_ORDER + ["epoch"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def main():
    paths = {
        "KOI": RAW / "kepler_cumulative_koi.csv",
        "K2": RAW / "k2_planets_candidates.csv",
        "TESS": RAW / "tess_toi.csv",
    }

    colmaps = {
        "KOI": KOI_COLMAP,
        "K2": K2_COLMAP,
        "TESS": TOI_COLMAP,
    }
    frames = []
    for mission, path in paths.items():
        if path.exists():
            print("[load]", mission, ":", path)
            frames.append(load_and_standardize(path, "KEPLER" if mission == "KOI" else mission, colmaps[mission]))
        else:
            print("[warn] Missing file for", mission, ":", path)
    if not frames:
        raise SystemExit("No input CSVs found in " + str(RAW))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["label_coarse"]).copy()
    df["is_planet"] = (df["label_coarse"].str.upper() == "CONFIRMED").astype(int)
    # Clip extreme outliers per feature (1st–99th percentile)
    for c in FEATURES_ORDER:
        if c in df.columns:
            lo, hi = df[c].quantile([0.01, 0.99])
            df[c] = df[c].clip(lower=lo, upper=hi)


    
    for c in ["source_id", "mission", "label_raw", "label_coarse"]:
        if c in df.columns:
            df[c] = df[c].astype("string")   # arrow-friendly string dtype
    df.to_parquet(PROC / "unified_raw.parquet", index=False)
    df.to_csv(PROC / "unified_raw.csv", index=False)
    # Splits
    X = df[FEATURES_ORDER].copy()
    y_bin = df["is_planet"].values
    y_multi = df["label_coarse"].astype(str).values
    X_train, X_temp, yb_train, yb_temp, ym_train, ym_temp = train_test_split(
        X, y_bin, y_multi, test_size=0.3, random_state=42, stratify=y_bin
    )
    X_val, X_test, yb_val, yb_test, ym_val, ym_test = train_test_split(
        X_temp, yb_temp, ym_temp, test_size=0.5, random_state=42, stratify=yb_temp
    )
    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)
    X_train_t = scaler.fit_transform(X_train_i)
    X_val_t = scaler.transform(X_val_i)
    X_test_t = scaler.transform(X_test_i)
    # Save splits
    def save_split(name, Xn, yb, ym):
        np.save(PROC / ("X_" + name + ".npy"), Xn)
        pd.DataFrame({ "y_binary": yb, "y_multiclass": ym }).to_csv(PROC / ("y_" + name + ".csv"), index=False)
    save_split("train", X_train_t, yb_train, ym_train)
    save_split("val", X_val_t, yb_val, ym_val)
    save_split("test", X_test_t, yb_test, ym_test)
    # Save unified processed (for reference)
    X_full_i = imputer.transform(df[FEATURES_ORDER])
    X_full_t = scaler.transform(X_full_i)
    proc_df = pd.DataFrame(X_full_t, columns=["z_" + c for c in FEATURES_ORDER])
    proc_df["label_binary"] = y_bin
    proc_df["label_multiclass"] = y_multi
    proc_df["mission"] = df["mission"].values
    proc_df.to_parquet(PROC / "unified_clean.parquet", index=False)
    proc_df.to_csv(PROC / "unified_clean.csv", index=False)
    # Save artifacts
    joblib.dump(scaler, ART / "numeric_scaler.joblib")
    joblib.dump(imputer, ART / "numeric_imputer.joblib")
    json.dump({
        "label_classes": sorted(df["label_coarse"].dropna().str.upper().unique().tolist()),
        "features": FEATURES_ORDER,
    }, open(ART / "label_map.json", "w"))
    print("[done] Wrote processed datasets and artifacts to", PROC, "and", ART)

if __name__ == "__main__":
    main()