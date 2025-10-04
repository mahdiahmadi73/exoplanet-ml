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
    
    out["label_binary"] = out["label_coarse"].map(
    {"CONFIRMED": 1, "CANDIDATE": 0, "FALSE POSITIVE": 0}
        ).astype("Int64")
    
    # Ensure numeric casting
    for c in FEATURES_ORDER + ["epoch"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out



def deduplicate_k2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedupe K2 rows using ONLY unified columns.
    - Assumes `source_id` is already set during standardization
      (pl_name, or hostname-pl_letter fallback).
    - If a date-like column exists (rowupdate/pl_pubdate/releasedate), use it
      to keep the newest; otherwise keep the last occurrence.
    """
    if df.empty or "mission" not in df.columns:
        return df

    is_k2 = df["mission"].str.upper().eq("K2")
    d2 = df[is_k2].copy()
    d_other = df[~is_k2].copy()

    if d2.empty or "source_id" not in d2.columns:
        return df  # nothing to do / no IDs to dedupe on

    # Prefer a date column *if it exists* (but we don't require it)
    date_col = next((c for c in ("rowupdate", "pl_pubdate", "releasedate") if c in d2.columns), None)
    if date_col is not None:
        d2["_dedupe_date"] = pd.to_datetime(d2[date_col], errors="coerce", utc=True)
        d2 = d2.sort_values("_dedupe_date", na_position="first").drop_duplicates(subset=["source_id"], keep="last")
        d2 = d2.drop(columns=["_dedupe_date"])
    else:
        # No usable date column carried through → just drop dups by source_id
        d2 = d2.drop_duplicates(subset=["source_id"], keep="last")

    return pd.concat([d_other, d2], ignore_index=True)


def main():
    paths = {
        "KOI":  RAW / "kepler_cumulative_koi.csv",
        "K2":   RAW / "k2_planets_candidates.csv",
        "TESS": RAW / "tess_toi.csv",
    }
    colmaps = {"KOI": KOI_COLMAP, "K2": K2_COLMAP, "TESS": TOI_COLMAP}

    frames = []
    for mission, path in paths.items():
        if path.exists():
            print("[load]", mission, ":", path)
            frames.append(load_and_standardize(path, "KEPLER" if mission == "KOI" else mission, colmaps[mission]))
        else:
            print("[warn] Missing file for", mission, ":", path)

    if not frames:
        raise SystemExit("No input CSVs found in " + str(RAW))

    # 1) Merge missions
    df = pd.concat(frames, ignore_index=True)

    # 2) Keep only rows with usable labels; make binary int
    df = df.dropna(subset=["label_coarse", "label_binary"]).copy()
    df["label_binary"] = df["label_binary"].astype(int)

    # 3) Deduplicate K2 using unified columns
    df = deduplicate_k2(df)

    # 4) Clip outliers (1%..99%) on numeric features
    for c in FEATURES_ORDER:
        if c in df.columns:
            lo, hi = df[c].quantile([0.01, 0.99])
            df[c] = df[c].clip(lower=lo, upper=hi)

    # 5) Cast string cols (Arrow-friendly) and save unified raw
    for c in ["source_id", "mission", "label_raw", "label_coarse"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROC / "unified_raw.parquet", index=False)
    df.to_csv(PROC / "unified_raw.csv", index=False)

    # 6) Build X/y
    X = df[FEATURES_ORDER].copy()
    y_bin   = df["label_binary"].astype(int).values
    y_multi = df["label_coarse"].astype(str).values

    # 7) Train/val/test (60/20/20), stratify by binary label
    X_tmp, X_test, y_tmp, y_test, m_tmp, m_test = train_test_split(
        X, y_bin, df["mission"].astype(str).values, test_size=0.2, random_state=42, stratify=y_bin
    )
    X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
        X_tmp, y_tmp, m_tmp, test_size=0.25, random_state=42, stratify=y_tmp
    )  # 0.25 of 0.8 => 0.2

    # 8) Impute + scale (fit on train only)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_i = imputer.fit_transform(X_train)
    X_val_i   = imputer.transform(X_val)
    X_test_i  = imputer.transform(X_test)

    X_train_z = scaler.fit_transform(X_train_i)
    X_val_z   = scaler.transform(X_val_i)
    X_test_z  = scaler.transform(X_test_i)

    # 9) Save artifacts & splits
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, ART / "numeric_imputer.joblib")
    joblib.dump(scaler,  ART / "numeric_scaler.joblib")
    with open(ART / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    np.save(PROC / "X_train.npy", X_train_z)
    np.save(PROC / "X_val.npy",   X_val_z)
    np.save(PROC / "X_test.npy",  X_test_z)

    pd.DataFrame({"y_binary": y_train}).to_csv(PROC / "y_train.csv", index=False)
    pd.DataFrame({"y_binary": y_val}).to_csv(PROC / "y_val.csv", index=False)
    pd.DataFrame({"y_binary": y_test}).to_csv(PROC / "y_test.csv", index=False)

    # 10) Save unified_clean.csv with z-scored features (for EDA)
    X_all_i = imputer.transform(X)
    X_all_z = scaler.transform(X_all_i)
    zcols = [f"z_{c}" for c in FEATURES_ORDER]
    df_z = pd.DataFrame(X_all_z, columns=zcols, index=df.index)
    out_clean = pd.concat(
        [df_z, df[["label_binary", "label_coarse", "mission"]].reset_index(drop=True)],
        axis=1
    )
    out_clean.to_csv(PROC / "unified_clean.csv", index=False)

    print("[done] unified_raw:", df.shape, " -> saved to data/processed/")
    print("[done] splits: X_train", X_train_z.shape, "X_val", X_val_z.shape, "X_test", X_test_z.shape)
    print("[done] artifacts:", ART)

if __name__ == "__main__":
    main()