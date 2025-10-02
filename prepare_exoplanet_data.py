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


BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
ART = BASE / "artifacts"

# Create directories if they don't exist
for p in [RAW, PROC, ART]:
    p.mkdir(parents=True, exist_ok=True)

FEATURES_ORDER = ["period_days", "transit_depth_ppm", "planet_radius_re", "stellar_radius_rs", "snr"]
KOI_COLMAP = {"source_id": "kepoi_name", "period_days": "koi_period", "transit_depth_ppm": "koi_depth", "planet_radius_re": "koi_prad", "stellar_radius_rs": "koi_srad", "snr": "koi_snr", "epoch": "koi_time0bk", "label_raw": "koi_disposition"}
K2_COLMAP = {"source_id": "epic_id", "period_days": "pl_orbper", "transit_depth_ppm": "pl_trandep", "planet_radius_re": "pl_rade", "stellar_radius_rs": "st_rad", "snr": "k2_snr", "epoch": "pl_tranmid", "label_raw": "archive_disposition"}
TOI_COLMAP = {"source_id": "toi", "period_days": "pl_orbper", "transit_depth_ppm": "pl_trandep", "planet_radius_re": "pl_rade", "stellar_radius_rs": "st_rad", "snr": "snr", "epoch": "pl_tranmid", "label_raw": "tfopwg_disp"}
LABEL_MAP = {"CONFIRMED": ["CONFIRMED", "CONF", "CANDIDATE CONFIRMED", "KP", "CP", "KNOWN PLANET"], "CANDIDATE": ["CANDIDATE", "PC", "APC", "AMBIGUOUS", "CAND", "CANDIDATE AMBIGUOUS"], "FALSE POSITIVE": ["FALSE POSITIVE", "FP", "FALSE-POSITIVE", "NOT DISPOSITIONED - FALSE POSITIVE"]}

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
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame()
    for unified_col, src_col in colmap.items():
        out[unified_col] = df[src_col] if src_col in df.columns else np.nan
    out["mission"] = mission
    out["label_coarse"] = out["label_raw"].apply(normalize_label)
    if mission == "TESS":
        if out["period_days"].isna().all():
            for c in ["orbital_period", "toi_period"]:
                if c in df.columns:
                    out["period_days"] = pd.to_numeric(df[c], errors="coerce")
                    break
        if out["transit_depth_ppm"].isna().all():
            for c in ["transit_depth", "toi_transit_depth"]:
                if c in df.columns:
                    out["transit_depth_ppm"] = pd.to_numeric(df[c], errors="coerce")
                    break
    for c in FEATURES_ORDER + ["epoch"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def main():
    paths = {
        "KOI": RAW / "cumulative_2025.10.02_13.01.38.csv",
        "K2": RAW / "k2pandc_2025.10.02_13.03.14.csv",
        "TESS": RAW / "TOI_2025.10.02_13.02.56.csv",
    }

