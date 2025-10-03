
1. **Kepler**  
   - Label: `koi_pdisposition` (preferred) / `koi_disposition` (fallback)  
   - Features: `koi_period`, `koi_depth`, `koi_prad`, `koi_srad`, `koi_model_snr`

2. **K2**  
   - Label: `disposition`  
   - Features: `pl_orbper`, `pl_trandep`, `pl_rade`, `st_rad`, (SNR rarely present)

3. **TESS O(TOI)**  
   - Label: `tfopwg_disp` (`CP`, `KP`, `PC`, `FP`, `APC`)  
   - Features: `pl_orbper`, `pl_trandep`, `pl_rade`, `st_rad`  

---

## 🧾 Labels
I normalize all mission labels into two forms:

- **Multiclass (`label_multiclass`)**  
  - `CONFIRMED`  
  - `CANDIDATE`  
  - `FALSE POSITIVE`

- **Binary (`label_binary`)**  
  - `1` = Confirmed planet  
  - `0` = Candidate or False Positive  

---

## 📊 Features
The pipeline extracts five core **tabular features** (common across missions):

- `period_days` → orbital period [days]  
- `transit_depth_ppm` → transit depth [ppm]  
- `planet_radius_re` → planet radius [Earth radii]  
- `stellar_radius_rs` → stellar radius [Solar radii]  
- `snr` → transit signal-to-noise ratio  

These are preprocessed into **z-scored features** (standardized: mean 0, std 1) and saved as:  

```
z_period_days, z_transit_depth_ppm, z_planet_radius_re, z_stellar_radius_rs, z_snr
```

---
