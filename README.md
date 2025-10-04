
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

##  Labels
I normalize all mission labels into two forms:

- **Multiclass (`label_multiclass`)**  
  - `CONFIRMED`  
  - `CANDIDATE`  
  - `FALSE POSITIVE`

- **Binary (`label_binary`)**  
  - `1` = Confirmed planet  
  - `0` = Candidate or False Positive  

---

##  Features
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

##
To remember,how we preapared the data, artifacts are availabe.
Some values were missing → we filled them in using the median from the training set. Features had very different scales (e.g. radius in Earth units vs period in days)  we rescaled them so they all fit into a standard range (z-scoring).

numeric_imputer.joblib : remembers what numbers we used to fill in missing values.
numeric_scaler.joblib : remembers the average and standard deviation for each feature, so we can scale new data exactly the same way.
label_map.json : documents how labels like CP, PC, or FP were converted into CONFIRMED, CANDIDATE, or FALSE POSITIVE.


##  Exploratory Data Analysis (EDA)

Before training machine learning models, it is essential to understand the dataset — this step is called **Exploratory Data Analysis (EDA)**.  
EDA helps us check data quality, reveal patterns, and make sure our features actually carry useful information.

In this project, we performed EDA on the unified Kepler, K2, and TESS dataset:

### Correlation Heatmap

We computed the pairwise correlations between the five main z-scored features:

- `z_period_days` (orbital period)  
- `z_transit_depth_ppm` (transit depth)  
- `z_planet_radius_re` (planet radius in Earth radii)  
- `z_stellar_radius_rs` (stellar radius in Solar radii)  
- `z_snr` (signal-to-noise ratio)

The heatmap shows which features are strongly related. For example:
	•	Larger planets (planet_radius_re) often cause deeper transits (transit_depth_ppm).
	•	Stellar radius can influence transit depth and detectability.

This helps us understand redundancy and which features may add unique value.
**Key observations:**
- Transit depth and planet radius are strongly correlated (larger planets usually produce deeper transits).  
- Transit depth also correlates with SNR, since deeper transits are easier to detect.  
- Stellar radius shows moderate correlation with planet radius and transit depth.  
- Orbital period is mostly independent — providing unique information.  


⸻

### Principal Component Analysis (PCA)

PCA reduces the 5-dimensional feature space into 2 dimensions (PC1, PC2) while preserving as much variance as possible.
This allows us to visualize the structure of the dataset:
	•	PCA by binary label
	•	Points are colored as “Planet” (confirmed) or “Non-Planet” (candidate/false positive).
	•	If separation appears, it means our features are informative for classification.
	•	PCA by mission
	•	Points are colored by mission (Kepler, K2, TESS).
	•	This reveals how mission-specific biases or noise affect the feature space.
	•	Example: TESS may cluster differently due to brighter target stars and shorter baselines.

⸻

### Insights from EDA
	•	Some features are highly correlated (e.g., radius vs transit depth), so models may not need all of them.
	•	Outliers exist (extreme radius or depth values), which require clipping or scaling.
	•	PCA shows partial separation between planets and false positives, suggesting the dataset is suitable for ML classification.
	•	Missions have distinct distributions, so including “mission” as a feature may help the model generalize.

⸻

 Figures generated (in reports/figures/):
	•	correlation_heatmap.png — feature correlations
	•	pca_binary.png — PCA colored by planet vs non-planet
	•	pca_mission.png — PCA colored by mission


-------------------------


⸻

### UMAP Visualization

I applied UMAP (Uniform Manifold Approximation and Projection), a non-linear dimensionality reduction method. Unlike PCA, which is linear, UMAP captures more complex structures and is widely used to visualize high-dimensional data in 2D.

## UMAP colored by mission
	•	Each point represents a candidate from Kepler, K2, or TESS.
	•	Kepler (orange) and TESS (green) tend to form distinct regions in the embedding space, while K2 (blue) overlaps with both.
	•	This reflects the domain shift between missions — each survey has different noise levels, target stars, and observational strategies.
	•	It shows why adding mission as a feature can help the ML model learn across datasets.

## UMAP colored by label (planet vs non-planet)
	•	When points are colored by label_binary (1 = confirmed planet, 0 = candidate/false positive), we see partial separation.
	•	Confirmed planets cluster more densely in some regions, while false positives spread differently.
	•	The overlap means the classification task is challenging, but the visible trends confirm the features (period, depth, radius, stellar size, SNR) contain real signal.
	•	This supports using ML models to learn decision boundaries beyond what is visible in raw scatter plots.

⸻

## Takeaway
 UMAP reveals both mission-specific biases and label-driven structure in the dataset. It justifies:
	1.	Including “mission” as a feature during training.
	2.	Expecting ML models to perform better than naive thresholds, since structure is present but nonlinear.

⸻






⸻

Label Mapping

Mission Raw Label (examples)	label_coarse (normalized)	label_binary (numeric)
CONFIRMED, CONF, KP, CP, KNOWN PLANET	CONFIRMED	1
CANDIDATE, PC, APC, AMBIGUOUS, CANDIDATE AMBIGUOUS	CANDIDATE	0
FALSE POSITIVE, FP, FALSE-POSITIVE, FA, NOT DISPOSITIONED - FALSE POSITIVE	FALSE POSITIVE	0


⸻

 Explanation
	•	label_coarse is the human-readable normalized label across Kepler, K2, and TESS.
	•	label_binary is the simplified machine-learning target:
	•	1 = confirmed exoplanet
	•	0 = everything else (candidate or false positive).

⸻

