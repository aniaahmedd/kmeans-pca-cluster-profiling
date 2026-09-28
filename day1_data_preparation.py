"""Day 1: select features, handle missing values, standardize."""
import pandas as pd, numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from common import *

raw = load_raw()
print("Shape:", raw.shape)
miss = raw.isna().sum().rename("missing").to_frame()
miss["pct"] = (miss.missing / len(raw) * 100).round(2)
print(miss); miss.to_csv(TAB / "d1_missing_values.csv")

# Feature selection: 4 continuous morphology measurements.
# Excluded: species (label, would leak the answer), island/sex (categorical; keep for profiling only).
df = raw.copy()
# Rows missing ALL measurements carry no information -> drop; partial gaps -> median impute.
all_missing = df[FEATURES].isna().all(axis=1)
print("Rows with all features missing (dropped):", all_missing.sum())
df = df[~all_missing].reset_index(drop=True)
imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(df[FEATURES]), columns=FEATURES)
print("Partial-missing rows imputed:", int(df[FEATURES].isna().any(axis=1).sum()))

# Outlier check (IQR) - reported, not removed (k-means is outlier sensitive, so we flag it)
q1, q3 = X_imp.quantile(.25), X_imp.quantile(.75); iqr = q3 - q1
out = ((X_imp < q1 - 1.5*iqr) | (X_imp > q3 + 1.5*iqr)).sum()
print("IQR outliers per feature:\n", out)

# Standardize (needed: body_mass_g is ~1000x larger scale than bill depth)
X = pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=FEATURES)
print(X.describe().loc[["mean", "std"]].round(3))

df[FEATURES] = X_imp
df.to_csv(TAB / "d1_clean_unscaled.csv", index=False)
X.to_csv(TAB / "d1_scaled_features.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
X_imp.boxplot(ax=ax[0], rot=25); ax[0].set_title("Before scaling (raw units)")
X.boxplot(ax=ax[1], rot=25); ax[1].set_title("After standardization")
plt.tight_layout(); plt.savefig(FIG / "d1_scaling.png", dpi=130)
