"""Day 5: profile clusters; external categorical variables used ONLY for interpretation."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from common import *

X = pd.read_csv(TAB / "d1_scaled_features.csv"); df = pd.read_csv(TAB / "d1_clean_unscaled.csv")
k = int(pd.read_csv(TAB / "d3_selection.csv", index_col=0).loc["chosen_k"].iloc[0])
df["cluster"] = KMeans(k, n_init=25, random_state=SEED).fit_predict(X)
size = df.cluster.value_counts().sort_index().rename("n").to_frame(); size["pct"] = (size.n/len(df)*100).round(1)
prof = df.groupby("cluster")[FEATURES].mean().round(1); print(size, "\n", prof)
z = pd.DataFrame(X.values, columns=FEATURES).assign(cluster=df.cluster).groupby("cluster").mean().round(2); print("\nZ-score profile:\n", z)
sp = pd.crosstab(df.cluster, df.species); isl = pd.crosstab(df.cluster, df.island); sx = pd.crosstab(df.cluster, df.sex)
print("\n", sp, "\n\n", isl, "\n\n", sx)
print("\nARI vs species (external check only):", round(adjusted_rand_score(df.species, df.cluster), 3))
pd.concat([size, prof], axis=1).to_csv(TAB / "d5_cluster_profiles.csv"); z.to_csv(TAB / "d5_cluster_zprofile.csv")
sp.to_csv(TAB / "d5_cluster_vs_species.csv"); isl.to_csv(TAB / "d5_cluster_vs_island.csv"); sx.to_csv(TAB / "d5_cluster_vs_sex.csv")
df.to_csv(TAB / "d5_clustered_data.csv", index=False)

import seaborn as sns
plt.figure(figsize=(6, 3.5)); sns.heatmap(z, annot=True, cmap="coolwarm", center=0); plt.title("Cluster profiles (z-scores)")
plt.tight_layout(); plt.savefig(FIG / "d5_profile_heatmap.png", dpi=140)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
for a, f in zip(ax, FEATURES): sns.boxplot(data=df, x="cluster", y=f, ax=a)
plt.tight_layout(); plt.savefig(FIG / "d5_feature_boxplots.png", dpi=130)
