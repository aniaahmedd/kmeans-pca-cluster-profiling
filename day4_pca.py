"""Day 4: PCA, variance retained, 2D/3D visualizations."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from common import *

X = pd.read_csv(TAB / "d1_scaled_features.csv")
k = int(pd.read_csv(TAB / "d3_selection.csv", index_col=0).loc["chosen_k"].iloc[0])
labels = KMeans(k, n_init=25, random_state=SEED).fit_predict(X)
pca = PCA().fit(X); evr = pca.explained_variance_ratio_
print("Explained variance ratio:", evr.round(4), "cumulative:", np.cumsum(evr).round(4))
pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "explained": evr, "cumulative": np.cumsum(evr)}).round(4).to_csv(TAB / "d4_variance.csv", index=False)
load = pd.DataFrame(pca.components_.T, index=FEATURES, columns=[f"PC{i+1}" for i in range(len(evr))]).round(3)
print(load); load.to_csv(TAB / "d4_loadings.csv")

Z = PCA(n_components=3).fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].bar(range(1, len(evr)+1), evr); ax[0].plot(range(1, len(evr)+1), np.cumsum(evr), "o-r"); ax[0].set_title("Scree + cumulative variance")
sc = ax[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="viridis", s=25, alpha=.8)
ax[1].set_xlabel(f"PC1 ({evr[0]:.1%})"); ax[1].set_ylabel(f"PC2 ({evr[1]:.1%})"); ax[1].set_title(f"k-means (k={k}) clusters in PCA space")
plt.tight_layout(); plt.savefig(FIG / "d4_pca_visualization.png", dpi=140)

fig = plt.figure(figsize=(7, 6)); a = fig.add_subplot(111, projection="3d")
a.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=labels, cmap="viridis", s=20)
a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.set_zlabel("PC3"); a.set_title("3D PCA view")
plt.savefig(FIG / "d4_pca_3d.png", dpi=140)
pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "PC3": Z[:, 2], "cluster": labels}).to_csv(TAB / "d4_pca_coords.csv", index=False)
