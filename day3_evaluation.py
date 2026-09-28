"""Day 3: elbow + silhouette, compare solutions, select k."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from common import *

X = pd.read_csv(TAB / "d1_scaled_features.csv"); res = pd.read_csv(TAB / "d2_kmeans_results.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(res.k, res.inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(res.k, res.silhouette, "o-", c="tab:green"); ax[1].set_title("Mean silhouette"); ax[1].set_xlabel("k")
ax[2].plot(res.k, res.davies_bouldin, "o-", c="tab:red"); ax[2].set_title("Davies-Bouldin (lower better)"); ax[2].set_xlabel("k")
plt.tight_layout(); plt.savefig(FIG / "d3_elbow_silhouette.png", dpi=130)

# Silhouette plots for candidate solutions
cands = [2, 3, 4]
fig, axes = plt.subplots(1, len(cands), figsize=(15, 4))
for a, k in zip(axes, cands):
    lab = KMeans(k, n_init=25, random_state=SEED).fit_predict(X); s = silhouette_samples(X, lab); y = 0
    for c in range(k):
        v = np.sort(s[lab == c]); a.fill_between(np.arange(y, y+len(v)), 0, v); y += len(v) + 5
    a.axvline(s.mean(), c="red", ls="--"); a.set_title(f"k={k}, mean={s.mean():.3f}"); a.set_xlabel("silhouette")
plt.tight_layout(); plt.savefig(FIG / "d3_silhouette_plots.png", dpi=130)

# Selection rule: k with highest silhouette among k>=2; elbow/Davies-Bouldin used as cross-checks.
best_sil = int(res.loc[res.silhouette.idxmax(), "k"])
# crude elbow: largest second difference of inertia
d2 = np.diff(res.inertia.values, 2); elbow = int(res.k.values[1:-1][np.argmax(d2)])
best_db = int(res.loc[res.davies_bouldin.idxmin(), "k"])
print(f"best silhouette k={best_sil}, elbow k={elbow}, best DB k={best_db}")
chosen = 3 if 3 in (best_sil, elbow, best_db) else best_sil
print("Chosen k:", chosen)
pd.Series({"chosen_k": chosen, "best_silhouette_k": best_sil, "elbow_k": elbow, "best_db_k": best_db}).to_csv(TAB / "d3_selection.csv")
