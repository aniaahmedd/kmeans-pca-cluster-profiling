"""Day 2: k-means for k=2..10, record results."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from common import *

X = pd.read_csv(TAB / "d1_scaled_features.csv")
rows = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=25, random_state=SEED).fit(X)
    rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X, km.labels_),
                     davies_bouldin=davies_bouldin_score(X, km.labels_),
                     calinski_harabasz=calinski_harabasz_score(X, km.labels_),
                     min_cluster_size=int(pd.Series(km.labels_).value_counts().min())))
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False)); res.to_csv(TAB / "d2_kmeans_results.csv", index=False)

# Stability: ARI between runs with different seeds
from sklearn.metrics import adjusted_rand_score
stab = []
for k in range(2, 7):
    base = KMeans(k, n_init=1, random_state=0).fit_predict(X)
    aris = [adjusted_rand_score(base, KMeans(k, n_init=1, random_state=s).fit_predict(X)) for s in range(1, 21)]
    stab.append(dict(k=k, mean_ari=round(sum(aris)/len(aris), 3), min_ari=round(min(aris), 3)))
stab = pd.DataFrame(stab); print(stab.to_string(index=False)); stab.to_csv(TAB / "d2_stability.csv", index=False)
