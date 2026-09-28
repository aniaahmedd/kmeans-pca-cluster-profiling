# Clustering Project: K-Means, PCA and Cluster Profiling

5-day unsupervised learning project on the Palmer Penguins dataset (344 rows; 4 continuous measurements).
Species is **never** used for clustering; it is only used on Day 5 as an external sanity check.

## Run
```
pip install pandas numpy scikit-learn matplotlib seaborn
cd src
python day1_data_preparation.py
python day2_kmeans.py
python day3_evaluation.py
python day4_pca.py
python day5_profiling.py
```
Data source: `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv` (saved as `penguins.csv`).

## Structure
| Day | Script | Main outputs |
|---|---|---|
| 1 | `day1_data_preparation.py` | missing-value table, cleaned + scaled data, scaling plot |
| 2 | `day2_kmeans.py` | k=2..10 results table, seed-stability table |
| 3 | `day3_evaluation.py` | elbow / silhouette / Davies-Bouldin plots, silhouette plots, chosen k |
| 4 | `day4_pca.py` | variance table, loadings, **PCA visualization (2D + 3D)** |
| 5 | `day5_profiling.py` + `REPORT.md` | cluster profiles, heatmap, **profile report + final interpretation** |
