# Cluster Profile Report and Final Interpretation

**Dataset:** Palmer Penguins (344 birds, 3 species, 3 islands). **Clustering features:** bill length, bill depth, flipper length, body mass.

## 1. Data preparation (Day 1)
- Selected 4 continuous morphology features. Species was excluded (it is a label and would leak the answer); island and sex were held out for profiling only.
- Missing values: 2 rows (0.6%) had all four measurements missing and were dropped, leaving **342 rows**. No rows had partial gaps, so no imputation was needed (a median imputer is in the pipeline for safety). Sex had 11 missing values but is not a clustering feature.
- No IQR outliers were found. All features were standardized (mean 0, sd 1) because body mass (~4000 g) would otherwise dominate distance calculations.

## 2. K-means and evaluation (Days 2-3)
| k | Inertia | Silhouette | Davies-Bouldin |
|---|---|---|---|
| 2 | 565.7 | **0.532** | **0.714** |
| 3 | 379.4 | 0.447 | 0.944 |
| 4 | 300.4 | 0.400 | 0.944 |
| 5 | 232.6 | 0.379 | 0.971 |
| 6 | 204.4 | 0.368 | 0.995 |

- **Silhouette and Davies-Bouldin favour k=2.** The elbow method points to **k=3**. Seeds are stable at k=2 (ARI 1.00) and reasonably stable at k=3 (mean ARI 0.89, worst 0.47), and less stable at k=4-5.
- **Selected: k=3.** This is a judgment call, not a metric win. k=2 is the statistically cleanest split (large birds vs. the rest) but merges two visibly different groups; k=3 keeps most of the silhouette quality, sits at the elbow, and is stable. k=2 remains a defensible alternative.

## 3. PCA (Day 4)
- PC1 explains **68.8%** of variance, PC2 **19.3%**; the first two together retain **88.2%**, three retain **97.3%**. A 2D plot is therefore a faithful view of the data.
- PC1 is essentially "overall size" (flipper length and body mass load positively, ~0.55-0.58). PC2 mostly separates bill shape (bill depth 0.80 and bill length 0.60).
- See `outputs/figures/d4_pca_visualization.png` and `d4_pca_3d.png`.

## 4. Cluster profiles (Day 5)
| Cluster | n (%) | Bill length | Bill depth | Flipper | Body mass |
|---|---|---|---|---|---|
| 0 | 87 (25%) | 47.5 mm (high) | 18.8 mm (high) | 196.9 mm (below avg) | 3,902 g (below avg) |
| 1 | 123 (36%) | 47.5 mm (high) | 15.0 mm (very low) | 217.2 mm (very high) | 5,076 g (very high) |
| 2 | 132 (39%) | 38.2 mm (low) | 18.1 mm (above avg) | 188.4 mm (low) | 3,585 g (low) |

- **Cluster 1, "large, long-flippered, slim-billed":** the heaviest birds by a wide margin, with shallow bills and long flippers.
- **Cluster 0, "medium body, long and deep bill":** average-to-small body with a notably long, deep bill.
- **Cluster 2, "small body, short bill":** the lightest birds with the shortest bills and flippers.

## 5. Practical meaning (interpreted cautiously)
Checking against the held-out labels: cluster 1 contains all 123 Gentoo penguins and nothing else; cluster 2 is 127 Adelie plus 5 Chinstrap; cluster 0 is 63 Chinstrap plus 24 Adelie. The agreement with species is strong (Adjusted Rand Index 0.79). Cluster 1 is also found only on Biscoe island.
This suggests the clusters reflect real morphological groups that largely align with species, but they are **not** species labels: 29 Adelie birds landed in the "Chinstrap-like" cluster and 5 Chinstrap in the "Adelie-like" one. Sex mixes across all clusters, so sex is not what drives the split (although males are larger on average, which blurs boundaries).

## 6. Limitations
1. **k is a choice.** Metrics disagree (silhouette says 2, elbow says 3); results depend on how we weigh them.
2. **K-means assumes roughly spherical, similar-sized clusters** and is sensitive to scaling and to initialization (worst-case ARI at k=3 was 0.47 across single-init runs; we use 25 inits to mitigate).
3. **Only 4 correlated size features** were used (PC1 alone carries ~69% of variance), so clusters are mostly "size" groups. Other traits could change the picture.
4. **Cross-tabs with species are validation, not proof.** Unsupervised clusters have no ground truth, and agreement on this dataset will not necessarily transfer to others.
5. **Small sample (342), imputation/dropping choices and a single snapshot** of the data; no significance testing or uncertainty estimates for cluster profiles were performed.
6. **PCA plots are a projection**: 11.8% of variance is lost in 2D, so apparent overlaps in the plot may not reflect true overlaps.
7. Cluster names are descriptive shorthand, not established categories.
