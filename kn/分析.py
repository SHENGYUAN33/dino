# clustering_pipeline.py
# 完整聚類流程腳本：
# - 選對 k：掃描不同 k，找最大 Silhouette
# - 精煉特徵：PCA 維度、UMAP、標準化/白化
# - 清除雜訊：DBSCAN／HDBSCAN 去除離群後再聚類
# - 試不同距離：L2 normalize → Cosine KMeans
# - 微調超參數：n_init、max_iter

from scipy.cluster.vq import whiten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
import hdbscan

# 1. 讀取特徵
feats = np.load(r"H:\dino\feats.npy")  # shape (N, D)
print(f"Loaded feats.npy, shape={feats.shape}")

# 2. 特徵標準化與白化
scaler = StandardScaler()
feats_std = scaler.fit_transform(feats)
feats_whiten = feats_std.copy()
# whiten 会返回白化後的數值
feats_whiten = whiten(feats)

# 3. PCA 維度掃描
pca_dims = [10, 20, 50, 100]
scores_pca = {}
for d in pca_dims:
    pca = PCA(n_components=d, random_state=42)
    Xp = pca.fit_transform(feats_std)
    print(
        f"PCA dim={d}, explained variance ratio sum={pca.explained_variance_ratio_.sum():.4f}")
    # 在每個維度上掃描 k
    ks = list(range(2, 11))
    best = (None, -1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42,
                    n_init=10, max_iter=300).fit(Xp)
        score = silhouette_score(Xp, km.labels_)
        if score > best[1]:
            best = (k, score)
    scores_pca[d] = best
    print(f"  Best k at dim {d}: k={best[0]}, silhouette={best[1]:.4f}")

# 4. UMAP 降維可視化（選用 PCA 50 維後的特徵）
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb2d = reducer.fit_transform(
    PCA(n_components=50, random_state=42).fit_transform(feats_std))
plt.figure(figsize=(6, 6))
plt.scatter(emb2d[:, 0], emb2d[:, 1], s=2, cmap='Spectral')
plt.title("UMAP 2D Projection of Retinal Features")
plt.show()

# 5. 清除雜訊：HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
labels_h = clusterer.fit_predict(
    PCA(n_components=50, random_state=42).fit_transform(feats_std))
mask = labels_h != -1
print(
    f"HDBSCAN removed {np.sum(mask==False)} noise points, kept {np.sum(mask)} points")
X_clean = feats_std[mask]

# 6. L2 Normalize + Cosine KMeans
X_norm = normalize(
    PCA(n_components=50, random_state=42).fit_transform(X_clean))
# 重複掃描最佳 k
ks = list(range(2, 11))
best_cosine = (None, -1)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42,
                n_init=20, max_iter=500).fit(X_norm)
    score = silhouette_score(X_norm, km.labels_)
    if score > best_cosine[1]:
        best_cosine = (k, score)
print(
    f"Cosine KMeans best: k={best_cosine[0]}, silhouette={best_cosine[1]:.4f}")

# 7. 最終聚類與結果展示
final_k = best_cosine[0]
km_final = KMeans(n_clusters=final_k, random_state=42,
                  n_init=20, max_iter=500).fit(X_norm)
labels_final = km_final.labels_

# UMAP 上展示最終結果
emb2d_final = reducer.fit_transform(X_norm)
plt.figure(figsize=(6, 6))
plt.scatter(emb2d_final[:, 0], emb2d_final[:, 1],
            c=labels_final, s=2, cmap='tab10')
plt.title(f"Final Cosine KMeans (k={final_k}) Clustering on UMAP")
plt.show()

# 8. Silhouette Score 最終打印
print(f"Final silhouette score: {silhouette_score(X_norm, labels_final):.4f}")
