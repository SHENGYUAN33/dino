from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import umap


# 假設你在 extract.py 裡 save 成了 feats.npy, paths.npy
feats = np.load(r"H:\dino\feats.npy")      # shape: (N, 2048)
paths = np.load(r"H:\dino\paths.npy")      # shape: (N,)  存每張圖對應的檔名


# PCA
for d in [10, 20, 50, 100]:
    pca = PCA(n_components=d, random_state=42)
    feats_pca = pca.fit_transform(feats)
    print("d=", d, "explained var ratio=", pca.explained_variance_ratio_.sum())
