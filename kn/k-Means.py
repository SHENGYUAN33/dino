from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# 假設你在 extract.py 裡 save 成了 feats.npy, paths.npy
feats = np.load(r"H:\dino\feats.npy")      # shape: (N, 2048)
paths = np.load(r"H:\dino\paths.npy")      # shape: (N,)  存每張圖對應的檔名


k = 5  # 先假設分 10 群，或用肘部法、輪廓係數找最優 k
km = KMeans(n_clusters=k, random_state=42)
labels_km = km.fit_predict(feats)

# 內部指標：Silhouette Score（越接近 1 越好，<0 表示分群很糟）
print("KMeans silhouette:", silhouette_score(feats, labels_km))
