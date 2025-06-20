import hdbscan
from sklearn.decomposition import PCA
import numpy as np

# 假設你在 extract.py 裡 save 成了 feats.npy, paths.npy
feats = np.load(r"H:\dino\feats.npy")      # shape: (N, 2048)
paths = np.load(r"H:\dino\paths.npy")      # shape: (N,)  存每張圖對應的檔名

clusterer = hdbscan.HDBSCAN(min_cluster_size=50, prediction_data=True)
labels_hdb = clusterer.fit_predict(feats)
print("HDBSCAN found clusters:", len(
    set(labels_hdb)) - (1 if -1 in labels_hdb else 0))
print("HDBSCAN silhouette (ignore noise):",
      silhouette_score(feats[labels_hdb >= 0], labels_hdb[labels_hdb >= 0]))
