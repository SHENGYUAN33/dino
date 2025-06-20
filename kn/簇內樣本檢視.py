import os
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# 假設你在 extract.py 裡 save 成了 feats.npy, paths.npy
feats = np.load(r"H:\dino\feats.npy")      # shape: (N, 2048)
paths = np.load(r"H:\dino\paths.npy")      # shape: (N,)  存每張圖對應的檔名
k = 30  # 先假設分 10 群，或用肘部法、輪廓係數找最優 k
km = KMeans(n_clusters=k, random_state=42)
labels_km = km.fit_predict(feats)

# 建立輸出資料夾
os.makedirs("cluster_examples", exist_ok=True)

for cluster_id in sorted(set(labels_km)):
    idxs = np.where(labels_km == cluster_id)[0]
    # 挑 9 張最接近中心的：
    center = km.cluster_centers_[cluster_id]
    dists = np.linalg.norm(feats[idxs] - center, axis=1)
    chosen = idxs[np.argsort(dists)[:9]]

    # 拼圖 3×3
    imgs = [Image.open(os.path.join(r"H:\glaucoma\test", paths[i])).resize(
        (128, 128)) for i in chosen]
    grid = Image.new("RGB", (128*3, 128*3))
    for j, im in enumerate(imgs):
        grid.paste(im, ((j % 3)*128, (j//3)*128))
    grid.save(f"cluster_examples/cluster_{cluster_id}.jpg")
