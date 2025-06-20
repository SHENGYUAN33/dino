# train_and_save_models.py

import os
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------- 配置区 ----------
FEATS_PATH = r"H:\dino\feats.npy"
PICKLE_DIR = r"H:\dino\models"      # 保存 pkl 的文件夹
PCA_DIM = 50                      # 你上次选的 PCA 维度
K = 6                       # 你上次选的簇个数
# -------------------------------

# 1. 载入特征
feats = np.load(FEATS_PATH)  # (N, 2048)
print(f"Loaded feats.npy, shape = {feats.shape}")

# 确保保存目录存在
os.makedirs(PICKLE_DIR, exist_ok=True)

# 2. 训练 PCA
pca = PCA(n_components=PCA_DIM, random_state=42)
feats_pca = pca.fit_transform(feats)
print(
    f"PCA done (n_components={PCA_DIM}), explained variance = {pca.explained_variance_ratio_.sum():.4f}")

# 保存 PCA 模型
pca_path = os.path.join(PICKLE_DIR, "pca_model.pkl")
with open(pca_path, "wb") as f:
    pickle.dump(pca, f)
print(f"Saved PCA model to {pca_path}")

# 3. 训练 KMeans
km = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
labels = km.fit_predict(feats_pca)
print(f"KMeans done (n_clusters={K}), sample labels: {labels[:10]}")

# 保存 KMeans 模型
km_path = os.path.join(PICKLE_DIR, "kmeans_model.pkl")
with open(km_path, "wb") as f:
    pickle.dump(km, f)
print(f"Saved KMeans model to {km_path}")
