# move_images_by_cluster.py
# 將未標註影像依照 cluster 標籤搬移到對應資料夾

import os
import shutil
import numpy as np
from sklearn.cluster import KMeans

# ---------- 使用者參數 ----------
# 特徵與檔名儲存路徑
FEATS_PATH = r"H:\dino\feats.npy"
PATHS_PATH = r"H:\dino\paths.npy"
# 原始影像資料夾
INPUT_DIR = r"H:\glaucoma\test"
# 輸出分類後資料夾（會在此目錄下建立 cluster_0, cluster_1, ... 等）
OUTPUT_ROOT = r"H:\dino\clusters"
# 分群數量 k
K = 30
# --------------------------------

# 1. 載入特徵與檔名
feats = np.load(FEATS_PATH)    # (N, D)
paths = np.load(PATHS_PATH)    # (N,)

# 2. 執行 KMeans
km = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=300)
labels = km.fit_predict(feats)

# 3. 建立輸出資料夾
os.makedirs(OUTPUT_ROOT, exist_ok=True)
for cid in range(K):
    folder = os.path.join(OUTPUT_ROOT, f"cluster_{cid}")
    os.makedirs(folder, exist_ok=True)

# 4. 移動每張影像到對應群資料夾
for i, fname in enumerate(paths):
    src = os.path.join(INPUT_DIR, fname)
    if not os.path.isfile(src):
        print(f"Warning: file not found: {src}")
        continue
    dest_folder = os.path.join(OUTPUT_ROOT, f"cluster_{labels[i]}")
    dest = os.path.join(dest_folder, fname)
    try:
        shutil.move(src, dest)
    except Exception as e:
        print(f"Failed to move {src} -> {dest}: {e}")

print("All images have been moved to cluster folders.")
