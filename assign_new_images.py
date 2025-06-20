# assign_new_images.py（修改后）

import os
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from moco import MoCo
from config import CFG

# --------- 参数区 ----------
MOCO_CKPT = r"H:\dino\moco_retina.pth"
PCA_PATH = r"H:\dino\models\pca_model.pkl"
KMEANS_PATH = r"H:\dino\models\kmeans_model.pkl"

NEW_DIR = r"H:\glaucoma\test"
OUTPUT_ROOT = r"H:\dino\new_clusters"
# ----------------------------

# 1. 载入模型、设定 device
cfg = CFG()
# 改为：如果没有 GPU 就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] using device: {device}")

model = MoCo(cfg)
model.encoder_q.load_state_dict(torch.load(MOCO_CKPT, map_location="cpu"))
# 无论如何都先把模型 load 到 CPU
model.encoder_q.to(device)
model.encoder_q.eval()

# 2. 加载 PCA & KMeans
with open(PCA_PATH,    "rb") as f:
    pca = pickle.load(f)
with open(KMEANS_PATH, "rb") as f:
    kmeans = pickle.load(f)

# 3. 图像预处理（与之前完全一致）
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 4. 批量提取新影像特征
new_paths = [f for f in os.listdir(
    NEW_DIR) if f.lower().endswith((".png", ".jpg", ".bmp"))]
new_feats = []
for fn in new_paths:
    img = Image.open(os.path.join(NEW_DIR, fn)).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encoder_q(x).cpu().numpy().squeeze()
    new_feats.append(feat)
new_feats = np.stack(new_feats, axis=0)  # (M, 2048)

# 5. 用同样的 PCA 降维
new_pca = pca.transform(new_feats)       # (M, PCA_dim)

# 6. 用 kmeans.predict 打标签
labels = kmeans.predict(new_pca)        # (M,)

# 7. 搬移到对应文件夹
os.makedirs(OUTPUT_ROOT, exist_ok=True)
for cid in set(labels):
    os.makedirs(os.path.join(OUTPUT_ROOT, f"cluster_{cid}"), exist_ok=True)

for fn, lbl in zip(new_paths, labels):
    src = os.path.join(NEW_DIR, fn)
    dst = os.path.join(OUTPUT_ROOT, f"cluster_{lbl}", fn)
    shutil.move(src, dst)

print(f"Assigned {len(new_paths)} images into {len(set(labels))} clusters.")
