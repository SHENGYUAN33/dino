# extract.py
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from moco import MoCo
from config import CFG

cfg = CFG()
device = cfg.device
model = MoCo(cfg).to(device)
model.encoder_q.load_state_dict(torch.load(r"H:\dino\moco_retina.pth"))
model.encoder_q.eval()

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

feats = []
paths = []
for fn in os.listdir(cfg.data_dir):
    if not fn.lower().endswith((".png", ".jpg", ".bmp")):
        continue
    img = Image.open(os.path.join(cfg.data_dir, fn)).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encoder_q(x)  # 1×2048
    feats.append(f.cpu().numpy().squeeze())
    paths.append(fn)

# 轉成 NumPy 陣列
feats = np.stack(feats, axis=0)   # shape: (N, 2048)
paths = np.array(paths)           # shape: (N,)

# 儲存到磁碟
np.save(r"H:\dino\feats.npy", feats)
np.save(r"H:\dino\paths.npy", paths)

print(f"Saved feats.npy with shape {feats.shape}")
print(f"Saved paths.npy with {paths.shape[0]} entries")
