# train_moco.py （只展示关键部分）
import torch
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm          # 进度条库，确保你已 install tqdm
from config import CFG
from dataset import get_moco_dataloader
from moco import MoCo


def train():
    cfg = CFG()
    device = torch.device(cfg.device)
    loader = get_moco_dataloader(cfg)

    model = MoCo(cfg).to(device)
    optimizer = SGD(
        model.encoder_q.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        # 用 tqdm 包装 DataLoader，就会显示进度条
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", ncols=100)
        for im_q, im_k in loop:
            im_q, im_k = im_q.to(device), im_k.to(device)

            logits, labels = model(im_q, im_k)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加 loss
            epoch_loss += loss.item()

            # 每个 batch 更新一下进度条后缀，显示当前平均 loss
            loop.set_postfix(loss=epoch_loss / (loop.n if loop.n > 0 else 1))

        # 一个 epoch 结束后，再打印一次最终平均 loss
        avg_loss = epoch_loss / len(loader)
        print(
            f"[MAIN] Epoch {epoch+1} finished, average loss = {avg_loss:.4f}")

    # 训练完后存模型
    torch.save(model.encoder_q.state_dict(), "moco_retina.pth")
    print("[MAIN] Training complete, checkpoint saved as moco_retina.pth")


if __name__ == "__main__":
    train()
